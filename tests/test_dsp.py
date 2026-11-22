"""Features, cross-correlation, ratio selection and level finding on in-memory signals."""

from fractions import Fraction

import numpy as np
import pytest
from conftest import PAL, music_and_effects

from dubweld import ratio as ratio_mod
from dubweld.features import FRAME_RATE, N_BANDS, band_edges, compute_features, moving_average, stretch
from dubweld.levels import Level, changepoint, find_levels, prune_levels, runs_of, viterbi
from dubweld.ratio import MAX_SLOPE, NAMED, candidates, choose_ratio, drift_slope, flat_count, parse_ratio
from dubweld.verify import _crisp_clusters
from dubweld.xcorr import Matcher, OffsetTrack, offset_track, scores_at_offset

SR = 16000


@pytest.fixture(scope="module")
def bed() -> np.ndarray:
    return music_and_effects(150, SR, seed=3)


def speed_up(x: np.ndarray, ratio: float, sr: int = SR) -> np.ndarray:
    """What a PAL transfer does: the same content played `ratio` times faster."""
    n = int(len(x) / ratio)
    return np.interp(np.arange(n) * ratio, np.arange(len(x)), x).astype(np.float32)


class TestFeatures:
    def test_rate_and_shape(self, bed: np.ndarray) -> None:
        f = compute_features(bed)
        assert f.bands.shape[0] == N_BANDS
        assert abs(f.n_frames / FRAME_RATE - len(bed) / SR) < 0.1
        assert f.env_db.shape == (f.n_frames,)

    def test_band_edges_strictly_increase(self) -> None:
        edges = band_edges()
        assert len(edges) == N_BANDS + 1
        assert np.all(np.diff(edges) >= 1)

    def test_zero_mean_unit_variance(self, bed: np.ndarray) -> None:
        f = compute_features(bed)
        assert np.allclose(f.bands.mean(axis=1), 0, atol=1e-3)
        assert np.allclose(f.bands.std(axis=1), 1, atol=1e-2)

    def test_gain_does_not_matter(self, bed: np.ndarray) -> None:
        a, b = compute_features(bed), compute_features(bed * 0.25)
        corr = np.sum(a.bands * b.bands) / np.sqrt(np.sum(a.bands**2) * np.sum(b.bands**2))
        assert corr > 0.98

    def test_moving_average_of_constant(self) -> None:
        assert np.allclose(moving_average(np.full((2, 50), 3.0), 9), 3.0)

    def test_stretch_length(self, bed: np.ndarray) -> None:
        f = compute_features(bed)
        assert abs(stretch(f, PAL).n_frames - f.n_frames * PAL) < 3


class TestMatcher:
    def test_finds_integer_shift(self, bed: np.ndarray) -> None:
        shift = int(3.0 * SR)
        donor = compute_features(bed[shift:])
        track = offset_track(Matcher(compute_features(bed)), donor)
        good = track.scores >= 0.5
        assert good.sum() > len(track) * 0.8
        assert np.allclose(track.offsets[good], 3.0, atol=0.002)

    def test_finds_fractional_shift(self, bed: np.ndarray) -> None:
        # 13 ms is between two 20 ms frames
        shift = int(0.013 * SR)
        track = offset_track(Matcher(compute_features(bed)), compute_features(bed[shift:]))
        good = track.scores >= 0.5
        assert abs(np.median(track.offsets[good]) - 0.013) < 0.003

    def test_speed_change_undone_by_stretch(self, bed: np.ndarray) -> None:
        donor = stretch(compute_features(speed_up(bed, PAL)), PAL)
        track = offset_track(Matcher(compute_features(bed)), donor)
        good = track.scores >= 0.5
        assert abs(np.median(track.offsets[good])) < 0.002

    def test_max_offset_limits_search(self, bed: np.ndarray) -> None:
        shift = int(20.0 * SR)
        track = offset_track(Matcher(compute_features(bed)), compute_features(bed[shift:]), max_offset=5.0)
        assert np.all(np.abs(track.offsets) <= 5.0 + 0.2)

    def test_score_at_offset(self, bed: np.ndarray) -> None:
        m = Matcher(compute_features(bed))
        donor = compute_features(bed)
        track = offset_track(m, donor)
        at0 = scores_at_offset(m, donor, track, 0.0)
        far = scores_at_offset(m, donor, track, 7.3)
        assert np.nanmedian(at0) > 0.95
        assert np.nanmedian(far) < 0.3


class TestRatio:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("auto", None),
            ("pal", float(NAMED["pal"])),
            ("pal-inv", float(1 / NAMED["pal"])),
            ("1", 1.0),
            ("25/24", 25 / 24),
            ("1.0427", 1.0427),
        ],
    )
    def test_parse(self, text: str, value: float | None) -> None:
        assert parse_ratio(text) == value

    @pytest.mark.parametrize("text", ["fast", "0.1", "3", "1/0"])
    def test_parse_rejects(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_ratio(text)

    def test_frame_rate_quotient_first(self) -> None:
        cands = candidates(Fraction(24000, 1001), Fraction(25))
        assert cands[0].name == "pal"
        assert len({round(c.ratio, 9) for c in cands}) == len(cands)

    def test_snaps_to_pal(self, bed: np.ndarray) -> None:
        donor = compute_features(speed_up(bed, PAL))
        choice = choose_ratio(Matcher(compute_features(bed)), donor, None, None)
        assert choice.method == "snapped"
        assert choice.ratio == pytest.approx(PAL, abs=1e-9)

    def test_slightly_wrong_ratio_drifts(self, bed: np.ndarray) -> None:
        # DubSync's estimate: 3e-4 away from PAL, which drifted 0.3 s over an episode
        m = Matcher(compute_features(bed))
        donor = compute_features(speed_up(bed, PAL))
        right = offset_track(m, stretch(donor, PAL), hop_s=4)
        wrong = offset_track(m, stretch(donor, 1.04304), hop_s=4)
        assert abs(drift_slope(right)) < 2e-5
        assert drift_slope(wrong) == pytest.approx(PAL / 1.04304 - 1, abs=1e-4)
        assert abs(drift_slope(wrong)) > MAX_SLOPE
        assert flat_count(right)[0] >= flat_count(wrong)[0]

    def test_estimates_non_standard_ratio(self, bed: np.ndarray) -> None:
        donor = compute_features(speed_up(bed, 1.0213))
        choice = choose_ratio(Matcher(compute_features(bed)), donor, None, None)
        assert choice.method == "estimated"
        assert choice.warning
        assert choice.ratio == pytest.approx(1.0213, abs=5e-5)

    def test_keeps_standard_ratio_when_estimate_is_no_flatter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A real release: PAL measured a noisy drift of 5.5e-5 (just over the limit); the estimate was no flatter
        def fake(_m: object, _d: object, cand: ratio_mod.Candidate, _hop: float) -> None:
            pal = abs(cand.ratio - PAL) < 1e-9
            near = abs(cand.ratio - PAL) < 1e-4
            cand.flat, cand.confident = (4 if near else 0), 20
            cand.slope = 5.5e-5 if pal else 0.0

        monkeypatch.setattr(ratio_mod, "_evaluate", fake)
        choice = choose_ratio(None, None, Fraction(24000, 1001), Fraction(25))  # type: ignore[arg-type]
        assert choice.method == "snapped"
        assert choice.ratio == PAL
        assert not choice.warning


def _track(times: np.ndarray, offsets: np.ndarray, scores: np.ndarray) -> OffsetTrack:
    return OffsetTrack(times=times, offsets=offsets, scores=scores, starts=np.zeros(len(times), int), window_frames=400)


class TestLevels:
    def test_crisp_level_beats_bigger_loose_cluster(self) -> None:
        rng = np.random.default_rng(0)
        # 10 crisp windows at 6.276 ±2 ms vs 30 loose ones drifting over 5.86-6.16 (more of them, similar scores)
        crisp_t = np.sort(rng.uniform(700, 1240, 10))
        loose_t = np.sort(rng.uniform(700, 1240, 30))
        t = np.concatenate([crisp_t, loose_t])
        o = np.concatenate([6.276 + rng.uniform(-0.002, 0.002, 10), rng.uniform(5.86, 6.16, 30)])
        s = np.concatenate([rng.uniform(0.5, 0.85, 10), rng.uniform(0.45, 0.6, 30)])
        order = np.argsort(t)
        levels = find_levels(_track(t[order], o[order], s[order]))
        kept = prune_levels(levels)
        assert [round(lv.offset, 3) for lv in kept] == [pytest.approx(6.276, abs=0.003)]

    def test_levels_need_three_windows(self) -> None:
        t = np.arange(6) * 2.0
        o = np.array([1.0, 1.001, 3.0, 3.001, 3.002, 9.0])
        s = np.full(6, 0.7)
        assert [round(lv.offset, 2) for lv in find_levels(_track(t, o, s))] == [3.0]

    def test_distinct_crisp_levels_both_kept(self) -> None:
        a = Level(offset=4.0, members=10, seeds=8, crispness=1.0, median_score=0.7, first=0, last=100, support=7)
        b = Level(offset=4.3, members=8, seeds=6, crispness=0.9, median_score=0.7, first=200, last=300, support=5)
        assert len(prune_levels([a, b])) == 2

    def test_video_does_not_rescue_a_loose_level(self) -> None:
        # A real release: the video favored loose clusters 0.18-0.27 s off; the donor's A/V sync varies by that much
        crisp = Level(offset=4.138, members=13, seeds=9, crispness=0.87, median_score=0.54, first=0, last=1, support=7)
        loose = Level(offset=3.956, members=8, seeds=4, crispness=0.30, median_score=0.49, first=0, last=1, support=4)
        kept = prune_levels([crisp, loose], {id(crisp): 0.18, id(loose): 0.0})
        assert kept == [crisp]

    def test_video_settles_two_crisp_levels(self) -> None:
        a = Level(offset=6.05, members=9, seeds=6, crispness=0.9, median_score=0.6, first=0, last=1, support=6)
        b = Level(offset=6.27, members=8, seeds=6, crispness=0.9, median_score=0.6, first=0, last=1, support=5)
        assert prune_levels([a, b], {id(a): 0.01, id(b): 0.22}) == [a]
        assert len(prune_levels([a, b], {id(a): 0.01, id(b): 0.05})) == 2

    def test_viterbi_ignores_single_window_flips(self) -> None:
        scores = np.array([[0.6, 0.1]] * 10 + [[0.1, 0.6]] + [[0.6, 0.1]] * 10 + [[0.1, 0.7]] * 10)
        path = viterbi(scores)
        assert list(path[:21]) == [0] * 21
        assert list(path[21:]) == [1] * 10

    def test_viterbi_holds_through_silence(self) -> None:
        scores = np.array([[0.6, 0.1]] * 10 + [[0.05, 0.05]] * 20 + [[0.1, 0.6]] * 10)
        path = viterbi(scores)
        assert path[0] == 0 and path[-1] == 1
        assert np.all(np.diff(path) >= 0)

    def test_unbacked_runs_are_merged(self) -> None:
        t = np.arange(20) * 2.0
        o = np.array([1.0] * 10 + [5.0] * 10)
        s = np.array([0.6] * 10 + [0.6] * 10)
        levels = [
            Level(offset=1.0, members=10, seeds=10, crispness=1, median_score=0.6, first=0, last=18, support=6),
            Level(offset=5.0, members=10, seeds=10, crispness=1, median_score=0.6, first=20, last=38, support=6),
        ]
        scores = np.stack([np.where(o == 1.0, 0.6, 0.1), np.where(o == 5.0, 0.6, 0.1)], axis=1)
        path = np.array([0] * 9 + [1] + [0] + [1] * 9)  # window 9 flips without support at level 1
        runs = runs_of(path, scores, _track(t, o, s), levels)
        assert [(r.level, r.i0, r.i1) for r in runs] == [(0, 0, 10), (1, 11, 19)]


class TestChangepoint:
    def test_finds_the_switch(self) -> None:
        t = np.arange(100.0)
        a = np.where(t < 37, 0.6, 0.1)
        b = np.where(t < 37, 0.1, 0.6)
        est, lo, hi, margin = changepoint(t, a, b)
        assert abs(est - 36.5) <= 0.5
        assert lo <= est <= hi
        assert margin > 10

    def test_flat_evidence_gives_wide_range(self) -> None:
        t = np.arange(50.0)
        _, lo, hi, _ = changepoint(t, np.full(50, 0.2), np.full(50, 0.2))
        assert hi - lo > 40


def test_changepoint_with_drop_longer_than_range() -> None:
    t = np.arange(0.0, 60.0)
    est, lo, hi, _ = changepoint(t, np.full(60, 0.3), np.full(60, 0.3), drop=300.0)
    assert est == t[0] and lo <= hi


def test_short_target_is_an_analysis_error() -> None:
    from dubweld.errors import AnalysisError

    short = compute_features(music_and_effects(5, SR))
    with pytest.raises(AnalysisError):
        offset_track(Matcher(short), compute_features(music_and_effects(30, SR)))


def test_unmatched_files_raise() -> None:
    from dubweld.errors import AnalysisError

    a = compute_features(music_and_effects(60, SR, seed=11))
    b = compute_features(music_and_effects(60, SR, seed=12))
    with pytest.raises(AnalysisError):
        choose_ratio(Matcher(a), b, None, None)


class TestVerifyClusters:
    # A dub locked onto a dialogue cluster: 4 windows in two pairs 2 minutes apart showed the M&E at +179 ms
    T = np.array([580.0, 584.0, 696.0, 700.0])

    def test_sparse_early_dub_is_out_of_sync(self) -> None:
        r = np.array([0.178, 0.179, 0.179, 0.181])
        found = _crisp_clusters(self.T, r, np.full(4, 0.55), np.ones(4, dtype=bool))
        assert [(round(med, 3), n) for _, _, med, n in found] == [(0.179, 4)]

    def test_sparse_dialogue_band_agreement_is_not(self) -> None:
        r = np.array([-0.230, -0.229, -0.231, -0.230])
        assert _crisp_clusters(self.T, r, np.full(4, 0.55), np.ones(4, dtype=bool)) == []
