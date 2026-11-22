"""End to end on generated releases: analyze, render, verify, with a known truth."""

from pathlib import Path

import pytest
from conftest import PAL, Release, make_release, needs_ffmpeg

from dubweld.analyze import AnalyzeOptions, analyze
from dubweld.plan import Plan
from dubweld.render import RenderOptions, render
from dubweld.verify import verify

pytestmark = [needs_ffmpeg, pytest.mark.slow]

OFFSET_TOLERANCE = 0.005
CUT_TOLERANCE = 0.5


def check_plan(plan: Plan, rel: Release) -> None:
    assert plan.ratio == pytest.approx(rel.ratio, abs=1e-9)
    assert len(plan.segments) == len(rel.truth), plan.describe()
    for seg, (a0, a1, off) in zip(plan.segments, rel.truth):
        assert seg.offset == pytest.approx(off, abs=OFFSET_TOLERANCE), plan.describe()
    for cut, (_, a_end, off), b_cut in zip(plan.cuts, rel.truth, rel.cuts_b):
        assert cut.b_time == pytest.approx(b_cut, abs=CUT_TOLERANCE), plan.describe()


@pytest.fixture(scope="module")
def act_plan(act_breaks: Release) -> Plan:
    return analyze(act_breaks.a, act_breaks.b, AnalyzeOptions())


class TestActBreaks:
    """PAL speed-up with a 2.5 s gap, a 15 s jump (DubSync dropped > 12 s) and a 15 s segment (it dropped < 25 s)."""

    def test_plan(self, act_plan: Plan, act_breaks: Release) -> None:
        assert act_plan.ratio_method == "snapped"
        check_plan(act_plan, act_breaks)

    def test_gaps_match_the_extra_material(self, act_plan: Plan) -> None:
        inner = [round(g.a_end - g.a_start, 1) for g in act_plan.gaps(0.5)][: len(act_plan.cuts)]
        assert inner == [2.5, 15.0, 3.0]

    def test_render_and_verify(self, act_plan: Plan, act_breaks: Release, tmp_path: Path) -> None:
        out = render(act_plan, act_breaks.a, act_breaks.b, tmp_path / "synced.mkv", RenderOptions(codec="aac"))
        report = verify(act_breaks.a, out)
        assert report.verdict == "pass", report.describe()
        assert report.median_ms < 5
        assert abs(report.bias_ms) < 3
        assert report.coverage > 0.6

    def test_verify_catches_a_wrong_offset(self, act_plan: Plan, act_breaks: Release, tmp_path: Path) -> None:
        bad = Plan.from_dict(act_plan.to_dict())
        bad.segments[3].offset += 0.2
        bad.normalize()
        out = render(bad, act_breaks.a, act_breaks.b, tmp_path / "synced-wrong.mkv", RenderOptions(codec="flac"))
        report = verify(act_breaks.a, out)
        assert report.verdict == "fail", report.describe()
        assert any(abs(r + 0.2) < 0.01 and start > 160 for start, _, r, _ in report.offsync)


def test_donor_with_extra_material(tmp_path: Path) -> None:
    """The donor has 4 s the target lacks: the offset goes down and that material is dropped."""
    rel = make_release(tmp_path, [(0.0, 80.0), (80.0, 170.0)], a_duration=175.0, extra={1: 4.0}, seed=4)
    plan = analyze(rel.a, rel.b, AnalyzeOptions())
    assert len(plan.segments) == 2, plan.describe()
    assert plan.segments[0].offset == pytest.approx(0.0, abs=OFFSET_TOLERANCE)
    assert plan.segments[1].offset == pytest.approx(-4.0, abs=OFFSET_TOLERANCE)
    assert plan.cuts[0].jump == pytest.approx(-4.0, abs=0.01)
    # The target stays continuous: the first segment ends where the second starts, near 80 s
    assert plan.segments[0].a_end == pytest.approx(80.0, abs=CUT_TOLERANCE), plan.describe()
    assert plan.segments[1].a_start == pytest.approx(plan.segments[0].a_end, abs=0.01)
    out = render(plan, rel.a, rel.b, tmp_path / "synced.mkv", RenderOptions(codec="flac"))
    assert verify(rel.a, out).verdict == "pass"


def test_cut_in_silence(tmp_path: Path) -> None:
    """No audio signal at all around the cut: the cut goes into the silence."""
    silent = (95.0, 106.0)
    rel = make_release(tmp_path, [(0.0, 100.0), (102.5, 200.0)], a_duration=205.0, silence=[silent], seed=5)
    plan = analyze(rel.a, rel.b, AnalyzeOptions())
    assert [round(s.offset, 3) for s in plan.segments] == [pytest.approx(0.0, abs=0.005), pytest.approx(2.5, abs=0.005)]
    cut = plan.cuts[0]
    assert cut.evidence[0].kind == "silence", plan.describe()
    assert silent[0] < cut.a_time < silent[1] - 2.5
    assert cut.confidence in ("medium", "high")


def test_video_pins_the_cut(tmp_path: Path) -> None:
    """Black frames in the donor's video place the cut where the audio alone is vague."""
    rel = make_release(tmp_path, [(0.0, 90.0), (92.4, 180.0)], a_duration=185.0, video=True, seed=6)
    plan = analyze(rel.a, rel.b, AnalyzeOptions(use_video=True))
    check_plan(plan, rel)
    cut = plan.cuts[0]
    assert cut.evidence[0].kind == "black", plan.describe()
    assert cut.confidence == "high"
    assert cut.b_time == pytest.approx(rel.cuts_b[0], abs=0.1)
    for seg in plan.segments:
        assert seg.video_offset == pytest.approx(seg.offset, abs=0.05)


def test_no_speed_change(tmp_path: Path) -> None:
    rel = make_release(tmp_path, [(5.0, 120.0)], a_duration=125.0, ratio=1.0, seed=8)
    plan = analyze(rel.a, rel.b, AnalyzeOptions())
    assert plan.ratio == 1.0
    assert plan.segments[0].offset == pytest.approx(5.0, abs=OFFSET_TOLERANCE)
    assert PAL != plan.ratio


class TestShortTail:
    """A closing jingle of 7 s after a jump: too short for 8 s windows to form a level."""

    def test_plan(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        rel = make_release(tmp_path_factory.mktemp("tail"), [(0, 100), (102.7, 110)], a_duration=113, seed=7)
        plan = analyze(rel.a, rel.b, AnalyzeOptions(cache_dir=None))
        check_plan(plan, rel)
