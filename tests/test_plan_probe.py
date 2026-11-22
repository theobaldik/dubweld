"""Plan files, ffprobe parsing, video helpers and cut evidence."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from conftest import find_media, find_releases

from dubweld.cuts import silence_runs
from dubweld.errors import PlanError, ProbeError
from dubweld.features import FRAME_RATE, T0, Features
from dubweld.plan import Cut, Evidence, InputRef, Plan, Segment, mmss
from dubweld.probe import parse_probe
from dubweld.video import Clip, black_runs, match_offset, scene_cuts


def make_plan() -> Plan:
    plan = Plan(
        a=InputRef(path="target.mkv", audio_index=0, duration=1299.359, size=1, mtime=1700000000.123456),
        b=InputRef(path="donor.m4v", audio_index=0, duration=1235.58),
        ratio=25 / (24000 / 1001),
        ratio_name="pal",
        ratio_method="snapped",
        segments=[
            Segment(1.547, 276.95, 1.547),
            Segment(279.41, 697.91, 4.011),
            Segment(700.18, 1250.38, 6.276),
            Segment(1252.74, 1296.92, 8.641),
        ],
        cuts=[Cut(276.95, 275.40, 1.547, 4.011, "high", [Evidence("black", 276.95, "fade")], (270.0, 280.0))],
        windows={"times": [1.0, 3.0], "offsets": [1.5, 1.5], "scores": [0.7, 0.2]},
    )
    plan.normalize()
    return plan


class TestPlan:
    def test_round_trip(self, tmp_path: Path) -> None:
        plan = make_plan()
        plan.save(tmp_path / "synced.plan.json")
        loaded = Plan.load(tmp_path / "synced.plan.json")
        assert loaded.ratio == plan.ratio
        assert loaded.a.mtime == plan.a.mtime
        assert [(s.a_start, s.a_end, s.offset) for s in loaded.segments] == [
            (s.a_start, s.a_end, s.offset) for s in plan.segments
        ]
        assert loaded.cuts[0].evidence[0].kind == "black"
        assert loaded.windows["scores"] == [0.7, 0.2]

    def test_windows_on_single_lines(self, tmp_path: Path) -> None:
        make_plan().save(tmp_path / "synced.plan.json")
        lines = (tmp_path / "synced.plan.json").read_text().splitlines()
        assert any(line.strip().startswith('"times": [1.0,3.0]') for line in lines)

    def test_donor_spans_follow_offsets(self) -> None:
        plan = make_plan()
        seg = plan.segments[1]
        assert seg.b_start == pytest.approx((279.41 - 4.011) / plan.ratio)
        assert seg.b_start * plan.ratio == pytest.approx(275.40, abs=0.01)

    def test_gaps(self) -> None:
        gaps = make_plan().gaps()
        assert [round(g.a_end - g.a_start, 2) for g in gaps] == [1.55, 2.46, 2.27, 2.36, 2.44]

    def test_overlap_is_trimmed(self) -> None:
        plan = make_plan()
        plan.segments[0].a_end = 280.0
        plan.normalize()
        assert plan.segments[0].a_end == plan.segments[1].a_start

    def test_hand_edit_is_picked_up(self, tmp_path: Path) -> None:
        make_plan().save(tmp_path / "synced.plan.json")
        text = (tmp_path / "synced.plan.json").read_text().replace('"offset": 4.011', '"offset": 4.02')
        (tmp_path / "synced.plan.json").write_text(text)
        assert Plan.load(tmp_path / "synced.plan.json").segments[1].offset == 4.02

    def test_rejects_bad_plans(self, tmp_path: Path) -> None:
        (tmp_path / "broken.plan.json").write_text("{nope")
        with pytest.raises(PlanError):
            Plan.load(tmp_path / "broken.plan.json")
        d = make_plan().to_dict()
        d["segments"] = []
        with pytest.raises(PlanError):
            Plan.from_dict(d)
        d = make_plan().to_dict()
        d["version"] = 99
        with pytest.raises(PlanError):
            Plan.from_dict(d)

    def test_describe_mentions_cuts(self) -> None:
        text = make_plan().describe()
        assert "4:36.9" in text and "black" in text and "pal" in text

    @pytest.mark.parametrize(
        ("seconds", "text"), [(0, "0:00.0"), (75.25, "1:15.2"), (3723.04, "1:02:03.0"), (-1.5, "-0:01.5")]
    )
    def test_mmss(self, seconds: float, text: str) -> None:
        assert mmss(seconds) == text


class TestProbe:
    DATA = {
        "format": {"duration": "1235.580227"},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "r_frame_rate": "25/1",
                "start_time": "0.0",
                "disposition": {"attached_pic": 0},
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "44100",
                "channels": 2,
                "start_time": "0.0",
                "tags": {"language": "eng"},
                "disposition": {"default": 1},
            },
            {
                "index": 2,
                "codec_type": "video",
                "codec_name": "png",
                "r_frame_rate": "90000/1",
                "disposition": {"attached_pic": 1},
            },
        ],
    }

    def test_skips_cover_picture(self) -> None:
        info = parse_probe(Path("donor.m4v"), self.DATA)
        assert info.video is not None and info.video.codec == "h264"
        assert info.video.fps == Fraction(25)

    def test_cover_first_still_skipped(self) -> None:
        data = {"format": {}, "streams": [self.DATA["streams"][2], self.DATA["streams"][0], self.DATA["streams"][1]]}
        info = parse_probe(Path("donor.m4v"), data)
        assert info.video is not None and info.video.codec == "h264" and info.video.video_index == 1

    def test_audio(self) -> None:
        info = parse_probe(Path("donor.m4v"), self.DATA)
        a = info.audio_stream(0)
        assert (a.sample_rate, a.channels, a.language, a.default) == (44100, 2, "eng", True)
        with pytest.raises(ProbeError):
            info.audio_stream(1)

    @pytest.mark.parametrize(
        ("codec", "rate", "start", "padding", "skip", "first"),
        [
            ("aac", 48000, "-0.021000", 1024, 1024, 0.0),  # ffmpeg 8 reading AAC in Matroska
            ("aac", 44100, "-0.023000", 1024, 1024, 0.0),
            ("aac", 48000, "0.000000", 1024, 0, 0.0),  # ffmpeg 9, or AAC in MP4
            ("opus", 48000, "-0.007000", 312, 0, -0.007),  # ffmpeg drops Opus priming itself
            ("flac", 48000, "0.000000", 0, 0, 0.0),
        ],
    )
    def test_untrimmed_priming(self, codec: str, rate: int, start: str, padding: int, skip: int, first: float) -> None:
        stream = {"index": 0, "codec_type": "audio", "codec_name": codec, "sample_rate": str(rate), "channels": 2}
        stream |= {"start_time": start, "initial_padding": padding}
        info = parse_probe(Path("donor.mkv"), {"format": {"start_time": start}, "streams": [stream]})
        a = info.audio_stream(0)
        assert a.untrimmed_priming == skip
        assert a.first_sample_time == pytest.approx(first, abs=1e-9)
        assert info.start_time == float(start)


class TestVideo:
    def test_black_runs(self) -> None:
        frames = np.full((100, 576), 120, np.uint8)
        frames[40:55] = 5
        frames[70:72] = 5  # too short
        times = np.arange(100) * 0.04
        runs = black_runs(Clip(frames, times), 0.04)
        assert len(runs) == 1
        assert runs[0].start == pytest.approx(1.6) and runs[0].end == pytest.approx(2.2)

    def test_scene_cut(self) -> None:
        rng = np.random.default_rng(0)
        a, b = rng.integers(30, 220, 576), rng.integers(30, 220, 576)
        frames = np.array([a] * 20 + [b] * 20, dtype=np.uint8)
        cuts = scene_cuts(Clip(frames, np.arange(40) * 0.04))
        assert cuts == [pytest.approx(0.8)]

    def test_match_offset(self) -> None:
        rng = np.random.default_rng(1)
        content = rng.integers(30, 220, (300, 576))
        target = Clip(content.astype(np.uint8), np.arange(300) / 24)
        donor = Clip(content[60:200].astype(np.uint8), np.arange(140) / 24 + 1.0)  # shows target time 2.5 s at 1.0 s
        found = match_offset(donor, target, center=1.3, radius=1.0, step=1 / 24)
        assert found is not None
        assert found[0] == pytest.approx(1.5, abs=0.02)
        assert found[1] > 0.99


def test_silence_runs() -> None:
    env = np.full(1000, -20.0, np.float32)
    env[400:450] = -70
    f = Features(bands=np.zeros((32, 1000), np.float32), env_db=env)
    runs = silence_runs(f, 0, 1000 / FRAME_RATE)
    assert len(runs) == 1
    start, end, level = runs[0]
    assert start == pytest.approx(T0 + 8.0) and end == pytest.approx(T0 + 9.0) and level == -70


class TestFindMedia:
    def test_any_container_extension(self, tmp_path: Path) -> None:
        for name in ("target.MP4", "donor.mkv", "donor.plan.json", "target-2.m4v", "synced.txt"):
            (tmp_path / name).touch()
        assert find_media(tmp_path, "target") == tmp_path / "target.MP4"
        assert find_media(tmp_path, "donor") == tmp_path / "donor.mkv"
        assert find_media(tmp_path, "target-2") == tmp_path / "target-2.m4v"
        assert find_media(tmp_path, "synced") is None
        assert find_media(tmp_path / "missing", "target") is None

    def test_prefers_matroska_when_several(self, tmp_path: Path) -> None:
        for name in ("donor.mp4", "donor.mkv"):
            (tmp_path / name).touch()
        assert find_media(tmp_path, "donor") == tmp_path / "donor.mkv"

    def test_releases_pair_by_id(self, tmp_path: Path) -> None:
        names = ["target.mkv", "donor.m4v", "synced.mkv", "target1.mp4", "donor1.mkv", "target-2.mkv", "donor-2.mkv"]
        names += ["synced-2.mka", "target3.mkv", "donor4.mkv", "target.plan.json"]
        for name in names:
            (tmp_path / name).touch()
        found = {r.id: (r.target.name, r.donor.name, r.synced and r.synced.name) for r in find_releases(tmp_path)}
        assert found == {
            "": ("target.mkv", "donor.m4v", "synced.mkv"),
            "1": ("target1.mp4", "donor1.mkv", None),
            "-2": ("target-2.mkv", "donor-2.mkv", "synced-2.mka"),
        }
