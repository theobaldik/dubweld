"""Rendering: speed, placement to the millisecond, crossfades, fill and muxing."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import PAL, needs_ffmpeg, needs_mkvmerge

from dubweld.errors import RenderError
from dubweld.ffmpeg import Tools, decode_pcm
from dubweld.plan import InputRef, Plan, Segment
from dubweld.probe import probe
from dubweld.render import RenderOptions, render, render_dub, speed_filter

SR = 48000


def write(path: Path, pcm: np.ndarray, sr: int, args: list[str]) -> Path:
    raw = path.with_suffix(".raw")
    pcm = np.asarray(pcm, dtype="<f4")
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    pcm.tofile(raw)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(sr),
            "-ac",
            str(channels),
            "-i",
            str(raw),
            *args,
            str(path),
        ],
        check=True,
    )
    raw.unlink()
    return path


def decode(path: Path, audio_index: int, sr: int) -> np.ndarray:
    """Decode a rendered stream like dubweld does, skipping AAC priming that ffmpeg 8 leaves in Matroska."""
    skip = probe(path).audio_stream(audio_index).untrimmed_priming
    return decode_pcm(path, audio_index, sr, skip=skip)


def target_with_video(tmp: Path, duration: float = 12.0) -> Path:
    """A target with video, 5.1 audio and subtitles, like a real release."""
    srt = tmp / "target.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")
    out = tmp / "target.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x36:rate=24000/1001:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=300:sample_rate=48000:d={duration}",
            "-i",
            str(srt),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:s",
            "-c:v",
            "ffv1",
            "-c:a",
            "flac",
            "-ac",
            "6",
            "-c:s",
            "srt",
            "-metadata:s:a:0",
            "language=eng",
            str(out),
        ],
        check=True,
    )
    return out


def clicks(times: list[float], duration: float, sr: int) -> np.ndarray:
    x = np.zeros(int(duration * sr), np.float32)
    for t in times:
        i = int(round(t * sr))
        x[i : i + 8] = 0.9
    return x


def click_times(pcm: np.ndarray, sr: int, threshold: float = 0.3) -> list[float]:
    """Onsets of clicks (first sample above threshold after quiet)."""
    loud = np.abs(pcm) > threshold
    onsets = np.where(loud & ~np.concatenate([[False], loud[:-1]]))[0]
    out: list[float] = []
    for i in onsets:
        if not out or i / sr - out[-1] > 0.1:
            out.append(i / sr)
    return out


def plan_for(a: Path, b: Path, a_duration: float, b_duration: float, segments: list[Segment], ratio: float) -> Plan:
    plan = Plan(a=InputRef(str(a), 0, a_duration), b=InputRef(str(b), 0, b_duration), ratio=ratio, segments=segments)
    plan.normalize()
    return plan


def test_speed_filter_resamples_first() -> None:
    f = speed_filter(PAL)
    assert f.startswith("aresample=48000,asetrate=")
    assert f.endswith(",aresample=48000")
    assert f"{48000 / PAL:.6f}" in f


@needs_ffmpeg
class TestRender:
    def test_44k_donor_is_slowed_not_sped_up(self, tmp_path: Path) -> None:
        # DubSync assumed 48 kHz input: a 44.1 kHz dub came out 4 % faster instead of slower
        sr_b = 44100
        t = np.arange(int(10 * sr_b)) / sr_b
        b = write(tmp_path / "donor.m4a", 0.5 * np.sin(2 * np.pi * 1000 * t), sr_b, ["-c:a", "aac", "-b:a", "192k"])
        a = write(tmp_path / "target.flac", np.zeros(12 * SR), SR, ["-c:a", "flac"])
        plan = plan_for(a, b, 12.0, 10.0, [Segment(0.0, 12.0, 0.0)], PAL)
        out = tmp_path / "synced.mka"
        render_dub(plan, a, b, out, RenderOptions(codec="flac"))
        pcm = decode(out, 0, SR)
        loud = np.where(np.abs(pcm) > 0.1)[0]
        assert (loud[-1] - loud[0]) / SR == pytest.approx(10 * PAL, abs=0.05)
        seg = pcm[SR : 5 * SR]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        peak = np.argmax(spec) * SR / len(seg)
        assert peak == pytest.approx(1000 / PAL, abs=1.0)

    @pytest.mark.parametrize("codec", ["aac", "opus", "flac"])
    @needs_mkvmerge
    def test_clicks_land_to_the_millisecond(self, tmp_path: Path, codec: str) -> None:
        sr_b = 44100
        b_clicks = [1.0, 2.0, 3.0, 6.0, 7.0, 8.0]
        b = write(tmp_path / "donor.m4a", clicks(b_clicks, 10.0, sr_b), sr_b, ["-c:a", "aac", "-b:a", "192k"])
        a = target_with_video(tmp_path)
        # Up-jump: target has 1.5 s more between donor 4.5 s and the rest
        cut = 4.5 * PAL
        segs = [Segment(0.3, cut + 0.3, 0.3), Segment(cut + 1.8, 11.9, 1.8)]
        plan = plan_for(a, b, 12.0, 10.0, segs, PAL)
        out = render(plan, a, b, tmp_path / "synced.mkv", RenderOptions(codec=codec, language="cze", name="Test"))
        pcm = decode(out, 1, SR)
        expected = [t * PAL + (0.3 if t * PAL < cut else 1.8) for t in b_clicks]
        found = click_times(pcm, SR)
        assert len(found) == len(expected)
        assert np.max(np.abs(np.array(found) - expected)) < 0.002

    def test_ffmpeg_mux_fallback(self, tmp_path: Path) -> None:
        b = write(tmp_path / "donor.m4a", clicks([1.0, 2.0], 4.0, SR), SR, ["-c:a", "aac"])
        a = target_with_video(tmp_path, 5.0)
        plan = plan_for(a, b, 5.0, 4.0, [Segment(0.5, 4.5, 0.5)], 1.0)
        tools = Tools(mkvmerge="dubweld-no-such-mkvmerge")
        out = render(plan, a, b, tmp_path / "synced.mkv", RenderOptions(language="cze"), tools=tools)
        found = click_times(decode(out, 1, SR), SR)
        assert np.allclose(found, [1.5, 2.5], atol=0.002)

    def test_crossfade_has_no_click(self, tmp_path: Path) -> None:
        # Down-jump: two adjacent segments from different donor times meet mid-tone
        t = np.arange(10 * SR) / SR
        b = write(tmp_path / "donor.flac", 0.5 * np.sin(2 * np.pi * 220 * t), SR, ["-c:a", "flac"])
        a = write(tmp_path / "target.flac", np.zeros(8 * SR), SR, ["-c:a", "flac"])
        plan = plan_for(a, b, 8.0, 10.0, [Segment(0.0, 4.0, 0.0), Segment(4.0, 8.0, -1.0013)], 1.0)
        out = tmp_path / "synced.mka"
        render_dub(plan, a, b, out, RenderOptions(codec="flac"))
        pcm = decode(out, 0, SR)
        join = pcm[4 * SR - 2400 : 4 * SR + 2400]
        smooth_step = 2 * np.pi * 220 / SR * 0.5
        assert np.max(np.abs(np.diff(join))) < 1.5 * smooth_step
        assert np.min(np.abs(pcm[SR : 7 * SR])) < 0.01  # sanity: it is a tone, not silence

    def test_fill_original(self, tmp_path: Path) -> None:
        b = write(tmp_path / "donor.flac", np.zeros(4 * SR), SR, ["-c:a", "flac"])
        t = np.arange(6 * SR) / SR
        a = write(tmp_path / "target.flac", 0.4 * np.sin(2 * np.pi * 300 * t), SR, ["-c:a", "flac"])
        plan = plan_for(a, b, 6.0, 4.0, [Segment(1.0, 5.0, 1.0)], 1.0)
        out = tmp_path / "synced.mka"
        render_dub(plan, a, b, out, RenderOptions(codec="flac", fill="original"))
        pcm = decode(out, 0, SR)
        rms = lambda x: float(np.sqrt(np.mean(x**2)))  # noqa: E731
        assert rms(pcm[: int(0.9 * SR)]) > 0.2
        assert rms(pcm[int(1.1 * SR) : int(4.9 * SR)]) < 0.01
        assert rms(pcm[int(5.1 * SR) :]) > 0.2

    def test_silence_fill_is_silent(self, tmp_path: Path) -> None:
        t = np.arange(4 * SR) / SR
        b = write(tmp_path / "donor.flac", 0.4 * np.sin(2 * np.pi * 300 * t), SR, ["-c:a", "flac"])
        a = write(tmp_path / "target.flac", np.zeros(6 * SR), SR, ["-c:a", "flac"])
        plan = plan_for(a, b, 6.0, 4.0, [Segment(1.0, 5.0, 1.0)], 1.0)
        out = tmp_path / "synced.mka"
        render_dub(plan, a, b, out, RenderOptions(codec="flac"))
        pcm = decode(out, 0, SR)
        assert len(pcm) == pytest.approx(6 * SR, abs=SR // 100)
        assert np.max(np.abs(pcm[: int(0.98 * SR)])) < 1e-4
        assert np.max(np.abs(pcm[int(5.02 * SR) :])) < 1e-4

    @needs_mkvmerge
    def test_mux_keeps_target_tracks_and_flags(self, tmp_path: Path) -> None:
        b = write(tmp_path / "donor.m4a", clicks([1.0], 3.0, SR), SR, ["-c:a", "aac"])
        a = target_with_video(tmp_path, 4.0)
        plan = plan_for(a, b, 4.0, 3.0, [Segment(0.5, 3.5, 0.5)], 1.0)
        out = render(
            plan, a, b, tmp_path / "synced.mkv", RenderOptions(language="cze", name="Czech dub", default="dub")
        )
        info = json.loads(subprocess.run(["mkvmerge", "-J", str(out)], capture_output=True, check=True).stdout)
        tracks = [
            (t["type"], t["properties"].get("language"), t["properties"].get("default_track")) for t in info["tracks"]
        ]
        assert tracks[0][0] == "video"
        assert ("audio", "eng", False) in tracks
        assert ("audio", "cze", True) in tracks
        assert tracks[-1][0] == "subtitles"
        names = [t["properties"].get("track_name") for t in info["tracks"]]
        assert "Czech dub" in names

    def test_refuses_to_overwrite(self, tmp_path: Path) -> None:
        b = write(tmp_path / "donor.flac", np.zeros(SR), SR, ["-c:a", "flac"])
        a = write(tmp_path / "target.flac", np.zeros(SR), SR, ["-c:a", "flac"])
        plan = plan_for(a, b, 1.0, 1.0, [Segment(0.0, 1.0, 0.0)], 1.0)
        (tmp_path / "synced.mkv").write_bytes(b"x")
        with pytest.raises(RenderError):
            render(plan, a, b, tmp_path / "synced.mkv")
        with pytest.raises(RenderError):
            render(plan, a, b, a, RenderOptions(force=True))
        assert (tmp_path / "synced.mkv").read_bytes() == b"x"


@needs_ffmpeg
def test_tape_rewinds_for_moved_scenes(tmp_path: Path) -> None:
    # The second segment plays donor material from before the first one: the tape must restart, not misplace
    b = write(tmp_path / "donor.flac", clicks([1.0, 7.0], 9.0, SR), SR, ["-c:a", "flac"])
    a = write(tmp_path / "target.flac", np.zeros(10 * SR), SR, ["-c:a", "flac"])
    plan = plan_for(a, b, 10.0, 9.0, [Segment(0.0, 4.0, -5.0), Segment(4.0, 8.0, 4.0)], 1.0)
    out = tmp_path / "synced.mka"
    render_dub(plan, a, b, out, RenderOptions(codec="flac"))
    assert np.allclose(click_times(decode(out, 0, SR), SR), [2.0, 5.0], atol=0.001)


def test_output_must_be_matroska(tmp_path: Path) -> None:
    plan = plan_for(tmp_path / "target.flac", tmp_path / "donor.flac", 1.0, 1.0, [Segment(0.0, 1.0, 0.0)], 1.0)
    with pytest.raises(RenderError, match="Matroska"):
        render(plan, tmp_path / "target.flac", tmp_path / "donor.flac", tmp_path / "synced.mp4")


@needs_ffmpeg
def test_damaged_donor_does_not_hang(tmp_path: Path) -> None:
    # A truncated, partly garbage file makes ffmpeg log an error per frame; a full stderr pipe used to block it
    good = write(tmp_path / "donor-intact.m4a", clicks([1.0], 30.0, SR), SR, ["-c:a", "aac"])
    data = bytearray(good.read_bytes())
    rng = np.random.default_rng(0)
    for i in range(len(data) // 3, len(data), 97):
        data[i] = int(rng.integers(0, 256))
    bad = tmp_path / "donor.m4a"
    bad.write_bytes(bytes(data))
    a = write(tmp_path / "target.flac", np.zeros(31 * SR), SR, ["-c:a", "flac"])
    plan = plan_for(a, bad, 31.0, 30.0, [Segment(0.0, 31.0, 0.0)], 1.0)
    try:
        render_dub(plan, a, bad, tmp_path / "synced.mka", RenderOptions(codec="flac"))
    except RenderError:
        pass
