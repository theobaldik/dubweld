"""Synthetic releases: a music-and-effects bed shared by both, different "dialogue", edits and a speed change."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from dubweld.ratio import NAMED

SR = 48000
PAL = float(NAMED["pal"])

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
needs_mkvmerge = pytest.mark.skipif(shutil.which("mkvmerge") is None, reason="mkvmerge not installed")

# Containers the real-data tests look for, in order of preference when several files share a name
MEDIA_EXTENSIONS = (".mkv", ".mp4", ".m4v", ".mov", ".webm", ".avi", ".ts", ".m2ts", ".mka", ".m4a")


def find_media(directory: Path, stem: str) -> Path | None:
    """The media file named `stem` with any container extension (case-insensitive) in `directory`, or None."""
    if not directory.is_dir():
        return None
    found = {p.suffix.lower(): p for p in directory.iterdir() if p.is_file() and p.stem == stem}
    return next((found[ext] for ext in MEDIA_EXTENSIONS if ext in found), None)


@dataclass
class RealRelease:
    """A real target/donor pair (`target<id>.*`, `donor<id>.*`) and an optional hand-made `synced<id>.*`."""

    id: str
    target: Path
    donor: Path
    synced: Path | None


def find_releases(directory: Path) -> list[RealRelease]:
    """Every `target<id>.*` in `directory` that has a `donor<id>.*`, by id (any container extension, any id)."""
    if not directory.is_dir():
        return []
    ids = sorted(
        {
            p.stem[len("target") :]
            for p in directory.iterdir()
            if p.is_file() and p.stem.startswith("target") and p.suffix.lower() in MEDIA_EXTENSIONS
        }
    )
    out: list[RealRelease] = []
    for i in ids:
        target, donor = find_media(directory, f"target{i}"), find_media(directory, f"donor{i}")
        if target is not None and donor is not None:
            out.append(RealRelease(id=i, target=target, donor=donor, synced=find_media(directory, f"synced{i}")))
    return out


def music_and_effects(duration: float, sr: int = SR, seed: int = 1) -> np.ndarray:
    """Notes, chords and percussive hits with distinct band-energy patterns every few hundred ms."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    out = np.zeros(n, dtype=np.float64)
    t_note = 0.0
    while t_note < duration:
        length = rng.uniform(0.15, 0.9)
        freq = 110 * 2 ** (rng.integers(0, 40) / 12)
        i0, i1 = int(t_note * sr), min(n, int((t_note + length) * sr))
        tt = np.arange(i1 - i0) / sr
        env = np.exp(-tt * rng.uniform(2, 8)) * np.minimum(1, tt / 0.01)
        tone = sum(np.sin(2 * np.pi * freq * h * tt + rng.uniform(0, 6)) / h for h in (1, 2, 3, 5))
        out[i0:i1] += 0.12 * env * tone
        t_note += rng.uniform(0.08, 0.45)
    hits = np.sort(rng.uniform(0, duration, int(duration * 3)))
    for h in hits:
        i0 = int(h * sr)
        m = min(n - i0, int(0.25 * sr))
        noise = rng.standard_normal(m)
        spec = np.fft.rfft(noise)
        f = np.fft.rfftfreq(m, 1 / sr)
        center = rng.uniform(200, 9000)
        spec *= np.exp(-0.5 * (np.log(f + 1) - np.log(center)) ** 2 / 0.3)
        burst = np.fft.irfft(spec, m)
        burst *= np.exp(-np.arange(m) / sr * rng.uniform(10, 40))
        out[i0 : i0 + m] += 0.6 * burst / (np.abs(burst).max() + 1e-9)
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def dialogue(duration: float, sr: int = SR, seed: int = 2) -> np.ndarray:
    """Speech-like band-limited noise in phrases with a ~4 Hz syllable envelope."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1 / sr)
    spec[(f < 250) | (f > 3500)] = 0
    voiced = np.fft.irfft(spec, n)
    # Syllables of random length: a periodic envelope would correlate between the two dubs and bias the match
    env = np.zeros(n)
    pos = 0
    while pos < n:
        length = int(rng.uniform(0.08, 0.35) * sr)
        env[pos : pos + length] = np.sin(np.linspace(0, np.pi, min(length, n - pos))) ** 2
        pos += length + int(rng.uniform(0.02, 0.15) * sr)
    phrases = np.zeros(n)
    pos = 0.0
    while pos < duration:
        length = rng.uniform(1.0, 4.0)
        phrases[int(pos * sr) : int(min(duration, pos + length) * sr)] = 1
        pos += length + rng.uniform(0.5, 3.0)
    out = voiced * env * phrases
    return (out / (np.abs(out).max() + 1e-9) * 0.25).astype(np.float32)


@dataclass
class Release:
    """A generated pair: target A, donor B, and the true segments as (a_start, a_end, offset)."""

    a: Path
    b: Path
    ratio: float
    truth: list[tuple[float, float, float]]
    cuts_b: list[float] = field(default_factory=list)
    silent: list[tuple[float, float]] = field(default_factory=list)


def _write_audio(path: Path, pcm: np.ndarray, sr: int, args: list[str], video: Path | None = None) -> None:
    raw = path.with_suffix(".f32")
    np.asarray(pcm, dtype="<f4").tofile(raw)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if video is not None:
        cmd += ["-i", str(video)]
    cmd += ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", str(raw)]
    if video is not None:
        cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy"]
    cmd += [*args, str(path)]
    subprocess.run(cmd, check=True)
    raw.unlink()


def _write_video(path: Path, frames: np.ndarray, fps: Fraction) -> None:
    h, w = frames.shape[1:]
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps.numerator}/{fps.denominator}",
        "-i",
        "-",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "gray",
        str(path),
    ]
    subprocess.run(cmd, input=frames.astype(np.uint8).tobytes(), check=True)


def _scene(content_time: np.ndarray, seed: int = 7) -> np.ndarray:
    """A frame (36x64) for each content time: smooth noise that drifts, changing scene every 6 s."""
    rng = np.random.default_rng(seed)
    keys = rng.uniform(40, 220, size=(400, 9, 16))
    out = np.empty((len(content_time), 36, 64))
    for i, ct in enumerate(content_time):
        k = int(ct // 2)
        frac = (ct / 2) - k
        small = keys[k % 400] * (1 - frac) + keys[(k + 1) % 400] * frac
        if int(ct // 6) % 2:
            small = small[:, ::-1]
        out[i] = np.kron(small, np.ones((4, 4)))
    return out


def make_release(
    tmp: Path,
    pieces: list[tuple[float, float]],
    a_duration: float,
    ratio: float = PAL,
    extra: dict[int, float] | None = None,
    silence: list[tuple[float, float]] | None = None,
    video: bool = False,
    seed: int = 1,
) -> Release:
    """
    Build A (FLAC in MKV) and B (AAC 44.1 kHz in MP4, `ratio` times faster).

    :param pieces: A-time ranges of the shared bed that B contains, in order
    :param extra: B-only material (seconds) inserted before piece index -> length (offset goes down after it)
    :param silence: A-time ranges silenced in both releases (a cut inside one has no audio signal)
    :param video: also give both files a tiny video with black frames at B's cuts
    """
    extra = extra or {}
    bed = music_and_effects(a_duration, SR, seed)
    a_pcm = bed + dialogue(a_duration, SR, seed + 10)
    for s0, s1 in silence or []:
        a_pcm[int(s0 * SR) : int(s1 * SR)] = 0
    foreign = music_and_effects(60.0, SR, seed + 99)

    parts: list[np.ndarray] = []
    truth: list[tuple[float, float, float]] = []
    cuts_b: list[float] = []
    stretched = 0.0  # B time on A's clock
    content: list[tuple[float, float, float]] = []  # (stretched start, stretched end, A time at start or nan)
    for j, (a0, a1) in enumerate(pieces):
        if j in extra:
            n_extra = int(extra[j] * SR)
            parts.append(foreign[:n_extra])
            content.append((stretched, stretched + extra[j], float("nan")))
            stretched += extra[j]
        if j > 0:
            cuts_b.append(stretched)
        parts.append(bed[int(a0 * SR) : int(a1 * SR)])
        truth.append((a0, a1, a0 - stretched))
        content.append((stretched, stretched + a1 - a0, a0))
        stretched += a1 - a0
    b_bed = np.concatenate(parts)
    b_slow = b_bed + dialogue(len(b_bed) / SR, SR, seed + 20)
    for x0, x1 in silence or []:
        for s0, s1, a0 in content:
            if np.isfinite(a0):
                lo, hi = max(s0, s0 + x0 - a0), min(s1, s0 + x1 - a0)
                if hi > lo:
                    b_slow[int(lo * SR) : int(hi * SR)] = 0
    # EQ tilt and level change between the mixes
    spec = np.fft.rfft(b_slow)
    f = np.fft.rfftfreq(len(b_slow), 1 / SR)
    spec *= 0.7 * (1 + 0.5 * np.exp(-f / 2000))
    b_slow = np.fft.irfft(spec, len(b_slow)).astype(np.float32)
    # Speed up: B plays `ratio` times faster (B raw time = stretched / ratio), pitch rises like a PAL transfer
    sr_b = 44100
    n_b = int(len(b_slow) / SR / ratio * sr_b)
    t_b = np.arange(n_b) / sr_b * ratio
    b_pcm = np.interp(t_b, np.arange(len(b_slow)) / SR, b_slow).astype(np.float32)

    a_path, b_path = tmp / "target.mkv", tmp / "donor.mp4"
    a_video = b_video = None
    if video:
        fps_a, fps_b = Fraction(24000, 1001), Fraction(25)
        t_a = np.arange(int(a_duration * fps_a)) / float(fps_a)
        frames_a = _scene(t_a, seed)
        for s0, s1 in silence or []:
            frames_a[(t_a >= s0) & (t_a < s1)] = 4
        # Target's act breaks: black over the material B lacks
        for (p0, p1), (n0, n1) in zip(pieces, pieces[1:]):
            frames_a[(t_a >= p1 - 0.3) & (t_a < n0 + 0.3)] = 4
        st = np.arange(int(len(b_pcm) / sr_b * fps_b)) / float(fps_b) * ratio
        ct = np.full(len(st), np.nan)
        for s0, s1, a0 in content:
            sel = (st >= s0) & (st < s1)
            ct[sel] = a0 + st[sel] - s0 if np.isfinite(a0) else 1000 + st[sel]
        frames_b = _scene(ct, seed)
        for c in cuts_b:
            frames_b[np.abs(st - c) < 0.3] = 4
        a_video, b_video = tmp / "target-video.mkv", tmp / "donor-video.mkv"
        _write_video(a_video, frames_a, fps_a)
        _write_video(b_video, frames_b, fps_b)
    if video:
        b_path = tmp / "donor.mkv"  # MP4 cannot hold FFV1
    _write_audio(a_path, a_pcm, SR, ["-c:a", "flac"], a_video)
    _write_audio(b_path, b_pcm, sr_b, ["-c:a", "aac", "-b:a", "160k"], b_video)
    return Release(a=a_path, b=b_path, ratio=ratio, truth=truth, cuts_b=cuts_b, silent=list(silence or []))


@pytest.fixture(scope="session")
def act_breaks(tmp_path_factory: pytest.TempPathFactory) -> Release:
    """PAL donor, three cuts: a 2.5 s gap, a 15 s jump and a 15 s segment."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    tmp = tmp_path_factory.mktemp("act_breaks")
    pieces = [(0.0, 70.0), (72.5, 130.0), (145.0, 160.0), (163.0, 230.0)]
    return make_release(tmp, pieces, a_duration=235.0)
