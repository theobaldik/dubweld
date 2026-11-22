"""Tiny grayscale frames: black runs, scene cuts and video offsets between two releases."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dubweld.ffmpeg import Tools, run
from dubweld.probe import MediaInfo

log = logging.getLogger(__name__)

WIDTH = 32
HEIGHT = 18
BLACK_LUMA = 20.0
MIN_BLACK_FRAMES = 3
SCENE_CUT_CORRELATION = 0.5
MATCH_BLOCK = 12


@dataclass
class Clip:
    """Consecutive frames of one video; `times` are on the clock of the file's chosen audio stream (seconds)."""

    frames: np.ndarray
    times: np.ndarray

    def __len__(self) -> int:
        return len(self.times)


@dataclass
class Run:
    """A stretch of time [start, end) in seconds."""

    start: float
    end: float

    @property
    def middle(self) -> float:
        return (self.start + self.end) / 2

    @property
    def length(self) -> float:
        return self.end - self.start


class VideoReader:
    """
    Decodes short ranges of a file's video as tiny gray frames, on the audio clock scaled by `scale`
    (the speed ratio for the donor, 1 for the target).
    """

    def __init__(self, info: MediaInfo, audio_index: int, scale: float = 1.0, tools: Tools | None = None) -> None:
        if info.video is None:
            raise ValueError(f"{info.path.name} has no video stream")
        self.path: Path = info.path
        self.video = info.video
        self.audio_start: float = info.audio_stream(audio_index).start_time
        self.scale: float = scale
        self.tools: Tools = tools or Tools()
        self.duration: float = info.duration

    @property
    def frame_duration(self) -> float:
        return self.scale / float(self.video.fps)

    def clip(self, start: float, end: float) -> Clip:
        """Frames whose (scaled audio-clock) time lies in [start, end)."""
        fps = float(self.video.fps)
        pts0 = max(0.0, start / self.scale + self.audio_start)
        pts1 = min(self.duration, end / self.scale + self.audio_start)
        if pts1 <= pts0:
            return Clip(np.zeros((0, WIDTH * HEIGHT), np.uint8), np.zeros(0))
        # Nothing exists before the stream's first frame; labels must start at the frames ffmpeg really returns
        first = max(0, math.ceil((pts0 - self.video.start_time) * fps - 1e-6))
        seek = self.video.start_time + first / fps
        # Deblocking is invisible at 32x18 and costs ~10 % of the decode
        cmd = [self.tools.ffmpeg, "-nostdin", "-v", "error", "-skip_loop_filter", "all"]
        if seek > 0:
            cmd += ["-ss", f"{seek - 0.5 / fps:.6f}"]
        cmd += ["-t", f"{pts1 - seek:.6f}", "-i", str(self.path), "-map", f"0:v:{self.video.video_index}"]
        cmd += ["-an", "-sn", "-dn", "-vf", f"scale={WIDTH}:{HEIGHT}:flags=area,format=gray"]
        cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-"]
        raw = np.frombuffer(run(cmd, binary=True), dtype=np.uint8)
        size = WIDTH * HEIGHT
        frames = raw[: len(raw) // size * size].reshape(-1, size)
        pts = self.video.start_time + (first + np.arange(len(frames))) / fps
        return Clip(frames=frames.copy(), times=(pts - self.audio_start) * self.scale)


def black_runs(clip: Clip, frame_duration: float) -> list[Run]:
    """Runs of at least `MIN_BLACK_FRAMES` frames with mean luma below `BLACK_LUMA`."""
    if len(clip) == 0:
        return []
    dark = clip.frames.mean(axis=1) < BLACK_LUMA
    edges = np.diff(np.concatenate([[0], dark.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [
        Run(float(clip.times[s]), float(clip.times[e - 1] + frame_duration))
        for s, e in zip(starts, ends)
        if e - s >= MIN_BLACK_FRAMES
    ]


def _normalize(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32)
    f -= f.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    return f / np.where(norm > 1e-3, norm, np.inf)


def scene_cuts(clip: Clip) -> list[float]:
    """Times of hard cuts: consecutive frames whose correlation drops below `SCENE_CUT_CORRELATION`."""
    if len(clip) < 2:
        return []
    f = _normalize(clip.frames)
    corr = np.einsum("ij,ij->i", f[:-1], f[1:])
    bright = (clip.frames.mean(axis=1) >= BLACK_LUMA)[1:] & (clip.frames.mean(axis=1) >= BLACK_LUMA)[:-1]
    return [float(clip.times[i + 1]) for i in np.where((corr < SCENE_CUT_CORRELATION) & bright)[0]]


def match_offset(donor: Clip, target: Clip, center: float, radius: float, step: float) -> tuple[float, float] | None:
    """
    Video offset (target time - donor time) that best matches the clips, searched within center ± radius.

    :return: (offset seconds, mean correlation), or None when the clips are too dark or flat to tell
    """
    if len(donor) < MATCH_BLOCK or len(target) < MATCH_BLOCK:
        return None
    df = _normalize(donor.frames)
    tf = _normalize(target.frames)
    usable = np.isfinite(df).all(axis=1) & (np.abs(df).sum(axis=1) > 0)
    if usable.sum() < MATCH_BLOCK:
        return None
    best: tuple[float, float] | None = None
    for v in np.arange(center - radius, center + radius + step / 2, step):
        idx = np.searchsorted(target.times, donor.times + v)
        idx = np.clip(idx, 1, len(target) - 1)
        prev_closer = np.abs(target.times[idx - 1] - (donor.times + v)) < np.abs(target.times[idx] - (donor.times + v))
        idx = idx - prev_closer
        inside = (donor.times + v >= target.times[0]) & (donor.times + v <= target.times[-1]) & usable
        if inside.sum() < MATCH_BLOCK:
            continue
        corr = float(np.mean(np.einsum("ij,ij->i", df[inside], tf[idx[inside]])))
        if best is None or corr > best[1]:
            best = (float(v), corr)
    return best
