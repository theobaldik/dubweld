"""Probe media files with ffprobe and pick the streams dubweld works on."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from dubweld.errors import ProbeError
from dubweld.ffmpeg import Tools, run

log = logging.getLogger(__name__)


@dataclass
class AudioStream:
    """An audio stream; `audio_index` is its position among the file's audio streams (ffmpeg's `0:a:N`)."""

    index: int
    audio_index: int
    codec: str
    sample_rate: int
    channels: int
    start_time: float
    language: str = "und"
    title: str = ""
    default: bool = False


@dataclass
class VideoStream:
    """A real video stream (attached cover pictures are skipped); `fps` is the exact frame rate."""

    index: int
    video_index: int
    codec: str
    fps: Fraction
    start_time: float
    width: int = 0
    height: int = 0


@dataclass
class MediaInfo:
    """The probed streams of one file."""

    path: Path
    duration: float
    size: int
    mtime: float
    audio: list[AudioStream] = field(default_factory=list)
    video: VideoStream | None = None

    def audio_stream(self, audio_index: int) -> AudioStream:
        """
        The audio stream at position `audio_index` among the audio streams.

        :raises ProbeError: when there is no such stream
        """
        for stream in self.audio:
            if stream.audio_index == audio_index:
                return stream
        raise ProbeError(f"{self.path.name} has no audio stream a:{audio_index} ({len(self.audio)} audio streams)")

    @property
    def signature(self) -> dict[str, float | int]:
        """Size and mtime, the notion of "this file changed" stored in plans."""
        return {"size": self.size, "mtime": self.mtime}


def _fraction(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "0"):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def parse_probe(path: Path, data: dict, size: int = 0, mtime: float = 0.0) -> MediaInfo:
    """Build a `MediaInfo` from ffprobe's `-show_streams -show_format` JSON."""
    fmt = data.get("format", {})
    info = MediaInfo(path=path, duration=_float(fmt.get("duration")), size=size, mtime=mtime)
    audio_index = 0
    video_index = 0
    for s in data.get("streams", []):
        kind = s.get("codec_type")
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        disposition = s.get("disposition") or {}
        if kind == "audio":
            info.audio.append(
                AudioStream(
                    index=int(s["index"]),
                    audio_index=audio_index,
                    codec=s.get("codec_name", "?"),
                    sample_rate=int(_float(s.get("sample_rate"), 0)),
                    channels=int(s.get("channels") or 0),
                    start_time=_float(s.get("start_time")),
                    language=tags.get("language", "und"),
                    title=tags.get("title", ""),
                    default=bool(disposition.get("default")),
                )
            )
            audio_index += 1
        elif kind == "video":
            this_index = video_index
            video_index += 1
            # A cover image (PNG/JPEG in MP4/MKV) shows up as a video stream too
            if disposition.get("attached_pic") or info.video is not None:
                continue
            fps = _fraction(s.get("r_frame_rate")) or _fraction(s.get("avg_frame_rate"))
            if fps is None or fps > 1000:
                continue
            info.video = VideoStream(
                index=int(s["index"]),
                video_index=this_index,
                codec=s.get("codec_name", "?"),
                fps=fps,
                start_time=_float(s.get("start_time")),
                width=int(s.get("width") or 0),
                height=int(s.get("height") or 0),
            )
    if info.duration <= 0:
        durations = [_float(s.get("duration")) for s in data.get("streams", [])]
        info.duration = max(durations, default=0.0)
    return info


def probe(path: Path, tools: Tools | None = None) -> MediaInfo:
    """
    Probe `path` with ffprobe.

    :raises ProbeError: when the file is missing, unreadable or has no audio
    """
    tools = tools or Tools()
    path = Path(path)
    if not path.is_file():
        raise ProbeError(f"{path} does not exist")
    out = run([tools.ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        raise ProbeError(f"ffprobe returned garbage for {path}: {e}") from e
    st = path.stat()
    info = parse_probe(path, data, size=st.st_size, mtime=st.st_mtime)
    if not info.audio:
        raise ProbeError(f"{path.name} has no audio stream")
    return info
