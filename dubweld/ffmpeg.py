"""Thin wrappers around the ffmpeg, ffprobe and mkvmerge executables."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dubweld.errors import ToolError

log = logging.getLogger(__name__)


@dataclass
class Tools:
    """
    The external executables, as paths or names looked up on PATH.

    :param ffmpeg: ffmpeg, for decoding, encoding and muxing
    :param ffprobe: ffprobe, for reading the inputs' streams
    :param mkvmerge: mkvmerge (MKVToolNix), preferred for muxing; ffmpeg muxes when it is missing
    """

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    mkvmerge: str = "mkvmerge"

    def has(self, name: str) -> bool:
        """
        Whether an executable can be found.

        :param name: the field naming it: `ffmpeg`, `ffprobe` or `mkvmerge`
        """
        return shutil.which(getattr(self, name)) is not None


def _tail(data: bytes, lines: int = 8) -> str:
    return "\n".join(data.decode(errors="replace").strip().splitlines()[-lines:])


def run(cmd: list[str], binary: bool = False) -> str | bytes:
    """
    Run `cmd` and return its stdout.

    :raises ToolError: when the executable is missing or exits non-zero
    """
    log.debug("run: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as e:
        raise ToolError(f"{cmd[0]} not found; install it or put it on PATH") from e
    if proc.returncode != 0:
        raise ToolError(f"{Path(cmd[0]).name} failed ({proc.returncode}):\n{_tail(proc.stderr)}")
    return proc.stdout if binary else proc.stdout.decode(errors="replace")


def audio_args(
    path: Path,
    audio_index: int,
    start: float | None = None,
    duration: float | None = None,
) -> list[str]:
    """Input arguments selecting one audio stream, optionally seeking (accurately) before decoding."""
    args: list[str] = []
    if start is not None and start > 0:
        args += ["-ss", f"{start:.6f}"]
    if duration is not None:
        args += ["-t", f"{duration:.6f}"]
    return args + ["-i", str(path), "-map", f"0:a:{audio_index}", "-vn", "-sn", "-dn"]


def decode_pcm(
    path: Path,
    audio_index: int,
    sample_rate: int,
    channels: int | None = 1,
    filters: str | None = None,
    start: float | None = None,
    duration: float | None = None,
    tools: Tools | None = None,
    skip: int = 0,
) -> np.ndarray:
    """
    Decode one audio stream to float32 PCM in memory.

    :param channels: output channel count; None keeps the stream's own
    :param skip: samples (at the stream's own rate) to drop from the start, see `AudioStream.untrimmed_priming`
    :return: shape (samples,) for mono, (samples, channels) otherwise
    """
    tools = tools or Tools()
    cmd = [tools.ffmpeg, "-nostdin", "-v", "error", *audio_args(path, audio_index, start, duration)]
    filters = _with_skip(filters, skip, start)
    if filters:
        cmd += ["-af", filters]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    cmd += ["-ar", str(sample_rate), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    raw = run(cmd, binary=True)
    data = np.frombuffer(raw, dtype="<f4")
    if channels == 1:
        return data.copy()
    n_ch = channels or _channel_count(data, path, audio_index, tools)
    return data[: len(data) // n_ch * n_ch].reshape(-1, n_ch).copy()


def _channel_count(_data: np.ndarray, path: Path, audio_index: int, tools: Tools) -> int:
    out = run(
        [
            tools.ffprobe,
            "-v",
            "error",
            "-select_streams",
            f"a:{audio_index}",
            "-show_entries",
            "stream=channels",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return int(str(out).strip().splitlines()[0])


def _with_skip(filters: str | None, skip: int, start: float | None) -> str | None:
    if skip <= 0:
        return filters
    if start:
        raise ValueError("skipping priming samples only works when decoding from the start")
    # First in the chain, so it counts samples at the stream's own rate
    return f"atrim=start_sample={skip}" + (f",{filters}" if filters else "")


def gray_frames(
    path: Path, video_index: int, width: int = 32, height: int = 18, tools: Tools | None = None
) -> np.ndarray:
    """
    Decode a video stream to tiny grayscale frames, one per source frame.

    :return: uint8 array of shape (frames, height * width)
    """
    tools = tools or Tools()
    cmd = [
        tools.ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:v:{video_index}",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"scale={width}:{height}:flags=area,format=gray",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-",
    ]
    raw = run(cmd, binary=True)
    frame = width * height
    data = np.frombuffer(raw, dtype=np.uint8)
    return data[: len(data) // frame * frame].reshape(-1, frame).copy()


def open_decoder(
    path: Path,
    audio_index: int,
    sample_rate: int,
    channels: int,
    filters: str | None = None,
    start: float | None = None,
    duration: float | None = None,
    tools: Tools | None = None,
    skip: int = 0,
) -> subprocess.Popen:
    """
    Start an ffmpeg process streaming float32 PCM of one audio stream to its stdout.

    :param skip: samples (at the stream's own rate) to drop from the start, see `AudioStream.untrimmed_priming`
    """
    tools = tools or Tools()
    cmd = [tools.ffmpeg, "-nostdin", "-v", "error", *audio_args(path, audio_index, start, duration)]
    filters = _with_skip(filters, skip, start)
    if filters:
        cmd += ["-af", filters]
    cmd += ["-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    log.debug("decode: %s", " ".join(cmd))
    try:
        return _popen(cmd, stdout=subprocess.PIPE)
    except FileNotFoundError as e:
        raise ToolError(f"{cmd[0]} not found; install it or put it on PATH") from e


def open_encoder(
    output: Path,
    sample_rate: int,
    channels: int,
    codec_args: list[str],
    tools: Tools | None = None,
) -> subprocess.Popen:
    """Start an ffmpeg process encoding float32 PCM read from its stdin into `output`."""
    tools = tools or Tools()
    cmd = [
        tools.ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        *codec_args,
        str(output),
    ]
    log.debug("encode: %s", " ".join(cmd))
    try:
        return _popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError as e:
        raise ToolError(f"{cmd[0]} not found; install it or put it on PATH") from e


def _popen(cmd: list[str], **kwargs: object) -> subprocess.Popen:
    # stderr goes to a file: a pipe nobody reads fills up on a damaged input and blocks ffmpeg forever
    err = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stderr=err, **kwargs)  # type: ignore[call-overload]
    except BaseException:
        err.close()
        raise
    proc.dubweld_stderr = err  # type: ignore[attr-defined]
    return proc


def stderr_tail(proc: subprocess.Popen) -> str:
    """The last lines a streaming process wrote to stderr (and close that file)."""
    err = getattr(proc, "dubweld_stderr", None)
    if err is None or err.closed:
        return ""
    err.seek(0)
    text = _tail(err.read())
    err.close()
    return text


def stop(proc: subprocess.Popen) -> None:
    """Kill a streaming process if it still runs, reap it and close its stderr file."""
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    err = getattr(proc, "dubweld_stderr", None)
    if err is not None and not err.closed:
        err.close()


def finish(proc: subprocess.Popen, what: str) -> None:
    """
    Wait for a streaming process and check it exited cleanly.

    :raises ToolError: on a non-zero exit
    """
    proc.communicate()
    tail = stderr_tail(proc)
    if proc.returncode != 0:
        raise ToolError(f"{what} failed ({proc.returncode}):\n{tail}")
