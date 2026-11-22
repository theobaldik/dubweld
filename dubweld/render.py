"""Render the synced dub from a plan (streaming, bounded memory) and mux it into a copy of the target."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dubweld.errors import RenderError, ToolError
from dubweld.ffmpeg import Tools, finish, open_decoder, open_encoder, run, stderr_tail, stop
from dubweld.plan import Plan
from dubweld.probe import probe

log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
BLOCK_SECONDS = 10.0
CODECS: dict[str, tuple[str, str, str]] = {
    # name: (ffmpeg encoder, default bitrate, file suffix)
    # Matroska audio for all: mkvmerge ignores the MP4 edit list that hides AAC priming, adding 21 ms of delay
    "aac": ("aac", "192k", ".mka"),
    "opus": ("libopus", "128k", ".mka"),
    "flac": ("flac", "", ".mka"),
}


@dataclass
class RenderOptions:
    """
    Render and mux settings.

    :param fill: what plays where the donor has nothing: `silence` or `original` (the target's own audio)
    :param pitch: `shift` resamples (undoes a PAL speed-up's pitch rise), `keep` time-stretches (pitch-corrected donor)
    :param default: which audio track is flagged default in the output: `original` or `dub`
    :param crossfade: length of the raised-cosine crossfade at every join, seconds
    """

    language: str = "und"
    name: str = ""
    fill: str = "silence"
    codec: str = "aac"
    bitrate: str = ""
    default: str = "original"
    pitch: str = "shift"
    crossfade: float = 0.030
    force: bool = False


def speed_filter(ratio: float, pitch: str = "shift", sample_rate: int = SAMPLE_RATE, tools: Tools | None = None) -> str:
    """
    ffmpeg filter making donor audio `ratio` times longer at `sample_rate`.

    Resamples to a known rate *first*: `asetrate` alone assumes the input already runs at `sample_rate`.
    """
    if pitch == "keep":
        tempo = 1.0 / ratio
        if tools is None or _has_filter(tools, "rubberband"):
            return f"aresample={sample_rate},rubberband=tempo={tempo:.10f}"
        return f"aresample={sample_rate},atempo={tempo:.10f}"
    return f"aresample={sample_rate},asetrate={sample_rate / ratio:.6f},aresample={sample_rate}"


def _has_filter(tools: Tools, name: str) -> bool:
    try:
        out = run([tools.ffmpeg, "-hide_banner", "-filters"])
    except ToolError:
        return False
    return any(line.split()[1:2] == [name] for line in str(out).splitlines() if line.strip())


class _Tape:
    """
    One decoder read from the start, addressed by sample index, with a rolling buffer behind the read head.

    Never seeks: ffmpeg's input seeking is not sample-accurate for every codec (AAC in MP4 lands ~21 ms off).
    Going back past the buffer, as a scene moved earlier needs, restarts the decoder from the start.
    """

    KEEP = 5 * SAMPLE_RATE

    def __init__(self, opener: Callable[[], subprocess.Popen], channels: int, what: str) -> None:
        self._opener = opener
        self.channels = channels
        self.what = what
        self.proc: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        self.close()
        self.proc = self._opener()
        self.buf = np.zeros((0, self.channels), dtype=np.float32)
        self.buf_start = 0
        self.ended = False

    @property
    def buf_end(self) -> int:
        return self.buf_start + len(self.buf)

    def read(self, m0: int, m1: int) -> np.ndarray:
        """Samples [m0, m1) of the stream; zeros before its start and after its end."""
        out = np.zeros((m1 - m0, self.channels), dtype=np.float32)
        if m1 <= 0:
            return out
        if max(m0, 0) < self.buf_start:
            log.debug("%s: rewinding to sample %d", self.what, m0)
            self._start()
        self._fill(m1)
        lo, hi = max(m0, self.buf_start), min(m1, self.buf_end)
        if hi > lo:
            out[lo - m0 : hi - m0] = self.buf[lo - self.buf_start : hi - self.buf_start]
        keep_from = max(self.buf_start, m1 - self.KEEP)
        if keep_from > self.buf_start:
            self.buf = self.buf[keep_from - self.buf_start :]
            self.buf_start = keep_from
        return out

    def _fill(self, upto: int) -> None:
        frame = 4 * self.channels
        chunks = [self.buf]
        end = self.buf_end
        while end < upto and not self.ended and self.proc is not None and self.proc.stdout is not None:
            # Bounded reads, so a long skip passes through the trimming below instead of landing in memory
            want = min(max(upto - end, SAMPLE_RATE), self.KEEP)
            raw = self.proc.stdout.read(want * frame)
            got = len(raw) // frame
            if got < want:
                self.ended = True
                self._check()
            if got:
                chunks.append(np.frombuffer(raw[: got * frame], dtype="<f4").reshape(-1, self.channels))
                end += got
            # Drop what nobody can ask for any more, so a long skip does not pile up
            if end - self.KEEP > self.buf_start and end < upto - self.KEEP:
                merged = np.concatenate(chunks)
                drop = len(merged) - self.KEEP
                self.buf_start += drop
                chunks = [merged[drop:]]
        self.buf = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    def _check(self) -> None:
        if self.proc is None:
            return
        self.proc.wait()
        tail = stderr_tail(self.proc)
        if self.proc.returncode != 0:
            raise ToolError(f"decoding {self.what} failed:\n{tail}")

    def close(self) -> None:
        if self.proc is not None:
            stop(self.proc)
            self.proc = None


@dataclass
class _Piece:
    """Output samples [n0, n1) taken from `tape` at sample n - shift."""

    n0: int
    n1: int
    tape: _Tape
    shift: int


def _ramp(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)).astype(np.float32)


def render_dub(
    plan: Plan,
    a_path: Path,
    b_path: Path,
    out_audio: Path,
    options: RenderOptions | None = None,
    tools: Tools | None = None,
    progress: Callable[[float], None] | None = None,
) -> None:
    """
    Encode the synced dub track, as long as the target, into `out_audio`.

    :raises RenderError: when decoding or encoding fails
    """
    options = options or RenderOptions()
    tools = tools or Tools()
    if options.codec not in CODECS:
        raise RenderError(f"unknown codec {options.codec!r} (choose from {', '.join(CODECS)})")
    if options.fill not in ("silence", "original"):
        raise RenderError(f"unknown fill {options.fill!r}")
    b_info = probe(Path(b_path), tools)
    b_stream = b_info.audio_stream(plan.b.audio_index)
    channels = b_stream.channels or 2
    sr = SAMPLE_RATE
    total = int(round(plan.a.duration * sr))
    pad = max(1, int(round(options.crossfade * sr / 2)))
    filt = speed_filter(plan.ratio, options.pitch, sr, tools)

    codec, default_rate, _ = CODECS[options.codec]
    args = ["-c:a", codec]
    if options.bitrate or default_rate:
        args += ["-b:a", options.bitrate or default_rate]
    tapes: list[_Tape] = []
    encoder: subprocess.Popen | None = None
    try:
        # Donor tape sample m is stretched donor time m / sr; target sample n = m + offset * sr
        donor = _Tape(
            lambda: open_decoder(
                b_path, plan.b.audio_index, sr, channels, filt, tools=tools, skip=b_stream.untrimmed_priming
            ),
            channels,
            "the donor",
        )
        tapes.append(donor)
        target: _Tape | None = None
        if options.fill == "original":
            a_skip = probe(Path(a_path), tools).audio_stream(plan.a.audio_index).untrimmed_priming
            target = _Tape(
                lambda: open_decoder(
                    a_path, plan.a.audio_index, sr, channels, f"aresample={sr}", tools=tools, skip=a_skip
                ),
                channels,
                "the target",
            )
            tapes.append(target)
        pieces = _pieces(plan, total, sr, donor, target)
        encoder = open_encoder(out_audio, sr, channels, args, tools)
        _mix(pieces, encoder, total, pad, channels, progress)
    except BrokenPipeError as e:
        tail = stderr_tail(encoder) if encoder is not None else ""
        raise RenderError(f"the encoder stopped:\n{tail}") from e
    except ToolError as e:
        raise RenderError(str(e)) from e
    finally:
        for tape in tapes:
            tape.close()
        if encoder is not None:
            stop(encoder)


def _pieces(plan: Plan, total: int, sr: int, donor: _Tape, target: _Tape | None) -> list[_Piece]:
    pieces: list[_Piece] = []
    cursor = 0
    for seg in sorted(plan.segments, key=lambda s: s.a_start):
        n0, n1 = int(round(seg.a_start * sr)), min(total, int(round(seg.a_end * sr)))
        if target is not None and n0 > cursor:
            pieces.append(_Piece(cursor, n0, target, 0))
        pieces.append(_Piece(n0, n1, donor, int(round(seg.offset * sr))))
        cursor = max(cursor, n1)
    if target is not None and cursor < total:
        pieces.append(_Piece(cursor, total, target, 0))
    return pieces


def _mix(
    pieces: list[_Piece],
    encoder: subprocess.Popen,
    total: int,
    pad: int,
    channels: int,
    progress: Callable[[float], None] | None,
) -> None:
    """Sum the pieces block by block, crossfading at their edges, into the encoder's stdin."""
    block = int(BLOCK_SECONDS * SAMPLE_RATE)
    up, down = _ramp(2 * pad), _ramp(2 * pad)[::-1]
    if encoder.stdin is None:
        raise RenderError("encoder has no input")
    for b0 in range(0, total, block):
        b1 = min(total, b0 + block)
        out = np.zeros((b1 - b0, channels), dtype=np.float32)
        for p in pieces:
            lo, hi = max(b0, p.n0 - pad), min(b1, p.n1 + pad)
            if lo >= hi:
                continue
            data = p.tape.read(lo - p.shift, hi - p.shift)
            idx = np.arange(lo, hi)
            gain = np.ones(hi - lo, dtype=np.float32)
            fade_in = idx < p.n0 + pad
            if p.n0 > 0:
                gain[fade_in] = up[idx[fade_in] - (p.n0 - pad)]
            fade_out = idx >= p.n1 - pad
            if p.n1 < total:
                gain[fade_out] *= down[idx[fade_out] - (p.n1 - pad)]
            out[lo - b0 : hi - b0] += data * gain[:, None]
        encoder.stdin.write(np.clip(out, -1.0, 1.0).astype("<f4").tobytes())
        if progress:
            progress(b1 / total)
    encoder.stdin.close()
    finish(encoder, "encoding the dub")


def _mkv_tracks(path: Path, tools: Tools) -> list[dict]:
    out = run([tools.mkvmerge, "-J", str(path)])
    return json.loads(out).get("tracks", [])


def mux(
    a_path: Path, dub: Path, output: Path, options: RenderOptions, delay: float = 0.0, tools: Tools | None = None
) -> None:
    """
    Copy every track of the target untouched and add the dub (mkvmerge, or ffmpeg when it is missing).

    :param delay: seconds to delay the dub by (the target audio's start time)
    """
    tools = tools or Tools()
    name = options.name or "Dub"
    dub_default = options.default == "dub"
    if tools.has("mkvmerge"):
        cmd = [tools.mkvmerge, "-q", "-o", str(output)]
        if dub_default:
            for t in _mkv_tracks(a_path, tools):
                if t.get("type") == "audio":
                    cmd += ["--default-track-flag", f"{t['id']}:no"]
        cmd += [
            str(a_path),
            "--language",
            f"0:{options.language}",
            "--track-name",
            f"0:{name}",
            "--default-track-flag",
            f"0:{'yes' if dub_default else 'no'}",
        ]
        if abs(delay) > 1e-4:
            cmd += ["--sync", f"0:{round(delay * 1000)}"]
        cmd.append(str(dub))
        try:
            run(cmd)
        except ToolError as e:
            # mkvmerge exits 1 on warnings with a usable file
            if not output.is_file() or "(1)" not in str(e):
                raise
            log.warning("mkvmerge: %s", e)
        return
    log.warning("mkvmerge not found; muxing with ffmpeg")
    a_info = probe(Path(a_path), tools)
    n_audio = len(a_info.audio)
    # -copyts: by default ffmpeg shifts the dub's first packet (AAC priming, at -21 ms) to 0, making it 21 ms late
    cmd = [tools.ffmpeg, "-nostdin", "-v", "error", "-y", "-copyts", "-i", str(a_path)]
    if abs(delay) > 1e-4:
        cmd += ["-itsoffset", f"{delay:.6f}"]
    cmd += [
        "-i",
        str(dub),
        "-map",
        "0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        f"-metadata:s:a:{n_audio}",
        f"language={options.language}",
        f"-metadata:s:a:{n_audio}",
        f"title={name}",
    ]
    for i in range(n_audio):
        cmd += [f"-disposition:a:{i}", "0" if dub_default else ("default" if i == 0 else "0")]
    cmd += [f"-disposition:a:{n_audio}", "default" if dub_default else "0", str(output)]
    run(cmd)


def check_output(output: Path) -> None:
    """
    The output is always Matroska.

    :raises RenderError: for any other file name suffix
    """
    if Path(output).suffix.lower() not in (".mkv", ".mka"):
        raise RenderError(f"{Path(output).name}: the output is Matroska, so name it .mkv (or .mka)")


def render(
    plan: Plan,
    a_path: Path,
    b_path: Path,
    output: Path,
    options: RenderOptions | None = None,
    tools: Tools | None = None,
    progress: Callable[[float], None] | None = None,
    keep_audio: Path | None = None,
) -> Path:
    """
    Render the dub and write `output`: the target with every track copied plus the dub. Video is never re-encoded.

    :param keep_audio: also keep the encoded dub track at this path
    :raises RenderError: on failure; no partial output is left behind
    """
    options = options or RenderOptions()
    tools = tools or Tools()
    output = Path(output)
    check_output(output)
    if output.exists() and not options.force:
        raise RenderError(f"{output} exists (use --force to overwrite)")
    if output.resolve() in (Path(a_path).resolve(), Path(b_path).resolve()):
        raise RenderError("the output must not be one of the inputs")
    plan.validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = CODECS.get(options.codec, ("", "", ".mka"))[2]
    a_start = probe(Path(a_path), tools).audio_stream(plan.a.audio_index).first_sample_time
    with tempfile.TemporaryDirectory(prefix=".dubweld-", dir=output.parent) as tmp:
        dub = Path(tmp) / f"dub{suffix}"
        partial = Path(tmp) / f"out{output.suffix.lower()}"
        render_dub(plan, a_path, b_path, dub, options, tools, progress)
        mux(Path(a_path), dub, partial, options, delay=a_start, tools=tools)
        if keep_audio is not None:
            shutil.copyfile(dub, keep_audio)
        partial.replace(output)
    return output
