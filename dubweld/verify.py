"""Check a rendered dub against the target: it should match at offset 0 wherever it has signal."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dubweld.analyze import load_features
from dubweld.errors import ProbeError
from dubweld.features import FRAME_RATE, Features
from dubweld.ffmpeg import Tools
from dubweld.plan import mmss
from dubweld.probe import probe
from dubweld.xcorr import Matcher, parabolic, offset_track

log = logging.getLogger(__name__)

WINDOW = 8.0
HOP = 4.0
LOCAL_REACH = 1.0
NEAR_ZERO = 0.040
MATCH = 0.3
CONFIDENT = 0.5
CLUSTER_SCORE = 0.4
CLUSTER_TOLERANCE = 0.010
CLUSTER_SPAN = 90.0
# Outside the dialogue band nothing but a wrong offset makes windows agree, so sparser agreement counts there
CLUSTER_SPAN_WIDE = 150.0
# Dubbed dialogue lines up loosely with the original, reading as the dub 0.1-0.55 s late; it wanders, the M&E doesn't
DIALOGUE = (-0.550, -0.100)
ACTIVE_DB = -60.0
UNCOVERED = 60.0
MAX_BIAS_MS = 10.0


@dataclass
class VerifyReport:
    """
    How well a rendered dub matches the target at offset 0, judged on 8 s windows every 4 s.

    Only the shared music and effects can be checked: stretches of dialogue match loosely or not at all, so read
    `uncovered` and listen there. Times are target seconds; residuals are seconds, negative where the dub is late.

    :param verdict: `pass`, `warn` or `fail`; a fail means a consistent non-zero offset somewhere (`offsync`), while
        such a cluster in the dialogue band where the dub matches nearby (`doubtful`) only warns
    :param median_ms: median absolute residual of the confident windows, milliseconds (NaN when none matched)
    :param max_ms: largest absolute residual of the confident windows, milliseconds
    :param bias_ms: median signed residual, milliseconds: a constant delay of the whole dub
    :param confident: windows matching at offset 0 with a high score
    :param matched: windows matching at offset 0 (within ±40 ms)
    :param active: windows with enough sound in the dub to check
    :param windows: all windows
    :param coverage: `matched` / `active`
    :param uncovered: `(start, end)` stretches over a minute with nothing matching, to listen to
    :param offsync: `(start, end, residual, windows)` of each consistently off stretch
    :param outliers: `(time, residual, score)` of isolated confident windows off by more than 40 ms, usually noise
    :param doubtful: `(start, end, residual, windows)` like `offsync`, but probably dialogue
    :param dialogue: windows showing the loose dialogue match (the dub 0.1-0.55 s late), ignored
    :param messages: explanations of the verdict
    :param times: center of every window
    :param residuals: residual of every window
    :param scores: normalized correlation of every window (1 = identical)
    """

    verdict: str
    median_ms: float
    max_ms: float
    bias_ms: float
    confident: int
    matched: int
    active: int
    windows: int
    coverage: float
    uncovered: list[tuple[float, float]] = field(default_factory=list)
    offsync: list[tuple[float, float, float, int]] = field(default_factory=list)
    outliers: list[tuple[float, float, float]] = field(default_factory=list)
    doubtful: list[tuple[float, float, float, int]] = field(default_factory=list)
    dialogue: int = 0
    messages: list[str] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def describe(self) -> str:
        """A short text report with target timestamps."""
        lines = [
            f"verification: {self.verdict.upper()}",
            f"  {self.matched} of {self.active} audible windows match at offset 0 (±{NEAR_ZERO * 1000:.0f} ms); "
            f"{self.confident} confident ones: median {self.median_ms:.1f} ms, max {self.max_ms:.1f} ms, "
            f"bias {self.bias_ms:+.1f} ms",
        ]
        for start, end, residual, n in self.offsync:
            lines.append(f"  OUT OF SYNC {mmss(start)} - {mmss(end)}: {n} windows agree on {residual * 1000:+.0f} ms")
        for start, end, residual, n in self.doubtful:
            lines.append(
                f"  probably dialogue {mmss(start)} - {mmss(end)}: {n} windows agree on {residual * 1000:+.0f} ms "
                "where the dub matches nearby (listen)"
            )
        if self.outliers:
            lines.append("  isolated confident windows off by more than 40 ms (probably noise; listen if unsure):")
            lines += [f"    {mmss(t)}  {r * 1000:+.0f} ms (score {s:.2f})" for t, r, s in self.outliers]
        if self.dialogue:
            lines.append(f"  {self.dialogue} windows show the loose dialogue match (dub 0.1-0.55 s late); ignored")
        if self.uncovered:
            lines.append(
                f"  stretches over {UNCOVERED:.0f} s with no match at offset 0 (nothing to check there; listen):"
            )
            lines += [f"    {mmss(a)} - {mmss(b)} ({b - a:.0f} s)" for a, b in self.uncovered]
        lines += [f"  {m}" for m in self.messages]
        return "\n".join(lines)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def local_residuals(matcher: Matcher, dub: Features, starts: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Best offset within ±1 s of 0 for each window (sub-frame), and its score."""
    reach = int(LOCAL_REACH * FRAME_RATE)
    lags = np.arange(-reach, reach + 1)
    res = np.zeros(len(starts))
    sc = np.full(len(starts), -1.0)
    for i, s in enumerate(starts):
        pos = s + lags
        ok = (pos >= 0) & (pos <= matcher.n - m)
        if ok.sum() < 3:
            continue
        v = np.full(len(lags), -1.0)
        v[ok] = matcher.ncc(dub.bands[:, s : s + m], pos[ok])
        j = int(np.argmax(v))
        delta, peak = (0.0, v[j]) if j in (0, len(v) - 1) else parabolic(v[j - 1], v[j], v[j + 1])
        res[i], sc[i] = (lags[j] + delta) / FRAME_RATE, peak
    return res, sc


def _in_sync_nearby(
    t: np.ndarray, zero: np.ndarray, t0: float, t1: float, pieces: list[tuple[float, float]] | None
) -> bool:
    """
    Whether the dub matches at offset 0 (`zero`) between `t0` and `t1`, or elsewhere in the piece holding them: a
    piece plays continuously at one offset, so a dialogue-band cluster there is most likely the dialogue.
    """
    if np.any(zero & (t > t0) & (t < t1)):
        return True
    piece = next(((a, b) for a, b in pieces or [] if a <= t0 and t1 <= b), None)
    return piece is not None and bool(np.any(zero & (t >= piece[0]) & (t <= piece[1])))


def _crisp_clusters(
    t: np.ndarray, r: np.ndarray, s: np.ndarray, cand: np.ndarray
) -> list[tuple[float, float, float, int]]:
    """
    Groups of >= 3 candidate windows within ±90 s (±150 s outside the dialogue band) agreeing to ±10 ms and
    outnumbering looser neighbors.
    """
    out: list[tuple[float, float, float, int]] = []
    used = np.zeros(len(t), dtype=bool)
    for i in np.where(cand)[0][np.argsort(-s[cand])]:
        if used[i]:
            continue
        dialogue = DIALOGUE[0] <= r[i] <= DIALOGUE[1]
        span = np.abs(t - t[i]) <= (CLUSTER_SPAN if dialogue else CLUSTER_SPAN_WIDE)
        tight = cand & span & (np.abs(r - r[i]) <= CLUSTER_TOLERANCE) & ~used
        loose = cand & span & (np.abs(r - r[i]) <= 0.150)
        if tight.sum() >= 3 and tight.sum() >= 0.5 * loose.sum():
            used |= tight
            out.append((float(t[tight].min()), float(t[tight].max()), float(np.median(r[tight])), int(tight.sum())))
    return sorted(out)


def verify(
    target: Path,
    output: Path,
    target_track: int = 0,
    dub_track: int | None = None,
    tools: Tools | None = None,
    cache_dir: Path | None = None,
    progress: Callable[[float], None] | None = None,
    pieces: list[tuple[float, float]] | None = None,
) -> VerifyReport:
    """
    Check a rendered dub against the target: synced right, it matches at offset 0 wherever it has music and effects.

    :param target: the file the dub was synced to
    :param output: the file holding the dub, usually `render`'s output
    :param target_track: the target's audio stream to compare with, counted among its audio streams
    :param dub_track: the audio stream of `output` holding the dub; None takes its last audio stream
    :param tools: paths of the external executables; found on PATH when None
    :param cache_dir: where the target's features are cached; None disables the cache
    :param progress: called with the fraction done, 0 to 1
    :param pieces: target-time spans (seconds) each played at one offset, `(target_start, target_end)` of the plan's
        segments; with them, a dialogue-like cluster in a segment that matches elsewhere only warns
    :return: the report; print `describe()` for a summary
    :raises ProbeError: when a file can't be read, lacks the stream, or both name the same stream of the same file
    :raises ToolError: when ffmpeg or ffprobe is missing or fails
    """
    tools = tools or Tools()
    a_info, o_info = probe(Path(target), tools), probe(Path(output), tools)
    if dub_track is None:
        dub_track = len(o_info.audio) - 1
    if Path(target).resolve() == Path(output).resolve() and dub_track == target_track:
        raise ProbeError("comparing a track with itself")
    fa = load_features(a_info, target_track, cache_dir, tools)
    fo = load_features(o_info, dub_track, None, tools)
    matcher = Matcher(fa)
    glob = offset_track(matcher, fo, WINDOW, HOP, progress=progress)
    t, starts, m = glob.times, glob.starts, glob.window_frames
    res, sc = local_residuals(matcher, fo, starts, m)

    k = (starts[:, None] + np.arange(0, m, 10)[None, :]).clip(0, fo.n_frames - 1)
    active = (fo.env_db[k] > ACTIVE_DB).mean(axis=1) >= 0.9
    matched = active & (sc >= MATCH) & (np.abs(res) <= NEAR_ZERO)
    confident = matched & (sc >= CONFIDENT)
    dialogue = active & (sc >= MATCH) & (res >= DIALOGUE[0]) & (res <= DIALOGUE[1])

    signed = res[confident if confident.any() else matched] * 1000
    median_ms = float(np.median(np.abs(signed))) if len(signed) else float("nan")
    max_ms = float(np.abs(signed).max()) if len(signed) else float("nan")
    bias_ms = float(np.median(signed)) if len(signed) else float("nan")

    # A consistent non-zero offset is an error; the dialogue match wanders and does not form crisp clusters
    off = active & (sc >= CLUSTER_SCORE) & (np.abs(res) > NEAR_ZERO)
    clusters = _crisp_clusters(t, res, sc, off)
    # Dialogue in one scene can agree with itself for a few windows; next to in-sync matches it's only worth a listen
    doubtful = [
        c for c in clusters if DIALOGUE[0] <= c[2] <= DIALOGUE[1] and _in_sync_nearby(t, matched, c[0], c[1], pieces)
    ]
    offsync = [c for c in clusters if c not in doubtful]
    far = active & (glob.scores >= CONFIDENT) & (np.abs(glob.offsets) > LOCAL_REACH)
    for i0, i1 in _runs(far):
        if i1 - i0 >= 2 and np.ptp(glob.offsets[i0:i1]) <= 0.020:
            offsync.append((float(t[i0]), float(t[i1 - 1]), float(np.median(glob.offsets[i0:i1])), int(i1 - i0)))
    in_cluster = np.zeros(len(t), dtype=bool)
    for start, end, residual, _ in offsync + doubtful:
        in_cluster |= (t >= start) & (t <= end) & (np.abs(res - residual) <= CLUSTER_TOLERANCE)
    outliers = active & (sc >= CONFIDENT) & (np.abs(res) > NEAR_ZERO) & ~dialogue & ~in_cluster

    half = WINDOW / 2
    uncovered = []
    for i0, i1 in _runs(active & ~matched):
        a, b = float(t[i0] - half + HOP / 2), float(t[i1 - 1] + half - HOP / 2)
        if b - a >= UNCOVERED:
            uncovered.append((max(0.0, a), b))

    messages: list[str] = []
    if not matched.any():
        verdict = "fail"
        messages.append("no window matches the target at offset 0")
    elif offsync:
        verdict = "fail"
        messages.append("part of the dub is consistently off; edit the plan's offsets there")
    elif abs(bias_ms) > MAX_BIAS_MS:
        verdict = "warn"
        late = "late" if bias_ms < 0 else "early"
        messages.append(f"the whole dub is about {abs(bias_ms):.0f} ms {late} (a muxing or encoder delay?)")
    elif median_ms > 10 or outliers.any() or doubtful:
        verdict = "warn"
    else:
        verdict = "pass"
    return VerifyReport(
        verdict=verdict,
        median_ms=median_ms,
        max_ms=max_ms,
        bias_ms=bias_ms,
        confident=int(confident.sum()),
        matched=int(matched.sum()),
        active=int(active.sum()),
        windows=len(t),
        coverage=float(matched.sum() / max(1, active.sum())),
        uncovered=uncovered,
        offsync=sorted(offsync),
        doubtful=sorted(doubtful),
        outliers=[(float(t[i]), float(res[i]), float(sc[i])) for i in np.where(outliers)[0]],
        dialogue=int(dialogue.sum()),
        messages=messages,
        times=t.tolist(),
        residuals=res.tolist(),
        scores=sc.tolist(),
    )
