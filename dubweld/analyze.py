"""Analysis: from two files to a sync plan (ratio, segments, cuts)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dubweld import cache
from dubweld.cuts import localize
from dubweld.errors import AnalysisError, ToolError
from dubweld.features import FRAME_RATE, SAMPLE_RATE, T0, Features, compute_features, stretch
from dubweld.ffmpeg import Tools, decode_pcm
from dubweld.levels import COMPETE_DISTANCE, CRISP, Level, find_levels, prune_levels, runs_of, viterbi
from dubweld.plan import Cut, InputRef, Plan, Segment, mmss
from dubweld.probe import MediaInfo, probe
from dubweld.ratio import choose_ratio
from dubweld.video import VideoReader, match_offset
from dubweld.xcorr import DECIMATION, Matcher, OffsetTrack, offset_track, scores_at_offset

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]
CHANGEPOINT_REACH = 120.0
VIDEO_CLIP = 6.0
VIDEO_MIN_CORRELATION = 0.6
# The donor's unmatched head and tail get a second look with short windows: a closing jingle a few seconds long
# yields too few 8 s windows to form a level
EDGE_WINDOW = 4.0
EDGE_HOP = 1.0
EDGE_MIN = 6.0


@dataclass
class AnalyzeOptions:
    """
    Analysis settings.

    :param a_track: audio stream of the target, counted among its audio streams
    :param b_track: audio stream of the donor (the dub)
    :param ratio: donor-to-target speed ratio, or None to choose automatically
    :param use_video: use the videos to place cuts and settle ambiguous offsets
    :param max_offset: only search offsets within ± this many seconds (faster on long files)
    :param cache_dir: where decoded features are cached; None disables the cache
    """

    a_track: int = 0
    b_track: int = 0
    ratio: float | None = None
    use_video: bool = True
    max_offset: float | None = None
    cache_dir: Path | None = None
    window: float = 8.0
    hop: float = 2.0


def load_features(info: MediaInfo, audio_index: int, cache_dir: Path | None, tools: Tools) -> Features:
    """Decode one audio stream to 16 kHz mono and compute its features (cached)."""

    def compute() -> Features:
        log.info("decoding %s (audio a:%d)", info.path.name, audio_index)
        return compute_features(decode_pcm(info.path, audio_index, SAMPLE_RATE, tools=tools))

    return cache.features(info.path, audio_index, info.size, info.mtime, cache_dir, compute)


def _input_ref(info: MediaInfo, audio_index: int) -> InputRef:
    stream = info.audio_stream(audio_index)
    return InputRef(
        path=str(info.path),
        audio_index=audio_index,
        duration=info.duration,
        size=info.size,
        mtime=info.mtime,
        fps=str(info.video.fps) if info.video else "",
        sample_rate=stream.sample_rate,
        channels=stream.channels,
    )


class _Video:
    """Both videos, for matching frames at a given stretched donor time."""

    def __init__(self, a: VideoReader, b: VideoReader) -> None:
        self.a = a
        self.b = b
        self._memo: dict[tuple[float, float], float | None] = {}

    def offset_at(self, b_time: float, audio_offset: float) -> float | None:
        """Video offset near stretched donor time `b_time`, searched within audio_offset ± 1 s."""
        key = (round(b_time, 1), round(audio_offset, 2))
        if key not in self._memo:
            self._memo[key] = self._offset_at(b_time, audio_offset)
        return self._memo[key]

    def _offset_at(self, b_time: float, audio_offset: float) -> float | None:
        try:
            donor = self.b.clip(b_time, b_time + VIDEO_CLIP)
            target = self.a.clip(b_time + audio_offset - 1.5, b_time + audio_offset + VIDEO_CLIP + 1.5)
        except ToolError as e:
            log.debug("video match at %.1f failed: %s", b_time, e)
            return None
        found = match_offset(donor, target, audio_offset, 1.0, self.a.frame_duration)
        if found is None or found[1] < VIDEO_MIN_CORRELATION:
            return None
        return found[0]

    def offset_of(self, times: list[float], audio_offset: float) -> float | None:
        values = [v for v in (self.offset_at(t - VIDEO_CLIP / 2, audio_offset) for t in times) if v is not None]
        return float(np.median(values)) if values else None


def _level_times(track: OffsetTrack, offset: float, idx: np.ndarray | None = None, n: int = 3) -> list[float]:
    """Centers of the best windows at `offset` (±10 ms), spread over time."""
    sel = np.arange(len(track)) if idx is None else idx
    near = sel[np.abs(track.offsets[sel] - offset) <= 0.010]
    near = near[np.argsort(-track.scores[near])]
    picked: list[float] = []
    for i in near:
        if all(abs(track.times[i] - p) > 30 for p in picked):
            picked.append(float(track.times[i]))
        if len(picked) == n:
            break
    return picked


def _refine_offset(track: OffsetTrack, idx: np.ndarray, level: float) -> tuple[float, int]:
    near = idx[(np.abs(track.offsets[idx] - level) <= 0.010) & (track.scores[idx] >= 0.4)]
    if len(near) == 0:
        return level, 0
    o, w = track.offsets[near], track.scores[near]
    order = np.argsort(o)
    cum = np.cumsum(w[order])
    return float(o[order][np.searchsorted(cum, cum[-1] / 2)]), int(len(near))


def analyze(
    a_path: Path,
    b_path: Path,
    options: AnalyzeOptions | None = None,
    tools: Tools | None = None,
    progress: ProgressCallback | None = None,
) -> Plan:
    """
    Work out how to lay the donor's audio onto the target.

    :param a_path: target file (the video and timeline to keep)
    :param b_path: donor file (has the dub)
    :return: the plan; nothing is written
    :raises AnalysisError: when the files cannot be matched
    """
    options = options or AnalyzeOptions()
    tools = tools or Tools()

    def step(name: str) -> Callable[[float], None] | None:
        return (lambda f: progress(name, f)) if progress else None

    a_info, b_info = probe(Path(a_path), tools), probe(Path(b_path), tools)
    a_ref, b_ref = _input_ref(a_info, options.a_track), _input_ref(b_info, options.b_track)
    if len(b_info.audio) > 1 and options.b_track == 0:
        log.warning("%s has %d audio streams; using a:0 (choose with --b-track)", b_info.path.name, len(b_info.audio))

    fa = load_features(a_info, options.a_track, options.cache_dir, tools)
    fb = load_features(b_info, options.b_track, options.cache_dir, tools)
    matcher = Matcher(fa)

    fps_a = a_info.video.fps if a_info.video else None
    fps_b = b_info.video.fps if b_info.video else None
    choice = choose_ratio(matcher, fb, fps_a, fps_b, forced=options.ratio, progress=step("speed ratio"))
    log.info("speed ratio %.10f (%s, %s)", choice.ratio, choice.name, choice.method)
    donor = stretch(fb, choice.ratio)

    track = offset_track(matcher, donor, options.window, options.hop, options.max_offset, progress=step("offsets"))
    levels = find_levels(track)
    if not levels:
        raise AnalysisError("no offset is confirmed by several windows; the files may not share music and effects")

    video: _Video | None = None
    if options.use_video and a_info.video and b_info.video:
        video = _Video(
            VideoReader(a_info, options.a_track, 1.0, tools),
            VideoReader(b_info, options.b_track, choice.ratio, tools),
        )

    av_consistency = _av_consistency(levels, track, video) if video else None
    kept = prune_levels(levels, av_consistency)
    for lv in levels:
        if lv.dropped:
            log.info("dropping level %+.3f s: %s", lv.offset, lv.dropped)

    level_scores = np.stack([scores_at_offset(matcher, donor, track, lv.offset) for lv in kept], axis=1)
    runs = runs_of(viterbi(level_scores), level_scores, track, kept)

    offsets: list[float] = []
    counts: list[int] = []
    spans: list[tuple[float, float]] = []  # stretched donor seconds each run claims, before cut placement
    level_times: list[list[float]] = []
    for r in runs:
        idx = np.arange(r.i0, r.i1 + 1)
        off, n = _refine_offset(track, idx, kept[r.level].offset)
        offsets.append(off)
        counts.append(n)
        spans.append((float(track.times[r.i0]), float(track.times[r.i1])))
        level_times.append(_level_times(track, off, idx, n=2))

    for at_end in (False, True):
        j = len(runs) - 1 if at_end else 0
        conf = runs[j].confident
        edge_from = track.times[conf[-1]] if at_end else 0.0
        edge_to = donor.duration if at_end else track.times[conf[0]]
        edge = _edge_level(matcher, donor, edge_from, edge_to, offsets[j], at_end, options.max_offset)
        if edge is None:
            continue
        log.info(
            "short segment at the %s: %+.3f s (%d windows of %.0f s)",
            "end" if at_end else "start",
            edge.offset,
            edge.members,
            EDGE_WINDOW,
        )
        levels.append(edge)
        if at_end:
            spans[j] = (spans[j][0], float(edge_from))
            offsets.append(edge.offset)
            counts.append(edge.members)
            spans.append((edge.first, donor.duration))
            level_times.append([edge.first, edge.last])
        else:
            spans[0] = (float(edge_to), spans[0][1])
            offsets.insert(0, edge.offset)
            counts.insert(0, edge.members)
            spans.insert(0, (0.0, edge.last))
            level_times.insert(0, [edge.first, edge.last])

    video_offsets: list[float | None] = []
    for times, off in zip(level_times, offsets):
        video_offsets.append(video.offset_of(times, off) if video else None)

    cuts: list[Cut] = []
    for j in range(len(offsets) - 1):
        (start0, end0), (start1, end1) = spans[j], spans[j + 1]
        boundary = (end0 + start1) / 2
        lo = max(start0, boundary - CHANGEPOINT_REACH)
        drop = max(0.0, offsets[j] - offsets[j + 1])
        hi = min(end1 + drop, boundary + CHANGEPOINT_REACH + drop)
        shifts = [
            v - o for v, o in ((video_offsets[j], offsets[j]), (video_offsets[j + 1], offsets[j + 1])) if v is not None
        ]
        av_shift = float(np.mean(shifts)) if shifts else 0.0
        cut = localize(matcher, donor, offsets[j], offsets[j + 1], lo, hi, video.b if video else None, av_shift)
        cuts.append(cut)
        log.info(
            "cut at %s: %+.3f -> %+.3f s (%s, %s)",
            mmss(cut.a_time),
            cut.offset_before,
            cut.offset_after,
            cut.confidence,
            cut.evidence[0].kind,
        )

    segments = _segments(offsets, counts, video_offsets, cuts, donor, a_ref.duration)
    plan = Plan(
        a=a_ref,
        b=b_ref,
        ratio=choice.ratio,
        ratio_name=choice.name,
        ratio_method=choice.method,
        segments=segments,
        cuts=cuts,
        levels=[{**asdict(lv), "kept": not lv.dropped} for lv in levels],
        candidates=[asdict(c) for c in choice.candidates],
        windows={
            "times": track.times.tolist(),
            "offsets": track.offsets.tolist(),
            "scores": track.scores.tolist(),
        },
    )
    if choice.warning:
        plan.warnings.append(choice.warning)
    for c in cuts:
        if c.confidence == "low":
            plan.warnings.append(f"the cut at {mmss(c.a_time)} is a guess; listen there and edit the plan if needed")
    for lv in kept:
        if lv.crispness < 0.5:
            plan.warnings.append(f"offset {lv.offset:+.3f} s is a loose match (crispness {lv.crispness:.2f})")
    plan.normalize()
    plan.validate()
    return plan


def _edge_level(
    matcher: Matcher,
    donor: Features,
    t0: float,
    t1: float,
    neighbor: float,
    at_end: bool,
    max_offset: float | None,
) -> Level | None:
    """
    A crisp level in donor [t0, t1] (stretched seconds) that only short windows reveal, or None.

    It must differ from the neighboring segment's offset and keep the target time moving forward: a theme tune that
    recurs elsewhere in the target matches tightly too, but at an offset that runs backwards.
    """
    if t1 - t0 < EDGE_MIN:
        return None
    m = int(round(EDGE_WINDOW * FRAME_RATE))
    m -= m % DECIMATION
    hop = int(round(EDGE_HOP * FRAME_RATE))
    f0 = max(0, int((t0 - T0) * FRAME_RATE) - m // 2)
    f1 = min(donor.n_frames, int((t1 - T0) * FRAME_RATE) + m // 2)
    if f1 - f0 < m:
        return None
    starts = np.unique(np.append(np.arange(f0, f1 - m + 1, hop), f1 - m)).astype(int)
    positions, scores = matcher.search(donor.bands, starts, m, max_offset=max_offset)
    fine = OffsetTrack(
        times=T0 + (starts + m / 2) / FRAME_RATE,
        offsets=(positions - starts) / FRAME_RATE,
        scores=scores,
        starts=starts,
        window_frames=m,
    )
    ok = []
    for lv in find_levels(fine):
        # Close to the neighbor it is more likely the loose dialogue match than a jump
        if lv.crispness < CRISP or abs(lv.offset - neighbor) <= COMPETE_DISTANCE:
            continue
        forward = lv.first + lv.offset >= t0 + neighbor if at_end else lv.last + lv.offset <= t1 + neighbor
        if forward:
            ok.append(lv)
    return max(ok, key=lambda lv: lv.quality, default=None)


def _av_consistency(levels: list[Level], track: OffsetTrack, video: _Video) -> dict[int, float]:
    """
    Per level: how far its audio-minus-video difference is from the other levels' (seconds). Only computed when
    levels compete, for the competitors and a few crisp references, since every sample decodes video.
    """
    competing = {id(x) for x in levels for y in levels if x is not y and abs(x.offset - y.offset) <= COMPETE_DISTANCE}
    if not competing:
        return {}
    crisp = [lv for lv in levels if lv.crispness >= CRISP and id(lv) not in competing][:3]
    diffs: dict[int, float] = {}
    for lv in [lv for lv in levels if id(lv) in competing] + crisp:
        v = video.offset_of(_level_times(track, lv.offset, n=1), lv.offset)
        if v is not None:
            diffs[id(lv)] = lv.offset - v
    # Loose levels do not vote: their audio offsets are off by design
    voters = {id(lv) for lv in levels if lv.crispness >= CRISP and id(lv) in diffs}
    out: dict[int, float] = {}
    for lv in levels:
        if id(lv) not in diffs:
            continue
        others = [diffs[k] for k in voters if k != id(lv) and abs(diffs[k] - diffs[id(lv)]) < 1.0]
        if others:
            out[id(lv)] = abs(diffs[id(lv)] - float(np.median(others)))
    return out


def _segments(
    offsets: list[float],
    counts: list[int],
    video_offsets: list[float | None],
    cuts: list[Cut],
    donor: Features,
    a_duration: float,
) -> list[Segment]:
    donor_end = donor.duration
    segments: list[Segment] = []
    for j, off in enumerate(offsets):
        b0 = 0.0 if j == 0 else cuts[j - 1].b_time + max(0.0, offsets[j - 1] - off)
        b1 = donor_end if j == len(offsets) - 1 else cuts[j].b_time
        a0, a1 = max(0.0, b0 + off), min(a_duration, b1 + off)
        if a1 - a0 <= 0.0:
            continue
        segments.append(Segment(a_start=a0, a_end=a1, offset=off, windows=counts[j], video_offset=video_offsets[j]))
    return segments
