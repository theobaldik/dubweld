"""Pin down where a cut is: audio changepoint for the region, then video black, scene cuts or silence inside it."""

from __future__ import annotations

import logging

import numpy as np

from dubweld.errors import ToolError
from dubweld.features import FRAME_RATE, T0, Features
from dubweld.levels import changepoint
from dubweld.plan import Cut, Evidence, mmss
from dubweld.video import VideoReader, black_runs, scene_cuts
from dubweld.xcorr import Matcher, OffsetTrack, scores_at_offset, window_starts

log = logging.getLogger(__name__)

FINE_WINDOW = 4.0
FINE_HOP = 1.0
SEARCH_MARGIN = 2.0
SILENCE_DB = -45.0
MIN_SILENCE = 0.2
SHARP_AUDIO = 2.0


def fine_track(donor: Features, lo: float, hi: float) -> OffsetTrack:
    """Short windows covering stretched donor time [lo, hi], for scoring at fixed offsets (no search)."""
    starts, m = window_starts(donor.n_frames, FINE_WINDOW, FINE_HOP)
    times = T0 + (starts + m / 2) / FRAME_RATE
    keep = (times >= lo) & (times <= hi)
    starts, times = starts[keep], times[keep]
    return OffsetTrack(
        times=times, offsets=np.zeros(len(times)), scores=np.zeros(len(times)), starts=starts, window_frames=m
    )


def silence_runs(donor: Features, lo: float, hi: float) -> list[tuple[float, float, float]]:
    """Quiet runs (start, end, level dBFS) in stretched donor time [lo, hi]."""
    k0 = max(0, int((lo - T0) * FRAME_RATE))
    k1 = min(donor.n_frames, int((hi - T0) * FRAME_RATE) + 1)
    env = donor.env_db[k0:k1]
    if len(env) == 0:
        return []
    quiet = env < SILENCE_DB
    edges = np.diff(np.concatenate([[0], quiet.astype(int), [0]]))
    out = []
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if (e - s) / FRAME_RATE >= MIN_SILENCE:
            out.append((T0 + (k0 + s) / FRAME_RATE, T0 + (k0 + e) / FRAME_RATE, float(np.min(env[s:e]))))
    return out


def _distance(t: float, estimate: float, drop: float) -> float:
    """Distance of a clue from the cut, or from the far end of the dropped material in a down-jump."""
    return min(abs(t - estimate), abs(t - estimate - drop)) if drop > 0 else abs(t - estimate)


def localize(
    matcher: Matcher,
    donor: Features,
    off_a: float,
    off_b: float,
    lo: float,
    hi: float,
    video: VideoReader | None = None,
    av_shift: float = 0.0,
) -> Cut:
    """
    Find the cut between a stretch at offset `off_a` and one at `off_b` in stretched donor time [lo, hi].

    :param av_shift: donor video-to-audio correction (video offset - audio offset): a black frame at stretched
        video time v belongs to stretched audio time v + av_shift
    """
    ft = fine_track(donor, lo, hi)
    if len(ft) < 3:
        mid = (lo + hi) / 2
        return Cut(
            a_time=mid + off_a,
            b_time=mid,
            offset_before=off_a,
            offset_after=off_b,
            confidence="low",
            evidence=[Evidence("midpoint", mid + off_a, "too little audio to compare")],
            search=(lo, hi),
        )
    sa = scores_at_offset(matcher, donor, ft, off_a)
    sb = scores_at_offset(matcher, donor, ft, off_b)
    # A down-jump (the donor has extra material) leaves a stretch matching neither level
    drop = max(0.0, off_a - off_b)
    estimate, c_lo, c_hi, margin = changepoint(ft.times, sa, sb, drop=drop)
    s_lo, s_hi = max(lo, c_lo - SEARCH_MARGIN), min(hi, c_hi + drop + SEARCH_MARGIN)
    evidence: list[Evidence] = []
    width = c_hi - c_lo
    audio_detail = f"changepoint {mmss(estimate + off_a)}, ±{width / 2:.1f} s"
    audio_ev = Evidence("audio", estimate + off_a, audio_detail)

    if video is not None:
        try:
            clip = video.clip(s_lo - av_shift - 1.0, s_hi - av_shift + 1.0)
        except ToolError as e:
            log.warning("cannot read donor video near %s: %s", mmss(estimate + off_a), e)
            clip = None
        if clip is not None and len(clip):
            blacks = [r for r in black_runs(clip, video.frame_duration) if s_lo <= r.middle + av_shift <= s_hi]
            if blacks:
                best = min(blacks, key=lambda r: (_distance(r.middle + av_shift, estimate, drop) - 2 * r.length))
                t = best.middle + av_shift
                evidence.append(
                    Evidence(
                        "black", t + off_a, f"donor fades to black {best.length:.2f} s at {mmss(best.start + off_a)}"
                    )
                )
            cuts = [c + av_shift for c in scene_cuts(clip) if s_lo <= c + av_shift <= s_hi]
            if cuts:
                t = min(cuts, key=lambda c: _distance(c, estimate, drop))
                evidence.append(Evidence("scene", t + off_a, f"donor scene cut at {mmss(t + off_a)}"))

    quiet = silence_runs(donor, s_lo, s_hi)
    if quiet:
        s, e, level = min(quiet, key=lambda q: (_distance((q[0] + q[1]) / 2, estimate, drop) - 2 * (q[1] - q[0])))
        evidence.append(Evidence("silence", (s + e) / 2 + off_a, f"donor quiet ({level:.0f} dBFS) for {e - s:.2f} s"))
    evidence.append(audio_ev)

    chosen = evidence[0]
    if chosen.kind == "black":
        confidence = "high"
    elif chosen.kind in ("scene", "silence"):
        confidence = "medium"
    else:
        confidence = "medium" if width <= SHARP_AUDIO and margin > 1.0 else "low"
    b_time = chosen.a_time - off_a
    # In a down-jump the clue may mark where level b starts, the far end of the dropped material
    if drop > 0 and chosen.kind != "audio" and abs(b_time - estimate - drop) < abs(b_time - estimate):
        b_time -= drop
    return Cut(
        a_time=b_time + off_a,
        b_time=b_time,
        offset_before=off_a,
        offset_after=off_b,
        confidence=confidence,
        evidence=evidence,
        search=(s_lo + off_a, s_hi + off_a),
    )
