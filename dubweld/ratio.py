"""Choose the speed ratio between donor and target: snap to a standard ratio, fall back to an estimate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from dubweld.errors import AnalysisError
from dubweld.features import Features, stretch
from dubweld.xcorr import Matcher, OffsetTrack, ProgressFn, offset_track

log = logging.getLogger(__name__)

FILM = Fraction(24000, 1001)
NAMED: dict[str, Fraction] = {
    "1": Fraction(1),
    "pal": Fraction(25) / FILM,  # 23.976 -> 25 (PAL speed-up of a film-rate master)
    "pal24": Fraction(25, 24),  # 24 -> 25
    "ntsc": Fraction(24) / FILM,  # 23.976 -> 24, also 29.97 -> 30
}
# Low enough for weak matches (a dialogue-heavy episode peaks around 0.4-0.6): flatness, not the score, rejects
# chance matches
CONFIDENT = 0.4
FLAT_TOLERANCE = 0.010
FLAT_HORIZON = 120.0
SPARSE_HOP = 6.0
# |relative drift| a snapped ratio may leave: 5e-5 is 3 ms per minute
MAX_SLOPE = 5e-5
MIN_FLAT = 4


@dataclass
class Candidate:
    """One ratio tried, with how many confident windows had flat neighbors under it."""

    name: str
    ratio: float
    flat: int = 0
    confident: int = 0
    slope: float = 0.0


@dataclass
class RatioChoice:
    """The ratio used and how it was found (`method`: snapped, forced, estimated)."""

    ratio: float
    name: str
    method: str
    candidates: list[Candidate] = field(default_factory=list)
    warning: str = ""


def parse_ratio(text: str) -> float | None:
    """`auto` -> None, a name from `NAMED`, a float or a fraction like `25/23.976`."""
    text = text.strip().lower()
    if text == "auto":
        return None
    if text in NAMED:
        return float(NAMED[text])
    if text.endswith("-inv") and text[:-4] in NAMED:
        return float(1 / NAMED[text[:-4]])
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            value = float(num) / float(den)
        else:
            value = float(text)
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(f"invalid ratio {text!r}") from e
    if not 0.5 < value < 2.0:
        raise ValueError(f"ratio {value} is out of range (0.5, 2)")
    return value


def candidates(fps_a: Fraction | None, fps_b: Fraction | None) -> list[Candidate]:
    """Ratios to try: the frame-rate quotient first (when both are known), then the standard ones and inverses."""
    out: list[Candidate] = []
    seen: list[float] = []

    def add(name: str, value: Fraction | float) -> None:
        v = float(value)
        if all(abs(v - s) > 1e-6 for s in seen):
            seen.append(v)
            out.append(Candidate(name=name, ratio=v))

    if fps_a and fps_b:
        q = Fraction(fps_b) / Fraction(fps_a)
        named = [n for n, f in NAMED.items() if abs(float(f) - float(q)) < 1e-6]
        named += [f"{n}-inv" for n, f in NAMED.items() if abs(float(1 / f) - float(q)) < 1e-6]
        add(named[0] if named else f"fps {float(fps_b):.3f}/{float(fps_a):.3f}", q)
    for name, value in NAMED.items():
        add(name, value)
        if value != 1:
            add(f"{name}-inv", 1 / value)
    return out


def flat_count(track: OffsetTrack) -> tuple[int, int]:
    """
    Confident windows with at least two other confident windows at the same offset (±10 ms) within ±120 s.

    :return: (flat windows, confident windows)
    """
    conf = track.scores >= CONFIDENT
    t, o = track.times[conf], track.offsets[conf]
    if len(t) == 0:
        return 0, 0
    near = (np.abs(t[:, None] - t[None, :]) <= FLAT_HORIZON) & (np.abs(o[:, None] - o[None, :]) <= FLAT_TOLERANCE)
    return int(np.sum(near.sum(axis=1) >= 3)), int(len(t))


def drift_slope(track: OffsetTrack) -> float:
    """Median offset drift per second between confident windows on (nearly) the same level, 20-300 s apart."""
    conf = track.scores >= CONFIDENT
    t, o = track.times[conf], track.offsets[conf]
    if len(t) < 2:
        return 0.0
    dt = t[None, :] - t[:, None]
    do = o[None, :] - o[:, None]
    pair = (dt >= 20) & (dt <= 300) & (np.abs(do) < 0.08)
    if not pair.any():
        return 0.0
    return float(np.median(do[pair] / dt[pair]))


def _evaluate(matcher: Matcher, donor: Features, cand: Candidate, hop: float) -> None:
    track = offset_track(matcher, stretch(donor, cand.ratio), hop_s=hop)
    cand.flat, cand.confident = flat_count(track)
    cand.slope = drift_slope(track)


def choose_ratio(
    matcher: Matcher,
    donor: Features,
    fps_a: Fraction | None,
    fps_b: Fraction | None,
    forced: float | None = None,
    progress: ProgressFn | None = None,
    hop: float = 2.0,
) -> RatioChoice:
    """
    Pick the speed ratio (donor seconds × ratio = target seconds) by how flat the confident offsets are.

    :param hop: window spacing (s) of the analysis; candidates are re-scored at it when the sparse pass is too thin

    :raises AnalysisError: when no ratio, standard or estimated, gives a usable match
    """
    if forced is not None:
        return RatioChoice(ratio=forced, name=f"{forced:.10g}", method="forced")
    cands = candidates(fps_a, fps_b)
    for i, cand in enumerate(cands):
        _evaluate(matcher, donor, cand, SPARSE_HOP)
        log.debug(
            "ratio %s %.10f: %d flat of %d confident, drift %.2e",
            cand.name,
            cand.ratio,
            cand.flat,
            cand.confident,
            cand.slope,
        )
        if progress:
            progress((i + 1) / len(cands))

    # Neighboring ratios (PAL and 25/24 differ by 0.1%) can be about as flat; the drift tells them apart, even when
    # noise pushes the right one's over MAX_SLOPE
    def rank(c: Candidate) -> tuple[bool, bool, float, int]:
        enough = c.flat >= MIN_FLAT
        return enough and abs(c.slope) <= MAX_SLOPE, enough, -abs(c.slope) if enough else 0.0, c.flat

    best = max(cands, key=rank)
    spacing = SPARSE_HOP
    if not rank(best)[0] and hop < SPARSE_HOP:
        spacing = hop
        # Too few matches for the sparse pass (a dialogue-heavy episode): look again as densely as the analysis will
        for cand in cands:
            if cand.flat:
                _evaluate(matcher, donor, cand, spacing)
                log.debug("ratio %s at %.1f s spacing: %d flat, drift %.2e", cand.name, hop, cand.flat, cand.slope)
        best = max(cands, key=rank)
    if best.flat >= MIN_FLAT and abs(best.slope) <= MAX_SLOPE:
        return RatioChoice(ratio=best.ratio, name=best.name, method="snapped", candidates=cands)

    if best.flat >= MIN_FLAT:
        start = best
    else:
        start = _scan(matcher, donor, cands)
        if start is None:
            raise AnalysisError(
                "no speed ratio gives a usable match; are these two releases of the same video, and is the audio "
                "track right (--target-track / --donor-track)?"
            )
    cand = Candidate(name="estimated", ratio=start.ratio * (1 + start.slope))
    for _ in range(3):
        _evaluate(matcher, donor, cand, spacing)
        if abs(cand.slope) <= MAX_SLOPE / 4:
            break
        refined = Candidate(name="estimated", ratio=cand.ratio * (1 + cand.slope))
        _evaluate(matcher, donor, refined, spacing)
        if (refined.flat, -abs(refined.slope)) <= (cand.flat, -abs(cand.slope)):
            break
        cand = refined
    cands.append(cand)
    if best.flat >= MIN_FLAT and cand.flat < best.flat + max(2, best.flat // 4):
        # Drift measured between a few windows is noisy (2 ms over 40 s is 5e-5): keep the standard ratio unless
        # the estimate is clearly flatter
        log.info("estimated ratio %.8f is not flatter than %s; keeping %s", cand.ratio, best.name, best.name)
        return RatioChoice(ratio=best.ratio, name=best.name, method="snapped", candidates=cands)
    if cand.flat < MIN_FLAT:
        raise AnalysisError(
            f"no speed ratio gives a usable match (best estimate {cand.ratio:.6f}); are these two releases of the "
            "same video, and is the audio track right (--target-track / --donor-track)?"
        )
    estimate = cand.ratio
    warning = (
        f"no standard speed ratio is flat (best: {best.name}, drift {best.slope:.1e}); using the estimated ratio "
        f"{estimate:.8f}. Check the result, and pass --ratio if you know the real one."
    )
    log.warning(warning)
    return RatioChoice(ratio=estimate, name="estimated", method="estimated", candidates=cands, warning=warning)


def _scan(matcher: Matcher, donor: Features, cands: list[Candidate]) -> Candidate | None:
    """Coarse search over 0.9-1.1 in steps an 8 s window tolerates (about 1 frame of smear)."""
    log.warning("no standard ratio matches; scanning 0.90-1.10, this takes a while")
    best: Candidate | None = None
    for r in np.arange(0.90, 1.10, 0.0015):
        cand = Candidate(name="scan", ratio=float(r))
        _evaluate(matcher, donor, cand, SPARSE_HOP * 2)
        if best is None or (cand.flat, cand.confident) > (best.flat, best.confident):
            best = cand
    if best is None or best.confident < 3:
        return None
    _evaluate(matcher, donor, best, SPARSE_HOP)
    return best
