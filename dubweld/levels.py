"""Offset levels (where the music and effects match exactly) and the segmentation of the donor into them."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from dubweld.xcorr import OffsetTrack

log = logging.getLogger(__name__)

SEED_SCORE = 0.5
MEMBER_SCORE = 0.4
TOLERANCE = 0.010
NEIGHBORHOOD = 0.150
MERGE_DISTANCE = 0.015
MIN_MEMBERS = 3
# Dubbed dialogue roughly lining up with the original forms a loose cluster 0.2-0.3 s from the true level
COMPETE_DISTANCE = 0.6
CRISP = 0.5
SWITCH_PENALTY = 1.5
UNINFORMATIVE = 0.3


@dataclass
class Level:
    """
    A candidate offset (seconds) seen in several windows.

    `crispness` is the share of nearby matches (±150 ms) that sit within ±10 ms of it: about 1 for the music and
    effects, low for the loose dialogue cluster.
    """

    offset: float
    members: int
    seeds: int
    crispness: float
    median_score: float
    first: float
    last: float
    support: float
    dropped: str = ""

    @property
    def quality(self) -> float:
        return self.crispness * self.support


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def find_levels(track: OffsetTrack) -> list[Level]:
    """Tight clusters (±10 ms) of high-score windows, best first; never a mean over all windows."""
    t, o, s = track.times, track.offsets, track.scores
    seeds = np.where(s >= SEED_SCORE)[0]
    seeds = seeds[np.argsort(-s[seeds])]
    used = np.zeros(len(t), dtype=bool)
    centers: list[float] = []
    for i in seeds:
        if used[i]:
            continue
        near = (s >= SEED_SCORE) & (np.abs(o - o[i]) <= TOLERANCE) & ~used
        center = _weighted_median(o[near], s[near])
        near = (s >= SEED_SCORE) & (np.abs(o - center) <= TOLERANCE)
        center = _weighted_median(o[near], s[near])
        used |= near
        centers.append(center)
    merged: list[float] = []
    for c in sorted(centers):
        if merged and c - merged[-1] <= MERGE_DISTANCE:
            continue
        merged.append(c)
    levels: list[Level] = []
    for c in merged:
        members = (s >= MEMBER_SCORE) & (np.abs(o - c) <= TOLERANCE)
        neighborhood = (s >= MEMBER_SCORE) & (np.abs(o - c) <= NEIGHBORHOOD)
        seeds_n = int(np.sum(members & (s >= SEED_SCORE)))
        if members.sum() < MIN_MEMBERS or seeds_n == 0:
            continue
        c = _weighted_median(o[members], s[members])
        levels.append(
            Level(
                offset=c,
                members=int(members.sum()),
                seeds=seeds_n,
                crispness=float(members.sum() / max(1, neighborhood.sum())),
                median_score=float(np.median(s[members])),
                first=float(t[members].min()),
                last=float(t[members].max()),
                support=float(np.sum(s[members])),
            )
        )
    levels.sort(key=lambda lv: -lv.quality)
    return levels


def prune_levels(levels: list[Level], av_consistency: dict[int, float] | None = None) -> list[Level]:
    """
    Drop loose levels that compete with a crisp one 0.1-0.6 s away; prefer crispness over window count.

    :param av_consistency: optional per-level distance (s) of its audio-video offset difference from the other
        crisp levels'; between two crisp (or two loose) levels, the one the video disagrees with by > 0.1 s more
        loses
    :return: the kept levels; dropped ones are marked in place (`dropped`)
    """
    kept: list[Level] = []
    for lv in sorted(levels, key=lambda x: -x.quality):
        rival = next((k for k in kept if abs(k.offset - lv.offset) <= COMPETE_DISTANCE), None)
        if rival is not None:
            loses = lv.crispness < CRISP or lv.quality < 0.5 * rival.quality
            # Video only settles contests crispness cannot: the donor's A/V sync itself varies by 0.2 s
            same_kind = (lv.crispness >= CRISP) == (rival.crispness >= CRISP)
            if av_consistency is not None and same_kind:
                a = av_consistency.get(id(lv))
                b = av_consistency.get(id(rival))
                if a is not None and b is not None and abs(a - b) > 0.1:
                    loses = a > b
            if loses:
                lv.dropped = f"loose cluster next to {rival.offset:+.3f} s"
                continue
        kept.append(lv)
    return sorted(kept, key=lambda x: x.first)


@dataclass
class Run:
    """Consecutive windows [i0, i1] assigned to one level."""

    level: int
    i0: int
    i1: int
    confident: list[int] = field(default_factory=list)


def viterbi(level_scores: np.ndarray, penalty: float = SWITCH_PENALTY) -> np.ndarray:
    """
    Most likely level per window: maximize the summed scores at each window's level minus `penalty` per switch.

    :param level_scores: (windows, levels) score of each window at each level's exact offset (NaN = unusable)
    """
    n, k = level_scores.shape
    e = np.nan_to_num(level_scores, nan=0.0)
    e = np.clip(e, 0, None)
    weak = e.max(axis=1, initial=0) < UNINFORMATIVE
    e[weak] = 0
    acc = e[0].copy()
    back = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        best_prev = int(np.argmax(acc))
        stay = acc
        switch = acc[best_prev] - penalty
        choose_switch = switch > stay
        back[i] = np.where(choose_switch, best_prev, np.arange(k))
        acc = np.where(choose_switch, switch, stay) + e[i]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(acc))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def runs_of(path: np.ndarray, level_scores: np.ndarray, track: OffsetTrack, levels: list[Level]) -> list[Run]:
    """Runs of the Viterbi path, keeping only those backed by at least one confident window at their level."""
    runs: list[Run] = []
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            lv = int(path[start])
            idx = np.arange(start, i)
            conf = idx[
                (np.nan_to_num(level_scores[idx, lv]) >= MEMBER_SCORE)
                & (np.abs(track.offsets[idx] - levels[lv].offset) <= 2 * TOLERANCE)
            ]
            runs.append(Run(level=lv, i0=start, i1=i - 1, confident=[int(c) for c in conf]))
            start = i
    # An unbacked run is noise: give its windows to a neighbor and merge equal neighbors
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, r in enumerate(runs):
            if r.confident:
                continue
            if j > 0:
                runs[j - 1].i1 = r.i1
            else:
                runs[j + 1].i0 = r.i0
            del runs[j]
            changed = True
            break
        merged: list[Run] = []
        for r in runs:
            if merged and merged[-1].level == r.level:
                merged[-1].i1 = r.i1
                merged[-1].confident += r.confident
                changed = True
            else:
                merged.append(r)
        runs = merged
    return runs


def changepoint(
    times: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, slack: float = 1.0, drop: float = 0.0
) -> tuple[float, float, float, float]:
    """
    Where the donor switches from level a to level b: the split c maximizing Σ score_a before c + Σ score_b after.

    :param drop: seconds of donor material between the levels that matches neither (a down-jump: the donor has
        extra material); windows in [c, c + drop] count at a baseline score instead
    :return: (estimate, low, high, margin): the split time, the range of splits within `slack` of the best one,
        and how much better the best split is than the worst end of the range
    """
    a = np.nan_to_num(score_a)
    b = np.nan_to_num(score_b)
    edges = np.concatenate([[times[0]], (times[:-1] + times[1:]) / 2, [times[-1]]])
    if drop > 0:
        fits = edges[edges + drop <= times[-1] + (times[1] - times[0] if len(times) > 1 else 0)]
        # Dropped material longer than the searched range: the split can only be at its start
        edges = fits if len(fits) else edges[:1]
    cum_a = np.concatenate([[0.0], np.cumsum(a)])
    cum_b = np.concatenate([[0.0], np.cumsum(b)])
    base = float(np.median(np.minimum(a, b)))
    i_c = np.searchsorted(times, edges)  # windows before the split
    i_d = np.searchsorted(times, edges + drop) if drop > 0 else i_c
    gain = cum_a[i_c] + (cum_b[-1] - cum_b[i_d]) + base * (i_d - i_c)
    k = int(np.argmax(gain))
    ok = np.where(gain >= gain[k] - slack)[0]
    margin = float(gain[k] - min(gain[0], gain[-1]))
    return float(edges[k]), float(edges[ok.min()]), float(edges[ok.max()]), margin
