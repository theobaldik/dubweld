"""Normalized cross-correlation of feature windows against a whole target, coarse to fine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dubweld.errors import AnalysisError
from dubweld.features import FRAME_RATE, T0, Features

log = logging.getLogger(__name__)

DECIMATION = 4
TOP_CANDIDATES = 5
REFINE_RADIUS = 12
# Coarse peaks closer than this (decimated frames) are one peak; small enough to keep a level 0.2 s away
NMS_RADIUS = 2
BATCH_BYTES = 256 * 1024 * 1024
# The coarse search runs on the target's top principal components: 3x faster, the refined result is unchanged
COARSE_COMPONENTS = 12

ProgressFn = Callable[[float], None]


@dataclass
class OffsetTrack:
    """
    Best match of each window: `times` are window centers on the (stretched) donor clock, `offsets` the seconds to add
    to land on the target, `scores` the normalized correlation at that offset (1 = identical).
    """

    times: np.ndarray
    offsets: np.ndarray
    scores: np.ndarray
    starts: np.ndarray
    window_frames: int

    def __len__(self) -> int:
        return len(self.times)


def _decimate(bands: np.ndarray, factor: int) -> np.ndarray:
    n = bands.shape[1] // factor * factor
    return bands[:, :n].reshape(bands.shape[0], -1, factor).mean(axis=2)


def _fast_len(n: int) -> int:
    """Smallest 2^a * 3^b * 5^c >= n, a size pocketfft handles quickly."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


def parabolic(y_m: float, y_0: float, y_p: float) -> tuple[float, float]:
    denom = y_m - 2 * y_0 + y_p
    if denom >= 0:
        return 0.0, y_0
    delta = float(np.clip(0.5 * (y_m - y_p) / denom, -0.5, 0.5))
    return delta, y_0 - 0.25 * (y_m - y_p) * delta


class Matcher:
    """
    Finds where windows of a donor feature matrix sit in a target feature matrix.

    :param target: target features (the clock offsets are measured on)
    """

    def __init__(self, target: Features) -> None:
        self.bands: np.ndarray = np.ascontiguousarray(target.bands, dtype=np.float32)
        self.n: int = self.bands.shape[1]
        energy = np.einsum("bn,bn->n", self.bands, self.bands, dtype=np.float64)
        self._cum: np.ndarray = np.concatenate([[0.0], np.cumsum(energy)])
        k = min(COARSE_COMPONENTS, self.bands.shape[0])
        _, vectors = np.linalg.eigh(np.cov(self.bands.astype(np.float64)))
        self._basis: np.ndarray = vectors[:, ::-1][:, :k].T.astype(np.float32)
        coarse = _decimate(self._basis @ self.bands, DECIMATION).astype(np.float32)
        self.n_coarse: int = coarse.shape[1]
        energy_c = np.einsum("bn,bn->n", coarse, coarse, dtype=np.float64)
        self._cum_coarse: np.ndarray = np.concatenate([[0.0], np.cumsum(energy_c)])
        self.nfft: int = _fast_len(self.n_coarse)
        self._spec: np.ndarray = np.fft.rfft(coarse, n=self.nfft, axis=1).astype(np.complex64)

    def _seg_energy(self, cum: np.ndarray, positions: np.ndarray, length: int) -> np.ndarray:
        return cum[positions + length] - cum[positions]

    def ncc(self, window: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Normalized correlation of one (bands, M) window at integer target positions (clipped to range)."""
        m = window.shape[1]
        positions = np.clip(np.asarray(positions, dtype=int), 0, self.n - m)
        w_norm = float(np.sqrt(np.sum(window.astype(np.float64) ** 2)))
        if w_norm < 1e-9 or len(positions) == 0:
            return np.zeros(len(positions))
        lo, hi = int(positions.min()), int(positions.max())
        seg = self.bands[:, lo : hi + m]
        # Per-band np.correlate over the covered range beats einsum on a strided view several times over
        num = np.zeros(hi - lo + 1)
        for b in range(seg.shape[0]):
            num += np.correlate(seg[b], window[b], "valid")
        num = num[positions - lo]
        den = w_norm * np.sqrt(np.maximum(self._seg_energy(self._cum, positions, m), 1e-12))
        return num / den

    def score_at(self, window: np.ndarray, position: float) -> float:
        """Normalized correlation at a fractional target position (linear between the two integer neighbors)."""
        p0 = int(np.floor(position))
        frac = position - p0
        vals = self.ncc(window, np.array([p0, p0 + 1]))
        return float(vals[0] * (1 - frac) + vals[1] * frac)

    def search(
        self,
        donor: np.ndarray,
        starts: np.ndarray,
        m: int,
        max_offset: float | None = None,
        progress: ProgressFn | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Best target position of each donor window `donor[:, s:s+m]`.

        :param max_offset: limit the search to |position - start| <= this many seconds
        :return: (positions in fractional target frames, scores)
        """
        md = m // DECIMATION
        n_valid = self.n_coarse - md + 1
        if n_valid <= 0 or self.n < m:
            raise AnalysisError("the target is shorter than one analysis window")
        donor_c = _decimate(self._basis @ donor, DECIMATION).astype(np.float32)
        n_bands = donor_c.shape[0]
        batch = max(1, BATCH_BYTES // (n_bands * (self.nfft // 2 + 1) * 8 * 2))
        positions = np.zeros(len(starts))
        scores = np.zeros(len(starts))
        seg_c = np.sqrt(np.maximum(self._seg_energy(self._cum_coarse, np.arange(n_valid), md), 1e-12))
        for b0 in range(0, len(starts), batch):
            idx = np.arange(b0, min(b0 + batch, len(starts)))
            sd = starts[idx] // DECIMATION
            wins = np.stack([donor_c[:, s : s + md] for s in sd])  # (nb, bands, md)
            wins_norm = np.sqrt(np.einsum("nbm,nbm->n", wins, wins, dtype=np.float64))
            spec_w = np.fft.rfft(wins, n=self.nfft, axis=2)
            cross = np.einsum("nbf,bf->nf", np.conj(spec_w).astype(np.complex64), self._spec)
            corr = np.fft.irfft(cross, n=self.nfft, axis=1)[:, :n_valid]
            corr /= np.maximum(wins_norm[:, None], 1e-9) * seg_c[None, :]
            for row, i in enumerate(idx):
                c = corr[row]
                if max_offset is not None:
                    radius = int(np.ceil(max_offset * FRAME_RATE / DECIMATION))
                    lo, hi = max(0, sd[row] - radius), min(n_valid, sd[row] + radius + 1)
                    masked = np.full_like(c, -np.inf)
                    masked[lo:hi] = c[lo:hi]
                    c = masked
                positions[i], scores[i] = self._refine(donor[:, starts[i] : starts[i] + m], c)
            if progress:
                progress(min(1.0, (b0 + len(idx)) / len(starts)))
        return positions, scores

    def _refine(self, window: np.ndarray, coarse: np.ndarray) -> tuple[float, float]:
        c = coarse.copy()
        candidates = []
        for _ in range(TOP_CANDIDATES):
            k = int(np.argmax(c))
            if not np.isfinite(c[k]):
                break
            candidates.append(k * DECIMATION)
            c[max(0, k - NMS_RADIUS) : k + NMS_RADIUS + 1] = -np.inf
        m = window.shape[1]
        best = (0.0, -np.inf)
        for p in candidates:
            lags = np.arange(p - REFINE_RADIUS, p + REFINE_RADIUS + DECIMATION + 1)
            lags = lags[(lags >= 0) & (lags <= self.n - m)]
            if len(lags) == 0:
                continue
            vals = self.ncc(window, lags)
            j = int(np.argmax(vals))
            if vals[j] <= best[1]:
                continue
            if 0 < j < len(vals) - 1:
                delta, peak = parabolic(vals[j - 1], vals[j], vals[j + 1])
            else:
                delta, peak = 0.0, float(vals[j])
            best = (lags[j] + delta, peak)
        return best


def window_starts(n_frames: int, window_s: float, hop_s: float) -> tuple[np.ndarray, int]:
    """Start frames of windows of `window_s` every `hop_s` that fit in `n_frames`."""
    m = int(round(window_s * FRAME_RATE))
    m -= m % DECIMATION
    hop = max(1, int(round(hop_s * FRAME_RATE)))
    if n_frames < m:
        return np.zeros(0, dtype=int), m
    return np.arange(0, n_frames - m + 1, hop, dtype=int), m


def offset_track(
    matcher: Matcher,
    donor: Features,
    window_s: float = 8.0,
    hop_s: float = 2.0,
    max_offset: float | None = None,
    progress: ProgressFn | None = None,
) -> OffsetTrack:
    """Dense offset track of a donor (already on the target's clock) against the matcher's target."""
    starts, m = window_starts(donor.n_frames, window_s, hop_s)
    positions, scores = matcher.search(donor.bands, starts, m, max_offset=max_offset, progress=progress)
    times = T0 + (starts + m / 2) / FRAME_RATE
    offsets = (positions - starts) / FRAME_RATE
    return OffsetTrack(times=times, offsets=offsets, scores=scores, starts=starts, window_frames=m)


def scores_at_offset(matcher: Matcher, donor: Features, track: OffsetTrack, offset: float) -> np.ndarray:
    """Score of every window of `track` at one fixed offset (seconds)."""
    out = np.zeros(len(track))
    m = track.window_frames
    for i, s in enumerate(track.starts):
        p = s + offset * FRAME_RATE
        if p < 0 or p > matcher.n - m - 1:
            out[i] = np.nan
            continue
        out[i] = matcher.score_at(donor.bands[:, s : s + m], p)
    return out
