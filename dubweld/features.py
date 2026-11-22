"""Language-agnostic band-energy features at 50 frames per second, plus a loudness envelope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
WIN = 1024
HOP = 320
FRAME_RATE = SAMPLE_RATE / HOP  # 50 frames per second
N_BANDS = 32
# Time of frame 0's center; identical for every file, so it cancels out of offsets but not out of cut times
T0 = WIN / 2 / SAMPLE_RATE
MOVING_AVERAGE_S = 5.0
CHUNK_FRAMES = 8192


@dataclass
class Features:
    """
    Per-band features of one audio stream.

    `bands` is (N_BANDS, frames) float32, zero-mean unit-variance per band after removing a 5 s moving average
    (which cancels level and EQ differences between mixes). `env_db` is the per-frame RMS level in dBFS.
    Frame i is centered at `T0 + i / FRAME_RATE` seconds of decoded audio.
    """

    bands: np.ndarray
    env_db: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.bands.shape[1]

    @property
    def duration(self) -> float:
        return self.n_frames / FRAME_RATE


def band_edges(n_bins: int = WIN // 2 + 1, n_bands: int = N_BANDS) -> np.ndarray:
    """Integer FFT-bin edges of `n_bands` log-spaced bands, strictly increasing."""
    raw = np.geomspace(3, n_bins - 1, n_bands + 1)
    edges = np.round(raw).astype(int)
    for k in range(1, len(edges)):
        edges[k] = max(edges[k], edges[k - 1] + 1)
    if edges[-1] > n_bins:
        raise ValueError("too many bands for the FFT size")
    return edges


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average along the last axis; edges average over the samples that exist."""
    n = x.shape[-1]
    half = width // 2
    c = np.zeros(x.shape[:-1] + (n + 1,), dtype=np.float64)
    np.cumsum(x, axis=-1, out=c[..., 1:])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return ((c[..., hi] - c[..., lo]) / (hi - lo)).astype(x.dtype)


def compute_features(pcm: np.ndarray) -> Features:
    """
    Features of 16 kHz mono float PCM.

    :param pcm: float32 samples in [-1, 1] at `SAMPLE_RATE`
    """
    pcm = np.asarray(pcm, dtype=np.float32)
    if len(pcm) < WIN:
        pcm = np.pad(pcm, (0, WIN - len(pcm)))
    n_frames = 1 + (len(pcm) - WIN) // HOP
    window = np.hanning(WIN).astype(np.float32)
    norm = 1.0  # raw power of [-1, 1] samples: log1p(1e4 E) then behaves like dB down to near silence
    edges = band_edges()
    energy = np.empty((N_BANDS, n_frames), dtype=np.float32)
    env = np.empty(n_frames, dtype=np.float32)
    frames_view = np.lib.stride_tricks.sliding_window_view(pcm, WIN)[::HOP]
    for start in range(0, n_frames, CHUNK_FRAMES):
        block = frames_view[start : start + CHUNK_FRAMES]
        env[start : start + len(block)] = np.sqrt(np.mean(block.astype(np.float32) ** 2, axis=1))
        spec = np.fft.rfft(block * window, axis=1)
        power = (spec.real**2 + spec.imag**2).astype(np.float32) / norm
        csum = np.concatenate([np.zeros((len(block), 1), np.float32), np.cumsum(power, axis=1)], axis=1)
        energy[:, start : start + len(block)] = (csum[:, edges[1:]] - csum[:, edges[:-1]]).T
    bands = np.log1p(1e4 * energy)
    bands -= moving_average(bands, int(MOVING_AVERAGE_S * FRAME_RATE))
    std = bands.std(axis=1, keepdims=True)
    bands -= bands.mean(axis=1, keepdims=True)
    bands /= np.where(std > 1e-6, std, 1.0)
    env_db = 20 * np.log10(np.maximum(env, 1e-6))
    return Features(bands=bands.astype(np.float32), env_db=env_db.astype(np.float32))


def stretch(features: Features, ratio: float) -> Features:
    """
    Resample features in time by `ratio` (donor clock -> target clock): frame k of the result holds the donor at
    stretched time `T0 + k / FRAME_RATE`, i.e. donor time `(T0 + k / FRAME_RATE) / ratio`.
    """
    if ratio == 1.0:
        return features
    n_out = int(np.floor((features.n_frames - 1 + T0 * FRAME_RATE) * ratio - T0 * FRAME_RATE)) + 1
    k = np.arange(n_out, dtype=np.float64)
    src = ((T0 + k / FRAME_RATE) / ratio - T0) * FRAME_RATE
    src = np.clip(src, 0, features.n_frames - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, features.n_frames - 1)
    frac = (src - i0).astype(np.float32)
    bands = features.bands[:, i0] * (1 - frac) + features.bands[:, i1] * frac
    env = features.env_db[i0] * (1 - frac) + features.env_db[i1] * frac
    return Features(bands=bands.astype(np.float32), env_db=env.astype(np.float32))
