"""On-disk cache of decoded features, keyed by file signature, so re-analyzing a pair is fast."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from dubweld.features import Features

log = logging.getLogger(__name__)

# Bump when feature extraction changes, so stale entries are never read
FEATURE_VERSION = 3


def default_cache_dir() -> Path:
    """`$XDG_CACHE_HOME/dubweld`, or `~/.cache/dubweld`."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "dubweld"


def _key(path: Path, audio_index: int, size: int, mtime: float) -> str:
    raw = f"{Path(path).resolve()}|{size}|{mtime:.6f}|{audio_index}|v{FEATURE_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()


def features(
    path: Path,
    audio_index: int,
    size: int,
    mtime: float,
    cache_dir: Path | None,
    compute: Callable[[], Features],
) -> Features:
    """Load features from `cache_dir` or compute and store them; with no `cache_dir`, always compute."""
    if cache_dir is None:
        return compute()
    file = Path(cache_dir) / f"features-{_key(path, audio_index, size, mtime)}.npz"
    if file.is_file():
        try:
            with np.load(file) as data:
                log.debug("features of %s from cache", Path(path).name)
                return Features(bands=data["bands"], env_db=data["env_db"])
        except (OSError, ValueError, KeyError) as e:
            log.debug("ignoring unreadable cache entry %s: %s", file, e)
    result = compute()
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp.npz")
        np.savez(tmp, bands=result.bands, env_db=result.env_db)
        tmp.replace(file)
    except OSError as e:
        log.debug("cannot write cache %s: %s", file, e)
    return result
