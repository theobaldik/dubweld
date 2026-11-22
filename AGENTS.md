# AGENTS.md

## Commands

- **Install deps:** `uv sync`
- **Run tests:** `uv run pytest` (about 4 minutes; `-m "not slow and not real"` for the fast unit tests only)
- **Run the CLI:** `uv run dubweld run TARGET DONOR -o tries/synced.mkv` (or `uv run python -m dubweld`)
- **Build:** `uv build`
- **Format:** `black .` (line length 120, configured in `pyproject.toml`; not a declared dev dependency)
- No linter or type checker is configured

## Requirements

- Python 3.11+, numpy (the only runtime dependency)
- `ffmpeg` and `ffprobe` on PATH; `mkvmerge` recommended (muxing falls back to ffmpeg without it)

## Test data and tests

- `tests/conftest.py` generates releases: `music_and_effects()` (the shared bed), `dialogue()` (different per
  release, with random syllable lengths: a periodic envelope correlates between the two dubs and biases matches by
  several ms), and `make_release()`, which cuts, speeds up (44.1 kHz AAC donor), adds donor-only material, silences
  ranges in both mixes and optionally adds tiny FFV1 videos with black frames at the cuts. `Release.truth` holds the
  true `(a_start, a_end, offset)` per segment.
- `tests/test_real.py` is marked `real` and runs one case per set of real files in `$DUBWELD_DATA` or the repo root:
  `target<id>.<ext>` and `donor<id>.<ext>`, plus optionally `synced<id>.<ext>`, a hand-made checked result with the
  dub as its last audio stream. `<id>` is anything shared by a set (`target.mkv`, `target1.mkv`, `target-2.mp4`, ...)
  and `<ext>` any container in `conftest.MEDIA_EXTENSIONS` (`conftest.find_releases`). The render must verify
  against the target and, when present, match the `synced` dub (the ground truth). These files are never committed.
- `tests/test_render.py::TestRender::test_clicks_land_to_the_millisecond` guards sample-accurate placement through
  every codec and muxer, and `test_44k_donor_is_slowed_not_sped_up` the resample-before-asetrate fix. Keep both.
- Write outputs of manual runs to `tries/` (git-ignored). Never modify the input files.

## Architecture

`dubweld/` is a library with a thin CLI on top. The library has no global state, reads no environment variables
(except `cache.default_cache_dir()`, which only the CLI calls), and never configures logging.

| Module | Role |
|---|---|
| `probe.py` | ffprobe JSON → `MediaInfo`; skips attached cover pictures (they show up as video streams) |
| `ffmpeg.py` | `Tools` (executable paths), `run()`, PCM decoding, streaming decoder/encoder processes |
| `features.py` | 16 kHz mono → 50 fps, 32 log-spaced bands, `log1p(1e4·E)` on raw power, 5 s moving average removed, z-scored per band; RMS envelope. `stretch()` resamples features by the speed ratio |
| `xcorr.py` | `Matcher`: normalized cross-correlation of windows against the whole target. Coarse: decimated ×4 and projected on the target's top 12 principal components, top 5 peaks. Fine: exact NCC within ±12 frames (per-band `np.correlate`), parabolic sub-frame step |
| `ratio.py` | standard ratio candidates (frame-rate quotient first), flatness and drift scoring, free-form fallback (used only when clearly flatter than the best standard ratio) |
| `levels.py` | tight clusters (`find_levels`), crisp-vs-loose pruning (`prune_levels`), `viterbi`, `runs_of`, `changepoint` (with a dropped-material model for down-jumps) |
| `video.py` | `VideoReader` decodes short ranges as 32×18 gray frames; black runs, scene cuts, time-based frame matching |
| `cuts.py` | `localize()`: changepoint range, then video black → scene cut → silence → audio, with evidence and confidence |
| `analyze.py` | `analyze()` orchestrates everything into a `Plan`; a second look at the donor's unmatched head and tail with 4 s windows finds short end segments (a closing jingle) |
| `plan.py` | `Plan` / `Segment` / `Cut` / `Evidence` / `Gap`, JSON save/load, the text summary |
| `render.py` | streaming render (`_Tape` readers, raised-cosine crossfades), encode, mux |
| `verify.py` | local ±1 s residuals of the rendered track, crisp non-zero clusters (= out of sync; sparser agreement counts outside the dialogue band), coverage |
| `report.py` | self-contained HTML with inline SVG |
| `cli.py` | argparse front end; progress on stderr, summaries on stdout |

Time convention everywhere: **target time = donor time × ratio + offset**. "Stretched" donor time means donor time ×
ratio. Feature frame *i* is centered at `T0 + i / 50` s. A plan's segments are authoritative by `a_start`, `a_end`,
`offset`; `b_start` / `b_end` are recomputed on load.

## Invariants (each one fixed a real bug)

- **Snap the ratio.** Score standard ratios by flatness; never average a free-form estimate when one fits.
- **Crispness over count.** A level is chosen by how tightly its windows agree, never by window count or a mean over
  all windows: the dubbed dialogue forms a loose cluster 0.2–0.3 s from the M&E level, sometimes a bigger one.
- **Video never overrules crispness.** The audio-video check only settles contests between two crisp (or two loose)
  levels, and only crisp levels vote on what the A/V difference should be: on a real release the video sided with loose
  clusters in three segments.
- **Snapping beats a noisy drift.** Drift measured between a few sparse windows is noisy (2 ms over 40 s is 5×10⁻⁵);
  keep the standard ratio unless the free-form estimate has clearly more flat windows.
- **No limits on jump size or segment length**, and cuts are placed by evidence, never guessed in matchless gaps.
- **Resample before `asetrate`.** `asetrate` alone assumes the input already runs at the target rate.
- **No input seeking for audio placement.** ffmpeg `-ss` before `-i` is not sample-accurate for AAC in MP4 (21 ms
  off); `render._Tape` decodes sequentially and restarts only to go backwards.
- **AAC goes into Matroska audio (`.mka`), never `.m4a`, before mkvmerge**: mkvmerge ignores the MP4 edit list
  hiding the encoder priming and adds 21 ms of delay.
- **Skip AAC priming that ffmpeg leaves in.** ffmpeg 8 decodes the 1024 priming samples of AAC in Matroska (and
  reports a start time of -21 ms) where ffmpeg 9 drops them: every decode passes
  `AudioStream.untrimmed_priming` as `skip`, start times come from `AudioStream.first_sample_time`, video seeks
  subtract `MediaInfo.start_time`, and the ffmpeg fallback mux uses `-copyts`. CI runs ffmpeg 8 and catches a miss.
- **Final offsets come from audio.** Video only places cuts and settles contested levels: the donor's own A/V sync can
  be off by 0.2–0.36 s and change mid-file.
- Never re-encode video; never touch the inputs; `render()` leaves no partial output behind.

## Conventions

- Dataclasses for structured data; `pathlib.Path`; type hints on every signature
- `logging.getLogger(__name__)`; the CLI prints summaries to stdout, everything else is logging
- US English everywhere: code, identifiers, comments, docs, CLI messages, the changelog and commit messages
  (*analyze*, *color*, *behavior*, *license*, *normalize*, *canceled*)
- Comments only for a non-obvious why; public library functions get a short docstring with units and `:param:`
  fields where they help

## Commits and changelog

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `<type>[(scope)]: <description>`,
  types `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `chore`, `style`; the scope is the module
  (`render`, `levels`, `cli`, ...)
- `CHANGELOG.md` follows [Common Changelog](https://common-changelog.org/), with unreleased work under
  `## [Unreleased]`; groups `Changed`, `Added`, `Removed`, `Fixed`, in that order
- Releases are tagged `vX.Y.Z`; `project.version` in `pyproject.toml` is the single source of truth
- Pushing a `vX.Y.Z` tag runs `.github/workflows/release.yml`: it checks the tag against `project.version` and
  creates the GitHub release with that version's `CHANGELOG.md` section as notes; `/release` prepares all of it
