# dubweld

Lay a dubbed audio track from one release of a video onto a different release of the same video, lip-synced.

You have the release you want to keep (the **target**: better picture, original audio) and another release that
has the dub you want (the **donor**). The two rarely line up: the donor may be a PAL transfer running 4 % fast,
start at a different point, and be cut differently, with longer or shorter fades at act breaks, or scenes added
or missing. A single offset or a single stretch can't fix that. dubweld finds the speed ratio, every offset and
every cut automatically. It renders the dub onto the target's timeline, muxes it in as an extra track, and then
checks its own result.

```console
$ dubweld run target.mkv donor.m4v -o synced.mkv --lang cze --name "Czech dub"
speed ratio 1.0427083333 (pal, snapped)

segments (target time):
   1     0:01.5 - 4:37.0     offset   +1.5443 s  (donor 0:00.0 - 4:24.1)
   2     4:39.4 - 11:37.9    offset   +4.0070 s  (donor 4:24.1 - 11:05.5)
   3    11:40.2 - 20:50.4    offset   +6.2702 s  (donor 11:05.5 - 19:53.1)
   4    20:52.7 - 21:36.9    offset   +8.6356 s  (donor 19:53.1 - 20:35.5)

cuts (listen around these):
     4:37.0  +1.544 -> +4.007  target has 2.46 s extra, filled  [high; black: donor fades to black 0.54 s at 4:36.5]
    11:37.9  +4.007 -> +6.270  target has 2.26 s extra, filled  [high; black: donor fades to black 0.75 s at 11:37.2]
    20:50.4  +6.270 -> +8.636  target has 2.37 s extra, filled  [high; black: donor fades to black 0.67 s at 20:49.7]

verification: PASS
  39 of 316 audible windows match at offset 0 (±40 ms); 18 confident ones: median 0.9 ms, max 3.1 ms, bias +0.8 ms
```

The output is a copy of the target with every track untouched (video is never re-encoded) plus the dub. It comes
with `synced.plan.json`, an editable plan, and `synced.html`, a report with the offset plot, the cuts and their
evidence, and the verification.

## How it works

A dub keeps the original **music and effects** (M&E): only the dialogue is replaced. dubweld matches the two
releases on that shared M&E, using language-agnostic band-energy features that ignore loudness and EQ differences
between the mixes. The matches are accurate to about a millisecond.

1. **Speed ratio.** dubweld tries the ratio implied by the frame rates first (25 vs 23.976 fps means a PAL speed-up),
   then the other standard ratios. It picks the one under which confident matches stay flat and don't drift (PAL
   and 25/24 differ by only 0.1%). It never settles for
   a loose estimate: being 3×10⁻⁴ off drifts 0.3 s over an episode. A free-form estimate is used only when no
   standard ratio fits, and dubweld warns loudly when that happens.
2. **Offsets.** It slides 8 s windows of the donor over the whole target (normalized cross-correlation, coarse to
   fine, sub-frame accurate), then looks for *levels*: tight clusters (±10 ms) of high-scoring windows. The dubbed
   dialogue roughly lines up with the original dialogue and forms a second, looser cluster 0.2–0.3 s away, sometimes
   with *more* windows than the true level. dubweld prefers crispness over window count; the video only breaks
   ties between equally crisp levels. A donor edit of a single frame (a 40 ms step) shows up as two levels, and
   neither makes the other look loose.
3. **Segments and cuts.** A Viterbi pass assigns every window to a level. Offsets may jump up (the target has extra
   material, left as a gap), jump down (the donor has extra material, which is dropped) or return to an earlier
   level. There is no limit on jump size or segment length: the donor's unmatched start and end get a second look
   with 4 s windows, which finds a closing jingle a few seconds long. A cut is first narrowed by an audio
   changepoint, then pinned by the strongest clue in that range: a fade to black in the donor's video, a scene cut,
   silence. Each cut records its evidence and confidence.
4. **Render.** The donor is resampled (not time-stretched), which also undoes a PAL speed-up's pitch rise. Each
   segment is placed at its offset with 30 ms crossfades, and the gaps are filled with silence or the target's own
   audio. The donor is decoded in one sequential pass, never seeked, so placement is sample-accurate.
5. **Verify.** The rendered track is re-matched against the target. dubweld reports the residual offsets, the stretches
   it could not check, and any part where several windows agree on a non-zero offset. Such a part fails the
   verification, unless it looks like the dialogue match in a segment that matches elsewhere: then it only warns.

## Requirements

- Python 3.11+ and numpy
- `ffmpeg` and `ffprobe`
- `mkvmerge` (MKVToolNix), recommended; without it dubweld muxes with ffmpeg

## Install

```console
uv tool install git+https://github.com/theobaldik/dubweld
# or
pipx install git+https://github.com/theobaldik/dubweld
```

## Usage

```
dubweld run     TARGET DONOR -o OUT.mkv [options]       # analyze + render + verify
dubweld analyze TARGET DONOR [-o plan.json] [--report plan.html] [analysis options]
dubweld render  TARGET DONOR plan.json -o OUT.mkv [render options]
dubweld verify  TARGET OUT.mkv [--a-track N] [--track N]
```

Analysis options:

| Option                       | Meaning                                                                                                                                          |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `--a-track N`, `--b-track N` | audio stream of the target / donor, counted among audio streams (default 0)                                                                      |
| `--ratio R`                  | `auto` (default), `pal` (25/23.976), `pal24` (25/24), `ntsc` (24/23.976), `1`, any of them with `-inv`, a number, or a fraction like `25/23.976` |
| `--no-video`                 | don't use the videos to place cuts                                                                                                               |
| `--max-offset S`             | only search offsets within ±S seconds (faster on long files)                                                                                     |

Render options:

| Option                                 | Meaning                                                                                                       |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `--lang`                               | language of the dub track (default `und`)                                                                     |
| `--name`                               | track name (default `Dub`)                                                                                    |
| `--fill silence\|original`             | what plays where the donor has nothing (default silence)                                                      |
| `--codec aac\|opus\|flac`, `--bitrate` | dub codec (default AAC 192k)                                                                                  |
| `--default original\|dub`              | which audio track is flagged default (default original)                                                       |
| `--pitch shift\|keep`                  | `shift` resamples (right for PAL donors); `keep` time-stretches, for donors that were already pitch-corrected |
| `--crossfade S`                        | crossfade at joins (default 0.03 s)                                                                           |
| `--force`                              | overwrite the output                                                                                          |

Global options: `-v` / `-vv` for more output, `-q` for less, `--no-cache` / `--cache-dir` for the feature cache (by
default `~/.cache/dubweld`, keyed by file size and modification time).

Exit codes: 0 ok, 1 error, 2 usage error, 3 verification failed, 130 interrupted.

### The listen-and-fix loop

1. Run `dubweld run`, or `dubweld analyze` followed by `dubweld render`.
2. Listen around the cuts it prints, and in the stretches the verification lists as unchecked.
3. If something is off, edit the plan. The segments' `a_start`, `a_end` and `offset` (target seconds) are what
   renders; everything else is informational. Then re-render with `dubweld render TARGET DONOR plan.json -o synced.mkv`.

## Limitations

- The releases must share the music and effects. A re-scored or remixed dub can't be matched.
- A segment shorter than about 10 s is hard to see with 8 s windows, except at the very start or end of the donor.
- Verification sees only the music and effects. In dialogue-heavy stretches it can't check anything, and it lists
  them; listen there.
- At a downward jump (the donor has extra material), the target is assumed continuous across the cut.
- `--pitch keep` uses ffmpeg's `rubberband` (or `atempo`). Their latency is not compensated, so check the
  verification.
- The output is always Matroska.

## Credits

The approach builds on the experience of syncing a real dub with
[DubSync Studio](https://github.com/bleronmorina/dubsync) (ideas only; no code is used). Related tools:
[risenxxx/dubsync](https://github.com/risenxxx/dubsync), [DubaLign](https://github.com/Arata12/DubaLign),
[AudioAlign](https://github.com/protyposis/AudioAlign).

## License

[MIT](LICENSE)
