# Changelog

## [Unreleased]

_Initial release._

### Added

- Add `dubweld analyze`, `render`, `verify` and `run`
- Snap the speed ratio to a standard one (PAL, 25/24, NTSC and inverses) by how flat the matches are
- Find offsets as crisp levels of the shared music and effects, ignoring the loose dubbed-dialogue match
- Place cuts by fades to black, scene cuts or silence, and record the evidence and confidence of each
- Support offsets going up or down, recurring levels, any jump size and any segment length
- Find short segments at the donor's start and end, such as a closing jingle after a final jump
- Render sample-accurately with crossfades, filling gaps with silence, or the target's own audio
- Mux the dub into a copy of the target without re-encoding anything else
- Verify the rendered dub automatically and list the stretches it could not check
- Write an editable JSON plan, and a self-contained HTML report
- Use dubweld as a Python library: `analyze`, `render`, `verify` and an editable `Plan`
- Map times between the two releases with `Plan.to_target` and `Plan.to_donor`, for example to retime subtitles
