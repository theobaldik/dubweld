"""The sync plan: speed ratio, segments, cuts and gaps; saved as editable JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dubweld.errors import PlanError

log = logging.getLogger(__name__)

PLAN_VERSION = 1


def mmss(seconds: float, decimals: int = 1) -> str:
    """`75.25` -> `1:15.3`; hours are added past 60 minutes (`1:02:03.0`)."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    total = round(seconds, decimals)
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    width = 3 + decimals if decimals else 2
    sec = f"{s:0{width}.{decimals}f}"
    return f"{sign}{int(h)}:{int(m):02d}:{sec}" if h else f"{sign}{int(m)}:{sec}"


@dataclass
class InputRef:
    """
    One input file as the analysis saw it; `render` uses only `audio_index` and `duration`.

    :param path: the file's path when it was analyzed
    :param audio_index: the audio stream used, counted among the file's audio streams
    :param duration: the file's duration, seconds
    :param size: file size in bytes, to notice a different file at render time
    :param mtime: modification time (POSIX seconds), for the same reason
    :param fps: video frame rate as a fraction (`24000/1001`), empty without video
    :param sample_rate: sample rate of the audio stream, Hz
    :param channels: channel count of the audio stream
    """

    path: str
    audio_index: int
    duration: float
    size: int = 0
    mtime: float = 0.0
    fps: str = ""
    sample_rate: int = 0
    channels: int = 0


@dataclass
class Segment:
    """
    A stretch of the target filled from the donor: target time t plays donor time (t - offset) / ratio.

    Only `target_start`, `target_end` and `offset` drive the render; edit those by hand to fix a plan.

    :param target_start: where the segment starts, target seconds
    :param target_end: where it ends, target seconds
    :param offset: seconds added to the stretched donor time to land on the target
    :param donor_start: where it starts in the donor, donor seconds (recomputed by `Plan.normalize`)
    :param donor_end: where it ends in the donor, donor seconds (recomputed by `Plan.normalize`)
    :param windows: how many analysis windows matched at this offset
    :param video_offset: the offset the videos agree on, or None when not checked; a difference from `offset` is
        the donor's own audio-video sync error
    """

    target_start: float
    target_end: float
    offset: float
    donor_start: float = 0.0
    donor_end: float = 0.0
    windows: int = 0
    video_offset: float | None = None


@dataclass
class Evidence:
    """
    One clue to where a cut is.

    :param kind: `black` (a fade to black in the donor's video), `scene` (a scene cut), `silence`, `audio` (where the
        matches change) or `midpoint` (no clue: the middle of the search range)
    :param target_time: where the clue puts the cut, target seconds
    :param detail: a human-readable description
    """

    kind: str
    target_time: float
    detail: str = ""


@dataclass
class Cut:
    """
    A change of offset between two segments.

    :param target_time: where the earlier segment ends, target seconds
    :param donor_time: the same point in stretched donor time (donor seconds × ratio)
    :param offset_before: offset of the earlier segment, seconds
    :param offset_after: offset of the later segment, seconds
    :param confidence: `high` (video black), `medium` (scene cut, silence, a sharp audio change) or `low` (a guess)
    :param evidence: the clues found, strongest first
    :param search: the target-time range searched for the cut, seconds
    """

    target_time: float
    donor_time: float
    offset_before: float
    offset_after: float
    confidence: str
    evidence: list[Evidence] = field(default_factory=list)
    search: tuple[float, float] = (0.0, 0.0)

    @property
    def jump(self) -> float:
        """
        Change of offset, seconds: positive where the target has extra material (left as a gap), negative where the
        donor has (dropped).
        """
        return self.offset_after - self.offset_before


@dataclass
class Gap:
    """
    Target time with no donor audio, filled with silence or the target's own audio.

    :param target_start: target seconds
    :param target_end: target seconds
    """

    target_start: float
    target_end: float


@dataclass
class Plan:
    """
    How to lay the donor's audio onto the target: target time = donor time × `ratio` + the segment's offset.

    Everything `render` needs, plus what the analysis saw for reports. Save it with `save` to edit it by hand.

    :param target: the file whose video and timeline are kept
    :param donor: the file the dub comes from
    :param ratio: donor-to-target speed ratio (25/23.976 for a PAL speed-up: the donor runs faster)
    :param ratio_name: the standard ratio's name (`pal`, `ntsc`, ...), `estimated`, or the forced ratio's value
    :param ratio_method: how the ratio was found: `snapped`, `forced` or `estimated`
    :param segments: the donor stretches to play, sorted by target time and not overlapping
    :param cuts: the offset changes between segments
    :param warnings: things to listen to or check by hand
    :param levels: the offset clusters the analysis found (informational, for the report)
    :param candidates: the speed ratios tried and their scores (informational)
    :param windows: per analysis window: `times` (stretched donor seconds), `offsets` (seconds) and `scores`
        (normalized correlation); informational
    :param version: plan format version
    """

    target: InputRef
    donor: InputRef
    ratio: float
    ratio_name: str = ""
    ratio_method: str = ""
    segments: list[Segment] = field(default_factory=list)
    cuts: list[Cut] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    levels: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    windows: dict[str, list[float]] = field(default_factory=dict)
    version: int = PLAN_VERSION

    def gaps(self, min_length: float = 0.005) -> list[Gap]:
        """
        Target time not covered by any segment, from 0 to the target's end.

        :param min_length: ignore gaps shorter than this, seconds
        """
        out: list[Gap] = []
        t = 0.0
        for seg in sorted(self.segments, key=lambda s: s.target_start):
            if seg.target_start - t >= min_length:
                out.append(Gap(t, seg.target_start))
            t = max(t, seg.target_end)
        if self.target.duration - t >= min_length:
            out.append(Gap(t, self.target.duration))
        return out

    def normalize(self) -> None:
        """Sort segments, trim overlaps (a later segment wins) and recompute the donor spans."""
        self.segments.sort(key=lambda s: s.target_start)
        for prev, nxt in zip(self.segments, self.segments[1:]):
            if prev.target_end > nxt.target_start:
                log.warning("segments overlap at %s; trimming the earlier one", mmss(nxt.target_start))
                prev.target_end = nxt.target_start
        self.segments = [s for s in self.segments if s.target_end > s.target_start]
        for seg in self.segments:
            seg.donor_start = (seg.target_start - seg.offset) / self.ratio
            seg.donor_end = (seg.target_end - seg.offset) / self.ratio

    def validate(self) -> None:
        """
        Check the plan is renderable.

        :raises PlanError: on a non-positive ratio, an empty plan or a segment reaching before the donor's start
        """
        if not self.ratio > 0:
            raise PlanError(f"invalid ratio {self.ratio}")
        if not self.segments:
            raise PlanError("the plan has no segments")
        for seg in self.segments:
            if seg.target_end <= seg.target_start:
                raise PlanError(f"segment at {mmss(seg.target_start)} ends before it starts")
            if (seg.target_start - seg.offset) / self.ratio < -0.01:
                raise PlanError(f"segment at {mmss(seg.target_start)} starts before the donor does")

    def to_target(self, donor_time: float) -> float | None:
        """
        Map a donor time onto the target, e.g. to retime the donor's subtitles.

        :param donor_time: donor seconds
        :return: target seconds, or None where the donor's material isn't played (dropped at a cut, or outside the
            segments)
        """
        for seg in self.segments:
            if seg.donor_start <= donor_time < seg.donor_end:
                return donor_time * self.ratio + seg.offset
        return None

    def to_donor(self, target_time: float) -> float | None:
        """
        Map a target time onto the donor.

        :param target_time: target seconds
        :return: donor seconds, or None in a gap without donor audio
        """
        for seg in self.segments:
            if seg.target_start <= target_time < seg.target_end:
                return (target_time - seg.offset) / self.ratio
        return None

    def to_dict(self) -> dict[str, Any]:
        """The plan as JSON-ready data, with the gaps added for readers of the file."""
        d = asdict(self)
        d["gaps"] = [asdict(g) for g in self.gaps()]
        return d

    def save(self, path: Path) -> None:
        """
        Write the plan as JSON; the long per-window arrays go on single lines so the rest stays editable.

        :param path: the JSON file to write
        """
        d = self.to_dict()
        windows = d.pop("windows")
        d = _round(d)
        d["ratio"] = self.ratio
        d["target"]["mtime"], d["donor"]["mtime"] = self.target.mtime, self.donor.mtime
        text = json.dumps(d, indent=2, ensure_ascii=False)
        if windows:
            compact = ",\n".join(
                f'    "{k}": {json.dumps(_round(v, 4), separators=(",", ":"))}' for k, v in windows.items()
            )
            text = text[: text.rfind("}")].rstrip() + ',\n  "windows": {\n' + compact + "\n  }\n}"
        Path(path).write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Plan:
        """
        Read a plan written by `save` (and possibly edited by hand); normalizes and validates it.

        :param path: the JSON file to read
        :raises PlanError: when the file is not a valid plan
        """
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise PlanError(f"cannot read plan {path}: {e}") from e
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Plan:
        """
        Build a plan from data like `to_dict`'s; normalizes and validates it.

        :param d: the parsed JSON
        :raises PlanError: when the data is not a valid plan
        """
        try:
            if int(d.get("version", 0)) > PLAN_VERSION:
                raise PlanError(f"plan version {d['version']} is newer than this dubweld understands")
            plan = cls(
                target=InputRef(**d["target"]),
                donor=InputRef(**d["donor"]),
                ratio=float(d["ratio"]),
                ratio_name=d.get("ratio_name", ""),
                ratio_method=d.get("ratio_method", ""),
                segments=[Segment(**s) for s in d["segments"]],
                cuts=[
                    Cut(
                        **{k: v for k, v in c.items() if k not in ("evidence", "search")},
                        evidence=[Evidence(**e) for e in c.get("evidence", [])],
                        search=tuple(c.get("search", (0.0, 0.0))),
                    )
                    for c in d.get("cuts", [])
                ],
                warnings=list(d.get("warnings", [])),
                levels=list(d.get("levels", [])),
                candidates=list(d.get("candidates", [])),
                windows=dict(d.get("windows", {})),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise PlanError(f"malformed plan: {e}") from e
        plan.normalize()
        plan.validate()
        return plan

    def describe(self) -> str:
        """Human-readable summary: ratio, segments and cuts with target timestamps to listen at."""
        lines = [
            f"speed ratio {self.ratio:.10f} ({self.ratio_name}, {self.ratio_method})",
            "",
            "segments (target time):",
        ]
        for i, s in enumerate(self.segments, 1):
            lines.append(
                f"  {i:2d}  {mmss(s.target_start):>9} - {mmss(s.target_end):<9}  offset {s.offset:+9.4f} s"
                f"  (donor {mmss(s.donor_start)} - {mmss(s.donor_end)})"
            )
        if self.cuts:
            lines += ["", "cuts (listen around these):"]
        for c in self.cuts:
            if c.jump >= 0:
                what = f"target has {c.jump:.2f} s extra, filled"
            else:
                what = f"donor has {-c.jump:.2f} s extra, dropped"
            ev = c.evidence[0] if c.evidence else None
            how = f"{ev.kind}: {ev.detail}" if ev else "no evidence"
            lines.append(
                f"  {mmss(c.target_time):>9}  {c.offset_before:+.3f} -> {c.offset_after:+.3f}  {what}  [{c.confidence}; {how}]"
            )
        gaps = [g for g in self.gaps() if g.target_end - g.target_start >= 0.05]
        if gaps:
            lines += ["", "gaps without dub:"]
            lines += [
                f"  {mmss(g.target_start):>9} - {mmss(g.target_end):<9} ({g.target_end - g.target_start:.2f} s)"
                for g in gaps
            ]
        for w in self.warnings:
            lines += ["", f"WARNING: {w}"]
        return "\n".join(lines)


def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    return value
