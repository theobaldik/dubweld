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
    """One input file as the plan saw it."""

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

    `a_start`, `a_end` and `offset` drive the render; `b_start` / `b_end` (donor seconds) are informational and
    recomputed on load.
    """

    a_start: float
    a_end: float
    offset: float
    b_start: float = 0.0
    b_end: float = 0.0
    windows: int = 0
    video_offset: float | None = None


@dataclass
class Evidence:
    """One clue to where a cut is (`kind`: black, scene, silence, audio, midpoint); times in target seconds."""

    kind: str
    a_time: float
    detail: str = ""


@dataclass
class Cut:
    """
    A change of offset. `a_time` is the target time where the earlier segment ends; `b_time` the stretched donor
    time of the same point. Confidence is high (video black), medium (scene cut, silence, sharp audio) or low.
    """

    a_time: float
    b_time: float
    offset_before: float
    offset_after: float
    confidence: str
    evidence: list[Evidence] = field(default_factory=list)
    search: tuple[float, float] = (0.0, 0.0)

    @property
    def jump(self) -> float:
        return self.offset_after - self.offset_before


@dataclass
class Gap:
    """Target time with no donor audio (filled with silence or the target's own audio)."""

    a_start: float
    a_end: float


@dataclass
class Plan:
    """Everything `render` needs, plus what the analysis saw (levels, windows) for reports."""

    a: InputRef
    b: InputRef
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
        """Target time not covered by any segment, from 0 to the target's end."""
        out: list[Gap] = []
        t = 0.0
        for seg in sorted(self.segments, key=lambda s: s.a_start):
            if seg.a_start - t >= min_length:
                out.append(Gap(t, seg.a_start))
            t = max(t, seg.a_end)
        if self.a.duration - t >= min_length:
            out.append(Gap(t, self.a.duration))
        return out

    def normalize(self) -> None:
        """Sort segments, trim overlaps (a later segment wins) and recompute the donor spans."""
        self.segments.sort(key=lambda s: s.a_start)
        for prev, nxt in zip(self.segments, self.segments[1:]):
            if prev.a_end > nxt.a_start:
                log.warning("segments overlap at %s; trimming the earlier one", mmss(nxt.a_start))
                prev.a_end = nxt.a_start
        self.segments = [s for s in self.segments if s.a_end > s.a_start]
        for seg in self.segments:
            seg.b_start = (seg.a_start - seg.offset) / self.ratio
            seg.b_end = (seg.a_end - seg.offset) / self.ratio

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
            if seg.a_end <= seg.a_start:
                raise PlanError(f"segment at {mmss(seg.a_start)} ends before it starts")
            if (seg.a_start - seg.offset) / self.ratio < -0.01:
                raise PlanError(f"segment at {mmss(seg.a_start)} starts before the donor does")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gaps"] = [asdict(g) for g in self.gaps()]
        return d

    def save(self, path: Path) -> None:
        """Write the plan as JSON; the long per-window arrays go on single lines so the rest stays editable."""
        d = self.to_dict()
        windows = d.pop("windows")
        d = _round(d)
        d["ratio"] = self.ratio
        d["a"]["mtime"], d["b"]["mtime"] = self.a.mtime, self.b.mtime
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
        Read a plan written by `save` (and possibly edited by hand).

        :raises PlanError: when the file is not a valid plan
        """
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise PlanError(f"cannot read plan {path}: {e}") from e
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Plan:
        try:
            if int(d.get("version", 0)) > PLAN_VERSION:
                raise PlanError(f"plan version {d['version']} is newer than this dubweld understands")
            plan = cls(
                a=InputRef(**d["a"]),
                b=InputRef(**d["b"]),
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
                f"  {i:2d}  {mmss(s.a_start):>9} - {mmss(s.a_end):<9}  offset {s.offset:+9.4f} s"
                f"  (donor {mmss(s.b_start)} - {mmss(s.b_end)})"
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
                f"  {mmss(c.a_time):>9}  {c.offset_before:+.3f} -> {c.offset_after:+.3f}  {what}  [{c.confidence}; {how}]"
            )
        gaps = [g for g in self.gaps() if g.a_end - g.a_start >= 0.05]
        if gaps:
            lines += ["", "gaps without dub:"]
            lines += [f"  {mmss(g.a_start):>9} - {mmss(g.a_end):<9} ({g.a_end - g.a_start:.2f} s)" for g in gaps]
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
