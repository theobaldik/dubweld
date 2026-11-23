"""
Lay a dubbed audio track from one release of a video onto another release, lip-synced.

plan = analyze(target, donor); render(plan, target, donor, output); verify(target, output)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dubweld")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

from dubweld.analyze import AnalyzeOptions, ProgressCallback, analyze
from dubweld.errors import AnalysisError, DubweldError, PlanError, ProbeError, RenderError, ToolError
from dubweld.ffmpeg import Tools
from dubweld.plan import Cut, Evidence, Gap, InputRef, Plan, Segment
from dubweld.render import RenderOptions, render
from dubweld.report import write_report
from dubweld.verify import VerifyReport, verify

__all__ = [
    "AnalysisError",
    "AnalyzeOptions",
    "Cut",
    "DubweldError",
    "Evidence",
    "Gap",
    "InputRef",
    "Plan",
    "PlanError",
    "ProbeError",
    "ProgressCallback",
    "RenderError",
    "RenderOptions",
    "Segment",
    "ToolError",
    "Tools",
    "VerifyReport",
    "__version__",
    "analyze",
    "render",
    "verify",
    "write_report",
]
