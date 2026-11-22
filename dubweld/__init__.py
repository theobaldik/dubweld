"""dubweld: lay a dubbed audio track from one release of a video onto another release, lip-synced."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dubweld")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

from dubweld.analyze import AnalyzeOptions, analyze
from dubweld.errors import AnalysisError, DubweldError, PlanError, ProbeError, RenderError, ToolError
from dubweld.plan import Cut, Evidence, Gap, Plan, Segment
from dubweld.render import RenderOptions, render
from dubweld.verify import VerifyReport, verify

__all__ = [
    "AnalysisError",
    "AnalyzeOptions",
    "Cut",
    "DubweldError",
    "Evidence",
    "Gap",
    "Plan",
    "PlanError",
    "ProbeError",
    "RenderError",
    "RenderOptions",
    "Segment",
    "ToolError",
    "VerifyReport",
    "__version__",
    "analyze",
    "render",
    "verify",
]
