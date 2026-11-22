"""Errors raised by the dubweld library; the input files are never modified."""


class DubweldError(Exception):
    """Base class of every error dubweld raises on purpose."""


class ProbeError(DubweldError):
    """A file could not be probed, or it lacks the requested stream."""


class ToolError(DubweldError):
    """An external tool (ffmpeg, ffprobe, mkvmerge) is missing or failed."""


class AnalysisError(DubweldError):
    """The donor could not be matched to the target."""


class PlanError(DubweldError):
    """A plan file is malformed or does not fit the given inputs."""


class RenderError(DubweldError):
    """The dub track could not be rendered or muxed."""
