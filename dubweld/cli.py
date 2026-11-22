"""The `dubweld` command: analyze, render, verify, run."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from dubweld import __version__
from dubweld.analyze import AnalyzeOptions, analyze
from dubweld.cache import default_cache_dir
from dubweld.errors import DubweldError
from dubweld.ffmpeg import Tools
from dubweld.plan import Plan
from dubweld.ratio import parse_ratio
from dubweld.render import CODECS, RenderOptions, check_output, render
from dubweld.report import write_report
from dubweld.verify import VerifyReport, verify

log = logging.getLogger("dubweld")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_VERIFY_FAILED = 3
EXIT_INTERRUPTED = 130


class Console:
    """One progress line on stderr that log records end before printing."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._open = False
        self._last = 0.0

    def progress(self, step: str, fraction: float) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if fraction < 1.0 and now - self._last < 0.2:
            return
        self._last = now
        sys.stderr.write(f"\r{step}: {fraction * 100:5.1f} %")
        sys.stderr.flush()
        self._open = True
        if fraction >= 1.0:
            self.end_line()

    def step(self, name: str) -> Callable[[float], None]:
        return lambda f: self.progress(name, f)

    def end_line(self) -> None:
        if self._open:
            sys.stderr.write("\n")
            self._open = False


class _Handler(logging.StreamHandler):
    def __init__(self, console: Console) -> None:
        super().__init__(sys.stderr)
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        self.console.end_line()
        super().emit(record)


def _track(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("track numbers start at 0")
    return value


def _ratio(text: str) -> float | None:
    try:
        return parse_ratio(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _language(text: str) -> str:
    if not (text.isalpha() and len(text) in (2, 3)) and not ("-" in text and text.replace("-", "").isalnum()):
        raise argparse.ArgumentTypeError(f"{text!r} is not a language code (ISO 639-2 like cze, or BCP 47 like cs)")
    return text


def _analyze_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("analysis")
    g.add_argument("--a-track", type=_track, default=0, metavar="N", help="audio stream of the target (default 0)")
    g.add_argument("--b-track", type=_track, default=0, metavar="N", help="audio stream of the donor (default 0)")
    g.add_argument(
        "--ratio",
        type=_ratio,
        default=None,
        metavar="R",
        help="speed ratio donor->target: auto (default), pal (25/23.976), pal24 (25/24), ntsc (24/23.976), "
        "1, an -inv variant, a number or a fraction",
    )
    g.add_argument("--no-video", action="store_true", help="do not use the videos to place cuts")
    g.add_argument("--max-offset", type=float, metavar="S", help="only search offsets within ±S seconds (faster)")


def _render_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("render")
    g.add_argument("--lang", type=_language, default="und", help="language of the dub track (default und)")
    g.add_argument("--name", default="", help='track name of the dub (default "Dub")')
    g.add_argument(
        "--fill",
        choices=("silence", "original"),
        default="silence",
        help="what plays where the donor has nothing (default silence)",
    )
    g.add_argument("--codec", choices=tuple(CODECS), default="aac", help="dub codec (default aac)")
    g.add_argument("--bitrate", default="", help="dub bitrate (default 192k for aac, 128k for opus)")
    g.add_argument(
        "--default",
        choices=("original", "dub"),
        default="original",
        dest="default_track",
        help="which audio track is flagged default (default original)",
    )
    g.add_argument(
        "--pitch",
        choices=("shift", "keep"),
        default="shift",
        help="shift: resample, undoing a PAL pitch rise (default); keep: time-stretch",
    )
    g.add_argument("--crossfade", type=float, default=0.030, metavar="S", help="crossfade at joins (default 0.03)")
    g.add_argument("--force", action="store_true", help="overwrite the output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dubweld",
        description="Lay a dubbed audio track from one release of a video (the donor) onto another release (the "
        "target), lip-synced: speed, offsets and cuts are corrected automatically.",
    )
    parser.add_argument("--version", action="version", version=f"dubweld {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more log output (-vv for debug)")
    parser.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")
    parser.add_argument("--no-cache", action="store_true", help="do not cache decoded features")
    parser.add_argument("--cache-dir", type=Path, default=None, help=f"feature cache (default {default_cache_dir()})")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("analyze", help="work out the sync plan", description="Analyze and write a plan (JSON).")
    p.add_argument("target", type=Path, help="file whose video and timeline to keep (A)")
    p.add_argument("donor", type=Path, help="file with the dub (B)")
    p.add_argument("-o", "--output", type=Path, help="plan file (default <target>.dubweld.json)")
    p.add_argument("--report", type=Path, help="also write an HTML report here")
    _analyze_args(p)

    p = sub.add_parser(
        "render",
        help="render a plan into a new file",
        description="Render the dub from a plan and mux it into a copy of the target.",
    )
    p.add_argument("target", type=Path)
    p.add_argument("donor", type=Path)
    p.add_argument("plan", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True, help="output file (.mkv)")
    p.add_argument("--no-verify", action="store_true", help="skip the verification after rendering")
    p.add_argument("--report", type=Path, help="HTML report (default next to the output)")
    _render_args(p)

    p = sub.add_parser("verify", help="check a rendered file", description="Check a dub track against the target.")
    p.add_argument("target", type=Path)
    p.add_argument("output", type=Path, help="file holding the dub")
    p.add_argument("--a-track", type=_track, default=0, metavar="N", help="target audio stream (default 0)")
    p.add_argument(
        "--track",
        type=_track,
        default=None,
        metavar="N",
        help="audio stream of the dub in the output (default: the last one)",
    )

    p = sub.add_parser(
        "run",
        help="analyze, render and verify",
        description="Analyze, render and verify in one go; writes the plan and report next to the output.",
    )
    p.add_argument("target", type=Path)
    p.add_argument("donor", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True, help="output file (.mkv)")
    _analyze_args(p)
    _render_args(p)
    return parser


def _setup_logging(console: Console, verbose: int, quiet: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose >= 2 else logging.INFO)
    handler = _Handler(console)
    handler.setFormatter(logging.Formatter("%(message)s" if verbose == 0 else "%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("dubweld")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False


def _options(args: argparse.Namespace, cache_dir: Path | None) -> AnalyzeOptions:
    return AnalyzeOptions(
        a_track=args.a_track,
        b_track=args.b_track,
        ratio=args.ratio,
        use_video=not args.no_video,
        max_offset=args.max_offset,
        cache_dir=cache_dir,
    )


def _render_options(args: argparse.Namespace) -> RenderOptions:
    return RenderOptions(
        language=args.lang,
        name=args.name,
        fill=args.fill,
        codec=args.codec,
        bitrate=args.bitrate,
        default=args.default_track,
        pitch=args.pitch,
        crossfade=args.crossfade,
        force=args.force,
    )


def _check_inputs(*paths: Path) -> None:
    for p in paths:
        if not p.is_file():
            raise DubweldError(f"{p} does not exist")


def _verify_and_report(
    target: Path, output: Path, plan: Plan, a_track: int, cache_dir: Path | None, report_path: Path, console: Console
) -> VerifyReport:
    log.info("verifying %s", output.name)
    report = verify(target, output, a_track=a_track, cache_dir=cache_dir, progress=console.step("verify"))
    console.end_line()
    print(report.describe())
    write_report(report_path, plan, report)
    log.info("report: %s", report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console(enabled=not args.quiet and sys.stderr.isatty())
    _setup_logging(console, args.verbose, args.quiet)
    cache_dir = None if args.no_cache else (args.cache_dir or default_cache_dir())
    tools = Tools()
    try:
        if args.command == "analyze":
            _check_inputs(args.target, args.donor)
            plan = analyze(args.target, args.donor, _options(args, cache_dir), tools, console.progress)
            console.end_line()
            out = args.output or args.target.with_suffix(".dubweld.json")
            plan.save(out)
            print(plan.describe())
            log.info("plan: %s", out)
            if args.report:
                write_report(args.report, plan)
                log.info("report: %s", args.report)
            return EXIT_OK

        if args.command == "render":
            _check_inputs(args.target, args.donor, args.plan)
            check_output(args.output)
            plan = Plan.load(args.plan)
            for ref, path in ((plan.a, args.target), (plan.b, args.donor)):
                if ref.size and path.stat().st_size != ref.size:
                    log.warning("%s differs from the file the plan was made for (%s)", path.name, Path(ref.path).name)
            render(plan, args.target, args.donor, args.output, _render_options(args), tools, console.step("render"))
            console.end_line()
            log.info("wrote %s", args.output)
            if args.no_verify:
                return EXIT_OK
            report_path = args.report or args.output.with_suffix(".html")
            report = _verify_and_report(
                args.target, args.output, plan, plan.a.audio_index, cache_dir, report_path, console
            )
            return EXIT_VERIFY_FAILED if report.verdict == "fail" else EXIT_OK

        if args.command == "verify":
            _check_inputs(args.target, args.output)
            report = verify(
                args.target, args.output, args.a_track, args.track, tools, cache_dir, console.step("verify")
            )
            console.end_line()
            print(report.describe())
            return EXIT_VERIFY_FAILED if report.verdict == "fail" else EXIT_OK

        if args.command == "run":
            _check_inputs(args.target, args.donor)
            check_output(args.output)
            plan_path = args.output.with_suffix(".plan.json")
            for path in (args.output, plan_path):
                # A hand-edited plan is work too
                if path.exists() and not args.force:
                    raise DubweldError(f"{path} exists (use --force to overwrite)")
            plan = analyze(args.target, args.donor, _options(args, cache_dir), tools, console.progress)
            console.end_line()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            plan.save(plan_path)
            print(plan.describe())
            print()
            log.info("plan: %s", plan_path)
            render(plan, args.target, args.donor, args.output, _render_options(args), tools, console.step("render"))
            console.end_line()
            log.info("wrote %s", args.output)
            report = _verify_and_report(
                args.target, args.output, plan, args.a_track, cache_dir, args.output.with_suffix(".html"), console
            )
            return EXIT_VERIFY_FAILED if report.verdict == "fail" else EXIT_OK
    except KeyboardInterrupt:
        console.end_line()
        log.error("interrupted")
        return EXIT_INTERRUPTED
    except DubweldError as e:
        console.end_line()
        log.error("error: %s", e)
        return EXIT_FAILED
    parser.error(f"unknown command {args.command}")
    return EXIT_USAGE
