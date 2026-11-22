"""The command line, driven through main() on a generated release."""

from pathlib import Path

import pytest
from conftest import Release, needs_ffmpeg

from dubweld.cli import EXIT_FAILED, EXIT_OK, main
from dubweld.plan import Plan

pytestmark = [needs_ffmpeg, pytest.mark.slow]


def test_run_analyze_render_verify(act_breaks: Release, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "synced.mkv"
    args = [
        "--no-cache",
        "run",
        str(act_breaks.a),
        str(act_breaks.b),
        "-o",
        str(out),
        "--ratio",
        "pal",
        "--lang",
        "cze",
    ]
    assert main(args) == EXIT_OK
    assert out.is_file() and out.with_suffix(".plan.json").is_file() and out.with_suffix(".html").is_file()
    text = capsys.readouterr().out
    assert "verification: PASS" in text and "cuts (listen around these)" in text

    plan_path = tmp_path / "synced.plan.json"
    assert (
        main(
            [
                "--no-cache",
                "analyze",
                str(act_breaks.a),
                str(act_breaks.b),
                "-o",
                str(plan_path),
                "--no-video",
                "--report",
                str(tmp_path / "synced.html"),
            ]
        )
        == EXIT_OK
    )
    assert len(Plan.load(plan_path).segments) == 4
    assert (tmp_path / "synced.html").read_text().startswith("<!doctype html>")

    out2 = tmp_path / "synced2.mkv"
    assert (
        main(
            [
                "--no-cache",
                "render",
                str(act_breaks.a),
                str(act_breaks.b),
                str(plan_path),
                "-o",
                str(out2),
                "--codec",
                "opus",
                "--no-verify",
            ]
        )
        == EXIT_OK
    )
    assert main(["--no-cache", "verify", str(act_breaks.a), str(out2)]) == EXIT_OK
    # Existing output without --force
    assert (
        main(
            [
                "--no-cache",
                "render",
                str(act_breaks.a),
                str(act_breaks.b),
                str(plan_path),
                "-o",
                str(out2),
                "--no-verify",
            ]
        )
        == EXIT_FAILED
    )


def test_missing_input(tmp_path: Path) -> None:
    assert main(["analyze", str(tmp_path / "target.mkv"), str(tmp_path / "donor.mkv")]) == EXIT_FAILED


@pytest.mark.parametrize("bad", [["--lang", "c z"], ["--ratio", "fast"], ["--fill", "noise"]])
def test_usage_errors(bad: list[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as e:
        main(["run", "target.mkv", "donor.mkv", "-o", str(tmp_path / "synced.mkv"), *bad])
    assert e.value.code == 2
