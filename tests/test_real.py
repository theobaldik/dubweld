"""Real releases (never in the repo), one test case per set found in $DUBWELD_DATA or the repo root:

    target<id>.<ext>   the release to keep
    donor<id>.<ext>    the release with the dub
    synced<id>.<ext>   optional: a hand-made, checked result (target plus the dub as its last audio stream)

<id> is anything, the same for the files of one set (`target.mkv`, `target1.mkv`, `target-2.mp4`, ...); <ext> is any
container in `conftest.MEDIA_EXTENSIONS`.

Expected values can't follow arbitrary files, so the checks hold for any correct sync. The `synced` file is the ground
truth: a dub locked onto the loose dialogue match, a wrong ratio or a missed short segment all disagree with it by
0.2 s or more somewhere, which these checks catch.
"""

import os
from pathlib import Path

import pytest
from conftest import RealRelease, find_releases, needs_ffmpeg

from dubweld.analyze import AnalyzeOptions, analyze
from dubweld.plan import Plan
from dubweld.probe import probe
from dubweld.render import RenderOptions, render
from dubweld.verify import verify

DATA = Path(os.environ.get("DUBWELD_DATA", Path(__file__).resolve().parent.parent))
RELEASES = find_releases(DATA)

pytestmark = [
    pytest.mark.real,
    needs_ffmpeg,
    pytest.mark.skipif(not RELEASES, reason=f"no target<id>.* with donor<id>.* in {DATA}"),
]


@pytest.fixture(scope="module", params=RELEASES, ids=lambda r: r.target.stem)
def release(request: pytest.FixtureRequest) -> RealRelease:
    return request.param


@pytest.fixture(scope="module")
def plan(release: RealRelease, tmp_path_factory: pytest.TempPathFactory) -> Plan:
    return analyze(release.target, release.donor, AnalyzeOptions(cache_dir=tmp_path_factory.mktemp("cache")))


@pytest.fixture(scope="module")
def rendered(release: RealRelease, plan: Plan, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("render") / "synced.mkv"
    return render(plan, release.target, release.donor, out, RenderOptions(language="cze", name="Dub"))


def test_plan_is_consistent(plan: Plan) -> None:
    plan.validate()
    assert plan.segments
    assert len(plan.cuts) == len(plan.segments) - 1


def test_render_verifies_against_the_target(release: RealRelease, rendered: Path) -> None:
    report = verify(release.target, rendered)
    assert report.verdict == "pass", report.describe()
    assert report.median_ms <= 10
    assert report.max_ms <= 40


def test_render_matches_the_synced_reference(release: RealRelease, rendered: Path) -> None:
    if release.synced is None:
        pytest.skip(f"no synced{release.id}.*")
    dub_track = len(probe(release.synced).audio) - 1
    cross = verify(release.synced, rendered, a_track=dub_track)
    assert cross.verdict == "pass", cross.describe()
    assert cross.coverage > 0.9
    assert cross.median_ms <= 10
