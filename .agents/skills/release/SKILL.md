---
name: release
description: Prepare the dubweld repo for a new release — fix outdated docs, run the tests, bump the version and turn every change since the last tag into the release's changelog section. Use when the user asks to prepare, cut or make a release, to bump the version for a release, or invokes /release (optionally with `patch`, `minor` or `major`). It only prepares the working tree; it never commits, tags, pushes or publishes.
---

# Prepare a release

Prepares the working tree for `chore: release X.Y.Z`. The user commits, tags and publishes; never run `git commit`,
`git tag`, `git push` or `uv publish` here. AGENTS.md holds the rules this follows (version
source, changelog format, commit types); read its *Commits and changelog* section first.

Work through the steps in order and stop at the first one that fails: a failure must leave no version bump behind.
The bump is the last thing that changes files.

## 1. Preflight

```bash
git status --short
git branch --show-current
git describe --tags --abbrev=0 --match 'v*'      # the last release reachable from HEAD
git log --oneline "$(git describe --tags --abbrev=0 --match 'v*')"..HEAD
```

- **Branch:** releases are cut from the default branch (`master`) only. On any other branch, stop.
- **No tag yet:** this is the first release. Its version is the one already in `pyproject.toml` (no bump in step 6
  unless the user wants a different one), and step 4's log is the whole history (`git log --reverse`).
- **No commits at all:** stop; there is no history to release (ask the user to commit first).
- **Nothing since the last tag:** stop, there is nothing to release.
- **Uncommitted changes:** they are not in the history, so step 4 cannot see them and the release would ship them
  undocumented. Ask whether to stop so the user commits them first (recommended) or to go on.
- **Remote tags:** if a remote exists, `git ls-remote --tags origin 'v*'` (read-only). A newer tag there than locally
  means the local repo is behind: stop and say so.

## 2. Check and fix outdated docs

Compare every doc with the code as it is now, not with what the commit messages claim. Fix what is outdated, in the
docs' own style and in American English. Never edit released changelog sections. Note each fix for the summary.

| Check | How |
|---|---|
| CLI flags and exit codes | `uv run dubweld --help` and `uv run dubweld <analyze\|render\|verify\|run> --help` against the option table, the usage examples and the exit-code line in `README.md`; the `EXIT_*` constants in `dubweld/cli.py` against that line |
| Example output | the sample `dubweld run` output in `README.md` still matches the format of `Plan.describe()` and `VerifyReport.describe()` (wording, columns), not necessarily the numbers |
| Library API | `__all__` in `dubweld/__init__.py` and the `DubweldError` subclasses in `dubweld/errors.py` against any library usage the docs describe |
| Features and limitations | each user-facing change since the last tag (step 4's log) reflected in `README.md` *How it works* / *Limitations* |
| Requirements | `requires-python` and the dependencies in `pyproject.toml` against *Requirements* / *Install* in `README.md` and AGENTS.md |
| AGENTS.md | every module of `dubweld/` in the architecture table, every test file it names existing, the suite's runtime roughly right (from step 3), and every backticked identifier it names still existing (`grep -rn` it in the code) |
| Pins | the `uv` version in `.github/workflows/tests.yml` and the `uv_build` range in `pyproject.toml` still fit the `uv` in use (`uv --version`); the CI Python matrix covers `requires-python`'s lower bound |
| Project URLs | `[project.urls]` in `pyproject.toml` and links in `README.md` point at `github.com/theobaldik/dubweld` |

If a doc is wrong in a way the code can't settle (is it the doc or the code that's wrong?), ask instead of guessing.

## 3. Run the checks

```bash
uv run pytest -rs                      # note which `real` tests ran or were skipped
black --check .
uv build -o "$SCRATCH/dist" && uvx twine check "$SCRATCH/dist"/*   # $SCRATCH: the session's scratchpad, never dist/ in the repo
```

All must pass. The `real` tests skip without the user's media (`target<id>.*`, `donor<id>.*` in `$DUBWELD_DATA` or the
repo root); say in the summary whether they ran. Don't fix a failing test as part of the release; report it and stop,
unless the fix is obviously part of this release's changes and the user agrees.

## 4. Decide the version

Read every commit since the last tag, the body too, and the diff where a subject is unclear:

```bash
git log --reverse --format='--- %h %s%n%b' "$(git describe --tags --abbrev=0 --match 'v*')"..HEAD
```

From the Conventional Commits:

- A `!` after the type or a `BREAKING CHANGE:` footer → **major**. While the version is `0.x`, a breaking change
  bumps the **minor** instead (semver allows it below 1.0); say so. For dubweld, breaking includes a CLI flag or exit
  code removed or changed in meaning, a plan JSON that older plans no longer load into, and a removed name from
  `__all__`.
- Otherwise, any `feat` → **minor**.
- Otherwise → **patch**.

Show the proposed version together with the commits that decide it, and confirm it with the user (AskUserQuestion,
recommended level first) unless they already named the level. The version is their call.

## 5. Write the changelog section

`CHANGELOG.md` follows Common Changelog with the AGENTS.md rules: groups `### Changed`, `### Added`, `### Removed`,
`### Fixed` in exactly that order; one imperative line per entry that makes sense without its heading; `**Breaking:**`
entries first in their group; no commit hashes, no `feat:` prefixes, no pasted commit subjects.

1. Start from the existing `## [Unreleased]` section and check every entry against the code. It may be stale.
2. Go through the commits from step 4. A user-facing change with no entry gets one; skip what users never see
   (`test`, `ci`, `style`, internal `refactor` / `build` / `chore`, docs-only changes unless they document something
   new). A change in sync accuracy, in what the plan JSON holds or in the report is user-facing.
3. A fix for something introduced **since the last tag** gets no `### Fixed` entry: the released version never had the
   bug. Fold it into the entry of the feature it fixes, if that changes its wording at all.
4. Merge entries that describe one change. Keep it short: a user reads it to decide whether to upgrade.
5. Rename the heading to `## [X.Y.Z] - YYYY-MM-DD` (plain version, today's date) and leave no empty `## [Unreleased]`
   behind. Keep a first release's `_Initial release._` line. Add
   `[X.Y.Z]: https://github.com/theobaldik/dubweld/releases/tag/vX.Y.Z` at the top of the link list at the end of the
   file (start the list if there is none).
6. Check that the section is what the release workflow will publish: `.github/workflows/release.yml` extracts
   everything from the heading up to the next `## ` heading or link definition.

## 6. Bump the version

`project.version` in `pyproject.toml` is the single source of truth (`dubweld.__version__` reads it from the installed
metadata); never edit it by hand:

```bash
uv version --bump <patch|minor|major> --dry-run
uv version --bump <patch|minor|major>          # also updates uv.lock
grep -rn --exclude=CHANGELOG.md --exclude=uv.lock --exclude-dir=.git --exclude-dir=.venv -F '<old version>' .   # only historical mentions may remain
uv lock --check
```

`git diff --stat` should show `pyproject.toml` and `uv.lock`, plus the changelog and step 2's doc fixes. Any other
file that pins the version is a bug: say so and fix it to read the version from the metadata instead.

## 7. Hand over

Summarize:

- the version and why it is that level
- the doc fixes from step 2, each in one line
- the check results (test count, which `real` tests ran, black, twine)
- the changelog section, in full, as it will appear on GitHub

Then give the commands for the user to run, without running any of them:

1. If step 2 changed docs, commit them first, as a separate `docs: ...` commit (stage those files only); then the
   rest as `chore: release X.Y.Z`. Both messages follow Conventional Commits, scope as in AGENTS.md.
2. `git tag vX.Y.Z && git push origin master vX.Y.Z`. The tag push makes `.github/workflows/release.yml` check that
   the tag matches `project.version` and create the GitHub release from the changelog section.
3. Only if the user publishes to PyPI: `uv publish "$SCRATCH/dist"/*` (rebuild first if anything changed after step 3).
