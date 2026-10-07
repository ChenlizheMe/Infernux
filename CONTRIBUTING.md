# Contributing to Infernux

Thanks for contributing.

## Before you start

- Read the main `README.md` for project scope and current limitations.
- Search existing issues and discussions before opening a new thread.
- Keep changes focused. Mixed refactors and feature work are much harder to review in an engine codebase.

## Local setup

The repository provides one Conda environment definition for the supported
development ABI, Python 3.13:

```powershell
./scripts/setup/configure_development.ps1
conda activate infernux
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-install-wheel
```

The install preset builds the wheel and installs it into the Python environment
selected when CMake was configured. Activate `infernux` before configuring, and
close Editors and Players using that environment before replacing its package.
`windows-msvc-wheel` only builds the artifact; it does not install it.

To validate the installed package, leave `PYTHONPATH` and
`INFERNUX_NATIVE_MODULE_DIR` unset and disable the repository's source-path
setting for that test run:

```powershell
python -m pytest -o pythonpath= tests/python tests/contracts -q -ra
```

This uses the active interpreter's installation rather than `python/infernux`
in the checkout. Hub and source-contract tests retain their own source scopes.

On Ubuntu or Debian, run `scripts/setup/install_linux_dependencies.sh` once and
then `bash scripts/setup/configure_development.sh`. The setup scripts initialize
submodules and create or repair the `infernux` environment from
`environment.yml`. CMake intentionally rejects another Python minor version so
a local build cannot silently produce an incompatible wheel.

For Hub development:

```bash
conda activate infernux
python packaging/launcher.py
```

## Workspace output layout

Generated files have one canonical home:

- `out/build/<preset>/` contains CMake configure and build trees.
- `out/stage/<preset>/` contains disposable wheel and Hub assembly trees; verified wheels, Hub update archives, and installers are written to `dist/releases/<version>/`.
- `dist/releases/<version>/` contains final, upload-ready release assets only.

Do not create new top-level `build-*`, `release-*`, or package-output directories.
Run `./scripts/maintenance/clean_workspace.ps1` from PowerShell to remove all
generated output, including all local releases, scratch work, native staging,
plugin payloads, dependency build trees, and caches. Use `-WhatIf` to inspect
the paths first. Tracked source files are protected. Repository-level
automation is indexed in `scripts/README.md`; website-only tools remain under
`docs/tools/`.

For test-state cleanup without deleting native builds or acceptance reports,
stop running tests and use
`./scripts/maintenance/clean_workspace.ps1 -Scope TestArtifacts`.
This removes test bytecode, pytest caches, and generated fixture runtime data;
it preserves tracked fixtures and pending, untracked test sources. Preview it
with `-WhatIf` before running a new acceptance pass.

## What to include in a change

- A clear problem statement.
- The smallest practical implementation that solves it at the root cause.
- Updates to docs when public APIs, workflows, or user-facing behavior change.
- Validation notes in the PR describing what you built, ran, or manually verified.

## Validation expectations

The right validation depends on what you changed:

Start with the maintained [critical-workflow regression guide](tests/README.md).
It maps user workflows to their owning tests, exact commands, CI jobs, and
native/GPU prerequisites. Changes to a listed workflow must add or update a
regression in its owning suite; update the matrix when ownership changes.
Report unexpected skips as missing validation, not as a pass.

- Python API or tooling changes: run targeted Python tests or static validation.
- Native runtime changes: build the relevant CMake targets and describe runtime checks.
- Docs and website changes: regenerate generated docs when the API surface changed.

Documentation regeneration:

```bash
conda activate infernux
scripts\docs\update_api_docs.bat
```

## Pull request guidance

- Explain the problem first, then the implementation.
- Call out behavior changes, migration impact, and follow-up work explicitly.
- Include screenshots for editor, Hub, or website changes when relevant.
- If a change is intentionally incomplete, say so directly.

## Coding guidelines

- Preserve existing style within the touched area.
- Avoid unrelated cleanup unless it is required to make the change correct.
- Prefer explicit ownership and readable control flow over clever abstractions.
- Do not check in generated binaries or local environment artifacts.

## Discussions and questions

Use the [Infernux community](https://infernux-engine.discourse.group/) for open-ended design conversations or evaluation questions. Use Issues for actionable bugs, feature requests, and task-shaped work.
