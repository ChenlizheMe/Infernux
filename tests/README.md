# Tests

Run commands from the repository root in `conda activate infernux` (Python 3.13).
See [CONTRIBUTING.md](../CONTRIBUTING.md) for build prerequisites.

## Native engine

Build the matching Editor and Player before testing. Use the Windows or Linux
CMake presets from `.github/workflows/ci.yml`; do not mix a source checkout with
an unrelated installed native module.

```sh
python -m pytest tests/python/test_shader_declaration_discovery.py -q -ra
python -m pytest tests/python tests/isolated --ignore=tests/python/test_shader_declaration_discovery.py -q -ra
```

Shader declaration discovery runs in its own process because it changes global
builtin shader declarations. `tests/python` shares a native Editor/renderer;
`tests/isolated` gives each lifetime or crash scenario its own native process.
Both require the current native runtime. Missing native binaries are setup errors.

For installed-wheel validation, clear `PYTHONPATH` and
`INFERNUX_NATIVE_MODULE_DIR` and pass `-o pythonpath=`. On headless Linux, use
Xvfb and the Vulkan configuration from the desktop workflow.

The retained suites focus on scene and asset roundtrips, Play/Stop, script
publication, Undo/Redo, GUID ownership, native lifetimes, JIT execution, rendering,
audio and physics. Add a focused regression for a distinct failure; prefer
extending a real workflow over another default snapshot or mock-only matrix.
The current API and file formats are authoritative; historical migrations do not
need a parallel test suite.

## Portable Hub

No compiled engine or Vulkan device is required:

```sh
python -m pip install pytest PySide6 packaging PyYAML
python -m pytest tests/hub tests/contracts -q -ra
```

Set `QT_QPA_PLATFORM=offscreen` without a display. Linux Qt also requires its
system EGL/OpenGL libraries. These suites retain installation, runtime publication,
permissions, project launch, updater and release identity workflows. Tests use
disposable projects; cleanup tests require Git and PowerShell 7.

## GPU and Player acceptance

CTest runs native C++ tests. Hosted desktop CI also runs the software Vulkan gate
and the native Python suite; software drivers are test dependencies, not shipped
renderer fallbacks.

```sh
python tests/gpu/run.py --list
```

Use `.github/workflows/platform-player.yml` for Windows, Linux, Android and Web
Player build and runtime commands. Work on a copy of
`tests/fixtures/multiplatform_player` under `out/`. Its acceptance harnesses verify
startup, input, rendered output and shutdown. Hardware audio/device acceptance
remains separate from software-driver tests.

Inspect pytest's `-ra` output: an unavailable platform, tool or device is not a
successful validation. Keep reports and temporary probes under ignored `out/` or
`dev/`, not alongside authored engine resources.
