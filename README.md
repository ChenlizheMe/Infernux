<p align="center">
  <img src="docs/assets/logo.png" alt="Infernux logo" width="128" />
</p>

<h1 align="center">Infernux</h1>

<p align="center">
  <strong>Open-source game engine with a C++17 / Vulkan runtime and a Python production layer.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/version-0.1.1-orange.svg" alt="Version" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey.svg" alt="Platform" />
  <img src="https://img.shields.io/badge/python-3.12+-brightgreen.svg" alt="Python" />
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++ 17" />
  <img src="https://img.shields.io/badge/graphics-Vulkan-red.svg" alt="Vulkan" />
</p>

<p align="center">
  <a href="README-zh.md">中文文档</a> ·
  <a href="https://chenlizheme.github.io/Infernux/">Website</a> ·
  <a href="https://chenlizheme.github.io/Infernux/wiki.html">Docs</a> ·
  <a href="#quick-start">Quick Start</a>
</p>

---

## Overview

Infernux is a from-scratch engine for teams that want to own the runtime, the tools, and the iteration loop.

The project combines:

- a native C++17 / Vulkan runtime for rendering, scene systems, platform integration, and physics
- a Python layer for gameplay, editor tooling, content workflows, and render-stack authoring
- an MIT license with no royalties, runtime fees, or closed engine surface

The goal is straightforward: keep the hot path native, keep the authoring loop fast, and keep the architecture readable enough that you can extend it without fighting hidden engine policy.

## Why Infernux

- Scriptable rendering: the render pipeline is exposed through RenderGraph and RenderStack APIs instead of being locked behind editor-only configuration.
- Python where iteration matters: gameplay, editor extensions, asset workflows, and automation all live in the same scripting ecosystem.
- Repository-first development: the codebase is intended to be inspectable and modifiable, not treated as a black box.
- Honest scope: the current release is a Windows technical preview with a working runtime/editor core, not a finished multi-platform engine.

## Current Status

The project already has a usable foundation for real engine work:

- Vulkan forward and deferred rendering, PBR, cascaded shadows, MSAA, shader reflection, and post-processing.
- RenderGraph and RenderStack systems that can be authored from Python.
- Jolt physics integration with rigidbodies, colliders, queries, callbacks, and layer filtering.
- SDL3-based audio, scene/resource management, GUID-based assets, and dependency tracking.
- A built-in editor with Hierarchy, Inspector, Scene View, Game View, Project, Console, UI editor, build settings, and play-mode isolation.
- Python-side component lifecycle, coroutine utilities, prefab workflows, serialized fields, and hot-reload support.
- Game UI primitives including Canvas, Text, Image, Button, and event routing.
- Hub packaging flows for both a standalone bundle and a Windows installer.

Near-term roadmap priorities are animation, advanced UI controls, more onboarding material, and a stronger content pipeline.

## Architecture

The engine is split by responsibility rather than ideology.

| Layer | Responsibility |
|:------|:---------------|
| C++17 / Vulkan | Renderer, resource ownership, physics, scene systems, platform services |
| pybind11 bridge | Native bindings exposed to Python |
| Python | Gameplay, editor logic, tooling, automation, render authoring |

This division keeps performance-sensitive systems native while leaving day-to-day production code in a language that is faster to iterate on and easier to integrate with external tooling.

## Quick Start

### Prerequisites

<details>
<summary><b>Windows</b></summary>

| Dependency | Version |
|:-----------|:--------|
| Windows | 10 / 11 (64-bit) |
| Python | 3.12+ |
| Vulkan SDK | 1.3+ |
| CMake | 3.22+ |
| Visual Studio | 2022 (MSVC v143) |
| pybind11 | 2.11+ |

</details>

<details>
<summary><b>macOS</b></summary>

| Dependency | Version |
|:-----------|:--------|
| macOS | 12+ (Apple Silicon or Intel) |
| Python | 3.12+ |
| Vulkan SDK | 1.3+ (LunarG SDK with MoltenVK) |
| CMake | 3.22+ |
| Ninja | 1.10+ |
| Xcode Command Line Tools | Latest |
| pybind11 | 2.11+ |

Install the Vulkan SDK from <https://vulkan.lunarg.com/sdk/home> (macOS tab). After installation, source the setup script:

```bash
source ~/VulkanSDK/<version>/setup-env.sh
```

Install Ninja and CMake via Homebrew if not already present:

```bash
brew install cmake ninja
```

</details>

Use any Python 3.12 environment you prefer. The examples below use Conda because that is the most common workflow in this repository.

### Clone the repository

```bash
git clone --recurse-submodules https://github.com/ChenlizheMe/Infernux.git
cd Infernux
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Build the engine

<details>
<summary><b>Windows</b></summary>

```bash
conda create -n infengine python=3.12 -y
conda activate infengine
pip install -r requirements.txt
cmake --preset release
cmake --build --preset release
```

</details>

<details>
<summary><b>macOS</b></summary>

```bash
conda create -n infengine python=3.12 -y
conda activate infengine
pip install -r requirements.txt
cmake --preset release-macos
cmake --build --preset release-macos
```

</details>

This builds the native module, copies the required runtime dependencies, and installs the Python package into the active environment so `import Infernux` works directly from the workspace.

### Launch the Hub in development mode

```bash
conda activate infengine
python packaging/launcher.py
```

Development mode uses your current Python environment and local build outputs. It does not install the Hub's private runtime.

### Run tests

```bash
conda activate infengine
cd python
python -m pytest test/ -v
```

## Documentation

- Website: <https://chenlizheme.github.io/Infernux/>
- Documentation hub: <https://chenlizheme.github.io/Infernux/wiki.html>
- API reference: generated from the Python package and published as static HTML under `docs/wiki/site/`

To regenerate the API Markdown and HTML site locally:

```bash
conda activate infengine
python docs/wiki/generate_api_docs.py
python -m mkdocs build --clean -f docs/wiki/mkdocs.yml
```

The equivalent CMake targets are `generate_api_docs` and `build_wiki_html`.

## Packaging

There are two supported Hub distribution paths.

### Standalone bundle

```bash
cmake --build --preset packaging
```

This produces the portable PyInstaller output under `dist/Infernux Hub/`. It is useful for local validation and developer-facing distribution.

### Windows installer

```bash
cmake --build --preset packaging-installer
```

This produces the graphical Windows installer for the Hub. The installer flow downloads and stages the matching Python 3.12 runtime for the host architecture, then provisions project runtimes from that private Hub-managed base.

## Contributing

Bug reports, feature requests, and workflow feedback are useful right now. If you are filing an issue, include the engine version, environment details, reproduction steps, and whether the problem is in the native runtime, the Python layer, or packaging.

Contribution and support policies live in the repository community files:

- `CONTRIBUTING.md`
- `SECURITY.md`
- `SUPPORT.md`

## License

Infernux is released under the MIT License. See `LICENSE` for details.
- installs the selected Infernux version into each project's runtime

This means each project owns a complete, self-contained Python copy. It does not share a runtime with other projects.

---

## Architecture

```text
Python authoring layer
  -> editor panels, components, RenderGraph authoring, tooling, project workflows
  -> pybind11 binding seam
C++ engine core
  -> renderer, scene, resources, physics, audio, platform services
External stack
  -> Vulkan, SDL3, Jolt, ImGui, Assimp, GLM, glslang, VMA
```

### Practical flow

1. Author gameplay or rendering logic in Python.
2. Bind that logic to editor-visible data and scene objects.
3. Describe render passes through the RenderGraph API.
4. The native backend handles scheduling, memory, and GPU execution.

This is the main architectural promise of the engine: **high-level iteration without surrendering low-level ownership**.

---

## Status

### Working

- Vulkan rendering (forward + deferred), PBR, shadows, 8 post-processing effects
- Python scripting with hot-reload and editor integration
- Full editor (12 panels, gizmos, undo/redo, play mode)
- Jolt physics (rigidbodies, colliders, raycasting, collision layers)
- SDL3 audio with 3D spatialization
- Asset pipeline (GUID-based AssetDatabase, .meta files, dependency graph)
- Prefab system (save/instantiate/override tracking)
- Game UI system (Canvas, Text, Image, Button, event system)
- Standalone game build via Nuitka
- Hub launcher and Windows installer

### In progress

- Animation system (skeletal animation, state machines)
- Advanced UI controls (ScrollView, Slider, layout groups)
- Documentation, tutorials, and example projects

---

## Roadmap

| Version | Focus |
|:--------|:------|
| v0.1 | **Current** — Rendering, physics, audio, scripting, editor, prefabs, game UI, standalone build. Usable for basic games without animation |
| v0.2 | Animation system, advanced UI controls (ScrollView, Slider, layout), asset rename safety |
| v0.3 | Particles, terrain, model/content pipeline improvements |
| v0.4 | Networking foundations |
| v1.0 | Documentation, examples, production readiness |

---

## Contributing

1. Read the README and the docs site first.
2. Check the roadmap to understand current priorities.
3. Open an issue or discussion before pushing broad architectural changes.
4. Submit focused pull requests with a clear engineering goal.

This repository benefits most from contributions that preserve the core idea of the project: explicit architecture, short iteration loops, and a stack the team actually owns.

---

## Contact

- Email: [chenlizheme@outlook.com](mailto:chenlizheme@outlook.com)
- GitHub: [https://github.com/ChenlizheMe/Infernux](https://github.com/ChenlizheMe/Infernux)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
