<p align="center">
  <img src="docs/assets/logo.png" alt="InfernuxEngine Logo" width="128" />
</p>

<h1 align="center">InfernuxEngine</h1>

<p align="center">
  <strong>Open-source game engine · C++17 / Vulkan · Python scripting · MIT licensed</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/version-0.1.0-orange.svg" alt="Version" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey.svg" alt="Platform" />
  <img src="https://img.shields.io/badge/python-3.12+-brightgreen.svg" alt="Python" />
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++ 17" />
  <img src="https://img.shields.io/badge/graphics-Vulkan-red.svg" alt="Vulkan" />
</p>

<p align="center">
  <a href="README-zh.md">🇨🇳 中文文档</a> · <a href="#quick-start">Quick Start</a> · <a href="#architecture">Architecture</a> · <a href="https://chenlizheme.github.io/InfEngine/">Website</a>
</p>

---

## Overview

InfernuxEngine is a from-scratch game engine built with a **C++17 / Vulkan runtime** and a **Python scripting layer**. The C++ side handles rendering, physics, and resource management; the Python side handles gameplay logic, editor tools, and content workflows.

Key points:

- **C++17 / Vulkan core** — forward and deferred rendering, PBR, RenderGraph-based pipeline, Jolt physics
- **Python scripting** — Unity-style component model, hot-reload, editor extensions, access to the Python ecosystem
- **MIT licensed** — no royalties, no runtime fees, full source access

---

## Design Stance

### Scriptable rendering

The render pipeline is exposed through a RenderGraph API. You can author and modify render passes from Python instead of working against a closed editor pipeline.

### Fast iteration

Python handles not just gameplay code but also editor extensions, asset workflows, and tooling — keeping the inner loop short.

### No licensing surprises

MIT license. No royalties, no runtime fees, no vendor lock-in.

---

## Features

### Runtime foundation

- Vulkan forward and deferred rendering
- PBR materials, cascaded shadows, MSAA, post-processing
- RenderGraph-based pass orchestration
- Jolt physics (rigidbodies + colliders)
- Input, audio groundwork, scene and resource systems

### Python layer

- Unity-style component lifecycle
- `serialized_field` metadata for Inspector-driven authoring
- Decorators for component requirements and editor execution
- Hot-reload for scripts and content
- Full access to the Python ecosystem

### Editor

- Hierarchy, Inspector, Scene View, Game View, Console, Project panel
- Selection, gizmos, undo/redo, play-mode scene isolation

---

## Why the C++ / Python Split Matters

The engine is structured around a practical division of labor.

| Layer | Role |
|:------|:-----|
| C++17 / Vulkan | Renderer, scene systems, resources, physics, platform |
| pybind11 bridge | Bindings between native systems and the Python API |
| Python | Gameplay, editor logic, render authoring, tooling |

---

## Quick Start

### Prerequisites

| Dependency | Version |
|:-----------|:--------|
| Windows | 10 / 11 (64-bit) |
| Python | 3.12+ |
| Vulkan SDK | 1.3+ |
| CMake | 3.22+ |
| Visual Studio | 2022 (MSVC v143) |
| pybind11 | 2.11+ |

### Clone

```bash
git clone --recurse-submodules https://github.com/ChenlizheMe/InfEngine.git
cd InfEngine
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

### Build

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cmake --preset release
cmake --build --preset release
```

The build produces the native Python module, copies required runtime dependencies, and installs the package so `import InfEngine` works immediately inside the workspace environment.

### Run

```bash
python packaging/launcher.py
```

### Test

```bash
cd python
python -m pytest test/ -v
```

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

- Rendering and render-pipeline
- Python scripting and editor integration
- Core editor authoring loop
- Asset identification and project launcher
- Physics integration and scene interaction

### In progress

- Prefab workflows
- UI pipeline
- Animation systems
- Standalone build / export
- Fuller documentation, examples, and production path

---

## Roadmap

| Version | Focus |
|:--------|:------|
| v0.1 | **Current** — Scripting, rendering, physics, editor. Usable for basic games without animation |
| v0.2 | Prefab workflows, UI completion, asset rename improvements |
| v0.3 | Animation system, model/content pipeline |
| v0.4 | Standalone build, particles, terrain |
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
- GitHub: [https://github.com/ChenlizheMe/InfEngine](https://github.com/ChenlizheMe/InfEngine)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
