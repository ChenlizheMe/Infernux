<p align="center"><img src="docs/assets/logo.png" width="112" alt="Infernux logo"></p>

<h1 align="center">Infernux · 熔炉</h1>
<p align="center"><a href="https://github.com/ChenlizheMe/Infernux/releases"><img src="https://img.shields.io/badge/version-0.4.1-orange.svg" alt="Published release 0.4.1"></a></p>
<p align="center"><strong>Build worlds. Give them intelligence.</strong><br>A Python-first engine advancing toward a Neural Network-Native Engine.</p>

<p align="center">
  <a href="README-zh.md">简体中文</a> ·
  <a href="https://infernux-engine.com/">Website</a> ·
  <a href="https://infernux-engine.com/start.html">Get started</a> ·
  <a href="https://infernux-engine.com/wiki.html">Documentation</a> ·
  <a href="https://infernux-engine.com/roadmap.html">Roadmap</a> ·
  <a href="https://infernux-engine.discourse.group/">Community</a>
</p>

Infernux is an open-source game engine for building playable worlds and bringing the Python computing ecosystem into them. Author gameplay, editor tools and rendering pipelines in Python. Let C++ handle the runtime, Vulkan power native graphics, and WebGPU bring your project to the browser.

Our destination is a **Neural Network-Native Engine (3N)**: an engine where models can observe a world, influence its simulation and become part of the game you ship. The route starts with a capable game engine, then connects world data, neural computation and authoring tools through explicit, inspectable interfaces.

**MIT licensed. Windows and Linux Editors. Windows, Linux, Android and Web Players.** Infernux is in active development; 041 is the current development line, the published release is **0.4.1**.

<img src="docs/assets/demo.png" width="1920" height="1032" alt="An Infernux editor scene containing 65,536 GameObjects">
*A real capture from the 0.3.4 showcase: 65,536 ordinary GameObjects, with a Python-authored RenderStack.*

## A world you can build today

### Python from the first component to your own tools

Write gameplay components with Inspector fields, lifecycle callbacks and coroutines. Assemble prefabs, edit multiple scenes, and iterate while the editor is running. Extend the editor with your own panels and nested menus. Project scripts and plugin components participate in the same transactional hot-reload workflow.

Python is also the home of render authoring, asset workflows and numerical code. CPU/GPU JIT and Taichi integration let project scripts run numerical workloads and connect their results to gameplay. Startup warmup and build preparation move supported compilation work out of the first interaction.

### Shape the look, own the pipeline

Build on PBR materials, lights, shadows, cameras and post-processing. Define a RenderStack in Python and let the native RenderGraph execute it. Work with Forward, Forward+ and Deferred rendering, imported models and skeletal animation, GPU particles, screen UI and world-space text.

Vulkan and WebGPU share engine-facing rendering interfaces. Platform capabilities still matter: a browser has different compute and Python-extension constraints from a desktop or Android device. Target preparation belongs to the engine and its platform plugins, so game projects can share their authoring code.

### Build a game across scenes

Load, unload and edit scenes together; move objects between them and keep persistent objects alive. Bring in FBX and Blender content, split meshes, assign materials and textures, and synchronize external source changes. Add Jolt rigid bodies, collision callbacks, animation, audio and interactive UI. Inspect the result through Scene/Game views, Gizmos, the Hierarchy and the Console.

Assets keep their **GUID identity** throughout import, editing and cooking. Player builds consume a cooked asset index and packed content, carrying the resources the game needs without depending on the editor's project layout.

### Make the editor your own

InxPackage plugins can contain components, tools, assets and platform exporters. Keep gameplay in `runtime/`, authoring tools in `editor/`, and general files alongside them. Add localized panels and menus, document the plugin inside the editor through `plugin_pages/`, and package a folder as `.inxpkg` from the Project/File Manager context menu. Local folders can provide `inx_package.json` metadata; otherwise the exporter generates it. The GitHub template places distributable content in `package/`, with a standalone `package.py` packager outside it.

The optional MCP plugin exposes editor operations to agents through the same command and undo paths used by the interface. Automate scene work, inspect logs and capture the actual viewport while keeping those actions observable.

## One project, several places to play

| Target | Editor | Player | Graphics |
| --- | --- | --- | --- |
| Windows x64 | Yes | Yes | Vulkan |
| Linux x86_64 | Yes | Yes | Vulkan |
| Android arm64/x86_64 | No | APK/AAB | Vulkan |
| Web | No | HTML/JS/WASM | WebGPU |

See [platform requirements and limitations](SUPPORT.md#platform-support) and the [platform support matrix](docs/platform-support.json) for target details.

Platform plugins declare their build options in the editor and supply their runtime payloads. InfernuxHub manages engine installations, Python environments and shared Android tools. Exported Players include runtime components and assets; editor tools stay with the editor.

## What 041 brings

The 041 development line deepens the everyday workflow: multi-scene editing, model import and source synchronization, CPU/GPU JIT preparation, physics queries, world UI, more reliable Player startup and scene switching, and a reworked plugin lifecycle.

Plugin work includes Player component registration, live updates to existing component behavior, cleanup of panels and callbacks, plugin localization, and platform-owned build settings. Read the full [English](UpdateLog.md) or [Chinese](UpdateLog-zh.md) changelog for the scope and upgrade notes.

## The road to 3N

The engine already provides Python authoring, native rendering and simulation, batch world-data APIs, compute integration and programmable editor tools. The next steps make those foundations serve neural systems directly:

| Direction | What it unlocks |
| --- | --- |
| Model workflows and portable inference | Develop with the Python ML ecosystem; ship the inference runtime each target needs. |
| Shared world schemas | Give tools and models a precise description of state and valid operations. |
| Snapshots, deltas and deterministic replay | Reproduce interactions, diagnose simulation and prepare repeatable learning workloads. |
| Batch worlds and a tensor data plane | Step many environments and exchange world state with models efficiently. |
| Governed tools and model packages | Distribute extensions with clear capabilities, ownership and lifecycle. |

**These are roadmap commitments; the complete neural training and deployment loop is still ahead.** The [roadmap](https://infernux-engine.com/roadmap.html) tracks the stages through 0.5.2. Correctness, observability and explicit platform behavior guide the implementation.

## Start building

Install [InfernuxHub](https://infernux-engine.com/download.html), choose an engine version and create a project. Hub manages the Python environment; you can start from the editor without setting up a native compiler. Continue with the [learning guides](https://infernux-engine.com/learn.html) and [API reference](https://infernux-engine.com/wiki/site/en/api/index.html).

For plugin authors: [authoring guide](https://infernux-engine.com/wiki/site/en/plugin-package-content.html) · [plugin template](https://github.com/ChenlizheMe/infernux_plugin_template).

### Build the engine from source

Windows needs Visual Studio 2022 with MSVC v143, CMake 3.25+, a Vulkan SDK and the Python 3.13 development environment:

```powershell
git clone --recurse-submodules https://github.com/ChenlizheMe/infernux.git
cd Infernux
./scripts/setup/configure_development.ps1
conda activate infernux
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
cmake --build --preset windows-msvc-install-wheel
python packaging/launcher.py
```

On Linux, install native prerequisites with `scripts/setup/install_linux_dependencies.sh`, run `bash scripts/setup/configure_development.sh`, activate `infernux`, and use the `linux-clang-release` configure/build presets followed by `linux-clang-install-wheel`. Presets select the matching configuration and install the packaged engine. See [CONTRIBUTING.md](CONTRIBUTING.md) for development details.

## Join the forge

Build a game, share a plugin, or show us a workflow that should feel better. Bring questions and demos to the [community](https://infernux-engine.discourse.group/); report reproducible bugs on [GitHub](https://github.com/ChenlizheMe/Infernux/issues). Contributions to the engine, documentation and examples are welcome.

Infernux is released under the [MIT license](LICENSE). Free code signing is provided by [SignPath.io](https://signpath.io/), with a certificate from [SignPath Foundation](https://signpath.org/). See the [code signing policy](CODE_SIGNING_POLICY.md).

## Citation

```bibtex
@software{chen2026infernux,
  author  = {Chen, Lizhe},
  title   = {Infernux},
  year    = {2026},
  version = {0.4.1},
  url     = {https://github.com/ChenlizheMe/Infernux}
}
```
