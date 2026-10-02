<p align="center"><img src="docs/assets/logo.png" width="112" alt="Infernux logo"></p>

<h1 align="center">Infernux · 熔炉</h1>
<p align="center"><a href="https://github.com/ChenlizheMe/Infernux/releases"><img src="https://img.shields.io/badge/version-0.4.1-orange.svg" alt="Published release 0.4.1"></a></p>
<p align="center"><strong>Build worlds. Give them intelligence.</strong><br> A Python-first game engine moving toward a Neural Network-Native Engine.</p>

<p align="center">
  <a href="README-zh.md">简体中文</a> ·
  <a href="https://infernux-engine.com/">Website</a> ·
  <a href="https://github.com/ChenlizheMe/Infernux">Repository</a> ·
  <a href="https://infernux-engine.com/start.html">Get started</a> ·
  <a href="https://infernux-engine.com/wiki.html">Documentation</a> ·
  <a href="https://infernux-engine.com/roadmap.html">Roadmap</a> ·
  <a href="https://infernux-engine.discourse.group/">Community</a>
</p>

Infernux is an open-source engine for building playable worlds with Python. Use Python for gameplay, editor tools, asset workflows and render authoring; let a C++ runtime handle execution; and target native graphics through Vulkan or the browser through WebGPU.

The long-term direction is a **Neural Network-Native Engine (3N)**: worlds that models can observe, influence and learn from through explicit, inspectable interfaces. The current releases focus on a capable, extensible engine that makes that direction practical.

**MIT licensed. Windows and Linux editors; Windows, Linux, Android and Web players.** The published release is **0.4.1**; the **041** line is in active development.

<img src="docs/assets/demo.png" width="1920" height="1032" alt="An Infernux editor scene containing 65,536 GameObjects">

## Engine foundations

- **Python-first authoring** — Build gameplay components, scenes, tools and rendering pipelines in Python, with lifecycle callbacks, coroutines and live iteration.
- **Native runtime and rendering** — A C++ runtime executes Python-authored work. Vulkan powers desktop and Android players; WebGPU powers browser players.
- **Scenes, assets and simulation** — Work across scenes, import common model content, and combine materials, lighting, animation, physics, audio, particles and UI.
- **Extensible editor and plugins** — Add project tools and plugins without leaving the engine workflow. Optional integrations expose observable editor operations to automation and agents.
- **Compute-friendly workflows** — Use Python’s numerical ecosystem and supported CPU/GPU JIT paths alongside gameplay and world data.

## Supported targets

| Target | Editor | Player | Graphics |
| --- | --- | --- | --- |
| Windows x64 | Yes | Yes | Vulkan |
| Linux x86_64 | Yes | Yes | Vulkan |
| Android arm64/x86_64 | No | APK/AAB | Vulkan |
| Web | No | HTML/JS/WASM | WebGPU |

See the [platform requirements and limitations](SUPPORT.md#platform-support) and the [support matrix](docs/platform-support.json) for details.

## Current direction

The 041 development line expands multi-scene editing, model import and source synchronization, compute preparation, physics queries, world UI, player startup and plugin lifecycle reliability. The [roadmap](https://infernux-engine.com/roadmap.html) tracks the next stages through 0.5.2.

The 3N roadmap connects engine foundations to model workflows, shared world schemas, snapshots and replay, batch environments, tensor data exchange, and governed model/tool packages. Full neural training and deployment workflows are still ahead.

## Get started

The fastest path is [InfernuxHub](https://infernux-engine.com/download.html): install Hub, choose an engine version, create a project and launch the editor. Hub manages the project’s Python runtime for you.

Continue with the [learning guides](https://infernux-engine.com/learn.html) and [API reference](https://infernux-engine.com/wiki/site/en/api/index.html). Plugin authors can use the [authoring guide](https://infernux-engine.com/wiki/site/en/plugin-package-content.html) and [plugin template](https://github.com/ChenlizheMe/infernux_plugin_template).

### Build from source

For Windows, use Visual Studio 2022 with MSVC v143, CMake 3.25+, a Vulkan SDK and Python 3.13. For Linux, install native prerequisites with `scripts/setup/install_linux_dependencies.sh`. Then run the platform setup script, configure with the matching CMake preset, build, install the wheel and launch:

```text
git clone --recurse-submodules https://github.com/ChenlizheMe/Infernux.git
cd Infernux
# Windows: ./scripts/setup/configure_development.ps1
# Linux:   bash scripts/setup/configure_development.sh
# Activate the created `infernux` environment, then use the matching
# windows-msvc-release or linux-clang-release CMake presets.
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete development workflow.

## Join the forge

- Ask questions and share projects in the [community](https://infernux-engine.discourse.group/).
- Report reproducible bugs on [GitHub Issues](https://github.com/ChenlizheMe/Infernux/issues).
- Read the [changelog](UpdateLog.md), [support guide](SUPPORT.md) and [security policy](SECURITY.md).
- Contributions to the engine, documentation and examples are welcome.

Infernux is released under the [MIT license](LICENSE). Code signing is provided through [SignPath.io](https://signpath.io/); see the [code signing policy](CODE_SIGNING_POLICY.md).

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
