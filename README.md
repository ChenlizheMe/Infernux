<p align="center">
  <a href="https://infernux-engine.com/"><img src="docs/assets/logo.png" width="96" alt="Infernux logo"></a>
</p>

<h1 align="center">Infernux · 熔炉</h1>

<p align="center">
  <strong>Build worlds. Give them intelligence.</strong><br>
  An open-source game engine you write in Python, on its way to 3N: a <b>Neural Network-Native</b> engine.
</p>

<p align="center">
  <a href="https://github.com/ChenlizheMe/Infernux/releases"><img src="https://img.shields.io/badge/version-0.4.1-orange.svg" alt="Published release 0.4.1"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.13-3776AB.svg" alt="Python 3.13">
  <img src="https://img.shields.io/badge/graphics-Vulkan%20%C2%B7%20WebGPU-c8444b.svg" alt="Vulkan and WebGPU">
  <a href="https://github.com/ChenlizheMe/Infernux/actions/workflows/ci.yml"><img src="https://github.com/ChenlizheMe/Infernux/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="https://infernux-engine.com/">Website</a> ·
  <a href="https://infernux-engine.com/download.html">Download</a> ·
  <a href="https://infernux-engine.com/tutorials.html">Tutorials</a> ·
  <a href="https://infernux-engine.com/wiki/site/en/api/index.html">API</a> ·
  <a href="https://infernux-engine.com/roadmap.html">Roadmap</a> ·
  <a href="https://infernux-engine.discourse.group/">Community</a> ·
  <a href="README-zh.md">简体中文</a>
</p>

<p align="center">
  <img src=".github/media/space-battle.gif" width="600" alt="Infernux runtime: hundreds of ships and particle bursts driven by gameplay logic">
</p>

<p align="center">
  <img src=".github/media/npr-pipeline.gif" width="196" alt="A programmable render pipeline rebuilt at runtime">
  <img src=".github/media/rigid-coins.gif" width="196" alt="1,000 interacting rigid bodies with a custom bloom">
  <img src=".github/media/animated-cats.gif" width="196" alt="Cats driven by animation state machines and Timeline">
  <img src=".github/media/rendergraph-grid.gif" width="196" alt="RenderGraph composing different effects across split screens">
</p>
<p align="center"><sub>Real engine footage from the <a href="https://www.bilibili.com/video/BV1538P6jELT/?p=2">0.3.4 demo reel</a>. No concept art.</sub></p>

## What is Infernux?

Infernux is a game engine where **Python is where you work** and **C++ is where it runs**. You write gameplay, editor tools, render pipelines and numerical code in Python; a C++ runtime, a Vulkan renderer and Jolt physics do the heavy lifting. The same project ships to Windows, Linux, Android and the browser.

It is also built toward a specific goal: **3N, a Neural Network-Native engine**. In most engines, AI is a plugin bolted on after the game is finished. In a 3N engine the world is data a model can read, the simulation is a place where it can act, and the shipped game is somewhere it can run.

```python
import infernux as inx


class Spin(inx.InxComponent):
    speed: float = inx.serialized_field(default=90.0, range=(0.0, 360.0))

    def update(self, delta_time: float) -> None:
        self.transform.rotate(inx.Vector3(0.0, self.speed * delta_time, 0.0))
```

<sub>An ordinary Python class. `speed` appears in the Inspector and can be tuned while the game runs.</sub>

> [!WARNING]
> Infernux is **alpha software** under active development; the current release is **0.4.1**. APIs still change between releases, and the neural-network features below are roadmap items rather than shipped features. It is ready for experiments, tools and small games, not yet for a production schedule.

## Highlights

- **Python first.** Components with Inspector fields, coroutines and prefabs. Project scripts and plugins hot-reload transactionally, including the behaviour of instances that already exist.
- **A renderer you can author.** Describe a RenderStack in Python and the native RenderGraph runs it: Forward, Forward+ and Deferred paths, PBR, shadows, post-processing, GPU particle graphs, skinned animation, screen and world-space UI.
- **A complete game loop.** Multi-scene editing, FBX and Blender import with live source sync, Jolt rigid bodies and scene queries, audio, animation state machines and Timeline.
- **Compute in the game.** Batch NumPy access to world data, CPU/GPU JIT and Taichi integration for numerical gameplay. This is the data path models will use.
- **An editor you can extend.** Dockable panels and menus written in Python. InxPackage plugins carry components, tools, assets and platform exporters. An optional MCP plugin lets agents drive the editor through the same command and undo paths you use.
- **Built to be inspected.** Assets keep a GUID identity from import to cooked Player. Players load a packed asset index, never your project folder.

## Platforms

| Target | Editor | Player | Graphics |
| --- | :---: | :---: | --- |
| Windows x64 | ✅ | ✅ | Vulkan |
| Linux x86_64 | ✅ | ✅ | Vulkan |
| Android arm64 / x86_64 | — | APK / AAB | Vulkan |
| Web | — | HTML + WASM | WebGPU |

Requirements and known limits per platform are in [SUPPORT.md](SUPPORT.md#platform-support).

## Get started

**Use the engine.** Install [InfernuxHub](https://infernux-engine.com/download.html), pick an engine version and create a project. The Hub manages Python environments and Android tools for you; no native compiler is needed. Then follow the [five-step start](https://infernux-engine.com/start.html) to a component spinning a cube, and continue with the [courses](https://infernux-engine.com/learn.html).

**Build from source.** Windows needs a Visual Studio installation with the
Desktop C++ workload, CMake 3.25+, the Vulkan SDK and Python 3.13. The Windows
preset lets CMake select the newest compatible Visual Studio generator installed
on the machine.

```powershell
git clone --recurse-submodules https://github.com/ChenlizheMe/Infernux.git
cd Infernux
./scripts/setup/configure_development.ps1
conda activate infernux
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
cmake --build --preset windows-msvc-install-wheel
python packaging/launcher.py
```

If configuration cannot create the selected generator, install a CMake version
that lists a compatible Visual Studio generator in `cmake --help`, or set
`CMAKE_GENERATOR` to one of the listed generators and configure again.

On Linux, run `scripts/setup/install_linux_dependencies.sh` and `bash scripts/setup/configure_development.sh`, activate `infernux`, then use the `linux-clang-release` and `linux-clang-install-wheel` presets. [CONTRIBUTING.md](CONTRIBUTING.md) covers the full workflow.

## The road to 3N

| Stage | Status | What it unlocks |
| --- | --- | --- |
| Python authoring, batch world APIs, CPU/GPU compute | ✅ Available | Models and gameplay share one data path into the world |
| Deterministic stepping, snapshots and replay | 🔜 Planned | Reproducible runs for debugging and learning |
| Batched worlds and a tensor data plane | 🔜 Planned | Many environments stepping side by side, zero-glue data exchange |
| Portable inference in the Player | 🔜 Planned | Trained models ship inside the game on every target |

Ten interactive maps chart every system on the [roadmap](https://infernux-engine.com/roadmap.html). Release notes live in [UpdateLog.md](UpdateLog.md).

## Community

- 💬 Questions, demos and ideas: [community forum](https://infernux-engine.discourse.group/)
- 🐛 Reproducible bugs: [GitHub issues](https://github.com/ChenlizheMe/Infernux/issues)
- 🧩 Plugins: [authoring guide](https://infernux-engine.com/wiki/site/en/plugin-package-content.html) and [template](https://github.com/InfernuxEngine/infernux_plugin_template)
- 🛠️ Contributing: start with [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Infernux is released under the [MIT license](LICENSE): free to use, no royalties. Third-party notices are indexed in [licenses](licenses/README.md). Windows builds are signed free of charge by [SignPath.io](https://signpath.io/) with a certificate from the [SignPath Foundation](https://signpath.org/) ([policy](licenses/signing/CODE_SIGNING_POLICY.md)).

<details>
<summary>Citing Infernux</summary>

```bibtex
@software{chen2026infernux,
  author  = {Chen, Lizhe},
  title   = {Infernux},
  year    = {2026},
  version = {0.4.1},
  url     = {https://github.com/ChenlizheMe/Infernux}
}
```

</details>
