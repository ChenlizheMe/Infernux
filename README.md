# Infernux

Infernux is an open-source game engine for building interactive worlds with Python authoring, a native C++ runtime, and Vulkan/WebGPU rendering. It is designed for projects that want gameplay code, tools, simulation, and future model workflows to live in one coherent engine.

## What it is

- Python-first gameplay, components, scenes, prefabs, UI, tools, and automation
- Native rendering, physics, audio, asset loading, and platform runtimes
- Vulkan and WebGPU paths with Windows, Linux, Android, and Web targets
- An editor workflow that keeps authoring, inspection, undo, and export connected
- An extensible plugin and package model for engine tools and project workflows
- A long-term direction toward engines where models can observe and act on world state safely

<p align="center"><a href="https://github.com/ChenlizheMe/Infernux/releases"><img src="https://img.shields.io/badge/version-0.4.1-orange.svg" alt="Published release 0.4.1"></a></p>

## Start here

- [Start guide](https://infernux-engine.com/start.html)
- [Learn](https://infernux-engine.com/learn.html)
- [API reference](https://infernux-engine.com/wiki/site/en/api/index.html)
- [Roadmap](https://infernux-engine.com/roadmap.html)
- [Community](https://infernux-engine.discourse.group/)

The quickest route is [InfernuxHub](https://infernux-engine.com/download.html): install Hub, create a project, and open it in the editor. The tutorials are the source of truth for the current authoring workflow.

Current published release: **0.4.1**. The roadmap describes capabilities and dependencies rather than turning every page into a version notice.

## Build from source

The repository contains the C++ runtime, Python package, editor, platform plugins, documentation site, and tests. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change. The regular checks live in [TESTING.md](TESTING.md).

```bash
git clone https://github.com/ChenlizheMe/Infernux.git
cd Infernux
cmake --preset linux-debug
cmake --build --preset linux-debug
```

Submodules are only needed for native engine builds. Website-only work can use the `docs/` tree without fetching them.

## Contributing

Issues and pull requests are welcome. Keep public APIs, tutorials, and examples aligned with the implementation. For visual changes, preserve readability, keyboard access, responsive layouts, and the documented workflow.

```toml
[project]
version = {0.4.1}
```

## License

Infernux is released under the [MIT License](LICENSE).
