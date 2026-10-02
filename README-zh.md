# Infernux

Infernux 是一个开源游戏引擎：用 Python 编写玩法、组件、工具和模拟逻辑，用原生 C++ 运行时承载性能关键路径，并通过 Vulkan / WebGPU 把世界渲染出来。它希望把创作、运行、调试和未来的模型工作流放进同一套清晰的引擎体系。

## 它是什么

- Python 优先的玩法、组件、场景、Prefab、UI、工具和自动化
- 原生渲染、物理、音频、资源加载与多平台运行时
- Vulkan 与 WebGPU 渲染路径，覆盖 Windows、Linux、Android 和 Web
- 连贯的编辑器工作流：创作、检查、撤销和导出彼此衔接
- 面向引擎工具和项目工作流的插件与包体系
- 长期方向是让模型能够在安全边界内观察并作用于世界状态

<p align="center"><a href="https://github.com/ChenlizheMe/Infernux/releases"><img src="https://img.shields.io/badge/version-0.4.1-orange.svg" alt="当前公开版本 0.4.1"></a></p>

## 从这里开始

- [开始使用](https://infernux-engine.com/start.html)
- [学习教程](https://infernux-engine.com/learn.html)
- [API 参考](https://infernux-engine.com/wiki/site/zh/api/index.html)
- [路线图](https://infernux-engine.com/roadmap.html)
- [社区](https://infernux-engine.discourse.group/)

最快的方式是使用 [InfernuxHub](https://infernux-engine.com/download.html)：安装 Hub、创建项目，然后在编辑器中打开。当前的创作方式以教程为准。

当前公开版本为 **0.4.1**。网站路线图按能力和依赖组织，其他页面不会反复强调版本号。

## 从源码构建

仓库包含 C++ 运行时、Python 包、编辑器、多平台插件、文档网站和测试。提交改动前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，常用检查见 [TESTING.md](TESTING.md)。

```bash
git clone https://github.com/ChenlizheMe/Infernux.git
cd Infernux
cmake --preset linux-debug
cmake --build --preset linux-debug
```

只有构建原生引擎时才需要拉取 submodule。只做网站改动时可以直接使用 `docs/`，不必下载它们。

## 参与贡献

欢迎提交 Issue 和 Pull Request。请让公开 API、教程和示例与实现保持一致；网站改动需要同时照顾可读性、键盘访问、响应式布局和现有教程流程。

```toml
[project]
version = {0.4.1}
```

## 许可证

Infernux 使用 [MIT 许可证](LICENSE) 发布。
