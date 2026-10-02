<p align="center"><img src="docs/assets/logo.png" width="112" alt="Infernux 标志"></p>

<h1 align="center">Infernux · 熔炉</h1>
<p align="center"><a href="https://github.com/ChenlizheMe/Infernux/releases"><img src="https://img.shields.io/badge/version-0.4.1-orange.svg" alt="Published release 0.4.1"></a></p>
<p align="center"><strong>创造世界，让智能成为世界的一部分。</strong><br>一款以 Python 为主要创作语言、面向神经网络原生方向发展的开源游戏引擎。</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://infernux-engine.com/">官网</a> ·
  <a href="https://github.com/ChenlizheMe/Infernux">仓库</a> ·
  <a href="https://infernux-engine.com/start.html">开始使用</a> ·
  <a href="https://infernux-engine.com/wiki.html">文档</a> ·
  <a href="https://infernux-engine.com/roadmap.html">路线图</a> ·
  <a href="https://infernux-engine.discourse.group/">社区</a>
</p>

Infernux 是一款开源游戏引擎，让你用 Python 创造可以游玩的世界，并把 Python 的计算生态带进游戏。Python 负责玩法、编辑器工具、资源流程与渲染编排；C++ 负责运行时；Vulkan 面向原生图形，WebGPU 面向浏览器。

长期方向是 **Neural Network-Native Engine（3N）**：让模型通过明确、可检查的接口观察世界、参与模拟，并成为最终游戏的一部分。当前优先把游戏引擎本身做好，再把世界数据、模型计算和工具连接起来。

**MIT 协议。Windows、Linux 编辑器；Windows、Linux、Android、Web Player。** 引擎仍在积极开发，041 是当前开发主线，公开发布版本为 **0.4.1**。

<img src="docs/assets/demo.png" width="1920" height="1032" alt="Infernux 编辑器中的 65,536 个 GameObject">

## 引擎基础

- **Python 优先的创作流程**：编写玩法组件、场景、工具和渲染管线，支持生命周期回调、协程与实时迭代。
- **原生运行时与跨平台渲染**：C++ 执行运行时，桌面和 Android 使用 Vulkan，浏览器使用 WebGPU。
- **完整的世界工作流**：场景、材质、灯光、动画、物理、音频、粒子、UI 和资源导出在一条流程里协同工作。
- **可扩展的编辑器**：插件可以加入组件、工具、资源和平台导出能力；可选 MCP 插件让自动化操作保持可观察。
- **面向计算的工作流**：Python 数值生态、CPU/GPU JIT 和世界数据 API 为更复杂的模拟与模型工作流打基础。

## 支持的平台

| 平台 | 编辑器 | Player | 图形后端 |
| --- | --- | --- | --- |
| Windows x64 | 有 | 有 | Vulkan |
| Linux x86_64 | 有 | 有 | Vulkan |
| Android arm64/x86_64 | 无 | APK/AAB | Vulkan |
| Web | 无 | HTML/JS/WASM | WebGPU |

详细限制见[平台支持说明](SUPPORT.md#platform-support)和[支持矩阵](docs/platform-support.json)。

## 当前方向

041 主线继续完善多场景编辑、模型导入、计算准备、物理查询、世界 UI、Player 启动和插件生命周期。[路线图](https://infernux-engine.com/roadmap.html)记录了后续阶段。

3N 方向会逐步连接模型工作流、统一世界 Schema、快照与回放、批量环境、张量数据交换，以及有治理边界的工具和模型包。完整的训练与部署闭环仍在建设中。

## 开始使用

最快的方式是安装 [InfernuxHub](https://infernux-engine.com/download.html)，选择引擎版本、创建项目并启动编辑器。Hub 会管理项目所需的 Python 环境。

接着阅读[学习指南](https://infernux-engine.com/learn.html)与 [API 文档](https://infernux-engine.com/wiki/site/zh/api/index.html)。插件作者可以参考[创作指南](https://infernux-engine.com/wiki/site/zh/plugin-package-content.html)和[插件模板](https://github.com/ChenlizheMe/infernux_plugin_template)。

### 从源码构建

Windows 需要 Visual Studio 2022（MSVC v143）、CMake 3.25+、Vulkan SDK 和 Python 3.13。Linux 需要先运行 `scripts/setup/install_linux_dependencies.sh`。随后运行对应平台的配置脚本，使用匹配的 CMake preset 构建并安装：

```text
git clone --recurse-submodules https://github.com/ChenlizheMe/Infernux.git
cd Infernux
# Windows: ./scripts/setup/configure_development.ps1
# Linux:   bash scripts/setup/configure_development.sh
# 激活生成的 infernux 环境，再使用 windows-msvc-release
# 或 linux-clang-release 等匹配的 CMake preset。
```

完整开发流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 加入熔炉

- 在[社区](https://infernux-engine.discourse.group/)提问并分享项目。
- 在 [GitHub Issues](https://github.com/ChenlizheMe/Infernux/issues) 提交可复现的问题。
- 阅读[更新日志](UpdateLog-zh.md)、[支持说明](SUPPORT.md)和[安全政策](SECURITY.md)。
- 欢迎参与引擎、文档和示例的建设。

Infernux 使用 [MIT 协议](LICENSE)。代码签名由 [SignPath.io](https://signpath.io/) 提供，详见[代码签名政策](CODE_SIGNING_POLICY.md)。

## 引用

```bibtex
@software{chen2026infernux,
  author  = {Chen, Lizhe},
  title   = {Infernux},
  year    = {2026},
  version = {0.4.1},
  url     = {https://github.com/ChenlizheMe/Infernux}
}
```
