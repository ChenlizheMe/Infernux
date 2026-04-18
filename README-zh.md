<p align="center">
  <img src="docs/assets/logo.png" alt="Infernux logo" width="128" />
</p>

<h1 align="center">Infernux</h1>

<p align="center">
  <strong>开源游戏引擎，采用 C++17 / Vulkan 原生运行时，以及负责生产工作流的 Python 层。</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/version-0.1.4-orange.svg" alt="Version" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey.svg" alt="Platform" />
  <img src="https://img.shields.io/badge/python-3.12+-brightgreen.svg" alt="Python" />
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++ 17" />
  <img src="https://img.shields.io/badge/graphics-Vulkan-red.svg" alt="Vulkan" />
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://chenlizheme.github.io/Infernux/">官网</a> ·
  <a href="https://chenlizheme.github.io/Infernux/wiki.html">文档</a> ·
  <a href="https://arxiv.org/pdf/2604.10263">技术报告</a> ·
  <a href="#快速开始">快速开始</a>
</p>

## 运行画面

<p align="center">
  <img src="docs/assets/demo.png" alt="Infernux 编辑器中运行 10000 个立方体场景的静态截图" width="100%" />
</p>

<p align="center">
  <em>编辑器处于 Play 模式时的实际运行画面截图。</em>
</p>

## 项目概览

Infernux 是一个从零开始构建的游戏引擎项目，目标不是把运行时和编辑器变成黑盒，而是让开发者能够真正掌控运行时能力、编辑器工作流与脚本层扩展面。

当前架构由三部分组成：

- 使用 C++17 / Vulkan 编写的原生运行时，负责渲染、场景系统、物理、音频和平台服务。
- 使用 pybind11 构建的绑定层，把原生能力暴露给 Python。
- 负责玩法、编辑器工具、内容工作流、构建自动化和渲染编排的 Python 生产层。

这个分层的目标很直接：性能敏感部分留在原生层，日常生产与迭代留给 Python，同时保持代码结构足够清晰，便于团队理解、扩展和维护。

## 当前能力范围

Infernux 目前以 Windows 为主的技术预览版。项目已具备可用的编辑器与运行时核心，但仍应按持续演进的引擎来评估，而非成品商业平台。

当前已具备的核心能力包括：

- Vulkan 前向/延迟渲染、PBR、级联阴影、MSAA、Shader 反射与后处理。
- 可在 Python 中编排的 RenderGraph 与 RenderStack API。
- Jolt 物理：刚体、碰撞体、场景查询、回调与层过滤。
- 基于 GUID 的资产、依赖追踪、场景序列化、预制体工作流与 Play 模式隔离。
- 集成编辑器：Hierarchy、Inspector、Scene View、Game View、Project、Console、UI 编辑与构建设置。
- Python 侧组件生命周期、协程、序列化字段与脚本热重载。
- 基础运行时 UI：Canvas、Text、Image、Button 与指针事件。
- **2D 动画（预览）：** 精灵 `SpiritAnimator`、`AnimClip2D` 资产、动画状态机资产，以及用于编辑的面板。在栈稳定前可能出现不兼容变更。
- Hub、独立目录包与 Windows 安装器等打包路径。

## 快速开始

### 环境要求

<details>
<summary><b>Windows</b></summary>

| 依赖 | 版本 |
|:-----|:-----|
| Windows | 10 / 11（64 位） |
| Python | 3.12+ |
| Vulkan SDK | 1.3+ |
| CMake | 3.22+ |
| Visual Studio | 2022（MSVC v143） |
| pybind11 | 2.11+ |

</details>

<details>
<summary><b>macOS</b></summary>

| 依赖 | 版本 |
|:-----|:-----|
| macOS | 12+ |
| Python | 3.12+ |
| Vulkan SDK | 1.3+（LunarG SDK + MoltenVK） |
| CMake | 3.22+ |
| Ninja | 1.10+ |
| Xcode Command Line Tools | 最新版 |
| pybind11 | 2.11+ |

安装 Vulkan SDK 后，先执行环境脚本：

```bash
source ~/VulkanSDK/<version>/setup-env.sh
brew install cmake ninja
```

</details>

你可以使用任意 Python 3.12 环境。下面示例使用 Conda，因为这是当前仓库里最常见的工作流。

### 克隆仓库

```bash
git clone --recurse-submodules https://github.com/ChenlizheMe/Infernux.git
cd Infernux
```

如果之前没有带上子模块：

```bash
git submodule update --init --recursive
```

### 构建引擎

```bash
conda create -n infengine python=3.12 -y
conda activate infengine
pip install -r requirements.txt
cmake --preset release
cmake --build --preset release
```

在 macOS 上，将 `release` 替换为 `release-macos`。

构建流程会编译原生模块、复制运行时依赖，并把 Python 包安装进当前环境，使你可以直接在工作区中 `import Infernux`。

### 以开发模式启动 Hub

```bash
conda activate infengine
python packaging/launcher.py
```

开发模式使用当前 Python 环境和本地构建产物，不会安装 Hub 的托管运行时。

### 运行测试

```bash
conda activate infengine
cd python
python -m pytest test/ -v
```

## 文档

- 官网：<https://chenlizheme.github.io/Infernux/>
- 文档入口：<https://chenlizheme.github.io/Infernux/wiki.html>
- 技术报告：[Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting（arXiv:2604.10263）](https://arxiv.org/pdf/2604.10263)
- API 参考：从 Python 包自动生成，并发布到 `docs/wiki/site/`

本地重新生成 API Markdown 和静态站点：

```bash
conda activate infengine
python docs/wiki/generate_api_docs.py
python -m mkdocs build --clean -f docs/wiki/mkdocs.yml
```

对应的 CMake 目标是 `generate_api_docs` 和 `build_wiki_html`。

## 打包与分发

Hub 目前支持两条分发路径。

### 独立目录打包

```bash
cmake --build --preset packaging
```

这会生成位于 `dist/Infernux Hub/` 的 PyInstaller 目录。

### Windows 安装器

```bash
cmake --build --preset packaging-installer
```

这会生成图形化 Windows 安装器。安装流程会为当前主机准备匹配的 Python 3.12 运行时，再基于这套托管运行时为项目创建独立环境。

## 引用

如果你在论文、技术报告或公开材料中使用 Infernux，可以按软件条目引用：

```bibtex
@software{chen2026infernux,
  author  = {Chen, Lizhe},
  title   = {Infernux},
  year    = {2026},
  version = {0.1.4},
  url     = {https://github.com/ChenlizheMe/Infernux},
  note    = {Open-source game engine with a C++17/Vulkan runtime and a Python production layer}
}
```

## 参与贡献

当前阶段，Bug 报告、功能建议和工作流反馈都很有价值。提交 issue 时，建议附带引擎版本、环境信息、复现步骤，以及问题位于原生运行时、Python 层还是打包链路。

贡献和支持相关说明见：

- `CONTRIBUTING.md`
- `SECURITY.md`
- `SUPPORT.md`

## 许可证

Infernux 基于 MIT 协议发布，详见 `LICENSE`。
