<p align="center">
  <img src="docs/assets/logo.png" alt="Infernux logo" width="128" />
</p>

<h1 align="center">Infernux</h1>

<p align="center">
  <strong>开源游戏引擎，采用 C++17 / Vulkan 原生运行时，以及负责生产工作流的 Python 层。</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/version-0.1.3-orange.svg" alt="Version" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey.svg" alt="Platform" />
  <img src="https://img.shields.io/badge/python-3.12+-brightgreen.svg" alt="Python" />
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++ 17" />
  <img src="https://img.shields.io/badge/graphics-Vulkan-red.svg" alt="Vulkan" />
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://chenlizheme.github.io/Infernux/">官网</a> ·
  <a href="https://chenlizheme.github.io/Infernux/wiki.html">文档</a> ·
  <a href="#快速开始">快速开始</a>
</p>

## 演示画面

<p align="center">
  <img src="docs/assets/demo.gif" alt="Infernux 编辑器中实时渲染的 10000 个立方体" width="100%" />
</p>

<p align="center">
  <em>30 FPS 实机录制：编辑器进入 Play 模式后不做任何操作，屏幕上持续渲染 10000 个立方体，并展示整片场景中的受光变化。</em>
</p>

---

## 项目概览

Infernux 是一个从零开始构建的引擎，目标不是把所有东西都做成黑盒，而是让你真正掌控运行时、工具链和迭代流程。

这个项目由三部分组成：

- 原生 C++17 / Vulkan 运行时，负责渲染、场景系统、平台服务和物理。
- Python 层，负责玩法逻辑、编辑器工具、内容工作流和渲染栈编排。
- MIT 协议，不收版税，不收运行时费用，也不把核心能力关在闭源层里。

核心思路很简单：性能敏感的部分留在原生层，日常开发最频繁的部分交给 Python，同时把架构保持在一个能读懂、能扩展、能接管的状态。

## 为什么是 Infernux

- 渲染可编排：RenderGraph 和 RenderStack 对脚本开放，不需要完全依赖封闭的编辑器配置。
- Python 用在迭代环路里：玩法、编辑器扩展、资产工作流、自动化脚本都能放在一个生态中完成。
- 仓库优先：项目默认你会去读代码、改代码，而不是把引擎当成不可触碰的黑盒。
- 范围表述诚实：当前版本是 Windows 上的技术预览版，核心能力可用，但还不是一个完成态的跨平台商业引擎。

## 当前状态

目前已经具备一套可以继续向真实项目推进的基础能力：

- Vulkan 前向/延迟渲染、PBR、级联阴影、MSAA、Shader 反射、后处理。
- 可从 Python 编排的 RenderGraph / RenderStack 渲染体系。
- Jolt 物理，支持刚体、碰撞体、查询、回调和层过滤。
- SDL3 音频、GUID 资产系统、依赖追踪、场景与资源管理。
- 内置编辑器，包含 Hierarchy、Inspector、Scene View、Game View、Project、Console、UI 编辑器、Build Settings 等面板。
- Python 组件生命周期、协程工具、预制体工作流、序列化字段和热重载。
- 游戏 UI 基础能力，包括 Canvas、Text、Image、Button 和事件系统。
- Hub 的独立打包与 Windows 安装器流程。

接下来的重点是动画系统、高级 UI 控件、更完整的上手资料，以及更强的内容生产链路。

## 架构分层

Infernux 的分层不是为了概念好看，而是按职责划分。

| 层 | 职责 |
|:---|:-----|
| C++17 / Vulkan | 渲染器、资源所有权、物理、场景系统、平台服务 |
| pybind11 绑定层 | 把原生系统暴露给 Python |
| Python | 玩法、编辑器逻辑、工具开发、自动化、渲染编排 |

这样做的目的是让性能关键路径维持在原生侧，而日常频繁改动的生产代码放在更适合快速迭代和接入外部生态的语言中。

## 快速开始

### 环境要求

| 依赖 | 版本 |
|:-----|:-----|
| Windows | 10 / 11（64 位） |
| Python | 3.12+ |
| Vulkan SDK | 1.3+ |
| CMake | 3.22+ |
| Visual Studio | 2022（MSVC v143） |
| pybind11 | 2.11+ |

你可以使用任意 Python 3.12 环境。下面的示例使用 Conda，因为这是这个仓库里最常见的开发方式。

### 克隆仓库

```bash
git clone --recurse-submodules https://github.com/ChenlizheMe/Infernux.git
cd Infernux
```

如果你之前没有带上子模块：

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

这个流程会编译原生模块、复制运行时依赖，并把 Python 包安装到当前环境里，使你可以直接在工作区中 `import Infernux`。

### 以开发模式启动 Hub

```bash
conda activate infengine
python packaging/launcher.py
```

开发模式会使用当前 Python 环境和本地构建产物，不会安装 Hub 的私有运行时。

### 运行测试

```bash
conda activate infengine
cd python
python -m pytest test/ -v
```

## 文档

- 官网：<https://chenlizheme.github.io/Infernux/>
- 文档入口：<https://chenlizheme.github.io/Infernux/wiki.html>
- API 参考：从 Python 包自动生成，并以静态 HTML 发布到 `docs/wiki/site/`

本地重新生成 API Markdown 和 HTML 文档：

```bash
conda activate infengine
python docs/wiki/generate_api_docs.py
python -m mkdocs build --clean -f docs/wiki/mkdocs.yml
```

对应的 CMake 目标是 `generate_api_docs` 和 `build_wiki_html`。

## 打包与分发

Hub 目前有两条正式支持的分发路径。

### 独立目录打包

```bash
cmake --build --preset packaging
```

这会生成位于 `dist/Infernux Hub/` 的 PyInstaller 目录，适合本地验证或开发者分发。

### Windows 安装器

```bash
cmake --build --preset packaging-installer
```

这会生成 Hub 的图形化 Windows 安装器。安装流程会根据主机架构准备对应的 Python 3.12 运行时，再由 Hub 基于这套私有运行时为项目创建运行环境。

## 参与贡献

现在这个阶段，Bug 报告、功能建议和工作流反馈都很有价值。提交 issue 时，尽量说明引擎版本、环境信息、复现步骤，以及问题属于原生运行时、Python 层还是打包链路。

贡献和支持相关说明位于这些社区文件中：

- `CONTRIBUTING.md`
- `SECURITY.md`
- `SUPPORT.md`

## 许可证

Infernux 基于 MIT License 发布，详见 `LICENSE`。

这意味着每个项目拥有自己独立完整的 Python 副本，不与其他项目共享运行时。

---

## 架构

```text
Python 创作层
  -> 编辑器面板、组件系统、RenderGraph 编排、工具工作流
  -> pybind11 绑定接缝
C++ 引擎核心
  -> 渲染器、场景、资源、物理、音频、平台服务
外部技术栈
  -> Vulkan、SDL3、Jolt、ImGui、Assimp、GLM、glslang、VMA
```

### 实际工作流

1. 用 Python 编写玩法或渲染逻辑。
2. 绑定到编辑器可见的数据和场景对象。
3. 通过 RenderGraph API 描述渲染 Pass。
4. 原生后端负责调度、内存管理和 GPU 执行。

---

## 当前状态

### 已完成

- Vulkan 渲染（前向 + 延迟）、PBR、阴影、8 种后处理效果
- Python 脚本与热重载、编辑器集成
- 完整编辑器（12 个面板、Gizmo、撤销重做、Play 模式）
- Jolt 物理（刚体、碰撞体、射线检测、碰撞层）
- SDL3 音频（3D 空间化）
- 资产管线（基于 GUID 的 AssetDatabase、.meta 文件、依赖图）
- 预制体系统（保存/实例化/覆盖追踪）
- 游戏 UI 系统（Canvas、Text、Image、Button、事件系统）
- 基于 Nuitka 的独立游戏构建
- Numba JIT 集成，提供 `@njit` 装饰器、自动降级回退、`.py` 属性显式调用纯 Python 路径
- Hub 启动器与 Windows 安装器

### 进行中

- 动画系统（骨骼动画、状态机）
- 高级 UI 控件（ScrollView、Slider、布局组件）
- 文档、教程与示例项目

---

## 路线图

| 版本 | 重点 |
|:-----|:-----|
| v0.1 | **当前** — 渲染、物理、音频、脚本、编辑器、预制体、游戏 UI、独立构建均已可用，支持开发不含动画的基础游戏 |
| v0.2 | 动画系统、高级 UI 控件（ScrollView、Slider、布局）、资产重命名改进 |
| v0.3 | 粒子系统、地形、模型/内容管线改进 |
| v0.4 | 网络基础 |
| v1.0 | 文档、示例、生产就绪 |

---

## 仓库结构

```text
cpp/infernux/        原生引擎运行时
python/Infernux/     Python 引擎层与编辑器系统
packaging/            启动器与项目管理工具
docs/                 网站与生成文档入口
external/             第三方依赖与子模块
dev/                  发布用脚本
```

---

## 参与贡献

1. 先读 README 和文档站。
2. 查看路线图了解当前优先级。
3. 大改动前请先开 Issue 或 Discussion。
4. 提交目标明确的 Pull Request。

---

## 致谢

- 架构方向受到王希 [GAMES104](https://games104.boomingtech.com/) 课程启发
- 使用了 [Jolt Physics](https://github.com/jrouwe/JoltPhysics)、[SDL3](https://github.com/libsdl-org/SDL)、[Dear ImGui](https://github.com/ocornut/imgui)、[Assimp](https://github.com/assimp/assimp)、[GLM](https://github.com/g-truc/glm)、[glslang](https://github.com/KhronosGroup/glslang) 与 [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)

---

## 联系方式

- 作者：Lizhe Chen
- 邮箱：[chenlizheme@outlook.com](mailto:chenlizheme@outlook.com)
- GitHub：[https://github.com/ChenlizheMe/Infernux](https://github.com/ChenlizheMe/Infernux)

## 许可证

MIT 协议。详见 [LICENSE](LICENSE)。
