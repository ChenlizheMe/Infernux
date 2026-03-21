<p align="center">
  <img src="docs/assets/logo.png" alt="InfernuxEngine Logo" width="128" />
</p>

<h1 align="center">InfernuxEngine</h1>

<p align="center">
  <strong>开源游戏引擎 · C++17 / Vulkan · Python 脚本 · MIT 协议</strong>
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
  <a href="README.md">🇬🇧 English</a> · <a href="#快速开始">快速开始</a> · <a href="#架构">架构</a> · <a href="https://chenlizheme.github.io/InfEngine/">网站</a>
</p>

---

## 项目简介

InfernuxEngine 是一个从零开始构建的游戏引擎，使用 **C++17 / Vulkan 运行时**和 **Python 脚本层**。C++ 负责渲染、物理和资源管理；Python 负责玩法逻辑、编辑器工具和内容工作流。

主要特点：

- **C++17 / Vulkan 核心** — 前向/延迟渲染、PBR、RenderGraph 管线、Jolt 物理
- **Python 脚本** — 类 Unity 组件模型、热重载、编辑器扩展、Python 生态直接可用
- **MIT 协议** — 无版税、无运行时费用、完全开源

---

## 设计立场

### 渲染可编排

渲染管线通过 RenderGraph API 对外开放。你可以用 Python 编写和修改渲染 Pass，而不是面对一个封闭的编辑器黑箱。

### 快速迭代

Python 不只处理玩法代码，也负责编辑器扩展、资产工作流和工具开发，让内循环尽可能短。

### 无授权顾虑

MIT 协议。无版税、无运行时费用、无供应商锁定。

---

## 功能

### 运行时基础

- Vulkan 前向与延迟渲染
- PBR 材质、级联阴影、MSAA、后处理
- 基于 RenderGraph 的 Pass 编排
- Jolt 物理（刚体 + 碰撞体）
- 输入、音频基础、场景与资源系统

### Python 层

- 类 Unity 的组件生命周期
- `serialized_field` 元数据，支持 Inspector 编辑
- 组件依赖与编辑器执行的装饰器
- 脚本和内容热重载
- 完整接入 Python 生态

### 编辑器

- Hierarchy、Inspector、Scene View、Game View、Console、Project 面板
- 选择、Gizmo、撤销重做、Play 模式场景隔离

---

## C++ / Python 分层

| 层 | 角色 |
|---|---|
| C++17 / Vulkan | 渲染、场景系统、资源、物理、平台服务 |
| pybind11 | 原生系统与 Python API 之间的绑定层 |
| Python | 玩法、编辑器逻辑、渲染编排、工具开发 |

---

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

### 克隆仓库

```bash
git clone --recurse-submodules https://github.com/ChenlizheMe/InfEngine.git
cd InfEngine
```

如果克隆时没有带子模块：

```bash
git submodule update --init --recursive
```

### 构建

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cmake --preset release
cmake --build --preset release
```

构建流程会生成原生 Python 模块、复制运行时依赖，并把包安装到当前工作区环境里，之后即可直接 `import InfEngine`。

### 运行

```bash
python packaging/launcher.py
```

### 测试

```bash
cd python
python -m pytest test/ -v
```

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

- 渲染与渲染管线
- Python 脚本与编辑器集成
- 编辑器核心创作流程
- 资产标识与项目启动
- 物理集成与场景交互

### 进行中

- Prefab 工作流
- UI 管线
- 动画系统
- 独立构建与导出
- 大规模项目的生产化完善

---

## 路线图

| 版本 | 重点 |
|:-----|:-----|| v0.1 | **当前** — 脚本、渲染、物理、编辑器均可用，已支持开发不含动画的基础游戏 || v0.2 | Prefab 工作流、UI 完善、资产重命名改进 |
| v0.3 | 动画系统、模型/内容管线 |
| v0.4 | 独立构建、粒子、地形 |
| v1.0 | 文档、示例、生产就绪 |

---

## 仓库结构

```text
cpp/infengine/        原生引擎运行时
python/InfEngine/     Python 引擎层与编辑器系统
packaging/            启动器与项目管理工具
docs/                 网站与生成文档入口
external/             第三方依赖与子模块
dev/                  规划文档与内部设计记录
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
- GitHub：[https://github.com/ChenlizheMe/InfEngine](https://github.com/ChenlizheMe/InfEngine)

## 许可证

MIT 协议。详见 [LICENSE](LICENSE)。
