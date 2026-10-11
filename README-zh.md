<p align="center">
  <a href="https://infernux-engine.com/"><img src="docs/assets/logo.png" width="96" alt="熔炉标志"></a>
</p>

<h1 align="center">熔炉 · Infernux</h1>

<p align="center">
  <strong>创造世界，让智能成为其中一部分。</strong><br>
  一款用 Python 创作的开源游戏引擎，正驶向 3N：<b>神经网络原生</b>引擎。
</p>

<p align="center">
  <a href="https://github.com/ChenlizheMe/Infernux/releases"><img src="https://img.shields.io/badge/version-0.4.1-orange.svg" alt="已发布版本 0.4.1"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT 协议"></a>
  <img src="https://img.shields.io/badge/python-3.13-3776AB.svg" alt="Python 3.13">
  <img src="https://img.shields.io/badge/graphics-Vulkan%20%C2%B7%20WebGPU-c8444b.svg" alt="Vulkan 与 WebGPU">
  <a href="https://github.com/ChenlizheMe/Infernux/actions/workflows/ci.yml"><img src="https://github.com/ChenlizheMe/Infernux/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="https://infernux-engine.com/">官网</a> ·
  <a href="https://infernux-engine.com/download.html">下载</a> ·
  <a href="https://infernux-engine.com/tutorials.html">教程</a> ·
  <a href="https://infernux-engine.com/wiki/site/zh/api/index.html">API</a> ·
  <a href="https://infernux-engine.com/roadmap.html">路线图</a> ·
  <a href="https://infernux-engine.discourse.group/">社区</a> ·
  <a href="README.md">English</a>
</p>

<p align="center">
  <img src=".github/media/space-battle.gif" width="600" alt="熔炉运行时：由玩法逻辑驱动的数百艘飞船与粒子爆发">
</p>

<p align="center">
  <img src=".github/media/npr-pipeline.gif" width="196" alt="可在运行时重建的可编程渲染管线">
  <img src=".github/media/rigid-coins.gif" width="196" alt="1000 个刚体交互与自定义辉光">
  <img src=".github/media/animated-cats.gif" width="196" alt="由动画状态机与 Timeline 驱动的猫群">
  <img src=".github/media/rendergraph-grid.gif" width="196" alt="RenderGraph 分屏组合多种渲染效果">
</p>
<p align="center"><sub>以上均为引擎实机画面，来自 <a href="https://www.bilibili.com/video/BV1538P6jELT/?p=2">0.3.4 演示视频</a>，没有概念图。</sub></p>

## 熔炉是什么？

熔炉是一款 **用 Python 创作、由 C++ 运行** 的游戏引擎。玩法、编辑器工具、渲染管线与数值计算都用 Python 编写；C++ 运行时、Vulkan 渲染器和 Jolt 物理负责重活。同一个项目可以发布到 Windows、Linux、Android 和浏览器。

它也朝着一个明确的目标构建：**3N，即神经网络原生（Neural Network-Native）引擎**。在多数引擎里，AI 是游戏做完后再接上去的插件；而在 3N 引擎里，世界是模型读得懂的数据，模拟是它能行动的场所，发布的游戏也是它能运行的地方。

```python
import infernux as inx


class Spin(inx.InxComponent):
    speed: float = inx.serialized_field(default=90.0, range=(0.0, 360.0))

    def update(self, delta_time: float) -> None:
        self.transform.rotate(inx.Vector3(0.0, self.speed * delta_time, 0.0))
```

<sub>一个普通的 Python 类。`speed` 会出现在 Inspector 中，游戏运行时也能随时调整。</sub>

> [!WARNING]
> 熔炉仍是 **Alpha 阶段** 的软件，正在积极开发中，当前发布版本为 **0.4.1**。不同版本之间 API 仍会变化，下文中的神经网络能力属于路线图目标，尚未交付。它适合实验、工具开发和小型游戏，暂不适合有严格排期的商业项目。

## 亮点

- **Python 优先。** 带 Inspector 字段的组件、协程与 Prefab；项目脚本和插件支持事务式热重载，已经存在的实例也会随之更新行为。
- **可以编写的渲染器。** 用 Python 描述 RenderStack，由原生 RenderGraph 执行：Forward、Forward+ 与 Deferred 路径，PBR、阴影、后处理、GPU 粒子图、蒙皮动画，以及屏幕与世界空间 UI。
- **完整的游戏工作流。** 多场景编辑，FBX 与 Blender 导入并实时同步源文件，Jolt 刚体与场景查询，音频、动画状态机与 Timeline。
- **把计算带进游戏。** 用 NumPy 批量读写世界数据，CPU/GPU JIT 与 Taichi 集成支撑计算驱动的玩法，这也是模型将来接入世界的数据通道。
- **可以扩展的编辑器。** 用 Python 编写可停靠面板与菜单；InxPackage 插件可携带组件、工具、资源与平台导出器；可选的 MCP 插件让 Agent 通过与你相同的命令和撤销路径操作编辑器。
- **经得起检查。** 资源从导入到 Player 构建始终保有 GUID 身份；Player 读取打包后的资源索引，而不依赖你的项目目录。

## 平台

| 平台 | 编辑器 | Player | 图形后端 |
| --- | :---: | :---: | --- |
| Windows x64 | ✅ | ✅ | Vulkan |
| Linux x86_64 | ✅ | ✅ | Vulkan |
| Android arm64 / x86_64 | — | APK / AAB | Vulkan |
| Web | — | HTML + WASM | WebGPU |

各平台的要求与已知限制见 [SUPPORT.md](SUPPORT.md#platform-support)。

## 开始使用

**直接使用引擎。** 安装 [InfernuxHub](https://infernux-engine.com/download.html)，选择引擎版本并创建项目。Hub 会替你管理 Python 环境与 Android 工具，无需安装原生编译器。然后跟着 [五步入门](https://infernux-engine.com/start.html) 写出第一个让方块旋转的组件，再继续学习 [系列课程](https://infernux-engine.com/learn.html)。

**从源码构建。** Windows 需要 Visual Studio 2022（MSVC v143）、CMake 3.25+、Vulkan SDK 与 Python 3.13：

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

Linux 上先运行 `scripts/setup/install_linux_dependencies.sh` 与 `bash scripts/setup/configure_development.sh`，激活 `infernux` 环境，再使用 `linux-clang-release` 与 `linux-clang-install-wheel` 预设。完整流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 通往 3N 的航线

| 阶段 | 状态 | 带来的能力 |
| --- | --- | --- |
| Python 创作、批量世界 API、CPU/GPU 计算 | ✅ 已可用 | 模型与玩法共用同一条进入世界的数据通道 |
| 确定性步进、快照与回放 | 🔜 规划中 | 可复现的运行，用于调试与学习 |
| 批量世界与张量数据面 | 🔜 规划中 | 多个环境并行推进，数据交换无需胶水代码 |
| Player 内的可移植推理 | 🔜 规划中 | 训练好的模型随游戏发布到每个平台 |

十张交互式星图标出了 [路线图](https://infernux-engine.com/roadmap.html) 上的每个系统；版本说明见 [更新日志](UpdateLog-zh.md)。

## 社区

- 💬 提问、展示与讨论：[社区论坛](https://infernux-engine.discourse.group/)
- 🐛 可复现的问题：[GitHub Issues](https://github.com/ChenlizheMe/Infernux/issues)
- 🧩 插件开发：[创作指南](https://infernux-engine.com/wiki/site/zh/plugin-package-content.html) 与 [插件模板](https://github.com/InfernuxEngine/infernux_plugin_template)
- 🛠️ 参与贡献：从 [CONTRIBUTING.md](CONTRIBUTING.md) 开始

## 许可

熔炉以 [MIT 协议](LICENSE) 开源，永久免费，不抽成。第三方版权声明集中索引在 [licenses](licenses/README.md)。Windows 构建由 [SignPath.io](https://signpath.io/) 免费签名，证书由 [SignPath Foundation](https://signpath.org/) 提供，详见 [代码签名政策](licenses/signing/CODE_SIGNING_POLICY-zh.md)。

<details>
<summary>引用熔炉</summary>

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
