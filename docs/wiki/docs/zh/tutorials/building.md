# 构建与导出你的游戏

<div class="class-info">
教程 &nbsp;|&nbsp; <a href="../../en/tutorials/building.html">English</a>
</div>

## 概述

你做出了一个好东西。现在该让别人玩到了。InfEngine 使用 [Nuitka](https://nuitka.net/) 把你的 Python 游戏编译成**独立的原生 EXE** — 玩家的电脑上不需要安装 Python。最终产出是一个包含可执行文件、引擎 DLL 和游戏数据的文件夹。双击即玩。

构建流程如下：

```
你的项目
    ↓
GameBuilder（Python）
    ↓
NuitkaBuilder（Python → C → 原生 EXE）
    ↓
注入引擎 DLL（_InfEngine.pyd、SDL3.dll 等）
    ↓
复制游戏数据（Assets/、场景、材质）
    ↓
编译用户脚本（.py → .pyc）
    ↓
生成 BuildManifest.json
    ↓
✅ 独立游戏文件夹
```

## 前置条件

构建之前需要：

1. **Nuitka** — 如果没有会自动安装（`pip install nuitka ordered-set`）
2. **C 编译器** — MSVC（推荐，自动检测）或 MinGW64（Nuitka 自动下载）
3. **至少一个场景**在你的 Build Settings 中

> **为什么推荐 MSVC？** MinGW 编译的可执行文件更容易被杀毒软件误报。MSVC 还原生支持 Unicode 路径，避免了非 ASCII 用户名导致的崩溃（对，就是说你，中文用户名的朋友们）。

## Build Settings（构建设置）

在编辑器菜单栏打开：**File → Build Settings**。

### 场景列表

把场景添加到构建列表。**第一个场景**是启动时加载的。你可以：

- 点击 **"Add Open Scene"** 添加当前打开的场景
- 从 Project 面板拖拽场景文件
- 拖拽排序
- 点 X 按钮移除

### 显示模式

| 模式 | 说明 |
|---|---|
| Fullscreen Borderless（无边框全屏） | 填满整个屏幕，没有标题栏。默认选项。 |
| Windowed（窗口化） | 在窗口中运行，可配置尺寸。 |

### 开场画面

添加图片和/或视频，在游戏开始前播放：

- **图片**：设置淡入时间、显示时长、淡出时间
- **视频**：直接播放视频文件
- 项目按从上到下的顺序播放

### 输出目录

选择构建后文件的存放位置。

## 从编辑器构建

1. 打开 **File → Build Settings**
2. 添加场景
3. 配置显示模式和开场画面
4. 点 **Build**（或 **Build & Run** 直接测试）

进度条显示构建状态。构建在后台线程运行，编辑器不会卡住。

### 构建过程详解

`GameBuilder` 负责整个流程：

1. **验证** — 检查 BuildSettings.json 存在，且所有列出的场景文件都在
2. **生成启动脚本** — 创建临时入口点：
   - 设置 `_INFENGINE_PLAYER_MODE=1`（跳过编辑器 UI）
   - 从 BuildSettings.json 加载第一个场景
   - 包含崩溃报告（写入 `crash.log` 并弹出消息框）
3. **Nuitka 编译** — 将启动脚本 + InfEngine 编译为原生代码
4. **注入原生库** — 复制 `_InfEngine.pyd`、SDL3.dll 和其他引擎 DLL
5. **复制游戏数据** — Assets/、ProjectSettings/、materials/ → `Data/`
6. **编译用户脚本** — `.py` → `.pyc` 保护源代码
7. **处理开场画面** — 复制图片并打包视频数据
8. **嵌入 UTF-8 清单** — 确保 Unicode 路径在 Windows 上正常工作
9. **自签名可执行文件** — 减少杀毒软件误报
10. **生成 BuildManifest.json** — 显示模式、窗口尺寸、开场配置

### 输出目录结构

```
MyGame/
    MyGame.exe              ← 原生可执行文件
    python312.dll           ← CPython 运行时
    SDL3.dll, imgui.dll … ← 引擎原生 DLL
    InfEngine/              ← 引擎包
        lib/
            _InfEngine.pyd  ← pybind11 扩展模块
            SDL3.dll …      ← DLL（用于 os.add_dll_directory）
    Data/
        Assets/             ← 你的场景、脚本（.pyc）、贴图、模型
        ProjectSettings/    ← 构建 & Tag/Layer 设置
        materials/          ← 材质定义
        Splash/             ← 开场图片 + 视频数据
        BuildManifest.json  ← 运行时配置
```

## 从脚本构建

你也可以用代码触发构建：

```python
from InfEngine.engine.game_builder import GameBuilder

builder = GameBuilder(
    project_path="C:/MyProject",
    output_dir="C:/Builds/MyGame",
    display_mode="fullscreen_borderless",  # 或 "windowed"
    window_width=1920,
    window_height=1080,
    splash_items=[
        {"type": "image", "path": "Assets/splash.png",
         "fade_in": 0.5, "duration": 2.0, "fade_out": 0.5},
    ],
)

def on_progress(message, percent):
    print(f"[{percent:.0%}] {message}")

output = builder.build(on_progress=on_progress)
print(f"构建完成：{output}")
```

## 第三方依赖

构建系统自动检测你的游戏使用的第三方包：

1. **requirements.txt**（你的项目根目录，最高优先级）
2. **AST 扫描** Assets/ 下所有 `.py` 文件（自动兜底）

只有当前环境中实际安装的包才会被包含。标准库和引擎包自动排除。

```
# 示例 project requirements.txt
torch>=2.0
pillow
requests
```

## InfEngine Hub

Hub 是一个独立的启动器（PySide6 + PyInstaller 构建），用于在不打开编辑器的情况下管理 InfEngine 项目。

### 功能

- **项目列表** — 创建、打开、管理项目
- **引擎版本管理** — 从 GitHub Releases 下载引擎 wheel 包
- **Python 运行时管理** — 自动安装 Python 3.12 嵌入式运行时
- **虚拟环境模板** — 预构建的 venv，快速创建项目

### 安装 Hub

Hub 以一键安装程序（`InfEngineHubInstaller.exe`）的形式发布，安装过程：

1. 将 Hub 应用解压到你选择的目录
2. 下载并安装 Python 3.12 嵌入式运行时
3. 创建可复用的带 pip 的 venv 模板
4. 在 Windows 添加/删除程序中注册
5. 创建开始菜单快捷方式

### 构建 Hub（开发者用）

```powershell
# 构建 Hub 应用
cd packaging
pyinstaller infengine_hub.spec --clean

# 构建安装程序（包含 Hub + Python 运行时）
pyinstaller infengine_hub_installer.spec --clean
```

也可以通过 CMake：
```powershell
cmake --build --preset release --target infengine_hub
```

## 常见问题

### "BuildSettings.json not found"

在编辑器中打开 Build Settings 并至少添加一个场景。文件会自动创建。

### 杀毒软件标记 EXE

这在 Nuitka 编译的可执行文件中很常见。构建系统通过以下方式缓解：
- 优先使用 MSVC 而非 MinGW（更少误报）
- 用代码签名证书自签名可执行文件
- 嵌入正确的应用程序清单

要正式分发，建议购买 EV 代码签名证书。

### 非 ASCII 路径崩溃

构建系统通过以下方式处理：
- 编译期间使用 ASCII 安全的暂存目录（`C:\_InfBuild\`）
- 在 EXE 中嵌入 UTF-8 活动代码页清单
- 将 TEMP/TMP 重定向到 ASCII 安全的位置

### 运行时缺少 DLL

如果游戏启动失败并报告缺少 DLL：
1. 检查构建输出中的 `InfEngine/lib/` 是否包含 `_InfEngine.pyd` 和所有 `.dll` 文件
2. 确认引擎 DLL 也在根目录中（备用搜索路径）

### Nuitka 编译很慢

- 首次构建会下载 MinGW64（~200MB），如果没有 MSVC
- 后续构建复用缓存的 C 编译结果
- 使用 `--jobs` 控制并行编译（默认为 CPU 核心数 - 1）

## 小贴士

- **尽早、频繁地测试构建。** 别等到发布当天才发现构建问题。
- **开发时使用 "Build & Run"** 快速测试独立版本。
- **保持 Assets/ 整洁。** Assets/ 里的所有东西都会被复制到构建输出。删除不用的文件。
- **player.log 文件** 在游戏目录中，包含启动日志和崩溃信息。让你的玩家发 bug 报告时带上它。
- **源码保护**：用户脚本被编译为 `.pyc` 字节码。不是加密，但能防止他人直接阅读源代码。

## 另请参阅

- [物理教程](physics.md) — Rigidbody、碰撞体、射线检测
- [音频教程](audio.md) — 音效和音乐
- [渲染教程](rendering.md) — 后处理和自定义管线
- [协程教程](coroutines.md) — 异步游戏逻辑
- [UI 教程](ui.md) — Canvas、按钮、血条
