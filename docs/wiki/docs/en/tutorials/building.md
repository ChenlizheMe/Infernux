# Building & Exporting Your Game

<div class="class-info">
Tutorial &nbsp;|&nbsp; <a href="../../zh/tutorials/building.html">中文</a>
</div>

## Overview

You've built something awesome. Now you want to share it. InfEngine compiles your Python game into a **standalone native EXE** using [Nuitka](https://nuitka.net/) — no Python installation required on the player's machine. The result is a self-contained folder with an executable, engine DLLs, and your game data. Double-click and play.

The build pipeline looks like this:

```
Your Project
    ↓
GameBuilder (Python)
    ↓
NuitkaBuilder (Python → C → native EXE)
    ↓
Inject engine DLLs (_InfEngine.pyd, SDL3.dll, etc.)
    ↓
Copy game data (Assets/, scenes, materials)
    ↓
Compile user scripts (.py → .pyc)
    ↓
Generate BuildManifest.json
    ↓
✅ Standalone game folder
```

## Prerequisites

Before building:

1. **Nuitka** — auto-installed if missing (`pip install nuitka ordered-set`)
2. **A C compiler** — MSVC (recommended, auto-detected) or MinGW64 (auto-downloaded by Nuitka)
3. **At least one scene** in your Build Settings

> **Why MSVC?** MinGW-compiled executables are more likely to trigger antivirus false positives. MSVC also handles Unicode paths natively, which avoids crashes on non-ASCII usernames (looking at you, 中文用户名).

## Build Settings

Open Build Settings from the editor menu bar: **File → Build Settings**.

### Scene List

Add scenes to the build list. The **first scene** is the one that loads on startup. You can:

- Click **"Add Open Scene"** to add the currently open scene
- Drag scene files from the Project panel
- Reorder scenes by dragging
- Remove scenes with the X button

### Display Mode

| Mode | Description |
|---|---|
| Fullscreen Borderless | Fills the screen, no title bar. Default. |
| Windowed | Runs in a window with configurable size. |

### Splash Screen

Add images and/or videos that play before the game starts:

- **Images**: Set fade-in time, display duration, and fade-out time
- **Videos**: Plays the video file directly
- Items play in order, top to bottom

### Output Directory

Choose where the built game files will be placed.

## Building from the Editor

1. Open **File → Build Settings**
2. Add your scenes
3. Configure display mode and splash screen
4. Click **Build** (or **Build & Run** to test immediately)

A progress bar shows the build status. The build runs in a background thread so the editor stays responsive.

### What Happens During Build

The `GameBuilder` orchestrates the full process:

1. **Validates** — checks that BuildSettings.json exists and all listed scenes are present
2. **Generates boot script** — creates a temporary entry point that:
   - Sets `_INFENGINE_PLAYER_MODE=1` (skips editor UI)
   - Loads the first scene from BuildSettings.json
   - Has crash reporting (writes to `crash.log` and shows a message box)
3. **Nuitka compilation** — compiles the boot script + InfEngine into native code
4. **Injects native libraries** — copies `_InfEngine.pyd`, SDL3.dll, and other engine DLLs
5. **Copies game data** — Assets/, ProjectSettings/, materials/ → `Data/`
6. **Compiles user scripts** — `.py` → `.pyc` for source protection
7. **Processes splash items** — copies images and packages video data
8. **Embeds UTF-8 manifest** — ensures Unicode paths work on Windows
9. **Self-signs the executable** — reduces antivirus false positives
10. **Generates BuildManifest.json** — display mode, window size, splash config

### Output Layout

```
MyGame/
    MyGame.exe              ← Native executable
    python312.dll           ← CPython runtime
    SDL3.dll, imgui.dll … ← Engine native DLLs
    InfEngine/              ← Engine package
        lib/
            _InfEngine.pyd  ← pybind11 extension
            SDL3.dll …      ← DLLs (for os.add_dll_directory)
    Data/
        Assets/             ← Your scenes, scripts (.pyc), textures, models
        ProjectSettings/    ← Build & tag-layer settings
        materials/          ← Material definitions
        Splash/             ← Splash images + video data
        BuildManifest.json  ← Runtime configuration
```

## Building from Script

You can also trigger builds programmatically:

```python
from InfEngine.engine.game_builder import GameBuilder

builder = GameBuilder(
    project_path="C:/MyProject",
    output_dir="C:/Builds/MyGame",
    display_mode="fullscreen_borderless",  # or "windowed"
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
print(f"Build complete: {output}")
```

## Third-Party Dependencies

The build system automatically detects third-party packages your game uses:

1. **requirements.txt** in your project root (highest priority)
2. **AST scanning** of all `.py` files in Assets/ (automatic fallback)

Only packages that are actually installed in your environment are included. Standard library and engine packages are excluded automatically.

```
# Example project requirements.txt
torch>=2.0
pillow
requests
```

## The InfEngine Hub

The Hub is a standalone launcher (built with PySide6 + PyInstaller) for managing InfEngine projects without opening the editor.

### Features

- **Project list** — create, open, and manage projects
- **Engine version management** — download engine wheels from GitHub Releases
- **Python runtime management** — auto-installs Python 3.12 embeddable
- **Virtual environment templates** — pre-built venvs for fast project creation

### Installing the Hub

The Hub ships as a one-click installer (`InfEngineHubInstaller.exe`) that:

1. Extracts the Hub application to your chosen directory
2. Downloads and installs Python 3.12 embeddable runtime
3. Creates a reusable venv template with pip
4. Registers in Windows Add/Remove Programs
5. Creates a Start Menu shortcut

### Building the Hub (for developers)

```powershell
# Build the Hub application
cd packaging
pyinstaller infengine_hub.spec --clean

# Build the installer (wraps Hub + Python runtime)
pyinstaller infengine_hub_installer.spec --clean
```

Or via CMake:
```powershell
cmake --build --preset release --target infengine_hub
```

## Troubleshooting

### "BuildSettings.json not found"

Open Build Settings in the editor and add at least one scene. The file is created automatically.

### Antivirus flags the EXE

This is common with Nuitka-compiled executables. The build system mitigates this by:
- Preferring MSVC over MinGW (fewer false positives)
- Self-signing the executable with a code-signing certificate
- Embedding a proper application manifest

For distribution, consider purchasing an EV code-signing certificate.

### Non-ASCII path crashes

The build system handles this by:
- Using an ASCII-safe staging directory (`C:\_InfBuild\`) during compilation
- Embedding a UTF-8 active-code-page manifest in the EXE
- Redirecting TEMP/TMP to ASCII-safe locations

### Missing DLLs at runtime

If the game fails to start with missing DLL errors:
1. Check that `InfEngine/lib/` in the build output contains `_InfEngine.pyd` and all `.dll` files
2. Verify the engine DLLs are also in the root directory (fallback search path)

### Nuitka compilation is slow

- First build downloads MinGW64 (~200MB) if MSVC isn't available
- Subsequent builds reuse cached C compilation results
- Use `--jobs` to control parallel compilation (defaults to CPU count - 1)

## Tips

- **Test your build early and often.** Don't wait until release day to discover build issues.
- **Use "Build & Run"** during development to quickly test the standalone version.
- **Keep your Assets/ clean.** Everything in Assets/ gets copied to the build. Remove unused files.
- **The player.log file** in the game directory contains startup logs and crash information. Tell your players to send it during bug reports.
- **Source protection**: User scripts are compiled to `.pyc` bytecode. It's not encryption, but it prevents casual reading of your source code.

## See Also

- [Physics Tutorial](physics.md) — Rigidbody, colliders, raycasting
- [Audio Tutorial](audio.md) — Sound effects and music
- [Rendering Tutorial](rendering.md) — Post-processing and custom pipelines
- [Coroutines Tutorial](coroutines.md) — Async game logic
- [UI Tutorial](ui.md) — Canvas, buttons, health bars
