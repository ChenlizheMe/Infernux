# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for InfEngine Hub.

Build with:
    cd packaging
    pyinstaller infengine_hub.spec --clean
or via CMake:
    cmake --build --preset release --target infengine_hub
"""

import os
import sys
import urllib.request

block_cipher = None

_PACKAGING_DIR = os.path.dirname(os.path.abspath(SPEC))
_VENDOR_DIR = os.path.join(_PACKAGING_DIR, "_vendor")
_INSTALLER_NAME = "python-3.12.0-amd64.exe"
_INSTALLER_URL = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
_INSTALLER_PATH = os.path.join(_VENDOR_DIR, _INSTALLER_NAME)

os.makedirs(_VENDOR_DIR, exist_ok=True)

if not os.path.isfile(_INSTALLER_PATH):
    print(f"[spec] Downloading {_INSTALLER_NAME} for bundled offline install ...")
    urllib.request.urlretrieve(_INSTALLER_URL, _INSTALLER_PATH)

a = Analysis(
    [os.path.join(_PACKAGING_DIR, "launcher.py")],
    pathex=[_PACKAGING_DIR],
    binaries=[],
    datas=[
        (os.path.join(_PACKAGING_DIR, "resources", "icon.png"), "resources"),
        (os.path.join(_PACKAGING_DIR, "resources", "PingFangTC-Regular.otf"), "resources"),
        (_INSTALLER_PATH, os.path.join("_inner", "runtime")),
    ],
    hiddenimports=[
        "hub_resources",
        "hub_utils",
        "python_runtime",
        "version_manager",
        "database",
        "splash_screen",
        "style",
        "ui_project_list",
        "model",
        "model.project_model",
        "model.new_project_model",
        "view",
        "view.control_pane_view",
        "view.new_project_view",
        "view.sidebar_view",
        "view.installs_view",
        "viewmodel",
        "viewmodel.control_pane_viewmodel",
        "viewmodel.new_project_viewmodel",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Don't bundle the engine itself — it's installed per-project
        "InfEngine",
        # Heavy dev packages that aren't needed
        "numpy",
        "watchdog",
        "PIL",
        "cv2",
        "matplotlib",
        "scipy",
        "pandas",
        "tkinter",
        "unittest",
        "test",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="InfEngine Hub",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(_PACKAGING_DIR, "resources", "icon.png"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="InfEngine Hub",
)
