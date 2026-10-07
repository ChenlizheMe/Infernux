"""Make every Hub test runnable alone, without collection-order side effects."""

from pathlib import Path
import importlib.metadata
import shutil
import sys

import pytest


PACKAGING_ROOT = (Path(__file__).resolve().parents[2] / "packaging")
sys.path.insert(0, str(PACKAGING_ROOT))
sys.path.insert(0, str(PACKAGING_ROOT.parent / "python"))


@pytest.fixture
def private_python_factory():
    """Real isolated executable/pip with the host stdlib; never installs into the test host."""
    def create(root: Path) -> Path:
        root.mkdir(parents=True)
        python = root / ("python.exe" if sys.platform == "win32" else "bin/python")
        python.parent.mkdir(exist_ok=True)
        shutil.copy2(sys.executable, python)
        if sys.platform == "win32":
            dll = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
            shutil.copy2(Path(sys.base_prefix) / dll, root / dll)
            shutil.copy2(Path(sys.executable).with_name("pythonw.exe"), root / "pythonw.exe")
            # A Windows venv expects Scripts/python.exe. This fixture uses the
            # full-runtime layout, so give it its own stdlib landmark as well.
            ignore = shutil.ignore_patterns("site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter")
            shutil.copytree(Path(sys.base_prefix) / "Lib", root / "Lib", ignore=ignore)
            shutil.copytree(Path(sys.base_prefix) / "DLLs", root / "DLLs")
            for dependency in Path(sys.base_prefix).glob("*.dll"):
                shutil.copy2(dependency, root / dependency.name)
            # Conda keeps ctypes/OpenSSL dependencies apart from CPython's DLLs.
            for pattern in ("*ssl*.dll", "*crypto*.dll", "*ffi*.dll", "*expat*.dll", "*bz*.dll", "*lzma*.dll", "*sqlite*.dll", "*zlib*.dll"):
                for dependency in (Path(sys.base_prefix) / "Library/bin").glob(pattern):
                    shutil.copy2(dependency, root / "DLLs" / dependency.name)
        else:
            (root / "pyvenv.cfg").write_text(
                f"home = {Path(sys.executable).parent}\ninclude-system-site-packages = false\n", encoding="utf-8")
        packages = root / ("Lib/site-packages" if sys.platform == "win32"
                           else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
        distribution = importlib.metadata.distribution("pip")
        for path in distribution.files:
            if path.parts[0] == "pip" or path.parts[0].endswith(".dist-info"):
                if path.suffix == ".pyc":
                    continue
                dest = packages / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(distribution.locate_file(path), dest)
        return python
    return create
