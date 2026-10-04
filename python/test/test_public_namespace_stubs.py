"""Public lowercase imports must expose the canonical type declarations."""

from pathlib import Path
import runpy

import pytest
import setuptools


def _generator(monkeypatch):
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    source = Path(__file__).resolve().parents[2] / "setup.py"
    return runpy.run_path(str(source))["generate_public_namespace_stubs"]


def test_public_stub_package_preserves_declarations_and_removes_deleted_modules(tmp_path, monkeypatch):
    generate = _generator(monkeypatch)
    monkeypatch.setitem(generate.__globals__, "_case_insensitive_module_paths", lambda _path: True)
    canonical = tmp_path / "Infernux"
    (canonical / "renderstack").mkdir(parents=True)
    (canonical / "__init__.pyi").write_text("from Infernux.components import InxComponent as InxComponent\n")
    (canonical / "renderstack/__init__.py").write_text("from Infernux.renderstack.render_stack import RenderStack\n")
    (canonical / "renderstack/render_stack.py").write_text("raise RuntimeError('runtime body')\n")
    declaration = "class RenderStack:\n    def refresh(self) -> None: ...\n"
    (canonical / "renderstack/render_stack.pyi").write_text(declaration)
    (canonical / "renderstack/gone.py").write_text("class Removed: pass\n")
    (canonical / "native.pyd").write_bytes(b"native")
    generate(str(tmp_path))
    public = tmp_path / "infernux-stubs"
    assert (public / "renderstack/render_stack.pyi").read_text() == declaration
    assert (public / "renderstack/__init__.pyi").read_text() == (
        canonical / "renderstack/__init__.py"
    ).read_text()
    assert (public / "__init__.pyi").read_text() == (canonical / "__init__.pyi").read_text()
    assert (public / "py.typed").read_text() == "partial\n"
    assert not list(public.rglob("*.py"))
    assert not list(public.rglob("*.pyd"))
    (canonical / "renderstack/gone.py").unlink()
    generate(str(tmp_path))
    assert not (public / "renderstack/gone.pyi").exists()
    assert (canonical / "renderstack/render_stack.pyi").read_text() == declaration


def test_public_stub_generation_requires_authoritative_root_declarations(tmp_path, monkeypatch):
    generate = _generator(monkeypatch)
    with pytest.raises(RuntimeError, match="public type declarations"):
        generate(str(tmp_path))
    assert not (tmp_path / "infernux-stubs").exists()


def test_case_sensitive_stub_package_reexports_canonical_types(tmp_path, monkeypatch):
    generate = _generator(monkeypatch)
    monkeypatch.setitem(generate.__globals__, "_case_insensitive_module_paths", lambda _path: False)
    canonical = tmp_path / "Infernux"
    (canonical / "renderstack").mkdir(parents=True)
    (canonical / "__init__.pyi").write_text("__version__: str\n")
    (canonical / "renderstack/__init__.py").write_text("from .render_stack import RenderStack\n")
    (canonical / "renderstack/render_stack.py").write_text("class RenderStack: pass\n")
    generate(str(tmp_path))
    public = tmp_path / "infernux-stubs"
    assert (public / "__init__.pyi").read_text() == (
        "from Infernux import *\nfrom Infernux import __version__ as __version__\n"
    )
    assert (public / "renderstack/__init__.pyi").read_text() == "from Infernux.renderstack import *\n"
    assert (public / "renderstack/render_stack.pyi").read_text() == (
        "from Infernux.renderstack.render_stack import *\n"
    )
