from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
import pytest

import infernux
import infernux as inx


_FORBIDDEN_PUBLIC_IMPORTS = (
    "from Infernux import ",
    "from Infernux.",
    "import Infernux",
    "from infernux import *",
)


def _owned_public_documents(repository: Path) -> list[Path]:
    paths = {
        repository / name
        for name in (
            "README.md",
            "README-zh.md",
            "UpdateLog.md",
            "UpdateLog-zh.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "SUPPORT.md",
            "scripts/README.md",
            "docs/tools/README.md",
        )
    }
    paths.update((repository / "docs" / "learn").glob("*.md"))
    paths.update((repository / "docs" / "learn").glob("*.html"))
    paths.update((repository / "docs" / "wiki" / "docs").rglob("*.md"))
    paths.add(repository / "docs" / "tools" / "apply-api-curation.mjs")
    paths.update((repository / "external" / "plugins").glob("*/README*.md"))
    paths.update((repository / "external" / "plugins").glob("*/package/plugin_pages/*.md"))
    paths.update((repository / "tests" / "fixtures").glob("**/README*.md"))
    return sorted(path for path in paths if path.is_file())


def _owned_public_fixture_scripts(repository: Path) -> list[Path]:
    return sorted(
        (repository / "tests" / "fixtures").glob("**/Assets/Scripts/*.py")
    )


def _forbidden_import_violations(
    repository: Path, paths: list[Path]
) -> list[str]:
    violations: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for marker in _FORBIDDEN_PUBLIC_IMPORTS:
            if marker in text:
                violations.append(f"{path.relative_to(repository)}: {marker}")
    return violations


def test_lowercase_namespace_exposes_gameplay_api() -> None:
    assert inx.__version__
    assert inx.InxComponent.__module__.startswith("infernux.")
    assert inx.GameObject.__module__.startswith("infernux.")
    assert inx.Vector3.__module__.startswith("infernux.")
    assert inx.InxComponent is infernux.InxComponent
    assert inx.GameObject is infernux.GameObject


def test_lowercase_namespace_exposes_audio_device_bus_and_voice_control() -> None:
    from infernux.lib import AudioEngine

    assert inx.AudioEngine is AudioEngine
    assert callable(inx.AudioEngine.instance().fade_bus_volume)
    assert callable(inx.AudioEngine.instance().cancel_bus_fade)


def test_resource_type_members_do_not_pollute_native_module_namespace() -> None:
    import infernux.lib as native

    assert native.ResourceType.Mesh.name == "Mesh"
    assert native.ResourceType.Material.name == "Material"
    assert not hasattr(native, "Mesh")
    assert not hasattr(native, "Material")
    assert not hasattr(native, "Texture")
    assert not hasattr(native, "Shader")
    assert inx.Mesh is infernux.core.Mesh


def test_lowercase_namespace_is_an_actual_package() -> None:
    spec = importlib.util.find_spec("infernux")
    assert spec is not None
    assert spec.submodule_search_locations is not None
    assert Path(spec.origin).name == "__init__.py"
    assert Path(spec.origin).parent.name == "infernux"
    assert not any(type(finder).__name__ == "_PublicNamespaceFinder" for finder in sys.meta_path)


def test_incorrect_package_case_cannot_load_a_second_engine() -> None:
    component = inx.InxComponent
    with pytest.raises(ImportError):
        importlib.import_module("Infernux")
    assert "Infernux" not in sys.modules
    assert inx.InxComponent is component


def test_lowercase_namespace_lazily_forwards_subsystems() -> None:
    for name in (
        "components",
        "core",
        "editor",
        "input",
        "lifecycle",
        "physics",
        "rendergraph",
        "renderstack",
        "resources",
        "scene",
        "ui",
    ):
        assert name in inx.__all__
    assert inx.input.__name__ == "infernux.input"
    assert inx.lifecycle.__name__ == "infernux.lifecycle"
    assert inx.physics.__name__ == "infernux.physics"
    assert inx.renderstack.__name__ == "infernux.renderstack"
    assert inx.resources.__name__ == "infernux.resources"
    assert callable(inx.renderstack.discovery_import_failures)


def test_editor_namespace_reuses_authoritative_registries():
    from infernux.engine.interaction import EditorCommand, EditorCommandRegistry, ShortcutRouter

    assert inx.editor.EditorCommand is EditorCommand
    assert inx.editor.EditorCommandRegistry is EditorCommandRegistry
    assert inx.editor.ShortcutRouter is ShortcutRouter


def test_public_editor_panel_api_reuses_lifecycle_and_translation_authorities():
    from infernux.engine.interaction import PanelInteractionDescriptor
    from infernux.engine.ui.editor_panel import EditorPanel
    from infernux.engine.ui.panel_registry import editor_panel
    from infernux.engine.i18n import t

    assert inx.editor.EditorPanel is EditorPanel
    assert inx.editor.editor_panel is editor_panel
    assert inx.editor.PanelInteractionDescriptor is PanelInteractionDescriptor
    assert inx.editor.translate is t


def test_runtime_ui_public_import_does_not_load_editor_theme() -> None:
    repository = Path(__file__).parents[2]
    python_root = repository / "python"
    code = r'''
import builtins
import sys

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "infernux.engine.ui.theme":
        raise AssertionError("runtime UI imported editor theme")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import infernux as inx

assert inx.ui.UIText().font_size == 18.0
assert inx.ui.UIButton().background_color == [0.922, 0.341, 0.341, 1.0]
assert "infernux.engine.ui.theme" not in sys.modules
'''
    env = os.environ.copy()
    previous_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(python_root), previous_pythonpath) if value
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_lowercase_namespace_exports_lifecycle_with_shared_identity() -> None:
    assert inx.InxPreload is infernux.InxPreload
    assert inx.PreloadContext is infernux.PreloadContext


def test_lowercase_namespace_export_record_is_unique_and_complete() -> None:
    assert len(inx.__all__) == len(set(inx.__all__))
    assert set(infernux.__all__).issubset(inx.__all__)
    for name in inx.__all__:
        assert hasattr(inx, name), name


def test_lowercase_type_stub_explicitly_covers_runtime_exports() -> None:
    stub = (Path(__file__).parents[2] / "python") / "infernux" / "__init__.pyi"
    tree = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    explicit: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        assert all(alias.name != "*" for alias in node.names)
        explicit.update(alias.asname or alias.name for alias in node.names)
    explicit.update(node.target.id for node in tree.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name))
    explicit.update(node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    assert set(inx.__all__) <= explicit



def test_lowercase_namespace_reload_preserves_runtime_type_identity() -> None:
    component_type = inx.InxComponent
    reloaded = importlib.reload(inx)
    assert reloaded.InxComponent is component_type
    assert reloaded.InxComponent is infernux.InxComponent


@pytest.mark.parametrize("suffix", [
    "renderstack", "renderstack.render_stack", "components", "components.component",
    "components.registry", "rendergraph.graph", "engine.path_utils", "lib", "version",
])
def test_lowercase_submodule_imports_share_canonical_identity(suffix) -> None:
    canonical = importlib.import_module("infernux." + suffix)
    original_spec = canonical.__spec__
    public = importlib.import_module("infernux." + suffix)
    assert public is canonical
    assert public.__spec__ is original_spec
    assert public.__spec__.name == "infernux." + suffix


def test_lowercase_submodule_from_import_and_nested_aliases() -> None:
    import infernux.renderstack as rendering
    from infernux.renderstack import RenderStack
    from infernux.renderstack.render_stack import RenderStack as NestedStack
    from infernux.components.component import InxComponent

    assert rendering is infernux.renderstack
    assert RenderStack is NestedStack is rendering.RenderStack
    assert RenderStack is inx.RenderStack
    assert InxComponent is inx.InxComponent


def test_public_gameplay_docs_keep_the_lowercase_namespace_contract() -> None:
    repository = Path(__file__).parents[2]
    gameplay_docs = sorted((repository / "docs" / "learn").glob("gameplay-*.md"))
    assert gameplay_docs

    violations = _forbidden_import_violations(repository, gameplay_docs)
    assert not violations, "\n".join(violations)


def test_public_docs_never_import_the_internal_package() -> None:
    repository = Path(__file__).parents[2]
    public_docs = _owned_public_documents(repository)
    assert len(public_docs) >= 50
    violations = _forbidden_import_violations(repository, public_docs)
    assert not violations, "\n".join(violations)


def test_curated_wiki_examples_reference_existing_public_api() -> None:
    repository = Path(__file__).parents[2]
    checked = 0
    for path in (repository / "docs/wiki/docs").glob("*/api/*.md"):
        text = path.read_text(encoding="utf-8")
        for source in re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL):
            if "import infernux as inx" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            compile(tree, str(path), "exec")
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                attributes = []
                root = node
                while isinstance(root, ast.Attribute):
                    attributes.append(root.attr)
                    root = root.value
                if not isinstance(root, ast.Name) or root.id != "inx":
                    continue
                value = inx
                for attribute in reversed(attributes):
                    assert hasattr(value, attribute), (path, ast.unparse(node))
                    value = getattr(value, attribute)
            checked += 1
    assert checked >= 28


def test_public_fixture_scripts_use_the_lowercase_namespace() -> None:
    repository = Path(__file__).parents[2]
    scripts = _owned_public_fixture_scripts(repository)
    assert scripts
    violations = _forbidden_import_violations(repository, scripts)
    assert not violations, "\n".join(violations)
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        assert "import infernux as inx" in source, path.relative_to(repository)
