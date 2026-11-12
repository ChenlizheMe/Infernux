from pathlib import Path

from infernux.engine.library_sync import sync_resources


def test_builtin_update_invalidates_startup_catalog_but_unchanged_launch_reuses_it(tmp_path, monkeypatch):
    source = tmp_path / "package"
    shader = source / "shaders" / "lit.frag.py"
    shader.parent.mkdir(parents=True)
    shader.write_text('shader_id = "example/lit"\n', encoding="utf-8")
    monkeypatch.setattr("infernux.resources.get_package_resources_path", lambda: str(source))
    project = tmp_path / "project"
    resources = Path(sync_resources(str(project)))
    cache = project / "Library" / "AssetIndex.startup-cache.json"
    cache.write_text("cached catalog", encoding="utf-8")

    sync_resources(str(project))
    assert cache.read_text(encoding="utf-8") == "cached catalog"

    shader.write_text('shader_id = "example/materials/lit"\n', encoding="utf-8")
    sync_resources(str(project))
    assert not cache.exists()
    assert (resources / "shaders" / "lit.frag.py").read_text(encoding="utf-8") == shader.read_text(encoding="utf-8")

    cache.write_text("cached catalog", encoding="utf-8")
    shader.unlink()
    sync_resources(str(project))
    assert not cache.exists()
    assert not (resources / "shaders" / "lit.frag.py").exists()
