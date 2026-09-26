"""Bounded matching is explicit, deterministic and never publishes imports."""
from types import SimpleNamespace

import pytest

from Infernux.engine.ui.model_material_search import find_material_candidates


@pytest.mark.parametrize("scope,expected", [("local", 2), ("upwards", 2), ("project", 3)])
def test_directory_scope_and_ambiguity(tmp_path, scope, expected):
    root = tmp_path / "Assets"
    model = root / "Models/Car/body.fbx"
    paths = [root / p for p in ("Models/Car/Red.mat", "Models/Car/Materials/Red.mat",
                                "Elsewhere/Red.mat", "Models/Car/Red.mat.bak")]
    paths.extend([tmp_path / "AssetsOther/Red.mat", root / "Models/Car/Red.mat"])
    result = find_material_candidates(model, [{"source_id": "material/Red"}], paths, root, scope=scope)
    assert len(result["material/Red"]) == expected
    assert len(set(result["material/Red"])) == expected
    assert result == find_material_candidates(model, [{"source_id": "material/Red"}], reversed(paths), root, scope=scope)


def test_nearest_parent_per_material_relative_paths_and_naming(tmp_path):
    root = tmp_path / "Assets"
    paths = ["Assets/Models/Shared/Red.mat", "Assets/Red.mat", "Assets/Blue.mat",
             "Assets/Models/Car/body-Green.mat"]
    slots = [{"source_id": "material/" + n} for n in ("Red", "Blue", "Missing")]
    slots.append({"source_id": ""})
    result = find_material_candidates("Assets/Models/Car/body.fbx", slots, paths, root, scope="upwards")
    assert result["material/Red"] == (str(root / "Models/Shared/Red.mat"),)
    assert result["material/Blue"] == (str(root / "Blue.mat"),)
    assert result["material/Missing"] == ()
    assert "" not in result
    result = find_material_candidates("Assets/Models/Car/body.fbx", [{"source_id": "material/Green"}],
                                      paths, root, naming="model_material")
    assert result["material/Green"] == (str(root / "Models/Car/body-Green.mat"),)


@pytest.mark.parametrize("options", [{"scope": "guess"}, {"naming": "guess"}])
def test_invalid_rules_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        find_material_candidates(tmp_path / "a.obj", [], [], tmp_path, **options)


def test_package_model_can_search_project_without_walking_outside_assets(tmp_path):
    paths = [tmp_path / "Assets/Red.mat"]
    slots = [{"source_id": "material/Red"}]
    model = tmp_path / "Packages/demo/a.fbx"
    assert find_material_candidates(model, slots, paths, tmp_path / "Assets", scope="upwards")["material/Red"] == ()
    assert find_material_candidates(model, slots, paths, tmp_path / "Assets", scope="project")["material/Red"] == (str(paths[0]),)


def test_search_requires_explicit_confirmation_and_invalidates_catalog_changes(monkeypatch, tmp_path):
    from Infernux.core.assets import AssetManager
    from Infernux.engine.interaction import asset_reference_catalog
    from Infernux.engine.ui import asset_details_renderer as ui

    root = tmp_path / "Assets"
    path = root / "Red.mat"
    db = SimpleNamespace(assets_root=str(root), query_generation=1)
    monkeypatch.setattr(AssetManager, "_asset_database", db)
    monkeypatch.setattr(ui, "field_label", lambda *a: None)
    monkeypatch.setattr(AssetManager, "_get_guid_from_path", lambda _: "red-guid")
    scans = []
    monkeypatch.setattr(asset_reference_catalog, "items", lambda *a: scans.append(a) or [("Red.mat", str(path))])
    state = SimpleNamespace(extra={}, file_path=str(root / "a.fbx"))
    mesh = SimpleNamespace(generation=1)
    assigned = []
    class Context:
        click = ""
        def combo(self, label, index, choices): return index
        def record_semantic_item(self, *a): pass
        def button(self, label): return self.click in label if self.click else False
        def label(self, *a): pass
        def text_wrapped(self, *a): pass
        def separator(self): pass
    ctx = Context()
    def draw():
        ui._render_model_material_search(ctx, state, mesh, [{"source_id": "material/Red"}],
                                        lambda *a: assigned.append(a))
    draw()
    assert not scans and not assigned
    ctx.click = "##model_material_search"
    draw()
    assert len(scans) == 1 and not assigned
    ctx.click = "##material_search_material/Red_red-guid"
    draw()
    assert assigned == [("material/Red", "red-guid")]
    db.query_generation += 1
    draw()  # A stale candidate cannot be applied after delete/move/reimport.
    assert len(assigned) == 1 and len(scans) == 1
    assert "results" not in state.extra["model_material_search"]
