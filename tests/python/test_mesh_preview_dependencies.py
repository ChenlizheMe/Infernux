import json
import pytest

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, _Infernux
from tests.gpu.mesh_preview_dependency_case import MeshPreviewDependencyCase, measure_preview_pixels
from test_material_preview_retirement import pump_until


@pytest.mark.parametrize(
    "mode,consumer", [("edit", "python"), ("history", "native"), ("texture_import", "python"), ("snapshot", "native")]
)
def test_model_and_submesh_previews_follow_published_dependencies(engine, scene, tmp_path, monkeypatch, consumer, mode):
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", engine.get_asset_database())
    case = MeshPreviewDependencyCase(engine, tmp_path / mode, mode, consumer)
    try:

        def tick(_):
            # The native fixture does not install Engine's Python post-draw
            # owner. Drain the same deferred texture publication at this point.
            AssetManager.flush_pending_gpu_texture_reloads(paths=[str(case.texture)])
            return case.tick()["done"]

        pump_until(engine, tick)
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0, console._get_visible_log_snapshot(100)
    finally:
        (tmp_path / "preview-dependencies.json").write_text(json.dumps(case.observations, indent=2), encoding="utf-8")
        case.close()


@pytest.mark.parametrize("stage", [pytest.param("rendering", id="uploading")])
def test_mesh_preview_finishes_latest_request_without_repeated_queries(engine, scene, tmp_path, monkeypatch, stage):
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", engine.get_asset_database())
    case = MeshPreviewDependencyCase(engine, tmp_path / stage, "edit", "native")
    key, path = case.keys["model"], case.paths["model"]
    phase = "baseline"
    baseline = None
    settled = 0
    gate = None
    published_before_edit = None
    observations = []

    def publish(color):
        case.material.set_vector4("base_color", *color, 1.0)
        AssetManager.note_asset_edit(str(case.material_path), material_json=case.material.serialize())

    def supersede(row):
        nonlocal phase, published_before_edit
        observations.append(row)
        published_before_edit = row["ready_generation"]
        publish((1.0, 0.0, 0.0))
        phase = "latest"

    def after_host(_foreground_ready):
        nonlocal gate
        if phase != "supersede":
            return
        row = next(dict(r) for r in engine.preview_task_snapshots if r["resource_key"] == key)
        if not row["has_render_ticket"] and row["ready_generation"] != baseline["generation"] + 1:
            return
        # Late service can submit and publish a small preview in one host
        # frame. Observe either its pending ticket or its completed image.
        gate = engine.frame_work.add_source(
            None, lambda: _Infernux._FrameWorkFacts(gpu_busy=True), lambda gpu, lane: False
        )
        engine.frame_work.wake(gate)
        supersede(row)

    try:
        engine.query_or_schedule_mesh_preview(key, path, 0)
        if stage == "rendering":
            engine.set_end_frame_callback(after_host)

        def tick(_):
            nonlocal phase, baseline, settled, gate
            row = next(dict(r) for r in engine.preview_task_snapshots if r["resource_key"] == key)
            if phase == "latest":
                assert row["ready_generation"] in (published_before_edit, baseline["generation"] + 2), row
            ready = row["texture_id"] and row["generation"] == row["ready_generation"] and not row["in_flight"]
            if phase == "latest" and gate is not None:
                # Unrelated background GPU work must not hold up an interactive
                # preview. Keep it busy until the newest image is published;
                # only then release the gate for the verification readback.
                if not ready:
                    return False
                assert row["generation"] == baseline["generation"] + 2, row
                observations.append(row)
                engine.frame_work.remove_source(gate)
                gate = None
            if ready:
                row = measure_preview_pixels(engine, row, case.pixel_reads)
                if row is None:
                    return False
            if phase == "baseline" and ready:
                assert engine.query_or_schedule_mesh_preview(key, str(case.source), 0) == row["texture_id"]
                same_source = next(r for r in engine.preview_task_snapshots if r["resource_key"] == key)
                assert same_source["generation"] == row["generation"], same_source
                baseline = row
                observations.append(row)
                publish((0.0, 0.0, 1.0))
                engine.query_or_schedule_mesh_preview(key, path, 0)
                phase = "supersede"
            if phase == "supersede" and stage == "queued":
                supersede(next(dict(r) for r in engine.preview_task_snapshots if r["resource_key"] == key))
            elif phase == "latest":
                if ready:
                    assert row["generation"] == baseline["generation"] + 2, row
                    assert row["measured_generation"] == row["generation"], row
                    assert row["measured_hash"] == baseline["measured_hash"], (baseline, row)
                    settled += 1
                    if settled >= 30:
                        observations.append(row)
                        return True
            return False

        pump_until(engine, tick)
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0, console._get_visible_log_snapshot(100)
    finally:
        if stage == "rendering":
            engine.set_end_frame_callback(None)
        if gate is not None:
            engine.frame_work.remove_source(gate)
        (tmp_path / "preview-supersession.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
        case.close()


@pytest.mark.parametrize("replacement", ["recreate", "reimport"])
def test_material_content_deduplication_does_not_outlive_its_source(engine, scene, tmp_path, monkeypatch, replacement):
    from infernux.lib import InxMaterial

    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", engine.get_asset_database())
    directory = tmp_path / replacement
    case = MeshPreviewDependencyCase(engine, directory, "edit", "native")
    try:
        pump_until(engine, lambda _: case.tick()["done"])
        if replacement == "recreate":
            case.close()
            case = None
            case = MeshPreviewDependencyCase(engine, directory, "edit", "native")
        else:
            authored = InxMaterial.create_default_unlit()
            authored.set_vector4("base_color", 1.0, 0.0, 0.0, 1.0)
            assert authored.save_to(str(case.material_path))
            assert AssetManager.reimport_asset(str(case.material_path))
            case.material = case.registry.load_material(str(case.material_path))
            case.step, case.frames, case.phase = "baseline", 0, 0
        pump_until(engine, lambda _: case.tick()["done"])
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0, console._get_visible_log_snapshot(100)
    finally:
        if case is not None:
            (tmp_path / "preview-source-replacement.json").write_text(
                json.dumps(case.observations, indent=2), encoding="utf-8"
            )
            case.close()
