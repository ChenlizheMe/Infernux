"""On-demand renderer observations reuse native state without submitting work."""

from types import SimpleNamespace

from infernux.host import EditorAutomationHost


def test_performance_observation_includes_texture_publication_and_gpu_residency(monkeypatch):
    window = {"sample_count": 5, "timings": {"frame": {"max_ms": 3.0}}}
    residency = {
        "device_wait_idle_count": 2,
        "pending_material_texture_descriptor_set_count": 1,
        "retired_material_descriptor_set_count": 3,
        "material_descriptor_set_count": 7,
    }
    editor_ui = {
        "gui_frame": 20, "gui_frame_consume_count": 100,
        "gui_frame_approved_count": 20, "gui_frame_forced_count": 10,
        "gui_frame_request_count": 50, "ui_panel_times_ms": {"Game": 0.1},
        "ui_panel_sub_times_ms": {"Game.Draw": 0.05},
    }
    renderer = {
        "frame": 25, "scene_view_visible": True,
        "scene_target_width": 640, "scene_target_height": 480,
        "game_target_width": 320, "game_target_height": 240,
        "scene_draw_call_count": 12, "game_draw_call_count": 6,
    }
    native = SimpleNamespace(
        get_renderer_performance_window=lambda: window,
        renderer_frame_snapshot={**editor_ui, **renderer},
        resident_mesh_vertex_buffer_count=1,
        pending_mesh_gpu_upload_count=0,
        submitted_mesh_gpu_upload_count=5,
        completed_mesh_gpu_upload_count=5,
        pending_texture_cpu_load_count=2,
        pending_texture_gpu_upload_count=1,
        submitted_texture_gpu_upload_count=4,
        completed_texture_gpu_upload_count=3,
        gpu_residency_snapshot=residency,
    )
    host = EditorAutomationHost()
    monkeypatch.setattr(host, "_native_engine", lambda: native)

    observed = host.renderer_performance_window()

    assert observed["resources"]["pending_texture_cpu_loads"] == 2
    assert observed["resources"]["pending_texture_uploads"] == 1
    assert observed["resources"]["submitted_texture_uploads"] == 4
    assert observed["resources"]["completed_texture_uploads"] == 3
    assert observed["gpu_residency"] == residency
    assert observed["gpu_residency"] is not residency
    assert observed["editor_ui"] == editor_ui
    assert observed["renderer"] == renderer
    assert "resources" not in window  # The native timing result is not modified.
