"""On-demand renderer observations reuse native state without submitting work."""

from types import SimpleNamespace

from Infernux.host import EditorAutomationHost


def test_performance_observation_includes_texture_publication_and_gpu_residency(monkeypatch):
    window = {"sample_count": 5, "timings": {"frame": {"max_ms": 3.0}}}
    residency = {
        "device_wait_idle_count": 2,
        "pending_material_texture_descriptor_set_count": 1,
        "retired_material_descriptor_set_count": 3,
        "material_descriptor_set_count": 7,
    }
    native = SimpleNamespace(
        get_renderer_performance_window=lambda: window,
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
    assert "resources" not in window  # The native timing result is not modified.
