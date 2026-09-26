"""Exercise the public native material-refresh API before any UI draw."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from Infernux import Engine
from Infernux.lib import InxMaterial


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="infernux-ui-refresh-") as root:
        project = Path(root)
        assets = project / "Assets"
        assets.mkdir()
        (project / "ProjectSettings").mkdir()
        vertex = assets / "RefreshScreenUI.vert"
        fragment = assets / "RefreshScreenUI.frag"
        world_fragment = assets / "RefreshWorldUI.frag"
        vertex.write_text(
            'ShaderInfo { Name "RefreshScreenUIVertex" Capabilities [ScreenUI] }\n'
            '#version 450\nvoid main() { gl_Position = vec4(0.0); }\n',
            encoding="utf-8",
        )
        fragment.write_text(
            'ShaderInfo { Name "RefreshScreenUIFragment" Capabilities [ScreenUI] }\n'
            '#version 450\nlayout(location=0) out vec4 color; '
            'void main() { color = vec4(1.0); }\n',
            encoding="utf-8",
        )
        world_fragment.write_text(
            'ShaderInfo { Name "RefreshWorldUIFragment" Capabilities [WorldUI] }\n'
            '#version 450\nlayout(location=0) out vec4 color; '
            'void main() { color = vec4(1.0); }\n',
            encoding="utf-8",
        )

        frontend = Engine()
        engine = frontend.get_native_engine()
        try:
            try:
                frontend.init_renderer(64, 64, str(project))
            except (OSError, RuntimeError) as error:
                print(f"UI material refresh smoke test skipped: {error}")
                return 77
            database = frontend.get_asset_database()
            vertex_guid = database.get_guid_from_path(str(vertex))
            fragment_guid = database.get_guid_from_path(str(fragment))
            world_fragment_guid = database.get_guid_from_path(str(world_fragment))
            assert vertex_guid and fragment_guid and world_fragment_guid

            material = InxMaterial("UndrawnScreenUI")
            material.vert_shader_reference = {
                "guid": vertex_guid, "shader_id": "RefreshScreenUIVertex", "path_hint": ""
            }
            material.frag_shader_reference = {
                "guid": fragment_guid, "shader_id": "RefreshScreenUIFragment", "path_hint": ""
            }
            assert not engine.is_shader_loaded("RefreshScreenUIVertex", "vertex")
            assert engine.refresh_material_pipeline(material)
            assert not engine.is_shader_loaded("RefreshScreenUIVertex", "vertex"), (
                "refresh published an ownerless UI program before the first draw"
            )
            # Exercise the actual native particle graph publication entry. Its
            # bytecode is only a decoder fixture: domain rejection must happen
            # before any particle GPU pipeline can consume it.
            spirv_header = (0x07230203).to_bytes(4, "little") + bytes(16)
            particle_stages = (
                "bootstrap", "init", "update", "contact_prepare", "contact_solve",
                "contact_dispatch", "render_reset", "rendering",
            )
            particle_program = {
                "id": 101, "graph_instance_id": 77, "graph_emitter_index": 0,
                "owner_object_id": 1, "owner_layer_mask": 1,
                "artifact_revision": 1, "stable_id": "wrong-ui-domain",
                "capacity": 1, "state_stride": 16, "event_type_count": 0,
                "collision_enabled": False, "parameter_words": [],
                "continuation": None,
                "update_render_fusion": {"eligible": False, "fused_stage": ""},
                "stages": {stage: spirv_header for stage in particle_stages},
                "billboard": {stage: spirv_header for stage in (
                    "vertex", "picking_fragment", "motion_vertex", "motion_fragment")},
                "mesh_shaders": {stage: spirv_header for stage in (
                    "vertex", "shadow_fragment", "picking_fragment", "motion_vertex", "motion_fragment")},
                "outputs": [{
                    "id": 102, "stable_id": "sprite", "output_type": "sprite",
                    "mesh": None, "material": {
                        "native": material, "render_queue": 3000, "blend_enabled": True,
                        "depth_test_enabled": True, "depth_write_enabled": False,
                    },
                    "receive_scene_lighting": False, "receive_shadows": False,
                    "cast_shadows": False, "soft_particles": False,
                    "soft_distance": 1.0, "sort_mode": "none",
                    "ribbon_uv_mode": "stretch", "ribbon_uv_scale": 1.0,
                    "flipbook_columns": 1, "flipbook_rows": 1,
                    "sprite_alignment": "camera_plane", "alignment_axis": [0.0, 1.0, 0.0],
                }],
            }
            particle_error = engine._replace_gpu_particle_graph(77, [particle_program], [])
            assert "domain mismatch" in particle_error.lower(), particle_error
            assert not engine.is_shader_loaded("RefreshScreenUIVertex", "vertex"), (
                "particle output published an ownerless UI program"
            )
            mixed = InxMaterial("MixedUIDomains")
            mixed.vert_shader_reference = material.vert_shader_reference
            mixed.frag_shader_reference = {
                "guid": world_fragment_guid, "shader_id": "RefreshWorldUIFragment", "path_hint": ""
            }
            assert not engine.refresh_material_pipeline(mixed)
            assert not engine.is_shader_loaded("RefreshWorldUIFragment", "fragment")
            # An ordinary material keeps the existing eager mesh refresh path.
            assert engine.refresh_material_pipeline(InxMaterial.create_default_lit())
        finally:
            engine.cleanup()
    print("UI material refresh smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
