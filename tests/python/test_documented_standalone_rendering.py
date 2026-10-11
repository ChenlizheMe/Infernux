"""The tutorial's standalone host must execute a real native camera graph."""
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("language", ("en", "zh"))
@pytest.mark.parametrize("samples", (1, 4))
def test_documented_provider_executes_native_standalone(tmp_path, language, samples):
    tutorial = Path(__file__).resolve().parents[2] / "docs/learn/rendergraph-advanced.md"
    script = textwrap.dedent('''
        import json, re, sys
        from pathlib import Path
        import infernux as inx
        from infernux.lib import PrimitiveType, SceneManager

        project = Path(sys.argv[1])
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        english, chinese = Path(sys.argv[2]).read_text(encoding="utf-8").split("<!-- language:zh -->")
        text = english if sys.argv[3] == "en" else chinese
        code = next(block for block in re.findall(r"```python\\n(.*?)```", text, re.S)
                    if "class BaseColorPresentPipeline(" in block)
        namespace = {"__name__": __name__}
        exec(compile(code, sys.argv[2], "exec"), namespace)
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        frames, observations, errors = 0, [], []
        try:
            frontend.init_renderer(128, 128, str(project))
            scene = SceneManager.instance().create_scene("Standalone Provider")
            SceneManager.instance().set_active_scene(scene)
            camera = scene.create_game_object("Camera")
            camera.transform.position = inx.Vector3(0, 0, -5)
            camera.add_component("Camera")
            scene.create_primitive(PrimitiveType.Cube, "Provider Cube")
            pipeline = namespace["BaseColorPresentPipeline"]()
            native.set_msaa_samples(int(sys.argv[4]))
            native.resize_game_render_target(128, 128)
            native.set_scene_view_visible(False)
            native.set_game_camera_enabled(True)
            native.set_render_pipeline(pipeline)

            def after_draw():
                global frames
                frames += 1
                try:
                    snapshot = native.renderer_frame_snapshot
                    if frames >= 3:
                        names = snapshot["game_render_graph_pass_names"]
                        assert snapshot["game_render_graph_current_executed"], snapshot
                        assert all(name in names for name in ("DepthPrepass", "opaque_preview_color", "CopyToCamera", "Present")), names
                        assert names.count("_DisplayEncode") == names.count("_ScreenUI_Camera") == names.count("_ScreenUI_Overlay") == 1, names
                        assert snapshot["game_draw_call_count"] >= 1, snapshot
                        observations.append({"frame": snapshot["frame"], "draws": snapshot["game_draw_call_count"], "passes": names})
                    if frames >= 5:
                        native.exit()
                except BaseException as error:
                    errors.append(repr(error))
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            assert not errors, errors
            assert len(observations) == 3
            assert pipeline._defining_graph is None
            assert len(pipeline._standalone_graphs) == 1
            output_samples, = pipeline._standalone_graphs
            # Screen output reports zero: the pipeline owns screen MSAA. The
            # exact tutorial selects 1x, overriding the initial engine setting.
            assert output_samples == 0
            assert pipeline._standalone_desc.msaa_samples == native.get_msaa_samples() == 1
            assert observations[-1]["frame"] > observations[0]["frame"]
            print("STANDALONE_PROVIDER " + json.dumps({"samples": native.get_msaa_samples(), "observations": observations}))
        finally:
            native.set_post_draw_callback(None)
            native.set_render_pipeline(None)
            native.cleanup()
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(tutorial), language, str(samples)],
        capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    record, = [line.removeprefix("STANDALONE_PROVIDER ") for line in result.stdout.splitlines()
               if line.startswith("STANDALONE_PROVIDER ")]
    assert len(json.loads(record)["observations"]) == 3
    assert not any(marker in output for marker in ("[ERROR]", "[FATAL]", "VUID-", "SYNC-HAZARD", "Validation Error", "Traceback (most recent call last)")), output
