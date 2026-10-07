"""Queries use startup configuration only when the author omits the layer mask."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


QUERIES = ["raycast", "raycast_all", "raycast_batch", "overlap_sphere", "overlap_box",
           "overlap_capsule", "sphere_cast", "box_cast", "capsule_cast", "query_rigidbodies_in_bounds"]
MODES = {"default": None, "none": None, "empty": 0, "layer0": 1, "layer2": 4, "all": 0xFFFFFFFF}
SINGLE = {"raycast", "raycast_batch", "sphere_cast", "box_cast", "capsule_cast"}


@pytest.fixture(scope="module", params=[0, 4, 0x80000000, 0xFFFFFFFB])
def observed(request, tmp_path_factory):
    project = tmp_path_factory.mktemp("query-mask")
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
        str(project), str(request.param)], env=os.environ.copy(), capture_output=True, text=True,
        encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads((project / "query-masks.json").read_text(encoding="utf-8"))
    assert value["configured"] == request.param
    return value


@pytest.mark.parametrize("api", ["native", "public"])
@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("mode", MODES)
def test_query_mask_resolution(observed, api, query, mode):
    mask = observed["configured"] if MODES[mode] is None else MODES[mode]
    selected = [int(value) for layer, value in observed["owners"].items() if mask & (1 << int(layer))]
    if query in SINGLE:
        selected = selected[-1:]
    assert observed["results"][api][query][mode] == sorted(selected)


@pytest.mark.parametrize("mode", MODES)
def test_screen_ray_uses_configured_mask(observed, mode):
    assert observed["screen"][mode] == observed["results"]["public"]["raycast"][mode]


@pytest.mark.parametrize("mode", MODES)
def test_mapped_buffer_query_uses_configured_mask(observed, mode):
    assert_buffer_query(observed, "combined", mode)


@pytest.fixture(scope="module", params=[0, 4, 0x80000000, 0xFFFFFFFB])
def observed_gpu(request, tmp_path_factory):
    project = tmp_path_factory.mktemp("query-mask-gpu")
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
        str(project), str(request.param), "gpu"], env=os.environ.copy(), capture_output=True,
        text=True, encoding="utf-8", timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(token in result.stdout + result.stderr for token in ("VUID-", "SYNC-HAZARD"))
    value = json.loads((project / "query-masks.json").read_text(encoding="utf-8"))
    assert value["configured"] == request.param
    return value


@pytest.mark.parametrize("query", ["box", "combined"])
@pytest.mark.parametrize("mode", MODES)
def test_gpu_buffer_query_uses_configured_mask(observed_gpu, query, mode):
    assert_buffer_query(observed_gpu, query, mode)


def assert_buffer_query(observed, query, mode):
    mask = observed["configured"] if MODES[mode] is None else MODES[mode]
    selected = [int(value) for layer, value in observed["owners"].items() if mask & (1 << int(layer))]
    row = observed["buffers"][query][mode]
    assert row["owners"] == sorted(selected)
    assert row["count"] == len(selected)
    expected_y = [i * 2 for i, layer in enumerate((0, 2, 31)) if mask & (1 << layer)]
    assert row["centers_y"] == expected_y
    if query == "combined":
        assert row["positions_y"] == expected_y


def exercise_buffers(evidence, gpu):
    import infernux as inx
    import numpy as np
    from infernux.physics import Physics

    # Exercise the engine's actual Web CPU mapping route on the native test host.
    if not gpu:
        os.environ["INFERNUX_WEB_RUNTIME"] = "1"
    state = {name: inx.buffer(shape=3, dtype=inx.vector3, device="gpu") for name in
             ("position", "center_of_mass", "linear_velocity", "angular_velocity", "inverse_mass")}
    state["rotation"] = inx.buffer(shape=3, dtype=inx.vector4, device="gpu")
    state["inverse_inertia"] = inx.buffer(shape=(3, 3, 3), dtype=np.float32, device="gpu")
    boxes = {name: inx.buffer(shape=3, dtype=dtype, device="gpu") for name, dtype in (
        ("body_index", np.int32), ("center", inx.vector3), ("rotation", inx.vector4),
        ("half_extents", inx.vector3), ("friction", np.float32), ("bounciness", np.float32))}
    try:
        rows = evidence["buffers"] = {}
        for query in (["box", "combined"] if gpu else ["combined"]):
            rows[query] = {}
            for mode, mask in MODES.items():
                kwargs = {} if mode == "default" else {"layer_mask": mask}
                if query == "box":
                    Physics.query_rigidbody_box_states_in_bounds((-2, -2, -2), (2, 6, 2), boxes, **kwargs)
                else:
                    Physics.query_rigidbody_states_and_box_states_in_bounds(
                        (-2, -2, -2), (2, 6, 2), state, boxes, **kwargs)
                count = boxes["count"]
                read = lambda buffer: buffer.get_data().numpy() if gpu else buffer.numpy()
                row = rows[query][mode] = {
                    "owners": sorted(body.game_object.id for body in boxes["rigidbodies"]),
                    "count": count, "centers_y": sorted(read(boxes["center"])[:count, 1].tolist()),
                }
                if query == "combined":
                    row["positions_y"] = sorted(read(state["position"])[:count, 1].tolist())
    finally:
        for values in (state, boxes):
            for value in values.values():
                if isinstance(value, inx.compute.Buffer):
                    value.close()
        if not gpu:
            del os.environ["INFERNUX_WEB_RUNTIME"]


def batch_output():
    import numpy as np
    return {name: np.zeros(shape, dtype=dtype) for name, shape, dtype in (
        ("hit", 1, "uint8"), ("point", (1, 3), "float32"), ("normal", (1, 3), "float32"),
        ("distance", 1, "float32"), ("body_id", 1, "uint32"), ("sub_shape_id", 1, "uint32"),
        ("triangle_index", 1, "uint32"), ("collider_id", 1, "uint64"), ("game_object_id", 1, "uint64"))}


def exercise(project, configured, gpu=False):
    import numpy as np
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import EngineConfig, LogLevel, Physics, RuntimeMode, SceneManager, Vector3
    from infernux.physics import Physics as PublicPhysics

    EngineConfig.get().default_query_layer_mask = configured
    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Graphical if gpu else RuntimeMode.Headless)
    evidence = {"owners": {}, "results": {}}
    try:
        if gpu:
            engine.init_renderer(64, 64, str(project))
        else:
            engine.init_headless(str(project))
        evidence["configured"] = EngineConfig.get().default_query_layer_mask
        scene = SceneManager.instance().get_active_scene()
        for index, layer in enumerate((0, 2, 31)):
            owner = scene.create_game_object("Layer" + str(layer))
            owner.layer = layer
            owner.transform.position = Vector3(0, index * 2, 0)
            owner.add_component("BoxCollider")
            owner.add_component("Rigidbody").use_gravity = False
            evidence["owners"][layer] = owner.id
        Physics.sync_transforms()
        if gpu:
            exercise_buffers(evidence, gpu=True)
            return
        origin, direction = Vector3(0, 8, 0), Vector3(0, -1, 0)
        for api_name, api in (("native", Physics), ("public", PublicPhysics)):
            rows = evidence["results"][api_name] = {}
            for query in QUERIES:
                rows[query] = {}
                for mode, mask in MODES.items():
                    kwargs = {} if mode == "default" else {"layer_mask": mask}
                    fn = getattr(api, query)
                    if query == "raycast_batch":
                        output = fn(np.array([[0, 8, 0]], dtype=np.float32),
                            np.array([[0, -1, 0]], dtype=np.float32), batch_output(), 12, **kwargs)
                        ids = [int(output["game_object_id"][0])] if output["hit"][0] else []
                    else:
                        if query in ("raycast", "raycast_all"):
                            value = fn(origin, direction, 12, **kwargs)
                        elif query == "sphere_cast":
                            value = fn(origin, .25, direction, 12, **kwargs)
                        elif query == "box_cast":
                            value = fn(origin, Vector3(.25, .25, .25), direction, max_distance=12, **kwargs)
                        elif query == "capsule_cast":
                            value = fn(Vector3(0, 7.75, 0), Vector3(0, 8.25, 0), .25, direction, 12, **kwargs)
                        elif query == "overlap_sphere":
                            value = fn(Vector3(0, 2, 0), 6, **kwargs)
                        elif query == "overlap_box":
                            value = fn(Vector3(0, 2, 0), Vector3(2, 6, 2), **kwargs)
                        elif query == "overlap_capsule":
                            value = fn(Vector3(0, 0, 0), Vector3(0, 4, 0), 2, **kwargs)
                        else:
                            value = fn(Vector3(-2, -2, -2), Vector3(2, 6, 2), **kwargs)
                        values = ([value] if value is not None else []) if query in SINGLE else value
                        ids = [item.game_object.id for item in values]
                    rows[query][mode] = sorted(ids)
        class CameraRay:
            def screen_point_to_ray(self, *_):
                return origin, direction
        evidence["screen"] = {}
        for mode, mask in MODES.items():
            kwargs = {} if mode == "default" else {"layer_mask": mask}
            hit = PublicPhysics.raycast_screen(CameraRay(), (32, 32), (64, 64), 12, **kwargs)
            evidence["screen"][mode] = [] if hit is None else [hit.game_object.id]
        exercise_buffers(evidence, gpu=False)
    finally:
        engine.exit()
        (project / "query-masks.json").write_text(json.dumps(evidence), encoding="utf-8")


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), int(sys.argv[2]), len(sys.argv) > 3 and sys.argv[3] == "gpu")
