import json
from pathlib import Path

import numpy as np

from infernux import compute
from infernux._compiler import slang as frontend
from infernux.engine.build import compute_aot
from infernux.engine.build.compute_aot import (
    ensure_engine_compute_artifacts,
    stage_compute_artifacts,
)
from infernux.engine.game_builder import GameBuilder
from infernux.engine.player_package_native import read_manifest


def _write_artifact(path: Path, function: str) -> None:
    from infernux._compiler.source_metadata import embed_compute_sources
    from infernux.engine.project_context import using_project_root

    module, name = function.rsplit(".", 1)
    project = path.parents[3]
    source = project / "Assets/Scripts" / (module.rsplit(".", 1)[-1] + ".py")
    source.parent.mkdir(parents=True, exist_ok=True)
    text = ("import infernux as inx\n@inx.compute.kernel\n" + f"def {name}(domain):\n"
            "    i=inx.compute.index(domain)\n    domain[i]+=1\n")
    source.write_text(text, encoding="utf-8")
    namespace = {"__name__": module, "__file__": str(source)}
    exec(compile(embed_compute_sources(text), str(source), "exec"), namespace)
    params = (compute_aot._gpu_buffer_descriptor((1,), np.int32),)
    with using_project_root(project):
        frontend.compile_kernel(namespace[name].function, params)
        plan = frontend.prepare_kernel(namespace[name].function, params)
    (path.parent / (plan.artifact_key + ".inxgpu")).replace(path)


def _write_kernel(project: Path) -> Path:
    source = project / "Assets/Scripts/Jelly.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import infernux as inx\n"
        "from infernux import compute as gpu\n"
        "@inx.compute.function\n"
        "def helper(value):\n"
        "    return value\n"
        "@gpu.kernel\n"
        "def step(domain):\n"
        "    i = inx.compute.index(domain)\n"
        "    domain[i] = helper(domain[i])\n",
        encoding="utf-8",
    )
    return source


def test_stage_compute_artifacts_seals_selected_and_engine_kernels(tmp_path, monkeypatch):
    from infernux._compiler.source_metadata import embed_compute_sources
    from infernux.engine.project_context import using_project_root

    source = _write_kernel(tmp_path)
    cache = tmp_path / "Library/Artifacts/Compute"
    namespace = {"__name__": "Scripts.Jelly", "__file__": str(source)}
    exec(compile(embed_compute_sources(source.read_text(encoding="utf-8")), str(source), "exec"), namespace)
    with using_project_root(tmp_path):
        frontend.compile_kernel(namespace["step"].function, (compute_aot._gpu_buffer_descriptor((1,), np.int32),))
    _write_artifact(cache / "other.inxgpu", "Scripts.EditorOnly.step")
    data = tmp_path / "build/Data"

    result = stage_compute_artifacts(tmp_path, (source,), data)

    destination = data / "Library/Artifacts/Compute"
    assert result.artifact_count == 4
    assert result.kernel_count == 1
    assert len(list(destination.glob("*.inxgpu"))) == 4
    assert not (destination / "other.inxgpu").exists()
    manifest = json.loads((destination / "AotManifest.json").read_text(encoding="utf-8"))
    assert [record["function"] for record in manifest["artifacts"]] == [
        "Scripts.Jelly.step",
        "infernux.compute._mesh_attribute_kernel",
        "infernux.compute._transform_anchor_points",
        "infernux.compute._transform_anchor_vectors",
    ]


def test_engine_compute_aot_prewarms_all_player_specializations(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        frontend,
        "compile_kernel",
        lambda function, params: calls.append((function.__qualname__, params)),
    )

    ensure_engine_compute_artifacts(tmp_path)

    assert [name for name, _params in calls] == [
        "_transform_anchor_points",
        "_transform_anchor_vectors",
        "_mesh_attribute_kernel",
    ]
    assert calls[0][1][1].dtype == "vector3"
    assert calls[-1][1][1].shape == (1, 25)


def test_stage_compute_artifacts_keeps_unwarmed_kernels_for_runtime_preparation(tmp_path):
    source = _write_kernel(tmp_path)
    cache = tmp_path / "Library/Artifacts/Compute"
    _write_artifact(cache / "other.inxgpu", "Scripts.Other.step")

    data = tmp_path / "build/Data"
    result = stage_compute_artifacts(tmp_path, (source,), data)
    assert result.kernel_count == 1 and result.artifact_count == 3
    rows = json.loads((data / "Library/Artifacts/Compute/AotManifest.json").read_text(encoding='utf8'))['artifacts']
    assert not any(row['function'].startswith('Scripts.') for row in rows)


def _runtime_missing_kernel(domain):
    i = compute.index(domain)
    domain[i] = 1


def test_cold_player_prepares_a_program_with_the_resident_compiler(tmp_path):
    from infernux.engine.project_context import using_project_root

    domain = compute_aot._gpu_buffer_descriptor((1,), np.int32)
    with using_project_root(tmp_path):
        artifact = frontend.compile_kernel(_runtime_missing_kernel, (domain,))
    assert artifact.target == 'spirv' and artifact.tasks


def test_compute_aot_is_sealed_inside_content_package(tmp_path):
    project = tmp_path / "project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    final = tmp_path / "player"
    compute_root = final / "TestGame_Data/Library/Artifacts/Compute"
    compute_root.mkdir(parents=True)
    # Packaging copies opaque sealed content; compilation is exercised above.
    (compute_root / "kernel.inxgpu").write_bytes(b"sealed kernel payload")
    builder = GameBuilder(str(project), str(tmp_path / "output"), game_name="TestGame")

    builder._pack_content_archive(str(final))

    manifest = read_manifest(final / "TestGame_Data/Content.inxpkg")
    assert {entry["path"] for entry in manifest["files"]} == {
        "Library/Artifacts/Compute/kernel.inxgpu",
    }
