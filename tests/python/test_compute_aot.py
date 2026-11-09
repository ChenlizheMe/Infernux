import json
from pathlib import Path

import pytest
import numpy as np

from infernux import compute
from infernux._compiler.taichi import frontend
from infernux.engine.build import compute_aot
from infernux.engine.build.compute_aot import (
    ComputeAotBuildError,
    declared_kernel_names,
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
    assert (destination / "AotOnly").is_file()
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


def test_stage_compute_artifacts_rejects_incomplete_selected_closure(tmp_path, monkeypatch):
    source = _write_kernel(tmp_path)
    cache = tmp_path / "Library/Artifacts/Compute"
    _write_artifact(cache / "other.inxgpu", "Scripts.Other.step")

    with pytest.raises(ComputeAotBuildError) as error:
        stage_compute_artifacts(tmp_path, (source,), tmp_path / "build/Data")

    assert error.value.missing == ("Scripts.Jelly.step",)
    assert not (tmp_path / "build/Data/Library/Artifacts/Compute").exists()


def _runtime_missing_kernel(domain):
    i = compute.index(domain)
    domain[i] = 1


def test_aot_only_player_never_falls_back_to_private_compiler(tmp_path, monkeypatch):
    (tmp_path / "AotOnly").write_text("sealed\n", encoding="utf-8")
    domain = object.__new__(compute.Buffer)
    domain._dtype = compute._buffer_dtype(np.int32)
    domain._shape = (1,)
    domain._device = "gpu"
    monkeypatch.setattr(frontend, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(
        frontend,
        "_load_vendor",
        lambda: pytest.fail("AOT-only Player must not initialize the compiler"),
    )

    with pytest.raises(RuntimeError, match="Player GPU AOT artifact is missing"):
        frontend.compile_kernel(_runtime_missing_kernel, (domain,))


def test_compute_aot_is_sealed_inside_content_package(tmp_path):
    project = tmp_path / "project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    final = tmp_path / "player"
    compute_root = final / "TestGame_Data/Library/Artifacts/Compute"
    compute_root.mkdir(parents=True)
    # Packaging copies opaque sealed content; compilation is exercised above.
    (compute_root / "kernel.inxgpu").write_bytes(b"sealed kernel payload")
    (compute_root / "AotOnly").write_text("sealed\n", encoding="utf-8")
    builder = GameBuilder(str(project), str(tmp_path / "output"), game_name="TestGame")

    builder._pack_content_archive(str(final))

    manifest = read_manifest(final / "TestGame_Data/Content.inxpkg")
    assert {entry["path"] for entry in manifest["files"]} == {
        "Library/Artifacts/Compute/AotOnly",
        "Library/Artifacts/Compute/kernel.inxgpu",
    }
