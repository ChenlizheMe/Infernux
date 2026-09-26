import json
from pathlib import Path
import struct

import pytest
import numpy as np

from Infernux import compute
from Infernux._compiler.taichi import frontend
from Infernux.engine.build.compute_aot import (
    ComputeAotBuildError,
    declared_kernel_names,
    stage_compute_artifacts,
)
from Infernux.engine.game_builder import GameBuilder
from Infernux.engine.player_package_native import read_manifest


def _write_artifact(path: Path, function: str) -> None:
    header = json.dumps(
        {
            "task_sizes": [4],
            "task_metadata": [{"entry_point": "main"}],
            "domain_parameter": 0,
            "argument_layout": {"size": 0, "parameters": []},
            "required_capabilities": {"spirv_version": 0x10500},
            "diagnostic_locations": [{"function": function}],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"INXGPU\x01" + struct.pack("<I", len(header)) + header + b"SPV0")


def _write_kernel(project: Path) -> Path:
    source = project / "Assets/Scripts/Jelly.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import Infernux as inx\n"
        "from Infernux import compute as gpu\n"
        "@inx.compute.function\n"
        "def helper(value):\n"
        "    return value\n"
        "@gpu.kernel\n"
        "def step(domain):\n"
        "    i = gpu.index(domain)\n"
        "    domain[i] = helper(domain[i])\n",
        encoding="utf-8",
    )
    return source


def test_declared_kernel_names_uses_runtime_module_identity(tmp_path):
    source = _write_kernel(tmp_path)

    assert declared_kernel_names((source,), tmp_path) == ("Scripts.Jelly.step",)


def test_declared_kernel_names_accepts_explicit_static_class_kernel(tmp_path):
    source = tmp_path / "Assets/Scripts/Jelly.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import Infernux as inx\n"
        "class Jelly:\n"
        "    @staticmethod\n"
        "    @inx.compute.kernel\n"
        "    def step(domain):\n"
        "        i = inx.compute.index(domain)\n"
        "        domain[i] = 1\n",
        encoding="utf-8",
    )

    assert declared_kernel_names((source,), tmp_path) == ("Scripts.Jelly.Jelly.step",)


def test_declared_kernel_names_accepts_declared_instance_receiver_field(tmp_path):
    source = tmp_path / "Assets/Scripts/Jelly.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import infernux as inx\n"
        "class Jelly:\n"
        "    scale: float\n"
        "    @inx.compute.kernel\n"
        "    def step(self, domain):\n"
        "        i = inx.compute.index(domain)\n"
        "        domain[i] = self.scale\n",
        encoding="utf-8",
    )

    assert declared_kernel_names((source,), tmp_path, target="Android/AOT") == (
        "Scripts.Jelly.Jelly.step",
    )


def test_declared_kernel_names_reports_implicit_class_receiver_with_location(tmp_path):
    source = tmp_path / "Assets/Scripts/Jelly.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import Infernux as inx\n"
        "class Jelly:\n"
        "    @inx.compute.kernel\n"
        "    def step(self, domain):\n"
        "        i = inx.compute.index(domain)\n"
        "        domain[i] = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ComputeAotBuildError) as error:
        declared_kernel_names((source,), tmp_path, target="Android/AOT")
    message = str(error.value)
    assert "Scripts.Jelly.Jelly.step" in message
    assert f"{source}:4:5" in message
    assert "Android/AOT" in message
    assert "implicit instance receiver 'self'" in message
    assert "@staticmethod" in message


def test_declared_kernel_names_reports_positional_only_class_receiver(tmp_path):
    source = tmp_path / "Assets/Scripts/Jelly.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import Infernux as inx\n"
        "class Jelly:\n"
        "    @inx.compute.kernel\n"
        "    def step(self, /, domain):\n"
        "        i = inx.compute.index(domain)\n"
        "        domain[i] = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ComputeAotBuildError, match="implicit instance receiver 'self'"):
        declared_kernel_names((source,), tmp_path, target="Android/AOT")


def test_declared_kernel_names_reports_unbound_class_field_with_attribute_location(tmp_path):
    source = tmp_path / "Assets/Scripts/Jelly.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import Infernux as inx\n"
        "class Jelly:\n"
        "    @inx.compute.kernel\n"
        "    def step(domain):\n"
        "        i = inx.compute.index(domain)\n"
        "        domain[i] = self.scale\n",
        encoding="utf-8",
    )

    with pytest.raises(ComputeAotBuildError) as error:
        declared_kernel_names((source,), tmp_path, target="Web/Player")
    message = str(error.value)
    assert f"{source}:6:21" in message
    assert "unbound receiver field 'self.scale'" in message
    assert "target 'Web/Player'" in message
    assert "pass the required scalar or inx.buffer explicitly" in message


@pytest.mark.parametrize(
    "target",
    ("Editor/Desktop", "Player/Windows", "Player/Linux", "Android/AOT"),
)
def test_kernel_contract_fixture_reports_the_same_identity_for_native_targets(target):
    fixture_root = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "compute_kernel_contract"
    source = fixture_root / "Assets/Scripts/InvalidInstance.py"
    with pytest.raises(ComputeAotBuildError) as error:
        declared_kernel_names((source,), fixture_root, target=target)
    message = str(error.value)
    assert "Scripts.InvalidInstance.JellyKernel.step" in message
    assert f"{source}:8:5" in message
    assert f"target '{target}'" in message
    assert "implicit instance receiver 'self'" in message


@pytest.mark.parametrize(
    "target",
    ("Editor/Desktop", "Player/Windows", "Player/Linux", "Android/AOT"),
)
def test_kernel_contract_fixture_accepts_the_same_static_kernel_for_native_targets(target):
    fixture_root = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "compute_kernel_contract"
    source = fixture_root / "Assets/Scripts/ValidStatic.py"
    assert declared_kernel_names((source,), fixture_root, target=target) == (
        "Scripts.ValidStatic.JellyKernel.step",
    )


def test_stage_compute_artifacts_seals_selected_and_engine_kernels(tmp_path):
    source = _write_kernel(tmp_path)
    cache = tmp_path / "Library/Artifacts/Compute"
    _write_artifact(cache / "selected.inxgpu", "Scripts.Jelly.step")
    _write_artifact(cache / "engine.inxgpu", "Infernux.compute._transform_anchor_points")
    _write_artifact(cache / "other.inxgpu", "Scripts.EditorOnly.step")
    data = tmp_path / "build/Data"

    result = stage_compute_artifacts(tmp_path, (source,), data)

    destination = data / "Library/Artifacts/Compute"
    assert result.artifact_count == 2
    assert result.kernel_count == 1
    assert sorted(path.name for path in destination.glob("*.inxgpu")) == [
        "engine.inxgpu",
        "selected.inxgpu",
    ]
    assert (destination / "AotOnly").is_file()


def test_stage_compute_artifacts_rejects_incomplete_selected_closure(tmp_path):
    source = _write_kernel(tmp_path)
    cache = tmp_path / "Library/Artifacts/Compute"
    _write_artifact(cache / "other.inxgpu", "Scripts.Other.step")

    with pytest.raises(ComputeAotBuildError) as error:
        stage_compute_artifacts(tmp_path, (source,), tmp_path / "build/Data")

    assert error.value.missing == ("Scripts.Jelly.step",)
    assert not (tmp_path / "build/Data/Library/Artifacts/Compute").exists()


def test_stage_compute_artifacts_uses_platform_target_for_source_diagnostics(tmp_path):
    source = tmp_path / "Assets/Scripts/Jelly.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import Infernux as inx\n"
        "class Jelly:\n"
        "    @inx.compute.kernel\n"
        "    def step(self, domain):\n"
        "        i = inx.compute.index(domain)\n"
        "        domain[i] = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ComputeAotBuildError, match="target 'Android/AOT'"):
        stage_compute_artifacts(
            tmp_path,
            (source,),
            tmp_path / "build/Data",
            target="Android/AOT",
        )


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
    monkeypatch.setattr(frontend, "_load_artifact", lambda _key: None)
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
    _write_artifact(compute_root / "kernel.inxgpu", "Scripts.Jelly.step")
    (compute_root / "AotOnly").write_text("sealed\n", encoding="utf-8")
    builder = GameBuilder(str(project), str(tmp_path / "output"), game_name="TestGame")

    builder._pack_content_archive(str(final))

    manifest = read_manifest(final / "TestGame_Data/Content.inxpkg")
    assert {entry["path"] for entry in manifest["files"]} == {
        "Library/Artifacts/Compute/AotOnly",
        "Library/Artifacts/Compute/kernel.inxgpu",
    }
