"""Current-source AOT selection and actual source-less Vulkan execution."""
import json
import linecache
import sys
import types

import numpy as np
import pytest

from infernux import compute
from infernux.application import Application
from infernux._compiler.source_metadata import embed_compute_sources
from infernux._compiler.taichi import frontend
from infernux.engine.build.compute_aot import (
    ComputeAotBuildError, _gpu_buffer_descriptor, stage_compute_artifacts,
)
from infernux.engine.game_builder import GameBuilder
from infernux.engine.project_context import using_project_root


def source_text(value, kind="body"):
    prefix = "import infernux as inx\n"
    expression = str(value)
    if kind == "constant":
        prefix += f"GAIN = {value}\n"
        expression = "GAIN"
    elif kind == "helper":
        prefix += f"@inx.compute.function\ndef gain(value):\n    return value + {value}\n"
        expression = "gain(0)"
    return (prefix + "@inx.compute.kernel\ndef solve(values):\n"
            "    i = inx.compute.index(values)\n" + f"    values[i] += {expression}\n")


def load_source(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    linecache.clearcache()
    module = types.ModuleType("Scripts.Work")
    module.__file__ = str(path)
    exec(compile(embed_compute_sources(source), str(path), "exec"), module.__dict__)
    return module


def warm(root, declaration, *, receiver=False):
    params = (_gpu_buffer_descriptor((4,), np.int32),)
    bindings = None
    if receiver:
        bindings, params = declaration._effective_params(params)
    with using_project_root(root):
        return frontend.compile_kernel(declaration.function, params, receiver_fields=bindings)


def project_records(data):
    document = json.loads((data / "Library/Artifacts/Compute/AotManifest.json").read_text(encoding="utf-8"))
    return [row for row in document["artifacts"] if row["function"].startswith("Scripts.")]


@pytest.mark.parametrize("kind", ["body", "helper", "constant"])
def test_aot_selects_only_current_warmed_revision(tmp_path, kind):
    path = tmp_path / "Assets/Scripts/Work.py"
    old = warm(tmp_path, load_source(path, source_text(1, kind)).solve)
    latest = warm(tmp_path, load_source(path, source_text(2, kind)).solve)
    data = tmp_path / "build/Data"
    stage_compute_artifacts(tmp_path, (path,), data)
    records = project_records(data)
    assert len(records) == 1
    artifact = frontend._decode_artifact(data / "Library/Artifacts/Compute" / records[0]["artifact"])
    assert artifact.spirv_tasks == latest.spirv_tasks != old.spirv_tasks


@pytest.mark.parametrize("kind", ["body", "helper", "constant"])
def test_unwarmed_source_edit_fails_before_replacing_previous_build(tmp_path, monkeypatch, kind):
    path = tmp_path / "Assets/Scripts/Work.py"
    old_module = load_source(path, source_text(1, kind))
    warm(tmp_path, old_module.solve)
    # The editor has a valid but stale published module. Build must read disk.
    monkeypatch.setitem(sys.modules, "Scripts.Work", old_module)
    data = tmp_path / "build/Data"
    stage_compute_artifacts(tmp_path, (path,), data)
    before = {p.name: p.read_bytes() for p in (data / "Library/Artifacts/Compute").iterdir()}
    path.write_text(source_text(2, kind), encoding="utf-8")
    with pytest.raises(ComputeAotBuildError) as error:
        stage_compute_artifacts(tmp_path, (path,), data)
    assert error.value.missing == ("Scripts.Work.solve",)
    assert {p.name: p.read_bytes() for p in (data / "Library/Artifacts/Compute").iterdir()} == before
    assert sys.modules["Scripts.Work"] is old_module


@pytest.mark.parametrize("import_line", [
    "from infernux import compute as gpu", "import infernux.compute as gpu",
    "import infernux as owner\ngpu = owner.compute",
])
@pytest.mark.parametrize("receiver", [False, True])
def test_cooked_source_less_public_prepare_and_launch(tmp_path, engine, monkeypatch, import_line, receiver):
    # The last form keeps a supported explicit root decorator spelling.
    decorator = "owner.compute" if "gpu =" in import_line else "gpu"
    lines = f"import infernux as inx\n{import_line}\n"
    if receiver:
        lines += ("class Controller:\n    scale: int = 2\n"
                  f"    @{decorator}.kernel\n    def solve(self, values):\n"
                  "        i = inx.compute.index(values)\n        values[i] += self.scale\n")
    else:
        lines += (f"@{decorator}.kernel\ndef solve(values):\n"
                  "    i = inx.compute.index(values)\n    values[i] += 2\n")
    path = tmp_path / "Assets/Scripts/Work.py"
    module = load_source(path, lines)
    declaration = module.Controller().solve if receiver else module.solve
    warm(tmp_path, declaration, receiver=receiver)
    data = tmp_path / "build/Data"
    stage_compute_artifacts(tmp_path, (path,), data)

    builder = GameBuilder(str(tmp_path), str(tmp_path / "output"), game_name="AotTest")
    builder.include_jit_runtime = False
    cooked = builder._cook_compute_source(lines)
    namespace = {"__name__": "Scripts.Work"}
    exec(compile(cooked, "<source-less-player>", "exec"), namespace)
    path.unlink()
    linecache.clearcache()
    declaration = namespace["Controller"]().solve if receiver else namespace["solve"]
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(frontend, "_load_vendor", lambda: pytest.fail("Player must not load a compiler"))
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, "_native_compute_host", lambda: host)
    values = compute.buffer(shape=4, dtype=np.int32, device="gpu", data=[10, 20, 30, 40])
    try:
        with using_project_root(data):
            compute.prepare(declaration, (values,))
            compute.launch(declaration, (values,))
        readback = values.get_data()
        try:
            np.testing.assert_array_equal(readback.numpy(), [12, 22, 32, 42])
        finally:
            readback.close()
    finally:
        declaration._release_engine_resources()
        values.close()


def test_same_source_has_same_artifact_identity_across_clean_project_roots(tmp_path):
    records = []
    binaries = []
    for directory in ("author A", "author B"):
        root = tmp_path / directory
        path = root / "Assets/Scripts/Work.py"
        module = load_source(path, source_text(2, "helper"))
        artifact = warm(root, module.solve)
        data = root / "build/Data"
        stage_compute_artifacts(root, (path,), data)
        records.append(project_records(data))
        binaries.append(artifact.spirv_tasks)
    assert records[0] == records[1]
    assert binaries[0] == binaries[1]


def test_player_rejects_manifest_from_another_source_revision(tmp_path, monkeypatch):
    path = tmp_path / "Assets/Scripts/Work.py"
    warm(tmp_path, load_source(path, source_text(1)).solve)
    data = tmp_path / "build/Data"
    stage_compute_artifacts(tmp_path, (path,), data)
    current = load_source(path, source_text(2))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(frontend, "_load_vendor", lambda: pytest.fail("Player must not compile"))
    with using_project_root(data), pytest.raises(RuntimeError, match="missing or ambiguous"):
        frontend.compile_kernel(current.solve.function, (_gpu_buffer_descriptor((4,), np.int32),))


def test_aot_uses_the_frozen_cooked_source_instead_of_a_later_disk_edit(tmp_path):
    path = tmp_path / "Assets/Scripts/Work.py"
    cooked_source = source_text(1)
    compiled = warm(tmp_path, load_source(path, cooked_source).solve)
    path.write_text(source_text(2), encoding="utf-8")
    data = tmp_path / "build/Data"
    stage_compute_artifacts(tmp_path, (path,), data, source_snapshots={str(path): cooked_source})
    row, = project_records(data)
    assert frontend._decode_artifact(data / "Library/Artifacts/Compute" / row["artifact"]).spirv_tasks == compiled.spirv_tasks
    with pytest.raises(ComputeAotBuildError, match="snapshots disagree"):
        stage_compute_artifacts(tmp_path, (path,), data, source_snapshots={})


def test_new_signature_does_not_publish_or_require_historical_arities(tmp_path):
    path = tmp_path / "Assets/Scripts/Work.py"
    warm(tmp_path, load_source(path, source_text(1)).solve)
    current = source_text(2).replace("solve(values)", "solve(values, amount)").replace("+= 2", "+= amount")
    module = load_source(path, current)
    with using_project_root(tmp_path):
        frontend.compile_kernel(module.solve.function, (_gpu_buffer_descriptor((4,), np.int32), 2))
    data = tmp_path / "build/Data"
    stage_compute_artifacts(tmp_path, (path,), data)
    row, = project_records(data)
    assert json.loads(row["specialization"])[-1] == ["int32"]


def test_imported_helper_edit_is_read_from_current_private_closure(tmp_path, monkeypatch):
    from infernux.engine.candidate_import import CandidateImportTransaction

    path = tmp_path / "Assets/Scripts/Work.py"
    source = ("import infernux as inx\nfrom Scripts.Shared import gain\n"
              "@inx.compute.kernel\ndef solve(values):\n"
              "    i = inx.compute.index(values)\n    values[i] += gain(0)\n")
    helper = path.with_name("Shared.py")
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    original = None
    for value in (1, 2):
        helper_source = ("import infernux as inx\n@inx.compute.function\n"
                         f"def gain(value):\n    return value + {value}\n")
        helper.write_text(helper_source, encoding="utf-8")
        with using_project_root(tmp_path):
            broker = CandidateImportTransaction()
            try:
                broker.register("Scripts.Shared", str(helper), source=embed_compute_sources(helper_source))
                broker.register("Scripts.Work", str(path), source=embed_compute_sources(source))
                module = broker.load("Scripts.Work")
                compiled = warm(tmp_path, module.solve)
                if value == 1:
                    original = compiled
                    # Keep the editor's old helper published while disk advances.
                    monkeypatch.setitem(sys.modules, "Scripts.Shared", broker.module_for("Scripts.Shared"))
            finally:
                broker.rollback()
    data = tmp_path / "build/Data"
    stage_compute_artifacts(tmp_path, (path, helper), data)
    row, = project_records(data)
    selected = frontend._decode_artifact(data / "Library/Artifacts/Compute" / row["artifact"])
    assert selected.spirv_tasks == compiled.spirv_tasks != original.spirv_tasks


def test_nested_closure_free_kernel_preserves_qualified_source_in_player(tmp_path, monkeypatch):
    source = ("import infernux.compute as gpu\nimport infernux as inx\n"
              "def factory():\n    @gpu.kernel\n    def solve(values):\n"
              "        i = inx.compute.index(values)\n        values[i] += 2\n"
              "    return solve\n")
    path = tmp_path / "Assets/Scripts/Work.py"
    module = load_source(path, source)
    compiled = warm(tmp_path, module.factory())
    data = tmp_path / "build/Data"
    stage_compute_artifacts(tmp_path, (path,), data)
    row, = project_records(data)
    assert row["function"] == "Scripts.Work.factory.<locals>.solve"
    path.unlink()
    monkeypatch.setattr(frontend, "_load_vendor", lambda: pytest.fail("Player must not compile"))
    with using_project_root(data):
        loaded = frontend.compile_kernel(module.factory().function, (_gpu_buffer_descriptor((4,), np.int32),))
    assert loaded.spirv_tasks == compiled.spirv_tasks
