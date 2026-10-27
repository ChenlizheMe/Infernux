from __future__ import annotations

import dataclasses
import gc
import importlib.util
import os
import sys
import types
import weakref
from pathlib import Path

import pytest

from infernux.engine.candidate_import import (
    CandidateImportError,
    CandidateImportTransaction,
)
from infernux.engine.project_context import (
    get_project_root,
    get_script_import_paths,
    get_script_module_name,
    set_project_root,
)


@pytest.fixture
def candidate_project(tmp_path):
    previous = get_project_root()
    project = tmp_path / "CandidateImportProject"
    assets = project / "Assets"
    assets.mkdir(parents=True)
    set_project_root(str(project))
    try:
        yield assets
    finally:
        set_project_root(previous)


def _broker(assets: Path, name: str, source: str) -> CandidateImportTransaction:
    path = assets / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    broker = CandidateImportTransaction()
    broker.register(name, str(path), source=source)
    return broker


def test_candidate_policy_allows_enum_member_names_that_shadow_imports():
    from infernux.engine.script_candidate_policy import analyze_script_candidate

    source = '''from enum import Enum
from infernux.components.builtin import Rigidbody
class TargetKind(str, Enum):
    Rigidbody = "Rigidbody"
Rigidbody = None
del Rigidbody
'''
    assert not analyze_script_candidate(source).is_rejected
    for statement in ("Rigidbody.enabled = False", "Rigidbody.settings[0] = False", "Rigidbody += [1]"):
        report = analyze_script_candidate("from shared import Rigidbody\n" + statement)
        assert report.is_rejected and report.blocked[0].operation.startswith("Rigidbody")


def test_preload_library_uses_python_identity_and_lazy_children(candidate_project, monkeypatch):
    from infernux.engine.project_context import register_preload_python_library, release_preload_python_libraries

    library = candidate_project.parent / "Packages/vendor/tween/runtime/owned_tween"
    library.mkdir(parents=True)
    (library / "__init__.py").write_text("VALUE = object()\n", encoding="utf-8")
    (library / "lazy.py").write_text("from . import VALUE\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(library.parent))
    module = importlib.import_module("owned_tween")
    register_preload_python_library("fixture-tween", str(library))
    candidate = _broker(candidate_project, "tween_consumer",
                        "from owned_tween import VALUE\nfrom owned_tween.lazy import VALUE as CHILD\n")
    try:
        result = candidate.load("tween_consumer")
        assert result.VALUE is module.VALUE and result.CHILD is module.VALUE
        assert candidate.loaded_module_names == ("tween_consumer",)
        candidate.rollback()
        assert sys.modules["owned_tween"] is module
        release_preload_python_libraries("fixture-tween")
        unowned = _broker(candidate_project, "unowned_consumer", "import owned_tween\n")
        with pytest.raises(CandidateImportError, match="path/name mismatch"):
            unowned.load("unowned_consumer")
        unowned.rollback()
    finally:
        candidate.rollback()
        release_preload_python_libraries("fixture-tween")
        sys.modules.pop("owned_tween.lazy", None)
        sys.modules.pop("owned_tween", None)


def test_serializable_candidate_is_private_until_module_commit(candidate_project):
    from infernux.components.serializable_object import get_serializable_class

    broker = _broker(
        candidate_project,
        "candidate_data",
        (
            "from infernux.components import SerializableObject, serialized_field\n"
            "class Rules(SerializableObject):\n"
            "    score: int = serialized_field(default=17)\n"
            "VALUE = Rules._deserialize(Rules()._serialize())\n"
        ),
    )
    identity = "candidate_data:Rules"
    try:
        module = broker.load("candidate_data")
        assert type(module.VALUE) is module.Rules
        assert module.VALUE.score == 17
        assert get_serializable_class(identity) is None
        broker.commit()
        assert get_serializable_class(identity) is sys.modules["candidate_data"].Rules
    finally:
        broker.rollback()
    assert get_serializable_class(identity) is None


def test_serializable_failed_candidate_keeps_live_type(candidate_project):
    from infernux.components.serializable_object import get_serializable_class

    source = (
        "from infernux.components import SerializableObject, serialized_field\n"
        "class Rules(SerializableObject):\n"
        "    score: int = serialized_field(default=17)\n"
    )
    live = _broker(candidate_project, "failed_data", source)
    candidate = None
    try:
        live_module = live.load("failed_data")
        live.commit()
        candidate = _broker(
            candidate_project, "failed_data", source.replace("17", "29") + "raise RuntimeError('invalid candidate')\n"
        )
        with pytest.raises(RuntimeError, match="invalid candidate"):
            candidate.load("failed_data")
        assert get_serializable_class("failed_data:Rules") is live_module.Rules
        assert sys.modules["failed_data"] is live_module
    finally:
        if candidate is not None:
            candidate.rollback()
        live.rollback()


def test_serializable_replacement_retires_removed_types_and_can_rollback(candidate_project):
    from infernux.components.serializable_object import get_serializable_class

    source = (
        "from infernux.components import SerializableObject\n"
        "class Rules(SerializableObject):\n"
        "    score: int = 17\n"
        "class Removed(SerializableObject):\n"
        "    pass\n"
    )
    live = _broker(candidate_project, "replaced_data", source)
    candidate = None
    try:
        live_module = live.load("replaced_data")
        live.commit()
        candidate = _broker(candidate_project, "replaced_data", source.replace("17", "29").split("class Removed")[0])
        module = candidate.load("replaced_data")
        assert get_serializable_class("replaced_data:Rules") is live_module.Rules
        assert get_serializable_class("replaced_data:Removed") is live_module.Removed
        candidate.commit()
        assert get_serializable_class("replaced_data:Rules") is module.Rules
        assert get_serializable_class("replaced_data:Removed") is None
        candidate.rollback()
        assert get_serializable_class("replaced_data:Rules") is live_module.Rules
        assert get_serializable_class("replaced_data:Removed") is live_module.Removed
    finally:
        if candidate is not None:
            candidate.rollback()
        live.rollback()


def test_serializable_helper_defaults_use_private_candidate_types(candidate_project):
    from infernux.components.serializable_object import get_serializable_class

    source = (
        "from infernux.components import SerializableObject, serialized_field\n"
        "from data_helper import Stats\n"
        "class Rules(SerializableObject):\n"
        "    stats: Stats = serialized_field(default=Stats())\n"
        "VALUE = Rules._deserialize(Rules()._serialize())\n"
    )
    helper = candidate_project / "data_helper.py"
    helper.write_text(
        "from infernux.components import SerializableObject\nclass Stats(SerializableObject):\n    hp: int = 100\n",
        encoding="utf-8",
    )
    broker = _broker(candidate_project, "data_consumer", source)
    broker.register("data_helper", str(helper))
    try:
        module = broker.load("data_consumer")
        stats = broker.module_for("data_helper").Stats
        assert type(module.VALUE.stats) is stats
        assert get_serializable_class("data_helper:Stats") is None
        assert get_serializable_class("data_consumer:Rules") is None
        broker.commit()
        assert get_serializable_class("data_helper:Stats") is stats
        assert get_serializable_class("data_consumer:Rules") is module.Rules
    finally:
        broker.rollback()


def test_serializable_candidate_cannot_claim_another_modules_identity(candidate_project):
    from infernux.components.serializable_object import get_serializable_class

    source = (
        "from infernux.components import SerializableObject\n"
        "class Rules(SerializableObject):\n"
        "    __serialized_type_id__ = 'tests:owned-rules'\n"
    )
    owner = _broker(candidate_project, "data_owner", source)
    intruder = _broker(candidate_project, "data_intruder", source)
    try:
        live = owner.load("data_owner")
        owner.commit()
        intruder.load("data_intruder")
        with pytest.raises(ValueError, match="owned by module 'data_owner'"):
            intruder.commit()
        assert get_serializable_class("tests:owned-rules") is live.Rules
        assert "data_intruder" not in sys.modules
    finally:
        intruder.rollback()
        owner.rollback()


def test_serializable_rollback_preserves_unrelated_publication(candidate_project):
    from infernux.components.serializable_object import get_serializable_class

    source = "from infernux.components import SerializableObject\nclass Rules(SerializableObject): pass\n"
    first = _broker(candidate_project, "first_data_owner", source)
    second = _broker(candidate_project, "second_data_owner", source)
    try:
        first.load("first_data_owner")
        other = second.load("second_data_owner")
        first.commit()
        second.commit()
        first.rollback()
        assert get_serializable_class("first_data_owner:Rules") is None
        assert get_serializable_class("second_data_owner:Rules") is other.Rules
        assert sys.modules["second_data_owner"] is other
    finally:
        second.rollback()
        first.rollback()


def test_candidate_scc_uses_private_table_until_commit(candidate_project):
    assets = candidate_project
    (assets / "scc_a.py").write_text(
        "VALUE_A = 'a'\nimport scc_b\nVALUE_FROM_B = scc_b.VALUE_B\n",
        encoding="utf-8",
    )
    (assets / "scc_b.py").write_text(
        "VALUE_B = 'b'\nimport scc_a\nVALUE_FROM_A = scc_a.VALUE_A\n",
        encoding="utf-8",
    )
    broker = CandidateImportTransaction()
    broker.register("scc_a", str(assets / "scc_a.py"))
    broker.register("scc_b", str(assets / "scc_b.py"))

    module = broker.load("scc_a")

    assert module.VALUE_FROM_B == "b"
    assert broker.module_for("scc_b").VALUE_FROM_A == "a"
    assert "scc_a" not in sys.modules
    assert "scc_b" not in sys.modules

    broker.commit()
    assert sys.modules["scc_a"] is module
    assert sys.modules["scc_b"] is broker.module_for("scc_b")
    broker.rollback()
    assert "scc_a" not in sys.modules
    assert "scc_b" not in sys.modules


def test_candidate_child_does_not_mutate_live_parent_package_before_commit(
    candidate_project,
):
    package = candidate_project / "live_parent_pkg"
    package.mkdir()
    package_init = package / "__init__.py"
    child_path = package / "child.py"
    package_init.write_text("MARKER = 'lkg'\n", encoding="utf-8")
    child_path.write_text("VALUE = 'candidate'\n", encoding="utf-8")

    live_parent = type(sys)("live_parent_pkg")
    live_parent.__file__ = str(package_init)
    live_parent.__path__ = [str(package)]
    sentinel = object()
    live_parent.child = sentinel
    previous_parent = sys.modules.get("live_parent_pkg")
    previous_child = sys.modules.get("live_parent_pkg.child")
    sys.modules["live_parent_pkg"] = live_parent
    sys.modules.pop("live_parent_pkg.child", None)

    root_path = candidate_project / "live_parent_root.py"
    root_path.write_text(
        "from live_parent_pkg import child\nVALUE = child.VALUE\n",
        encoding="utf-8",
    )
    broker = CandidateImportTransaction()
    broker.register("live_parent_root", str(root_path))
    broker.register("live_parent_pkg.child", str(child_path))

    try:
        root = broker.load("live_parent_root")
        candidate_child = broker.module_for("live_parent_pkg.child")
        assert root.VALUE == "candidate"
        assert live_parent.child is sentinel
        assert "live_parent_pkg.child" not in sys.modules

        broker.commit()
        assert sys.modules["live_parent_pkg"] is live_parent
        assert sys.modules["live_parent_pkg.child"] is candidate_child
        assert live_parent.child is candidate_child

        broker.rollback()
        assert live_parent.child is sentinel
        assert "live_parent_pkg.child" not in sys.modules
    finally:
        if previous_parent is None:
            sys.modules.pop("live_parent_pkg", None)
        else:
            sys.modules["live_parent_pkg"] = previous_parent
        if previous_child is None:
            sys.modules.pop("live_parent_pkg.child", None)
        else:
            sys.modules["live_parent_pkg.child"] = previous_child


@pytest.mark.parametrize("auto_parallel", [False, True])
def test_jit_candidate_publication_keeps_live_revision_until_commit(candidate_project, auto_parallel):
    source = (
        "from infernux import jit\n"
        "FACTOR = 2\n"
        "def helper(value): return value * FACTOR\n"
        f"@jit.compile(cache=True, auto_parallel={auto_parallel})\n"
        "def scale(value): return helper(value)\n"
    )
    name = "jit_revision_candidate"
    live = _broker(candidate_project, name, source)
    candidate = None
    try:
        original = live.load(name)
        live.commit()
        assert original.scale(3) == 6
        candidate = _broker(candidate_project, name, source.replace("FACTOR = 2", "FACTOR = 5"))
        replacement = candidate.load(name)
        assert replacement.scale(3) == 15
        assert sys.modules[name] is original
        assert original.scale(3) == 6
        candidate.commit()
        assert sys.modules[name].scale(3) == 15
        candidate.rollback()
        assert sys.modules[name] is original
        assert original.scale(3) == 6
    finally:
        if candidate is not None:
            candidate.rollback()
        live.rollback()


def test_dynamic_project_import_stays_transaction_private(candidate_project):
    broker = _broker(
        candidate_project,
        "dynamic_project_authoring",
        ("import importlib\ndependency = importlib.import_module('dynamic_helper')\nvalue = dependency.VALUE\n"),
    )
    helper = candidate_project / "dynamic_helper.py"
    helper.write_text("VALUE = 41\n", encoding="utf-8")
    broker.register("dynamic_helper", str(helper), source="VALUE = 42\n")
    try:
        candidate = broker.load("dynamic_project_authoring")
        assert candidate.value == 42
        assert "dynamic_helper" not in sys.modules
        assert candidate.dependency is broker.module_for("dynamic_helper")
    finally:
        broker.rollback()


def test_execution_failure_restores_existing_module_entry(candidate_project):
    sentinel = object()
    previous = sys.modules.get("failure_candidate", sentinel)
    sys.modules["failure_candidate"] = sentinel
    broker = _broker(
        candidate_project,
        "failure_candidate",
        "raise RuntimeError('candidate execution failed')\n",
    )

    with pytest.raises(RuntimeError, match="candidate execution failed"):
        broker.load("failure_candidate")
    broker.rollback()

    if previous is sentinel:
        sys.modules.pop("failure_candidate", None)
    else:
        sys.modules["failure_candidate"] = previous


def test_durable_body_reload_releases_previous_modules(candidate_project, monkeypatch):
    from infernux.components.script_loader import (
        ComponentBodyReloadRequest,
        stage_component_body_reload_batch,
    )

    name = "durable_reload_helper"
    path = candidate_project / f"{name}.py"
    path.write_text("VALUE = -1\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    revisions = []
    for value in range(12):
        transaction = stage_component_body_reload_batch((
            ComponentBodyReloadRequest(
                str(path),
                source=f"VALUE = {value}\nPAYLOAD = bytearray(65536)\ndef read(): return VALUE\n",
            ),
        ))
        transaction.commit()
        transaction.finalize()
        assert transaction.finalized
        revisions.append(weakref.ref(sys.modules[name]))
        del transaction

    gc.collect()
    assert sys.modules[name].read() == 11
    assert all(reference() is None for reference in revisions[:-1])
    assert revisions[-1]() is sys.modules[name]


@pytest.mark.parametrize("kind", ["package", "namespace", "plugin"])
def test_finalized_import_hooks_follow_published_relative_and_dynamic_dependencies(
    candidate_project, monkeypatch, kind,
):
    if kind == "plugin":
        control = candidate_project.parent / "Packages" / "durable" / "tool"
        control.mkdir(parents=True)
        (control / "inx_package.json").write_text("{}\n", encoding="utf-8")
        package = control / "runtime"
    else:
        package = candidate_project / "durable_imports"
    package.mkdir(parents=True)
    entry = package / "entry.py"
    helper = package / "helper.py"
    entry.write_text(
        "import importlib\nimport dataclasses\n"
        "def read():\n"
        "    from . import helper\n"
        "    from .helper import VALUE\n"
        "    return helper.VALUE, VALUE, importlib.import_module('.helper', __package__).VALUE\n"
        "def record():\n"
        "    @dataclasses.dataclass\n"
        "    class Record:\n"
        "        value: int = 7\n"
        "    return Record()\n",
        encoding="utf-8",
    )
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    entry_name = get_script_module_name(str(entry))
    helper_name = get_script_module_name(str(helper))
    parent_name = entry_name.rpartition(".")[0]
    for name in (parent_name, entry_name, helper_name):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    initial_parent = sys.modules[parent_name]
    initial_parent.__path__ = [str(package)]
    initial_parent.__file__ = None
    broker = CandidateImportTransaction()
    if kind == "package":
        (package / "__init__.py").write_text("MARKER = 'package'\n", encoding="utf-8")
        broker.register(parent_name, str(package / "__init__.py"))
    broker.register(entry_name, str(entry))
    broker.register(helper_name, str(helper))
    broker.load(parent_name)
    module = broker.load(entry_name)
    original_helper = broker.load(helper_name)
    previous_helper = weakref.ref(original_helper)
    del original_helper
    broker.commit()
    broker.finalize()
    assert module.read() == (1, 1, 1)
    assert module.record().value == 7
    helper.unlink()
    assert module.read() == (1, 1, 1)
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    if kind == "plugin":
        manifest = control / "inx_package.json"
        manifest.unlink()
        assert module.read() == (1, 1, 1)
        manifest.write_text("{}\n", encoding="utf-8")

    replacement = CandidateImportTransaction()
    replacement.register(helper_name, str(helper), source="VALUE = 2\n")
    replacement.load(helper_name)
    replacement.commit()
    replacement.finalize()
    gc.collect()
    # The old entry's captured builtins/importlib proxy still work, without
    # pinning either the old helper or a private overlay of its live parent.
    assert module.read() == (2, 2, 2)
    assert module.record().value == 7
    assert previous_helper() is None


def test_finalization_preserves_published_data_and_effect_registries(candidate_project, monkeypatch):
    from infernux.components.serializable_object import (
        _restore_serializable_types,
        get_serializable_class,
    )
    from infernux.renderstack.render_effect_compiler import (
        _retire_source_effect_features,
        _snapshot_effect_features,
    )

    name = "durable_registry_owner"
    identity = name + ":Rules"
    effect_id = "tests.candidate.durable-registry"

    def effect_type():
        # Inspect publication itself: the public getter independently scans
        # authored files and can replace declarations under a discovery name.
        return _snapshot_effect_features((effect_id,), ())[effect_id].effect_class

    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    source = (
        "from infernux.components import SerializableObject\n"
        "from infernux.renderstack import render_effect_feature\n"
        "class Rules(SerializableObject):\n    value: int = 1\n"
        f"@render_effect_feature('{effect_id}')\n"
        "class Effect:\n    value = 1\n"
    )
    live = _broker(candidate_project, name, source)
    try:
        original = live.load(name)
        live.commit()
        live.finalize()
        assert get_serializable_class(identity) is original.Rules
        assert effect_type() is original.Effect

        candidate = _broker(candidate_project, name, source.replace("= 1", "= 2"))
        changed = candidate.load(name)
        candidate.commit()
        assert get_serializable_class(identity) is changed.Rules
        assert effect_type() is changed.Effect
        candidate.rollback()
        assert sys.modules[name] is original
        assert get_serializable_class(identity) is original.Rules
        assert effect_type() is original.Effect

        durable = _broker(candidate_project, name, source.replace("= 1", "= 3"))
        current = durable.load(name)
        durable.commit()
        durable.finalize()
        assert get_serializable_class(identity) is current.Rules
        assert effect_type() is current.Effect
        with pytest.raises(CandidateImportError, match="durable"):
            durable.rollback()
        assert get_serializable_class(identity) is current.Rules
        assert effect_type() is current.Effect
    finally:
        _restore_serializable_types({identity: None})
        _retire_source_effect_features(str(candidate_project / f"{name}.py"))


def test_candidate_finalization_requires_commit_and_closes_rollback(candidate_project):
    broker = _broker(candidate_project, "durable_states", "VALUE = 1\n")
    with pytest.raises(CandidateImportError, match="must commit"):
        broker.finalize()
    broker.load("durable_states")
    broker.commit()
    broker.finalize()
    broker.finalize()
    broker.commit()
    try:
        with pytest.raises(CandidateImportError, match="durable"):
            broker.rollback()
        with pytest.raises(CandidateImportError, match="cannot register"):
            broker.register("another", str(candidate_project / "durable_states.py"))
        with pytest.raises(CandidateImportError, match="no published module"):
            broker.load("not_published_project_module")
    finally:
        sys.modules.pop("durable_states", None)

    rejected = _broker(candidate_project, "rolled_back_states", "VALUE = 1\n")
    rejected.load("rolled_back_states")
    rejected.rollback()
    with pytest.raises(CandidateImportError, match="rolled back"):
        rejected.finalize()


def test_rejected_finalization_preserves_the_rollback_window(candidate_project):
    from infernux.components._component_lifecycle import RuntimeExecutionScheduler

    broker = _broker(candidate_project, "finalize_safe_point", "VALUE = 1\n")
    broker.load("finalize_safe_point")
    broker.commit()
    scheduler = RuntimeExecutionScheduler()
    frame = scheduler.begin_frame()
    try:
        with pytest.raises(RuntimeError, match="safe point"):
            broker.finalize()
        assert not broker.finalized
    finally:
        frame.close()
        broker.rollback()
    assert "finalize_safe_point" not in sys.modules
