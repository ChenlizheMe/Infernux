"""Private candidate dataclasses retain CPython semantics without module publication."""
import dataclasses
import inspect
import pickle
import sys
import types

import pytest

from infernux.engine.candidate_import import CandidateImportTransaction
from test_candidate_import_transaction import candidate_project


IMPORTS = {
    'named': ('from dataclasses import dataclass, field, InitVar, KW_ONLY\nfrom typing import ClassVar\n',
              'dataclass', 'field', 'InitVar', 'KW_ONLY', 'ClassVar'),
    'aliases': ('from dataclasses import dataclass as dc, field as f, InitVar as IV, KW_ONLY as KO\nfrom typing import ClassVar as CV\n',
                'dc', 'f', 'IV', 'KO', 'CV'),
    'qualified': ('import dataclasses as dc\nimport typing as t\n',
                  'dc.dataclass', 'dc.field', 'dc.InitVar', 'dc.KW_ONLY', 't.ClassVar'),
}


def load_reference(name, path, source):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


@pytest.mark.parametrize('future', [False, True])
@pytest.mark.parametrize('style', IMPORTS)
@pytest.mark.parametrize('slots', [False, True])
@pytest.mark.parametrize('frozen', [False, True])
@pytest.mark.parametrize('live', [False, True])
def test_candidate_dataclass_matches_standard_module(candidate_project, future, style, slots, frozen, live):
    imports, decorator, field, initvar, kwonly, classvar = IMPORTS[style]
    name = 'candidate_dataclass_rules'
    path = candidate_project / (name + '.py')
    source = ('from __future__ import annotations\n' if future else '') + imports + f'''
@{decorator}(slots={slots}, frozen={frozen})
class Rules:
    shared: {classvar}[int] = 13
    seed: {initvar}[int] = 7
    value: int = 5
    _: {kwonly}
    extra: int = 3
    following: 'Rules | None' = None
    items: list[int] = {field}(default_factory=list, compare=False)
    def __post_init__(self, seed):
        object.__setattr__(self, 'value', self.value + seed)
'''
    path.write_text(source, encoding='utf-8')
    reference = load_reference(name, path, source)
    broker = CandidateImportTransaction()
    try:
        expected_fields = [(f.name, f.type, f.kw_only) for f in dataclasses.fields(reference.Rules)]
        expected_signature = str(inspect.signature(reference.Rules))
        expected = reference.Rules(seed=11)
        if not live:
            sys.modules.pop(name)
        else:
            # A stale live namespace must not decide the candidate's annotation kinds.
            reference.ClassVar = reference.CV = reference.InitVar = reference.IV = int
        broker.register(name, str(path), source=source)
        candidate = broker.load(name)
        cls = candidate.Rules
        assert cls.__module__ == name
        assert cls.__annotations__.keys() == reference.Rules.__annotations__.keys()
        for key, annotation in cls.__annotations__.items():
            expected_annotation = reference.Rules.__annotations__[key]
            if isinstance(annotation, dataclasses.InitVar):
                assert isinstance(expected_annotation, dataclasses.InitVar)
                assert annotation.type == expected_annotation.type
            else:
                assert annotation == expected_annotation
        assert [(f.name, f.type, f.kw_only) for f in dataclasses.fields(cls)] == expected_fields
        assert str(inspect.signature(cls)) == expected_signature
        actual = cls(seed=11)
        assert dataclasses.asdict(actual) == dataclasses.asdict(expected)
        assert repr(actual) == repr(expected)
        assert dataclasses.replace(actual, seed=1).value == dataclasses.replace(expected, seed=1).value
        assert actual.items is not cls().items
        assert ('__dict__' in dir(actual)) == (not slots)
        if frozen:
            assert hash(actual) == hash(expected)
            with pytest.raises(dataclasses.FrozenInstanceError):
                actual.value = 0
        assert sys.modules.get(name) is (reference if live else None)
        broker.commit()
        assert sys.modules[name] is candidate
        assert candidate.Rules.__module__ == name
        restored = pickle.loads(pickle.dumps(actual))
        assert type(restored) is cls and restored == actual
    finally:
        broker.rollback()
        assert sys.modules.get(name) is (reference if live else None)
        sys.modules.pop(name, None)


@pytest.mark.parametrize('failure', [False, True])
def test_private_dataclass_keeps_strings_unevaluated_and_live_module_unchanged(candidate_project, failure):
    name = 'candidate_annotation_effect'
    source = '''from __future__ import annotations
from dataclasses import dataclass
calls = []
def annotation_effect():
    calls.append('evaluated')
    return int
@dataclass(slots=True)
class Rules:
    value: annotation_effect() = 1
'''
    path = candidate_project / (name + '.py')
    path.write_text(source, encoding='utf-8')
    live = load_reference(name, path, source)
    broker = CandidateImportTransaction()
    try:
        candidate_source = source + ('\nraise RuntimeError("reject candidate")\n' if failure else '')
        broker.register(name, str(path), source=candidate_source)
        if failure:
            with pytest.raises(RuntimeError, match='reject candidate'):
                broker.load(name)
        else:
            candidate = broker.load(name)
            assert candidate.calls == []
            assert dataclasses.fields(candidate.Rules)[0].type == 'annotation_effect()'
            assert candidate.Rules.__module__ == name
        assert live.calls == []
        assert sys.modules[name] is live
        assert live.Rules.__module__ == name
    finally:
        broker.rollback()
        sys.modules.pop(name, None)


@pytest.mark.parametrize('slots', [False, True])
@pytest.mark.parametrize('frozen', [False, True])
def test_dataclass_factory_and_inheritance_use_private_module_namespaces(candidate_project, slots, frozen):
    helper_name, root_name = 'candidate_dc_helper', 'candidate_dc_derived'
    helper_source = f'''from __future__ import annotations
from dataclasses import dataclass, InitVar
from typing import ClassVar
@dataclass(slots={slots}, frozen={frozen})
class Base:
    shared: ClassVar[int] = 13
    seed: InitVar[int] = 7
    value: int = 5
    def __post_init__(self, seed):
        object.__setattr__(self, 'value', self.value + seed)
'''
    root_source = f'''from __future__ import annotations
import dataclasses as dc
from typing import ClassVar
from candidate_dc_helper import Base
@dc.dataclass(slots={slots}, frozen={frozen})
class Derived(Base):
    extra: int = 3
    shared: ClassVar[int] = 17
Factory = dc.make_dataclass('Factory', [('shared', 'ClassVar[int]', 19), ('value', int, 4)],
                            slots={slots}, frozen={frozen})
'''
    helper_path, root_path = (candidate_project / (name + '.py') for name in (helper_name, root_name))
    helper_path.write_text(helper_source, encoding='utf-8')
    root_path.write_text(root_source, encoding='utf-8')
    live_helper = load_reference(helper_name, helper_path, helper_source)
    live_root = load_reference(root_name, root_path, root_source)
    broker = CandidateImportTransaction()
    try:
        broker.register(helper_name, str(helper_path), source=helper_source)
        broker.register(root_name, str(root_path), source=root_source)
        candidate = broker.load(root_name)
        for name in ('Derived', 'Factory'):
            actual, expected = getattr(candidate, name), getattr(live_root, name)
            assert actual.__module__ == root_name
            assert str(inspect.signature(actual)) == str(inspect.signature(expected))
            assert [f.name for f in dataclasses.fields(actual)] == [f.name for f in dataclasses.fields(expected)]
            assert dataclasses.asdict(actual()) == dataclasses.asdict(expected())
        assert candidate.Base is broker.module_for(helper_name).Base
        assert candidate.Base is not live_helper.Base
        assert sys.modules[root_name] is live_root
        assert sys.modules[helper_name] is live_helper
    finally:
        broker.rollback()
        sys.modules.pop(root_name, None)
        sys.modules.pop(helper_name, None)


@pytest.mark.parametrize('future', [False, True])
def test_class_local_alias_follows_standard_string_annotation_rules(candidate_project, future):
    name = 'candidate_dc_local_alias'
    source = ('from __future__ import annotations\n' if future else '') + '''from dataclasses import dataclass
from typing import ClassVar
@dataclass
class Rules:
    CV = ClassVar
    shared: CV[int] = 11
    value: int = 5
'''
    path = candidate_project / (name + '.py')
    path.write_text(source, encoding='utf-8')
    live = load_reference(name, path, source)
    broker = CandidateImportTransaction()
    try:
        broker.register(name, str(path), source=source)
        candidate = broker.load(name)
        assert [f.name for f in dataclasses.fields(candidate.Rules)] == [f.name for f in dataclasses.fields(live.Rules)]
        assert dataclasses.asdict(candidate.Rules()) == dataclasses.asdict(live.Rules())
    finally:
        broker.rollback()
        sys.modules.pop(name, None)
