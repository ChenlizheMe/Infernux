"""Imported numeric helpers keep Python scopes and original source diagnostics."""

import ast
from dataclasses import replace
from types import ModuleType

import pytest

from infernux import compute
from infernux._compiler.gpu_artifact import _encode_artifact, _read_artifact_payload, _specialization_identity
from infernux._compiler.slang import compile_kernel, prepare_kernel
from infernux._compiler.work_item import normalize_work_item
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor, _specialization_parameters
from infernux.engine.project_context import using_project_root


def load_module(root, name, source, **bindings):
    path = root / (name + '.py')
    path.write_text(source, encoding='utf8')
    module = ModuleType(name)
    module.__dict__.update(bindings)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def evaluate(item, values):
    namespace = dict(item.globals_map)
    namespace['_infernux_gpu_function'] = lambda function: function
    tree = ast.fix_missing_locations(ast.Module(body=[*item.helpers, item.loop_definition()], type_ignores=[]))
    exec(compile(tree, '<normalized-numeric-library>', 'exec'), namespace)
    domain = type('Domain', (list,), {'shape': (len(values),)})(values)
    output = [0] * len(values)
    namespace[item.definition.name](domain, output)
    return output


LIBRARY = '''import infernux as inx
GAIN = {gain}
@inx.compute.function
def apply(value):
    return value * GAIN
'''

KERNEL = '''import infernux as inx
GAIN = 100
@inx.compute.kernel
def run(domain, output):
    i = inx.compute.index(domain)
    output[i] = {expression}
'''


@pytest.mark.parametrize('qualified', [False, True])
def test_same_names_in_separate_modules_keep_their_own_constants(tmp_path, qualified):
    first = load_module(tmp_path, '第一套数学', LIBRARY.format(gain=2))
    second = load_module(tmp_path, '第二套数学', LIBRARY.format(gain=5))
    expression = 'first.apply(domain[i]) + second.apply(domain[i])' if qualified else 'first(domain[i]) + second(domain[i])'
    owner = load_module(tmp_path, '主核', KERNEL.format(expression=expression),
                        first=first if qualified else first.apply, second=second if qualified else second.apply)
    item = normalize_work_item(owner.run.function, 2)
    first.GAIN, second.GAIN = 200, 500
    assert evaluate(item, [1, 2, -3]) == [7, 14, -21]


def test_transitive_helpers_and_aliases_share_one_definition(tmp_path):
    leaf = load_module(tmp_path, '叶子库', LIBRARY.format(gain=3))
    middle = load_module(tmp_path, '中间库', '''import infernux as inx
GAIN = 2
@inx.compute.function
def apply(value):
    return leaf.apply(value) + GAIN
''', leaf=leaf)
    owner = load_module(tmp_path, '调用者', KERNEL.format(
        expression='middle.apply(domain[i]) + alias(domain[i]) + leaf.apply(domain[i])'),
        middle=middle, alias=leaf.apply, leaf=leaf)
    item = normalize_work_item(owner.run.function, 2)
    assert len(item.helpers) == 2
    assert evaluate(item, [1, 2]) == [11, 20]


def test_helper_defaults_keep_the_value_evaluated_at_declaration(tmp_path):
    library = load_module(tmp_path, '默认值', LIBRARY.format(gain=2).replace(
        'apply(value):', 'apply(value, /, bias=GAIN):').replace('value * GAIN', 'value * GAIN + bias'))
    library.GAIN = 5
    owner = load_module(tmp_path, '入口', KERNEL.format(expression='library.apply(domain[i])'), library=library)
    item = normalize_work_item(owner.run.function, 2)
    assert evaluate(item, [1, 2]) == [7, 12]


def test_library_constant_edit_changes_prepared_program_identity(tmp_path):
    library = load_module(tmp_path, '数学库', LIBRARY.format(gain=2.0))
    owner = load_module(tmp_path, '入口', KERNEL.format(expression='library.apply(domain[i])'), library=library)
    descriptor = _gpu_buffer_descriptor((2,), 'float32')
    before = prepare_kernel(owner.run.function, (descriptor, descriptor))
    library.GAIN = 5.0
    after = prepare_kernel(owner.run.function, (descriptor, descriptor))
    assert before.artifact_key != after.artifact_key
    assert before.generated_source != after.generated_source
    assert any(location[0].endswith('数学库.py') and location[1] == 5 for location in before.lowering.source_ranges)


def test_helper_error_points_to_its_file_and_statement(tmp_path):
    library = load_module(tmp_path, '错误库🚀', LIBRARY.format(gain=2).replace('value * GAIN', 'unknown(value)'))
    owner = load_module(tmp_path, '入口', KERNEL.format(expression='library.apply(domain[i])'), library=library)
    descriptor = _gpu_buffer_descriptor((2,), 'float32')
    with pytest.raises(SyntaxError) as raised:
        prepare_kernel(owner.run.function, (descriptor, descriptor))
    assert raised.value.filename == str(tmp_path / '错误库🚀.py')
    assert raised.value.lineno == 5
    assert 'unknown(value)' in raised.value.text


@pytest.mark.parametrize('arguments', ['domain[i], unknown=1', 'domain[i], value=2'])
def test_invalid_helper_keyword_points_to_the_call(tmp_path, arguments):
    library = load_module(tmp_path, '数学库', LIBRARY.format(gain=2))
    owner = load_module(tmp_path, '入口', KERNEL.format(expression=f'library.apply({arguments})'), library=library)
    descriptor = _gpu_buffer_descriptor((2,), 'float32')
    with pytest.raises(SyntaxError, match='keywords') as raised:
        prepare_kernel(owner.run.function, (descriptor, descriptor))
    assert raised.value.filename == str(tmp_path / '入口.py')
    assert raised.value.lineno == 6


def test_alias_recursion_is_rejected_before_emission(tmp_path):
    library = load_module(tmp_path, '递归库', '''import infernux as inx
@inx.compute.function
def apply(value):
    return alias(value)
''')
    library.alias = library.apply
    owner = load_module(tmp_path, '入口', KERNEL.format(expression='library.apply(domain[i])'), library=library)
    with pytest.raises(TypeError, match='compute helper recursion.*递归库.apply'):
        normalize_work_item(owner.run.function, 2)


@compute.kernel
def affine(values, output, scale, bias):
    i = compute.index(values)
    output[i] = values[i] * scale + bias


@compute.kernel
def square(values, output):
    i = compute.index(values)
    output[i] = values[i] * values[i]


@pytest.mark.parametrize('target', ['spirv', 'wgsl'])
def test_target_program_and_layout_roundtrip(tmp_path, target):
    descriptor = _gpu_buffer_descriptor((8,), 'float32')
    params = descriptor, descriptor, 2.0, 3.0
    with using_project_root(tmp_path):
        artifact = compile_kernel(affine.function, params, target=target)
    specialization = _specialization_identity(params, target=target)
    header, loaded = _read_artifact_payload(_encode_artifact(artifact, specialization), specialization=specialization)
    assert loaded.target == artifact.target == header['target'] == target
    assert loaded.tasks == artifact.tasks and loaded.argument_layout == artifact.argument_layout
    assert loaded.task_metadata[0]['entry_point'] == 'main'
    assert loaded.task_metadata[0]['threads_per_group'] == 64
    assert loaded.argument_layout['size'] > 0
    if target == 'spirv':
        assert loaded.tasks[0][:4] == b'\x03\x02\x23\x07'
    else:
        assert '@compute' in loaded.tasks[0].decode('utf8')
    _, request = _specialization_parameters(specialization)
    assert request['target'] == target


def test_target_changes_identity_but_runtime_scalars_do_not():
    descriptor = _gpu_buffer_descriptor((8,), 'float32')
    params = descriptor, descriptor, 2.0, 3.0
    spirv = prepare_kernel(affine.function, params, target='spirv')
    wgsl = prepare_kernel(affine.function, params, target='wgsl')
    changed = prepare_kernel(affine.function, (descriptor, descriptor, 8.0, -1.0), target='spirv')
    assert spirv.generated_source == wgsl.generated_source
    assert spirv.artifact_key != wgsl.artifact_key
    assert spirv.artifact_key == changed.artifact_key
    assert _specialization_identity(params, target='spirv') != _specialization_identity(params, target='wgsl')


@pytest.mark.parametrize('target', ['spirv', 'wgsl'])
def test_reverse_program_uses_the_same_explicit_target(tmp_path, target):
    descriptor = _gpu_buffer_descriptor((8,), 'float32')
    params = descriptor, descriptor
    with using_project_root(tmp_path):
        artifact = compile_kernel(square.function, params, target=target, autodiff_mode='reverse', gradient_parameters=(0, 1))
    assert artifact.target == target and artifact.gradient_parameters == (0, 1)
    assert {row['is_grad'] for row in artifact.task_metadata[0]['buffer_bindings'] if row['resource_kind'] == 'external_buffer'} == {True, False}


def test_program_target_cannot_disagree_with_its_specialization(tmp_path):
    descriptor = _gpu_buffer_descriptor((8,), 'float32')
    params = descriptor, descriptor, 2.0, 3.0
    with using_project_root(tmp_path):
        artifact = compile_kernel(affine.function, params, target='spirv')
    specialization = _specialization_identity(params, target='spirv')
    with pytest.raises(RuntimeError, match='compilation identity mismatch'):
        _read_artifact_payload(_encode_artifact(replace(artifact, target='wgsl'), specialization))
