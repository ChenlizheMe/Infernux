"""Imported numeric helpers keep Python scopes and original source diagnostics."""

import ast
from types import ModuleType

import pytest

from infernux._compiler.slang import prepare_kernel
from infernux._compiler.work_item import normalize_work_item
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor


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
