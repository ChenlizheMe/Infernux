"""Exact source retention and JSONC grammar at the configuration edit boundary."""
import pytest

from infernux.engine.jsonc_document import JsoncDocument


@pytest.mark.parametrize("source,expected", [
    ('{"url":"https://host/a//b", "text":"/* hello */"}',
     {"url": "https://host/a//b", "text": "/* hello */"}),
    ('\ufeff{/* header */ "嵌套": [1, {"a": [true, null,],},],}//tail',
     {"嵌套": [1, {"a": [True, None]}]}),
    ('{"x"/* a */:/* b */"comma,} and ]", /* c */ "y":-1.5e2,}',
     {"x": "comma,} and ]", "y": -150.0}),
    ('{\n// comment-only object\n}', {}),
    ('{"x":1, "x":2,}', {"x": 2}),
])
def test_jsonc_decodes_supported_syntax_without_changing_source(source, expected):
    document = JsoncDocument(source)
    assert document.value == expected
    assert document.render(dict(expected)) == source


@pytest.mark.parametrize("source", [
    '{,}', '{"x":[,]}', '{"x":{,}}', '{"x":1,,}', '{"x":NaN}', '{"x":Infinity}',
    '{"x":1 /* unfinished}', '{"x":1} garbage', '{"x":1 "y":2}',
])
def test_jsonc_rejects_invalid_documents_without_rewriting_them(source):
    with pytest.raises(ValueError):
        JsoncDocument(source)


@pytest.mark.parametrize("source,updated,expected", [
    ('{"a":1}', {"a": 1, "b": 2}, '{"a":1,\n    "b": 2\n}'),
    ('{/*keep*/"a":1}', {"a": 3, "b": 2}, '{/*keep*/"a":3,\n    "b": 2\n}'),
    ('{"a":1,"b":2}', {"a": 1}, '{"a":1}'),
    ('{"a":1,"b":2,}', {"a": 1}, '{"a":1,}'),
    ('{"a":1,"a":2}', {"a": 3}, '{"a":1,"a":3}'),
    ('{"a":1, /*comment*/ "b":2}', {"b": 2}, '{ /*comment*/ "b":2}'),
    ('{"a":1, /*comment*/ "b":2}', {}, '{ /*comment*/ }'),
    ('{\r\n\t"a": 1, // keep\r\n}\r\n', {"a": 2, "b": 3},
     '{\r\n\t"a": 2, // keep\r\n\t"b": 3,\r\n}\r\n'),
])
def test_property_edits_preserve_every_unowned_text_span(source, updated, expected):
    rendered = JsoncDocument(source).render(updated)
    assert rendered == expected
    assert JsoncDocument(rendered).value == updated


@pytest.mark.parametrize("indent", [None, 2, "\t"])
@pytest.mark.parametrize("retained_mask", range(16))
def test_property_removal_and_addition_preserve_standard_json_grammar(indent, retained_mask):
    import json

    original = {"first": ["quote\\\"", {"items": [1, 2]}], "second": "https://host/*text*/",
                "third": False, "last": None}
    source = json.dumps(original, indent=indent)
    updated = {key: value for index, (key, value) in enumerate(original.items())
               if retained_mask & (1 << index)}
    for additions in ({}, {"new": ["Assets", "库"]}):
        target = {**updated, **additions}
        rendered = JsoncDocument(source).render(target)
        # The standard library is an independent grammar oracle for plain
        # JSON input; edits must not introduce an unnecessary trailing comma.
        assert json.loads(rendered) == target
