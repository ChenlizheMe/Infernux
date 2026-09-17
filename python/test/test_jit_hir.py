"""Corpus tests for the standalone Typed HIR legality front-end."""

from __future__ import annotations

import pytest

from Infernux.jit_hir import (
    AliasRiskKind,
    BasicBlockKind,
    BufferAccessKind,
    DiagnosticCode,
    EffectKind,
    HIRParseError,
    ValueType,
    analyze_buffer_aliases,
    analyze_function,
    build_hir,
    hir_fingerprint,
)


def _loop(source: str):
    hir = build_hir(source)
    assert len(hir.loops) == 1
    return hir, hir.loops[0]


@pytest.mark.parametrize("body,tail", [
    ("if source[i] > 0:\n            value = source[i]\n        output[i] = value", "return output"),
    ("output[i] = value\n        value = source[i]", "return output"),
    ("value = source[i]\n        output[i] = value", "return value"),
    ("i = 0\n        output[i] = source[i]", "return output"),
    ("output[i] = source[i]", "return i"),
    ("value = value + source[i]\n        output[i] = value", "return output"),
    ("if source[i] > 0:\n            value = source[i]\n        elif source[i] < 0:\n            continue\n        output[i] = value", "return output"),
    ("value = source[i]\n        output[i] = value", "if n > 0:\n        value = 9\n    return value"),
])
def test_scalar_dependencies_cannot_be_lowered_as_independent_iterations(body, tail):
    hir, loop = _loop(
        "def kernel(source, output, n):\n    value = 0\n"
        f"    for i in range(n):\n        {body}\n    {tail}\n"
    )
    assert not loop.parallel_eligible
    assert any(item.code == DiagnosticCode.LOOP_CARRIED_SCALAR for item in hir.diagnostics)


@pytest.mark.parametrize("body,tail", [
    ("if source[i] > 0:\n            value = source[i]\n        else:\n            value = -source[i]\n        output[i] = value", "return output"),
    ("if source[i] <= 0:\n            continue\n        value = source[i]\n        output[i] = value", "return output"),
    ("value = source[i]\n        output[i] = value", "value = 9\n    return value"),
    ("value = source[i]\n        value = value * 2\n        output[i] = value", "return output"),
    ("if source[i] <= 0:\n            continue\n        else:\n            value = source[i]\n        output[i] = value", "return output"),
    ("value = source[i]\n        output[i] = value", "if n > 0:\n        value = 9\n    else:\n        value = 3\n    return value"),
])
def test_iteration_local_scalars_and_dead_loop_outputs_remain_parallel(body, tail):
    _hir, loop = _loop(
        "def kernel(source, output, n):\n    value = 0\n"
        f"    for i in range(n):\n        {body}\n    {tail}\n"
    )
    assert loop.parallel_eligible, loop.diagnostics


@pytest.mark.parametrize("body", [
    "total += source[i]\n        total *= 2",
    "if source[i] > 0:\n            total += source[i]\n        else:\n            total *= 2",
    "total = 1\n        total += source[i]",
    "total += source[i]\n        total = 1",
])
def test_mixed_or_reset_accumulators_are_not_parallel_reductions(body):
    _hir, loop = _loop(
        "def kernel(source, n):\n    total = 1\n"
        f"    for i in range(n):\n        {body}\n    return total\n"
    )
    assert not loop.parallel_eligible
    assert DiagnosticCode.INVALID_REDUCTION in _codes(loop)


def test_repeated_additions_to_one_accumulator_remain_a_reduction():
    _hir, loop = _loop("""def kernel(source, n):
    total = 1
    for i in range(n):
        total += source[i]
        if source[i] > 0:
            total += 2
    return total
""")
    assert loop.parallel_eligible, loop.diagnostics


@pytest.mark.parametrize("read,expected", [("i, 1", False), ("i-1, 0", True)])
def test_multidimensional_row_access_dependence(read, expected):
    hir = build_hir(f"def kernel(x,n):\n    for i in range(1,n):\n        x[i,0] = x[{read}] + 1\n        x[i,1] = 2\n")
    hazards = {DiagnosticCode.LOOP_CARRIED_READ, DiagnosticCode.LOOP_CARRIED_WRITE}
    assert any(item.code in hazards for item in hir.diagnostics) is expected
    # Tuple-index lowering is not yet part of the minimal CPU HIR.
    assert not hir.eligible_loops


@pytest.mark.parametrize("body,expected", [
    ("first[i] = second[i - 1] + 1", DiagnosticCode.LOOP_CARRIED_READ),
    ("first[i] = 1\n        second[i + 1] = 2", DiagnosticCode.LOOP_CARRIED_WRITE),
    ("first[i] += 1\n        second[i] += 2", None),
    ("first[2 * i] = 1\n        second[2 * i + 1] = 2", None),
    ("output[i] = first[i - 1] + second[i + 1]", None),
])
def test_actual_equal_layout_aliases_share_the_symbolic_dependence_rules(body, expected):
    hir = build_hir(f"def kernel(first, second, output, n):\n    for i in range(1, n):\n        {body}\n")
    assert analyze_buffer_aliases(hir, ()) == ()
    diagnostics = analyze_buffer_aliases(hir, (("first", "second"),))
    if expected is None:
        assert diagnostics == ()
    else:
        assert diagnostics[0].code == expected
        assert diagnostics[0].location.line >= 3


def test_nested_aliases_are_not_proven_by_missing_hir_accesses():
    hir = build_hir("""def kernel(first, second):
    for i in range(first.shape[0]):
        for j in range(3):
            first[i] += second[j]
""")
    diagnostics = analyze_buffer_aliases(hir, (("first", "second"),))
    assert diagnostics[0].code == DiagnosticCode.ALIAS_RISK
    assert "not proven" in diagnostics[0].message


def test_alias_residue_proof_does_not_require_affine_trip_count():
    hir = build_hir("""def kernel(first, second):
    for i in range(first.shape[0] // 2):
        first[2 * i] = 5
        second[2 * i + 1] = 9
""")
    assert any(item.code == DiagnosticCode.UNSUPPORTED_RANGE for item in hir.diagnostics)
    assert analyze_buffer_aliases(hir, (("first", "second"),)) == ()


def test_unresolved_local_array_binding_is_not_mistaken_for_independent_storage():
    hir = build_hir("""def kernel(first, second):
    view = first
    for i in range(1, first.shape[0]):
        view[i] = second[i - 1] + 1
""")
    diagnostics = analyze_buffer_aliases(hir, (("first", "second"),))
    assert diagnostics[0].code == DiagnosticCode.ALIAS_RISK
    assert "indirect binding" in diagnostics[0].message


def _codes(loop):
    return {diagnostic.code for diagnostic in loop.diagnostics}


class TestSafeCorpus:
    def test_even_and_odd_writes_do_not_conflict(self):
        _, loop = _loop('''
def fill(values, n):
    for i in range(n):
        values[2 * i] = 1
        values[2 * i + 1] = 2
''')
        assert loop.parallel_eligible

    def test_same_shifted_element_read_write_is_independent(self):
        _, loop = _loop('''
def update(values, n):
    for i in range(n - 1):
        values[i + 1] = values[i + 1] + 1
''')
        assert loop.parallel_eligible

    def test_reduction_feedback_is_not_a_parallel_reduction(self):
        _, loop = _loop('''
def scan(values, output):
    total = 0
    for i in range(len(values)):
        total += values[i]
        output[i] = total
    return total
''')
        assert not loop.parallel_eligible
        assert DiagnosticCode.REDUCTION_FEEDBACK in _codes(loop)

    def test_different_writes_to_one_buffer_are_not_independent(self):
        _, loop = _loop('''
def overwrite(values, n):
    for i in range(n - 1):
        values[i] = 1
        values[i + 1] = 2
''')
        assert not loop.parallel_eligible
        assert DiagnosticCode.LOOP_CARRIED_WRITE in _codes(loop)

    def test_direct_elementwise_write_and_pure_math(self):
        hir, loop = _loop(
            """
            import math

            def scale(out: array, values: array, n: int):
                for i in range(n):
                    out[i] = math.sqrt(values[i]) * 2.0
            """
        )

        assert hir.parallel_eligible
        assert loop.parallel_eligible
        assert loop.index_name == "i"
        assert loop.range_spec is not None and loop.range_spec.is_affine
        assert loop.buffer_writes[0].kind == BufferAccessKind.WRITE
        assert loop.buffer_writes[0].unique
        assert loop.buffer_reads[0].same_iteration
        assert any(effect.kind == EffectKind.PURE_CALL for effect in loop.effects)
        assert not loop.diagnostics

    def test_shape_query_is_a_valid_range_bound(self):
        _, loop = _loop(
            """
            def copy(out: array, values: array):
                for i in range(values.shape[0]):
                    out[i] = values[i]
            """
        )

        assert loop.parallel_eligible
        assert loop.range_spec is not None
        assert loop.range_spec.stop_affine is not None
        assert loop.range_spec.stop_affine.variables == ("values.shape[0]",)

    def test_len_and_numpy_scalar_allowlist(self):
        hir, loop = _loop(
            """
            import numpy as np

            def transform(out: array, values: array):
                for i in range(len(values)):
                    out[i] = np.maximum(np.sin(values[i]), 0.0)
            """
        )

        assert hir.parallel_eligible
        assert loop.parallel_eligible
        assert not _codes(loop)
        assert sum(effect.kind == EffectKind.PURE_CALL for effect in loop.effects) >= 2

    def test_strided_affine_range_and_unique_affine_write(self):
        _, loop = _loop(
            """
            def fill(out: array, n: int):
                for i in range(1, n, 2):
                    out[2 * i + 1] = 1.0
            """
        )

        assert loop.parallel_eligible
        assert loop.range_spec is not None and loop.range_spec.step_value == 2
        assert loop.buffer_writes[0].index is not None
        assert loop.buffer_writes[0].index.coefficient("i") == 2

    def test_scalar_sum_reduction(self):
        hir, loop = _loop(
            """
            def sum_values(values: array, n: int):
                total = 0.0
                for i in range(n):
                    total += values[i]
                return total
            """
        )

        assert hir.parallel_eligible
        assert loop.parallel_eligible
        assert len(loop.reductions) == 1
        assert loop.reductions[0].target == "total"
        assert loop.reductions[0].operator == "+"
        assert any(effect.kind == EffectKind.REDUCTION for effect in loop.effects)

    def test_elementwise_alias_risk_is_reported_without_rejecting_same_index_work(self):
        _, loop = _loop(
            """
            def copy(out: array, values: array, n: int):
                for i in range(n):
                    out[i] = values[i]
            """
        )

        assert loop.parallel_eligible
        assert loop.alias_risks[0].kind == AliasRiskKind.POSSIBLE
        assert "out" in loop.alias_risks[0].buffers

    def test_source_position_and_stable_id_are_transformer_facing(self):
        source = """
        def kernel(out: array, n: int):
            for i in range(n):
                out[i] = i
        """
        first = build_hir(source).loops[0]
        second = build_hir(source).loops[0]

        assert first.stable_id == second.stable_id
        assert first.source_location.line == 3
        assert first.source_location.end_line == 4
        assert "out[i] = i" in first.source

    def test_stable_id_survives_source_line_shift(self):
        first = build_hir(
            """
            def kernel(out: array, n: int):
                for i in range(n):
                    out[i] = i
            """
        ).loops[0]
        shifted = build_hir(
            """


            def kernel(out: array, n: int):
                for i in range(n):
                    out[i] = i
            """
        ).loops[0]

        assert first.stable_id == shifted.stable_id
        assert shifted.source_location.line == 5

    def test_function_object_entry_point_does_not_execute_function(self):
        calls = []

        def kernel(out: list, n: int):
            calls.append("must not run")
            for i in range(n):
                out[i] = i

        hir = analyze_function(kernel)
        assert hir.parallel_eligible
        assert calls == []

    def test_structured_cfg_contains_loop_back_edge_and_exit_edge(self):
        hir = build_hir(
            """
            def kernel(out: array, n: int):
                scale = 2
                for i in range(n):
                    out[i] = i * scale
                return out
            """
        )

        kinds = [block.kind for block in hir.blocks]
        assert kinds == [
            BasicBlockKind.ENTRY,
            BasicBlockKind.LINEAR,
            BasicBlockKind.LOOP_HEADER,
            BasicBlockKind.LOOP_BODY,
            BasicBlockKind.LINEAR,
            BasicBlockKind.EXIT,
        ]
        header = next(block for block in hir.blocks if block.kind == BasicBlockKind.LOOP_HEADER)
        body = next(block for block in hir.blocks if block.kind == BasicBlockKind.LOOP_BODY)
        tail = hir.blocks[-2]
        assert header.successors == (body.stable_id, tail.stable_id)
        assert body.successors == (header.stable_id,)
        assert tail.successors == (hir.exit_block.stable_id,)

    def test_hir_fingerprint_changes_with_semantics_not_whitespace(self):
        first = build_hir(
            """
            def kernel(out: array, n: int):
                for i in range(n):
                    out[i] = i * 2
            """
        )
        spaced = build_hir(
            """

            def kernel(out: array, n: int):
                for i in range(n):
                    out[i] = i * 2
            """
        )
        changed = build_hir(
            """
            def kernel(out: array, n: int):
                for i in range(n):
                    out[i] = i * 3
            """
        )

        assert hir_fingerprint(first) == hir_fingerprint(spaced)
        assert hir_fingerprint(first) != hir_fingerprint(changed)


class TestUnsafeCorpus:
    @pytest.mark.parametrize(
        ("body", "code"),
        [
            ("out[i] = out[i - 1] + 1", DiagnosticCode.LOOP_CARRIED_READ),
            ("out[index[i]] = values[i]", DiagnosticCode.INDIRECT_WRITE),
            ("out[i] = make_value(values[i])", DiagnosticCode.UNKNOWN_CALL),
            ("values.append(i)", DiagnosticCode.CONTAINER_MUTATION),
            ("break", DiagnosticCode.UNSUPPORTED_CONTROL_FLOW),
            ("return", DiagnosticCode.UNSUPPORTED_CONTROL_FLOW),
            ("yield i", DiagnosticCode.UNSUPPORTED_CONTROL_FLOW),
            ("await values[i]", DiagnosticCode.UNSUPPORTED_CONTROL_FLOW),
        ],
    )
    def test_unsafe_constructs_are_rejected_with_specific_diagnostic(self, body, code):
        source = f"""
        def kernel(out: array, values: array, index: array, n: int):
            for i in range(n):
                {body}
        """

        hir, loop = _loop(source)
        assert not loop.parallel_eligible
        assert code in _codes(loop)
        assert any(diagnostic.message for diagnostic in loop.diagnostics)

    def test_try_is_rejected_even_when_body_looks_pure(self):
        hir, loop = _loop(
            """
            def kernel(out: array, values: array, n: int):
                for i in range(n):
                    try:
                        out[i] = values[i]
                    except ValueError:
                        out[i] = 0.0
            """
        )

        assert not hir.parallel_eligible
        assert DiagnosticCode.UNSUPPORTED_CONTROL_FLOW in _codes(loop)

    def test_unknown_range_and_zero_step_are_rejected(self):
        _, dynamic = _loop(
            """
            def kernel(out: array, n: int):
                for i in range(start(), n):
                    out[i] = 1.0
            """
        )
        _, zero = _loop(
            """
            def kernel(out: array, n: int):
                for i in range(n, 0, 0):
                    out[i] = 1.0
            """
        )

        assert DiagnosticCode.UNSUPPORTED_RANGE in _codes(dynamic)
        assert DiagnosticCode.UNSUPPORTED_RANGE in _codes(zero)

    def test_non_affine_read_is_rejected_conservatively(self):
        _, loop = _loop(
            """
            def kernel(out: array, values: array, index: int, n: int):
                for i in range(n):
                    out[i] = values[index]
            """
        )

        assert not loop.parallel_eligible
        assert DiagnosticCode.NON_AFFINE_INDEX in _codes(loop)

    def test_offset_access_between_distinct_buffers_is_rejected_for_alias_safety(self):
        _, loop = _loop(
            """
            def kernel(out: array, values: array, n: int):
                for i in range(n):
                    out[i] = values[i - 1]
            """
        )

        assert not loop.parallel_eligible
        assert DiagnosticCode.ALIAS_RISK in _codes(loop)

    def test_uninitialized_scalar_update_is_not_a_reduction(self):
        _, loop = _loop(
            """
            def kernel(values: array, n: int):
                for i in range(n):
                    total += values[i]
            """
        )

        assert not loop.parallel_eligible
        assert DiagnosticCode.INVALID_REDUCTION in _codes(loop)

    @pytest.mark.parametrize("operator", ["-=", "/="])
    def test_non_associative_augmented_scalar_updates_are_not_reductions(self, operator):
        _, loop = _loop(
            f"""
            def kernel(values: array, n: int):
                total = 1.0
                for i in range(n):
                    total {operator} values[i]
            """
        )

        assert not loop.parallel_eligible
        assert DiagnosticCode.INVALID_REDUCTION in _codes(loop)

    def test_nested_loop_is_rejected(self):
        hir = build_hir(
            """
            def kernel(out: array, values: array, n: int, m: int):
                for i in range(n):
                    for j in range(m):
                        out[i] = values[j]
            """
        )
        loop = hir.loops[0]

        assert not loop.parallel_eligible
        assert DiagnosticCode.UNSUPPORTED_NESTED_LOOP in _codes(loop)

    def test_unknown_attribute_call_is_rejected(self):
        _, loop = _loop(
            """
            def kernel(out: array, values: array, n: int, service):
                for i in range(n):
                    out[i] = service.sample(values[i])
            """
        )

        assert not loop.parallel_eligible
        assert DiagnosticCode.UNKNOWN_CALL in _codes(loop)

    def test_normal_function_return_after_loop_does_not_block_candidate(self):
        hir, loop = _loop(
            """
            def kernel(out: array, n: int):
                for i in range(n):
                    out[i] = i
                return out
            """
        )

        assert loop.parallel_eligible
        assert hir.parallel_eligible

    def test_no_output_loop_is_not_a_candidate(self):
        _, loop = _loop(
            """
            import math

            def kernel(values: array, n: int):
                for i in range(n):
                    math.sin(values[i])
            """
        )

        assert not loop.parallel_eligible
        assert "no analyzable output" in loop.reason

    def test_parse_error_is_explicit(self):
        with pytest.raises(HIRParseError) as error:
            build_hir("def broken(:\n    pass\n")
        assert "parse_error" in str(error.value)
