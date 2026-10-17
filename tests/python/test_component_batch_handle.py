"""Reusable component field batches stay validated and live."""

import numpy as np
import pytest

from infernux.batch import (
    batch_read,
    batch_write,
    create_component_batch_handle,
)
from infernux.components import InxComponent
from infernux.components.fields import serialized_field


class Probe(InxComponent):
    speed: float = serialized_field(default=1.0, range=(0.0, 5.0))
    label: str = serialized_field(default="")


class Other(InxComponent):
    speed: float = serialized_field(default=0.0)


def test_handle_reads_and_writes_the_same_values_as_the_list_api():
    components = [Probe() for _ in range(3)]
    handle = create_component_batch_handle(components, "speed")

    handle.write(np.array([2.0, 8.0, 100.0]))
    np.testing.assert_array_equal(handle.read(), [2.0, 5.0, 5.0])
    np.testing.assert_array_equal(batch_read(components, "speed"), [2.0, 5.0, 5.0])

    components[0].speed = 4.0
    assert handle.read()[0] == 4.0


def test_handle_accepts_the_batch_api_without_a_property():
    components = [Probe() for _ in range(2)]
    handle = create_component_batch_handle(components, "speed")

    batch_write(handle, np.array([1.5, 2.5]))
    np.testing.assert_array_equal(batch_read(handle), [1.5, 2.5])


def test_handle_rejects_mixed_concrete_types():
    with pytest.raises(TypeError, match="same concrete type"):
        create_component_batch_handle([Probe(), Other()], "speed")


def test_handle_requires_a_cds_backed_field():
    components = [Probe()]
    with pytest.raises(TypeError, match="not a CDS-backed numeric field"):
        create_component_batch_handle(components, "label")
