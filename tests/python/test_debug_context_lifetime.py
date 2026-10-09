"""Retained diagnostics must not own gameplay objects or their native resources."""
from datetime import datetime
import gc
import weakref

import pytest

from infernux.core.material import Material
from infernux.debug import Debug, DebugConsole, LogEntry, LogType


class ContextOwner:
    def __init__(self):
        self.material = Material.create_unlit("Logged runtime material")


@pytest.fixture
def console():
    previous = DebugConsole._instance
    current = DebugConsole()
    yield current
    current.clear()
    DebugConsole._instance = previous


@pytest.mark.parametrize("logger", [Debug.log, Debug.log_warning, Debug.log_error])
def test_retained_log_releases_context_and_native_resource(console, logger):
    owner = ContextOwner()
    reference = weakref.ref(owner)
    resource = weakref.ref(owner.material)
    native = weakref.ref(owner.material.native)
    logger("Gameplay observation remains available", owner)
    entry = console.get_entries()[-1]
    assert entry.context is owner

    del owner
    gc.collect()
    assert reference() is None
    assert resource() is None
    assert native() is None
    assert entry.context is None
    assert entry.message == "Gameplay observation remains available"
    assert console.get_entries()[-1] is entry


def test_context_assignment_releases_previous_owner():
    first, second = ContextOwner(), ContextOwner()
    first_ref, second_ref = weakref.ref(first), weakref.ref(second)
    entry = LogEntry("replace context", LogType.LOG, datetime.now(), context=first)
    entry.context = second
    del first
    assert first_ref() is None
    assert entry.context is second
    del second
    assert second_ref() is None
    assert entry.context is None


def test_nonweak_context_container_does_not_retain_scene_resources(console):
    owner = ContextOwner()
    reference = weakref.ref(owner)
    Debug.log("unsupported context is observational", {"owner": owner})
    del owner
    assert reference() is None
    entry = console.get_entries()[-1]
    assert entry.context is None
    assert entry.message == "unsupported context is observational"


def test_none_context_and_clearing_context_are_safe():
    owner = ContextOwner()
    entry = LogEntry("optional", LogType.LOG, datetime.now())
    assert entry.context is None
    entry.context = owner
    assert entry.context is owner
    entry.context = None
    assert entry.context is None
