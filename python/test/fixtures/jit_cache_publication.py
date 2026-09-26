"""Unchanged source used to test disk-cache reuse across process launches."""

import os

from Infernux import jit
import jit_cache_dependency as settings
from jit_cache_dependency import multiply


FACTOR = int(os.environ["INFERNUX_TEST_JIT_FACTOR"])


def leaf(value):
    return value * FACTOR


def helper(value):
    return leaf(value)


@jit.compile(cache=True, auto_parallel=os.environ["INFERNUX_TEST_JIT_AUTO"] == "1")
def direct(value):
    return value * FACTOR


@jit.compile(cache=True, auto_parallel=os.environ["INFERNUX_TEST_JIT_AUTO"] == "1")
def indirect(value):
    return helper(value)


@jit.compile(cache=True, auto_parallel=os.environ["INFERNUX_TEST_JIT_AUTO"] == "1")
def module_constant(value):
    return value * settings.OFFSET


@jit.compile(cache=True, auto_parallel=os.environ["INFERNUX_TEST_JIT_AUTO"] == "1")
def foreign_helper(value):
    return multiply(value)


@jit.compile(cache=True, parallel_policy="required")
def fill(values):
    for index in range(len(values)):
        values[index] = index * FACTOR


def source_less(value):
    return value * FACTOR


source_less.__code__ = source_less.__code__.replace(co_filename="<Content.inxpkg>/cpu_kernel.py")
source_less = jit.compile(source_less, cache=True, auto_parallel=False)
