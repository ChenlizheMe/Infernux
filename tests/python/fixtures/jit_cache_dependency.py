"""Foreign module whose constants are changed without modifying the caller."""

import os

from numba.extending import register_jitable


FACTOR = int(os.environ["INFERNUX_TEST_JIT_FACTOR"])
OFFSET = FACTOR


@register_jitable
def multiply(value):
    return value * FACTOR
