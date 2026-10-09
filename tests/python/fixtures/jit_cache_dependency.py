"""Foreign module whose constants are changed without modifying the caller."""

from numba.extending import register_jitable


FACTOR = 2
OFFSET = FACTOR
AUTO_PARALLEL = False


def configure(factor, auto_parallel):
    global FACTOR, OFFSET, AUTO_PARALLEL
    FACTOR = OFFSET = factor
    AUTO_PARALLEL = auto_parallel


@register_jitable
def multiply(value):
    return value * FACTOR
