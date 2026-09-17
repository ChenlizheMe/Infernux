"""A cached recursive CPU specialization with cross-signature calls."""

from Infernux._jit_kernels import _compile_njit


def factorial(value):
    if value < 2:
        return 1
    return value * factorial(value - 1)


factorial = _compile_njit(factorial, {"cache": True})
