"""Gradient recording mode; storage and execution remain in the common plan."""
from contextlib import contextmanager
from contextvars import ContextVar

_enabled = ContextVar('infernux_nn_grad_enabled', default=True)


def is_grad_enabled():
    return _enabled.get()


@contextmanager
def no_grad():
    token = _enabled.set(False)
    try:
        yield
    finally:
        _enabled.reset(token)
