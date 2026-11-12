"""Static CPU/Vulkan networks on Infernux's shared typed WorkItem compiler.

Each input signature captures a fixed value graph over shared model parameters.
Call Module.recompile() after structure changes. This
package shares the same Plan with offline calls and Engine-managed continuation.
Model/checkpoint IO is cold by default, or managed with a completion window.
Baked CPU/GPU models reuse built programs without importing the source compiler.
"""
from numpy import float32, int32, uint32, int64

from ._tensor import Tensor, Parameter, tensor, as_tensor
from .modules import Module, Linear, Conv2d, PixelShuffle, Sigmoid, ReLU, SiLU, ELU, Tanh, Softmax, LogSoftmax, Flatten, Sequential
from .functional import cat, stack, reshape, flatten, transpose, permute, sum, mean, sin, cos, square, exp, log, sqrt, argmax
from .functional import squeeze, unsqueeze, ones_like, zeros_like
from .functional import clamp, minimum, maximum
from ._grad import no_grad, is_grad_enabled
from ._random import rand, manual_seed, get_rng_state, set_rng_state
from . import functional, optim, distributions, utils
from .model import bake, export, load, save_checkpoint, load_checkpoint
from .program import program, Program, Budget
from .sources import Channel
from infernux.execution import Clock
from infernux.runtime_services import Signal

__all__ = ["float32", "int32", "uint32", "int64", "Tensor", "Parameter", "tensor", "as_tensor", "Module", "Linear", "Conv2d", "PixelShuffle", "Sigmoid", "ReLU", "SiLU", "ELU", "Tanh", "Softmax", "LogSoftmax",
           "Flatten", "cat", "stack", "reshape", "flatten", "transpose", "permute", "sum", "mean", "sin", "cos", "square", "exp", "log", "sqrt", "argmax",
           "squeeze", "unsqueeze", "ones_like", "zeros_like", "no_grad", "is_grad_enabled",
           "clamp", "minimum", "maximum",
           "rand", "manual_seed", "get_rng_state", "set_rng_state",
           "Sequential", "functional", "optim", "distributions", "utils", "program", "Program", "Budget", "Channel", "Clock", "Signal",
           "bake", "export", "load", "save_checkpoint", "load_checkpoint"]
