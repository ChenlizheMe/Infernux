"""Fixed-shape distributions composed inside the shared neural value graph.

Compose policy and distribution operations in one Module.forward for training.
Parameters must satisfy their distribution's mathematical domain. Runtime value
validation is opt-in in Torch; this static backend currently supports trusted
inputs only, without downloading GPU parameters to validate them on the CPU.
"""
from functools import cached_property
import math

import numpy as np

from . import functional as F
from ._tensor import Tensor
from .modules import Module


class Categorical:
    """A categorical policy with the last axis containing class probabilities.

    Supply either nonnegative finite ``probs`` with positive row sums, or
    ``logits`` with at least one finite class per row and no NaN/+infinity.
    ``log_prob`` takes integer Tensor actions. Shapes and dtypes are checked
    during preparation; dynamic argument validation is not yet implemented.
    """
    has_rsample = False

    def __init__(self, probs=None, logits=None, validate_args=None):
        from ._plan import _Symbol
        if (probs is None) == (logits is None):
            raise ValueError("specify exactly one of probs or logits")
        if validate_args not in (None, False, True):
            raise TypeError("validate_args must be bool or None")
        if validate_args:
            raise NotImplementedError("dynamic distribution validation is not available in a static graph")
        self._raw = probs if probs is not None else logits
        self._is_probs = probs is not None
        if not isinstance(self._raw, (Tensor, _Symbol)) or not self._raw.shape:
            raise ValueError("Categorical requires a Tensor with a nonempty class axis")
        if self._raw.dtype != np.dtype(np.float32):
            raise TypeError("Categorical parameters must be float32")
        self.batch_shape, self.event_shape = self._raw.shape[:-1], ()

    def _call(self, operation, *arguments, sample_shape=()):
        from ._plan import _Symbol
        if isinstance(self._raw, Tensor):
            return _CategoricalOperation(self._is_probs, operation, sample_shape)(self._raw, *arguments)
        capture = self._raw.capture
        if operation == "probs":
            return self._raw / self._raw.sum(-1, keepdim=True) if self._is_probs else F.softmax(self._raw, -1)
        if operation == "logits":
            return capture.unary("probability_log", self.probs) if self._is_probs else F.log_softmax(self._raw, -1)
        if operation == "mode":
            return self.probs.argmax(-1)
        if operation == "log_prob":
            action, = arguments
            if not isinstance(action, _Symbol) or action.capture is not capture:
                raise TypeError("Categorical actions must belong to the same captured graph")
            return capture.categorical_select(self.logits, action)
        if operation == "entropy":
            return capture.categorical_entropy(self.probs if self._is_probs else self.logits, self._is_probs)
        if operation == "sample":
            return capture.categorical_sample(self.probs, sample_shape)
        raise ValueError("unknown categorical operation")

    @cached_property
    def probs(self):
        return self._call("probs")

    @cached_property
    def logits(self):
        return self._call("logits")

    @property
    def mode(self):
        return self._call("mode")

    def log_prob(self, value):
        return self._call("log_prob", value)

    def entropy(self):
        return self._call("entropy")

    def sample(self, sample_shape=()):
        from ._random import _shape
        return self._call("sample", sample_shape=_shape(sample_shape))


class _CategoricalOperation(Module):
    def __init__(self, is_probs, operation, sample_shape=()):
        super().__init__()
        self._is_probs, self._operation = is_probs, operation
        self._sample_shape = sample_shape

    def forward(self, raw, *arguments):
        distribution = Categorical(probs=raw) if self._is_probs else Categorical(logits=raw)
        return distribution._call(self._operation, *arguments, sample_shape=self._sample_shape)


class Normal:
    """Independent scalar Gaussians with broadcast float32 parameters.

    ``scale`` must be positive and finite. ``sample`` stops gradients;
    ``rsample`` differentiates location and scale using reparameterization.
    Tensor values follow the same trusted-input contract as Categorical.
    Compose trainable operations in one Module.forward.
    """
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        from ._plan import _Symbol, _capture
        if validate_args not in (None, False, True):
            raise TypeError("validate_args must be bool or None")
        if validate_args:
            raise NotImplementedError("dynamic distribution validation is not available in a static graph")
        parameters = (loc, scale)
        symbols = [value for value in parameters if isinstance(value, _Symbol)]
        tensors = [value for value in parameters if isinstance(value, Tensor)]
        if symbols and tensors:
            raise TypeError("captured Normal parameters must belong to one graph")
        capture = _capture.get()
        anchor = (symbols or tensors or ([capture.inputs[0]] if capture is not None else [None]))[0]

        def parameter(value):
            if isinstance(value, (_Symbol, Tensor)):
                if value.dtype != np.dtype(np.float32):
                    raise TypeError("Normal parameters must be float32")
                if symbols and value.capture is not anchor.capture:
                    raise ValueError("Normal parameters must belong to one captured graph")
                if tensors and value.device != anchor.device:
                    raise ValueError("Normal parameters must use one device")
                return value
            if not isinstance(value, (int, float, np.number)):
                raise TypeError("Normal parameters must be float scalars or Tensors")
            if isinstance(anchor, _Symbol):
                return anchor.capture.constant(anchor, float(value), shape=(), dtype=np.float32)
            return Tensor(value, device="cpu" if anchor is None else anchor.device)

        self._loc, self._scale = (parameter(value) for value in parameters)
        self.batch_shape = tuple(np.broadcast_shapes(self._loc.shape, self._scale.shape))
        self.event_shape = ()

    def _call(self, operation, *arguments, sample_shape=()):
        if isinstance(self._loc, Tensor):
            values = tuple(value if isinstance(value, Tensor) else Tensor(value, device=self._loc.device)
                           for value in arguments)
            return _NormalOperation(operation, sample_shape)(self._loc, self._scale, *values)
        from ._plan import _Symbol
        from ._random import rand
        capture = self._loc.capture
        loc, scale = (capture._expand(value, self.batch_shape) for value in (self._loc, self._scale))
        if operation == "loc":
            return loc
        if operation == "scale":
            return scale
        if operation == "variance":
            return scale.square()
        if operation == "entropy":
            return scale.log() + .5 * (1. + math.log(2. * math.pi))
        if operation == "log_prob":
            value, = arguments
            if isinstance(value, (int, float, np.number)):
                value = capture.constant(loc, float(value))
            if not isinstance(value, _Symbol) or value.capture is not capture:
                raise TypeError("Normal observations must belong to the same captured graph")
            return -.5 * ((value-loc)/scale).square() - scale.log() - .5*math.log(2.*math.pi)
        if operation == "rsample":
            # Two independent counter ranges. 1-U is in (0,1], including a
            # finite radius when U=0; no epsilon clamp changes the density.
            shape = sample_shape + self.batch_shape
            noise = ((-2. * (1.-rand(shape)).log()).sqrt() * (2.*math.pi*rand(shape)).cos()).detach()
            return loc + scale * noise
        raise ValueError("unknown normal operation")

    @cached_property
    def loc(self):
        return self._call("loc")

    @cached_property
    def scale(self):
        return self._call("scale")

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self._call("variance")

    def log_prob(self, value):
        return self._call("log_prob", value)

    def entropy(self):
        return self._call("entropy")

    def rsample(self, sample_shape=()):
        from ._random import _shape
        return self._call("rsample", sample_shape=_shape(sample_shape))

    def sample(self, sample_shape=()):
        from ._grad import no_grad
        with no_grad():
            return self.rsample(sample_shape)


class _NormalOperation(Module):
    def __init__(self, operation, sample_shape):
        super().__init__()
        self._operation, self._sample_shape = operation, sample_shape

    def forward(self, loc, scale, *arguments):
        return Normal(loc, scale)._call(self._operation, *arguments, sample_shape=self._sample_shape)
