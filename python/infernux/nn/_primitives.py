"""Finite typed recipes lowered through existing CPU ranges or GPU kernels."""
from contextlib import contextmanager
import math

import infernux as inx
from infernux._work import (prepare as _prepare_work, _deferred_prepares, _prepare_gpu, _prepare_gpu_family,
    program_family, inner_range, domain_parameter, accesses, inner_domain, gpu_arguments, retire)

_K, _M, _B, _COUNT, _INV_N = 1, 1, 1, 1, 1.0
_WIDTH, _STRIDE = 1, 1
_BATCH, _FROM_PROBS = 1, 0
_SAMPLES, _IGNORE_INDEX, _MEAN = 1, -100, 1
_MOMENT_POWER = 1
_BINARY_OP, _SIDE = 0, 0
_ALPHA, _MINIMUM, _MAXIMUM = 1.0, 0.0, 0.0
_HAS_MINIMUM, _HAS_MAXIMUM = 0, 0
_SCALE, _OFFSET = 1.0, 0
_SHIFT = 0.0
_LEFT, _RIGHT, _PART = 1, 1, 1
_CI, _CO, _CIG, _COG = 1, 1, 1, 1
_IH, _IW, _OH, _OW, _KH, _KW = 1, 1, 1, 1, 1, 1
_SH, _SW, _DH, _DW, _H_SHIFT, _W_SHIFT = 1, 1, 1, 1, 1, 1
_TOP, _FACTOR = 0, 1
_AXIS_A, _AXIS_B, _BETWEEN, _TAIL = 1, 1, 1, 1
_STEP, _START, _NUMERATOR, _DENOMINATOR, _BIAS = 1, 0, 1, 1, 0
_INV_DENOMINATOR = 1.0
_INTERPOLATE = 1
_PAD = 0
_HAS_BIAS = 1
_X_SCALE, _Y_SCALE, _X_SHIFT, _Y_SHIFT, _X_LIMIT, _Y_LIMIT = 1.0, 1.0, 0.0, 0.0, 1.0, 1.0


def linear(x, weight, bias, y):
    i = inx.compute.index(y)
    total = 0.0
    for k in range(_K):
        total += x[(i // _M) * _K + k] * weight[(i % _M) * _K + k]
    y[i] = total + (bias[i % _M] if _HAS_BIAS else 0.0)


def pull_dx(weight, dy, dx):
    j = inx.compute.index(dx)
    total = 0.0
    for m in range(_M):
        total += weight[m * _K + j % _K] * dy[(j // _K) * _M + m]
    dx[j] = total


def pull_dw(x, dy, dw):
    j = inx.compute.index(dw)
    total = 0.0
    for b in range(_B):
        total += x[b * _K + j % _K] * dy[b * _M + j // _K]
    dw[j] = total


def pull_db(dy, db):
    j = inx.compute.index(db)
    total = 0.0
    for b in range(_B):
        total += dy[b * _M + j]
    db[j] = total


def conv2d(x, weight, bias, y):
    i = inx.compute.index(y)
    batch = i // (_CO * _OH * _OW)
    channel = (i // (_OH * _OW)) % _CO
    row = (i // _OW) % _OH
    column = i % _OW
    total = 0.0
    for k in range(_CIG * _KH * _KW):
        total += (
            x[((batch * _CI + (channel // _COG) * _CIG + k // (_KH * _KW)) * _IH
               + row * _SH + ((k // _KW) % _KH) * _DH) * _IW + column * _SW + (k % _KW) * _DW]
            * weight[channel * (_CIG * _KH * _KW) + k])
    y[i] = total + (bias[channel] if _HAS_BIAS else 0.0)


def conv2d_dx(weight, dy, dx):
    i = inx.compute.index(dx)
    batch = i // (_CI * _IH * _IW)
    channel = (i // (_IH * _IW)) % _CI
    row = (i // _IW) % _IH
    column = i % _IW
    total = 0.0
    # Complete output periods keep even inactive stride lanes in bounds.
    # On contributing lanes the modulo coordinates are exactly the inverse
    # stride mapping, rather than a clamped approximation of the derivative.
    for k in range(_COG * _KH * _KW):
        total += (
            weight[(((channel // _CIG) * _COG + k // (_KH * _KW)) * _CIG + channel % _CIG)
                   * (_KH * _KW) + k % (_KH * _KW)]
            * dy[((batch * _CO + (channel // _CIG) * _COG + k // (_KH * _KW)) * _OH
                  + ((row + _H_SHIFT - ((k // _KW) % _KH) * _DH) // _SH) % _OH) * _OW
                 + ((column + _W_SHIFT - (k % _KW) * _DW) // _SW) % _OW]
            if row >= ((k // _KW) % _KH) * _DH and column >= (k % _KW) * _DW
            and (row - ((k // _KW) % _KH) * _DH) % _SH == 0 and (column - (k % _KW) * _DW) % _SW == 0
            and (row - ((k // _KW) % _KH) * _DH) // _SH < _OH
            and (column - (k % _KW) * _DW) // _SW < _OW else 0.0)
    dx[i] = total


def conv2d_dw(x, dy, dw):
    i = inx.compute.index(dw)
    channel = i // (_CIG * _KH * _KW)
    input_channel = (channel // _COG) * _CIG + (i // (_KH * _KW)) % _CIG
    row = (i // _KW) % _KH
    column = i % _KW
    total = 0.0
    for k in range(_B * _OH * _OW):
        total += (
            x[(((k // (_OH * _OW)) * _CI + input_channel) * _IH
               + ((k // _OW) % _OH) * _SH + row * _DH) * _IW + (k % _OW) * _SW + column * _DW]
            * dy[((k // (_OH * _OW)) * _CO + channel) * (_OH * _OW) + k % (_OH * _OW)])
    dw[i] = total


def conv2d_db(dy, db):
    i = inx.compute.index(db)
    total = 0.0
    for k in range(_B * _OH * _OW):
        total += dy[((k // (_OH * _OW)) * _CO + i) * (_OH * _OW) + k % (_OH * _OW)]
    db[i] = total


def pad2d(x, output):
    i = inx.compute.index(output)
    row = (i // _OW) % _OH
    column = i % _OW
    value = 0.0
    if row >= _TOP and row < _TOP + _IH and column >= _LEFT and column < _LEFT + _IW:
        value = x[(i // (_OH * _OW)) * (_IH * _IW) + ((row + _IH - (_TOP % _IH)) % _IH) * _IW + (column + _IW - (_LEFT % _IW)) % _IW]
    output[i] = value


def pad2d_vjp(upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = upstream[(i // (_IH * _IW)) * (_OH * _OW) + ((i // _IW) % _IH + _TOP) * _OW + i % _IW + _LEFT]


def pixel_shuffle(x, output):
    i = inx.compute.index(output)
    batch = i // (_CO * _OH * _OW)
    channel = (i // (_OH * _OW)) % _CO
    row = (i // _OW) % _OH
    column = i % _OW
    output[i] = x[((batch * _CI + channel * _FACTOR * _FACTOR + (row % _FACTOR) * _FACTOR + column % _FACTOR) * _IH + row // _FACTOR) * _IW + column // _FACTOR]


def pixel_shuffle_vjp(upstream, downstream):
    i = inx.compute.index(downstream)
    batch = i // (_CI * _IH * _IW)
    channel = (i // (_IH * _IW)) % _CI
    row = (i // _IW) % _IH
    column = i % _IW
    downstream[i] = upstream[((batch * _CO + channel // (_FACTOR * _FACTOR)) * _OH + row * _FACTOR + (channel // _FACTOR) % _FACTOR) * _OW + column * _FACTOR + channel % _FACTOR]


def sigmoid(x, y):
    i = inx.compute.index(x)
    v = x[i]
    if v >= 0.0:
        z = math.exp(-v)
        y[i] = 1.0 / (1.0 + z)
    else:
        z = math.exp(v)
        y[i] = z / (1.0 + z)


def sigmoid_vjp(x, upstream, downstream):
    # CPU analytic primitive recipe, not a second automatic differentiation system.
    i = inx.compute.index(x)
    v = x[i]
    if v >= 0.0:
        z = math.exp(-v)
    else:
        z = math.exp(v)
    denominator = 1.0 + z
    downstream[i] = upstream[i] * z / (denominator * denominator)


def relu(x, y):
    i = inx.compute.index(y)
    if x[i] > 0.0:
        y[i] = x[i]
    else:
        y[i] = 0.0


def relu_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    if x[i] > 0.0:
        downstream[i] = upstream[i]
    else:
        downstream[i] = 0.0


def elu(x, y):
    i = inx.compute.index(y)
    value = x[i]
    if value != value:
        y[i] = value
    elif value > 0.0:
        y[i] = value
    elif value > -.01:
        # exp(x)-1 cancels around zero; this local expansion preserves tiny
        # negative activations without a second backend-specific math path.
        y[i] = _ALPHA * value * (1.0 + value * (.5 + value * (0.16666666666666666 + value * 0.041666666666666664)))
    else:
        y[i] = _ALPHA * (math.exp(value) - 1.0)


def elu_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    # Select the input NaN independently of GLSL Exp's special-value result;
    # form the bounded local derivative before multiplying the seed.
    downstream[i] = x[i] if x[i] != x[i] else upstream[i] if x[i] > 0.0 else upstream[i] * (_ALPHA * math.exp(x[i]))


def clamp(x, y):
    i = inx.compute.index(y)
    value = _MINIMUM if _HAS_MINIMUM and x[i] < _MINIMUM else x[i]
    y[i] = _MAXIMUM if _HAS_MAXIMUM and value > _MAXIMUM else value


def clamp_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    active = (not _HAS_MINIMUM or x[i] >= _MINIMUM) and (not _HAS_MAXIMUM or x[i] <= _MAXIMUM)
    downstream[i] = upstream[i] if active else 0.0


def tanh(x, y):
    i = inx.compute.index(y)
    y[i] = math.tanh(x[i])


def tanh_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    value = math.tanh(x[i])
    downstream[i] = upstream[i] * (1.0 - value * value)


def sine(x, y):
    i = inx.compute.index(y)
    y[i] = math.sin(x[i])


def sine_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = upstream[i] * math.cos(x[i])


def cosine(x, y):
    i = inx.compute.index(y)
    y[i] = math.cos(x[i])


def cosine_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = -upstream[i] * math.sin(x[i])


def square(x, y):
    i = inx.compute.index(y)
    y[i] = x[i] * x[i]


def square_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = 2.0 * x[i] * upstream[i]


def exponential(x, y):
    i = inx.compute.index(y)
    y[i] = math.exp(x[i])


def exponential_vjp(y, upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = upstream[i] * y[i]


def logarithm(x, y):
    i = inx.compute.index(y)
    y[i] = math.log(x[i])


def logarithm_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = upstream[i] / x[i]


def square_root(x, y):
    i = inx.compute.index(y)
    y[i] = math.sqrt(x[i])


def square_root_vjp(y, upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = upstream[i] * .5 / y[i]


def silu(x, y):
    i = inx.compute.index(y)
    value = x[i]
    if value >= 0.0:
        z = math.exp(-value)
        y[i] = value / (1.0 + z)
    else:
        z = math.exp(value)
        y[i] = value * z / (1.0 + z)


def silu_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    value = x[i]
    if value >= 0.0:
        z = math.exp(-value)
        s = 1.0 / (1.0 + z)
    else:
        z = math.exp(value)
        s = z / (1.0 + z)
    downstream[i] = upstream[i] * s * (1.0 + value * (1.0 - s))


def softmax(x, y):
    i = inx.compute.index(y)
    base = (i // (_WIDTH * _STRIDE)) * (_WIDTH * _STRIDE) + i % _STRIDE
    maximum = x[base]
    for j in range(_WIDTH):
        maximum = max(maximum, x[base + j * _STRIDE])
    total = 0.0
    for j in range(_WIDTH):
        total += math.exp(x[base + j * _STRIDE] - maximum)
    y[i] = math.exp(x[i] - maximum) / total


def softmax_vjp(y, upstream, downstream):
    i = inx.compute.index(downstream)
    base = (i // (_WIDTH * _STRIDE)) * (_WIDTH * _STRIDE) + i % _STRIDE
    total = 0.0
    for j in range(_WIDTH):
        total += upstream[base + j * _STRIDE] * y[base + j * _STRIDE]
    downstream[i] = y[i] * (upstream[i] - total)


def log_softmax(x, y):
    i = inx.compute.index(y)
    base = (i // (_WIDTH * _STRIDE)) * (_WIDTH * _STRIDE) + i % _STRIDE
    maximum = x[base]
    for j in range(_WIDTH):
        maximum = max(maximum, x[base + j * _STRIDE])
    total = 0.0
    for j in range(_WIDTH):
        total += math.exp(x[base + j * _STRIDE] - maximum)
    y[i] = x[i] - maximum - math.log(total)


def log_softmax_vjp(y, upstream, downstream):
    i = inx.compute.index(downstream)
    base = (i // (_WIDTH * _STRIDE)) * (_WIDTH * _STRIDE) + i % _STRIDE
    total = 0.0
    for j in range(_WIDTH):
        total += upstream[base + j * _STRIDE]
    downstream[i] = upstream[i] - math.exp(y[i]) * total


def probability_log(x, y):
    i = inx.compute.index(y)
    y[i] = math.log(min(0.9999998807907104, max(0.00000011920928955078125, x[i])))


def random_uniform(output, seed0, seed1, counter0, counter1):
    # Threefry2x32-20 (Salmon et al., SC11). Canonical signed words keep
    # Python/LLVM and SPIR-V's 32-bit wrap arithmetic on the same bit sequence.
    # Each row owns a counter; scheduling and continuation never advance it.
    i = inx.compute.index(output)
    low = (counter0 & 65535) + (i & 65535)
    high = ((counter0 >> 16) & 65535) + (i >> 16) + (low >> 16)
    x0 = ((high & 65535) << 16) + (low & 65535) + seed0
    x1 = counter1 + (high >> 16) + seed1
    k2 = 466688986 ^ seed0 ^ seed1
    for j in range(20):
        rotation = (13 if j % 8 == 0 else 15 if j % 8 == 1 else 26 if j % 8 == 2 else 6 if j % 8 == 3
                    else 17 if j % 8 == 4 else 29 if j % 8 == 5 else 16 if j % 8 == 6 else 24)
        added = x0 + x1
        x0 = (added & 2147483647) + ((added >> 31) & 1) * -2147483648
        mixed = ((x1 << rotation) | ((x1 >> (32 - rotation)) & ((1 << rotation) - 1))) ^ x0
        x1 = (mixed & 2147483647) + ((mixed >> 31) & 1) * -2147483648
        injection = (j + 1) // 4
        key0 = seed0 if injection % 3 == 0 else seed1 if injection % 3 == 1 else k2
        key1 = seed1 if injection % 3 == 0 else k2 if injection % 3 == 1 else seed0
        x0 = x0 + key0 if (j + 1) % 4 == 0 else x0
        x1 = x1 + key1 + injection if (j + 1) % 4 == 0 else x1
    output[i] = (float((x0 >> 8) & 16777215) * 0.000000059604644775390625) * _SCALE + _SHIFT


def categorical_sample(probs, uniform, output):
    i = inx.compute.index(output)
    total = 0.0
    for j in range(_WIDTH):
        total += probs[(i % _BATCH) * _WIDTH + j]
    threshold = uniform[i] * total
    cumulative = 0.0
    selected = 0
    for j in range(_WIDTH):
        selected = j if probs[(i % _BATCH) * _WIDTH + j] > 0.0 and cumulative <= threshold else selected
        cumulative = cumulative + probs[(i % _BATCH) * _WIDTH + j]
    output[i] = selected


def probability_log_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = (upstream[i] / x[i] if x[i] >= 0.00000011920928955078125 and x[i] <= 0.9999998807907104 else 0.0)


def categorical_select(logits, action, output):
    i = inx.compute.index(output)
    selected = -math.inf
    for j in range(_WIDTH):
        selected = logits[i * _WIDTH + j] if action[i] == j else selected
    output[i] = selected


def categorical_select_vjp(action, upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = upstream[i // _WIDTH] if action[i // _WIDTH] == i % _WIDTH else 0.0


def categorical_entropy(x, output):
    i = inx.compute.index(output)
    total = 0.0
    for j in range(_WIDTH):
        value = x[i * _WIDTH + j]
        probability = value if _FROM_PROBS else math.exp(value)
        logarithm = (math.log(min(0.9999998807907104, max(0.00000011920928955078125, value)))
                     if _FROM_PROBS else max(-3.4028234663852886e38, value))
        total += -logarithm * probability
    output[i] = total


def categorical_entropy_vjp(x, upstream, downstream):
    i = inx.compute.index(downstream)
    value = x[i]
    derivative = (-math.log(min(0.9999998807907104, max(0.00000011920928955078125, value)))
                  - (1.0 if value >= 0.00000011920928955078125 and value <= 0.9999998807907104 else 0.0)
                  if _FROM_PROBS else -math.exp(value) * (max(-3.4028234663852886e38, value)
                      + (1.0 if value >= -3.4028234663852886e38 else 0.0)))
    downstream[i] = derivative * upstream[i // _WIDTH]


def mse_mean(prediction, target, loss):
    i = inx.compute.index(loss)
    total = 0.0
    for k in range(_COUNT):
        total += (prediction[k] - target[k]) * (prediction[k] - target[k]) * _INV_N
    loss[i] = total


def mse_vjp(prediction, target, gradient):
    i = inx.compute.index(prediction)
    gradient[i] = 2.0 * (prediction[i] - target[i]) * _INV_N


def weighted_log_mean(log_probability, target, loss):
    i = inx.compute.index(loss)
    total = 0.0
    for k in range(_COUNT):
        total += -log_probability[k] * target[k] * _INV_N
    loss[i] = total


def indexed_log_rows(log_probability, target, rows):
    i = inx.compute.index(rows)
    base = (i // _STRIDE) * _WIDTH * _STRIDE + i % _STRIDE
    selected = math.nan
    for k in range(_WIDTH):
        selected = log_probability[base + k * _STRIDE] if target[i] == k else selected
    rows[i] = 0.0 if target[i] == _IGNORE_INDEX else -selected


def scaled_class_loss(rows, scale, loss):
    i = inx.compute.index(loss)
    total = 0.0
    for sample in range(_SAMPLES):
        total += rows[sample]
    loss[i] = math.nan if _MEAN and scale[0] == 0.0 else total * scale[0]


def class_normalization(target, scale):
    i = inx.compute.index(scale)
    count = 0.0
    for sample in range(_SAMPLES):
        count += 1.0 if target[sample] != _IGNORE_INDEX else 0.0
    scale[i] = 1.0 / count if count > 0.0 else 0.0


def indexed_cross_entropy_vjp(prediction, target, scale, gradient):
    i = inx.compute.index(gradient)
    sample = (i // (_WIDTH * _STRIDE)) * _STRIDE + i % _STRIDE
    label = target[sample]
    base = (sample // _STRIDE) * _WIDTH * _STRIDE + sample % _STRIDE
    maximum = prediction[base]
    for k in range(_WIDTH):
        maximum = max(maximum, prediction[base + k * _STRIDE])
    total = 0.0
    for k in range(_WIDTH):
        total += math.exp(prediction[base + k * _STRIDE] - maximum)
    value = (math.exp(prediction[i] - maximum) / total
             - (1.0 if label == (i // _STRIDE) % _WIDTH else 0.0)) * scale[0]
    selected = value if label >= 0 and label < _WIDTH else math.nan
    gradient[i] = 0.0 if label == _IGNORE_INDEX else selected


def cross_entropy_vjp(prediction, target, gradient):
    i = inx.compute.index(gradient)
    base = (i // (_WIDTH * _STRIDE)) * _WIDTH * _STRIDE + i % _STRIDE
    maximum = prediction[base]
    for k in range(_WIDTH):
        maximum = max(maximum, prediction[base + k * _STRIDE])
    total = 0.0
    for k in range(_WIDTH):
        total += math.exp(prediction[base + k * _STRIDE] - maximum)
    target_sum = 0.0
    for k in range(_WIDTH):
        target_sum += target[base + k * _STRIDE]
    gradient[i] = (math.exp(prediction[i] - maximum) / total * target_sum - target[i]) * _INV_N


def sgd(current, gradient, updated, learning_rate):
    i = inx.compute.index(current)
    updated[i] = current[i] - learning_rate * gradient[i]


def adam_moment(current, gradient, previous, updated, beta, complement, weight_decay, reset):
    i = inx.compute.index(updated)
    value = gradient[i] + weight_decay * current[i]
    if _MOMENT_POWER == 2:
        value = value * value
    old = 0.0
    if reset == 0.0:
        old = previous[i]
    updated[i] = beta * old + complement * value


def adam_update(current, first, second, updated, learning_rate, correction1, correction2, epsilon):
    i = inx.compute.index(updated)
    mean = first[i] * correction1
    variance = second[i] * correction2
    updated[i] = current[i] - learning_rate * mean / (math.sqrt(variance) + epsilon)


def add(left, right, output):
    i = inx.compute.index(output)
    output[i] = left[i] + right[i]


def binary(left, right, output):
    i = inx.compute.index(output)
    if _BINARY_OP == 0:
        output[i] = left[i] - right[i]
    elif _BINARY_OP == 1:
        output[i] = left[i] * right[i]
    elif _BINARY_OP == 2:
        output[i] = left[i] / right[i]
    else:
        # Explicit NaN propagation and tie selection on both CPU and Vulkan.
        a = left[i]
        b = right[i]
        output[i] = a if a != a else b if b != b else (a if a <= b else b) if _BINARY_OP == 3 else (a if a >= b else b)


def binary_vjp(left, right, upstream, downstream):
    i = inx.compute.index(downstream)
    if _BINARY_OP == 0:
        if _SIDE == 0:
            downstream[i] = upstream[i]
        else:
            downstream[i] = -upstream[i]
    elif _BINARY_OP == 1:
        if _SIDE == 0:
            downstream[i] = upstream[i] * right[i]
        else:
            downstream[i] = upstream[i] * left[i]
    elif _BINARY_OP == 2:
        if _SIDE == 0:
            downstream[i] = upstream[i] / right[i]
        else:
            downstream[i] = -upstream[i] * left[i] / (right[i] * right[i])
    else:
        a = left[i]
        b = right[i]
        inactive = (a > b if _SIDE == 0 else a < b) if _BINARY_OP == 3 else (a < b if _SIDE == 0 else a > b)
        downstream[i] = 0.0 if inactive else upstream[i] * .5 if a == b else upstream[i]


def affine(x, output):
    i = inx.compute.index(output)
    output[i] = x[i] * _SCALE + _SHIFT


def affine_vjp(upstream, downstream):
    i = inx.compute.index(downstream)
    downstream[i] = upstream[i] * _SCALE


def copy_values(x, output):
    i = inx.compute.index(output)
    output[i] = x[i]


def maximum_tiles(x, output):
    # Bounded rows: scale before squaring so large finite gradients do not
    # overflow merely because their squares exceed float32.
    i = inx.compute.index(output)
    maximum = 0.0
    for k in range(_WIDTH):
        value = x[(i * _WIDTH + k) % _COUNT] if i * _WIDTH + k < _COUNT else 0.0
        magnitude = value if value >= 0.0 else -value
        maximum = magnitude if magnitude != magnitude or magnitude > maximum else maximum
    output[i] = maximum


def norm_tiles(x, maxima, output):
    i = inx.compute.index(output)
    maximum = maxima[i]
    total = 0.0
    for k in range(_WIDTH):
        value = x[(i * _WIDTH + k) % _COUNT] if i * _WIDTH + k < _COUNT else 0.0
        scaled = value / maximum if maximum > 0.0 and maximum <= _MAXIMUM else 0.0
        total += scaled * scaled
    output[i] = maximum * math.sqrt(total) if maximum <= _MAXIMUM else maximum


def norm_merge(left, right, output):
    i = inx.compute.index(output)
    a = left[i]
    b = right[i]
    maximum = a if a != a or a >= b else b
    x = a / maximum if maximum > 0.0 and maximum <= _MAXIMUM else 0.0
    y = b / maximum if maximum > 0.0 and maximum <= _MAXIMUM else 0.0
    output[i] = maximum * math.sqrt(x * x + y * y) if maximum <= _MAXIMUM else maximum


def clip_gradient(x, norm, output, max_norm):
    i = inx.compute.index(output)
    coefficient = max_norm / (norm[0] + 1e-6)
    coefficient = 1.0 if coefficient > 1.0 else coefficient
    output[i] = x[i] * coefficient


def fill_values(output):
    i = inx.compute.index(output)
    output[i] = _SHIFT


def expand_axis(x, output):
    i = inx.compute.index(output)
    output[i] = x[(i // (_WIDTH * _STRIDE)) * _STRIDE + i % _STRIDE]


def expand_axis_vjp(upstream, downstream):
    i = inx.compute.index(downstream)
    base = (i // _STRIDE) * _WIDTH * _STRIDE + i % _STRIDE
    total = 0.0
    for j in range(_WIDTH):
        total += upstream[base + j * _STRIDE]
    downstream[i] = total


def reduce_axis(x, output):
    i = inx.compute.index(output)
    base = (i // _STRIDE) * _WIDTH * _STRIDE + i % _STRIDE
    total = 0.0
    for k in range(_WIDTH):
        total += x[base + k * _STRIDE]
    output[i] = total


def argmax_axis(x, output):
    i = inx.compute.index(output)
    base = (i // _STRIDE) * _WIDTH * _STRIDE + i % _STRIDE
    maximum = x[base]
    position = 0
    for k in range(_WIDTH):
        value = x[base + k * _STRIDE]
        selected = value > maximum or (value != value and maximum == maximum)
        maximum = value if selected else maximum
        position = k if selected else position
    output[i] = position


def transpose(x, output):
    i = inx.compute.index(output)
    outer = i // (_AXIS_B * _BETWEEN * _AXIS_A * _TAIL)
    b = (i // (_BETWEEN * _AXIS_A * _TAIL)) % _AXIS_B
    middle = (i // (_AXIS_A * _TAIL)) % _BETWEEN
    a = (i // _TAIL) % _AXIS_A
    output[i] = x[(((outer * _AXIS_A + a) * _BETWEEN + middle) * _AXIS_B + b) * _TAIL + i % _TAIL]


def concatenate(left, right, output):
    i = inx.compute.index(output)
    outer = i // (_WIDTH * _STRIDE)
    axis = (i // _STRIDE) % _WIDTH
    offset = i % _STRIDE
    output[i] = (left[outer * _LEFT * _STRIDE + (axis % _LEFT) * _STRIDE + offset] if axis < _LEFT else
                 right[outer * _RIGHT * _STRIDE + ((axis + _RIGHT - (_LEFT % _RIGHT)) % _RIGHT) * _STRIDE + offset])


def concatenate_vjp(upstream, downstream):
    i = inx.compute.index(downstream)
    source = (i // (_PART * _STRIDE)) * _WIDTH * _STRIDE + ((i // _STRIDE) % _PART + _OFFSET) * _STRIDE + i % _STRIDE
    downstream[i] = upstream[source]


def zero(output):
    i = inx.compute.index(output)
    output[i] = 0.0


def slice_axis(x, output):
    i = inx.compute.index(output)
    source = ((i // (_COUNT * _STRIDE)) * _WIDTH * _STRIDE
              + (_START + ((i // _STRIDE) % _COUNT) * _STEP) * _STRIDE + i % _STRIDE)
    output[i] = x[source]


def slice_axis_vjp(upstream, downstream):
    i = inx.compute.index(downstream)
    coordinate = (i // _STRIDE) % _WIDTH
    distance = max(coordinate, _START) - _START
    source = ((i // (_WIDTH * _STRIDE)) * _COUNT * _STRIDE
              + min(_COUNT - 1, distance // _STEP) * _STRIDE + i % _STRIDE)
    downstream[i] = (upstream[source] if coordinate >= _START and distance % _STEP == 0
                     and distance // _STEP < _COUNT else 0.0)


def resample_axis(x, output):
    i = inx.compute.index(output)
    coordinate = (i // _STRIDE) % _COUNT
    # Integer rational coordinates are identical on both devices. The clamp
    # is the sampling boundary rule, and keeps every gather address in range.
    numerator = min((_WIDTH - 1) * _DENOMINATOR,
                    max(_BIAS, coordinate * _NUMERATOR + _NUMERATOR // 2) - _BIAS)
    left = numerator // _DENOMINATOR
    right = min(_WIDTH - 1, left + 1)
    fraction = float(numerator % _DENOMINATOR) * _INV_DENOMINATOR
    base = (i // (_COUNT * _STRIDE)) * _WIDTH * _STRIDE + i % _STRIDE
    value = x[base + left * _STRIDE]
    if _INTERPOLATE:
        value = value * (1.0 - fraction) + x[base + right * _STRIDE] * fraction
    output[i] = value


def resample_axis_vjp(upstream, downstream):
    i = inx.compute.index(downstream)
    coordinate = (i // _STRIDE) % _WIDTH
    base = (i // (_WIDTH * _STRIDE)) * _COUNT * _STRIDE + i % _STRIDE
    total = 0.0
    for k in range(_COUNT):
        numerator = min((_WIDTH - 1) * _DENOMINATOR,
                        max(_BIAS, k * _NUMERATOR + _NUMERATOR // 2) - _BIAS)
        left = numerator // _DENOMINATOR
        right = min(_WIDTH - 1, left + 1)
        fraction = float(numerator % _DENOMINATOR) * _INV_DENOMINATOR
        total += (upstream[base + k * _STRIDE] * ((1.0 - fraction if coordinate == left else 0.0)
                    + (fraction if coordinate == right else 0.0))
                  if coordinate == left or (_INTERPOLATE and coordinate == right) else 0.0)
    downstream[i] = total


def grid_sample(x, grid, output):
    i = inx.compute.index(output)
    batch = i // (_CI * _OH * _OW)
    channel = (i // (_OH * _OW)) % _CI
    point = (batch * _OH * _OW + i % (_OH * _OW)) * 2
    gx = grid[point]
    gy = grid[point + 1]
    px = (gx if gx == gx else -1.0) * _X_SCALE + _X_SHIFT
    py = (gy if gy == gy else -1.0) * _Y_SCALE + _Y_SHIFT
    left = int(min(float(_IW - 1), max(0.0, px if px == px else 0.0)))
    top = int(min(float(_IH - 1), max(0.0, py if py == py else 0.0)))
    right = min(_IW - 1, left + 1)
    bottom = min(_IH - 1, top + 1)
    fx = min(_X_LIMIT, max(0.0, px if px == px else 0.0)) - float(left)
    fy = min(_Y_LIMIT, max(0.0, py if py == py else 0.0)) - float(top)
    nearest_x = min(_IW - 1, left + (1 if fx > .5 or (fx == .5 and (left + _PAD) % 2 == 1) else 0))
    nearest_y = min(_IH - 1, top + (1 if fy > .5 or (fy == .5 and (top + _PAD) % 2 == 1) else 0))
    base = (batch * _CI + channel) * _IH * _IW
    value = x[base + nearest_y * _IW + nearest_x]
    if _INTERPOLATE:
        value = ((x[base + top * _IW + left] * (1.0 - fx) + x[base + top * _IW + right] * fx) * (1.0 - fy)
                 + (x[base + bottom * _IW + left] * (1.0 - fx) + x[base + bottom * _IW + right] * fx) * fy)
    output[i] = value


def grid_sample_dx(grid, upstream, downstream):
    i = inx.compute.index(downstream)
    batch = i // (_CI * _IH * _IW)
    channel = (i // (_IH * _IW)) % _CI
    row = (i // _IW) % _IH
    column = i % _IW
    total = 0.0
    for k in range(_OH * _OW):
        gx = grid[(batch * _OH * _OW + k) * 2]
        gy = grid[(batch * _OH * _OW + k) * 2 + 1]
        px = (gx if gx == gx else -1.0) * _X_SCALE + _X_SHIFT
        py = (gy if gy == gy else -1.0) * _Y_SCALE + _Y_SHIFT
        left = int(min(float(_IW - 1), max(0.0, px if px == px else 0.0)))
        top = int(min(float(_IH - 1), max(0.0, py if py == py else 0.0)))
        right = min(_IW - 1, left + 1)
        bottom = min(_IH - 1, top + 1)
        fx = min(_X_LIMIT, max(0.0, px if px == px else 0.0)) - float(left)
        fy = min(_Y_LIMIT, max(0.0, py if py == py else 0.0)) - float(top)
        nearest_x = min(_IW - 1, left + (1 if fx > .5 or (fx == .5 and (left + _PAD) % 2 == 1) else 0))
        nearest_y = min(_IH - 1, top + (1 if fy > .5 or (fy == .5 and (top + _PAD) % 2 == 1) else 0))
        weight_x = (1.0 - fx if column == left else 0.0) + (fx if column == right else 0.0)
        weight_y = (1.0 - fy if row == top else 0.0) + (fy if row == bottom else 0.0)
        weight = weight_x * weight_y if _INTERPOLATE else (1.0 if column == nearest_x and row == nearest_y else 0.0)
        total += (upstream[(batch * _CI + channel) * _OH * _OW + k] * weight if weight != 0.0 else 0.0)
    downstream[i] = total


def grid_sample_dgrid(x, grid, upstream, downstream):
    i = inx.compute.index(downstream)
    batch = i // (_OH * _OW * 2)
    point = i // 2
    gx = grid[point * 2]
    gy = grid[point * 2 + 1]
    px = (gx if gx == gx else -1.0) * _X_SCALE + _X_SHIFT
    py = (gy if gy == gy else -1.0) * _Y_SCALE + _Y_SHIFT
    left = int(min(float(_IW - 1), max(0.0, px if px == px else 0.0)))
    top = int(min(float(_IH - 1), max(0.0, py if py == py else 0.0)))
    right = min(_IW - 1, left + 1)
    bottom = min(_IH - 1, top + 1)
    fx = min(_X_LIMIT, max(0.0, px if px == px else 0.0)) - float(left)
    fy = min(_Y_LIMIT, max(0.0, py if py == py else 0.0)) - float(top)
    active = (((px >= 0.0 if _PAD else px > 0.0) and px < _X_LIMIT) if i % 2 == 0 else
              ((py >= 0.0 if _PAD else py > 0.0) and py < _Y_LIMIT))
    scale = _X_SCALE if i % 2 == 0 else _Y_SCALE
    total = 0.0
    for k in range(_CI):
        base = (batch * _CI + k) * _IH * _IW
        a = x[base + top * _IW + left]
        b = x[base + top * _IW + right]
        c = x[base + bottom * _IW + left]
        d = x[base + bottom * _IW + right]
        derivative = ((b - a) * (1.0 - fy) + (d - c) * fy) if i % 2 == 0 else ((c - a) * (1.0 - fx) + (d - b) * fx)
        total += (upstream[(batch * _CI + k) * _OH * _OW + point % (_OH * _OW)] * derivative * scale
                  if _INTERPOLATE and active else 0.0)
    downstream[i] = total


def prepare(function, params, *, own_deferred_samples=False, source=None, **facts):
    """Add a model build recipe to the common numerical program contract."""
    if source is None:
        from ._programs import recipe
        program = _prepare_work(function, params, **facts)
        program._nn_recipe = recipe(function, params, facts)
    else:
        program = source.prepare(function, params, **facts)
    if isinstance(program, inx.compute.Kernel):
        _prepare_gpu_family(program, params, own_deferred_samples=own_deferred_samples)
    return program


def run(program, *params, reverse=False):
    if isinstance(program, inx.compute.Kernel):
        arguments, gradients = gpu_arguments(params, reverse=reverse)
        inx.compute.launch(program, arguments,
                           autodiff_mode="reverse" if reverse else "none", gradients=gradients)
    else:
        program(*params, begin=0, end=len(params[program.domain_parameter]))


def prepare_reverse(program, params):
    _prepare_gpu(program, params, own_deferred_samples=True,
                 autodiff_mode="reverse", gradient_parameters=(0, 1))


@contextmanager
def stage(device, programs):
    """Synchronous offline boundary over the original recording/FIFO, no new executor."""
    if device == "cpu":
        yield
        return
    from ._tensor import _check_options
    _check_options(None, device)
    try:
        with inx.compute.recording():
            yield
    except Exception as error:
        for declaration in programs:
            for executable in declaration._executables.values():
                stream = inx.compute._command_streams().get(int(executable.host.identity))
                if stream is not None:
                    inx.compute._fail_command_stream(stream, error)
        raise
    finally:
        inx.compute._finish_kernels(programs)


def release_bindings(values):
    """Completed run values may outlive a stage, but its LRU must not own them."""
    for value in values:
        if value.device == "gpu" and not value.closed:
            inx.compute._discard_buffer_launches(value)
