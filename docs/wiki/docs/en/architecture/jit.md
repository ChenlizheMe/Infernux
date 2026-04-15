---
category: Architecture
tags: ["jit", "performance", "python", "numba", "parallel"]
date: "2026-04-16"
---

# Deep Dive: JIT-Accelerated Scripting in Infernux

Infernux is a game engine whose entire gameplay layer is written in Python. That is a deliberate architectural choice — Python gives us hot reload, a massive ecosystem, and a low barrier to entry — but it also means that every per-frame update loop is, by default, interpreted. When you need to touch 100,000 transforms each frame, interpretation is not fast enough. This page walks through how the engine's JIT subsystem solves that problem, what design decisions it makes under the hood, and where the work is heading next.

For the formal evaluation and benchmark tables, see the full technical report: [*Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting* (arXiv:2604.10263)](https://arxiv.org/pdf/2604.10263).

---

## 1 &ensp; The problem: death by a thousand boundary crossings

Consider what happens when a naive Python script updates every entity's position:

```python
for obj in scene.objects:
    pos = obj.transform.position          # Python → C++ → Python
    pos.y += math.sin(time + pos.x)       # pure Python math
    obj.transform.position = pos          # Python → C++ → Python
```

Each iteration crosses the pybind11 boundary twice: once to read, once to write. Every crossing acquires the GIL, converts between C++ and Python types, and dispatches through pybind11's type-caster machinery. At 10 objects this is invisible. At 100,000 objects it burns more time in boundary overhead than in the actual math.

The engine's performance story therefore has two chapters, and they must be read in order.

---

## 2 &ensp; Chapter one: the batch data bridge

Before any JIT compilation can help, we need to eliminate per-object boundary crossings. The batch bridge does this by moving engine state through contiguous NumPy arrays in a single call:

```python
positions = batch_read(targets, "position")   # one crossing, returns (N,3) array
jit_wave_kernel(positions, time_value, count)  # pure array math, no crossings
batch_write(targets, positions, "position")    # one crossing, writes back
```

Internally, `batch_read` dispatches through a tiered fast path:

1. **C++ Transform path** — for `position`, `rotation`, `scale` and other Transform properties, the call goes directly to `TransformECSStore::GatherVec3` / `GatherQuat` in C++, which memcpys from the SoA store into a pre-allocated NumPy buffer with the GIL released.
2. **ComponentDataStore path** — for user-defined `InxComponent` fields backed by the typed CDS (Float64, Int64, Bool, Vec2–Vec4), C++ scatter/gather routines handle the bulk copy.
3. **Python fallback** — for anything else, a `getattr` loop collects values one by one. This is the slow path, but it is always correct.

A `create_batch_handle()` call can further cache the underlying C++ `Transform*` pointers so that repeated frames skip the O(N) pybind11 cast of the target list.

The result: boundary-crossing cost becomes O(1) per frame per property instead of O(N). The data that arrives in the kernel is a dense, cache-friendly NumPy ndarray — exactly the input shape that LLVM vectorizers love.

---

## 3 &ensp; Chapter two: compiling the kernel

With boundary crossings eliminated, the remaining cost is the Python interpreter evaluating the inner loop. A sine-and-add loop over 100k floats in CPython takes milliseconds. The same loop compiled to native SIMD code takes microseconds. That is where JIT comes in.

### 3.1 &ensp; The `njit` decorator

The engine exposes a single entry point: `from Infernux.jit import njit`. Depending on argument style, it does different things:

| Usage | Behavior |
|---|---|
| `@njit` | Compile one serial variant via Numba, cache by bytecode key. |
| `@njit(cache=True, auto_parallel=True)` | Compile **two** variants (serial + parallel), AST-rewrite the parallel one, and build an auto-selecting dispatcher. |

When Numba is not installed, `njit` degrades to a transparent pass-through: the original Python function runs as-is, and a runtime warning is emitted so the performance loss is visible rather than silent.

### 3.2 &ensp; Bytecode-keyed compilation cache

Traditional Numba caching keys on `(module, function_name)`. That breaks in a game engine where scripts are reloaded dozens of times per session — the module identity changes, but the code may not have.

Infernux keys the cache on:

```
(co_filename, func_name, sha256(co_code)[:16], kwargs_tag)
```

The `sha256` of the raw bytecode means that if the function body has not changed, the cached machine code is reused even after a hot reload cycle destroys and recreates the module object. If the bytecode *has* changed, only that function is recompiled; every other kernel keeps its cached variant.

This is stored in an in-process `dict` (`_compiled_cache`). There is no filesystem lock contention, no stale `.nbi` files to manually clean, and no edge cases from Numba's built-in file cache interacting with Nuitka's frozen importers.

### 3.3 &ensp; Graceful degradation and self-healing

The JIT subsystem anticipates three runtime environments:

- **Full Numba available** — compile and cache normally.
- **Numba missing** — fall back to plain Python. `ensure_jit_runtime()` can optionally auto-install Numba via pip (a self-healing path: import fails → install → reimport → succeed).
- **Nuitka-compiled distribution** — Numba's file-backed cache is disabled (because `co_filename` no longer points to a real `.py` file), but in-memory compilation still works. This gives packaged builds JIT performance without requiring the end user to have a compiler toolchain.

Every compiled function object carries a `.py` attribute pointing to the original Python callable, so introspection and debugging always have a fallback.

---

## 4 &ensp; Auto-parallelization: from `range` to `prange`

This is the most engine-specific part of the JIT subsystem, and it is worth understanding in detail.

### 4.1 &ensp; The idea

Numba supports `prange` — a parallel range that distributes loop iterations across threads. But asking every gameplay programmer to manually write `prange`, verify there are no data races, and handle the serial fallback when parallelism is not profitable is unrealistic. The engine should do this automatically when it is safe.

### 4.2 &ensp; The AST rewriter

When `auto_parallel=True` is set, the engine takes the source of the decorated function, parses it into a Python AST, and runs `_AutoParallelRangeTransformer` — a `ast.NodeTransformer` subclass that visits every `for` node and asks:

1. **Is this a counted loop?** — The iterator must be a direct `range(...)` call. Iterator-style loops (`for x in arr`) and `while` loops are never touched.

2. **Is the body free of unsupported control flow?** — `return`, `break`, `yield`, and `try/except` inside the loop body all abort the rewrite. These constructs either have unclear parallel semantics or are not supported by Numba's parallel backend.

3. **Is the write pattern safe?** — The rewriter checks for two patterns:
   - **Supported reductions**: augmented assignments like `total += arr[i]`, `product *= arr[i]`. These are recognized as commutative reductions that Numba can partition across threads.
   - **Embarrassingly parallel indexed stores**: array writes of the form `out[i] = expr` where the index is the loop variable. Because each iteration writes to a unique index, there are no data races.

   If neither pattern is found, the loop is left serial.

4. **Rewrite**: If all checks pass, `range` is replaced with `prange` in the AST, the modified source is compiled via `exec`, and the resulting function object is handed to Numba with `parallel=True`.

```
Original:    for i in range(n):  out[i] = arr[i] * scale
                                      ↓ AST rewrite
Rewritten:   for i in prange(n): out[i] = arr[i] * scale
```

### 4.3 &ensp; Why syntactic analysis is enough

A real compiler would need full alias analysis to prove that `out[i]` and `arr[j]` don't alias for `i ≠ j`. The engine's rewriter deliberately skips that and uses a syntactic heuristic: if the write target is `<array>[<loop_var>]`, it is assumed disjoint.

This is sound for the dominant game-engine use case: SoA property updates indexed by entity ID. `positions[i]`, `velocities[i]`, `scales[i]` — every kernel the engine ships looks like this. The heuristic rejects anything more complex, which is the conservative-safe direction to err in: you might miss a parallelization opportunity, but you never introduce a data race.

### 4.4 &ensp; Prebuilt sidecars for distribution

For packaged builds, live AST rewriting at startup adds latency. The build system can pre-generate `.autop.pyc` sidecar files via `build_auto_parallel_sidecar_source()`. At runtime, `_load_prebuilt_auto_parallel_variant()` checks for sidecars first and only falls back to live rewriting if none are found.

---

## 5 &ensp; The dual-variant dispatcher

Parallel execution is not always faster. For small N, thread-pool overhead dominates. For large N, parallel throughput wins. The engine handles this with a dual-variant dispatch architecture.

### 5.1 &ensp; Compilation

When `auto_parallel=True`, the engine compiles **two** Numba functions from the same source:

- `fn_serial` — compiled with `parallel=False`, the original `range` loops.
- `fn_parallel` — compiled with `parallel=True`, `range` rewritten to `prange`.

Both are cached under their respective `kwargs_tag` so the bytecode cache distinguishes them.

### 5.2 &ensp; Warm-up and pinning

The `warmup()` function does more than just trigger compilation. For dual-variant kernels, it runs a **two-pass benchmark**:

1. Call `fn_serial(*args)` and `fn_parallel(*args)` once each to force compilation (cold run, discarded).
2. Measure steady-state latency of both variants on the actual warm-up arguments.
3. **Pin** the faster variant as the default dispatch target.

This pinning happens at scene load time. The rationale: if your scene has 50 objects, the serial path is probably faster, and the dispatcher should not pay thread-pool overhead every frame. If your scene has 500,000 objects, the parallel path wins, and the dispatcher should use it from the first real frame.

### 5.3 &ensp; Runtime fallback

Even after pinning, the parallel variant can fail at runtime (e.g., Numba's thread pool hits an OS limit). The dispatcher catches the exception and **self-heals** to the serial variant for the rest of the session, logging a warning.

---

## 6 &ensp; Warm-up and hot reload in practice

### 6.1 &ensp; Cold-start cost

Compiling a serial + parallel pair through Numba/LLVM typically costs 50–200 ms per function on a modern desktop CPU. That is acceptable at scene load but unacceptable in the middle of gameplay.

### 6.2 &ensp; The warm-up helper

```python
from Infernux.jit import warmup

def on_scene_load():
    dummy_positions = np.zeros((1000, 3), dtype=np.float32)
    warmup(jit_wave_kernel, dummy_positions, 0.0, 1000)
```

By calling `warmup()` during scene load (or a loading screen), the compilation cost is hidden from the player. The dummy arguments also serve as the benchmark input for dual-variant pinning.

### 6.3 &ensp; Hot reload integration

When a script is modified in the editor and reloaded:

1. The module is re-imported, creating new function objects.
2. `njit` computes the bytecode key for each function.
3. If the key matches an existing cache entry → reuse, zero compilation cost.
4. If the key is new (code changed) → recompile only that function. Other kernels are untouched.

This means editing a single script function triggers at most one 50–200 ms recompile, not a full-project rebuild. For an editor-first engine, this is as important as raw throughput.

---

## 7 &ensp; Performance results

The cleanest measurement from the technical report is the *pure-compute benchmark* — no rendering, no Transform write-back, just scripting throughput.

| Element count | Auto-parallel JIT | NumPy (no JIT) | Unity IL2CPP |
|---|---|---|---|
| 10k | >3000 FPS | ~800 FPS | ~2400 FPS |
| 100k | ~2200 FPS | ~120 FPS | ~600 FPS |
| 1M | **848 FPS** | ~80 FPS | ~123 FPS |

At 1M elements: **6.9× faster than Unity IL2CPP**, **10.5× faster than the non-JIT NumPy path**.

The takeaway is not one number but a scaling curve: as entity count grows, the JIT path degrades gracefully while the interpreted path collapses. Python remains viable for real-time authoring when the inner loop is compiled and boundary crossings are batched.

---

## 8 &ensp; Current limitations

The system is honest about what it does not solve:

- **The boundary is now the bottleneck.** Each `batch_read` / `batch_write` call still crosses pybind11, acquires the GIL, performs type conversion, and returns. Kernel compute is no longer the ceiling; boundary latency is.
- **Static variant pinning assumes stable workloads.** The warm-up benchmark pins serial or parallel based on a single measurement at scene load. If the workload changes dramatically at runtime, the pinned choice may become suboptimal.
- **Syntactic alias analysis is conservative.** Some loops that are actually safe to parallelize are rejected because the write pattern does not match the `arr[i] = ...` template.

---

## 9 &ensp; What comes next

### 9.1 &ensp; Lock-free command ring

The next planned boundary optimization is a lock-free ring buffer between the Python scripting thread and the C++ runtime. Instead of synchronous pybind11 calls, batch dispatches would be enqueued as command packets and consumed by the native side without GIL contention. This attacks the boundary-crossing bottleneck that the JIT path has exposed.

### 9.2 &ensp; Adaptive JIT: dynamic problem-scale monitoring

This is the direction we are most actively exploring, and it addresses a fundamental tension in game workloads.

**The core observation:** problem scale in a game is not static. A player might face 20 enemies one second and 300 the next. A particle system might emit 500 particles in a calm scene and 50,000 during an explosion. The optimal parallelization strategy depends on N — and N changes every frame.

Static variant pinning (as described in Section 5.2) picks serial or parallel once at warm-up time and sticks with it. That is a reasonable default, but it leaves performance on the table in dynamic scenarios:

- If pinned to parallel and N drops below the crossover point, thread-pool overhead wastes cycles.
- If pinned to serial and N spikes, the kernel misses an opportunity to scale across cores.

**The idea under exploration** is an *adaptive JIT dispatcher*: a lightweight monitor running on a dedicated sub-thread that continuously observes the actual problem scale (N) each operator is being called with. When it detects that the current variant is no longer optimal for the observed N, it asynchronously triggers a variant swap:

```
Main thread:    kernel(data, N=30)  →  serial variant (fast for small N)
                    ...
                kernel(data, N=8000) → serial variant (suboptimal!)
                    ...
Monitor thread: observes N trending upward → signals "switch to parallel"
                    ...
Main thread:    kernel(data, N=12000) → parallel variant (now optimal)
```

The key design constraints are:

- **Zero contention on the hot path.** The main thread must never block waiting for the monitor's decision. Variant swaps are *eventual* — the main thread reads a single atomic pointer to decide which variant to call, and the monitor updates that pointer asynchronously.
- **Hysteresis to avoid thrashing.** Switching variants has a cost (cache-line invalidation, thread-pool wake-up). The monitor should use windowed averaging or exponential smoothing to distinguish sustained scale changes from momentary spikes.
- **Per-operator granularity.** Different kernels have different serial-parallel crossover points. The monitor tracks N independently for each registered operator.

This would make the JIT subsystem not just a static compiler but a runtime-adaptive execution layer — one that continuously tunes its parallelization strategy to match the actual workload the game is producing, frame by frame.

This work is in the early exploration stage, but it addresses a real gap: existing JIT systems in scientific computing assume stable problem sizes, while game workloads are inherently dynamic. Bridging that gap is the next frontier for the engine's performance story.

---

## Further reading

- [Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting (arXiv:2604.10263)](https://arxiv.org/pdf/2604.10263) — full technical report with benchmark methodology and evaluation.
- [Architecture Overview](about.md) — how the C++ runtime, pybind11 bindings, and Python production layer fit together.