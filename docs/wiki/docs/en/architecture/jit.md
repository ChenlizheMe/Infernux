---
category: Architecture
tags: ["jit", "performance", "python"]
date: "2026-04-16"
---

# JIT-Accelerated Scripting

This page condenses the JIT section of the technical report, [*Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting*](https://arxiv.org/pdf/2604.10263), into a shorter project-facing note.

## Why the JIT path exists

Pure Python is expressive, reloadable, and fits the engine's editor-first workflow, but per-object property access across the Python and C++ boundary is too expensive for large per-frame workloads. Infernux addresses that bottleneck with two mechanisms that are designed to work together:

- A batch data bridge that moves engine state through contiguous NumPy arrays in one boundary crossing.
- An optional JIT path built on Numba that compiles annotated Python update functions to LLVM machine code.

The practical goal is not to replace Python with a different gameplay language. It is to keep Python as the authoring language while moving the tight inner loop closer to native throughput.

## Batch first, then JIT

The report treats JIT as part of a larger performance bridge rather than a standalone compiler trick. Typical frame logic follows this shape:

```python
positions = batch_read(targets, "position")
jit_wave_kernel(positions, time_value, count)
batch_write(targets, positions, "position")
```

That pattern keeps the number of Python-to-C++ crossings constant per frame instead of scaling with object count. The JIT kernel then works on dense arrays, which is exactly the case where Numba performs well.

## What Infernux adds on top of Numba

The engine's JIT decorator is intentionally more opinionated than calling `numba.njit` directly.

- It degrades gracefully to pure Python when Numba is not installed, while emitting a runtime warning so the slowdown is visible.
- It uses a bytecode-keyed cache that survives module reloads and editor hot reload cycles.
- It preserves partial compatibility with Nuitka-based packaging. Numba still requires CPython at runtime, so this is a distribution convenience rather than a full AOT replacement.

In other words, the JIT path is integrated into the engine's reload and packaging model instead of being treated as an isolated optimization utility.

## Auto-parallelization

The paper's most engine-specific JIT feature is the auto-parallel mode. When enabled, an AST rewriter inspects counted loops of the form `for i in range(n)` and promotes them to `prange` when the loop body satisfies a conservative set of rules.

The rewriter does **not** try to parallelize every loop. It rejects:

- Iterator-style loops such as `for x in arr`.
- `while` loops.
- Bodies containing unsupported control flow such as `break`, `yield`, exception handling, or early return.
- Writes that do not look like disjoint indexed array stores.

The aliasing check is intentionally syntactic rather than fully semantic. In practice, that is acceptable for the dominant engine use case: updating positions, rotations, scales, or other SoA columns indexed by entity ID.

## Warm-up and hot reload

Compiling both a serial and a parallel variant introduces cold-start cost. The report places typical compile latency at roughly 50-200 ms per decorated function on the benchmark machine. Infernux hides that cost in two ways:

- A warm-up helper can precompile registered kernels during scene load.
- During hot reload, only the modified function is recompiled; unchanged kernels keep their cached machine code.

This is an important design choice for editor workflows. The goal is not just high throughput after compilation, but also predictable iteration while scripts are being changed live.

## What the report shows

The cleanest JIT result in the report is the pure-compute benchmark, where no rendering and no Transform write-back are involved. In that workload:

- The auto-parallel JIT path sustains high frame rates as element count scales from 10k to 1M.
- At 1M elements, the runtime result in the paper reaches 848 FPS.
- That is reported as 6.9x faster than the Unity IL2CPP reference and 10.5x faster than the non-JIT NumPy path for the same workload.

The larger point is not a single headline number. It is that Python remains viable for real-time authoring when engine state is batched and the heavy loop is compiled.

## Limits and next step

The report is explicit that the remaining bottleneck is no longer arithmetic throughput inside the compiled kernel. It is the communication boundary itself. Each batch dispatch still crosses pybind11, acquires the GIL, performs type conversion, and returns.

That is why the roadmap after JIT focuses on a lock-free command ring for the Python-to-native path. The JIT path already reduces compute cost sharply; the next ceiling is the boundary crossing around it.

## Read the full report

For the complete evaluation, benchmark tables, and system context, read the full report:

- [Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting (arXiv:2604.10263)](https://arxiv.org/pdf/2604.10263)