# Current CPU JIT demo

Copy `Assets/Scripts/CpuJitDemo.py` into an Infernux project's `Assets/Scripts`
folder, add `CpuJitDemo` to an empty GameObject, then enter Play mode.

The acceptance marker is `INFERNUX_CURRENT_JIT_DEMO_READY`. Desktop Players
must report `mode=parallel`; Web reports `mode=ordinary-python` because that
target deliberately builds the same decorated function as regular Python.

The sample exercises only the current public surface:

- `@inx.jit.compile`
- `inx.jit.warmup`
- `inx.jit.statistics` on targets that provide the CPU JIT runtime

It does not require a resource path, a project plugin, an AOT product, or a
legacy Numba decorator.
