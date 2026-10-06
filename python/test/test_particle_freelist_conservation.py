"""Run generated particle stages on Vulkan and verify resident slot ownership."""
import re
import struct

import numpy as np
import pytest

from infernux.lib import _Infernux
from infernux.particle import GpuParticleGlslLowerer, ParticleKernelLowerer, ParticleScriptCompiler


def _source(rejection):
    body = {
        "half": 'particles.kill_if(particles.id < ctx.parameter("reject-before"))',
        "all": "particles.kill_if(True)",
        "none": "particles.kill_if(False)",
        "nonfinite": "particles.set_position((1e30, 0.0, 0.0))\n            particles.multiply_position((1e30, 1.0, 1.0))",
    }[rejection]
    script = f"""
from infernux.particle import ParticleScript, ParticleEmitter, EmitterSettings, Parameter
class Conservation(ParticleScript):
    parameters = (Parameter(stable_id="reject-before", name="Reject Before", value_type="u32", default=0),)
    class Emitter(ParticleEmitter):
        stable_id = "slots"
        settings = EmitterSettings()
        def init(self, ctx, particles):
            particles.set_position((0.0, 0.0, 0.0))
            particles.set_lifetime(1000.0)
            {body}
        def update(self, ctx, particles):
            particles.kill_if(ctx.delta_time > 0.5)
        def rendering(self, ctx, particles):
            particles.sprite()
"""
    hir = ParticleScriptCompiler().compile(script, source_name="Conservation.particle.py")
    return GpuParticleGlslLowerer().lower(ParticleKernelLowerer().lower(hir)).emitters[0]


@pytest.fixture(scope="module")
def particle_kernels(engine):
    host = engine._acquire_compute_host()
    bundles = {}
    # ComputeHost accepts one descriptor set. Flatten addresses only; keep
    # layouts, control flow, atomics and barriers exactly as generated.
    addresses = {(0, index): index for index in range(16)}
    addresses.update({(3, index): 16 + index for index in range(5)})
    addresses.update({(6, index): 21 + index for index in range(7)})
    bindings = [(index, "uniform" if index == 5 else "storage") for index in range(28)]
    pattern = re.compile(r"set\s*=\s*(\d+)\s*,\s*binding\s*=\s*(\d+)")
    state = [host, bundles, bindings]
    try:
        for rejection, portable in [(mode, portable) for mode in ("half", "all", "none", "nonfinite")
                                    for portable in (False, True)]:
            emitter = _source(rejection)
            stages = {name: pattern.sub(lambda match: f"set = 0, binding = {addresses[int(match[1]), int(match[2])]}", source)
                      for name, source in emitter.stages().items()
                      if name in ("bootstrap", "init", "update", "update_rendering_fused", "render_reset")}
            if portable:
                stages = {name: source.replace("#version 450", "#version 450\n#define INX_WEBGPU 1", 1)
                          for name, source in stages.items()}
            compiled = _Infernux._compile_compute_glsl_batch(stages, f"test-particle-slots-{rejection}")
            kernels = {name: host._create_kernel_from_compiler(code, bindings, 48)
                       for name, code in compiled.items()}
            bundles[rejection, portable] = (emitter, kernels)
        yield state
    finally:
        for emitter, kernels in bundles.values():
            for kernel in kernels.values():
                kernel.wait()
            kernels.clear()
        bundles.clear()
        state.clear()
        kernel = kernels = host = None


@pytest.mark.parametrize("native_schedule,portable", [(True, False), (False, True)])
@pytest.mark.parametrize("capacity,rejection,fused,use_alive", [
    (1, "all", False, True), (257, "half", False, True),
    (65536, "half", False, True), (1048576, "half", False, True),
    (65536, "all", False, True), (65536, "nonfinite", False, True),
    (65536, "none", False, True), (65536, "half", True, True),
    (65536, "all", True, True), (65536, "nonfinite", True, True),
    (65536, "half", False, False), (65536, "all", False, False),
    (65536, "nonfinite", False, False), (65536, "half", True, False),
    (65536, "all", True, False), (65536, "nonfinite", True, False),
])
def test_init_rejection_preserves_free_alive_partition(
    particle_kernels, capacity, rejection, fused, use_alive, native_schedule, portable,
):
    host, bundles, bindings = particle_kernels
    emitter, kernels = bundles[rejection, portable]
    sizes = {0: capacity * emitter.state_stride // 4, 1: capacity, 2: 16,
             3: capacity * 28, 4: 4, 5: 64, 6: capacity,
             7: capacity, 8: capacity, 9: 8, 10: 4, 11: capacity * 4,
             15: 4, 17: 8}
    buffers = []
    buffer = None
    stages = None
    try:
        for index, _kind in bindings:
            buffer = host.create_buffer(sizes.get(index, 4), "uint32")
            buffers.append(buffer)
        buffers[5].set_bytes(np.tile(np.eye(4, dtype=np.float32).reshape(-1), 4).tobytes())
        buffers[15].set_bytes(struct.pack("<4I", 1, 1, 0, 0))
        buffers[20].set_bytes(struct.pack("<4I", 1, 0, 0, 0))
        accesses = ["read_write"] * len(bindings)
        accesses[5] = "read"
        def constants(*, step=0, read=0, alive=False, dt=0.0, flags=0):
            return struct.pack("<6If5I", capacity, capacity, step * capacity, 1, 17, step,
                               dt, flags, read, read ^ 1, int(alive), int(not native_schedule))
        def dispatch(name, push, count):
            return (kernels[name], buffers, accesses, push, (count + 255) // 256, 1, 1)
        stages = [dispatch("bootstrap", constants(), capacity)]
        host.dispatch_batch(stages)
        read_slot = 0
        free_count = capacity
        all_slots = np.arange(capacity, dtype=np.uint32)
        for step in range(7):
            # Initial full-capacity scanning models the first frame after state
            # migration. Subsequent frames use the compact list just published.
            list_ready = use_alive or step != 0
            buffers[18].set_bytes(struct.pack("<4I", step * capacity + capacity // 2, 0, 0, 0))
            # Native SpawnPrepare limits accepted requests to the available
            # slots. Portable hosts use push constants without this pass.
            buffers[17].set_bytes(struct.pack("<8I", free_count, step * capacity, 1, 0,
                                              (free_count + 255) // 256, 1, 1, free_count))
            push = constants(step=step, read=read_slot, alive=list_ready)
            prepare = constants(step=step, read=read_slot, alive=list_ready, flags=8)
            update = constants(step=step, read=read_slot, alive=list_ready, dt=1.0 if step == 5 else 0.0)
            stages = [dispatch("init", push, capacity), dispatch("render_reset", prepare, 1),
                      dispatch("update_rendering_fused" if fused else "update", update, capacity)]
            host.dispatch_batch(stages)
            read_slot ^= 1
            counters = np.frombuffer(buffers[2].get_bytes(64), np.uint32)
            control = np.frombuffer(buffers[10].get_bytes(16), np.uint32)
            free_count, alive_count = int(counters[0]), int(control[read_slot])
            assert free_count + alive_count == capacity, (step, free_count, alive_count, capacity)
            free = np.frombuffer(buffers[1].get_bytes(capacity * 4), np.uint32)[:free_count]
            alive = np.frombuffer(buffers[7 + read_slot].get_bytes(capacity * 4), np.uint32)[:alive_count]
            np.testing.assert_array_equal(np.sort(np.concatenate((free, alive))), all_slots,
                                          err_msg=f"step={step} capacity={capacity} mode={rejection}")
            states = np.frombuffer(buffers[0].get_bytes(capacity * emitter.state_stride), np.uint32).reshape(capacity, -1)
            np.testing.assert_array_equal(np.flatnonzero(states[:, 0] & 1), np.sort(alive))
            assert np.all(states[free, 0] == 0), "Recycled slots retained lifecycle state"
            if rejection in ("all", "nonfinite") or step == 5:
                assert alive_count == 0 and free_count == capacity
            if rejection == "none" and step != 5:
                assert alive_count == capacity
            if rejection == "half":
                id_offset = next(field[3] // 4 for field in emitter.attribute_fields if field[0] == "builtin.id")
                assert np.all(states[alive, id_offset] % capacity >= capacity // 2)
    finally:
        for kernel in kernels.values():
            kernel.wait()
        stages = None
        buffers.clear()
        buffer = kernel = host = None
