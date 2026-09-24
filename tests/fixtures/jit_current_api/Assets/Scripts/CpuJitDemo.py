"""Copyable CPU JIT demo for the current Infernux authoring API.

Attach :class:`CpuJitDemo` to an empty GameObject and enter Play mode.  The
component prepares one numeric signature without mutating gameplay data, runs
the function once, and prints a deterministic acceptance marker.

Web Players intentionally execute the decorated function as ordinary Python;
they do not expose native compilation statistics.  That platform contract is
selected before execution and is not an error-time fallback.
"""

import numpy as np

import infernux as inx


@inx.jit.compile(parallel_policy="required")
def integrate(positions, velocities, delta_time, gravity):
    """Advance independent rows in place; the loop must be CPU-parallel."""
    for index in range(len(positions)):
        velocities[index, 1] -= gravity * delta_time
        positions[index, 0] += velocities[index, 0] * delta_time
        positions[index, 1] += velocities[index, 1] * delta_time
        positions[index, 2] += velocities[index, 2] * delta_time


def create_state(sample_count):
    """Create deterministic NumPy state accepted by desktop and Web Players."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    positions = np.zeros((sample_count, 3), dtype=np.float32)
    velocities = np.empty((sample_count, 3), dtype=np.float32)
    velocities[:, 0] = 1.5
    velocities[:, 1] = 2.0
    velocities[:, 2] = -0.5
    return positions, velocities


def run_current_jit_demo(sample_count=4096, delta_time=0.125, gravity=9.0):
    """Run the complete current API flow and return its observable result."""
    positions, velocities = create_state(sample_count)
    positions_before = positions.copy()
    velocities_before = velocities.copy()

    inx.jit.warmup(integrate, positions, velocities, delta_time, gravity)
    if not np.array_equal(positions, positions_before):
        raise RuntimeError("jit.warmup mutated positions")
    if not np.array_equal(velocities, velocities_before):
        raise RuntimeError("jit.warmup mutated velocities")

    integrate(positions, velocities, delta_time, gravity)
    expected_velocity_y = 2.0 - gravity * delta_time
    expected_first_position = np.array(
        [1.5 * delta_time, expected_velocity_y * delta_time, -0.5 * delta_time],
        dtype=np.float32,
    )
    if not np.allclose(positions[0], expected_first_position):
        raise RuntimeError("compiled integration produced an unexpected position")

    if not inx.jit.JIT_AVAILABLE:
        return {
            "mode": "ordinary-python",
            "sample_count": int(sample_count),
            "first_position": tuple(float(value) for value in positions[0]),
            "specialization_count": 0,
        }

    report = inx.jit.statistics(integrate)
    if report.selected_mode != "parallel":
        raise RuntimeError(
            f"CPU JIT selected {report.selected_mode!r}; this demo requires parallel"
        )
    if not report.specializations:
        raise RuntimeError("CPU JIT did not publish a specialization")
    return {
        "mode": report.selected_mode,
        "sample_count": int(sample_count),
        "first_position": tuple(float(value) for value in positions[0]),
        "specialization_count": len(report.specializations),
    }


class CpuJitDemo(inx.InxComponent):
    """Play-mode component entry point for the CPU JIT acceptance sample."""

    sample_count: int = 4096
    delta_time: float = 0.125
    gravity: float = 9.0

    def start(self):
        result = run_current_jit_demo(
            sample_count=self.sample_count,
            delta_time=self.delta_time,
            gravity=self.gravity,
        )
        inx.Debug.log(
            "INFERNUX_CURRENT_JIT_DEMO_READY "
            f"mode={result['mode']} samples={result['sample_count']} "
            f"specializations={result['specialization_count']} "
            f"first_position={result['first_position']}",
            self,
        )
