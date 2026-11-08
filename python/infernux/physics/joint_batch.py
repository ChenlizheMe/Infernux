"""Ordered live hinge IO on the existing physics owner and numerical buffers."""
import numpy as np

from infernux import lib
from infernux.compute import Buffer


class HingeJointBatch:
    """Bind once after Play has created the joint constraints.

    Motion and targets use radians; component Inspector angles use degrees.
    Methods run on the physics owner thread. Recreate after a joint rebind or
    a new Play session; scaling preserves the original joint reference.
    """
    def __init__(self, joints):
        self._native = lib.HingeJointBatch([joint._require_cpp_component() for joint in joints])

    def __len__(self):
        return len(self._native)

    @property
    def world_id(self):
        return self._native.world_id

    @property
    def simulation_epoch(self):
        return self._native.simulation_epoch

    def validate(self):
        self._native.validate()

    def read_state(self, out=None):
        """Return motion[N,2] and physics_step for the observed solver state.

        A reusable contiguous float32 CPU array or Buffer may supply more
        rows than needed; its tail is untouched. No GPU transfer is issued.
        Use nn.Channel.hinges with nn.program for managed GPU input.
        """
        if out is None:
            out = np.empty((len(self),2),dtype=np.float32)
        if isinstance(out,Buffer):
            if out.device != 'cpu' or out.dtype != np.dtype(np.float32):
                raise TypeError('hinge snapshots require float32 CPU storage')
            array = out.numpy(copy=False).reshape(-1,2)
        else:
            array = out
        step = self._native.read_motion(array)
        return {'motion':out,'physics_step':step}

    def set_motor_targets(self, targets, *, expected_physics_step=None):
        """Apply radians[N] or [N,1], without enabling motors or stepping.

        Pass the snapshot's physics_step to reject actions spanning another
        completed physical integration. GPU inference should request a CPU
        result through its existing managed result transfer before this call.
        """
        if expected_physics_step is not None and (type(expected_physics_step) is not int or expected_physics_step < 0):
            raise ValueError('expected_physics_step must be a non-negative integer')
        if isinstance(targets,Buffer):
            if targets.device != 'cpu' or targets.dtype != np.dtype(np.float32):
                raise TypeError('hinge targets require float32 CPU storage')
            targets = targets.numpy(copy=False)
        self._native.set_motor_targets(targets,expected_step=expected_physics_step)
