# MuJoCo Warp physics port

These Slang entries follow MuJoCo Warp's shared model and world-indexed data. The port is exercised as a standalone compute sequence before scene integration. The reference revision is MuJoCo Warp 4.0.1, `2db59f4d929dab714d85fab946d93427d2aa56e7`. [Upstream source](https://github.com/google-deepmind/mujoco_warp/tree/2db59f4d929dab714d85fab946d93427d2aa56e7) · [Apache 2.0 license](../../../../../../licenses/mujoco_warp/LICENSE)

## Ported stages

| Module | MuJoCo Warp stages | Slang entries |
|---|---|---|
| `position.slang` | Branch kinematics, body inertial frames, geometry transforms | `kinematics_branch`, `body_inertial_frames`, `geom_local_to_global` |
| `inertia.slang` | `com_pos`, subtree composite inertias and ancestor-row mass assembly | `subtree_com`, `subtree_com_world`, `cinert`, `cdof`, `crb_subtree`, `assemble_mass` |
| `bias.slang` | `com_vel` and `rne` forward dynamics bias | `comvel_branch`, `bias_acceleration_branch`, `bias_force_body`, `bias_force_subtree`, `bias_force_world`, `project_bias` |
| `euler.slang` | Velocity, position, time and warm-start advancement for semi-implicit Euler | `advance_velocity`, `advance_position`, `advance_time` |

The Euler entries consume the final `qacc` for the step. Passive and applied forces, mass matrix factorization, collision, constraint assembly, and the constraint solver are needed to compute that acceleration from the state. Implicit damping, actuator and tendon armature, sleeping, CCD, and alternative integrators also require their corresponding stages before full-step comparisons.

The model contains the topology and inertial data shared by all worlds. `model.slang` defines records using 16-byte fields. Scalar generalized coordinates and the mass matrix remain tightly packed. Dispatch X addresses branches, bodies, joints or degrees of freedom, and dispatch Y addresses worlds. Dynamics arrays use a separate contiguous row per world. Quaternion order is `wxyz`. Matrix records store three padded rows.

`branches` packs the `nbranch + 1` offsets first, followed by Warp's branch body indices. Each leaf branch evaluates its parent chain locally. The first branch containing a body owns the output stores for that body and its joints. Seed world body zero with the identity pose before dispatching. Inertia entries consume contiguous subtree ranges, and the mass rows consume the parent DOF chain and CSR row metadata from the same model. `DofModel.armature.x` holds the generalized diagonal armature for the row.

Run the position, inertia, and bias entries in table order. The bias entries use one gravity vector per world. Euler advances velocity before position and time. Resource dependencies between those dispatches require compute barriers.

Each entry compiles to SPIR-V or WGSL. The following command compiles the branch traversal from this directory.

```sh
slangc position.slang -entry kinematics_branch -target spirv -profile spirv_1_5 -o kinematics_branch.spv
```
