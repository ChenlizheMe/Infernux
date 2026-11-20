# MuJoCo Warp physics port

These Slang entries follow MuJoCo Warp's shared model and world-indexed data. The port is exercised as a standalone compute sequence before scene integration. The reference revision is MuJoCo Warp 4.0.1, `2db59f4d929dab714d85fab946d93427d2aa56e7`. [Upstream source](https://github.com/google-deepmind/mujoco_warp/tree/2db59f4d929dab714d85fab946d93427d2aa56e7) · [Apache 2.0 license](../../../../../../licenses/mujoco_warp/LICENSE)

## Ported stages

| Module | MuJoCo Warp stages | Slang entries |
|---|---|---|
| `position.slang` | Branch kinematics, body inertial frames, geometry transforms | `kinematics_branch`, `body_inertial_frames`, `geom_local_to_global` |
| `inertia.slang` | `com_pos`, subtree composite inertias and ancestor-row mass assembly | `subtree_com`, `subtree_com_world`, `cinert`, `cdof`, `crb_subtree`, `assemble_mass` |
| `bias.slang` | `com_vel` and `rne` forward dynamics bias | `comvel_branch`, `bias_acceleration_branch`, `bias_force_body`, `bias_force_subtree`, `bias_force_world`, `project_bias` |
| `passive.slang` | Rigid joint spring and damping forces | `joint_passive_force` |
| `acceleration.slang` | Smooth generalized forces | `assemble_smooth_force` |
| `wrench.slang` | Cartesian body wrenches, gravity compensation and generalized projection | `center_body_wrenches`, `add_generalized_wrenches` |
| `mass_solve.slang` | `factor_m` and `solve_m` with compact, scalar, tiled and sparse blocks | `factor_small_mass`, `factor_tiled_mass`, `factor_sparse_mass`, `solve_small_mass`, `solve_tiled_mass`, `solve_sparse_mass` |
| `euler.slang` | Velocity, position, time and warm-start advancement with optional implicit joint damping | `assemble_euler_mass`, `advance_velocity`, `advance_position`, `advance_time` |
| `broadphase.slang` | Segmented sweep and prune with plane, sphere, axis-aligned box and oriented box filters | `sap_project`, `sap_merge`, `sap_find_ranges`, `sap_scan_ranges`, `sap_emit_pairs` |
| `primitive_contacts.slang`, `box_contacts.slang`, `capsule_box.slang` | Plane against sphere, capsule, ellipsoid, cylinder or box, sphere against sphere, capsule, cylinder or box, capsule against capsule or box, and oriented box against box | `clear_contact_count`, `collide_primitives` |
| `contact_constraints.slang` | Sparse contact Jacobian, constraint row parameters and pyramidal or elliptic contact objective | `clear_constraint_counts`, `reserve_contact_rows`, `project_contact_rows`, `finish_contact_rows`, `seed_contact_acceleration`, `contact_residual`, `evaluate_contact_objective` |
| `contact_cg.slang` | Mass-preconditioned nonlinear conjugate gradient for rigid contact | `begin_contact_cg`, `cg_initial_mass_product`, `cg_project_iteration`, `cg_gradient`, `cg_search_direction`, `cg_ray_mass_and_gauss`, `cg_ray_constraint_product`, `cg_line_search`, `cg_apply_step`, `cg_publish_force` |
| `contact_newton.slang` | Sparse rigid-contact Hessian, reverse block Cholesky and Newton direction | `clear_newton_pattern`, `seed_newton_mass_pattern`, `seed_newton_contact_pattern`, `seal_newton_pattern`, `clear_newton_matrix`, `add_newton_mass`, `build_newton_contact_metric`, `add_newton_contacts`, `factor_newton_matrix`, `solve_newton_direction`, `finalize_newton_direction` |

The mass solve produces the smooth acceleration. The CG and Newton contact solves supply the acceleration consumed by Euler in supported rigid-contact cases. The port accepts applied generalized forces and actuator forces as inputs. Actuator dynamics and transmissions, tendon forces and armature, actuator-specific gravity compensation, the remaining narrowphase pairs and row producers, sleeping, CCD, and alternative integrators require their corresponding stages.

The model contains the topology and inertial data shared by all worlds. `model.slang` defines records using 16-byte fields. Scalar generalized coordinates and the mass matrix remain tightly packed. Dispatch X addresses branches, bodies, joints or degrees of freedom, and dispatch Y addresses worlds. Dynamics arrays use a separate contiguous row per world. Quaternion order is `wxyz`. Matrix records store three padded rows.

`branches` packs the `nbranch + 1` offsets first, followed by Warp's branch body indices. Each leaf branch evaluates its parent chain locally. The first branch containing a body owns the output stores for that body and its joints. Seed world body zero with the identity pose before dispatching. Inertia entries consume contiguous subtree ranges, and the mass rows consume the parent DOF chain and CSR row metadata from the same model. `DofModel.armature.x` holds the generalized diagonal armature for the row.

Run the position, inertia, and bias entries in table order. The bias entries use one gravity vector per world. Factor the mass blocks after assembling the mass matrix, and solve each block after both its factor and the generalized force are available. Euler advances velocity before position and time. Resource dependencies between those dispatches require compute barriers.

Assemble the generalized force after the joint passive step, and add any body wrenches before the mass solve. A body wrench gives world-space torque and force at the body's inertial origin. The projection first moves each wrench to its tree center of mass. With implicit Euler damping, assemble the effective mass and solve with the resolved generalized force as the right-hand side. `advance_velocity` integrates the effective acceleration and stores the resolved, undamped acceleration for warm-starting. Bind the same acceleration to both slots for the explicit damping case.

## Mass layout

Classify each kinematic tree once when building the shared model, following upstream `m_block_layout`. Compact diagonal and full triangular blocks through six DOFs use the scalar entries. Other blocks through 64 DOFs use a workgroup tile, with the upper triangle gathered from CSR using `tileElements`. Absent entries carry `NoIndex`. Larger trees use the sparse entries. Each tiled or sparse workgroup owns a tree in one world. The sparse workgroup gathers contributions into a single writer per row and advances through tree levels on-device.

Factor rows store packed dense block allocations followed by an optional CSR-shaped sparse region. Compact and sparse blocks store reciprocal diagonals in the per-DOF scratch array. The shared `sparseTopology` buffer starts with a four-word header containing the offsets of the level, row and contribution sections. Four-word block records follow. Within each block, the levels list DOFs shallowest first. A row's contribution records list each descendant DOF and the CSR entry for the pair. Submit an entry only when its block list is populated.

## Broadphase

Project bounds after geometry kinematics. The merge and inclusive scan entries use alternating buffers and strides of 1, 2, 4, and so on for the shared geometry count. The sweep workers take their work count from the final prefix sum for their world. The pair rule table follows the reference model's packed upper triangle and carries authored exclusions, explicit pair margins and sensor ids. Dispatches for an empty geometry set can be omitted when preparing the model.

Set pair storage capacity per world through `sweepSettings`. The emission stage reports the total requested slot count. A following stage must consume at most the allocated range and propagate an overflow if the total exceeds it. Sleep filtering and per-world missing meshes require the matching model and state columns.

## Primitive contacts

Clear the contact counters once before dispatching narrowphase. The primitive entry reads each world's candidate count on-device. Its material table stores geometry rows first and explicit pair overrides second. For geometry rows, the first three friction coefficients are sliding, torsional and rolling friction. For explicit pairs, supply the five contact coefficients. Contacts carry signed distance, frame, material, feature ordinal, and constraint and sensor flags. Oriented box pairs test all separating axes and clip the incident face or construct an edge witness pair. Face contacts preserve polygon order and carry up to eight points. Contact counts retain any overflow beyond allocated storage, just as candidate counts do.

The box-pair reference is Warp's primitive SAT route. A full Warp step selects the convex route for box pairs unless `DisableBit.NATIVECCD` selects the primitive route. Plane-cylinder cap contacts use an in-plane radius direction when the cap is parallel to the plane, including rotated planes.

## Convex distance

`collision_geometry.slang` shares the collision shape layout and frame transforms with primitive narrowphase. `convex_support.slang` evaluates sphere, capsule, ellipsoid, cylinder, box and mesh-hull supports. Mesh geometry references its local hull vertices and optional adjacency table in read-only model buffers. The adjacency representation uses Warp's packed offsets and vertex IDs. The support cache tracks the last hull vertex for the life of the query.

`convex_distance.slang` uses the signed-volume simplex reduction, duality gap and support-plane cutoff from Warp's GJK. The query runs in a frame translated to the first geometry's center. For shallow contacts the convex phase inflates the reduced-geometry witnesses by the radii and contact margins. For overlaps it retains the support pairs and vertex IDs needed by EPA. Close-to-collinear triangles compute normalized barycentric coordinates from a common origin to limit FP32 cancellation. The distance functions are available for the convex pass. Penetration expansion, contact manifold generation and publication require their corresponding stages.

## Contact constraints

The contact row entries reserve contiguous CSR rows for each contact. They walk the two DOF chains, omitting shared ancestors, then project the contact frame onto the point and angular Jacobians. The continuous-time row update computes the reference acceleration and regularization from the projected velocity, material and world timestep. The setup entries select the initial acceleration, evaluate the constraint residual, and compute the force, cost and state for the chosen friction cone.

The per-world row and nonzero counts retain overflow. Skip any incomplete matrix and report the exceeded capacity before publishing or integrating the step. Equality, limits, DOF friction loss and discrete integrator rows need their own producers.

Prepare the symmetric mass adjacency once per model. Store `nv + 1` CSR offsets followed by pairs of a column DOF and its packed mass index, with the diagonal first in every row. CG reads the iteration and line search limits and packed matrix size from `cgOptions`. The world records in the constraint settings supply the convergence and line search tolerances in their last two words.

Initialize the solver after the row assembly, mass factorization and warm-start selection. Clear and compute `Jᵀf` in a workgroup per world, form the gradient, apply the existing mass solve as the preconditioner, and update the Polak–Ribière direction. The mass search product and smooth directional derivative share one dispatch. Compute the Jacobian search product before the bracketed Newton line search. The accepted step updates acceleration, its mass product and the constraint residual. Re-evaluate the contact objective at that residual before forming the next gradient. A final gradient and direction check distinguishes convergence from hitting the budget. Use a final projection for publication, including zero force for worlds without contacts. Worlds track status and line search exhaustion on-device. The bounded sequence can be recorded in one compute packet, with no intermediate host readback.

## Newton contact direction

Build the symbolic pattern once from the mass and contact supports for the step. A world with `t = ceil(nv / 8)` uses `t * ceil(t / 32) + 5 * t + 3` words. These hold the tile bitset, numeric tile offsets, component offsets and members, temporary roots and cursors, and the factorization error flag. Reverse elimination adds fill and counts packed upper tiles. The fourth word of the solver options is the numeric tile capacity per world. Each packed tile has 64 floats. A second option record supplies the number of X workgroups for contact assembly, which loops over rows at that stride.

After each gradient update, assemble the mass and the contact Hessian into that pattern. The elliptic middle region uses the local cone curvature. Symbolic elimination assigns tiles to independent components. Dispatch factorization and triangular solve with `t` groups along X. Each nonempty group processes one component, preserving descending DOF order for Cholesky. The finalization dispatch uses one group per world to check convergence and combine the direction norms. The direction uses the shared search products, line search, and accepted-step update. Capacity exhaustion and an invalid diagonal are reported separately before publishing a step.

## Shader compilation

Each entry compiles to SPIR-V or WGSL. The following command compiles the branch traversal from this directory.

```sh
slangc position.slang -entry kinematics_branch -target spirv -profile spirv_1_5 -o kinematics_branch.spv
```
