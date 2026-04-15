# fix(scene): gizmo rigidbody drag flash-back, double-sided shadow, shader hot-reload cache

## Summary

Fix three regressions/bugs related to play-mode scene view interaction and forward-lit rendering. Gizmo drag on dynamic rigidbodies caused a one-frame flash-back to the pre-drag position on release; double-sided (backface) lit surfaces incorrectly self-shadowed; and shader template edits did not take effect on hot-reload due to stale template cache.

Also includes a zero-allocation refactor for physics ECS iteration (`ForEachAlive*` lambdas replacing `GetAlive*Handles` + index loops), sprite shader double-sided / alpha clip fixes, and the quaternion-based rotation gizmo rewrite that eliminates Euler↔quaternion round-trip jumps.

## What changed

- **Rigidbody kinematic→dynamic interpolation cache** (`Rigidbody.cpp`): `SetIsKinematic(false)` now reinitializes `previousPhysicsPosition`/`currentPhysicsPosition` and `lastSyncedPosition`/`lastSyncedRotation` from the current Jolt body and Transform. Prevents `ApplyInterpolatedTransform` from overwriting the Transform with stale pre-kinematic values on the first frame after the switch.
- **Gizmo drag in play mode** (`_scene_view_gizmo.py`, `scene_view_panel.py`): Rewritten to temporarily switch dynamic rigidbodies to kinematic during drag and write Transform directly + `Physics.sync_transforms()` per frame for immediate visual feedback, instead of relying on `MoveBodyKinematic` (which defers to next physics step).
- **Rotation gizmo** (`_scene_view_gizmo.py`): Stores drag start as `quatf` and applies delta via `quatf.angle_axis()` directly to `transform.rotation`, eliminating Euler→Quaternion→Euler round-trip that caused normalization jumps.
- **Forward-lit double-sided shadow fix** (`fragment_outputs_lit.glsl`): `GetMainLight()` macro now uses `gl_FrontFacing`-aware normal for shadow sampling, so backface shadow bias pushes in the correct direction.
- **Surface main backface normal flip** (`surface_main.glsl`, `surface_main_gbuffer.glsl`): Added `if (!gl_FrontFacing) s.normalWS = -s.normalWS` after `surface()` in both forward and gbuffer paths.
- **Sprite shaders** (`sprite_lit.frag`, `sprite_unlit.frag`): Added `@cull: none` and `@alpha_clip: on` for proper double-sided rendering.
- **Shader template cache invalidation** (`InxShaderLoader.cpp/.hpp`, `Infernux.cpp`): Added `InvalidateTemplateCache()` and call it from `ReloadShader()` so template file edits under `_templates/` take effect on hot-reload.
- **Physics ECS zero-allocation iteration** (`SceneManager.cpp`, `PhysicsECSStore.h`): Replaced `GetAliveColliderHandles()`/`GetAliveRigidbodyHandles()` (returns `std::vector` copy) with `ForEachAliveCollider`/`ForEachAliveRigidbody` lambda-based traversal in all sync paths.
- **Other** (`InxContiguousPool.h`, `BoxCollider.cpp`, `PhysicsWorld.cpp`, `VkCoreDraw.cpp`, `VkCoreMaterial.cpp`, `MaterialPipelineManager.h`, `EngineConfig.h`): Renderer double-sided pipeline support, contiguous pool `ForEach`, box collider shape rebuild fixes.

## Verification

- [x] Built the affected targets (release preset, exit code 0)
- [x] Ran static validation on all modified Python and C++ files — no errors
- [ ] Manual play-mode test: drag dynamic rigidbody with gizmo — no flash-back on release
- [ ] Manual test: sprite backface lighting — no self-shadow artifact
- [ ] Manual test: edit shader template file, Ctrl+R reload — change takes effect without restart

## Notes for reviewers

- The `SetIsKinematic` cache reset reads from Jolt body position via `PhysicsWorld::GetBodyPosition` which uses `BodyInterface_NoLock`. This is safe because `SetIsKinematic` is only called from the main thread (gizmo code in `Update`, inspector property changes), never from the physics step.
- `Physics.sync_transforms()` during gizmo drag calls `SyncCollidersToPhysics()` which uses `MoveKinematic` under the hood for kinematic bodies — this is correct and matches Unity's `Physics.SyncTransforms()` behavior.
- The `ForEachAlive*` refactor is a drop-in replacement with identical iteration order; the only difference is avoiding the temporary `std::vector` allocation per frame.
- Sprite `@cull: none` + `@alpha_clip: on` are metadata-only changes that affect pipeline creation — existing scenes using these shaders will recompile pipelines on first load.
