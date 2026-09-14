# 041 执行记录

## Latest — 2026-09-14 multi-config build guidance

- Added an explicit Windows multi-config build note to the root README.
- Direct Visual Studio builds now document `--config Release` for staging and
  wheel packaging, preventing Debug/Release runtime mixing with Vulkan/native
  artifacts. The existing `windows-msvc-*` build presets already select this
  configuration automatically.

## Latest — 2026-09-14 RenderTexture and camera regression

- Release Vulkan regression passed **6/6** for RenderTexture, RenderTexture
  artifacts, camera culling isolation, RenderTexture Vulkan, Screen UI Vulkan,
  and multi-camera GPU mesh retirement.

## Latest — 2026-09-14 UI value and asset API regression

- Python UI value controls, RenderTexture asset references and integer
  Inspector batching passed **49/49**.
- The first command referenced a non-existent test filename; it was corrected
  to the repository's current test layout before recording the result.

## Latest — 2026-09-14 Raycast batch binding hot path

- Released the Python GIL while the native `PhysicsWorld::RaycastBatch` runs;
  Python reacquires it only to publish results into caller-owned arrays.
- Rebuilt the Release `_Infernux` module and restaged its native dependencies.
- Physics query integration regression passed **29/29** selected tests.

## Latest — 2026-09-14 Release regression after Raycast change

- Full Windows Release CTest passed **90/90** in 109.09 seconds, including
  physics, RenderGraph, UI, asset, compute, and GPU mesh suites.

## Latest — 2026-09-14 wheel publication staging after Raycast change

- Rebuilt `infernux-0.4.0-2-cp313-cp313-win_amd64.whl` through the Release
  packaging target; native payload verification passed.
- Installed that wheel into the isolated audit directory and imported it
  successfully: `WHEEL_RAYCAST_RELEASE_OK 0.4.0`.

## Latest — 2026-09-14 full Python regression after wheel rebuild

- Full `python/test/` regression passed **6358 tests / 12 skipped** in 309.29
  seconds after rebuilding the Release wheel and native Raycast binding.
- The suite emitted one expected diagnostic from the transactional rejection
  test for an asymmetric tag/layer collision matrix; it did not fail a test or
  change the result.

## Latest — 2026-09-14 official Windows wheel preset

- Ran the documented `windows-msvc-wheel` CMake preset under `conda activate
  infernux`; the preset rebuilt the integrated GPU JIT/Taichi compiler payload,
  native bindings, Player host and official plugins.
- Wheel creation and native payload verification completed successfully for
  `infernux-0.4.0-2-cp313-cp313-win_amd64.whl`.

## Latest — 2026-09-14 wheel payload audit

- Audited the generated wheel directly: **959 files**.
- Confirmed the native `_Infernux` module, integrated GPU JIT compiler,
  Taichi `NOTICE`, and wheel metadata are all present; no required payload is
  missing.

## Latest — 2026-09-14 A17 contract promotion

- Promoted the verified batch-RayCast GIL boundary into the 041 plan as an
  explicit completed engine contract; progress is now 104/257 in the main
  plan and 36/76 in the compute plan (140/333 combined).

## Latest — 2026-09-14 UI material contract regression

- UI material serialization, Button background/text slots, dependency tracking,
  command packets and native dependency publication passed **233 tests / 2
  skipped**.
- Inspector/Player shader-property and cross-platform visual acceptance remains
  intentionally open; these tests do not close that broader item.

## Latest — 2026-09-14 runtime UI material submission

- Screen UI runtime submission and native material dependency tests passed
  **30/30**, confirming material references reach the runtime UI command path
  rather than existing only in serialized component state.

## Latest — 2026-09-14 RenderTexture Python resource chain

- RenderTexture resource creation, binary artifacts, asset references and UI
  sampling tests passed **101/101**.
- Camera-target and persistent-resource contracts remain covered separately by
  the Release Vulkan suite; editor/Player visual acceptance is still open.

## Latest — 2026-09-14 RenderGraph camera-target contract review

- Reviewed the `copy_texture` validation path for the reported `CommitGrade`
  message. Camera targets are intentionally persistent outputs, not transient
  copy sources/destinations; the supported public paths remain `present()` or
  a fullscreen pass. Existing regression coverage keeps this contract strict.

## Latest — 2026-09-14 isolated wheel runtime import

- Installed the rebuilt wheel into an isolated project-local audit directory with `--no-deps`, ahead of the conda site-packages path.
- Runtime import passed: `WHEEL_RUNTIME_API_OK 0.4.0`; `UIRawImage is UIImage` and all promoted UI exports resolve from the wheel itself.

## Latest — 2026-09-14 wheel content audit

- Audited the newest Windows wheel directly: **959 files**, including lowercase `infernux.pyi`, top-level UI stubs, `UIProgressBar` implementation and Taichi `NOTICE`; no required package file is missing.

## Latest — 2026-09-14 cross-platform Player contract regression

- Android, Linux, multiplatform Player, platform plugin and build/export contract suites passed **136 tests / 2 skipped**.
- This confirms the public API and rebuilt wheel did not regress platform packaging contracts; device-specific APK execution and final per-platform 041Lab visual acceptance remain open.

## Latest — 2026-09-14 GPU compute and mesh residency regression

- Full Vulkan GPU compute/mesh execution subset passed **10/10**: buffer, kernel, mesh publication, Gizmo compute, physics exchange, continuous publication, retirement, multi-camera retirement and resident mesh resources.
- This confirms the compiler output reaches the shared Vulkan resource path without per-test failures; it does not replace the 041Lab real soft-body/rigid-body performance gate.

## Latest — 2026-09-14 Taichi compiler contract regression

- Main Release CTest compiler contracts passed **3/3**: compiler contract, bit contract and compiler-only path.
- The fork's current compiler-only Vulkan lowering remains build-integrated and independently covered; runtime buffer ownership and full 041Lab soft-body acceptance remain separate open items.

## Latest — 2026-09-14 composite model artifact regression

- Mesh artifact, skinned mesh artifact, AssetDatabase refresh and dependency graph native tests: **4/4 passed**.
- Existing binary artifacts preserve the current mesh/submesh/material/skinning contracts; the missing A16 work is specifically an authored composite hierarchy and stable child-asset identity, not a regression in the existing mesh artifact path.

## Latest — 2026-09-14 model asset format smoke regression

- Confirmed the shared asset-format registry continues to classify `.blend` as a mesh source and the core asset-category/import contract remains green: **63 passed**.
- This verifies the format is recognized by the asset pipeline; full Blender hierarchy/material preservation and cooked native-asset acceptance remain open plan work.

## Latest — 2026-09-14 full Python regression and stub export fix

- Full `python/test` regression reached **6357 passed / 12 skipped** with one public namespace stub failure.
- Fixed `python/infernux.pyi` to explicitly mirror the newly public UI exports (`UICanvas`, `UIFrame`, `UIGroup`, `UIProgressBar`, `UISlider`, `UIText`, `UIImage`, `UIRawImage`, `UIButton`, `UIEvent`, `UIEvent1`).
- Re-ran the affected namespace suite: **11 passed**. The full run's only failure is resolved; the broad suite should be rerun before publishing a new wheel.
- Full suite rerun after the stub fix: **6358 passed / 12 skipped** in 308.17 seconds.
- Restaged and rebuilt the Windows Release wheel with explicit `--config Release`; native payload verification passed for `infernux-0.4.0-2-cp313-cp313-win_amd64.whl`. The earlier Debug/Release linker mismatch came only from invoking a multi-config build without its configuration selector.
- The rebuilt wheel was produced after the full regression fix and contains the synchronized lowercase `infernux.pyi` UI exports; the configured Release staging path completed native-payload verification.
- Packaging validation also confirmed the multi-config requirement: invoking the Windows targets without `--config Release` selects Debug and can produce a runtime-library mismatch against Release-only third-party archives; the supported preset/explicit Release path is green.

## Latest — 2026-09-14 full Release native regression

- Re-ran the complete configured Windows Release CTest suite with the required `Release` configuration: **90/90 passed** in 83.41 seconds.
- This includes Vulkan RenderTexture, screen UI, compute buffer/kernel/mesh/Gizmo/physics exchange, mesh publication/retirement and scene residency soak tests; no native regression was observed.

## Latest — 2026-09-14 Inspector/Camera/RenderTexture integration regression

- Ran the complete Inspector, Camera and RenderTexture test groups together: **309 passed**.
- The editor-created RenderTexture description, Camera target/reference surface and Inspector property batching remain compatible across the integrated test set. Expected no-pipeline fixture diagnostics remain isolated to tests.

## Latest — 2026-09-14 camera/inspector integer contract regression

- Re-ran integer property batch decoding together with native Inspector and Camera contract/projection coverage: **83 passed**.
- The unsigned camera culling-mask range and integer Inspector batch path remain covered without a test failure; the renderer warnings are intentional fixtures with no active pipeline, not a new regression.

## Latest — 2026-09-14 RenderTexture/UI contract regression

- Re-ran the focused UI value-control, world-input, RenderTexture component, artifact and runtime texture suites after the public API changes: **117 passed**.
- This confirms managed RenderTexture references, camera-target-facing asset contracts, UI texture aliases, value notifications and world-input paths remain compatible; full Player/cross-platform acceptance is still open.

## Latest — 2026-09-14 public API and registry regression

- Ran the component catalog, component registration, RenderTexture asset and plugin publication regression set after promoting the UI surface to the top-level package API: **65 passed**.
- No import-order, registry, serialization or plugin-publication regression was observed; the rebuilt wheel remains the validated artifact.

## Latest — 2026-09-14 UI value-control contract

- `UIProgressBar` now exposes the same value-change event contract as `UISlider`; both controls clamp through `set_value(value, notify=True)` and provide Unity-style `set_value_without_notify(value)` for state restoration and internal updates.
- Added focused regression coverage for progress notifications, silent writes, clamping, pointer-driven slider changes and existing mouse/touch capture: **15 passed**.
- This closes a public API inconsistency but does not by itself close the full A06.1 item; Inspector/Player acceptance and the remaining UI resource matrix stay open.
- The wider UI/material/RenderTexture/world-input regression after the change passed **280 tests / 2 skipped**.
- Hot-reload, command publication and Play-mode component reload coverage after the API change passed **154 tests / 1 skipped**.
- Follow-up resource and hit-snapshot regression passed **59 tests**; managed `UIImage` RenderTexture references, camera target references, UI dependencies and hit snapshots remain intact.
- Added a legacy scene-load regression proving authored `x/y/rotation` values migrate to the attached GameObject Transform; value-control tests now pass **7/7**.
- Exposed `UIRawImage` as the Unity-compatible public name for the existing `UIImage` managed texture/RenderTexture component; it shares the serialized type and renderer. RenderTexture and value-control regression passed **45 tests**.
- Promoted the UI component surface, including `UIRawImage`, to the top-level `Infernux` API for Unity-style scripts; local source import check reported `UI_TOPLEVEL_OK`.
- Full UI component, command, dependency, world-input and RenderTexture regression after top-level exports passed **282 tests / 2 skipped**; no import-cycle or serialization-order regression was observed.
- Re-staged the Windows Release wheel through `stage_python_package`; staged source imports report `Infernux 0.4.0` and `UIRawImage is UIImage`, with the private compiler payload and license files retained.
- Rebuilt the actual wheel with `package_python` after detecting the previous stale archive. The new `infernux-0.4.0-2-cp313-cp313-win_amd64.whl` contains 959 files; package inspection reports `WHEEL_UI_API_OK` and CMake's native-payload verifier passes.

## Latest — 2026-09-14 world UI batch submission optimization

- World UI no longer treats ImGui canvas `ClipRect` as a batch boundary, matching its camera/depth-only contract; compatible elements can merge even when authored clip rectangles differ.
- The camera view-projection push constant is now submitted once per world replay instead of once per draw command.
- Rebuilt the renderer and passed the focused Vulkan regression **1/1**; the test now verifies clipped world elements remain visible and merge into one draw.

## Latest — 2026-09-14 formal wheel restage

- Re-ran the supported `stage_python_package` target after the native rebuild. The Windows wheel was staged successfully and contains 940 files, private Taichi compiler payload under `Infernux/_compiler/taichi`, and Taichi `LICENSE`/`NOTICE`; no top-level `taichi/` package is emitted.

## Latest — 2026-09-14 incremental native build

- Rebuilt the current Windows Release native runtime after the accumulated renderer/resource/physics changes; `InfernuxRuntime` and all dependent native modules staged successfully.

## Latest — 2026-09-14 full Python regression baseline

- Full `python/test` regression passed **6354 tests / 12 skipped in 284.64 s**. Reported shader and tag-layer errors are expected negative-fixture assertions; the suite completed successfully.

## Latest — 2026-09-14 raycast boundary audit

- Audited `Physics.raycast_batch`: one physics-world synchronization is performed before the batch, inputs/outputs cross Python/C++ once, and each query reuses Jolt's published broad/narrow-phase world. No per-ray Python callback or scene scan is present.
- The remaining A17 work is therefore engine-side query throughput and hit publication (including non-convex BVH scaling), not API batching. Existing performance evidence remains below the 041 budget and the item stays open.

## Latest — 2026-09-14 RenderTexture authoring/cook regression

- RenderTexture resource creation, importer coordination, Camera GUID target binding, material/UI sampling and game-builder closure passed **334 passed, 1 skipped**.
- This confirms the existing resource-file path is functional; remaining RenderTexture work is editor UX/performance and cross-platform visual acceptance, not a second runtime target implementation.

## Latest — 2026-09-14 regression sweep

- Focused Python regression for asset formats, screen/world UI, RenderTexture and physics passed **206/206**.
- Windows Release native CTest passed **90/90** in 101.12 s, including RenderTexture, UI, GPU buffer/kernel/mesh/gizmo/physics exchange, mesh retirement and scene residency soak.

## Latest — 2026-09-14 Blender source-format audit

- Confirmed the existing model asset boundary already classifies `.blend` alongside FBX/GLTF through the single native `kMeshExtensions` registry; the AssetDatabase, Project panel and `ModelImporter` all consume that same list, and the Python asset-category contract covers `.blend`.
- Focused asset-type and RenderTexture regression passed **99/99**. This is format-recognition evidence only: the full A16 composite Blender importer, stable sub-resource identity, material/texture mapping, reimport and no-source Player audit remain unchecked until a real `.blend` fixture and conversion backend are selected.

## Latest — 2026-09-14 Taichi compiler test closure

- Fixed the fork's SPIR-V dependency policy so compiler-only builds force `SPIRV_SKIP_EXECUTABLES` and `SPIRV_SKIP_TESTS` in the cache, and no longer enter the upstream SPIR-V test subtree when tests are disabled. This keeps the shipped compiler focused on the engine path and avoids stale/unrelated test targets.
- A fresh Python 3.13 Release configure/build completed successfully. The private `taichi_python.cp313-win_amd64.pyd` and both Infernux contract executables were produced.
- Fresh CTest result: **2/2 passed** (`infernux.compiler_contract`, `infernux.bit_contract`); no SPIR-V unit tests were registered. Existing compiler warnings remain non-fatal upstream cleanup candidates.
- Strict 041 progress remains **139/332 (41.9%)**; this closes only the Taichi build-policy substep, not the full Taichi/runtime or 041 acceptance gates.
- After the fork CMake change, the existing Windows Release engine build completed its full native regression at **90/90 passed in 95.29 s**, including RenderTexture, screen UI, compute/GPU exchange, mesh publication/retirement and scene residency soak.
- RenderTexture authoring/runtime coverage was rechecked after the build-policy change: asset creation, GUID resolution, Camera target binding, material/UI sampling and binary artifact cooking passed **110/110 Python tests**.
- World UI no longer applies ImGui canvas clip rectangles in its fragment path; it remains constrained only by the camera render area and scene depth. The native screen/world UI, RenderTexture and transform dependency regression passed **3/3** after rebuilding the renderer.
- World UI collection, per-element transform/Z picking, pointer routing, layer filtering and native dependency boundaries passed **61/61 Python tests**. This confirms the editor/runtime picking contract is already present; remaining work is visual Scene outline integration and performance, not another parallel picking implementation.
- Full Windows Release native regression after the world-UI clip removal passed **90/90 in 96.83 s**, including all RenderTexture, UI, compute/GPU exchange, mesh retirement and residency tests.
- Removed the now-dead world-UI clip rectangle from Vulkan push constants as well as the fragment decision, reducing per-draw state to the view-projection matrix. Rebuilt the renderer; `screen_ui_vulkan` and `render_texture_vulkan` passed **2/2**.
- Audited the RenderGraph `Copy` validation: rejecting camera-target copies is intentional because a persistent camera target must be exported with `present()` or consumed by a fullscreen pass; transient copy semantics remain unchanged. RenderGraph plus RenderTexture asset/runtime tests passed **154/154**.
- Complete Python regression after the world-UI shader/push-constant changes passed **6354 tests / 12 skipped in 274.94 s**. The emitted error lines are expected negative-fixture coverage and did not fail the suite.
- Rechecked the current engine-level GPU boundary before advancing A03.4: resident GPU buffer exchange, Physics feedback exchange, continuous mesh publication and scene residency soak passed **4/4**. These prove the reusable transport/lifetime path, but not yet GPU narrow-phase or bidirectional soft-body coupling.
- Cleaned two Taichi fork MSVC warnings in active compiler transforms: explicit `AccessFlag` comparison and pointer hashing through `uintptr_t`. The Python 3.13 Release rebuild completed without those warnings; compiler contracts remain **2/2 passed**.
- Ran the formal `stage_python_package` target (the supported packaging target; there is intentionally no standalone `gpu-jit-wheel` target). It rebuilt and staged the GPU JIT into the Windows wheel source, preserved Taichi `LICENSE`/`NOTICE`, and emitted `infernux-0.4.0-2-cp313-cp313-win_amd64.whl`; wheel/player contract tests passed **20/20**.
- Audited the emitted wheel: 940 files, no top-level `taichi/`, AOT, or SNode payloads; 40 private compiler entries under `Infernux/_compiler/taichi`; Taichi `LICENSE` and `NOTICE` present. Packaging contract tests passed **20/20**.

## Latest — 2026-09-14 screen UI native pose path

- Screen UI now records each submitted control's local vertex span with its Transform handle, then applies position/Z-rotation deltas during native publication. Pose-only motion no longer requires rebuilding CPU ImGui geometry, while layout, clipping, material, text and draw order remain unchanged.
- The pose pass uses an immutable per-list local-position snapshot rather than the previously transformed GPU array, so repeated movement/rotation cannot accumulate drift across frames.
- Transform scale is now part of the same native pose calculation: the control's local vertices are scaled around its recorded pivot, while layout sizing remains owned by the UI layout system.
- Windows Release rebuilt `_Infernux` and `infernux_screen_ui_vulkan_tests` successfully. Focused native tests passed **2/2**; focused Python UI regression passed **152 passed / 1 skipped**. No fallback or extra hash-validation path was introduced.
- This is an implementation step only: full-frame screen benchmarking and visible-editor verification are still required before claiming an FPS improvement. Scale/size changes, GPU upload granularity, cross-platform measurements and the remaining 041 checklist remain open. Strict progress remains **139/332 (41.9%)**.
- After the scale correction, the complete Windows native suite passed **90/90 in 94.89 s** and the complete Python suite passed **6352 / 12 skipped in 266.43 s**. The remaining diagnostics are expected negative-fixture coverage; no new failure was reported.
- Corrected pose upload invalidation: a screen Transform change now marks only the affected frame-slot vertex upload dirty, so native pose changes cannot be silently skipped by the geometry-revision cache. Windows Release rebuilt successfully and focused native tests passed **2/2** after the fix. Full-frame upload bandwidth and partial-range upload remain the next optimization boundary.
- Implemented the next upload step: pose-only screen changes now flush one merged vertex byte range covering dirty controls; topology/content changes retain the existing full vertex/index upload. Focused native tests remain **2/2** after the change. A full-frame bandwidth measurement is still required before checking off the performance item.
- Full Windows native validation after partial uploads passed **90/90 in 91.76 s**; focused Python UI validation passed **152/1 skipped in 3.81 s**. No RenderGraph, resource-retirement, Compute/JIT or multi-camera regression appeared.
- Added a native screen-bound packet regression: a retained screen control moved through Transform, its rendered pixels changed, and the pose-only path passed Vulkan readback. `infernux.screen_ui_vulkan` passed after the new case.
- Extended that regression to consecutive Z rotation and non-uniform Scale changes; both produce distinct Vulkan readback frames without rebuilding the retained packet.
- Current Windows Release native `--performance` baseline (validation disabled) records 1000 static screen labels at **0.0007 ms P50**, 1000 dynamic overlay labels at **0.0724 ms**, and 1000 dynamic world labels at **0.1512 ms**; 1000 textured controls with distinct descriptors record **0.0530 ms** (overlay) / **0.0787 ms** (world). These are native recording measurements, not full-frame FPS or GPU execution, and remain baseline evidence rather than a completed performance checkbox.
- RenderTexture/RenderGraph/UI resource regression after the screen upload changes passed **162 tests** (`test_ui_render_texture`, RenderGraph, render effects and declarative RenderStack inspector). No resource ownership or target-binding regression was found.

## Latest — 2026-09-14 screen UI bulk-animation audit

- Continued after the scene-bound world UI optimization. Added an explicit real-load benchmark for 1000 Canvas controls with 64 materials, changing every control's Transform each frame. Screen position animation measured **102.48275 ms P50** and rotation **100.1121 ms P50**; both extract 1000 elements and publish one packet batch. This is a Python submission benchmark and excludes native upload/GPU execution (`041-ui-screen-all-{position,rotation}.log`).
- An earlier parent-only screen benchmark was intentionally rejected as evidence: the fixture moved a Canvas owner that is not itself a screen geometry entry, so it left all child screen geometry unchanged and reported 0.32695 ms. The corrected per-control benchmark above demonstrates the actual cost. No screen-performance checkbox was marked.
- The result establishes the next implementation boundary: screen UI position/rotation is still baked into CPU ImGui vertices, unlike world UI's native scene-bound local geometry. A screen-native pose segment (or equivalent layout/pose separation) is required before claiming bulk screen animation. Layout/size invalidation, intrinsic text/flow changes, GPU upload granularity, full-frame/cross-platform measurements and the world-button MCP semantic discovery gap remain open. No fallback or compatibility retry was introduced.

## Latest — 2026-09-14 scene-bound world UI geometry

- Previous goal turn was verified progress (Transform-local invalidation). Continued the full041 goal and the user-prioritized UI path; newer builtin-Taichi and visible-editor decisions override stale plugin/headless wording. World UI already emits local-coordinate geometry, but any pose change still recaptured its text/shapes. Added native `BeginWorldObject`: packets retain a generational Transform handle, and publication resolves current world position/rotation and layer. Local geometry remains immutable. Explicit-matrix `BeginWorldElement` remains a distinct immediate/custom-authoring API, not a fallback. Own scale remains ignored by the UI contract; inherited parent motion/scale affects world position. Expired handles reject publication instead of binding a recycled object. Python no longer invalidates world geometry for pose-only edits; screen pose propagation is unchanged.
- Added a parent-animation regression: 60 world labels move/rotate with no measurement visit or packet capture, and output matches a fresh submission. It failed before the implementation. Updated test renderer doubles to model scene-bound commands resolved at publication, plus screen/world pose, parent, mixed-space and layer assertions. Focused **152 passed / 1 skipped**. The first focused command referenced a nonexistent dispatch test file and ran no tests; the corrected command above passed. No runtime compatibility shim was added for the old test doubles.
- Native Vulkan regression compares scene-bound retained packets with explicit fresh geometry pixel-for-pixel under existing sample configurations, rotation, parent scale, ignored own scale, changing layers/culling masks and uncommitted frame-cache positions. Visible frames are required to change and masked frames to remain clear. Destroy/recycle explicitly checks handle lifetime rejection. Added a 1000-object parent-animation native benchmark with identical-frame comparisons and exact capture counts (12000 recaptures versus 1000 initial captures over 12 frames). Windows Release rebuilt `_Infernux` and `infernux_screen_ui_vulkan_tests` successfully. Final complete native suite **90/90, 92.32 s**, 17 device-required tests (`041-ui-world-pose-full-native.log`), terminal exit0.
- Isolated Python submission, 1000 world labels /64 materials, rotating their parent: **57.54945 → 0.3992 ms P50**, extractions **1000 → 0**, no actual text measurements, one native batch (`041-ui-world-pose-before.log`, `041-ui-world-pose-after.log`). Both runs terminal success. Separate native same-backend A/B with validation explicitly off (`--performance`): build **0.8775 → 0.134 ms**, CPU render recording **0.1584 → 0.1523 ms** (`041-ui-world-pose-native-bench.log`). A first invocation used unsupported `--benchmark`, failed the argument assertion, and was corrected; this was not a renderer failure. Timings ran serially without editor/Hub/other tests. These are separate phase/workload measurements, not full-frame FPS or GPU duration.
- Final full Python **6352 passed /12 skipped, 256.74 s** (`041-ui-world-pose-full-python.log`). The additional skip is the screen variant of a world-only regression. The original exec handle65463 was unavailable after the continuation; authoritative final pytest summary and process enumeration confirmed completion, and the suite was not restarted. All test/build/helper sessions are terminal. Whitespace checks passed; editor/native logs have no new Traceback/native ERROR/VUID/Validation Error/DEVICE_LOST.
- Reopened ordinary visible editor **46132** with rebuilt modules. Real MCP checks passed independent warm screen/world movement and rotation, Chinese/font switches, parent transforms, immediate amber/blue shared material changes, four screen flow pointer callbacks (including parent-only motion), and actual world-button callback **0→0.75** after parent translation+rotation at observed editor point794,455. Human-reviewed engine-owned captures; no OS screenshot or algorithmic pixel processing. Console0warnings/0errors. Logs `041-ui-world-pose-{dependency,flow,material}-visible.log`, `041-ui-world-pose-world-{setup,click}.log`. Functional checks overlapped full pytest, not isolated timings.
- Helpers restored clean03 Edit and removed only their own temporary material through undoable asset deletion; no UITransientReview assets/metas remain. A later read-only project check found the user had moved the live editor to clean **12_OutlineOwners / Playing**. Preserve this live state; do not reopen03 or stop the user's scene. Hub **28240** reopened. No commit/push/release.
- Strict plan recount remains main103/256 plus compute36/76 = **139/332 (41.9%)**. World pose animation now reuses local geometry, but screen bulk-animation/layout invalidation, bulk visual-content updates, GPU upload granularity, full-frame/load/cross-platform matrix and Canvas-free MCP semantic discovery remain open. No whole-plan checkbox increment and no complete-goal claim.

## Latest — 2026-09-14 Transform-local UI invalidation

- Continued the user-prioritized screen/world UI path after the verified event-driven packet-group slice. Native geometry dependencies already compared individual poses, but Python included the aggregate geometry revision in every packet epoch. Moving one control therefore rebuilt every control. The existing native snapshot now publishes changed entry indices for its latest poll; Python invalidates only those consumers. Screen-space parent/child dependencies are built at membership changes (including ordinary non-UI bridges), so affected descendants update without invalidating unrelated branches. World entries retain native inherited-pose and layer detection. Removed the aggregate geometry revision from the packet epoch; the whole-list publication signature still observes it. No delta history, retry, fallback or new recovery manager.
- Added native changed-index ordering, unchanged-poll, frame-cache, world-layer and mixed screen/world checks. New Python tests cover individual position/rotation changes, parent propagation, unrelated branches, layers and mixed spaces. The first eight cases exposed seven failures against the old policy before the fix. Final focused Python: **250 passed / 1 skipped** (`041-ui-transform-local-focused-final.log`). Windows Release rebuilt `_Infernux` and the dependency test target successfully; final complete native suite: **90/90, 95.02 s**, including 17 device-required tests (`041-ui-transform-local-full-ctest.log`).
- Isolated A/B uses the same rebuilt native module and 1000-control/64-material workload. The baseline test-only switch restores the old aggregate geometry epoch policy; it is not an old-checkout comparison. For one position edit, Python submission P50 changed **screen 92.17135 → 0.44045 ms; world 57.95375 → 0.40970 ms**. Rotation: **screen 92.34880 → 0.43915 ms; world 57.85875 → 0.38085 ms**. Extraction count falls from 1000 to 1, with zero actual text measurements and one native packet batch. Evidence: `041-ui-transform-{global,local}-{position,rotation}-isolated.log`, four terminal-success runs. Measurements ran serially with editor/Hub/other tests closed, and exclude native geometry construction, upload and GPU execution. They do not establish full-frame FPS or the cost of animating all 1000 controls.
- Final full Python: **6351 passed / 11 skipped, 262.37 s** (`041-ui-transform-local-full-python.log`). The process has terminated; after context compaction its already-consumed exec handle was unavailable, so completion was rechecked from the exact final pytest summary and process list rather than rerunning the suite. Invalid shader/document fixtures emitted expected diagnostics. No test helper remains. Source whitespace check passed.
- Ordinary visible editor **50880** loaded the rebuilt module. Engine-owned captures and actual MCP input verified warm screen/world text movement and rotation independently, Chinese/font changes, parent transforms, immediate first/second material edits, and four screen flow clicks including a parent-only position change after caches were warm. Each click set the witness to 0.75. A world panel moved +1 X and rotated +20 degrees; clicking its newly observed button position (794,455) likewise produced 0 → 0.75. Logs: `041-ui-transform-local-dependency-visible.log`, `041-ui-transform-local-flow-visible.log`, `041-ui-transform-local-world-{setup,click}.log`, `041-ui-transform-local-material-visible.log`. Human-viewed engine captures; no OS screenshots or automated pixel analysis. These functional checks overlapped full pytest and are not timing evidence.
- Stop/reopen discarded temporary scene edits. Only the material created by the helper was removed through undoable asset deletion; no UITransientReview asset/meta remains. Final MCP project info confirms clean `03_WorldSpaceUI`, Edit, ready, loading=false. Console checks reported no warnings/errors, and editor logs contain no Traceback/native ERROR/VUID/Validation Error/DEVICE_LOST. Hub **30472** reopened and minimized; only this Hub and editor50880 remain.
- Strict checklist remains **139/332 (41.9%)**; no whole-item checkbox increment. Remaining UI work includes size/intrinsic/flow-layout invalidation scope, many simultaneously animated controls, native upload granularity, complete frame/load/cross-platform measurements, and Canvas-free world-button MCP semantic discovery. Actual pointer success does not resolve semantic discovery. Full041 remains open. No commit, push or release.

## Latest — 2026-09-14 event-driven UI packet groups

- The preceding turn revalidated the existing UI result but added no implementation. This turn resumed the user-prioritized screen/world UI path, keeping the full041 goal intact and the newer builtin-Taichi/visible-editor decisions authoritative. Strict checklist re-count: main103/256 plus compute36/76 =139/332 (41.9%). No complete-plan checkbox change.
- Replaced per-element warm-cache polling with ordered, target-owned command groups. Elements hold weak subscriptions to their live groups; edits dirty only the affected indices. The existing resource publication notifies actual material/RenderTexture consumers. Each owner retains its own pending changes, including when it misses a frame. Unchanged geometry is appended as a batch without1000 Python measurement/key-check calls. Membership/target/font/shared state still resets groups, and a changed measured layout updates sibling geometry before publication. Removed the unused per-element revision counter and the previous measurement/packet-key maps instead of keeping two mechanisms.
- Custom renderers still receive the real renderer in authored order. A new regression exposed a custom draw changing a preceding sibling's width while a later cached sibling stayed at its old position; observing the layout revision at the custom-draw boundary fixes this within the same publication. Already-drawn elements remain dirty for the next publication. Failed capture discards the incomplete group, with no stale-packet fallback or immediate retry. Weak subscriptions release with their owner; no global edit history or recovery manager.
- New screen/world tests cover one changed element being the only measurement visit (both failed against the previous implementation), two independent submission owners with missed frames, weak-group release, and layout mutation during a custom draw. Focused233 passed/1 skipped (`041-ui-dirty-batches-focused-final.log`); the final obsolete-counter cleanup also passed233/1 (`041-ui-dirty-batches-delivery-focused.log`). The custom-layout regression was red before its fix. Existing custom order, font/hidden-font, intrinsic sizing, material aliases/document restoration/removal, RenderTexture, lifecycle, hierarchy and partial-capture cases remain green.
- Isolated1000-control/64-material Python submission P50 (`041-ui-dirty-batches-label-isolated.log`): color/text edits screen0.4336/0.4457ms, world0.38515/0.4165ms, versus1.0884/1.0631 and0.81785/0.83770ms. One extraction,0/1 actual text measure, one native batch. Material0 edits (`041-ui-dirty-batches-material-isolated.log`) screen1.5343/world1.48655ms versus2.0850/1.8084ms; exactly16 consumers extracted. Benchmarks exclude native/GPU time, run serially without editor/Hub/other tests, and are not full-frame FPS. No native changes or new CTest claim in this slice.
- Initial full Python before the final dead-counter cleanup:6341 passed/11 skipped,267.46s, terminal exit0 (`041-ui-dirty-batches-full-python.log`, session99178). **Final source full Python:6341 passed/11 skipped,256.87s, terminal exit0** (`041-ui-dirty-batches-delivery-full-python.log`, session88661). Deliberately invalid shader/document fixtures print expected diagnostics after the summary; no test failed. All benchmark, helper and pytest handles are terminal.
- Ordinary visible editor35844 passed realMCP material first amber/second blue updates, three flow-layout pointer callbacks, world-button callback0→0.75 at observed editor coordinates835,455, Chinese/font/parent-pose changes, Stop and clean reopen. Human-viewed engine-owned captures; no OS screenshots or algorithmic pixel analysis. Console0warnings/0errors and native log scan clean. Only the helper-created material was deleted through undoable asset deletion. This editor was then closed normally for the final source refresh; final editor45824 reopened clean03 and repeated the material test successfully, deleting only its own temporary material. Functional checks overlap full pytest and are not timing evidence.
- Final editor45824 repeated all visible checks on the cleanup version: immediate material updates, screen horizontal/resized-Fill/vertical real pointer callbacks0.75, world button0→0.75, Chinese/font/parent-pose changes (`041-ui-dirty-batches-delivery-*-visible.log`, `041-ui-dirty-batches-delivery-world-click.log`). Human-viewed the engine-owned editor and final font-switch captures. Console0warnings/0errors; editor log scan has no Traceback/nativeERROR/VUID/ValidationError/DEVICE_LOST. Temporary authoring was discarded via Stop/reopen; only its own temporary material was deleted through undoable asset deletion, and no UITransientReview asset/meta remains. Final03 is clean Edit/ready/loading=false. Hub55632 reopened/minimized. Only editor45824 and Hub55632 remain; no surviving test helper. Source whitespace check passed. Native source unchanged; previous90/90 is still the latest native suite, not a new test result.
- Remaining: complete UI load/full-frame/cross-platform matrix, native upload granularity and layout/Transform-local invalidation. The separate world-button MCP semantic-discovery gap persists: GameViewPanel's semantic publication enumerates Canvas elements only, whereas real input already includes Canvas-free world surfaces; actual pointer success does not close discovery. No commit, push or release.

## Latest — 2026-09-14 UI binding membership and warm packet lookup

- Previous goal turn was verified progress: native retained geometry and visible UI regression. Continue the full041 goal, following the newer builtin GPU-compiler/visible-editor decisions rather than the stale plugin/headless wording in the goal boilerplate. UI remains the user-prioritized path; no full-plan checkbox increment.
- Profiling showed every label/color/hover edit rebuilt all material/image binding membership and reacquired serialized-field storage, even when no resource was reassigned. Separate resource-binding edits within the existing UI revision module; ordinary visual/size/hover updates now keep the binding tuple. Topology, material/text-material/texture assignments and explicit global invalidation rebuild membership. Retain the actual raw references, polling current owners/versions each publication so unresolved aliases, hot resource replacement and RenderTexture revisions still work. Consumer resource tuples are rebuilt only when their resource states change, not on an unrelated visual edit. No representative-owner cache that hides independently replaced aliases, no retries, compatibility fallback or checksum validation.
- Warm native packets and text measurements already imply a retained builtin renderer. Move capability lookup after the existing warm-cache hit instead of repeating it for all controls; public renderer registration already invalidates the shared epoch. Custom renderers still execute directly in order. Canvas scope uses its identity; Canvas parameters, topology and target size already belong to the invalidating epoch, so avoid hashing the entire layout argument tuple for every control.
- Added screen/world regressions: text/color/width/hover preserve binding membership (8 cases failed before fix), independently replaced material reference owner, typed document material assignment/removal, warm renderer capability lookup only for the changed element (2 failed before fix), and public custom renderer replacement/restoration. Two older flow-only fakes used unhashable SimpleNamespace rather than real component identity; updated them to an identity-based test stub instead of adding a runtime fallback. A new-test assertion incorrectly checked extraction counts after intentionally fresh comparison, and its edit briefly introduced indentation trouble; both corrected. Final focused225 passed/1 skipped (`041-ui-bindings-focused-delivery-final.log`).
- Final isolated Python submission with1000controls/64materials (`041-ui-bindings-label-final-isolated.log`): color/text edits **screen1.0884/1.0631ms, world0.81785/0.83770ms**, versus prior2.7640/2.7524 and2.19095/2.29570ms. Exactly1 extraction,0/1 text measurement and1 native packet batch. Shared material0 edit **screen2.0850/world1.8084ms** (`041-ui-bindings-material-final-isolated.log`), versus3.07905/2.35655; exactly16 consumers extracted and1 batch. Four+two cases passed, terminal exit0. These exclude native tessellation, GPU upload/record/execute and cannot be converted to full-frame FPS. Editor44932 and Hub8512 were closed normally before the isolated runs; no tests or builds overlapped those measurements.
- Final complete Python: **6333 passed / 11 skipped, 266.17 seconds, terminal exit0** (`041-ui-bindings-full-python.log`, session5620). Native source/module unchanged in this slice, so no new CTest claim; the previous90/90 native result remains the last native suite. All benchmark, helper and pytest sessions are terminal.
- Reopened ordinary visible editor **49812** on clean Scene03. Real MCP material assignment and immediate first amber/second blue edits worked for screen text, world text and world-button text (`041-ui-bindings-material-visible.log`). Human-reviewed engine-owned Scene/Game captures confirm the colors. Screen flow horizontal/resized-Fill/vertical real pointer callbacks each returned0.75 (`041-ui-bindings-flow-visible.log`); the world button likewise produced0→0.75 at observed editor coordinates835,455 (`041-ui-bindings-world-click.log`). Repeated Chinese/font-switch/parent-pose review (`041-ui-bindings-font-visible.log`) and inspected the engine capture. No OS screenshots or algorithmic processing of human-review-only captures. Functional editor checks overlapped the full suite, not isolated performance measurements.
- Console0warnings/0errors and no Traceback/nativeERROR/VUID/ValidationError/DEVICE_LOST in editor logs. Stop/reopen discarded temporary scene edits; only the material created by the review was removed through undoable MCP asset deletion, with no UITransientReview assets/metas remaining. Scene03 is clean Edit/ready/loading=false. Hub reopened/minimized as **51784**; only that Hub and editor49812 remain, no pytest/helper processes. FullUI/041, GPU partial upload/full-frame/cross-platform verification and the separate world-button MCP semantic-discovery gap remain open. Strict plan count139/332 unchanged; no commit/push/release.

## Latest — 2026-09-14 retained native UI geometry and batch publication

- Final complete Python against the final native module: **6317 passed / 11 skipped, 273.43 seconds, terminal exit0** (`041-ui-packets-full-python-final.log`, session56763). The test's deliberately invalid shader/document fixtures print expected diagnostics after the summary; no test failed. All build/benchmark/test handles have completed. Isolated benchmarks ran before this full suite and before reopening the editor; visible functional checks later overlapped the tail of the full suite and are not used as timing evidence.
- Reopened ordinary visible editor **44932**; verified non-minimized SDL editor through its own process metadata. Real MCP material review passed first amber and subsequent blue updates for screen/world/title/button text; human-viewed engine-owned Game/Scene captures show the changes (`041-ui-packets-material-visible.log`). Repeated CJK/font switch and parent Transform captures (`041-ui-packets-font-visible.log`), inspected the resulting Chinese and transformed panel. Three real screen flow callbacks (horizontal, resized-Fill, vertical) each changed the witness to0.75 (`041-ui-packets-flow-visible.log`); actual world-button pointer input at the observed editor coordinate835,455 likewise changed0→0.75 (`041-ui-packets-world-click.log`). These use no OS screenshots or algorithmic analysis of human-review-only captures.
- Console0warnings/0errors and no Traceback/nativeERROR/VUID/ValidationError/DEVICE_LOST in the editor logs. Stop/reopen discarded transient scene authoring. Only the material created by this review was deleted through the undoable asset operation; no UITransientReview assets/metas remain. Scene03 is clean Edit/ready/loading=false. Hub reopened as **8512** and minimized; only it and editor44932 remain, with no surviving pytest/helper process. World-button MCP semantic discovery still returns[]; actual pointer success does not close that separate known gap. CompleteUI/041 remains open and strict progress139/332 unchanged.

- Final isolated Python submission (native tessellation/upload/GPU excluded), 1000 controls/64 materials: changing one label color/text **screen 2.7640/2.7524 ms, world 2.1910/2.2957 ms**, one extraction/AddText and one native packet batch; editing material0 **screen 3.0791/world 2.3566 ms**, exactly 16 consumer extractions and one batch (`041-ui-packets-label-final-isolated.log`, `041-ui-packets-material-final-isolated.log`, 4+2 cases passed, terminal exit0). The corresponding prior argument-replay baselines were 2.9763/3.0082, 2.5406/2.6116 and 3.2080/2.8170 ms, respectively. Remaining Python O(N) work is not eliminated; this slice's larger benefit is avoiding unchanged native tessellation and repeated binding calls.
- The first full Python run was **6316 passed/11 skipped/1 failed**: a pre-existing unit-test Renderer fake still implemented only AddText and lacked the new packet protocol. Updated the fake and explicit flush assertions, without adding a production compatibility fallback; the entire test_ui_logic module then passed **134/1 skipped**. A final full run is tracked separately and is not declared green until its terminal result. LastTest.log scan found no VUID/Validation Error/DEVICE_LOST findings. Web's existing submission calls the common static canvas helper without packets, so this desktop-native retention does not require or claim a new Web adapter.

- Previous goal turn was verified progress (resource-local UI invalidation and the first-edit Color/Float4 fix). Continue the full041 goal and user-prioritized screen/world UI work; newer builtin GPU compiler and visible-editor decisions supersede the older goal's plugin/headless wording. No full-plan checkbox increment.
- Replaced Python command-argument recording/replay with immutable native geometry packets. Changed built-in controls use the same existing AddText/AddImage/AddFilledRect tessellators; unchanged controls reuse their vertices, indices, clip/texture commands, HDR ranges and world element metadata. Python publishes ordered packet batches rather than replaying every draw argument through the binding. Custom renderers retain the real native renderer and flush pending packets at their authored position. There is no parallel legacy recorder/fallback path.
- Renderer-owned scratch ImDrawLists are reused during capture; retained packets hold only plain CPU arrays and freeze texture IDs, not ImGui atlas-data pointers or GPU owners. Packets can be released after renderer/ImGui shutdown. Appending preserves shared 16-bit vertex buckets and merges compatible screen commands; world element boundaries remain intact for the existing camera-specific depth/layer ordering. Large packets retain their own base-vertex splits. GPU frame-slot buffers and resource retirement remain the existing main path; this slice does not yet add partial GPU uploads.
- Font atlas/default-font/cache generations invalidate native and Python snapshots. A changed font also invalidates intrinsic text measurement, including text hidden during font replacement and reactivated later. Failed capture aborts its unpublished packet and invalidates the incomplete native frame, with no retry or stale-geometry fallback. Null packet/uninitialized renderer rejection is at the native API boundary, not a per-frame validation service.
- Expanded the actual Vulkan test: fresh versus retained GPU readback across 20 dynamic frames/1000 labels or 64 alternating textures; Camera/Overlay/World clips, rotation/mirror, HDR, alpha, mixed direct/retained drawing, world culling, large multi-base-vertex geometry, abort/empty capture, font invalidation and CPU-packet release after context shutdown. Retained dynamic text still draws once, while 64 alternating textures retain the expected1000 draws. Capture count is1000 initially plus two changed labels per subsequent frame, not1000 retessellations each frame.
- Validation-off same-binary A/B (`041-ui-packets-native-performance.log`): 1000 dynamic text native build P50 **screen0.5067→0.0391ms / world0.5287→0.0579ms**; native render recording remains~0.06/~0.15ms. 1000 images with64 alternating descriptors build **screen0.0348→0.0226 / world0.0572→0.0382ms**, render record~0.05/~0.08ms. These exclude Python extraction, driver execution and GPU elapsed time; not full-frame FPS. Validation-on counterparts are separately recorded, not mixed into production performance claims.
- Full native CTest90/90 in93.58s (`041-ui-packets-full-ctest.log`). Final build after formatting and capture-count assertions passed (`041-ui-packets-delivery-build.log`); final validation-enabled ScreenUI Vulkan test passed0.89s (`041-ui-packets-delivery-vulkan.log`). Python focused initial76passed/2failed were new-test author mistakes (UIButton exposes label, not text); corrected focused78passed, with no engine compatibility alias added. Added packet batching/font changes, abort, custom interleaving and hidden-font-metrics cases. Final full Python, isolated submission and visible editor results follow only once observed.

## Latest — 2026-09-14 resource-local UI packet publication

- Continued both screen and world UI after the user accepted the Outline main path at 600+ FPS. Shared material versions are inspected once per unique resource within each publication, including unresolved GUID aliases. The existing dependency snapshot now also identifies each element's material/text-material/RenderTexture inputs. Only topology and inherited geometry/layout changes clear all packets; a material or image revision rebuilds its actual consumers. Native list publication still observes the complete resource dependency set. No extra fallback, retry, hash verification or per-control polling service.
- Added nine Python regressions: shared/independent material owners, unresolved GUID aliases, background and text material changes/rebinding, RenderTexture resize and per-consumer extraction in both screen/world modes. Retained draw command tuples must exactly match a fresh submission. Focused suite 72 passed (`041-ui-material-local-focused-final.log`); the full Python run before the additional native material fix passed 6306/11 skipped in 255.58 seconds, terminal exit 0 (`041-ui-material-full-python.log`).
- Isolated 1000-control/64-material Python submission: shared material polling initially reduced a one-label edit from screen 4.34–4.65/world 3.95–3.98 ms to screen 2.62–2.63/world 2.21–2.21 ms. More importantly, editing one shared material previously extracted all 1000 controls (~62 ms); local resource keys extract its 16 consumers (~3.24 screen/2.73 world ms). These exclude native tessellation, upload and GPU time; final isolated measurements are recorded below, not converted into FPS.
- Extended the real Vulkan UI test with dynamic 1000-text and 64-texture submission cases. Default tests retain validation; explicit `--performance` disables it only for measurement. Validation-on 1000 alternating image draws cost ~2.96/2.84 ms screen/world to record; the same validation-off path is ~0.051/0.081 ms. This is a measurement correction, not an engine optimization. Validation-off dynamic text geometry builds remain ~0.51/0.53 ms (`041-ui-dynamic-native-perf-baseline.log`); they are still remaining work.
- Visible MCP material review caught a bug despite green unit tests: a newly created .mat bound to screen text, world text and world-button text stayed white after its first amber edit; the second blue edit worked. Waiting for asset IO did not change this. Temporary launch-only tracing proved the live material changed to amber/version2, then pipeline refresh reset it to white/version3. UI invalidation was functioning. Project material creation incorrectly declared baseColor as Float4 while Unlit declares Color; shader-default synchronization discarded the matching numeric value because its semantic tag differed.
- Fixed the authoring template to emit native Color semantics. Native shader synchronization now preserves the same vec4 value while adopting the shader's Color/Float4 semantic type; genuinely incompatible shapes still use the declared typed default. Added native bidirectional transition/no-op/version/serialized-type coverage, proven red before the fix (`041-ui-material-transition-red.log`), plus a Python template/native-contract regression. No permissive conversion of unrelated types. Final rebuild passed (`041-ui-material-final-build.log`). The MCP helper now inspects the actual live material after publication instead of trusting the requested-document echo; temporary delay was removed. Final suites and human-viewed first-edit captures follow below once verified.
- Final full native CTest: **90/90 passed in 99.74 seconds**, including 17 device-required tests (`041-ui-material-final-ctest.log`); LastTest.log has no VUID/Validation Error/DEVICE_LOST findings. The UI Vulkan test itself explicitly enables validation in this default run; no claim that this command force-enabled validation for every other test.
- Final isolated Python submission P50 with 1000 controls/64 materials: one shared material edit **3.2080 screen / 2.81695 world ms**, exactly **16** extractions (`041-ui-material-edit-final-isolated.log`, 2 cases passed). One label color/text edit **2.9763/3.0082 screen**, **2.5406/2.6116 world ms**, exactly **1** extraction (`041-ui-material-label-final-isolated.log`, 4 cases passed). The per-element resource map adds some cost relative to the intermediate polling-only 2.2–2.6 ms result, but prevents a shared edit from rebuilding all 1000 controls. All cases still issue 1000 draw calls to the native API; native tessellation/GPU time is excluded. Benchmarks and complete suites ran serially, with Hub/editor normally closed before the native rebuild.

- Final complete Python against the final native module: **6307 passed / 11 skipped in 268.56 seconds**, terminal exit 0 (`041-ui-material-final-python.log`). Intentional invalid-document/shader fixtures print diagnostics after the summary; no test failed. All benchmark/build/test sessions are terminal.
- Reopened the ordinary visible editor PID52216 with the final module and Hub PID4036. Cold Scene03 was clean/ready before Play. MCP material authoring now passes the immediate post-publication live-document assertion with no added delay; human-reviewed engine captures show the **first amber edit and second blue edit** correctly on screen text, world text and world-button text (`041-ui-material-fixed-visible.log`, `review/041-ui-material-amber-game.png`, `review/041-ui-material-amber-scene.png`, `review/041-ui-material-blue-game.png`). The temporary tracing launcher was removed, not retained in production.
- Repeated real screen layout/button callbacks in horizontal, resized-Fill and vertical arrangements: each actual pointer click changes the witness from 0 to 0.75 (`041-ui-material-flow-final.log`). Repeated the actual world-button pointer click at the human-observed editor position, with the same 0→0.75 callback (`041-ui-material-world-final-click.log`). Captures came from engine render targets, not OS screenshots or pixel analysis. Console0warnings/0errors and no native/traceback/validation diagnostics in the editor logs. Temporary .mat assets and their .meta files were deleted through the undoable asset operation; temporary scene changes were discarded with Stop/reopen. No UITransientReview files remain. Only editor52216 and Hub4036 remain; Scene03 is clean Edit/ready/loading=false.
- Remaining: native per-element command/geometry reuse, dynamic text geometry, dense resource binding and cross-platform/full-frame scalability. The world-button MCP semantic discovery gap from the earlier slice is still open; real pointer response is not a claim that semantic discovery is fixed. No commit/push/release or complete041 checkbox increment; strict full-plan progress remains139/332 (41.9%).

## Latest — 2026-09-14 UI owner resolution and resize authority

- Continue the screen/world UI path after the accepted Outline stage. Reuse the already-validated owner, Canvas resolution and Transform inside each `get_rect` query; remove two private helper indirections. The existing hierarchy/version cache and native lifetime validation remain authoritative. Reference-field reads reuse one immutable type set instead of constructing it for every scalar field access. No new guard/fallback/hash layer or C++ change.
- The 300-control flow profile drops owner lookups from 2106 to 605 and enum hash calls from 16905 to 2415. Intermediate isolated CPU layout is 2.6407/7.2156 ms for 100/300 labels (`041-ui-rect-profile.log`), compared with the preceding slice's final 4.2650/12.4507 ms. Each group still measures exactly 100/300 desired sizes. These are CPU layout times, not GPU or complete frame times; final isolated evidence is recorded below when complete.
- Converted old unattached/mock screen-UI tests to real Canvas/Transform fixtures. This exposed a real Inspector bug: a document-only size edit does not include the native position needed to preserve the rotated top-left corner. Resize now submits width, height and Transform position through the existing compound property command, without a temporary serialized-document round trip. World UI changes its own geometry without moving its Transform. Six regression cases cover 0/37/90-degree screen/world resizing, one Undo action, restored position/size on Undo and Redo, and preserved Z. Focused Inspector/layout 105 passed; prior focused Inspector/submission 93 passed. First full run: 6290 passed, 11 skipped, one failure at this newly real resize fixture. Do not treat that first run as green; the final full rerun follows below.
- Added two native-scene tests for a single owner resolution on cached rect reads, live screen/world reparenting and destroyed-owner invalidation. Updated the static-dependency test to reject actual `get_rect` traversal instead of patching a removed helper. Earlier focused UI 203 passed/1 skipped.
- Visible editor 50620 exercised screen flow horizontal/resized-Fill/vertical with actual callbacks and world font/text/parent transforms. The world button set a witness 0→0.75 through real pointer input. Engine-owned captures were inspected. After the Inspector fix, normally closed that editor via MCP and opened visible editor 53780; repeated cold Scene03, fonts, parent transforms and all three screen-flow callbacks with Console 0 errors/warnings. Stop/reopen discards temporary objects and leaves the authored Lab unchanged in clean Edit. MCP world-button semantic discovery still returns no targets; actual pointer interaction works, and that separate semantic gap is not claimed fixed.
- Final full Python **6297 passed / 11 skipped**, 261.73 seconds, terminal exit 0 (`041-ui-rect-full-python-final.log`). Final isolated 100/300 flow CPU layout P50 **2.3796 / 7.0758 ms**, exactly 100/300 desired-size evaluations, two cases passed (`041-ui-rect-final-isolated.log`). The full suite's intentional invalid-document/shader cases print native errors after pytest's summary; its terminal result is green. Visible editor 53780 is in Scene03 ready/clean Edit, Hub 6204 remains intact, and pytest has exited. This editor run has no traceback, native error or Vulkan validation error. No new native CTest run, commit, push, release or full-plan checkbox increment in this slice. Complete UI scalability and 041 remain open.

## Latest — 2026-09-14 batched automatic layout

- Previous goal turn made verified progress in shared font resolution/layout. Continue the accepted Outline stage into UI performance; complete 041 remains active, with current user decisions (visible MCP editor and builtin GPU compiler) taking precedence over older goal boilerplate. No completion/checklist increment in this slice.
- Found quadratic sibling work in `UIFrame`: each child's rectangle query rediscovered and measured the entire flow group, then repeated measurement while finding the requested child. A fixed-size frame also measured descendants whose sizes could not affect it. The existing layout cache now holds one arranged group and nested Hug intrinsic sizes, invalidated at the same geometry/publication boundaries as rectangles. A fixed frame skips unused intrinsic descendant walks. Each arrangement reuses its desired-size array for main/cross axes. Existing constrained Fill redistribution, gaps, padding, anchors, justify/align, absolute children and world-UI rules are preserved; no new hashes, polling, retry or fallback layer.
- Native-scene CPU layout benchmark (no drawing/GPU): 100/300 flow labels, change gap and clear the layout epoch each iteration. Baseline **128.4638 / 1123.4706 ms**, **15150 / 135450** desired-size evaluations (`041-ui-flow-baseline.log`). Batched result **4.1300 / 12.0565 ms**, exactly **100 / 300** evaluations (`041-ui-flow-batched.log`). This removes repeated sibling traversal but does not claim complete UI scalability; per-element Transform/rect extraction and native geometry work remain open.
- Added 14 integration cases to `test_ui_auto_layout.py`: late-child-first query, reuse/invalidation, horizontal/vertical groups, nested Hug bounded measurement counts, padding/direction/Fill/absolute/intrinsic/viewport changes against fresh layout, native create/destroy/reparent/parent motion publication, and world UI explicitly bypassing screen arrangement/clipping. Final focused layout **27 passed**; preceding layout+command packet run **59 passed** predates the four native lifecycle cases. A first combined invocation mixed dev's explicit fixture plugin with python/test's conftest and attempted to initialize AssetRegistry twice (49 passed, two harness errors); running the dev benchmark separately resolved that test invocation issue, with no engine workaround.
- Normally closed editor 46764 through MCP and opened visible editor 18232 with the new Python code; Hub 6204 remained intact. Real Scene03 temporary flow controls changed from horizontal to resized+Fill to vertical, with engine-owned captures reviewed. A persistent UIButton callback moved with layout and successfully set a temporary witness 0→0.75 at all three distinct hit coordinates (`041-ui-flow-click-review-final.log`). First helper run incorrectly wrote the witness's already-zero value; MCP rejects unchanged writes, so the helper now skips that redundant first reset. This was not an engine failure. Stop/reopen removed all temporary objects/bindings without saving authored Lab assets; Console 0 warnings/errors and scene ready/clean Edit.
- Full Python terminal result and final isolated benchmark follow below. No C++ production change/rebuild or new CTest run is claimed here; prior native 90/90 belongs to the preceding turn. No subagent, commit, push or release.

- Final full Python **6289 passed / 11 skipped**, 274.31 seconds, exit 0 (`041-ui-flow-full-python.log`). Final isolated 100/300 flow layout P50 **4.2650 / 12.4507 ms**, exactly 100/300 desired-size evaluations, two benchmark cases passed (`041-ui-flow-final-isolated.log`). The earlier 4.13/12.06 ms values remain above as intermediate evidence; neither run is total frame time or a completed dynamic UI performance gate. Visible editor 18232 is ready in Scene03 clean Edit, Hub 6204 remains restored, and test subprocesses have exited. No new traceback/native error/Vulkan validation error in this editor run. Current full-plan scope and prior strict checklist count remain unchanged.

## Latest — 2026-09-14 shared UI font resolution and glyph layout

- Continue screen/world UI optimization after the accepted Outline 600+ FPS stage. This slice changes the shared native text path, not project assets, public UI APIs or the full 041 completion status.
- Found an expensive main-path mistake: even an already-loaded explicit font performed physical filesystem resolution twice per layout. Look up the existing cache using a lexical absolute key first, resolve the physical path only at the loading boundary, and retain canonical aliases in the same cache. Relative paths still follow the current working directory. Existing font-cache clearing remains the reload boundary; no new hash, file polling, retry or fallback layer.
- Layout now retains decoded codepoints, selected faces, sizes and advances in its transient result. Rendering consumes those metrics rather than decoding, selecting, measuring and reading SFNT tables for every glyph again. Compute fallback face sizes once per layout and reuse whitespace-run endpoints during wrapping. Keep ImGui's live glyph/atlas rendering; no long-lived UV cache. Removed `FontDataOwnedByAtlas=false` from file font loading: ImGui allocates those bytes, and that flag caused a clone while abandoning the original allocation.
- Enabled assertions in the Release native text test. Expanded it with 960 layout/draw geometry cases covering explicit Latin/CJK chains, wrapping, spacing, alignment, line height, CR/LF, malformed UTF-8, embedded NUL, emoji and long whitespace runs. Independently re-decode and measure each resulting line, then compare positions/UV/colors and vertex/index counts against retained-glyph rendering. This is not a separate legacy line-break oracle. Relative-path aliases, changed working directory, font clearing/reload and ImGui allocator accounting also pass; loaded file font data is freed on atlas destruction.
- CPU-only benchmark: 1,000 explicitly file-fonted labels, layout plus glyph tessellation every frame, 20 iterations with four warm-up iterations. Baseline P50 Latin **463.222 ms**, mixed Latin/CJK **747.225 ms** (`041-ui-text-baseline.log`); removing repeated physical resolution alone **4.6151 / 5.2747 ms** (`041-ui-text-resolve.log`); final focused run **2.3992 / 3.2588 ms** (`041-ui-text-final-focused.log`). These are not Vulkan recording, GPU execution or whole-editor frame times, and the extreme baseline specifically includes explicit file-font resolution. Do not generalize its speedup to default-font UI or FPS.
- Complete native build confirmed exit 0 (`041-ui-text-build-confirm.log`); complete CTest **90/90**, including 17 Vulkan tests, 105.17 seconds (`041-ui-text-full-ctest.log`). Full Python suite and final isolated benchmark are recorded below after their terminal results.
- Restored visible source editor 46764 and Hub 6204 after normal shutdown for native replacement. In real Scene03, cold Game/Scene display, live Latin/CJK fallback, switching to another explicit face without restart, parent movement/rotation and actual world-button callback 0→0.75 passed through MCP. Engine-owned captures reviewed under `review/041-ui-text-*`; no authored Lab changes saved. World semantic button discovery still returns no targets, so the separate pointer/callback assertion supplies interaction evidence, not a discovery-fix claim. Remaining UI gates include frequent relayout, per-element native geometry updates, many distinct resources and editor/Player platform matrices. No new 041 checkbox, commit, push or release.

- Final complete Python regression **6275 passed / 11 skipped**, 268.61 seconds, exit 0 (`041-ui-text-full-python.log`); trailing native shader/type rejection diagnostics are intentional negative fixtures. Scene12 all six actual screen buttons and Stop/reopen passed, graph topology stayed stable, Console 0 warnings/errors (`041-ui-text-screen-review.log`). Reviewed the pressed world button and restored screen UI captures. Final isolated text benchmark after other tests exited: Latin **2.3867 ms**, mixed Latin/CJK **3.2540 ms**, all 960 geometry cases passed (`041-ui-text-final-isolated.log`). Final Scene03 is ready, clean Edit; only the visible editor 46764 and restored/minimized Hub 6204 remain, no test subprocess. This run's editor logs contain no traceback, Vulkan validation error or native `[ERROR]`. Source whitespace checks pass. Full UI/041 remains open; the strict checklist was not recounted or incremented.

## Latest — 2026-09-14 retained native UI geometry and sorted world batches

- User accepts the Outline 600+ FPS main path for this stage; screen/world UI performance remains the active priority. This slice does not close all UI workloads or the full 041 goal.
- Native renderer previously converted and uploaded unchanged draw lists on every camera render. Retain converted geometry by the existing publication path, upload each content revision once per engine-owned frame slot, and reuse it across cameras. Frame buffers now match VkCore's configured frame count (1–4), using its completed-slot contract; no new waits, hashes, polling or recovery layer. Flush noncoherent VMA allocations at the upload boundary. World geometry remains camera-independent; camera depth is computed once per element before stable sorting.
- World element boundaries previously forced separate Vulkan draws even for contiguous compatible geometry. Merge adjacent index ranges after camera sorting only when texture, clip and base vertex match. Keep blend/primitive order, distinct clips/textures, layer masks and 16-bit base-vertex boundaries. Also remove empty screen draw submissions.
- New real Vulkan test `infernux.screen_ui_vulkan` checks pixel readback, clear/disable/reenable, four-slot retained replay, two differently colored/positioned slot recordings before GPU execution, reversed camera depth, masks, stable equal-depth alpha blending, clip and texture batch boundaries, and 4,000 labels crossing the 16-bit index range. Initial baseline fixture mistakenly attached screen depth and omitted the final imported-color read state; corrected before collecting a valid baseline. The first batching assertion exposed an existing empty screen draw; production now skips it.
- Valid baseline for 1,000 same-font labels: Camera 0.0892 ms, Overlay 0.0876 ms, World 2.6491 ms. Retained geometry alone: 0.0136 / 0.0144 / 2.3141 ms. Compatible world batching: 0.0176 / 0.0122 / 0.0255 ms, one nonempty draw per list. These are CPU Vulkan command-recording P50 measurements with validation enabled, excluding layout, tessellation, GPU execution and total editor frame time; the world fixture is deliberately batch-compatible. Logs: `041-ui-native-baseline-2.log`, `041-ui-native-retained.log`, `041-ui-native-batched-2.log`. Final isolated rerun follows full tests below.
- Final Release build exited 0. Complete CTest **90/90 passed**, including **17 Vulkan tests**, 106.83 s (`041-ui-native-full-ctest.log`). Complete Python **6275 passed / 11 skipped**, 272.59 s, exit 0 (`041-ui-native-full-python.log`). Native error lines after the Python summary are intentional negative-fixture diagnostics, not failing tests.
- Normally closed editor 50568 and Hub 46436 before native rebuild, then restored visible editor 32236 and Hub 13196. Real 03_WorldSpaceUI cold Scene/Game display, live text/font and parent TRS changes passed; actual world pointer-down/up changed the transient witness from 0 to 0.75 and showed pressed tint. Real 12_OutlineOwners six screen buttons, state restoration and Stop/reopen passed; Console has 0 warnings/errors. Engine-owned captures `review/041-ui-native-*` reviewed. Temporary Play changes were discarded; no Lab asset edits saved. World semantic-target discovery still returns an empty list (pre-existing open item, not counted as fixed). No commit/push/release or subagent in this slice.

- One existing MCP Windows transport issue recurred after successful client disconnect: asyncio Proactor `_call_connection_lost` reports `ConnectionResetError` / WinError 10054 in the editor stderr. All requested operations/captures completed and engine Console remained clear. Do not describe this run's entire stderr as error-free or count that separate transport issue as fixed.
- Final isolated Vulkan replay after all suites: Camera **0.0112 ms**, Overlay **0.0112 ms**, World **0.0277 ms**, each 1 compatible nonempty draw (`041-ui-native-final-isolated.log`, passed). All pixel/batching/frame-slot assertions passed, including the 4,000-label range split. Real 12_OutlineOwners three 240-frame windows average **0.967–0.982 ms**, P95 **1.77–1.78 ms**; GUI median is 0 because existing idle frames reuse GUI work, so these are not a GPU-only timing or fully redrawn-editor FPS claim (`041-ui-native-outline-profile.log`). Stop/reopen restored 03_WorldSpaceUI to clean Edit. Only editor 32236 and Hub 13196 remain; test processes exited. No new 041 checkbox is marked.

## Latest — 2026-09-14 retained UI draw arguments

- Outline600+FPS remains stage accepted. This continuation targets screen/world UI dynamic Python command construction, not the whole041 completion gate.
- New baseline uses1000 real native scene UIText components, with only one label changing color or text each frame. The renderer sink counts submitted commands and measurement calls, excluding native tessellation and GPU rendering. P50screen/color56.41685ms,world/color47.66895ms,screen/text56.89605ms,world/text48.89690ms (`041-ui-dynamic-baseline.log`,4passed). Each change still traversed1000 components even though native font measurement ran0/1times.
- Replaced the separate text/image/button extraction caches with one submission-owned built-in UI command packet cache. Existing scene/pose/resource dependencies form the shared epoch; per-element visual revisions preserve unchanged packets. Group/Canvas/global changes invalidate inherited state; selectable hover/press publishes its element revision. Custom renderer registration keeps its original renderer contract and is not intercepted by the built-in recorder. No new public UI authoring API, hashes, retry/fallback path, lower update rate, or scene-specific optimization.
- Text premeasurement runs before sibling layout. FixedSize text now invalidates layout only when its effective bounds change, not because unused intrinsic glyph width changed. AutoWidth/AutoHeight still propagate sibling layout changes; the whole native UI snapshot observes measured-layout revision too. UIFrame clip changes invalidate descendant clip packets. Pending texture publication, material versions/live RenderTexture revisions, viewport, scene membership, renderer replacement and explicit broad invalidation refresh retained arguments.
- A real child-to-child world reparent regression exposed missing Scene structure publication: GameObject::SetParent bypassed Scene root attach/detach and left hierarchy consumers stale. Publish the completed hierarchy change in the native Scene path; add native persistent-scene and Python retained/fresh-equivalence checks. This is a general hierarchy fix, not a world-UI fallback.
- Preliminary retained-argument benchmark:screen/color2.1402ms,world/color1.6369ms,screen/text2.22ms (`041-ui-dynamic-packets.log`,4passed). Native AddText calls remain1000 because this caches Python extraction, not native geometry. Final isolated benchmark with explicit extraction counts is recorded below after complete regression. Do not convert these CPU-sink numbers into FPS or claim full native/GPU UI completion.
- First full Python run:3failed/6236passed/11skipped; fixed two tests for the intentionally moved cache boundary and FixedSize effective-layout return value, and retained the existing per-element +1 revision semantics. First new focused run had two tests call nonexistent GameObject.set_active (correct API is active property), plus the genuine world reparent failure above. Final focused206passed/1skipped; complete native89/89passed,105.07s,including16Vulkan tests. Build exited0. Final complete Python6275passed/11skipped,268.46s,exit0 (`041-ui-packets-full-python-final.log`), including36new retained-command cases. Native error logs after the Python summary are intentional negative-fixture output, not failed tests.
- Normal MCP/Qt close ended editor50296/Hub24808 before native rebuild. Restored visible source editor50568 and Hub46436 (Hub minimized); no Lab asset changes saved. Scene12 six actual screen button clicks, state restoration and Stop/reopen passed with Console0warnings/0errors. Scene03 cold display, screen/world text edits, parent movement/rotation and Stop/reopen passed. Its world semantic-target list is still empty; actual world-button pointer/callback validation is separate. Engine-owned captures under `review/041-ui-packets-*` preserve previous runs' files. No commit, push, release or subagent in this slice.
- Real world-button pointer down/up passed: transient UIProgressBar target changed0→.75; pressed-state tint and callback captures were reviewed. Stop/reopen discarded the temporary binding/object, leaving Scene03 clean Edit. Source editor50568 and Hub46436 are the only related Python app processes; no new server traceback/Vulkan validation error in this run. Existing world semantic-target discovery remains a separate open issue.
- Final isolated1000-label benchmark after full suites and with editor in clean Edit:color screen2.0487ms/world1.61305ms; text screen2.12865ms/world1.65705ms (`041-ui-dynamic-final-isolated.log`,4passed,25.99s). Every case reextracts exactly1element and replays1000AddTextcalls; color requires0fontmeasures, text1. This improves sparse visual updates. Native tessellation/upload, dense bound resources, moving-layout workloads and full editor/Player GPU matrices remain open; no new041checkbox and no full-goal completion claim.

## Latest — 2026-09-14 batched world UI input and frontmost hit

- Continue UI optimization after the user's acceptance of the Outline600+FPS main path. No 041 checkbox or goal completion claim. This is input/dependency work, not dynamic draw-list or GPU UI completion.
- Extended the existing native UITransformDependencies generational snapshot with batch ray projection. Runtime pointers and Scene picking now share that implementation. Layout stays Python-owned; resolved dimensions are retained against the existing layout revision (including intrinsic text measurement). Scene input membership is retained, and all-miss event processing no longer repeats each component's owner/enable query. There is no new public authoring API, fallback, hashing, reduced update rate, separate world Canvas, or raw stale owner pointer.
- Projection retains unbounded local coordinates for captured drags, ignores UI's own scale, follows parent/world rotation and translation, preserves layer31, parallel/behind-camera rejection, closest world depth and physics occlusion. Native tests cover pre-commit FrameCache movement and expired handles; Python tests add parent TRS matrices, size changes, mouse/touch capture, policy, destruction, selection and stable depth ties. Two old mocked Transform tests now use real native scene UI. The first new destruction test omitted process_pending_destroys; fixed its deferred-destruction assumption, not engine lifetime semantics.
- Initial complete regression before front-to-back query ordering: Python6238passed/11skipped,275.32s,exit0. Complete native89/89passed,103.18s,including16real Vulkan tests. Negative-fixture native error logs after the Python summary are not test failures. Final post-ordering Python rerun is recorded below when terminal.
- Input-only all-miss1000-world baseline16.37875ms becomes0.5697ms (`041-ui-world-batch-bench.log`). The added end-to-end CPU input test (collection+projection+event dispatch, stationary pointer) exposed expensive checks of every overlapping world quad. Event queries now visit surface priority front-to-back and stop on the first eligible hit; equal depths retain the first original surface. Screen Canvas priority and non-interactable blocker semantics are preserved. Added an explicit100-overlap policy-call-count test. This changes lookup order, not scene/render order.
- The first post-order benchmark (`041-ui-world-batch-bench-final.log`,12passed) overlapped editor startup and is preliminary, not the final isolated performance window. A clean rerun is required after regression and startup stop. No FPS estimate is derived from these CPU input microbenchmarks.
- Final post-order complete Python regression:6239passed/11skipped,330.09s,exit0 (`041-ui-world-batch-full-python-final.log`). Focused event/screen tests152passed/1skipped. Final benchmark ran after the full suite, with the editor in clean Edit and no startup/other tests:12passed,46.10s (`041-ui-world-batch-bench-isolated.log`). All-miss mapping+raycast1000:screen0.0624ms,world0.5757ms (world pre-batch16.37875ms). Full CPU input stages1000 (collection+mapping+events):screen miss0.0720ms/overlap0.0718ms;world miss0.7240ms/overlap0.6334ms. The pre-front-to-back overlap baseline was16.3935ms after native projection was already installed. Do not mix the full-input and mapping-only denominators or treat either as frame FPS.
- Normal MCP window close ended editor30952; Hub45500 exited through its inspected Qt tray termination route. Native module/test build exit0; visible editor50296 and Hub24808 restored. Scene03 cold Edit, screen/world text edits, parent translation/rotation and Stop/reopen passed without saving. Scene12 six actual screen buttons and owner/state restoration passed; Console0warnings/0errors. Engine-owned full-window/Game/Scene captures used, no OS screenshots.
- Added real world-button callback proof: temporarily created a far-away UIProgressBar through MCP, bound the existing World Action button's persistent event to set_value(.75), then sent actual pointer-down/up at the engine-captured button. Target value changed0→.75 with no Console error. Stop/reopen removed all temporary changes. The helper's first click run expected state 'play' instead of the actual 'playing' and stopped before injecting input; corrected the helper and the real callback passed (`041-ui-world-callback-click-final.log`). World semantic-button listing remains empty; coordinate input plus observed callback now supplies actual interaction evidence, without claiming that separate MCP discovery issue fixed.
- Final Scene12 whole-frame windows after the microbenchmark summary:240frames each,means0.95673/0.95029/0.97892ms,P951.633/1.642/1.639ms (`041-outline-profile-ui-world-batch.jsonl`). 1080p,4xAA,13passes retained. GUI P50 is0 (the editor does not rebuild its idle UI every frame); no claim that all improvement is caused by this slice, no GPU-timestamp claim, and no replacement of the user's600+FPS stage decision with a new1000FPS gate. Continuous semantic collection stayed disabled; engine captures followed timed windows. Native modules are current in visible editor50296; Hub24808 restored/minimized, only these two Python app processes remain. Editor left ready/clean Edit. Server stderr contains three Windows asyncio connection-close WinError10054 traces with subsequent operations succeeding; this separate MCP transport issue remains, not suppressed or claimed fixed.
- Remaining gates: dynamic layout/text workloads, per-element command updates, dense bound materials and complete editor/Player UI performance matrices. Full041 remains active; no commit, push, release or new agent.

## Latest — 2026-09-14 UI input broad-phase snapshots

- Previous goal turn was progress: native dependency batching, full Python/native regression and visible Scene03/12 evidence. Continue the full 041 scope; Outline600+FPS remains stage accepted, UI performance remains open. No checklist increment.
- Measured the actual mapping + per-surface raycast functions with 100/1000 real scene UI elements and a pointer outside all rectangles. Baseline input-only P50: screen100/1000 = 2.3063/28.9876 ms; world100/1000 = 2.23185/26.73385 ms (`041-ui-input-baseline.log`). The supplied camera ray is fixed to isolate mapping/hit work; physics queries still run. These are all-miss scaling costs, not whole-frame FPS or a claim about topmost-hit cost. Profiling identifies repeated component-owner resolution, ancestor/group walks, geometry and field reads.
- Screen Canvas now retains ordered eligible AABBs and clip rectangles, using the existing native Transform batch plus scene/world/epoch, resolved-layout and input-policy revisions. Only broad-phase candidates enter the existing exact/custom hit test. `raycast` and `raycast_all` share one implementation. Color/alpha animations do not invalidate hit bounds; policy changes, native enable/activation, measured text size, clipping, viewport sizes and parent motion do. Empty canvases have no retained rows. No device fallback, reduced update rate, content hashing or additional authoring API.
- A real regression test proved render-only parent motion left a child's cached rect at x=0 instead of x=200. Runtime command dependencies now invalidate derived child layouts on native pose changes even without an input query. Both consumers retain Transform as the position authority. UIEventProcessor no longer ignores stationary pointers: UI moving under a stationary mouse, or becoming disabled, updates Hover. This follows the unconditional pointer raycast in Unity's [PointerInputModule](https://raw.githubusercontent.com/Unity-Technologies/uGUI/main/com.unity.ugui/Runtime/UGUI/EventSystem/InputModules/PointerInputModule.cs); AABB-first candidate filtering is also present in [GraphicRaycaster](https://raw.githubusercontent.com/Unity-Technologies/uGUI/main/com.unity.ugui/Runtime/UGUI/UI/Core/GraphicRaycaster.cs). No Unity source was copied.
- World targets now reject an out-of-bounds point before resolving owner/group policy. The per-object Python ray mapping and event-loop surface queries remain too costly. Intermediate input-only rerun: screen100/1000 = 0.0261/0.06385 ms; world100/1000 = 1.5881/18.8948 ms (`041-ui-input-cached.log`). This does not accept world input or dynamic full-list rebuilding.
- Added nine real-scene tests for snapshot reuse, color animation, group policy, clip/rotation/tolerance/order, native enable, measured size/resize mode, stationary hover, destruction and render-only parent changes. Converted old unattached/mock-Canvas tests to their actual screen Canvas or canvas-free world target. The first test run named a nonexistent layout test file; later failures exposed test assumptions about set_active() and old anchor positioning, corrected to the real active property/Transform-backed layout. Final focused selection:165 passed/1 skipped; the preceding broader selection191 passed/1 skipped predates the color-policy split. Final complete Python regression:6230 passed/11 skipped,297.99s,exit0 (`041-ui-input-full-python.log`). Native rejection diagnostics after its summary are intentional negative fixtures; no native code changed or CTest rerun is claimed this turn.
- Visible source editor30952 loaded the new Python code after old editor52600 exited normally through MCP. Scene12 six screen buttons, state restoration and Stop/reopen passed; Scene03 cold Edit, screen/world text and world-parent pose changes passed. Engine-owned captures reviewed, Console0warnings/0errors, no authored Lab asset saves. The first Scene12 helper required clean Edit but found Play; after normal Stop it passed. World semantic button discovery remains empty, so that helper does not claim a world-button callback assertion. Hub45500 remains available; no native rebuild or release this turn.
- Final input-only all-miss rerun after the complete suite: screen100/1000 =0.0244/0.0622ms; world100/1000 =1.46665/16.37875ms,4 tests passed (`041-ui-input-final.log`). Earlier intermediate world18.8948ms is retained rather than omitted; neither result is acceptable world-input scalability. Remaining repeated owner access and per-object Python projection are still visible in the profile.
- Final uninstrumented Scene12 three240-frame means1.03788/1.04612/1.10963ms (901–964FPS), P951.978/1.977/1.983ms, with1920x1080,4x sample AA and13passes retained (`041-outline-profile-ui-hit-snapshot.jsonl`). Timed windows ran after tests/benchmarks finished, captures outside the windows, continuous semantic collection disabled. The earlier noisy windows remain in the preceding log; this is current-run evidence, not a controlled attribution of all frame-time improvement to this change. Engine full-window capture reviewed; OS window facts confirm visible=true,minimized=false (not an OS screenshot). The editor is left in ready, clean Edit and only editor30952/Hub45500 remain. MCP stderr again contains one asyncio connection-close WinError10054, while later calls and captures succeed; this is a separate transport issue, not a claim of fully clean transport.
- Next UI gates: native batched world input/projection, cached input membership and actual nearest-hit/drag/touch matrices, dynamic layout/per-element draw-command reuse, bound-material workloads and whole-frame scale tests. The existing nested-Canvas traversal comment should also be checked against `_walk_children` behavior; do not assume it already excludes nested islands. Full 041 remains active; no commit, push or completion claim.

## Latest — 2026-09-14 native UI dependency snapshots

- Previous goal turn was progress: engine dependency simplification, real UI checks and full Python6217passed/11skipped. Outline600+FPS remains stage accepted; full041 and UI performance remain open.
- Replaced per-element Python Transform/property/matrix expansion with an internal native UITransformDependencies batch. It retains ECS generational handles, compares screen XY/Z rotation or world position/rotation/layer, and returns one revision. Membership is rebuilt on existing scene/world/topology changes; UI field notifications rebuild sparse resource bindings. Empty material slots no longer enter each-frame material resolution; unresolved nonempty references and live RenderTextures remain tracked. No public authoring API, new fallback, retry, device path or crypto check was introduced.
- Regression caught an important boundary: the native global serial advances on clean-to-dirty changes, not every repeated local write. Screen poses therefore read their three native values directly; world poses also read while matrices are dirty or the deferred frame cache is active. C++ tests cover repeated world overrides before EndFrameCache and expiry after unload. Python tests cover static/no-Python-geometry reads, unrelated motion, repeated local edits without world sync, parent motion, ignored UI Scale, world layers, detached materials and destroyed handles. An existing unbound RenderTexture mock was replaced with a real attached UI component, retaining the live-target revision fixture.
- Initial build reached linking but Hub40988 held the native module. The editor exited normally through MCP. After verifying the exact owning process, Hub was closed through Qt's tray-window WM_CLOSE application-termination route; no force kill or setting change. Retried only after the old build was terminal and Hub had exited. Final native module and new test target build exit0 (`041-ui-native-dependencies-build-final.log`). Final editor52600 and Hub45500 were reopened; no test/benchmark Python process remains.
- Full native CTest:89/89passed, including16real Vulkan tests,116.05s (`041-ui-native-dependencies-ctest.log`). Focused Python before the final frame-cache addition:166passed/1skipped. Final full Python:6221passed/11skipped,314.98s,exit0 (`041-ui-native-dependencies-full-python.log`). Invalid shader/asset diagnostics after the summary belong to intentional negative fixtures.
- Final local A/B (`test_041_ui_native_revision_bench.py`, `041-ui-native-dependencies-bench-final.log`): real native scenes,100/1000UIText,12ancestors,10warmups +35samples, optimized→previous polling→optimized repeat. For1000controls: screen static13.3896→0.0119/0.0119ms; world static15.2767→0.0028/0.0028ms; unrelated moving/resolved3D object, screen13.5410→0.0120/0.0119ms and world15.3696→0.1279/0.1283ms; one UI element moving, screen13.5389→0.0124/0.0120ms and world15.4445→0.1269/0.1273ms. These are dependency-only CPU P50s, not layout, input, GPU, command construction or end-to-end FPS. The first draft repeated one world GameObject lookup and left the unrelated object's world matrix unresolved; final rerun corrects both, superseding that draft.
- Visible-editor regression: `041-ui-native-world-review.log` completed with Scene03 cold Edit captures before Play, screen/world text and font changes, world parent translation/rotation, Stop/reopen ready and clean. Engine-owned cold editor, updated Game and Scene captures were reviewed. World semantic-button targets are still absent; the helper does not claim semantic click coverage. A separate coordinate press/release was delivered, with editor capture `041-ui-native-world-pressed-editor.png`; no click callback assertion is claimed. Scene12 `041-ui-native-outline-review.log` passed all six real screen buttons, owner/state restoration and Stop/reopen, without graph topology rebuilds; captures reviewed, Console0warnings/0errors. No Lab assets were saved.
- Uninstrumented Scene12,1920x1080,4x sample AA and13passes retained: first three240-frame means1.88111/2.18457/1.90325ms, P954.421/4.408/4.214ms (`041-outline-profile-ui-native-dependencies.jsonl`). After buttons and clean reopen, a second series gave1.68228/1.21857/1.19666ms, P953.193/2.547/2.248ms (`041-outline-profile-ui-native-after-buttons.jsonl`). Captures were outside timed windows and continuous semantic collection was disabled. All windows are retained, not just the fastest; this variability does not prove end-to-end improvement or an accepted UI budget. Foreground-window inspection showed WeChat, but no causal claim or forced focus switch was made. Game-only and GUI timers are CPU timings, not GPU durations.
- Editor Console has no runtime error and native logs have no Vulkan validation failure. Separately, the MCP server stderr contains one asyncio Proactor connection-close WinError10054 at01:04:56, with subsequent calls succeeding; retain it as transport evidence, not a clean-transport claim and not an excuse for suppressing exceptions in the UI path. Editor remains clean Edit after verification. No screenshot outside the engine, commit, push or release was performed.
- Remaining UI work is explicitly not hidden by this benchmark: bound-material/texture-heavy cases, input/raycast costs, layout/text updates, per-element command reuse and full-frame/multi-Camera/editor/Player matrix. World pose inspection is still native O(N) after transform changes, and one visual edit can still rebuild the shared draw list. These next steps remain in A06; no full-checklist completion, commit, push or release is claimed.

## Latest — 2026-09-14 UI dependency work

User accepted the Outline main path at600+FPS and moved priority to screen/world UI. This turn simplifies Canvas ancestry, private bookkeeping and material dependency checks; full details/evidence are in the2026-09-14 section at the end of this log. Scope remains unfinished: final1000-element dependency-only A/B improves screen19.881→11.374–11.423ms and world14.819→13.113–13.222ms, still far too costly to accept as scalable UI. Real Scene12 six buttons and Scene03 screen/world text/transform rendering passed. Final full Python regression:6217passed/11skipped. Historical1000FPS-blocking statements below no longer reflect the user's current Outline decision.

## 2026-09-13 — GPU submission batching and MCP one-shot capture (local verification complete; performance open)

- Continued the actual 4x Outline AA/performance target without changing the Lab's assets, resolution, sample count, effects, or cameras. Previous acceptance remains open; no checklist item is added.
- Obtained a real Nsight Vulkan trace of the visible Scene12 editor through its MCP Play/input controls. Nsight initially failed on the Unicode TEMP path; process-local TEMP/TMP under `dev/041-nsys-local` fixed the tool launch, without changing system/user configuration. Initial default injection produced no Vulkan data and is not evidence. Explicitly enabling Nsight's installed Vulkan layer produced `041-outline-gpu-layer.nsys-rep` and its SQLite export.
- The 3-second trace contains 2071 presents and 16568 `vkQueueSubmit` calls: exactly 8 host submissions per frame. GPU intervals are grouped by the correlated submit and next CPU present; 2057 complete frames have median first-to-last GPU span1.2146ms, including queue gaps, not pure shader time. Dominant submit-slot GPU intervals have medians0.6727ms and0.0712ms. This instrumented trace is diagnostic, not the uninstrumented FPS acceptance. CPU render/scene timers remain distinct from GPU measurements.
- The Vulkan queue owner now atomically reserves contiguous serial ranges. The existing executor hands adjacent same-role VkSubmitInfo batches to a single Vulkan call, retaining each command buffer, timeline wait/signal, cross-queue boundary, upload dependency and final fence. No outline-specific rendering branch or alternative fallback path was added. Expanded real Vulkan tests split fill/modify/readback across three same-role batches on Graphics, Compute and Transfer, gated by an external upload timeline.
- Initial build used std::span despite the repository's C++17 contract; corrected the private API to pointer/count. Final full native build succeeds without compiler warnings. Focused submission/queue replay/RenderTexture Vulkan tests3/3 pass. Full native regression and visible post-change measurements are pending at this entry. Test editor and Hub exited through their normal MCP/window/tray actions before rebuilding; no process kill, commit, push or release.
- Final native regression88/88 passed (101.74s), including16 Vulkan tests; Compute/CPU-JIT Python126/126 passed. Post-change Nsight reports4192 submits for2096 complete frames:2/frame instead of8, retaining8 command buffers/frame. The remaining compute/graphics role boundary is preserved. GPU first-to-last span median0.43185ms is instrumented diagnostic evidence, not a standalone claim about shader speed or final FPS; CPU submit cost did not fall in proportion to call count.
- A real post-interaction regression exposed the MCP plugin's `_snapshot` unconditionally enabling continuous semantic capture. The native one-shot mechanism was correct; the plugin permanently opted into expensive UI collection. Whole-frame2.43–2.69ms and UI-build P956–8ms after six button clicks are retained as failure evidence (`041-outline-profile-submit-batched-final.jsonl`). Removed the implicit enable, using RequestSnapshot directly without toggling an explicitly selected continuous policy. Added success/timeout tests for both preexisting policies. Initial test clock patched the shared time module and exhausted its iterator; isolated the fake clock to ui_operations and reran. Focused22/22, complete MCP suite88/88 pass (86.47s).
- Built/verified local official MCP808768-byte inxpkg and updated Lab through PluginManager's normal installer:38members, no removals, no dependency install, preload error or restart-required flag. No public release was made. Visible editor45900 then passed all six buttons, owner/depth/projection restoration, Stop and reopen. After interaction, continuous=false, active=false, requested=completed=27. Publish count stays27 while ordinary GUI frames advance1255→1852: the expensive collection no longer remains active.
- Final fresh normal editor45900 (no Nsight/validation modules) at1920×1080,4x scene/owner/depth,13 authored passes: three240-frame means1.30137/1.31660/1.31248ms (**760–768FPS**), P952.074/2.108/2.077ms, average GUI build0.071–0.077ms. `041-outline-profile-one-shot-final.jsonl`; final Game and full editor captures are engine-owned, Game image visually reviewed. Console0warnings/0errors during interaction. Editor remains playing for user review; Hub is restored separately. This is improvement, not the~1000FPS target or final visual acceptance. Strict041 stays139/332(41.9%). No scene/material edits, commits, pushes or releases in this slice.

## 2026-09-13 — Actual per-sample Outline AA (local regression passed; performance open)

- Corrected the rejected color-only MSAA boundary. ShaderInfo Resources now accepts `Texture2DMS`/`Texture2DMSUInt`; SPIR-V reflection retains image MS state and SampleRateShading requirements. Supported Vulkan devices enable sample-rate shading; unsupported fullscreen shaders report the missing capability instead of silently changing algorithms.
- The common fullscreen graph reads raw multisampled color/depth when the reflected shader requests it, and resolves ordinary Texture2D inputs. Independent depth inputs retain their own source/version/extent/sample count. RenderTexture multisample color is explicitly sampleable. Shader hot publication rebuilds affected resolve topology, and an incomplete graph is not executed after Reset. No outline-specific engine class, blur fallback, content hash or per-frame reflection was added.
- Lab Scene12 uses its existing 13-pass topology, 4x owner/depth attachments and per-sample fullscreen shader. Its 1x offscreen Camera selects the 1x entry from the graph's effective sample count. Both shaders share the same effect code through ShaderInfo Imports; IDs/flags are never averaged. Scene and materials were not replaced.
- Initial Vulkan test correctly failed: persistent multisample color lacked Sampled usage. Corrected the allocation contract; real Vulkan regression now verifies per-sample owner IDs 1/9 and depth, separate sample shading and final red/blue coverage resolve (not an averaged ID). Structured shader compile test passes for float and uint MS inputs. Full native/Python regression still pending at this entry.
- Initial Lab integration used unsupported raw GLSL `#include`; changed to the existing ShaderInfo Imports mechanism. Failed-launch logs (`041-outline-sample-aa-editor*`) are retained as failure evidence and are not acceptance evidence. Corrected visible editor45816 compiled both entries and rendered the scene. Six actual buttons, split/restore owners, orthographic/perspective, Stop/reopen passed (`041-outline-sample-aa-interaction.log`), Console 0 warnings/0 errors and no Vulkan validation errors in corrected logs. Engine-owned1920x1080 Game and full editor captures were visually reviewed. Native files changed after that run require a final restart.
- This is progress on AA, not final user acceptance or the ~1000FPS/1ms performance target. The earlier823-840FPS mask1 result is not carried forward as the performance of this shader. Final uninstrumented frame windows remain to be measured. Strict041 checklist remains139/332 (41.9%).
- Final allocator regression initially caught an unnecessary filtering sampler on multisample color; kept the existing no-sampler lifetime contract for MS images and reran the complete build/tests. Final native88/88 (16 real Vulkan tests,112.70s), Python6207 passed/11 skipped (320.85s),20 generated tutorial outputs verified. Native shader/asset rejection tests intentionally log malformed-input errors; these are not failures of the visible scene.
- Final uninstrumented visible editor3900: Game1920x1080,4x scene/owner/depth sampling,13 passes and offscreen Camera preserved. Three240-frame windows average1.57555/1.62834/1.42766ms (635/614/700FPS), P95 2.59469/2.83936/2.53884ms (`041-outline-profile-sample-aa-final.jsonl`). Final engine-owned full editor capture reviewed, Console0 warnings/0 errors. These measurements do not meet the1ms target; no GPU timestamps are available yet, and CPU render timing must not be called GPU time. Editor remains playing for user review. No commits, pushes or releases performed.

## 2026-09-13 — Outline CPU submission and 4×MSAA performance work (in progress)

- Latest user visual rejection: outline edges remain jagged. **Scene color/depth4× with a single-sample owner mask is not the requested completed AA path.** The categorical mask uses `texelFetch`, and the composite uses discrete neighbor selection/coverage maxima; shading the fullscreen triangle into a4× target does not create missing per-sample outline coverage. Do not mark AA or the~1000FPS acceptance complete. Inspect the common sampled-resource/reflection/fullscreen path before implementing per-sample ID/coverage consumption and final color resolve; do not average owner IDs or substitute blur as MSAA. Current ShaderInfo Resources only exposes Texture2D/Texture2DUInt; SampledImageInfo lacks multisample metadata. Original Unity Runner-Long source also explicitly sets mask/depth msaaSamples=1, so its smooth appearance cannot be attributed to a proven per-sample-mask implementation without checking the final camera AA chain.
- Property-descriptor guard focused validation:229passed/1skipped (`041-outline-property-guard-focused.log`). The preceding UI/method revision passed6206/11 (`041-outline-optimized-full-python.log`,302.17s); that full run predates the final property change. Visible normal editor20816 loads the latest property/UI changes. With Game visible,1920×1080, main color/depth4× and owner mask1×, three240-frame windows give1.1903/1.2148/1.1963ms (823–840FPS), P95 1.85–1.90ms (`041-outline-profile-msaa4-final.jsonl`); engine-owned Game/editor captures were reviewed. This is intermediate performance evidence only, **not full outline-AA acceptance or1000FPS achieved**. It remains open and visible for user inspection.
- User tightened scene12 acceptance to approximately 1 ms actual editor frames / 1000 FPS at 1920×1080 and 4×MSAA, retaining existing effects, world UI, interaction and the authored offscreen Camera. This is not yet accepted and does not increase the 139/332 checklist count.
- Replaced per-lookup native callable closures with class-level descriptor guards. Follow-up also moves property error translation onto native accessors, removing blanket Python field interception. Destroyed-object errors, falsy liveness and ordinary Python hot replacement remain explicit; no owner/callable cache or retry path is added.
- World UI membership now shares a structure/scene-epoch snapshot across rendering, input and picking. Static world UI uses the existing native command cache rather than forcing every screen/world element to rebuild every frame. World pose (including parent changes), layer, enabled/active state, materials, textures and persistent-scene structure participate in the existing revision. Text/image/button packet reuse now respects parent UIGroup state.
- Intermediate method-guard regression: 6203 passed /11 skipped,294.39s (`041-outline-binding-full-python.log`). Initial final-UI focused run exposed two list-to-tuple signature mistakes and a stale test double; corrected, then196 passed /1 skipped (`041-outline-ui-tests.log`). Later property-guard and parent-alpha test require final rerun; do not attribute the earlier full pass to those later edits.
- Intermediate visible uninstru­mented MSAA1 measurement: with Game tab actually visible, three240-frame windows average1.295/1.331/1.283ms (751–779 FPS), P95 about2.09–2.17ms; previous no-validation-layer baseline was3.52–3.63ms. The initial post-restart Scene-tab measurement1.89–1.96ms is a different workload and is not substituted for Game-tab comparison. Captures are engine-owned; target stays1920×1080. Scene/Game differences remain important to any performance claim.
- Scene12 pipeline now requests4× scene color/depth; owner-ID and corresponding mask-depth stay explicitly1× because categorical IDs must not be averaged. Grade intermediate remains1×. Existing native color/depth resolve is reused. This is not a claim that owner masks have per-sample ID coverage AA. The original MSAA1 script is retained locally at `041-OutlineGalleryPipeline-before-msaa.py`; no scene/material assets are rewritten.
- Visible validation editor25536:4× pipeline renders with13 authored passes; six real buttons, owner splitting, projection, selection, Stop/reopen passed; Console0warnings/0errors and no Vulkan VUID/validation errors. Windows asyncio logged two MCP socket reset callbacks (WinError10054); they are not shader errors and are not suppressed. Native editor exited normally. Final property-guard regression and uninstrumented4× timings still pending at this entry.

## 2026-09-13 — Compiler metadata/device boundary; Outline performance follow-up

- Previous goal turn: verified progress, not completion. Continued the builtin JIT-only compiler path, preserving engine buffer/Vulkan ownership. Removed the active legacy transfer/command implementation and its `ti_device_api` target, plus the unused LLVM kernel dispatcher. Texture formats now live in compiler metadata; capabilities are explicit value inputs. Replaced transitive device includes and consolidated build policy in `InfernuxJitPolicy.cmake`. Removed four tracked legacy files (recoverable from Git); no project assets deleted.
- Actual MSVC dependency traces cover all 137 current core/frontend/SPIR-V/util translation units: RHI includes are only `arch.h` and `device_capability.h`; no Device/CommandList/CUDA/DirectX or LLVM runtime headers remain in this active closure. The native contract rejects reintroducing competing device/runtime declarations through public compiler headers. Unused upstream source outside this closure, shared IR/type-factory pruning and the formal compiler ABI/service remain open; B05 is not complete.
- Final compiler target builds and stages normally (`041-compiler-metadata-final-build.log`), including preserved Apache LICENSE/NOTICE. Final 88/88 CTest (16 Vulkan,3 compiler;101.48s),126 compute/CPU-JIT Python tests,57 Hub update/default-setting tests and7 fork tests passed. Fork diff check and workflow YAML passed. Found and corrected the fork CI's stale output-path assertion; Windows locally builds/installs the small native output fixture and verifies private frontend and attribution. Ubuntu job updated but not run remotely; this is not a manylinux acceptance claim.
- Visible editor44948 ran existing Snow and early Jelly through MCP, restored scene12 clean Edit and later closed normally. Snow's engine-owned1920x1009 image was inspected; no scene/material/gameplay asset changes. GPU numerical/cold compiler validation belongs to formal tests, not cached visible playback. Source Hub27636 remained open. Counts stay139/332(41.9%),193unchecked; full-scope12–18solo engineering-week estimate unchanged. No commit,push or release.
- User then reported Outline around200FPS withoutMSAA. Diagnosed the real scene before further implementation. `OutlineGalleryPipeline.py` explicitly sets samples1; capture confirms1920x1080 Game target and1920x1009editor. Three240-frame windows with Khronosvalidation loaded:5.560/5.642/5.607ms. Fresh editor7604 without that layer, same authored scene/settings:3.595/3.519/3.631ms(~275–284FPS); screenshot briefly302FPS is not the aggregate. CPU pipeline/camera preparation remains2.209/2.229/2.286ms; these are CPU timers, not GPU timestamp results. Both instances normally exited. A cProfile editor36488 was opened to locate that remaining CPU overhead; its FPS must not be used as the benchmark. No reduction in resolution,drawn effects orsamplecount was made.
- Profiling editor36488 ran63s of Play and exited normally, writing `041-outline-cpu.pstats` / `041-outline-cpu-summary.log`. The instrumented startup/Edit/Play session recorded397,438`get_canvas` calls and2,411,469`_wrap_native_callable` constructions; `functools.update_wrapper` ran2,412,148times. This identifies repeated UI hierarchy traversal and eager binding wrapper creation as real engine CPU work; cProfile callback/reentry overhead means these cumulative seconds are not GPU timings or an uninstrumented percentage. Inspection confirmed `_guarded_getattribute` creates a new guarded function with `@wraps` on each method read. No guard bypass, UI feature removal or engine performance fix was applied in this diagnostic slice. GPU pass timestamps and matched Unity/content/MSAA scaling remain unmeasured. The next performance work should address these shared UI/binding/submission paths while preserving destroyed-object errors and scene-change invalidation.

## 2026-09-13 — Independent Hub update audit

- A new independent agent reproduced the reported update-discovery failure mechanisms, rather than blaming Python3.13 or GitHub rate limits alone: legacy0.3.7 expects retired release filenames, initial0.4.0 rejects newer optional catalog metadata, and a repeated0.4.0/build-2 cannot compare greater than0.4.0. The current source uses the website's static catalog and Cloudflare asset URLs. Early incompatible clients require a full installer; source edits cannot repair shipped binaries.
- Fixed release-candidate comparison, preserved real network errors, joined manual/startup checks with visible feedback, and treated unpublished catalogs as invalid rather than up-to-date. Only a failed network request tries the same static catalog mirror; successful-but-current or malformed catalogs are not replaced by unrelated sources. Public Hub changes must use a higher application version, not an invisible wheel build-number increment.
- The live site is legacy GitHub Pages behind Cloudflare. The existing bot catalog push does not start a Pages build. The release workflow now requests the build, waits within a fixed budget for the target commit, and verifies the exact public URL/headers used by Hub (not a cache-busting substitute). The agent read actual Pages configuration and official API requirements; it did not run remote writes. Actual release/deployment acceptance remains for the next publication. Partial-publication recovery without overwriting public artifacts is documented in `scripts/release/README.md`.
- User explicitly selected default-on startup update checks: unset/enabled check, explicit disabled remains disabled, downloads and installation still require confirmation. Settings, installer and EN/CH disclosures are consistent; installer layout accommodates both languages. Final isolated Hub suite375passed/3skipped, release automation3passed; workflow YAML and diff checks pass. Windows/Linux Cloudflare archive HEAD requests returned200 with catalog-matching lengths; no claim of full archive download or remote installation.
- Reopened source Hub27636 after engine tests, with the visible main window restored. The only remaining Python process is this intentional Hub, not a test leak. Engine/MCP results are recorded below. No version bump, commit, push, public release or forum response occurred in this slice; full041 remains active.

## 2026-09-13 — Forum MCP feedback and independent audit

- Final combined-state CTest passed88/88 in129.91s (16Vulkan,3compiler contracts), after the complete6199/11Python result. Rebuilt/verified local MCP package is808768bytes and was installed through PluginManager.update;38members, no removed members, no dependency install, no restart-required flag or preload error (`041-mcp-lab-update.log`). This is still a local Lab update, not an external release.
- Visible editor37320 loaded the Release module and Khronos validation layer; its actual1920x1009 window was visible/not minimized. MCP Snow, early Jelly and saved800x450 RenderTexture Edit/Play/Stop completed with0Consolewarnings/errors. Engine-owned captures were inspected directly: snow indentation, deformed sphere/cubes and occluded outline RenderTexture are present (`041-mcp-audit-snow-visible.log`, `...jelly-visible.log`, `...rt-visible.log`). Existing compiler caches were consumed; this does not claim visible cold compilation or300FPS/scaling/physical-quality acceptance. Authored gameplay assets and jelly scale were not rewritten.
- Scene12 returned to clean Edit withdocument_state=ready. Normal MCP window.close fully exited editor37320; flushed stdout/stderr contain0validation/VUID/device-loss/traceback/error findings and no Python/CTest/build process remains. Full041 remains active at139/332(41.9%),193unchecked, rough12–18solo engineering weeks at unchanged scope. No commit, push, public release or forum reply occurred.
- Complete Python regression after the accepted fixes passed6199/11skipped in285.31s, process exit0 (`041-mcp-audit-full-python.log`). Expected invalid-shader/asymmetric-collision-matrix negative fixtures still log their deliberate errors. The final combined-state CTest run is being repeated because some native test fixtures also import the Python scene/document layer; the visible Lab package update follows only after it finishes.
- Final independent re-audit now accepts the original report plus its discovered regressions. The reviewer read the real JUnit report:185passed,0failed/errors/skipped in10.37s (`dev/artifacts/mcp-feedback-targeted-final.xml`). Coverage includes a consumed watcher event between DOCUMENT_READY and publication, new/same paths, clean/dirty content, actual queued reload failure/conflict, existing reimport retry after a first failure, and already-imported staging files moved onto existing/new Python targets. Loaded durable state and latest observed external state are now separate facts; only publication performs one comparison, not a per-frame hash loop or automatic replay. The new full Python run is in progress.
- Combined native rebuild succeeded; full CTest passes88/88 in103.62s, including16 Vulkan tests (`041-mcp-audit-full-build.log`, `041-mcp-audit-full-ctest.log`). Subsequent Python audit changes are not yet covered by a new complete pytest run. The reviewer found that a watcher event consumed during a scene read can be lost at publication; imported staging files also need reclassification when moved to a new path with another extension. These are under active repair, not accepted as completed.
- The Lab update dry-run correctly refused three locally changed MCP files (`material_operations.py`, `operation_support.py`, `scene_operations.py`). Inspection shows the incoming changes add explicit MCP renderer ownership and replace duplicated scene/helper code with the engine Host authority; existing scene operations remain available there. Preserved the entire installed MCP directory and plugin registry at `dev/artifacts/041-mcp-lab-before-feedback/` before the planned update. No gameplay asset was overwritten; the actual package update and visible validation are still pending.
- The user requested a GPT-6 Astra/xhigh implementation agent for the complete report at https://infernux-engine.discourse.group/t/topic/23/2, followed by a fresh agent with no inherited discussion. Both agents read the original report; work is split between engine document/asset/scene authorities and MCP response schema/docs. No forum reply, commit, push or release was sent; the user received a draft reply only.
- Initial fixes made stale conflict choices visible without retry, preserved reload parser errors, normalized unknown-source atomic replacements and reported asynchronous scene.open as scheduled, with last-load state observable through project.info. Failed/cancelled Save Copy now preserves an existing conflict while ending its ticket. Three malformed fields through startup/deferred/conflict paths are covered by actual scene tests. First full Python regression passed6174/11skipped in277.03s (`041-compiler-context-final-python.log`); this is explicitly not final acceptance of later audit fixes.
- The independent audit rejected first-pass completion: keep-local could retain an obsolete CAS baseline or remain clean/unsaveable; successful same-path reload did not always replace the full baseline; an already-imported staging file could steal the target GUID; reading A and publishing after disk B could incorrectly acknowledge B. These are being fixed with real file/save/cross-frame regressions rather than loosening format validation or adding automatic retry.
- Native scene reads now publish their exact loaded-byte state alongside the validated document. ReadTextFileSnapshot reuses the existing AtomicFile identity/hash algorithm without reopening the file for content; SceneDocumentReadTicket retains that token after consuming the JSON and clears it on cancellation. Python must carry it through successful publication, never recapture later bytes as if they were loaded. Build succeeded and AtomicFile tests pass1/1 including existing interruption gates (`041-mcp-read-snapshot-build.log`, `041-mcp-read-snapshot-ctest.log`). Independent native code review found no new blocker; Python integration and final complete regression remain pending.
- Built a local MCP .inxpkg using its standard standalone packaging script (`dev/artifacts/041-mcp-feedback.inxpkg`,808384bytes,source version0.1.1). This is a local test artifact, not a published release. The Lab's installed MCP is a physical package rather than a source junction; its update will use the normal installer and refuse overwriting modified files without inspection.

## 2026-09-13 — Direct SPIR-V context and value-only compiler output

- Classified the previous goal turn as verified progress: helper/borrowed-view pruning, 85/85 native tests, 6146 Python passes / 11 skips, reviewed visible Lab consumers and normal process exit. Rechecked the actual branch and live processes; only the user's source Hub45060 was running. No editor/native-test/build overlap is permitted.
- Reproduced the remaining process-wide Program restriction in the formal no-device test (`041-compiler-context-before.log`): a second live compilation context throws "Only one instance at a time". The new test retains first-context IR while compiling/releasing another context, then recompiles the original IR and compares exported SPIR-V/metadata. This is serial ownership isolation, not parallel frontend support.
- Removed the single-implementation ProgramImpl/InfernuxCompilerProgramImpl hierarchy. Program now invokes the stateless SPIR-V compiler directly, preserving the same argument/return layout algorithm. Context destruction no longer resets global IR/SNode counters, and dead host-arch/timing/kernel-ID/finalize APIs are removed. The private Program name and necessary shared IR remain; no new context registry, cleanup poll or fallback was added.
- Compiler output is now a concrete SPIR-V/metadata value, not an optional-backend base carrying launch handles and file state. Removed TIC load/dump, duplicate binary/JSON serialization and SHA256/error translation, the unused compiled-output check and dynamic-cast guards; removed the associated source files and PicoSHA2 include path from the active target. Infernux's existing .inxgpu cache remains unchanged. Disabled upstream runtime/AOT/other-backend source still needs its planned dependency cleanup; this slice does not claim all Taichi source is pruned.
- Compiler rebuild and focused compiler/Vulkan tests initially passed 2/2, but the first full CTest failed 1/85: compute_gizmo_gpu segfaulted during native cleanup (`041-compiler-context-full-ctest.log`). This was not accepted as flaky or hidden by repeat passes. The Windows dump identified ComputeBuffer::~ComputeBuffer accessing a destroyed Python ComputeHost lease through renderer-held native buffer ownership; the Vulkan core also destroyed its compute queue before clearing those owners. The machine subsequently rebooted at19:53; the earlier source Hub process is no longer running.
- Added a deterministic no-GPU service-lifetime regression: replacing a host wrapper at the same address changed an existing buffer's queue under the old implementation (`041-compute-lifetime-before.log`). Buffer, kernel and asynchronous readback now retain their own device/queue service view; equality compares those real services instead of wrapper addresses. Renderer teardown releases native consumers before the compute queue. External Python lease protection is retained. No resource registry, retry or new null fallback was added. The new regression and real Vulkan buffer/kernel/Gizmo tests pass4/4 (`041-compute-lifetime-focused.log`).
- Replaced the fork's obsolete Gfx runtime/TIC contract test with compiler context/layout/value contracts, preserving its IR cloning and use-replacement coverage. Both fork contracts are now built with native tests and registered in root CTest; test executables stay in the configured build directory and are not shipped in the compiler wheel. Standalone fork tests passed4/4 (`041-compiler-contract-tests.log`). Full root rebuild succeeded (`041-compiler-context-final-build.log`); final full regression is in progress. No new completion checkmarks or release claim yet; totals remain139/332 (41.9%).
- Root CTest after the compute-lifetime/compiler changes passed88/88 in98.03s, including16 real Vulkan tests and3 compiler contracts (`041-compiler-context-final-ctest.log`). LastTest.log and final log contain no VUID/validation/device-loss diagnostics. Subsequent user-requested document/MCP changes are tracked above and still require combined-state validation. Full041 remains active.

## 2026-09-13 — Inline-only helper frontend and borrowed compiler views

- The prior goal turn made verified progress on request-owned kernels and failure/global-cycle lifetimes. This turn inspected current source/build state and found that Infernux helpers already lower inline; the alternate real_func path had no engine caller and even expected a Program.create_function Python method no longer exported.
- Removed real_func, function specialization/ID/callback caches, its AST mode branches, native Function/FunctionKey/call exports and Program's unused function ownership/map. Shared native optimizer Function IR is retained: deleting an unused authoring path is not permission to break the optimizer dependency graph. The engine's kernel/function APIs and execution model are unchanged.
- New borrowing coverage first failed because an ASTBuilder view outlived its owning native Kernel (`041-compiler-inline-borrow-before.log`). Program config and Kernel ASTBuilder now use owner-retaining binding policies. No proxy registry, polling validation, reset path or fallback was added. Standalone API absence initially failed as expected (`041-compiler-inline-before.log`).
- Nested engine helpers with defaults, branches and local loops executed correctly before and after pruning. A new typed-pair helper exposed an existing tuple-mutation error in return casting (`041-compiler-inline-final-focused.log`); the compiler now casts via a mutable value sequence. The supported Python range is3.13, so old3.7/3.8 annotation branches and private typing._GenericAlias checks were replaced by get_origin/get_args. Final focused native/device-independent + real Vulkan tests pass2/2 in5.00s (`041-compiler-inline-tuple-focused.log`). Native module size after pruning is9,496,576bytes; this is not a complete wheel-size or performance claim.
- Full regression and visible validation follow below. B05 remains open for the rest of the native export/dependency cleanup, authoritative compiler ABI/service and final platform/Player/performance delivery. Checklist totals remain139/332 (41.9%).
- Final full regression passed: 85/85 CTest in 134.20s, including 16 Vulkan tests and the device-independent compiler test (`041-compiler-inline-full-ctest.log`); 6146 Python passes / 11 skips in 348.59s, exit 0 (`041-compiler-inline-full-python.log`). No Vulkan validation/VUID/device-loss findings in either full log or LastTest.log. Deliberate invalid-scene/shader/collision-matrix fixtures emit expected negative diagnostics, not failures. The final CMake target restaged the same compiler plus updated LICENSE/NOTICE (`041-compiler-inline-delivery-stage.log`).
- Fresh visible editor 45936 loaded the current Release engine and the actual Khronos validation layer. MCP Snow, early Jelly and the authored 800x450 RenderTexture passed Edit/Play/Stop checks (`041-compiler-inline-snow-visible.log`, `...jelly-visible.log`, `...rt-visible.log`). Engine-owned images were reviewed directly. The editor used existing cached artifacts and did not load the private compiler DLL; cold compilation is covered by the separate formal tests, not these images. No Lab asset or authored softbody scale was changed, and no full physical-quality/scaling or 300 FPS acceptance is claimed.
- Restored scene12 clean Edit, Console 0 warnings / 0 errors; normal MCP close fully exited process45936. Complete flushed editor stdout/stderr have no traceback, error, Vulkan validation or device-loss findings. Only the user's source Hub45060 remains running; its main window4787370 is visible and not minimized. The user reconfirmed normal Hub close/reopen permission during verification, but no additional shutdown was necessary for this compiler-only target. Both plan summaries now reflect this slice. Counts remain 139/332 (41.9%), 193 unchecked; the full-scope 12–18 solo engineering-week estimate is unchanged. No commit, push or release in this slice.

## 2026-09-13 — Request-owned compiler kernels

- The previous goal turn made verified progress: metadata-only input, single forward kernels,85/85CTest,6146Python passes/11skips and visible consumers. This continuation confirmed the remaining actual retention: PyTaichi.kernels and Program.kernels both kept compiled kernels indefinitely, while the engine's bounded artifact cache retained only exported values.
- Native Program::kernel now returns unique ownership to its caller. The private Python binding keeps the Program alive for that kernel, not vice versa. Removed the Python kernel registry, its cache/reset traversal, duplicate Kernel reset initialization and the stale native kernel pointer after AST lowering. Generated-source linecache entries now have the same try/finally lifetime as the compilation request. This is deterministic ownership, not a polling cleanup service or alternate runtime.
- The new engine ownership test first failed against the old code (`041-compiler-ownership-before.log`). Compiler-only native rebuild then passed (`041-compiler-ownership-build.log`), without closing/restarting the source Hub because it did not load the compiler DLL. Focused device-independent and real Vulkan tests passed2/2 in4.21s (`041-compiler-ownership-focused.log`). The no-device test now exercises32fresh successful lowerings interleaved with32failing ASTs, weak-reference release, Program lifetime retained by its kernel and exported SPIR-V surviving release. The engine test also checks generated-source removal and specific source-line diagnostics after failure.
- Full regression and visible validation follow below. Native Function/IR/type-factory dependency pruning, full compiler ABI/service boundary and final Player/platform/performance acceptance remain open; this does not complete B05 or change the139/332checklist total.
- First-phase full results:85/85CTest in128.08s and6146Python passes/11skips in346.67s (`041-compiler-ownership-full-ctest.log`, `...full-python.log`). Further boundary review then found two real remaining defects, each reproduced by a failing formal test (`041-compiler-ownership-boundaries-before.log`): native Kernel construction called Python before its owning handle existed, so a retained failure traceback could outlive the native owner; exec-generated function/global cycles also retained copied project objects until a later GC pass. The editor deliberately schedules GC, so request cleanup must not depend on automatic collection.
- Second-phase fixes allocate and return the owning native Kernel before invoking Python AST lowering, remove the private manual-finalize/reset escape hatch, and clear only the request's copied globals after lowering. Original project globals are never cleared. Exported artifacts remain usable independently. Final verification of this combined state follows below; first-phase passing results alone do not cover these last changes.
- Final combined-state verification passed: the two formerly failing boundary tests now pass2/2 in4.51s (`041-compiler-ownership-boundaries-focused.log`), including immediate release with automatic GC disabled and retained source-line diagnostics. Full native85/85,16Vulkan plus compiler-only,146.86s (`041-compiler-ownership-final-ctest.log`); full Python6146passed/11skipped in349.49s (`...final-python.log`). Both logs were checked for VUID/validation/device-loss findings; none. Release compiler payload is staged through CMake, with33Python files and the9,535,488-byte native module built at18:44:34.
- Visible editor17752 loaded the Release engine and actual Khronos layer, then passed Snow, early Jelly and saved800x450RenderTexture Play/Stop checks through MCP (`041-compiler-ownership-snow-visible.log`, `...jelly-visible.log`, `...rt-visible.log`). Engine screenshots were human-reviewed; project assets and authored scale were not edited. These are existing cached-artifact consumer checks, not an editor cold-compilation timing claim; fresh compilation/ownership is covered by the independent and Vulkan formal tests. Scene12 was restored clean Edit, Console0warnings/0errors; normal close fully exited and flushed native logs contain no errors/traceback/VUID/device-loss. Source Hub45060 remained visible; no test worker/editor remains.
- Both plan summaries and fork READMEs/NOTICE reflect this result. Strict progress remains103/256 +36/76 =139/332 (41.9%),193unchecked, with12–18solo full-time engineering weeks still a rough full-scope estimate. No whole-B05/full041/performance/multiplatform/release/PR-green claim and no commit/push this turn.

## 2026-09-13 — Metadata-only GPU compilation and single-kernel ownership

- Replaced synthetic NumPy array/scalar arguments at the private compiler boundary with the engine's buffer type annotations and scalar type-only placeholders. Runtime values remain the responsibility of the engine's packed argument layout. No new upload, synchronization or device owner was introduced.
- Removed automatic adjoint kernel creation, class-stackframe probing, data-oriented class wrappers and the unused gradient AST checker. One entry point now creates one forward compiler kernel. Numeric Matrix/Vector helpers retain their small dual-scope implementation: they are needed for literal arithmetic and IR lowering, not GPU storage or execution.
- The initial contract test failed against the old placeholder path (`041-compiler-signature-before.log`). Removing every dual-scope helper initially broke matrix_ops imports; that over-pruning was corrected before acceptance. A later added host determinant assertion also targeted a kernel-only method and was removed, while host norm/transpose/inverse coverage remains. These failed runs are not reported as passes.
- Promoted the fork's compiler-only test into the main CTest suite as `infernux.compute_compiler_only`. It loads the actual installed private/native compiler, emits SPIR-V and task metadata without importing Infernux.lib or creating a GPU device, and is not labeled as a Vulkan-device test. Real Vulkan execution remains separately tested. The installed Python frontend now contains33files; LICENSE/NOTICE and both fork READMEs describe the narrower boundary.
- Full regression passed:85/85CTest, including16actual Vulkan tests plus the device-independent compiler test,127.26s (`041-compiler-signature-full-ctest.log`);6146Python tests passed/11skipped in334.87s (`041-compiler-signature-full-python.log`), exit0. LastTest.log has no Vulkan validation/VUID/device-loss findings. Negative shader and collision-matrix fixtures deliberately emit errors after the passing Python summary; those are not editor failures. Final CMake install restaged the same compiler source and updated NOTICE (`041-compiler-signature-delivery-stage.log`).
- Visible editor30788 loaded the current Release module and the real Khronos validation DLL. Snow Play/Stop, early softbody frames and the saved800x450RenderTexture camera all passed via MCP (`041-compiler-signature-snow-visible.log`, `...jelly-visible.log`, `...rt-visible.log`). Engine-owned frames were reviewed directly, without OS screenshots or pixel analysis. Authored softbody scale and scene assets were preserved; early frames show the deformed mesh and rigid cubes, not proof of the full interaction/performance matrix. Restored scene12clean Edit, Console0warnings/0errors; normal MCP close fully exited, and the complete stdout/stderr contain no VUID/validation/device-loss/traceback/error findings.
- At this continuation's start the previously reopened Hub was no longer running, so no second shutdown was needed. Reopened the source Hub in the infernux environment as process45060 after verification; its actual main window4787370 is visible and not minimized. No test editor/helper remains. This slice does not complete B05: native IR/source pruning, compiler ABI/lifetime, Player/multiplatform packaging and the rigid-softbody scale/performance matrix remain open. Checklist totals remain139/332 (41.9%),193unchecked, with12–18solo engineering weeks as a rough full-scope estimate. No commit, push or release in this slice.

## 2026-09-13 — Python storage-owner removal and presentation-independent maintenance

- Continued the agreed built-in compiler path. Removed Matrix/Vector/Struct field factories, MatrixField/StructField, host-storage access branches, root/FieldsBuilder and SNode authoring helpers, runtime storage materialization/synchronization, and obsolete dynamic-SNode AST recovery. Numeric Matrix/Vector/Struct expressions and code-generation IR remain. Deleted `_compiler_types.py`, `runtime_ops.py`, `mesh.py`, and `snode.py`; the last two were previously only excluded from installation. Git retains their history. The definition-based removal alone covered 1,063 lines; that is not a net repository LOC claim.
- The real CMake-installed private frontend now has34 Python files, down from36. Updated both fork READMEs and NOTICE to describe implemented private imports, engine-owned execution and remaining IR/source/release work. Restaged LICENSE/NOTICE through `infernux_gpu_jit_compiler`, not a manual file copy. This remains an in-progress compiler dependency, not a released plugin or a completed cross-platform wheel.
- Regression first reproduced the obsolete marker import, then passed real Vulkan kernel/helper/atomic/vector-buffer results and numeric matrix/vector mutation with the removed APIs absent. Complete native regression:84/84 (16 Vulkan),149.78s (`041-compiler-field-full-ctest.log`). Complete Python regression:6143passed/13skipped,308.30s (`041-compiler-field-full-python.log`); process14828 exited. No editor ran concurrently with those tests/builds.
- Visible validation editor72120 exercised existing Snow, Jelly and scene12 RenderTexture through MCP. Initial scene-open observation timed out because the real window was minimized (Win32 IsIconic=true, client0x0); the original request was still pending, not failed. Restoring that same window completed the original load. The failed probe is preserved in `041-compiler-field-snow-visible.log` and does not count as a pass. The subsequent snow/jelly/RenderTexture probes exited0; reviewed engine-owned images show a snow indentation, deformed jelly/cube interaction, and an800x450 outlined RenderTexture. Saved asset references and authored jelly scale were preserved. This is functional compiler regression evidence, not300FPS/scaling or physical-quality acceptance.
- Scene12 returned to clean Edit. Editor72120 closed through the normal MCP input lifecycle and exited; its fully flushed logs contain no Vulkan validation or Python traceback findings. Found a general frame-lifecycle defect: minimized, surface-recreation and MSAA early returns skipped deferred owner work. Added one shared callback invocation body at the existing safe points; skipped pre-Update frames run it without simulation, MSAA exits first end the active scene frame, normal frames retain post-submission ordering. No retry/replay or new fallback manager was introduced.
- The new source-boundary regression fails on the old code and passes after the change (`041-minimized-maintenance-before.log`, `041-minimized-maintenance-focused.log`). Initial native linking failed because the user's source Hub66916 held the native module. After explicit permission, invoked its tray Exit action through UI Automation, verified normal process exit, and rebuilt successfully (`041-minimized-maintenance-build-retry.log`). Hub must be reopened after verification. Complete post-fix regression and a real minimized-window scene switch are still pending at this record point; a source assertion alone does not prove the runtime fix.
- Counts remain main103/256 plus Compute36/76 =139/332 (41.9%),193 unchecked. Native IR/legacy source removal, final compiler ABI, full rigid/soft-body scale/performance matrix and multi-platform delivery remain open. No commit/push/release or full041 completion is claimed.

Final verification for this slice:

- Post-fix CTest84/84 (16 Vulkan),126.68s; complete Python6144passed/13skipped,264.61s, process exit0 (`041-minimized-maintenance-full-ctest.log`, `041-minimized-maintenance-full-python.log`). Expected invalid-shader and asymmetric-collision-matrix diagnostics from negative tests remain in the full Python log; they are not editor failures. Installed34-file AST audit has no absolute public Taichi imports or the four retired modules.
- Fresh visible editor7020 loaded the rebuilt native module and the Khronos validation DLL. With its actual Win32 window kept minimized throughout, MCP opened Snow from scene12 in0.768s, then an engine-owned1920x1009 capture completed after restoring the window (`041-minimized-owner-visible.log`). This covers the real skipped-presentation path, not only a source-text assertion. It does not claim a tested display-surface-loss fault injection or all-platform lifecycle coverage.
- Visible Snow Edit/Play/Stop and scene12 RenderTexture Edit/Play/Stop then passed (`041-minimized-owner-snow-play.log`, `041-minimized-owner-rt-visible.log`). Engine-owned snow/RenderTexture captures were reviewed; Console0warning/0error. Restored scene12 clean Edit, closed through normal MCP input, verified editor7020 exited and fully flushed stdout/stderr have no VUID/sync/device-loss/traceback/error findings. No native-build/pytest/editor overlap occurred.
- Reopened the user's source Hub after verification as process21624, using the same infernux environment and launcher. The two plan headers now reflect this result without adding premature checkmarks. The next compiler work is the remaining metadata-only input/AD/class-kernel and native IR/source dependency cleanup, followed by the existing full performance and delivery requirements; no new fallback path was added.

## 2026-09-13 — Priority: legacy Hub installs Linux CPython313 into Windows Python312

- User reported Windows `.runtime/python312` containing only CPython313 Linux `.so` files; the affected machine/Hub version was not supplied and its project path is not present locally. User permits either a fix or an evidence-backed response for the reporter. Do not claim a remote installation was repaired.
- Reproduced the historical selector directly from `v0.3.7:packaging/version_manager.py` against the live GitHub v0.4.0 asset list (`041_wheel_selection_repro.py`). It returns the first wheel, `infernux-0.4.0-1-cp313-cp313-manylinux_2_35_x86_64.whl`, without platform/ABI filtering. That Hub's embedded runtime is fixed at312. Current selection returns Windows wheels only, preferring build2 and requiring Python3.13. The published v0.4.0 source already contains platform/ABI filtering and defaults to3.13.
- Downloaded the current PyPI build2 Windows wheel into the explicit local audit directory. WHEEL declares `cp313-cp313-win_amd64`; native payload has `_Infernux.cp313-win_amd64.pyd`, matching bootstrap, SDL3/Jolt/assimp and engine DLLs, not Linux `.so` files. GitHub also publishes the Windows0.4.0 Hub installer and Windows manifest. This strongly supports the old-Hub explanation, but the reporter's exact version is still unconfirmed.
- Current Hub version-manager/project-model regressions:26/26 passed in4.27s (`041-wheel-selection-tests.log`). No extra production validation/fallback was added after establishing the existing fix. Reporter guidance: update Hub itself, not just the engine; install/select0.4.0 in the new Hub and verify with a new empty project; preserve the existing project. If it persists, request the Hub version and complete downloaded wheel filename.

## 2026-09-13 — Private GPU frontend imports and obsolete array storage pruning

- Continued A05/B05 against the actual CMake-installed compiler payload, not an imagined new plugin. The shipped closure has36 Python files;31 had upstream absolute imports. These now resolve inside `Infernux._compiler.taichi._vendor.taichi`. Removed the engine frontend's temporary `taichi.*` alias bridge, whole-process module scan, displaced-module restoration, and rejection of an occupied public namespace. The existing compiler lock remains; this is not concurrent compilation support.
- Baseline regression `041-compiler-namespace-before.log` reproduces the old initialization failure with three unrelated module sentinels occupying `taichi`, `taichi.lang`, and `taichi._lib`. The real Vulkan kernel fixture now runs with those identities untouched, checks the private module name and removed storage APIs, and exercises mathematical intrinsics in a compute helper. Sentinels prove import-namespace isolation, not coexistence with a second loaded upstream Taichi native extension.
- Removed dead `MatrixNdarray`, `VectorNdarray`, their four Matrix/Vector/type factory methods, and `impl.ndarray`. Compiler buffer arguments use the existing external-buffer expression, not an upstream host/device allocation. Matrix/Vector arithmetic and compiler IR remain. Field/SNode/Program-related legacy structures still need deeper review; full B05 stays unchecked. Removed an error-message-only `exec('from taichi import ...')` probe and its broad catch; source errors retain their cause without guessing a replacement API.
- CMake's real `infernux_gpu_jit_compiler` target installed the changed closure and preserved Taichi LICENSE/NOTICE. An initial manual indentation error was caught by the GPU fixture and corrected; its failed log is not passing evidence. Final staged math/helper/namespace test passes (`041-compiler-final-stage.log`, `041-compiler-final-focus.log`). Complete native regression passes84/84 in145.00s, including16 Vulkan tests (`041-compiler-private-full-ctest.log`). Complete Python and visible Lab validation remain pending at this record point.
- No Lab asset was rewritten, no new runtime fallback/installation path was added, and no release/commit/push occurred. Counts remain main103/256 + Compute36/76 =139/332 (41.9%),193 pending. Scope and rough12–18solo engineering-week estimate are unchanged; this is progress within deep compiler pruning, not full041 or300FPS acceptance.

- Final full Python result:6143passed/13skipped in259.33s, process exit0 (`041-compiler-private-full-python.log`). Staged36-file AST audit found no remaining absolute public `taichi` imports. Visible editor PID15420 loaded the Khronos validation DLL. MCP snow and jelly Edit/Play/Stop and saved scene12 RenderTexture800x450 passed operation/Console checks; engine-owned captures were reviewed. The first5s jelly image did not show the body, so it was not accepted blindly: follow-up captures show deformation and moving cubes, with Transform moving from approximately(0,1.66,0) at0.32s to(3.26,-13.05,-7.24) at5.24s, outside the finite floor/view. The authored X scale2.7839 was preserved. This does not establish physical-quality/scaling performance acceptance or a compiler regression. No scene parameters were changed to keep it on screen.
- Priority shifted to the user's Hub report. Scene12 was restored to clean Edit; the editor closed via the normal MCP input lifecycle and PID15420 exited. All build/test/probe processes are terminal. Further B05 pruning and the two plan-header updates remain for continuation; no full041 completion or release is claimed.
- Final log scan found no Vulkan validation findings, but editor stderr contains one asyncio transport traceback that still needs triage; do not call the complete stderr clean. A separately launched source Hub process66916 appeared during the user-priority investigation and was left untouched.

## 2026-09-13 — RenderGraph queue ownership across repeated execution

- The preceding ordinary editor PID60080 was closed normally before native work. Its fully flushed stdout/stderr contain no VUID, validation, device-loss, ERROR or traceback findings. No editor/test overlap was used.
- Found a general mismatch: compilation described only one execution's queue transfers, while graph-owned allocations retain their final owner across frames. Added final-owner -> first-consumer release metadata to the existing pre-setup submission path. The first execution has no prior release; graph move/reset/destruction now transfers/clears these existing lists. No CPU wait, download, hash, fallback renderer or second ownership manager was added to production.
- Corrected the initial probe's queue claim: a Transfer pass's Compute override was rejected, so the original missing-release reproduction was Graphics -> Transfer, not Graphics -> Compute. Transfer passes now accept Compute, which supports transfer commands. The expanded fixture asserts actual batch queues and fails on every engine ERROR. Invalid test-only color clears on dedicated Transfer and depth clears/copies on non-Graphics were replaced with color upload/copy and real compute depth sampling, respecting Vulkan command capabilities.
- Color images and buffers cover all nine Graphics/Compute/Transfer pairs. Depth is cleared on Graphics and sampled by the existing test compute shader on Graphics or Compute. Six executions per case also test graph move construction/assignment, recompilation without resource replacement, and Reset: **120 executions**, every returned component checked. Actual machine families are Graphics0, Compute2, Transfer1. Ordinary validation passed (`041-queue-replay-legal-test.log`). Test readback fences are measurement scaffolding, not new runtime waits or performance evidence.
- Additional synchronization validation exposed a second real bug (`041-queue-replay-syncval.log`): shader-only semaphore waits did not order ownership/layout operations. Queue-family transfers now wait at AllCommands; same-family dependencies retain their narrower scopes. Whole-graph external setup dependencies also cover acquires before the root shader. This follows the [Vulkan ownership-transfer scope requirement](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-queue-transfers); it does not enable optional maintenance extensions or serialize unrelated queues. The final focused synchronization-validation run passes (`041-queue-replay-root-test.log`), and this layer setting is now part of the CTest fixture.
- Intermediate test failures are retained as diagnostic history, not passing evidence. Full Release rebuild, complete native/Python regression and a fresh visible Lab/shutdown check are pending at this record point. Main103/256 + Compute36/76 = **139/332 (41.9%)**,193remaining. A09's full scheduling/resource contract stays unchecked; this fix alone does not complete that larger item. Full041 remains active, estimated12–18solo full-time engineering weeks at unchanged scope. No commit, push, release or new cross-platform/300FPS claim.

- Final verification: `041-queue-replay-full-build.log` Release ALL_BUILD exits0; `041-queue-replay-full-ctest.log`84/84 in143.89s,16Vulkan, LastTest.log contains0VUID/Validation Error/SYNC-HAZARD/device-loss findings. Full Python `041-queue-replay-full-python.log`:6143passed/13skipped in260.14s. Skips remain platform/filesystem/helper availability, not newly failing tests.
- Visible editor PID55412 loaded the real Khronos validation DLL. `041_queue_replay_visible.py` uses MCP only, does not rewrite Lab assets, and passes saved scene12 Edit/Play/Stop at800x450 with the original target GUID and0Consolewarnings/errors (`041-queue-replay-visible.log`). Engine-owned editor/Play/stopped captures were human-reviewed; GPU deformation and occluded outlines remain visible. MCP normal close returned, process exited, and full stdout/stderr through shutdown contain0validation/error/traceback findings (`041-queue-replay-editor.log/.err.log`). This verifies the actual Editor/RenderTexture consumer, not new cross-platform or300FPS acceptance. One ordinary visible editor PID23476 was reopened for handoff; no test/build process remains.

- Handoff confirmed by MCP: PID23476 is the only Python process, scene12 is clean Edit, and RenderTexture Camera is selected. Human-reviewed engine capture `review/041-queue-replay-handoff.png` shows the saved `OutlineCapture.rendertexture` in Camera's target slot and the rendered Gallery. No OS screenshots, pixel analysis or additional scene writes were used. All test/build sessions are terminal; working-tree changes remain uncommitted.

## 2026-09-13 — A10 logical-owner outline consumer

- New visible Lab scene `Assets/Scenes/12_OutlineOwners.scene` (GUID `0a0f2f2304a7a495ac339576e0e7f9f0`) was authored through MCP. Scene11 and existing materials/scenes were preserved. The new controller uses Inspector references, file materials, ordinary UI buttons, two native Rigidbody owners, builtin meshes and the existing GPU mesh-buffer kernel path. Outline registration, grouping and the effect remain project code; no Runner Long-specific engine component was added.
- Read the original game's `PuzzleScreenSpaceOutlineRegistry`, Feature and Mask/Composite shaders. The project registry merges independent callers' invalid/hatch/depth flags, groups parts by nearest parent Rigidbody, and updates only registered objects. It reuses `RendererSelection` and captured `DrawParameterBlock` values. Unchanged registration does not republish; movement and membership/state changes do not rebuild the Python graph. The current geometry/Transform path also supplies GPU-deformed vertices.
- The data mask uses single-sample RGBA32F (owner ID, positive eye depth, invalid, flags) and an independent D32 depth attachment after opaque geometry. Zero is empty; dense IDs are limited to `2**24 - 1`. IDs/flags use unfiltered texel reads, not half-float IDs or interpolated category values. The composite implements white/red owner boundaries, authored hatch, occluded hatch and explicit scene-depth rejection. Scene depth is copied before compositing; inverse-view-projection math reconstructs depth for perspective and orthographic views, and retained fragments write device depth. Neighbor-depth bias remains project-owned. This is not yet proof of thin-grass behavior or the full late-world-text/postprocess contract.
- Authoring failures were caught rather than hidden: a newly created shader used a different ShaderInfo Name than its asset name; only that new, unreferenced shader was deleted through the undoable asset operation and recreated with a consistent name. Missing EngineGlobals capability and incorrectly cased PushConstants types were corrected. A material document omitted its declared float properties; these were added through the material document operation.
- The helper update initially kept the old failing call. Logs identified noncanonical sibling imports (`OutlineRegistry` instead of `Scripts.OutlineRegistry`), not a need to relax candidate-module identity validation. Qualified asset-root imports fixed publication without restarting the editor. The new script also incorrectly called the `is_valid` boolean property; it now uses the actual property contract. No alias fallback, runtime import repair or expanded allowlist was introduced.
- First button-only checks passed while the pixels were still wrong. Engine-owned human-reviewed captures revealed an error-material checkerboard in the data mask: raw `main()` output had not opted out of generated shadow/depth/picking/motion/normal/base-color variants. The data-only shader now explicitly declares its supported pass contract. Hot publication restored the correct shader without an editor restart. Added a generic native shader-link regression for this material form, not an engine special case. The bilingual advanced RenderGraph guide documents these boundaries and the real API.
- `041-outline-exercise-3.log`: six actual SDL UI clicks changed INVALID/HATCH/DEPTH/GROUP/PROJECTION/SELECT, then restored state; owners changed2→3→0→2, with graph build count fixed at2. GPU deformation, rotation, overlapping parts and clean Stop/reopen were exercised. `041-outline-matrix.log`:16 flag/projection combinations plus4 fully-occluded combinations; all preserved topology and had zero Console warnings/errors. Captures were generated by the engine and reviewed with the image viewer, never OS capture or image-processing scripts. Perspective and orthographic occlusion/hatch and strict hiding were visually confirmed. Test-only wall changes stayed in Play and were discarded normally.
- Registry consumer tests (`dev/test_041_outline_registry.py`, using the existing real native engine fixtures, not fake renderers): **11 passed in2.74s**. They cover all8 caller flag combinations, independent removal, unchanged publication, reparenting, retired handles, and precision-limit rejection before mutation. Native Release shader-test build passed. Complete CTest with the Khronos layer: **83/83 in96.94s**, including15 real Vulkan tests; LastTest.log has no Validation Error/VUID entries (`041-outline-native-ctest.log`). Generated bilingual guide consistency:20 pages verified; whitespace check exit0.
- Full Python regression: **6110 passed / 13 skipped in215.87s** (`041-outline-python.log`). All13 skipped nodes were individually rerun with reasons (`041-outline-skips.log`): platform-specific paths, symlink permission, unavailable8.3short paths and one absent legacy UI helper. The two additional skips compared with the prior11 are8.3cases on this run's E: volume, not new test failures. Intentional invalid shader/asymmetric collision-matrix tests emit native error messages; pytest exits0.
- Fresh visible editor PID33884 loaded `VkLayer_khronos_validation.dll` and rendered scene12 directly in Edit with zero Console warnings/errors. Repeated the six actual clicks and all20 matrix cases (`041-outline-cold-exercise.log`, `041-outline-cold-matrix.log`), still without graph topology rebuilds. Engine-owned screenshots were human-reviewed, including perspective/orthographic full occlusion. A240frame diagnostic window at1920×1080 with validation: full-frame mean2.581ms/P95 4.428ms; game-only mean1.207ms/P95 1.478ms. This is this consumer's native timing window, not a GPU timestamp, Unity benchmark or softbody300FPS acceptance (`041-outline-performance.json`).
- Normal Stop and MCP window close terminated PID33884, with zero native ERROR, exception, DEVICE_LOST or validation entries through shutdown (`041-outline-cold-editor.log`, `.err.log`). Build/CTest/pytest did not run concurrently with the visible editor. Reopened normal visible editor PID28564 without the diagnostic layer; MCP confirmed scene12 clean Edit and startup logs contain no errors (`041-outline-delivery-state.json`, `041-outline-delivery-editor.log`, `.err.log`). Final whitespace check exit0. Do not treat the earlier erroneous checkerboard captures as acceptance evidence.
- Marked only the first4 A10.1 checkboxes: main **98/256**, Compute **36/76**, combined **134/332 (40.4%)**,198 pending. Remaining full scope still estimated at12–18 solo full-time engineering weeks, not a calendar guarantee. Transparent/clipped edges, thin occluders, late special world text and the complete postprocess/history order remain pending. A10 is not wholly complete and the041 goal remains active; no commit, push, PR or release claimed.

## 2026-09-13 — Selected Renderer/submesh passes on the existing render path

- Strict recount after one A09 checkbox: main94/256 + Compute36/76 = **130/332 (39.2%)**,202 pending. Estimate remains **12–18 solo full-time engineering weeks**, not a calendar commitment or a code-volume estimate. Full A09/A10, Runner Long integration, performance matrices, compiler trimming and platform/release work remain incomplete.
- Added `RendererSelection` as the owner of a material and explicit generation-safe Renderer/submesh entries. Each `set` captures an immutable DrawParameterBlock; an exact submesh entry takes precedence over an all-submeshes entry. The existing RenderWorld publication supplies current transforms, skinning and GPU mesh buffers. No copied geometry, global material mutation, extra frame hash, recovery loop or second renderer. Empty selection is intentionally no selected draw, not an error fallback.
- DrawRenderers uses effective per-draw parameters in material sorting, both batch merge paths, uniform batching and descriptor binding. Selection passes exclude automatic particle injection. RenderTexture reads use the selection material and normal graph dependency ownership. Membership changes do not change graph topology; actual resource-access changes may still require normal dependency recompilation. Public setters perform boundary validation once; mutation belongs to the owner update phase before native recording.
- `CommandBuffer.draw_mesh` now consumes the shared matrix bridge, matching Camera/material/Renderer NumPy `(4,4)` and strided inputs while retaining the existing flat document format. Graph Python APIs, native projections, stubs and the advanced Learn guide were updated. Unity comparison is limited to [CommandBuffer.DrawRenderer](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Rendering.CommandBuffer.DrawRenderer.html) Renderer/submesh/replacement-material responsibilities, not complete Unity SRP parity. An override shader must implement any source-material-only VS deformation; that differs from already-deformed shared GPU geometry.
- Release rebuild verified exit0 (`041-renderer-selection-build-verified.log`). Full native **83/83 passed**, including15 real Vulkan cases,126.22s (`041-renderer-selection-ctest.log`). CommandBufferTests explicitly enables assertions in Release and covers parameter capture, exact/all-submesh precedence, world/object/component generations and failed-update atomicity.
- First focused Python run had5 fixture errors (157passed): new test material setter incorrectly passed a tuple instead of its real positional-float API. Fixed the fixture;9 selection cases passed. First complete Python run **6085passed/11skipped/1failed** (`041-renderer-selection-full-python.log`) found new docs importing the internal uppercase package. Subsequent focused suites116and51passed. A later real authoring run exposed that changing this to `from infernux.rendergraph` was also wrong: the public API is the `infernux` facade module, not a duplicated lowercase package tree. The guide now uses `import infernux as inx` and `inx.rendergraph`; an additional test executes the actual guide against real native material/Renderer/graph resources. Final focused **52passed** (`041-selection-last-focused.log`); final complete Python run pending below. No broad candidate-import allowlist or alias fallback was added to hide an invalid example.
- Permanent visible041Lab consumer: `Assets/Scenes/11_SelectionStudies.scene`, GUID63108423ec98ed6a986f2b79c52b41f1. Pipeline/controller/material/shader and ordinary UI authored through MCP, normal refresh and serialized Inspector references. Play uses builtin Sphere topology with a resident GPU buffer and public kernel; the rotating cube uses a public two-submesh Mesh. Original10/09 and BuildSettings preserved. This is the foundation for A10, not the original game's complete grouped-ID/eye-depth/invalid/hatch/fragment-depth composite.
- Initial authored pipeline omitted `camera_target=True`, producing black Game captures despite no Console error. Corrected the project pipeline's explicit output declaration through MCP; hot reload restored visible output without an engine fallback. Initial black-run logs are not acceptance evidence. Actual accepted run: `041-selection-exercise-accepted.log`, Console0warnings/0errors. Engine-only captures moving/coral/submesh-zero/submesh-one/empty/restored were human-viewed: current GPU-deformed geometry is outlined, submesh colors are distinct, removing selection leaves ordinary geometry intact, and independent mask depth preserves the silhouette behind a foreground wall. No pixel analysis used.
- Accepted240-frame editor window at1920×1080: mean1.127882ms, P95 1.837515ms, P99 1.900817ms, max2.0772ms; game-only mean0.663316ms. One resident mesh vertex buffer, zero pending uploads,9submitted/9completed mesh uploads. These are editor CPU/frame observations, not GPU timestamps, a general-engine benchmark or the soft/rigid300FPS acceptance. `selection_updates`4→13 while `topology_builds` stayed2. Stop/reopen restored saved Edit state; complete physical UI click and Player/platform coverage remains pending.
- Found a generic authoring issue: discovery invalidation lived in live RenderStack callbacks, so creating a pipeline when no RenderStack existed left the project catalog stale. Moved the single invalidation to ResourcesManager before consumer notification; gameplay-only scripts preserve the cache. Removed the duplicated invalidation from RenderStack. Two regression tests cover no-live-consumer and invalidation-before-consumer ordering.
- Real no-RenderStack authoring regression passed without restart (`041-selection-catalog-visible-accepted.log`): empty scene→create pipeline→add Camera/RenderStack→select new pipeline→9pass graph ready. First harness used an unsupported dotted lowercase import; that failure remains in `041-selection-catalog-visible.log`. Corrected the harness to public `inx.renderstack.DefaultForwardPipeline`; accepted run produced no new Console errors (old failed-authoring entries retained). Both temporary scenes/scripts were deleted through recoverable Project history;11restored clean Edit. No native tests ran while the visible editor was alive.

- Final complete Python run **6089passed/11skipped/0failed in360.61s**, exit0 (`041-selection-final-full-python.log`). Expected rejected-input shader/layer messages after the pytest summary are negative fixtures, not failures. All20 generated Learn pages and ordinary whitespace checks passed (`041-selection-docs-check-final.log`, `041-selection-whitespace-final.log`). Native code is unchanged since the verified83/83 run.
- After pytest exited, launched the real visible editor PID30724 (`041-selection-final-editor.log/.err.log`). Updated both11demo scripts through normal MCP asset publication to use only the public `import infernux as inx` facade. Pipeline/component hot reload succeeded, then Play ran the GPU kernel normally. Six real pointer clicks on engine-published semantic button locations toggled sphere/sculpture/color off/on correctly and kept topology_builds unchanged. `041-selection-buttons-visible.log` exited0, Console0warnings/0errors; Stop and scene reopen restored clean Edit. Final engine captures `041-selection-buttons-editor.png` and `041-selection-buttons-verified-game.png` were human-viewed. Only the intended editor remains; no test worker or temporary catalog-probe asset remains. No commit/push, remote release, full041 completion or newly exported Player is claimed.

## 2026-09-13 — Shared NumPy matrix boundary and authored reflection gallery

- Strict recount remains main 93/256 + Compute 36/76 = 129/332 (38.9%), with 203 pending. Full-scope estimate remains 12–18 solo full-time engineering weeks; a narrow API/demo slice does not complete all of A09.
- Audited the existing Camera view override, oblique clipping, per-view history and scheduling paths before implementing. The missing public bridge was matrix handoff: Camera already returned NumPy `(4,4)` while material/renderer/draw inputs required manual flat column-major conversion. A shared `MatrixPyBridge.h` now handles row/column NumPy values, strides and copies at the binding boundary; existing persistent flat column-major documents remain unchanged. Public `Material.set_matrix` exposes the native operation. No extra render-target type, mirror renderer, runtime hash or fallback chain was added.
- Native Release build succeeded (`041-matrix-bridge-build.log`). Matrix/Camera/material/renderer/RenderGraph focused tests: 159 passed. Full Python before the later editor lifecycle fix: 6063 passed / 11 skipped in 359.35s. Full native CTest: 83/83, including 15 Vulkan cases, in 138.77s. Logs: `041-matrix-bridge-focused.log`, `041-matrix-bridge-full-python.log`, `041-matrix-bridge-ctest.log`. Generated Learn pages (20) and whitespace checks passed.
- MCP authored a permanent `10_PlanarReflectionGallery` scene in the real desktop 041Lab, with imported RenderTexture/material/shader/script assets, Inspector component references, a mirror-exclusion layer, clipping witness, orbit controls and UI buttons. Original 09 and Build Settings were preserved. Authoring harness mistakes (missing shader variant, changing ShaderInfo name, unchanged field edits, copied scene name) were corrected in the harness; they are not engine runtime fallbacks. Initial authoring logs record these failures honestly.
- Edit-mode preview exposed a generic lifecycle gap: `execute_in_edit_mode` dispatched Update but not LateUpdate. The scheduler now performs all preview Updates before all preview LateUpdates against one native frame snapshot, without enabling fixed-step physics. Tests cover ordinary components remaining inactive, phase ordering and same-frame disable. Focused regression: 57 passed (`041-editor-late-focused.log`). Full regression and actual gallery visual acceptance are pending at this record's creation; subsequent evidence is appended below.
- Unity comparison: [Material.SetMatrix](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Material.SetMatrix.html) and [Camera.CalculateObliqueMatrix](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Camera.CalculateObliqueMatrix.html) provide comparable material-parameter and near-plane responsibilities. Infernux retains its own +Z/[0,1] conventions and per-camera culling policy. This is not a claim of complete Unity ExecuteAlways or SRP parity.
- The editor LateUpdate change passed a full Python run: 6065 passed / 11 skipped (`041-editor-late-full-python.log`). Visible Edit preview then worked, but Play exposed a native fast-snapshot omission: records excluded native-only objects, so serialized Camera references failed preflight. Removed that extra filtering pass, retained native identities for every object, and added eight root/nested Camera/Light/Rigidbody/Transform cases. Release rebuild succeeded (`041-gallery-snapshot-build.log`); focused tests passed after correcting a test's inherited post-load delta-time state.
- Stop/reopen exposed two reference bugs rather than a rendering fallback need. SerializedField bypassed reference ownership by directly returning `_cached`; that duplicate fast path was removed. ComponentRef now checks Python wrapper liveness as well as its native identity. Play no longer invalidates still-live wrappers on entry or redundantly clears the cache around transactional Stop restoration.
- Reproduction then showed that retired callbacks could resolve old authored IDs into the replacement world. Ordinary persistent-reference reconnection is an existing supported contract, so a trial globally pinning refs to dead native generations was rejected and undone. The committed implementation uses a narrowly scoped retiring-world exclusion only during scene transaction finalization; other resident worlds remain accessible, and normal references reconnect afterward. Tests cover cached/first-use references, exception-safe scope exit, additive worlds, Undo and the unchanged persistent-rebind contract. Final focused result: 409 passed (`041-gallery-reference-retirement-final.log`). No per-frame hash, recovery branch, or second reference catalog was added.
- The real gallery also exposed its own invalid cleanup lookup: after world retirement it called `self.game_object.get_component` on the retired owner. The authored script now carries its surface Renderer as an Inspector component reference, just like its two Camera references, and only clears surviving references in OnDisable. This does not claim that a retired owner's `game_object` is accessible; that broader lifetime contract is not closed by this demonstration. Updated and saved through MCP, not by rewriting the scene document.
- Human-viewed captures proved PBR file materials, Edit/Play reflected geometry, real UI orbit/clipping buttons, below-plane exclusion and a translated/tilted mirror. Earlier failed Stop captures remain recorded rather than relabelled as passes. The initial unlit material choice was corrected in the authoring harness. Final full regression and a clean repeated visible lifecycle test follow this record; no complete A09 checkbox is awarded yet.
- Full Python on the final reference path initially reported 2 failed / 6075 passed / 11 skipped (`041-gallery-final-full-python.log`,355.21s). Both failures were incomplete UI material stand-ins without the real Material `name` property, previously hidden by the field cache bypass. Corrected the fixtures, not engine resolution policy; UI/reference focused regression passed 231 / 1 skipped (`041-gallery-ui-reference-focused.log`). Full rerun is recorded separately below.
- Visible editorPID17836 passed two complete Play→mirror deactivate/reactivate→Stop→scene reopen cycles, then returned to clean Edit with Console0warning/0error (`041-gallery-lifecycle-verified.log`). Game1920×1080,reflectionCamera1280×720,and full-editor1920×1009 captures `041-gallery-lifecycle-final-{game,camera,editor}.png` were produced by engine capture and human-viewed (no pixel analysis). Native frame windows240samples each: averages1.335879/1.344314ms,P952.120895/2.108275ms,P992.403002/2.289728ms. These are this small Windows gallery's frame measurements, not GPU timestamps or general engine/soft-body acceptance. Both failed editorPID64408 and successfulPID17836 exited normally and were confirmed absent before native/Python regression.
- Final native CTest succeeded 83/83, including all15 `requires_vulkan_device` cases,141.20s (`041-gallery-final-ctest.log`). Full Python rerun starts only after its exit0; no simultaneous visible editor/build/GPU test workload.
- Audited the next consumer in the real Unity Runner-Long sources: `PuzzleScreenSpaceOutlineFeature.cs` draws selected Renderers/submeshes with per-owner ID and invalid/hatch/depth flags into RGBA16F plus independent D32, then composites against scene depth and writes fragment depth. A10 already describes this correctly. Infernux RenderPassBuilder currently filters queues/tags/material programs and can override a material, but it does not yet accept the explicit Renderer set/per-draw data required by that consumer. Existing DrawParameterBlock capture should be reused; do not substitute ordinary picking or an enlarged-mesh outline. `CommandBuffer.draw_mesh`'s transform argument also still accepts only legacy flat16 even though parameter matrices now accept NumPy4×4. These remain next A08/A09 implementation work, not completed claims.
- Final full Python rerun succeeded **6077 passed /11 skipped in363.72s**, exit0 (`041-gallery-verified-full-python.log`). Combined with native83/83,20generated Learn pages and `git diff --check`, this closes the current matrix/preview/reference implementation regression, not full041 or all A09 acceptance. Plan header and demonstration table updated; strict counters remain129/332. No commits, pushes, branch switches or remote release claims were made.

## 2026-09-13 — Cooked RenderTexture consumers in the visible Windows Player

- Strict recount remains main93/256 + Compute36/76 =129/332 (38.9%),203pending; estimate12–18 solo full-time engineering weeks. No complete A09 checkbox is awarded for this Windows-only slice.
- Closed visible editorPID46688 normally before the native build. Formal CMake `prebuild_player_runtime` succeeded (`041-ui-player-runtime-build.log`) and wrote the Windows subrepository's player payload and `.inxpkg` directly. The project had an ordinary copied plugin directory, not a link; the normal PluginManager update reported10files,0removed,no local conflicts,loaded/enabled true,no restart required (`041-ui-player-plugin-check.log`, `041-ui-player-plugin-update.log`). No manual binary copy, commit, push or remote release.
- Reopened visible source editorPID19820. MCP authored temporary `CookedTargetProbe` scene/material/RenderTexture plus one offscreen Camera, two UIImage consumers and a MeshRenderer consumer. The target is517×291,rgba16_sfloat,4×MSAA,D32. Scene save/reopen preserved the concrete UIImage GUID. First17.41s build succeeded but launch correctly rejected an undeclared start scene; the fixture—not runtime policy—was corrected by temporarily including the scene in Build Settings and rebuilding. Job0b975d4bcde642f680b39a32eef6819f completed successfully with no diagnostics. Build Settings were subsequently restored exactly to the original Jelly-only scene list.
- First visible Debug PlayerPID73176 reached ready and showed Camera output on Mesh material, direct UIImage.texture and UIImage.material. Capture `041-cooked-render-target-player.png` was human-viewed. Project-side temporary source assets were then removed through recoverable Project deletion, and original09 restored clean Edit. No user-authored scene/material was deleted.
- The unchanged package cold-started again as visible PlayerPID64068 after all three temporary source files were absent. Capture `041-cooked-render-target-without-source.png` again showed all three consumers, with updated audio-field activity. Both Player processes exited through normal Supervisor shutdown and were confirmed absent. Engine-only captures were human-reviewed; no screenshot pixel analysis. Logs: `041-ui-player-prepare.log`, `041-ui-player-launch.log`, `041-ui-player-observe.log`, `041-ui-player-clean-source.log`, `041-ui-player-cold-launch.log`.
- `041-ui-player-package-audit.log` read the actual pack through the native reader:43 Content entries; target descriptor begins INXRTEX1 and is187bytes; scene→material→target catalog edges are present. Output contains Content.inxpkg,AssetCatalog.inxcat,Runtime,Modules and manifests, with no loose Assets/Library/ProjectSettings/Packages. Scene/material documents inside the container remain serialized documents; this does not assert cryptographic protection or that every payload is a custom binary format.
- No engine implementation changed this continuation, so the previous6048passed/11skipped full Python result remains the latest implementation regression rather than a newly claimed run. Original041Lab09 is clean Edit,Console0warning/0error. A09 View/history/reflection and other-platform matrices still pending. Next scope: remaining offscreen Camera/view contracts and a permanent authored monitor/reflection demo.
- Final gates:20 generated Learn pages verified and normal git diff --check passed (`041-ui-player-docs-check.log`, `041-ui-player-whitespace.log`). Captured build job evidence reports17.55s,diagnostics=[],executable_exists=true (`041-ui-player-build-job.log`). Process inspection confirms only editorPID19820 remains; both test Players are absent. Final project query confirms09_AudioVoiceField,dirty=false,play_state=edit.

## 2026-09-13 — UIImage persistent asset slot and document-backed Inspector

- Strict recount: main93/256, Compute36/76, total129/332=38.9%,203pending. Full-scope estimate12–18 solo full-time engineering weeks. A09 remains incomplete; no checkbox is awarded for source/API existence alone.
- UIImage now declares one `texture` asset field accepting concrete Texture or RenderTexture references. Old `texture_path` documents migrate once at component decoding; the old property is authoring shorthand, not another serialized slot or missing-GUID fallback. Anonymous RenderTextures retain a private live override, while raw field/document reads keep the authored GUID. UI static draws use the existing GUID cache, not eager CPU pixel loading. Imported target draws use the shared native GPU owner.
- `Texture.Sampled` is a field constraint, not a new resource identity: codecs, Inspector drops, clipboard, nulls and deleted references preserve the concrete type. UIImage uses the common asset drawer. Python asset drawers now use the existing component document transaction and raw value reads; native property setters retain their own adapter.
- First regression:268passed/1skipped; transaction suite66passed; first full suite6046passed/11skipped in345.05s. Actual visible authoring then found two missing drawer connections: semantic FieldType→asset-kind projection and the old first-drag-type override. Both were corrected in the common drawer/model, not with runtime exception suppression. Regression now exercises the real IGUI model construction, missing/cleared references and override/Undo. Final focused143passed in9.63s (`041-ui-asset-drawer-final.log`); the first full result is not final evidence for these later fixes.
- Harness corrections were separate from engine fixes: set the camera target's required depth attachment, search for a collapsed/offscreen hierarchy child, and locate the actual Inspector object-field rather than the similarly named Project item. A failed visible run exited the editor; its temporary probe assets were then removed through normal recoverable Project deletion after reopening the unchanged original09 scene. Source fixtures and diagnostic logs remain local under dev.
- Final complete Python regression **6048passed/11skipped/0failed in349.19s** (`041-ui-asset-final-full-python.log`). The trailing broken-shader/invalid-layer messages belong to negative fixtures; pytest exited0. The later143-test focused run and this complete run include the real drawer/model tests.20 generated Learn pages and normal git diff --check passed. No native rebuild was required by these Python-only changes.
- Accepted visible run: `041-ui-asset-visible-accepted.log`. Actual Project→UIImage Inspector drag assigns the concrete RenderTexture GUID. Save/switch/reopen displays output already in Edit, Play displays live camera output, and384×216→517×291/HDR/4× reimport retains consumption. Engine-only captures `041-ui-asset-inspector-drop.png`, `041-ui-asset-producer.png`, `041-ui-asset-edit-game.png`, `041-ui-asset-reimport-game.png` were human-viewed; no pixel analysis. Temporary probe assets were deleted through recoverable Project history; original09 restored clean Edit, Console0warning/0error. Only visible editorPID46688 remains (`041-ui-asset-accepted-editor.log/.err.log`); no pytest/build runs remain.
- English/Chinese RenderGraph guide documents direct image authoring and transient override semantics. Full Player/multi-platform acceptance is not claimed. Next scope: final Cook→Windows Player consumption with persistent Camera/material/UIImage identities, then remaining A09 platform/view matrix. No commit/push or remote artifact publication.

## 2026-09-10：隔离 CTest headless 插件缓存

- 可见编辑器运行时会持有 Hub 共享插件缓存；原 `infernux.headless` CTest 直接使用用户缓存，编辑器同时打开时可能在 `.inxpkg.tmp` 写入阶段被 Windows 拒绝。这是测试隔离缺口，不是引擎运行时回退问题。
- `cpp/tests/CMakeLists.txt` 现在为 headless 测试设置构建目录下独立的 `INFERNUX_PACKAGE_CACHE_ROOT`，不再和可见编辑器或 Hub 争用用户级缓存。没有加入重试或吞错路径。
- 重新配置 Windows Release 后，`infernux.headless` 单测通过；在独立缓存根下，headless、GPU buffer/kernel/mesh/gizmo/physics exchange、mesh publication/retirement、input manager 和 mesh artifact 共 `11/11` 通过。
- 完整 Windows Release CTest 随后 `75/75` 通过（总耗时约 81.82 秒）。Taichi RHI 测试中非法的 3 字节 upload/readback 按公开 `RhiResult::invalid_usage` 返回合同断言，未改成异常吞咽或重试。

## 2026-09-10：真实编辑器启动与 GPU 编译器载荷回归

- 源码工作区使用 `PYTHONPATH=python` 时，公开 Python 包不再假定源码树内已经有 wheel 私有载荷；`INFERNUX_NATIVE_MODULE_DIR` 现在会按构建约定解析 `gpu-jit-wheel/Infernux/_compiler/taichi/_vendor/taichi`。显式 `INFERNUX_GPU_JIT_VENDOR_DIR` 仍是发布/隔离安装时的权威路径，不增加运行时下载或猜测式回退。
- 在 `conda activate infernux` 下启动真实 Windows Infernux041Lab 编辑器，使用最新 Release 原生模块，经 MCP 进入 Play。GPUJelly 的 `inx.compute.launch` 连续运行无 GPU compiler import error；MCP console 读取结果为 `errors=0, warnings=0`。
- 10 秒热态性能窗口采集 240 帧：完整帧平均 `0.7661 ms`、P50 `0.5288 ms`、P95 `2.3106 ms`、P99 `3.8691 ms`、最大 `4.8875 ms`；game-only 平均 `0.2512 ms`，render 平均 `0.0551 ms`。这只证明当前启动/准备路径与热态计算可运行，不代表真实动态刚体交互、完整帧硬门槛或 1/8/32/128 阶梯已经通过。
- 因此本轮不新增计划勾选。当前硬计数保持 `16/235`（约 6.8%）；按可复用工程纵向切片约 65%，按最终用户验收约 36%。A03.4、A05 正式双 JIT 发行、A01/A02/A04/A06—A15 的完整消费者验证仍未收口。

## 2026-09-10：补齐真实求解后冲量公共边界

- Jolt `ContactConstraintManager` 新增按需的 applied-contact-impulse snapshot，在临时约束缓冲释放前读取每个接触点的实际总求解 lambda，并保留 body/sub-shape 身份、世界接触点、A→B 法线和线冲量；普通物理默认关闭，不增加常规项目的复制成本。
- PhysicsWorld/pybind/`Infernux.physics.Physics` 暴露 `set_contact_impulse_stream_enabled()` 与 `get_contact_impulses()`，以 NumPy SoA 交付给软体耦合器；初始化前开启的请求也会在 PhysicsSystem 建立后生效。接口文档明确这是最近完成固定步的 resolved snapshot，不是 Transform 推导或质量乘速度估算。
- 新增 `test_contact_impulse_stream_publishes_actual_solver_impulses`，真实下落球与静态地面验证数组布局、body/sub-shape 身份、有限值和非零 solver impulse；`python/test/test_integration_physics.py` 为 `107 passed`。Windows Release 增量构建成功，8 项 Vulkan/网格/GPU CTest 为 `8/8 passed`。
- 这完成 A03.1 的接触点、法线和求解后冲量公共边界，但不宣称 A03.4：GPU 驻留窄相/反馈、多接触归并、固定步内双向耦合、真实 041Lab Rigidbody↔软体和 300 FPS+ 仍未完成。
- 进度更新为：细粒度勾选 `16/235`（约 6.8%）；按可复用工程切片约 64%，按 041 最终验收约 36%。剩余单人全职约 9–13 周，含性能长尾和跨平台回归的保守日历约 13–18 周。

## 2026-09-10：041 计划硬口径盘点与重新勾选

- 重新按“有实现 + 有独立回归证据 + 未混入项目特判”盘点计划。当前明确勾选条目为 `15/235`（约 6.4%）；这个数字只反映计划的细粒度硬门槛，不把大量已完成的局部纵向切片误报为整项完成。
- 本次新增勾选的是 A03.1 的两项：Box/Sphere/Capsule/Cylinder/已 ready 凸 Mesh 的预测穿透覆盖（包含 center、旋转、缩放、compound 边界）以及 Collider 直接几何查询不使用世界 layer/trigger/pair 过滤。真实物理集成回归与 041 Vulkan/网格/GPU 8 项 CTest 均通过。
- 当前仍未勾选且决定收口的主项：A01 权威 schema/DataAsset/MCP 全链路、A02 多场景、A03.2 关节、求解后真实冲量与 GPU 驻留窄相、041Lab 刚软体双向 300 FPS+ 及 1/8/32/128 阶梯、A04 资源代际完整模型、B01/B04/B05/B06 双 JIT 正式分发、A06—A10 作者/渲染能力、A12—A15 迁移与发布。
- 本轮验证：`python/test/test_integration_physics.py` + `python/test/test_mesh_renderer_numpy.py` 为 `125 passed`；`infernux.input_manager`、`mesh_artifact`、`compute_buffer_gpu`、`compute_kernel_gpu`、`compute_mesh_gpu`、`compute_physics_exchange_gpu`、`mesh_publication_gpu`、`runtime_mesh_retirement_gpu` 为 `8/8 passed`。pytest 只留下工作区 `.pytest_cache` 权限 warning，无测试失败。
- 按工程切片而非 checkbox 计数，当前约 63%；按最终用户验收约 35%。300 FPS 双向软硬体仍是未通过硬门槛，不因本轮物理 API/法线回归通过而提前收口。单人全职剩余约 9—13 周，包含性能长尾与平台回归的保守日历约 13—18 周。

## 2026-09-10：补齐通用固定步接触快照边界

- 物理层新增按需开启的 `Physics.set_contact_event_stream_enabled()` 与 `Physics.get_contact_events()`。开启后，Jolt 最近完成固定步的 resolved 接触事件以 NumPy SoA 返回：事件类型、body/sub-shape 身份、世界接触点、接触法线和相对速度；默认关闭，普通项目不承担接触事件缓冲成本。
- 该接口保留固定步语义：快照在 `Jolt Step → contact resolve → contact dispatch` 后可读，并在下一固定步替换；不把质量乘速度冒充 solver impulse，软体反馈继续通过现有批量 `apply_rigidbody_impulses` 提交。
- 新增真实 Jolt 集成回归，验证下落球与静态地面的接触快照布局、有限值和 body 身份；`TestRigidbodyStateBatch::test_contact_stream_publishes_resolved_geometry_for_custom_solvers` 通过。Windows Release 增量构建和 8 项 Vulkan/网格/GPU 回归全部通过（100%）。
- 这只是 A03.1 的公共数据入口，不宣称完成双向软硬体耦合：solver 后冲量、GPU 驻留窄相/反馈、多接触归并、041Lab 300 FPS+ 门槛仍然未完成。
- 当前估算微调为：可复用工程切片约 63%，按 041 最终验收约 35%；A03.4 的严格门槛仍由真实 041Lab 双向交互、300 FPS+ 以及四平台交付决定。单人全职剩余约 9–13 周，含性能长尾和跨平台回归的保守日历约 13–18 周。

## 2026-09-10：收紧 Player 启动边界并完成全量回归复核

- 原生 `init_renderer` 的异常跨 Python 边界统一转为 UTF-8。Windows 的 `std::filesystem`/系统 API 可能用活动代码页生成 `what()`，旧 Player 会把它二次解释为 UTF-8，最终只留下误导性的 `UnicodeDecodeError`；现在保留真实启动异常文本，不增加启动重试或回退路径。
- 在 `conda activate infernux` 下重建 Windows Release，并执行 `prebuild_player_runtime`，同步刷新官方 Windows Player 的 `Runtime.inxrt` 与 `.inxpkg`，避免编辑器源码已更新而导出 Player 仍锁定旧原生模块。
- MCP 缺失的完整编辑器→导出→Player 启动用例重新通过；此前的 Player 启动失败已消失。MCP 官方包卸载/重启用例单独以真实子进程重跑通过。
- Python 全量回归在可写的同盘临时根下达到 `5653 passed, 13 skipped`；在仓库 E: 下直接使用 pytest `--basetemp` 时，唯一失败是测试环境把引擎项目留在 C:、shader 临时文件放在 E:，Windows `os.path.relpath` 跨盘必然拒绝。将 `TEMP/TMP` 统一到 E: 后，该用例通过；这不是引擎行为回归。
- `git diff --check` 通过。当前估算按可复用工程切片约 62%，按 041 最终验收约 34%，严格 checkbox 仍约 1% 级别；A03.4 的 300 FPS 双向软硬体、GPU 驻留窄相/冲量、资源代际生命周期、四平台玩家验收仍未完成。剩余单人全职约 9–13 周，含性能长尾与跨平台回归的保守日历约 13–18 周。

## 2026-09-10：补齐凸 Mesh 的只读穿透查询

- `Physics.compute_penetration` 现在与 `Collider.closest_point`/`raycast` 一样，支持已经发布且 `convex = true` 的 `MeshCollider`。查询复用 Jolt 的已 cook 形状，不修改 Transform、不注册 body、不同步 broadphase，也不偷偷启动异步 cook。
- 非凸 Mesh 或尚未完成 cook 的 Mesh 继续直接报错；没有用近似盒体、无命中或 fallback 掩盖资产状态。Python 公共文档与 041 计划已同步为这一边界。
- 在 `conda activate infernux` 下完成 Windows Release 增量构建。真实集成回归 `python/test/test_integration_physics.py`：`105 passed`（仅工作区 `.pytest_cache` 权限 warning）。新增用例覆盖凸 Cube Mesh 与 Box 的穿透、分离后的无命中，以及非凸 Mesh 的显式错误。
- 该切片完成了 A03.1 的凸 Mesh 穿透入口，但不宣称 A03.1 收口：compound ignore、过滤合同、批量复用热路径和 solver 接触/冲量输出仍待实现。后续优先继续补引擎公共物理生命周期，不修改 041Lab 参数来制造验收结果。
- 进度估算（按本轮结束）：按已完成的可复用工程切片约 61%，按 041 全部最终验收约 33%，严格 checkbox 约 1% 级别；后者仍被 300 FPS 双向软硬体、四平台与最终用户交付门槛压低。剩余工期按单人全职约 9–13 周，包含性能长尾和跨平台回归的保守日历约 13–18 周。
- 复核说明：GPU smoke 必须显式指向 Release 构建产生的私有 compiler vendor 目录；未设置该环境变量时，源码树本身不会假装拥有 wheel 载荷而会准确报缺失。设置 `INFERNUX_GPU_JIT_VENDOR_DIR` 后，`compute_mesh_gpu_test.py` 输出 `INFERNUX_COMPUTE_RESIDENT_MESH_OK`，`compute_physics_exchange_gpu_test.py` 输出 `INFERNUX_COMPUTE_PHYSICS_EXCHANGE_OK`。全 Python 回归在可写 basetemp 下已跑过 2134 项后遇到与本轮无关的 headless `Input.warp_cursor` 平台语义失败；物理+网格核心切片为 `124 passed`。
- 同一 Release 通过 CTest 的 7 项引擎侧切片：`mesh_artifact`、`compute_buffer_gpu`、`compute_kernel_gpu`、`compute_mesh_gpu`、`compute_physics_exchange_gpu`、`mesh_publication_gpu`、`runtime_mesh_retirement_gpu`，结果 `100% tests passed`，Vulkan 设备测试总耗时约 17 秒。
- 修复输入合同的最后一个已知回归：`Input.warp_cursor` 在 `confined` 状态下不再与窗口管理器竞争绝对定位，必须先解除 confinement；这与 visible/non-relative warp 规则一致。Windows Release 重建通过，`test_input.py` 为 `42 passed`，`infernux.input_manager` CTest 通过。

## 2026-09-10：Taichi compiler-only 主目标继续收窄

- 在 `external/taichi_for_infernux/cmake/TaichiCore.cmake` 中把上游 `taichi/jit/jit_session.cpp` 从 `taichi_core` 主目标排除。它只提供 LLVM/CUDA/AMDGPU 的运行期 JIT session 工厂，而 041 的 Taichi 角色是由 Infernux 自有 RHI 接管的 Python→SPIR-V 编译器；保留它会把已禁用的 LLVM-facing runtime 表面继续编进主库。没有删除上游源码，仍保留可追溯的 Apache 许可证与 NOTICE。
- 在 `conda activate infernux` 下重新生成 Windows Release，gpu-jit-wheel 安装阶段成功，`taichi_core`、`taichi_python`、`infernux_taichi_rhi` 及引擎本体全部重新链接。首次构建遇到可见编辑器占用制品目录，停止该已知进程后重建通过；这属于文件锁，不是编译或链接失败。
- 真实 Vulkan 回归通过：`compute_kernel_gpu_test.py` 输出 `INFERNUX_PUBLIC_GPU_KERNEL_OK`，`compute_mesh_gpu_test.py` 输出 `INFERNUX_COMPUTE_RESIDENT_MESH_OK`，`compute_physics_exchange_gpu_test.py` 输出 `INFERNUX_COMPUTE_PHYSICS_EXCHANGE_OK`。编译对象清单不再包含 `jit_session.cpp`。
- 受影响的 Python 侧法线/物理合同回归也通过：`test_mesh_renderer_numpy.py` 与 `test_integration_physics.py` 共 `123 passed`（仅因工作区 `.pytest_cache` 权限限制产生一个 pytest cache warning）。
- 这一步只关闭了一个明确无用的 LLVM runtime translation unit，不宣称 Taichi 已完成整体裁剪、GPU 常驻绘制链路或 041Lab 300 FPS 双向软硬体验收；下一步继续按依赖图裁剪字段/ndarray/AOT 周边，并优先补齐系统侧法线、Mesh 发布时序与 GPU 物理交换的公共生命周期。

## 2026-09-09：Taichi SPIR-V 首次由 Infernux RHI 直接执行，Jolt 宽相候选落地

- 本轮继续拆分编译与执行：默认构造的内部编译上下文无需 Infernux 图形引擎、VkDevice、队列、runtime materialization、显存或 Taichi ndarray 分配。独立 Python 进程用外部连续数组参数生成 SPIR-V，得到 `INFERNUX_COMPILER_ONLY_SPIRV_OK`；显式 `Program(None)` 被拒绝，不把空设备误作自动探测。当前仍借用 `Program` 类型壳和上游内部 Python AST，独立 compiler service 与 Infernux `kernel/index` 前端尚未完成。
- 正式 Windows wheel 重建审计为 25,169,674 bytes；Taichi 相关仅 5 个条目：私有 loader/包标记、单一原生模块、Apache LICENSE 与 NOTICE。隔离安装后能从 wheel 创建无设备 Vulkan compiler context；没有安装 field/ndarray/SNode/AOT/UI/profiler 作者表面。原生模块仍链接待裁的 gfx runtime/AOT 对象，因此文件清单通过不等于源码依赖图已裁完。

- 在通用物理接口新增 `Physics.query_rigidbodies_in_bounds`，直接调用 Jolt broadphase AABox 查询并只返回范围内实际拥有 Rigidbody 的唯一对象；layer/trigger 规则在进入结果前执行。真实集成用例放置 3 个近场 Rigidbody、128 个远场 Rigidbody 与 1 个近场静态 Collider，查询稳定只返回 3 个近场刚体。该能力用于让软体耦合复杂度依赖空间候选/活跃接触，而不是世界总刚体数；它不是 041Lab 专用分支，也不声称已完成 GPU 窄相。
- 增加 `Physics.get_rigidbody_box_states`，在一次 native 调用中扁平化候选刚体的 BoxCollider，输出 body 索引、世界中心/旋转/半尺寸、摩擦和弹性 SoA；可写入容量大于当前结果的 CPU `inx.buffer`，以 count 标记有效前缀。刚体运动状态快照也允许超额容量并只写候选前缀，候选数波动不再要求重新分配。该实现复用 Collider/Transform 权威参数并与 Jolt 的最小半尺寸一致，不遍历世界、不读渲染插值、不把 Sphere/Mesh 猜成 Box。旋转、非均匀缩放、center、混合 body 和复用容量测试通过。
- 编译 fork 的 `CompiledKernelData` 新增 Infernux 私有产物边界，输出各 task 的 SPIR-V、线程信息、参数索引以及具名 `resource_kind` / `binding_type`。不再把 Taichi 内部 BufferType 整数暴露给引擎作为 ABI。真实 affine kernel 揭示动态数组 range-for 的 advisory 总线程数固定为 131072；最终派发不能沿用这个数制造小负载开销，而应由 `inx.compute.index(buffer)` 的权威域和实际 buffer 长度决定。
- 通用 `rhi::ComputeKernel` 从“连续 storage binding”扩展为稀疏 slot 及 uniform/storage 类型布局；`ComputeBuffer` 增加 Uniform usage，以承载只读 kernel 参数块。保留已有连续 storage 构造，普通引擎 kernel 不需要理解 Taichi。真实 Vulkan 测试把 Taichi 生成的 arguments uniform（binding 0）和外部数据 storage（binding 2）交给 Infernux RHI，按 128 个元素/128 线程组宽度只派发 1 个 group，逐项得到 `i * 3 + 11`；没有调用 `Program.launch_kernel`、Taichi ndarray 分配或 Taichi command list。
- 初次直连失败得到全零，原因是引擎把 arguments 错误声明为 storage descriptor；修正的是通用编译产物/RHI binding 合同，没有增加重试或兼容 fallback。测试失败清理也显式释放 RHI 对象，避免 engine compute lease 保护遮蔽原始断言。重新编译后 `INFERNUX_SPIRV_ENGINE_RHI_OK`、编译边界、传输顺序、旧 runtime 对照与 host release 全部通过。
- Windows Release `_Infernux` 和 fork 原生模块增量构建成功。JIT/compute 122 项通过，无设备编译 native 准入 6 项及 SPIR-V 独立进程测试通过，Jolt 宽相/批量状态/Box SoA 定向通过，独立引擎 SPIR-V buffer Vulkan 测试通过，Taichi→Infernux RHI 真实 Vulkan 测试通过。旧 hpc 缺后端提示已改成“内置 Vulkan kernel frontend 未初始化”，不再错误要求安装 Taichi 插件。当前 wheel 只安装私有 native binding/loader，不向用户安装 field/ndarray/SNode/AOT 前端；但 native 链接闭包和测试编译阶段仍依赖 Program/gfx runtime，Infernux 作者 `kernel/index/launch` 前端尚未完成。
- 重新估算：严格主计划 checkbox 仍为 2/235（0.9%），因为刚软体双向作用、300 FPS、四平台与最终消费者均未验收；按代码垂直切片工程约 34%，最终验收约 17%。单人全职剩余仍约 10—14 周，包含四平台、Runner Long 迁移与性能长尾的保守日历为 14—20 周。未修改 041Lab 参数换分，未提交、推送或发布。

## 2026-09-09：回到实际编辑器 Play，排除遗留 Player 干扰

- 用户指出先前局部进展没有解决实际开始 5 FPS、稳定 70 FPS，并明确对照 Unity 编辑器内同任务约 300 FPS。本轮停止扩大局部数值探针，直接经现有 MCP 查询桌面项目并进入/退出真实 Play；项目参数为 9 子步而非此前探针 8 子步，scene dirty=true 始终保留，没有保存场景、切场景或关闭编辑器。
- 发现两个历史测试 Player（原始与 Updated）仍运行且无 MCP TCP 监听，但均有原调试控制文件。通过只读进程环境取得指定通道键、复用原认证 observe 查询，两个 GPUJelly 均为 broken_script，实际没有求解。不要把其约 1200 次/秒 renderer counter 当作有效游戏性能或屏幕实际呈现率。它们持续提交图形工作，NVIDIA 全卡快照当时约 70% 利用率；不是这次创建的 headless 求解器。
- 读取三个历史 Player Content.inxpkg 的 pyc 但不执行，确认原始、Updated、Latest 都还是优化前的 project_color（无局部顶点 p00…p32）。前几轮改动没有进入这些交付包。此事实不能替代用户明确的编辑器内问题。
- 临时分析工具 psutil/py-spy 安装在 dev/profiling-tools，不进入引擎依赖、插件或运行期自动安装。外部采样成功采集 Updated 903 栈、编辑器第一次 Play 2440 栈；原生无完整符号的最近导出名不视为准确函数定位。首次窗口出现 max 2467.20 ms、fixed 多次补算；但外部栈采样会扰动目标，且 avg120 含编辑空闲帧，所以 265.73 ms 混合均值不能作为干净启动 FPS。已向用户纠正这一点。
- 首次采样结束后，完整 Play 热态窗口约 12–13 ms/帧；fixedFrame 约 17 ms，无固定步帧约 5.2–5.5 ms，CPU gameplay 约 1.8 ms/帧。当前编辑器还在使用安装的旧 backend.py，重复 inspect/类型/别名工作尚未更新；加载的 Vulkan validation 也在栈采样中明确存在，不是独立 Release 探针。
- 用户确认只看编辑器后，通过两个既有 Player 调试通道发送 normal shutdown，进程退出已确认，没有强杀或删除制品。随后无外部采样器重新 Play，两个完整连续窗口为 8.38/8.40 ms 平均帧时间（约 119 FPS）、P95 14.68/14.63 ms，fixedFrame 13.94/13.90 ms，无固定步帧 4.36/4.42 ms，每帧平均固定步 0.42。此前核已预热，不作为冷启动修复证明；没有把全部跨轮变化归因于单一因素。仍明显没有达到用户参考目标。
- 最后已通过 MCP Stop，project.info 确认 edit、原 01_XPBD_Jelly、dirty=true。新增只读包审计、受限实时 Player 观察/退出及栈摘要脚本，更新计划的真实编辑器性能验收要求。当前回合没有宣称已修复 5 FPS、达到 300 FPS、解决 Player broken script 或 Taichi 路径已收口。后续需在实际可加载交付中更新后端、分离准备与固定步、接入驻留渲染并作同条件对照；完整 041 保持未完成。

## 2026-09-09：求解、蒙皮与法线串联 GPU 驻留验证

- 上轮 JIT batch 准入简化和实际回归属于进展。本轮继续 A04/A05 的驻留到绘制路径。源码确认 CompiledGraph 仍逐 dispatch 构造 LaunchContext 和绑定；没有在未证明生命周期安全/性能收益前增加持久 context 缓存。另确认桌面 GPUJelly 每个 update 用 NumPy 做蒙皮、面法线及 np.add.at 顶点聚合，再发布完整网格。
- 新增 dev/041_jelly_skin_kernels.py 作者计算原型，沿 inx.compute.hpc(device="gpu") 实现蒙皮及法线；顶点到三角形的邻接表在初始化时构造，法线采用每顶点遍历相邻面而非原子累加。所有外部数据仍为 NumPy/标量，使用插件绑定的引擎 Vulkan，不新建 GPU API 或更换设备。
- 探针 --skin-profile 从真实桌面 GPUJelly.py 提取纯 make_skin 和 _publish_mesh 函数，以捕获器替代网格发布，只比较 CPU 数值处理。没有导入/注册用户 Component、保存或重启编辑器，也不把捕获器耗时算作原生上传/绘制。
- 3458 个表面顶点、6912 个三角形、60 步含一次冲击，GPU 求解后直接在同一 batch 蒙皮和求法线。逐帧与原 NumPy 方法比较：最大位置差 4.76837e-7，法线分量差 4.78327e-6，全部有限。第一轮 CPU 处理中位 1.5961 ms，求解加 GPU 蒙皮法线同步调用 6.1321 ms；这些是范围不同的绝对成本，不能相除作为加速比。
- 第二轮新增私有 resident state 串联全部步骤，冲击也由 GPU kernel 写入，初始化后不逐帧读回或更新主机状态。末尾复制前验证 host 节点仍是初始位置、输出 host 数组仍为零，随后明确 copy_to_host；最终节点/表面位置/法线与普通同步序列对照通过，表面位置及法线差 0。
- 第二轮普通求解加表面处理同步调用中位 5.88505 ms，CPU 表面处理单独 1.47225 ms，驻留异步流水调用中位 3.22660 ms，最后 drain+copy 另需 1.63890 ms。异步数字不是 GPU 完成延迟或 FPS；有界队列可在槽用尽时等待。未测真实材质透明绘制和网格上传，未宣称低 FPS 已解决。
- 主仓 diff --check 通过。当前内核保留在 dev 实验原型，尚未接入真实 MeshRenderer 消费、AOT/Player 或项目脚本，不把验证通过冒充公开驻留 API 和演示已交付。完整 041 及 GPU 网格资源保活/生产者依赖/多相机等验收仍未完成；未提交、推送或发布。

## 2026-09-09：复用 batch 准入结果，减少 JIT 热路径重复检查

- 用当前桌面 JellyMaterial 和真实插件复测，cProfile 定位到每步 80 个作者调用重复做数组重叠扫描及成员查询。batch 入口已经验证全部声明数组，void JIT 调用改为按现有数组键一次完成成员确认及同跨度别名分组，保留非 batch/标量返回路径的原有语义。不以地址缓存推断主机内容是否变化，不跳过上传或并发资源边界。
- 纯生命周期/准入测试 22 通过（1.64 秒），覆盖同跨度 view、互不重叠切片，以及未声明复制、偏移、reshape、dtype view 的拒绝。实际 GPU 用例在 batch 入口之后禁止重复 overlap scanner，验证相同跨度两个参数继续得到正确计算结果。
- 首次 Ninja 构建缺少 MSVC include 环境而失败；在 conda infernux 下加载现有 VS Developer PowerShell 后构建完成，没有安装 SDK。新 infernux_taichi.inxpkg 的六项真实回归全部通过（137.62 秒），含引擎宿主、GPU 计算和独立进程无源码 AOT。
- 优化前后 20 步 cProfile 调用数 395680→322240。新分段探针普通同步中位 5.6384 ms、同步驻留 4.45835 ms，最终位置差 0；Python 调度约 1.19/1.16 ms，原生图准备仍约 2 ms。该对照没有并行运行其它 GPU 测试，但未控制完整桌面绘制，不把跨轮微小差别宣传为严格收益。
- 新版同算法 CPU/GPU 180 步冲击对照通过：同步 GPU 中位 5.751 ms、CPU 3.8965 ms，最大位置差 3.09497e-5 米。GPU 全调用目前仍比 CPU 慢，不能把上轮设备约 0.71 ms 当成完整求解或 FPS。桌面 GPUJelly 的 CPU 蒙皮/法线/网格再上传未改变。
- 更新路径调查文档，区分最新数据和历史测量。原生 CompiledGraph 逐 dispatch 创建 LaunchContext、查找参数/绑定的下一步机会仅检查源码，未盲目增加长期 context 缓存。尚未公开驻留 API、接入 GPU 网格渲染、替换桌面安装插件或发布 Player；完整 041 保持未完成。

## 2026-09-09：定位约束核并合入局部顶点复用

- 上轮真实设备计时和回归属于进展。本轮探针新增 --kernel-profile：按原算法顺序将每个作者 kernel 单独提交/同步，热态统计后与未拆分的同算法比较；最终位置差 0。拆分改变提交边界，计时只作定位，不冒充原始整图性能。20 个热态步骤中 predict/finish 每步设备合计各约 0.052 ms，project_color 64 次合计约 4.441 ms，主要设备耗时明确位于约束核。
- 排除 Python 标量被意外升级为 FP64：当前后端将普通 float 绑定 f32，项目数值数组也为 float32。编译优化已启用，fast_math=False 保持不变。九次相同分母除法复用倒数的局部实验没有明显收益（设备仍约 4.44 ms），未合入并移除该实验源文件。
- 在单个四面体内部先读取四个顶点，两个约束在局部标量上按原顺序更新，最后写回。保持六个四面体的顺序、八颜色、八子步、材质和接触参数不变。该作者算法明确使用互异四面体顶点，不把此变换无条件推广为允许任意别名的编译器优化。
- 局部实验设备区间降至 0.714816/0.712816 ms（普通/驻留）；普通同步全调用中位约 6.0865 ms、驻留同步约 4.85355 ms。300 步、两次冲击与旧算法交替验证通过：最大位置差 2.65613e-5 米、速度差 2.39918e-4 m/s，无倒置，体积始终在初始 90%–110%；同轮普通调用中位旧 9.99860、新 6.43410 ms。
- 在核对项目文件与实验副本只有预期核内差异后，用 apply_patch 将修改合入桌面 Infernux041Lab/Assets/Scripts/JellyMaterial.py。保存旧参考至 dev/041_jelly_pre_local_vertices.py，未改参数/schema/meta/GUID，未保存场景、切场景、关闭或重启编辑器。MCP 随后确认原场景仍 edit、dirty=true。移除已合入的重复实验源文件，保留真实项目源及旧参考。
- 新项目脚本独立 JIT 导出及无源码 AOT 驻留通过：dev/041-jelly-local-aot-cacb9f9f/Content.inxpkg；60 步位置差 0，体积比例 0.98901266，最小四面体体积 9.13105e-7。导出参考普通路径中位 6.05585 ms，AOT 异步驻留调用中位 5.75325 ms、末尾 drain+copy 1.85 ms。AOT 数字仍含调用链，不是设备时间或实际 FPS；探针原生路径已统一读取显式环境目录。
- 本轮实际性能修改在演示算法，并非宣称所有 GPU kernel 都有相同收益。GPU 网格/蒙皮/法线消费、准备执行计划、生成器高细分退化和完整 041 其它内容仍未完成；未重新打可启动桌面 Player、未推送或发布，视觉/手感没有因此自动通过。

## 2026-09-09：计算队列 GPU 时间戳与实际果冻测量

- 上轮独立 Release 回归恢复属于进展。本轮复用 GpuTimestampQueries 为 VulkanComputeQueue 增加显式启停的诊断功能；默认不创建 query pool，不插入 GPU 时间戳。启停是该队列的排空边界；平时仅在已有 fence 完成路径收集，没有为取时间另加等待。私有 _ComputeHost 提供启停与快照，NumPy/HPC 作者接口不变。
- GpuTimestampQueries 允许在 Release 中使用，但不会自动启用；指定实际计算 queue family 取得 timestampValidBits 和 timestampPeriod，不把图形队列位数当成所有队列的位数。保留现有回绕计算，并修正乱序收集旧槽覆盖最新样本的问题。
- 原生覆盖真实 compute 提交、失败记录、单槽复用、禁用时退休未完成提交、乱序收集以及关闭清空。RHI query/Vulkan executor/Taichi adapter 三项 CTest 通过（0.70 秒）；实际 .inxpkg 六项回归通过（124.13 秒），包括宿主绑定诊断启停、真实 GPU 与无源码 AOT。最后补充 queue-family timestampPeriod 取值后原生及实际果冻测试继续通过。
- 果冻探针增加 --gpu-profile。首次冷态一步产生数百次准备提交，原单提交断言正确阻止了错误计时；随后限定只统计预热后的 50 步，并确认每步恰好一条测量提交。探针改为 with 管理每步 batch，并在退出前释放 Program 引用，修正断言失败后活动 batch/租约未释放导致的次生清理错误。
- 原始 JellyMaterial（343 节点、1296 四面体、8 子步）设备提交区间中位：普通 batch 4.346976 ms、同步驻留 4.465760 ms；相应 CPU 完成等待约 4.69705/4.73360 ms。该区间从命令开始到结束，包含 dispatch/barrier 等设备执行，并不是单个 shader 的纯算术耗时；不再把这部分全部归为主机驱动等待。
- 固定范围诊断副本（此前已核对等价算法，只适用于固定拓扑）对应设备区间 4.526208/4.554416 ms，未观察到减少派发规模的明显收益。不是随机交错基准，不推导微小倒退；没有将固定节点数写入引擎或桌面脚本。
- 增加 --cells 诊断参数后尝试 12 格细分，原项目 make_body 的正体积断言失败，在求解前退出。其圆角投影会将多层同向外侧格点压到同一边界；没有绕过断言、没有产生大型网格 GPU 性能结论。这个生成器限制仍需修复并验证，不将当前小网格测量推断为任意规模表现。
- 未修改桌面项目、替换安装副本、提交或发布。GPU 网格/蒙皮/法线直接消费、准备执行计划、果冻质量及完整 041 其余项仍未收口；队列时间戳是进展，不是整帧性能已达标。

## 2026-09-09：修复独立 Release 构建状态并恢复实际包回归

- 上轮完成了原生同步验证，并发现 Release 启动访问异常，属于有证据的进展。本轮用不导入 Taichi 的最小 init_renderer 复现；给独立 Release 链接增加本地 PDB（保留优化），从 Windows 错误偏移 0x625d41 定位到 ShadowCameraResources unordered_map 插入。
- 检查发现 InxVkCoreModular.obj 时间为 06:00:43，而头文件已在 06:50/09:04 改变布局，VkCoreDraw.obj 已于 09:09 重编。MSBuild 的 CL.read.1.tlog 中构造函数、InxRenderer 和 SceneRenderGraph 对应源项均为 0 条依赖，VkCoreDraw 有 570 条。先前增量构建成功不能证明这些对象按新头文件构建。
- 使用现有 MSBuild 对独立目录的 InfernuxRuntime 与 _Infernux 做 Rebuild（BuildProjectReferences=false，保留已构建第三方库）。构造函数等依赖记录恢复为 524/690/548 条，相关对象统一重编；未修改运行逻辑、未增加启动重试或容器修复。随后同一完整实际插件回归 6 通过（122.52 秒），原生计算→图形消费测试 1 通过（0.37 秒）。这证明本机混合增量构建异常已消除；未证明导致旧依赖记录为空的历史原因。
- GPU 探针改为从 INFERNUX_NATIVE_MODULE_DIR 取引擎目录并输出实际模块路径，避免绑定选择 Release 而引擎构造参数仍写死 dev/engine-build；同算法 CPU/GPU 对照加入位置误差门槛，不只打印误差。
- 独立 Release 实际 JellyMaterial 三路 60 步结果：普通 batch 中位 9.86330 ms、同步驻留 8.52875 ms、异步驻留 4.75945 ms（P95 5.06170 ms），最后 drain+copy 15.34140 ms，三路最终位置差 0。异步路径 Python 调度约 1.70925 ms、原生图记录约 1.91640 ms。均是 CPU 墙钟/流水测量，不是纯 GPU shader 时间或游戏 FPS；本轮没有关闭系统验证层或修改驱动设置。
- 同一 Release 随后完成 180 步 CPU/GPU 同算法交替对照，第 90 步相同冲击：普通同步 GPU 中位 10.01620 ms、CPU 4.01165 ms，最大位置差 1.70693e-5 米，满足新增门槛。不能用异步多步吞吐与单步 CPU 延迟直接宣布 GPU 已反超，也没有降低子步或约束质量。
- MCP 确认桌面 Infernux041Lab/01_XPBD_Jelly 处于 edit、dirty=true，未保存/重载/关闭编辑器；未安装或发布新制品。普通网格 GPU 消费接入、真正 GPU timestamp、果冻画面/交互及完整 041 其余内容仍未完成。

## 2026-09-09：计算到图形队列的完成依赖验证

- VulkanComputeQueue 保存最近成功提交的 ExecuteResult，提供完成 timeline/value；失败提交不覆盖成功记录，销毁清空。timeline 是队列借用句柄，消费者必须在队列销毁前完成，不能当成独立保活所有权。
- 真实原生测试让计算提交后立即由 Graphics 角色的独立提交等待 timeline 并复制结果，CPU 只等待消费者 fence。连续八轮改变输入，单槽复用后结果正确；没有先在 CPU 等生产者。本机逻辑队列可能共用物理队列，不宣称独立物理队列拓扑已覆盖。
- dev/engine-build 的 DLL 被当前编辑器占用，链接返回 LNK1168，没有关闭或保存用户编辑器。改用独立 out/build/windows-msvc-release，原生测试重复通过（1 项，0.35 秒），相关引擎绑定及依赖重建成功。
- 独立 Release 的完整实际插件回归未通过：3 通过、3 失败（68.90 秒）。三个图形用例均返回 0xc0000005；faulthandler 定位在 engine.init_renderer，Windows 事件记录故障模块为该 Release 目录的 _Infernux.cp313-win_amd64.pyd，偏移 0x625d41。发生在获取 ComputeHost/执行 kernel 之前，根因尚未确认，不能将其归因于 Taichi 求解，也不能宣布 Release 可交付。
- 本轮只证明底层同步通路，未接入普通 MeshRenderer 实际绘制，未修改桌面项目或安装副本。GPU timestamp、作者生命周期、求解到蒙皮/法线/绘制的设备内链路，以及 Release 启动异常仍未收口。

## 2026-09-09：无源码 Player AOT 的异步驻留回归

- 上轮异步状态及资源释放属于已验证进展。本轮补足 AOT 证据：新进程删除导出目录、禁止 inspect 源码读取，通过 Content.inxpkg 内 pyc/AOT 加载作者函数；浮点/整数两组数据连续运行 12 步，禁止每步 ti.sync，主机数组保持旧值直到读回。数值精确符合预期，保留 state 引用跨卸载后已 closed 且设备数组引用清空。定向实际包/AOT 回归 1 通过（63.74 秒）。
- 更新实际 JellyMaterial 导出探针，增加 load-resident 模式，不修改桌面项目。使用新构建插件从当前脚本重新导出，在 `dev/041-jelly-resident-7db2dd84` 产生 Content.inxpkg；另一进程只读取封包代码/计算制品并执行 60 步驻留计算。
- 实際果冻参考导出路径中位 9.80115 ms/P95 11.10690 ms；封包 AOT 驻留中位 5.83820 ms/P95 6.09078 ms，最后 drain+copy 另计 5.13 ms。位置差为 0，速度满足 1e-2 m/s 容差，体积比例 0.98901248、所有四面体正体积。不是最终游戏帧率或实机渲染验收，不能称为所有 Player 平台已完成。
- 核对普通网格与粒子 GPU 网格的消费接口，将 RHI 句柄/保活引用、顶点布局和计算→图形依赖的差异记入 `041-mesh-publication-map.md`。不以粒子专用渲染器冒充普通 MeshRenderer，也不把共享设备当作同步完成。
- 子仓 diff --check 通过。本轮未修改引擎公共 API、桌面场景、安装副本或发布远程制品；GPU 渲染衔接、作者生命周期及完整 041 其余内容仍待完成。

## 2026-09-09：驻留状态异步提交与销毁释放

- 上轮独立状态及数值/生命周期回归属于进展。本轮将私有驻留 batch 的退出从 compiler.sync 改为 Program._submit_pending，沿用已有 EngineStream 和 VulkanComputeQueue；普通同步 batch 不变。队列默认为 3 个槽，满时仍使用既有有界等待，不能宣称永不阻塞或无限提交。读回、关闭、插件卸载继续是完成边界。
- 实包测试禁止驻留步调用 compiler.sync，连续提交 12 步，再验证多实例交替、同步 batch 混用和结果。补充提交后丢弃 owner 的弱引用回收及后续同步，并把 Vulkan Validation Error 纳入实际 preload 测试失败条件。此测试不证明 GC 时 GPU 必然仍未执行完，仅覆盖提交后释放路径。
- 源码检查发现 Program::delete_ndarray 在 used_in_kernel 时跳过删除，没有后续删除记录，可能一直保留到 Program reset。改为销毁在用数组时 synchronize 后 erase，沿 Ndarray 析构进入引擎 RHI 退休，不新增待删队列或每帧扫描。销毁可以等待，不在普通驻留步等待。没有宣称已用显存曲线量化全部资源占用。
- 原生 program.cpp 已在 conda infernux/现有 MSVC 下重建并生成 `.inxpkg`。异步版首轮完整实包 6 通过（122.50 秒），新增 owner 丢弃/validation 条件后定向实包 1 通过（25.38 秒）；最终原生释放修复后的完整实包 6 通过（122.12 秒），单元 21 通过（1.48 秒）。子仓 diff --check、探针语法检查通过。
- 最终实际果冻 60 步三路：普通 batch 9.77195 ms、同步长驻留 8.76325 ms、异步独立状态 4.67290 ms（中位）；异步 P95 5.03743 ms，最后 drain+copy 另计 14.4718 ms，三路最终位置误差为 0。异步 scope_exit 包含提交及可能的队列背压，不是读回；该数据为不绘制果冻的 CPU 调度/流水吞吐探针，不能转换成游戏 FPS 或纯 GPU 时间。
- 仍未把驻留状态接入公开作者生命周期或 GPU 网格/法线/渲染消费者，也未修改桌面场景、安装项目副本、提交、推送或发布。完整 041 目标不变；GPU timestamp、准备好的原生计算计划和渲染衔接继续待完成。

## 2026-09-09：独立驻留状态的后端所有权

- 上轮参数绑定准备化是已验证进展。本轮继续 A05 跨帧基础：抽取共用 batch 存储分配，新增私有 `_ResidentState`，独立持有自己的 host/device 映射，不借用普通 batch 的可复用槽。创建时上传，后续 batch 退出不读回；内部 copy_to_host 明确同步，公开 compute.batch/NumPy 语义未变，没有新增公开 GPU 数组类型。
- 状态每步仍等待完成；异常退出完成已记录工作并清除活动作用域，不承诺回滚。传输边界拒绝 host 地址/dtype/shape 改变。backend 用弱集合跟踪状态，插件卸载先完成提交再释放所有状态和编译器；保留状态引用也不能继续执行或持有设备数组。
- 实包覆盖两个状态交替执行、穿插普通 batch、主机值不提前变化、按需读回、多实例数据隔离、异常后继续、活动作用域拒绝交叉读回、布局变化拒绝、关闭后拒绝、卸载清理及重新加载。补充测试首次因引用已删除的局部 provider 失败，改为读取当前正式 provider 后通过；没有把测试脚本错误当成引擎失败。
- 本地 CMake 已生成 `.inxpkg`，完整实包 6 通过（121.02 秒）；追加生命周期边界后的定向实包 1 通过（25.47 秒）；单元 21 通过（3.11 秒）。子仓 diff --check 通过。
- 实际 JellyMaterial 60 步三路对照通过：普通 batch/长作用域驻留/独立状态中位 9.8906/8.79605/8.93365 ms，P95 10.5957/9.67824/10.29826 ms，三路最终位置满足绝对误差 2e-5 m。独立状态仍有每步等待，不宣称异步 GPU 渲染或性能瓶颈已解决。探针后来将 scope_enter/scope_exit 命名改为准确边界，并将最终误差汇总扩展到所有参与路径。
- 未修改桌面场景、未安装新的项目副本、未提交/推送/发布。作者层生命周期、计算计划、GPU 时间测量与计算到渲染共享仍待实现；不能把内部 copy_to_host 方法当作要求用户手动管理显存的公开 API，完整 041 目标继续保持。

## 2026-09-09：固定位置参数布局在准备阶段生成

- 上轮类型键/编译注解分离是已验证进展。本轮继续 A05：JIT 和 AOT kernel 共享 `_prepare_argument_binding`，准备时确定参数名和固定位置布局；完整位置调用直接建立名称映射，不再重复 Signature.bind/apply_defaults。关键字、缺省值和参数数量错误仍走原 Python 绑定规则，没有捕获异常后重试或新缓存。
- 定向测试比较位置专用参数、默认值、关键字和非法调用的结果/错误消息；禁止 Signature.bind 后，完整位置调用仍正确。21 项定向通过（3.20 秒）；CMake 重新生成实际 `.inxpkg`，6 项实包回归通过（125.35 秒），覆盖实际 GPU 及独立进程 AOT。
- 最新相同 JellyMaterial 探针：普通/驻留中位 10.0850/8.7711 ms，Python 调度 1.8304/1.7573 ms，最终位置差 0。编辑器此时已处于 edit/dirty=true，与上轮 paused 条件不同，且完成等待/传输也下降，不能将全部差值归因于绑定优化或当作稳定提速承诺。未保存、重载或切换用户当前场景。
- 子仓 diff --check 通过。未提交、推送、发布或替换桌面副本；跨帧状态正式 API、native 准备图、GPU 时间测量和计算到渲染仍待完成，完整 041 目标不变。

## 2026-09-09：GPU 调用键与编译注解分离

- 上轮调查提供了实际瓶颈证据，本轮继续 A05 热路径：将 `_argument_key` 与 `_argument_type` 分开，前者只提取调用类型键并检查连续性/标量范围，后者只在需要编译类型时构造注解。JIT 特化命中、图键、AOT 特化查找不再反复构造 ndarray 编译注解；AOT 图入参验证也移除随即丢弃的数组注解。不增加新的缓存或失败后重试。
- 新增无 compiler 实例的键合同测试，覆盖长度变化不改 rank 键、rank/dtype/标量区分、非连续数组、大整数和非法标量。实包 JIT 测试在已准备后禁止调用注解构造，仍能实际执行 batch 并核对 GPU 输出；实包 AOT 测试禁止构造 ndarray 注解，加载后的图仍处理不同 dtype/长度并输出正确结果。
- 首轮 JIT 改动定向+实包共 26 通过（123.41 秒）。AOT 改动后重新通过 CMake 生成 `.inxpkg`，定向 20 通过（3.31 秒），完整实包 6 通过（134.69 秒），包括独立进程 AOT。子仓 diff --check 通过。
- 同一 JellyMaterial、关闭实验进程 validation 的驻留对照：普通/驻留中位 11.8849/9.81145 ms，P95 12.8365/10.5770 ms；Python 调度中位 2.1658/2.1347 ms，最终位置差为 0。相较上轮约 2.5 ms 调度有小幅改善，但不是随机交叉基准，不能承诺稳定提速幅度，更不代表求解/渲染瓶颈已解决。
- 保留完整 041 目标和原 NumPy 公共合同。新的跨帧状态/准备执行计划及 GPU 渲染衔接尚未实现；当前只是去除热路径中明确冗余的编译对象构造。未提交、推送、发布或替换桌面安装副本。
- 收尾尝试恢复编辑器时 resume 被拒绝；随即查询显示它已是 playing，场景 dirty=false，因此没有再次切换或重启编辑器。

## 2026-09-09：GPU 全路径、驻留与动态派发调查

- 新增 `041-compute-path-investigation.md`，区分源码事实、实际计时、对照实验和待验证判断。当前 ndarray 是引擎 Vulkan device-local 存储，但 batch 每次上传/读回全部数组，跨帧没有设备权威值；CPU 表皮/法线计算后仍重新发布网格。
- 探针通过私有 `_submit_pending` 分离原生记录、Vulkan 编码/提交与完成等待。60 步跨步驻留对照最终位置完全一致；validation 开启时普通/驻留中位 15.52/14.88 ms，关闭当前子进程 Khronos validation 后两轮约 12.34/10.79 和 12.23/10.44 ms。空队列原生/Python 同步分别仅约 0.0002/0.0011 ms。有任务完成等待约 5.5 ms 仍需 GPU timestamp 分离，不能等同纯 shader 时间。
- 从本轮实际执行后重新导出的 metadata 确认：动态 range 每次 131072 线程，project_color 另有 serial task；80 次作者调用实际为 144 次 dispatch。仅在 dev 建立固定 343 点/27 单元的诊断副本，降为 80 dispatch；300 步两次冲击位置差 3.25e-5 m，无倒置且总体积 90%–110%。关闭 validation 后驻留约 10.31 ms，未证明这个调整单独解决性能，禁止把硬编码拓扑放入产品。
- 记录后续顺序：准确 GPU 时间 → 准备一次的执行计划 → 规模正确的派发 → 跨帧状态与明确 NumPy 同步边界 → GPU 蒙皮/法线/渲染共享资源。保持现有 host/RHI，不能靠 SHA 扫描 NumPy 判断变化，也不靠暗中 CPU 回退掩盖性能。
- 上轮内部屏障调整后的完整 payload 为 6 通过（140.39 秒）；本轮重建诊断绑定后的实际 `.inxpkg` payload 再次 6 通过（128.16 秒）。主仓/子仓 diff --check 无空白错误，仅已有行尾提示。未发布/推送，也未将实验副本安装进桌面项目。
- 为减少测量干扰通过 MCP 暂停了原本处于 Play 的编辑器，结束后已执行 runtime.resume 并查询确认 playing，场景未变且 dirty=false。完整 041 和果冻视觉/性能验收仍未完成。

## 2026-09-09：内部计算屏障范围与耗时归因

- 检查共享 RHI command adapter，原先每条内部 Barrier 都覆盖 AllCommands/Host。现仅将内部边界限定为 ComputeShader/Transfer 及对应读写（含 UniformRead）；入口保留跨提交同步，出口保留 HostRead。没有删除必要依赖或增加缓存/回退路径，范围与此列表支持的命令匹配。
- 正常构建新 `.inxpkg`；相同算法 180 步 GPU 中位 15.0052 ms/P95 17.47645 ms，CPU 3.9541 ms/P95 4.80014 ms，最大位置差仍 1.70693e-5 m。没有显著性能改善，不能把此调整描述为瓶颈已解决。
- `infernux.taichi_compute_pipeline` 真实 Vulkan CTest 通过（0.37 秒），覆盖填充→计算→拷贝、跨提交依赖和命令列表销毁等。完整 payload 回归随后执行。
- 进一步检查 GfxRuntime::synchronize→flush→EngineStream::submit→VulkanComputeQueue::Submit 的调用链，确认 probe 原 `sync_ms` 同时包含 CPU Vulkan 编码、提交和等待，不能当成纯 GPU 时间。改名 `native_encode_submit_wait_ms`；之前的记录保留原测量值但必须按此边界解读。下一步需定位 CPU 编码与 GPU 执行各自成本，避免把全部 6 ms 归咎于求解数学。

## 2026-09-09：并行 Jacobi 实验提速但物理失败，不进入场景

- 查阅并行 PBD/XPBD 文献检索结果。综述 `https://mmacklin.com/EG2015PBD.pdf` 讨论并行 Jacobi 的修正汇总及欠松弛；完整 PDF 超出浏览器读取大小，另一份 block Neo-Hookean 预印本超时，未假称通读或实现其算法。
- 新增仅在 dev 的 `041_jelly_jacobi.py` 实验：保留 343 节点/1296 四面体、材料参数和 8 子步，四面体分别生成修正、顶点按 incidence 汇总，无浮点原子和跨线程写同一位置。由原 80 次小 dispatch 变成 48 次较大 dispatch，未改公共 NumPy/标量边界、未替换桌面项目脚本。
- 真实 Taichi/Vulkan 热步样本约 5.35–5.59 ms，但第 10 帧体积比仅 0.47769，min J=-0.04367；第 20 帧 min J=-0.60688。简单平均的收敛/体积保持失败，不能把提速当作验收成功，不能交付到用户场景。
- 为现有 probe 补逐帧体积比 0.9–1.1 与 J>0 的拒绝门槛，测量区间外计算质量指标，不污染求解计时。实验 28/35 帧失败，进程明确 exit 1；同一工具跑当前桌面 JellyMaterial 35 帧通过，热样本约 13.76–13.85 ms。此检查仅排除明显体积塌缩/翻转，不证明物理观感达标。
- 下一步需兼顾并行性与收敛，不能把原 GS 各颜色顺序更新直接换成平均后就宣称相同物理质量。原有桌面项目和 Player 未更新；完整 041 及果冻性能/回弹仍未完成。

## 2026-09-09：参数缓冲复用实验无收益，撤回额外状态

- 实验在 GfxRuntime 中按 dispatch 槽位复用参数 UBO，只在 stream 完成后重置，异步 flush 不重置。正常 CMake native 编译与 `.inxpkg` 构建通过；完整包回归 6 项通过（124.26 秒）。新增同 kernel 80/17/80 组不同标量参数，以及批次中途 scalar return 后继续 dispatch 的真实 GPU 检查。
- 同算法 180 步复测 GPU 中位 15.32895 ms/P95 17.57734 ms，CPU 4.01625 ms/P95 5.09295 ms，最大位置差仍 1.70693e-5 m。与前次无实质改善，不能证明新增缓冲缓存值得保留。
- 按 fire-forced 撤回本轮原生参数缓冲池和 cursor/recycle 状态，仅保留有效的连续参数覆盖/同步回归测试。没有回滚用户或早先的原生修改；正在重建恢复后的包，用户桌面安装与现有 Player 未被实验包替换。
- 性能下一步转向求解组织：80 次 dispatch、每颜色仅 27 单元的并行度；不再把小规模缓存优化作为主要解决方案。不降低子步、材料或网格规模来冒充改进。完整 041、真实果冻回弹和 GPU 性能仍未完成。
- 撤回后 native 与 `.inxpkg` 重建完成；新增连续参数/中途标量同步测试在恢复后的包中通过（`test_preload_compute_entry`，27.09 秒）。本轮没有给主干遗留实验参数缓冲池。

## 2026-09-09：计算图热路径去除重复符号构造

- `_flush_batch` 只在图缓存未命中时转换数组的 Taichi 符号类型和建立编译参数；热路径保留数组布局/别名及标量类型作为键，并绑定当前值，不保留作者旧数组、不跳过形状合同。
- 新增重复执行时更换数组对象/长度与标量值仍绑定新值、同秩复用图、不同秩重新编译的测试。preload/lifecycle 定向 19 项通过（1.53 秒），子仓定向 diff --check 通过。正常 CMake `infernux_package` 重建 `.inxpkg`，未手动复制库。
- 使用新包同算法 180 步复测：GPU 中位 15.2322 ms/P95 17.7707 ms，CPU 4.0123 ms/P95 4.6748 ms，位置差仍 1.70693e-5 m。相对前次两端均有小幅波动，P95 无明显改善，不能声明主要性能瓶颈解决。80 次小 dispatch 与物理质感问题仍未验收。
- native 编译管理器检查确认 kernel key 在 Kernel 对象缓存，并非每帧重新计算 SHA；不进行无证据的哈希删减。桌面已安装插件和独立 Player 暂未替换，不混淆开发包验证与用户制品交付。
- 新 `.inxpkg` 完整 payload 回归 6 项通过（128.69 秒），包含真实 Vulkan 包载入/计算与 AOT 路径；这证明本次热路径修改未破坏这些已有合同，不代表性能达标。

## 2026-09-09：用户否决果冻速度与回弹后的同算法基线

- 用户指出 GPU 仅 50 多 FPS、CPU 更快、回弹不像果冻，故暂停视觉美化优先定位。此前通过 MCP 添加 Jelly Softbox（Area light，3×2，intensity 8）与 Camera FOV 42 并保存场景；截图 `review/jelly-softbox.png` 只证明光照/构图变化，不构成果冻验收。编辑器已停止 Play，未重建此场景的新 Player。
- 用实际桌面 `JellyMaterial.py` 进行分段 Vulkan 测量，343 节点、1296 四面体、8 个颜色组、8 子步，每步 80 次 dispatch，每颜色组仅 27 个单元并行。热步 13.60–14.93 ms；Python 记录约 2.51–2.58 ms，图运行/提交约 3.70–4.82 ms，显式等待约 6.11–6.36 ms；上传约 0.45–0.72 ms、读回约 0.42–0.84 ms。这是包含同步的分段墙钟，不冒称纯 GPU timestamp。
- 为现有 `dev/041_gpu_jelly_probe.py` 加入 `--cpu-compare`，将同一作者函数分别由公共 hpc CPU/GPU 编译，保留同网格、参数、步长与第 90 帧相同冲量，不拿不同求解器的老 CPU 场景充当基线。
- 180 步、dt=0.02、8 子步，排除前 10 步：GPU 中位 15.6769 ms/P95 17.7987 ms，CPU 中位 4.1583 ms/P95 5.2185 ms；最大位置差 1.70693e-5 m。两端按压后的高度轨迹几乎一致，表明 GPU 慢与回弹设计差是两项独立问题。性能与真实果冻质感均未解决、未通过，不用数值一致替代验收。
- 后续优先减少细碎 GPU 调度/图提交及不必要往返，并补局部按压/释放的物理响应，保持 NumPy/标量接口、明确 GPU，不通过隐藏 CPU 回退或降质量掩盖瓶颈。完整 041 仍在推进。

## 2026-09-09：最新版平台包的果冻 Player 回归与下一批演示边界

- 重新启动桌面 Infernux041Lab 编辑器，通过 MCP Play 准备计算特化、Pause，再走正常 Windows 构建任务 `1b1e3a5a2ed14e8a9ad3ec2d3171f2b8`。22.125 秒完成，输出 `D:/Users/Chenlizhe/Desktop/Infernux041JellyPlayer-Latest/Infernux041Lab.exe`，无构建 diagnostics。
- 实际启动独立 Player。帧 744 与 5134 的运行时网格均为 2 项、403432 字节；gameplay_ready 为 true。空格交互已送达，0.1006 秒按压前后 height 从 1.556168 变为 1.476669，模拟时间持续推进。读取的运行日志中没有 ERROR、Traceback 或 failed 匹配；这不是全平台或长期稳定性结论。
- 实际 Game 渲染目标截图位于项目 `.infernux/mcp_sessions/20260909-071707-2f0385d6/review/jelly-latest.png`。已目视检查，当前仍是绿色半透明技术样例，不认定漂亮果冻 Demo 验收通过。
- A15.1 补充阶段交付边界：果冻视觉待验收；雪地首版高度场连续压痕、重置，先比较 128/256 网格，堆积独立实现，不宣称完整体积雪仿真；后续轮廓/世界 UI/物理场景依公共功能推进。没有新增私有 Demo 框架或运行时兜底。
- 检查确认 AOT 标量返回仍被显式拒绝。本轮未修改此合同，不能声明该缺口已解决；041 总目标仍未完成。

## 2026-09-09：primitive 切换修复进入正式平台包

- 正式windows-msvc-player构建session59398 exit0，CMake直接生成子仓库平台包。桌面编辑器未运行，PluginManager更新Windows包session9753 exit0。尚未重开桌面编辑器或重新导出带此修复的游戏。
- 扩大回归mesh_renderer_numpy、integration_scene、camera_contract共211 passed /8.09秒。此前四项Vulkan CTest证据保持有效。
- A05.2进度段更新为实际Windows AOT构建/Player/交互/驻留证据，删除已过时的“构建钩子未完成”描述；明确普通AOT Player编译器裁剪、AOT标量返回、显式Player JIT与跨平台仍待完成，未勾选整项或缩小041范围。

## 2026-09-09：双 Game Camera 运行时网格退休回归

- runtime_mesh_retirement_gpu_test新增--multiple-cameras，两台不同depth相机与Scene视图共存，创建64×64 Game target；要求game_camera_count=2、两条game_render_view_ids、图实际执行且draw_call_count>0。只创建相机但未创建target的早期测试正确失败，没有把相机登记当作渲染验证。
- 沿原192次双对象共享变形、删除一个仍共享、删除全部回到Gizmo基线断言。注册infernux.runtime_mesh_retirement_multi_camera_gpu。
- session46038四项requires_vulkan_device CTest全部通过15.95秒，覆盖资产间隔/连续发布、inline共享删除及多相机。本轮无新正式制品；上轮primitive→inline修复尚未入平台包。041其它功能和演示交付仍未完成。

## 2026-09-09：共享 inline 网格回归与 primitive 切换修复

- 新增真实 Vulkan runtime_mesh_retirement_gpu_test，先记录编辑器 Gizmo 驻留基线，再让两个对象进行192次相同inline变形；删除第一个后仍比基线多一份共享mesh，删除第二个后字节和条目都回到基线，且无预算淘汰/pending upload。
- 测试发现已渲染Cube再SetProceduralMesh时旧DrawCall仍保留builtin身份，变形上传未发生。MeshRenderer::SetProceduralMesh在HasSharedInlineMesh→自有inline切换时发结构通知，普通动态帧仍只发geometry通知。开发模块session27214重建通过；修复前测试因缺少新mesh失败，修复后通过。
- 注册CTest infernux.runtime_mesh_retirement_gpu；与已有mesh_publication_gpu及continuous共同运行，session57641三项通过11.01秒。此用例是Scene绘制、双对象共享及删除，不把它宣称为多Game Camera验收。
- 桌面编辑器已正常关闭以更新DLL，目前未重开。此轮primitive切换修复尚未进入正式平台插件/Player；远程制品和完整041仍待推进。

## 2026-09-09：正式 Player 动态网格驻留与重置验证

- session61284正式构建exit0，更新桌面Windows平台插件后编辑器exec97898、9741。NumPy mesh接口12 passed /2.60秒。导出Residency目录jobdf5ee396baea436fb57c445132c4cb99完成21.189秒；启动job5da4aa8d2d06479598b83c9b02fba1e8。
- 实际Player帧674、2718、5731三次采样runtime_mesh_entry_count均2，runtime_mesh_bytes均403432；旧路径曾269/583项、107/233MB。截图review/jelly-retired-mesh.png（session20260909-065527-1e32aa97）确认正常变形体仍被绘制，非删掉可见数据造成的低占用。
- MCP R重置：simulated_seconds80.26→0.12，高度1.55617→1.69657，重置后继续运行到5731帧仍保持上述占用。日志无ERROR；正常shutdown。此为当前单果冻场景验证，多对象共享/多相机/资源压力仍需扩大，不代替041全局GPU生命周期验收。
- 最新正式Player preload11906.4ms、bootstrap13649.8ms。Taichi中英文README更新Windows正常导出+AOT+键盘交互证据，保留其它平台/纹理互操作/完整发布未完成说明，未push。

## 2026-09-09：动态运行时网格无引用后退休

- VkCoreDraw 的运行时内容哈希缓存原先仅随预算淘汰。新增帧末/对象清理后的退休：仅 assetGuid 为空且顶点/索引 buffer 都只剩缓存引用时移出缓存，沿用现有deletion queue和lease计数。资产缓存与仍被对象/绘制引用的buffer不动，不等待device idle。
- 首次dev构建缺MSVC标准库路径失败；完整VsDevShell并保留Unicode用户环境后session72887编译exit0。新版编辑器exec49449、9741运行果冻87.54秒后暂停，Console error为空。此证据尚不证明GPU占用收敛。
- 正式Windows构建session61284仍运行，最近已完成VkCoreDraw编译、Release/LTO链接，后续需跟踪同一进程，再更新桌面平台插件、导出并用Player现有遥测采样运行时mesh数量/字节与实际画面。不可将此修改提前标为内存验收通过。

## 2026-09-09：全量 preload 扫描去除重复角色判断

- 保留 _read_path_declarations 的插件 enabled 检查：新作者脚本尚未写入归属表时仍必须按所在插件判断。全量 _source_paths 已执行的 Editor/Runtime 角色判断只做一次；单路径刷新默认仍独立检查。
- 首版把禁用检查一并略过，既有回归发现4个disabled新增脚本失败，已修正，未弱化测试。最终session77312：145 passed、5 skipped /48.70秒。
- 旧content-2e3c缓存已被正常回收，第一次0.003秒空扫描结果弃用。确认最新content-1d73d0e714a97afdce56bb08含registry后，最终93条路径扫描9.2975秒（无cProfile，非完整冷启动）。本轮尚未重建平台运行时，上轮Player实测18.0147秒仍是当前已验证制品结果。

## 2026-09-09：预加载优化正式 Player 冷启动回归

- 正式 cmake --build --preset windows-msvc-player（session22054）exit0，直接发布Windows子仓库 .inxpkg（85,655,936字节）及Runtime.inxrt；PluginManager更新桌面平台插件后重启编辑器exec44579、9741。
- Taichi完整包级回归session13801：6 passed /159.72秒，覆盖本轮实际包/独立进程AOT路径。
- 新导出Preload目录job85571bc5cbc54c9baeab438f08f49352完成25.148秒；启动job7ed5ee4ff1124a1b84f2cd6b868e3c0c。真实Player日志preload16176ms、bootstrap18014.7ms，相比先前约30.5秒bootstrap明显改善，仍非足够快的最终体验。
- observe GPUJelly awake/started、broken_script=false、simulated_seconds20.5、高度1.55622，日志无ERROR。优化没有跳过GPU初始化。随后MCP shutdown，编辑器保留暂停。尚无远程发布/CI验收，不标记041或演示完整完成。

## 2026-09-09：Player preload 扫描定位与单轮索引优化

- 对实际 AOT Player 的 content-2e3c47527f7f483615ef6c94 缓存运行 cProfile，仅 _refresh_declaration_catalog 就32.066秒；93次逐文件 _package_file_record 消耗16.653秒，19459次 resolved_path 消耗31.096秒累计。
- 改为每轮刷新构造一个局部 package_files 索引，无跨轮持久缓存。单文件刷新也走同一个索引构造函数。空 preload_declarations 不计算模块名。相同缓存复测13.388秒，约减少58%；不是完整Player冷启动测量，尚未重建正式Windows运行时。
- 新增test_preload_compiled_index.py覆盖一轮一次索引、空声明不求模块名及下一轮文件更新。首次整组回归发现 _load_path 遗漏旧函数调用（37失败），已同步改为新索引入口。重跑session65876：145 passed、5 skipped，44.88秒。

## 2026-09-09：独立 Player GPU AOT 果冻与真实键盘交互首次通过

- PluginManager 更新最新插件后重启桌面编辑器，exec84715、MCP9741。正常导出 AOT 目录（job a3a0630865dd4ba6b793ff64c27c3ff1，21.858秒，无diagnostics），启动job5d2c925ab7be43eb98683c6a5de173aa返回ready。
- observe确认 GPUJelly awake/started、broken_script=false、simulated_seconds=6.16；高度1.5529、宽度2.2970，已不是静态未执行的模型。日志未发现ERROR。MCP空格按压0.1608秒：模拟时间22.72→22.90，高度1.55618→1.63111，宽度2.67399→2.60691；真实输入已到达计算脚本。
- 实际Game渲染截图：桌面项目 .infernux/mcp_sessions/20260909-063051-9344eaf8/review/jelly-aot-running.png，1280×720。可见变形透明绿色体，但单张截图与一次按键不是完整漂亮/稳定性验收。冷启动、持续交互、重置、Gizmo与更完整性能仍待完成。
- 单次frame遥测game_only_frame_ms18.2753，不能外推稳定FPS。runtime_mesh缓存采样269项/107491792字节，后583项/233430912字节，再降到156项/62169752字节；存在回收，不能称无限泄漏。VkCoreDraw.cpp PublishSharedMeshBuffers只立即淘汰有assetGuid的旧generation，运行时内容哈希缓存需复核动态网格保留策略。
- 已通过MCP正常shutdown此Player。编辑器仍暂停，项目保留。041全计划及漂亮演示交付未完成，目标保持active。

## 2026-09-09：真实 Player 逐层打通 pyc、Windows DLL 与 AOT 制品根目录

- 上轮有实际代码/测试/包变更，属于 progress。更新桌面插件后导出 Cooked（job 6a4775fc90e743d0a9bd61ef4942a1d0，22.105秒）；pyc 加载通过，原生 DLL 则受重定向后长路径影响。限定 Windows vendor 路径使用扩展长度形式，源码/pyc 回归包含路径断言；独立 conda 进程直接导入真实导出缓存 taichi_python.cp313-win_amd64.pyd 成功。
- 再经 PluginManager 更新、正常重启编辑器（exec17632，9741），导出 LongPath（job811424f3d8f1489dbde78d77a094e04e，22.524秒）；启动job46ee559c5e0c4dfbaee63135a23bb66d。日志无 preload 导入错误，但 bind_aot 错把 Application.data_path 的物化缓存当作制品根，找不到 Content.inxpkg。启动仍约30.5秒，preload28.9秒，性能问题未解决。
- bind_aot 改用 Player 启动契约已有 _INFERNUX_PLAYER_DATA_ROOT，不回退搜索缓存。真实 AOT 测试入口显式提供同一契约；新增缓存根与制品根不同的单测，18 passed / 1.42秒。新 .inxpkg 再次生成，尚未安装到桌面/重新导出。当前 LongPath Player 不算 GPU 演示通过。

## 2026-09-09：独立果冻 Player 暴露并修正编译包加载边界

- MCP 启动桌面 Infernux041JellyPlayer-GPU，job 139d31c2a5d943e79d9b9aba2678fe03 返回 ready，Player PID 42260。实际截取 Game render target：项目 .infernux/mcp_sessions/20260909-061305-725f0ec0/review/jelly-gpu-first.png。画面是静态果冻，不能算 GPU 模拟通过。
- 新日志已无 Engine.runtime_mode 错误，但 preload 硬编码 vendor/taichi/__init__.py；项目导出已将文件 cook 为 pyc，因此 compiler 未加载。改用限定插件 vendor 目录的标准 PathFinder，源码/无源码包共用 Python 加载规则，未新增全局搜索路径、失败重试或 CPU 兜底。
- 新增 source/sourceless 两组真实包导入测试，包括相对导入及重复加载；preload 回归 17 passed / 1.52 秒。infernux_package 已重新生成 .inxpkg。此轮新修复尚未更新桌面安装和再次导出，当前 Player 仍是失败样本，不能交付为可用演示。
- preload 仍耗时约 34.7 秒，尚需分段定位。A15.1 已包含五类持续演示与真实交互/性能/保存重载验收，本次未把窗口 ready 或单帧截图标记成视觉通过。

## 2026-09-09：Taichi preload 不再以 runtime 标志判断宿主类型

- lifecycle沿用宿主get_native_engine接口解包Python Engine，直接原生宿主原样传递；runtime只是运行阶段，不再当作宿主类型标志。不增加失败重试或GPU到CPU兜底。
- preload回归参数化runtime真假、wrapper/direct四种组合并断言GpuBackend收到同一原生对象；15passed/0.61秒。
- infernux_package成功生成新Taichi .inxpkg（含原NOTICE流程）。确认桌面编辑器dirty=false后正常关闭35944，PluginManager更新插件成功；新editor exec59628运行、MCP9741。随后启动果冻，继续真实构建验证；此前独立Player不是修复后的插件，不作为通过证据。

## 2026-09-09：新平台插件完整导出通过，真实 preload 宿主接入待修

- 正常重启编辑器加载审计修改（exec26263，MCP9741）；新构建job9896871898d942c9a191625536f2a5d2通过，20.2637秒，无diagnostics，输出桌面Infernux041JellyPlayer-Updated；没有手动替换产物。
- 启动job4df95dd4c2704e2db30dd7cdb78a39c8成功就绪，observe确认gameplay_ready=true、camera_available=true、camera_count=1、Game1280x720。启动日志的camera=False不是最终相机状态。
- GPU场景仍未通过：TaichiPreload报Engine没有runtime_mode。lifecycle.py按context.runtime选择直接使用context.engine，但实际Player传入的是Python Engine wrapper；Editor路径调用get_native_engine反而正确。需统一PreloadContext宿主访问，兼顾已有原生headless测试宿主，不通过加长等待或静默CPU回退掩盖。插件preload阶段仍约32秒，需后续定位。

## 2026-09-09：新 Windows 插件发布完成，统一 Gizmos 审计边界

- exec92064已成功exit0，CMake直接发布子仓库payload与dist/infernux.platform-windows.inxpkg（85,655,936字节）。包内Runtime.inxrt实际包含analyze_buffer_aliases及gizmos公共API，不含collector；mt提取插件PlayerHost确认longPathAware=true。经PluginManager更新到桌面项目成功。
- 源码编辑器exec58175运行、MCP9741；果冻Play12.72秒后暂停。导出job c87477aecd744db3bf7e1dca9aac9247已失败：审计尚禁止整个gizmos前缀。现审计改为只禁止collector，并加打包/审计同边界测试。需重启editor加载修复再导出；不再需要因这项纯构建审计修改重编译插件。
- 新目标输出Infernux041JellyPlayer-Updated尚未成功。旧Player40588仍是之前的诊断副本，不作为新插件验证依据。定向及GameBuilder测试exec1319随后记录。

## 2026-09-09：Release 原生编译完成，修复本机构建环境乱码

- exec70715终止exit1：Release/LTO原生编译、静态契约生成与安装成功，但Python发布进程在PreferencesStore创建用户目录时失败，路径出现乱码。
- 前后对照证实Enter-VsDevShell使USERPROFILE/LOCALAPPDATA/APPDATA的中文变为U+FFFD；不是引擎路径算法。新命令在激活conda后保存这三项，进入VsDevShell后恢复原值，再使用既有native-temp。
- 重试正式windows-msvc-player构建exec92064仍运行，已通过全部原生stage，输出“Prebuilding LTO Release Player Runtime Pack and optional parallel build cache”。继续跟踪92064，不重启；尚未报告新inxpkg成功。

## 2026-09-09：正式 Windows Release 插件构建进行中

- 配置session36996已完成exit0（configure59.7秒、generate1.7秒）。启动cmake --build --preset windows-msvc-player，exec70715保持运行；最近输出到引擎资源、物理组件编译，未报失败，继续跟踪该句柄，不重复构建。
- 核对缓存：_player_compile_input_fingerprint包含引擎Python源码，当前jit_hir变化会改变发布编译输入；不是简单按0.4.0版本号复用旧包。正常游戏导出仍只消费平台预编译包，不在用户构建时编译引擎。
- 补独立Python进程Gizmos导入测试，确认Gizmos API可导入且sys.modules没有collector；本文件2passed/3.62秒。完整新平台插件、桌面项目更新与Player运行仍待此次构建完成。

## 2026-09-09：区分开发链接配置与正式 Player 配置

- 检查 stage_player_native_contract.cmake：STATIC_RUNTIME=OFF 明确删除契约，因而上轮缺文件不是生成依赖竞态；InfernuxInstall 却无条件安装它。现仅在 INFERNUX_RUNTIME_STATIC 为真时安装，保持契约语义，不伪造 static 标记。
- dev/engine-build 为动态开发配置，不能作为正式 Player 发布输入。启动已有 windows-msvc-release preset，静态运行库、Release、PlayerHost开启，使用 conda infernux、VsDevShell及既有native-temp。配置exec36996仍运行，最后确认正在Zstd编译器特性检测，不重复启动；配置完成后使用windows-msvc-player构建preset。
- diff --check通过；完整Release插件重建尚未开始，实际Player40588旧副本仍未验收，测试editor已正常关闭。下一轮继续跟踪36996。

## 2026-09-09：修复 Player Gizmos 导入边界并重建平台运行库

- 读取存活 Player 40588 的 debug 日志，确认 Taichi preload 因制品 jit_hir 缺 analyze_buffer_aliases 失败；GPUJelly 因顶层 import Infernux.gizmos 失败。并非单纯启动慢。源码有该计算 API，需重新发布平台运行库。
- Player 保留 gizmos 公共绘制 API，只排除 gizmos.collector；包入口延迟导入显式请求的收集器，避免普通游戏脚本导入 Gizmos 就携入 Editor。相关回归23 passed / 2.76s，diff --check通过。
- 正式 prebuild_player_runtime 首次因编辑器占用 InfernuxRuntime.dll 而 LNK1168。确认停止Play、dirty=false后正常关闭 editor10304，WaitForExit成功；重试exec86284已终止exit1：原生链接通过，但stage安装找不到python-sync/PlayerNativeContract.json。下一步检查生成contract与stage依赖顺序。实际 Player40588使用桌面副本，未强制终止。

## 2026-09-09：长路径 EXE 编译与实际启动前进

- dev/engine-build 启用 INFERNUX_BUILD_PLAYER_HOST；使用 VsDevShell 和既有 dev/native-temp 后编译成功。manifest 改为 CMake target_sources 输入，避免 /MANIFEST:EMBED 与 CMake vs_link_exe 冲突。mt.exe 从实际 EXE 提取确认 longPathAware=true、asInvoker。
- 将新启动器定点部署到桌面诊断 Player，并重新编译还原 player_package_native.pyc，移除临时异常注释；这是诊断制品部署，正式平台 .inxpkg 尚需重建。
- 启动 job 0eee4cac331545abbb93c48e637aaee8 已越过长路径失败，PID 40588 存活并创建 Vulkan 窗口。真实重定向缓存日志显示 content_ready=0.470s、native init=1481ms、preload project plugins=28366.8ms、bootstrap ready=30107.5ms；场景21对象但 camera=False。需继续检查相机、图像、GPU模拟及异常长的插件初始化，不能宣称场景已验收。

## 2026-09-09：定位应用容器长路径与 Player manifest 缺失

- 临时对实际 Player 的解包 Python 字节码添加异常注释后，独立启动 job 0191f8e9e4da483abb06e994cff7d600 揭示 native_bytes：create_directories: The filename or extension is too long。实际 LOCALAPPDATA 被重定向到 C:/Users/陈立哲/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local，较普通 shell 环境长，Taichi 深层目录超过限制。UnicodeDecodeError 是中文路径出现在本地编码 C++ 异常消息的次生错误。
- 源码临时异常注释已撤回，不留下运行期诊断层；桌面诊断制品的 pyc 仍有注释，需通过正式重构建替换，不能发布该诊断制品。
- PlayerHost 增加 longPathAware manifest，并由 MSVC linker 嵌入。dev/engine-build 当前未启用 INFERNUX_BUILD_PLAYER_HOST，尝试构建该 target 返回 unknown target；下一步需启用正确 Player 构建配置，检查链接后的 manifest 并重新验收长路径独立启动。此处尚未证实修复生效。

## 2026-09-09：缩小独立 Player 解包异常范围

- dev/041_pack_path_probe.py 分别直接载入实际制品的 _InfernuxBootstrap.pyd 与 Runtime/Infernux/lib/_Infernux.pyd（DLL 搜索目录也来自制品），读取同一 Content.inxpkg 的 142 条记录并解包。
- 两模块在 ASCII dev 路径、中文 dev 路径、真实中文用户 AppData/Local/Infernux/Players/Infernux041Lab/Cache 下均成功。最长输出路径 215 字符，不能据现有证据归因为中文路径普遍不支持或 MAX_PATH 超限。
- 这些实验使用 conda Python 宿主，实际失败发生在独立 EXE 宿主，下一步需检查独立进程路径/环境与原生异常原始字节。诊断目录为脚本创建的独立临时目录，未替换游戏缓存、未对引擎加 ASCII 回退。实际 Player 仍未通过启动验收。

## 2026-09-09：完整 Windows Player 构建成功、启动暴露路径错误

- 正常停止 Play、确认场景 dirty=false，关闭 PID 30652 后重启源代码编辑器（exec 46572，MCP 9741）。运行果冻约 12.8 秒并暂停，重新准备计算。
- MCP 构建 job 29dbeccec94c45a1b2139703dee3096b 成功，耗时 18.0489 秒，diagnostics 为空。实际 EXE：D:/Users/Chenlizhe/Desktop/Infernux041JellyPlayer/Infernux041Lab.exe；正式封包审计通过。
- 独立启动 job 2b68fde73eae485883f8f1111e0b7bc3 已失败退出（4294967295），未 ready。player.stdout.log 指向 platform_player_bootstrap._content_cache → player_package_native.extract_pack 的 UnicodeDecodeError（0xb3，position70）。缓存路径位于中文用户 AppData，尚不能断言具体根因；需对照分发的 _InfernuxBootstrap.pyd 与当前源码/原生构建并复现。
- 日志在桌面项目 .infernux/mcp_sessions/20260909-054149-99fd990d/player.stdout.log。制品已生成但不可验收，不将构建成功等同于 Player 运行成功。

## 2026-09-09：首次实际果冻完整构建抵达封包审计

- 桌面 BuildSettings 启动场景设为 Assets/Scenes/01_XPBD_Jelly.scene，enable_jit=true（项目仍含 CPU HPC）。MCP job b1fe76db77304024a58aabd85b639416 已终止失败；准备、Cook 和封包执行后，审计报告三内核的 __content__/__version__ 以及 numba/packaging 的 py.typed 为重复内容。没有最终输出目录，不能报告制品已完成。
- 将包内格式/类型标记从重复游戏载荷判定排除；仅 Compute 范围内的两类模块标记与 py.typed，真实 SPIR-V 或混合普通资产仍拒绝。未关闭整体审计，未新增哈希或退回 JIT。
- 定向及 GameBuilder 回归 275 passed、1 skipped / 15.27 秒。当前 editor session 99002 仍加载修复前审计模块；后续需要载入修复再执行新构建，不能重用旧 job 作为通过证据。

## 2026-09-09：桌面实验室更新插件并恢复果冻运行

- MCP 确认旧编辑器 edit、dirty=false 后通过 CloseMainWindow 正常关闭 PID 60076，WaitForExit 成功；没有强制终止。经 PluginManager.install_package 更新当前 Taichi .inxpkg 并安装本地 Windows 平台 .inxpkg，两次安装均返回对应 reference、无 preload failures。
- 用 conda infernux 源码启动编辑器，exec session 99002 保持运行、MCP 9741。targets 确认 windows-x64 available=true，当前 host target 正确；其它平台未安装，仍 unavailable。
- 通过 MCP GUID 打开 01_XPBD_Jelly，进入 Play 成功（进入耗时约 54.23ms）；运行约 11.34 秒后通过 MCP 暂停，保留当前进程及内核特化用于随后构建验证。没有修改作者脚本。此处只证明生命周期操作，尚未完成视觉/性能或完整 Player 导出验收。

## 2026-09-09：MCP 构建入口携带计算产物

- 实际 MCP targets 确认桌面项目没有安装任何平台 exporter；只有 mcp 与 taichi 目录，平台目标均 unavailable。项目内 Taichi backend 尚无 prepare_build，需在完整导出验证前更新安装，未直接覆盖运行中的 native 插件。
- 检查发现 EditorAutomationHost.build_player 不经过面板，此前没有 GPU 准备。现使用 owner queue 执行模块收集/prepare_build，用本次 TemporaryDirectory 持有产物直到 BuildService.execute 结束，并通过同一个 BuildRequest 字段传递。
- 增加真实后台线程 → owner queue 准备回归；compute preparation、host player build、catalog preflight 合计 16 passed / 2.83 秒。MCP 实际完整 Player 构建仍未完成，下一步需安装当前插件及重启测试 editor 加载改动。

## 2026-09-09：构建准备线程与生命周期回归

- 新增实际后台线程驱动的面板回归，覆盖成功、取消、执行失败、准备失败及线程启动失败。确认 prepare 在调用线程、worker 在不同线程，读取时计算文件仍存在，结束后临时目录移除、面板不再处于 building。此文件 7 passed / 3.73 秒；后端使用测试替身，不代替真实 GPU 验收。
- 检查实际 preflight：完成回调从 post_present_tick 进入，而非后台目录扫描线程。
- 实机 MCP 9741 可用，PID 60076 确认为 conda infernux 的 dev/041_launch_lab.py。project.info 返回桌面 Infernux041Lab、06_SharedMeshPublication、dirty=false、edit 状态，未关闭或重启该编辑器。项目 BuildSettings 仍为 Start 场景、enable_jit=false，尚未配置成 GPU 果冻完整导出验收。

## 2026-09-09：构建面板连接 GPU 准备入口

- 新增 loaded_script_snapshot：沿用项目 canonical module name 和本次资产目录选择已加载模块；不导入作者脚本，不扫描整个 sys.modules。模块来自另一项目时直接拒绝，避免把别的项目的 GPU 特化封进本次游戏。
- BuildSettingsPanel 在 preflight 完成、启动 worker 前调用 provider.prepare_build，并传入 BuildRequest。专用 TemporaryDirectory 由 worker 闭包持有到结束，准备失败或线程启动失败也清理；没有插件时保持 CPU 构建路径。
- 定向测试 40 passed / 3.50 秒，包含模块快照、构建设置、preflight 和请求契约。尚未通过实际编辑器点击构建验收线程归属、目录生命周期与最终 Player；不得把这组测试解释为完整 GUI 验收。冷构建特化覆盖、跨平台 GPU ABI 和演示场景视觉仍未完成。

## 2026-09-09：构建请求传递计算产物快照

- BuildRequest 增加 compute_artifacts，递归冻结签名与文件引用；拒绝模块、数组和 GPU 对象进入请求。只冻结记录，不宣称冻结磁盘文件内容；准备目录仍需由构建前端持有到 worker 结束。
- 桌面导出与公共平台内容 Cook 均在资产目录冻结后将这些记录登记到 GameBuilder，沿用现有 Content.inxpkg 流程，不新增摘要校验、补救编译或第二套包格式。
- 测试覆盖嵌套记录不可修改、来源对象修改不影响请求、两条构建入口确实传递记录。相关回归 300 passed、1 skipped / 21.26 秒；定向 diff --check 通过。
- 下一步仍需连接构建面板的权威模块快照、owner-thread 准备与临时产物生命周期；不能将本次传递层完成宣称为 GUI 一键 GPU 构建已经验收。A15.1 的果冻、雪地、轮廓、世界 UI 和物理演示继续按功能阶段交付，视觉与性能验收独立于这些单测。

## 2026-09-09：按当前模块快照准备 GPU 构建输入

- 检查 BuildSettingsPanel：资产目录完成回调在启动 build worker 前创建 BuildRequest，而具体 GameBuilder 由 exporter 在 worker 使用。GPU 导出必须在 owner thread 完成，不能直接塞进 worker 的 copy/stage 阶段。
- Provider 新增 prepare_build(script_modules, destination)，接收明确的 GUID/module 快照；发现模块自有 HPC 函数和词法嵌套类的 staticmethod，排除导入函数与别名重复，不扫描整个 sys.modules。返回只有 GUID、qualified name 和已导出文件产物的记录，交给构建器登记。期间不调用作者内核。
- 回归覆盖静态类方法、类自引用不递归、函数/类别名、外部导入函数不被误收集；compute/lifecycle 共 45 passed / 4.07 秒。
- 桌面果冻诊断已使用这一入口，不再自己枚举/逐个导出函数。dev/041-jelly-build-prepare-20260909 导出真实三内核、正常完成 60 步 JIT 参考；独立 AOT 加载验证随后记录。
- 此为正式准备流程可调用的 provider 入口，BuildSettingsPanel 的模块快照来源和异步请求生命周期尚未接入，不能宣称用户点击 Build 已自动准备。当前特化来自已有准备/调用，完整类型覆盖与冷构建策略仍未验收。
- 独立 AOT 进程通过：343 粒子、1296 四面体、60 固定步，位置最大差 0.0，所有体积为正（min 9.13096699e-7）；AOT 计算/传输 median 15.5328 ms、P95 16.6364 ms，不含渲染。未修改桌面作者项目、未远程发布。

## 2026-09-09：多特化 AOT 的 Cook 与 Player 选择

- 构建登记改为 register_gpu_aot_artifacts，一次接收同函数全部已准备特化；绑定统一为 variants 列表（含签名、graph 和包内 files），不保留此前未发布的单图运行分支。相同导出源文件在本次登记中复用同一包内 entry，不以内容哈希去重。
- Player bind_aot 载入一次共享模块，为各特化构造 NumPy runner；每次按已有类型/rank/别名签名选择，不重新查询源码或编译。未准备签名报 specialization unavailable；成功调用数按整个函数累计，插件卸载检查仍在入口。
- 正式真实 GPU 测试扩展到同一个 Cook 后的插件函数在 batch 中交替接受 float32/int32；float64 未准备时拒绝且不触发 JIT。构建/compute/lifecycle 合计 317 passed、1 skipped / 18.15 秒；真实包测试进行中。
- dev/041_jelly_aot.py 去掉“必须只有一个特化”的探针断言，登记实际导出列表。dev/041-jelly-aot-variants-20260909 独立 JIT/AOT 各跑 60 固定步，343 粒子、1296 四面体，位置差 0.0，最小体积 9.13096699e-7。此次 AOT median/P95 15.5951/17.6878 ms（同时有另一 GPU 回归进程，不作为独占性能基线）。
- 尚未接正式构建准备钩子，也未证明有限观察/有界缓存包含所有运行分支的类型；完整 041、Player 场景视觉与跨平台交付仍继续。
- 最终真实 GPU 包回归 6 passed / 111.11 秒，无跳过；多特化封包字节码调用、批次选择、缺失签名拒绝、样本释放和卸载全部通过。

## 2026-09-09：从已准备特化导出，不重复执行样本

- GPU 特化有界缓存现在同时持有内核及其参数类型/rank 信息；export_prepared_aot 遍历现有特化，输出各图及签名，不调用作者函数，不需要保存 NumPy 样本。现有显式样本导出复用同一图生成逻辑。
- 首次真实回归的弱引用断言暴露 clone_call_arguments 的递归闭包暂存原数组直到 cyclic GC。现于复制结束清空 memo/array_roots；新增禁用 cyclic GC 的单测确认原数组和返回副本都能立即释放，不增加周期性 GC。计算/生命周期/JIT runtime 合计 57 passed / 3.68 秒。
- 正式包测试补 float32/int32 两种已准备特化，导出前后调用次数不变，逐图以 NumPy 执行；首次 5 passed、1 failed / 67.32 秒为上述引用问题，修正后重跑中。
- 桌面果冻诊断改为正常 step 一次后发现模块内的 HPC dispatchers，直接导出已有特化，不再手写 predict/project_color/finish 的样本参数列表。dev/041-jelly-aot-collected-20260909 的独立 AOT 进程 343 粒子/1296 四面体/60 固定步，最大位置差 0.0；AOT median/P95 13.6983/15.5454 ms，最小体积 9.13096699e-7。
- 边界：当前发现入口仍在诊断脚本，不是构建面板的正式准备钩子。真实果冻每函数只有一个特化，探针对此明确断言；普通多特化 Player 自动选择、缓存已淘汰特化的完整性、构建期准备范围仍未完成，不能依赖有限观察宣称穷尽所有游戏分支。
- 修正后正式包真实 GPU 回归 6 passed / 105.68 秒，无跳过；两种准备类型均可导出/加载、导出不增加调用次数、弱引用样本释放及原有封包/卸载验证全部通过。

## 2026-09-09：AOT 构建产物登记与暂存

- GPU provider.export_aot_graph 返回 graph 与导出 sources；GameBuilder.register_gpu_aot_artifact 将其按脚本 GUID/qualified name 登记为包内引用，作者机器的 source 路径只留在构建侧。文件名和身份在登记边界校验，避免逃出暂存根；不新增文件摘要或运行期重试。
- 两条正式构建路径（平台内容 Cook/桌面完整构建）在 copy_game_data 后调用 _stage_gpu_aot_artifacts，沿用 Data/Library → Content.inxpkg 的选择与打包流程。特化样本的发现/准备仍需后续接入，不把产物登记当成自动特化收集完成。
- 构建/compute 回归 306 passed、1 skipped / 15.14 秒；diff --check 无错误。测试覆盖包内引用不带本机路径、暂存不修改源产物、重复登记和逃逸文件名拒绝。
- 真实果冻诊断脚本改用返回产物登记与构建器暂存，不再手工构造计算文件引用。dev/041-jelly-aot-staged-20260909/Content.inxpkg 为 26,624 字节；独立 JIT/AOT 进程均跑 343 粒子/1296 四面体/60 固定步，位置差 0.0，最小体积仍为 9.13096699e-7。JIT median/P95 11.9337/12.9786 ms，AOT 13.6278/15.4097 ms（仅计算与传输，不含渲染）。
- 样本准备、实际项目 GUI 构建、完整 Player/场景视觉、跨平台和远程发布仍未完成，未更新桌面原项目插件。

## 2026-09-09：真实 XPBD 果冻的 AOT 封包数值验证

- dev/041_jelly_aot.py 读取桌面 Infernux041Lab 的原 JellyMaterial.py 及显式 meta GUID（57c9d83936683b0dcae2f6f762899f8d）；作者文件不修改。对 predict/project_color/finish 的真实 NumPy 样本导出 AOT，确认导出 warmup 不改变样本，走 GameBuilder Assets 脚本 Cook，将三组 AOT 文件与 pyc 写入 Content.inxpkg。
- 第一轮在第二个 Module 创建时失败：上游 _finalize_root_fb_for_aot 将已 finalize 布局当成错误。调整为对未完成布局 finalize、后续模块复用不可变布局，不重启设备。正式 live_aot_graph 回归补同进程再次导出并比对 metadata.json。
- 修正后 dev/041-jelly-aot-20260909-b 产出封包；export/load 为两个独立进程，加载进程禁止源码查询，从封包 pyc 绑定原 hpc。相同 343 粒子、1296 四面体、60 个 dt=0.02 固定步（每步 8 substeps），位置最大差 0.0，所有值有限且最小四面体体积 9.13096699e-7 > 0。JIT 稳态计算步 median 12.0496 ms / P95 12.8768 ms；AOT median 14.3899 ms / P95 15.4689 ms，含 NumPy 批次上传/回传，不含场景渲染，不等于 Player FPS。
- 本次证据确认真实果冻算法可使用现有 AOT 封包路径，不是完整透明场景/交互/构建 GUI 自动化验收。特化样本和文件清单在诊断脚本里显式准备；正式自动构建收集、长期稳定性、视觉和跨平台继续。桌面作者插件副本、远程制品未更新。
- 最终插件包正式回归 6 passed / 101.88 秒，无跳过；同进程重复 AOT 导出、Assets/Packages sourceless GPU 调用及卸载仍通过。实际果冻 Content.inxpkg 为 26,176 字节，只含该算法和计算产物，不是完整场景包。

## 2026-09-09：Packages 与 Assets 共用计算 Cook

- 将 GPU AOT 绑定和 CPU 并行源码转换抽为 GameBuilder._cook_compute_source，Assets 与 Packages 编译均调用它。插件取导出 InxPlugins.json 文件记录的 GUID，不新建路径派生身份；已有 preload 声明/compiled_path_hint 更新保留。
- 参数化回归同时覆盖 Assets 和 Packages：缺失 GPU 特化不产生 pyc；已准备的 GUID 绑定进入 sourceless 字节码，装饰器调用 provider.bind_aot，不执行作者函数体或进入 JIT。GameBuilder 与 compute 合计 301 passed、1 skipped / 13.76 秒；diff --check 无错误。
- 真实 GPU 正式测试新增通过 GameBuilder._compile_player_plugin_scripts Cook 插件算法、以 package_scale.pyc 装入 Content.inxpkg、在禁止源码查询的独立进程执行其 hpc 调用。验证进行中。
- 最终真实 GPU 包测试 6 passed / 106.36 秒，无跳过。插件 Cook 后的字节码实际驱动 GPU，NumPy 结果与调用计数正确；原有 AOT/NumPy/batch/卸载回归仍通过。主仓及 external/plugins 内无依赖原静态方法调用方式的残留。
- 此步没有自动收集特化或自动 stage AOT 文件；当前 GPU 图仍是已显式准备的单特化。全目标 A05/041、实际果冻 Player、平台 ABI 与远程交付仍未完成。

## 2026-09-09：Cook 字节码的 HPC/AOT 自动绑定

- 引擎 build_gpu_aot_embedded_source 按 Python qualified name 将构建侧引用嵌入 GPU hpc 装饰器的私有 `_aot` 参数；区分同名类方法，缺失/多余绑定或作者手填私有字段在构建阶段报错。不嵌 GPU 源码，不把 AOT 缺失解释为 Player 隐式 JIT。
- GameBuilder 增加构建输入 gpu_aot_bindings（脚本资产 GUID → qualified name → provider 引用），在原有 Assets 脚本 pyc/Cook 过程中应用，CPU 并行转换仍沿用原路径。当前尚需插件构建阶段自动准备此清单和导出文件；Packages 脚本的统一接入也未完成。
- GPU provider.bind_aot 从活动数据根的 Content.inxpkg 读取既有 AOT 文件，在装饰器执行时载入并绑定原函数签名。运行调用无源码查询，位置参数、关键字和默认值由同一 Python signature 绑定后进入 NumPy 图。测试从实际封包读取 pyc 并执行，而非另写手工 AOT 用户调用代码。
- Compute/lifecycle 合计 44 passed / 3.51 秒；GameBuilder 全文件 266 passed、1 skipped / 14.11 秒，含当前资产 GUID 绑定后的 sourceless 脚本执行与缺失特化拒绝。
- 首轮新 GPU 回归 5 passed、1 failed / 70.85 秒，发现 Taichi 不接受参数默认值；HPC 调度已 apply_defaults，因此特化函数不再携带 __defaults__，仍向 backend 传完整位置参数。新包已生成，真实 GPU 回归重跑中。
- 最终真实 GPU 新包回归 6 passed / 102.69 秒，无跳过；封包 pyc 中的原 hpc 装饰器成功绑定 AOT，位置/默认/关键字调用数值正确，仍禁止源码查询。正常卸载及 cleanup 通过。未宣称完整果冻 Player 或跨平台验收；桌面插件及远程尚未更新。

## 2026-09-09：统一 HPC 的 AOT 导出与 NumPy 适配

- GPU provider 新增内部 `load_aot_graph`；原生 LoadedAotGraph 暴露自身参数元数据，Python 不另读/重建 graphs.json schema。NumPy 数组与标量按此元数据准入，复用现有 batch 的驻留数组、上传/回传；AOT 提交前排出已有 JIT 指令以保持调用顺序。记录请求/实际设备、模式及成功调用次数，卸载后拒绝调用。
- 新增内部 `export_aot_graph`，由已有 hpc 特化与隔离 warmup 导出原生图。正式回归的算法改为单份 `compute.hpc(device="gpu")`，导出前后作者的样本数组不变，不再另写 ti.kernel。当前只导出无标量返回的图，标量返回 kernel 的 AOT 结果通道仍待实现，不静默丢返回值。
- 适配层版本的真实包测试 6 passed / 103.80 秒；引擎 compute 与插件 lifecycle 合计 39 passed / 4.20 秒，含 JIT 指令先于 AOT 图提交的合同测试。进一步的 HPC 导出新版包正在独立完整回归，结果随后补记。
- HPC 导出首次回归为 5 passed / 1 failed（77.88 秒）：导出成功，但 cleanup 检出 compute host 引用尚未释放。定位特化函数对整个作者 globals 的拷贝留住了与内核无关的原生 Program；改为仅复制源码引用的全局名称与 Python 模块基础信息，不补 GC/retry 或放宽 native 生命周期检查。修改后 compute/lifecycle 39 passed / 3.59 秒，真实 GPU 包正在重跑。
- 最终新包完整真实 GPU 回归 6 passed / 104.28 秒，无跳过。单份 hpc 算法导出、中文目录 AOT 文件加载、移除测试临时导出目录后的 Content.inxpkg 加载、NumPy 单次/批次计算、错误参数准入、旧图及旧 NumPy callable 卸载拒绝全部通过；导出进程无需 gc.collect 即正常 cleanup。正式 Player 自动 Cook/绑定仍未完成。
- 边界：这是构建工具/Player 调度将调用的内部适配，不是新增作者 API。完整 kernel 身份/特化清单与 Cook/装饰器自动绑定未接入；不能据此声称果冻 Player、标量返回、多参数别名或跨平台 AOT 已验收。未更新桌面插件副本和远程制品。

## 2026-09-09：AOT 封包字节接入既有 VirtualDir

- Program/GfxProgramImpl 的同一 AOT 加载入口接受 VirtualDir；新增内部 `_load_aot_files` 将字符串文件名与 bytes 映射适配到既有读取器。加载同步消费字节，不持有 Python 对象，不解包临时目录，不另建 Vulkan 设备。文件系统入口保留供开发导出检查，不作为封包失败时的回退。
- 原生 44 项增量构建与 `.inxpkg` 打包成功。正式真实 GPU 回归扩展为导出、文件系统加载、Content.inxpkg 加载三个独立进程；封包测试移除自身临时导出目录、释放输入字节映射，再检查两个 NumPy 数组规模的结果和卸载后拒绝旧图。
- 新包 `test_package_payload.py` 全部 6 项通过，99.98 秒，无跳过；新增封包进程验证通过，无 Vulkan 验证错误。测试移除的仅为 pytest 新建临时 AOT 导出目录，产物保留在同级 Content.inxpkg，可重新导出。
- 这里的文件名清单仅为测试夹具；产品仍需接入 hpc 构建变体清单、公共 NumPy 调度与 Cook 资产身份，不能将此底层测试记作完整果冻 Player 或 A05 验收。桌面插件副本及远程制品未更新。

## 2026-09-09：AOT 双进程正式回归与 Unicode 修复

- 插件 tests/infernux 新增 live_aot_graph.py 及 test_aot_graph_loads_without_source_in_a_new_process：分别解包官方开发 .inxpkg，在独立进程导出/加载。加载进程禁止 inspect.getsource/getsourcelines（若尝试 JIT 查源码立即失败），数组长度 13 和 257 计算逐项一致，删除 module 后 graph 保活，Preload unload 后拒绝旧图。导出与载入均检查进程退出/Vulkan 验证错误。
- 正式测试采用中文目录后暴露原生 std::fstream 窄路径问题；首次日志还被 GBK 解码遮蔽，测试子进程与接收端显式 UTF-8 后确认 metadata.json 无法读取。AOT builder 和 FilesystemVirtualDir 改用 std::filesystem::u8path；builder 启用流错误异常并显式 close，避免文件没写成仍假成功。不增加 ASCII 临时路径重试。
- 原生重编译/重打包成功，新包六项全通过（86.66 秒），包括新增中文目录的 AOT 新进程测试。仍不是 Content.inxpkg VirtualDir 或用户 hpc 构建变体的最终验收；原桌面副本未更新、远程未发布，完整 041 继续。

## 2026-09-09：AOT 加载与新进程 GPU 执行成立

- Program 新增专用 load_aot_module，接 GfxProgramImpl 的既有 GfxRuntime 和 gfx::make_aot_module，不创建第二个设备。LoadedAotModule 拥有 native Module，Graph 持有所属 Module；Program finalize 同步后使所有仍存活的 Module 失效，旧图在提交前检查已结束/线程归属。弱引用列表不强留已无人使用的模块。
- 实测先暴露 get_program_impl 为 LLVM 专用历史接口，改为 Program 专用方法而未放宽它。随后纯 ndarray 模块在 get_graph 原生崩溃：loader 硬编码 num_snode_trees=1；按 root_buffer_size 为零时设为零后通过图创建。又发现 loader 未重建 graph.args，补充从 dispatch.symbolic_args 恢复，使已有 Python 参数转换和 AOT context 一致。
- 多轮编译/重新打包后，dev/041_load_gpu_aot.py 在全新进程加载 dev/041-aot-eqhif5ge/aot，脚本无 Python kernel 定义。将长度 13 的 NumPy float32 数组上传，经预编译 scale_graph 乘 4，读回逐项正确。删除 module 变量后 graph 保持可用；Preload unload 后 graph.run({}) 报 finalized，释放探针保留的 Program 引用后引擎 cleanup 成功。输出 AOT_FRESH_PROCESS_NUMPY_AND_UNLOAD_OK，退出码 0。
- 这是底层 AOT 文件系统加载/执行与生命周期验证，不是用户 hpc 构建变体选择或 Content.inxpkg 挂载验收。仍需正式回归固化、封包 VirtualDir、provider 统一 NumPy 包装、构建期变体与果冻 Player。桌面插件副本未同步，远程未提交/发布，完整 041 未完成。
- 新包五项真实 GPU/JIT、Preload、Headless 与包体回归全部通过（48.98 秒）。

## 2026-09-09：AOT Graph Python 执行入口

- export_lang.cpp 将已有 Graph 参数转换提取为局部共享调用函数：jit_run(config,args) 沿原编译路径，新增 run(args) 调用 CompiledGraph::run 的预编译内核路径。没有复制标量/ndarray/矩阵绑定规则，没有 AOT 失败重跑 JIT。矩阵临时缓冲区原为固定 128 字节，现按实际元素字节数分配。
- taichi_python 完整编译/链接成功；infernux_package 重新生成 .inxpkg，包含本轮 Python binding 和上一轮 loader 清理。首次包测试因未指定实时引擎参数得到 2 passed/3 skipped；补齐 INFERNUX_ENGINE_NATIVE_DIR/RESOURCES 后五项全部通过（52.29 秒），覆盖真实 GPU/JIT、Preload、Headless、包检查。
- 新包运行 AOT 导出探针成功（dev/041-aot-9ed7kpkq），上一轮补充的参数维度/标量与 SPIR-V magic 断言均通过。
- 这些结果不证明新 run 已执行加载的 AOT 图：仍需 Module 加载、graph/module/runtime 所有权和无源码独立进程测试。当前桌面已安装的副本尚未更新，不改称最终交付；未提交/推送/发布，041 持续推进。

## 2026-09-09：AOT 加载来源清理与调用合同确认

- gfx AOT loader 原先即使传入 VirtualDir 也构造一个备用 FilesystemVirtualDir（空路径会指向 ./）。改为只有明确使用文件系统输入时才构造，包内 VirtualDir 路径不创建另一来源；删除未调用的 read_spv_file 及其重复路径成员/include。不是增加 fallback。
- 原生 gfx_runtime 目标重编译与静态库链接成功。没有重打插件包，因此运行中的插件暂未包含本轮 C++ 变更；也未声称已经完成包内 AOT 执行。
- 确认 CompiledGraph.run 使用 AOT compiled_kernel + LaunchContextBuilder；现有 Python 暴露的 jit_run 使用 ti_kernel 并重新编译，不能拿它调用加载的 AOT graph。后续应共享参数转换代码但分清两个入口，并保持 Module 拥有 kernel、graph 引用 Module、provider 卸载先结束提交再释放 runtime 的顺序。不能只把裸 GfxRuntime* 暴露给 Python 后任其跨 reset 存活。
- 完整目标仍有加载绑定、封包接入、NumPy 变体和真实无源码执行待实现；无提交/推送/发布。

## 2026-09-09：引擎 RHI 上的现有 AOT 导出验证

- 通过实际 infernux_taichi.inxpkg 解包/Preload，使用引擎 renderer 的 compute host 初始化裁剪编译器。dev/041_probe_gpu_aot.py 运行 NumPy float32 数组乘 3 并校验结果，再使用现有 ti.aot.Module/add_kernel/save 导出；直接 ti.kernel 仅是编译器底层探针，不是用户 API 或最终 inx.compute 构建方案。
- 首次探针未创建 Module.save 所要求的目标目录而失败；补建探针目录后成功。输出 dev/041-aot-t_pkx_3h/aot：metadata.json 1646 字节，scale_c58_1_0_t00.spv 1832 字节，graphs.json/__content__/__version__。实读元数据包含 ndarray 维度 1、float32 标量、访问模式、任务 buffer_binds 与 SPIR-V 能力要求。后来补充了这些结构和 SPIR-V magic 的断言，待下一轮随加载回归重跑。
- 确认既有 gfx::make_aot_module 接受 GfxRuntime*，该 runtime 本身已建立在引擎设备上；优先复用其加载与参数信息，不建立第二套 SPIR-V 反射或 Vulkan 设备。下一步仍需暴露给 provider 的加载/调用入口、封包资产装载和 NumPy 变体选择，并验证全新进程无源码执行。
- 当前只证明 AOT 导出存在且实际可用，不证明 AOT 载入执行或 inx.compute Player 完成。未提交、推送、发布，完整 041 继续。

## 2026-09-09：封包 GPU 内核源码依赖已复现

- 检查当前 GPU provider 的 compile：backend.py 直接 inspect.getsourcelines(fn)，随后 Taichi kernel_impl.py 经 _wrap_inspect 再次取源码；CPU build_auto_parallel_embedded_source 只覆盖 CPU 克隆，没有 GPU Player 输入路径。
- dev/041_probe_cooked_gpu_source.py 从真实 Content.inxpkg 读取 Assets/Scripts/JellyMaterial.pyc，提取 predict/project_color 的实际 code object，按当前 backend 同一 inspect 操作读取。两者都报 source code not available，co_filename=Assets/Scripts/JellyMaterial.py。探针只验证源码查询失败，没有宣称已执行 GPU。不能借作者机器原 .py、临时还原源码或全局 inspect monkeypatch 使 Player 假通过。
- A05 已要求插件构建集成与 AOT 内容载入。下一步应接正式 GPU 内核编译输入/构建产物到 Player provider，并和动态运行时 JIT 的来源生命周期区分；不能用整项目明文副本替代，也不能仅让 Cook 检测报错就算完成。必须覆盖 NumPy 参数变体、GPU 执行和无作者源/全新进程验证。
- 本轮证据推翻“Cook 成功意味着 GPU Player 可运行”的推断，完整 041 仍有此真实缺口；未修改求解器、未发布制品、未标记完成。

## 2026-09-09：桌面 Lab 真实内容 Cook

- 正常关闭可见编辑器后，使用 headless publish_player_asset_catalog_for_host 刷新真实桌面项目索引，通过 GameBuilder.cook_platform_content 执行完整内容处理/脚本编译/封包。没有安装平台插件或伪造可执行文件，验证 host identity 明确为 041-content-validation-only，entry_point=unprovided-player.exe，不能运行，也不作最终制品。
- 首次停在 CPU compute 需要 JIT：Assets 仍有 XPBDJelly.py 的 advance/surface_normals CPU 参考计算。dev/041_cook_lab.py 增加显式 --enable-jit 重跑后完成。未删除参考脚本、未把 GPU 求解改成 CPU、未改动项目 BuildSettings（其中 enable_jit 仍 false）；后续正式构建需通过正常设置入口启用。
- 成功输出 dev/041-lab-content-v0mfgq3_/Infernux041Lab_Data。仅四个文件：Content.inxpkg 4243456 字节、AssetCatalog.inxcat 18496 字节、PackageIndex.inxmanifest 190 字节、Player.inxmanifest 5654 字节。内容包 176 条目，raw_bytes=17174214。
- 实读封包目录确认 SavedSlope GUID f75f22fde3a4c19c9b3f0332c337647e 的 mesh_artifact=620 字节、空骨骼伴随=45 字节，均为 compiled_artifact 且 unresolved_dependencies=[]，源路径作为目录元数据保留，不作为散文件。尚未从 Player 挂载该包并渲染。
- 失败输出 dev/041-lab-content-eb4jxlyn 保留供定位；成功内容包也只作本地验证，无发布。已重新启动可见 Lab 编辑器。完整 041 仍未完成。

## 2026-09-09：实际导入网格接入构建暂存测试

- 扩展原生网格 Cook 集成用例，将真实 AssetIndex 条目交给 GameBuilder.freeze_asset_index_entries 和 _stage_library_runtime_artifacts，验证导入二进制暂存字节、源替换路径集合、GUID binding，并确认暂存目录没有作者源。随后 Content.inxpkg 使用实际暂存产物而不是直接取 Library 文件。
- 首次测试使用不存在的 set_asset_index_entries，修正为现有 freeze 接口；引擎 fixture 缺少 BuildSettings，补上测试设置并在 finally 恢复，不绕过构建器的配置合同。定向用例通过，资产库与构建闭包 43 项通过（16.55 秒）。
- 现有策略实际选择所有已导入 Assets，并非场景可达性裁剪；本用例仅单网格索引/无构建场景，不能证明完整场景封包或 Player 启动。桌面项目 BuildSettings 当前仍指向存在的 Start.scene，后续完整 demo Player 验收需显式选择演示入口并配置相机/必要运行时内容。没有修改桌面构建设置，无提交/推送/发布。

## 2026-09-09：作者网格与 Cook 制品边界

- 检查发现 runtime_artifact_catalog 仅凭 .inxmesh 后缀把 Assets/Packages 的作者源也分类为 mesh_artifact。现将这两种作者目录下的 .inxmesh 标为 model_source，Library/Artifacts/Mesh 仍为 mesh_artifact；不新增格式或备用加载路径。
- 增加真实原生资产库集成用例：OBJ 几何保存为 .inxmesh 作者源→原 ModelImporter 导入→读取实际 AssetIndex→validate_artifact 通过；导入产物嵌入源内容身份而非作者角色标记，字节不等同作者源。使用正式 write_pack/read_entry 将导入 Mesh 放入 Content.inxpkg 并验证往返字节，移除作者源后失效驻留并重新加载，恢复几何与原源序列化一致。
- 用例验证现有原生加载器读 Library 导入产物，不是从挂载包启动完整 Player；没有把局部包读写称为完整游戏制品验收。尚需场景闭包、封包映射与 Player 启动/画面验证。
- GameBuilder、资产闭包、平台 Cook、原生资产库扩大回归：319 passed / 1 skipped，28.35 秒；定向四项先通过。主目标保持完整且活动，无提交/推送/发布。

## 2026-09-09：Inspector 静态网格保存副本入口

- MeshRenderer Inspector 增加中英文“保存网格副本…”按钮，位于原有网格/材质区域，原生批量字段与 Python 字段布局均共用该入口。读取当前资产网格，复用统一 save_file_dialog 和 ProjectAssetCommandService.save_mesh_copy，成功后定位新资产，不自动换引用、不覆盖原资产。
- 仅对有资产且无骨骼载荷的 Mesh 显示；内置/纯内联网格的新建作者能力与骨骼保存仍未交付，不以静态格式静默丢弃骨骼。正常取消无事务，写入/路径错误通过 UI 错误日志呈现，不重试或另造格式。
- Inspector/资产/组件命令 111 项测试通过（18.17 秒）；追加按钮触发与错误反馈测试后 Inspector 67 项通过（3.99 秒）。测试覆盖对话框取消、项目 Assets 默认目录、保存命令参数、定位结果、不可保存来源不提供按钮，以及按钮使用当前几何。真实 OS 对话框点击与视觉布局本轮未验证，不能将 mock 测试称为人工 UI 验收。
- 上轮真实桌面网格引用证据仍有效；完整 041、漂亮 demo、Cook/Player 与各平台交付继续待办。没有提交、推送或发布。

## 2026-09-09：统一网格赋值命令与无脚本资产引用验证

- Mesh 来源选择是聚合编辑，不是把序列化 document.meshAssetGuid 当作 Python 标量字段写入。新增 ComponentCommandService.assign_mesh_asset，使用原生 setter 切换来源，并复用 edit_document 保存完整前后文档与 Undo。Inspector 的模型选择与新 MCP infernux.scene.mesh.assign 经 Host 共用此命令；没有新增 MCP 私有 JSON 改写或回退路径。普通字段的完整 schema/序列化名称映射仍是后续语义工作，不能把这个聚合操作当作所有字段问题已解决。
- 原生引擎集成测试覆盖方块→模型、Undo 恢复完整旧来源、Redo、重复赋值不产生历史、无效 GUID 不修改对象。测试最初错误使用不存在的 Scene 构造器和 AssetDatabase.instance，已改为引擎 scene fixture 与现有 AssetManager 数据库入口。扩大网格/资产/命令/Inspector/MCP 回归 140 项通过，33.97 秒。
- 桌面 Infernux041Lab 的场景 06 新增 Saved Slope Asset（对象 48，MeshRenderer 139），只包含 Transform/MeshRenderer/BoxCollider，无生成脚本。通过 MCP 绑定 SavedSlope.inxmesh，保存后切换果冻场景再返回，GUID 与 boundsMax=[1,1.5,1] 保持，实际捕获并查看斜面正常显示。证据：.infernux/mcp_sessions/20260909-032900-35fdbdf9/review/saved-mesh-reopened-bound.png。
- 此场景仅为技术验证，不算漂亮演示交付。碰撞未按保存斜面重建、Player/Cook 未验，现场未操作 Undo（由集成测试覆盖）。A15.1 已规定果冻、雪地、轮廓、世界 UI、物理游乐场分阶段交付，继续保持。无提交/推送/制品发布，完整 041 未完成。

## 2026-09-09：MCP 资产类型修正与场景赋值缺口

- asset_identity 改用原生元数据 get_resource_type().name，移除读取不存在的 meta.type 后显示 unknown 的兜底。文件缺失元数据明确报 asset.not_found；目录返回 folder。增加 native-method 合同测试，20 项 MCP 回归通过（20.48 秒）。安装副本同步并重启后，SavedSlope.inxmesh 的实际 MCP inspect 返回 resource_type=mesh。
- 通过 MCP 创建普通 MeshRenderer，尝试用 hierarchy 查询输出的 meshAssetGuid 字段绑定保存副本，实际失败：MeshRenderer has no attribute 'meshAssetGuid'。来源是查询序列化字段名与 Host/组件命令普通属性赋值不一致；需在权威字段/组件服务处解决，不用脚本重新生成几何冒充持久引用成功。
- 新建未绑定成功的对象 48 已通过正常 MCP 删除并保存，现有探针与 SavedSlope 资产保留。review/saved-mesh-asset.png 的请求发生在失败后，不作为保存 Mesh 渲染成功的证据。没有宣称场景绑定或 Player 完成；完整 041 仍活动，未提交/推送/发布。

## 2026-09-09：MCP 保存 Mesh 副本接入

- 官方 MCP 插件新增 infernux.asset.mesh.save-copy，输入源 asset_guid 与 destination，声明 asset.write 和可撤销副作用。操作经 EditorAutomationHost.save_mesh_copy 调用上一轮 Project 命令，不绕过资产历史；支持项目相对目标路径，最终路径/覆盖限制由命令层负责。
- 初始实现直接导入引擎内部模块，被现有架构测试拒绝；已移入 Host API，未放宽架构约束。操作注册数量从 82 更新为 83，必需操作集合加入新入口；19 项 MCP 回归通过（20.22 秒）。
- 桌面实验项目安装副本同步此源码后重启编辑器，通过真实 MCP 在共享网格场景调用保存，得到 Assets/Meshes/SavedSlope.inxmesh，新 GUID f75f22fde3a4c19c9b3f0332c337647e，源 GUID 78a4791f339f4da174692aec7b8ac663。真实会话证明操作可执行/导入；本轮未另验文件几何或会话 Undo/Redo，后者此前由原生资产库集成测试覆盖，不混淆证据范围。
- MCP asset_identity 的旧字段访问把摘要 resource_type 显示为 unknown，已记录待修；不以此宣称类型展示正确。UI 保存菜单、场景引用与 Player Cook 仍未交付。未提交/推送/发布，完整 041 继续推进。

## 2026-09-09：Project 命令层保存 Mesh 副本

- 新增 ProjectAssetCommandService.save_mesh_copy(mesh, target_path)，限定项目 Assets/Packages 下的新 .inxmesh 文件。使用现有 NativeDocumentStore 原子二进制写入，再经 AssetManager 正常导入，外层复用 ProjectAssetCreateCommand 的 Undo/Redo，不新增历史栈或资源注册表。
- 目标存在时拒绝覆盖；提交要求目标仍不存在。条件写入失败直接传播，不让创建命令清理外部新出现的文件；导入失败则沿原创建命令清理本次创建内容。没有生成备用文件格式。
- 真实资产库测试验证独立 GUID、撤销移除文件/记录、重做保留 GUID/字节并可加载；额外测试在编码期间模拟目标被占用，保存冲突后外部文件内容保持。资产/命令/DocumentStore 51 项通过（14.99 秒），追加冲突用例单测通过（2.84 秒），diff --check 通过。
- 这是编辑器命令服务入口，尚未接 UI 菜单和 MCP operation，也未做完整场景引用/Player Cook 验收。不将保存副本视为原模型覆盖或完整 Mesh 作者 API；041 继续推进，未提交/推送/发布。

## 2026-09-09：原生 .inxmesh 源接入资产导入

- 在统一 AssetFormatRegistry 注册 .inxmesh；MeshLoader.ImportSourceDetailed 对此源调用已建立的 DeserializeSource，保留作者几何，绕过 Assimp 转换。结果回到现有 ModelImporter，生成同样的带源哈希主制品和空骨骼伴随制品，运行时 MeshLoader 路径不分叉。
- InxMesh.serialize_source 暴露 bytes 编码，不负责落盘/导入。测试将编辑后的模型编码为独立源，经 AssetDatabase.import_asset 获得不同 GUID，再清缓存重载，完整源编码仍一致；修改原模型不影响副本。重写副本源并重新导入，副本 GUID 保持，清缓存后读到新数据。
- Windows 模块构建成功；110 项资产导入、类型与网格回归通过（15.34 秒），diff --check 通过。更新 MeshLoader 过时的“无中间二进制”注释。该入口只支持静态作者几何，不应用 Assimp 导入转换参数。
- 文件写入目前仅在测试中进行，尚无用户可用的编辑器保存事务/Undo/MCP 保存 Mesh 命令，也未做场景保存与 Player Cook 验证。因此不将该步骤算作完整 Mesh 持久化交付，完整 041 仍活动；未提交/推送/发布。

## 2026-09-09：静态 Mesh 作者源编解码入口

- 核对现有导入链：MeshLoader 读取绑定源哈希的 MeshArtifact 和独立 SkinnedMeshArtifact，不能把 Library 缓存直接当作可编辑源文件。新增 MeshArtifact.SerializeSource / DeserializeSource，复用现有二进制布局和校验；用固定源角色标识区分作者源与绑定源文件哈希的导入缓存，不新增平行几何格式。身份由未来正常 import 分配，不继承原模型 GUID/路径。
- 往返测试验证法线、切线、UV、索引和材质槽，重新编码保持相同字节；作者源与导入缓存不能互相误读。存在 skinned payload 时明确拒绝保存为静态源，避免静默丢失动画/骨骼。
- 首次原生测试失败来自空 skinned fixture 本身不满足 SetSkinnedData 合同，修正为有顶点/索引的有效载荷后通过。Windows 原生模块构建成功，mesh_artifact CTest 通过（0.01 秒），46 项资产/网格回归通过（14.70 秒）。
- 本轮仅编解码基础，不代表已可在编辑器保存 Mesh：源扩展注册、导入器接入、写入事务、作者 API、保存重载及 Player 尚未完成。完整 041 保持活动，未提交/推送/发布。

## 2026-09-09：每帧共享 Mesh 发布回归

- 扩展真实 Vulkan 回归为间隔 24 次与连续每帧 192 次两个模式，均在 240 帧检查上传队列清空、当前资源存在 GPU 数据、旧版本 GPU 字节为 0，且预算淘汰计数不变。连续模式允许在途版本被更新替代，不要求每个中间状态都完成上传。
- 本机连续模式结果：192 updates，采样 peak pending = 0，final pending = 0，stale GPU bytes = 0。现有 PumpPendingMeshUploads 已逐帧处理完成任务并丢弃过期资产版本，因此本轮没有凭推测添加取消、限流或重试分支。
- 新增 CTest infernux.mesh_publication_gpu_continuous，与原测试都归 requires_vulkan_device。需要明确：当前采样没有观察到积压，不证明慢传输设备上的队列峰值上界；不能把小网格 192 次测试当作大规模计算或帧率验收。完整 Mesh 作者能力与完整 041 继续推进，未提交/推送/发布。

## 2026-09-09：共享 Mesh 旧 GPU 版本退休

- 核对发现共享资产几何的旧版本此前只在超预算时淘汰，持续发布会保留过期载荷直到预算上限。PublishSharedMeshBuffers 现在在新版本成功进入缓存后，将同 GUID 的较旧缓存版本送入既有 GPU 延迟释放队列；已发出的绘制引用继续持有租约，不立即销毁在途缓冲，不新增回收框架。
- 新增真实 Vulkan mesh_publication_gpu_test，两个对象共享，24 次位置发布，每次间隔 8 帧，最后等待至 240 帧检查旧 GPU 字节为 0。禁用本轮退休逻辑重新构建，测试失败并报告 6912 stale_gpu_bytes；恢复后通过。该测试证明已完成上传版本退休，不覆盖上传速度落后于逐帧修改时的 pending 队列积压。
- 已注册 CTest infernux.mesh_publication_gpu（requires_vulkan_device、30 秒超时），正式 CTest 通过（3.46 秒）；553 项 Python 回归通过（22.28 秒），diff --check 通过。没有提交、推送或发布；完整 A04/041 仍活动。

## 2026-09-09：网格变形与法线共同发布

- 扩展同一 update_mesh_positions 入口，可选传入 NumPy normals (N, 3)，与位置一次准备、一次发布。省略法线保持旧值；法线形状/数量及有限值在发布前验证，无效法线不会留下新位置旧法线的半更新。没有自动猜测平滑规则或每帧额外烘焙。
- 双 Renderer 集成测试检查新法线可见、无效法线不修改几何/版本、省略法线保持已发布属性，碰撞仍只显式 recook。Windows 原生模块重建成功，553 项回归通过（23.15 秒），diff --check 通过。
- 桌面 SharedMeshPublication 探针按斜面坡度计算单位法线后和位置一起提交。编辑器用新模块重开；本轮未重新做灯光效果对比，不把数据测试当作法线贴图完整验收。切线/UV 等完整编辑、新建/复制/持久化与完整 041 仍未完成。未提交、推送或发布。

## 2026-09-09：共享 Mesh 的实际 Vulkan 画面验证

- 在桌面 Infernux041Lab 通过 MCP 创建并维护独立 06_SharedMeshPublication.scene，两个对象共享 SharedSurface.obj；SharedMeshPublication 的 lift 驱动同一 registry 位置发布。第二对象只消费，不写共享资源，避免保存重开时两个作者互相覆盖。该场景是技术探针，不是最终雪地演示。
- 从新编译 Windows 模块启动可见编辑器，固定 Scene 相机，MCP 引擎渲染目标捕获 flat / raised 两帧（800×600）。实际查看图片确认两块平面同时成为斜面，证明修改不仅进入 CPU getter，也进入 Vulkan 绘制。没有做图片哈希来代替视觉判断。
- MCP 保存后重开，并切换到果冻场景再切回，捕获 switched-back 仍显示两个斜面。这里恢复的是组件参数并重新发布几何，不代表运行时 Mesh 数据已经持久化，也不代表 Player 路径可用。重开前控制台计数 0 errors / 0 warnings。
- 证据位于桌面项目 .infernux/mcp_sessions/20260909-024046-dd6290fe/review/shared-mesh-*.png。当前编辑器保持该技术场景打开；没有提交、推送或发布。创建/复制 Mesh、完整属性编辑、持久化与完整 041 仍待推进。

## 2026-09-09：共享 Mesh 位置发布接入 registry

- 新增 UpdateMeshPositions / update_mesh_positions(guid, first, positions)，在 owner thread 上准备候选后保留原实例发布。更新 CPU 驻留、runtime version、加载代际，沿 RuntimeModified 通知共享消费者，不重新导入、不写源文件。位置输入为 NumPy (N, 3)，保留拓扑和非位置属性；空更新不发布，无效输入在修改前报错。
- 实测原 Python 引用读到更新，输入数组后续修改不污染 Mesh；源文件不变，旧 worker ticket 不能覆盖新发布。两个 Renderer 同时读到新位置且不提交新碰撞任务；只 recook 一个对象后其射线高度变为新高度，另一对象仍保持旧碰撞。
- Windows 原生模块重建成功；新增两项集成测试通过，扩大回归最终 553 passed（19.55 秒），diff --check 通过。不是完整 Mesh 作者入口：创建/显式复制、法线编辑、GPU 画面/缓存验证、持久化和 Player 仍待推进。本轮未提交、推送、发布，未标记 A04 或 041 完成。

## 2026-09-09：运行时 Mesh 通知分流

- 在同一 AssetDependencyGraph 增加 RuntimeModified：只传播给运行时依赖，不触发源资产依赖链。MeshRenderer 刷新 bounds、材质槽与缓冲脏标记后只通知可见内容变化；文件修改保留原碰撞通知。新分支不输出逐帧导入 INFO，没有新增重试或恢复链。
- Windows 原生模块构建成功，asset_dependency_graph 与 mesh_artifact 两项 CTest 通过（0.33 秒）；551 项 Python 网格/物理/组件/场景/资产回归通过（19.77 秒）。图测试区分了 runtime-only 与文件事件的依赖并集。
- 尚未接入 registry 内存发布器，也未证明多个 Renderer 的端到端更新；本次为通知基础，不勾选整个 A04。演示矩阵继续按功能成熟逐批交付；更新果冻旧 2 FPS 描述，但不将性能改善当作视觉验收。未提交、推送或发布。

## 2026-09-09：Mesh 运行时发布接入核对

- 本轮源码核对确认 InxMesh generation 不等于 registry runtime version，绘制端还检查 IsLoaded 与 AssetRef.cachedVersion。仅开放原生修改绑定不能让共享绘制自动成立。
- 当前文件事件分支会 recook 并逐依赖对象写 INFO；ReloadAsset 又要求磁盘路径/loader。因此后续采用同一 registry 的内存发布边界和明确的运行时内容通知，而不是伪造临时 GUID、模拟文件修改或新增平行资源库。
- 新增 041-mesh-publication-map.md，逐项固定驻留统计、版本/加载任务代际、引用刷新、碰撞边界、GPU 缓存、持久化的接入点与真实验证要求，并链接 A04。本轮是源码调查及实施清单更新，没有新增运行时 API、执行新回归或发布制品；完整目标保持活动。

## 2026-09-09：原生 Mesh 顶点区间更新

- 在现有 InxMesh 增加 UpdateVertexRange，区间边界在准备前检查；索引、submesh 划分和材质槽保持，更新顶点后按导入器使用的 vertexStart/vertexCount 重算 submesh bounds，并沿 SetData 发布总体 bounds 和新几何代际。空更新不发布；越界失败保留旧快照/版本。
- 原生断言覆盖前后未修改顶点、被修改法线、旧快照、索引与材质槽、整体/分块 bounds、空区间、末尾越界与极大索引。开启断言的 mesh_artifact CTest 通过（0.02 秒）。Windows 模块重建后 505 项扩大回归通过（18.59 秒），diff --check 通过，编辑器已重开报告 ENGINE_LOADED。
- 此入口目前是 CPU 原生层，保留旧代际需要复制几何，未声称局部 GPU 上传、零拷贝或性能完成。公开 Python Mesh 创建/编辑、registry/runtime version 通知、GPU 消费和持久化仍待串接；完整 A04/041 未完成。本轮未提交/推送/发布。

## 2026-09-09：共享 Mesh 的不可变几何代际底座

- 核对 A04 后确认已有 InxMesh/AssetRef/SceneRenderExtractor 是主路径，Python InxMesh 仍为只读加载资产，普通 Renderer 依赖 registry 的已发布 runtime version。没有先添加脱离注册表的运行时 GUID 或平行 Mesh 对象来绕过这些合同。
- InxMesh 几何改为 `shared_ptr<const MeshGeometry>`；SetData 准备完整顶点/索引/submesh/总体 bounds 后替换几何代际，旧 getter、资源身份、Cook 格式和 generation 入口保持。SceneRenderExtractor 持有所引用的几何快照，而非只保活仍可被 SetData 改写的 InxMesh；skinned 专用 pose/model 快照路径未改写。
- 原生测试验证更换数据后旧顶点/索引/submesh/bounds 不变，复制 Mesh 不被另一副本 SetData 改写，空几何 bounds 归零，源 Mesh 析构后快照可读，最后一个引用释放后退休。CPU 字节统计改为计算当前几何载荷，不声称统计所有仍在途的历史代际。
- 发现 mesh_artifact 测试在 RelWithDebInfo 下因 NDEBUG 未执行 assert，为该测试目标显式启用断言。启用后旧球体 UV 三角形跨度断言失败：极点经度未按面展开。修正极点采用相邻两点展开后的面内 UV 中值并按面分离极点顶点，未放宽断言或改变几何细分。最终原生 mesh_artifact CTest 通过（0.02 秒）。
- Windows 模块已重建，扩大到场景/网格/物理/组件/资产数据库/资产类型共 505 项通过（18.65 秒）；diff --check 通过。桌面编辑器用新模块重开报告 ENGINE_LOADED。没有把启动成功当成共享动态 Mesh 的完整渲染或多平台验收。
- 仍缺公开可编辑 Mesh 创建/共享/复制/区间更新、registry 与 GPU buffer 版本消费、容量/CPU-readable/资源退休与持久化的完整串接。本轮是现有主路径的存储准备，不新增临时私有格式或把 A04 勾选完成。未提交、推送、更新平台载荷或发布制品。

## 2026-09-09：碰撞快照复制隔离与在途 owner 退役回归

- 针对 A04 生命周期补实际消费者测试，不改变运行时机制。新用例连续创建对象、确认 `is_cooking` 后删除，再创建替代对象；完成同步后无 pending，射线只命中新对象的高度 7，旧完成结果没有写入新 owner。
- 初版测试只在首次物理边界前请求烘焙，尚无 body，不能证明真正的在途删除。将测试改为先初始化 body，再发起新几何请求并断言 is_cooking，随后销毁；修正后 12 项网格回归通过。
- 在普通/被新请求替代的快照测试中加入公共 `GameObject.instantiate()`。复制体保持源对象选定的碰撞高度而不是可见面新高度；复制体显式 recook 后采用当前可见几何，原对象仍命中原快照，证明复制隔离。
- 加入完整场景集成，最终 410 项场景/网格/物理/组件测试通过（7.43 秒），diff --check 通过。本轮没有添加新的恢复链、构建原生模块或修改桌面场景，编辑器保持上一轮状态。
- 尚未验证快照跨保存重载/Player Cook 的持久资源语义，也未完成共享 Mesh、区间更新、GPU 直连、多平台和 041 其它阶段；不因本轮生命周期用例通过而勾选整个 A04。未提交/推送/发布。

## 2026-09-09：显式烘焙源快照与缩放同步顺序

- 新增请求期间可见几何继续变化的真实测试，基线失败：初始化后请求高度 1，再仅把可见面改为高度 2，worker 完成回调重读了最新 Renderer 并提交第三次烘焙，而非采用请求时的几何。
- MeshCollider 现在在首次使用或显式几何变更请求时捕获未缩放源顶点/索引，后续创建形状从这个来源应用当前缩放。recook 替换来源，center/scale/convex 变更不重新选择可见网格；完成回调继续使用现有 worker 缓存和 revision，不增加第二套作业或重试链。Clone 复制源数据但不复制 pending 作业，完整克隆/持久化验收仍待补。
- 加入缩放断言又发现原有 SyncTransforms 在发现缩放变更前已经等待 cooking，导致缩放新任务逃出显式同步边界。顺序改为发现并同步变更、等待由此产生的 cooking、发布/同步结果。只调整原来的边界顺序，不循环等待猜测状态。
- 测试覆盖未被替代/被第二次 recook 替代的在途请求：仅可见更新不额外提交 cooking；射线命中选定高度，center 和缩放正确应用到同一来源，新的显式 recook 才采用最新高度。单次 Physics.sync_transforms 返回时缩放 cooking 已结束。
- 本机 conda infernux 下重建 Windows 原生模块，网格/物理/组件 231 项通过（6.59 秒），diff --check 通过。桌面可见编辑器已重开并报告 ENGINE_LOADED；没有声称完整资源生命周期、GPU 数据发布、跨平台或整个 A04 已完成。本轮未提交/推送/发布。

## 2026-09-09：失败的复合碰撞替换不丢失旧网格

- 沿 A04 显式 recook 继续检查替换。真实物理测试先复现：单 MeshCollider 的无效新几何保留旧形状，但同物体存在 BoxCollider 时，BuildShapeForColliderSet 跳过失败的 mesh 子形状，随后用只有 box 的形状替换原复合体，射线不再命中旧网格。
- 复合形状构建现在要求所有启用、非排除的子形状都就绪才返回候选；仍访问全部子形状以启动各自工作，不遇到首个 pending 就短路。原 UpdateBodyShape 的成功后替换边界保持不变，失败/未完成不写当前 body。没有备用 box、重复验证或失败重放。
- 新测试覆盖单体/复合体与主线程三角索引不完整/工作线程退化三角形失败四种组合；失败后旧网格射线命中保持，随后有效高度 2 网格可重建，复合 box 仍存在。测试最初漏设 BoxCollider.size，被既有 AutoFitToMesh 拟合成薄平面；明确设置单位大小后按真实合同验证，未修改 AutoFit 迁就样本。
- Windows 原生模块重建，网格/物理/组件 229 项通过（6.59 秒），diff --check 通过。可见桌面编辑器已用新模块重开、报告 ENGINE_LOADED；本轮不重复宣称视觉或多平台验收。
- 尚需验证请求期间可见几何继续变化时使用的烘焙快照、复合 GPU 碰撞数据与 body 同步发布、销毁/取消和最终各平台消费；没有把这四种失败测试视为完整 A04。未提交、推送或发布，完整 041 目标仍活动。

## 2026-09-09：程序化可见网格与显式碰撞烘焙边界

- 本轮回到 A04，发现现有 Python NumPy 写入绑定调用 `SetMesh`，会触发兄弟 MeshCollider 的 `OnMeshGeometryChanged`。新增真实物理测试首先失败：第一次可见顶点更新使 async_submissions 从 1 变成 2，违背逐帧几何更新不隐式 recook 的计划合同。
- 绑定改用已有 `SetProceduralMesh`，将该原生入口从 protected 提到 public；复用现有 bounds、渲染内容失效与上传退休通道，没有再建网格更新实现。已有 primitive/资产重新赋值行为保留，未混同逐帧顶点修改与换资产。
- 公开 Python/native `MeshCollider.recook()`，直接调用现有烘焙失效/重建入口，补 wrapper 与 stub。NumPy 写入文档明确复制进原生存储、后续渲染上传、碰撞不随之重建，不声称零拷贝。
- 同一场景网格连续抬升 0.5/1/2：两层 API 均无新增 cooking，射线仍命中高度 0；recook 后新增一次 cooking，命中高度 2，shape_error 为空。未绑定 wrapper 明确拒绝 recook。Windows 原生模块已重新构建。
- 扩大回归初次 224 通过、1 失败，失败是已有静态世界同步计数测试假定固定步总会同步碰撞；当前 RunFixedSimulationStep 在没有 Rigidbody 时跳过模拟，由查询按需同步。显式 sync_transforms 会主动标脏全部 collider，不能用来验证增量候选。测试初始化时完成同步，移动单个对象后用实际 raycast 触发查询，断言新位置命中与候选数为 1；未改变引擎静态优化。最终 225 项网格/物理/组件测试通过（6.69 秒）。
- 已保存桌面项目并关闭编辑器进行构建，随后用新原生模块重开。首次启动漏传本地 MCP 端口，确认进程仍存活后正常关闭并按项目启动脚本的 9741 设置重开，不将连接错误当成引擎崩溃。
- 新模块下 MCP 果冻单步捕获通过，高度 1.801745/1.160678/2.183489 与之前一致；查看压缩帧确认程序化渲染更新可见。捕获目录 `20260909-015631-a91becfc/review`，控制台无 warning/error；没有将此单场景证据扩展到多平台或最终视觉验收。
- 共享 Mesh/区间更新/显存直连、烘焙失败的完整替换保证、跨平台载荷与完整 041 仍待完成；不据此勾选整个 A04。本轮未提交/推送/发布。

## 2026-09-09：GPU 果冻局部变量求解与连续演示推进

- 延续 A15.1 的连续演示矩阵；本轮先解决果冻与后续物理场景共用计算路径的性能问题，没有将雪地、描边或世界 UI 标记为已实现。
- 将桌面 `Infernux041Lab/Assets/Scripts/JellyMaterial.py` 的 Neo-Hookean 约束临时矩阵/梯度移入 GPU 线程局部标量，删除两个全局 scratch 数组和参数。仍为 343 粒子、1296 四面体、8 单元颜色、8 子步，材料参数与算法不变；项目使用公开 `inx.compute.hpc`/`batch`，不增加第二套求解器或自动回退。
- 对照旧 GPU 求解器连续运行 300 步（6 秒，含两次 0.9 m/s 挤压），逐步比较位置与速度，检查有限值、正四面体体积和总体积比例。最大位置差 0.000056751 m、速度差 0.000334248 m/s，全部通过。旧版仅作为 ignored dev 测试参考，不进入项目运行路径。去除 scratch 后再次对照通过。
- 单独运行初版局部变量内核约 13.0–13.5 ms/步，提交/等待约 5.1–5.5 ms，旧版对应约 18–20 ms/步。交替运行两版的正确性测试涉及布局切换，得到 26.66/20.44 ms 中位数，不能冒充独立运行性能。
- MCP 刷新实际项目、保存并进入 Play。暂停单步 0.10/0.20/0.36 秒的高度为 1.801745/1.160678/2.183489，压缩/回弹断言通过；真实 Space 挤压为 1.600741→1.372904→1.778414，G 往返与 R 重置通过。首次 review 因 Game 面板隐藏导致测试取空列表，改用已有 show_view 先打开面板，不修改引擎行为来迁就测试。
- Scene 显示时重新捕获内部 Gizmo 开关，Game 捕获已人工查看：材料仍是技术样本，透明外观和布景不算最终视觉验收。证据位于桌面项目 `.infernux/mcp_sessions/20260909-012804-176210b5/review/`。
- 真实可见 Scene + 内部 Gizmo 连续运行，两个后续采样窗口固定步帧均值 17.41/17.80 ms、总体帧均值 13.89/14.58 ms、p95 18.28/18.97 ms；这不是所有帧低于 16.67 ms 的保证。控制台 0 warning/0 error，22 项 Gizmo 单测通过（2.41 秒）。编辑器留在暂停状态。
- 本轮未修改插件 native 载荷或发布制品，未提交/推送。完整 041、其余演示、Player/AOT 与最终平台/CI 验收继续保持未完成。

## 2026-09-09：GPU 果冻的启动边界、真实性与首轮性能修复

- 修正 Taichi preload 对 Python 实现库的所有权：整个 `runtime/infernux_taichi` 归 preload/unload 管理，不只是 `_vendor`。普通项目/包内作者组件仍进入热刷新；库文件仍作为资产打包。真实编辑器启动不再把 backend 的 `contextlib` 导入当成项目依赖报错。补成功/失败/卸载归属回归，不扩大候选导入白名单。
- 桌面项目已显式安装开发 `.inxpkg`，MCP 将旧 CPU 组件替换为 GPUJelly 并保存。343 粒子、1296 四面体、8 个无顶点冲突的单元颜色、8 子步；连续体剪切/体积 XPBD、按体积质量和独立平滑表面。使用 `inx.compute.hpc(device="gpu")`，不是 CPU 回放。内部工作数组由 `inx.compute.batch` 管理，用户边界仍是 NumPy/标量。
- 暂停单步真实 Game 捕获：模拟约 0.10/0.20/0.36 秒，高度约 1.80/1.16/2.18。图像位于项目 `.infernux/mcp_sessions/20260909-010415-4758b22f/review/gpu-jelly-*.png`。初次按墙钟采样因 JIT 延迟失败，已改为实际 R 重置和暂停单步；这组结果仍不代表最终视觉验收。
- 用户指出约 2 FPS，实机日志证实固定步帧平均约 546 ms；20 ms 固定步预算下单次求解约 38–41 ms，导致持续追赶。分段实测上传约 11–12 ms、Python 记录约 3 ms、原生图记录约 7–10 ms、提交/等待约 9–11 ms、回读约 6 ms；不把渲染器 GPU 时间戳冒充求解时间。
- 修正 Taichi CMake 覆盖配置级 flags 的错误：Release 缓存虽为 `/O2 /Ob2 /DNDEBUG`，生成的编译命令却丢了它们。现在保留配置参数，MSVC/Clang 和 LTO 开关的四个 CMake 实测配置回归通过；重新编译生成 `.inxpkg`。构建进程显式载入 VS 开发环境、使用本仓库 ASCII 临时目录，未安装工具链或修改用户全局环境。
- NumPy batch 改为一批原生传输，移除逐数组 copy kernel/sync，并复用上一批的设备布局；每次仍读取新主机值，不靠内容 hash 或缓存旧值判断。相同 8 子步独立实测降到约 18–20 ms，上传/回读合计约 1 ms，物理采样数值与优化前一致；当前仍接近预算，真实编辑器帧率待复测。真实 `.inxpkg` GPU 集成测试覆盖多数组、布局替换、主机修改、图参数更新和正常卸载通过。
- Gizmo 增加 NumPy 批量线段，native 生成的 sphere/arc 不再转成 Python 嵌套列表；打包直接写连续类型缓冲。相同 125 节点/4624 条线的 CPU 收集+打包中位数从 2.50 ms 降至批量 0.056 ms，几何/颜色逐端点比较一致。该微基准不代表完整渲染性能验收，C++ 上传/绘制和可见 Scene 尚待测。
- A15 新增持续演示矩阵：果冻、交互雪地、可交互轮廓、场景 UI、物理游乐场，按功能依赖逐步交付场景与交互/画面/性能证据，不以演示替代 Runner Long 完整关卡。桌面 README 已撤回旧 CPU 场景的现行说明并写明未验收。
- 更新开发 `.inxpkg` 后，真实可见编辑器的固定步帧约 23–28 ms，控制台无错误，未宣称稳定 60 FPS。普通定向回归 254 通过、5 跳过；另测新 CMake 参数和生命周期共 14 项通过。后端存储复用没有改变相同物理采样结果。
- Space 原强度 0.45 m/s 在稳定材料上只产生约 0.10 的高度差；增加可编辑 `poke_speed`，默认 0.9 m/s，不改变材料刚度。MCP 在冻结模拟时输入，再单步采样，验证高度约 1.57→1.39→1.77，R 重置与 G 往返开关通过。Scene 隐藏时本来就不收集 Gizmo；修正测试先显示 Scene，再实际捕获开/关差异，内部节点与连接线显示正常。
- 最终使用新 `.inxpkg` 重跑包集成套件，5 项通过（59.04 秒），包含真实 GPU 数组读写、批次存储复用、卸载/重载与 headless 路径。桌面保留已保存的 GPU 场景，在暂停状态显示内部 Gizmo，用户可继续检查。
- 完整 041、普通 Player/AOT、多平台、官方在线安装、最终 CI/发布均未完成；本轮没有提交或推送。

## 2026-09-09：撤回果冻视觉验收，转入 GPU 连续体材料重做

- 用户否决上一版 CPU 软化效果；已有高度变化和截图只能证明发生形变，不能证明果冻观感合格。A15 已拆分项目创建与果冻验收，后者保持未完成。
- 调研 Macklin/Müller 的 Stable Neo-Hookean Materials (2021) 及补充梯度公式、Small Steps (2019) 和作者交互示例，准备替换距离弹簧主模型；明确 GPU 并行求解、NumPy 边界和共享引擎 Vulkan，不用 CPU 回放伪装 GPU。
- A07 新增 Gizmo 渲染性能收口：真实负载分阶段测量、批量输入/共享拓扑/缓冲复用、可见性与深度语义、优化前后对比。当前大量逐节点 wire sphere 和逐边小批次是待测热点，不把隐藏内部核视作修复。

## 2026-09-09：按用户反馈重做软果冻与内部 Gizmo 验收

- 用户指出果冻接近硬块且未使用自定义 Gizmo。独立参数扫描证实原拉伸柔度 1.5e-5 的落地最低高度约 1.764 m，初始高度 1.8 m，形变仅约 2%；此前有限值/体积保持和静态图不足以证明果冻观感合格。
- 桌面项目 XPBDJelly 默认及场景持久化拉伸柔度改为 0.006；体积约束继续保留 1e-9。阻尼改为按子步时长计算的 exp(-rate*h)，默认 0.35/秒。Space 对上部粒子施加向下挤压和少量剪切，不再主要表现为整体跳跃；R 仍从空中重置落下，CPU 仍不依赖 Taichi。
- 新增可选 show_internal_core/show_core_constraints/core_node_radius，沿公开 on_draw_gizmos + Gizmos.draw_line/draw_wire_sphere 绘制实际内部节点与连接，使用物体 Transform，恢复共用绘制状态。没有额外模拟网格或绕过 Gizmo collector。G 在 Game 输入中切换，Scene 视图显示，默认关闭；Inspector 也可编辑。
- 新加入的测量字段最初误用了 readonly，实际 awake 写入触发异常停用组件。修正声明后通过真实 Inspector MCP 输入恢复组件并保存，不能把这次失败当通过。测试途中编辑器状态/会话发生变化，按实际 MCP 状态重读，不沿用旧截图路径。
- 实际 MCP 播放/暂停与渲染采样为 0.16 s/1.80 m、0.80 s/0.979 m、1.52 s/1.650 m；压扁时宽度 2.974 m，真实截图确认压缩和回弹。MCP Space 测试高度从 1.468 m 到 1.111 m，再到 1.696 m；G 两次切换、R 计时重置通过。Scene 截图分别确认内部线框开启与关闭。
- 证据位于桌面项目 `.infernux/mcp_sessions/20260909-001158-1e8b1f0a/review/soft-jelly-*.png`，项目 README 已更新。独立五秒仿真最低高度 0.966 m、正四面体体积、末态体积比 0.999952。未宣称 GPU XPBD、Player、多平台或完整 041 验收完成。

## 2026-09-09：数据类型随候选模块事务发布

- 上轮重新核对了桌面项目的实际 MCP 运行画面与 35 项 CPU/插件生命周期回归。本轮继续 A01 DataAsset 前置，发现 SerializableObject 虽然在字段编译后注册，但仍会在隔离的候选模块执行中覆盖进程全局类型。
- 三个最小测试先失败，分别证明候选新类型提前可见、失败候选替换旧类型、成功准备已改变旧注册。将临时声明写集交给既有 CandidateImportTransaction；声明和嵌套默认值的 codec 查询只在执行上下文看到私有类型，普通消费者继续使用原来的唯一发布表。
- safe-point 模块提交同时替换所属数据类型，删除该模块不再声明的类型；保留逐键 before-image，外层组件/dispatch/CDS 事务失败时沿原 rollback 恢复。未增加内容哈希、自动重试或第二套长期注册表。独立引擎模块的普通声明不归项目候选所有，跨模块抢占显式 ID 在写入前拒绝。
- 覆盖候选依赖中的嵌套默认值编解码、纯数据脚本沿真实 reload batch 提交、删除与恢复旧类型、其它模块在撤销期间仍保有自己的注册，以及组件模块写入 sys.modules 后外层失败同时撤销数据类型。定向 119 项通过（12.05 秒），扩大组件/字段/codec/候选/重载/compute 回归 722 项通过（15.29 秒），改动文件 diff --check 通过。
- 所有命令在 conda infernux，使用 dev/engine-build 原生模块；本轮只有 Python 实现和本地计划变更，未重启用户正在使用的桌面编辑器，未提交、推送或发布。DataAsset 的独立资产实现、稳定 GUID、活实例嵌套数据迁移、完整 owner 卸载/退休和 Player 目录仍未完成，041 完整目标保持活动。

## 2026-09-08：裁剪 Taichi 前端的张量、工具链与启动副作用

- 上轮 GPU 数组别名修复与真实 GPU 验收属于有效进展。本轮继续 A05 实际载荷裁剪，而非新增恢复层。
- 核对实际代码发现 SourceBuilder 仍可调用 clang++/llvm-as/CUDA bitcode 路径；util、field/matrix/struct、kernel 调用及 autodiff 中仍有 Torch/Paddle 直接转换、自动探测和 CUDA→CPU→CUDA 张量复制分支，与 NumPy/标量外部边界不符。
- 删除上述转换/探测实现和两个已跟踪模块 `lang/source_builder.py`、`_version_check.py`；移除失效导入，内部 compiler Field/Ndarray 自动微分保留，不引入 NN 后端或兼容别名。删除的原模块可从子仓 Git 历史恢复。上游 tools 源码保留作参考，但从启动 exports 和 `.inxpkg` 载荷排除，不伪称整个上游源树已经裁净。
- 去掉 import 时修改宿主 warnings 过滤的语句。解包子进程安装拒绝 Torch/Paddle 导入的测试钩子，并核对 PATH、dlopen、源码查找与 warnings 状态不变；不存在框架时不扫描/安装/修复。最初两项回归失败确认旧 tools 和入口仍存在；修正测试中把 `taichi.lang.field` 导出函数当模块的命名冲突后，按实际类型导入验证移除的转换 API。
- CMake 直接重新造 `.inxpkg`，native 无需重编或手动复制；包集成、真实 GPU/preload/headless、native 入口与动态源码检查 22 项通过（58.97 秒），原作者署名、LICENSE/NOTICE 与第三方文件仍在包中。中英文 README 与 NOTICE 同步实际边界。
- 本轮没有发布在线制品、提交或推送；完整 041、官方可安装、普通 Player/AOT 与剩余跨平台验收仍未完成。

## 2026-09-08：GPU 运行时数组别名与编译前依赖分析

- 上轮桌面 MCP 果冻、真实输入/持久化及 707 项引擎回归属于有效进展。本轮继续 A05 共同数值合同，保留整个 041 的原始交付范围。
- 新增两项禁止调用编译器的最小回归，均先失败：同范围 NumPy 数组经不同参数名传入后，原有静态 HIR 名称判定漏掉跨参数前缀读取和重叠写入。不是为假设故障增加恢复路径。
- 抽取已有 HIR 的读写依赖规则供静态分析与实参别名分析共享，删除不再需要的 AST 位置转接辅助函数。GPU 准备要求共享 span 的 dtype/shape 一致，部分重叠仍拒绝。别名模式加入原有有界特化键；准备时分析一次，正常调用仅检查数组布局/地址关系，不哈希源码/数组内容、不比较 CPU/GPU 输出、不重放。
- 实机测试覆盖先缓存合法的不重叠特化，再传危险同范围别名：报错且输出/调用次数不变，随后合法调用仍可执行。保留同索引累加、同范围 view 与奇偶位置写入；新增 dtype 重解释和 reshape 别名拒绝。第一次包回归 18 通过、1 失败，定位为 affine trip-count 条件误限制安全的奇偶位置写入；修正为对范围内任意迭代值成立的索引不相交证明，未改样例回避失败。
- CMake 的 infernux_package 目标已重新生成 `.inxpkg`，无手动搬运 native 文件。定向引擎 HIR/JIT/compute 与插件生命周期 109 项通过（4.47 秒）；包集成、真实宿主 GPU/preload/headless、native 准入共 19 项通过（67.81 秒）。中英文 README 与插件说明同步。
- 现有 HIR 未分析的嵌套/间接绑定别名明确拒绝；完整嵌套和间接索引语言、数值精度、兼容性 CPU lowering、AOT/Player、官方发布及其余 041 未完成。本轮未提交、推送或发布在线制品。

## 2026-09-08：桌面 MCP XPBD 果冻实验

- 按用户新要求创建 `D:/Users/Chenlizhe/Desktop/Infernux041Lab`，沿用 Hub 项目模板，仅默认 MCP。通过实际 MCP 端点创建/编辑相机、灯光、RenderStack、果冻、地面、参考网格与 HUD，保存 `Assets/Scenes/01_XPBD_Jelly.scene`。Python 求解脚本走正常资产刷新；未用任意 Python 执行接口绕过 MCP 场景操作。
- XPBD 使用 `inx.compute.hpc(device="cpu")` 与 NumPy/标量；216 粒子、750 四面体、1115 边，固定步子步求解距离、体积和地面约束。透明 Lit 材质不代表真实折射。独立热编译后四秒仿真约 0.92 ms/步、体积比 0.999933，无 NaN/翻转四面体；不包括网格上传和渲染成本，不宣称是 GPU 求解。
- 实际 MCP 点击 Game 后注入 Space，截图确认跃起旋转/形变；R 后模拟时间恢复约 0.04 秒。保存/重载及完整关闭重开后场景保留，重新 Play 无控制台错误。首轮地面颜色编辑未实时显示，重载后生效；未据此宣称已修复材质热刷新。
- 场景暴露 `inx.compute.hpc` 尚未被候选脚本声明策略识别：补充受控装饰器导入形式及候选导入可信模块，不开放整个模块的任意顶层调用。同步全类 FieldSchema 在发布前编译的回归与 UIEventArgument 前向定义问题；相关 242 项测试通过。给 MeshRenderer 增加 NumPy inline mesh 的薄公共入口，不重复形状验证，两项真实 native 上传/非法形状/解绑回归通过；项目不再访问私有 native wrapper。
- Taichi 仍不是默认项目插件。缺插件 CPU/GPU 错误路径回归通过；官方目录当前没有 Taichi、远程没有开发版 Release，在线可安装仍是后续交付项，不添加指向不存在制品的下载链接。
- 本轮再次使用当前原生构建及现有开发 `.inxpkg` 运行包集成/preload 生命周期/native 入口套件，17 项通过（56.39 秒），包含真实宿主 GPU kernel、统一 compute、无 GPU headless 与卸载场景；不向桌面实验项目安装插件。最终截图位于项目 `.infernux/mcp_sessions/20260908-232425-32e17c0e/review/jelly-final.png`。
- 最终扩大字段/组件/候选导入/compute/JIT/网格回归 707 项通过（15.34 秒），当前编辑器控制台无错误，项目 Packages 实际仅有 MCP。编辑器留在已保存的编辑模式，可直接点击 Play；未提交或发布本轮改动。

## 2026-09-08：显式字段身份与实际 CDS 重命名事务

- 上轮 compute 命名与数据边界确认、文档更新和 27 项测试属于进展。本轮继续 A01/T.1，不缩小完整 041 目标。
- `FieldMetadata` / `serialized_field` 增加可选 field_id，省略时仍使用声明名；FieldSchema 导出独立身份，属性路径不变。空值/错误类型/首尾空白在声明时拒绝，有效 MRO 中不同字段的重复 ID 在 CDS/type 注册前拒绝；允许同名字段覆写，不加入帧内校验或哈希。
- 共享活实例迁移先按身份匹配，未匹配才使用显式 former_names。同名但更换 ID 是移除/新增，不能借同名取回旧值；保持 ID 的属性重命名和两个属性名称交换均保留对应值。把原来隐含的名称 ID 显式写出不改变存储语义。
- 增加组件和 SerializableObject 的继承/多继承、重复 ID、非法声明、注册失败不覆盖旧数据类型，以及数据文档仍使用 authored keys 的回归。首次定向 64 通过、1 失败是新增测试错把 typed document 的 fields 当顶层，核对现有格式后修正测试，未改动场景格式。
- PlayModeManager 集成覆盖真实原生 CDS 重命名：准备阶段原值/类型未变，提交后新属性与 batch_read 保留值，旧 runtime epoch 的槽位值仍可读；分别验证 finalize 后继续写入及 rollback 恢复原属性/槽位。没有增加运行期属性别名或失败后重试。
- 定向包含重载 102 项通过（7.38 秒）；最终扩大到相机、组件、场景、Inspector、codec、目录、View 服务等 1091 项通过（18.94 秒）。所有命令使用 conda infernux，加载 dev/engine-build 中的原生模块；本轮 Python 改动无需重新编译 native。主仓、Taichi/Web 子仓 diff --check 均通过。
- 历史文档的 schema-version 迁移、全类声明预编译、父子类型同批重载、原生/脚本 owner 发布、DataAsset/MCP/cooked catalog 和完整 041 各平台交付仍未完成。未提交、推送或发布，不把 Windows 回归测试当成完整设备验收。

## 2026-09-08：开工基线与第一批字段事务收敛

### 计量与估时

- 主计划原有 196 个执行条目，语义附录 63 个细分条目；两者存在包含关系，不能直接相加作为工作量。
- 此前仅完成计划与源码核对，实施验收进度为 0%。本次完成 A00 的基线记录条目，主清单 1/196（约 0.5%）；A01 仅部分实现，不勾选整项。
- 全部范围初估 20–36 人周，单人全职约 5–9 个月。该数值是低置信度工程工作量区间，不是自动代理运行时间或交付承诺。
- Taichi Python 3.13、同设备 Vulkan 互操作、Web CPU/Wasm 是关键风险。完成最小构建/运行验证后重新估算，不凭测试条数推算完成率。

### 实际基线

- 引擎：`E:/project/InfEngine`，`master`，HEAD `d0eac0f2d29c0e606288e5f2d64b5b2fd30ae203`；本次实施开始前跟踪工作区干净，与 origin/master 同步。
- 游戏：`E:/project/UnityProject/Runner-Long`，HEAD `134d85215f01af4305b4ed1411b02cf2aec11f7d`，已不同于规划时基线。
- 游戏用户未跟踪文件：`Assets/Blender/cha1-level4.blend1` 和其 `.meta`。未修改、移动或删除。
- Level02 场景：`Assets/Scenes/MainFlow/03_Level02_Custom.unity`，GUID `83ab4f5624566554e9d439fd1d4d1200`。
- 关卡数据：`Assets/Data/Levels/Level02_第二部.asset`，GUID `9fcdfea6af6772b4ebb8efa237be8698`，`levelId=level-02`，`sceneName=03_Level02_Custom`。
- 关卡依赖：chapter GUID `7e721f3308259874f8983a99a1d1fe51`；configuration GUID `0ae4add1e14bfdc48bc6972ff5a752a6`；初始单位 `core.kilogram`；前置 `level-01`，后续 `level-03`。
- 尚未采集新的 Unity 运行参考、完整玩法记录、音频听测和帧时数据；其它 A00 项目保持未完成。

### A01：已有 Python 属性事务主路径

- 修改 `make_python_component_property_transaction`，必须取得声明字段元数据，不再吞异常后跳过字段合同。
- 类型、只读状态和标准输入转换由字段声明提供；删除函数及 Inspector 调用方重复传递的 `value_type/read_only/normalize` 参数。
- 复用已有 `coerce_serialized_field_input`、`normalize_runtime_field_value`、common codec 与 `PythonComponentDocumentCommand`，不新增备用 schema/恢复系统。
- 在比较和入 Undo 前完成范围规范化，重复越界输入归一到相同值时不产生新的修改；多目标任一只读即禁止该次编辑。
- 原手写假组件用例换成真实 `InxComponent`；增加只读、未声明字段、枚举与 Undo、三个编辑入口范围/非法输入/无变更、多目标只读回归。
- 先验证 3 个新增用例失败，再改实现；最终相关 200 项测试通过（9.81 秒），`git diff --check` 通过。
- 不代表 native/Python 完整 schema、DataAsset、MCP 目录迁移或 Player 目录已实现；A01 整体验收仍未完成。

### 测试环境

- 所有命令先 `conda activate infernux`，解释器为 `C:/ProgramData/anaconda3/envs/infernux/python.exe`。
- Python 源码通过 `PYTHONPATH=E:/project/InfEngine/python` 加载，已打印并核实 `Infernux.__file__`。
- 源码目录没有 `_Infernux.pyd`。本次 Python 层回归显式设置 `INFERNUX_NATIVE_MODULE_DIR=C:/ProgramData/anaconda3/envs/infernux/Lib/site-packages/Infernux/lib`，使用已安装 cp313 原生扩展；未修改已安装 wheel。
- 此次使用真实 native/Vulkan 测试 fixture，不是全新 native 构建的验收。原生改动后必须重建，不沿用该证据。
- 测试文件：`test_serialized_property_core.py`、`test_serialized_field.py`、`test_component_schema_migration.py`、`test_value_codec.py`、`test_inspector_contracts.py`、`test_inspector_asset_references.py`、`test_editor_automation_host.py`、`test_host_operations.py`、`test_mcp_automation_operations.py`。

### A05：Taichi 准备

- 用户已有 public fork：`https://github.com/ChenlizheMe/taichi`，master `ff251e1145fabff95b0682fe3f0d5a8ec23187ad`（2023-11-03）。尚未改动远程。
- 本次查询上游 `taichi-dev/taichi` master 为 `ba0e81dce559fb63a5958bf82feb1d00c55c02fe`（2025-07-30）。
- 在忽略目录 `dev/taichi` 拉取现有 fork 以核对升级和裁剪；尚未作为默认插件或子模块发布，不声称计算后端可用。
- 确認 fork 相对上游无独有提交（`0 / 79`）后，本地 master 已 `--ff-only` 前进至上述上游基线；远程没有推送，用户旧历史仍保留。
- LLVM CPU 构建明确依赖 LLVM 开发库和 Clang；当前激活环境能找到 CMake/Ninja，但 PATH 中未发现 Clang。尚需检查机器已有工具链并建立可复现依赖配置，不将缺依赖改成自动选其它后端。

## 2026-09-08：Taichi fork 命名与 Vulkan 所有权修订

- 上一目标轮有实际源码改动及 200 项通过证据，属于进展，不是无进展等待。
- 按用户新指示，GitHub public fork 已改名为 `https://github.com/ChenlizheMe/infernux_taichi`；fork 关系保留，已核实 `isPrivate=false`、`isFork=true`，本地 origin 已更新。当前源码暂在 `dev/taichi`，接入主仓时使用规范子模块路径。
- 不再以 Taichi 自有 Vulkan RHI（即使导入同一个 VkDevice）作为最终架构。适配其通用 `Device/GfxRuntime` 与引擎 RHI，复用引擎的分配、pipeline/bindings、命令和退休所有权。
- 核对发现 upstream `ti_import_vulkan_runtime` 仍创建 `VulkanDevice` 包装并自行初始化 Vulkan 结构，且带固定能力猜测和额外窗口扩展；不能把该入口直接当成最终集成验收。
- fork 已移除 GUI/GGUI Python 绑定源码及顶层 GUI 导入，解除 Vulkan 自动启用 GGUI/GLFW 的耦合，去掉 kernel 初始化的联网版本检查和后端轮询/CPU fallback；尚未通过 native 编译，不能宣称可用。
- CPU/Vulkan 构建入口先只构建 CPU 编译验证，`TI_WITH_VULKAN=OFF` 明确不编译 upstream 独立 Vulkan 运行栈；GPU 测试须等引擎 RHI 适配完成后执行，不用独立设备样本替代。
- CMake 配置首次停在 Windows SDK 的 rc/mt 搜索路径，不是 Taichi 编译错误；已安装 MSVC，接下来修正构建环境，禁止掩盖编译器检查。
- LLVM15 官方 upstream workflow 使用的 Windows 归档正在通过 curl 下载到 `dev/toolchains`（命令会话 54200）；已完成所需 10 个源依赖子模块下载（会话 55296 已成功退出）。未安装新的 Vulkan SDK。
- 最新 GPU 所有权验收新增一条，主清单变为 197 条；当前仍只有 A00 基线条目完整验收，不据此扩大进度。
- public 仓库 description/topics 已更新，名称指向 Infernux；本地中英文 README 已重写，明确关联引擎、CPU/引擎 RHI 职责及尚未验证的制品状态。源码/README 改动暂未推送，等待构建链路贯通。

## 2026-09-08：inxpkg 交付边界与署名

- 用户再次确认最终交付是 `.inxpkg`，不是 Taichi wheel；原生编译只是插件载荷的生产步骤。
- 已将整个 `dev/taichi` checkout 移至 `external/plugins/infernux_taichi`，注册 `.gitmodules`，并检查迁移后子仓与已初始化嵌套依赖的 Git 状态正常。没有删除源码/编译产物；旧 CMake 缓存含旧绝对路径，后续重新配置，不直接复用。
- fork 新增根目录 `NOTICE`，保留上游原始 `LICENSE` 和 C++ 作者头，已修改的上游实现文件补充 Infernux 修改说明。
- 041 末尾新增第 22 节、5 项交付/许可验收；主清单为 202 条，仍只有此前完整验收的 1 条勾选。本轮准备动作不计为最终插件交付完成。
- 子模块和相关代码尚未提交/推送；父仓发布前须先发布子仓可获取的提交，再固定其 gitlink。LICENSE/NOTICE 随包复制及最终 Player 核验尚待包装实现，不声称已有可用 `.inxpkg`。

## 2026-09-08：原生编译环境与 Python 3.13 配置

- 上一轮已迁移子模块并修改源码/计划，属于进展。下载会话 54200 在本轮重新轮询仍存活、文件持续增长，未重新发起下载。
- 最小 C 程序普通 `/MD` 编译成功，而 `/Zi /FS` 失败并报 D8050。只将当前进程 TEMP/TMP 改为工作区 ASCII 路径后，同一调试编译成功；无需重装 SDK 或跳过检查。
- 重新配置后，MSVC C/C++ compiler ABI 检测均通过。此前 rc/mt 路径及 D8050 不再阻塞；本轮本机复现命令保存在忽略的 `dev/build_taichi_native.ps1`，明确激活 infernux 和已有 VS 开发环境。
- Taichi 构建改用 CMake FindPython 的 Interpreter/Development.Module/NumPy 同次查找和 pybind11 3，实测选中当前 conda CPython 3.13.15、pybind11 3.1.0，移除 legacy FindPython 警告链和错误的 setup.py 安装引导。
- CPU runtime bitcode 构建改为有输出/依赖文件的增量规则，产物写 build 而非源码；删除其废弃 CUDA 自定义编译入口。需实际 Clang 构建继续核验。
- Python 全目录 compileall 通过（原上游 image.py 有一处文档转义 SyntaxWarning）；该检查只证明语法可解析，不证明原生 ABI、内核运行或插件安装。

## 2026-09-08：JIT 分工修订与原生基线结果

- LLVM 下载会话 54200、Clang 下载会话 97945 均已退出 0；官方上游构建使用的 LLVM 15.0.1 和 Clang 14.0.6 已解压到 `dev/toolchains`。仅维护者构建依赖，未修改全局 SDK。
- Windows 中文 MSVC `/showIncludes` 与 CMake 探测编码不一致导致 Ninja 未识别依赖前缀；进程 code page 65001 下重新配置后正确识别，无手写语言前缀兜底。
- Taichi 原生构建会话 77481 成功退出 0，216 步完成，生成 CPython 3.13 扩展和 C API；尚未做 kernel 运行/插件安装验证。这份 CPU 基线不是新的最终交付方向。
- 现有 JIT 核对：`Infernux.jit` 包装 Numba，Typed HIR 用于循环合法性/并行分析。真实数组平方和得到 14.0，生成 CPUDispatcher 与 nopython signature。本机 Numba 0.67.0、llvmlite 的 LLVM 22.1.0；与 Taichi CPU 重合，不能直接共用二进制 LLVM。
- 用户提出统一内置 JIT 的 CPU/GPU 分工：CPU 保留 Numba，Taichi 作为 `.inxpkg` GPU 后端，设备/资源/同步仍归引擎 RHI。A05 已修订已确定部分，Web 计算验收冲突明确保留待确认，没有缩减 041 完成定义。
- FieldSchema 已由编辑器事务模块迁至运行时值模块 `Infernux.field_schema`，旧入口重导出同一类型；元数据深冻结并支持脱离实例的文档往返。会话 35247 验证相关 5 文件共 182 项通过；完整 canonical 字段编译/C++ 目录/Player 发布尚未完成。

## 2026-09-08：新计算装饰器与兼容性回退计划修订

- 按用户新要求修订 041 A05、讨论决议及 042 引用：新装饰器暂定 `@inx.compute(device="auto" | "cpu" | "gpu")`，取代并列的旧 njit/ti.kernel 作者入口；本轮只调整计划，没有修改运行实现或新增完成勾选。
- 原生 JIT 只提供 Numba CPU；Taichi 作为 GPU `.inxpkg` provider，裁剪 CPU/LLVM、其它后端和独立 Vulkan 运行栈，继续共享引擎 RHI。
- 核对当前源码：自动调度是 CPU 串行/并行选择，已有 Typed HIR、签名桶、有界决策缓存、预热和原因/耗时跟踪；新增跨设备选择要保留这些能力，不将其记成已有 GPU 自动调度。
- 实际 GPU 请求缺插件时走 `Debug.log_error` 并终止；用户补充确认设备不支持时允许 CPU 兼容性回退，前提是合法 CPU kernel、平台运行时及资源合同均成立。GPU 优先请求也遵循此规则，编译/算法/执行错误不重放到 CPU。
- 保留 Web 计算验收，移除已经过时的强制 Taichi CPU/Wasm 实施描述；新的 Web 路径仍待确认，不用 Numba 桌面支持替代浏览器证明。
- 补充自动选择成本/资源归属、稳定决策、无逐帧哈希验证、错误分类、统一完成语义、导出迁移及验收矩阵；既有历史执行记录保持原样，不改写成新方向已经完成。

### 后续确认：显式设备，设备内自动并行（取代上段 auto 设备草案）

- 用户最终明确 CPU 自动串并行、GPU 自动并行、必须显式声明设备。041 正文/决议和 042 引用已同步：暂定 `@inx.compute(device="cpu")` / `@inx.compute(device="gpu")`，移除 auto 设备选项与跨设备性能竞速计划，保留 CPU HIR/预热/跟踪机制。
- 上一轮明确允许的设备不支持→CPU 兼容性例外仍保留；不扩展到插件缺失、编译或执行错误，不重建 Taichi CPU 后端。
- 明确 Taichi CPU 与 LLVM/Clang CPU bitcode 工具链从构建、CI 和载荷裁剪；无 LLVM 环境构建 GPU provider 作为验收。Numba CPU 自身仍需要 llvmlite/LLVM，不能据此声称全引擎无 LLVM。
- 本轮仅修订本地 dev 计划，未改运行代码，未将任何新能力标为完成。

## 2026-09-08：移除 Taichi 新引入的 LLVM 依赖

- 用户要求先移除新增 LLVM。已将 `dev/toolchains/llvm15`、`clang14`、对应两个下载压缩包及旧 `external/plugins/infernux_taichi/build/infernux-windows` CPU 构建目录移入 Windows 回收站，约 2.79 GiB，可恢复；未清空回收站，不声称磁盘空间已永久释放。未删除 conda、Numba/llvmlite 或机器已有工具链。
- 规范 CMake 预设改为 LLVM OFF，禁止再次启用 LLVM 或 upstream 独立 Vulkan。移除根 CMake 的 Clang 查找/bitcode 生成链和 TaichiCore 的 LLVM 查找/链接/CPU backend 构建段；本地维护脚本不再传 LLVM_DIR/CLANG_EXECUTABLE。
- 无 LLVM 开发包参数的原生 Python/C API 构建会话 83931 完成 167 步并退出 0；删除工具后再次运行维护脚本，配置成功、增量构建无待执行工作。缓存只保留 `TI_WITH_LLVM=OFF`，没有 LLVM_DIR/CLANG_EXECUTABLE。该结果是编译证据，不是 GPU kernel 已通过引擎 RHI 执行。
- 同一 infernux conda 环境实测 Numba 0.67.0、LLVM 22.1.0，nopython `(int64)->int64` 平方函数得到 49，确认原生 CPU JIT 仍正常。
- 尚未完成上游所有 CPU 源码/旧 CI 文件的整体裁剪、GPU RHI provider、新装饰器运行实现或 .inxpkg 验收；保持 041 目标未完成。

## 2026-09-08：新 compute 的 CPU 链路与真实预热

- 上轮依赖清理属于实际进展；本轮开始实现统一入口。`inx.compute(device="cpu")` 必须显式设备，拒绝 auto/其它名称和 Numba GPU 参数；直接复用已有 njit 的 CPU HIR、自动串并行、有界决策与诊断，不另建计算调度器。
- GPU 尚未接通 RHI provider。新入口当前对 GPU 声明明确报错，没有用 Numba CUDA、独立 Taichi VkDevice 或 Python 循环代替；该错误分支还不是完整的插件注册/生命周期实现，设备不支持时的 CPU 兼容性回退也未实现。
- 找到并修复旧预热缺陷：静态判定串/并行，以及没有并行变体时，以前只记录决定就返回；现在在隔离输入上真正编译/执行所选实现。实测小数组串行、百万元素并行、循环依赖串行均生成 nopython signature 且真实输入不被预热改写。准备错误不再被外层 warmup 吞掉。
- 新 CPU 声明参与导出的 HIR 嵌入，真实 `_compile_user_scripts` 产生 .pyc 并删除源码后，仍可执行并保留并行实现/指纹。声明 CPU compute 却不带 Numba/JIT 运行时的构建提前失败，不留到游戏启动时报错。
- 删除 JIT 公共模块中的运行期 pip/ensurepip/自动安装、重载自修复与相应过时测试，移除 ensure_jit_runtime 导出；保留正常安装流程。旧 njit 仍处于本期迁移中，其内部遗留 fallback/全部作者调用尚未清完，不声称新旧迁移完成。
- 先新增测试得到 8 项失败，再实现；最终源码 Python 相关 10 文件共 274 项通过（6.68 秒），导出相关另 9 项通过（其余 254 项未运行）。使用 conda infernux、源码 PYTHONPATH 和既有安装版 native 扩展，不是新 native 构建或完整 Player 实机验收。
- `git diff --check` 通过，仅有仓库既有 CRLF 转换提示。未提交或推送；没有将 A05 整体或 041 标记完成。

## 2026-09-08：GPU 算术能力与着色器合同键

- 延续前一轮 GPU 原生修改，重新轮询构建会话 38108，确认生产 `VkDeviceContext.cpp` 编译及两个 RHI 单元测试成功；没有因观测间隔重启构建。
- 引擎既有 `DeviceCapabilityState`/Vulkan feature chain 加入 shaderInt16、shaderInt64、shaderFloat64 的物理支持和逻辑启用状态。设备创建仅开启支持的算术能力，并保留 feature chain 的 core features，避免后续渲染 feature 赋值将数值能力清零。它们不隐含 16-bit storage 或 64-bit atomic 支持。
- 找到现有 `ComputeDeviceShaderContractKey` 固定常量占用低位的问题。新增 128 种启用组合的唯一性测试先失败，再改用低位空闲的版本化固定前缀加位编码；不引入 SHA256、逐帧能力探测或额外缓存体系。设备仅支持但未启用的能力不改变合同键。
- 同步既有 ResourceIndex 测试：bindless 必须满足全部必要子能力，一位 descriptorIndexing 不构成可用 ABI；补上 Release 构建的断言启用。补测 float64 feature chain 成功路径和请求失败后的状态复位。
- 聚焦构建使用仓库正式测试源码、现有 VS/Vulkan SDK，不修改全局环境。三个原生测试全部通过（0.22 秒），生产设备创建编译单元亦通过。
- 忽略目录 `dev/rhi-tests` 的本机真实设备检查直接链接生产 VkDeviceContext/VulkanRhiDevice/VMA 与仓库 SDL；会话 23672 构建完成 277 步并退出 0。隐藏 SDL 窗口无人工操作，在 NVIDIA GeForce RTX 5070 Ti / Vulkan 1.4.325 初始化成功，三种算术能力 physical/supported/enabled 均为 1，设备 ID 为 1，VMA 正常销毁。没有用 Taichi 独立 Vulkan 设备代替引擎路径。
- 该实机检查证明本机引擎设备创建及能力发布，不证明 Taichi kernel 已通过 RHI 提交执行；GPU provider、通用资源/同步接口、插件包装及跨平台验收仍未完成，不新增 A04/A05 整体验收勾选。

## 2026-09-08：独立计算提交、Taichi 宿主注入与 preload 边界

- 引擎 VulkanSubmissionExecutor 支持不含 Graphics batch 的 Compute/Transfer 计划；沿用设备 epoch、队列 ticket 和终端依赖合流，不伪造相机或图形 pass。无 Graphics 时明确拒绝 presentation 信号，并正确等待前一任务的完成依赖。新增回归先失败，再修复；包含真实 Vulkan 队列提交在内的四项原生测试通过。空命令提交测试不等于数值 kernel 正确性验证。
- Taichi Program 改为显式接收宿主 Device 适配器，移除多后端自建初始化和宿主 CPU 浮点模式修改；GfxProgramImpl 使用实例持有的结果缓冲，不依赖全局 HostMemoryPool。具体 Infernux RHI Device 适配器仍未完成。
- 移除 GfxRuntime 自行读写 rhi_cache.bin、空 flush 的虚假提交和 2ms 墙钟自动提交。同步缩小到计算流，原生管线缓存归引擎；没有删除必要的资源完成依赖，也没有声称整个编译器不再使用磁盘缓存。
- 修复 LLVM/独立 Vulkan 后端均关闭时 GPU program 的链接遗漏和 SPIR-V 工厂重复定义；GPU 编译数据直接使用单一 SPIR-V 工厂。未启用的上游 CPU 源码/测试仍需后续裁剪，不能据此声称完整上游测试全部通过。
- 会话 27917 原生 Python 扩展、C API 和合同测试完成 51 步构建并退出 0；原生资源所有权/同步/缓存合同测试通过，直接加载新 cp313 扩展的 6 项入口测试通过。合同测试使用计数设备，不冒充 GPU kernel 执行；删除了之前与最终方向冲突的 CPU/独立 Vulkan 测试草稿。
- 用户确认初始化可走插件 preload。核对 EditorBootstrap 和 PlayerBootstrap 均先初始化引擎再调用 PluginManager，之后加载场景；计划和中英文插件说明已冻结 runtime InxPreload 接入、准备阶段预热、unload 退休任务及必要 require_restart 的边界，不增设另一套启动系统。尚未添加伪 provider 或声称插件初始化已贯通。
- 本轮未提交/推送，未发布插件制品；041/A04/A05 保持未完成。

## 2026-09-08：通用 GPU 填充与 Taichi 根缓冲区简化

- 前一轮属于进展：核对实际启动顺序并固定 preload 合同。本轮补入 RHI TransferCommandEncoder.FillBuffer，使用显式四字节对齐范围和重复 uint32 模式；零尺寸/无效编码器/未实现 dispatch/无效句柄/未对齐/越界返回失败，不截断范围，不走 CPU 替代。TransferDestination usage 和同步由调用方按合同声明。
- 生产 VulkanRhiDevice 直接记录 vkCmdFillBuffer，复用引擎句柄及缓冲区尺寸。没有新建资源池、队列或插件 VkDevice。
- 扩展正式 RhiContractsTests 和 VulkanSubmissionExecutorTests。后者在引擎 RTX 5070 Ti 的 Compute 与独立 Transfer 队列分别分配设备缓冲区，执行整块模式填充、局部清零、尾部模式填充，再经 GPU copy 复制到 RHI Readback 缓冲区，等待完成后逐项校验 16 个 uint32。验证整块、子区间、相邻数据保留及非法范围拒绝，不再只有空命令提交证据。
- 五项聚焦原生测试通过（首次 0.95 秒），再次启用 CTest 对 Vulkan Validation Error 的失败规则后亦全部通过。详细运行显示真实 Khronos validation layer、独立 Compute family 2 / Transfer family 1，未报告 Vulkan validation error；两个 presentation 错误属于既有预期拒绝用例，不等于验证层错误。首次构建因 PowerShell 未获得 MSVC INCLUDE 而失败；显式 Enter-VsDevShell 后构建成功，未安装或改动 SDK。
- Taichi 回读已有 readback_data 等待提交 semaphore 并同步完成复制，移除调用前额外 device.wait_idle。该改动基于现有同步合同；真实 Taichi kernel 的回读仍待适配器贯通后验证，不拿 RHI 填充结果替代。
- 合并根缓冲区 allocation 与 byte_size 为一个条目，删除指针→尺寸 unordered_map；修复 get_root_buffer_size 先索引后检查造成越界的次序，空/负/越界 ID 在访问前明确失败，SNode 指针获取复用同一检查。保留空根分配 4 字节的既有合法布局语义。
- Taichi Python/C API/原生测试完成 15 步构建。新增错误路径测试最初未捕获上游 TI_ERROR 抛出的 std::string（不是 std::runtime_error），修正测试后会话 43824：原生生命周期/根缓冲区测试通过 0.52 秒，6 项直接原生 Python 入口测试通过。未悄悄改写整个日志异常合同来让测试通过。
- 主仓与 fork 的 diff --check 通过。修改保留于现有工作区，未提交/推送；具体 Taichi RHI 适配器、preload provider 注册、数值 kernel、inxpkg 交付和其余 041 验收仍待完成。

## 2026-09-08：CMake 直接生成插件开发包与前端加载裁剪

- 前轮属于进展：通用 FillBuffer 与实机回读验证完成。本轮新增 fork 内 `package/inx_package.json`、独立中英文介绍页和 `infernux_package_payload` / `infernux_package` CMake 目标。原生 Python 扩展直接链接到 `package/runtime/infernux_taichi/_vendor/taichi/_lib/core`；PDB/import library 留在构建树，不再需要维护者手动搬运原生文件。C API 仍作为单独开发目标构建，不将其未接通的上游 runtime API 宣称为 Player 后端。
- 复用现有官方插件的标准库独立 `package.py`，无需 import Infernux；只将 package 目录内容写入既有 INXPKG 格式。Python 前端是运行所需载荷，C++ 源码、CMake、仓库 README、CLI、示例、GUI 和 pycache 不进包。生成目录的 Python 文件在重新整理前清理，避免被删除的模块残留，原生扩展保留；生成副本可由构建恢复，作者源码未删除。
- 随包携带 LICENSE/NOTICE、FP16/PicoSHA2/spdlog/SPIRV-Headers/SPIRV-Tools/Eigen/pybind11 的许可证，以及 fmt 与嵌入 json/miniz/stb 的原文声明。更新中英文 README、NOTICE 及本机维护脚本。首次整理失败于 Windows pybind11 许可证路径反斜杠被 CMake 当转义，转换为 CMake 路径后成功，未修改安装环境。
- 完整包内前端导入暴露 dill 依赖。移除旧 Blender/IPython/REPL 临时源码探测和全局 inspect monkeypatch，直接使用标准 inspect；新增 authored 源码、linecache 注册的动态源码以及未登记源码明确失败的测试。没有删除动态 JIT 需求；引擎发布器的源码注册/退休与 AOT 链路仍须后续贯通。
- 原生加载器不再修改 PATH/dlopen flags，不读版本 timestamp、不打印 LLVM banner、不扫描 wheel/pip 并推荐自修复；去掉前端 colorama 依赖。完整前端现在从包内模块加载，不通过 pip 临时安装 dill 或改用已安装的上游 Taichi。
- 发现 Catch 测试运行器与 bit 测试被编入 runtime。移除 runtime 的 run_tests task/测试库构建输入和 SPIR-V 头的测试依赖，将原位操作断言迁入独立 infernux.bit_contract 测试目标；同时移除两个剩余 LLVM_INCLUDE_DIRS 引用。没有删除这组测试来降低门槛，原位操作断言继续执行。
- 会话 56735 完成重新构建及打包，2 项原生测试通过（0.08 秒），11 项 Python/包集成测试通过（2.15 秒）。包集成使用现有 Infernux 读取器检查元数据、运行时角色、文档、许可证与排除内容；真实解包到中文/空格项目路径后，以隔离子进程加载包内 Python+cp313 原生模块，验证无 PATH/inspect/dlopen 改动、未创建 Program，用户 ti.init() 无宿主设备时明确拒绝。
- 开发包约 29 MiB，位置为 fork 的 `build/infernux-gpu-compiler/infernux_taichi.inxpkg`。这是编译器开发载荷，不是已可用的 GPU 插件；元数据与文档明确标记 0.1.0.dev1，未发布到官方列表/Release。Linux/Android/Web 载荷、RHI 适配器、preload 注册、GPU kernel 与 Player 验收仍未完成。
- 主仓/fork diff --check 通过，未提交或推送，未改变 041 完成门槛或添加整阶段完成勾选。

## 2026-09-08：宿主可见映射与 Taichi 缓冲区适配

- RHI 增加 MapBuffer/UnmapBuffer 与 Read/Write/ReadWrite 访问合同，生产 Vulkan 实现复用 VMA 已映射内存与 invalidate/flush。调用方负责完成依赖；接口不等待 GPU、不创建中间数组、不新增映射注册表。ReadBuffer/WriteBuffer 复用该入口。
- 实机提交测试增加上传缓冲区映射写入、GPU copy、回读验证和局部 ReadWrite 修改；覆盖设备内存不可映射、访问权限、零长度、越界、失效句柄。五项原生测试通过后，增加 Taichi 缓冲区适配实机测试，六项全部通过（会话 85594，0.68 秒）。
- fork 新增 EngineBufferStore，通过宿主 RHI 创建/释放缓冲区、映射与冲刷，保留 RHI 句柄 generation，不引入第二套显存分配器或 Vulkan 队列。映射状态仅用于将 Taichi 无范围 unmap 接口对接明确的 RHI 范围合同；错误不改走拷贝或 CPU 路径。
- 修复 public_device.h 隐式依赖 fmt 及静态设备 API 错误导出 C++ DLL 符号的问题；格式化器移到已有日志依赖的 device.h，内部静态类型不再 dllexport。CMake 将缓冲区适配组件编入插件目标；独立克隆需显式指定匹配引擎头文件，中英文 README 与 NOTICE 已说明。
- 会话 85594 完成 161 步 Python 扩展、C API、测试及开发 inxpkg 构建。随后两项 Taichi 原生测试通过（0.07 秒），11 项 Python/包集成测试通过（3.44 秒）。完整 Device、pipeline/命令提交适配、preload provider 注册与真实 Taichi GPU kernel 仍未贯通，缓冲区实机测试不能替代数值 kernel 验收。
- 再次确认用户要求：初始化沿用现有 runtime InxPreload；计划 A05 与插件中英文说明已有明确约束，不引入平行启动流程。修改尚未提交/推送，041 保持进行中。

## 2026-09-08：计算管线适配与编译元数据直通

- 前轮属于实际进展，缓冲区适配与包回归已验证。本轮新增 EngineComputePipeline：通过宿主 RHI 创建 shader、绑定布局及计算管线，不调用插件私有 Vulkan，不新建反射解析器或缓存。shader 在管线创建完成后释放；管线与布局的使用期由后续 stream 的完成依赖管理，不在析构中等待设备。
- PipelineSourceDesc 接收编译器已产出的资源绑定信息。CompiledTaichiKernel 从现有 TaskAttributes 填入 Args/ArgPack uniform、其余 storage buffer 及 sampled/storage image；JIT/AOT 共用这条注册路径，元数据序列化格式未改。计数设备合同测试覆盖六项绑定映射，明确不是 GPU 执行测试。
- 新增实机 compute_affine 样本，使用已有 SDK glslc 编译测试 shader；EngineBufferStore 创建宿主参数/结果资源，EngineComputePipeline 创建真实管线，生产 RHI 创建绑定组，VulkanSubmissionExecutor 在 Compute 队列执行 128 元素 i*3+11 并等待完成后回读逐项验证。包含非连续 binding 3/7、不同参数顺序以及不支持的 stage/错误源码长度拒绝。未引入第二个 VkDevice 或队列实现。
- 会话 89446 完成构建，七项聚焦原生测试全部通过（1.08 秒），验证层错误会使测试失败。本样本不是 Taichi Python 编译产物，不据此声称 GPU provider 已贯通。
- 主仓原生 CMake 增加显式维护者选项 INFERNUX_BUILD_TAICHI_ADAPTER_TESTS，链接生产 InfernuxVulkanBackend，提供 buffer_store/compute_pipeline 两个 requires_vulkan_device 测试。当前运行证据来自 dev 聚焦构建；主仓完整构建及跨平台 CI 尚未执行这一选项。中英文 README 说明使用方式及验收边界。
- 会话 61620 完成 156 步 Taichi Python 扩展/C API/测试/开发包构建，2 项原生合同测试通过（0.07 秒）。更新 NOTICE 后重新整理包体，会话 52475 的 11 项 Python/包集成测试通过（3.93 秒）。现有指针转 unsigned long 和 get_err_msg 返回路径编译警告仍待单独处理，没有用忽略警告掩盖。
- 完整 Device/资源集/命令流适配、preload provider 注册、真实 Taichi 数值 kernel、各平台 Player、其他 041 功能仍在范围内且未完成。修改未提交/推送，未发布 GPU 正式制品，041 保持进行中。

## 2026-09-08：Taichi Device 与缓冲区资源集接线

- 前轮属于进展：计算管线与编译绑定元数据已验证。本轮新增 EngineDevice，直接实现 Taichi Device 的分配/释放、映射、管线创建和资源集接口，复用已有 EngineBufferStore/EngineComputePipeline。构造接收宿主 RHI 与宿主集成的 Stream，不创建 VkDevice/队列；wait_idle 仅委托 stream.command_sync，后续必须由真实命令流实现完成依赖。
- 只向编译器声明宿主已启用的 int16/int64/float64 算术能力，采用 Vulkan 1.2 可用的 SPIR-V 1.3 基线；不推断 storage16/atomic64/物理地址/子组能力。EngineResourceSet 将逻辑绑定快照为宿主 RHI bind group，支持整块、范围、重复 binding 替换；拒绝越界、外部设备、已释放资源及超过 RHI 容量的绑定。纹理适配仍缺失，显式报未完成，不让基类 assert/unreachable 伪装成已支持。
- 实机测试从直接调用辅助组件改为通过 Taichi Device/ShaderResourceSet 接口进行分配、映射、管线创建与绑定，再由生产 VulkanSubmissionExecutor 执行同一 128 元素计算。测试中的 ProbeStream 只计数同步委托，不实现/模拟成功提交；真实提交仍在测试宿主端，因此不声称插件命令流或 Taichi Python kernel 已贯通。
- 修正上一轮测试 AllocParams host_write/host_read 顺序写反的问题。参数改用 Upload 写入并 flush，结果改用 Readback invalidate 后读取，不依赖当前 Windows GPU 内存恰好缓存一致。补充跨设备资源拒绝、绑定容量上限/满额替换、释放后无法重新物化绑定组的回归。
- 删除公共 Device 中仅供已停用后端/GUI 使用的跨 CPU/CUDA/Vulkan 分派、未实现的 share_to/host/staging 拷贝入口。直接拷贝严格限定同一宿主设备；核心设备操作不再依赖 Taichi 日志头。RHI CMake 缩为静态设备 API，不再构建 interop_rhi、common_rhi、独立 device DLL 或隐藏下载 MoltenVK 的分支；尚未删除全部旧后端源码，也不宣称完整上游 GUI/后端测试仍可运行。
- 聚焦七项原生测试通过（首次 0.99 秒，完善边界后 1.07 秒）。会话 85409 完成 151 步原生模块、C API、合同测试与开发 inxpkg 构建，2 项合同测试通过（0.08 秒）。Python/包集成 11 项通过（4.21 秒）。更新中英文说明与 NOTICE；未改变插件未完成的发布定位。
- 下一步仍是通用宿主计算提交服务及 Taichi 命令流：复用引擎提交、屏障、完成 ticket 和资源退休，不能以测试端直接执行代替生产接入。随后接 preload provider、真实数值 kernel、纹理/Player 验收，并继续其余 041 功能。修改未提交/推送，目标保持进行中。

### 2026-09-08：宿主计算提交服务与 preload 边界复核

- 新增通用 RHI ComputeQueue / ComputeRecordingContext 及宿主 VulkanComputeQueue，复用现有设备、队列管理和提交执行器，以有界命令槽、fence、完成 ticket 与全局退休 epoch 管理异步任务；仅槽位满时等待对应任务，不建立 Taichi 私有 Vulkan 运行系统。录制失败不提交、不重放，保留原始异常。
- InxVkCoreModular 提供 PrepareComputeQueue，按需准备计算服务并纳入引擎退休/退出顺序。Taichi 生产 Stream 和 Python provider 尚未接通；测试中的 ProbeStream 仍不是可用插件命令流。
- 合并 RenderGraph 与提交执行器重复的 Vulkan stage/access 转换，保留渲染图空阶段为 0、信号量等待空阶段按 ALL_COMMANDS 处理的语义差异；补充合同测试。
- 主构建 dev/engine-build 完成 _Infernux 和两项 Taichi 适配器测试目标。初次集成暴露公共转换函数的 ADL 重名问题，已移除旧重复实现；两项测试最初因构建树 SDL DLL 搜索路径缺失超时，已在 CTest 中从 SDL CMake target 明确加入路径，不修改用户全局 PATH、不增加运行时自修复。修复后两项实机测试通过（0.64 秒）；独立聚焦七项测试通过（1.33 秒）。
- 再次核对用户确认：Taichi 初始化唯一入口为运行时目录中的 InxPreload.preload(context)，清理走 unload()。现有 EditorBootstrap 与 PlayerBootstrap 均使用 PluginManager.startup；沿用先初始化引擎、再预加载插件、再加载场景的顺序。A05 已包含此合同及真实发现链验收，不重复新增启动机制或将尚未实现的 provider 标为完成。
- 后续仍需真实 Taichi CommandList/Stream、宿主适配器绑定与 preload provider 注册、数值 kernel 端到端及 Player 验收；本记录不表示 A04/A05 或整个 041 完成。改动未提交或推送。

### 2026-09-08：Taichi 命令流接入宿主计算队列

- 新增 EngineStream / EngineCommandList，EngineDevice 改为接收宿主 ComputeQueue，并拥有薄命令适配器，不再注入测试用 ProbeStream。填充、拷贝、绑定快照、计算调度与屏障经宿主录制接口执行；完成由宿主 ticket 表示，不新建 GPU 队列、原生分配器、提交线程或完成状态表。
- 同队列依赖沿用提交顺序与内存屏障，不插 CPU 等待；外部队列依赖明确留给宿主资源调度集成，当前不伪装成支持。命令列表只允许提交一次，先完整检查录制结果再提交；noexcept 录制 API 保存原始异常，提交时原样抛出，不提交半份命令、不重放。
- 管线和缓冲区由调用方保持到提交，绑定组是命令列表持有的不可变快照；同步录制完成后，原生资源存活由引擎完成 epoch 保护。实机测试明确连接生产 RHI 退休序号源，覆盖异步提交后立即销毁列表，以及修改原资源集不改变已录制绑定。
- 拷贝与填充采用明确的范围、对齐和重叠合同，不静默裁剪；管线重新绑定后必须重新提供所需描述符，调度不能超过宿主工作组上限。上传/回读原有 noexcept 不符合其分配和提交行为，已移除，使错误正常传播而不是 terminate；零批次不提交，负批次数拒绝。
- 最终聚焦原生七项通过（1.12 秒）；主构建两个实机适配器测试通过（0.63 秒），覆盖 128 元素计算、连续八次提交、填充/拷贝、同步拷贝、上传/回读、无效录制无提交、错误依赖/外部列表/重复提交拒绝及绑定重置。
- 原生编译器与开发 .inxpkg 经 146 步构建成功；包/Python 集成 11 项通过（2.71 秒）。README 中英文和 NOTICE 已同步命令流边界，仍保留 Apache 作者与许可证。编译仍存在既有 CSE 指针截断及 get_err_msg 缺返回路径警告，未掩盖或宣称已解决。
- 另行重建 infernux_runtime_contract_tests / infernux_bit_contract_tests 后，两项合同测试通过（0.07 秒），不是仅运行旧测试二进制。当前开发包 30,330,112 字节，未上传为正式 release。
- 后续关键路径为宿主设备生命周期绑定、preload 注册统一 GPU provider、真实 Python kernel 编译执行，然后纹理/渲染依赖及 Player 验收。当前测试使用预编译 affine shader，不能代替真实 Taichi kernel 验收；A04/A05 和整个 041 均保持未完成。修改未提交或推送。

### 2026-09-08：宿主计算租约与真实 Python kernel

- 新增通用 RHI ComputeHost 租约，由 InxRenderer 返回现有 Device / ComputeQueue；Infernux 仅在图形引擎初始化后允许获取。原生绑定提供私有 _acquire_compute_host，不暴露整数 Vulkan 句柄或第二套设备创建入口。
- 插件新增 _attach_engine_compute_host 工厂，返回已有 EngineDevice 适配器。通过 pybind keep_alive 建立 Program → Device wrapper → ComputeHost → Engine 的保活关系；原生服务仍由引擎拥有。显式 cleanup 在清理场景/回调前要求租约全部释放，不在 GPU 已销毁后尝试自修复。正常卸载顺序仍是 preload 对应 unload 释放计算运行时，再清理引擎。
- 主仓与插件原生构建通过。实机首次启动揭示 copy_dependencies.cmake.in 的 Windows 分支未区分 Ninja 单配置目录，漏收集 SDL/Assimp/Jolt DLL；改为按 CMAKE_CONFIGURATION_TYPES 选择路径后，依赖由主构建直接归集到模块目录，不在测试中额外扫描/回退。
- 新增显式 live_host_kernel.py：真实 Taichi Python kernel 在宿主 Vulkan 上运行 NumPy 输入输出、GPU 常驻 ndarray、连续 affine/increment 调度和整数归约；验证 128 个结果。检查初始化前无法获取设备、活跃租约阻止 cleanup、删除外部 engine 引用后 Program 仍保活宿主、reset 释放后正常退出。
- 新增插件包实机测试：从 .inxpkg 解包到中文目录，在隔离 Python 子进程中运行上述 kernel，使用新构建的引擎和包内编译器。最终插件/Python 12 项通过（10.47 秒）；主仓两个原生适配器测试和 headless 回归共 3 项通过（37.82 秒）。这些证据覆盖 Windows 内部编译入口，不代表跨平台、公开统一装饰器或插件发现链完成。
- 中英文 README、NOTICE 更新真实 kernel 和租约边界；开发 .inxpkg 已重建，仍未上传正式 release。下一步需要把工厂通过 runtime InxPreload 接入统一 compute provider，并验证禁用/重载/失败初始化、CPU 兼容性回退、Player/纹理/渲染依赖及其余 041 功能。A04/A05 和完整目标保持未完成，修改未提交/推送。

### 2026-09-08：runtime preload 与统一 GPU compute 入口

- 在插件 runtime 目录新增 TaichiPreload，沿用 InxPreload 的发现、preload(context) 和 unload()；获取已初始化的引擎计算宿主，注册统一 GPU provider，卸载时完成本后端工作、释放编译器并注销。不新增启动钩子、私有 Vulkan 设备或运行时安装/修复路径。
- 主引擎 compute 将显式 GPU 请求交给已注册 provider；未安装插件仍明确报错。插件支持连续 NumPy 数组、32 位标量、整数/浮点返回类型和按参数类型特化，接入已有 warmup 与 dispatch 跟踪。预热复制参数，不修改游戏中的原始数组。编译错误直接传播，不在 CPU 上重放。
- Windows 实机测试从 .inxpkg 解包到中文项目路径，通过真实 PreloadManager 发现并加载插件，执行数组填充、整数归约、浮点特化、关键字参数与预热；验证非宿主线程拒绝、编译失败不改数据、卸载后旧函数不可调用、重新加载后恢复计算。不是完整 EditorBootstrap 或构建后 Player 验收。
- 插件/Python 集成最终 13 项通过（46.81 秒）；CPU/provider 注册回归此前 13 项通过（3.64 秒）。主仓与插件 git diff --check 通过。测试包解包与清理交给父进程，在加载原生模块的子进程退出后清理，避免 Windows 已加载 pyd 的文件锁。
- 中英文说明与 NOTICE 同步现状。公开 GPU 常驻数组、纹理/渲染依赖、持久缓存、fastmath、设备兼容性 CPU 回退、初始化失败场景及 Player/AOT 仍待完成；不把基础 preload 跑通视为 A05 或整个 041 收口。未提交、推送或发布正式制品。

### 2026-09-08：headless preload 与卸载失败边界

- 上一目标轮完成 runtime preload/provider 接入和实机测试，本轮归类为继续推进，不是停滞重试。沿用真实工作区，不重建分支或替换完整 041 目标。
- GpuBackend 在引擎已声明 Headless 模式下只登记已安装但设备不可用的状态，不导入 Taichi 原生编译器、不创建渲染器或私有 Vulkan 服务。GPU 请求准确说明缺少 headless 设备，不误报为需要安装插件；CPU compute 仍独立使用 Numba。图形路径先获取宿主租约，未初始化的引擎在编译器导入前报错。
- 修正卸载顺序：先关闭新任务入口，再同步已提交工作并 reset；完成等待失败时保留编译器和宿主租约，不允许新 dispatch，不提前销毁宿主设备。成功释放后清空本实例的编译器引用，重复 close 不会 reset 后续实例的运行时。该行为复用已有 preload 卸载失败/重启标记，不新建恢复调度器。
- 新增真实 headless 包集成：中文目录解包、原生 headless 初始化、真实 PreloadManager 两轮加载卸载；确认没有 renderer、没有 taichi 模块导入或原生重启标记，CPU 预热/执行结果正确，GPU 请求明确拒绝，卸载清除 provider。
- 增加六项定向生命周期合同测试：无初始化宿主不导入编译器、部分初始化 reset 且不发布、重复注册只关闭新实例、卸载失败保留注册者、等待失败先阻止新调用且保留资源、既有 compiler Program 不被第二次初始化 reset。此处故障注入是单元测试，不声称已经在真实丢失设备/驱动故障下完成验收。
- 最终插件套件 20 项通过（83.34 秒），包含既有真实 GPU kernel/统一入口与新 headless 测试；此前四项生命周期加 CPU/provider 的组合 17 项通过（3.60 秒）。中英文 README、插件页面、NOTICE 和主计划实施进度同步，保留作者署名与许可证。
- 设备兼容性 CPU 降级仍未完成。下一步需要同一份数值算法的 CPU/GPU 精度、别名、归约和输入输出合同，以及合法 CPU lowering；不能把任意 Numba 可编译认定为等价，也不捕获 GPU 编译/执行错误重放。GPU 常驻资源、纹理/渲染依赖、Player/AOT、完整 Editor/跨平台以及其余 041 项仍继续推进。改动未提交、推送或正式发布。

### 2026-09-08：数组传输、预热别名与 CPU 并行合法性

- 上一轮推进了 headless/preload 生命周期，本轮继续从当前源码检查数值合同。发现上游 external-array 传输将 WRITE 当作整数组覆盖的证明，且按参数独立分配，即使它们指向同一块内存。新增同一算法 CPU/GPU 对照后，旧包实机明确失败：写入前 3 个元素导致其余 125 个从 123456 变成 0。
- 修正 GfxRuntime 的宿主数组传输：初始化输入数据保留未写入元素；相同地址和长度合并为一个宿主 RHI 分配，去重上传与回读，仍使用已有 allocator/stream。访问标签只描述读写，不再用它推断全覆盖。无须新增设备、缓存校验或异常后重跑。
- 不重叠的数组切片保持独立；同一字节范围的 ndarray/view 共用分配。部分重叠范围在录制前明确拒绝，避免被默默转成两块不相关内存；这项限制应由后续 offset-aware GPU 资源绑定解除，不作为最终缩减目标。共享存储也不表示跨迭代依赖在 GPU 上合法。
- 引擎 clone_call_arguments 改为保留重复对象和公共 NumPy 存储根的视图关系，包括负 stride、Fortran 布局、dtype 重解释与只读标记。独立外部存储根但范围重叠、对象数组等不能隔离的输入明确拒绝，不拿改变过语义的副本做预热。仍不宣称覆盖任意 Python 对象图。
- CPU 自动并行的运行签名加入数组别名状态；可能重叠时选择串行，不复用无别名参数的并行决定。parallel_policy='required' 与该输入冲突时在执行前报错。此为循环合法性选择，不是异常后的串行兜底。
- 原生插件增量构建成功并直接生成 .inxpkg。旧包对照失败后，新包部分写入与同数组双参数对照通过；扩展实机测试覆盖同范围 view、相邻不重叠切片、双顺序部分重叠拒绝且输入不变、共享参数预热。插件全套 20 项通过（97.39 秒）；CPU/JIT/预热单测 45 项通过（5.33 秒），包含真实 Numba 原地跨参数依赖。
- README 中英文与 NOTICE 同步边界。共同数值类型/精度/归约、GPU 依赖合法性、合法 CPU lowering 与设备兼容性降级仍未完成；A05 及完整 041 继续进行。其余 GPU 常驻/渲染资源、Player/AOT、跨平台与 Runner Long 功能不从目标中移除。当前未提交/推送或正式发布。

### 2026-09-08：并行依赖诊断与循环上界 IR 引用修正

- 上一目标轮已修正数组传输和别名，本轮继续推进共同数值合同，没有重跑已终止会话或缩减完整目标。HIR 区分归约和归约中间值反馈：total += values[i] 后把 total 写进 output[i] 是扫描，CPU 必须保留串行。补充不同写入间的依赖检查；同一仿射偏移读写和不同余数类（2*i 与 2*i+1）不误判为冲突。
- GPU 声明在编译准备阶段复用 HIR 的跨迭代读写和 reduction_feedback 诊断，错误包含真实源码文件、行号及诊断代码，并公开已有 HIR 结果。它不套用 CPU 全部并行资格规则：嵌套 GPU 循环仍可执行。运行时跨参数别名、嵌套/间接索引的完整依赖证明仍待完成；当前不能据此宣称任意 GPU kernel 安全或可降级 CPU。
- 新增实机测试暴露真实编译缺陷：合法交错写入的 range(values.shape[0] // 2) 在 SPIR-V 阶段仍读到旧 floordiv 节点。补充运算名/类型/源码的原生诊断后定位根因：OffloadedStmt 的 end_stmt 属于任务元数据引用，不是支配任务的输入 operand，StatementUsageReplace 未随操作降级/简化更新该引用。
- 修正统一 IR 使用替换中的整树和向上遍历两条入口，让捕获的循环上界跟随原语句替换；不新增整除实现兜底、不重复运行降级 pass、不拆出额外 GPU 提交。重建插件后原失败用例通过；新增原生合同测试覆盖两种引用替换入口，重建后通过（0.94 秒，CTest 总 0.96 秒）。
- 实机 GPU 回归覆盖已知依赖在声明时拒绝、正常归约、合法嵌套循环与带整除上界的交错写入，以及既有数组/预热/生命周期案例。最终插件 20 项通过（81.79 秒）；CPU/JIT/HIR/游戏构建测试 343 项通过、1 项跳过（19.74 秒）。跳过不计作跨平台验收。主仓与插件 diff --check 通过。
- 本次编译失败的未处理异常还显示：Python traceback 持有 Program 时，宿主租约会阻止引擎提前 cleanup。现有保护避免销毁被引用的设备，但异常保留与显式 finalize/宿主租约释放仍需进一步生命周期验收；没有删除保护来掩盖问题。
- README 中英文、NOTICE 和主计划实施进度同步；开发 inxpkg 重建，仍未提交、推送或正式发布。A05、其余 A00–A15 及完整 041 目标保持进行中，下一步继续共同数值/兼容 CPU lowering、GPU 常驻资源及 Player/AOT 等实际缺口。

### 2026-09-08：按用户确认统一 NumPy/标量数据边界

- 撤回本地刚新增的公开 GpuArray、ResidentArray、jit.create_array 及手动 upload/readback 生命周期要求，删除对应包装实现。计算输入输出统一为数值 NumPy 数组或标量，无输出的原地 kernel 可返回 None；不把后端资源类型交给用户管理。现有原生 RHI 缓冲区、提交和内部计算能力保留。
- CPU compute 在公开边界拒绝容器、对象数组和非数值对象，沿用原有自动串并行与预热机制；GPU 延续 NumPy 输出参数写回，补充 NumPy 数值标量按 dtype 特化。不承诺 GPU 已支持直接创建并返回新 ndarray。
- 明确同步调用返回前写回数组、Python 在调用间的修改对下一次 GPU 计算可见。普通 ndarray 没有自动脏跟踪，因此不能只凭对象身份复用旧显存内容；内部批处理/常驻优化不能破坏此语义。沿用 InxPreload 初始化、注册与卸载，不另造启动机制。
- 真实插件回归新增连续 8 次 Python 修改/ GPU 计算/即时 NumPy 读取，保留标量返回、部分写入、共享数组、预热隔离、卸载重载等用例。CPU/JIT/HIR/构建测试 347 通过、1 跳过（18.22 秒）；签名反射移至声明阶段后，定向测试 19 通过（4.10 秒）。主仓与插件 diff --check 通过。
- 041 计划、插件中英文 README 和内置介绍页同步调整；开发 inxpkg 已重建。最终插件整套 20 项通过（67.69 秒），包括真实 GPU 的 NumPy 标量用例与连续写回。未提交、推送或正式发布，不宣称完整 041、Player/AOT、跨平台、兼容性 CPU lowering 或渲染直连已完成。

### 2026-09-08：修正计算任务 IR 克隆主路径

- 上一轮属于实际进展：NumPy/标量边界已修改且有 CPU/GPU 回归。本轮重新核对完整 A00—A15 与 A01 语义附录，完整目标仍未完成；未缩减到 Taichi 子任务，也未标记收口。
- 原生最小用例证明 OffloadedStmt 克隆丢失 end_stmt：未改实现时 CTest 在动态上界指向副本的断言失败（2.96 秒）。源码还显示 IRCloner 在两阶段遍历中把已克隆的 tls/bls/mesh 前置/收尾块重置为空块，会破坏遍历和映射。
- 修复 OffloadedStmt::clone 保留范围提示、上界、返回类型和源码诊断；IRCloner 只遍历已复制块，复用原有 operand_map 重映射捕获上界，不新建克隆注册表、恢复流程或 hash 校验。保留现有外部引用语义。
- 原生合同覆盖动态上界、源码提示、五种辅助块的内容/父节点、跨块操作数、LoopIndex 所属任务、重复克隆以及原始 IR 释放后副本仍有效。扩展后原生测试通过（0.04 秒，CTest 总 0.05 秒）。这些是 IR 合同证据，不是已实现 AOT 导出的证据。
- CMake 已将新 native 载荷直接输出到插件 package 并重建开发 inxpkg（30,455,616 bytes）。新包整套插件测试 20 通过（63.37 秒），包含真实 Windows GPU、NumPy 读写、预热、InxPreload 生命周期和 headless 无 GPU 副作用。主仓及子仓 diff --check 通过。
- 主计划、插件中英文 README、随包 NOTICE 同步。未提交/推送/正式发布。A01 权威字段与 native/Python 发布仍是完整计划的重要前置：现有 FieldSchema 只是不可变值描述，get_serialized_fields、CDS 注册、属性事务和操作目录仍需统一，不能把当前局部类型当作完成的语义目录。其余 A02—A15、Runner Long、跨平台和发行验收全部保留。

### 2026-09-08：统一数据字段声明，并分离 HPC/NN 场景入口

- 上轮 IR 克隆修复有实际源码和原生/实机证据，属于进展。本轮继续完整 041 的 A01；用户中途进一步确认计算命名空间，作为新增要求一并实现，没有舍弃原目标。
- 先用对照测试复现 SerializableObject 相对 InxComponent 的三处不一致：漏纯注解/显式私有字段与 Annotated 标记、继承字段不执行范围约束、错误类型声明覆盖已有注册。旧实现 3 失败、1 通过。
- 把组件声明解析提取为共享 _compile_serialized_fields，删除 SerializableObject 重复解析；保留组件/CDS 与普通数据对象各自的存储方式。运行时规范化及原始引用读取共用 get_serialized_fields，避免继承引用先被解析成场景对象而丢掉保存身份。数据类型只在声明解析成功后注册，不添加回退解析器。
- 扩展测试又发现嵌套数据类注解未被解析；在同一 resolve_annotation 中补齐 SerializableObject 及列表元素类型/枚举元数据。覆盖注解字符串、嵌套对象/列表、默认值隔离、引用与私有字段的 deepcopy/保存往返。字段、CDS、候选发布、脚本加载、value codec 与编辑事务等 307 项通过（9.91 秒）。完整 native/Python catalog、CDS 继承布局与 owner 事务仍待实现，A01 未勾选完成。
- 按用户确认新增 compute 模块命名空间：公开装饰器迁为 inx.compute.hpc，原数值 HIR、CPU 串并行、GPU provider、预热/跟踪及 NumPy/标量边界不变；删除未发布的 compute(...) 与 jit.compute 作者入口。compute.nn 仅明确抛出 NotImplementedError，不加载 Torch 或借用 HPC 分析冒充 NN。具体 NN AST/模型执行属于后续独立实现，不自动扩大本期范围。
- Cook 通过明确引擎导入识别 HPC，包括模块/直接函数导入别名；NN 和第三方同名装饰器不进入 HPC CPU lowering。新增无源码 cooked 执行用例。CPU/JIT/HIR/构建测试 355 通过、1 跳过（15.20 秒），并非跨平台验收。
- 更新运行时声明/stub、插件示例与中英文 README/介绍页、041 决议及 042 消费边界。CMake 重建开发 inxpkg 后，插件整套 20 项通过（59.01 秒），包含新 HPC 入口的真实 Windows GPU、NumPy 写回与 InxPreload/headless 生命周期。主仓、子仓 diff --check 通过；未提交、推送或正式发布，完整 041 保持进行中。

### 2026-09-08：继承字段的原生布局与重载声明归属

- 上轮核对 HPC/NN 的 27 项入口测试已通过；本轮继续 A01，未收窄完整 041。先前两个原生复现分别证明子类 CDS 缺父类字段、同 GUID/布局重注册后新描述符未绑定（新实例原生默认值为 0 而不是 3）。
- CDS 数值筛选改用统一继承字段视图，布局标识包含父类字段；每个具体组件持有独立描述符绑定，共享声明元数据，父类和兄弟类的原生存储不被子类注册覆盖。同布局复用也完成新描述符绑定。原生包装类显式关闭 CDS 时不创建重复布局。
- 补充真实原生候选发布/撤销、多层继承、数值类型覆写、批量读写及序列化测试。稳定父类下子类重载又复现 __set_name__ 二次注册污染声明归属；发布直接采用已编译字段声明，不重新执行这一注册步骤，保留其他用户描述符的 Python 绑定语义。
- 删除重载中的重复原生槽位复制：先前复制结果随后被完整语义快照覆盖。现在分配候选槽位后只写一次已准备好的重命名、默认值和转换结果，删除无消费者的 Python migrate_slot 包装及未读取的分配列表；原生迁移 API 保留。千实例迁移测试明确禁止重复调用原生复制。
- 字段、组件、CDS、脚本重载及运行时 dispatch 扩展回归：447 通过（简化前 13.83 秒，简化后 14.54 秒；不是性能比较）。测试走 dev/engine-build 原生模块和实际 Vulkan 初始化，Play 管理器的场景/资产依赖使用 fixture，不冒称完整桌面游戏验收。主仓、子仓 diff --check 通过。
- 发现另一未解决边界：两个测试先后使用相同 type GUID，先发布新增字段布局，再从旧布局迁移回这个已存在目标布局时，事务报 CDS layout reuse disagrees with live class ID。继承声明测试现使用各自独立 GUID，以分别验证 finalize/rollback；此布局复用问题保留在主计划，不能据 447 项通过声称任意反复改 schema 已支持。父子同时更新、完整原生语义目录、MCP/Player 和 041 其余阶段仍待推进。未提交、推送、重建插件或发布制品。

### 2026-09-08：迁移到已发布布局的隔离存储代际

- 上轮源码修复、447 项回归和新边界发现属于实际进展；本轮继续 A01，不缩减完整 041。先补原生同名称重新准备的合同，旧实现 CTest 因不允许 prepared class name 已发布而失败。首次构建缺 MSVC include 环境，随后在 conda infernux 下显式加载现有 VS Developer PowerShell；没有安装 SDK 或修改引擎运行路径。
- PrepareClass 支持为已发布名称准备独立 ClassStorage；seal 只构建候选名称映射，commit 仍使用新 class ID/候选存储并退休被替换代际，rollback 恢复旧映射和旧代际状态。普通无 @ 的名称也按相同名称替换规则退休。未向当前可见存储分配半完成迁移实例，不增加失败后旁路写入。
- Python owner 根据实际活实例的数值布局迁移需要声明 private_layout_types；目标布局已存在时也走原有候选分配、字段写入、seal/commit，而非直接复用不匹配槽位。数值布局未变的普通方法重载仍不重建原生数据。
- 原生 CDS 测试明确检查 prepare/seal 不可见、旧值和 alive count 不变、commit 后新旧值同时存在、rollback 恢复旧映射及可分配性、finalize 保留旧槽位到显式释放；修复后 1 项通过（0.01 秒，CTest 总 0.02 秒）。已重建 InfernuxRuntime.dll 和 _Infernux 原生绑定供 Python 实测。
- 完整 Python 重载 fixture 显式预先发布目标布局并保留值 71，分别验证提交和撤销均不覆盖它；迁移实例保留值 23，父类保留值 11，声明归属不变。另加同一实例连续八轮字段增删、返回旧布局、每轮继续数值读写。定向继承测试 4 通过；组件/字段/序列化/重载/dispatch 扩展回归 450 通过（13.29 秒）。
- 主仓 diff --check 通过。仅已验证 Windows 本机原生模块与测试 fixture，不是完整编辑器/MCP/Runner Long/Player/跨平台验收。完整原生/Python semantic catalog、父子同批重载及元数据约束变更规范化仍未完成；041 目标保持活动。未提交、推送或发布插件/引擎制品。

### 2026-09-08：候选字段约束与原生值发布一致

- 上轮原生隔离代际、450 项扩展回归属于实际进展。本轮继续 A01；整数/浮点最小测试先复现 range 从 0–10 收紧到 0–5 时，候选迁移保留值 8 而普通赋值会得到 5，两项均失败。
- prepare_instance_values 使用现有 normalize_runtime_field_value 对迁移后的候选值规范化，先完成这一过程再启动原生 schema 事务。新增默认值继续使用共享默认值复制/规范化函数；引用从 raw 描述符读取，不先解析成场景对象。
- 改为对具有活实例的变更 schema 使用私有数值存储，不再只按布局 hash 判断是否需要迁移。范围、只读等元数据的变化不等价于“可直接复用旧值”；纯方法体重载仍不进入 schema 迁移。相同物理布局替换时也持有旧 epoch 的存储租约，同时保留新代际所需注册键。
- 删除准备阶段重复 schema 签名扫描与两次 _class_key 计算，以第一次确认的迁移集合驱动后续流程，不增加回退尝试。非法候选范围的测试禁止调用原生 begin，证明失败发生在存储/类体变更前，旧字段/槽位保持不变。
- 真实 Play 重载 fixture 验证值 8 在 commit 后成为 5、rollback 恢复精确旧槽位和值、finalize 后旧 epoch 被持有时仍可读取旧值、后续 Python 赋值仍遵守新范围。整数/浮点预备值测试通过；组件/字段/序列化/重载/dispatch 扩展回归最终 455 通过（14.02 秒）；原生 CDS 合同 1 通过（0.01 秒，CTest 总 0.02 秒）。主仓与子仓 diff --check 通过。
- 未新增运行期校验框架或备用解析链；未宣称完整 canonical catalog、所有字段的约束与只读一致性、父子同批迁移、DataAsset/MCP/Player、Runner Long 及跨平台已完成。未提交、推送或发布制品，完整 041 保持活动。

### 2026-09-08：声明投影进入实际 Python 属性事务

- 上轮约束规范化及 455 项回归属于实际进展。本轮回到 A01/S1 的声明目录前置；源码确认 InxTypeRegistry 只负责资源 any 值/文本类型转换，RuntimeTypeRegistry 只负责 Player 脚本身份/生命周期，均不是完整组件字段语义目录。新增 dev/041-semantic-catalog-migration-map.md 记录已核实来源、消费者与切换条件。
- 新增 field_schema_compiler，将 FieldMetadata 投影到现有 FieldSchema 的不可变属性，不再让 Python 属性事务只生成路径/类型/readonly 三个字段。保留默认值、范围、展示信息、旧名、枚举成员、嵌套数据标识及引用类型限制。默认文档沿用 VALUE_CODECS；回调留在声明侧，不编码或执行。编译失败提供 code/path，UNKNOWN、未解析枚举/元素/嵌套类型和非法默认值明确拒绝。
- get_field_schema 使用按类/字段的投影缓存，和既有元数据缓存在重载发布/撤销时一起失效。Python 属性事务直接持有这个对象，测试验证对象身份相同；多对象 readonly 仍聚合。当前是按需投影缓存，不冒称已在类注册前完成全类编译；后续由同一 native/Python catalog 替代该投影缓存。
- 对照发现实际组件默认初始化直接写 CDS，而描述默认值已经过范围规范化。将共享 copy_serialized_field_default 接入既有 normalize_runtime_field_value，真实 CDS 与纯 Python 存储的默认值现在都与描述一致。没有新增 codec 或另一套数值约束规则。
- 测试覆盖不可变/文档往返、回调不执行、明确错误码、普通值/向量/颜色/引用/枚举/嵌套对象/列表/曲线/渐变/通用资产默认值、缓存失效、重载 range 投影更新及撤销。组件/字段/重载/dispatch 回归 482 通过（14.54 秒）；加入 Inspector/value codec 后 679 通过（16.60 秒）。单独无组件实例扫描现有引擎 Python 目录 137 个字段、0 编译失败，随后加入永久目录回归；最终定向 59 项通过（2.97 秒）。主仓/子仓 diff --check 通过。
- 仍缺完整 type/owner/version/field identity、原生不可变目录与并发读取、同一发布事务、DataAsset/MCP/Player、继承树同批更新及 Runner Long/跨平台。未提交、推送或发布制品。完整 041 目标保持活动。
- 加入永久引擎字段目录回归后，最终扩大测试集 680 通过（16.85 秒），主仓 diff --check 再次通过。

### 2026-09-08：原生语义快照与 Python 字段文档贯通

- 重新核对实际工作区及原生模块，确认 SemanticTypeRegistry 和桌面内部绑定已编译可加载；没有依据旧构建状态重复启动任务。本轮继续完整 041 的 A01，不将底座或单平台测试视作整体完成。
- C++ SemanticTypeRegistry 持有不可变值快照，Prepare 成批替换 owner 类型集合但不改变当前目录，Publish 在最终提交边界递增 revision；空类型集合退休 owner。旧快照继续拥有原描述，过期候选不重放，禁止跨 owner 抢占 GUID，候选继承关系需完整且无环。尚未接到正式 ComponentFactory 或 CDS/dispatch 事务。
- 新增真实原生绑定回归。第一轮两个正常发布/旧快照用例通过，八个输入边界用例失败：其中存在负数/浮点版本号被整数转换、空对象被当作空列表的实际错误；另有错误消息不一致。绑定在 schema_version 窄化前检查整数与范围，并要求 fields/types 为数组，复用原有 JsonPyBridge，不新增转换器、自动重试或备用解析链。
- 补充 C++ 继承环、缺失父类型、owner 不匹配测试，消除测试忽略 nodiscard 返回值的警告。使用现有 conda infernux + VS Developer PowerShell 重建 _Infernux 和原生测试，没有安装新 SDK。
- Python 测试验证编译后的默认值/范围能原样进入候选、输入及导出文档修改不污染快照、发布前不可见、退休后旧快照仍可读、过期发布不更改当前状态。现有引擎 Python 组件字段额外通过原生目录往返，测试使用独立测试 owner/身份，不冒充引擎正式注册。
- 定向字段/目录测试 50 通过（2.66 秒）；随后加入整数边界与引擎字段批量往返，组件/字段/重载/Inspector/codec 扩大回归 693 通过（16.66 秒）。原生 semantic_type_registry 与 component_data_store 两个 CTest 通过（总 0.03 秒）。这些是 Windows 本机模块和 fixture 证据，不是编辑器/MCP/Runner Long/Player/跨平台验收。
- 更新目录迁移映射和 A01 详细合同状态。仍需正式 native 字段声明、完整字段合同、统一类编译与发布事件、CDS/dispatch owner 同批发布、DataAsset/MCP/Player，以及 041 其余阶段。未提交、推送或发布制品，完整目标保持活动。

### 2026-09-08：Transform 正式声明进入现有编辑服务

- 上轮目录绑定修复和 693 项回归属于进展。本轮继续 A01 原生生产者/消费者接入，源码确认 ComponentFactory 已统一 creator、document validator 和约束，Transform Inspector 已走 ComponentCommandService.set_field，但后者仅生成 Any 简化 schema。
- 工厂增加静态语义声明回调，在静态注册完成后的模块导入/引擎构造时一次发布 engine:native 类型集合。Transform 显式声明局部位置、欧拉旋转、缩放的 VEC3、默认值、非空、存储方式及序列化字段映射；不实例化组件来探测声明，不从当前值推导类型。其余工厂注册尚无声明，仍是待迁移状态。
- get_native_field_schema 从当前原生快照读取不可变 FieldSchema。既有通用属性事务允许持有该描述，组件服务的三个局部 Transform 属性沿 VALUE_CODECS 规范化后进入同一 Undo 命令。Inspector 调用入口未另造私有流程。其余 native/generic 路径及原生直接 setter 尚未统一，记录为后续工作。
- 已使用真实 Scene/Transform 验证默认值与声明相符，修改实例不改变描述；编辑、一次 Undo、Redo 及短向量/NaN/null/字符串拒绝均不污染对象和历史。首轮三个失败是测试错用 ValueError/TypeError，实际组件服务按既有 commit_or_raise 合同抛 RuntimeError，修正测试期望而未增加异常转译层。
- 扩大组件/字段/重载/Inspector/codec 回归 697 通过（16.14 秒）。补场景持久化断言时发现测试错误地把 intrinsic Transform 当成 components 数组成员；源码与实际文档都使用独立 transform 字段，修正测试验证旧结构原样保留，不改变格式迁就测试。最终目录/场景集成/组件服务/属性事务 222 通过（4.31 秒），两个原生 CTest 通过（总 0.03 秒）。
- Windows _Infernux/InfernuxRuntime 已重建。Web 的 core 源码已由现有递归清单收集，补共同 BindingModule 调用与 Web 专用绑定源码清单；未重建 Web 制品，不宣称浏览器验收通过。尚缺其余原生字段、完整类型合同、CDS/dispatch owner 同批发布及完整 041 其余交付。未提交、推送或发布制品。

### 2026-09-08：Camera 声明、包装层与共享枚举编码

- 上轮 Transform 实际接入和回归属于进展。本轮核对 Camera 原生/包装层：原生远裁剪面默认值为 5000，Python 元数据却声明 1000；原生 CameraProjection/CameraClearFlags 不能被共享 VALUE_CODECS 编码，两项最小测试先失败。
- Camera 在同一 ComponentFactory 声明十二个持久化属性及原有序列化字段映射，包含类型、默认值、枚举、显示文案及已有 FOV Inspector 范围。新增 CppProperty.from_native，在类定义时投影到现有 Inspector metadata；删除 Camera 包装层中重复的字段类型、默认值、范围和 i18n 文案。可见性回调及 COLOR 转换留在执行绑定侧，不放入目录。真实原生构造默认值和场景格式未修改。
- 持有 canonical schema 的 CppProperty 通过原组件编辑服务将同一个 FieldSchema 交给属性事务，数值与枚举/颜色用 VALUE_CODECS 编解码，范围沿既有 normalize_runtime_field_value；没有另造 UI 命令、备用解析链或安装修复。未迁移的属性仍是后续工作，不将过渡状态宣称完整统一。
- VALUE_CODECS 将注册在当前原生模块中的枚举编码为现有 enum 文档，未声明枚举成员拒绝。源码确认 Web 使用 _Infernux 导入再映射完整包名，因此原生枚举身份改用 native:infernux.*，不依赖 DLL/Python 模块导入名；CppProperty 执行绑定只查当前原生模块。Python 字段使用同一原生枚举时，field_schema_compiler 也生成相同身份/成员/default。
- Windows 原生模块已重建。测试核对所有 Camera 声明默认值与真实 native/场景文档、包装层 metadata 相符；六种真实包装属性编辑覆盖 FOV 规范化、far_clip、投影/清屏枚举、背景颜色和 dithering，均同一描述、一次 Undo、Redo，旧格式不变。另验证原生模块名变化不改变枚举编码。
- 定向目录/codec 52 通过（3.01 秒）；加入场景集成的扩大回归先 885 通过（16.22 秒），加入导入名测试 886 通过（17.52 秒），最终补 Python/native 枚举声明对照后 887 通过（16.99 秒）。两个原生 CTest 通过（总 0.08 秒），主仓、Taichi/Web 子仓 diff --check 通过。
- 仍缺 Camera 原生直接 setter 的完整约束、near/far 跨字段规则统一、raw native/包装层全入口、多对象编辑和 Cook；其余 native 类型、DataAsset、CDS/dispatch owner 发布及完整 041 的其它阶段未完成。本轮未提交、推送或发布，未把 Windows fixture/源代码检查当成 Web/Android/Linux 或 Runner Long 实机验收。

### 2026-09-08：Camera 原生约束与多对象预检

- 上轮 Camera 实际目录消费和 887 项回归属于进展。本轮核对直接 setter、文档载入和 Inspector 多选。最初用 NaN 构造 vec4 的测试在收集时已被既有向量边界拒绝，移除此错误测试目标；随后发现 add_component 返回包装层，改为明确取得真实 native 对象。有效基线为 17 失败、1 通过，证明 native setter 接受非法值以及原子裁剪面 API 缺失，不把 fixture 错误当引擎失败。
- 将投影枚举、FOV、正数尺寸、clip planes 和 clear flags 的约束放入共用原生函数，setter 在状态变更前执行，文档验证复用；depth 和背景色要求有限值。保留极窄正数 aspect 的原有 0.01 下限，拒绝非正/非有限输入。新增 native/Python set_clip_planes，一次设置合法的 0 < near < far，避免先改 near 时越过旧 far 的中间非法状态；未新增失败后顺序重试。
- Camera 文档载入改用 SetDepth 通知场景排序失效。测试验证校验本身不改变对象或 scene revision，合法 depth 修改生效，非法整个候选不改变值或排序版本。
- 在原生 Component 绑定公开无副作用 validate_document，直接调用已注册 document validator，复用 JsonPyBridge。已声明 CppProperty 提取共同 normalize_value/validate_value，单选/多选属性事务共享；新增按目标 validator 回调，保证所有候选预检完成再创建命令。完整文档检查只在作者编辑准备阶段，不放到常规帧读取中。
- 多 Camera 测试使用不同 near：新的 far 对第一个合法、对第二个非法。测试禁止构造 SetPropertyCommand，证明不是先写第一项再回滚；对象与历史都不变。合法提交只增加一项 Undo，Undo/Redo 恢复两个对象。另验证包装层拒绝非法 near 后仍可访问并继续修改原生相机。
- 定向 116 通过（4.86 秒）；加入相机、组件/场景集成和 View 服务后的扩大回归 1068 通过（18.70 秒）。本机所有构建/测试均在 conda infernux，原生模块已重建；本轮没有下载安装额外 SDK、提交、推送或发布制品。
- 仍缺其他原生类型、完整跨字段约束的目录投影、一般多字段批量编辑、CDS/dispatch owner 同批发布、DataAsset/MCP/Player 与 041 其余交付。FOV 的 Inspector 范围规范化和 low-level native 合法区间需要在最终公共合同中明确，不把本轮通过视为 T.4 或完整平台验收完成。

### 2026-09-09：实际编辑器更新、果冻米制与接触修正

- 上轮经用户同意保存场景并重启编辑器，实际 PID 69032 使用 out/build/windows-msvc-release/Release 原生模块、项目中更新后的 Taichi 和 MCP 包；旧 Player 已退出。通过现有 native bounded performance window 暴露 MCP begin/get，不另建 profiler。网关目录由 84 项更新为 86 项；MCP 自动化/server 回归 21 通过（22.86 秒）。先前旧尺寸模型的真实编辑器热态/重新掉落/冲击平均帧时约 3.12/3.15/3.28 ms，P95 约 7.4–7.8 ms，不等于每帧 300 FPS 或与 Unity 同条件验收。
- 用户继续指出下落过快与震荡不自然。源码确认模拟体为 0.18 米、表面和 Gizmo 整体乘十，而重力仍为 9.81；实际对象 Transform.scale 为 1。实际编辑器墙钟 10.038 秒内游戏时钟推进 10.032 秒，time_scale 为 1。未发现倍速调度证据；不能把昂贵层级序列化前后的非同步样本差误当时钟漂移。
- 桌面 Infernux041Lab 的 JellyMaterial/GPUJelly 改为同一米制：保留原先可见的 1.8 米尺寸，取消显示放大，重新计算该尺寸下质量、体积和 inverse rest。初始几何的共享长度常量同时供求解体和表面使用。通过 MCP 将当前作者参数从 1200/120000 Pa 改为 12000/1200000 Pa；阻尼 0.45 改为 0.1423025 /s，当前冲击速度 0.408 改为 1.2902093 m/s。此为有记录的项目参数调整，不是放慢引擎时钟；透明度、9 子步、几何规模和用户 Gizmo 开关保留。
- 保存上一固定步位置，按 Time.time/Time.fixed_time 的余量在表面发布时插值；重置同时重置两个位置端点，求解仍只在 fixed_update。没有用插值结果反馈物理或减少子步。当前仍 CPU skin/回读上传，不能声称 GPU 直供渲染完成。
- 新运动探针先发现 5 秒 CPU/GPU 最大位置差 3.90625 毫米，原定 2 毫米门限失败，没有放宽门限。继续检查发现接触摩擦错误地使用上一位置离地距离，已贴地节点几乎无摩擦；改为使用本次法向位置修正量。最终真实 GPU/CPU 各 250 步、含冲击测试通过，最大差 0.6493 毫米。GPU 前 0.3 秒无阻尼自由落体对半隐式重力参考最大误差 0.1295 毫米，CPU 0.2516 毫米；首次接触均 0.36 秒；GPU 体积比 0.98471–1.00346，无翻转。另验证贴地摩擦阻止微小滑动，以及实际表面发布 alpha=0/.25/.5/.75/1 的一米制位移和单位法线。
- 项目更新在同一 PID 69032 内经资源刷新与 MCP 属性编辑生效，没有再重启编辑器。用户进入 Play 后先保留其播放，随后说明并停止 Play 完成接触修正。独立探针与正式编辑器性能复测串行，避免将探针 GPU 负载混入编辑器结果。仅本机数值与当前编辑器证据，不代表果冻美术质量、真实刚体同场景对照、插件原生二进制热替换、GPU 直供 mesh 或完整 041 验收完成。
- 插件 Python preload 相关回归 16 通过（8.60 秒）；已加载 .pyd/DLL 的实际替换尚未实现。主计划补充此边界，不将普通 Python 刷新与引擎本体原生模块升级混同。未提交、推送或发布新制品，完整 041 保持活动。

### 2026-09-09：已有 HPC 声明跟随插件重新加载

- 继续核对发现 hpc 直接返回旧 provider 的闭包；插件卸载后旧声明仍调用旧实例。JIT/AOT 两个新增用例首先失败，证实不是只有提示文案问题。
- 改为引擎持有稳定 GPU 声明，调用/准备/导出/追踪属性读取使用当前 preload owner；owner 未变不重新绑定，替换后使用新编译或 AOT 绑定。无插件时仍报需要 infernux_taichi，不执行 Python 函数体、不转 CPU、不联网或自动安装；不增加第二套 provider 注册表。
- 构建准备扫描识别稳定声明，保留原有模块/类词法归属过滤。真实包回归直接重用卸载前的 fill 声明，移除重新装饰函数的测试做法；新进程验证真实 Vulkan 执行、卸载、重新 preload 和 AOT 路径。
- conda infernux 下 CMake infernux_package 生成新的本地 .inxpkg；定向 65 项通过，HPC/构建准备/JIT 扩大回归 120 项通过（4.60 秒），真实插件载荷 6 项通过（106.22 秒）。未修改正在运行编辑器的引擎 Python 模块，也未安装该包到当前项目；源码/本地包已更新，不宣称当前老进程已经采用新的声明实现。
- 用户新增内建网格工具及果冻与普通刚体持续交互约 300 FPS 的验收。已核对现有 add_force_at_position 支持 Impulse/Jolt 路径；完整双向 GPU/CPU 耦合与批量物理状态仍未完成。主计划新增 A03.4 和 A04.1 工具边界，非功能完成记录。上次修正尺寸/摩擦后的真实编辑器热态、重新掉落和冲击平均约 3.50/3.51/3.52 ms，P95 8.19/8.29/8.52 ms，进入 Play 247.6 ms；该测试不包含普通刚体交互。

### 2026-09-09：内建法线计算与刚体接触速度底座

- Rigidbody 新增 get_point_velocity(world_point)、get_point_velocities(points, output)，后者接受 caller-owned、连续 float32 (N,3) NumPy 数组，C++ 读取一次物理线速度/角速度/质心后完成整批计算。无逐粒子 Python 调用、无显示 Transform 推导、无额外物理注册表。验证偏移质心、旋转速度、原地输出及输入/输出边界。真实 Jolt 偏心球测试验证 Impulse 线速度/角速度与再次 step 不重复施加；初始测试错误假设已发布刚体的 Impulse 延后生效，经核对 SubmitForceCommand 修正测试为立即生效，没有改变引擎行为去迎合测试。
- set_inline_mesh_data 沿原有网格发布路径新增 normals=None，原生累计面积加权法线并归一化；手动法线保持不变。覆盖共享/分裂顶点、退化/孤立顶点、非法三角形不发布与空网格。桌面 GPUJelly 删除每帧 np.add.at，传 None 使用内建计算。当前仍 CPU skin、CPU 法线、几何回读上传，不宣称 GPU-resident Mesh 完成；切线与完整 Mesh 工具还未完成。
- conda infernux 下先失败用例确认接口缺失，然后实现；物理/网格定向 87 项通过，加入组件/HPC 扩大回归 272 项通过（8.14 秒）。原生 dev 与正式 Windows Release 模块均已构建。首次 dev 编译因 shell 缺 MSVC 标准 include 环境失败，使用现有 VS DevShell 修正环境后成功；没有安装 SDK 或修改工具链。
- MCP 确认原编辑器 edit、dirty=false 后 CloseMainWindow 正常退出，更新引擎原生模块后以 PID 68104 显式重开可见编辑器。项目脚本已用新接口，首次 Play/重新掉落/冲击均执行成功；日志没有 ERROR/Traceback，测试结束 edit、dirty=false。重启是引擎本体原生升级，不是宣称插件脚本需要重启。截图 review/041-native-mesh-normals.png 已生成并检查，场景内网格/透明材质正常呈现。
- 实际 MCP 原生性能窗口各取 240 帧：首次玩法平均 2.478 ms / P95 6.989 ms；热态 2.501 / 7.354；重新掉落 2.353 / 6.758；冲击 2.427 / 7.186。进入 Play 364.24 ms，过渡帧最大 490.26 ms。保留 9 子步、343 粒子、1296 四面体和原始表面规模，未把短窗口当成长时间持续统计。平均对应约 400–425 FPS，但长帧依然存在，且不含普通刚体接触，不视为用户双向交互 300 FPS 验收完成。
- 运动探针的表面验证改为调用真实原生 MeshRenderer（不在测试里伪造自动法线结果）；新 Release 上真实 GPU/CPU 各 250 步含冲击通过，最大位置差仍 0.6493 mm；一米制插值和原生单位法线通过。独立探针在编辑器 Stop 后运行，不混入上述性能测量。
- 完整软体/刚体接触求解、有效质量/惯性批量状态、逐刚体反馈归并、同固定步同步与持续交互场景仍未交付。未提交/推送/发布，不标记 A03.4、A04 或 041 完成。

### 2026-09-09：批量刚体状态与真实 GPU 接触反馈

- 上轮判定为 progress：原生法线与接触点速度接口已实现，桌面场景使用新法线入口，真实 Play 性能和运动回归已产生有效证据。本轮继续 A03.4，不缩减完整 041 目标。
- 新增 Physics.get_rigidbody_states：按输入刚体顺序返回独立 float32 NumPy 数组，包含物理 origin/xyzw/COM/线速度/角速度、世界平移轴逆质量对角项和世界逆惯性。PhysicsWorld 在单个 body 读锁中从实际 Jolt 取值，Rigidbody 复用既有 Collider body 身份。没有新的注册表、不从显示 Transform 推导、不反解一套惯性、不为不存在的物理 body 生成兜底快照。
- 新增 Physics.apply_rigidbody_impulses：接收归并好的 (N,3) 世界线冲量和 COM 角冲量，复用 AddForce/AddTorque 的 Impulse 主路径。类型/形状/有限值及 body 可用性在变更之前检查；零反馈跳过提交以保持休眠，运动学/静态响应遵守 Jolt。立即应用、无 dt 乘法、无自动 step；调用者必须选择正确固定步阶段。这里只实现归并结果的批量入口，未实现多接触归并求解器。
- 新增回归先确认 API 缺失；旋转非均匀 Box 的逆惯性与解析质量公式、冻结平移/旋转轴、偏移质心、实际偏心冲量后的线/角速度一致。覆盖快照所有权、运动学、空批次、缺 Collider body、批次后项无效时前项不受力、零反馈保持休眠、float64 拒绝转换。6 项定向通过；物理/组件/Mesh/HPC 扩大回归 278 项通过（7.44 秒）。所有本机命令在 conda infernux，dev 原生模块重建成功，diff --check 通过。
- Taichi 插件增加真实 .inxpkg 接触桥接回归：从提取包经 preload 启动 GPU，以 NumPy 传入实际球体/旋转 Box 的状态和给定偏心接触，在 GPU 计算粒子与刚体有效质量及无摩擦非弹性接触冲量，再批量反馈真实 Jolt。验证接触后法向相对速度为零、线动量与角动量守恒、能量不增加、刚体同时平移/旋转，以及再次提交分离接触不产生吸引力或残留 GPU 反馈。单项初跑通过，补完能量/分离断言后完整包回归 7 项通过（129.68 秒）。没有用 CPU 结果替代 GPU 执行。
- 该回归每个刚体只有一个外部提供的接触点，没有碰撞检测、摩擦、多接触迭代、完整 XPBD 果冻、长期稳定性或持续交互帧率；不算 A03.4 完成，也没有宣称 300 FPS。GPU 仍使用普通 NumPy batch 同步，GPU 驻留数据和逐刚体紧凑反馈的最终热路径尚待接入。
- 桌面编辑器 PID 68104 保持既有 Release 原生模块，MCP 返回 edit/dirty=false；本轮没有再重启或修改其场景，不能宣称它已加载这些新接口。新代码在独立 dev-native 测试进程验证。未提交、推送、发布；完整 041 保持活动。

### 2026-09-09：只读穿透查询与几何驱动 GPU 接触

- 上轮为 progress：批量物理状态和真实 .inxpkg 的 GPU/Jolt 反馈已实现并通过动量测试。本轮检视当前代码确认只有 overlap 对象列表，没有 compute_penetration；不把 overlap 返回对象当成已获得接触深度。
- 新增 Physics.compute_penetration(collider_a, position_a, rotation_a, collider_b, position_b, rotation_b)，通过既有 Jolt CollisionDispatch 查询，返回 None 或只读 PenetrationResult(direction, distance, point_a, point_b)。direction 为把 A 推离 B 的单位方向，distance 为米，两点为各自表面世界坐标；姿态参数为对象 origin 而不是 COM。内部在 A 原点相对坐标下查询再还原，正确合成 shape COM/center 与预测旋转。
- Box/Sphere/Capsule/Cylinder 复用原有只读形状构建，含 signed scale、center、方向轴和圆角，查询不改 Transform、不注册/移动 body、不同步 broadphase、不使用世界 layer/trigger 过滤，也不夹带 compound 其它成员。MeshCollider 的 CreateJoltShapeRaw 会更新缓存并可能启动 cook，所以当前显式拒绝，不借“查询”副作用启动构建或把不支持返回 None。凸 Mesh 仍属待实现范围，不以此缩减 A03.1。
- 12 项新回归先确认缺失接口，再验证预测分离方向/深度、沿结果移动后分离、远离返回 None、实际世界射线结果与 body count 不变、center/负缩放/预测旋转、禁用 compound 成员、Capsule/Cylinder 三个方向轴、非法姿态和不支持 Mesh 的错误。dev 原生构建成功；加入物理/组件/Mesh/HPC 后 290 项通过（7.32 秒），Python 公开结果类型可导入，diff --check 通过。
- 真实插件 GPU 接触测试改为用禁用的小 SphereCollider 作为纯查询几何，与实际球/旋转箱体求交；将检测出的表面接触点与法线传给 GPU，然后反馈 Jolt。没有手填接触点/法线代替检测。法向球面接触应穿过 COM，测试要求球不凭空自转；偏心箱面接触仍要求产生转矩。动量/能量/分离无吸引力等断言保留，新端到端测试通过（17.31 秒）。此前手给偏心球面法线只是验证任意冲量接口，不等于真实无摩擦球面碰撞，本轮明确纠正验证范围。
- 仍只有每刚体一个接触，检测在 CPU，不是最终 GPU 驻留接触检测；多接触求解、摩擦、完整果冻耦合、固定步同步与 300 FPS 验收没有完成。桌面场景未修改、可见编辑器未重启，新的 query/native 接口仍仅在独立 dev 测试进程使用。未提交、推送或发布，A03.1/A03.4/041 保持未完成。

### 2026-09-09：Sphere 取代旧果冻，修订最终性能与引擎改动边界

- 用户将 Taichi 验收改为内置 Sphere 变形、文件材质和多个 Cube 交互，并进一步明确：真正 Rigidbody 与自定义软体双向交互保持 300 FPS+，增加刚体数量不应大幅退化；core 的改动必须服务整个引擎的访存/资源/调度等通用机制，而不是此场景的专用快速路径。A03.4 已记录以上约束、阶梯测试矩阵和通用机制/项目算法分层。
- 桌面 Infernux041Lab/01_XPBD_Jelly 已移除自建可见网格生成器，直接读取 PrimitiveType.Sphere 的 2619 顶点、5120 三角形与原始 UV/索引，保留拓扑并由球形 XPBD cage 驱动。内部仍是 343 节点、1296 四面体、9 子步。UV seam 保留，显式法线按原始相同位置顶点共享。当前仍是 NumPy 蒙皮、CPU mesh 提交，不是 GPU 直供渲染完成。
- 通过 MCP 创建三个 primitive.cube，保留实际 BoxCollider；使用 JellyMint.mat 和 ContactCubes.mat 文件，renderer 保存 GUID，不再由 GPUJelly.awake 临时 create_lit。MCP 修改材质时遇到新 Lit 文档 property_order 与裁剪 properties 不一致，最终文件保留完整材质语义与 renderStateOverrides。脚本删除 opacity 参数，透明度归材质资产管理。编辑器中用户随后移动了部分 Cube，后续读取其新 Transform/Collider，未恢复为初始摆放。
- 初始静态粗 cage 接触虽短窗口约 331–339 FPS，但实际 Sphere 表面在箱面穿入约 7 cm，不能验收。已把可见 Sphere 的嵌入接触约束并入既有 GPU 八色 cell 求解，直接修正物理节点，不在显示端伪造分离。增加保守 cell 包围盒筛选及两个紧凑 NumPy 表，未减少 Sphere 顶点、子步或实际接触面。最新实际编辑器 R/方向键/Space/重置采样未检出超过 2 cm 的表面顶点穿透，接触样本约有 1.4 cm contact offset；不等于全三角形连续碰撞或长期稳定性通过。
- 更完整表面接触带来明显成本：最新短窗口完整帧 drop/push/squish/reset 平均 4.347/5.887/5.537/4.350 ms（约 230/170/181/230 FPS），P95 14.412/16.887/15.740/14.105 ms。没有达到 300 FPS+，更没有完成真实动态刚体双向耦合与数量扩展性。此前静态/不完整接触的 300+ 数字作废为验收证据。MCP 持续 UI 采集有额外明显开销，测试在完成点击定位后关闭采集，并保留普通编辑器绘制。
- 中间分步编辑造成一次旧 ROUND_CORE 导入失败，随后更新完整依赖并通过正常 asset.refresh 重新发布，不重启编辑器；新 Play 已确认静态 Sphere 中心为 2.1 m，实际新求解执行，不能拿此前旧模块结果替代。MCP 回归结束 Stop、保存，最后 edit/dirty=false。所有本轮场景算法留在项目脚本；未新增果冻专用引擎接口，未提交/推送/发布。
- 前一阶段 JellyContacts 多接触 GPU/Jolt 数值探针的动量、摩擦锥、法向残差检查成功，但进程因 TemporaryDirectory 中加载的 .pyd 未释放而清理失败；整个命令不算成功回归。该探针生命周期问题尚待以父进程管理子进程退出后的目录清理解决，不加 ignore_errors 兜底。

### 2026-09-09：通用驻留存储的选择性 CPU/GPU 交换

- 上一轮为 progress：场景、计划和真实性能/穿透证据已更新。继续根据现状审查公开 compute.batch 与私有 _ResidentState：已有跨步驻留和共享引擎队列，缺少选择性输入/输出传输，不应重建第二套资源管理器。
- 在现有 _ResidentState._transfer 路径增加数组选择，copy_to_host(*arrays) 只下载指定存储，copy_from_host(*arrays) 只发布指定主机更新；空参数保留全量语义。未选中的设备状态和主机镜像均保持原状；完全别名合并为一次传输。对未声明数组/部分视图、非 NumPy、布局变化、只读目标、活动 batch、关闭状态的边界错误在提交前拒绝；无全量内容哈希、自动脏检测或兜底重传。
- 新契约测试记录真实调用参数，证明选择反馈不会夹带大型状态、一次无效批量请求不会先传入有效前缀。一般计算 GPU 回归用 32768 个 int32 状态、1 个 int32 控制参数、4 个 int32 反馈，选择性回读只需 16 字节，控制上传只需 4 字节；CPU 故意将未上传大型状态改为 -777，后续 GPU 结果仍依据正确设备状态。另验证显式重新上传该状态后的结果。
- sourceless AOT / Content.inxpkg 路径验证两个不同 dtype 数组跨 12 步驻留后只回读其中一个，上传另一个并继续原 AOT kernel，最后全量回读得到正确结果。没有把 CPU 计算或手写设备数组代替引擎 hpc 路径。
- 全部命令在 conda infernux。preload+compute 单测 58 通过；修正测试中 NumPy deprecated shape 赋值后 preload 单测 23 通过且无该警告。通过既有 CMake infernux_package 目标重新生成本地 .inxpkg；完整真实插件包回归 7 项通过（109.46 秒），包含 JIT、Jolt 单接触、无图形 preload、源码移除后的 AOT/Content 包加载与卸载。
- 本轮没有重启或替换桌面项目已加载插件，没有把新机制接进特定 Sphere 快速通道；没有公开后端数组类型，也没有改变 compute.batch 退出时同步主机可见的契约。没有承诺零拷贝/异步读回或 300 FPS 已实现。
- 后续证据指向通用原生传输调度：_transfer_numpy 当前调用 program.synchronize，通用 Device::upload_data/readback_data 每次数组传输重新创建 staging，且 submit_synced；需要在既有 RHI 所有权/提交依赖之下解决复用与同步，而非无依据删等待。GPU Mesh 直连、真实多刚体双向耦合、数量扩展性和完整 041 仍未完成。未提交、推送或发布远程制品。

### 2026-09-09：传输前提交而不预先阻塞 CPU，明确 Compute Shader 风格数据边界

- 上一轮主要是用户要求的接口讨论，没有代码交付；本轮重新核对原生队列/内存屏障、Program 的 flush/synchronize 与资源清理，不把对话里的示意 API 当成现成功能。计划纠正旧的“任何模式都不显式 upload/readback”限制：便捷调用自动传输，性能路径可选长期驻留和选择性显式交换，NumPy/标量仍是数值边界，具体公开命名尚未定稿。
- _transfer_numpy 原本先等已有计算完成，再准备 staging/提交同步复制。现在先 program.flush 将尚未提交的计算发到同一引擎 compute stream，继而准备/提交传输；沿用 EngineCommandList 入口 AllCommands 内存屏障和顺序队列，保证之前的读写先于复制。不新增设备、队列、场景判断、自动 dirty 扫描或回退。
- 传输仍是同步合同，复制完成后调用 program.synchronize 清理 profiler、ndarray 使用记录等；此时最新队列 ticket 已完成，VulkanComputeQueue::Wait 的已完成路径不会再调用 vkWaitForFences。不是完全删掉 synchronize，也不是异步回读。没有数据的请求直接返回，不因空传输触发计算提交。staging 仍逐数组分配，复用池和通用 GPU 渲染消费未完成，不能宣称零拷贝或具体 FPS 收益。
- 真实打包原生 GPU 用例新增 capture kernel：32 轮验证尚未显式提交的 kernel 先读取旧数据、随后主机上传覆盖、回读先前结果，再执行增量写入并立即回读。验证顺序和数值；不以它代替性能测量。首轮所有数值检查已通过，但新增测试局部变量保留 Program，触发引擎正确的 compute lease 清理保护；修正测试引用释放后完整重跑，不削弱引擎保护或忽略清理失败。
- 所有命令在 conda infernux。首次原生链接因中文系统 TEMP 路径 LNK1104 失败，沿用已有 dev/native-temp 后同一增量构建成功，无新增 SDK/环境安装。通过既有 infernux_package 生成 .inxpkg（13,705,024 bytes）；完整真实包 GPU/JIT/AOT/Jolt/卸载回归 7 项通过（108.74 秒），preload+compute 单测 58 项通过（6.48 秒），已有 CTest 4 项通过（1.12 秒），Python 编译检查通过。CTest 不是新传输顺序的数值证明，该证明来自新包的 live_host_kernel。
- 本轮未修改或重启桌面编辑器，也未将新包安装进其正在使用的项目；未提交、推送或发布。Sphere 与多刚体双向交互的 300 FPS+、数量扩展性、GPU Mesh 直连、完整 041 均仍未完成。

### 2026-09-09：JIT-only、双 JIT 构建与仓库重定位

- 用户将构建方案纳入调整：引擎 wheel/适用 Player 同时携带 CPU Numba/llvmlite 和 GPU JIT 工具。AOT 导出、特化收集、加载与 C-API 发行暂不进入项目；依赖清单、Nuitka 字节码/受管代码输入、动态库、平台兼容与体积实测写入 B06。未宣称 JIT-only 必然缩小 Player。
- 线上 fork 已改名 ChenlizheMe/taichi_for_infernux；保留上游 fork 关系/署名，更新英文描述、homepage、compiler/jit/spirv/vulkan 等 topics。提交 9beb91c9e 已推送 master，重写中英文 README/NOTICE，新增 JIT preset、禁用 AOT C-API 的策略和原生 wheel 输出合同；移除依赖上游私有服务的 11 个遗留 Actions 工作流，替换为 Windows/Linux 配置验收。该提交未夹带尚未验证的本地原生实验。
- 新工作流 run 34322002984 的 Windows/Linux 两项均成功，仅证明 preset/policy/wheel 输出配置，不是完整编译器或 Player 验收。本地在 infernux 环境及完整 MSVC Developer Shell 下完成新目录 CMake configure/generate；未执行完整 native build。
- 本地子模块文件完整迁至 external/taichi_for_infernux，更新 .gitmodules/remote/登记及 cpp/tests 的源路径，主仓 gitlink 指向新提交。文件系统拒绝整目录改名后采用逐文件迁移；确认旧目录剩余文件数为 0，部分空目录壳保留。主仓旧的空 index.lock 被移动至 .git/index.lock.stale-041-compiler 保留，不删除用户数据。
- 本地 CMake 退役 InfernuxPackage/StagePackage，编译输出指向 wheel staging；既有未提交宿主适配暂留 InfernuxEngineBridge，仍待纯编译输入输出替换。原生 field/Program/AOT 源码和完整双 JIT 导出尚未裁净，不标记交付完成。
- 主仓大量既有改动保留；没有提交/推送主仓，没有构建或发布最终 wheel，也没有恢复所有 041 实施。

### 2026-09-09：用户确认 Taichi 仅保留编译职责并内置 wheel（仅规划）

- 取代上一轮“内置/插件待决策”：裁剪后的 infernux_taichi 确定作为引擎本体 wheel 组成部分，不再交付项目插件或单独 Taichi wheel。编译器输入为数值代码/IR、布局和目标描述，输出 Vulkan SPIR-V 与必要元数据；不拥有活动 VkDevice、buffer、执行或生命周期。
- B05 新增 field/SNode/设备 ndarray、Program 执行 runtime、分配/调度/同步/回读的完整裁剪映射；已有可复用 IR/codegen 与 CPU JIT llvmlite/LLVM 保留，不以删除所有 LLVM 为目标。引擎负责所有资源、AOT 打包加载与执行。
- 加入 external/plugins/infernux_taichi → external/infernux_taichi 的受控子模块迁移、CMake 直接产 wheel、Windows/manylinux 安装、官方 catalog/preload/provider/项目依赖退役，以及无 GPU 编译/无项目插件编译测试。旧项目迁移保留用户内容，不盲删 Packages。
- 同步 041 主计划、决议、语义附录与 042。普通 Player AOT 与显式动态 GPU JIT 的载荷继续分离，编辑器 wheel 的编译模块不是可选联网下载；署名和 NOTICE 随实际分发保留。
- 本轮仅修改 dev 计划与记录，没有移动子模块、修改引擎代码、构建、安装或发布；此前实现仍暂停。前一条“待决策”日志作为历史讨论保留，执行依据为本次确认。

### 2026-09-09：计划改为引擎 Buffer、独立 JIT/Compute（仅规划）

- 按最新确认使用 inx.buffer、set_data/get_data、小写 vector3；CPU @inx.jit.compile 普通调用并自动串并行，GPU @inx.compute.kernel + launch 描述单 work-item。NumPy 改为可选交换，废止旧 hpc/resident 和 GPU 自动 CPU 回退作为最终合同。
- 新增 041-buffer-jit-compute-contract.md，覆盖原生连续存储/布局/生命周期、GPU Mesh 直连、材质/实例/粒子/Gizmo/物理交换的接入与 GPU 化审计、CPU 数据流/类型反馈/优化分层/失效与 V8 对照、Taichi 不使用 field 的深度改造及平台交付。
- Taichi 是否内置列为待决策，比较完整内置、内置执行层加可选编译器、继续插件的成本后确认；本轮没有移动 external 子模块、修改默认安装或构建分发。无论选择何种形态，都保留 LICENSE/NOTICE/原作者。
- 直接替换主计划 A05 的冲突条款，更新 A03/A04/A14/A15/交接及决议文档；042 复用 buffer 与新编译入口，并补动态 owner 的代码/资源失效和在途退休。旧调查与实现日志标为历史证据，不折算新接口完成比例。
- 本轮只编辑 dev 下计划/记录；未改引擎代码、未启动构建或场景、未提交推送。此前暂停的实施仍暂停；300 FPS+ 刚软体真实交互与扩展性验收不变。

### 2026-09-09：公开驻留接口实验，用户要求暂停定型

- 基于既有 _ResidentState 做了 compute.resident / Resident 的工作区实验：dispatch/upload/download/close，GPU 委托原提供者，CPU 直接读写 NumPy。增加线程局部作用域，HPC 在执行前拒绝与作用域不匹配的设备；GPU 数组接纳复用原后端，CPU 检查已声明存储。不引入另一种公开数值数组或另一套 GPU 资源管理器。
- 生命周期退出不隐式回读；插件卸载关闭状态，重新加载不自动拿过时主机镜像重建原状态。实验接口补充 pyi 和中英文插件 README。GPU/JIT 选择性传输与 sourceless AOT 测试改为经过公开入口，覆盖未上传主机污染不改变设备值、退出作用域主机仍旧、显式回读、设备不匹配拒绝、关闭不回读、卸载后旧状态拒绝与新状态正常。headless CPU 同入口不加载 Taichi。
- 验证：compute+preload 61 项通过（6.05 秒），JIT/runtime/build preparation 40 项通过（6.26 秒），完整真实插件包 GPU/JIT/AOT/Jolt/生命周期 7 项通过（113.88 秒）。未重新构建原生，因为复用上一轮 native .inxpkg；本轮 Python 引擎源码配合该包运行。未替换桌面编辑器，未提交/推送/发布。
- 用户随后明确认为 state.upload/with state.dispatch/state.download 写法笨重，要求先看 Unity Compute Shader 对应脚本，偏好 Taichi 形式并讨论 compute.cpu / compute.gpu 拆分。即刻停止扩展/推广实验接口；上述代码和 README 暂留本地工作区用于比较，不视为最终 API 或用户批准，后续设计要按讨论修订/替换，不能以测试通过反推这个接口已经被接受。先澄清“NumPy 对外交换”与“作者可见持久化 field”是否允许共存，不自行定型新的字段对象或装饰器。完整 041 仍活动。

### 2026-09-09：恢复实施，主仓 Windows 双 JIT wheel 首个闭环

- 重新统计主计划：严格端到端 checkbox 为 2/235（0.9%）；该数字不反映已完成但尚未覆盖全平台/全场景的局部能力。按当前代码和证据评估，工程实现约 28%，最终可验收约 13%。单人全职余量约 10–14 周；包含四平台、Runner Long 迁移和 300 FPS 性能波动的保守日历工期为 14–20 周。
- 主仓新增 `InfernuxGpuJit.cmake`：以隔离子构建消费 `external/taichi_for_infernux`，避免上游全局 CMake 状态污染主构建；`stage_python_package` 显式依赖该编译器，安装阶段直接进入 `PythonWheel`，没有 release.py 或手工移动。
- Windows Release 主配置成功；首次完整 GPU 编译器构建成功，原生模块 12,705,792 bytes。随后完整 `package_python` 成功，产生 25,445,882-byte wheel；`verify_python_wheel.cmake` 强制核对单一 GPU 编译模块、私有 Python→IR 前端、Apache LICENSE/NOTICE 及 CPU Numba/llvmlite 声明。
- 从完整 staging tree 独立导入 `Infernux._compiler.taichi` 成功，加载到 `taichi._lib.core.taichi_python`。wheel 内暂含 92 个私有编译前端 Python 文件，已排除 examples/ui/shaders；它们仍含 field/SNode/AOT 等遗留，且 C++ 链接仍构建 Program/gfx runtime/AOT loader。此结果仅证明发行主链，不代表纯编译职责或 GPU 执行验收完成。
- 用户再次冻结最终 Taichi 验收：Infernux041Lab 中引擎自带 Rigidbody 与自定义软体真实双向作用；刚体被软体反冲，软体受刚体接触影响；按 1/8/32/128 数量矩阵避免远处刚体导致全遍历，真实编辑器完整帧约 300 FPS。优化优先进入通用 RHI compute、inx.buffer、批量刚体快照/反馈和空间候选机制，不为演示项目增加特判。

### 2026-09-09：引擎自有 inx.buffer 的首个真实 Vulkan 闭环

- 新增通用 `rhi::ComputeBuffer`：权威存储为引擎 DeviceLocal buffer，usage 同时覆盖 Storage/Vertex/Transfer，Upload/Readback 仅是传输 staging；创建、CopyBuffer、barrier、等待与释放全部复用 RhiDevice/ComputeQueue/BufferResource，没有 Taichi field/ndarray、第二 Vulkan core、场景名称或软体常量。
- pybind 私有面仅给 ComputeHost 增加 create_buffer 及字节传输，不公开 VkBuffer。Python 增加最终合同方向的 `inx.buffer(shape=..., dtype=..., device=...)`、CPU/GPU Buffer、标量/vector2/3/4、set_data/get_data、CPU 索引、NumPy 可选互操作和显式关闭。GPU 普通索引会拒绝，不隐式回读；get_data 返回可复用 CPU Buffer 快照。
- Windows `_Infernux` Release 增量编译通过；真实 64×64 Vulkan Engine 中用 12 个 vector3 进行初始上传/回读、覆盖上传/复用输出回读，数值完全一致并正常清理。compute 单测 41 项通过，包含模拟引擎存储、布局拒绝和无隐式设备猜测。
- 这只是 B01 全量同步纵向切片：区间传输、staging 复用/异步依赖、资源退休、kernel 参数绑定、Mesh 直连、物理交换和 Linux/Player 尚未完成。旧 Resident/hpc 测试暂时仍存在，待 kernel/launch 主链能替代后一次删除，不将双栈长期保留为 fallback。
- 用户进一步要求大规模裁剪/魔改 fork：Taichi 只做 Infernux Python kernel/HIR 到 Vulkan SPIR-V；Python ndarray/field/SNode/AOT/AD/profiler/UI、独立 Program/device/queue/runtime 及其导出/链接闭包全部属于退役范围。当前 92 个前端文件和 12.7 MB 原生模块仍明显超出目标，不计作深裁完成。

### 2026-09-09：引擎 RHI 直接执行驻留 Buffer 的 SPIR-V kernel

- 核对桌面 Infernux041Lab 后确认其 GPUJelly 仍在每个 fixed_update 收集项目字典中几乎全部 NumPy 数组，再由旧插件 `compute.batch` 建 Taichi ndarray、Program、graph/AOT 并整批上传/回读；表面蒙皮/法线/网格也仍回到 CPU。该结构是启动约 5 FPS、热态仍明显低于 Unity Compute Shader 的架构原因，不继续靠减少项目粒子、子步或碰撞体掩盖。
- 新增通用 `rhi::ComputeKernel`：接收 SPIR-V、storage-buffer binding 数及 push-constant 大小，经现有 RhiDevice 创建 layout/pipeline，经 ComputeQueue 异步 dispatch；计算前后使用 RHI barrier。稳定 resident buffer 组合只建立一个 bind group，重复 dispatch 不逐次分配；绑定组合变化是明确 Wait/退休边界。在途提交保留 buffer，kernel 销毁等待并释放；没有原始 VkDevice、Taichi runtime 或场景特判。
- 私有 pybind 暴露 ComputeHost.create_kernel 及 dispatch/collect/wait，只作为 Infernux compiler/runtime 连接面，不作为作者输入 SPIR-V 的公共 API。真实 Windows Vulkan 测试由现有编译工具生成标量缩放 SPIR-V，直接原地修改引擎 GPU buffer，随后一次显式 get_data 得到正确 257 项结果；证明执行阶段不依赖 Taichi ndarray/Program/graph。
- `inx.buffer` 补齐逻辑元素区间 set_data/get_data、复用 CPU 输出、fill/zero；原生描述固定 float32/int32/uint32、元素数、1—4 lane、元素步长，vector3 为明确 12 字节元素而不是假定 std430 vec3 数组步长。GPU 向量区间上传/回读同一真实 Vulkan 测试通过；compute 单测 42 项通过。
- Jolt 批量状态接口已可直接写入复用 CPU `inx.buffer`，反馈冲量接受同类 buffer；真实 Jolt 集成 7 项通过。它消除了固定步七组 NumPy 分配，但当前仍是 CPU 快照/反馈边界，不是 GPU 宽相或零回读。
- 当前工程实现按垂直切片约 30%，最终验收约 15%；严格计划 checkbox 仍约 0.9%，因为 B01/B03/B05 和场景矩阵都未完整勾选。单人全职余量仍估 10—14 周，含四平台、Runner Long 和 300 FPS 尾延迟的保守日历为 14—20 周。作者 `@compute.kernel/launch/index`、Taichi HIR→SPIR-V、单次提交多 launch、GPU Mesh 直连、真实多刚体接触/空间候选和性能矩阵尚未完成，未提交、推送或发布。

### 2026-09-09：多 kernel 单次 RHI 提交与 CPU JIT 正式入口

- 将多个已编译 compute dispatch 记录进一个引擎 RHI command buffer，并只调用一次 ComputeQueue submit。首个 dispatch 前建立通用写→compute 屏障，各 kernel 之间建立 compute write→read/write 屏障，最后发布给后续引擎消费者；没有新增 Vulkan device/queue、场景分支或 solver 常量。
- 稳定 kernel/buffer 组合继续复用 bind group。同一 batch 中同一 kernel 偷换一套资源绑定会在提交前明确拒绝，因为提前释放首个 bind group 会破坏尚未提交的命令；跨 batch 换绑仍沿已有 Wait/退休边界。一个 ticket 只在每个参与 kernel 登记一次，kernel/buffer 生命周期继续覆盖在途执行。
- 真实 Windows Vulkan 用同一 257 元素 GPU buffer 在单次 batch 内连续执行两次缩放，只在末尾显式 get_data；6.25 倍结果逐项一致并正常释放。`inx.buffer` 构造补充元素数×步长溢出硬错误。Release `_Infernux` 编译成功；compute 单测 42 通过，真实 Vulkan 脚本通过，Jolt 批量状态 7 通过。
- `@inx.jit.compile` 成为 CPU 新入口，默认启用现有 typed-HIR 自动串并行选择并保留 warmup、有限缓存与执行后不回放；没有让 CPU/GPU 共用装饰器，也没有恢复 Numba GPU。JIT/HIR/runtime 定向 79 项通过。旧 njit/hpc/resident 仍待消费者迁移后删除，本轮不把入口改名算作完整 V8 级能力。
- 041Lab 源码复核继续确认现有瓶颈：GPUJelly 每个 fixed_update 仍从项目字典取几乎全部 NumPy 状态并进入旧 batch，表面插值、蒙皮、法线和整网格发布仍在 CPU。下一主链是 GPU buffer 直供动态 Mesh 与 Taichi 无设备纯编译；未修改项目参数来换取帧率，也未提交、推送或发布制品。

### 2026-09-09：Buffer 有序上传与动态 Mesh 代际发布

- `@inx.jit.compile` 补齐 CPU `inx.buffer` 直接调用：包装层把 buffer 的受控连续 NumPy 存储交给既有 Numba/HIR dispatcher，原地返回同一数组时恢复原 Buffer 身份；GPU buffer 在调用前明确拒绝并指向 `inx.compute.launch`，不隐式回读或改走 Python。JIT/HIR/runtime/compute 定向回归 122 项通过。
- GPU ComputeBuffer 的 Upload/Readback staging 改为首次使用时创建，不再让每个只驻留计算的 buffer 默认占用 storage+upload+readback 三份空间。`set_data` 写入 staging 后提交有序复制并立即返回；只有同一 staging 再次被 CPU 覆盖时才等待上一使用，随后 kernel 与同步 `get_data` 依靠同一 ComputeQueue 顺序保证可见性。没有增加后台线程、dirty hash 或失败重试路径。
- 删除动态 Mesh GPU 发布的逐顶点/逐索引 FNV 内容哈希。资产和内建共享网格继续以 GUID+runtime version 共享；无资产运行时网格以 object id+单调发布代际拥有缓存，指针/大小仅保留不变帧快速判断。两份独立动态网格不再因为内容碰巧相等被昂贵扫描后合并；真正共享将在 B02 通过显式 buffer/resource view 表达。
- Windows Release `_Infernux` 重建成功。真实 Vulkan buffer 全量/区间上传、两 kernel 单提交、最终回读通过；动态 Mesh 24 次普通更新、192 次连续替换，以及单/多 Camera 的 192 帧双对象更新与逐对象销毁均通过，最终 pending=0、stale GPU bytes=0、resident 回到基线。物理/mesh 定向 25 项通过。
- 这一步消除了两个确定的 CPU/同步成本，但尚未让计算结果直接成为 Mesh 顶点流：041Lab 仍走旧 Taichi ndarray/Program 和 CPU surface publication，因此不能据此更新 300 FPS 验收状态。当前工程实现仍约 30%，最终验收约 15%；余量保持单人全职 10—14 周、含四平台与性能长尾保守 14—20 周。未提交、推送或发布。

### 2026-09-09：公开 kernel/index/launch 与参数传输合批

- 新增 Infernux 私有 GPU kernel frontend：作者使用 `@inx.compute.kernel` 声明单 work item，以唯一 `inx.compute.index(buffer)` 指定一维执行域，并用 `inx.compute.launch(kernel, params=(...))` 提交；调用方不填 dim/group。特化键包含 Buffer dtype/rank 和标量类别，不包含 Buffer 长度或标量值，避免常见 N/value 造成无界重新编译。GPU kernel 不返回 Python 值、不接受 CPU Buffer、不回退 CPU。
- 编译在 compiler-only Program 中完成，不创建 VkDevice、queue、显存或 Taichi ndarray；产物携带 SPIR-V、任务线程宽度、uniform/storage binding 和参数索引。执行只走 Infernux ComputeHost/RHI。私有绝对导入只在编译锁内短暂使用 `taichi.*` 名称，结束后迁移回 Infernux 私有命名空间；已导入公共 Taichi 时明确拒绝，不覆盖第三方模块。
- 修正多次取得 ComputeHost 时错误按 lease 包装地址判异的问题：同一渲染器以其有序 ComputeQueue 地址作为兼容身份，多 Buffer/Kernel 可以共享同一设备执行，lease 生命周期仍各自保留。真实隔离 wheel 回归通过 int32 标量变换与 vector3 位置/速度双 Buffer 积分，且结束后无顶层 Taichi 模块残留。
- `ComputeBuffer` 增加可由 batch 记录的 prepared upload；kernel arguments uniform 更新与编译器产生的多个 task 进入同一次 ComputeQueue submission。相同参数字节不重复上传；没有扫描大型状态 Buffer。343 个 vector3、固定参数的 1000 次单 kernel launch+最终回读耗时 33.737 ms（约 33.7 微秒/launch）；每次交替标量并更新 uniform 为 63.098 ms（约 63.1 微秒/launch）。这是 64×64 独立 GPU 微基准，不是编辑器完整帧或 300 FPS 验收。
- 私有 Python 分发从约 412 个上游条目裁为 61 个 vendor 条目，删除 examples/UI/AOT/AD/graph/linalg/sparse/tools 等明显无关目录；最终 wheel 为 25,333,545 bytes，`Infernux/_compiler` 压缩后约 4.28 MB。实际导入闭包仍有 field/SNode/ndarray/profiler，原生链接仍有 Program/gfx/AOT 对象，必须继续解耦后删除，不能把隐藏名称算作深裁完成。
- Windows Release native 与 wheel 构建通过；隔离 wheel 真实 Vulkan 回归通过；compute/JIT/HIR/runtime/physics 定向 Python 回归 218 项通过。当前工程实现约 36%，最终 041 验收约 18%，严格端到端 checkbox 仍约 0.9%。单人全职余量仍按 10—14 周，含四平台、Runner Long 和性能长尾的保守日历为 14—20 周。041Lab 仍使用旧插件/hpc/NumPy/CPU Mesh 路径，真实多刚体双向交互、数量矩阵和完整编辑器约 300 FPS 均未通过；未提交、推送或发布。

### 2026-09-09：生命周期合批与 GPU/Jolt 双向数据交换纵向切片

- GPU Buffer 改为同一活动 Engine 共享一个 ComputeHost lease；资源统一在 `compute._release_engine_resources` 后释放该 lease。此前每个 Buffer 各自取得 host，导致同设备的多 buffer 不能组成一次原生批处理；真实测试发现后直接纠正所有权主链，没有增加设备地址猜测或重试 fallback。
- 组件 lifecycle phase 自动记录 `inx.compute.launch` 和 buffer update，阶段末一次提交；显式回读/物理反馈是依赖边界。同一 kernel 可在一个 command buffer 中切换缓存的 bind group，update-only 批次有明确 barrier。1000 launch 的独立微基准从立即提交约 34.9/38.0 微秒降为批量约 19.0/17.7 微秒；该数据不替代编辑器完整帧验收。
- C++ 新增多 ComputeBuffer 单 submission/readback；`Physics.get_rigidbody_states` 与 `get_rigidbody_box_states` 可将完整 GPU `inx.buffer` 输出集由 C++ 直接一次异步上传，`apply_rigidbody_impulses` 将线性/角冲量一次回读等待后直接施加 Jolt。隔离 wheel 的真实 Vulkan→Jolt 回归验证质量 2/4 刚体速度、反向状态上传和 Box body index 均正确，无 Python/NumPy 中间数组。
- 公开 kernel 真实回归扩展到一维 order 域、二维 float buffer 间接索引、局部串行循环和分支。私有 Taichi types 根包移除 quant 导入，wheel 安装规则及审计同步禁止 `types/quant.py`；field/SNode/ndarray/Program 静态闭包仍在，深裁未完成。重新封包审计确认 quant 不在隔离安装中，最新 Windows wheel 为 25,324,438 bytes。
- 通过项目自带 MCP 打开真实桌面 Infernux041Lab/01_XPBD_Jelly 并执行可见编辑器 Play。旧场景热态 240 帧平均 2.927 ms（约 342 FPS），但 p95 8.694 ms；当前三个动态 Cube 尚未接入 GPUJelly 的双向求解，项目仍是旧插件 hpc/NumPy/CPU Mesh，因此该数字只记录迁移前基线，不构成 300 FPS 验收。MCP 捕获确认内置 Sphere、文件材质与三个真实 BoxCollider/Rigidbody 可见；旧的 0.18 秒高度断言与米制下落时间不符而失败，未据此修改材料或放宽验收。
- Windows `_Infernux`、wheel、177 项定向 Python 回归、公开 kernel 真 Vulkan 与 compute-physics 真 Vulkan 回归通过。当前工程实现约 39%，最终 041 验收约 19%；严格主计划仍为 2/235（0.9%）。剩余仍估单人全职 10—14 周，包含四平台、Runner Long 与 300 FPS 尾延迟的保守日历 14—20 周。GPU Mesh 直供、软体窄相/反馈接线、1/8/32/128 矩阵及 Taichi 原生深裁是当前硬阻塞；未提交、推送或发布。

### 2026-09-09：真实编辑器 MCP 复测，确认双向接触与尾延迟尚未收口

- 连接当前可见的桌面编辑器 PID 27748/MCP 9741；`project.info` 确认项目为 `D:\Users\Chenlizhe\Desktop\Infernux041Lab`、活动场景 `01_XPBD_Jelly`、场景未脏且初始为 Edit。层级确认 GPUJelly 为 343 粒子/1296 四面体，三个 Contact Cube 均使用引擎内建 BoxCollider 与 Rigidbody。
- 将真实编辑器帧探针改为即使 Game viewport 暂未出现在语义快照中也继续核心采样，并记录 GPUJelly 与 Contact Cube 的前后状态；若 viewport 可见仍执行 R/Space 实际输入。没有修改项目参数、材料、粒子数或求解子步。
- 本次 Enter Play 状态机总耗时 60.36 ms。首个 5 秒窗口 240 帧平均 9.62 ms、p95 19.91 ms；热运行 240 帧平均 5.40 ms（约 185 FPS）、p95 14.53 ms。R 重置阶段平均 8.83 ms、p95 20.21 ms；Space 挤压阶段平均 8.21 ms、p95 18.31 ms。中位帧可以低至 1.84 ms，但周期性 scene/game 尖峰仍使稳定 300 FPS 不成立。
- 果冻高度/宽度随运行和 Space 输入发生变化，说明旧 GPUJelly 求解及交互输入确实在真实编辑器运行；三个 Contact Cube 的运行时位置与旋转从初始到最终逐值不变，说明演示尚未把软体反馈接入 Jolt 刚体，不能宣称双向交互。该结论与 `JellyContacts.py` 尚未挂入 GPUJelly 主路径一致，不以“场景能跑”替代验收。
- Play 后通过 `infernux.console.read` 读取 native console：errors=0、warnings=0；当前失败属于能力尚未接线与性能尾延迟，而非本轮运行时异常。探针最终正常 Stop，恢复 Edit 且场景 dirty=false，编辑器保持打开。
- 进度与工期不因一次可运行复测上调：工程实现约 39%，最终 041 验收约 19%，严格 checklist 2/235（0.9%）；单人全职余量仍为 10—14 周，保守日历 14—20 周。下一主链仍是通用 GPU Mesh 直供、软体窄相与 GPU 聚合反馈接入 Jolt、数量矩阵和原生编译器深裁，而不是继续调演示参数。

### 2026-09-10：冻结果冻纵向切片，补齐动态 Mesh 法线/切线主路径

- 用户决定当前阶段冻结果冻场景并转向其它 041 缺口。最终 1/8/32/128 Rigidbody 矩阵和严格 300 FPS 长窗口仍是 A03.4 发布门槛，不以当前三个 Cube 或静止数字提前勾选；后续优化继续落在通用 RHI、物理交换和空间候选，不继续用项目参数打磨。
- 驻留 Mesh 派生属性从法线扩展为一次 GPU 法线+切线 pass。项目 kernel 只写 position，阶段末在同一有序 compute batch 内重建，渲染直接消费同一 `inx.buffer`。删除按相同浮点坐标焊接 Sphere UV seam 的场景化猜测，严格按作者索引拓扑保持 UV/硬边分裂。
- CPU inline Mesh 增加 tangent 手动输入与读取，以及 `recalculate_normals`、`recalculate_tangents`、`recalculate_bounds`。默认遗漏 tangent 时从 position/normal/UV 派生；显式 normal/tangent 保留。派生属性不触发 Collider cooking，resident Mesh 明确拒绝 CPU 重算以避免隐藏整网格 readback。
- 新增真实 Vulkan 断言：Sphere shear 后法线与切线均变化、归一且正交，handedness 为 ±1；同位置但拓扑分裂的两个面保持各自法线。63/64 项定向 Python 回归在修正“不完整三角形+显式法线”的既有合同后全部通过；`compute_mesh_gpu`、`mesh_publication_gpu`、`runtime_mesh_retirement_gpu` 三项真实 Vulkan 回归通过。
- 可见桌面 041Lab 经 MCP 重开并真实 Play；首次错误来自本地测试启动器漏传已构建的私有 lowering 目录，普通 wheel 并不依赖该环境变量。启动器改为从固定 CMake staging 绑定唯一编译器产物后，控制台 warning/error=0，1920×1080 Game 捕获显示变形 Sphere 光照连续，瞬时语义读数 1226 FPS（0.8 ms）；该短样本不替代 A03.4 长窗口性能矩阵。
- 当前工程实现估约 46%，端到端可验收约 24%，严格 checklist 仍为 2/235（0.9%）。剩余单人全职约 9—13 周；含四平台、Runner Long 实迁移与性能长尾的保守日历约 13—18 周。A04 仍缺统一 Mesh 资源编辑模型、GPU Bounds/剔除、实例/纹理计算消费和完整跨平台验收。

### 2026-09-10：Gizmo 直接消费 GPU 驻留数据

- 定位到 041Lab 内部核的主瓶颈：`on_draw_gizmos` 每帧对软体位置执行同步 `get_data()`，随后 NumPy 重建、pybind 转换、C++ 分片并再次上传 GPU。原有 `draw_lines` 批量接口只减少 Python 小对象，没有消除 GPU→CPU→GPU 往返。
- `Gizmos.draw_lines` 现可直接接受 GPU `inx.buffer`。引擎在同一 Gizmo compute 批次把位置扩展成 canonical Vertex 流，绘制绑定驻留缓冲；不可变索引拓扑仅在身份首次出现时上传，不做逐帧内容哈希或比较。NumPy immediate-mode 路径保持原合同。
- 新增通用 `draw_wire_spheres`：一个单位三圆线框和中心索引在 GPU 批量展开，内部节点不再逐个生成小列表/矩阵，也没有为了性能删减显示数量。源 Buffer 显式拥有派生资源生命周期，关闭时一并退休。
- 041Lab 的 GPU 约束线和全部内部节点迁移到公共接口。通过 MCP 在可见编辑器 Play，按 G 后节点随变形球体运动；重编原生模块并冷重启后再次通过，控制台 warning/error=0。捕获：`.infernux/mcp_sessions/20260910-020228-2acfc942/review/041-resident-gizmos-cold-restart.png`。
- 新增真实 Vulkan `infernux.compute_gizmo_gpu`，验证驻留位置更新、线与批量球接入及至少两个 resident vertex binding；连同 `compute_mesh_gpu`、`mesh_publication_gpu`、`runtime_mesh_retirement_gpu` 为 4/4 通过，定向 Python Gizmo/compute/Mesh 回归 86/86 通过。
- 当前工程实现估约 47%，端到端可验收约 25%。A07.1 仍缺 Scene View 分段计时、普通组件大负载和开关/不可见负载对照，不以 Game View 的短时约 1327 FPS 代替该验收。

### 2026-09-10：音频 2D/3D 主链与实时回调

- 现有 AudioClip 解码与转换一直保留双声道，真正错误位于 `FeedVoiceStream`：所有源在输出前无条件把左右声道平均为 mono，因此把 AudioSource 放到 Listener 附近并不等于 2D 音频。删除该伪 2D 约定，增加 `spatial_blend`：0 保持左右声道，1 为带距离衰减和等功率声像的点声源，中间值连续混合；源/轨道增益只乘一次。编辑器素材预览固定使用 2D。
- 新建 AudioSource 默认 2D，C++、pybind、Python CppProperty、stub 与严格文档序列化一致。旧场景唯一兼容规则是缺字段按旧的全 3D 语义 1.0 读取，之后保存写入新字段；未知字段仍由严格文档合同拒绝，没有多级 fallback。
- SDL 实时回调删除逐次 `std::vector` 分配和参数互斥等待，改用固定 1024-frame 栈分块；gain/spatial gain/pan/blend/pitch/loop/finished 通过原子快照交换，PCM 与 cursor 保持回调线程单所有者。新增纯混音数学回归覆盖 2D 声道保持、点声源中心/右声像、距离静音和中间 blend。
- 建立固定 Bus 合同 `Master/Music/SFX/Ambience/UI`。AudioSource 在配置/反序列化边界解析并拒绝未知名称；实时回调只接收数值。source×track×bus 在 voice gain 中应用一次，空间增益在 2D/3D blend 中应用一次，Master 只由 SDL device gain 应用一次；bus mute 不覆盖保存的音量。
- Windows Release `_Infernux` 编译通过；`infernux.audio_mixer` 与真实 Vulkan `infernux.compute_gizmo_gpu` 2/2 通过；组件集成 152/152、音频 Inspector 定向 9/9 通过。可见 041Lab 最终冷启动由 MCP 确认活动场景未脏、console warning/error=0。严格 checklist 现为 5/235（2.1%）；工程实现约 49%，端到端可验收约 27%，余量仍估单人全职 9—13 周、保守日历 13—18 周。A11 后续仍包含虚拟 voice、流式解码、增益平滑和真实听音压力矩阵。

### 2026-09-10：共享音频播放数据、采样级平滑与独立光标状态

- AudioClip 新增按输出采样率准备的不可变 F32 stereo 播放像；同一 clip 的多个 track/one-shot voice 共享这一所有权，不再各自 SDL 转换并复制整段 PCM。不同输出采样率产生新代际，旧活动 voice 继续安全持有旧像；Unload 后不再发放新像。修正 AudioClip move 时漏移 GUID 的原有所有权缺口。
- 音频回调新增 callback-owned 当前参数，gain、距离增益、pan 和 spatial blend 以约 5ms 的固定采样级斜坡趋近主线程原子目标。属性变化、mute 与 Bus 改变不再直接形成波形断点；没有加入动态分配、回调锁、音频图或多套平滑 fallback。长音乐仍未流式解码，有界真实/虚拟 voice 与压力听测仍未完成，故不勾选对应完整条目。
- A12 开始拆分游戏光标语义：visible、confined 与 locked 成为三个独立逻辑状态；locked/Scene 相机捕获共享一个 SDL 相对模式主链，失焦统一解除 relative/grab 并显示指针，聚焦恢复请求。现有 `set_cursor_locked` 兼容保留。新增逻辑窗口坐标 `warp_cursor`，拒绝 relative/无窗口/Web 宿主，匹配的 SDL warp motion 只更新位置、不进入 gameplay delta；Python/pybind/stub 使用同一表面。Game View 坐标/DPI 与真实边缘拖拽仍未验收。
- 核对 A03.3 后没有重写计时系统，而是给既有生产 accumulator 补权威回归：覆盖单帧零/一/多 fixed step、1 秒长帧按 0.05 秒截断、timeScale 0/2、scaled/unscaled fixed time、Pause 不推进和显式 Step 恰好一次。`infernux.runtime_frame_barrier` 使用同一生命周期/物理屏障主链通过，据此只勾选固定累积与时间/暂停/追赶两项；扩展阶段、手动物理 API、GPU 依赖和插值仍未完成。
- Windows Release `_Infernux`、`infernux.audio_mixer`、`infernux.audio_clip`、`infernux.input_manager` 与 `infernux.runtime_frame_barrier` 通过；输入/Player/组件/Inspector Python 回归 294/294，追加 cursor 定向回归 75/75 通过。用新模块再次可见启动 `D:/Users/Chenlizhe/Desktop/Infernux041Lab`，MCP 确认 `01_XPBD_Jelly`、edit、dirty=false、warning/error=0。严格 checklist 为 7/235（3.0%）；工程实现约 51%，端到端可验收约 29%，剩余仍估单人全职 9—13 周、保守日历 13—18 周。未提交、推送或发布。

### 2026-09-10：权威物理扩展阶段与 GPU JIT-only Player 构建

- 在既有 `RunFixedSimulationStep` 主链公开 `physics_pre_step`/`physics_post_step`，分别位于 Collider/Transform 输入同步之后、Jolt 之前，以及接触事件和 Rigidbody/Transform 发布之后。两阶段复用同一个 native frame、生命周期快照、变更日志与 compute 记录批次，没有新增物理循环或项目专用调度。
- 真实 Jolt 集成测试中，pre 阶段提交 3 m/s VelocityChange，post 阶段在同一次 Step 读到 3 m/s 权威速度和 `velocity × fixed_delta` 的已发布位移；FixedTiming/RigidbodyNumerics 13/13、物理/运行时基线与调度定向 173/173、原生 runtime frame barrier 通过。
- 按已确认的 041 JIT-only 决议删除旧 GPU AOT 双轨：BuildRequest 不再搬运 compute artifacts，Build Settings/自动化构建不再执行 owner-thread `prepare_build`，GameBuilder 不再生成 `Library/Compute` AOT staging 或给 `hpc` 注入 `_aot`。Player 中 `@inx.compute.kernel` 以 sourceless bytecode 保留，首次 launch 使用 wheel 自带 compiler-only frontend 和 Infernux Vulkan RHI；Assets 与 package 脚本两种 Cook 回归通过。
- 构建/compute/GameBuilder 定向回归 329/329（另有 1 个预期跳过），Windows Unicode junction 主路径在允许 ProgramData 的真实权限下 1/1 通过。删除的是已经否决的 Python GPU AOT 管线，粒子图的独立离线 artifact 未被混删。
- 严格 checklist 为 9/235（3.8%）；工程能力估约 53%，端到端可验收约 30%。A03.4 的 1/8/32/128 刚体矩阵、真实双向软体交互和完整编辑器稳定 300 FPS 仍未完成；剩余仍估单人全职 9—13 周，保守日历 13—18 周。未提交、推送或发布。

### 2026-09-10：果冻样板冻结后的单 Collider 查询切片

- 真实可见 041Lab 经 MCP 进入 Play、运行约 40 秒并退出，场景保持 `dirty=false`，native console warning/error=0。240 帧内部阶段窗口平均 1.259 ms、P95 6.035 ms；该统计不包含完整编辑器 UI/交换链，明确不拿来替代 A03.4 的 300 FPS 完整帧验收。
- 新增 `Collider.raycast`，直接对目标 Collider 所属 Jolt body 的 subshape 做射线查询。复合体会跳过前方兄弟形状；显式目标查询不受 IgnoreRaycast layer、trigger 和 pair-ignore 过滤。没有复制世界或维护第二套碰撞世界。
- 同一轮补齐 `Collider.closest_point`：内部点原样返回，外部最近点由 Jolt 几何接触求得；覆盖 Box/Sphere/Capsule/Cylinder 及 compound 兄弟隔离。凸 Mesh 最近点仍明确未支持，不以 AABB 或采样点兜底伪造。
- Windows Release `_Infernux` 增量构建通过；新增旋转/复合/过滤/四形状测试 7/7，完整物理加动态 Mesh 派生属性定向回归 120/120。自动法线/切线的三项真实 Vulkan 回归再次通过。项目不再承担基础重建；发布代际、GPU bounds 和共享 Mesh 完整合同仍未勾选。
- 严格 checklist 更新为 11/235（4.7%）；工程能力小幅更新至约 54%，端到端约 30%。剩余仍估单人全职 9—13 周，保守日历 13—18 周。A03.4 冻结为硬门槛而非取消，未提交、推送或发布。
## 2026-09-10 — Taichi AOT removal and dynamic Mesh closure

- Removed the private compiler binding surface for AOT modules and graphs, then removed AOT build/load ownership from `Program`, `GfxProgramImpl`, and `gfx_runtime`. `graph_builder` no longer enters the compiler target. This is a real compile/link removal, not an API alias or runtime fallback.
- The private Windows compiler module decreased from about 12,056.5 KiB to 11,710 KiB. The remaining import trace still includes Field/SNode/mesh/simt modules because the upstream AST frontend imports them eagerly; that deeper frontend separation remains open.
- Real Vulkan `compute_kernel_gpu_test.py` and `compute_mesh_gpu_test.py` pass after the removal. The kernel regression now asserts that `AotModuleBuilder`, `GraphBuilder`, and `LoadedAotModule` are absent from the native compiler binding.
- Added public `MeshRenderer.inline_mesh_version`. CPU tests verify successful/failed publication boundaries, explicit normal/tangent rebuild generations, local-to-world bounds, manual attributes, degenerate/isolated/empty geometry and explicit collider recook. `test_mesh_renderer_numpy.py`: 18 passed.
- Real Vulkan shared-mesh publication passed 24 generations with two Renderers, no pending upload, no eviction and no stale GPU bytes. A04.1 topology/derived-attribute and unified CPU/GPU Mesh-path checklist items are complete; exact GPU bounds-driven culling and A03.4 coupling remain open.

## 2026-09-10 — Rigidbody native schema slice and compiler-loader cleanup

- Added the first native physics component to the authoritative semantic catalog: `native:infernux.Rigidbody`. Its serialized fields, defaults, ranges, enum members, constraint identity, owner and runtime profiles are now described by the C++ registration instead of being duplicated in the Python wrapper. The public Python enum classes remain the author-facing API, but their `CppProperty` schemas point at the native declaration.
- Added English and Chinese Inspector labels/tooltips for every Rigidbody field and enum member. `ComponentCommandService` now edits mass, drag and enum fields through the same native schema used by serialization; Undo/Redo and rejection boundaries are covered by four parameterized command tests. `test_semantic_type_registry.py`: 29 passed.
- Completed the shared automatic-geometry evidence after the prior dynamic Mesh work: `_Infernux` rebuilt successfully; real Vulkan `compute_mesh_gpu_test.py` and `compute_kernel_gpu_test.py` both pass. A resident vertex buffer can be deformed by a kernel, then the same ordered compute batch derives area-weighted normals and tangent handedness before rendering consumes the buffer. CPU inline meshes keep explicit authored attributes and expose separate normal/tangent/bounds rebuild calls; no resident path performs a hidden readback.
- Fixed two unrelated test-contract regressions exposed by the full baseline run: the compiler-only Taichi loader no longer performs ad-hoc `Path.resolve()` normalization, and editor barrier/gizmo helpers tolerate the minimal `Engine.__new__` test harness. Player build routing also restores its missing `tempfile` import. The ASCII build-cache junction now prefers a writable ASCII ancestor beside the project cache instead of requiring `ProgramData` ACLs.
- Safe full-suite run: 5,627 passed, 13 skipped, 24 failures remain in pre-existing platform/GUI/MCP/path-isolation cases (including a no-MCP Player Unicode decode path and line-renderer native environment cases); the new semantic and compiler/mesh slices are green. Do not call the repository CI green from this local result.
- Strict plan remains **13/235 (5.5%)** because the Rigidbody work is a partial A01.1 slice and the A03.4/A05/A07.1 broad gates remain open. Practical engineering capability is about **57–58%**, end-to-end acceptance about **32%**. Remaining estimate stays at **9–13 full-time engineering weeks**, or roughly **13–18 calendar weeks** including cross-platform, long-tail performance and release validation. The hard 300 FPS bidirectional Rigidbody/softbody gate is intentionally not claimed complete.

## 2026-09-10 — Visible 041Lab MCP regression and source-launch fix

- Source-tree editor launches now read the authoritative `Infernux.version.ENGINE_VERSION` instead of requiring installed wheel metadata. This keeps source and wheel launches on one version path and removes the startup-only `PackageNotFoundError`.
- Supervisor-launched source sessions pass the locally built GPU-JIT vendor directory to the child editor when a packaged vendor is not installed. This is a developer-source path; packaged users continue to use the wheel payload.
- Reopened `D:/Users/Chenlizhe/Desktop/Infernux041Lab` through the real visible MCP supervisor and ran `01_XPBD_Jelly` in Play Mode. The corrected session reached Play Mode in 833.9 ms, produced zero console warnings/errors, and rendered the jelly/cube scene successfully.
- Real MCP performance window: 240 frames, frame average 1.278 ms, p50 0.843 ms, p95 6.232 ms, max 7.966 ms. This is a diagnostic source-editor result, not the 300 FPS acceptance claim; rigid/soft bidirectional scaling remains open.
- Captured the live Game target to `review/041-jelly-current.png` for human review, then returned the visible editor to Edit Mode. Resident Mesh Vulkan and 29 semantic registry tests remain green.

## 2026-09-10 — Taichi container surface reduced to compiler markers

- The real Infernux kernel import trace showed that upstream `_ndarray`, `_texture` and `field` modules were still imported even though Infernux owns all public storage through `inx.buffer`.
- Replaced those three runtime container implementations with small compiler-only marker boundaries. They preserve the upstream names needed for AST classification, but construction and host/storage access now fail explicitly; no `to_numpy`, `from_numpy`, sampling or field ownership API remains in the private wheel surface.
- The required `SNodeHostAccess` marker is retained only because upstream matrix/struct code checks the type while compiling unsupported paths. No SNode, field, ndarray or texture storage is restored.
- Rebuilt the Windows compiler staging target `infernux_gpu_jit_compiler` under `conda activate infernux`. The final staged GPU compiler payload is 11,801,444 bytes; this is source staging, not a published engine wheel.
- Real Vulkan `compute_kernel_gpu_test.py` and `compute_mesh_gpu_test.py` pass after the trim. The kernel contract additionally asserts that the reduced container classes expose no upstream host-storage methods. The import trace remains limited to compiler frontend modules plus the explicit marker modules; `_snode`, `simt`, mesh, sparse and AOT modules remain absent.
- This is a compiler payload reduction, not a claim that Taichi A05 or the complete 041 plan is complete. Cross-platform wheel generation, Player launch, resident physics coupling and the 300 FPS bidirectional rigid/soft gate remain open.

## 2026-09-10 — CPU tangent-frame normalization for dynamic meshes

- The native CPU tangent builder now normalizes authored normals for the orthogonal projection step. Authored normal data is preserved, while derived tangents use the same unit-frame assumption as the renderer shader path. This removes lighting-frame drift when procedural/imported meshes provide scaled normals.
- Added a regression covering non-unit authored normals: generated tangents remain unit length and orthogonal to the normalized normal direction.
- Rebuilt the Windows Release native module under `conda activate infernux`; `python/test/test_mesh_renderer_numpy.py` is green at 19 passed. Real Vulkan `compute_kernel_gpu_test.py` and `compute_mesh_gpu_test.py` remain green (`INFERNUX_PUBLIC_GPU_KERNEL_OK`, `INFERNUX_COMPUTE_RESIDENT_MESH_OK`).
- This closes one system-side normal/tangent correctness edge, but resident GPU meshes still intentionally require their geometry kernel to publish the normal/tangent fields; no hidden GPU readback or project-specific jelly optimization was introduced.

## 2026-09-10 — Source Supervisor uses the checkout Python surface

- A fresh visible 041Lab launch exposed that the Supervisor only forwarded the staged Taichi vendor path; the child then imported an older installed wheel and failed before Play with `Infernux has no attribute compute`.
- The Supervisor now prepends the checkout `python/` root when it is present, while leaving installed/plugin-only sessions unchanged. This keeps the source Python API, native `_Infernux` module and compiler vendor payload on one build.
- Reopened the real visible editor through the MCP Supervisor, entered `01_XPBD_Jelly` Play successfully, and returned to Edit. Console result was one render-graph INFO and zero warnings/errors; the transition completed in 665.9 ms. This validates the source launch path, not the outstanding 300 FPS rigid/soft acceptance gate.

## 2026-09-10 — Convex MeshCollider closest-point query

- Extended the generic Jolt `Collider.closest_point` path to accept cooked convex `MeshCollider` shapes. The query reuses the collider's existing shape/cooking contract, preserves interior-point identity, and reports an explicit not-ready/error state if asynchronous cooking has not produced a shape; concave mesh colliders remain rejected instead of silently approximating them.
- Added a real integration case using the engine primitive Cube as a convex MeshCollider. Full `python/test/test_integration_physics.py` passes: 104 passed; the focused closest-point/cooking slice passes 7/7.
- Rebuilt the native Release module under `conda activate infernux`, reopened the real visible 041Lab through MCP, and completed Play/Console/Stop with zero warnings/errors. This is an A03.1 generic geometry-query slice and does not change the 041Lab soft-body algorithm or claim the 300 FPS bidirectional gate.

## 2026-09-10 — Native broad-phase to resident box-state path

- Added a generic `Physics.query_rigidbody_box_states_in_bounds(...)` API. Jolt broad-phase candidate collection, layer/trigger filtering, world-space BoxCollider projection, and SoA upload now happen in one native call; the public result returns only the candidate wrappers plus a valid-prefix count.
- The path targets reusable `inx.buffer(device="gpu")` outputs and preserves the existing explicit buffer contract. It avoids the former Python query-then-rewalk sequence for GPU soft-body/contact solvers; no jelly-specific shortcut or hidden readback was added.
- Updated the public `.pyi` contract and rebuilt Windows Release under `conda activate infernux`. `python/test/test_integration_physics.py` and `python/test/test_mesh_renderer_numpy.py` pass together: 123 passed, 1 environment cache warning. The new GPU resident path still needs a graphical MCP/041Lab run before it can be considered part of the hard performance gate.
- Practical engineering progress is now approximately **59%**, strict checklist remains **13/235 (5.5%)**, and end-to-end acceptance remains approximately **32%**. The remaining estimate is still **9–13 full-time weeks** (about **13–18 calendar weeks**), dominated by generic rigid/soft coupling, GPU-resident contact feedback, Taichi compiler integration, gizmo/render ordering, and cross-platform acceptance. The 300 FPS bidirectional gate is not claimed complete.

## 2026-09-10 — Resident mesh tangent frame parity

- The GPU automatic mesh-attribute kernel now normalizes authored normals for tangent projection when `auto_normals=False`, while leaving the authored normal values untouched. CPU and GPU derived tangent frames now share the same unit-normal invariant.
- `python/test/test_compute.py` and `python/test/test_mesh_renderer_numpy.py` pass together: 60 passed, 1 environment cache warning. This is a renderer/mesh contract fix, not a per-project jelly adjustment.

## 2026-09-10 — Real Vulkan validation of native broad-phase upload

- Extended `python/Infernux/test/compute_physics_exchange_gpu_test.py` to exercise `Physics.query_rigidbody_box_states_in_bounds` with two live Jolt rigidbodies and resident GPU SoA buffers.
- Under `conda activate infernux`, with the freshly staged compiler vendor directory, the real Vulkan test passes: `INFERNUX_COMPUTE_PHYSICS_EXCHANGE_OK`. This proves the new native query/upload contract and candidate identity mapping; it is not yet the full soft-body narrow-phase or 300 FPS gate.

## 2026-09-10 — GPU authored-normal preservation regression

- Extended the real Vulkan resident-mesh test with a deliberately scaled authored normal and `auto_normals=False`. The renderer keeps the authored `[0, 0, 4]` values while the automatic tangent frame remains unit length and orthogonal to the normalized normal.
- `compute_mesh_gpu_test.py` passes with `INFERNUX_COMPUTE_RESIDENT_MESH_OK`. This validates the normal/tangent invariant on the actual Vulkan path rather than only the CPU helper.

## 2026-09-10 — Visible 041Lab post-build performance sample

- Relaunched the real desktop editor through the MCP Supervisor after the native rebuild, entered Play, collected a bounded 240-frame native timing window, and returned to Edit. The observed full-frame sample was average **1.471 ms**, P50 **0.903 ms**, P95 **7.504 ms**, P99 **8.299 ms**, max **8.918 ms**; game-only average was **0.821 ms**.
- This is useful evidence that the current visible session is responsive, but it is not the A03.4 acceptance result: the run still lacks the required controlled 1/8/32/128 rigidbody matrix, explicit continuous bidirectional contact phases, and proof that the scene was not in a low-work/idle state. No 300 FPS claim is made.

## 2026-09-10 — Mesh/Gizmo residency regression set

- Ran the real Vulkan mesh publication, runtime mesh retirement, and resident gizmo GPU tests with the rebuilt native module. Results: 24 mesh generations retired with zero pending/stale bytes, runtime mesh residency returned to baseline, and `INFERNUX_COMPUTE_RESIDENT_GIZMO_OK` passed.
- This confirms the existing resource-generation retirement path remains intact after the normal/tangent and resident-buffer changes. It does not close the broader Mesh asset persistence or A03.4 rigid/soft coupling gates.

## 2026-09-10 — Full 041 progress re-audit

- Recounted the complete A00—A15 checklist rather than extrapolating from the active compute/physics workstream. The strict result at that point was **16/235 (6.8%)**; no additional item had enough end-to-end evidence to be checked.
- Corrected the plan header so the former approximately 65% figure is explicitly scoped to the active semantic/physics/Mesh/Compute vertical slice. Whole-plan reusable implementation is currently estimated at **30%—35%**, and final user acceptance at **15%—20%**.
- Confirmed the visible 041Lab editor is alive in Edit state through the real MCP endpoint. The latest Windows Release native suite remains **75/75 CTest passed**. These are valid regression signals, but they do not satisfy the missing Additive Scene, joint, rigid/soft 300 FPS matrix, UI/Rect/RenderTexture, full audio, Runner Long migration, multi-platform artifact, or release gates.

## 2026-09-10 — Play-state investigation kept out of the main path

- A visible MCP run with Contact Cube A placed outside the jelly showed a clean fresh Play start and normal settling after the editor was rebuilt; the earlier extreme position sample was taken after a long-running session and was not a controlled stale-velocity reproduction.
- A candidate generic “zero every dynamic Rigidbody on Play” change was rejected because it broke the existing deferred-body contract for explicit initial velocity, Start-time forces, friction, and CCD. It was fully reverted under the fire-forced rule.
- After the revert and a fresh `_Infernux` binding rebuild, `python/test/test_integration_physics.py` is green: **107 passed**. No 041 checkbox was advanced from this investigation; a deterministic repro is still required before changing Play/physics state semantics.

## 2026-09-10 — Manual fixed-step contract closed

- Added an integration regression for `SceneManager.step`: while Play is running the call is a no-op, preventing a tooling/editor request from advancing a second World beside the automatic accumulator; once paused, one call reuses the authoritative fixed-step path exactly once.
- The same regression observes the phase order `fixed_update → physics_pre_step → physics_post_step → update → late_update`, and checks that fixed time and runtime-frame count advance by one step only. The public C++ binding and `_Infernux.pyi` now state this contract explicitly.
- Under `conda activate infernux` with the rebuilt Windows Release native module, the focused manual-step/stage tests pass **3/3** and the full `python/test/test_integration_physics.py` suite passes **108/108**. This closes the corresponding A03.3 checkbox; A03.3 GPU/CPU task dependencies and interpolation remain open.
- The same rebuilt Release tree also passes the complete native regression: **75/75 CTest**, including headless, Vulkan buffer/kernel/mesh/gizmo/physics-exchange, mesh publication/retirement and residency soak tests.

## 2026-09-10 — Checklist recount after manual-step closure

- Recounted the plan after the manual fixed-step contract was checked. The current strict result is **17/235 (7.2%)**: A00=1, A03=9, A04=3, A11=3, A15=1; all other sections remain open. The plan header and this log now use the same denominator and count.
- The strict count is intentionally low: the 041Lab GPU soft-body slice still lacks the required authoritative Rigidbody↔softbody bidirectional proof, 1/8/32/128 matrix, and complete-frame 300 FPS+ evidence. A04/A05 local contracts also do not substitute for the full multi-platform and release gates.
- Current estimate for the whole 041 remains approximately **30–35% reusable engineering implementation**, **15–20% end-to-end acceptance**, with about **9–13 full-time engineering weeks** remaining; allowing cross-platform artifact production, Runner Long migration, performance tail work, and release/CI verification, use **13–18 calendar weeks** as the conservative schedule.

## 2026-09-10 — Combined authoritative Rigidbody/Box GPU upload

- Added an engine-level `Physics.get_rigidbody_states_and_box_states(...)` path. It validates the existing `inx.buffer` layouts, gathers solver state and flattened BoxCollider descriptors from the same authoritative Jolt snapshot, and submits both groups through one engine-owned compute transfer batch.
- The previous separate APIs remain available and keep their behavior; the new path is an optimization of the submission boundary, not a project-specific jelly shortcut and not a second physics world. It is intended for fixed-step GPU consumers that need both motion and shape/material data.
- Updated the public and native stubs and the real Vulkan physics exchange regression. With the rebuilt Windows Release native module, `INFERNUX_COMPUTE_PHYSICS_EXCHANGE_OK` passes, the focused CTest pair (`headless` + `compute_physics_exchange_gpu`) is **2/2**, and the full `python/test/test_integration_physics.py` suite is **108/108**.
- This reduces one generic state-upload boundary but does not claim A03.4: the narrow-phase, bidirectional Rigidbody↔softbody coupling, 1/8/32/128 matrix, and complete-frame 300 FPS+ gate remain open.

## 2026-09-10 — 041Lab adopts the combined physics exchange path

- Updated the real desktop `Infernux041Lab/Assets/Scripts/GPUJelly.py` fixed-step exchange to call the engine-level combined state/BoxCollider upload instead of issuing the two separate public uploads. No solver constants, topology, material, or project-specific collision shortcut changed.
- Relaunched the visible editor through the MCP Supervisor with the rebuilt Release module and entered `01_XPBD_Jelly` Play. The editor reached Play in 674.4 ms; the Console contained only the render-graph INFO entry (`warnings=0`, `errors=0`).
- A bounded 240-frame MCP performance window reported complete-frame average **0.837 ms**, P50 **0.528 ms**, P95 **1.084 ms**, P99 **7.267 ms**, max **8.584 ms**; game-only average **0.383 ms**. This is a clean dynamic runtime regression for the new exchange boundary, not the A03.4 acceptance result: the controlled bidirectional contact phases and 1/8/32/128 matrix were not run here.
- The subsequent Windows Release CTest run remains **75/75 passed** (86.59 s), including headless, all GPU buffer/kernel/mesh/gizmo/physics exchange tests, mesh retirement and residency soak.

## 2026-09-10 — Native broad-phase plus combined state upload

- Added `Physics.query_rigidbody_states_and_box_states_in_bounds(...)`. Jolt broad-phase candidate selection, authoritative Rigidbody state extraction, BoxCollider projection, and the single GPU transfer submission now occur in one native call. Python only receives the candidate wrappers after the upload, preserving the mapping needed for feedback.
- The 041Lab fixed-step path now uses this query directly; it no longer performs a Python broad-phase wrapper pass followed by a second native state traversal. No jelly material, topology, timestep, or collision rule changed.
- Rebuilt the Windows Release module after closing the visible MCP editor that held staging files. Real Vulkan `compute_physics_exchange_gpu` passed, focused CTest (`headless` + physics exchange) passed **2/2**, and the full Python physics suite passed **108/108**.
- This is still an engine-side transfer/broad-phase optimization, not A03.4 completion. The narrow-phase, bidirectional impulse proof, quantity matrix and full-frame 300 FPS+ gate remain open.

## 2026-09-10 — 041 checklist re-audit after fresh MCP launch

- Recounted every checkbox in `041-runner-long-foundation-plan.md`: **17/235 checked (7.2%)**. The distribution is A00=1, A03=9, A04=3, A11=3, A15=1; no additional item was promoted because the latest native exchange path is still a partial prerequisite rather than a complete acceptance contract.
- Rebuilt Windows Release and launched the real `Infernux041Lab` editor through MCP after the staging files were released. A fresh Play → Console → Stop cycle completed successfully; the console contained only the render-graph INFO entry (`warnings=0`, `errors=0`), and the scene returned to Edit without dirtying the on-disk scene.
- Current progress remains approximately **30–35% reusable engineering implementation** and **15–20% end-to-end user acceptance**. The largest open gates are A01/A02 semantics and additive scenes, A03.4 rigid/soft bidirectional coupling with the 1/8/32/128 and 300 FPS matrix, A05 buffer/JIT/compute packaging, A06–A09 UI/Rect/Renderer/RenderTexture, A10 Runner Long migration, and A12–A15 release/multi-platform closure.

## 2026-09-10 — Public compute recording boundary

- Added the public `inx.compute.recording()` context and type stub. It is a thin engine-owned command-recording scope over the existing ordered `ComputeHost.dispatch_batch` path; nested scopes remain one submission, while explicit buffer reads and native consumers still flush at the observation boundary.
- Added a unit regression proving two ordered launches (including a nested scope) become one host submission. Under `conda activate infernux`, `python/test/test_compute.py` passes **42/42**.
- Updated the 041Lab fixed-step solver to record the full GPU soft-body step and skinning dispatches together. A fresh MCP Play run loaded the new API with zero console warnings/errors; the reported solver batch wall time fell from the earlier 4.79 ms sample to **2.38 ms** in the same editor session. This is a submission-overhead improvement, not the A03.4 300 FPS gate: the run still lacks the required controlled rigidbody matrix and full bidirectional contact evidence, so no checklist item was promoted.

## 2026-09-10 — Contact feedback regression and rollback to explicit substep semantics

- A direct-sum change to the Lab's rigidbody feedback was tested in the real editor and rejected: Contact Cube B accelerated to hundreds of metres in the first second. This proved that the GPU solver's rigid-body velocity snapshot is fixed across its authored substeps; summing stale-velocity corrections is not a valid same-step impulse contract.
- Replaced that experiment with an explicit mean-substep scale while still summing contact contributions within each substep (no contact-count averaging). The fresh MCP run remained bounded: the contact cube was displaced by roughly one metre laterally, stayed near the floor after settling, and the Console reported zero warnings/errors. The scene capture also shows the file-material sphere and authored Cube colliders rendering together.
- This is a project-side solver correction backed by a negative/positive regression pair; it does not close the engine's A03.4 gate. A generic engine-side GPU rigidbody state integration path is still required before a true full-sum same-step coupling claim is justified.

## 2026-09-10 — Lifecycle-owned compute recording

- Moved the recording boundary into the shared Python runtime scheduler: every enabled lifecycle callback now runs inside the public `inx.compute.recording()` scope, with a late import so CPU-only scheduler tests do not acquire a GPU backend. The 041Lab callback no longer owns a project-specific recording wrapper.
- After restarting the real editor, MCP Play remained clean (`warnings=0`, `errors=0`). A 240-frame complete-frame window measured average **0.854 ms**, P50 **0.518 ms**, P95 **1.242 ms**, P99 **7.395 ms**, max **7.963 ms**; game-only average was **0.407 ms**. The average is below the 3.33 ms reference, but the P99 synchronization tail and the missing 1/8/32/128 matrix still prevent A03.4 acceptance.
- Regression suites remain green: `test_compute.py` + lifecycle scheduler **69 passed**, and `test_integration_physics.py` **108 passed**. No checklist item was promoted; this closes a generic scheduling prerequisite only, not the rigid/softbody contract.

## 2026-09-10 — Full 041 re-audit after lifecycle recording

- Recounted every checklist entry in `041-runner-long-foundation-plan.md`: **17/235 checked (7.2%)**. The distribution is A00=1, A03=9, A04=3, A11=3, A15=1; all other sections remain open. No additional checkbox has all of its acceptance clauses satisfied by the current evidence.
- Confirmed the current 041Lab implementation uses the engine Sphere topology, file-backed material, `inx.buffer(device="gpu")`, `@inx.compute.kernel`/`inx.compute.launch`, engine-owned automatic mesh attributes, and the native broad-phase/rigidbody state exchange. These are valid prerequisites, but not the complete A03.4 gate.
- The remaining hard blocker is still generic Rigidbody↔softbody bidirectional coupling: continuous dynamic contact, compact feedback without a per-frame full-mesh readback, 1/8/32/128 rigidbody matrix, and complete-frame 300 FPS+ with P50/P95/P99/max and synchronization cost. The latest clean MCP window is useful evidence (full-frame average 0.854 ms, P50 0.518 ms, P95 1.242 ms, P99 7.395 ms, max 7.963 ms) but is not that matrix and does not close the tail.
- Other uncompleted product blocks remain A01/A02 semantics and additive scenes, A03.2 joints, A04.1 resource publication, A04.2 generic compute/instance/readback, A05 CPU JIT and wheel/platform packaging, A06–A09 UI/Rect/material/RenderTexture, A10 Runner Long visual pipeline, A11 audio stress/voice, A12–A14 author tools/plugin/Steam, and A15 migration, demos, multi-platform, documentation, CI and release.

## 2026-09-10 — Combined compute/readback boundary and visual diagnosis

- Added an engine-level compute submission that records buffer updates, ordered kernel dispatches and selected compact readbacks in one queue submission/wait. GPU `Buffer.get_data()` and the physics feedback path now consume this boundary instead of forcing a preceding submission followed by an independent readback submission.
- Focused regressions pass: compute/lifecycle **70/70**, physics **108/108**, and real Vulkan buffer/kernel/mesh/physics-exchange CTest **5/5**. A fresh 240-frame editor window measured full-frame average **0.695 ms**, P50 **0.555 ms**, P95 **1.155 ms**, P99 **3.655 ms**, max **4.297 ms**; game-only average was **0.243 ms**. This remains below the strict A03.4 evidence bar because it is not the required 1/8/32/128 controlled matrix.
- Added generic performance-window resource counters for resident mesh vertex buffers and pending/submitted/completed mesh uploads. The MCP operation contract and regression now return the same structure.
- The apparent disappearing jelly was isolated with real MCP operation rather than renderer changes: in Edit and at zero time after reset, the Sphere surface and resident vertex buffer are present; with dynamic contact cubes disabled, it falls and settles stably on the static floor. With those rigidbodies enabled it is propelled outside the authored camera/floor region. This rules out a resident-buffer/material/render synchronization regression, but does not prove the full rigid/soft coupling contract.

## 2026-09-10 — CPU/GPU automatic mesh attribute semantics aligned

- Closed a system-side lighting inconsistency in GPU automatic tangent generation. Degenerate geometry or an unreferenced vertex with no usable normal could previously produce a zero tangent, while the CPU mesh path produces the deterministic `(1, 0, 0, 1)` frame.
- The real Vulkan resident-mesh regression now covers an unreferenced zero-normal vertex as well as dynamic deformation, automatic area-weighted normals, UV tangents, topology splits and preserved non-unit authored normals. `infernux.compute_mesh_gpu` passes together with the buffer/kernel GPU tests **3/3**; compute, Mesh wrapper and MCP operation suites pass **83/83**.
- No checkbox was promoted. A04.1 still requires the complete shared Mesh resource/publication/retirement contract, and A03.4 still requires the controlled bidirectional rigidbody matrix. The strict plan result therefore remains **17/235 (7.2%)**.

## 2026-09-10 — Native HingeJoint first contract closed

- Added the public native `HingeJoint` backed by a Jolt Hinge in the existing `PhysicsWorld`. It supports a required local Rigidbody, an optional connected Rigidbody or fixed world, local anchor and axis, angular limits, current angle and connected-body collision control. The component participates in native fixed update so deferred Collider bodies create the constraint at the first authoritative physics boundary rather than requiring a second scene event.
- Added reference-counted body-pair collision suppression in the physics contact listener. Changing the collision option invalidates Jolt's contact cache and activates both bodies, so the new policy takes effect on the next step instead of waiting for incidental movement. The filter has no lock cost while no ignored pairs exist.
- Added a generic native component-reference remap hook. Direct hierarchy clones and scene documents loaded while the source scene remains alive now remap Hinge references to the fresh Rigidbody IDs; this is an engine graph mechanism rather than a Hinge-only clone workaround.
- Real Jolt regressions cover fixed-world anchor preservation, axis-constrained motion, ±35° limits, connected kinematic bodies, serialization, direct hierarchy cloning, duplicate live-scene loading, invalid settings and collision-off → collision-on contact behavior. The focused Hinge set passes **8/8** and the earlier full physics suite passed **113/113** before the final collision/remap cases were added.
- This closes only the first A03.2 checkbox. Inspector/Prefab/full lifecycle coverage, a single-axis translation joint and independently owned game-level pair-ignore remain open. Strict progress is now **18/235 (7.7%)**; reusable implementation is estimated at **31–36%**, end-to-end acceptance at **16–21%**, with approximately **9–13 engineering weeks / 13–18 calendar weeks** remaining.

## 2026-09-10 — SliderJoint and the joint authoring chain closed

- Added the public native `SliderJoint` backed directly by Jolt's slider constraint. It permits translation only on the authored local axis, locks every rotation and the other translation axes, supports an optional connected Rigidbody or fixed world, and enforces authored minimum/maximum distance without a per-frame Transform override. The pressure-plate regression reaches its lower travel limit under gravity while large lateral force and torque cannot escape the intended degree of freedom.
- Kept the public model intentionally narrow: `HingeJoint` and `SliderJoint` are two explicit single-axis components. Motor, spring, projection, break thresholds and a generic six-degree-of-freedom facade were not added without a current 041 consumer.
- Completed the native reference authoring path. Semantic fields remain named `connected_body`, while their canonical document field is explicitly `connected_body_component_id`. The generic `CppProperty` transaction validates component references using stable native component IDs, and the built-in Inspector now renders a reusable component picker instead of a text fallback. Undo/Redo, hierarchy clone, prefab/document instantiate, duplicate live-scene load, Disable/re-enable, Destroy and Stop-style scene rebuild all have focused regression coverage.
- Started the visible Windows Release editor on the real `Infernux041Lab` project and used MCP operations to create a temporary Cube with Rigidbody, HingeJoint and SliderJoint. Both component schema queries returned the same authoritative fields, including writable `connected_body`; the probe was then deleted without saving the project. Startup and render graph were clean.
- Validation is green: **184/184** Python physics/semantic/built-in component tests, **75/75** Windows Release CTest, Python compilation and both locale documents. A03.2 gains three checkboxes. The later-added `.blend` compound source-asset gate expands the denominator by one, so strict progress is **21/236 (8.9%)**. Reusable engineering implementation is estimated at **32–37%**, final user acceptance at **17–22%**; the overall schedule remains about **9–13 engineering weeks / 13–18 calendar weeks** because the A03.4 performance matrix, additive scenes, UI/RenderTexture, Runner Long migration and multi-platform release work still dominate.

## 2026-09-10 — Exact Collider pair policy and joint ownership closed

- Added public `Physics.ignore_collision/get_ignore_collision` on the existing Jolt World. The game-owned policy is an idempotent set keyed by the complete ordered pair of 64-bit Component identities; joint collision suppression remains a separate reference-counted body-pair policy.
- Contact validation resolves the actual compound subshape before applying the game policy, so ignoring one Collider does not suppress its siblings on the shared body. Changing the policy invalidates tracked events and Jolt contact caches and activates both bodies; destroying a Collider removes its owned pair entries.
- Real physics regressions cover a compound Box/Sphere ground where only the Sphere contact is ignored, runtime restoration, repeated idempotent writes, Collider replacement and a Hinge whose own suppression is released while the game policy remains authoritative. The focused set is **4/4**, the complete physics suite is **125/125**, and physics/semantic/built-in authoring is **187/187**.
- A clean Windows Release rebuild and complete CTest run pass **75/75** including all 15 real Vulkan tests. A03.1 and A03.2 each gain one checkbox. Strict progress is now **23/236 (9.7%)**; reusable engineering implementation is estimated at **33–38%**, final user acceptance at **18–23%**, and the remaining schedule stays approximately **9–13 engineering weeks / 13–18 calendar weeks**.

## 2026-09-10 — Preallocated batch Raycast closed

- Added `Physics.raycast_batch` as the hot-path world query rather than returning a list of per-hit Python objects. It accepts contiguous float32 origin/direction rows and writes hit flags, point/normal/distance, Jolt body identity, Collider component identity and GameObject identity into caller-owned typed NumPy arrays.
- Output validation is completed before the query or any write. Capacity may exceed the request and the unused suffix remains untouched; insufficient capacity is an explicit error rather than silent truncation. The physics world synchronizes authored transforms once for the batch, and the common single-ray path now uses a closest-hit Jolt collector when triggers are included instead of allocating an all-hit vector.
- The focused real-Jolt regressions pass **3/3**. A 256-hit call retained the exact dictionary/array objects and measured **55 bytes** of peak Python-traced allocation, proving that output cost does not scale through Python hit objects. The complete physics suite passes **128/128**.
- The rebuilt Windows Release tree passes **75/75 CTest**, including all 15 Vulkan tests. This closes the remaining A03.1 batch-query checkbox and advances strict progress to **24/236 (10.2%)**; the broader engineering/acceptance estimates remain **33–38% / 18–23%** because the unclosed blocks dominate.

## 2026-09-10 — Fixed-step Compute dependency and CPU/GPU exchange closed

- Unified lifecycle Compute recording at the phase boundary. The common scheduler now batches all enabled component callbacks in one fixed/update/late or physics extension phase for both Editor and Player; the redundant per-component nested boundary and the Player-only outer wrapper were removed.
- Preserved the compiler's authoritative external-buffer access analysis instead of re-parsing project kernels. READ/WRITE flags now travel from the trimmed Taichi SPIR-V artifact through `inx.compute` into the RHI dispatch. The ordered compute batch inserts barriers only for actual RAW/WAR/WAW resource hazards; completion and retirement remain owned by the existing compute submission ticket.
- Extended the real Vulkan→Jolt regression to warm JIT separately, dispatch a resident kernel, read only two compact `vector3` feedback ranges, apply the impulses, and advance the same Jolt step. It reports **48 bytes** and **0.4701 ms** steady-state wait on this Windows machine; the post-step body positions match the computed velocities.
- Validation so far: lifecycle/compute unit tests **70/70**, focused real Vulkan compiler/buffer/kernel/mesh/Gizmo/physics tests **6/6**, and a clean Windows Release `_Infernux` plus private GPU compiler rebuild. The access regression proves an in-place position buffer is READ_WRITE while its velocity input is READ_ONLY.
- Final regression for this slice is green: the complete real-Jolt physics suite passes **128/128** and Windows Release CTest passes **75/75**, including all 15 Vulkan-device tests. Focused diffs and the modified compiler fork pass whitespace validation.
- This closes all three A03.3 items and advances strict progress to **27/236 (11.4%)**. Reusable engineering implementation is estimated at **34–39%**, final user acceptance at **19–24%**; the dominant remaining schedule stays approximately **9–13 engineering weeks / 13–18 calendar weeks**.

## 2026-09-11 — A04.1 Mesh evidence re-audit

- Rechecked the public `MeshRenderer` wrapper/stub and native binding: whole-mesh authoring declares `(N,3)` positions/normals, `(N,2)` UVs, `(N,4)` tangents and flat uint32 indices; the binding force-casts contiguous float32/uint32 input and copies it into a new native geometry generation before publishing it for the next renderer update.
- Rechecked derived attributes and collision ownership. Authors may submit or explicitly rebuild normals/tangents/bounds; resident GPU meshes rebuild selected attributes on-device. Visual geometry edits never schedule MeshCollider cooking. `MeshCollider.recook()` snapshots the selected geometry at the existing worker/publication boundary, while invalid main-thread or worker results retain the complete previous collision shape rather than substituting another shape.
- Rechecked identity/lifetime evidence: asset Mesh uses GUID-backed `AssetRef`, immutable `MeshGeometry` generations and runtime dependency edges; inline Mesh uses a monotonic generation rather than per-frame hashing. Render extraction retains the referenced generation, resident compute buffers are held by shared resource leases/submission tickets, and real Vulkan retirement returns residency to baseline after repeated replacement and owner destruction.
- Existing tests covering these contracts remain part of the green **128/128** physics and **75/75** Windows Release runs. The public create/share/copy/submesh/range-edit Mesh resource model, explicit CPU-readable policy and complete Play-clone contract remain open. Four A04.1 boxes are promoted; strict plan progress is now **31/236 (13.1%)**, reusable engineering **36–41%**, and final acceptance **20–25%**.
# 2026-09-11：公开 Mesh 资源模型完成

- `inx.Mesh` 现可显式新建、从数组建立、按路径/GUID 加载及显式复制；运行时 Mesh 进入与导入资产相同的 `AssetRegistry` 版本/依赖发布主干。
- 整块和区间更新支持 position/normal/uv/tangent/color，submesh 与材质槽一次提交；公开顶点/索引 NumPy buffer，法线/切线保持显式重算语义。
- `MeshRenderer.mesh/shared_mesh` 共享同一资源，getter 不隐式复制；复制后拥有独立身份和版本。
- Windows Release 原生模块编译通过；新增公开 Mesh 测试 `2 passed`，Mesh/AssetDatabase/MeshCollider 组合回归 `58 passed, 1 skipped`。

## 2026-09-11 — 运行时 Mesh 生命周期收口

- 新增显式 `inx.Mesh.destroy()`；只接受运行时 Mesh 身份，并通过既有 AssetRegistry/依赖事件解除所有 Renderer 引用，不给导入资产增加危险的删除旁路。
- 新增真实 PlayMode 场景重建回归，确认进入与退出 Play 后仍共享同一运行时 Mesh；显式 copy 保持独立资源身份，普通对象 clone 不产生隐式 Mesh 副本。公开 Mesh 集中回归为 **3/3**。
- 新增 Vulkan 设备测试：两个 Renderer 已持有同一 Mesh 且存在在途帧时销毁资源，GPU cache 沿 completion retirement 回收，目标 GUID 的驻留、待上传和 Renderer 引用全部清空，未使用 `vkDeviceWaitIdle` 伪装安全。
- Windows Release 完整 CTest 通过 **76/76**，其中 **16** 项使用真实 Vulkan 设备；AssetDependencyGraph 原有缺失依赖语义也单独回归通过。
- A04.1 生命周期项完成。严格进度为 **33/236（14.0%）**，可复用工程实现约 **38%—43%**，最终用户验收约 **22%—27%**。剩余仍估计为单人全职 **9—13 个工程周**，包含性能长尾、四平台与发布验收的保守日历为 **13—18 周**。

## 2026-09-11 — World AudioListener 选择确定化

- 修正 Listener 备用选择依赖无序集合的问题：当前活动 Listener 保持所有权，禁用或销毁后按稳定 GameObject ID 选择下一位；新增 Listener 不会让听音位置随加载时序偷偷切换。
- 距离衰减由回调外独立实现迁到唯一的 `AudioMixer::DistanceAttenuation`，覆盖最小距离、线性区间、最大距离和退化范围；Python 侧增加只读活动 Listener owner ID 用于工具和验收。
- 音频相关真实组件测试 **8/8**，完整 Windows Release CTest 再次通过 **76/76**（16 项真实 Vulkan）。由于 A02 Additive Scene 还未完成，本轮不把“多场景单 World 音频”冒充已验收，严格进度保持 **33/236（14.0%）**。

## 2026-09-11 — A02 同 World 多场景主链形成

- SceneManager 的 Play/Update/FixedUpdate/LateUpdate/EndFrame、Transform 帧缓存、物理发布和原生组件注册表现在统一覆盖全部 loaded Scene；切换 active Scene 不再清空固定步累积，也不会暂停其它 Scene。active Scene 只承担默认作者归属、环境和主视图选择。
- 新增公开 `LoadSceneMode.SINGLE/ADDITIVE`、loaded Scene 查询与 Player prepare/publish 路径。Additive 提交不清空其它 Scene 的 Python/native 注册表；成功后仅启动新 Scene，失败或取消只卸载本次目标。
- 同一 authored Scene 可同时驻留多份。原生事务在提交前发现其它 World 中占用的 GameObject/Component ID，分配一次权威身份并由 commit token 将 GameObject 映射传给 Python 字段恢复；`GameObjectRef/ComponentRef` 现在遍历全部 loaded Scene，而不再只看 active/persistent Scene。
- 多 Camera 收集扩展到全部 loaded Scene，并沿单场景规则按 `depth`、再按稳定 Component ID 排序；灯光继续使用已有 World 注册表。原生集中回归 **3/3**，场景加载/Player 单元测试 **36/36**，真实公开 Additive 文件加载、跨语言引用与跨场景物理集中回归 **3/3**。
- 新增跨场景 Listener 回归：切 active Scene 不替换当前 Listener，禁用/卸载拥有者后才按稳定 ID 提升备用者。原生身份/灯光 soak 连续执行 32 轮 Additive 提交与卸载，A 的对象和注册表始终保留；Python 跨场景引用在 B 卸载后明确失效，不会误绑 A 中同名或相似对象。结合已有 persistent Scene 回归，A02 的“卸载与常驻/引用语义”项完成。
- A02 仍缺 RenderStack 实际出图、编辑器多场景选择/保存/Undo 和完整 Player 实机加载门槛。当前组合回归为 Python **374/374**、Windows Release CTest **77/77**（16 项真实 Vulkan），附加 World Listener 集中测试 **4/4**。严格进度为 **34/236（14.4%）**；工程底座约 **41%—46%**，最终用户验收约 **24%—29%**，剩余仍约 **9—13 个工程周 / 13—18 个自然周**。

## 2026-09-11 — A02 Player 事务与活动场景作者归属闭环

- 为 Player 添加真实 Cook 目录回归：`PlayerRuntimeAssetCatalog` 解析 GUID/runtime artifact，经 `PlayerSceneService.request_prepared_load(..., mode="additive")` 在 Play 中提交第二个 Scene；原活动 Scene、对象与 active 选择不变，新 Scene 可从 loaded 目录取得并独立卸载。
- SceneManager 新增唯一的 World 对象查询与根对象跨 Scene 移动主链。移动保留 GameObject/native/Python component 身份与层级，不重建对象；非根对象明确拒绝，避免跨 Scene 父子所有权不一致。公开 `GameObject.find`、tag/layer/ID 查询复用该主链。
- Hierarchy 创建服务增加双驻留 Scene 回归：active Scene 是唯一默认创建目标，切换 active 不会销毁或暂停另一 Scene。集中测试 **13/13** 通过。
- A02 的活动场景作者归属与 prepare/activate/residency/cancel/loaded-directory 两项完成。严格进度为 **36/236（15.3%）**；工程底座约 **42%—47%**，最终用户验收约 **25%—30%**。剩余工期仍约 **9—13 个工程周 / 13—18 个自然周**，下一主线是多场景编辑器选择、分 Scene 保存/Undo 与 RenderStack 实际出图。

## 2026-09-12 — Play-time Single 场景替换可逆化

- Play 边界现在记录每个 resident Scene 的作者名称、原生快照、文档身份、资源路径和 revision。运行期 `LoadSceneMode.SINGLE` 可继续销毁其它 native Scene，不需要引入第二套 Scene 容器。
- Stop 使用一条确定性恢复路径：先卸载进入 Play 后新增的 runtime-only Scene，按原 world identity 复用仍存活的 Scene，为被 Single 销毁的作者 Scene 创建新 World，然后逐一提交原生快照。
- `SceneFileManager` 在所有快照成功发布后一次性替换 Scene-to-document 映射，恢复原活动场景、文档 revision 与 dirty 状态；旧 native 指针不会留在作者目录中。
- 新集成回归覆盖两个作者场景进入 Play、Additive 场景被销毁、运行期临时场景创建及 Stop 后完整恢复。Play/脚本事务/场景/MCP 集中测试为 **272/272**。
- 严格计划进度保持 **38/242（15.7%）**；该工作补强已勾选的 A02 Play clone 项，不重复增加完成项。可复用工程实现约 **46%—49%**，最终用户验收约 **27%—31%**，剩余约 **6—9 个工程周**。

## 2026-09-12 — Editor Single 多文档关闭语义统一

- Open Scene 与 New Scene 不再只查看活动 Scene 的 dirty 位；它们把全部 resident Scene 文档交给已有 `CloseCoordinator`，由同一个弹窗依次完成 Save/Discard/Cancel。
- 成功提交新文档后，Editor 使用唯一 Single 主路径卸载所有非活动 Scene 及其绑定。Additive 编辑不再因普通 Open/New 留在当前 World。
- 新回归覆盖多脏文档顺序确认、SceneFileManager 请求范围，以及真实 Scene 提交后其它 World 和作者绑定一并退休。A02 集中回归为 **295/295**。
## 2026-09-12 — A02 multi-Camera / RenderStack visible proof

- Added `infernux.scene.active.set` as a thin MCP operation over the authoritative Editor Host and SceneManager resident-scene activation path; it does not introduce a second scene model.
- In the visible `Infernux041Lab` editor, loaded `01_XPBD_Jelly` and `06_SharedMeshPublication` together, then temporarily authored a Camera and RenderStack in the second scene.
- Captured two 1920×1080 Game targets: with the second Camera enabled (`DepthOnly`, depth 10) the same combined World was redrawn from its distinct view; disabling that Camera immediately restored the primary view. This proves actual output ordering rather than only registry state.
- Deleted both temporary objects, discarded the temporary scene revision, and returned the project to its original single-scene state.
- Combined with existing stable Camera sorting, active-scene environment ownership, cross-scene Light registration, and deterministic World AudioListener tests, checked the A02 Camera/RenderStack/Light/environment/AudioListener rule item.
## 2026-09-12 — A05 B04 public CPU JIT author/build path

- `@inx.jit.compile` is now recognized by the same typed-HIR build transform as the legacy `njit`/CPU `hpc` declarations. Supported static import forms and the bare decorator all embed the verified parallel implementation into ordinary Player bytecode; `auto_parallel=False` remains an explicit serial choice.
- Candidate-script policy admits `jit.compile` only in a declaration decorator. The ordinary Python `compile()` dynamic-code call is still rejected and cannot borrow this permission.
- CPU `inx.buffer` remains zero-copy NumPy storage for Numba. Public `vector2`/`vector3`/`vector4` values now cross the JIT boundary as typed float32 records with named fields and return as engine vector values; GPU buffers remain explicitly rejected in favor of `inx.compute.launch`.
- Migrated the Lab's legacy CPU XPBD sample declarations from `compute.hpc(device="cpu")` to `jit.compile` without restarting the visible editor; no new Console errors were published.
- Expanded JIT/compute/candidate-policy/Player-build regression: `151 passed`.

## 2026-09-11 — A11 World audio and Bus fade closure

- Completed the fixed `Master/Music/SFX/Ambience/UI` Bus envelope path in `AudioEngine`: frame-time interpolation, explicit cancellation and status observation are exposed through the native Python surface. Direct volume writes cancel the active envelope; mute only affects gain and does not seek, pause, recreate, or otherwise move a live voice cursor.
- Corrected the new integration test rather than weakening the runtime: a newly published Scene intentionally consumes one zero-delta boundary frame, so the test now consumes that boundary before measuring the authored fade. The production Scene transition timing remains unchanged.
- Combined the already completed Additive Scene World with deterministic Listener ownership. Active/additive Scene selection no longer changes the listening origin; disabling or unloading the owner promotes the stable-ID successor.
- Focused Python audio integration passed **10/10**. Native `infernux.audio_mixer` and `infernux.audio_clip` passed **2/2**, followed by a complete Windows Release CTest pass of **77/77**, including all 16 real-Vulkan tests.
- Reopened the real visible `D:/Users/Chenlizhe/Desktop/Infernux041Lab` editor with the rebuilt Release module. MCP entered Play in **677.4 ms**, the Console reported **0 warnings / 0 errors**, and MCP returned to Edit in **74.0 ms** without dirtying the project.

## 2026-09-11 — Resident Mesh capacity becomes a real Vulkan contract

- `MeshRenderer.create_vertex_buffer(capacity=...)` now separates the canonical Vertex allocation capacity from the authored Mesh's effective `vertex_count`. The actual GPU buffer is allocated for the requested capacity; rendering and automatic normal/tangent reconstruction continue to use only the authored range.
- The public binding accepts only `float32 (capacity, 23)` storage and rejects capacity below the Mesh count. Native scene and Vulkan boundaries independently require a canonical-stride allocation with at least the effective range, so bypassing the Python wrapper cannot produce an out-of-bounds draw.
- Added an observable `vertex_buffer_capacity`. The real Vulkan regression reserves 64 spare vertices, deforms the live sphere and rebuilds its normal/tangent frame while proving the spare suffix remains untouched.
- Windows Release rebuilt successfully. Focused Mesh tests pass **23/23** and the real Vulkan `infernux.compute_mesh_gpu` test passes **1/1**. This is a genuine slice of A04 capacity semantics, but the checklist remains **43/246 (17.5%)** because index capacity, dynamic effective-range changes and actual CPU-readable storage release are still open.

## 2026-09-11 — A12 cursor ownership and teardown closure

- Cursor visibility, relative lock, confinement and hardware warp remain independent public states on the one `InputManager` path. Logical Game View coordinates and screen pixel ratio are covered by the existing Python/native tests; the native warp path consumes the matching SDL motion instead of publishing a false gameplay delta, while Web/semantic hosts return unsupported.
- Deactivating or destroying the Game View now releases gameplay focus and cursor lock. Stop performs the release immediately after stopping gameplay, so a hidden dock cannot strand the editor cursor. Native window shutdown restores unlocked, unconfined and visible state before clearing the stored SDL window and destroying it.
- Focused Python input/Game View/Play Mode regression passes **97/97**. The focused native input test passes **1/1**, and the complete Windows Release CTest passes **77/77**, including all 16 real-Vulkan tests.
- The visible `D:/Users/Chenlizhe/Desktop/Infernux041Lab` editor entered Play through MCP in **679.5 ms**, ran for more than 24 seconds, returned to Edit in **72.5 ms**, and then exited normally through an actual window-close event with process code **0** and no shutdown error. Both A12 cursor items are complete; strict progress is **45/246 (18.3%)**.

## 2026-09-11 — A12 Tween/Sequence runtime service closure

- Added the public pure-Python `inx.Tween` / `inx.Sequence` timeline with the Runner Long value surface, easing set, sequence composition, Restart/Yoyo/Incremental loops, scaled/unscaled/fixed clocks, target kill and separate complete/cancel terminal states.
- Property paths replace getter/setter closures. Bound callbacks are weak, lifecycle-object closures are rejected at configuration time, and Disable/Destroy, Stop, next Play and Player shutdown cancel without publishing future callbacks.
- Tween phase demand is published into the existing native-backed runtime scheduler. A scene containing only a `start()` tween no longer needs a fake empty `update()` method, and the demand disappears when the last timeline terminates.
- Focused Tween/lifecycle/Play/Player/public-namespace regression passes **109/109**. The full Python run reached **5714 passed, 13 skipped, 25 failed**; its one new namespace failure was fixed and rechecked, while the remaining failures belong to other currently open 041 slices and prevent claiming a green aggregate run.
- Created the real `07_TweenTimeline` scene in the visible `D:/Users/Chenlizhe/Desktop/Infernux041Lab` project through MCP. The timeline moved the Sphere from authored `y=0.75` to observed `y=2.40977`, a 1920×1080 Game render-target capture completed, Stop restored `y=0.75`, and Console remained at **0 warnings / 0 errors**.
- Both A12 Tween items are complete. Strict progress is **47/246 (19.1%)**; reusable engineering implementation is about **52–55%**, final user acceptance about **33–37%**, with approximately **6–9 full-time engineering weeks** remaining before the complete 041 gate.

## 2026-09-12 — Full Python regression restored after World migration

- Fixed `RenderStack.on_after_deserialize()` to normalize an unbound document without accessing the public `game_object` property or publishing scene ownership. Live ownership still requires the component's actual Scene; no active-scene fallback was restored.
- Updated undo, gizmo, picking and object-execution fixtures to use the authoritative World-wide object lookup introduced by A02. Structural fixtures now carry their owning Scene and recreation target instead of relying on whichever Scene is active.
- Kept native component references behind the shared component-reference field boundary, synchronized the MCP contract with its real 89-operation directory, and made Shader relative-path coverage operate on the native AssetDatabase's actual project `Assets` root.
- The native engine begins with a resident Scene and additive tests may create more. The shared fixture now records the entering World set, unloads every Scene created by the test, permits a tested Single commit to retire an older Scene, and rejects any newly created Scene that survives teardown. This removed the stale `InlineMaterialCube` body that had polluted later Raycast, stack, CCD and pose-readback tests.
- Final full Python verification under `conda activate infernux`: **5741 passed, 13 skipped, 0 failed** in 214.61 seconds. Focused cross-order scene/physics verification passed **7/7** and the Single-commit teardown regression passed **2/2**. This restores the aggregate gate; it does not add a 041 checklist item or claim the remaining platform/demo work complete.

## 2026-09-11 — A01.2 DataAsset authoritative asset slice

- Added one `.inxdata` source document and one native `ResourceType::DataAsset` path through AssetDatabase, importer registration, text loading, metadata round-trip, Python bindings, Project icons and drag payloads. No alternate file format or document-driven module import was introduced.
- DataAsset subclasses require an explicit stable serialized type ID. Deserialization resolves only the published SerializableObject catalog and reuses its strict field/value codecs; malformed keys, unknown types and non-DataAsset types fail at that boundary.
- Added GUID-bound persistence, AssetManager shared identity, `DataAssetRef`, explicit identity-free `instantiate()`, Player write rejection and Assets/Packages authoring boundaries. Project creation enumerates published user DataAsset types and enters the existing Project command/Undo route.
- Focused DataAsset tests passed **5/5**; the combined DataAsset/Project menu/panel set passed **104/104**. The complete Python suite passed **5746**, skipped **13**, failed **0** in 212.53 seconds. Windows Release CTest passed **77/77**, including 16 real-Vulkan tests.
- A01.2 remains open: Inspector editing, Play isolation and save ownership, schema migrations, Cook/Player binary representation, Scene/Prefab round-trips and real gameplay configuration samples are still required. Strict plan progress therefore remains **47/246 (19.1%)**.

## 2026-09-11 — A16 `.blend` composite model import added to 041 scope

- Added a final 041 workstream for treating `.blend` as an editor-side composite model source rather than a runtime file or a single bare Mesh. Dragging an imported model into a Scene must preserve stable object hierarchy, local transforms, pivots, Mesh parts, material slots and the explicitly supported Blender presentation subset.
- Froze the intended mainline as source import → observable Infernux `ModelAsset` and stable subresources → Cooked Infernux binary assets in `Content.inxpkg`. Scene, Prefab, DataAsset, scripts and MCP consume GUID/subresource identities; Player does not parse or ship `.blend`, FBX or GLTF source files.
- The plan now calls out coordinate conversion, Principled BSDF/PBR mapping, embedded and external texture dependencies, skeletal animation, baked-only Blender features, stable reimport identity, user overrides and package auditing. Unsupported Blender semantics must be reported at import instead of silently producing a different result.
- This is newly accepted scope, not completed implementation. It contributes eleven open checklist items; strict progress remains **47/246 (19.1%)**.

## 2026-09-11 — A01.2 DataAsset Inspector, Cook and Play isolation closure

- Added DataAsset to the shared editable-document catalog. Inspector edits recursively cover scalar, list, nested SerializableObject and typed asset-reference fields, then commit through the existing Document/Undo/autosave transaction while preserving the loaded asset object's GUID identity.
- Added one deterministic binary Player representation, `.inxasset`, and staged it at `Library/Artifacts/Data/<guid>.inxasset`. Cook excludes the author `.inxdata` source; the native importer publishes nested GUID dependencies so closure still follows referenced assets without parsing a second runtime graph.
- Added a Play-owned DataAsset cache domain. A GUID resolves to one shared runtime clone during Play, author objects remain untouched, Stop discards the complete runtime domain, and reference-local caches cannot retain objects across the transition. Play clones reject writes to authored files; explicit Inspector authoring stays on the existing document transaction.
- The first complete Python run passed **5753**, skipped **13**, failed **0**. The DataAsset/Play concentrated run passed **47/47**. Four A01.2 combination items were complete at that checkpoint, bringing strict progress to **51/246 (20.7%)**; schema/version migration and the real configuration sample were still open.

## 2026-09-11 — A01.2 real configuration sample and Play-domain repair

- Added a real `RunnerLongConfig` sample in `Infernux041Lab` with nested unit rules, chapter configuration, visual parameters, audio groups and chapter ordering. `serialized_field(default_factory=...)` now constructs only plain registered `SerializableObject` defaults and rejects factories with custom initialization, keeping declaration-time execution bounded.
- MCP now exposes `data_asset` through the existing Project creation transaction and projects inspect/property edits from the shared DataAsset document controller. The installed development `.inxpkg` was rebuilt and updated through `PluginManager`; the live endpoint advertises 91 operations and created `Assets/Config/RunnerLong.inxdata` with GUID `1ca806a99472a33a71727a64539ccf52`.
- Fixed two general script-domain defects found by the real project: a candidate helper loaded before its namespace is now attached to the transaction-private parent, and retiring a module no longer erases immutable schema from class objects still owned by DataAsset documents, Undo history or a retiring runtime epoch.
- Created and saved the visible `08_DataAssetConfiguration` scene through MCP. Its Python component resolved the shared GUID and logged `RUNNER_LONG_CONFIG_READY chapter=chapter-01 speed=7.25 music=0.75 exposure=1.00`; the run had 0 warnings and 0 errors. The component deliberately changed its Play clone, then Stop restored Edit state and the authored asset still reported `move_speed=7.25`.
- Focused candidate import, script retirement, DataAsset and Play tests passed **89/89**; the MCP/package set passed **18/18**. The final-state complete Python suite passed **5761**, skipped **13**, failed **0** in 234.79 seconds. The real sample completes the fifth checked A01.2 item; only schema/version migration remains. Strict progress is now **52/246 (21.1%)**, reusable implementation about **55%—58%**, and final acceptance about **37%—41%**.

## 2026-09-11 — A01.2 versioned DataAsset migration boundary

- Added a positive `__serialized_schema_version__` to SerializableObject/DataAsset declarations and current typed documents. Existing unversioned documents are treated as the one legacy version-zero input; versions newer than the installed type are rejected.
- Migration runs only while loading: current field names and explicit `FormerlySerializedAs` declarations select retained values, new fields receive their declared defaults, removed fields disappear, and the resulting runtime document contains only current names. There is no old/new dual write, runtime alias, import-by-document string or arbitrary migration callback.
- The native DataAsset importer accepts both legacy and versioned author documents. Cook validates through the same Python type/codec path and always emits a current-version `.inxasset`, while leaving an old source `.inxdata` untouched.
- Reopened the visible `D:/Users/Chenlizhe/Desktop/Infernux041Lab` against the rebuilt Release module. Its intentionally unversioned `RunnerLong.inxdata` appeared through MCP as version 1, entered Play with `speed=7.25 / music=0.75 / exposure=1.00`, returned to Edit without changing the author asset, and reported 0 warnings / 0 errors.
- Focused schema/DataAsset/Cook regression passed **46/46**, including nested-object and typed asset-reference migration. The final complete Python suite passed **5763**, skipped **13**, failed **0** in 243.48 seconds; the two subsequently added migration cases passed in the focused set without further production changes. Native AssetDatabase refresh/dependency tests passed **2/2**. A01.2 is now fully checked; strict progress is **53/246 (21.5%)**, reusable engineering implementation about **56%—59%**, and final user acceptance about **38%—42%**.

## 2026-09-12 — A08 Renderer parameter override vertical slice

- Added persistent and runtime `MeshRenderer` parameter overrides backed by immutable draw snapshots. Runtime values win, removing one field reveals its persistent value, and clearing an override returns to the shared material without mutating or cloning that material. The public boundary accepts reflected shader parameters only; shader and pipeline state remain material/pass concerns.
- Added four thin MCP operations for get/set/remove/clear through the same public component API. The development MCP package hot-reloaded from 91 to 95 operations in the already-running editor, proving this Python plugin update did not require an editor restart.
- Real visible Vulkan validation used three objects sharing DefaultLit. The render target showed inherited white, runtime red and runtime green; removing green live restored inherited white while red remained. Console stayed at 0 warnings and 0 errors.
- The first visual run exposed a general cache defect: a content-only camera-cull cache refresh copied updated skin palettes but retained the old renderer parameter block. `SceneRenderer` now refreshes that complete mutable draw payload, and `CameraCullingIsolationTests` protects the contract. No fallback or parallel rendering path was added.
- Focused Python verification passed **48/48** and the native camera-culling regression passed **1/1**. Two A08 items are now complete. Strict progress is **55/246 (22.4%)**; arrays/buffers, multi-writer ownership, World/View/pass precedence, short-lived pass payloads, multi-submesh/batching and full resource-retirement coverage remain open.

## 2026-09-12 — A08 material-slot and runtime-writer ownership

- Replaced the single anonymous runtime override map with explicit owner layers. `script`, MCP and other runtime systems retain independent fields; the newest actual write wins a collision, while removing or clearing one owner reveals the preceding runtime value or authored value. No configurable priority tree or fallback namespace chain was added.
- Material slots retain independent immutable parameter publications. A native two-submesh fixture now uses a legitimately registered runtime Mesh, shares one material across both draws, and proves removing slot 0 does not mutate slot 1 or the shared material.
- Texture overrides validate their GUID at the public assignment/deserialization boundary and contribute one deduplicated runtime dependency set across authored and all owner layers. Builtin textures remain builtin identities rather than fabricated project assets.
- Windows Release native modules rebuilt. Focused Python renderer/MCP verification passed **39/39** and the native camera-culling regression passed **1/1**. The material-slot/writer-ownership A08 item is complete; strict progress is **56/246 (22.8%)**. Array/buffer parameters, World/View/pass precedence, short-lived pass payloads, batching performance and the complete texture-retirement matrix remain open.
- Renderer-local texture dependencies now receive the same asset generation event as material-owned textures. An effective texture override republishes a new immutable block while preserving its GUID; the superseded descriptor and texture lease remain on the existing GPU retirement queue. Six focused material/texture native tests and the complete Windows Release CTest suite passed **77/77**, including all 16 real-Vulkan tests.
- Rebuilt the bundled `infernux.mcp.inxpkg` from the updated source (`0.1.1`, 843,776 bytes) and verified its package contract. The install/uninstall/restart subprocess test passes with the authoritative 95-operation catalog.
- Reopened the visible `D:/Users/Chenlizhe/Desktop/Infernux041Lab` editor and exercised `08_DataAssetConfiguration` through MCP. Enter Play completed in **12.59 ms**, exit in **10.65 ms**, Console remained **0 warnings / 0 errors**, and a 240-frame observation reported complete-frame average **0.427 ms**, P95 **0.677 ms**. This is lifecycle evidence only and is not counted as the A03.4 jelly performance gate.
- The first complete Python suite reached **5770 passed, 13 skipped, 1 failed**; the sole failure was the stale expected operation count `91` after the already-delivered four renderer operations. After updating that test to the package/runtime truth of `95` and rebuilding the bundled package, the final complete suite passed **5771**, skipped **13**, failed **0** in **250.35 seconds**. No production fallback was added.

## 2026-09-12 — A08 explicit shader parameter domains

- Fixed the public binding contract to five explicit domains: Material/Renderer descriptors use set 0; World environment is resolved into each immutable RenderView snapshot and uses set 1; engine-frame and instance state use set 2; pass-local values use graph parameter blocks or push payloads; the bindless texture table remains set 3. There is no search-through-all-namespaces fallback chain.
- Removed `CommandBuffer.set_global_*` and `ScriptableRenderContext.set_global_*`. Those methods only populated temporary dictionaries that no shader binding ever consumed, while the command registry falsely reported them as implemented. Deleting the false surface is the fire-forced fix; no replacement compatibility dictionary was added.
- Added named descriptor-set constants in `ShaderProgram` and negative public-surface coverage proving the dead global API cannot return. Focused renderer/shader Python regression passed **120/120** and focused native shader/material/render-graph verification passed **6/6**.
- The complete Python suite passed **5772**, skipped **13**, failed **0** in **302.98 seconds**. The complete Windows Release CTest suite, run independently from Python load, passed **77/77** including all 16 real-Vulkan tests; the earlier residency fluctuation did not reproduce outside the concurrent stress run.
- Reopened the real visible `D:/Users/Chenlizhe/Desktop/Infernux041Lab` project with the rebuilt Release module. Scene `08_DataAssetConfiguration` entered Play in **11.92 ms**, ran for over 34 seconds with the Game View rendering, loaded the shared configuration asset, returned to Edit in **13.91 ms**, remained clean, and ended at **0 warnings / 0 errors**. The editor was then closed through normal window-close messages.
- The explicit-domain A08 item is complete. Strict progress is now **57/246 (23.2%)**; array/buffer parameters, captured short-lived pass payloads, batching performance and the complete texture-retirement matrix remain open.

## 2026-09-12 — A08 submission-time parameter capture

- Render-graph Effect parameters now have one explicit publication boundary. Python continues to update the live revisioned parameter blocks without rebuilding topology; `ScriptableRenderContext::SubmitCulling()` snapshots the changed generation together with the submitted renderer list, and the later Vulkan recording callback reads only that submitted snapshot. A value changed after submission therefore cannot rewrite commands that are already part of the frame.
- The implementation uses monotonic generations rather than per-frame hashes or unconditional copies. An unchanged generation reuses the preceding immutable submission, matching the fire-forced rule of one normal path without speculative recovery.
- Focused Python Effect/RenderStack coverage passed **196/196** and the complete Python suite passed **5772**, skipped **13**, failed **0** in **318.31 seconds**. The complete Windows Release CTest suite passed **77/77**, including all **16** real-Vulkan tests.
- Reopened the visible `D:/Users/Chenlizhe/Desktop/Infernux041Lab` project and corrected its stale author fixture by removing the retired read-only `Light.shadowBias` and `Light.shadowNormalBias` fields from `Start.scene`; no compatibility fallback was added to the engine. The authored Start scene then opened normally with its 21-pass Bloom + ACES stack.
- In Play, changing ACES exposure from `1.0` to `0.1` changed the 1920x1080 render capture from 154,520 bytes to 100,258 bytes; restoring `1.0` reproduced the original 154,520-byte capture. Stop returned to a clean document and Console remained at **0 warnings / 0 errors**. Bloom and ACES source values were restored to `0.8` and `1.0` before exit.
- This proves submission capture for graph-owned fullscreen Effect payloads, but does not yet provide the public per-draw payload for geometry passes. The corresponding A08 checklist item therefore remains open and strict progress stays **57/246 (23.2%)**.

## 2026-09-12 — B01 authoritative buffer description

- Added an immutable public `BufferDescription` for the existing `inx.buffer` mainline. It freezes logical shape, scalar/vector layout, lane offsets, stride, capacity, effective byte range, device, usage, ownership, wrapper generation and opaque RHI resource identity/generation without exposing a Vulkan handle or host address.
- GPU identity comes from the engine-owned `BufferHandle`; CPU buffers do not fabricate one. Closing a buffer invalidates the resource while an already-published description remains an immutable value snapshot.
- Confirmed that `inx.buffer` remains runtime storage rather than an asset or `serialized_field` value: the shared value codec rejects it instead of serializing a pointer/resource identity. Persisted reconstruction must continue through an authored GUID asset or explicit data.
- Focused public compute verification passed **45/45**. The real Windows Vulkan buffer test passed **1/1**, including stable resource identity and a 12-element `vector3` round trip. The Release build completed successfully.
- Two detailed B01 contract items are now complete. Borrowed views, shared Mesh ownership, owner destruction/layout changes and complete in-flight retirement remain open, so the main A05/B01 gate and strict plan progress remain **57/246 (23.2%)**.

## 2026-09-12 — B01 borrowed ranges and shared native ownership

- Added public `Buffer.view(offset, count)` as a no-copy borrowed range. CPU views share the same NumPy allocation; GPU views retain the same engine RHI identity and carry a byte range in the compiler argument ABI rather than allocating or copying another resource.
- Modified the trimmed Taichi SPIR-V lowering so descriptor-backed external pointers add the engine-provided byte offset before element addressing. This is the one kernel path used by ordinary buffers and views; there is no alternate copy path or descriptor-alignment fallback.
- Focused public compute verification passed **46/46**. The real Vulkan kernel test proved an offset view changed only elements 17—39 of a 257-element allocation and stayed usable after closing the owner wrapper.
- Added a real render-lifetime assertion: after `MeshRenderer` borrows a resident vertex allocation, closing the author-facing `Buffer` wrapper leaves subsequent frames drawable; explicit renderer unbinding then releases that ownership. The updated real Vulkan Mesh test passed **1/1**.
- The complete Windows Release native suite passed **77/77**, including all **16** real-Vulkan tests. The complete Python suite passed **5775**, skipped **13**, failed **0** in **300.22 seconds**. At this checkpoint, capacity/layout replacement and the complete in-flight consumer matrix still remained open, so the ownership detail item, A05/B01 gate and strict progress remained **57/246 (23.2%)**.

## 2026-09-12 — B01 exact-ticket compute binding retirement

- Replaced permanent strong ownership in native compute bind-group caches with weak cached identities plus an exact in-flight ownership lease. A successful direct or batched dispatch attaches its returned submission ticket and strong buffer set; collection releases those references only after that ticket completes and then destroys binding groups whose resources no longer have an owner.
- Closing a GPU `Buffer` removes its prepared Python launch entries and schedules only the affected native kernels for bounded frame-boundary collection. There is no device-idle wait, hash validation, replacement allocation or alternate submission path.
- Extended the real Vulkan public-kernel regression with resource generations of 7, 19, 41 and 83 elements. Each generation is launched, closed while submitted, waited by its exact kernel ticket and observed to return the native binding cache to the same baseline rather than growing across layout/capacity replacement.
- The focused public compute suite passed **46/46**; the focused real Vulkan upload/buffer/kernel/Mesh group passed **4/4**. This completes the detailed B01 ownership/retirement item. At this checkpoint the read-only and empty-shape acceptance cases still remained, so the aggregate gate stayed open.

## 2026-09-12 — B01 closure: read-only bindings and empty-shape policy

- Added an explicit read-only borrowed-view contract. CPU writes fail at the public buffer boundary; compiler-declared GPU write/read-write access fails before native submission, while a read-only buffer used by a read-only kernel binding executes normally.
- Fixed empty shape behavior as an explicit positive-capacity error. Infernux does not create a special zero-byte Vulkan allocation or infer a replacement dispatch domain.
- Focused public compute coverage passed **47/47** and the updated real Vulkan public-kernel test passed **1/1**. The final complete Windows Release suite passed **77/77**, including **16/16** real-Vulkan tests; the final complete Python suite passed **5776**, skipped **13**, failed **0** in **287.17 seconds**. B01 is now checked. Strict whole-plan progress advances to **58/246 (23.6%)**; B02 resource interoperability and staging/async performance remain separate work.

## 2026-09-12 — B02 resource-exact graphics dependency and async readback

- Replaced the resident-Mesh graphics wait on the latest global background-compute submission with a dependency carried by each `ComputeBuffer`'s exact last-write ticket. Vulkan resolves only an in-flight ticket to its timeline/value; completed writes need no semaphore wait.
- Added frame evidence for the resident write serial, latest background-compute serial and pending wait. The real Vulkan Mesh test submits unrelated compute after every resident deformation and proves the two serials diverge while rendering remains valid.
- Added `Buffer.get_data_async()` and an explicit `Readback`: each request owns its staging allocation and retains the source/host until its ticket completes. Polling `done` does not block; `get_data()` is the completion point and returns a CPU Buffer. Existing synchronous/batched `get_data()` behavior is unchanged.
- Focused CPU compute tests passed **48/48**; real Vulkan kernel and resident-Mesh tests passed **2/2**. The subsequent complete Windows Release CTest suite passed **77/77**, including all **16** real-Vulkan tests. B02's exact dependency and async-readback detail items are complete, but the aggregate B02 gate remains open pending candidate baselines, Texture lifecycle review and complete transfer/wait statistics. Strict whole-plan progress remains **58/246 (23.6%)**.

## 2026-09-12 — B02 bounded compute statistics

- Added one cumulative queue statistics window for Python/native calls, dispatches, submissions, upload/readback requests and bytes, staging allocations, host maps, CPU submit/wait time and current in-flight submissions. Reading the window performs only the queue's existing non-blocking collection.
- GPU duration remains opt-in: `compute.set_profiling_enabled(True)` creates timestamp resources at an explicit drain boundary; `compute.statistics()` then reports the last collected GPU duration and serial. No default per-frame query, logging, content scan or hash was added.
- The real Vulkan public-kernel regression proves every counter category, default profiling-off behavior and an explicitly enabled timestamp result. Kernel, resident-Mesh and GPU→Jolt physics exchange tests passed **3/3** after the first statistics slice; the extended kernel test passed again after completing timing and boundary coverage.
- The detailed statistics item is complete. B02 now has three of five detailed contract items complete; candidate baselines and the Texture/RenderTexture resource boundary remain open, so strict whole-plan progress stays **58/246 (23.6%)**.

## 2026-09-11 — B05 compiler-only native split

- Removed the retired Taichi engine-device adapter, project-plugin manifest/preload/provider surface, author-owned field/ndarray/texture containers, ArgPack/sparse Python branches and direct Python kernel-launch path. The retained private frontend can build compiler IR but cannot allocate or execute Taichi-owned resources.
- Replaced `GfxProgramImpl` with a device-free `InfernuxCompilerProgramImpl` that supplies only SPIR-V compilation, Vulkan target capabilities and the parameter-layout contract. The compiler subbuild no longer generates or links Taichi `gfx_runtime`, `gfx_program_impl`, kernel launcher or SNode tree manager; devices, buffers, queues, submission and synchronization remain exclusively in Infernux RHI.
- The seven real Windows Vulkan Buffer/Kernel/Mesh/Physics/resource-retirement tests passed **7/7** after the split. The visible `D:/Users/Chenlizhe/Desktop/Infernux041Lab` project compiled and ran on the new path; a 240-frame steady window averaged **0.780 ms**, P95 **1.160 ms**, with **0 warnings / 0 errors** and a successful 1920x1080 Game render-target capture.
- Cold frontend compilation still blocks the first Play for tens of seconds. B05 remains open until the Program/SNode/pybind surface is minimized and compiler preparation/prewarm, wheel/Player packaging, clean-install and platform validation are complete. Plugin-to-external migration and removal of the old plugin distribution entry are complete; strict combined progress is **92/324 (28.4%)**.

## 2026-09-11 — B05 Program resource-owner and pybind surface removal

- Removed `Program` ownership of devices, allocations, ndarray/ArgPack/Texture resources, SNode trees, materialization, synchronization and kernel execution. `LaunchContextBuilder` now retains only scalar and external-buffer argument packing required by `inx.buffer`.
- Removed the matching native Python exports: SNode/Axis/SNodeTree, field author helpers, Ndarray/ArgPack/Texture, Taichi Mesh author/runtime objects, device allocation, host helpers and crash/thread/debug test entries. CMake excludes the matching runtime/container/profiler/SNode-host-accessor sources from the active compiler target.
- The compiler-only `ProgramImpl` interface now contains only the compiler, target capabilities and parameter-layout services; runtime rejection stubs and the kernel-launcher owner were removed rather than retained as dead virtual branches. The disabled upstream `KernelCompilationManager` source directory and its `ticache` files, locks, SHA file keys, cleaner and disk serialization were removed from the fork and active build/call graph. `CompileConfig`, `Function` and `Kernel` no longer carry offline-cache-only flags, serialization or keys, and the offline-cache analysis/file-management sources are no longer built or retained. The existing bounded Python specialization/executable cache remains the single in-memory reuse path. The official main-tree `infernux_gpu_jit_compiler` target rebuilt and installed the compiler into wheel staging without a manual file move. The private native module decreased from **10,637,824** to **9,638,912 bytes**, a **9.4%** reduction for this cut.
- CPU compute/JIT tests passed **64/64**. Real Windows Vulkan Buffer, resident Gizmo, public Kernel, resident Mesh, GPU-to-physics exchange, Mesh publication and two resource-retirement tests passed **8/8** using the official staged compiler.
- Reopened the visible `D:/Users/Chenlizhe/Desktop/Infernux041Lab` editor with the final official payload. `02_InteractiveSnow` entered Play in **175.7 ms** and remains visibly running with **0 warnings / 0 errors**. The previous official cut's latest 240-frame window measured full-frame average **1.522 ms**, P95 **2.791 ms**, game-only average **0.691 ms**; neither number is substituted for the separate controlled rigid/soft acceptance matrix. A 1920x1080 engine render-target capture completed earlier in the same validation sequence.
- Strict progress remains **92/324 (28.4%)** because bounded engine-owned compiler disk cache/preparation, the remaining Python/IR dependency closure, final wheel/Player clean-install and multi-platform validation are not complete. Reusable implementation maturity is now estimated at **66%—68%** and end-to-end acceptance at approximately **45%—47%**.

## 2026-09-11 — B05 engine preparation and project-owned GPU artifact cache

- Added `inx.compute.prepare(kernel, params)` as the engine-owned preparation boundary. It compiles one typed specialization and creates native RHI pipelines without uploading argument values or dispatching authored work. Automatic resident-mesh normal/tangent rebuilding is prepared when its topology buffers are created; the 041 snow component prepares its two authored kernels in `start`, before the first interaction.
- Exported the compiler's exact argument layout and moved argument-block packing into Infernux. `CompilerArtifact` no longer retains a Taichi `Program` or `Kernel`; `LaunchContextBuilder`, kernel launch-context/return helpers, the matching pybind surface and source files were removed. The official private native module is now **9,538,048 bytes**, down **10.3%** from the pre-cut 10,637,824-byte module.
- Added one bounded Infernux `.inxgpu` artifact format containing SPIR-V, task/binding metadata, parameter layout and execution-domain identity. Editor artifacts live with the project's other derived products at `<project>/Library/Artifacts/Compute`; only Player runtime recompilation writes below that game's `Application.persistent_data_path()/Cache/Compute`. Both stores are bounded to 128 files/256 MiB. The key covers kernel AST, buffer schema and Vulkan compiler target; it never hashes runtime arrays or creates Taichi `ticache`/home-directory state. An existing artifact is trusted as the deterministic main path; malformed artifacts report an error instead of silently returning to another cache implementation.
- A real Vulkan regression removed the process-local executable, made private compiler initialization fail deliberately, and rebuilt the Infernux pipeline from the project artifact. CPU compute/JIT passed **66/66**; all **8/8** Windows Vulkan Buffer/Gizmo/Kernel/Mesh/Physics/resource-retirement regressions passed. After moving the Editor store into `Library`, the visible 041Lab scene created **3 files / 29,674 bytes** there; its first Play took **757.7 ms**, and a full editor close/reopen reduced the same Play to **183.9 ms** without changing the file set. The console remained at **0 warnings / 0 errors**. The obsolete test-only `<project>/Cache/Compute` directory was sent to the Recycle Bin, and the known Taichi cache locations on C: do not exist.
- The preparation/lifecycle and bounded device-free JIT artifact-service checklist items are now complete. Focused CPU Compute/JIT verification after the `Library` correction passed **125/125**, and the public Vulkan kernel regression passed again. Strict combined progress is **94/324 (29.0%)**. Remaining B05 work is the Infernux-owned typed HIR/frontend, further Python/IR closure reduction, clean wheel/Player installation and target-platform validation; reusable implementation maturity is estimated at **69%—71%**, with end-to-end acceptance at **46%—48%**.

## 2026-09-11 — B05 clean Windows wheel installation

- The formal `package_python` target produced and verified `infernux-0.4.0-2-cp313-cp313-win_amd64.whl` (**24,639,774 bytes**). Its installed private compiler payload contains **83 files / 10,508,610 bytes**; there is no separately installed top-level `taichi` package.
- Installed the wheel and ordinary declared dependencies into a new isolated venv, with source-tree `PYTHONPATH` and compiler overrides removed. `Infernux` imported from that venv's `site-packages`, compiled a fresh `@inx.compute.kernel`, executed it through the real Vulkan RHI, and produced the expected buffer result.
- The clean project produced exactly one **2,434-byte** `.inxgpu` artifact under its own `Library/Artifacts/Compute`; it did not create a repository-level cache, Taichi `ticache`, or user-home compiler state. This completes the clean-wheel B05.3 item. Strict combined progress is now **95/324 (29.3%)**; Windows Player and target-platform packaging remain separate open evidence.

## 2026-09-11 — B05 compiler failure boundary

- Added the public `ComputeCompilerError` for an incomplete or uninitializable private Python-to-Vulkan compiler payload. `ComputeCapabilityError` remains reserved for the active renderer/device, `KernelCompilationError` for authored kernel lowering, and `ComputeExecutionError` for a valid command failing at submission or execution. The compiler path performs no retry, plugin discovery, pip installation or self-repair.
- Rebuilt the formal Windows wheel and force-installed it into the clean venv. The installed API exposes the new exception and the real Vulkan compile/dispatch smoke still passes without source paths or a top-level Taichi installation. CPU Compute/JIT tests pass **125/125**; ordinary plugin Python site publication, preload index, candidate registration and import-transaction tests pass **63/63**.
- Native compiler modules remain process-lifetime modules and are not hot-unloaded during engine resource retirement; ordinary project plugins retain their existing transactional Python refresh. This completes the B05.3 failure/lifecycle item. Strict progress is **96/324 (29.6%)**, implementation maturity is estimated at **70%—72%**, and Player/manylinux/platform evidence remains open.

## 2026-09-12 — A06 Canvas-free world UI Vulkan vertical slice

- Added a third native UI command list for world geometry. Each Canvas-free UI root records one ordinary local-to-world matrix, local pixel origin and pixels-per-unit value; ImDrawList vertices are converted into a three-dimensional vertex stream and rendered by a depth-tested Vulkan pipeline in an explicit `_WorldUI` RenderGraph pass.
- Fixed the first live failure at its source: the pass initially received only the Camera view matrix, which put all world UI outside clip space. It now receives the current jittered `Projection × View`; no screen-projection fallback was added.
- Updated hierarchy creation so an explicit or selected ordinary scene parent remains authoritative even when the Scene also contains exactly one screen Canvas. Automatic use of that sole Canvas now applies only when there is no parent context.
- Windows Release `_Infernux` compiles. UI/RenderGraph/runtime focused tests pass **101/101**, and hierarchy/runtime-world tests pass **25/25**. In the visible 041Lab editor, MCP created an Empty → Frame → Button hierarchy with no Canvas; engine Game captures showed the button and text following ordinary Transform translation, scale and rotation. A placement behind an opaque rigidbody removed it from the capture, confirming that the retained Scene depth participates. Existing Screen UI remained visible and the Console reported **0 warnings / 0 errors**.
- This closes only A06.2's local-2D-to-world-geometry item. World pointer mapping, rotated clipping/masks, transparent ordering, explicit culling/billboard policies, Prefab/save/Play closure, font comparison and platform implementations remain open. Main-plan strict progress is **66/255 (25.9%)**; main plus Compute appendix is **102/331 (30.8%)**.

## 2026-09-12 — A06 world input and rotated local clipping

- Added one shared runtime input-surface model for Canvas-free UI. The active Camera ray is transformed into the root object's local plane and converted with the same root rect and pixels-per-unit used by rendering; the existing pointer/focus/drag event processor remains authoritative. Screen canvases sort above world surfaces, world surfaces sort by ray distance, and one physics ray on pointer mapping rejects surfaces behind an ordinary foreground collider. Captured drag/release keeps the original surface instead of changing ownership when the pointer leaves it.
- Reused the existing UI ancestor-clip result for world rendering, but did not reuse screen scissor. World vertices now carry local pixel coordinates and a dedicated fragment stage discards pixels outside the current draw command's local clip rectangle, so the clip rotates with the ordinary Transform. No RenderTexture, stencil fallback or second layout system was introduced.
- Fixed Scene integration at its ownership boundary: the existing screen UI renderer is now attached to the Scene RenderGraph as well as the Game graph, including graph replacement. Scene rendering excludes screen-overlay lists but consumes the same depth-tested world list. Scene marquee selection skips hidden Transform objects only inside a screen Canvas; Canvas-free UI remains ordinary world content.
- Windows Release `_Infernux` built successfully. The focused UI/runtime/input suite passed **191**, skipped **1**, failed **0**. In the visible 041Lab editor, an Empty → 200×120 clipping Frame → 300×160 Button was rotated around all three axes. Engine render-target captures proved the clipped region in both Game and Scene, and disabling `clip_content` restored the complete button. The Console remained at **0 warnings / 0 errors**; temporary objects were discarded by scene reload.
- The first real pointer attempt exposed two general integration defects rather than a scene problem. Runtime input read only the authored `Scene.main_camera`, while rendering uses the effective active camera when no explicit preference exists; it also built the ray from mutable Camera pixel dimensions that can be written by another Editor view. Game View and Player input now use `effective_game_camera`, and the public ray conversion accepts the exact target viewport width/height as one pair.
- UIButton runtime drawing now consumes its existing Selectable tint state instead of ignoring hover/pressed/disabled colors. In a fresh visible 041Lab editor, a rotated Canvas-free button changed from the authored salmon color to a deliberately bright green Hover color under an MCP-injected real Game View pointer. The engine render-target evidence is `review/world-ui-input-hit-green.png`; Console remained at **0 warnings / 0 errors**, and the temporary object was discarded by scene reload.
- Windows Release `_Infernux` rebuilt successfully. The latest focused Game View/runtime UI/UI logic/Input suite passed **199**, skipped **1**, failed **0**. A real hover and ray hit are now proven; full click callback, drag continuity, masks, transparent inter-surface ordering, culling/special policies, authoring lifecycle and multi-platform evidence remain open. Strict progress therefore stays **66/255** for the main plan and **102/331** combined. Reusable implementation maturity is estimated at **71%—73%**, with end-to-end acceptance at **47%—49%**.

## 2026-09-12 — A06 per-camera world transparency order

- World UI submission now records a non-mergeable native draw-command range for every Canvas-free root. The range is the transparency-sorting unit: commands inside one panel keep authored order, while `RenderWorld` stably sorts panel centers back-to-front from the ViewProjection of the Camera that is actually rendering. One camera-independent Python command list can therefore be replayed by multiple cameras without borrowing another camera's order.
- The world pipeline continues to depth-test against scene geometry without writing panel depth. This preserves ordinary scene occlusion, prevents coplanar child elements from fighting, and uses painter order rather than pretending that depth alone solves transparency. No global UI sorting order, alternate renderer or fallback pass was added.
- Windows Release `_Infernux` rebuilt successfully and the focused UI/runtime/native-boundary suite passed **260**, skipped **1**, failed **0**. In the visible 041Lab editor, MCP created overlapping 55%-alpha red and blue world buttons with near-red/far-blue while keeping red first in hierarchy; the Game render target became red-dominant purple. Only their world depths were swapped and the result became blue-dominant purple. Evidence: `review/world-ui-transparent-sort-final-red-near.png` and `review/world-ui-transparent-sort-final-blue-near.png`. Both remained depth-occluded by normal scene geometry. Temporary objects were discarded through scene reload; Console ended at **0 warnings / 0 errors**.
- This closes A06.2's default three-dimensional depth/transform/transparency rule and stable local-versus-spatial ordering items. Main-plan strict progress is now **68/255 (26.7%)**; combined with the Compute appendix it is **104/331 (31.4%)**. Mask, explicit special policies, complete click/drag callbacks, authoring lifecycle, fonts and multi-platform evidence remain open.

## 2026-09-12 — A06 complete world-surface pointer capture

- `UISlider` now consumes the authoritative `PointerEventData.canvas_size` instead of reaching back into screen-only `UICanvas.reference_width/reference_height`. The event producer already derives that size from each input surface, so screen Canvas and Canvas-free world UI now share one value-control mapping contract without aliases, feature probes or fallback dimensions.
- Focused value-control and UI-event coverage passed **128**, skipped **1**, failed **0**. The tests deliberately use an input surface with no screen-Canvas dimension fields.
- In the visible 041Lab editor, the reusable MCP review created a 300×50 Canvas-free Slider, entered Play, focused the real Game viewport, pressed its left edge and moved to the far right outside the control before release. The live serialized value reached **100.0 while still held**, proving that drag ownership remained with the original world surface after leaving its bounds. `review/world-slider-before.png`, `review/world-slider-after-drag.png`, and the engine-owned editor capture provide visual evidence of the empty-to-full transition.
- The temporary scene object was discarded through scene reload. The final Console contained **0 warnings / 0 errors**. Together with the prior foreground-occlusion, rotated-ray, Hover and event-state tests, this completes A06.2's default pointer/focus/drag item. Existing submission coverage also proves that ordinary Frame/Image components use the same Canvas-free root path, while live Text/Button/Slider captures and interaction use the same public component types as screen UI. This closes A06.1's unified screen/world control item without introducing a WorldCanvas type. Main-plan strict progress is **70/255 (27.5%)**; combined progress is **106/331 (32.0%)**. Reusable implementation maturity is estimated at **73%—75%**, with end-to-end acceptance at **50%—52%**.

## 2026-09-12 — A06 Figma/CSS em font-size semantics

- Added an exact-font comparison fixture for PingFangSC-Regular at 12/16/22/32/48/72px and captured both a browser CSS/Figma-style reference and the 041Lab native Game render target. The earlier engine widths at 22/32/48px were only about **66.3%—67.0%** of the reference, confirming the reported undersizing.
- The cause is the semantic difference between stb's `ScaleForPixelHeight` and CSS/Figma em pixels. `InxTextLayout` now reads each font's own big-endian SFNT `head` and `hhea` tables and derives the em-to-raster ratio. It does not hard-code PingFang or apply a global multiplier. Raster glyph size is corrected while authored baseline and line-height remain in logical font-size units.
- After rebuilding the Windows Release native module, visible glyph widths for 22/32/48/72px were **268/391/587/881px** in Infernux and **269/391/587/882px** in the browser reference. The new native SFNT parser test passed, the focused UI suite passed **142**, skipped **1**, failed **0**, and the live Console contained **0 warnings / 0 errors**.
- This closes A06.3's common authored `font_size` semantic item. Multiline/wrapping/bounds, alternate fonts and fallback, DPI/window/platform matrices, world-perspective sharpness and cooked-font evidence remain open. Strict progress is now **71/255 (27.8%)** for the main plan and **107/331 (32.3%)** combined. Reusable implementation maturity is estimated at **74%—76%**, with end-to-end acceptance at **51%—53%**.

## 2026-09-12 — A06 multiline, wrapping and clipping matrix

- Extended the exact PingFangSC browser/engine comparison with explicit multiline text, mixed Chinese/Latin wrapping in a fixed 520px box, 4px letter spacing and a deliberately short clipping box. The 32px/1.4 sample advances by 45px per line and the 22px/1.3 sample by 29px; both references wrap the mixed string at the same Chinese character boundary.
- Connected the existing `TextOverflow.Clip` author field to both editor draw-list text and the native Screen/World UI text submission. The renderer now applies the authored text box as the clip rectangle directly around the shared layout draw; no alternate text layout or overflow fallback was introduced.
- The real 041Lab Game render target contains only the first line of the 36px clipping sample, matching the CSS `overflow:hidden` reference. Evidence is `out/041-font-css-reference.png` and the MCP session artifact `review/font-metrics-engine.png`. The Windows Release module and native `infernux.text_layout` test pass; the focused Python UI group passes **143**, skips **1**, fails **0**; the visible editor Console has **0 warnings / 0 errors**.
- This completes the same-font Figma/CSS comparison matrix. Main strict progress is **72/255 (28.2%)** and combined progress is **108/331 (32.6%)**. Alternate-font fallback, hit rectangles, DPI/platform coverage, world-perspective sharpness and cooked-font evidence remain open.

## 2026-09-12 — A06 alternate-font metrics and atlas lifetime

- Added a second exact-file comparison using Roboto-Medium at 32px. Browser CSS and the real 041Lab Game render target both produced a 431px visible glyph span for `Roboto replacement Abg 0123`, proving that the SFNT-derived conversion follows each font rather than a PingFang-specific constant.
- Tied the custom text-layout font cache to the ImGui atlas lifetime. `ReloadGUIFont()` now clears cached custom and missing font identities before `io.Fonts->Clear()`, so a DPI-triggered atlas rebuild cannot leave stale `ImFont` pointers in the shared layout path.
- Windows Release `_Infernux` rebuilt and the real editor rendered the alternate font with **0 warnings / 0 errors**. This completes the stb/em, replacement-font and common measure/render-chain item. Explicit glyph fallback, font cook and the platform/DPI matrix remain open. Strict progress is **73/255 (28.6%)** main and **109/331 (32.9%)** combined.

## 2026-09-12 — A06 deterministic main-axis auto layout

- Added explicit Start/Center/End/SpaceBetween distribution on the auto-layout main axis while retaining the existing independent cross-axis alignment. The field is serialized, exposed in the Inspector and exported through the public UI API/stubs.
- Replaced one-shot per-child Fill clamping with a deterministic constrained allocation. Children that reach min/max are fixed in author order and the remaining main-axis extent is redistributed by weight; there is no iterative guess of parent size and no alternate layout path.
- The focused UI group passes **152**, skips **1**, fails **0**. In the real 041Lab Game render target, three fixed-width controls span a 1200px frame using SpaceBetween, while a second row caps two Fill controls at 150/250px and assigns all remaining width to the third. The temporary hierarchy was discarded and the Console has **0 warnings / 0 errors**. Evidence: `review/auto-layout-main-axis.png`.
- This completes the Auto Layout feature item. Strict progress is **74/255 (29.0%)** main and **110/331 (33.2%)** combined; reusable implementation maturity is **76%—78%** and end-to-end acceptance is **53%—55%**.

## 2026-09-12 — A06 single-authority UIText intrinsic layout

- Removed the three competing auto-size writers from the UI Editor, Inspector and runtime renderer. `UIText` now resolves one transient intrinsic size before geometry is consumed and keeps serialized `width` / `height` as authored data.
- Runtime measurement scales the font, wrap width and letter spacing together, then normalizes the result back to logical Canvas pixels. Editor preview zoom is therefore visual only. The previous editor-only wrap tolerance and per-backend padding guesses were removed.
- `get_rect`, `UIFrame`, clipping, drawing and Canvas raycast now consume the same derived rectangle. Focused tests prove the pre-layout runs before runtime command geometry, Auto Height uses the authored wrap width, hit testing uses the derived Auto Width, and neither a 1× nor 2× measurement writes scene geometry.
- In the visible 041Lab Game render target, fixed text, Auto Width and 260px mixed Chinese/Latin Auto Height rendered together. After Play, hierarchy documents still reported the authored Auto Width size as **520×42** and Auto Height as **260×12**. The focused suite passed **155**, skipped **1**, failed **0**; Console ended with **0 warnings / 0 errors**. This completes the shared auto-size/measure/clip/hit item. Strict progress is **75/255 (29.4%)** main and **111/331 (33.5%)** combined.

### A06 explicit missing-font boundary

- Empty `font_path` remains the one intentional request for the engine default font. A non-empty path that cannot be resolved, does not exist, or cannot be loaded now returns no font and emits one bounded native error; it no longer substitutes the default font and lets a comparison appear successful.
- `infernux.text_layout` covers two requests for the same missing file and proves `nullptr` on both with exactly one error. In the visible 041Lab Game render target, an Auto Width fixture assigned to a missing font disappeared instead of rendering with the default face, while the native terminal reported the missing path once.
- The fixture was removed and the scene returned to **0 warnings / 0 errors**. This is only the missing-selected-font half of the plan item; an explicit per-glyph fallback chain is still open, so strict progress does not change.

## 2026-09-12 — A06 exact text bounds and world alpha coverage

- Re-audited the text evidence against the literal checklist wording rather than visual similarity alone. Exact engine/browser widths, authored baseline and line starts, line-height, clipping, and the derived Auto Width hit rectangle all consume the same `UIText` layout result; the per-font SFNT metric conversion does not use a global empirical multiplier. This closes A06.3's exact-bounds item. Main strict progress is **76/255 (29.8%)** and combined progress is **112/331 (33.8%)**.
- Fixed a separate world-space UI depth defect in the shared Vulkan fragment path: fully transparent glyph/image texels are discarded before they can publish depth, while partially covered antialiased texels retain normal alpha blending. Windows Release `_Infernux` rebuilt successfully and the focused UI/RenderGraph suite passed **217**, skipped **1**, failed **0**.
- Rotated general-purpose masks are still absent, so the combined transparent-coverage/mask item remains unchecked. Reusable implementation maturity remains **77%—79%**, end-to-end acceptance **54%—56%**, and the full remaining estimate remains **12—18 full-time solo weeks** (desktop-first core **8—12 weeks**).

## 2026-09-12 — A06 explicit per-glyph fallback chain

- Added an authored ordered `fallback_font_paths` list to `UIText` and `UIButton`, exposed it in the custom Inspector, and carried it through the shared editor-measure/runtime Screen/World rendering path. Layout and draw now resolve each codepoint against primary then explicit fallback faces, using each face's own SFNT em conversion.
- A missing, empty or invalid explicitly selected fallback invalidates the layout and emits a bounded error; the engine never skips the authored error to consult an ambient system font. If every valid selected face lacks a codepoint, only the primary face's replacement glyph is used.
- Native coverage proves Roboto handles Latin while PingFangSC handles the Chinese glyph, plus missing and repeated-invalid chains. Focused Python UI/Inspector/RenderGraph coverage passed **286**, skipped **1**, failed **0**. The visible 041Lab Game target rendered `Roboto primary + 中文备用字体 0123`; its Play-cloned `UIText` document retained the ordered fallback path, Console ended with **0 warnings / 0 errors**, and the fixture was discarded by scene reload.
- This closes the explicit-font-chain item. Strict progress is now **77/255 (30.2%)** main and **113/331 (34.1%)** combined; reusable implementation maturity is **78%—80%**, end-to-end acceptance **55%—57%**. The full remaining estimate stays **12—18 full-time solo weeks** (desktop-first core **8—12 weeks**).
## 2026-09-12 — Rect Tool 使用真实世界 UI 布局矩形

- `EditorTools` 增加按对象登记的通用 Rect frame，组件可以给 Scene View 提供真实世界中心、平面轴和半尺寸；绘制、CPU 拾取与拖拽读取同一份 frame。
- UI Editor 与 Scene Rect 共用布局直改规则：被操作轴的 Fill/Hug 切 Fixed，UIText 切 FixedSize；世界 UI 修改布局字段，不再用 Transform scale 拉扁文字。
- Scene Rect 的连续编辑快照、回滚和 Undo 同时覆盖 UI 布局字段与必要的根 Transform 位置，普通 3D 对象继续走原 Transform 路径。
- Windows Release：`infernux.editor_tools` 通过；Python 定向回归 `140 passed, 1 skipped`。
- 真实 `Infernux041Lab`：320×180、100 px/unit 的世界 Frame 角点拖拽到约 448×176，Transform scale 保持 1；Ctrl+Z 恢复到 320×180 与原中心。MCP engine-owned Editor/Scene capture 可见 Rect frame，Console 0 warning / 0 error；临时对象随后通过场景 reload 丢弃。
- 同一真实窗口对旋转内建 Cube 做 Rect 角点拖拽：仅 Transform position/scale 改变，`Cube` 网格身份与材质 GUID 不变；Ctrl+Z 精确恢复，证明普通 3D 主路径不写顶点、不烘焙模型。

## 2026-09-12 — Rect Tool 无 Renderer 对象闭环

- 原生 `EditorTools` 回归覆盖没有 MeshRenderer 的对象：Rect frame 仍由 Transform 与固定编辑参考范围得到，绘制提交为完整 12 条边；Empty、Camera、Light 复用该路径，不新增作者资产宽高或类型专属回退。
- 在真实可见 `Infernux041Lab` 编辑器中选中 `Key Light` 并按 `T`，引擎自身 Editor render-target 捕获 `review/041-rect-light.png` 清晰显示位于灯光 Transform 的 Rect frame；Inspector 的 Light 文档保持不变，Console 为 0 warning / 0 error。
- 因此 A07 无 Renderer 条目完成。严格主计划为 **80/255（31.4%）**，Compute 附录 **36/76（47.4%）**，合计 **116/331（35.0%）**；可复用工程实现仍约 **79%—81%**，最终用户验收约 **56%—58%**。完整 041 剩余仍估单人全职 **12—18 周**，桌面核心优先约 **8—12 周**。

## 2026-09-12 — 永久世界 UI 样本与正尺度坐标主链

- 041Lab 保存了 `Assets/Scenes/03_WorldSpaceUI.scene`：普通场景对象直接承载 Frame/Image/Text/Button，没有 Canvas、RenderTexture 或私有渲染脚本。修正世界顶点局部 X 和世界射线局部 X 的共同约定后，根 Transform 使用正尺度、零旋转即可在 Scene 与 Game 保持相同左到右方向。
- Windows Release `_Infernux` 重建通过。真实编辑器重启后，MCP 引擎 render-target 捕获 `review/041-world-ui-game-positive-scale.png`、`review/041-world-ui-scene-positive-scale.png` 和 `review/041-world-ui-button-pressed.png`；Game View 实际指针按下使 Button 进入 pressed 色，证明显示与 hit test 不再镜像错位。Play/Stop 后 Console 0 warning / 0 error。
- Transform 的自动化字段同步改为权威 schema 的 `serialized_name → field_id` 映射，`position/rotation/scale` 写入本地 TRS 且可 Undo，不再把 list 与原生 Vector3 直接比较。Transform、世界 UI 与 Rect 路由集中回归 **40 passed**。
- 普通 Text/Button 已明确直接进入世界三维顶点流，不为每个元素分配离屏纹理，对应 A06.2 条目完成。此时严格主计划 **81/255（31.8%）**，合计 **117/331（35.3%）**。

## 2026-09-12 — A01 权威字段目录替代 JSON 类型猜测

- 自动化 `scene.component.schema` 对原生组件只枚举 C++ semantic catalog，对 Python 组件只枚举类注册/候选发布时编译的 `FieldSchema`。原生 Inspector 的 generic 与 multi 路径也只读取 catalog；删除了依据当前 JSON 中 bool/int/float/list 形状临时选择字段控件的分支。未声明 JSON 键保持不可编辑，不增加兼容回退。
- Native descriptor 与 Python `serialized_field` 都投影为不可变 `FieldSchema`，默认值和运行值使用 common `VALUE_CODECS`；范围、枚举成员、只读和隐藏状态随同一声明进入 Inspector 与 MCP。真实 Camera MCP schema 返回 12 个字段、FOV 1—179 和 Projection 枚举；真实编辑器 Camera Inspector 0 warning / 0 error。
- 字段、Inspector 和自动化集中回归 **81 passed**。A01.1 首项完成；完整字段类型矩阵、跨语言同批发布边界及 Player 语义目录仍保持未勾选。严格主计划 **82/255（32.2%）**，Compute 附录 **36/76（47.4%）**，合计 **118/331（35.6%）**。

## 2026-09-12 — Light 原生语义声明纵向迁移

- 将 Light 的十五个真实持久化字段加入 `engine:native` semantic catalog，声明覆盖三种枚举、范围、RGB/Vec2、位域整数和布尔字段；自动化暴露真实的 `lightType` / `spotAngle` / `influenceDomains` 等序列化键，不从当前实例值猜类型。
- Light 包装层中十个一一对应的属性改由 `CppProperty.from_native` 投影。颜色的 RGB↔RGBA 适配与两个 influence 位的派生布尔视图暂保留为明确的 adapter 边界，没有伪造存储字段或增加兼容分支。
- Windows Release `_Infernux` 重建通过，定向 semantic/Inspector/组件命令回归 **114 passed**。重新打开真实可见 041Lab 后，MCP 对对象 209 / 组件 1338 返回完整十五字段 Light schema；选中 Directional Light 的真实 Inspector 后 Console 仅有一条 RenderGraph INFO，**0 warning / 0 error**。该纵向迁移不构成完整字段类型矩阵，严格勾选数保持 **82/255**。

## 2026-09-12 — 世界 UI 普通对象生命周期完成

- 为 Canvas-free `UIFrame` + `UIButton` 增加普通对象 clone、严格 prefab save/instantiate 回归：不创建隐式 Canvas，Transform、子树、逻辑尺寸、pixels-per-unit 和文字保持，定向世界 UI/Rect/Prefab 回归 **49 passed**。
- 在真实可见 `03_WorldSpaceUI` 中选择永久面板，Ctrl+D 使用统一 Hierarchy 命令复制完整四组件子树；副本对象 214—217 和组件 1350—1357 均为独立身份。Ctrl+Z 随后移除副本，场景回到 `dirty=false`，Console **0 warning / 0 error**。
- 结合此前同一永久场景的保存/重开、Play clone/Stop、Scene 选择和通用 Rect Tool 证据，A06.2 的普通场景工作流项完成。严格主计划 **83/255（32.5%）**，Compute 附录 **36/76（47.4%）**，合计 **119/331（36.0%）**；可复用实现约 **81%—83%**，最终验收约 **58%—60%**。

## 2026-09-12 — 世界 UI 布局/Transform 权威边界完成

- 明确并回归唯一组合顺序：UI 布局生成根局部像素几何，`world_pixels_per_unit` 一次换算到局部单位，普通 Transform 一次变到世界；渲染收集不回写任何一侧。
- 世界提交测试在完整 begin/clip/draw/end 后逐字段核对 Transform 文档与 Frame 布局均未变化；结合 Rect Tool 的 UI-layout/3D-Transform 分流和真实正尺度 Scene/Game/输入证据，A06.1 对应项完成。定向回归 **42 passed**。
- 严格主计划 **84/255（32.9%）**，Compute 附录 **36/76（47.4%）**，合计 **120/331（36.3%）**。剩余完整 041 估时不变。
## 2026-09-12 — A01.3 engine-owned scene and asset operations

- Moved the 18 scene/object/component operations, 10 project-asset operations and 3 typed DataAsset operations, their schemas, GUID identity helpers, component serialization and owner-thread dispatch from the MCP package into `Infernux.host`. The engine bootstrap installs this authoring surface immediately after the interaction/document managers exist and before any project plugin is loaded.
- MCP now contributes only its protocol/session/capture and other transport-facing operation families. Gateway startup projects the already engine-owned authoring registry; shutdown removes 67 MCP-owned operations while the 31 engine-owned operations remain callable. Adapter status reports both the complete 98-operation surface and the plugin-owned subset instead of conflating the two lifetimes.
- Added an isolated registry regression that imports only `Infernux.host` and proves scene field query/edit plus asset query/edit schemas exist with owner `infernux/engine`. Compatibility modules in the MCP package only re-export the engine builders; they contain no duplicate operation definitions.
- `OperationRegistry` now enforces engine authority when an older compatible transport plugin republishes an identical schema: the transport projection is ignored and cannot replace the engine handler/lifetime; a different schema still raises a conflict. This allowed the real 041Lab project with its existing MCP 0.1.1 package to restart without reinstalling the plugin.
- Focused Host/MCP/DataAsset regression: **47 passed**, failed **0**. In the fresh visible editor, the existing MCP package projected the complete operation registry, engine-owned hierarchy query returned the permanent `03_WorldSpaceUI` scene, project status remained `dirty=false`, and Console reported **0 warnings / 0 errors**. This closes A01.3's ownership item. Strict progress is now main plan **85/255 (33.3%)**, Compute appendix **36/76 (47.4%)**, combined **121/331 (36.6%)**. Full 041 remains **12–18 full-time solo weeks**; desktop-core-first remains **8–12 weeks**.

## 2026-09-12 — A01.3 explicit operation contracts and live projection

- Extended `OperationSchema` with explicit Editor/Player availability and execution phase. Every engine-owned operation now declares owner-thread execution, `editor.read` or `editor.authoring`, capabilities, stable errors, reversibility and a concrete top-level output shape with required fields; successful results are validated against that schema instead of being accepted as arbitrary objects.
- `OperationRegistry` gives the engine owner unconditional precedence for the same stable operation ID. This is the authority rule that keeps older transport projections compatible without letting a plugin replace an engine handler; unrelated owner conflicts still fail.
- The MCP plugin no longer defines Scene, object, component, field, project-asset or DataAsset operations. Its three compatibility modules are imports only. Schema listing reads the live engine registry and revision, so there is no copied MCP catalog to refresh.
- Focused Host/MCP suite: **54 passed**, failed **0**. After a normal engine-owned close and fresh visible 041Lab restart with the existing MCP 0.1.1 package, `infernux.scene.component.schema` reported typed input/output, owner thread, `editor.read`, Editor-only availability and stable errors. Executing it against the real Light returned all 15 native fields and passed result validation; Console remained **0 warnings / 0 errors**.
- This closes two more A01.3 items. Strict progress is now main plan **87/255 (34.1%)**, Compute appendix **36/76 (47.4%)**, combined **123/331 (37.2%)**. Reusable implementation is estimated at **83%–85%**, final user acceptance at **60%–62%**; remaining estimates stay **12–18 weeks** full 041 or **8–12 weeks** desktop-core-first.

## 2026-09-12 — A01.3 shared authority and single-Undo acceptance

- Added `infernux.data_asset.schema` to the engine-owned catalog and made it expose the same registered `FieldSchema` documents used by DataAsset normalization and Inspector authoring. The catalog now contains **31 engine-owned + 67 MCP-owned = 98** operations.
- Native Light, a Python serialized component and a nested DataAsset field now share schema constraints, canonical values and the same editor history semantics in one integration suite. Invalid undeclared fields fail instead of creating shadow state; two independent component edits create two independent Undo entries.
- Fixed DataAsset reads to prefer the active DocumentRegistry resource over a second disk load. This makes MCP, Inspector and Undo observe the same live authoring document immediately, including before the asynchronous durability queue drains.
- Focused Host/MCP/DataAsset regression: **68 passed**, failed **0**. In the visible 041Lab, RunnerLong `unit.move_speed` changed from 7.25 to 6.75 through the engine operation, and one real Editor `Ctrl+Z` restored 7.25. The live document and durable transaction agreed; Console remained **0 warnings / 0 errors**.
- This closes A01.3's shared-authority/Undo item. Strict progress is now main plan **88/255 (34.5%)**, Compute appendix **36/76 (47.4%)**, combined **124/331 (37.5%)**. Reusable implementation is estimated at **84%–86%**, final user acceptance at **61%–63%**; remaining estimates stay **12–18 weeks** full 041 or **8–12 weeks** desktop-core-first.

## 2026-09-12 — A01.1 AudioSource native semantic catalog

- Moved AudioSource's 11 public/persistent editable fields from duplicated Python metadata into the C++ semantic catalog. The Python wrapper now projects the native descriptor, and the common projection retains hidden, rename, collection/asset and slider metadata for subsequent component migrations.
- Declared `track_count` as a setter-owned coupled document edit. Its single native setter changes both the scalar count and the dynamic `tracks` array atomically, while C++ validation and the public schema now agree on the existing 1—16 contract. This avoids manufacturing an invalid intermediate document and remains one normal property transaction, not an AudioSource-specific retry path.
- Windows Release native binding rebuilt successfully. Combined authority/semantic/audio/Inspector/Undo Python regression: **350 passed**; related native audio/document tests: **3/3 passed**.
- In the visible 041Lab, MCP observed all 11 fields, changed one AudioSource from 1 to 4 tracks, and one actual Editor Undo restored both the count and array. The component and probe object were then undone, leaving the scene `dirty=false` and Console at **0 warnings / 0 errors**.
- This advances A01.1 native coverage from six to seven component types. No complete checklist item is claimed, so strict progress remains main **88/255 (34.5%)**, Compute **36/76 (47.4%)**, combined **124/331 (37.5%)**.

## 2026-09-12 — A01.1 Collider family native semantic catalog

- Added native semantic declarations for BoxCollider, SphereCollider, CapsuleCollider, CylinderCollider and MeshCollider. Their common center, trigger and PhysicMaterial fields are produced by one C++ declaration builder, while concrete geometry remains owned by each concrete type.
- Kept the existing scene format intact: authoring exposes a typed PhysicMaterial asset reference, and its explicit setter boundary stores `physic_material_guid`. Python wrappers now project both common and concrete metadata from the native declaration instead of maintaining a second field description.
- Windows Release `_Infernux` rebuilt successfully. Semantic coverage passed **39**, the expanded component/physics/Mesh/Inspector/Undo suite passed **498**, and native semantic/document/GPU-physics coverage passed **3/3**.
- In the visible 041Lab, MCP created a temporary BoxCollider, changed center and PhysicMaterial identity, and real Ctrl+Z replay restored material, center, component and object in order. The scene returned clean and Console reported **0 warnings / 0 errors**.
- Native semantic coverage is now twelve component types. This remains an A01.1 migration slice, so strict progress stays main **88/255 (34.5%)**, Compute **36/76 (47.4%)**, combined **124/331 (37.5%)**.

## 2026-09-12 — A00 requirement-routing audit

- Recounted the locked Runner Long requirement map rather than leaving its checklist state stale. `041-runner-long-open-decisions.md` contains exactly **29 unique RL IDs**, with no duplicate or missing row.
- Every row points to its concrete A01—A15 acceptance section. Engine-owned public capability, official-plugin delivery and game-only migration work are explicitly separated: Taichi is an engine dependency, Steam remains an official plugin boundary, and Unity migration code stays in Runner Long.
- This closes A00's decision-document item. Strict progress is now main **89/255 (34.9%)**, Compute **36/76 (47.4%)**, combined **125/331 (37.8%)**. Reusable implementation is estimated at **85%–87%**, final user acceptance at **62%–64%**; full 041 remains **12–18 full-time solo weeks**, or **8–12 weeks** for the desktop core first.

## 2026-09-12 — Runtime frame ownership and AudioVoiceField investigation

- Fixed script-free native physics barriers implicitly opening a Python execution frame without a matching native owner. Physics callbacks now require an existing native frame; lifecycle phase entry no longer manufactures one. This caused the reported scene-publication safe-point error and subsequent Play rejection.
- Moved the existing pre-scene authoring queue ahead of presentation early-outs, so minimized windows still process MCP authoring commands. No retry or alternate publication path was added.
- Regression: **129 Python tests passed** across lifecycle, frame submission, change journal, automation, process exit and Play Mode. Native audio playback and RenderTexture tests passed **2/2**. The visible editor successfully played scene 08, stopped, loaded scene 09 and played it without console errors. Minimized-editor MCP project inspection completed in **43 ms**, then the same window was restored.
- Verified SDL's logical-device pause retains the physical device and emits silence. Stopping voices destroys their streams, not the output device. No device-lifecycle change was necessary for the user's keep-open requirement.
- Same-scene visible-editor A/B, five 240-frame windows per condition, repeated twice. Second run medians of window averages: normal frame **1.225 ms**, muted **1.215 ms**, restored normal **1.235 ms**, controller inactive **1.106 ms**, controller plus HUD inactive **1.095 ms**. Corresponding game-only work: **0.489 / 0.488 / 0.498 / 0.421 / 0.423 ms**. Normal showed 4 real + 8 virtual voices; muted showed 0 real + 12 virtual and zero output peak. Thus the mute comparison genuinely removed audible mixing but not logical voice scheduling. It does not measure total audio cost or explain all differences from other scenes.
- Script removal reduced scene work from about **0.090 ms** to **0.040 ms**; render preparation/submission remained roughly **0.37–0.39 ms**. Whole-frame timings fluctuated between runs, so no unsupported audio-only 0.5 ms regression or claimed optimization is recorded. All experimental state was restored; scene 09 is clean in Edit Mode with **0 warnings / 0 errors**.
- A09 implementation slice: generation-owned RHI RenderTexture attachments, transactional resize, sampled-view lifetime, optional MSAA/depth and RenderGraph imports compile and pass native resource tests. Public Camera/UI/Python integration and real Vulkan render/readback acceptance remain outstanding; this does not close A09.

## 2026-09-12 — A09 real Vulkan RenderTexture sampling and retirement

- Added a real-device render-graph test: six odd/non-workgroup-aligned extents alternate UNORM single-sample and HDR MSAA4, with D32 depth attachments. Each graph clears, resolves when needed, samples every pixel through a compute shader, and validates mapped readback values (including HDR values above 1).
- After recording, resize publishes a new generation; destroying the target and graph retires the old native handles without freeing their allocations before the submission epoch completes. The recorded commands still return the expected pixels. No queue-idle resize or old-content fallback was introduced.
- Fixed duplicate persistent-generation imports to share attachment identity and return the latest declared SSA write version. The test reimports under another label and checks both resource count and versions before consuming it, guarding against missing producer/consumer dependencies.
- Windows Release binding rebuild succeeded. Related native tests **4/4 passed**; Python RenderGraph suite **73 passed**. Public Camera/material/UI binding, cross-view scheduling and the monitor/reflection scenes are still pending; A09 and total completion counts are unchanged.

## 2026-09-12 — A09 public RenderTexture allocation boundary

- Introduced `inx.RenderTexture` with explicit pixel dimensions, existing PixelFormat values, optional depth, MSAA, filter and storage use; read-only properties and `resize()` forward to the same RHI generation owner already covered by GPU tests. Engine/renderer factories allocate on the existing device and assign process-local resource identities, not fake asset GUIDs.
- Added Python typing and exports. Public argument/descriptor/resize tests plus RenderGraph regressions: **86 passed**. Windows Release native bindings rebuilt successfully.
- Via real editor MCP operations, authored a temporary probe script/object in 041Lab, entered Play, constructed HDR/depth targets with 1/2/4 samples, resized each, verified revision and dimensions, and confirmed depth-as-color rejection. The script reported `passed=true`, payload **153700 bytes**, Console **0 warnings / 0 errors**. The created object and script were deleted through undoable editor operations and the scene was saved clean in Edit Mode.
- This is the fixed-size public allocation/lifetime slice only. Camera/material/UI consumption, automatic relative sizing, persistent asset authoring and monitor/reflection demos remain open. No full A09 checklist item was advanced.

## 2026-09-12 — A09 persistent owner in Python rendering topology

- Added `graph.import_texture(name, target)`: public graphs retain the resource owner while compiled native graphs borrow the current generation. Repeated imports share identity; occupied names cannot silently refer to another resource. Resizing refreshes graph extents and attachment/sample bindings before graph execution. Persistent writes are retained even when not consumed by the screen.
- The first public import path covers single-sample color produced and sampled in the same graph. The first writer must clear; reads before it completes are rejected at graph publication. Native MSAA/depth allocation remains supported, but their Python import surface and cross-view producer scheduling are not yet connected. These are explicit pending work, not implicit old-frame reads or automatic sample reduction.
- Removed the uncalled dictionary fallback builder and its constant switch. Editor, Player and tests now produce the same native description schema; public typing includes the import API.
- Windows Release rebuilt. **89 Python tests** passed, including live Vulkan allocations, native owner forwarding, resize, resource naming and graph regression; **3/3 native tests** passed, including real GPU clear/resolve/readback/retirement and persistent graph publication boundaries.
- Visible 041Lab MCP test created a temporary pipeline, component and fullscreen shader. During Play, sizes changed **37×23 → 53×29 → 113×67 → 37×23**, revisions **1 → 3 → 5 → 7** (width and height edited separately). The shader sampled target content on the left and encoded actual `textureSize` on the right. Engine captures showed the expected changing size colors, proving the sampler did not retain the old allocation. Console **0 warnings / 0 errors** in the completed run.
- An earlier test setup attempted to rename ShaderInfo during text replacement; rejection also reported a rollback publication failure. Shader identity was kept stable for this test. Atomic shader rename/failed-edit rollback remains an A13 follow-up, not claimed fixed. Another attempt correctly did not advance while the editor was minimized; restored the visible window before the completed run.
- Removed only the temporary test objects/assets via undoable MCP authoring and restored the original clean `09_AudioVoiceField` in Edit Mode. Captures are under the Lab MCP session's `review/041-rendertexture-graph-*.png`; the editor is left visible, not a detached background test Player.
- Closed A09's persistent-owner/transient-handle ownership item. Strict counts: main **91/256 (35.5%)**, Compute **36/76 (47.4%)**, combined **127/332 (38.3%)**. The next A09 work is Camera/material/UI consumption and cross-view scheduling; full 041 remains active, with **12–18 full-time solo weeks** still the scope-based estimate.

## 2026-09-12 — A09 attachment imports, shared target ownership and MCP field projection

- Unified Scene/Game render-target allocations and retirement with RHI RenderTexture generations; removed duplicated native allocation/destruction logic (SceneRenderTarget source/header combined: 209 fewer lines). Initial image transitions share one submission. This does not remove every existing cleanup-time device idle.
- Extended public graph imports to color, depth and resolve attachments. Single-sample resolve aliases color; MSAA color is sampled through explicit resolve, and sampled single-sample depth requires its declared usage. Multi-sample depth sampling and cross-graph/history scheduling remain unsupported, not silently replaced with older content.
- Fixed fixed-size fullscreen render extents, explicit fullscreen resolve publication, depth-only passes attaching the screen color unintentionally, single-sample depth entering automatic MSAA resolution, and Camera clear settings overriding an independent persistent target's clear. Persistent writers are graph side effects even when not sampled by the screen.
- Rebuilt Windows Release native bindings. **3/3 native tests** passed, including real Vulkan render/sample/readback/retirement. **185 Python tests** passed across RenderTexture, RenderGraph, RenderStack, Camera, host operations and component property transactions.
- Visible editor MCP regression used 4x MSAA gradient rendering and resolve, sampled depth at 0.25, and resize **37x23 -> 53x29 -> 113x67 -> 37x23**, revisions **1 -> 3 -> 5 -> 7**. The displayed size color changed accordingly. A second target retained its explicit blue clear even with the main Camera set to a red SolidColor clear. Engine-generated Game and editor captures were visually inspected. Console: **0 warnings / 0 errors**.
- Separately checked Scene camera/light icons and selected-object orange outline/XYZ gizmo after the shared native target refactor. Evidence: `review/041-rhi-target-selection.png`, `review/041-rhi-target-outline.png`, and `review/041-rendertexture-msaa-depth-*.png` within the Lab MCP session.
- The real test exposed a query/write mismatch in MCP native component fields. The host now accepts the native schema's serialized field names and value encoding, maps them through existing CppProperty setters and preserves transaction Undo/Redo. Tests cover Camera enum/color/FOV, Light enum/color, AudioSource mute/track count and invalid edits. Audio edits explicitly do not restore the whole document or reload all tracks.
- The final visible run exited successfully and restored the original clean `09_AudioVoiceField` in Edit Mode. Only the temporary probe object and three test assets were removed through undoable authoring operations. No extra test Player remains.
- Camera/material/UI consumers, cross-view dependencies and monitor/reflection demonstrations are still pending. Strict progress remains **127/332 (38.3%)**; no full A09 acceptance item was advanced by this partial slice.

## 2026-09-12 — A09 material pass contracts no longer use presentation defaults

- Investigated the Camera output route: each Camera has its own graph, but output allocation, MSAA negotiation and active-view gating still assume the shared screen target. A public target setter alone would not implement independent offscreen cameras.
- Replaced material callback derivation from global MaterialPipelineManager defaults with `ResolveMaterialPass`, which uses the current RenderView and the actual pass's attachments. Custom targets retain their own formats/sample counts; view depth aliases use the actual view depth format; sampled depth does not become a fixed-function attachment. Skybox draws now receive the same explicit pass contract as ordinary mesh draws.
- Callback publication tracks color/depth/sample changes, not only global samples. The steady-state revision check compares fixed fields without allocating a descriptor vector. Camera aspect/pixel dimensions now derive from its graph's RenderView rather than directly from the shared screen target.
- Native contract regression covers independent view descriptors, format-only changes, sample changes, explicit HDR/depth targets, read-only versus sampled depth, depth-only rendering and MRT slot order. Windows Release native build and related **3/3 native tests** passed; **144 Python tests** passed across RenderTexture, RenderGraph, RenderStack and Camera.
- Extended the ignored MCP regression helper with a geometry mode. Real 041Lab rendering writes identical scene geometry into **RGBA16F / MSAA4 / D32F** and **RGBA8 UNORM / 1 sample / D24S8** targets, then samples both side by side. Resize sequence **321x181 -> 533x299 -> 113x67 -> 321x181**, revisions **1 -> 3 -> 5 -> 7**, completed with no console warnings/errors.
- The initial run used SolidColor and did not exercise skybox drawing. Repeated with Skybox explicitly selected; engine captures show the procedural gradient on both targets instead of the blue clear, while foreground objects retain depth occlusion. This verifies real geometry and skybox drawing, not only a fullscreen clear. Evidence: Lab MCP session `review/041-rendertexture-geometry-*.png`.
- Each run removed its temporary object/script/shaders through undoable authoring and restored clean `09_AudioVoiceField` in Edit Mode. This is a single-view multi-target test, not yet a pair of independent offscreen Cameras. Camera target ownership/binding, cross-view scheduling, UI/material consumers and the monitor/reflection acceptance scenes remain open. Strict completion stays **127/332 (38.3%)**.

## 2026-09-12 — A09 public Camera output and native camera capture

- Added `Camera.target_texture`: a runtime strong reference to the existing `inx.RenderTexture`, with matching depth required. It does not fabricate an asset GUID or serialize GPU storage into the scene. Clone retains the resource owner. Invalid type/depth assignments preserve the previous binding.
- A per-Camera SceneRenderGraph borrows the allocation generation directly, without a second color allocation or ImGui descriptor. The target's format, sample count and dimensions are authoritative. Resize and same-size owner replacement refresh view identity/history; detach restores the screen route. Offscreen cameras do not participate in screen MSAA negotiation and remain active when the Game panel is hidden.
- RenderGraph descriptions carry a linear-output boundary resolved from the finished pass topology. Offscreen Camera graphs omit display encoding and screen UI; captures apply display encoding only to exported PNG pixels. World UI output, material/UI consumers and cross-view producer dependencies are still pending, not silently claimed supported.
- MCP `capture.request` now accepts `source="camera"` and `camera_component_id`. Capture reads the existing native target; it never captures the desktop. A changed/destroyed output before readback is reported explicitly. Updated the Lab's installed MCP capture operation to exercise the same source contract.
- Windows Release native build succeeded. Relevant native tests **3/3 passed (0.51 s)**. Initial public resource/graph/Camera/MCP tests **135 passed**; expanded final regression including RenderStack, MCP server/supervisor and Game panel **229 passed (30.85 s)**. The deliberate invalid Camera document test emits its expected rejection log; there was no real-editor error.
- Visible `09_AudioVoiceField` test, authored through MCP: two independent Cameras, distinct transforms and perspectives, HDR / MSAA4 / D32 output and RGBA8 UNORM / 1x / D24S8 output. Camera A resized **321x181 -> 533x299**, replaced its owner at the same size, resized **113x67**, detached/rebound, then returned to **321x181**. Camera B stayed **181x321** throughout. Both runs completed with Console **0 warnings / 0 errors**.
- Engine PNGs were inspected: scene geometry, skybox gradients, correct depth occlusion and distinct views, with no screen HUD copied into the offscreen targets. The complete Editor PNG retained the original on-screen HUD. In the second run MCP selected Scene (confirmed by engine Editor capture), hiding Game; camera captures continued at engine frames **7370 / 7383 / 7560** while the moving listener changed position. This is actual offscreen output, not a stale screen copy.
- Evidence lives under the Lab MCP session `20260912-022615-747a0787/review/041-camera-target-*.png`. Temporary Cameras, controller and test script were removed through undoable MCP operations; original `09_AudioVoiceField` restored clean Edit. The visible editor closed normally with exit 0 before final Python tests; no detached test Player was used.
- No additional full A09 item checked: material/UI consumption, world UI in offscreen Cameras, dependency-based multi-view scheduling, relative sizing, asset authoring, export and monitor/reflection demonstrations remain open. Strict completion remains **127/332 (38.3%)**, full 041 goal active.

## 2026-09-12 — A09 explicit cross-view producer scheduling

- Added a resource-owner-based view schedule, shared by immediate graph recording and frame-submission batches. Graph imports declare current-frame reads; camera outputs and persistent raster/resolve/copy destinations declare writes. Shared output writers preserve Camera depth order. Independent views have no artificial dependency on one another. A stable order is rebuilt only when the participating view identities or resource-access revisions change, without content hashes or an alternate old-frame execution path.
- Missing producers, same-target feedback and view dependency cycles are rejected. If schedule publication fails, non-editor captures fail instead of reporting previously rendered pixels as a newly rendered frame. Editor capture remains available to inspect the failure. Copy destinations now participate in the same first-writer/import classification and side-effect retention as raster outputs.
- Input-only persistent imports retain the producer's canonical shader-readable state. Camera single-sample sampled depth is exported for shader reads; MSAA color is consumed via resolve. The native regression now samples a producer in a separate reader graph and checks generation retention through resize, destruction and GPU completion.
- Windows Release binding rebuild succeeded. One added test initially used the wrong enum name (`Transfer` versus the existing `Copy`); corrected and rebuilt. An early Python run overlapped dependency staging and failed to load the locked DLL; rerun only after the build completed. Final native tests **3/3 passed (0.41 s)**, final Python RenderTexture/RenderGraph/Camera/RenderStack/Game panel tests **166 passed (4.57 s)**. Expected malformed Camera-document rejection is a test log, not an editor error. The native GPU test log contains no VUID/validation error.
- Real visible 041Lab MCP test: B (depth 100) renders scene geometry and skybox into UNORM color plus sampled D32 depth; A (depth 50) relays B into HDR/MSAA4; main Camera (depth 0) and Scene sample A/B/depth side by side. This intentionally reverses producer order. A resizes 321x181 -> 533x299, replaces its owner, resizes 113x67 -> 321x181; B remains 181x321. Hidden-Game captures continue to update. Engine Editor, Scene and Camera PNGs were visually inspected.
- The initial helper teardown attempted to detach Cameras already destroyed before the controller. Removed that redundant teardown; native component ownership performs release. Both repeated runs then restored clean Edit state. In the final run, disabling B produced the expected missing-producer publication error and Camera capture status `failed`; re-enabling B restored a completed current-frame capture. Cleared only these deliberate negative-test logs through the Console's semantic Clear button after checking their contents. Final console: **0 warnings / 0 errors**, including after Stop and object/asset removal.
- Evidence: `review/041-camera-consumer-editor.png`, `041-camera-consumer-scene-tab.png`, `041-camera-consumer-hidden-game-*.png`, `041-camera-consumer-producer-restored.png` in Lab MCP session `20260912-022615-747a0787`. Temporary objects and script/shader assets were removed through undoable MCP authoring. One visible editor remains; no probe Python or extra Player process remains.
- Acceptance limits: this is explicit RenderGraph consumption, not material/UI binding or completed monitor/reflection delivery. Demand-driven execution, temporal double buffering, offscreen world UI, persistent authoring and export remain open. Code review also flags Camera first-use layout assumptions: borrowed output attachments start undefined, while PrepareSubmissionExecution currently assumes canonical exported layouts. A real Camera validation-layer first-use/shared-output preserve test and a correct ownership-level transition must precede full A09 acceptance; the standalone producer/readback test does not prove that route. No synchronous per-bind initialization or old-image fallback was added to conceal it.
- Strict completion remains **127/332 (38.3%)**, full goal active. The user's 2000-versus-1000 FPS clarification is a cross-scene comparison, not an isolated audio measurement; the same-scene A/B and limitations above remain the applicable evidence.

## 2026-09-12 — A09 first-use attachment layouts and shared Camera outputs

- Reproduced the borrowed-target first-use defect in the visible Release editor with the installed Khronos validation layer explicitly enabled. The pre-fix log reached its 10-message cap for `VUID-vkCmdDraw-None-09600`: graph recording assumed shader-readable/color/depth attachment layouts while new allocations remained undefined. Vulkan's actual-layout rule is documented at https://docs.vulkan.org/spec/latest/chapters/resources.html.
- Initialization now belongs to the RenderTexture allocation generation and runs once for its color, optional MSAA and depth attachments. The existing Graphics frame setup records these barriers before ownership release/consumer work; direct graph execution follows the same contract. Resizing creates a new generation. No queue-idle wait, separate synchronous submission, per-bind clear, content hash or old-frame fallback was added.
- RenderGraph persistent imports declare the matching canonical state. Scene graphs export unsampled MSAA/depth attachments as attachment states as well as exporting sampleable color/depth for reads. The native Vulkan test adds a second graph loading the same target without clearing and checks that its original pixels survive, then samples/readbacks and checks retirement.
- The actual cross-view consumer test passed with core validation enabled, including resize, replacement, hidden Game rendering and the deliberate missing-producer rejection/recovery. The first separate shared-output helper run did not remain in Play; its teardown incorrectly attempted Stop from Edit, leaving three temporary objects/script. No cause for that Play exit was established. The helper now consults actual runtime state before Stop; the exact temporary objects/script were removed through undoable engine operations, without changing other authored objects.
- A subsequent independent visible-editor shared-output run passed: Camera A clears HDR/MSAA4 output green; Camera B loads/preserves it without clearing. Both share each new allocation through 321x181 -> 533x299 -> owner replacement -> 113x67 -> 321x181. Changing B's live culling mask from zero to all layers adds scene geometry over A's preserved background. Both engine captures were visually inspected. Hidden-Game captures advance through distinct engine frames. Final Stop/deletion restored clean `09_AudioVoiceField` and Console **0 warnings / 0 errors**.
- Core validation logs `041-camera-layout-after.log` and `041-camera-layout-inspector.log` are empty after these runs. This does not claim synchronization-validation or other-GPU/platform coverage. Screenshots: `review/041-camera-shared-*.png` under Lab MCP session `20260912-022615-747a0787`.
- Material/UI binding, temporal buffering, demand-driven updates, asset authoring/export and the full A09 demonstrations remain open. Strict completion stays **127/332 (38.3%)**; the full 041 goal remains active.

## 2026-09-12 — Camera Inspector unsigned layer-mask regression

- The new writable `Camera.culling_mask` projection exposed an existing Inspector narrowing boundary: valid uint32 mask `0xffffffff` was cast to C++ signed `int` while compiling the property batch. This caused the user's repeated Inspector error. Previous setter/serialization-only tests did not cover Inspector rendering.
- The common integer property batch now stores, decodes, refreshes, renders and returns signed 64-bit integer values, which include the full uint32 mask domain. Its slider/input use ImGui's matching scalar type; enum indices remain validated as ordinary int. Camera's native declaration owns the `[0, 0xffffffff]` UI range. The actual mask is neither truncated, converted to negative, hidden nor routed around the batch renderer.
- Added real native-GUI regression coverage for `0xffffffff`, bit 31, negative int32 and a value above uint32, covering plan creation plus cached value refresh. Before rebuilding, three of four cases reproduced the narrowing error; after the fix all passed. Expanded Inspector/Camera/schema/RenderTexture/RenderGraph/RenderStack suite: **286 passed (8.77 s)**. GUI-only unit runs log their intentionally absent scene pipeline; the malformed Camera-document test logs its expected rejection. These are not failures observed in the real editor.
- Visible MCP verification selected the permanent Main Camera in `03_WorldSpaceUI`, observed `4294967295` in the rendered Inspector, Ctrl-clicked the integer field, typed `2147483648`, verified that exact authored/native document value, then one actual Editor Undo restored `4294967295`. The scene returned clean with Console **0 warnings / 0 errors**. Engine screenshot `review/041-camera-inspector-mask.png` was inspected and shows the complete Camera panel and intact world UI.
- No release, commit or push was performed. The native binding was rebuilt locally in the activated `infernux` environment. Diagnostic validation is not intended to remain enabled in the user's normal performance-testing editor.
- After rebuilding the native regression executables against the current runtime, RenderTexture, scene-graph validation and real Vulkan sampling/retirement tests passed **3/3 (0.82 s)**. The diagnostic editor closed normally with exit 0 before testing; normal visible-editor startup then proceeded without forced Vulkan validation layers.

## 2026-09-12 — A09 Camera output sampled by ordinary materials

- Added runtime RenderTexture bindings to the existing material path: `material.set_texture(name, target)`. Native owners and clones retain the target; serialized asset references remain authored GUIDs/tokens, and this runtime assignment does not schedule a material save. Explicit asset assignment, clear, removal and non-texture replacement release the overlay. Deserialization retains only overlays whose texture property survives.
- RenderTexture publishes its sampled color through the existing immutable TextureGpuViewSlot. Bindless and ordinary sampler descriptors resolve that publication; resize replaces descriptors through existing retirement rather than changing an in-flight descriptor or forcing a GPU wait. Per-renderer descriptor keys already include the base descriptor generation identity.
- SceneRenderGraph collects material inputs after current camera culling submission, declares vertex/fragment reads, imports the real allocation and exposes the owner to the existing cross-view scheduler. Normal draws without runtime textures are filtered once before pass matching. No fake GUID/path, CPU readback, hidden file, content hash or alternate old-frame execution path was introduced.
- Visible-editor MCP exercise `041_camera_material_check.py` used a temporary cube monitor with the builtin Unlit material in the real 09 scene. Consumer depth -100 deliberately precedes producer depth 100. Both HDR 4xMSAA and single-sample runs passed 321x181 -> 533x299 -> owner replacement -> 113x67 -> 321x181; material images changed color and resolution accordingly, including hidden Game updates and Scene rendering. Engine captures were visually inspected, not programmatically pixel-analyzed.
- Both runs deliberately disabled the producer: capture failed with invalid schedule and the console reported only the expected missing-producer error. Restoring the producer resumed camera/material output. The expected diagnostic was recorded before clearing it through the real Console control. Temporary runtime objects, cameras and script were removed; the authored 09 scene is clean, with 0 warnings / 0 errors after teardown. No permanent monitor demo was claimed.
- Verified `VkLayer_khronos_validation.dll` loaded in the actual editor; `041-material-validation.log` has no Vulkan VUID or validation error/warning. Detailed test records: `041-material-msaa.log` and `041-material-single.log`. This is core validation on the local Windows GPU, not synchronization-validation or cross-platform coverage. Diagnostic editor closed normally afterwards.
- Public/material/Inspector/Camera/graph tests passed **167/167 (5.67 s)**. Updated native owner-slot and material binding/clone/serialization tests plus scene-graph and real Vulkan sampling/retirement passed **3/3 (0.39 s)** after the final runtime rebuild. Camera Inspector was independently rechecked through high-bit numeric editing and Undo, returning exact 0xffffffff with no errors.
- UI binding, same-graph production/material-consumption ordering, parameter-block/pipeline combinations, demand-driven/history behavior, persistent authoring/export and full A09 demos remain open. Strict completion remains **127/332 (38.3%)**. No commit, push or release was performed.

## 2026-09-12 — A09 camera output consumed by screen/world UI

- Added the runtime-only `UIImage.texture` override. It accepts the same public RenderTexture used by Camera and Material; `None` restores the authored material/texture_path. UI material `texSampler` bindings also accept that resource. Runtime overrides do not rewrite asset references, request asset path resolution, or schedule saves.
- GUI descriptors borrow the RHI sampled color (resolved color for MSAA). The existing descriptor retirement and allocation generations handle resize/replacement; no GPU readback/reupload, fake GUID, disk image, content hash, separate target type, or old-frame fallback was added. Cached screen commands now include direct and material-owned target revisions.
- Actual UI draw inputs join the existing RenderView dependency schedule. Fixed the old last-screen-camera-only renderer attachment: every camera renders world UI, while only the final screen camera composites screen command lists. World UI follows per-camera layer masks, depth and the actual attachment format/sample/stencil signature. Pointer mapping uses the same camera mask, including occluder queries.
- Fixed the display-space boundary: linear camera images sampled after display encoding are encoded once; Camera UI and World UI stay linear. Replaced the copied opaque screen shader bytecode arrays with the engine's shader compilation path and short source shaders. Existing font alpha/display-space uploads are not gamma transformed.
- Expanded regressions exposed stale screen-layout fixtures with no Canvas, and a real world AutoWidth/AutoHeight geometry bug. Screen fixtures now explicitly author a Canvas and the intended reference size/Transform. Effective text size is shared by world rendering, pointer/selection bounds and Rect/Gizmo geometry without overwriting authored width/height. Both screen and world intrinsic-size tests retain their geometric assertions.
- Real visible 041Lab MCP tests used temporary monitor UI in 09_AudioVoiceField, first HDR/D32/4xMSAA and then UNORM/D24S8/single-sample. Both ran 321x181 -> 533x299 -> owner replacement -> 113x67 -> 321x181, independent consumer target, deliberately reversed camera depths, hidden Game, and screen-only 411x233 resize/replacement. An auto-width world text label was also visible in the source scene and camera output. Captures were reviewed with the image viewer, never pixel-analyzed automatically.
- Both final runs disabled the producer, required a failed consumer capture with the expected missing-producer diagnostic, then restored it and captured current output. Expected diagnostics were recorded before clearing through the real Console. Temporary cameras, script and runtime UI were removed; the original 09 scene is clean, with 0 warnings / 0 errors after teardown. This is a functional regression fixture, not a permanent polished monitor demo.
- `VkLayer_khronos_validation.dll` was verified in the live process; final validation log has no Vulkan VUID or validation error/warning. Evidence: `041-ui-texture-msaa.log`, `041-ui-texture-single.log`, `041-ui-texture-validation-final.log`. An earlier run logged a Windows asyncio connection-reset during MCP transport teardown; no suppression/retry mechanism was added to hide it. This is local core Vulkan validation, not synchronization validation or Linux/Android/Web acceptance.
- Expanded Python UI/material/Camera/RenderGraph tests: **365 passed, 1 skipped (8.65 s)**. Rebuilt native owner/material, graph/attachment/scheduling and real Vulkan resource/retirement tests: **3/3 passed (0.43 s)**. Logs from negative contract tests intentionally include invalid-camera/no-pipeline messages; these are not editor-play failures.
- Checked A09's single public Camera/Material/UI resource item only. Main plan **92/256 (35.9%)**, Compute appendix **36/76 (47.4%)**, total **128/332 (38.6%)**; **204** items remain unchecked. Range remains **12–18 solo full-time engineering weeks** for the whole plan assuming no scope growth, not an unattended wall-clock promise. Same-graph internal producer/material ordering, pipeline/parameter combinations, demand/history, persistent authoring/export, reflection water, permanent demos and multi-platform acceptance remain open. No commit, push or release was performed.
- Closed the validation editor normally and reopened the ordinary visible editor (PID 24456, no forced Khronos layer). Restored clean 03_WorldSpaceUI, selected Main Camera, rechecked high-bit Inspector editing and Undo (2147483648 -> 4294967295), and reviewed the editor capture. Console remains 0 warnings / 0 errors. The normal editor is left open for the user; diagnostics are not left running as hidden Players.

## 2026-09-12 — A09 same-graph material/UI reads use the producing pass version

- Found two connected defects: implicit material/UI inputs were always declared external at view scheduling, and their graph handles were imported at graph creation (v0), before local writers published subsequent versions. Valid same-graph producer/consumer pipelines could therefore be rejected as cross-view feedback or acquire stale logical versions.
- SceneRenderGraph now distinguishes a preceding local writer from an external input. It resolves implicit draw inputs at their pass position through the existing deduplicating RenderGraph import operation. Writing a graph resource updates all texture aliases, including the output alias. Single-sample resolve and color are recognized as one physical attachment; depth-only writes do not fabricate a color producer.
- Reused the existing view-publication error boundary for unsupported read-before-write and same-pass feedback. No new fallback, copy, hash, previous-frame substitution, custom resource format or per-frame error/retry layer was added. The resource schedule is still revision driven.
- Added native classification tests for local/external input, before-producer rejection, same-pass feedback, later rewrite, aliases, single-sample resolve/color identity and depth-only exclusion. Extended the real Vulkan regression to sample into one readback buffer before a rewrite and into another afterward. Six independent UNORM/HDR, size and 1x/4x cases verify both expected outputs and recorded-generation lifetime through resize/destruction/retirement.
- Visible 041Lab MCP regression authored a temporary custom pipeline and ordinary unlit-material cubes plus UIImage in 09_AudioVoiceField. One RenderTexture carries a gradient, is read by the first cube, is overwritten blue, then is read by the second cube and UI. 1x and 4x both passed 321x181 -> 533x299 -> 113x67 -> 321x181. Engine Game/editor captures were reviewed visually; no automated pixel analysis of MCP captures was used. Runtime objects and temporary script/shaders were removed and the original scene restored clean.
- A live negative case moved the reader before its producer: capture correctly failed with an invalid-schedule result and Console identified the before-producer read. Restoring authored order restored current output without restarting. The expected diagnostic was recorded before clearing it through the real Console. This does not claim support for same-pass feedback, history or ping-pong authoring yet.
- Cross-camera UI regressions passed on rerun, including resize/replacement, hidden Game, independent consumer target, screen-only inputs and producer removal/restoration. The first run's test teardown failed because Console was not the active tab; it was not counted as a clean pass. Activating the real tab and rerunning completed with 0 warnings/errors. No test assertion was removed to hide this.
- Validation editor log `041-local-texture-validation.log` has no VUID, Vulkan validation error/warning or traceback. This was core Vulkan validation, not synchronization-validation or non-Windows acceptance. Functional logs: `041-local-texture-msaa.log`, `041-local-texture-single.log`, `041-local-texture-negative.log`, `041-local-texture-crossview-final.log`.
- Final rebuilt native tests: **3/3 passed (0.39 s)**; expanded Python selection: **365 passed, 1 skipped (7.61 s)**. Evidence: `041-local-texture-ctest.log`, `041-local-texture-pytest.log`. Python negative contract logs intentionally exercise invalid Camera/no-pipeline cases. Builds and pytest did not run concurrently with the editor or each other. Scoped diff whitespace checks passed.
- Recount unchanged: main **92/256**, Compute **36/76**, total **128/332 (38.6%)**. The dependency milestone is recorded but the encompassing A09 item remains unchecked: demand-driven updates, history/double buffering, asset authoring/export, reflection and platform coverage are outstanding. Whole-plan estimate remains **12–18 solo full-time engineering weeks**, assuming fixed scope; not an unattended completion promise. No commit, push or release was performed.
- After the final rebuild, a fresh ordinary visible editor also passed the single-sample explicit `resolve` alias case, including resize and read-before-write rejection/restoration (`041-local-texture-alias.log`); its restored capture was reviewed. Restored clean 03_WorldSpaceUI, exercised actual Main Camera Inspector mask editing and Undo (2147483648 -> 4294967295), and reviewed the editor capture. Console 0 warnings/errors. Visible editor PID **46088** is left running without the forced Khronos layer (`041-local-texture-inspector.log`); the earlier validation editor closed through MCP and no hidden Player was launched.

### 2026-09-12 — A09 custom projection and oblique clipping foundation

- Added one native runtime projection override with reset, pure oblique-near-plane calculation, and NumPy 4x4 Camera bindings. Override publication validates finite/invertible input once; no per-frame matrix verification, guessed fallback or hidden copy of a second camera state was added. Author FOV/near/far remain intact. Native rendering and culling consume the same projection.
- Screen rays now unproject the near plane and an interior depth sample using a revision/aspect-cached inverse projection, rather than reconstructing a different FOV-only frustum. This handles perspective, orthographic, asymmetric, oblique and distant/infinite far planes. Screen/world conversion consistently uses the engine's top-left pixel origin. Selected Camera gizmos derive a batched wire frustum from the same matrices, with a finite drawing-distance cap, rather than recreating the original FOV frustum.
- Consulted Unity's official [projectionMatrix](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Camera-projectionMatrix.html), [CalculateObliqueMatrix](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Camera.CalculateObliqueMatrix.html), [ScreenPointToRay](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Camera.ScreenPointToRay.html), and [worldToCameraMatrix](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Camera-worldToCameraMatrix.html) contracts. Adopted explicit override/reset and near-plane ray-origin semantics, not Unity's right-handed camera-space / CPU projection convention or bottom-left screen origin. Our LH +Z, Vulkan [0,1] depth and Y-down projection are documented explicitly. View override and reflected winding remain future work.
- Rebuilt native extension and Camera culling regression. Native **2/2 passed (0.49 s)**; expanded Python **383 passed, 1 skipped (6.19 s)**; Camera integration **5 passed (2.79 s)**. Evidence: `041-oblique-build.log`, `041-oblique-culling-build.log`, `041-oblique-ctest.log`, `041-oblique-python-wide.log`, `041-oblique-camera-integration.log`. Initial Python fixture incorrectly assigned the deliberately read-only public aspect_ratio; the corrected test exercises the renderer-owned native setter. Those initial failures are not counted as passing runs.
- Closed PID 46088 through MCP before building, then launched one visible editor PID 10628 with the Khronos validation layer requested, not a hidden Player. In clean 09_AudioVoiceField, an MCP-authored temporary component created two independent Camera/RenderTexture/UI monitors and ordinary unlit geometry. Reviewed engine captures for oblique perspective, off-axis perspective, oblique orthographic and reset; camera B stayed unchanged throughout. Both camera outputs and UI consumers completed at 641x401, including hidden Game capture. Reset visibly restored matching geometry in both monitors. No automated analysis of human-review-only MCP capture pixels was performed.
- The first visible fixture used a tuple instead of splatted Material.set_color arguments, and the next fixture hit an old test's hard-coded target size. These failed runs are retained (`041-oblique-visible.log`, `041-oblique-visible-final.log`); the shared helper now asserts the explicitly requested CASES size, not a weakened check. The final complete run `041-oblique-visible-complete.log` passed all modes and teardown with Console **0 warnings / 0 errors**. Temporary objects/assets were removed and original 09 restored clean. No VUID/validation error or warning was found in the editor log; that log also contains the initial fixture failure, so it is not represented as an error-free session or full synchronization-validation acceptance.
- Custom-frustum gizmo geometry and batching have unit coverage; an actual selected custom-camera gizmo interaction and post-change real world-UI clicking remain to be verified separately. Mirror view override, winding/recursive reflection, independent history, complete asset/export authoring and other platforms remain open. No whole A09 item was checked off: main **92/256**, compute **36/76**, total **128/332 (38.6%)**, fixed-scope estimate **12–18 solo full-time engineering weeks** unchanged.
- A user-reported 3072x1980 editor DPI/input mismatch takes priority next. At the user's explicit request, delegated it to `gpt-6-astra` / `ultra`, with official UE/Unity/Godot research, coordinate-domain review and regression requirements. A prior read-only investigation was stopped with no edits. The new investigation must distinguish confirmed coordinate defects from an as-yet-unreproduced Windows user configuration. Editor/build ownership was handed over after the Camera checks; no concurrent editor/build/pytest jobs are authorized.

### 2026-09-12 — Editor DPI coordinate contract; first stage closed by user (historical state)

- Research and evidence: `dev/041-editor-dpi-research.md`, with official Unity/UE/Godot, SDL, Microsoft references and explicit applicable/non-applicable mechanisms. Both release tags use the same SDL/ImGui revisions; 0.3.7 already scales fonts/styles, and normal Hub launches a separate editor process. No unsupported attribution to a dependency upgrade, Qt contamination or a guessed Windows multiplier was made.
- Preserved SDL window units for pointer/ImGui geometry. Centralized authored metric scale as window display scale / pixel density, eliminating repeated density scaling on Retina/Wayland while retaining normal Windows density=1 sizing. ImGui 1.92 owns framebuffer-density font rasterization. Added immediate GUI refresh on display/scale events and supplied actual SDL pixel extents to Vulkan's unspecified-surface-extent path. Existing DPI layout metadata rejects numerically incompatible older Retina layouts and preserves Windows layouts. No bulk layout wipe or font-size compensation was added.
- Native semantic publications now include main-editor coordinate provenance (origin, window extent, per-axis framebuffer ratio, UI metric scale). New MCP archives forward available metadata; independently updated plugins on older engines preserve semantic targets without inventing a pixel mapping. Multi-viewport captures are not claimed. The current lab's already installed MCP plugin still emits its old snapshot fields; new metadata forwarding is verified by unit tests, not falsely represented as live plugin migration.
- Release extension/tests and bundled MCP archive build/staging completed (`041-dpi-build.log`, `041-dpi-plugin-build.log`). Native **4/4 passed** (`041-dpi-ctest.log`), including 20 live ImGui-context draw-vertex/hit-test cases at 100/125/150/200% and density 1/1.25/1.5/2 with 3072x1980 model framebuffer and integer extent ratio differences. Python **50/50 passed** (`041-dpi-pytest.log`). These are model and local-device checks, not target high-DPI hardware acceptance.
- The actual local display is **1920x1080 at 96 DPI / 100%**, HWND PMv2. Editor client and engine capture both **1920x1009**, client origin (0,23). Baseline and rebuilt visible launch/capture recorded in `041-dpi-before.log` / `041-dpi-after.log`. Rebuilt Editor capture completed with 85 visible semantic targets and was reviewed manually. No automatic pixel analysis of human-review-only images occurred. Startup validation log contains no VUID/validation error or traceback at handoff.
- User's new screenshot shows menu/Hierarchy with a black central area and missing other panel content; one still image cannot prove a hit-test offset ratio or identify its cause. The original Windows high-DPI trigger remains unconfirmed. User explicitly requested current-stage closure and deferred real 3072x1980, 125/150/200%, runtime DPI and mixed-monitor acceptance until a new SSH-accessible machine; this is **not a current 041 blocker** and does not check off the full platform matrix. A prepared OS mouse/resize harness was not run after that direction.
- Visible 041Lab Editor **PID 62996**, HWND **14286896**, MCP **9713** remains open on the existing clean scene; Khronos validation requested, log `041-dpi-editor-validation.log`. Native build and pytest jobs are complete; no hidden Player, commit or push. Full 041 objectives and unchecked future platform work remain intact. Editor/build ownership returned to the root agent for A09.

### 2026-09-12 — DPI resumed on actual 2560x1440 / Windows 150%; local acceptance

- User connected a new monitor and explicitly requested real 150% testing. Read-only Win32 observations establish **2560x1440 /144 DPI/PMv2**, not an assumed meaning of "2K". Main client **2560x1334**, desktop origin **(0,34)**; native semantic `ui_scale=1.5`, framebuffer density **(1,1)**. Installed project MCP received only the optional coordinate/mouse forwarding patch and hot-refreshed, with no fabricated scale or full package migration.
- First observation of old PID62996 was minimized/client0; preserved that evidence before restoring. This is not proof of continuously-visible natural hot-switch behavior. Restored old process had full panels but stale18 font/menu38x22/button38x26 despite actual ui_scale1.5, excluding SDL-cache staleness as this observed defect.
- Fixed ImGui1.92 warm font-reload state publication: after Clear/Add, explicitly publish the new `Style.FontSizeBase`. ClearFonts otherwise restores the old current font size into the reset Style. Removed erroneous non-owned file-font data so atlas Clear releases the allocation. Menu/Toolbar authored style overrides and fixed metrics now use the one existing UI scale; the dedicated fixed toolbar's height is derived from font/padding before DockSpace layout. Only its main-dockspace single-window fixed leaf is adjusted; no saved-layout rebuild or game font/input compensation.
- Release build passed (`041-dpi-150-build.log`). Strengthened **one existing native test target**, **1/1 passed /0.11s** (`041-dpi-150-ctest.log`), retaining20 scale/density combinations and adding actual font/text-size and automatic font-sized button paint/press/release/miss assertions across Clear/Add in one ImGui context. Prior4/4 native and50/50 Python remain historical evidence, not repeated unchanged or relabeled real-device testing.
- Closed the clean old editor normally and cold-started the same visible project at150% with Khronos validation: **PID8796 /HWND13240798/MCP9713**. Menu57x33/Play57x39/Toolbar-height53 now match27 font plus scaled metrics. Main panels are visible in human-reviewed `041-dpi-150-cold.png`; native/client/capture sizes agree. Normalized font reload behavior is native-context evidence; no test-only engine API was added for a live reload experiment.
- User explicitly put the editor foreground after input-fixture checks rejected other-app focus. Real Win32 pointer events through SDL then passed project-menu expansion, Camera toolbar popup and Hierarchy selection (`041-dpi-150-os-surfaces.log`). Console Follow toggled true->false->true both maximized2560x1334 and resized1298x804 with nonzero desktop origin(111,115), then window restored maximized (`041-dpi-150-native-pointer.log`). Native cursor/ImGui coordinates agree within0.5px rounding.
- Inspector Add Component popup opened and canceled without adding a component. A manually reviewed Scene point(1234,770) selectedEmitter12 (`041-dpi-150-os-inspector-scene2.log`). Game tab and resolution popup worked; actual Toolbar Play->playing->Stop->edit passed. Final project clean09_AudioVoiceField and Console0warnings/0errors (`041-dpi-150-os-game.log`). Final human-reviewed engine Game capture completed at2560x1334 (`041-dpi-150-final.log`, `041-dpi-150-final-game.png`). No automatic PNG pixel analysis or OS display setting changes.
- Fixture failures remain recorded: initially minimized window/no targets, Unity/WeChat foreground/cursor constraints, and an Inspector test mistakenly interpreting component-header `selected` as collapse state. None is claimed as a DPI production failure. After inspecting the semantic contract the Inspector check used its explicit Add Component popup instead; production selection/collapse code was not changed.
- Full details/source reasoning/boundaries: `dev/041-editor-dpi-research.md`. Actual3072x1980,125/200%devices, mixed-monitor continuous hot DPI changes, secondary viewports, native OS file dialogs, complete Game-internal/Gizmo input and other-platform acceptance remain open; this local150% acceptance does not check the entire matrix. Final visible editor PID8796 is maximized, Game active, edit/clean, with no build/pytest running. `041-dpi-150-cold-editor.log` has no VUID/validation error/traceback. Editor/build ownership returns to root; full041 plan preserved, no commit/push/hiddenPlayer.

### 2026-09-12 — A09 reflected Camera view, per-view winding and actual shader eye

- Recounted main checklist **92/256 (35.9%)**, compute appendix **36/76 (47.4%)**, combined **128/332 (38.6%)**, **204 unchecked**. No partial A09 milestone was promoted to a completed whole item. Remaining estimate is **12–18 solo full-time engineering weeks**, including integration/release under unchanged scope; desktop core is roughly **8–12 weeks**, not the full release. Android/Web JIT feasibility, performance matrices and device availability remain uncertainty, not verified work.
- Added runtime `Camera.view_matrix` override, inverse `camera_to_world_matrix`, `has_custom_view_matrix` and `reset_view_matrix()`. Author Transform/projection fields and serialized documents are unchanged. Affine/invertibility/finite validation occurs once at the public setter before publication, not per consumer/frame. Reset follows the current Transform; explicit `invert_culling` remains independent. Rays, native view extraction/culling, selected-camera frustum, per-camera lighting position and shadow-camera bounds now consume the effective camera pose.
- `invert_culling` belongs to each Camera/SceneRenderGraph and material-pass cache key, including particle pipelines. It changes the effective front-face contract without modifying shared materials or a process-wide toggle. Setup publishes the flag before graph cache checks/callback capture, so repeated on/off switches use the right pipeline immediately. Light-space shadow-caster passes are not inverted by the observing camera. No old-frame copy/retry fallback was added.
- Comparison references: [Unity Camera.worldToCameraMatrix](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Camera-worldToCameraMatrix.html), [Unity GL.invertCulling](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/GL-invertCulling.html). The shared behavior is independent runtime view/reset and explicit reflection winding; Infernux retains its existing +Z/[0,1] camera conventions and uses per-camera rather than global raster state.
- Windows Release `_Infernux` rebuilt (`041-reflection-view-build.log`). Native **3/3 passed** (`041-reflection-view-ctest.log`): camera culling/clone/reset isolation, material-pass winding/cache signature, and scene graph depth versus shadow contracts. Assertions explicitly enabled for the two formerly release-disabled test targets. Python camera/view/projection contract **51 passed**, expanded Camera/RenderTexture/UI/render graph/pipeline suite **182 passed** (`041-reflection-view-python-wide.log`). Expected rejected-document error in the latter log is a negative test, not editor failure.
- Closed the clean old editor normally, confirmed its process exited, then reopened the real visible 041Lab (`041-reflection-view-editor.log`, MCP9713). All scene edits/test shaders used MCP. Two independent 641×401 Camera targets, HDR/4× and UNORM/1×, displayed side by side in UI. Modes **reflection -> wrong winding -> restored winding -> oblique clipping -> reset** passed direct Camera and Game capture review; reference Camera remains normal. The single-sided Quad disappears only with intentionally incorrect winding.
- Additional `getCameraPosition().x` fragment probe rendered reflected eye **-1.25 blue** and reference eye **+1.25 pink**; reset renders both pink. This checks actual shader-visible per-camera data, not only Python matrix getters. Human-reviewed engine captures: `041-reflection-view-case-0-mode-1-monitors.png`, `041-reflection-view-case-3-mode-3-monitors.png`, `041-reflection-view-case-4-mode-0-monitors.png`, under the project's `20260912-022615-747a0787/review` MCP session. No automatic analysis of human-review-only captures or desktop screenshots.
- Fixture mistakes remain recorded, not disguised as engine bugs: initial single-sided Quad faced away; initial fragment probe used `Material.set_shader`, which replaced both stage names despite only creating a fragment shader. Corrected the fixture to an authored front-facing Quad and `frag_shader_name`, retaining the built-in vertex shader. Final **`041-reflection-view-eye-final.log`** passes the complete mode sequence and cleanup with **0 warnings/0 errors**. Earlier diagnostic failure is retained in `041-reflection-view-eye-visible.log`; no production fallback or shadow disabling was added to hide it.
- At fixture teardown the project was clean **09_AudioVoiceField**, Edit mode. A later read-only MCP recheck reports the same clean scene **playing**; preserved this subsequent runtime state instead of stopping it again. Visible editor remains open. Temporary objects and test-owned script/shader assets were removed through MCP; user assets preserved. No native build/pytest runs while editor is live. Frame/history scheduling, persistent asset editing/cook/export, a permanent polished reflection/water demo, fullscreen-global shader helper auditing and other platforms remain open; this is a reusable Camera foundation, not complete A09 or complete041 acceptance. No commit/push or hidden Player.

### 2026-09-13 — A09 generic history resources and persistent attachment rebinding

- Strict count remains **128/332 (38.6%)**, with **204 unchecked**. Main **92/256**, compute **36/76**. Estimate under unchanged scope: **12–18 solo full-time engineering weeks** for complete041; desktop core **8–12 weeks** is only a subset. A09 remains partially implemented; no checkbox was added for this milestone.
- Graph history is no longer synonymous with TAA: `create_temporal_history(name, format=..., size=... / size_divisor=...)` exposes an existing read/write pair, while `set_temporal_jitter()` independently requests jitter. Built-in TAA opts in explicitly. The native owner is the existing RHI RenderTexture pair, replacing a separate raw-image allocator. Resize/format changes publish complete generations; existing GPU retirement owns lifetime.
- The first read after creation/invalidation is explicitly zero. Raster and copy writes are supported; the write slot must have a producer before a current-frame consumer. Contract validation rejects writes to the previous-frame slot, missing output writers, mismatched pair descriptions, read-before-write and same-pass self-read/write. These are graph publication checks, not per-frame hashes or recovery branches.
- A real Vulkan test caught a generic compiled attachment bug: changing an imported image view updated resource lookup but left cached VkRenderingAttachmentInfo pointing to the old image. The second ping-pong frame read the current write instead of the previous frame. Compilation now records reverse bindings for external color/resolve/depth attachments; persistent-color and existing raw import rebinding refresh only their affected attachment slots. No whole-graph rebuild, per-frame full graph scan, hidden copy or old-frame fallback.
- Final Windows Release build passed (`041-history-final-build.log`). Native **2/2 passed** (`041-history-final-ctest.log`): scene graph contracts plus real RenderTexture Vulkan tests. The latter runs eight alternating raster-write/readback frames across two physical owners with two zero-reset points, checking exact previous/current texels; existing ownership/format/resize cases also remain. Python Camera/view/projection/RenderTexture/RenderGraph/Effect regression **182 passed in 5.33s** (`041-history-final-python.log`).
- First real visible041Lab feedback run passed (`041-history-visible.log`): two Camera outputs (641×401 HDR/4× and 321×201 UNORM/1×), each with fixed1×1 clock history and half-view trail history. Resizing A to113×67 and back, changing A pose independently, and hiding Game while Scene is selected all preserve independent live outputs. Human-reviewed engine captures show cyan/pink feedback trails. The editor loaded Khronos validation; no desktop screenshots or automatic pixel inspection were used. This is a temporary functional probe, not the permanent polished demo.
- **Open regression, not accepted:** `041_taa_history_check.py` default mixed4×/1× targets failed; diagnostic `--single-sample` also failed. Logs `041-history-taa-visible.log` and `041-history-taa-single-visible.log` preserve repeated `opaque/Motion` color/depth sample mismatch and failed capture. Diagnostic changed only target samples, not the authored default4× pipeline, so it is not evidence of an independent1× pipeline failure. Both fixtures restored the original clean09 Edit scene; errors were not disguised as successful acceptance.
- Investigation: RenderStack caches one author-MSAA graph for all cameras; native ApplyPythonGraph overrides only frame MSAA for a Camera target, leaving explicit motion/normal MSAA resources and resolve topology authored for a different count. Next work is output-contract-aware graph construction/caching, including paired effect bindings/parameter revisions and custom/DSL pipelines. Do not fix the test by disabling TAA/MSAA, mutating author parameters per camera or adding old-frame fallback.
- After the final contract build, all test processes finished before reopening the visible editor (`041-history-final-editor.log`); MCP confirmed original09 clean Edit. A fresh visible history regression is recorded separately below. On-demand scheduling, complete TAA/history contracts, persistent asset editing/Cook/export, polished reflection/water scene and other platforms remain required. No commit/push or goal completion.
- Final re-run **passed** (`041-history-final-visible.log`, process exit0). Reviewed the fresh before/after-cut Game captures and full Scene-tab engine capture; both independent trails and Game-hidden updates remain visible after the new read-order checks. Teardown restored original09 clean Edit with **0 warnings / 0 errors**, temporary objects/script/shaders removed via MCP. Visible editor PID14116 remains open on the final build; native build and pytest are finished. TAA failure logs above remain retained and unresolved, not included in this passing fixture's Console counts.

### 2026-09-13 — A09 per-output MSAA graph specialization and TAA recovery

- Previous turn classified as progress: generic history, compiled attachment rebinding, real GPU and visible-editor evidence changed the mainline. This turn fixes the exposed TAA output-contract mismatch instead of suppressing MSAA or narrowing the required target combination. Full041 goal remains active; strict main/compute checklist still **128/332 (38.6%)**, **204 unchecked**, estimate **12–18 solo full-time weeks** under unchanged scope.
- Native `ScriptableRenderContext.output_samples` reports the actual fixed Camera target sample count before Python graph construction; zero denotes pipeline-owned screen quality. `RenderGraph(output_samples=...)` starts with that contract, and `set_msaa_samples(screen_preference)` returns its effective value. Built-in Forward, DSL routing/geometry providers, the no-RenderStack default and standalone RenderPipeline caches all consume this value. Explicitly unsupported Deferred MSAA targets are rejected at construction rather than silently changed.
- RenderStack now stores one state object per output-sample contract. Description, last accepted description, compiled effect bindings, upload revisions and build diagnostics stay paired with that variant; native-view identity still isolates parameter uploads within a shared variant. Alternating same-contract cameras reuses one description, not per-frame compilation. Author parameters and documents are never rewritten per Camera. Parameter/topology invalidation and repeated deserialization invalidate all variants, and destruction releases the whole variant directory.
- Candidate effect bindings are published only after complete graph construction; failed native publication restores bindings paired with the existing last-valid graph. This preserves the already-existing author-edit transaction behavior without adding a new fallback path. The hot render loop uses a local state reference, not repeated property lookup/full-cache scans. No hashes, per-frame cloning or hidden render target copies added.
- Windows Release native build passed (`041-view-msaa-build.log`); native **2/2 passed** (`041-view-msaa-ctest.log`). Initial targeted Python suite **281 passed** (`041-view-msaa-python-fixed.log`): 1/2/4/8 sample topology, Forward/DSL motion and resolve contracts, shared cache identity, whole-stack invalidation, per-view live parameters, prior Camera/projection/RT/Inspector/Effect tests. First new-test failures were fixture mistakes (write_colors is slot/name pairs; exposure belongs to tonemapping, not color_adjustments), fixed without weakening production behavior. Later deserialization/standalone tests are included in the full suite recorded below.
- Real visible041Lab after rebuild (`041-view-msaa-editor.log`, MCP9713) recovered the previously failing **4× HDR Camera A + 1× UNORM Camera B** TAA case. `041-view-msaa-taa-visible.log` completed captures of both independent targets,641×401→113×67→641×401 resize and Game-hidden/Scene-visible updates, with no Console errors. The current maximized editor surface is **2560×1369**; engine capture confirms the Scene-tab click still selected Scene and retained camera/light icons. No OS display settings were changed.
- Expanded live regression `041-view-msaa-taa-switch-fixed.log` passed **same-camera target4×→1×→4×**, additional resize, shared effect exposure **0.25→2→1**, and **TAA off→on**. Human-reviewed engine Game captures show both monitors changing brightness consistently and both valid again after TAA re-enable. Final Console **0 warning / 0 error**; clean original09 Edit restored, test-owned objects/scripts removed via MCP. Temporary monitoring shader/material objects remain test consumers, not a claimed polished reflection demo.
- The first expanded fixture used nonexistent GameObject.children after replacing Camera target and before updating UIImage. It correctly failed with AttributeError plus missing-producer errors; retained in `041-view-msaa-taa-switch-visible.log`. Corrected the fixture to the existing `get_children()` API and reran to completion. No engine fallback added to cover a broken producer/consumer fixture.
- Comparison: [Unity RenderTexture.antiAliasing](https://docs.unity3d.com/cn/6000.0/ScriptReference/RenderTexture-antiAliasing.html) makes raster sampling and resolve part of the target resource. [Unity6 URP anti-aliasing](https://docs.unity3d.com/cn/6000.0/Manual/urp/anti-aliasing.html) explicitly does not support simultaneous TAA+MSAA. Our comparison is resource/output ownership and lifecycle, not a claim that the supported effect combination is identical. Static target recovery is also not a dynamic reprojection/ghosting quality or performance acceptance.
- After visible tests, MCP confirmed clean09 Edit and closed the editor normally. Full Python regression started only after process exit, and includes all new caches and deserialization invalidation. Outcome is recorded below rather than inferred from the targeted suite. Complete A09 still requires demand-driven execution/retained-output policy, persistent asset/Cook/export, polished reflection/water and other-platform validation. No commit/push or goal completion.

#### Full-suite audit and follow-up fixes

- First complete run (`041-view-msaa-python-full.log`) finished **13 failed / 5929 passed / 11 skipped**, in 382.97s. Kept this baseline: targeted TAA success did not mean the entire branch was green.
- Fixed actual Windows compiler compatibility failure: a Unicode project under a Chinese user profile previously chose the first ASCII ancestor, `C:\Users`, and tried to create an unwritable `.infernux-build-links` directory. The build now creates one private temporary junction parent per invocation, using user TEMP when ASCII and Windows Temp when the user path is Unicode. This is a compiler-path compatibility boundary, not a new cache or fallback build. Actual artifacts stay in the requested project cache. Success, failure and cancellation remove the junction and its empty parent; there is no recursive target deletion or ACL/elevation change. Real local tests with Chinese TEMP and NTFS short names unavailable prove independent concurrent aliases, target identity, target preservation and failure cleanup. Standard-permission creation under Windows Temp was verified on this machine.
- Removed redundant Host asset containment/normalization implementations in favor of existing `path_utils` identity and containment operations. Listing/creation and DataAsset document identity now share the same resolved paths, including Windows case/alias handling. Added root-membership and directory-escape regressions. Added the missing explicit RenderTexture export to the lowercase public type stub.
- Updated strict tests to the current authoritative contracts: UIFrame/UIGroup/UIProgressBar/UISlider component catalog; Camera gizmo effective projection; RHI-owned sampled depth and deferred attachment retirement; resource-dependent per-Camera scheduling; 98 MCP operations; UI event logical input size; mandatory cooked type_id. No missing native interface was papered over with a production fallback.
- The Player gizmo import subprocess had tested the old installed 0.4.0 wheel rather than the source package under pytest. A read-only import trace proved that distinction. The regression now forwards the package-under-test import root to its fresh interpreter; source Gizmos remains lazy and the installed environment was not changed to make the test pass.
- Expanded focused suite: **412 passed / 1 failed / 1 skipped** (`041-view-msaa-regression-fixed.log`); the single failure was a missing subprocess import in the new negative test, fixed immediately. Follow-up build-link tests **3 passed**, Host/path/public namespace/Player gizmo tests **20 passed**. Full suite restarted in `041-view-msaa-python-full-fixed.log`; final outcome follows. No editor or native build runs alongside pytest.
- Final full Python suite **5946 passed / 11 skipped / 0 failed in 336.31s**, process exit0 (`041-view-msaa-python-full-fixed.log`). The trailing asymmetric collision-matrix error is from a deliberate rejected-document test, not a failed suite or editor startup. The added bilingual MSAA example builds under output contracts0/1/2/4/8; the subsequent RenderView-samples suite, including these five documentation checks, passed **20/20** (`041-view-msaa-doc-regression.log`). These five new example checks were added after the full run and are not included in5946.
- Updated the English/Chinese RenderGraph guide and relevant API signatures for effective MSAA, persistent RenderTexture ownership and explicit per-view history/jitter. Removed the obsolete claim that initial graph failure silently substitutes Default Forward. The same tested example now handles1× output without constructing an invalid resolve pass.
- Reopened visible041Lab only after all pytest processes finished: editor PID78064 (`041-view-msaa-final-editor.log`, MCP9713), current engine surface1920×1009. Final expanded TAA regression **passed**, process exit0 (`041-view-msaa-final-visible.log`); reviewed fresh Scene-tab, low/high-exposure and TAA-reenabled captures using the engine capture path. Both Camera outputs remain correct, Scene icons remain visible with Game hidden, original09 is restored to clean Edit and Console reports **0 warnings / 0 errors**. Temporary test objects/scripts were removed through MCP. No hidden Player, desktop screenshots or automatic image analysis used. Editor remains open for user inspection; no commit/push, no full041 completion claim.

### 2026-09-13 — A09 Game-relative RenderTexture ownership

- Audited explicit checkboxes before work: main92/256, Compute36/76, combined128/332 (38.6%). After the complete resource-description/relative-size slice below, marked only that A09 item: **main93/256 (36.3%), Compute36/76 (47.4%), combined129/332 (38.9%), 203 unchecked**. Full-scope estimate remains **12–18 solo full-time engineering weeks**, desktop core8–12 subset; not a calendar-date commitment. Full041 goal stays active; old goal wording about a Taichi plugin/headless acceptance does not supersede built-in compiler-only JIT and visible-editor requirements.
- Public `RenderTexture(scale=(0.5, 0.5), ...)` uses Game render pixels as the single reference, mutually exclusive with fixed width/height. Width/height report the actual allocation; fractional pixels round up, positive finite scales may exceed1. Mode/scale/format/MSAA remain immutable. Fixed targets retain resize; relative targets reject pixel resize. Explicit depth sampling/storage flags and existing format capability errors remain, with no quality downgrade or substitute format.
- InxRenderer holds weak references only to relative targets. The Game resize path collects live owners only on a reference-size change; it prepares every replacement generation before publishing any and before committing the new Game target. No frame loop registry scan/hash, script polling, extra device, hidden texture copy or CPU readback. Expired owners are pruned on creation/resize, not retained by the registry.
- Split RHI allocation preparation from publication without changing resource identity or exact GPU-retirement ownership. Native failure injection proves that a failure in the second resource leaves both first/second published generations and sampled-view slots unchanged; successful growth/shrink updates every target and preserves retained old generations. Native test also proves unchanged dimensions allocate nothing.
- Bootstrap prepares the configured Game output while registering panels, before project startup scripts, even when Scene is the visible tab. PlayerGUI prepares actual viewport×render_scale pixels before starting the project after splash/activation. Editor zoom/DPI/Scene dimensions and offscreen camera traversal do not choose the reference.
- Native C++17 build initially rejected accidental std::span/erase_if use; changed to vector/erase-remove, not a language-standard bump. Final Windows Release build passed (`041-relative-rt-build-fixed.log`); initial native3/3 passed (`041-relative-rt-ctest.log`), targeted Python165 passed (`041-relative-rt-python.log`). Full Python **5963 passed / 11 skipped / 0 failed in370.17s** (`041-relative-rt-python-full.log`), with no editor/native build running concurrently.
- Real visible041Lab via MCP9713 (`041-relative-rt-visible-editor.log`, `041-relative-rt-visible.log`) created a half-Game HDR4× Camera A and fixed641×401 Camera B. Two UIImage monitors plus a material in B sample the same A owner. The real Game toolbar changed1920×1080→1280×720→2560×1440→1920×1080: A automatically changed960×540→640×360→1280×720→960×540, revision1→2→3→4; B remained641×401. No project code called resize. Engine captures reviewed manually show both UI monitors and the dependent world material remaining visible; Scene-hidden/Game-hidden switching also preserved active offscreen output. Console **0 warnings / 0 errors**, temporary objects/script removed via MCP, original09 clean Edit and1920×1080 restored. No desktop screenshots or automated image analysis.
- Updated bilingual RenderGraph guide with the same executable relative-target example, ownership/rounding/usage rules and current runtime-only limitation. Regenerated Learn HTML through its existing builder; all20 generated pages passed its --check. Unity comparison uses the official RTHandle reference-size concept, not its maximum-ever reference allocation policy and not a second public resource wrapper.
- Expanded complete CTest audit exposed **80/82 passed**, failures retained in `041-relative-rt-ctest-full.log`. ShaderInfoSchema still failed after rebuilding: a native debug trace and explicit top-level exception report identified `recursive_directory_iterator: Access is denied`, not an RHI allocation failure. Bare virtual source filenames became build-tree source identities, making shader resolution scan unrelated build caches/junctions. Tests now anchor those virtual names under the declared shader root, retain every compile/schema assertion, and report exceptions to stderr with failure exit. No permission skipping/ACL change/production fallback was added. Focused schema test passed0.80s (`041-native-schema-root-fixed.log`).
- GPU Gizmo test had uploaded descriptors once before entering the native loop. The authoritative render-extraction barrier now re-collects live components and replaces those descriptors, correctly discarding the isolated pre-loop upload. Test now attaches a real ResidentGizmoProbe with on_draw_gizmos, preserving GPU line/sphere generation, transform matrix checks, readback values and the >=2 resident vertex buffer assertion. Actual Vulkan test passed (`041-native-gizmo-callback-fixed.log`). A temporary fixture call confused public SceneManager with its native singleton; corrected to the public static get_active_scene, without an engine change.
- Full native rerun and two new executable bilingual documentation cases follow below. Persistent RT asset/Inspector/Cook/export, on-demand view scheduling, complete reflection demo and other platforms remain open. No commit/push or full041 completion claim.

#### Final regression outcome

- Rebuilt native schema test passed with explicit source roots; removed temporary trace prints. Kept the test's failing-exception stderr report and Windows nonmodal assertion mode, not a runtime catch/fallback.
- Complete native suite **82/82 passed in123.84s**, including **15 real Vulkan tests** (`041-relative-rt-ctest-full-fixed.log`). This supersedes the earlier80/82 failure, which remains recorded.
- Post-documentation resource/View/Game/Player regression **101 passed in9.74s** (`041-relative-rt-doc-regression.log`), including two new tests that execute the identical English/Chinese sample against real Camera/RenderTexture owners at odd641×401 and1920×1080 Game resolutions. Those two new tests are not included in the earlier5963 full-suite count. No production sources changed after that full Python suite; later fixes affect native/GPU test fixtures and generated documentation.
- Closed every test process before reopening the visible editor. Follow-up verification uses the original09 scene, not a hidden Player; no commit, release or full041 completion claimed.
- Final MCP project/Console query confirms original09 **clean Edit**, **0 warnings / 0 errors** after a fresh launch (`041-relative-rt-final-editor.log`). Human-reviewed engine `review/041-relative-rt-final-clean.png` shows the normal Audio scene and its HUD immediately available. Editor intentionally remains open; the temporary regression runner and pytest are no longer running.

### 2026-09-13 — RenderTexture imported descriptions and same-frame camera lifetime

- Strict progress remains main93/256 + compute36/76 = **129/332 (38.9%)**,203 outstanding. No additional checkbox: persistent Inspector, Camera/material/UI GUID references and live GPU-owner reconfiguration remain unfinished. Full-scope estimate remains12–18 solo full-time engineering weeks; this is not a promised calendar completion date.
- Added the `.rendertexture` authoring description, validated and serialized through native RenderTextureDesc. CPU-only import produces versioned CBOR `.inxrtex` artifacts using the existing source-content identity, AssetIndex and import transaction. No rendered pixels, second package format, new per-frame hash or source-JSON runtime fallback. Registered the native CPU-description loader and resource type; same-GUID reimport updates the loaded description in place. Invalid source edits cannot overwrite the committed artifact.
- Added Project creation-service/MCP support using native defaults, without another Python schema. Asset Inspector/public persistent binding is explicitly not yet delivered. Cook includes binary descriptions in Content.inxpkg and excludes authoring source files, for Assets and selected Packages. A real package fixture exposed an existing general omission: selected plugin GUIDs were absent from the compiled-artifact closure. They now enter the same dependency traversal as project roots; disabled plugins are not indiscriminately included.
- Native targeted tests4/4 and initial complete CTest83/83 passed, including15 real Vulkan tests (`041-rt-asset-native-full.log`). First complete Python run:5986 passed,13 skipped,3 failed (`041-rt-asset-python-full.log`). One was a documentation sample importing internal PixelFormat; both languages now use public `inx.rendergraph.Format`, with executable example tests and generated Learn pages checked.
- The remaining rename errors also occurred in an independent stdlib-only100-directory probe: Windows returned transient WinError5 twice, and both same-target publications succeeded after the handle window elapsed (`041-directory-publish-probe.log`). This proves the symptom does not require the engine; the holding process was not identified. Added one build/Player IO-boundary helper matching existing native AtomicFile policy: normal one-rename path, Windows5/32/33 only, max8 attempts/254ms; permanent failures propagate. No permission changes, antivirus disablement, copy/merge fallback or old-generation recovery. Eight deterministic tests cover normal success, bounded Windows behavior and immediate failures elsewhere.
- A subsequent combined regression crashed instead of returning a test failure. Native exception tracing identified InxRenderer::EnsureGameRenderGraph through GetFrameTelemetrySnapshot (`041-rt-snapshot-debug.log`), not filesystem publication. Borrowed Camera pointers were cached until frame-begin even when scenes were unloaded or edited between queries in the same frame. Cache hits now compare existing loaded-scene world IDs/structure versions and active/persistent scene identity before dereferencing; misses rebuild through the normal camera enumeration. Camera target changes bump the existing structure version. No dangling-pointer probing or extra hash. Two regression cases cover repeated additive scene unload and enabled/order edits without rendering a frame.
- Latest native rebuild completed (`041-camera-cache-build.log`); focused Python **66 passed in4.85s** (`041-camera-cache-regression.log`). An earlier build attempt overlapped pytest and failed to stage the locked native DLL; serial rebuild succeeded. Subsequent builds, native/Python tests and the visible editor must run serially. Final complete reruns and visible-editor import/edit checks are recorded below when actually finished.
- Complete Python retry reached **5999 passed /13 skipped /1 failed** (`041-rt-assets-full-verified.log`): the remaining failure was the same WinError5 during JSON-file replacement, rather than directory replacement. Generalized the one helper to `replace_path` and used it consistently at GameBuilder/Player-bootstrap file and directory publication boundaries. Existing output rollback/error propagation is unchanged. Added real file/directory publication tests alongside deterministic failure-policy tests; no separate retry implementation.
- A related lifetime audit reproduced two further failures: removing an explicitly preferred Camera component or destroying its owner left Scene.main_camera referencing freed memory (`041-preferred-camera-lifetime.log`). Camera destruction now releases that borrowed scene reference, using the existing setter/structure version. This is owner cleanup, not a recurring validity check. The entire native build, including all test executables, was rebuilt successfully (`041-rt-camera-final-build.log`) before final regression; no tests/editor ran during the build.
- Final complete Python regression: **6004 passed /13 skipped /0 failed in306.76s** (`041-rt-assets-final-python.log`). This includes both Camera deletion cases, same-frame cache lifetime, RenderTexture descriptions/Cook, real file/directory publication and failure boundaries. The expected rejected-input native error logs are test fixtures, not test failures. All20 generated Learn pages and changed-file whitespace checks also passed.
- Final complete native regression: **83/83 passed in144.01s**, including **15 real Vulkan tests** (`041-rt-assets-final-native.log`). After all tests exited, launched the real visible041Lab editor (PID21572, `041-rt-assets-visible-editor.log/.err.log`); MCP confirms the original09_AudioVoiceField is clean Edit. Visible authoring verification follows.
- Visible MCP authoring check passed (`041-rt-assets-visible-check-fixed.log`): created RenderTextureAssetProbe under Assets/Rendering; imported184-byte INXRTEX1 descriptor; edited to relative(.5,.25),RGBA16_SFLOAT,D32,4×MSAA,sampled depth; same GUID1cddbde07e7ad3f5bd48ad7e4205b0af published the185-byte replacement. Deleted the temporary asset through Project history. Original09 remains clean Edit, Console0 warning/0 error. Reviewed engine-only1920×1009 `review/041-render-texture-asset-clean.png` visually; no pixel-analysis access was used. An initial diagnostic script failed before asset creation because its helper parameter `name` collided with asset.create's name argument; renamed the helper parameter, with no production-code change.
- Only the intended visible editor process remains; pytest/native-test/diagnostic processes exited. No release, remote push, new branch or full041 completion is claimed. The current editor is left open for the user's inspection.

#### Next A09 dependency slice (not delivered by this record)

- Keep CPU authoring/import independent of the graphics device. Resolve imported descriptions by the existing GUID registry into the same public RenderTexture owner, not another public texture class or a source-file parser in gameplay.
- Add persistent authoring references through the existing asset-reference/type/field catalog; Camera, material and UI should consume that same identity. Runtime-created targets still have no invented asset GUID. Add the Project entry and resource Inspector through the current authoring/Undo service rather than a separate MCP-only workflow.
- Reimport must publish a whole supported GPU target generation at an owner safe point and preserve every live consumer's owner identity. Fixed/relative changes must also update the existing relative-target ownership tracking. Do not introduce per-frame asset polling, reconstruction in each consumer, source fallback or partial attachment publication.
- Demonstrate create/edit/reference/save/reopen and a packaged Windows Player before checking the persistent-asset/export item. Other platforms remain explicit acceptance work. The current CPU artifact tests do not stand in for that end-to-end workflow.

### 2026-09-13 — Shared imported RenderTexture owners and real asset authoring

- Strict re-count: main93/256 + compute36/76 = **129/332 (38.9%)**,203 outstanding. No whole-item checkbox added: direct persistent built-in Camera/material/UIImage slots, final Player use and other platforms remain outstanding. Full-scope estimate remains **12–18 solo full-time engineering weeks**; not a calendar delivery promise.
- `RenderTexture.load/load_by_guid` and `RenderTextureRef` use the ordinary GUID/field/asset catalog. Runtime allocations do not invent GUIDs. `AssetManager` remains the Python cache authority; references have no extra cache that could conceal deletion/Undo. The renderer holds weak imported-owner records, and aliasing shared ownership pins the CPU description only while graphics consumers live. Restoring a deleted GUID reconnects a surviving graphics lease without creating an independent resource identity.
- Loader reimport calls the renderer before changing the committed CPU payload. A whole GPU generation is prepared first, including format/MSAA/depth/storage and fixed/relative tracking. Device-limit/allocation failures preserve the old owner and registry version. No per-frame source polling, content hash loop, silent format downgrade, source-loader fallback or additional disk cache.
- Camera retains its depth requirement across disable and clone. Reimport cannot drop a depth attachment while a Camera still holds it; releasing all Camera owners permits a depthless resource. Covered real native clone, component removal, deletion/restore, shared camera owner, registry CPU budget and relative/fixed transitions.
- Added Project creation entry, native drag payload, editable resource document category and Inspector. The Inspector is a CPU-only authoring adapter for the existing public resource; format choices come from the native codec. Writes, Undo/Redo and scene fields use existing authoring services. Numeric/boolean controls publish normal semantic targets for MCP; opening the asset alone does not allocate a GPU texture.
- Build logs: `041-rt-resident-assets-build.log`, `041-rt-resident-assets-build-final.log`, `041-rt-inspector-build.log`; all exited0. First focused ownership/fields run **138 passed in6.27s**. Inspector/document/menu run **68 passed in5.66s**.
- First complete Python run: **3 failed,6018 passed,11 skipped in377.77s** (`041-rt-resident-full-python.log`). Failures identified an omitted lowercase public stub export, an outdated exact Inspector-category contract and new documentation importing the internal uppercase package. Added the actual public stub export, updated the expected supported category and used the existing lowercase namespace; no assertion suppression. Complete rerun **6021 passed,11 skipped in356.23s** (`041-rt-resident-final-python.log`).
- Complete native **83/83 passed in136.85s**, including15 actual Vulkan tests (`041-rt-resident-native.log`). Subsequently added the missing Project native drag-map entry and rebuilt `_Infernux` successfully (`041-rt-drag-payload-build.log`); related Project/Inspector/public namespace/ownership regression **169 passed in8.22s**.
- Visible editor launched with source Python and current native module; all native builds/pytest ran only with the editor closed. An initial local harness assumed the wrong `.meta` nesting and failed before writes. A second run captured a correctly rendered Inspector but exposed missing numeric semantic targets; added those targets rather than relying on guessed coordinates. A later harness sent the next scene-open request before the previous deferred transaction completed; fixed the harness to wait for published active-scene identity. The renderer/source were not changed to hide the warning.
- Real Play then exposed an error in the newly written example: components access siblings through `self.game_object.get_component(...)`, not `self.get_component(...)`. Fixed both English/Chinese examples and the local probe. Added a regression that executes the actual identical Markdown lifecycle body against real Camera/RenderTexture resources. Final post-change focused suite **170 passed in13.76s** (`041-rt-resident-verified-focused.log`); this one new example test is not included in the preceding6021 full-suite count.
- Visible accepted workflow (`041-rt-authoring-visible-5.log` and `041-rt-authoring-visible-drag.log`): real Project search/selection; Inspector Ctrl-click width256→333 with autosave; actual drag from Project into the script Output asset field; scene save, switch away and reopen; persisted GUID94188a2765c845be416a871924f74aef restored; Play held that GPU owner and rendered333×256; source reimport changed it to517×291/HDR/4×MSAA, revision2, without reassigning the Camera. Captures from the Camera itself verified both dimensions; human-reviewed Editor and Camera images showed the real Audio test scene. No pixel-analysis access was used.
- Temporary RT/script/scene were deleted through the Project transaction system, with recoverable Undo backing; original09_AudioVoiceField was not overwritten and returned to clean Edit. Final accepted run Console **0 warnings /0 errors**. The failed harness logs remain recorded; Play's ordinary Clear-on-Play setting starts the subsequent clean run. No PR, remote release, full041 completion or Player/platform acceptance claimed.
- Final ordinary `git diff --check` passed; all20 generated Learn pages verified. After the170-test process exited, reopened the visible editor with the final sources (PID12120, `041-rt-authoring-final-editor.log/.err.log`). MCP confirms original09 clean Edit,0 warnings/0 errors; human-reviewed `review/041-rt-authoring-final-clean.png` is a normal full editor capture. Only that intended editor remains running; no test/build process or temporary authoring asset remains active.

### 2026-09-13 — Persistent Camera targets, dependency rekey and history safe point

- Recounted main93/256 + compute36/76 = **129/332 (38.9%)**,203 outstanding. Full scope remains **12–18 solo full-time engineering weeks**. No additional whole-item checkbox: persistent material/UIImage slots, packaged Player use and other platforms remain unfinished.
- Camera now declares one native ASSET field, `target_texture`, serialized as `targetTextureGuid`. Python CppProperty, native Inspector projection and MCP consume that schema. CPU scene deserialization stores identity only. The renderer resolves it at normal Camera-cache publication; a missing authored output does not become screen output. Runtime allocations still have no invented GUID. Native preflight/Cook follows the same typed dependency.
- Asset callbacks reconnect a deleted/restored target and preserve the same live graphics owner across reimport. No per-frame source polling, fallback texture, duplicate public resource class or additional cache. Retained Camera depth requirements continue through clone/disable.
- Found a generic scene-publication bug: committing staged component IDs left runtime dependency edges registered under staging IDs. Previously Scene only rebuilt some MeshRenderer edges. Added one dependency-owner rekey at Component and Scene commit boundaries, preserving all asset edges without rebuilding the immutable asset graph. Real Scene and GameObject subtree tests failed before the fix (2failed, `041-camera-asset-rekey-before.log`) and passed after it (2passed, `041-camera-asset-rekey-after.log`).
- Initial full Python run had4failures (`041-camera-asset-full-python.log`): outdated exact Camera field counts, generic Inspector trying to decode raw native GUID as a Python asset document, and an old local Windows Player payload. Fixed the Inspector through the declared setter-owned field projection and regenerated the Player through the existing CMake `prebuild_player_runtime` target, not manual file copying. All native build/release steps completed locally; no remote artifact publication is claimed.
- Complete post-rekey native regression **83/83 passed**,15 actual Vulkan tests,123.24s (`041-camera-asset-rekey-native.log`). Complete Python **6031 passed /11 skipped**,358.82s (`041-camera-asset-rekey-python-full.log`). Build, pytest, CTest and the visible editor ran serially.
- Visible authoring uncovered an asset label displaying Python repr; RenderTexture now provides the ordinary reference display name without extra asset resolution. The first deletion/Undo harness also clicked Project navigation after deletion, making navigation the actual next history entry. Moved that focus operation before deletion; did not weaken asset assertions.
- Manual cross-scene Undo then reproduced a real owner-safe-point error: UI immediately restored Scene/script dispatch while a native frame was active. UI now queues the existing Undo/Redo replay; the established pre-scene owner boundary drains it before native frame begin. Removed its pre-GUI drain. No exception suppression, retry loop or alternate scene loader. Added deferred replay coverage including an actual native scheduler frame.
- Historical focused run **147 passed in6.87s** (`041-camera-history-focused.log`). Final full Python rerun, including the real-frame history test, is in progress (`041-camera-history-full-python.log`); record its actual result below after completion.
- Visible accepted workflow (`041-camera-history-visible-final.log`): Project creates and edits the resource; drag directly to built-in Camera Target Texture; save, switch and reopen; camera capture in Edit and Play; reimport333×256/1× to517×291/HDR/4× with unchanged asset identity; delete disconnects the output without clearing GUID; ordinary Ctrl+Z restores the output. Then switch to original09, Ctrl+Z restores the temporary scene and its camera output, Ctrl+Y returns to09. All via visible editor/MCP, not headless scene substitutes. Final original09 clean Edit, Console **0 warnings /0 errors**.
- Engine-only captures include `041-rt-camera-slot-dragged-field.png`, `041-rt-camera-slot-delete-undo-camera.png`, `041-rt-camera-slot-history-restored-camera.png` and `041-rt-camera-slot-restored.png`. Human-viewed field/output captures confirm the friendly label and rendered scene; pixel-analysis permission remains disabled. Temporary scripts/scenes/targets from failed and accepted runs were deleted through Project history with recoverable backing; original09 was not overwritten.
- Unity comparison: [Unity6 Render Texture authoring](https://docs.unity.cn/6000.0/Documentation/Manual/output-to-render-texture.html), specifically Project resource creation and Camera Target Texture assignment. This does not claim complete Unity feature parity. English/Chinese RenderGraph guide now documents the direct field and asset lifecycle; generated20 Learn pages checked before the final history-only change.
- Final complete Python run **6032 passed /11 skipped /0 failed in357.15s** (`041-camera-history-full-python.log`), including actual native-frame deferred history coverage. Negative-test shader/layer errors after the pytest summary are intentional rejected-input fixtures, not failed tests. All20 generated Learn pages rechecked successfully (`041-camera-history-docs-check.log`); ordinary `git diff --check` passed (`041-camera-history-whitespace.log`).
- After pytest exited, reopened the final visible editor, PID28188 (`041-camera-history-final-editor.log/.err.log`). MCP confirms original09 clean Edit and Console0 warnings/0 errors; only this Python process remains, with no test/build worker. No temporary RenderTextureAuthoringProbe scene/target or RenderTextureAssetProbe script remains. Final engine capture: `review/041-camera-history-final-clean.png`. No commit/push, remote artifact release or full041 completion claimed.

## 2026-09-13 — Imported RenderTexture material consumers and authoring

- Recounted main93/256 and Compute36/76:129/332 (38.9%),203 pending. Full-scope estimate remains12–18 solo full-time engineering weeks. A09 is not checked complete: direct UIImage asset authoring, full Player/platform usage and other exits remain pending.
- Materials retain the existing Texture2D GUID document shape. Native preparation resolves imported RenderTexture descriptions only when bindings are pending, before render-graph material read collection. Imported targets share the renderer's existing owner; anonymous targets remain nonpersistent runtime overrides. No per-frame asset DB scan, hash, extra package format or CPU image readback.
- Existing AssetDependencyGraph runtime edges now cover live materials and clones under their unique runtime IDs; weak owner indexing uses existing periodic GPU maintenance and material destruction releases its edges. Reimport retains live graphics ownership until atomic publication; deletion clears only matching non-runtime bindings and keeps authored GUIDs. Scalar property updates do not trigger texture resolution.
- Material Inspector uses a sampled-texture field descriptor without widening static-only texture fields elsewhere. Removed its old explicit TEXTURE_FILE-only drag restriction. Clipboard projection preserves the concrete Texture/RenderTexture kind rather than exporting the field's union constraint as a resource type.
- Visible harness initially used search-row semantic IDs that only exist for ordinary folder items; selecting a search result navigates to its folder. Updated the harness to wait for asynchronous search publication, select its actual row, and drag between same-folder assets. Also waits for the ordinary material save debounce, traverses nested UI hierarchy, and supplies canonical serialized field names/reference documents. These were test-harness errors, not reasons to weaken production checks.
- Correctly named native Camera asset writes also exposed a host adapter mismatch: setter-owned fields were being passed through a native getter converter. They now use the same ComponentCommandService transaction as Inspector, including normalization, setters and Undo. A real Camera/asset regression covers set/clear/Undo/Redo; the focused component/material suite passed52tests (`041-material-rt-automation-focused.log`).
- Windows ALL_BUILD and formal CMake prebuild_player_runtime succeeded; the latter regenerates the local Windows platform payload directly. Native83/83 passed, including15 actual Vulkan tests,128.36s (`041-material-rt-native.log`). First full Python run:6036passed,11skipped,1failed from an outdated fake UI material; fixed the fixture rather than adding a production fallback. Then177passed/1skipped in the material/UI suite. Final full run is recorded below when complete.
- Accepted visible MCP run: `041-material-rt-visible-final.log`. Create target/material, drag target into material Inspector and save, author Camera output and UI material, save/switch/reopen, Play, Cube material clone and screen UIImage sample the same target, reimport333×187 to517×291/HDR/4×, delete/ordinary Ctrl+Z reconnects both consumers. Final original09 clean Edit,0 warnings/0 errors. Temporary assets deleted through recoverable Project history; original scene bytes preserved.
- Engine captures `041-material-rt-inspector-drag.png`, `041-material-rt-game.png`, `041-material-rt-reimport-game.png`, `041-material-rt-delete-undo-game.png` were human-viewed. The last shows both the in-world Cube and separate screen image, not two assertions over invisible data. Pixel analysis remains disabled.
- English/Chinese RenderGraph guide and material stub updated;20 generated Learn pages checked. Unity comparison is limited to its Project→Camera→Material RenderTexture workflow, not full feature parity. No commit/push or remote artifact publication.
- Final complete Python regression **6039 passed /11 skipped /0 failed in347.96s** (`041-material-rt-final-full-python.log`). The last clipboard-kind adjustment was then covered separately by **218 passed /1 skipped in9.30s** (`041-material-rt-clipboard-final.log`), including both newly added concrete-kind cases, reference models, material Inspector, Camera assets and UI. The clipboard-only change is not claimed to have another complete-suite run.20 generated Learn pages and normal git diff --check passed. No temporary MaterialRTProbe/MaterialTargetProbe files remain in Assets.
- After all tests exited, reopened the current visible editor (PID34912, `041-material-rt-restored-editor.log/.err.log`). MCP confirms original09 clean Edit,0 warnings/0 errors. Only this Python process remains. Next scoped work: direct UIImage persistent texture authoring and final Player consumption; full041 remains active and incomplete.

## 2026-09-13 — Fullscreen attachment state and the actual selection consumer

- Recounted explicit items: main **94/256**, Compute **36/76**, combined **130/332 (39.2%)**, **202 pending**. No partial A09/A10 slice is promoted to complete. Unchanged full-scope estimate: **12–18 solo full-time engineering weeks**, not elapsed calendar time. A02 is7/7; major open areas include CPU/GPU JIT delivery/performance, physics interaction matrix, remaining UI/author tools, complete Runner Long visuals, Steam/cloud, model import and packaged platform acceptance.
- Read the source game's registry, renderer feature, object-mask shader and composite shader. Its effect needs logical-owner IDs, eye depth, invalid/hatched/depth-tested state, occlusion handling and fragment depth. The11selection gallery is a generic public API consumer, not a claim that the complete A10 port is done. Unity6 ZTest/ZWrite/Blend references were checked and linked in both language sections of the public guide.
- Added `fullscreen_quad(shader, depth_test=DepthCompare|None, depth_write=False, alpha_blend=False)` through the public facade, Python IR, bindings and native fullscreen pipeline. Reuses RHI depth and straight-alpha state; cache identity includes depth format/test/write/compare and blending. Depth/blend passes preserve previous color unless explicitly cleared; sampled depth cannot simultaneously be its attachment. New transient loads are rejected at graph compilation, not repaired at runtime. Default replacement behavior remains unchanged.
- Actual Vulkan regression covers D32 and D24S8 at1×/4×: four successive draws prove fragment-depth publication, rejection behind the previous fragment, depth-write disabled, source-over color/alpha and load/resolve. The same fixture now copies a persistent depth attachment into a distinct transient snapshot. Initial full native83/83 and Python6112/11skipped passed, but subsequent live testing exposed issues that these earlier results did not cover.
- The first author helper incorrectly renamed the conventional root `depth`, so the automatic World UI insertion could not build. This was a helper error, corrected to the established pipeline contract. During the broken graph's hot update, a new shader's3input interface met the old2input graph and caused a GPU hang. Fullscreen pipeline creation now checks reflected set0 bindings against the graph's input count; no per-frame reflection, hash, old-shader fallback or CPU wait. Fault injection in the real editor subsequently rejected binding3, kept the editor alive, and restored the original Shader through MCP without a restart (`041-fullscreen-hot-reload.log`).
- The valid depth-copy consumer exposed a separate engine error: all viewport-sized depth names were aliased to the Camera depth, making the copy source and destination the same image. Registration now shares only the conventional root `depth`; all other transient names allocate independently, including full-resolution masks and snapshots. Merged duplicate color/depth registration code and removed the inline shadow fallback. Persistent depth images declare transfer source/destination support just like their color counterparts. View/material resolution uses the same identity rule.
- Python scene-graph Copy passes now stay on Graphics, like the existing MSAA resolve passes. They are interleaved with dependent raster work, so a dedicated transfer queue added ownership crossings without useful overlap. General native multi-queue Compute is not serialized. Separately, a transient first-on-Transfer/last-on-Graphics replay exposed an ownership-acquire mismatch in the general native queue path; this is not claimed repaired by the scene-graph scheduling choice and remains part of generic queue/lifetime acceptance.
- Final-depth build and native suite before the subsequent ComputeBuffer correction: **83/83 in99.42s**,15 real Vulkan tests. Live11Edit/Play, GPU deformation, submesh0/1, empty/restored selection and Stop/reopen passed;9selection changes did not alter the topology build count. Six real SDL UI clicks toggled sphere/sculpture/colour and restored the original values (`041-depth-final-buttons.log`). These captures were engine-owned and human-reviewed, never OS screenshots or pixel-analysis scripts. Actual view image: `review/041-selection-moving-game.png`.
- Validation during GPU deformation found `ComputeBuffer::SetData` requesting a VertexInput barrier on a dedicated Compute queue. The upload now barriers only to Compute; Graphics consumes its existing write ticket/timeline, which VkCoreDraw already acquires for resident vertex buffers. This removes the invalid stage without a global wait or graphics-only compute fallback. GPU CTests now fail on validation errors when a validation layer is enabled, rather than passing solely on exit status.
- The first complete suite with Khronos validation enabled failed two mesh-publication tests: numerical retirement passed, but device destruction reported four shader modules, two pipeline layouts and two descriptor layouts still alive per test (`041-depth-validated-ctest.log`). The cause was material identity, not a need for another shutdown sweep: imported model slots create independent runtime materials but stamp the same source path; `GetMaterialKey()` treated that path as shared identity, replacing the cache owner while other material instances still retained shader publications.
- Removed the source-path identity branch. GUID assets retain GUID identity; all non-GUID material instances retain the existing runtime ID, independent of display name or source provenance. Added native coverage for same-source instances, rename/move, copy construction, GUID sharing and cloning. Temporary lifetime instrumentation was removed; no extra runtime manager, retry or shutdown scan was introduced.
- Final Release build passed (`041-material-identity-build.log`). Complete native suite with the Khronos layer: **83/83 in99.53s**, including15 real Vulkan tests; the complete CTest `LastTest.log` contains zero validation errors and zero temporary trace messages. Both previously failing mesh-publication cases now pass (`041-material-identity-ctest.log`). Full Python suite against this final build: **6112 passed / 11 skipped in276.51s** (`041-material-identity-python.log`). Guide generation consistency and whitespace checks passed.
- Fresh visible editor PID50080 with Khronos validation passed the complete11exercise (`041-material-identity-visible.log`): GPU-deformed geometry, isolated submesh0/1, empty/restored selection, colour changes and clean Stop/reopen. Nine selection changes kept the topology count at2. The same three UI buttons were each toggled twice by actual SDL pointer events (`041-material-identity-buttons.log`). Engine-owned screenshots were manually reviewed, with no OS capture or pixel-analysis processing. A240frame diagnostic window measured full-frame mean1.796ms/P95 2.713ms at1920×1080 with validation enabled; this is only the11selection consumer, not proof of the softbody300FPS acceptance or a benchmark against Unity.
- MCP normal window close terminated PID50080. Raw stdout/stderr through shutdown contained zero validation errors, DEVICE_LOST, ERROR or traceback lines (`041-material-identity-editor.log` and `.err.log`); Console remained0warnings/0errors. Reopened one normal visible editor without the diagnostic layer, PID46912, confirmed `11_SelectionStudies` clean Edit via MCP. No build/tests ran concurrently with either editor. No commit, push, PR, release or full041 completion claimed. Strict progress remains130/332; continue A10 logical-owner ID/depth/state compositing through public APIs, with the general queue/View/MSAA boundaries above still explicit pending work.

## 2026-09-13 — World UI pass layers and Project-created Camera targets

- Recount at entry: main98/256 + Compute36/76 =134/332 (40.4%). The saved plan already contained the four accepted owner-outline items and scene12. Prior complete Python evidence on disk was6110passed/13skipped (`041-outline-python.log`). Do not replace the current saved plan with the older130/332 paragraph above.
- Added one unsigned32 World UI layer mask to the existing graph command, public `draw_world_ui(layer_mask=...)`, section helpers, binding and stubs. The native graph cache includes it. Both rendering and RenderTexture dependency collection intersect it with the current Camera mask. Ordinary World UI retains its existing renderer, attachment signature, depth test and no-depth-write state. No additional UI pipeline, per-frame validation, source scan or fallback.
- Release ALL_BUILD and focused native test rebuild exited0. Complete native suite with Khronos validation:83/83, including15 actual Vulkan tests,111.59s (`041-world-ui-pass-ctest.log`). LastTest.log has no validation errors/VUIDs. Initial Python focused run145passed/1failed because the new missing-depth assertion was incorrectly placed at Python projection; moved attachment rejection coverage to its actual native validation boundary. Full Python then6132passed/13skipped in233.28s (`041-world-ui-pass-python.log`).
- Visible editor PID25216 loaded `VkLayer_khronos_validation.dll`. Added ordinary layer29 and late layer30 UIText objects to scene12 through MCP. The project pipeline draws ordinary UI, applies a saturation grade, then draws late labels before display encoding, choosing either current scene depth or an explicit pre-outline depth snapshot. New `OutlineProjectGrade.frag` is a project asset; no built-in outline or grade effect was added.
- The first authoring attempt incorrectly copied the grade output into a camera-target texture. The existing compiler correctly rejected `CommitGrade`; the user reported that message during authoring. Replaced this project copy with the supported fullscreen blit and hot-published the script through MCP. No relaxation of the native camera-target/copy contract. These two historical Console errors are not counted as successful acceptance.
- Final live stage matrix (`041-world-ui-visible.log`): perspective/orthographic; ordinary-only versus split early/late masks; Camera excludes/restores layer30; both declared depth choices under a full wall. Console0warnings/0errors. Human-reviewed engine captures show both labels graded when drawn early, only the late label retaining orange when split, that label absent when its Camera layer is excluded, and both hidden behind the wall. Stop/reopen returns clean Edit. No image pixel analysis or OS screenshots.
- User requested persistent RenderTexture authoring again. Confirmed the real Project context menu has Create→Render Texture, then used actual SDL mouse/keyboard events to create/rename `Assets/Rendering/OutlineCapture.rendertexture`, edit its size to800×450, and drag it to the built-in Camera Target Texture slot. Added the ordinary `RenderTexture Camera` to scene12. GUID549c6024b563514b83dfb04935ab1707 is stored in the scene, not in a script. Save, switch to11, reopen12, Edit camera capture and Play camera capture all retained that identity and800×450 output (`041-rendertexture-delivery.log`). Human-reviewed camera-field and output captures confirm the actual binding and rendered geometry. The example remains in the project for the user; it is not a deleted temporary probe.
- This UI walkthrough exposed an awkward authoring default: new Project targets were color-only and required manual depth configuration before Camera use. Project creation now uses D32 depth by default; the low-level resource descriptor remains color-only, and users can still explicitly disable asset depth. Added a real native Camera assignment test for the newly created asset;96 related tests passed (`041-world-ui-rt-default-tests.log`). No auto-repair of previously authored targets. In the visible pre-restart walkthrough the example's depth was explicitly set through Inspector.
- Normal MCP close terminated PID25216. Native logs through shutdown have no VUID/validation error/DEVICE_LOST. Build, native tests, Python suites and the visible editor ran serially.20 generated Learn pages and whitespace checks pass. The final complete Python run and fresh-editor confirmation follow below once their actual results are available.
- Checked only the completed A10.1 late-world-text/postprocessing item: main99/256 + Compute36/76 =135/332 (40.7%),197 remaining. Estimate remains12–18 solo full-time engineering weeks for the complete scope. Thin occluders/alpha-clip edges, full effect migration, remaining queue/history/platform and JIT/physics gates are still pending. The old2.58ms scene12 number predates these extra stages and the Camera, so is not reused as a current performance claim. No commit/push/release or full041 completion.
- Final complete Python regression with the camera-ready creation default: **6133 passed /13 skipped in225.63s** (`041-world-ui-rt-final-python.log`), exit0. Expected broken-shader/collision-matrix negative fixtures log errors after the summary; no test failed. Documentation generation/check and final whitespace check exited0.
- After all test processes exited, opened the normal visible editor PID40776 without the diagnostic layer (`041-world-ui-delivery-editor.log/.err.log`). Cold check (`041-world-ui-cold-check.log`) verified scene12 clean Edit, persisted Camera GUID, actual800×450 camera output and1920×1080 Game output before Play. Created one temporary `CameraDefaultsCheck.rendertexture` through the ordinary Project asset command and confirmed D32 was present without editing depth, then removed only that temporary asset through recoverable Project history. The saved OutlineCapture resource and RenderTexture Camera remain. Human-reviewed final engine screenshot shows the selected Camera target field and Project resource. Console0warnings/0errors. Only the intended editor remains running; no build/test worker is left.

## 2026-09-13 — Thin occlusion coverage and nested Inspector semantics

- Revalidated the active visible editor, saved scene12, current source files and plan before continuing. Main99/256 + Compute36/76 was135/332. Read Runner Long's repository instructions and current engineering boundaries, then compared its actual outline mask/composite source without starting Unity or modifying the Unity project. Its replacement mask contains no source-material UV/alpha test; its occlusion hatch uses a cross-shaped maximum eye depth and absolute/relative bias. Do not invent automatic material inheritance as Unity behavior.
- Added project Inspector parameters for absolute/relative depth bias and pixel neighborhood radius. Radius0 provides a genuine per-pixel comparison; the default1 retains the original effect. Only automatic occlusion hatching uses neighborhood suppression; explicit depth-tested output still compares the current pixel. Kept existing categorical texelFetch data, independent depth, fragment-depth publication and public RenderGraph passes.
- Added `OutlineOccluder.frag`, file materials `OutlineThinFence` / `OutlineWideFence` / `OutlineGlass`, and the opt-in `Coverage Fence` quad to scene12 through MCP. The fence is saved inactive so the existing scene and persistent Camera target remain usable. Thin/wide coverage is a procedural alpha-cutout material, not many CPU-created thin meshes. Glass is a separate file material with alpha. The original game and shared Lab materials were not overwritten.
- Two author-helper mistakes were corrected: ShaderInfo rejects the helper's trailing-dot `80.` literal (changed to ordinary80.0), and object.create returns component names rather than full component documents (query the existing hierarchy). Initial shader publication was correctly rejected and retained its prior valid asset; no parser fallback or error suppression added. Both historical errors remain in `041-world-ui-delivery-editor.log`; acceptance starts in the fresh process below. Resumed only the newly created fence, preserving the rest of the scene.
- The real Inspector snapshot exposed a public engine defect: nested RenderPipeline parameters appeared as unnamed sliders, without stable semantic IDs. `InspectorSerializedTarget` now records each rendered field through the existing component-item helper, using the owner Component plus declared control/field names. Pipeline object addresses are not field identity. Request-only capture, hidden/read-only metadata and existing transaction callbacks are preserved. The production change is7lines, with no new storage, registry, mutation path or fallback.
- Closed PID40776 normally before tests. Related Inspector tests **100passed in4.64s** (`041-outline-semantics-tests.log`). Complete Python **6136passed /13skipped /0failed in229.76s** (`041-outline-semantics-full.log`). Negative-input fixtures retain expected logged errors after the passing summary. Native code did not change;83/83 CTest is previous-turn evidence, not a new native run.
- Opened visible editor PID23260 with Khronos validation and verified the actual layer DLL was loaded. `041-outline-coverage-visible.log` completed32combinations of perspective/orthographic, narrow/wide cutout occluders, opaque/alpha-blended selected sphere, explicit depth mode and radius0/1. Radius was edited through actual SDL Ctrl-click/text entry into the newly identifiable Inspector slider, and the persistent RenderStack document was checked after each edit. Four additional800×450 Camera captures exercise a different projected coverage size. No image pixels were analyzed; engine PNGs are human-review-only.
- Human-reviewed representative1920×1080 captures show per-pixel thin-line white noise reduced with radius1, retained wide-wall hatching, opaque scene coverage absent through the cutout holes, glass geometry retaining its white silhouette, and explicit depth-tested output hidden behind solid fence strips. `041-outline-coverage-motion.log` adds6captures with the thin fence translating and the GPU sphere deforming, across both projections; reviewed the changing geometry and outline. This is not a claim of every grass shader, source-material vertex override or selected alpha-cutout silhouette having been accepted.
- `041-outline-parameter-history.log` uses real Edit-mode Inspector input, Ctrl+Z/Ctrl+Y/Ctrl+Z: radius1→2→1→2→1, stable semantic field and persisted document verified, then clean scene reopen. `041-outline-coverage-buttons.log` rechecks all6existing gameplay buttons and owner regrouping. During flag/member changes Python topology build count stayed at3; Inspector topology inspection now accounts for an additional initial probe, so do not reuse the earlier2 count.
- All visible runs finished clean Edit with0warnings/0errors. MCP close terminated PID23260. Its complete stdout/stderr through shutdown contain no VUID, validation error, DEVICE_LOST, ERROR or traceback (`041-outline-coverage-editor.log/.err.log`). No test process overlapped either editor; all handles were observed terminal. English/Chinese rendering-guide explanations were updated and all20generated Learn pages checked; whitespace check passed.
- Checked only the neighbor-depth/bias/fragment-depth A10.1 item: main100/256 + Compute36/76 = **136/332 (41.0%)**,196remaining. Full fixed-scope estimate remains **12–18 solo full-time engineering weeks**. The final A10.1 combined matrix still awaits an explicitly matching mask for a selected alpha-cutout object; remaining effects, history/queue/platform and JIT/physics/publishing gates remain open. No commit/push/remote release or full041 completion is claimed. Fresh normal-editor cold verification follows below after completion.
- Final normal visible editor PID77944 (`041-outline-coverage-final-editor.log/.err.log`) cold-loads12clean Edit, Console0warnings/0errors, ordinary1920×1080 Game output and the original saved Camera target GUID. `Coverage Fence` remains inactive. Engine-owned `review/041-outline-capture-editor.png` was human-viewed; it shows the actual editor, scene geometry, Project assets and no console error. `041-outline-coverage-cold.log` records the state/capture results. A fresh Inspector snapshot identifies all3project parameters by owner/control/field names, not coordinates or runtime pipeline addresses. Only this intended Python editor is running, with no outstanding build/test handles. Final whitespace check passed. Next: finish the explicit selected-cutout mask consumer, then continue the remaining A10 effects/history work; the whole041goal stays active.

## 2026-09-13 — RenderTexture resource workflow follow-up

- Rechecked the user's Project-created RenderTexture / Camera-target request against current source and the visible Lab. Project's Create menu, resource Inspector, persistent Camera asset slot, default D32 attachment and cooked asset description are present. The saved `OutlineCapture.rendertexture` remains in `Assets/Rendering`, assigned to `RenderTexture Camera` in scene12. `CommitGrade` in the actual Lab script already uses the supported fullscreen blit instead of copying into a camera target. Added the concrete write-back example to both languages of the RenderGraph guide; all20generated guides pass their consistency check.
- The existing cold-check helper exposed a real create/delete race: GUID pairing converted same-path CREATED then DELETED into MODIFIED, trying to import an absent file. Two new regression cases reproduced the defect before the fix (2failed/11passed in `041-rt-event-before.log`). Restricted GUID move pairing to different paths; existing same-path ordering now owns cancellation versus atomic replacement. Removed the redundant same-path branch; no new retries, tombstones, hashes or fallback were introduced. Added one ResourceChangeHandler integration test and a reverse-order atomic replacement test.
- Closed editor PID77944 normally before tests. Focused asset/Camera/graph/import suites:278passed (`041-rt-request-tests.log`). Full Python:6140passed/13skipped in236.94s (`041-rt-request-full.log`), exit0. Negative shader/clip-plane fixtures log expected errors, not suite failures. The pending project outline registry consumer separately passes20tests (`041-rt-request-registry.log`); selected-cutout visual acceptance remains pending. Native code did not change in this follow-up, so no new native-build/CTest claim.
- Opened normal visible editor PID65672. Early delivery-helper calls incorrectly used a naked GUID, then an incomplete asset-reference document; both were rejected by the existing shared asset schema. Corrected the helper to include `$type`, `asset_type`, `guid` and `path_hint`, without weakening the engine contract. A subsequent helper reopen was correctly refused while its own target-slot edits marked the scene dirty. After restoring the original GUID, saved through MCP and canceled the pending unsaved-changes dialog. These attempts are not counted as passed helper runs.
- Final `041-rt-request-visible-accepted.log` exits0: three new Project RenderTextures each render256x256 through the same real Camera; restoring the original resource produces800x450. Temporary assets are deleted through recoverable Project history, never by a broad filesystem command. Scene12 saves/reopens clean with the original target GUID. `041-rt-request-final-check.log` then independently rechecks the original Camera capture,1920x1080 Game capture, Project defaults and actual Camera Inspector selection;0warnings/0errors. Human-viewed engine-owned images include the full editor, the restored Camera output and the subsequent Play snapshot; no OS screenshots or pixel analysis.
- A real Play/Stop cycle completes with the13pass Outline Gallery graph and0warnings/0errors. Final state is scene12 clean Edit, Camera selected, only the intended visible editor process remains. Editor logs so far contain no ERROR/Traceback/DEVICE_LOST/VUID; no shutdown-validation claim while the editor remains open. Whitespace checks pass. No commit/push/release or new checklist completion: main100/256 + Compute36/76 remains136/332 (41.0%); this verifies and fixes an already-counted resource workflow, not the whole041 goal.

## 2026-09-13 — Selected cutout acceptance and explicit Camera history reset

- Added the separate `13_CutoutBodies` Lab scene through MCP, retaining scene12. Selected surfaces and owner masks explicitly share their UV coverage parameters and back-face culling. Two real material assets use different stripe densities; solid draws explicitly restore solid coverage when reusing parameter blocks. The initially reversed Quad was corrected in the project, not hidden by disabling culling. Real visible runs cover36projection/material/occlusion/depth combinations,8state combinations, GPU-deformed glass, movement/back-face rejection, unselect, owner splitting, deletion and Stop/reopen. Human-reviewed engine captures show matching holes and outlines. Parameter changes leave Python graph topology at3builds. Evidence: `041-selected-cutout-matrix.log`, `041-selected-cutout-flags.log`. A10.1's final combined visual item is now checked; this is not automatic translation of arbitrary material shaders into masks.
- Added public `Camera.reset_history()`, delegating to a runtime-only Camera revision consumed independently by each existing SceneRenderGraph View. Existing invalidation clears only that View's history, previous VP and jitter; small movements remain continuous unless explicitly reset. No global history manager, fallback, pose rewrite or serialization field. Native tests cover two Views sharing one Camera, an unrelated Camera, repeated requests, small motion, previous-VP resumption and clear/rebind. Added Python wrapper tests and completed the public TemporalAAEffect stub/re-export contract. The API/stub A10.2 item is checked.
- First-phase verification: Release build,157focused Python tests and83/83native tests passed. Initial full Python run had1failure/6142passed/13skipped: a real Windows rename test assumed that its injected retry was the only filesystem contention. The deterministic mock tests retain exact retry assertions; the real-filesystem test now records actual bounded backoff, including real sleeps and unchanged source/target paths. No production filesystem retry policy changed. Re-run:6143passed/13skipped in288.99s (`041-temporal-final-full.log`).
- The temporary visible history consumer runs in scene09 and cleans up its own objects/assets through MCP. A shader-name mismatch and a test depth resource named the reserved View alias `depth` caused earlier helper runs to fail; those runs are not acceptance. The independent test attachment is now `geometry_depth`, with explicit matching1x samples. Successful latch runs (`041-history-reset-reviewed.log`, `041-history-reset-repeat.log`, `041-history-reset-views.log`) show that small Camera motion retains the previous real geometry frame, explicit reset refreshes only Camera A, Camera B remains unchanged, and large cuts, projection, resize and reflection invalidate independently. Reflection small motion versus explicit reset and actual Scene/Game tabs were human-reviewed from engine captures. Play/Stop/repeat cleanly restores scene09.
- Actual built-in TAA testing then exposed native Vulkan errors despite Console0: VUID00337 (unsampled persistent depth),04626 (queue-family mismatch) and02207 (missing release),10each in the fully flushed `041-history-review-editor.log`. That process is **not** Vulkan-clean acceptance. RenderTexture depth sampling remains explicit: a graph now rejects undeclared persistent-depth sampling at its declaration boundary, and the TAA consumer requests `sampled_depth=True`. Both native MSAA-depth-resolve insertion sites now execute their compute commands on the Graphics queue: they are immediate raster dependencies, not asynchronous work. No independent GPU stack or queue recovery mechanism was added. A native test checks accepted/rejected1x/4x real RenderTexture depth declarations.
- Second-phase Release rebuild passed (`041-depth-resolve-build.log`). Khronos CTest:83/83, including15Vulkan tests,168.76s (`041-depth-resolve-ctest.log`). Generated English/Chinese guides pass all20checks. Full Python and fresh visible TAA/shutdown validation are still pending at this record point. The broader asynchronous RenderGraph ownership work is not claimed fixed by keeping this immediate resolve on Graphics. Current confirmed count: main102/256 + Compute36/76 =138/332 (41.6%),194remaining;12–18solo full-time engineering weeks remain a rough fixed-scope estimate. No whole041, cross-platform, release,300FPS, commit or push claim.
- Second-phase completion: full Python6143passed/13skipped in312.43s, exit0 (`041-depth-resolve-full.log`). Fresh visible editor PID19984 loaded the actual Khronos DLL. `041-depth-fixed-taa.log` and `041-depth-fixed-history.log` both exit0, including1x/4x depth sampling, target replacement/resize, exposure/TAA toggles and the full independent-history/reflection matrix. Human-reviewed new engine frames show retained/updated Camera A and unchanged Camera B. MCP normal close returned and the complete stdout/stderr through shutdown contain0VUID/validation/DEVICE_LOST/ERROR/traceback findings (`041-depth-fixed-editor.log/.err.log`). Scene09 restored clean Edit; no temporary probe asset or process remains. The explicit history-reset A10.2 item is now checked:103/256 +36/76 =139/332 (41.9%),193remaining.
- Final error-path review moved the new persistent-depth usage rejection from PassBuilder's throwing declaration path to the existing boolean graph-compilation boundary. `EnsureGraphBuilt()` runs in a native frame: invalid author settings should disable that graph's submission, not throw out of the editor loop. No catch-all or alternative renderer was added. The new compile-negative test first failed (`041-depth-boundary-before.log`, assert at RenderTextureVulkanTests.cpp:42): imported depth is represented as generic Texture2D, so the check must use its actual RHI pixel format. Corrected that predicate; final Release build passes (`041-depth-boundary-final-build.log`). Final native/focused/visible rechecks follow below; the intermediate failed test is not reported as a pass.
- Final boundary verification is complete:83/83native tests,15Vulkan,152.46s (`041-depth-boundary-final-ctest.log`);237Camera/RenderTexture/graph/history Python tests pass in6.61s (`041-depth-boundary-focused.log`). The earlier full6143/13run remains the most recent full-suite result; no intervening Python production change. Native LastTest.log has0VUID/Validation Error/DEVICE_LOST findings; expected negative descriptor tests may log their rejected inputs. Visible validation editor PID18924 repeats TAA and the complete history matrix (`041-depth-boundary-taa.log`, `041-depth-boundary-history.log`), then opens the saved scene12, captures its original Camera at800x450 and completes Play/Stop with0warnings/0errors. Normal MCP close exits fully; final stdout/stderr scan has0VUID/validation/DEVICE_LOST/ERROR/traceback findings (`041-depth-boundary-editor.log/.err.log`).
- Final handoff opens one ordinary visible editor PID60080 on scene12, clean Edit, with RenderTexture Camera selected and its saved `OutlineCapture.rendertexture` GUID unchanged. Human-reviewed engine-owned `review/041-rendertexture-final-editor.png` shows the actual Camera target asset slot and rendered Outline Gallery; no OS screenshot or pixel processing. Console0warnings/0errors. Build/test/helper processes are terminal and no second editor remains. The scene's CommitGrade uses fullscreen write-back, not camera-target Copy; no native contract weakening. The latest plan header/checkmarks now agree at139/332 (41.9%),193remaining. Full041 stays active, with12–18solo engineering weeks as an estimate rather than a delivery promise. No commit, push, remote release or new cross-platform acceptance in this slice.
# 2026-09-14 — UI dependency-cost audit and first simplification (ongoing)

- Previous goal turn was progress: real Vulkan submit grouping and MCP one-shot semantic capture were fixed and tested. User now accepts the outline main path at roughly600+FPS; the historical1000FPS gate is no longer blocking that item. No claim that full041 or UI performance is complete.
- Current runtime reread confirmed the existing native editor and clean Scene12. Baseline three240-frame windows were1.301/1.332/1.400ms (`041-outline-profile-cpu-followup-baseline.jsonl`). Scoped UI profiling confirmed per-element material reference resolution, native component binding, Canvas ancestor traversal and repeated private `__setattr__` work; cProfile overhead means its timings are not normal frame budgets (`041-ui-scope-profile.log`).
- Engine changes: `UIText` reads the old value only for measurement-affecting fields; `InxUIScreenComponent` private bookkeeping bypasses visual invalidation work while retaining superclass setters. Canvas ancestry is reused against its actual owner/scene/structure revision, including cached absence. Material revision checks inspect owner/version/RenderTexture revision without color or texture-path resolution; native owner identity is included so replacement with the same GUID/version invalidates. Full draw material state is deliberately not cached across asset-path changes.
- Regression cost contracts: a twelve-level Canvas lookup performs no ancestor enumeration for100 unchanged reads, and reparent/add/remove/owner changes invalidate it. Material checks cannot call color/texture-property reads; material version, RenderTexture revision and native-owner replacement change the result. Internal bookkeeping must not read a getter or dirty layout; visible text still invalidates measurement.
- Local A/B (`dev/test_041_ui_revision_bench.py`, final rerun `041-ui-dependency-bench-world-identity.log`) uses real native scenes,100/1000UIText and12 ancestors. It compares pre-change Python policy against current code with warmups and35 samples, then repeats current code. Screen100:1.7529→0.9687/0.9712ms; screen1000:19.8805→11.4234/11.3740ms; world100:1.4216→1.2965/1.2551ms; world1000:14.8185→13.2220/13.1129ms. These are dependency-only P50 CPU costs, **still unacceptable at scale**, not layout/draw/GPU/frame claims. First draft benchmark included extra per-call imports in its baseline and is superseded by corrected reruns.
- Real visible editor Scene12: six actual button interactions and restored fields, Stop/reopen, no Console errors (`041-ui-dependency-review.log`). First post-change frame windows1.651/1.358/1.872ms were variable and do not establish an end-to-end speedup (`041-outline-profile-ui-dependency-optimized.jsonl`). Do not replace user-visible FPS with the microbenchmark gain.
- Scene03: real MCP changed screen and world text/font size and moved/rotated world parent during Play; engine-owned Game and Scene captures visually show updated text and world geometry. No authoring files were saved. Console0warnings/0errors (`041-ui-world-review.log`). This scene's semantic UI-button query returned no targets, so this run does **not** count as a world-button click test; screenshots alone do not prove input. Last scene reload was scheduled; its completion must be checked before final handoff.
- Final focused UI regression exposed a real scene-identity bug: native Scene address/wrapper reuse with an equal structure counter returned a destroyed world's UI component. World-element snapshots, Canvas ancestry and UI content revision now include the existing native world_id. Added deterministic reused-wrapper regression; this corrects the owner key rather than swallowing dead-component exceptions. An earlier failure was a test-author mistake (Material.set_color needs four scalar channels, not a tuple); that test was corrected without changing the engine API.
- Final focused UI:207passed/1skipped (`041-ui-dependency-final-tests-world-identity.log`). Final full Python after all fixes:6217passed/11skipped in276.55s, exit0 (`041-ui-dependency-final-full-python.log`). Intentional malformed asset/shader tests print native errors after the summary; they are not suite failures. No C++ changes, rebuild, Git publication or release this turn. Next major UI work is authoritative dirty dependency propagation and per-element/region command reuse, not more per-frame Python signatures.
- Final fresh normal visible editor PID45988 cold-loads Scene03 in clean Edit with document_state=ready/loading=false, then enters Play through MCP. Engine-owned full Editor captures `review/041-ui-dependency-final-cold-editor.png` and `review/041-ui-dependency-final-world-pressed.png` were human-viewed: world panel/text/button and screen text are present; an actual injected pointer press at the observed world button changes its tint. Release is delivered and no held input remains. This supplements the earlier run's missing world semantic-button targets; it proves pointer visual-state response, not an authored on_click game action. Stop/clean-state handoff is checked below.
- Handoff: MCP Stop completes, Scene03 remains clean Edit/ready/loading=false, Console0warnings/0errors. The real editor is left open for the user; Hub40988 was not stopped. All test sessions have terminal exit0; no overlapping benchmark/test worker remains. Changed tracked Python files pass git diff --check (only existing Windows line-ending notices). Main path is stage-accepted; UI scaling/per-element rebuild and full041 remain open, with no new full-checklist completion claimed.
# 2026-09-14 — Transform vector authoring coercion

- 统一 Transform 的 `position`、`local_position`、Euler 和 `local_scale` 写入规则：三项 tuple/list 现在在 Python 边界转换为 `Vector3`，与 Rigidbody 的向量 API 一致；三项以外的序列仍交给原生层报告错误，不增加宽泛 fallback。
- 新增 `test_lib_native_lifetime_guard.py` 回归，覆盖合法三项序列和非法长度保留行为；`23 passed`。这修复了脚本中 `transform.position = (x, y, z)` 的 pybind 类型错误，但不增加041整项勾选。
- 随后 UI 命令、屏幕 UI 提交、自动布局、RenderTexture、纹理缓存和生命周期组合回归 **175 passed / 1 skipped**；当前可见编辑器日志未发现新的错误、Traceback 或 Vulkan 设备错误。
- Windows Release 原生全量回归 **90/90 passed**（102.38s）；Python 全量回归 **6354 passed / 12 skipped**（275.13s）。输出中的错误行均来自专门验证拒绝路径的负向 fixture，不是测试失败；本轮没有新增清单勾选。
- 复跑物理查询集成中的非凸 Mesh Raycast：真实静态三角网格命中并返回 `triangle_index`，**1 passed / 129 deselected**。该结果确认非凸查询已有功能边界；大规模性能和软体耦合仍未验收。
- 全量回归结束后的进程与日志清理检查完成：仅保留用户可见的 041Lab 编辑器（PID 40016），没有遗留 pytest/ctest/cmake/Player 进程；编辑器日志未出现 Traceback、ERROR、VUID 或 DEVICE_LOST。
- 物理集成回归继续通过：`python/test/test_integration_physics.py` **130 passed**，覆盖 Collider 几何查询、非凸 Mesh Raycast、批量 Raycast、过滤、固定步和接触事件边界。该结果是稳定性证据，不等同于软体—刚体 300 FPS+ 验收。
- Taichi 发布链审计：fork 工作树已删除约 **14,845** 行、移除 Field/SNode/ndarray/旧 AOT 与多后端实现；主工程 wheel verifier 明确禁止顶层 `taichi` 包及退役目录，并要求唯一 GPU 编译模块和 NOTICE/L​​ICENSE。当前仍有上游 CMake/示例源码仅作为编译输入保留，不能据此宣称全部裁剪完成。
- GPU JIT/compute 主链原生回归 **13/13 passed**（29.93s）：编译器合同、bit lowering、buffer 生命周期、GPU buffer/kernel/mesh/gizmo、物理交换及连续网格发布均通过；该结果确认裁剪后的编译链可工作，但不代表 Taichi 源码裁剪或 041 整体收口。
- Taichi fork 默认构建策略收紧：`TI_BUILD_EXAMPLES` 与 `TI_WITH_C_API` 默认关闭，保持显式兼容开关但不让公共构建产生无用示例/C API。使用 `conda activate infernux` 并指定环境 Python 后，独立 CMake 配置成功；首次未指定旧式 `PYTHON_EXECUTABLE` 的配置失败仅为工具参数诊断，不影响主工程。
- 最小非 Python 配置实际构建完成，`taichi_core.lib` 成功产出；没有生成 C API 或示例目标。构建输出暴露了仍可继续裁剪的 SPIR-V 工具辅助目标和少量旧 transform 警告，暂不将其误判为运行时失败。
- 正式 Windows Release `_Infernux` 增量构建成功，重新生成并同步 `_Infernux.cp313-win_amd64.pyd` 及引擎 DLL 依赖；未引入新的链接或打包错误。
- 重新生成的本机原生模块在 `conda activate infernux` 环境中直接导入成功：`INFERNUX_NATIVE_IMPORT_OK`，并可正常构造 `Vector3`；确认同步目录的 ABI/依赖可加载。
- wheel/Player 发布合同回归 **20 passed**：覆盖 manylinux 规则、wheel verifier、GPU JIT 载荷声明和 Player 构建入口。仓库没有名为 `wheel-smoke` 的 CMake target，因此未把一次 target 探测失败记录为制品失败。
- Taichi/Infernux Python 编译前端回归 **36 passed**：覆盖 kernel lowering、GPU 编译合同、模块隔离及裁剪后旧 Field/SNode/ndarray 导出不可见性；确认 `inx.compute` 仍能使用当前私有编译器入口。
### 2026-09-14 RenderGraph history / output-sample contract recheck

- Re-ran `python/test/test_render_view_samples.py`, `python/test/test_temporal_history_api.py`, and `python/test/test_render_effect_runtime.py` after the current RenderTexture and temporal-history changes: **72 passed**.
- The public graph contract currently covers per-view output samples, typed temporal history, jitter toggling, and motion/depth attachment declarations without a Python-side fallback. This confirms the previously reported motion/depth mismatch is not present in the current Python contract tests; native mixed-sample and full Player/multiplatform acceptance remain open and are intentionally not checked off in A09.
- Current plan count remains **140/333 (42.0%)**: `041-runner-long-foundation-plan.md` 104/257 and `041-buffer-jit-compute-contract.md` 36/76.
### 2026-09-14 Audio listener regression

- `python/test/test_audio_listener_selection.py`: **2 passed**. Listener selection remains deterministic and does not regress while the remaining A11 voice virtualization, streaming, device-switch, and long-duration pressure items stay explicitly open.
### 2026-09-14 RenderGraph and audio contract combined regression

- `test_rendergraph.py`, `test_render_view_samples.py`, `test_temporal_history_api.py`, and `test_audio_listener_selection.py`: **105 passed**.
- The camera-target copy rejection remains intentional; fullscreen writes and `present()` remain the supported camera-output paths. No fallback or validation weakening was added.
### 2026-09-14 Native audio regression

- Release CTest `infernux.audio_mixer`, `infernux.audio_clip`, and `infernux.audio_playback`: **3/3 passed**.
- This verifies the current native mixer/clip/playback path and pooled voice lifecycle. It does not yet close the unchecked A11 items for bounded virtual voices, streaming, device switching, or long-duration pressure testing.
### 2026-09-14 Full Windows Release native regression

- `ctest --test-dir out/build/windows-msvc-release -C Release --output-on-failure -j 8`: **90/90 passed** in 129.81s.
- This covers the current RenderTexture/Vulkan, UI, audio, compute/GPU mesh, physics exchange, asset, scene, and editor-tool native paths. It is a regression gate only; platform Player acceptance and the still-open plan contracts remain open.
### 2026-09-14 Full Python regression

- `python -m pytest python/test -q`: **6358 passed, 12 skipped** in 394.46s.
- The emitted asymmetric collision-matrix diagnostic is from the intentional invalid-input regression and remains expected; no unexpected test failure occurred.
### 2026-09-14 World UI contract regression

- World UI input batching, pass-layer filtering, hit snapshots, native dependencies, and runtime Screen UI submission: **70 passed**.
- Scene/Game public contracts remain stable. This does not close the outstanding visible-editor performance and full cross-platform acceptance items.
### 2026-09-14 Inspector integer binding regression

- `test_inspector_integer_batch.py`, `test_inspector_nested_semantics.py`, and `test_ui_value_controls.py`: **14 passed**.
- The test fixture intentionally omits a render pipeline, so the native `No render pipeline set` diagnostics are expected setup messages; the prior Python-to-native integer cast failure did not reproduce.
### 2026-09-14 Compute/JIT and mesh resource regression

- Compute, field schema, mesh resource/renderer NumPy bridge, and GPU mesh resource tests: **105 passed**.
- The reflected-shader `texSampler` diagnostic is the intentional invalid-parameter fixture; no unexpected failure occurred. A03/A04 performance and cross-platform acceptance remain open.
### 2026-09-14 Renderer parameter and shader contract regression

- Renderer selection, matrix/NumPy material parameters, shader public API/stage contract, and UI command packets: **131 passed, 1 skipped**.
- The shader compiler diagnostics are from the intentional `Public_Broken_040.frag` invalid-shader fixture; no unexpected failure occurred.
### 2026-09-14 Gizmo and Scene View interaction regression

- Gizmo, Player gizmo surface, Scene View undo routing, and editor selection interaction tests: **78 passed**.
- The public transform/selection transaction path remains stable. Full visible-editor axis orientation, Rect Tool visual polish, and cross-platform acceptance are still open and were not marked complete from these unit tests alone.
### 2026-09-14 Physics query and joint regression

- `test_integration_physics.py -k "raycast or closest or penetration or rigidbody or joint"`: **83 passed, 47 deselected**.
- Jolt query, Rigidbody and joint public paths remain stable. The A03 soft-body coupling, rigid-body reaction and 300 FPS scaling gate remain separate and unclaimed.
### 2026-09-14 Taichi integration layout audit

- `.gitmodules`, CMake, and the compute contract all point to the public `https://github.com/ChenlizheMe/taichi_for_infernux` submodule at `external/taichi_for_infernux`.
- No `external/plugins/infernux_taichi` entry remains. The current design treats the fork as an engine wheel/Player GPU compiler payload, not a separately installed project plugin; this matches the latest 041 decision and keeps platform plugins separate.
### 2026-09-14 Rect Tool native binding recheck

- `test_scene_view_undo_routing.py` and `test_gizmos.py`: **38 passed**.
- `get_editor_rect_frame`, override/clear lifecycle, and Scene View transaction routing are present in the native binding and Python surface; the earlier missing-method report is therefore attributable to an outdated runtime payload, not the current source contract.
### 2026-09-14 Asset identity and publication regression

- DataAsset, serialized-field identity, filesystem publication, script dependency graph, and project asset command service: **89 passed**.
- GUID/lifecycle and publication foundations remain stable for future composite import work; `.blend` import itself remains a separate unchecked A16 deliverable.
### 2026-09-14 Composite model import audit

- The native registry recognizes `.blend` and routes it through the Assimp model importer. The importer does recursively preserve node metadata in `InxMesh`, but no Blender executable or real `.blend` fixture is available in the current environment, so hierarchy/material/texture round-trip acceptance remains open and A16 is not checked.
### 2026-09-14 Blend format registry contract

- Native `infernux.asset_database_refresh`: **1/1 passed**; Python `test_core_asset_types.py`: **63 passed**.
- `.blend` is classified as a mesh and participates in the same importer/asset refresh registry as other interchange formats. Real Blender hierarchy/material/texture import remains separately unverified until a fixture is available.
### 2026-09-14 Official Windows wheel publication staging

- `cmake --build --preset windows-msvc-wheel --target package_python -j 8` completed successfully.
- Produced and internally verified `out/stage/windows-msvc-release/wheels/infernux-0.4.0-2-cp313-cp313-win_amd64.whl` (24,824,948 bytes), including native engine libraries and the integrated GPU JIT/Taichi payload.
### 2026-09-14 Wheel payload audit

- Inspected the freshly staged wheel directly: **959 files**, native `_Infernux` payload present, integrated `taichi_python` payload present, and `infernux-0.4.0.dist-info/METADATA` present.
- This confirms packaging completeness at the archive level; isolated installation was not performed because the environment's package-install command was blocked, so no stronger runtime claim is made.
### 2026-09-14 Worktree diff hygiene check

- `git diff --check` reported no whitespace errors or conflict markers; output only contains the repository's existing LF→CRLF normalization notices.
### 2026-09-14 Plan count reconciliation

- Recounted checklist markers directly from both plan files: runner **104/257**, compute **36/76**, combined **140/333 (42.0%)**.
- The older progress numbers embedded in historical entries are retained as history; the current top summary and execution-log entries use the reconciled count.
### 2026-09-14 Taichi license and source provenance gate

- `external/taichi_for_infernux` contains Apache `LICENSE`, a dedicated `NOTICE` retaining Taichi authorship and identifying Infernux modifications, plus README links to both projects.
- Repository-wide search found no active `external/plugins/infernux_taichi` reference; the only remaining occurrence is the completed migration checklist entry.
### 2026-09-14 Scene lifecycle regression

- Scene manager API/runtime loading, integration scene lifecycle, Play Mode, and runtime dispatch epoch tests: **267 passed**.
- Native diagnostics are intentional invalid-document and identity-collision fixtures; no unexpected failure occurred.
### 2026-09-14 Plugin package and platform cook regression

- InxPackage native integration, plugin installation, platform content cook, and game-builder asset closure: **168 passed, 5 skipped**.
- Plugin assets and platform cook remain compatible with the integrated Taichi layout; cross-platform Player release acceptance is still open.
### 2026-09-14 Android Player contract regression

- Android Player smoke/protocol, platform content Cook, and Player format marker tests: **52 passed**.
- Android package/build protocol remains stable at the script layer; no claim is made for a real-device APK run in this environment.
### 2026-09-14 Web platform contract regression

- Web toolchain setup, shader pipeline, and exporter plugin tests: **31 passed**.
- Web build-side contracts remain stable; WebGPU browser rendering and full Player acceptance remain separate open items.
### 2026-09-14 MCP control-plane regression

- MCP server, supervisor, and automation operation tests: **68 passed**.
- The control plane used for desktop 041Lab maintenance remains stable; this does not replace visible MCP editor acceptance.
### 2026-09-14 Linux/manylinux packaging configuration audit

- The Linux wheel preset is `linux-clang-wheel`; packaging routes through `cmake/repair_linux_wheel.py` and enforces the single `manylinux_2_35_x86_64` tag.
- The wheel verifier checks the native binding, engine libraries, compiler payload, metadata, and rejects a top-level standalone Taichi package. No Windows-only artifact is selected by the Linux preset.
### 2026-09-14 Hub and plugin distribution regression

- Hub version/project model, plugin update/reload progress, release assets/versions, registry publication, Python storage, panel content, install progress, catalog refresh, and InxPackage plugin tests: **255 passed, 6 skipped**.
- Hub/plugin distribution contracts remain stable; real remote CDN download and cross-platform GUI acceptance remain separate open items.
### 2026-09-14 Hub publication policy regression

- Hub publication policy, plugin release assets/versions, and Hub version manager tests: **54 passed, 1 skipped**.
- manylinux wheel naming and Cloudflare/GitHub source selection rules remain consistent; remote endpoint availability is still an external acceptance step.
### 2026-09-14 High-DPI and editor viewport regression

- Game View/Scene View Python tests: **35 passed**.
- Native editor window bounds, sizing, display scale, and Windows DPI policy: **4/4 passed**.
- The coordinate-scaling contract is stable in tests; real 150% DPI visual acceptance remains a visible-editor check, not inferred from these unit tests.
### 2026-09-14 — 交付前子仓库审计

- 重新核对 041 两份主计划：runner-long 104/257、buffer/JIT/compute 36/76，合计 140/333（42.0%）。
- 官方 Android、Windows、Linux 子仓库当前干净；Taichi、Web、MCP 子仓库存在未提交修改，保留为当前工作区变更，不擅自覆盖。
- Taichi 集成路径仍为 `external/taichi_for_infernux`，未发现旧 `external/plugins/infernux_taichi` 活跃引用。
### 2026-09-14 — 渲染/UI/场景回归补测

- Windows Release 配置下补测 `render_texture`、`additive_scene_world`、`ui_transform_dependencies`、`camera_culling_isolation`、`screen_ui_vulkan`：5/5 通过。
- 首次未指定多配置生成器的 `ctest` 配置导致测试未运行，已按实际 Release 配置重跑；不计作失败。
### 2026-09-14 — Windows Release 全量原生回归

- `ctest --test-dir out/build/windows-msvc-release -C Release --output-on-failure`：**90/90 通过**，总耗时 125.40 秒。
- 覆盖资源、渲染、RenderTexture、Screen UI、场景生命周期、GPU upload/compute、GPU mesh publication、物理交换和 residency soak；结果不改变尚未完成的跨平台/真实编辑器验收项。
### 2026-09-14 — World UI / RenderTexture / DPI Python 回归

- `test_world_ui_pass_layers.py`, `test_world_ui_input_batch.py`, `test_ui_render_texture.py`, `test_ui_dpi_helpers.py`：**44 passed**。
- 该结果覆盖逻辑层、输入批处理、UI RenderTexture 契约和 DPI 辅助函数；真实编辑器视觉与跨平台包体验收仍按计划保留。
### 2026-09-14 — GPU 计算/物理交换/驻留回归

- Release Vulkan 定向回归 `compute_buffer_gpu`、`compute_kernel_gpu`、`compute_mesh_gpu`、`compute_physics_exchange_gpu`、Mesh publication/continuous、retirement、resident resource：**8/8 通过**，45.70 秒。
- 该证据确认共享 GPU 资源路径当前无回归；真实 041Lab 刚体—软体双向 300 FPS+仍未冒充完成。
### 2026-09-14 — Physics integration / non-convex Raycast 回归

- `python/test/test_integration_physics.py`：**130 passed**。
- 覆盖静态非凸 Mesh 三角形身份、批量 Raycast 输出复用、Jolt 候选查询与物理状态集成；A17 的性能预算和稀疏/密集增长矩阵仍未完成，因此不提前勾选整项。
### 2026-09-14 — Web toolchain / Player bootstrap 回归

- Web toolchain setup、shader pipeline、exporter plugin 与 Player bootstrap：**35 passed**。
- 该回归确认本机 Web 构建契约未回退；真实 Web Player 画面与 041Lab 端到端视觉验收仍需单独完成。
### 2026-09-14 — Platform export/cook/bootstrap contract regression

- Platform support/publication, Player workflow/bootstrap, platform content cook and Web exporter tests：**71 passed, 2 skipped**。
- Skips are environment-gated platform cases; no claim is made for remote Linux runtime without SSH authentication.
### 2026-09-14 — UI/Gizmo 全量 Python 回归

- UI、World UI、Gizmo 与 Player Gizmo surface：**393 passed, 2 skipped**。
- 本轮使用显式文件列表执行，避免 PowerShell 通配符未展开造成的假性“无测试”。
### 2026-09-14 — 041Lab 可见编辑器启动验收

- 通过 `dev/launch-041lab.ps1` 在当前 `infernux` 环境启动桌面 041Lab，收到 `ENGINE_LOADED`，RenderGraph `Outline Gallery` 成功建立（13 passes），Vulkan surface 从 1600×900 调整到 1920×1009。
- 项目 MCP HTTP 端点 `127.0.0.1:9741/mcp` 已监听；直接 GET 返回 406（需要 MCP 请求头/JSON-RPC），不是服务未启动。当前会话未提供 MCP 调用工具，因此不伪造交互结果。
### 2026-09-14 — 041Lab MCP 实际握手与 Ping

- 对可见运行中的 041Lab MCP 端点完成标准 JSON-RPC `initialize`、`tools/list` 与 `mcp_ping`。
- 服务返回 protocol `2025-06-18`、server `Infernux Editor 1.29.1`，`active=true`、revision 130、98 个 operation、14 个 gateway tool，`mcp_ping` 为 `ok=true`。
### 2026-09-14 — MCP 场景状态与层级真实查询

- 通过 MCP `infernux.project.info` 查询到 041Lab 当前工程，活动场景 `12_OutlineOwners`，文档 ready、已加载且无错误，Play 状态为 playing。
- 通过 `infernux.scene.hierarchy.get` 读取真实层级，确认 Main Camera、RenderStack、OutlineGallery、多个 MeshRenderer/Rigidbody 与 UIText 引用均由场景文档提供，而非测试脚本临时伪造。
### 2026-09-14 — MCP loaded-scene world query

- 通过 MCP `infernux.scene.loaded.get` 读取真实编辑器 World：active world id 1，当前 resident scene `12_OutlineOwners`，active=true，root_count=13。
- 这确认场景管理查询链路能返回 World/Scene 关系，不仅是单场景文档状态。
### 2026-09-14 — MCP runtime/performance/console 只读验收

- MCP 读取 `runtime.status`、`runtime.performance.get`、`console.read` 成功；退出 Play 后 runtime state 为 `edit`，transition 29.72 ms，当前性能采样为 0（没有把非 Play 数据冒充 FPS）。
- Console 仅返回 Outline Gallery RenderGraph ready 信息，无新的 error/warning；GPU mesh upload submitted/completed 均为 11。
### 2026-09-14 — 综合主线回归窗口

- 一次性执行场景、物理、资源数据库/索引、UI RenderTexture/World UI、平台支持、Player workflow 与 MCP server：**405 passed, 2 skipped**，28.89 秒。
- 输出中的错误均来自测试专门构造的非法文档/ID 诊断，未导致测试失败；未将该批回归误报为完整 041 端到端验收。
### 2026-09-14 — MCP Play/运行时批量状态

- MCP `infernux.runtime.play` accepted and completed transition in 34.06 ms; after a 3 s observation the editor had returned to `edit` automatically (total play time 13.89 s), with no transition pending.
- `runtime.performance.get` therefore returned zero samples; this is recorded as missing live-frame evidence, not an FPS result. Console continued to report valid 13-pass Outline Gallery RenderGraph and mesh uploads 13/13 completed.
### 2026-09-14 — MCP Play 稳定运行复测

- 排除误操作后重新调用 `infernux.runtime.play`，进入 Play transition 33.96 ms；持续观察约 5 秒后仍为 `playing=true`，累计 Play 时间 37.24 s。
- Resident mesh vertex buffer=1，mesh uploads 14/14 完成；Console errors/warnings=0，Outline Gallery 13-pass graph 持续有效。
- 当前性能采样器未启用（sample_count=0），因此不报告 FPS；这与 Play 生命周期稳定性分开记录。
### 2026-09-14 — MCP 性能采样入口审计

- `operation_schema_get(infernux.runtime.performance.get)` 确认该接口是只读、无参数查询；当前 MCP operation catalog 没有独立的性能采样 enable 命令。
- 因此真实 Play 可验证生命周期、资源驻留和 Console，但 FPS 统计必须由引擎启动配置/既有采样器开启，不能在查询端伪造或重置窗口。
# 2026-09-14 041 计划证据对账（本轮）

- 对照当前工作区计划与执行证据重新核对：主计划 `dev/041-runner-long-foundation-plan.md` 为 **104/257（40.5%）**，Compute 合同 `dev/041-buffer-jit-compute-contract.md` 为 **36/76（47.4%）**，合计 **140/333（42.0%）**。
- 本轮没有把“测试通过”误升格为聚合验收：真实 041Lab MCP 启动、层级/场景查询、Play/Stop 稳定运行、Windows Release CTest 90/90、Python 物理 130 项、UI/World UI/Gizmo 393 项、综合回归 405 项均已记录；但 A03.4 双向软硬体 300 FPS+、A05 B05/B06 Taichi 纯编译与发行载荷、A06 UI 性能/材质跨平台矩阵、A10 离屏 Camera/RenderTexture 完整契约、A17 非凸批量 Raycast 性能及 Runner Long 仍有明确未完成条目。
- 因此本轮按原子条目证据审计后不新增勾选；此前已由执行记录证明完成的条目已经在清单中勾选，剩余百分比不是测试记录滞后，而是尚未达到计划所写的范围或真实用户验收门槛。下一步继续以引擎公共路径和真实项目验收为主，不以扩大回归数量替代功能交付。

- 当前源码回归复核：Inspector、UI、RenderStack、物理查询和 MCP 公共链路 **385 passed / 1 skipped**（28.55 秒）。未发现回归；该结果作为后续性能和跨平台工作基线，不把测试数量等同于 041 完成。
- 重新启动桌面 `Infernux041Lab` 可见编辑器后，MCP 初始化成功（协议 `2025-06-18`，服务 `Infernux Editor 1.29.1`）；`mcp_ping` 返回 active=true、revision=130、98 个操作（67 个引擎自有操作、14 个网关操作），说明真实编辑器服务当前可用。
- 同一 MCP 会话查询 `infernux.project.info`：项目根目录为桌面 `Infernux041Lab`，活动场景 `12_OutlineOwners` 已加载、无 dirty、无错误，当前处于 Edit 状态。该结果确认后续场景验收可直接在可见编辑器上继续，不依赖孤立 headless fixture。
- 整体 Python 回归复核完成：`python/test` **6358 passed / 12 skipped**（350.35 秒）。末尾出现的错误日志均来自专门验证冲突、非法 shader 和非法层配置的负向 fixture，pytest 总结果为成功；未发现新的正向路径回归。
- Hub/发行链路全量复核完成：`packaging/tests` **375 passed / 3 skipped**（24.58 秒），覆盖更新发现、Cloudflare/GitHub 后备、安装状态、启动检查和发布策略；未发现回归。

### 2026-09-14 — Windows PC Player 源码构建验收

- 使用桌面 `Infernux041Lab`、源码引擎和 Windows Release 平台插件执行真实构建：首次在烹饪 DataAsset 时发现项目脚本类型未进入序列化注册表，报 `unknown SerializableObject type_id 'runner_long.config.game'`。
- 修复 `scripts/acceptance/build_player.py`：源码构建在烹饪项目前先同步项目资源、加载 Assets/Packages 脚本并依据 `.meta` GUID 发布组件/序列化类型；不依赖已运行的编辑器实例。
- 修复后重新构建成功：`D:\Users\Chenlizhe\Desktop\Infernux041Lab-PC-acceptance`，Windows x64 / Vulkan / Release / 压缩资源 / 无调试符号，耗时 **24.82 s**，`diagnostics=[]`；产物包含 `Infernux041Lab.exe`、`Infernux041Lab_Data\Content.inxpkg` 和 `AssetCatalog.inxcat`，未产生明文项目目录。
- 对生成的 Player 进行独立启动验证：进程启动后保持运行超过 8 秒，再正常终止，未在启动阶段退出。Release 包未开启 token-authenticated 调试控制服务，因此未将 debug-only smoke 脚本失败误判为 Release Player 失败。
-
### 2026-09-15 — PC 主链批量复核

- Windows Release CTest **90/90** 通过（134.74 s），覆盖场景驻留、RenderTexture、UI Transform、DPI、GPU buffer/mesh、compute physics exchange、材质、音频与运行时资源退休。
- Python 计算/JIT/MCP 组合回归 **148 passed**（16.73 s）。
- 真实 MCP 果冻复核的 `core` 路径可正常捕获；`input` 路径已不再触发运行期事务 safe-point 错误，但 Space 触发后的高度没有变化，故保留为未解决的输入/动作验收缺口，不将其标记为通过。
- 计划原子条目重新统计为主计划 **104/257**、Compute 合同 **37/76**，合计 **141/333（42.3%）**。本轮只更新有直接证据的统计，A03.4 双向耦合、Taichi 发行收口、完整 UI/RenderTexture、非凸 Raycast、`.blend` 导入和跨平台验收仍未完成。

### 2026-09-15 — Windows PC Player 独立启动复核

- 对 `Infernux041Lab-PC-acceptance-latest\Infernux041Lab.exe` 做独立启动验收；进程启动后持续运行超过 8 秒，随后通过窗口关闭并确认无残留 Player 进程。
- 本次只确认 Player 启动/存活与退出生命周期，不把它等同于完整场景交互或 FPS 验收；后两项仍需在可见编辑器和跨平台构建中分别取证。

### 2026-09-15 — PC Player 软体运行链路复核

- 导出包的 `Player.inxmanifest` 声明 `jit=true`，`Runtime/Infernux/_compiler/taichi` 内含 39 个编译器模块和 `taichi_python.cp313-win_amd64.pyd`；`Content.inxpkg` 也包含 `GPUJelly.pyc`、`JellyKernels.pyc`，因此“完全没有携带 Taichi”已排除。
- 但当前独立 Player 验收只证明进程存活，尚未取得可见运行帧或软体顶点/高度变化证据；“PC Player 软体不动”暂列为未解决的 Player 生命周期/计算执行阻塞项，不能用包体存在文件替代运行通过。下一步应在 Player 日志可见化后确认 `GPUJelly.fixed_update → compute.launch` 是否执行，再修公共 Player 调度或 GPU 编译路径。

### 2026-09-15 — Player splashless activation 修复复核

- Player bootstrap 在无 splash 项目中于首次 GUI present 前直接激活初始场景，避免场景进入 Play 依赖首帧相机纹理。
- Windows Development Player 重新构建通过（约 26.9 s，`diagnostics=[]`）；Player 服务/格式定向测试仍为 **18 passed**。软体 GPU kernel 的运行期位移观测仍待通过 Player 诊断通道确认。
