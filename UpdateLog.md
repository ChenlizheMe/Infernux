# Infernux v0.4.3-v5 · Hotfix 5: Game Input and Preview Control

This hotfix repairs game UI input in the editor and restores control of the Game preview resolution. The base engine version remains 0.4.3; wheels use build 5, with the matching Hub bundle at 0.4.3-6.

[中文更新日志](UpdateLog-zh.md)

**Baseline for comparison:** [`v0.4.3-v4...v0.4.3-v5`](https://github.com/ChenlizheMe/Infernux/compare/v0.4.3-v4...v0.4.3-v5).

### Input and Editor Performance

- Process game pointer transitions independently of editor panel redraws, so short clicks reach game UI without rebuilding every editor panel for each click.
- Request a GUI frame for each physical input batch even while continuous MCP semantic capture holds the frame scheduler active. Input edges no longer disappear before the next panel update.
- Keep raw editor input separate from game-owned mouse buttons, keys, wheel and text. Resolve actual window occlusion, popups and text editing before assigning new game input; preserve releases for presses already owned by the game.
- Keep retained UI geometry when an ordinary numeric field is assigned an equal integer or float value, avoiding needless invalidation in animated controls.
- Expose read-only UI input routing and existing frame timing details through the editor automation host for diagnosis without generating events or submitting UI work.

### Preview, Fonts and Plugins

- Restore the Game resolution selector when a project has a fixed render output. Users choose fixed or custom preview resolutions independently of project output and camera settings.
- Add Auto resolution, which follows the Game panel's available area and framebuffer scale. Fixed resolution previews retain their selected dimensions across Play and Stop.
- Map pointer coordinates independently on both axes, keeping game UI hit testing aligned with the preview at fractional scales and different aspect ratios.
- Preserve font asset handles when copying serialized UI styling, retaining their native asset identity and Player dependency tracking.
- Keep the renderer running when a UI material references a missing shader. Skip only the affected draw, report it once until the dependency changes, and resume after a successful import.
- Drain pending asset imports while their plugin modules are still loaded, before unloading plugins and releasing native resources during shutdown.
- Retire native Player scenes before releasing the asset catalog, so component disable and destroy callbacks can still resolve their font and other asset references during shutdown.
- Preserve signal handlers owned by embedded native hosts, avoiding an Android exit exception that interrupted normal engine cleanup.
- Ignore access-time and attribute-only script notifications when the recorded modification time and size are unchanged, preventing needless plugin restarts during Windows Player builds.
- Reload changed plugin helpers even when equal-size edits share an integer-second timestamp, while retaining unchanged import caches.
- Report unmatched GUI tree pops before changing layout or ID state, preventing malformed custom panels from leaving errors in subsequent frames.
- Register the independently released Text Animator 0.1.2 and MCP 0.1.21 packages. Text Animator scales its writable character-buffer views in place, including getter-only native views.
- Rebuild the matching Windows, Linux, Android and Web Player packages for the updated native input interfaces.

### Hub

- Preserve complete editor startup errors and include the process exit code. Wait for the stderr reader to finish without blocking the Hub, so the final exception is not lost.
- Give the Show details button the standard visible button border.

---

# Infernux v0.4.3-v4 · Hotfix 4: Character Geometry and Color Composition

This hotfix completes the native text interfaces used by the independent animation plugins. The base engine version remains 0.4.3; wheels use build 4, with the matching Hub bundle at 0.4.3-5.

[中文更新日志](UpdateLog-zh.md)

**Baseline for comparison:** [`v0.4.3-v3...v0.4.3-v4`](https://github.com/ChenlizheMe/Infernux/compare/v0.4.3-v3...v0.4.3-v4).

### Text and Rendering

- Add per-grapheme three-dimensional affine transforms and four-corner depth offsets. Screen text projects the transformed geometry; world text preserves depth through its retained draw packets, clipping and GPU vertex attributes.
- Compose the Manual, Effects and Appearance layers in order. Corner edits remain usable at zero scale and when another layer rotates or scales the character.
- Separate absolute character RGB and alpha overrides from ordinary layer tint. Alpha-only animation preserves rich-text colors, including multiple colored glyphs belonging to one grapheme. Reset restores the authored colors.
- Apply parent and element opacity independently from character alpha, and retain material tint in both runtime rendering and UI Editor previews. Characters can become visible after an authored zero-alpha base color.
- Treat text inside rich-text `<noparse>` spans as literal content while retaining original source offsets and the surrounding text styles.
- Rebuild the matching Windows, Linux, Android and Web Player packages for these native interfaces.

### Math and Reflection

- Return a valid identity rotation for zero forward vectors and a normalized forward rotation for collinear or zero up vectors in `Quaternion.look_rotation`.
- Make quaternion angle and rotation-towards calculations agree for opposite-sign representations of the same rotation; correct the quaternion constructor default in the Python type stub.
- Read string-valued Enum members through their declared values in semantic field descriptions, so plugin controls do not fail by coercing them to integers.
- Keep inherited component type identities separate from asset GUIDs in the Player dependency catalog, so valid plugin inheritance does not block builds while missing script dependencies remain errors.

Tween and Text Animator remain independently maintained official packages. Their own API and authoring changes are released through the plugin catalog.

---

# Infernux v0.4.3-v3 · Hotfix 3: Reliable Plugin Reload and Text Authoring

This hotfix fixes issues found while using independently installed animation plugins in real Windows and Linux projects. The base engine version remains 0.4.3; wheels use build 3. Hub 0.4.3-4 accompanies this distribution without changing its project-update workflow.

[中文更新日志](UpdateLog-zh.md)

**Baseline for comparison:** [`v0.4.3-v2...v0.4.3-v3`](https://github.com/ChenlizheMe/Infernux/compare/v0.4.3-v2...v0.4.3-v3).

### Plugin Lifecycle and Editor

- Load preload-owned Python libraries before publishing components during installation and re-enabling. Keep library modules out of component transactions and avoid repeating preload notifications during explicit package publication.
- Refresh project scripts that import a reloaded plugin library, using the in-memory dependency graph. Their dependent scripts join the same transaction, so animations do not silently retain an unloaded manager.
- Compile preload-owned libraries into Players without misregistering their internal helper classes as authored asset components. Player acceptance uses the same import boundary as the editor.
- Recognize cooked preload libraries as project dependencies and retain each plugin's legal notices even when their license text matches.
- Respect the existing edit-mode lifecycle declaration: gameplay Awake/OnEnable waits for Play, including prepared Player scene components. RenderStack explicitly runs in edit mode to retain editor rendering.
- Preserve live component base-class identities after disabling and re-enabling a package. Recognize package Runtime classes as project scripts, bind provisional imported classes to their asset identity transactionally, and initialize package namespaces for relative component imports.
- Give Inspector field groups stable IDs independent of their visible titles. Repeated groups no longer produce duplicate-item errors or share fold state. Apply field-label translation consistently to lists, asset references and nested objects.
- Accept ordinary Python name assignments that shadow imports, including Enum member declarations, without treating them as writes to an imported object. Mutating imported objects remains a rejected candidate side effect.

### Text and UI

- Expose renderer-free Unicode grapheme segmentation through `TextLayout.split_graphemes`, preserving original text, combining marks, emoji sequences and line breaks.
- Append the bundled font after authored primary and fallback fonts so mixed-language text retains its selected face and can display remaining Chinese glyphs. Missing authored font assets remain explicit errors.
- Route left, right and middle mouse buttons through independent UI press and drag state in Editor, desktop Player and Web. Buttons and sliders continue to activate with the left button; custom pointer handlers receive the actual button.
- Use adaptive arc subdivision for large rounded rectangles, improving large-radius UI edges without oversampling small corners.

Platform Player packages are rebuilt for the matching engine contract: Windows 0.2.13, Linux 0.2.12, Android 0.2.14 and Web 0.2.20. Tween and Text Animator remain independently maintained plugins; this engine hotfix does not bundle an unfinished game or claim a completed SNKRX replica.

---

# Infernux v0.4.3-v2 · Hotfix 2: Native Text and Plugin Foundations

This 0.4.3 hotfix adds reusable native text-animation and plugin-update interfaces, and repairs clocks and rendering state found during cross-platform validation. The base engine version remains 0.4.3; the engine uses build 2, with the separate Hub update at 0.4.3-3.

[中文更新日志](UpdateLog-zh.md)

**Baseline for comparison:** [`v0.4.3...v0.4.3-v2`](https://github.com/ChenlizheMe/Infernux/compare/v0.4.3...v0.4.3-v2).

### Hub 0.4.3-3

- Update installed engines to a newer wheel hotfix directly from their cards, with a background catalog refresh. Older cached wheels and project version pins remain unchanged by the download.
- Separate explicit project hotfix updates from engine version upgrades. Choose an installed release; Hub prepares its exact engine and Python runtime on the next launch. Open projects cannot be switched, and the version change preserves project assets.

### Runtime and Rendering

- Preserve the real unscaled frame duration and clamp scaled simulation consistently. Fix packaged Players treating the native pause method as a Boolean, which stopped Python clocks and time-driven effects.
- Provide removable Normal, Late, Fixed and Editor update registrations, with plugin ownership and automatic cleanup.
- Expose read-only grapheme, glyph, word and line layouts with original UTF-8 source ranges. Compose canonical combining accents during layout and correct whitespace wrapping.
- Add reusable character buffers for offsets, rotation, scale, skew, four-corner deformation and color, plus visible-character control and separate manual, effects and appearance layers.
- Support basic rich-text color, size and line-break spans; preserve clipping, group opacity, font atlases and material boundaries in screen and world UI.
- Invalidate camera state when replacing render graphs. Publish Web font atlases before recording UI geometry, fixing missing world text when animated screen text loads additional fonts.

### Editor and Distribution

- Add reversible editor preview scopes. Saving, entering Play, closing a panel, deleting a target, undoing changes or unloading a plugin restores captured values; saving a clean document also ends its preview.
- Prepare the official catalog for independently maintained animation plugins without adding their source repositories to the engine's submodules or build dependencies. Publish package files before the complete catalog.
- Retain the engine and Unicode notices in desktop, Android and Web Player distributions.
- Preserve each published hotfix's actual release tag in website history, keeping the original base release record.

---

# Infernux v0.4.3 · Faster Frames, Complete Players

0.4.3 brings the editor, animation authoring and cross-platform Players through a sustained real-game validation cycle. The focus is a correct, efficient main path: less repeated frame work, complete startup preparation, consistent asset identities and usable editing tools.

[中文更新日志](UpdateLog-zh.md)

**Baseline for comparison:** [`v0.4.2-v2...v0.4.3`](https://github.com/ChenlizheMe/Infernux/compare/v0.4.2-v2...v0.4.3). This includes the independent 0.4.2 Hub hotfixes and the changes merged in PRs #89 and #90.

### Editor Performance and Startup

- Reuse published component bindings, scene cameras, lights, shader classifications and asset catalogs instead of repeating discovery and filesystem work every frame.
- Reduce Python lifecycle dispatch, editor status synchronization, change-journal traversal, native pose submissions and allocation overhead. Leave automatic garbage collection in control of ordinary frame-loop collection.
- Complete declared startup work, material preparation and GPU uploads before revealing gameplay. Loading and GPU preparation no longer advance gameplay clocks or consume the first simulation delta.
- Show independent Play progress only in the editor, using the same presentation as build progress; packaged Players do not open the editor progress window.
- Keep the editor running when an asset cannot be loaded, including damaged initial scenes and files containing bytes from the wrong format. Failed assets use a generic file icon and a Delete/Cancel dialog; cancellation preserves the original file without attempting repairs or restoring cached content over it.
- Keep failed files visible with their original GUID during live imports. Wait for an actual source change instead of retrying identical invalid content, and close an obsolete error dialog after a successful reimport.
- Remove the experimental Cython packaging path. Engine wheels and Player builds use the supported Python/native architecture without an additional Cython build dependency.

### Animation and Authoring

- Repair 2D Animation Clip atlas selection, sprite-frame metadata, save operations and previews, and compact the editor layout. Sprite frame selection uses an appropriate numeric control.
- Route Ctrl+S to the visible editable document when another panel owns keyboard focus, preserving animation, scene and UI document ownership.
- Add multi-clip animation groups and improve state-machine editing, list controls, localized actions and direct directional transition links.
- Correct animation transitions, pose-stack bone masks, repeated pose publication and frame-zero events. Sprite, skeletal and grouped playback dispatch the first sample consistently.
- Preserve selections and scene identities through editing, Play/Stop restoration and runtime scene changes.

### Input, Console and UI

- Repair Game view mouse capture after Escape and Alt+Tab. Process game input before editor Escape handling, with explicit input-edge consumption for in-game menus.
- Make Console single-click selection target the originating object and double-click navigation open the relevant source location. Present project exceptions consistently and navigate to the deepest project call.
- Retain unchanged Canvas geometry, layout contexts and screen-UI command packets, with invalidation scoped to actual hierarchy, visibility and texture dependencies.
- Batch compatible clipped text and atlas-backed images while preserving material, texture and draw-order boundaries. Correct text publication, font invalidation and authored atlas sizing across platforms.
- Reduce editor panel bookkeeping and avoid unnecessary HUD layout and semantic capture work during normal gameplay.

### Rendering and Cross-Platform Players

- Repair material publication, complete frame presentation, shadow resource reuse and texture decoding through imported asset artifacts.
- Extend the Web Player with complete split scene bindings, layered cameras, per-camera particles, textured flipbooks and soft-particle scene-depth intersections. Preserve texture and uniform mappings during shader sampler splitting.
- Declare only the particle descriptor sets actually used by each pass, avoiding Android device descriptor limits. Repair mobile text, input and rendering publication paths exercised by real-device testing.
- Resolve cooked asset identities from the unpacked content catalog and install catalog snapshots without per-file probing. Preserve editor import identities in headless builds and keep editor-only imports out of Players.
- Keep incremental Android APK exports compact by replacing obsolete content entries instead of retaining old payloads.

### Plugins, Hub and Build Tooling

- Ship refreshed official MCP, Windows, Linux, Android and Web packages. Each platform package includes its precompiled Player payload; plugin installation does not require a user-managed compiler or toolchain download. Android SDK/device setup remains a separate build prerequisite.
- Separate plugin installation from target readiness, share the Hub installation root with independently launched plugin processes, and serialize MCP supervisor Player lifecycle operations.
- Include the Hub startup, single-instance, Python runtime-marker and Windows installation-access repairs released after 0.4.2. Repair inherited runtime permissions from older installations and avoid read-only installation files and invalid pip paths.
- Support Visual Studio 2022 and Visual Studio 2026 with an appropriate CMake version, and keep Hub packaging on the exact Visual Studio instance selected by CMake.
- Manage native dependencies through maintained, pinned submodules rather than cloning them during CMake configuration.
- Adopt the hollow striped monolith logo across engine, Hub, packaging and website assets.

### Delivery and Adoption

- Consolidate redundant tests around current authoring, runtime and delivery workflows, and remove historical reader fallbacks and obsolete helpers.
- Gate publication on matching Windows/Linux distributions and platform Player acceptance, then publish GitHub, PyPI and Cloudflare artifacts and derive website/download metadata from those actual files.
- Projects remain pinned to an exact engine release; teams should install 0.4.3 and use its matching precompiled platform packages together. Hub does not silently upgrade an existing project's engine pin.
- Infernux remains alpha software. Real-game validation improves the tested paths; it does not constitute a guarantee that every game, GPU or project format is supported.

---

# Infernux v0.4.2 · Reliable Worlds and Collaborative Authoring

0.4.2 strengthens the working paths introduced in 0.4.1: portable projects, stable asset identities, transactional editing, script reload, physics, rendering and delivery. This release brings the repair and acceptance work into the published engine, refreshes the website, and ships matching platform packages.

[中文更新日志](UpdateLog-zh.md)

**Baseline for comparison:** [`v0.4.1...v0.4.2-v2`](https://github.com/ChenlizheMe/Infernux/compare/v0.4.1...v0.4.2-v2). This includes the 0.4.1 v2/v3 hotfixes; the changes after the last published revision are also available in [`v0.4.1-v3...v0.4.2-v2`](https://github.com/ChenlizheMe/Infernux/compare/v0.4.1-v3...v0.4.2-v2).

### Publication revision 2

- Use the system Vulkan loader in Linux wheels so SDL and the native engine share one instance registry. Build 1 wheel files already uploaded to PyPI remain immutable; the corrected files use wheel build 2, Hub 0.4.2-2 and engine release 0.4.2-v2.
- Wait for newly uploaded wheels to become visible in PyPI's cached metadata before publishing the other channels. Missing metadata has a bounded wait; mismatched files and unrelated errors fail immediately.

### Projects, Assets and Collaboration

- **Pin the exact engine release.** Shared projects record the engine revision and CPython ABI. Hub disables projects whose matching engine is not installed and rebuilds their local runtime from the installed release during startup.
- **Keep project identities portable.** Store project asset metadata and references relative to the project root, separate persistent Scene GUIDs from runtime IDs, and resolve authored asset paths into cooked GUIDs for Players.
- **Make authored files mergeable.** Keep scene, material, prefab and package authoring in stable text documents; preserve independent edits and external revisions, and keep runtime caches, editor view state and MCP sessions outside Git. Git remains responsible for conflict resolution.
- **Publish complete asset changes.** Reject duplicate GUIDs, unsafe paths, unresolved dependencies and invalid fields before publication. Preserve authored metadata and imported subasset identities through rename, reimport, deletion and concurrent document saves.
- **Use current project templates.** Create projects with a complete default forward rendering setup, current scene/component formats and portable version-control rules.

### Scripts and Editor Workflows

- **Reload renamed components.** Resolve scripts by their asset identity and published component declarations, preserve references across class renames, and publish complete component sets and serialized field metadata together.
- **Restore authoring after Play.** Restore saved object data and remap object selections across Play/Stop scene replacement. Failed Play initialization also restores the authoring domain.
- **Preserve editing ownership.** Tighten Undo/Redo, Prefab propagation, animation/timeline saves, asset previews, project selections and pointer capture so retired scene or asset owners cannot receive later edits.
- **Keep panels efficient.** Share observed file and registry snapshots inside the editor's panel mechanism, remove repeated expensive preparation, and make custom-panel and Timeline text inherit the editor font size.
- **Strengthen MCP lifecycle.** Separate general editor operations from the plugin's transport service, retain local endpoint and permission settings, and clean up plugin commands, workers and services with their owners. Update the official MCP client for persistent sessions and virtual-environment launchers.
- **Keep release diagnostics quiet.** Make internal startup and runtime-origin diagnostics opt-in while retaining actionable errors and user logging.

### Physics, Animation and Audio

- **Keep joints physically coherent.** Preserve hinge frames and anchors across compound center-of-mass changes and collapsed scales, and prevent a swinging pendulum from sleeping prematurely.
- **Publish body changes consistently.** Retain authored mass and collider geometry across edits and clones, order nested body poses, compose kinematic targets, and retire joint constraints with their owners.
- **Correct query and contact contracts.** Return target-surface shape-cast contacts and correctly oriented normals, honor query masks and compound ignore rules, and resolve callback targets by their live native lifetime.
- **Repair animation timing.** Correct terminal frames, ordered loop events, Timeline completion and explicitly armed transition triggers; preserve clip events and graph identities through editing and save operations.
- **Respect audio ownership.** Retain shared clip dependencies until the last track releases them, register listeners only while enabled, and apply authored clip import settings in exported Players.

### Rendering, UI and Compute

- **Make first-frame rendering valid.** Resolve linked shadow variants and implicit DefaultLit before the first draw, collect shadow casters across resident scenes, and fit shadow cascades to the effective camera matrices.
- **Strengthen RenderGraph synchronization.** Declare exact texture/buffer consumers, queue sharing and particle dependencies; retire transient targets, descriptors, materials and camera resources after their GPU completion epoch.
- **Validate backend capabilities.** Expose a backend-neutral device capability contract and reject unsupported formats, commands and pipeline layouts before recording or allocation.
- **Correct material and effect composition.** Preserve shader texture defaults, alpha coverage, surface discard, GBuffer scalars, deformed lighting and slot-ordered Bloom/effect chains; retain exact pixels in equal-size transfers.
- **Align UI rendering and input.** Repair World UI depth replay, affine billboard picking, per-frame screen pose uploads, text/font invalidation and camera-local rendering state.
- **Keep CPU/GPU work ordered.** Correct asynchronous compute readback before later uploads, buffer/resource retirement and particle publication. Add parallel JIT material-kernel fixtures that verify mutation-free warmup and required CPU specialization; these fixtures are not a complete MPM solver.

### Players, Hub and Release Delivery

- **Deliver complete target payloads.** Cook enabled package dependencies and current document formats, reject missing scripts before scene replacement, and preserve package data layout in exported Players.
- **Run the Web Player through the full path.** Repair the static NumPy runtime, startup, keyboard input, camera clears and authored physics timing; validate development and release builds in Chromium with pixel and input checks.
- **Keep Windows launches portable.** Host editors outside deeply nested project runtimes, preserve long asset paths at loader boundaries, and report unsupported executable locations before graphics startup.
- **Publish one coherent release.** Require same-revision Desktop and Player acceptance before GitHub/PyPI/R2 publication, verify immutable release identities and update public catalogs only from the actual published files. Official plugins require Infernux >=0.4.2; this release's native payloads are built with 0.4.2.
- **Broaden automated evidence.** Add software-Vulkan correctness coverage, scene residency and headless physics gates, installed Player tests and isolated parallel Windows Python workers. Android CI covers builds and emulation; physical-device validation remains separate.

### Website and Upgrade Notes

- **Refresh the public site.** Ship the dark tape-futurist design, local GSAP animation, unified tutorials, categorized roadmap graphs, bilingual learning chapters and current API documentation.
- **Stream website showcase media efficiently.** Delay offscreen simulation/media work and keep large lazy-loaded content out of service-worker precache.
- **Use the canonical Python namespace.** Author scripts with `import infernux as inx`; update old `Infernux` imports and serialized package names. This release does not promise arbitrary old-file format compatibility.
- **Keep teams on one exact release.** Install 0.4.2-v2 on every development machine and use its matching platform packages. Existing 0.4.1 projects remain pinned to 0.4.1; Hub does not automatically upgrade them. Back up and deliberately update project pins and formats before adopting 0.4.2.
- **Keep the scope clear.** Infernux remains alpha software. Neural Network-Native Engine (3N) is the architectural goal; a complete neural-network inference runtime, networking stack and Tiny Glade/MPM game are not delivered by this release.

---

# Infernux v0.4.1 · Worlds, Compute and Live Tools

0.4.1 brings larger worlds, live Python authoring and target-aware computation into the same engine workflow. This release expands multi-scene editing, prefab authoring, model import, rendering and audio, then carries their resource and component lifecycles into exported Players. It also rebuilds key parts of the plugin experience in response to community reports.

Infernux is working toward a **Neural Network-Native Engine (3N)**. Here, “Neural Network” is one concept: neural networks should eventually participate directly in gameplay and simulation. 041 builds the world, compute and tooling foundations for that direction; it does not claim that the full neural inference and tensor runtime is already delivered.

[简体中文更新日志](UpdateLog-zh.md)

**Baseline for comparison:** [`v0.4.0...v0.4.1`](https://github.com/ChenlizheMe/Infernux/compare/v0.4.0...v0.4.1)

### 0.4.1 v3 · New Project Startup Hotfix

- Correct the Hub's generated scene metadata to use the native `DefaultText` resource type. The invalid `Scene` value prevented freshly created projects from starting the Editor.
- Give newly created scenes independent GUIDs, so separate projects with the same name no longer share a scene identity.
- Store the default scene GUID in Editor settings so new projects open their generated Start scene, including its render stack, instead of an unrelated untitled scene.
- Include the required color-temperature settings in the default light document so the Start scene passes current component validation.
- Fix first Play in projects without user scripts: Editor warmup no longer mistakes an empty script directory for a packaged Player or tries to load its runtime type registry.
- Add regression coverage for native metadata parsing, AssetDatabase scanning, complete scene/component restoration, effect references, Editor and build settings, repeated-scan GUID preservation, and empty-project warmup.
- Updating Hub corrects newly created projects; it does not rewrite existing assets. To repair a generated Start scene while retaining its GUID: change `metadata.resource_type.value` from `Scene` to `DefaultText` in `Assets/Scenes/Start.scene.meta`; add missing `useColorTemperature: false` and `colorTemperature: 6500.0` fields to the Light component's `data` in `Start.scene`; set `lastOpenedSceneGuid` in `ProjectSettings/EditorSettings.json` to the scene metadata's GUID. Use wheel build 3 for the Play Mode fix.
- Engine version remains **0.4.1**, with wheel build **3** and Hub update identity **0.4.1-3**.

### 0.4.1 v2 · Hub and Blender Hotfix

- Linux Hub now resolves the system certificate store when bundled OpenSSL points at the build machine's conda environment. HTTPS verification remains enabled, and explicit administrator certificate settings are preserved.
- Hub records full catalog and installation errors in persistent logs, displays errors as selectable plain text, and lets users retry catalog requests. Blender HTTP failures identify the download stage and source.
- Windows Blender extraction supports deep paths without requiring a global Windows policy change. Failed installations retain the verified archive and preserve the original error.
- The Editor can discover a native Blender installation through the system's `.blend` association on Windows and Linux. Opening a `.blend` file uses the same selected executable as import. External image identities remain consistent during Blender export.
- Rebuilt wheels keep the engine version **0.4.1** and use wheel build **2**. Hub presents one entry per engine version, selects the latest compatible build, and offers an update for a cached older build. Hub's own update identity is **0.4.1-2**.

### Multi-scene Worlds and Gameplay

- **Author several scenes together.** Open scenes additively, choose the active scene, inspect scene ownership in the Hierarchy, and move objects between scenes without treating the editor document as one global world.
- **Make scene transitions coherent.** Commit scene changes at safe points, update component membership and lifecycle plans, and replace the rendered world together with its UI. Fix transitions that changed interface content while leaving old GameObjects visible.
- **Preserve the right state.** Keep persistent objects across loads, isolate Edit and Play state, and release scene-owned resources and callbacks on unload.

### Prefab Authoring

- **Work in isolated Prefab Mode.** Edit prefab contents independently of scene instances, including nested prefabs and variants.
- **Apply and revert precisely.** Track property and structural overrides, including object/component additions, removals and references, against stable source identities and saved baselines.
- **Keep instances synchronized.** Propagate source edits through loaded scenes and variant relationships while preserving authored overrides; retain the same relationships after save and reopen.
- **Undo shared-asset edits.** Connect prefab editing and scene changes to the editor transaction and journal infrastructure instead of leaving shared assets outside Undo/Redo.

### Models, Materials and Animation

- **Import FBX and Blender sources through one model workflow.** Organize settings into Model, Rig, Animation and Materials pages, with explicit Apply and worker-based import rather than repeating expensive work while browsing settings.
- **Preserve source structure and identity.** Keep node-local mesh geometry and hierarchy, stable mesh/subasset references, and external-source synchronization across supported renames and reparenting. Preserve user overrides when the source changes.
- **Control materials and textures.** Extract, find and remap materials; retain embedded textures and external texture references as persistent assets; carry PBR factors, texture sampling and alpha/shadow behavior through import.
- **Control mesh shading.** Add normal/tangent import policies, smoothing-angle and weighted-normal handling, and MikkTSpace tangent generation.
- **Prepare animated assets.** Improve skeleton and skin-weight import, generic/humanoid rig settings, trimmed clips, morph target names and animation asset cooking. Preview imported animation in an isolated GPU-skinned view using the same clip assets.
- **Inspect imported content.** Expose structured model/subasset inventory to Python and MCP, and make imported resources easier to inspect and reuse.

### GUID Resources and Player Content

- **Use one internal asset identity.** Extend GUID references through scenes, prefabs, material shaders, camera target textures, UI assets, effects and build settings. Resolve controlled authoring paths into that identity before runtime delivery.
- **Cook the actual dependency graph.** Include resources reachable from the current documents and package components, preserve required raw runtime files and licenses, and reject uncooked model dependencies instead of relying on editor files being present.
- **Keep project and package access explicit.** Improve package asset indexing and `Application.asset_path` / `Application.package_path` handling, including Windows path aliases.
- **Repair asset command timing.** Run confirmed deletion through the editor command lifecycle after modal interaction; improve DataAsset authoring, inspection and deferred asset operations.

### CPU/GPU Compute and JIT Preparation

- **Integrate compute into the engine.** Move the internal Taichi computation path into the engine and retire the separate integration path; keep Python as the authoring interface for CPU and GPU work.
- **Prepare before interaction.** Connect startup, Play entry, script updates and target builds to warmup discovery. Reuse valid prepared results and report preparation timings rather than recompiling every time Play starts.
- **Strengthen CPU execution.** Improve LLVM specialization, buffer lifetime and alias handling, detached compilation and proven row-local parallel work, with observable compilation/optimization results.
- **Keep GPU work resident.** Improve compute buffers, kernel dispatch, queue ownership and storage bindings; reuse readback ownership and avoid unchanged transfers where supported.
- **Build for the target.** Prepare native GPU payloads for supported targets, including Android. For supported kernels on Web, prepare CPU execution during the build because the native Taichi GPU runtime is unavailable there; projects keep the same authored compute entry points.
- **Make boundaries explicit.** Validate supported kernel and closure contracts during preparation. This is not unrestricted Taichi support in the browser, and Android GPU computation does not require a CPU fallback.

### Physics Queries and Lifecycle

- **Scale scene queries.** Add batched ray work and shared query snapshots with explicit world generations, while reducing repeated compound lookup and body locking.
- **Broaden collider queries.** Improve static mesh closest-point queries and the handling of rotated, scaled and non-convex shapes; invalidate query state when collider cooking replaces geometry.
- **Keep callbacks and tools in the same world.** Tighten contact/impulse delivery and scene ownership, and use consistent scene data for runtime queries and editor inspection.

### Rendering and RenderGraph

- **Align Vulkan and WebGPU contracts.** Correct fullscreen resource binding, storage buffers, reflection/descriptor layouts and resource updates across the two graphics paths. Fix initial scene state that could leave the browser background different from later scene loads.
- **Scope rendering resources to the view.** Improve per-view RenderGraph buffers, camera light lists, shadow resources and deferred normal/geometry coverage, so multiple views do not share the wrong frame state.
- **Extend custom passes.** Support GUID texture imports and compute-buffer inputs in fullscreen passes, and improve material property, custom sampler and texture-update handling.
- **Retire resources in order.** Correct render-resource replacement and teardown, including scene changes and device shutdown, and initialize forward resources before dependent shadow variants.

### Cameras, Text and UI

- **Separate gameplay from observation.** Gameplay camera queries and camera-facing world UI use the game camera. Scene view rendering, navigation and selection rays continue to use the editor view's camera.
- **Author world UI explicitly.** Expose fixed or game-camera-facing orientation, constant screen-size behavior and on-top/occlusion policies. Keep world UI Inspector controls appropriate to world-space content.
- **Make what you see selectable.** Align rendered text, local UI bounds, rectangle handles and click picking; improve enlarged world text clarity and engine-font use.
- **Keep screen UI isolated.** Correct canvas placement and scene ancestry interactions, and improve custom UI material/shader resources and descriptor ownership.
- **Refine physical cameras and Gizmos.** Improve sensor, lens and gate-fit behavior, plus icon scale, tint, billboard orientation and picking.
- **Make editor panels fit.** Shrink the Game view scale slider before hiding FPS, remove excess toolbar spacing, localize layout controls and avoid duplicated visible Hierarchy IDs.

### Audio Runtime

- **Stream without decoding everything up front.** Add file-backed audio streaming with bounded buffering alongside resident PCM clips, and separate audio runtime ownership from editor-only services.
- **Control voices and buses.** Improve voice lifetime, spatial gain/panning, bus/master gain and parameter automation, with output meters and streaming diagnostics.
- **Define the real-time boundary.** Keep audio callback state bounded and publish control changes through explicit handoff. Establish a native output DSP stage; this is not a finished user-facing effects graph.

### Plugins, Components and Hot Reload

- **Retain familiar package roles.** Keep Runtime, Editor and general files, folder-context-menu export to `.inxpkg`, and the repository `package/` + `package.py` workflow.
- **Register Player types before scene construction.** Include packaged and locally authored runtime components in the cooked type registry and run runtime preloads in the proper order. Player startup does not discover host build tools or activate editor-only exporters.
- **Update existing behavior.** Reload Runtime/Editor components and their dependency modules transactionally, replacing behavior on live instances rather than retaining an old `update` implementation after a save. Reject a failed candidate before publishing a mixed module state.
- **Own the whole plugin lifetime.** Track panels, menus, commands, shortcuts, subscriptions and cleanup callbacks through activation, reload, disable and unload, including failed preload cleanup.
- **Load dependencies once when appropriate.** Allow preload to import substantial Python dependencies and manage reversible services; identify native extensions that require an editor restart when their owning package changes.
- **Place tools where authors need them.** Support custom panel placement and arbitrarily nested menu paths. Treat author-provided menu segments as text; use explicit localization keys for engine menus and plugin-owned editor translation catalogs.
- **Teach the complete workflow.** Update the online plugin template and `plugin_pages` tutorials with directory roles, component registration, preload cleanup, localization and File Manager packaging instructions.

### Editor Reliability and Authoring

- **Improve window startup and resizing.** Keep the editor hidden until its loading sequence is ready and fix resize/maximize handling reported on Windows.
- **Make inspection more useful.** Improve read-only/transient Inspector fields, component publication, localized camera/light settings and Console selection, copying and source navigation.
- **Make hot editing observable.** Improve deferred task errors, scene failure reporting, frame/compute diagnostics and MCP inspection so a successful launch is not mistaken for correct simulation.
- **Refresh learning material.** Recheck tutorial parameters and API usage against the implementation, expand hands-on examples, and publish bilingual, versioned release history on the website.

### Builds, Platforms and Hub

- **Give each target its own settings.** Let Windows, Linux, Web and Android exporters declare their build options through a shared editor interface.
- **Present a ready first frame.** Repair initialization in fresh export directories, packed resource startup and scene switching; delay Player presentation until the initial scene is ready to avoid premature black frames.
- **Keep desktop window semantics predictable.** Use borderless desktop mode for fullscreen and client/render-area dimensions for windowed mode, retaining the configured windowed resolution and taskbar behavior.
- **Follow the active Linux window backend.** Enable the Vulkan surface extensions required by SDL's actual X11/Wayland backend without forcing fullscreen or changing Player dimensions. Address dependency nullability warning noise at the dependency boundary.
- **Tighten mobile/browser input and lifecycle.** Improve Android surface pause/resume, native IME input, Back-to-Escape routing, same-frame touch delivery and persistent user-data placement; align browser touch and screen UI event ownership.
- **Ship matching runtime payloads.** Keep native asset/audio ownership consistent across library boundaries and bundle required Python extensions. Use owned dependency forks and checked-in source fixes instead of editing dependency sources during CMake configuration.
- **Improve Hub-managed delivery.** Expand managed Blender/build-tool integration and release catalog synchronization. Keep published versions monotonic and preserve actual audited Linux wheel tags rather than forcing a compatibility label.

### Upgrade Notes and Scope

- **Update engine and platform plugins together.** A precompiled Player payload must match its engine contract. Rebuild exported Players after upgrading rather than copying a new engine library into an old export.
- **Review plugin roles and lifecycle.** Put shipped components/services in Runtime and panels/local web tools/exporters in Editor. Follow the current template for cleanup, translations and `plugin_pages`; archive role directories are lowercase `runtime`/`editor`, while local authored Runtime/Editor folders retain their spelling.
- **Reimport and recook from source.** Preserve your project sources and GUID metadata, then regenerate imported/cooked output with the new engine. The release strengthens the current data path rather than maintaining every experimental intermediate format.
- **Understand compute portability.** Web's supported CPU preparation path differs from native GPU execution. Kernel support is checked at build time; numerical/performance identity across devices is not promised.
- **Keep the roadmap separate.** Full neural model deployment, the tensor data plane, batch worlds and deterministic replay remain future work. The deferred vk-torch delay optimization and a complete theme system are not part of 041.

---

# Infernux v0.4.0 · Multiplatform Builds and Distribution

0.4.0 adds Windows, Linux, Android, and Web Player builds, expands InxPackage authoring and runtime asset delivery, and puts shared build environments under Hub management.

[简体中文更新日志](UpdateLog-zh.md)

**Baseline for comparison:** [`v0.3.7...v0.4.0`](https://github.com/ChenlizheMe/Infernux/compare/v0.3.7...v0.4.0)

### Multiplatform

- Build Windows and Linux Editors and Players, Android APK/AAB packages, and Web Players from the shared build service.
- Deliver precompiled Players, runtimes, and target tools in the four platform plugins; normal exports require no engine checkout, submodules, CMake, or native engine compilation.
- Run native targets on Vulkan and browser targets on WebGPU, including Python gameplay, rendering, input, UI, audio, and particles.
- Exercise the same MultiPlatform040 project across targets, including button-triggered reads of packaged text assets.
- Add Windows/Linux desktop and platform Player CI builds, plus Web browser acceptance.

### Plugins and Assets

- Package only a repository's `package/` directory using a standalone `package.py`; keep README and language-specific build configuration outside the archive.
- Preserve direct local-folder authoring and generate default package metadata from the output filename when no manifest is supplied.
- Include scripts, materials, shaders, and arbitrary runtime files in plugins, with explicit asset metadata and live refresh under `Packages/`.
- Keep Player assets in `Content.inxpkg` instead of expanding the project's authoring directory tree. Resolve engine asset paths through cooked GUID identities and provide filesystem access for payloads that need it.
- Separate editor-only content from Player payloads and distribute platform build support as optional packages.
- Check compatible GitHub releases and explicitly select plugin updates while preserving GUIDs, enabled state, and user-added files; require consent before replacing local edits.
- Refresh the independent official catalog without upgrading project-pinned packages, and resolve former platform sources to their independent repositories.

### Hub and Build Environments

- Provide Windows and Linux Hub distributions and managed Python 3.13 environments.
- Install Android support from the Hub channel on Windows and Linux, sharing SDK, NDK, JDK, Gradle, and target Python dependencies; supply toolchain paths automatically and require installed support before enabling Android plugin import.
- Reuse downloads through the Hub Library while keeping project build caches inside the project.
- Group engine, Python, and Android installations into tabs under Installs, with background jobs, a compact progress strip, an expandable queue, and system-tray support.
- Keep interrupted Android kit downloads in Hub's shared cache so an explicitly restarted installation resumes the download; remove the download cache after successful installation.

---

# Infernux v0.3.7 · Plugins and Skeletal Animation

0.3.7 adds the InxPackage plugin system, moves MCP out of the engine core, and fixes skeletal animation imported from separate FBX files.

[简体中文更新日志](UpdateLog-zh.md)

**Baseline for comparison:** [`v0.3.6...v0.3.7`](https://github.com/ChenlizheMe/Infernux/compare/v0.3.6...v0.3.7)

### Plugins

- Install from a `.inxpkg`, a local folder, Git, or the official list.
- `Runtime/` ships with the game. `Editor/` stays in the Editor.
- Package docs come from `README.md`, `LICENSE`, and `InxPluginPages/`.
- MCP moved to the default official plugin `infernux/mcp`. New projects include it. You can disable or uninstall it.

### Skeletal Animation

- Retarget animation-only FBX files through exact joint identities while accepting Assimp-generated pivot and helper nodes between mapped joints.
- Reject renamed or structurally incompatible rigs instead of guessing from geometry and silently driving the wrong limbs.

### Editor and Authoring

- Use the Plugins window to install, enable, disable, reload, and uninstall packages with visible progress from download through activation.
- Keep Headless and MCP authoring on the same scene, command, permission, and undo paths used by the Editor.

---

# Infernux v0.3.6 · Unified Hierarchy and Hub Updates

Version 0.3.6 unifies the editor Hierarchy and restores automatic update discovery for packaged Infernux Hub builds.

**Baseline for comparison:** [`v0.3.5...v0.3.6`](https://github.com/ChenlizheMe/Infernux/compare/v0.3.5...v0.3.6)

### Editor

- Removed the separate Hierarchy UI mode. Scene and UI objects now share one tree, one selection model, and one context-menu path.
- Preserved Canvas root placement and UI Screen subtree constraints in the unified Hierarchy.

### Hub Updates

- Restored startup update checks in packaged Nuitka Hub builds.
- Kept the Install Engine Version list on the selected Hub theme instead of inheriting the desktop system palette.
- Added verified full-package update fallback and elevation for protected installation directories.
- Changed the release pipeline to publish independently installable full Hub packages instead of incremental patch assets.
