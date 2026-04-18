## Infernux v0.1.4

This release advances the **2D rendering and animation authoring stack**, tightens several **editor and asset workflows**, and lands a substantial **node-graph / animation FSM editor** refresh on top of the work already tagged as 0.1.4. Baseline for comparison: **v0.1.3** (`52d87ce0bfdfd922acc29071d0307ecf837d0273`).

---

### Rendering and 2D

- **Sprite pipeline:** Shader-based 2D sprite rendering with aspect handling, integrated with the asset system and editor inspectors.
- **Textures:** Filter, wrap, and anisotropic sampling are wired through the full material / render path; **`SPRITE`** is added as a first-class texture usage for 2D content.
- **Scene / lighting:** Improvements around double-sided shadows and more predictable shadow behaviour in common setups.

### Animation (authoring and runtime)

- **Clips and state machines:** `.animclip2d` authoring in the editor, `.animfsm` state-machine assets, runtime playback via **`SpiritAnimator`**, and dedicated editor panels for clips and FSM graphs.
- **2D animation preview:** Stack-style preview in the clip workflow and related inspector integration.
- **Node graph (FSM editor):**
  - **Camera model** uses a single graph-space view centre plus zoom; pan is derived from the real ImGui canvas item rectangle so pan/zoom and saved view state stay aligned across docking and window resizes.
  - **Per-state header colour** is stored on the asset (`AnimState.header_color`), edited from a **header swatch on the node** (not the side inspector), with undo integration.
  - **Narrower state nodes**, **larger title and pin (In/Out) labels**, optional **brighter pin label colour** for readability, and **transition link midpoint text removed** for a cleaner graph.
  - **Side panel widths** in the FSM editor are tightened to give more space to the graph.
- **Core graph model:** `NodeGraph` gains clearer schema/versioning and helpers; unit tests cover round-trip serialization.

### Editor, UI, and tooling

- **UI Editor:** Workspace and canvas chrome use the same dark palette as the node graph (`NODE_GRAPH_*` alignment); workspace is explicitly filled so the void colour matches the graph editor.
- **Inspector / theme:** Shared **`InspectorThemeBase`** styling (`inspector_theme.py`) and **`panel_spacing`** utilities reduce duplicated colour math and keep inspector stacks and materials consistent.
- **Materials:** Inspector material path and bootstrap inspector materials wiring are adjusted for the new texture and sprite options.
- **Input / ImGui:** Small key and binding surface updates for editor shortcuts.
- **Player mode:** Less editor UI overhead when running in player mode.
- **Misc editor fixes:** Console behaviour around play/clear, script lifecycle in edit mode, scene rename vs build settings, and related hotfixes from the 0.1.3 → 0.1.4 window.

### Documentation and distribution

- **README / site / wiki:** Brought up to date with 0.1.4, including demo callouts and API index refresh where applicable.
- **JIT documentation:** Architecture write-up expanded with a deeper dive and roadmap notes.
- **Python wheel:** `infernux-0.1.4-cp312-cp312-win_amd64.whl` (CPython 3.12, Windows amd64) is produced by the normal CMake **`release`** preset (`package_and_install_python`); the Hub and installers continue to live under `dist/` as in prior releases.

---

### Upgrade notes

- **Python 3.12** remains the supported baseline for the packaged wheel.
- **2D animation formats** (clips / FSM) are still evolving; treat advanced authoring features as **preview** where noted in docs.

---

### Thank you

Thanks to everyone reporting issues and trying early builds—feedback on the 2D and animation path directly shaped this release.
