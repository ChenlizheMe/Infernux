# Preview 2D animation stack (clips, FSM, SpiritAnimator) and docs

## Summary

This branch introduces a **preview** 2D animation system: sprite-driven playback via `SpiritAnimator`, `AnimClip2D` assets, animation state machine assets, and editor tooling (clip editor, FSM graph with `node_graph_view`, asset inspector integration). Supporting work touches the C++ editor shell and GUI bindings, resource loading, play mode, and serialization so new asset types behave like the rest of the pipeline.

Public messaging is updated so README, Chinese README, and the GitHub Pages landing site (capabilities card plus roadmap/status copy in `docs/js/i18n.js`) describe **2D animation as preview** and keep the next milestone focused on **3D skeletal** animation and advanced UI. Wiki API docs and the static MkDocs output under `docs/wiki/site/` were regenerated. The API doc generator now **YAML-quotes** manual nav titles that contain `:`, so `mkdocs.yml` stays valid when titles include colons.

## What changed

- Added `SpiritAnimator` (`animator2d.py`), `AnimStateMachine` / graph core (`anim_state_machine.py`, `node_graph.py`), and extended `animation_clip` / asset plumbing for 2D clips and FSM assets.
- New editor UI: `animclip2d_editor_panel.py`, `animfsm_editor_panel.py`, `node_graph_view.py`; expanded `asset_inspector`, dialogs, inspector reference handling, and panel/bootstrap wiring; removed legacy `menu_bar.py` in favor of consolidated menu handling.
- C++: editor panels (menu bar, project, toolbar, console, status bar), `InxGUI` / renderer hooks, material pipeline and pybinding updates for GUI and runtime integration.
- Documentation: `README.md`, `README-zh.md`, `docs/index.html`, `docs/js/i18n.js`; regenerated `docs/wiki/*` API pages and `docs/wiki/site/`, `docs/assets/wiki-docs.json`.
- Tooling: `docs/wiki/generate_api_docs.py` quotes nav keys containing `:` when writing `mkdocs.yml`.
- Tests: small adjustments in `test_integration_components.py` and `test_jit.py`; `jit.py` / `jit.pyi` trimmed where redundant.

## Verification

- [ ] Built the affected targets
- [ ] Ran the relevant tests or static validation
- [ ] Updated docs if behavior or public APIs changed

## Notes for reviewers

- **Preview contract:** 2D animation APIs, asset formats, and editor UX may still change; callouts were added in README and on the marketing site.
- **Large UI additions:** `animfsm_editor_panel.py` and `node_graph_view.py` are substantial; focus review on graph semantics, undo, and play/edit mode boundaries.
- **Native/Python boundary:** Confirm GUI bindings and resource reload paths for new asset types; watch for edge cases when entering/exiting play mode with open editor panels.
- **Docs noise:** `docs/wiki/site/` has many generated HTML assets; skim structural changes rather than line-by-line HTML.
- **MkDocs:** Regeneration requires `python docs/wiki/generate_api_docs.py` then `python -m mkdocs build --clean -f docs/wiki/mkdocs.yml` (also available as CMake targets `generate_api_docs` / `build_wiki_html` per README).
