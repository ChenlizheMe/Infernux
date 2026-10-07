# Website build tools

This directory contains deterministic website generation and validation tools.
They are intentionally kept beside the static site they maintain. The
repository-level maintainer entry point is
`scripts/docs/update_api_docs.bat`.

## Naming contract

| Prefix | Responsibility | May update checked-in output |
|:-------|:---------------|:-----------------------------|
| `build-*` | Generate a deterministic website artifact | Yes; most support `--check` |
| `apply-*` / `normalize-*` / `optimize-*` / `stamp-*` | Transform or normalize one documented part of the site | Yes; check mode where applicable |
| `check-*` | Enforce a static, deployment, performance, or accessibility contract | No, except an explicitly requested report file |
| `test-*` | Exercise browser-independent website behavior and regressions | No |
| `verify-site.mjs` | Aggregate repository, release, page, and metadata consistency checks | No |

`i18n-source.json` is the source of truth for route-localized text.
`build-i18n.mjs` produces the checked-in `docs/js/i18n*.js` bundles; do not edit
those bundles by hand. `normalize-i18n-fallbacks.mjs` rewrites the inline English
inside every `[data-i18n]` element to match that same source, so the pre-script
and crawler view of a page never drifts from the localized copy.

Roadmap node content lives in `docs/data/roadmap/<category>.json`: keep each goal's
English, Chinese and status together there. `docs/js/roadmap-data.js` is the category
index; update its node count and data URL version when editing a category. The page
loads only the selected category. The shell still uses `i18n-source.json`.
`live` denotes an implemented capability, `planned` a remaining or unverified goal, `progress`
explicitly confirmed active work, and `future` a longer-term research direction.
Only `progress` receives the slow GSAP pulse; do not infer it from partial completion.
Mixed branches with implemented leaves derive the static `partial` status. See
`docs/roadmap-status-audit.md` for implementation evidence and scope of the status review.
Only implemented planets use the category color. Remaining planets are charcoal;
partial parents show their implemented leaf fraction as a colored arc. Detailed
goals never inherit completion from an implemented, broader capability.
Neither category hubs nor a roadmap label
constitute completion or a promised release date. `roadmap-layout.js` lays out
the actual branch sizes radially with a stable seed and reserved bilingual label
footprints. Run `node docs/tools/test-roadmap-data.mjs` after edits to catch duplicate
goals, missing translations, disconnected nodes and overlapping footprints.

The GitHub workflows in `.github/workflows/build-wiki.yml` and
`.github/workflows/website-quality.yml` define the authoritative execution
order. When adding a new generator, add its check to the quality workflow and
document whether it mutates a committed artifact.

## Release versions

The current website, API baseline, Hub catalog, and engine use the same version
from `pyproject.toml`. Keep older release notes, download options, and API snapshots
as history; do not relabel their artifacts.

Generate `release.json` and `hub-catalog.json` from the versioned Windows/Linux
distribution directory with `python scripts/release/build_release_catalog.py
--release-dir dist/releases/<version>`. The optional `--linux-inventory` accepts
a verified CI archive inventory (`files` mapping archive paths to byte sizes,
plus its parsed Hub `manifest`). Artifact filenames and sizes come from those
inputs, not from the previous release.

Before publication, `published_at` is null and current-release downloads stay
disabled. After uploading the matching GitHub Release assets, rerun the generator
with its actual `--published-at` timestamp, regenerate release notes and the
Service Worker, and deploy the catalogs. Changing version metadata alone does
not upload binaries or publish a release.

## Visual system and roadmap performance

The website uses one dark palette in `css/style.css`. Keep the red brand accent
in the navigation and controls; category hues belong to implemented roadmap
capabilities. No global scanline, blur or constantly animated decorative layers.
The real renderer capture is reused in the hero and showcase (one asset transfer).

Roadmap stars live in a fixed viewport SVG, independent of camera zoom. Fine labels
appear when they have at least 10 screen pixels; branch labels remain readable in
the overview. GSAP animates one graph entrance and explicitly authored progress.
Progress pauses while offscreen or in a hidden browser tab; reduced motion removes
it. At most two rendered category SVGs remain in memory, with fetched JSON cached.
Changing these behaviors requires checking search focus, category switching during
slow requests, small screens and reduced-motion preferences in a real browser.

Roadmap interaction uses one demand-driven animation frame and a GSAP CSS setter
for the composited HTML graph plane. Pointer positions use that plane's untransformed
SVG viewport, including its letterboxing, never the decorative star SVG. Wheel deltas
are continuous and respect their unit; dragging may start on a planet and selects it
only below a 4px movement threshold. Stop camera frames when switching maps or hiding
the document. Measure all major labels in a single read pass on load/font/language
changes; do not call `getBBox()` inside camera or pointer updates. Shared SVG symbols
hold the six planet designs. Straight connectors retain a constant screen stroke.

Layouts pack each parent's local children first, then reserve the complete cluster
in the global map. Bilingual label footprints must remain disjoint. Child spacing
is independent of total category density; tests constrain distances and size ratios.
