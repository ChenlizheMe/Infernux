# Roadmap Three.js runtime

`roadmap-three.js` is a local, minified ES module built from three.js 0.186.1 and the official EffectComposer, RenderPass, UnrealBloomPass, and OutputPass addons. It is loaded only by the roadmap page.

Source: https://github.com/mrdoob/three.js (MIT; see `three-LICENSE`).

Build entry exports the renderer, scene/camera, points/line geometry, and postprocessing classes used by `docs/js/roadmap.js`. Built with esbuild 0.28.2, using `--bundle --format=esm --minify`.
