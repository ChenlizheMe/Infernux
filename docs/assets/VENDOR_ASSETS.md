# Website vendor assets

These files are committed so the GitHub Pages experience does not depend on third-party font or icon CDNs at runtime.

| Local file | Upstream | Version/source | SHA-256 |
|---|---|---|---|
| `fonts/source-han-sans-sc-subset.woff2` | Google Noto Sans SC (Source Han Sans family) | `NotoSansSC-VF.ttf`; website character subset | `98038ca3eba66feb91ddec25f83bf0115e54a021b1e1a2f480b173f012b5ed8c` |
| `fonts/inter-latin.woff2` | Google Fonts / Inter | `fonts.gstatic.com/s/inter/v20` | `3100e775e8616cd2611beecfa23a4263d7037586789b43f035236a2e6fbd4c62` |
| `fonts/jetbrains-mono-latin.woff2` | Google Fonts / JetBrains Mono | `fonts.gstatic.com/s/jetbrainsmono/v24` | `83c005d49d8a6a50474c73a5a36ac0468076e9c4a29da7bdb14995d80560a5be` |
| `fonts/space-grotesk-latin.woff2` | Google Fonts / Space Grotesk | `fonts.gstatic.com/s/spacegrotesk/v22` | `0640890476fc1198ab4de571fb658de443c4d85b66466ec09534a8737ab1ce9d` |
| `fonts/fa-solid-subset-900.woff2` | Font Awesome Free | `6.4.0`; 31-glyph site subset | `1ab0dea7613a56456bd30de51fee7d0fccb6def013fe1f46862e2eb204fba343` |
| `fonts/fa-brands-subset-400.woff2` | Font Awesome Free | `6.4.0`; GitHub + Python subset | `7d7c0b8449df96bbfdc8b4e6c6740ce2337af2c90363a5213713977df3e7ae76` |

Font license texts are preserved in `vendor-licenses/`. When any asset changes, update its version/source, checksum, license, and the static-site verifier in the same change.

The `source-han-sans-sc-subset.woff2` file is a character subset of Google Noto Sans SC, a Source Han Sans family font, distributed under SIL OFL 1.1; its license is preserved in `vendor-licenses/Noto-Sans-SC-OFL.txt`.

## Project-authored visual assets

`infernux-social-card-0.3.4.jpg` is the reviewed 1200×630 Open Graph/X card for release 0.3.4. It is a center-cropped derivative of the original 1920×1032 editor capture (`demo.png`, since retired from the repository) rather than separate promotional artwork. Its SHA-256 is `c1bb18887d484776433a14f43bf2d77dde6a9fe4f8eabc077e6fbb541c273159`. The site verifier locks its format, dimensions, release-scoped filename, and reviewed content hash.

The original `demo.png` capture and its `demo-runtime.webp` / `demo-runtime.avif` homepage derivatives of the 65,536-object voxel continent were retired with the homepage redesign. Both the homepage and the READMEs now show the demo reel below.

### README demo loops

The READMEs embed five GIF loops from the same reel, stored in `.github/media/` so the website never delivers them. Each is cropped to 852×388 like the clips, resampled with Lanczos, and quantised with a per-clip `palettegen` (`stats_mode=diff`) and `paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle`: `space-battle.gif` (600 px, 12 fps, 48 colours), `npr-pipeline.gif`, `rigid-coins.gif`, `animated-cats.gif` (320 px, 10 fps, 48 colours) and `rendergraph-grid.gif` (320 px, 8 fps, 32 colours). `check-image-variants.mjs` enforces their presence, GIF signature, per-file and total size limits, and that both READMEs reference them.

### Demo reel

The homepage reel is cut from the author's own Bilibili video `BV1538P6jELT`, part 2 ("熔炉0.3.4演示Demo纯净版"), downloaded as the 852×480 H.264 stream. Each clip is cropped to 852×388 to drop the title and subtitle bands (and the channel watermark), resampled with Lanczos to the listed width, re-timed to 24 fps and encoded with ffmpeg 7.1 (imageio-ffmpeg): `libx264 -preset veryslow -crf 30 -profile:v high -pix_fmt yuv420p -an -movflags +faststart`. Posters are single frames saved with Pillow 12.2.0 as WebP (quality 68; the feature poster 72). Clips carry `preload="none"`, a `data-src` and a poster; `js/fx-hud.js` attaches the source and plays a clip only while it is visible, and never under reduced motion unless the visitor presses play.

| Local file | Source time | Geometry | Bytes | SHA-256 |
|---|---|---|---:|---|
| `reel/space-battle.mp4` | 01:24.5, 4.5 s | 852×388 | 175,117 | `30dec9c1672fb13316b54240787167bd2078d35f5324a24f983abeae2c2a1af8` |
| `reel/fft-ocean.mp4` | 00:03.0, 4 s | 640×292 | 202,626 | `17062becf53d7c699724c04f6d06ce616796c8b17c1cb1ac073036c5fc656f7a` |
| `reel/npr-pipeline.mp4` | 00:19.0, 4 s | 640×292 | 111,851 | `ba8095dbdc583e0da7c58cd95d3608af3178d5b0b3cd3710d3c2fea0999c3453` |
| `reel/rigid-coins.mp4` | 00:33.5, 4 s | 640×292 | 290,880 | `5471e528ceac863eb59f2f2e156b51a6a47e48e6e3f8ebe9807b4d599da51bb8` |
| `reel/animated-cats.mp4` | 00:58.0, 4 s | 640×292 | 98,641 | `a6e4349dc2167253124ec1d1440c96fa81ad722c6786c9fd1dbaced199803856` |
| `reel/rendergraph-grid.mp4` | 01:12.5, 4 s | 640×292 | 73,309 | `5bc45cf8b26432f486c4d86210dd54c204169821b1e25ee82015a06211b372e0` |

`docs/tools/check-image-variants.mjs` locks every clip and poster by content hash, checks the MP4 `ftyp`/`moov`-before-`mdat` layout and poster dimensions, enforces per-clip and total size limits, and verifies the lazy `<video>` markup. The performance budget counts reel posters in the homepage first view while excluding the clips themselves.

### Install and touch icons

`logo-mark.webp` (76×76, Pillow 12.2.0 Lanczos, WebP quality 88; 3,410 bytes, SHA-256 `6b1b775c01bced8d73551bb11190df8e1828ae96009c49eed4e6b65474aaf123`) replaces the 256×256 `logo.png` in every navigation, footer and offline-page brand mark, which displays at 38 CSS pixels. `favicon-64.png` (64×64 Lanczos PNG; 5,520 bytes, SHA-256 `c127ac7f561a58aae6059e47591cb41cb372d7525417412a83448d26130e43a7`) is the page favicon. Both derive from the repository-owned `logo.png`, which remains the structured-data organisation logo.

The install icons are deterministic, project-authored derivatives of the repository-owned `logo.png`; they do not introduce an external artwork source or license. Pillow 12.2.0 in the repository `infernux` environment resized the source with Lanczos sampling, composited it over the site background `#0a0c11`, and wrote optimized 256-color opaque PNGs. The standalone maskable asset keeps the complete emblem inside the Web App Manifest safe-zone circle (radius 40% of the canvas); it is intentionally more padded than the ordinary launcher icons.

| Local file | Role and geometry | Bytes | SHA-256 |
|---|---|---:|---|
| `infernux-icon-192.png` | Chromium install icon; 192×192, opaque | 10,064 | `edfa0d3e709db4ac3100978575147579d4ccdb63c695c3d551e78bc7891c0f4a` |
| `infernux-icon-512.png` | Chromium install/splash icon; 512×512, opaque | 49,570 | `9f73c451f95f09decaf95702971099c1a6237a8e454c293f201dddfc7473e280` |
| `infernux-icon-maskable-512.png` | Adaptive launcher icon; 512×512, opaque, safe-zone padded | 25,603 | `54c43fee25612ce3d2d0fa4f14cff5191149201faa2d97d0b834860bfd3fbcc1` |
| `infernux-apple-touch-icon.png` | Apple home-screen icon; 180×180, opaque | 9,163 | `a4e54a3d319ab3badace561328c94233c07fc3181b116ca137033a68a31de7f5` |

`docs/tools/check-pwa-assets.mjs` locks each file's PNG signature, dimensions, opacity, reviewed hash, manifest role, HTML link, provenance, and Service Worker precache entry. Regenerate and review the complete set together if the emblem or background color changes.

## Font Awesome subsetting

The two Font Awesome files are generated from the official 6.4.0 `webfonts` files with the `pyftsubset` executable provided by the repository's `infernux` environment. The solid subset contains the code points declared by the first `unicode-range` in `css/fontawesome-subset.css`; the brand subset contains `U+F09B` (GitHub) and `U+F3E2` (Python). When a new icon class is introduced, regenerate the matching WOFF2, update its `unicode-range` and checksum, then run the website verifier. The verifier rejects missing CSS mappings, unexpected font hashes, and Font Awesome files large enough to indicate that a complete upstream font was restored.
