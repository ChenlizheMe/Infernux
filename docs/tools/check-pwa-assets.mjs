import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { inflateSync } from "node:zlib";

const docsRoot = path.resolve("docs");
const failures = [];
let maskableRadiusEvidence = null;
const expectedShortcuts = [
  { short_name: "Tutorials", url: "/tutorials.html" },
  { short_name: "API", url: "/wiki/site/en/api/index.html" },
  { short_name: "Community", url: "/community.html" },
  { short_name: "Download", url: "/download.html" },
];

const assets = [
  {
    src: "/assets/infernux-icon-192.png",
    width: 192,
    height: 192,
    purpose: "any",
    sha256: "edfa0d3e709db4ac3100978575147579d4ccdb63c695c3d551e78bc7891c0f4a",
  },
  {
    src: "/assets/infernux-icon-512.png",
    width: 512,
    height: 512,
    purpose: "any",
    sha256: "9f73c451f95f09decaf95702971099c1a6237a8e454c293f201dddfc7473e280",
  },
  {
    src: "/assets/infernux-icon-maskable-512.png",
    width: 512,
    height: 512,
    purpose: "maskable",
    sha256: "54c43fee25612ce3d2d0fa4f14cff5191149201faa2d97d0b834860bfd3fbcc1",
  },
  {
    src: "/assets/infernux-apple-touch-icon.png",
    width: 180,
    height: 180,
    purpose: "apple-touch-icon",
    sha256: "a4e54a3d319ab3badace561328c94233c07fc3181b116ca137033a68a31de7f5",
  },
];

function fail(message) {
  failures.push(message);
}

function readPng(buffer, label) {
  if (!buffer.subarray(0, 8).equals(Buffer.from("89504e470d0a1a0a", "hex"))) {
    throw new Error(`${label}: invalid PNG signature`);
  }

  let offset = 8;
  let ihdr = null;
  let hasTransparencyChunk = false;
  let palette = null;
  const imageData = [];
  while (offset + 12 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("ascii", offset + 4, offset + 8);
    const dataStart = offset + 8;
    const next = dataStart + length + 4;
    if (next > buffer.length) throw new Error(`${label}: truncated ${type || "PNG"} chunk`);
    if (type === "IHDR") {
      ihdr = {
        width: buffer.readUInt32BE(dataStart),
        height: buffer.readUInt32BE(dataStart + 4),
        bitDepth: buffer[dataStart + 8],
        colorType: buffer[dataStart + 9],
      };
    }
    if (type === "PLTE") palette = buffer.subarray(dataStart, dataStart + length);
    if (type === "IDAT") imageData.push(buffer.subarray(dataStart, dataStart + length));
    if (type === "tRNS") hasTransparencyChunk = true;
    offset = next;
    if (type === "IEND") break;
  }
  if (!ihdr) throw new Error(`${label}: missing IHDR chunk`);
  return { ...ihdr, opaque: !hasTransparencyChunk && ![4, 6].includes(ihdr.colorType), palette, imageData };
}

function paethPredictor(left, up, upperLeft) {
  const estimate = left + up - upperLeft;
  const leftDistance = Math.abs(estimate - left);
  const upDistance = Math.abs(estimate - up);
  const upperLeftDistance = Math.abs(estimate - upperLeft);
  if (leftDistance <= upDistance && leftDistance <= upperLeftDistance) return left;
  return upDistance <= upperLeftDistance ? up : upperLeft;
}

function indexedPixels(png, label) {
  if (png.bitDepth !== 8 || png.colorType !== 3 || !png.palette || !png.imageData.length) {
    throw new Error(`${label}: safe-zone audit requires an 8-bit indexed PNG`);
  }
  const compressed = Buffer.concat(png.imageData);
  const raw = inflateSync(compressed);
  const stride = png.width;
  if (raw.length !== (stride + 1) * png.height) throw new Error(`${label}: unexpected decoded scanline length`);

  const rows = [];
  let offset = 0;
  for (let y = 0; y < png.height; y += 1) {
    const filter = raw[offset];
    offset += 1;
    const encoded = raw.subarray(offset, offset + stride);
    offset += stride;
    const row = Buffer.alloc(stride);
    const previous = rows[y - 1] || Buffer.alloc(stride);
    for (let x = 0; x < stride; x += 1) {
      const left = x ? row[x - 1] : 0;
      const up = previous[x];
      const upperLeft = x ? previous[x - 1] : 0;
      let predictor = 0;
      if (filter === 1) predictor = left;
      else if (filter === 2) predictor = up;
      else if (filter === 3) predictor = Math.floor((left + up) / 2);
      else if (filter === 4) predictor = paethPredictor(left, up, upperLeft);
      else if (filter !== 0) throw new Error(`${label}: unsupported PNG filter ${filter}`);
      row[x] = (encoded[x] + predictor) & 0xff;
    }
    rows.push(row);
  }
  return rows;
}

function maxForegroundRadius(png, label, background = [10, 12, 17]) {
  const rows = indexedPixels(png, label);
  const centerX = png.width / 2;
  const centerY = png.height / 2;
  let maximum = 0;
  let foregroundPixels = 0;
  for (let y = 0; y < png.height; y += 1) {
    for (let x = 0; x < png.width; x += 1) {
      const paletteOffset = rows[y][x] * 3;
      const isBackground = background.every((channel, index) => png.palette[paletteOffset + index] === channel);
      if (isBackground) continue;
      foregroundPixels += 1;
      maximum = Math.max(maximum, Math.hypot(x + 0.5 - centerX, y + 0.5 - centerY));
    }
  }
  if (!foregroundPixels) throw new Error(`${label}: contains no foreground emblem pixels`);
  return maximum;
}

function iconHasPurpose(icon, purpose) {
  return String(icon?.purpose || "any").split(/\s+/).includes(purpose);
}

for (const asset of assets) {
  const file = path.join(docsRoot, asset.src.slice(1));
  try {
    const body = await readFile(file);
    const png = readPng(body, asset.src);
    if (png.width !== asset.width || png.height !== asset.height) {
      fail(`${asset.src}: expected ${asset.width}x${asset.height}, found ${png.width}x${png.height}`);
    }
    if (!png.opaque) fail(`${asset.src}: install and touch icons must have a deterministic opaque background`);
    if (asset.purpose === "maskable") {
      const actualRadius = maxForegroundRadius(png, asset.src);
      const safeRadius = Math.min(png.width, png.height) * 0.4;
      maskableRadiusEvidence = { actualRadius, safeRadius };
      if (actualRadius > safeRadius) {
        fail(`${asset.src}: foreground radius ${actualRadius.toFixed(2)}px exceeds the ${safeRadius.toFixed(2)}px maskable safe zone`);
      }
    }
    const hash = createHash("sha256").update(body).digest("hex");
    if (hash !== asset.sha256) fail(`${asset.src}: reviewed SHA-256 changed (${hash})`);
  } catch (error) {
    fail(error.message);
  }
}

let manifest = null;
try {
  manifest = JSON.parse(await readFile(path.join(docsRoot, "site.webmanifest"), "utf8"));
} catch (error) {
  fail(`site.webmanifest: ${error.message}`);
}

if (manifest) {
  for (const key of ["name", "short_name", "id", "start_url", "scope", "display", "icons"]) {
    if (!manifest[key] || (Array.isArray(manifest[key]) && !manifest[key].length)) fail(`site.webmanifest: missing '${key}'`);
  }
  if (manifest.id !== "/" || manifest.start_url !== "/" || manifest.scope !== "/") {
    fail("site.webmanifest: id, start_url, and scope must stay root-scoped");
  }
  if (!new Set(["fullscreen", "standalone", "minimal-ui"]).has(manifest.display)) {
    fail("site.webmanifest: display must remain installable");
  }
  if (manifest.prefer_related_applications !== false) {
    fail("site.webmanifest: prefer_related_applications must explicitly remain false");
  }

  const manifestIcons = Array.isArray(manifest.icons) ? manifest.icons : [];
  for (const asset of assets.filter((entry) => entry.purpose === "any" || entry.purpose === "maskable")) {
    const icon = manifestIcons.find((entry) => entry.src === asset.src);
    if (!icon) {
      fail(`site.webmanifest: missing ${asset.purpose} icon '${asset.src}'`);
      continue;
    }
    if (icon.sizes !== `${asset.width}x${asset.height}` || icon.type !== "image/png" || !iconHasPurpose(icon, asset.purpose)) {
      fail(`site.webmanifest: invalid metadata for '${asset.src}'`);
    }
  }
  if (!manifestIcons.some((icon) => icon.sizes === "192x192" && iconHasPurpose(icon, "any"))) {
    fail("site.webmanifest: Chromium installability requires a 192x192 any-purpose icon");
  }
  if (!manifestIcons.some((icon) => icon.sizes === "512x512" && iconHasPurpose(icon, "any"))) {
    fail("site.webmanifest: Chromium installability requires a 512x512 any-purpose icon");
  }
  if (!manifestIcons.some((icon) => icon.sizes === "512x512" && iconHasPurpose(icon, "maskable"))) {
    fail("site.webmanifest: missing the reviewed 512x512 maskable icon");
  }
  if (new Set(manifestIcons.map((icon) => icon.src)).size !== manifestIcons.length) {
    fail("site.webmanifest: icon sources must be unique");
  }

  const shortcuts = Array.isArray(manifest.shortcuts) ? manifest.shortcuts : [];
  if (shortcuts.length !== expectedShortcuts.length) {
    fail(`site.webmanifest: expected ${expectedShortcuts.length} task shortcuts, found ${shortcuts.length}`);
  }
  for (const [index, expected] of expectedShortcuts.entries()) {
    const shortcut = shortcuts[index];
    if (!shortcut) continue;
    if (shortcut.short_name !== expected.short_name || shortcut.url !== expected.url) {
      fail(`site.webmanifest: shortcut ${index + 1} must remain '${expected.short_name}' -> '${expected.url}'`);
    }
    if (!String(shortcut.name || "").trim() || !String(shortcut.description || "").trim()) {
      fail(`site.webmanifest: shortcut '${expected.short_name}' requires a full name and description`);
    }
  }
  if (new Set(shortcuts.map((shortcut) => shortcut.url)).size !== shortcuts.length) {
    fail("site.webmanifest: shortcut URLs must be unique");
  }
  if (new Set(shortcuts.map((shortcut) => shortcut.short_name)).size !== shortcuts.length) {
    fail("site.webmanifest: shortcut short names must be unique");
  }
  for (const shortcut of shortcuts) {
    if (!String(shortcut.url || "").startsWith("/")) fail(`site.webmanifest: shortcut '${shortcut.name}' must stay in root scope`);
    if (!shortcut.icons?.some((icon) => icon.src === "/assets/infernux-icon-192.png" && icon.sizes === "192x192")) {
      fail(`site.webmanifest: shortcut '${shortcut.name}' must use the reviewed 192px icon`);
    }
    try {
      await readFile(path.join(docsRoot, shortcut.url.slice(1)));
    } catch {
      fail(`site.webmanifest: shortcut '${shortcut.name}' targets a missing published route '${shortcut.url}'`);
    }
  }
}

const rootPages = ["index.html", "tutorials.html", "start.html", "learn.html", "roadmap.html", "community.html", "download.html", "offline.html"];
for (const page of rootPages) {
  try {
    const html = await readFile(path.join(docsRoot, page), "utf8");
    if (!html.includes('rel="manifest" href="site.webmanifest"')) fail(`${page}: missing root-scoped manifest link`);
    if (!html.includes('rel="apple-touch-icon" sizes="180x180" href="assets/infernux-apple-touch-icon.png"')) {
      fail(`${page}: missing reviewed Apple touch icon link`);
    }
  } catch (error) {
    fail(`${page}: ${error.message}`);
  }
}

for (const [page, prefix] of [
  ["404.html", "/"],
  [path.join("wiki", "theme", "main.html"), "/"],
  [path.join("wiki", "site", "en", "api", "index.html"), "/"],
]) {
  try {
    const html = await readFile(path.join(docsRoot, page), "utf8");
    const contract = `rel="apple-touch-icon" sizes="180x180" href="${prefix}assets/infernux-apple-touch-icon.png"`;
    if (!html.includes(contract)) fail(`${page}: missing reviewed Apple touch icon link`);
  } catch (error) {
    fail(`${page}: ${error.message}`);
  }
}

try {
  const serviceWorker = await readFile(path.join(docsRoot, "sw.js"), "utf8");
  const precacheMatch = serviceWorker.match(/const PRECACHE_URLS = (\[[\s\S]*?\]);/);
  if (!precacheMatch) throw new Error("deterministic core-shell list is missing");
  const precacheRoutes = JSON.parse(precacheMatch[1]);
  if (!precacheRoutes.includes("/site.webmanifest")) fail("sw.js: installable core shell must retain the manifest");
  for (const asset of assets) {
    if (precacheRoutes.includes(asset.src)) fail(`sw.js: platform-managed install icon '${asset.src}' must not consume core-shell precache bandwidth`);
  }
} catch (error) {
  fail(`sw.js: ${error.message}`);
}

try {
  const provenance = await readFile(path.join(docsRoot, "assets", "VENDOR_ASSETS.md"), "utf8");
  for (const asset of assets) {
    if (!provenance.includes(`\`${path.posix.basename(asset.src)}\``) || !provenance.includes(asset.sha256)) {
      fail(`assets/VENDOR_ASSETS.md: missing provenance and checksum for '${asset.src}'`);
    }
  }
} catch (error) {
  fail(`assets/VENDOR_ASSETS.md: ${error.message}`);
}

if (failures.length) {
  console.error(`PWA asset check failed with ${failures.length} issue(s):`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`PWA asset check passed: Chromium icon sizes, four task shortcuts, Apple touch icon, manifest metadata, provenance, and platform-managed loading are locked; maskable foreground radius ${maskableRadiusEvidence.actualRadius.toFixed(2)}px / ${maskableRadiusEvidence.safeRadius.toFixed(2)}px safe zone.`);
