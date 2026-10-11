import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Evidence media: the 0.3.4 demo reel as looping H.264 clips with WebP posters
// on the homepage, and as small GIF loops in .github/media for the READMEs.
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(docsRoot, "..");
const failures = [];
const fail = (message) => failures.push(message);

// Reviewed clips cut from the author's Bilibili video BV1538P6jELT (part 2,
// "熔炉0.3.4演示Demo纯净版"). The feature clip keeps the 852px source width.
const reel = [
    { name: "space-battle", width: 852, height: 388, mp4: "30dec9c1672fb13316b54240787167bd2078d35f5324a24f983abeae2c2a1af8", poster: "548dd95143120536460b25f93a5d98e9aba6193c68319aa7a10b7c290d7ed3cd" },
    { name: "fft-ocean", width: 640, height: 292, mp4: "17062becf53d7c699724c04f6d06ce616796c8b17c1cb1ac073036c5fc656f7a", poster: "fc33d5bd25649ab1a9f432b2dbb04173efb791a924a4a2c367e31cb05af18d3c" },
    { name: "npr-pipeline", width: 640, height: 292, mp4: "ba8095dbdc583e0da7c58cd95d3608af3178d5b0b3cd3710d3c2fea0999c3453", poster: "9f9a9c40c2fba0ebaa4a24809df1da8cfc502cfa116eef50f8f7d0342398d190" },
    { name: "rigid-coins", width: 640, height: 292, mp4: "5471e528ceac863eb59f2f2e156b51a6a47e48e6e3f8ebe9807b4d599da51bb8", poster: "baeb3229b80b8ffb27df27c0479489d3377c86c3d68c92993958efc4096ae6a8" },
    { name: "animated-cats", width: 640, height: 292, mp4: "a6e4349dc2167253124ec1d1440c96fa81ad722c6786c9fd1dbaced199803856", poster: "46688eeb7ec5f18d7881551ee41983197ac81e3188b68a0a7c180f49659c863d" },
    { name: "rendergraph-grid", width: 640, height: 292, mp4: "5bc45cf8b26432f486c4d86210dd54c204169821b1e25ee82015a06211b372e0", poster: "5ae93b2ced81e6afe699440603962f9a38bef938f5c6e2e07a82546e8e15f5e8" },
];
const limits = { clip: 320 * 1024, poster: 24 * 1024, reel: 1100 * 1024 };

const sha256 = (buffer) => createHash("sha256").update(buffer).digest("hex");


function webpDimensions(buffer) {
    if (buffer.length < 30 || buffer.subarray(0, 4).toString("ascii") !== "RIFF" || buffer.subarray(8, 12).toString("ascii") !== "WEBP") return null;
    const chunk = buffer.subarray(12, 16).toString("ascii");
    if (chunk === "VP8L" && buffer[20] === 0x2f) {
        return { width: 1 + buffer[21] + ((buffer[22] & 0x3f) << 8), height: 1 + ((buffer[22] & 0xc0) >> 6) + (buffer[23] << 2) + ((buffer[24] & 0x0f) << 10) };
    }
    if (chunk === "VP8X") return { width: 1 + buffer.readUIntLE(24, 3), height: 1 + buffer.readUIntLE(27, 3) };
    if (chunk === "VP8 " && buffer.subarray(23, 26).toString("hex") === "9d012a") {
        return { width: buffer.readUInt16LE(26) & 0x3fff, height: buffer.readUInt16LE(28) & 0x3fff };
    }
    return null;
}

// Top-level MP4 boxes: an ftyp brand, and moov ahead of mdat so playback can
// start before the whole clip arrives (ffmpeg -movflags +faststart).
function mp4Layout(buffer) {
    const boxes = [];
    for (let offset = 0; offset + 8 <= buffer.length && boxes.length < 32;) {
        const size = buffer.readUInt32BE(offset);
        const type = buffer.subarray(offset + 4, offset + 8).toString("ascii");
        boxes.push(type);
        if (size < 8) break;
        offset += size;
    }
    return boxes;
}


let reelBytes = 0;
for (const clip of reel) {
    const mp4 = await readFile(path.join(docsRoot, "assets", "reel", `${clip.name}.mp4`));
    const poster = await readFile(path.join(docsRoot, "assets", "reel", `${clip.name}.webp`));
    reelBytes += mp4.length;
    if (sha256(mp4) !== clip.mp4) fail(`reel/${clip.name}.mp4: content hash differs from the reviewed clip`);
    if (sha256(poster) !== clip.poster) fail(`reel/${clip.name}.webp: content hash differs from the reviewed poster`);
    if (mp4.length > limits.clip) fail(`reel/${clip.name}.mp4 is ${mp4.length} bytes; clip limit is ${limits.clip}`);
    if (poster.length > limits.poster) fail(`reel/${clip.name}.webp is ${poster.length} bytes; poster limit is ${limits.poster}`);
    const boxes = mp4Layout(mp4);
    if (boxes[0] !== "ftyp") fail(`reel/${clip.name}.mp4: not an MP4 container`);
    if (boxes.indexOf("moov") < 0 || boxes.indexOf("moov") > boxes.indexOf("mdat")) fail(`reel/${clip.name}.mp4: moov must precede mdat (faststart)`);
    const size = webpDimensions(poster);
    if (!size || size.width !== clip.width || size.height !== clip.height) fail(`reel/${clip.name}.webp: expected ${clip.width}x${clip.height}`);
}
if (reelBytes > limits.reel) fail(`demo reel totals ${reelBytes} bytes; reel limit is ${limits.reel}`);

const homepage = await readFile(path.join(docsRoot, "index.html"), "utf8");
const videos = [...homepage.matchAll(/<video\b[^>]*>/gi)].map((match) => match[0]);
if (videos.length !== reel.length) fail(`index.html: expected ${reel.length} reel videos, found ${videos.length}`);
for (const clip of reel) {
    const video = videos.find((tag) => tag.includes(`data-src="assets/reel/${clip.name}.mp4"`));
    if (!video) { fail(`index.html: missing reel clip ${clip.name}`); continue; }
    // Clips start only when scrolled into view; nothing is fetched up front.
    for (const token of ["muted", "loop", "playsinline", 'preload="none"', `poster="assets/reel/${clip.name}.webp"`, `width="${clip.width}"`, `height="${clip.height}"`]) {
        if (!video.includes(token)) fail(`index.html: reel clip ${clip.name} is missing ${token}`);
    }
    if (/\ssrc=/.test(video) || /\sautoplay/.test(video)) fail(`index.html: reel clip ${clip.name} must not load or autoplay before it is visible`);
}
if (!homepage.includes('"screenshot": "https://infernux-engine.com/assets/reel/space-battle.webp"')) fail("index.html: structured evidence must use a delivered reel poster");
if (/assets\/demo(?:-runtime)?\./.test(homepage)) fail("index.html: retired showcase captures must not be delivered");
if (/<link\b[^>]*rel=["']preload["'][^>]*reel\//i.test(homepage)) fail("index.html: below-the-fold reel media must not be preloaded");
if (!homepage.includes("https://www.bilibili.com/video/BV1538P6jELT")) fail("index.html: the reel must link to its full source video");

const hud = await readFile(path.join(docsRoot, "js", "fx-hud.js"), "utf8");
for (const contract of ["data-reel", "dataset.src", "IntersectionObserver", "prefers-reduced-motion"]) {
    if (!hud.includes(contract)) fail(`fx-hud.js: reel playback is missing '${contract}'`);
}

const budget = await readFile(path.join(docsRoot, "tools", "check-static-budget.mjs"), "utf8");
if (!budget.includes("poster=")) fail("check-static-budget.mjs: reel posters must count toward the homepage first view");

// README loops live outside docs/ so the website never delivers them.
const readmeGifs = ["space-battle", "npr-pipeline", "rigid-coins", "animated-cats", "rendergraph-grid"];
const gifLimits = { hero: 900 * 1024, tile: 450 * 1024, total: 2200 * 1024 };
let gifBytes = 0;
for (const name of readmeGifs) {
    const gif = await readFile(path.join(repoRoot, ".github", "media", `${name}.gif`)).catch(() => null);
    if (!gif) { fail(`.github/media/${name}.gif: README loop is missing`); continue; }
    gifBytes += gif.length;
    if (!/^GIF8[79]a/.test(gif.subarray(0, 6).toString("ascii"))) fail(`.github/media/${name}.gif: not a GIF`);
    const limit = name === "space-battle" ? gifLimits.hero : gifLimits.tile;
    if (gif.length > limit) fail(`.github/media/${name}.gif is ${gif.length} bytes; limit is ${limit}`);
}
if (gifBytes > gifLimits.total) fail(`README loops total ${gifBytes} bytes; limit is ${gifLimits.total}`);
for (const readmeName of ["README.md", "README-zh.md"]) {
    const readme = await readFile(path.join(repoRoot, readmeName), "utf8");
    for (const name of readmeGifs) if (!readme.includes(`src=".github/media/${name}.gif"`)) fail(`${readmeName}: missing demo loop ${name}.gif`);
    if (readme.includes("docs/assets/demo")) fail(`${readmeName}: retired showcase captures must not be referenced`);
}

const provenance = await readFile(path.join(docsRoot, "assets", "VENDOR_ASSETS.md"), "utf8");
for (const contract of ["BV1538P6jELT", "faststart", ...reel.flatMap((clip) => [`reel/${clip.name}.mp4`, clip.mp4])]) {
    if (!provenance.includes(contract)) fail(`VENDOR_ASSETS.md: missing reviewed reel provenance '${contract}'`);
}

for (const workflow of ["website-quality.yml", "build-wiki.yml"]) {
    const source = await readFile(path.join(repoRoot, ".github", "workflows", workflow), "utf8");
    if (!source.includes("node docs/tools/check-image-variants.mjs")) fail(`${workflow}: evidence media gate is not part of the published workflow`);
}

if (failures.length) {
    console.error(`Evidence media audit failed with ${failures.length} issue(s):`);
    for (const failure of failures) console.error(`- ${failure}`);
    process.exit(1);
}

console.log(`Evidence media audit passed: ${reel.length} lazy reel clips (${(reelBytes / 1024).toFixed(1)} KiB, faststart H.264, WebP posters); ${readmeGifs.length} README loops (${(gifBytes / 1024).toFixed(1)} KiB) outside website delivery.`);
