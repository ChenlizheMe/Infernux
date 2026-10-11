import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// README quality gate: both READMEs are the project's front door on GitHub
// and in the packaged wheel, so links must resolve, the two languages must
// stay structurally in sync, and the copy must keep its essential promises.
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..", "..");
const failures = [];
const fail = (message) => failures.push(message);
const exists = (relative) => stat(path.join(repoRoot, relative)).then(() => true, () => false);

// GitHub heading anchors: lowercase, punctuation dropped, spaces to hyphens.
function slug(heading) {
    return heading.trim().toLowerCase()
        .replace(/<[^>]+>/g, "")
        .replace(/[^\p{Letter}\p{Number}\s_-]/gu, "")
        .replace(/\s/g, "-");
}

async function anchorsOf(relative) {
    const source = await readFile(path.join(repoRoot, relative), "utf8");
    return new Set([...source.matchAll(/^#{1,6}\s+(.+)$/gm)].map((match) => slug(match[1])));
}

function references(markdown) {
    const refs = [];
    for (const match of markdown.matchAll(/\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)) refs.push(match[1]);
    for (const match of markdown.matchAll(/\b(?:href|src)="([^"]+)"/g)) refs.push(match[1]);
    return refs;
}

function structure(markdown) {
    const withoutCode = markdown.replace(/```[\s\S]*?```/g, "");
    return {
        headings: [...withoutCode.matchAll(/^(#{1,6})\s/gm)].map((match) => match[1].length).join(","),
        images: [...markdown.matchAll(/<img\b[^>]*\bsrc="([^"]+)"/g)].map((match) => match[1]).join("|"),
        codeBlocks: (markdown.match(/```[a-z]*\n/g) || []).length,
        tables: (withoutCode.match(/^\|\s*-{3,}/gm) || []).length,
    };
}

const readmes = { en: "README.md", zh: "README-zh.md" };
const sources = {};
for (const [language, relative] of Object.entries(readmes)) {
    const source = (await readFile(path.join(repoRoot, relative), "utf8")).replace(/\r\n/g, "\n");
    sources[language] = source;

    for (const ref of references(source)) {
        if (/^(?:https?:|mailto:)/i.test(ref)) {
            if (/^http:\/\//i.test(ref)) fail(`${relative}: use https for ${ref}`);
            continue;
        }
        const [target, anchor] = ref.split("#");
        if (!target) {
            if (anchor && !(await anchorsOf(relative)).has(anchor)) fail(`${relative}: in-page anchor #${anchor} has no heading`);
            continue;
        }
        const resolved = path.normalize(target);
        if (!(await exists(resolved))) { fail(`${relative}: broken relative reference ${ref}`); continue; }
        if (anchor && resolved.endsWith(".md") && !(await anchorsOf(resolved)).has(anchor)) {
            fail(`${relative}: ${ref} points to a missing heading`);
        }
    }

    for (const match of source.matchAll(/<img\b[^>]*>/g)) {
        if (!/\balt="[^"]+"/.test(match[0])) fail(`${relative}: image without alternative text: ${match[0].slice(0, 80)}`);
    }
    if (/docs\/assets\/demo/.test(source)) fail(`${relative}: references a retired showcase capture`);
    if (!/^## /m.test(source)) fail(`${relative}: has no sections`);
    if (!source.includes("[!WARNING]")) fail(`${relative}: must keep the alpha-status warning`);
}

if (!sources.en.includes('href="README-zh.md"')) fail("README.md: missing the 简体中文 language switch");
if (!sources.zh.includes('href="README.md"')) fail("README-zh.md: missing the English language switch");
if (!sources.en.includes("Neural Network-Native")) fail("README.md: must define 3N as Neural Network-Native");
if (!sources.zh.includes("神经网络原生")) fail("README-zh.md: must define 3N as 神经网络原生");

// scripts/release/sync_release_site.py bumps releases by plain string replacement,
// so each README must carry the current version literally or it silently goes stale.
const pyproject = await readFile(path.join(repoRoot, "pyproject.toml"), "utf8");
const version = pyproject.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
if (!version) fail("pyproject.toml: project version not found");
else for (const [language, relative] of Object.entries(readmes)) {
    if (!sources[language].includes(`version-${version}-`)) fail(`${relative}: version badge does not show ${version}`);
}

const en = structure(sources.en);
const zh = structure(sources.zh);
for (const key of Object.keys(en)) {
    if (en[key] !== zh[key]) fail(`README parity: ${key} differ between README.md (${en[key]}) and README-zh.md (${zh[key]})`);
}

for (const workflow of ["website-quality.yml", "build-wiki.yml"]) {
    const source = await readFile(path.join(repoRoot, ".github", "workflows", workflow), "utf8");
    if (!source.includes("node docs/tools/check-readme.mjs")) fail(`${workflow}: README quality gate is not enforced`);
    if (!source.includes('"README.md"') || !source.includes('"README-zh.md"')) fail(`${workflow}: README changes do not trigger the workflow`);
}

if (failures.length) {
    console.error(`README check failed with ${failures.length} issue(s):`);
    for (const failure of failures) console.error(`- ${failure}`);
    process.exit(1);
}

console.log(`README check passed: links, anchors and images resolve; ${en.headings.split(",").length} headings, ${en.codeBlocks} code blocks and ${en.tables} tables match across English and Chinese.`);
