import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

const docsRoot = path.resolve("docs");
const repoRoot = path.resolve(docsRoot, "..");
const failures = [];

function fail(message) {
    failures.push(message);
}

async function recursiveFiles(directory) {
    const output = [];
    for (const entry of await readdir(directory, { withFileTypes: true })) {
        const target = path.join(directory, entry.name);
        if (entry.isDirectory()) output.push(...await recursiveFiles(target));
        else if (entry.isFile()) output.push(target);
    }
    return output;
}

function runtimeReference(sourceFile, reference) {
    const clean = reference.split("#", 1)[0].split("?", 1)[0];
    if (!clean || /^(?:[a-z]+:|\/\/)/i.test(clean)) return null;
    const target = clean.startsWith("/")
        ? path.join(docsRoot, clean.slice(1))
        : path.resolve(path.dirname(sourceFile), clean);
    const relative = path.relative(docsRoot, target);
    if (relative.startsWith("..") || path.isAbsolute(relative)) return null;
    return path.normalize(target);
}

const htmlFiles = (await recursiveFiles(docsRoot)).filter((file) => file.endsWith(".html"));
const referencedRuntime = new Set();
for (const htmlFile of htmlFiles) {
    const html = await readFile(htmlFile, "utf8");
    for (const match of html.matchAll(/\b(?:href|src)=["']([^"']+\.(?:css|js)(?:[?#][^"']*)?)["']/gi)) {
        const target = runtimeReference(htmlFile, match[1]);
        if (target) referencedRuntime.add(target);
    }
}

const runtimeFiles = [
    ...(await readdir(path.join(docsRoot, "css"))).filter((name) => name.endsWith(".css")).map((name) => path.join(docsRoot, "css", name)),
    ...(await readdir(path.join(docsRoot, "js"))).filter((name) => name.endsWith(".js")).map((name) => path.join(docsRoot, "js", name)),
].map(path.normalize);

for (const file of runtimeFiles.filter((runtimeFile) => runtimeFile.endsWith(".js"))) {
    const source = await readFile(file, "utf8");
    for (const match of source.matchAll(/["'`](\/(?:[^"'`]+\/)*[^"'`]+\.js(?:\?[^"'`]*)?)["'`]/g)) {
        const target = runtimeReference(file, match[1]);
        if (target) referencedRuntime.add(target);
    }
}

for (const file of runtimeFiles) {
    if (!referencedRuntime.has(file)) fail(`${path.relative(repoRoot, file)}: runtime is neither referenced by HTML nor declared as a same-origin lazy dependency`);
}

const forbiddenVisualMutation = [
    { pattern: /\.style(?:\.|\[|\s*=)/, label: "element.style mutation" },
    { pattern: /\.setAttribute\(\s*["']style["']/, label: "style attribute mutation" },
    { pattern: /\.cssText\b/, label: "cssText mutation" },
    { pattern: /\.(?:insertRule|setProperty)\s*\(/, label: "runtime CSS rule/property mutation" },
    { pattern: /createElement\(\s*["']style["']\s*\)/, label: "runtime style element creation" },
];
for (const file of runtimeFiles.filter((file) => file.endsWith(".js"))) {
    const source = await readFile(file, "utf8");
    for (const contract of forbiddenVisualMutation) {
        if (contract.pattern.test(source)) fail(`${path.relative(repoRoot, file)}: ${contract.label} bypasses class/data-state design tokens`);
    }
}

const main = await readFile(path.join(docsRoot, "js", "main.js"), "utf8");
for (const contract of ["classList.toggle('is-scrolled'", "classList.add('reveal-pending'", "classList.remove('reveal-pending'", "classList.add('animate-in'"]) {
    if (!main.includes(contract)) fail(`docs/js/main.js: missing class-driven visual state '${contract}'`);
}
for (const deadRuntime of ["function copyCode", "function showTab", "#27ca40", "site-update-notice", "monitorServiceWorkerUpdates"]) {
    if (main.includes(deadRuntime)) fail(`docs/js/main.js: obsolete runtime '${deadRuntime}' was restored`);
}

const sharedStyle = await readFile(path.join(docsRoot, "css", "style.css"), "utf8");
for (const contract of [".navbar.is-scrolled", ".reveal-pending", "transition: opacity 0.5s ease, transform 0.5s ease"]) {
    if (!sharedStyle.includes(contract)) fail(`docs/css/style.css: missing class-driven visual contract '${contract}'`);
}
const generatedStyle = await readFile(path.join(docsRoot, "css", "wiki-generated.css"), "utf8");
if (!generatedStyle.includes(".clipboard-fallback")) fail("docs/css/wiki-generated.css: missing clipboard fallback class");

for (const file of runtimeFiles.filter((runtimeFile) => runtimeFile.endsWith(".css"))) {
    const source = await readFile(file, "utf8");
    for (const match of source.matchAll(/(?:font-size\s*:\s*|font\s*:[^;]*?\s)(0\.\d+)(rem|px)/gi)) {
        const value = Number(match[1]);
        const minimum = match[2].toLowerCase() === "px" ? 8 : 0.5;
        if (value < minimum) {
            fail(`${path.relative(repoRoot, file)}: font size ${match[1]}${match[2]} is below the 8px minimum`);
        }
    }
}

for (const workflowName of ["website-quality.yml", "build-wiki.yml"]) {
    const workflow = await readFile(path.join(repoRoot, ".github", "workflows", workflowName), "utf8");
    if (!workflow.includes("node docs/tools/check-runtime-assets.mjs")) fail(`${workflowName}: runtime asset/state gate is not enforced`);
    if (!workflow.includes("node docs/tools/test-runtime-visual-state.mjs")) fail(`${workflowName}: runtime visual-state test is not enforced`);
}

if (failures.length) {
    console.error(`Runtime asset audit failed with ${failures.length} issue(s):`);
    for (const failure of failures) console.error(`- ${failure}`);
    process.exit(1);
}

console.log(`Runtime asset audit passed: ${runtimeFiles.length} CSS/JS assets are referenced or explicitly lazy-loaded, local scripts use class/data-state styling, and obsolete roadmap code stays removed.`);
