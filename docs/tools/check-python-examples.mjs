import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// User-facing Python examples import the engine one way only:
//     import infernux as inx
// and reach everything through `inx` (inx.InxComponent, inx.jit, ...). This
// keeps copy-paste examples consistent and correct on case-sensitive systems.
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(docsRoot, "..");
const failures = [];

async function walk(directory, pattern) {
    const output = [];
    for (const entry of await readdir(directory, { withFileTypes: true })) {
        const target = path.join(directory, entry.name);
        if (entry.isDirectory()) output.push(...await walk(target, pattern));
        else if (pattern.test(entry.name)) output.push(target);
    }
    return output;
}

const decode = (html) => html.replace(/<[^>]+>/g, "")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, "&");

function markdownBlocks(source) {
    return [...source.matchAll(/```(?:python|py)\r?\n([\s\S]*?)```/g)].map((match) => match[1]);
}

function htmlBlocks(source) {
    return [...source.matchAll(/<pre\b[^>]*>\s*<code\b[^>]*>([\s\S]*?)<\/code>\s*<\/pre>/g)].map((match) => decode(match[1]));
}

const forbidden = /^\s*(?:from\s+[Ii]nfernux\b|import\s+Infernux\b|import\s+infernux(?!\s+as\s+inx\s*$)\S*)/;
let blocks = 0;
function inspect(relative, code) {
    blocks += 1;
    code.split(/\r?\n/).forEach((line, index) => {
        if (forbidden.test(line)) failures.push(`${relative}: example line ${index + 1} uses '${line.trim()}'; write 'import infernux as inx'`);
    });
}

const markdownFiles = [
    path.join(repoRoot, "README.md"),
    path.join(repoRoot, "README-zh.md"),
    ...await walk(path.join(docsRoot, "learn"), /\.md$/),
    ...await walk(path.join(docsRoot, "wiki", "docs"), /\.md$/),
];
for (const file of markdownFiles) {
    for (const code of markdownBlocks(await readFile(file, "utf8"))) inspect(path.relative(repoRoot, file), code);
}

const pageFiles = [
    ...(await readdir(docsRoot)).filter((name) => name.endsWith(".html")).map((name) => path.join(docsRoot, name)),
    ...await walk(path.join(docsRoot, "learn"), /\.html$/),
];
for (const file of pageFiles) {
    for (const code of htmlBlocks(await readFile(file, "utf8"))) {
        if (/\b(?:import|def|class)\b/.test(code)) inspect(path.relative(repoRoot, file), code);
    }
}

// The editor's "new script" template is the first example most users see.
const template = await readFile(path.join(repoRoot, "python", "infernux", "engine", "ui", "project_file_ops.py"), "utf8");
const scriptTemplate = template.match(/SCRIPT_TEMPLATE\s*=\s*'''([\s\S]*?)'''/);
if (!scriptTemplate) failures.push("project_file_ops.py: SCRIPT_TEMPLATE not found");
else {
    inspect("python/infernux/engine/ui/project_file_ops.py", scriptTemplate[1]);
    if (!/^import infernux as inx$/m.test(scriptTemplate[1])) failures.push("project_file_ops.py: new scripts must start from 'import infernux as inx'");
}

for (const workflow of ["website-quality.yml", "build-wiki.yml"]) {
    const source = await readFile(path.join(repoRoot, ".github", "workflows", workflow), "utf8");
    if (!source.includes("node docs/tools/check-python-examples.mjs")) failures.push(`${workflow}: Python example gate is not enforced`);
}

if (failures.length) {
    console.error(`Python example check failed with ${failures.length} issue(s):`);
    for (const failure of failures) console.error(`- ${failure}`);
    process.exit(1);
}

console.log(`Python example check passed: ${blocks} examples across READMEs, learning guides, API docs, site pages and the script template import the engine as 'import infernux as inx'.`);
