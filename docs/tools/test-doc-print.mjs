import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(scriptDir, "..");
const runtime = await readFile(path.join(docsRoot, "js", "wiki-generated.js"), "utf8");
const styles = await readFile(path.join(docsRoot, "css", "wiki-generated.css"), "utf8");
const sharedStyles = await readFile(path.join(docsRoot, "css", "style.css"), "utf8");
const template = await readFile(path.join(docsRoot, "wiki", "theme", "main.html"), "utf8");

assert.match(runtime, /print\.addEventListener\("click", function \(\) \{\s*window\.print\(\);\s*\}\);/, "printing must require an explicit user click");
assert.equal((runtime.match(/window\.print\(\)/g) || []).length, 1, "documentation runtime should have one consent-gated print call");
for (const contract of ["Print / Save PDF", "打印 / 保存 PDF", "print.dataset.docPrint", "fas fa-file-lines"]) {
    assert.ok(runtime.includes(contract), `print action is missing '${contract}'`);
}

for (const contract of [
    "@media print",
    "@page",
    ".site-header",
    ".api-sidebar",
    ".docs-search-dialog",
    ".doc-trail",
    ".doc-code-copy",
    "break-after: avoid-page",
    "break-inside: avoid-page",
    "white-space: pre-wrap",
    "display: table-header-group",
    "overflow-wrap: anywhere",
    ".doc-build-provenance",
    ".doc-diagram",
    ".learning-track"
]) {
    assert.ok(styles.includes(contract), `print stylesheet is missing '${contract}'`);
}
for (const contract of ["--print-paper", "--print-ink", "--print-rule", "--print-grid"]) {
    assert.ok(sharedStyles.includes(contract), `shared style tokens are missing '${contract}'`);
}
for (const contract of ["var(--print-paper)", "var(--print-ink)", "var(--print-rule)", "var(--print-grid)"]) {
    assert.ok(styles.includes(contract), `print stylesheet is missing semantic token '${contract}'`);
}
assert.match(styles, /\.site-header,[\s\S]*?\.doc-code-copy,[\s\S]*?display:\s*none\s*!important;/, "interactive chrome must be excluded from printed documentation");
assert.match(styles, /\.api-main table\s*\{[\s\S]*?display:\s*table\s*!important;[\s\S]*?width:\s*100%\s*!important;/, "printed tables should use page-width table layout");
assert.ok(template.includes('/css/wiki-generated.css?v=10'));
assert.ok(template.includes('/js/wiki-generated.js?v=15'));
assert.ok(template.includes('/css/style.css?v=27'));

async function htmlFiles(directory) {
    const files = [];
    for (const entry of await readdir(directory, { withFileTypes: true })) {
        const target = path.join(directory, entry.name);
        if (entry.isDirectory()) files.push(...await htmlFiles(target));
        else if (entry.isFile() && entry.name.endsWith(".html")) files.push(target);
    }
    return files;
}

const generatedPages = await htmlFiles(path.join(docsRoot, "wiki", "site"));
assert.ok(generatedPages.length > 100, "expected a complete generated documentation site");
for (const page of generatedPages) {
    const html = await readFile(page, "utf8");
    assert.ok(html.includes('/css/wiki-generated.css?v=10'), `${path.relative(docsRoot, page)} is missing print stylesheet v9`);
    assert.ok(html.includes('/js/wiki-generated.js?v=15'), `${path.relative(docsRoot, page)} is missing print action runtime v15`);
}

console.log(`Documentation print checks passed: explicit action, print-only layout, readable code/tables, and ${generatedPages.length} generated pages.`);
