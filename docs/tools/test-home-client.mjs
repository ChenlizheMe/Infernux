import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(scriptDir, "..");
const [source, html, css] = await Promise.all([
    readFile(path.join(docsRoot, "js", "home.js"), "utf8"),
    readFile(path.join(docsRoot, "index.html"), "utf8"),
    readFile(path.join(docsRoot, "css", "home.css"), "utf8")
]);

let copiedText = null;
const document = {
    documentElement: { lang: "en" },
    addEventListener() {},
    createElement() { throw new Error("clipboard fallback should not run when the Clipboard API succeeds"); }
};
const sandbox = {
    __INFERNUX_HOME_TEST__: true,
    document,
    navigator: { clipboard: { async writeText(value) { copiedText = value; } } },
    globalThis: null
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
new vm.Script(source, { filename: "home.js" }).runInContext(sandbox);

const api = sandbox.__infernuxHomeTest;
assert.ok(api, "home client must expose its isolated test surface");

const highlightedCode = {
    textContent: "from infernux import InxComponent\r\n\r\nclass Mover(InxComponent):\r\n    pass\r\n"
};
const button = {
    closest(selector) {
        assert.equal(selector, ".code-preview");
        return { querySelector(innerSelector) { assert.equal(innerSelector, "code"); return highlightedCode; } };
    }
};
const expected = "from infernux import InxComponent\n\nclass Mover(InxComponent):\n    pass\n";
assert.equal(api.extractStarterCode(button), expected, "copy must preserve exact code text, indentation, blank lines, and a final newline");
assert.equal(await api.copyHomeText(expected), true, "Clipboard API success should be reported");
assert.equal(copiedText, expected, "Clipboard API must receive the exact extracted source");
assert.equal(api.homeCopy("idle"), "Copy starter component");
assert.equal(api.homeCopy("success"), "Starter component copied");
document.documentElement.lang = "zh-CN";
assert.equal(api.homeCopy("idle"), "复制起步组件");
assert.equal(api.homeCopy("failure"), "无法复制起步组件，请手动选择代码。");

const starValue = { textContent: "—" };
const starCaption = { textContent: "stars in orbit" };
const starDeck = { dataset: {}, querySelector(selector) {
    if (selector === "[data-github-stars-value]") return starValue;
    if (selector === "[data-github-stars-caption]") return starCaption;
    return null;
} };
document.querySelector = (selector) => selector === "[data-github-stars]" ? starDeck : null;
sandbox.fetch = async (url) => {
    assert.equal(url, "https://api.github.com/repos/ChenlizheMe/Infernux", "home must request the engine repository's live star count");
    return { ok: true, async json() { return { stargazers_count: 1447 }; } };
};
await api.loadGithubStars();
assert.equal(starValue.textContent, "1,447", "the live GitHub signal must render the authoritative star count");
assert.equal(starDeck.dataset.githubStarsState, "ready");

for (const contract of [
    'href="css/home.css?v=3"',
    'data-home-code-copy',
    'type="button"',
    'aria-controls="home-starter-code"',
    'aria-describedby="home-code-copy-status"',
    'id="home-starter-code"',
    'id="home-code-copy-status"',
    'role="status"',
    'aria-live="polite"',
    'connect-src \'self\' https://api.github.com',
    'src="js/home.js?v=2"',
    'data-github-stars',
    'data-github-stars-value',
    'data-github-stars-caption'
]) assert.ok(html.includes(contract), `home page is missing '${contract}'`);
assert.equal((html.match(/id="home-starter-code"/g) || []).length, 1, "starter code id must be unique");
assert.equal((html.match(/id="home-code-copy-status"/g) || []).length, 1, "copy live-region id must be unique");
assert.doesNotMatch(html, /\son[a-z]+\s*=/i, "home copy must not rely on inline event handlers");
assert.doesNotMatch(html, /\sstyle\s*=/i, "home copy must not rely on inline styles");

assert.match(css, /\.code-copy-action\s*\{[\s\S]*?min-height:\s*44px;/, "starter copy must retain a 44px minimum touch target");
assert.match(css, /\.code-copy-action\[data-state="success"\]/, "copy success must have a visible state");
assert.match(css, /\.code-copy-action\[data-state="failure"\]/, "copy failure must have a visible state");
assert.match(css, /\.home-copy-fallback\s*\{/, "legacy copy fallback must be styled without inline declarations");
assert.match(css, /@media\s*\(max-width:\s*520px\)[\s\S]*?\.code-copy-action\s*\{[\s\S]*?width:\s*44px;[\s\S]*?\.code-copy-action span\s*\{[\s\S]*?clip:/, "phone layout must keep a compact 44px action while retaining its accessible label");
assert.match(source, /navigator\?\.clipboard\?\.writeText/, "home copy must prefer the Clipboard API");
assert.match(source, /document\.execCommand\("copy"\)/, "home copy must retain a legacy fallback");
assert.match(source, /dataset\.state/, "home copy must expose state to CSS");
assert.match(source, /stargazers_count/, "home must render GitHub's authoritative star count");
assert.doesNotMatch(source, /\.innerHTML\s*=/, "home copy must not inject HTML");
assert.doesNotMatch(source, /\.style\.|setAttribute\(["']style|\.cssText\s*=/, "home copy must not inject runtime styles");

console.log("Home starter-copy test passed: exact source extraction, bilingual feedback, safe clipboard fallback, 44px mobile action, and one live status.");
