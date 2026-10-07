import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const source = await readFile("docs/js/tutorials.js", "utf8");
const html = await readFile("docs/tutorials.html", "utf8");
const attributes = (tag) => Object.fromEntries(
    Array.from(tag.matchAll(/([\w-]+)="([^"]*)"/g), (match) => [match[1], match[2]])
);
const tabMarkup = Array.from(html.matchAll(/<button\b[^>]*data-tutorial-page="[^"]+"[^>]*>/g), (match) => attributes(match[0]));
const panelMarkup = Array.from(html.matchAll(/<section\b[^>]*data-tutorial-panel="[^"]+"[^>]*>/g), (match) => attributes(match[0]));
assert.deepEqual(tabMarkup.map((tab) => tab["data-tutorial-page"]), ["start", "learn"]);
assert.deepEqual(panelMarkup.map((panel) => panel["data-tutorial-panel"]), ["start", "learn"]);
for (const tab of tabMarkup) {
    assert.ok(panelMarkup.some((panel) => panel.id === tab["aria-controls"]), "each tab must control a shipped panel");
}

function open(hash, reducedMotion = false, motionAvailable = true) {
    const historyWrites = [];
    const animations = [];
    let focused = null;
    function node(attrs, dataset) {
        const classes = new Set((attrs.class || "").split(/\s+/));
        const handlers = new Map();
        return {
            attributes: { ...attrs }, dataset, hidden: false,
            classList: {
                contains(name) { return classes.has(name); },
                toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); }
            },
            setAttribute(name, value) { this.attributes[name] = value; },
            addEventListener(type, handler) { handlers.set(type, handler); },
            dispatch(type, event = {}) { handlers.get(type)?.(event); },
            focus() { focused = this; }
        };
    }
    const tabs = tabMarkup.map((attrs) => node(attrs, { tutorialPage: attrs["data-tutorial-page"] }));
    const panels = panelMarkup.map((attrs) => node(attrs, { tutorialPanel: attrs["data-tutorial-panel"] }));
    const windowHandlers = new Map();
    const window = {
        location: { hash },
        matchMedia() { return { matches: reducedMotion }; },
        addEventListener(type, handler) { windowHandlers.set(type, handler); }
    };
    const context = {
        document: {
            querySelector(selector) { return selector === "[data-tutorial-app]" ? { querySelectorAll() { return tabs; } } : null; },
            querySelectorAll(selector) { return selector === "[data-tutorial-panel]" ? panels : []; }
        },
        window,
        history: { replaceState(_state, _title, url) { historyWrites.push(url); window.location.hash = url; } }
    };
    if (motionAvailable) context.gsap = { fromTo(...args) { animations.push(args); } };
    vm.runInNewContext(source, context, { filename: "tutorials.js" });
    return {
        tabs, panels, historyWrites, animations,
        get focused() { return focused; },
        navigate(nextHash) { window.location.hash = nextHash; windowHandlers.get("hashchange")(); },
        assertSelected(name) {
            for (const tab of tabs) {
                assert.equal(tab.attributes["aria-selected"], String(tab.dataset.tutorialPage === name));
                assert.equal(tab.classList.contains("is-active"), tab.dataset.tutorialPage === name);
            }
            assert.deepEqual(panels.filter((panel) => !panel.hidden).map((panel) => panel.dataset.tutorialPanel), [name]);
        }
    };
}

for (const [hash, selected] of [["", "start"], ["#start", "start"], ["#learn", "learn"], ["#unknown", "start"]]) {
    const page = open(hash);
    page.assertSelected(selected);
    assert.deepEqual(page.historyWrites, [], "following a chapter link must preserve the incoming history entry");
    assert.equal(page.animations.length, 1);
    assert.equal(page.animations[0][0].dataset.tutorialPanel, selected);
}
const page = open("#learn");
page.tabs[0].dispatch("click");
page.assertSelected("start");
assert.deepEqual(page.historyWrites, ["#start"]);
page.navigate("#learn");
page.assertSelected("learn");
assert.deepEqual(page.historyWrites, ["#start"], "hash navigation must not write another history entry");

for (const [from, key, selected] of [[1, "ArrowRight", "start"], [0, "ArrowLeft", "learn"], [1, "Home", "start"], [0, "End", "learn"], [0, "ArrowDown", "learn"], [1, "ArrowUp", "start"]]) {
    let prevented = false;
    page.tabs[from].dispatch("keydown", { key, preventDefault() { prevented = true; } });
    assert.ok(prevented);
    page.assertSelected(selected);
    assert.equal(page.focused.dataset.tutorialPage, selected);
}
assert.equal(page.animations.at(-1)[0].dataset.tutorialPanel, "start", "animate the panel selected by the last keyboard action");
for (const [reducedMotion, motionAvailable] of [[true, true], [false, false]]) {
    const quiet = open("#learn", reducedMotion, motionAvailable);
    quiet.tabs[0].dispatch("click");
    quiet.assertSelected("start");
    assert.equal(quiet.animations.length, 0);
}
console.log("Tutorial tabs passed: shipped tab/panel bindings, incoming chapter links, hash navigation, keyboard focus, and reduced motion.");
