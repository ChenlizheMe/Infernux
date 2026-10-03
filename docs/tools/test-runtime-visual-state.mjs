import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const source = await readFile(path.join(scriptDir, "..", "js", "main.js"), "utf8");

function classList(initial = []) {
    const values = new Set(initial);
    return {
        add(...names) { names.forEach((name) => values.add(name)); },
        contains(name) { return values.has(name); },
        remove(...names) { names.forEach((name) => values.delete(name)); },
        toggle(name, force) {
            const add = force === undefined ? !values.has(name) : Boolean(force);
            if (add) values.add(name);
            else values.delete(name);
            return add;
        }
    };
}

function runScenario({ reduceMotion = false } = {}) {
    const documentListeners = new Map();
    const windowListeners = new Map();
    const navbar = { classList: classList() };
    const reveal = { classList: classList() };
    let observerInstance = null;

    class IntersectionObserver {
        constructor(callback) {
            this.callback = callback;
            this.observed = [];
            this.unobserved = [];
            observerInstance = this;
        }
        observe(element) { this.observed.push(element); }
        unobserve(element) { this.unobserved.push(element); }
    }

    function addListener(collection, type, handler) {
        if (!collection.has(type)) collection.set(type, []);
        collection.get(type).push(handler);
    }

    const documentElement = {
        lang: "en",
        attributes: new Map(),
        getAttribute(name) { return this.attributes.get(name) ?? null; },
        setAttribute(name, value) { this.attributes.set(name, value); }
    };
    const document = {
        activeElement: null,
        body: { classList: classList() },
        documentElement,
        addEventListener(type, handler) { addListener(documentListeners, type, handler); },
        dispatchEvent() {},
        getElementById() { return null; },
        querySelector(selector) {
            if (selector === ".navbar") return navbar;
            return null;
        },
        querySelectorAll(selector) {
            if (selector === "[data-reveal], .hero-slab, .subpage-hero, .hub-hero, .cta-panel") return [reveal];
            return [];
        }
    };
    const window = {
        IntersectionObserver,
        innerWidth: 1440,
        isSecureContext: false,
        pageYOffset: 0,
        scrollY: 0,
        addEventListener(type, handler) { addListener(windowListeners, type, handler); },
        matchMedia() { return { matches: reduceMotion }; },
        requestAnimationFrame(callback) { callback(); }
    };
    const sandbox = {
        CustomEvent: class {},
        IntersectionObserver,
        console,
        document,
        globalThis: null,
        localStorage: { getItem() { return null; }, setItem() {} },
        navigator: {},
        setTimeout() {},
        window
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    new vm.Script(source, { filename: "main.js" }).runInContext(sandbox);
    for (const handler of documentListeners.get("DOMContentLoaded") || []) handler();
    return { documentElement, documentListeners, navbar, observerInstance, reveal, sandbox, window, windowListeners };
}

const animated = runScenario();
assert.ok(animated.observerInstance, "motion-enabled pages should create one observer");
assert.deepEqual(animated.observerInstance.observed, [animated.reveal]);
assert.equal(animated.reveal.classList.contains("reveal-pending"), true);
assert.equal(animated.reveal.classList.contains("animate-in"), false);
animated.observerInstance.callback([{ isIntersecting: true, target: animated.reveal }]);
assert.equal(animated.reveal.classList.contains("reveal-pending"), false);
assert.equal(animated.reveal.classList.contains("animate-in"), true);
assert.deepEqual(animated.observerInstance.unobserved, [animated.reveal]);

animated.window.scrollY = 64;
for (const handler of animated.windowListeners.get("scroll") || []) handler();
assert.equal(animated.navbar.classList.contains("is-scrolled"), true);
animated.window.scrollY = 0;
for (const handler of animated.windowListeners.get("scroll") || []) handler();
assert.equal(animated.navbar.classList.contains("is-scrolled"), false);

const reduced = runScenario({ reduceMotion: true });
assert.equal(reduced.observerInstance, null);
assert.equal(reduced.reveal.classList.contains("reveal-pending"), false);
assert.equal(reduced.reveal.classList.contains("animate-in"), true);

assert.doesNotMatch(source, /\.style(?:\.|\[|\s*=)|#27ca40|function copyCode|function showTab/);
assert.doesNotMatch(source, /theme-toggle|toggleTheme|data-theme|color-scheme/);
console.log("Runtime visual-state test passed: scroll surface, reveal lifecycle, reduced motion, dark-only shell, and zero inline visual mutation.");
