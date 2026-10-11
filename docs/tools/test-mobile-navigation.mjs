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
        contains(value) { return values.has(value); },
        toggle(value, force) {
            const add = force === undefined ? !values.has(value) : Boolean(force);
            if (add) values.add(value);
            else values.delete(value);
            return add;
        }
    };
}

const listeners = new Map();
const documentElement = {
    lang: "en",
    attributes: new Map(),
    getAttribute(name) { return this.attributes.get(name) ?? null; },
    setAttribute(name, value) { this.attributes.set(name, value); }
};
const body = { classList: classList() };
const icon = { className: "fas fa-bars" };
const attributes = new Map();
const siteActionControls = [];
const document = {
    activeElement: null,
    body,
    documentElement,
    addEventListener(type, handler) { listeners.set(type, handler); },
    dispatchEvent() {},
    getElementById() { return null; },
    querySelector(selector) {
        if (selector === ".nav-links") return navLinks;
        if (selector === ".mobile-menu-btn") return button;
        return null;
    },
    querySelectorAll(selector) { return selector === "[data-site-action]" ? siteActionControls : []; }
};
function focusable(name) {
    return {
        name,
        focus() { document.activeElement = this; }
    };
}
const links = [focusable("start"), focusable("learn"), focusable("github")];
const navLinks = {
    classList: classList(["nav-links"]),
    querySelector(selector) { return selector === "a[href]" ? links[0] : null; },
    querySelectorAll(selector) { return selector === "a[href]" ? links : []; }
};
const button = {
    ...focusable("menu-button"),
    dataset: { siteAction: "menu" },
    eventHandlers: new Map(),
    addEventListener(type, handler) { this.eventHandlers.set(type, handler); },
    getAttribute(name) { return attributes.get(name) ?? null; },
    setAttribute(name, value) { attributes.set(name, value); },
    querySelector(selector) { return selector === "i" ? icon : null; }
};
function siteActionControl(action) {
    return {
        dataset: { siteAction: action },
        eventHandlers: new Map(),
        addEventListener(type, handler) { this.eventHandlers.set(type, handler); }
    };
}
const languageAction = siteActionControl("language");
siteActionControls.push(languageAction, button);
const window = {
    innerWidth: 375,
    isSecureContext: false,
    addEventListener() {},
    dispatchEvent() {},
    matchMedia() { return { matches: false }; },
    requestAnimationFrame(callback) { callback(); }
};
const sandbox = {
    __INFERNUX_NAV_TEST__: true,
    CustomEvent: class {},
    console,
    document,
    getComputedStyle() { return { getPropertyValue() { return ""; } }; },
    globalThis: null,
    localStorage: { getItem() { return null; }, setItem() {} },
    navigator: {},
    setTimeout() {},
    window
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
new vm.Script(source, { filename: "main.js" }).runInContext(sandbox);

const navigation = sandbox.__infernuxNavigation;
assert.ok(navigation, "mobile navigation test surface should be exported");

navigation.bindSiteActions();
assert.equal(languageAction.eventHandlers.has("click"), true, "language action should use an external event listener");
assert.equal(button.eventHandlers.has("click"), true, "menu action should use an external event listener");
button.eventHandlers.get("click")();
assert.equal(navLinks.classList.contains("mobile-open"), true);
navigation.setMobileMenuState(false);
navigation.bindSiteActions();
assert.equal(languageAction.dataset.siteActionBound, "true", "site actions should not be rebound");

navigation.setMobileMenuState(true, { moveFocus: true });
assert.equal(navLinks.classList.contains("mobile-open"), true);
assert.equal(body.classList.contains("mobile-menu-open"), true);
assert.equal(button.getAttribute("aria-expanded"), "true");
assert.equal(button.getAttribute("aria-label"), "Close navigation menu");
assert.equal(icon.className, "fas fa-xmark");
assert.equal(document.activeElement, links[0], "opening should move focus into the first task link");
const focusables = navigation.mobileMenuFocusables();
assert.equal(focusables.length, links.length + 1);
assert.equal(focusables[0], button);
for (const [index, link] of links.entries()) assert.equal(focusables[index + 1], link);

document.activeElement = links.at(-1);
let prevented = false;
navigation.handleMobileMenuKeydown({ key: "Tab", shiftKey: false, preventDefault() { prevented = true; } });
assert.equal(prevented, true);
assert.equal(document.activeElement, button, "Tab from the final link should wrap to the menu button");

prevented = false;
navigation.handleMobileMenuKeydown({ key: "Tab", shiftKey: true, preventDefault() { prevented = true; } });
assert.equal(prevented, true);
assert.equal(document.activeElement, links.at(-1), "Shift+Tab from the menu button should wrap to the final link");

navigation.handleMobileMenuKeydown({ key: "Escape", shiftKey: false, preventDefault() { prevented = true; } });
assert.equal(navLinks.classList.contains("mobile-open"), false);
assert.equal(body.classList.contains("mobile-menu-open"), false);
assert.equal(button.getAttribute("aria-expanded"), "false");
assert.equal(button.getAttribute("aria-label"), "Open navigation menu");
assert.equal(icon.className, "fas fa-bars");
assert.equal(document.activeElement, button, "Escape should return focus to the menu button");

navigation.setMobileMenuState(true, { moveFocus: true });
navigation.handleMobileMenuPointerDown({ target: { closest() { return null; } } });
assert.equal(navLinks.classList.contains("mobile-open"), false, "a pointer action outside the header should close the menu");

documentElement.lang = "zh-CN";
navigation.setMobileMenuState(true);
assert.equal(button.getAttribute("aria-label"), "关闭导航菜单");
navigation.setMobileMenuState(false);
assert.equal(button.getAttribute("aria-label"), "打开导航菜单");

window.innerWidth = 1440;
navigation.setMobileMenuState(true, { moveFocus: true });
assert.equal(navLinks.classList.contains("mobile-open"), false, "desktop layout must not open the mobile drawer");

console.log("Mobile navigation test passed: focus entry, tab loop, Escape return, outside close, localization, and desktop guard.");
