/* Interactive tree selector for the roadmap. Every tree is authored in HTML;
   this file only changes which subtree is visible and adds deterministic motion. */
(function () {
    const app = document.querySelector("[data-roadmap-app]");
    if (!app) return;

    const tabs = Array.from(app.querySelectorAll("[data-tree-page]"));
    const pages = Array.from(app.querySelectorAll("[data-tree-panel]"));
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function pageFromHash() {
        const value = window.location.hash.replace(/^#tree-/, "");
        return tabs.some((tab) => tab.dataset.treePage === value) ? value : "architecture";
    }

    function selectPage(name, updateHistory = true) {
        const selected = tabs.find((tab) => tab.dataset.treePage === name) || tabs[0];
        const pageName = selected.dataset.treePage;
        tabs.forEach((tab) => {
            const active = tab === selected;
            tab.classList.toggle("is-active", active);
            tab.setAttribute("aria-selected", String(active));
        });
        pages.forEach((page) => {
            const active = page.dataset.treePanel === pageName;
            page.classList.toggle("is-active", active);
            page.hidden = !active;
        });
        if (updateHistory) history.replaceState(null, "", `#tree-${pageName}`);
        const activePage = pages.find((page) => page.dataset.treePanel === pageName);
        if (!activePage || reduceMotion || !globalThis.gsap) return;
        globalThis.gsap.fromTo(activePage.querySelectorAll(".tree-page-heading, .tree-branch"),
            { autoAlpha: 0, y: 18 },
            { autoAlpha: 1, y: 0, duration: 0.48, stagger: 0.045, ease: "power3.out", overwrite: "auto" });
    }

    tabs.forEach((tab, index) => {
        tab.addEventListener("click", () => selectPage(tab.dataset.treePage));
        tab.addEventListener("keydown", (event) => {
            const direction = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1
                : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1
                    : event.key === "Home" ? -index
                        : event.key === "End" ? tabs.length - 1 - index : 0;
            if (!direction) return;
            event.preventDefault();
            const next = tabs[(index + direction + tabs.length) % tabs.length];
            next.focus();
            selectPage(next.dataset.treePage);
        });
    });
    window.addEventListener("hashchange", () => selectPage(pageFromHash(), false));
    selectPage(pageFromHash(), false);
}());
