/* Tutorial hub tabs: one navigation entry for the quick start and full courses. */
(function () {
    const app = document.querySelector("[data-tutorial-app]");
    if (!app) return;
    const tabs = Array.from(app.querySelectorAll("[data-tutorial-page]"));
    const panels = Array.from(document.querySelectorAll("[data-tutorial-panel]"));
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function pageFromHash() {
        const value = window.location.hash.replace(/^#/, "");
        return tabs.some((tab) => tab.dataset.tutorialPage === value) ? value : "start";
    }

    function selectPage(name, updateHistory = true) {
        const selected = tabs.find((tab) => tab.dataset.tutorialPage === name) || tabs[0];
        const pageName = selected.dataset.tutorialPage;
        tabs.forEach((tab) => {
            const active = tab === selected;
            tab.classList.toggle("is-active", active);
            tab.setAttribute("aria-selected", String(active));
        });
        panels.forEach((panel) => {
            const active = panel.dataset.tutorialPanel === pageName;
            panel.classList.toggle("is-active", active);
            panel.hidden = !active;
        });
        if (updateHistory) history.replaceState(null, "", `#${pageName}`);
        const activePanel = panels.find((panel) => panel.dataset.tutorialPanel === pageName);
        if (!activePanel || reduceMotion || !globalThis.gsap) return;
        globalThis.gsap.fromTo(activePanel, { autoAlpha: 0, y: 12 }, { autoAlpha: 1, y: 0, duration: 0.42, ease: "power3.out", overwrite: "auto" });
    }

    tabs.forEach((tab, index) => {
        tab.addEventListener("click", () => selectPage(tab.dataset.tutorialPage));
        tab.addEventListener("keydown", (event) => {
            const direction = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1
                : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1
                    : event.key === "Home" ? -index
                        : event.key === "End" ? tabs.length - 1 - index : 0;
            if (!direction) return;
            event.preventDefault();
            const next = tabs[(index + direction + tabs.length) % tabs.length];
            next.focus();
            selectPage(next.dataset.tutorialPage);
        });
    });
    window.addEventListener("hashchange", () => selectPage(pageFromHash(), false));
    selectPage(pageFromHash(), false);
}());
