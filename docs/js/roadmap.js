/* Deterministic roadmap atlas: one visible freeform vector cluster, selectable nodes, and local pan/zoom. */
(function () {
    const app = document.querySelector("[data-roadmap-app]");
    if (!app) return;

    const tabs = Array.from(app.querySelectorAll("[data-tree-page]"));
    const pages = Array.from(app.querySelectorAll("[data-tree-panel]"));
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const aliases = { architecture: "foundation", rendering: "pipeline", gameplay: "runtime", neural: "agents" };
    const cameras = new WeakMap();

    function defaultCamera(shell) {
        if (!window.matchMedia("(max-width: 820px)").matches) return { x: 0, y: 0 };
        const svg = shell.querySelector("svg");
        if (!svg) return { x: -420, y: 0 };
        const rect = svg.getBoundingClientRect();
        if (!rect.width) return { x: 0, y: 0 };
        const scale = rect.width / 1600;
        return {
            x: shell.clientWidth / (2 * scale) - 800,
            y: shell.clientHeight / (2 * scale) - 600,
        };
    }

    function cameraFor(shell) {
        if (!cameras.has(shell)) {
            const initial = defaultCamera(shell);
            cameras.set(shell, { ...initial, scale: 1, dragging: false, moved: false, initialized: false });
        }
        return cameras.get(shell);
    }

    function applyCamera(shell) {
        const camera = cameraFor(shell);
        const group = shell.querySelector("[data-graph-camera]");
        if (group) group.setAttribute("transform", `translate(${camera.x} ${camera.y}) scale(${camera.scale})`);
    }

    function resetCamera(shell) {
        const camera = cameraFor(shell);
        const initial = defaultCamera(shell);
        camera.x = initial.x;
        camera.y = initial.y;
        camera.scale = 1;
        camera.initialized = true;
        applyCamera(shell);
    }

    function graphPoint(shell, event) {
        const svg = shell.querySelector("svg");
        const rect = svg.getBoundingClientRect();
        return {
            x: ((event.clientX - rect.left) / rect.width) * 1600,
            y: ((event.clientY - rect.top) / rect.height) * 1200,
        };
    }

    function pageFromHash() {
        const raw = window.location.hash.replace(/^#tree-/, "");
        const value = aliases[raw] || raw;
        return tabs.some((tab) => tab.dataset.treePage === value) ? value : "foundation";
    }

    function selectNode(node) {
        const panel = node.closest("[data-tree-panel]");
        if (!panel) return;
        panel.querySelectorAll(".graph-node").forEach((candidate) => {
            const selected = candidate === node;
            candidate.classList.toggle("is-selected", selected);
            candidate.setAttribute("aria-pressed", String(selected));
        });
        if (!reduceMotion && globalThis.gsap) {
            globalThis.gsap.fromTo(node, { scale: 0.96 }, { scale: 1, duration: 0.24, ease: "back.out(2)", overwrite: "auto" });
        }
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
        const activeShell = activePage?.querySelector("[data-graph-canvas]");
        if (activeShell) {
            const camera = cameraFor(activeShell);
            if (!camera.initialized) {
                const initial = defaultCamera(activeShell);
                camera.x = initial.x;
                camera.y = initial.y;
                camera.initialized = true;
            }
            applyCamera(activeShell);
        }
        if (!activePage || reduceMotion || !globalThis.gsap) return;
        globalThis.gsap.fromTo(activePage.querySelectorAll(".graph-panel-heading, .graph-edge, .graph-node"),
            { autoAlpha: 0, y: 12 },
            { autoAlpha: 1, y: 0, duration: 0.42, stagger: 0.012, ease: "power3.out", overwrite: "auto" });
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

    app.querySelectorAll(".graph-node").forEach((node) => {
        node.setAttribute("aria-pressed", "false");
        node.addEventListener("click", () => selectNode(node));
        node.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            selectNode(node);
        });
    });

    app.querySelectorAll("[data-graph-canvas]").forEach((shell) => {
        const camera = cameraFor(shell);
        shell.addEventListener("pointerdown", (event) => {
            if (event.target.closest(".graph-node")) return;
            camera.dragging = true;
            camera.moved = false;
            camera.startX = event.clientX;
            camera.startY = event.clientY;
            camera.originX = camera.x;
            camera.originY = camera.y;
            shell.classList.add("is-panning");
            shell.setPointerCapture(event.pointerId);
        });
        shell.addEventListener("pointermove", (event) => {
            if (!camera.dragging) return;
            const svg = shell.querySelector("svg");
            const rect = svg.getBoundingClientRect();
            const dx = (event.clientX - camera.startX) * 1600 / rect.width / camera.scale;
            const dy = (event.clientY - camera.startY) * 1200 / rect.height / camera.scale;
            camera.x = camera.originX + dx;
            camera.y = camera.originY + dy;
            camera.moved = Math.abs(dx) + Math.abs(dy) > 2;
            applyCamera(shell);
        });
        const endDrag = (event) => {
            if (!camera.dragging) return;
            camera.dragging = false;
            shell.classList.remove("is-panning");
            if (shell.hasPointerCapture(event.pointerId)) shell.releasePointerCapture(event.pointerId);
        };
        shell.addEventListener("pointerup", endDrag);
        shell.addEventListener("pointercancel", endDrag);
        shell.addEventListener("wheel", (event) => {
            event.preventDefault();
            const before = graphPoint(shell, event);
            const nextScale = Math.min(2.2, Math.max(0.65, camera.scale * (event.deltaY < 0 ? 1.1 : 0.9)));
            const ratio = nextScale / camera.scale;
            camera.x = before.x - (before.x - camera.x) * ratio;
            camera.y = before.y - (before.y - camera.y) * ratio;
            camera.scale = nextScale;
            applyCamera(shell);
        }, { passive: false });
    });

    app.querySelector("[data-graph-reset]")?.addEventListener("click", () => {
        const active = pages.find((page) => !page.hidden);
        const shell = active?.querySelector("[data-graph-canvas]");
        if (shell) resetCamera(shell);
    });

    window.addEventListener("hashchange", () => selectPage(pageFromHash(), false));
    selectPage(pageFromHash(), false);
}());
