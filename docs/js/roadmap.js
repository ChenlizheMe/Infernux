/* Tape-futurist SVG galaxy map. SVG keeps labels crisp; GSAP carries the signal motion. */
(function () {
    const app = document.querySelector("[data-roadmap-app]");
    if (!app) return;

    const tabs = Array.from(app.querySelectorAll("[data-tree-page]"));
    const pages = Array.from(app.querySelectorAll("[data-tree-panel]"));
    const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
    let reduceMotion = motionPreference.matches;
    const aliases = { architecture: "foundation", rendering: "pipeline", animation: "runtime", physics: "agents" };
    const cameras = new WeakMap();
    const animations = new Map();
    const gsap = globalThis.gsap;
    const layouts = new WeakMap();
    const nodeData = new WeakMap();
    const dataRequests = new Map();
    let selectionEpoch = 0;
    const recentPanels = [];
    const search = app.querySelector('[data-roadmap-search]');
    const results = app.querySelector('[data-roadmap-results]');
    const detail = app.querySelector('[data-roadmap-detail]');
    const t = key => translateSiteKey(key);
    const label = item => item.key ? t(item.key) : item[document.documentElement.lang.startsWith('zh') ? 'zh' : 'en'];
    const statusLabel = status => t(`roadmap.graph.${status}`);
    const activePanel = () => pages.find(page => !page.hidden);

    // Targets and displayed values share the graph SVG's world coordinates.
    // Input only changes targets; one animation frame owns camera writes.
    function cameraFor(shell) {
        if (!cameras.has(shell)) cameras.set(shell, {
            x: 0, y: 0, scale: 1, shownX: 0, shownY: 0, shownScale: 1,
            dragging: false, frame: 0, time: 0, geometry: null,
            setTransform: gsap.quickSetter(shell.querySelector('[data-graph-plane]'), 'css')
        });
        return cameras.get(shell);
    }

    function geometryFor(shell) {
        const camera = cameraFor(shell);
        if (!camera.geometry) {
            const view = layouts.get(shell.closest('[data-tree-panel]'));
            const rect = shell.getBoundingClientRect();
            const width = shell.clientWidth, height = shell.clientHeight;
            const base = Math.min(width / view.width, height / view.height);
            const offsetX = (width - view.width * base) / 2, offsetY = (height - view.height * base) / 2;
            camera.geometry = { base, offsetX, offsetY, left: rect.left + shell.clientLeft + offsetX,
                top: rect.top + shell.clientTop + offsetY };
        }
        return camera.geometry;
    }

    function applyCamera(shell, smooth = false) {
        const camera = cameraFor(shell);
        camera.smooth = smooth && !reduceMotion;
        if (camera.frame) return;
        shell.classList.add("is-camera-moving");
        camera.time = performance.now();
        camera.frame = requestAnimationFrame(time => drawCamera(shell, time));
    }

    function drawCamera(shell, time) {
        const camera = cameraFor(shell);
        camera.frame = 0;
        if (shell.closest('[data-tree-panel]').hidden || document.hidden) return;
        const elapsed = Math.min(64, Math.max(1, time - camera.time));
        camera.time = time;
        const amount = camera.smooth ? 1 - Math.exp(-elapsed / (camera.dragging ? 28 : 65)) : 1;
        camera.shownX += (camera.x - camera.shownX) * amount;
        camera.shownY += (camera.y - camera.shownY) * amount;
        camera.shownScale += (camera.scale - camera.shownScale) * amount;
        const base = geometryFor(shell).base;
        const moving = Math.abs(camera.x - camera.shownX) * base > .1 || Math.abs(camera.y - camera.shownY) * base > .1 || Math.abs(camera.scale - camera.shownScale) > .0001;
        if (!moving) {
            camera.shownX = camera.x; camera.shownY = camera.y; camera.shownScale = camera.scale;
        }
        const geometry = geometryFor(shell);
        const x = camera.shownX * base + geometry.offsetX * (1 - camera.shownScale);
        const y = camera.shownY * base + geometry.offsetY * (1 - camera.shownScale);
        camera.setTransform({ x, y, scale: camera.shownScale, force3D: true });
        layouts.get(shell.closest('[data-tree-panel]')).edgeGroup.setAttribute('stroke-width', 1 / camera.shownScale);
        updateDetailLevel(shell);
        if (moving) camera.frame = requestAnimationFrame(next => drawCamera(shell, next));
        else shell.classList.remove("is-camera-moving");
    }

    function stopCamera(shell) {
        const camera = cameraFor(shell);
        cancelAnimationFrame(camera.frame);
        camera.frame = 0;
        camera.x = camera.shownX; camera.y = camera.shownY; camera.scale = camera.shownScale;
        camera.dragging = false;
        shell.classList.remove('is-panning', 'is-camera-moving');
    }

    function measureLabels(panel) {
        const view = layouts.get(panel);
        // Batch every read before any SVG writes; never measure during gestures.
        for (const entry of view.labels) entry.box = entry.text.getBBox();
        view.labelScale = null;
    }

    function updateDetailLevel(shell) {
        const view = layouts.get(shell.closest('[data-tree-panel]'));
        if (!view) return;
        const pixelsPerUnit = geometryFor(shell).base * cameraFor(shell).shownScale;
        if (view.labelScale === pixelsPerUnit) return;
        view.labelScale = pixelsPerUnit;
        shell.classList.toggle('map-detail-visible', pixelsPerUnit * 21 >= 10);
        const occupied = [];
        for (const { node, item, text, box } of view.labels) {
            const factor = Math.max(1, Math.min(6, (item.type === 'root' ? 12 / 30 : 9 / 25) / pixelsPerUnit));
            const y = item.radius + 27;
            text.setAttribute('transform', `translate(0 ${y * (1 - factor)}) scale(${factor})`);
            const padding = 3 / pixelsPerUnit;
            const bounds = { x: item.x + box.x * factor - padding, y: item.y + y + (box.y - y) * factor - padding,
                width: box.width * factor + padding * 2, height: box.height * factor + padding * 2 };
            const overlaps = factor > 1 && occupied.some(other => bounds.x < other.x + other.width && bounds.x + bounds.width > other.x && bounds.y < other.y + other.height && bounds.y + bounds.height > other.y);
            node.classList.toggle('is-label-muted', overlaps);
            if (!overlaps) occupied.push(bounds);
        }
    }

    function pageFromHash() {
        const raw = window.location.hash.replace(/^#tree-/, "");
        const value = aliases[raw] || raw;
        return tabs.some((tab) => tab.dataset.treePage === value) ? value : tabs[0]?.dataset.treePage;
    }

    function svgElement(tag, attributes) {
        const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
        Object.entries(attributes || {}).forEach(([name, value]) => element.setAttribute(name, String(value)));
        return element;
    }

    // Two-tone instrument symbols, not shaded miniature planets.
    function celestialBody(kind, id, radius) {
        const art = svgElement("g", { class: "celestial-art celestial-" + kind, transform: `scale(${radius / 24})`, "aria-hidden": "true" });
        const shape = (tag, attributes, parent = art) => {
            const element = svgElement(tag, attributes);
            parent.append(element);
            return element;
        };
        const asteroid = "M-22-8 -10-23 12-19 24-3 16 19 -6 24 -24 9Z";
        const defs = shape("defs", {});
        const clip = shape("clipPath", { id: "planet-" + id }, defs);
        shape(kind === "asteroid" ? "path" : "circle", kind === "asteroid" ? { d: asteroid } : { r: 24 }, clip);
        if (kind === "ringed") {
            shape("ellipse", { class: "planet-ring-back", rx: 38, ry: 10, transform: "rotate(-25)" });
        }
        if (kind === "star") {
            const rays = shape("g", { class: "planet-rays" });
            for (let i = 0; i < 4; i++) shape("path", { d: "M0-32V-39", transform: `rotate(${i * 90})` }, rays);
        }
        const surface = shape("g", { "clip-path": `url(#planet-${id})` });
        shape("circle", { class: "planet-base", r: 36 }, surface);
        if (kind === "gas") {
            [-11, 0, 11].forEach(y => shape("rect", { class: "planet-cut", x: -28, y, width: 56, height: y === 0 ? 5 : 2 }, surface));
        } else if (kind === "rock") {
            shape("circle", { class: "planet-cut", cx: -7, cy: -5, r: 6 }, surface);
            shape("path", { class: "planet-etch", d: "M6 11H17" }, surface);
        } else if (kind === "ice") {
            shape("path", { class: "planet-etch", d: "M-7-26 3-5-6 5 7 26M3-5 21-11" }, surface);
        } else if (kind === "asteroid") {
            shape("circle", { class: "planet-cut", cx: -7, cy: -4, r: 5 }, surface);
            shape("circle", { class: "planet-cut", cx: 9, cy: 8, r: 3 }, surface);
        } else if (kind === "star") {
            shape("circle", { class: "planet-cut", r: 10 }, surface);
            shape("rect", { class: "planet-cut", x: -26, y: -1, width: 52, height: 2 }, surface);
        }
        shape('circle', { class: 'planet-veil', r: 36 }, surface);
        if (kind === "ringed") {
            shape("path", { class: "planet-ring-front", d: "M-38 0A38 10 0 0 0 38 0", transform: "rotate(-25)" });
        }
        return art;
    }

    function localPoint(shell, event) {
        const geometry = geometryFor(shell);
        return { x: (event.clientX - geometry.left) / geometry.base, y: (event.clientY - geometry.top) / geometry.base };
    }

    function deterministicRandom(seed) {
        let state = seed >>> 0;
        return () => {
            state = (1664525 * state + 1013904223) >>> 0;
            return state / 4294967296;
        };
    }

    function loadGalaxy(panel) {
        const entry = GALAXIES.find(galaxy => galaxy.key === panel.dataset.galaxy);
        if (!dataRequests.has(entry.key)) {
            dataRequests.set(entry.key, fetch(entry.file).then(response => {
                if (!response.ok) throw new Error(`Roadmap data HTTP ${response.status}`);
                return response.json();
            }).catch(error => { dataRequests.delete(entry.key); throw error; }));
        }
        return dataRequests.get(entry.key);
    }

    function renderGalaxy(panel, data) {
        if (!panel || panel.dataset.rendered === "true") return;
        const svg = panel.querySelector(".node-graph");
        const starfield = panel.querySelector("[data-starfield]");
        const cameraGroup = panel.querySelector("[data-graph-camera]");
        cameraFor(panel.querySelector("[data-graph-canvas]")).geometry = null;
        if (!data || !svg || !starfield || !cameraGroup) return;

        const galaxyIndex = GALAXIES.findIndex(galaxy => galaxy.key === data.key);
        const random = deterministicRandom(0x1f4a + galaxyIndex * 977);
        const view = layoutRoadmap(data);
        view.labels = [];
        view.adjacentEdges = new Map();
        view.selected = null;
        layouts.set(panel, view);
        svg.setAttribute("viewBox", `0 0 ${view.width} ${view.height}`);
        svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
        starfield.replaceChildren();

        const addStar = (x, y, radius, className) => {
            starfield.append(svgElement("circle", { class: "galaxy-star " + (className || ""), cx: x.toFixed(1), cy: y.toFixed(1), r: radius.toFixed(1) }));
        };
        for (let index = 0; index < 240; index += 1) {
            const x = random() * 1600;
            const y = random() * 1000;
            const bright = index % 31 === 0;
            addStar(x, y, bright ? 1.7 : .4 + random() * .8, bright ? 'star-bright' : '');
            if (bright) starfield.append(svgElement('path', {
                class: 'star-cross', d: `M${x - 5} ${y}h10M${x} ${y - 5}v10`
            }));
        }

        cameraGroup.replaceChildren();
        svg.querySelector('[data-planet-symbols]')?.remove();
        const symbols = svgElement('defs', { 'data-planet-symbols': '' });
        for (const kind of ['star', 'gas', 'ringed', 'rock', 'ice', 'asteroid']) {
            const symbol = celestialBody(kind, `${data.key}-${kind}`, 24);
            symbol.setAttribute('id', `body-${data.key}-${kind}`);
            symbol.removeAttribute('class');
            symbols.append(symbol);
        }
        svg.insertBefore(symbols, cameraGroup);
        view.edgeGroup = svgElement('g', { class: 'graph-connections', 'stroke-width': 1 });
        cameraGroup.append(view.edgeGroup);
        const addLine = (from, to, status) => {
            const edge = svgElement("path", {
                class: "graph-edge state-" + status,
                "data-from": from.id, "data-to": to.id,
                d: `M${from.x} ${from.y}L${to.x} ${to.y}`
            });
            view.edgeGroup.append(edge);
            for (const id of [from.id, to.id]) {
                if (!view.adjacentEdges.has(id)) view.adjacentEdges.set(id, []);
                view.adjacentEdges.get(id).push(edge);
            }
        };
        const addNode = (item) => {
            const { type, id, status, x, y, radius, kind } = item;
            const node = svgElement("g", { class: "graph-node graph-" + type + " state-" + status, transform: `translate(${x} ${y})`, tabindex: 0, role: "button", "data-node-id": id, "aria-label": label(item) });
            nodeData.set(node, item);
            node.append(svgElement("circle", { class: "graph-hit", r: Math.max(28, radius * 1.7) }));
            node.append(svgElement("circle", { class: "graph-halo", r: radius * 1.85 }));
            if (status === 'partial') node.append(svgElement('circle', {
                class: 'graph-completion', r: radius * 1.85, pathLength: 100,
                'stroke-dasharray': `${item.completion * 100} 100`, transform: 'rotate(-90)',
                'aria-hidden': 'true'
            }));
            const art = svgElement('g', { class: `celestial-art celestial-${kind}`, transform: `scale(${radius / 24})`, 'aria-hidden': 'true' });
            art.append(svgElement('use', { href: `#body-${data.key}-${kind}` }));
            node.append(art);
            const text = svgElement("text", {
                class: type === "leaf" ? "graph-leaf-label" : "graph-label" + (type === "root" ? " graph-root-label" : ""),
                x: 0,
                y: radius + 27,
                "text-anchor": 'middle'
            });
            node.append(text);
            updateNodeLabel(node, item);
            cameraGroup.append(node);
            node.setAttribute('aria-pressed', 'false');
            if (type !== 'leaf') view.labels.push({ node, item, text });
        };

        const byId = new Map(view.nodes.map(node => [node.id, node]));
        view.edges.forEach(edge => addLine(byId.get(edge.from), byId.get(edge.to), edge.status));
        view.nodes.forEach(addNode);
        panel.dataset.rendered = "true";
    }

    function updateNodeLabel(node, item) {
        const text = node.querySelector('text');
        text.replaceChildren();
        roadmapLabelLines(label(item)).forEach((line, index) => {
            const span = svgElement('tspan', { x: 0, dy: index ? '1.25em' : 0 });
            span.textContent = line;
            text.append(span);
        });
        node.setAttribute('aria-label', `${label(item)} · ${statusLabel(item.status)}`);
    }

    function animateGalaxy(panel) {
        if (reduceMotion) return;
        const animation = gsap.context(() => {
            // One entrance for the graph; no per-node or per-label tween fanout.
            gsap.fromTo('[data-graph-camera]', { opacity: 0 }, { opacity: 1, duration: .45, ease: 'power2.out' });
            const progressing = panel.querySelectorAll('.state-progress .celestial-art');
            if (progressing.length) gsap.fromTo(progressing, { opacity: .45 }, { opacity: 1, duration: 2.6, repeat: -1, yoyo: true, ease: 'sine.inOut' });
        }, panel);
        animations.set(panel, animation);
        syncAnimationVisibility();
    }

    let mapInView = true;
    function syncAnimationVisibility() {
        for (const [panel, animation] of animations) {
            for (const tween of animation.getTweens()) tween.paused(document.hidden || !mapInView || panel.hidden);
        }
    }
    new IntersectionObserver(entries => {
        mapInView = entries[0].isIntersecting;
        syncAnimationVisibility();
    }).observe(app);
    document.addEventListener('visibilitychange', () => {
        syncAnimationVisibility();
        if (document.hidden) pages.forEach(page => stopCamera(page.querySelector('[data-graph-canvas]')));
    });
    motionPreference.addEventListener('change', event => {
        reduceMotion = event.matches;
        for (const animation of animations.values()) animation.revert();
        animations.clear();
        if (layouts.has(activePanel())) {
            applyCamera(activePanel().querySelector('[data-graph-canvas]'));
            animateGalaxy(activePanel());
        }
    });
    function refreshLabelMetrics() {
        // Hidden SVG text has no usable metrics; measure it when selected.
        const panel = activePanel();
        if (layouts.has(panel)) {
            measureLabels(panel);
            updateDetailLevel(panel.querySelector('[data-graph-canvas]'));
        }
    }
    document.fonts.ready.then(refreshLabelMetrics);
    const sizeObserver = new ResizeObserver(entries => entries.forEach(({ target: shell }) => {
        cameraFor(shell).geometry = null;
        if (layouts.has(shell.closest('[data-tree-panel]')) && !shell.closest('[data-tree-panel]').hidden) applyCamera(shell);
    }));
    app.querySelectorAll('[data-graph-canvas]').forEach(shell => sizeObserver.observe(shell));
    window.addEventListener('scroll', () => {
        pages.forEach(page => { cameraFor(page.querySelector('[data-graph-canvas]')).geometry = null; });
    }, { passive: true, capture: true });

    function selectNode(node) {
        const panel = node.closest("[data-tree-panel]");
        if (!panel) return;
        const layout = layouts.get(panel);
        const previous = layout.selected;
        if (previous) {
            previous.classList.remove('is-selected'); previous.setAttribute('aria-pressed', 'false');
            layout.adjacentEdges.get(nodeData.get(previous).id)?.forEach(edge => edge.classList.remove('is-related'));
        }
        layout.selected = node;
        node.classList.add('is-selected'); node.setAttribute('aria-pressed', 'true');
        const item = nodeData.get(node);
        layout.adjacentEdges.get(item.id)?.forEach(edge => edge.classList.add('is-related'));
        const parent = layout.nodes.find(candidate => candidate.id === item.parent);
        detail.textContent = [label(layout.nodes[0]), parent && label(parent), item.type !== 'root' && label(item), statusLabel(item.status)].filter(Boolean).join(' / ');
        if (item.type === 'branch') focusNode(node);
        if (item.type === 'root') fitMap();
    }

    function focusNode(node) {
        const panel = node.closest('[data-tree-panel]');
        const shell = panel.querySelector('[data-graph-canvas]');
        const view = layouts.get(panel);
        const item = nodeData.get(node);
        const baseScale = geometryFor(shell).base;
        const bounds = item.subtreeBounds;
        const scale = bounds ? Math.min(32, .84 * Math.min(view.width / (bounds.right - bounds.left), view.height / (bounds.bottom - bounds.top)))
            : Math.min(32, Math.max(1.8, 15 / (21 * baseScale)));
        const x = bounds ? (bounds.left + bounds.right) / 2 : item.x;
        const y = bounds ? (bounds.top + bounds.bottom) / 2 : item.y + item.radius + 38;
        Object.assign(cameraFor(shell), { scale, x: view.width / 2 - x * scale, y: view.height / 2 - y * scale });
        applyCamera(shell, true);
    }

    function fitMap() {
        const shell = activePanel().querySelector('[data-graph-canvas]');
        Object.assign(cameraFor(shell), { x: 0, y: 0, scale: 1 });
        applyCamera(shell, true);
    }

    function zoomMap(shell, factor, point) {
        const camera = cameraFor(shell);
        if (!layouts.has(shell.closest('[data-tree-panel]'))) return;
        const scale = Math.min(32, Math.max(.65, camera.scale * factor));
        const ratio = scale / camera.scale;
        camera.x = point.x - (point.x - camera.x) * ratio;
        camera.y = point.y - (point.y - camera.y) * ratio;
        camera.scale = scale;
        applyCamera(shell, true);
    }

    function updateSummary() {
        const view = layouts.get(activePanel());
        if (!view) return;
        const counts = { live: 0, partial: 0, progress: 0, planned: 0, future: 0 };
        view.nodes.forEach(node => counts[node.status]++);
        app.querySelector('[data-roadmap-count]').textContent = `${view.nodes.length} ${t('roadmap.graph.nodes')} · ` + Object.entries(counts).map(([status, count]) => `${statusLabel(status)} ${count}`).join(' / ');
    }

    function updateSearch() {
        const query = search.value.trim().toLocaleLowerCase();
        const panel = activePanel();
        const nodes = Array.from(panel.querySelectorAll('.graph-node'));
        const matches = nodes.filter(node => {
            const item = nodeData.get(node);
            return [label(item), item.en, item.zh].some(value => value?.toLocaleLowerCase().includes(query));
        });
        panel.classList.toggle('is-searching', Boolean(query));
        nodes.forEach(node => node.classList.toggle('is-match', Boolean(query) && matches.includes(node)));
        results.replaceChildren();
        results.hidden = !query;
        if (!query) return;
        if (!matches.length) { results.textContent = t('roadmap.graph.empty'); return; }
        matches.slice(0, 12).forEach(node => {
            const item = nodeData.get(node);
            const button = document.createElement('button');
            button.type = 'button';
            button.textContent = `${label(item)} · ${statusLabel(item.status)}`;
            button.addEventListener('click', () => {
                selectNode(node);
                if (item.type !== 'root') focusNode(node);
                search.value = '';
                updateSearch();
                node.focus({ preventScroll: true });
            });
            results.append(button);
        });
    }

    search.addEventListener('input', updateSearch);
    search.addEventListener('keydown', event => {
        if (event.key === 'Escape') { search.value = ''; updateSearch(); }
        if (event.key === 'Enter') results.querySelector('button')?.click();
    });
    app.querySelectorAll('[data-map-zoom]').forEach(button => button.addEventListener('click', () => {
        if (button.dataset.mapZoom === 'fit') { fitMap(); return; }
        const panel = activePanel();
        const view = layouts.get(panel);
        zoomMap(panel.querySelector('[data-graph-canvas]'), button.dataset.mapZoom === 'in' ? 1.35 : 1 / 1.35, { x: view.width / 2, y: view.height / 2 });
    }));

    async function selectPage(name, updateHistory) {
        const epoch = ++selectionEpoch;
        const selected = tabs.find((tab) => tab.dataset.treePage === name) || tabs[0];
        if (!selected) return;
        const pageName = selected.dataset.treePage;
        tabs.forEach((tab) => {
            const active = tab === selected;
            tab.classList.toggle("is-active", active);
            tab.setAttribute("aria-selected", String(active));
            tab.tabIndex = active ? 0 : -1;
        });
        pages.forEach((page) => {
            animations.get(page)?.revert();
            animations.delete(page);
            stopCamera(page.querySelector("[data-graph-canvas]"));
            const active = page.dataset.treePanel === pageName;
            page.classList.toggle("is-active", active);
            page.hidden = !active;
        });
        if (updateHistory) history.replaceState(null, "", "#tree-" + pageName);
        const activePage = pages.find((page) => page.dataset.treePanel === pageName);
        const activeShell = activePage?.querySelector("[data-graph-canvas]");
        if (!activeShell) return;
        search.value = '';
        search.disabled = true;
        results.hidden = true;
        app.querySelectorAll('[data-map-zoom]').forEach(button => { button.disabled = true; });
        activePage.setAttribute('aria-busy', 'true');
        app.querySelector('[data-roadmap-count]').textContent = t('roadmap.graph.loading');
        detail.textContent = t('roadmap.graph.loading');
        try {
            const data = await loadGalaxy(activePage);
            if (epoch !== selectionEpoch) return;
            renderGalaxy(activePage, data);
            const previous = recentPanels.indexOf(activePage);
            if (previous >= 0) recentPanels.splice(previous, 1);
            recentPanels.push(activePage);
            // Bound SVG memory after touring the catalog; JSON remains cached.
            if (recentPanels.length > 2) {
                const outgoing = recentPanels.shift();
                outgoing.querySelector('[data-graph-camera]').replaceChildren();
                outgoing.querySelector('[data-starfield]').replaceChildren();
                outgoing.querySelector('[data-planet-symbols]')?.remove();
                delete outgoing.dataset.rendered;
                layouts.delete(outgoing);
            }
        } catch (error) {
            if (epoch !== selectionEpoch) return;
            activePage.setAttribute('aria-busy', 'false');
            app.querySelector('[data-roadmap-count]').textContent = t('roadmap.graph.loadError');
            detail.textContent = t('roadmap.graph.loadError');
            return;
        }
        activePage.setAttribute('aria-busy', 'false');
        search.disabled = false;
        app.querySelectorAll('[data-map-zoom]').forEach(button => { button.disabled = false; });
        cameraFor(activeShell).geometry = null;
        measureLabels(activePage);
        applyCamera(activeShell);
        animateGalaxy(activePage);
        updateSearch();
        updateSummary();
        detail.textContent = t('roadmap.graph.select');
    }

    tabs.forEach((tab, index) => {
        tab.addEventListener("click", () => selectPage(tab.dataset.treePage, true));
        tab.addEventListener("keydown", (event) => {
            const direction = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1
                : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1
                    : event.key === "Home" ? -index
                        : event.key === "End" ? tabs.length - 1 - index : 0;
            if (!direction) return;
            event.preventDefault();
            const next = tabs[(index + direction + tabs.length) % tabs.length];
            next.focus();
            selectPage(next.dataset.treePage, true);
        });
    });

    app.querySelectorAll('[data-graph-canvas]').forEach(shell => {
        const camera = cameraFor(shell);
        shell.addEventListener('pointerdown', event => {
            if (!event.isPrimary || event.button !== 0 || !layouts.has(shell.closest('[data-tree-panel]'))) return;
            stopCamera(shell);
            camera.geometry = null;
            const point = localPoint(shell, event);
            camera.dragging = true;
            camera.pointer = event.pointerId;
            camera.startPoint = point;
            camera.startClient = { x: event.clientX, y: event.clientY };
            camera.originX = camera.x; camera.originY = camera.y;
            camera.pressedNode = event.target.closest('.graph-node');
            camera.moved = false;
            shell.setPointerCapture(event.pointerId);
        });
        shell.addEventListener('pointermove', event => {
            if (!camera.dragging || event.pointerId !== camera.pointer) return;
            const point = localPoint(shell, event);
            if (!camera.moved && Math.hypot(event.clientX - camera.startClient.x, event.clientY - camera.startClient.y) < 4) return;
            camera.moved = true;
            shell.classList.add('is-panning');
            camera.x = camera.originX + point.x - camera.startPoint.x;
            camera.y = camera.originY + point.y - camera.startPoint.y;
            applyCamera(shell, true);
        });
        const endDrag = event => {
            if (!camera.dragging || event.pointerId !== camera.pointer) return;
            camera.dragging = false;
            shell.classList.remove('is-panning');
            if (shell.hasPointerCapture(event.pointerId)) shell.releasePointerCapture(event.pointerId);
            if (event.type === 'pointerup' && !camera.moved && camera.pressedNode) selectNode(camera.pressedNode);
            camera.pressedNode = null;
        };
        shell.addEventListener('pointerup', endDrag);
        shell.addEventListener('pointercancel', endDrag);
        shell.addEventListener('lostpointercapture', endDrag);
        shell.addEventListener('wheel', event => {
            if (!layouts.has(shell.closest('[data-tree-panel]')) || camera.dragging) return;
            event.preventDefault();
            const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? shell.clientHeight : 1);
            // Continuous deltas preserve trackpad precision and wheel acceleration.
            zoomMap(shell, Math.exp(-Math.max(-240, Math.min(240, pixels)) * .0025), localPoint(shell, event));
        }, { passive: false });
        shell.addEventListener('click', event => {
            const node = event.target.closest('.graph-node');
            if (node && event.detail === 0) selectNode(node);
        });
        shell.addEventListener('keydown', event => {
            const node = event.target.closest('.graph-node');
            if (node && (event.key === 'Enter' || event.key === ' ')) {
                event.preventDefault(); selectNode(node); return;
            }
            if (event.target !== shell || !layouts.has(shell.closest('[data-tree-panel]'))) return;
            const directions = { ArrowLeft: [80, 0], ArrowRight: [-80, 0], ArrowUp: [0, 80], ArrowDown: [0, -80] };
            if (directions[event.key]) {
                event.preventDefault();
                const base = geometryFor(shell).base;
                camera.x += directions[event.key][0] / base; camera.y += directions[event.key][1] / base;
                applyCamera(shell, true);
            } else if (event.key === '0') fitMap();
        });
    });

    window.addEventListener("hashchange", () => selectPage(pageFromHash(), false));
    document.addEventListener('site:language-changed', () => {
        if (activePanel().getAttribute('aria-busy') === 'true' || !layouts.has(activePanel())) {
            const key = activePanel().getAttribute('aria-busy') === 'true' ? 'loading' : 'loadError';
            detail.textContent = t(`roadmap.graph.${key}`);
            app.querySelector('[data-roadmap-count]').textContent = detail.textContent;
            return;
        }
        app.querySelectorAll('.graph-node').forEach(node => {
            updateNodeLabel(node, nodeData.get(node));
        });
        refreshLabelMetrics();
        updateSummary();
        updateSearch();
        detail.textContent = t('roadmap.graph.select');
    });
    selectPage(pageFromHash(), false);
}());
