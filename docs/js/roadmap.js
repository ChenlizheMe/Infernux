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
        // While the camera moves only the composited plane transform changes:
        // no SVG attribute writes, so the GPU scales the cached raster. Stroke
        // compensation, label scaling and the HUD are applied once it settles.
        if (moving) {
            if (time - (camera.hudTime || 0) > 120) { camera.hudTime = time; updateHud(shell); }
            camera.frame = requestAnimationFrame(next => drawCamera(shell, next));
            return;
        }
        // A held drag keeps the plane composited until the pointer is released.
        if (camera.dragging) return;
        settleCamera(shell);
    }

    function settleCamera(shell) {
        const camera = cameraFor(shell);
        const view = layouts.get(shell.closest('[data-tree-panel]'));
        if (view && view.strokeScale !== camera.shownScale) {
            view.strokeScale = camera.shownScale;
            view.edgeGroup.setAttribute('stroke-width', (1 / camera.shownScale).toFixed(4));
            view.plotGroup?.setAttribute('stroke-width', (1 / camera.shownScale).toFixed(4));
            // Dash patterns are rescaled in CSS by power-of-two zoom bucket.
            const bucket = String(Math.min(32, Math.max(1, 2 ** Math.round(Math.log2(Math.max(1, camera.shownScale))))));
            if (shell.dataset.zoom !== bucket) shell.dataset.zoom = bucket;
        }
        updateDetailLevel(shell);
        updateHud(shell);
        shell.classList.remove("is-camera-moving");
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

    // #node-<map>-<branch>-<leaf> deep-links straight to a node on its map.
    function routeFromHash() {
        const match = window.location.hash.match(/^#node-(([a-z]+)(?:-\d+){0,2})$/);
        const panel = match && pages.find(page => page.dataset.galaxy === match[2]);
        return panel ? { page: panel.dataset.treePanel, focusId: match[1] } : { page: pageFromHash() };
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

    // Two-tone instrument symbols, not shaded miniature planets. Every cut is
    // pre-clipped geometry: no clipPath, so each <use> instance stays a plain
    // vector draw without an offscreen mask.
    function chordBand(y, height, r = 24) {
        const y2 = y + height;
        const x1 = Math.sqrt(r * r - y * y).toFixed(2), x2 = Math.sqrt(r * r - y2 * y2).toFixed(2);
        return `M${-x1} ${y}L${x1} ${y}A${r} ${r} 0 0 1 ${x2} ${y2}L${-x2} ${y2}A${r} ${r} 0 0 1 ${-x1} ${y}Z`;
    }

    function celestialBody(kind, id, radius) {
        const art = svgElement("g", { class: "celestial-art celestial-" + kind, transform: `scale(${radius / 24})`, "aria-hidden": "true" });
        const shape = (tag, attributes, parent = art) => {
            const element = svgElement(tag, attributes);
            parent.append(element);
            return element;
        };
        const asteroid = "M-22-8 -10-23 12-19 24-3 16 19 -6 24 -24 9Z";
        const body = kind === "asteroid" ? { d: asteroid } : { r: 24 };
        const bodyTag = kind === "asteroid" ? "path" : "circle";
        if (kind === "ringed") shape("ellipse", { class: "planet-ring-back", rx: 38, ry: 10, transform: "rotate(-25)" });
        if (kind === "star") {
            const rays = shape("g", { class: "planet-rays" });
            for (let i = 0; i < 4; i++) shape("path", { d: "M0-32V-39", transform: `rotate(${i * 90})` }, rays);
        }
        shape(bodyTag, { class: "planet-base", ...body });
        if (kind === "gas") {
            [[-11, 2], [0, 5], [11, 2]].forEach(([y, h]) => shape("path", { class: "planet-cut", d: chordBand(y, h) }));
        } else if (kind === "rock") {
            shape("circle", { class: "planet-cut", cx: -7, cy: -5, r: 6 });
            shape("path", { class: "planet-etch", d: "M6 11H17" });
        } else if (kind === "ice") {
            shape("path", { class: "planet-etch", d: "M-6-22 3-5-6 5 6 22M3-5 20-10" });
        } else if (kind === "asteroid") {
            shape("circle", { class: "planet-cut", cx: -7, cy: -4, r: 5 });
            shape("circle", { class: "planet-cut", cx: 9, cy: 8, r: 3 });
        } else if (kind === "star") {
            shape("circle", { class: "planet-cut", r: 10 });
            shape("path", { class: "planet-cut", d: chordBand(-1, 2) });
        }
        shape(bodyTag, { class: "planet-veil", ...body });
        if (kind === "ringed") shape("path", { class: "planet-ring-front", d: "M-38 0A38 10 0 0 0 38 0", transform: "rotate(-25)" });
        return art;
    }

    function localPoint(shell, event) {
        const geometry = geometryFor(shell);
        return { x: (event.clientX - geometry.left) / geometry.base, y: (event.clientY - geometry.top) / geometry.base };
    }

    // Mission plot: range rings, bearing scale and axes around the single
    // center. Static world-space geometry drawn once per map, below edges.
    function plotGraticule(view) {
        const group = svgElement('g', { class: 'plot-grid', 'aria-hidden': 'true', 'stroke-width': 1 });
        const cx = view.width / 2;
        const cy = view.height / 2;
        const reach = Math.max(...view.nodes.map(node => Math.hypot(node.x - cx, node.y - cy) + node.radius * 2));
        const outer = reach * 1.06;
        group.append(svgElement('path', { class: 'plot-axis', d: `M${cx - outer * 1.12} ${cy}H${cx + outer * 1.12}M${cx} ${cy - outer * 1.12}V${cy + outer * 1.12}` }));
        for (let ring = 1; ring <= 4; ring += 1) {
            const radius = (outer * ring) / 4;
            group.append(svgElement('circle', { class: `plot-ring${ring % 2 === 0 ? ' major' : ''}`, cx, cy, r: radius.toFixed(1) }));
            const label = svgElement('text', { class: 'plot-label', x: (cx + radius * 0.7071 + 8).toFixed(1), y: (cy - radius * 0.7071 - 8).toFixed(1) });
            label.textContent = `R·${String(Math.round(radius)).padStart(4, '0')}`;
            group.append(label);
        }
        let minor = '';
        let major = '';
        for (let degree = 0; degree < 360; degree += 5) {
            const angle = (degree - 90) * Math.PI / 180;
            const long = degree % 30 === 0;
            const inner = outer + (long ? 0 : 6);
            const tip = outer + (long ? 26 : 14);
            const segment = `M${(cx + Math.cos(angle) * inner).toFixed(1)} ${(cy + Math.sin(angle) * inner).toFixed(1)}L${(cx + Math.cos(angle) * tip).toFixed(1)} ${(cy + Math.sin(angle) * tip).toFixed(1)}`;
            if (long) {
                major += segment;
                const label = svgElement('text', {
                    class: 'plot-bearing', 'text-anchor': 'middle', 'dominant-baseline': 'middle',
                    x: (cx + Math.cos(angle) * (outer + 50)).toFixed(1), y: (cy + Math.sin(angle) * (outer + 50)).toFixed(1)
                });
                label.textContent = String(degree).padStart(3, '0');
                group.append(label);
            } else minor += segment;
        }
        group.append(svgElement('path', { class: 'plot-tick', d: minor }));
        group.append(svgElement('path', { class: 'plot-tick major', d: major }));
        group.append(svgElement('circle', { class: 'plot-ring outer', cx, cy, r: outer.toFixed(1) }));
        return group;
    }

    function nodeCode(item) {
        const [key, ...indices] = item.id.split('-');
        return [key.slice(0, 3).toUpperCase(), ...indices.map(index => String(Number(index) + 1).padStart(2, '0'))].join('-');
    }

    function renderDetail(item, parent, root) {
        detail.replaceChildren();
        const part = (tag, className, text) => {
            const element = document.createElement(tag);
            element.className = className;
            if (text !== undefined) element.textContent = text;
            detail.append(element);
            return element;
        };
        part('span', `detail-led state-${item.status}`).setAttribute('aria-hidden', 'true');
        part('span', 'detail-code', nodeCode(item));
        const path = part('span', 'detail-path');
        [root, parent, item.type !== 'root' ? item : null].filter(Boolean).forEach((entry, index, list) => {
            const crumb = document.createElement(index === list.length - 1 ? 'strong' : 'span');
            crumb.textContent = label(entry);
            path.append(crumb);
            if (index < list.length - 1) path.append(' / ');
        });
        part('span', `detail-status state-${item.status}`, statusLabel(item.status));
        if (item.type !== 'leaf') {
            const share = Math.round((item.completion || 0) * 100);
            const meter = part('span', 'detail-meter');
            meter.setAttribute('aria-hidden', 'true');
            const fill = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            fill.setAttribute('viewBox', '0 0 100 6');
            fill.setAttribute('preserveAspectRatio', 'none');
            fill.append(svgElement('rect', { class: 'detail-meter-track', width: 100, height: 6 }), svgElement('rect', { class: 'detail-meter-fill', width: share, height: 6 }));
            meter.append(fill);
            part('span', 'detail-share', `${share}%`);
        }
    }

    function updateHud(shell) {
        const hud = shell.querySelector('.map-hud');
        if (!hud) return;
        const camera = cameraFor(shell);
        const view = layouts.get(shell.closest('[data-tree-panel]'));
        if (!view) return;
        const scale = camera.shownScale;
        const centerX = (view.width / 2 - camera.shownX) / scale - view.width / 2;
        const centerY = (view.height / 2 - camera.shownY) / scale - view.height / 2;
        const sign = value => (value < 0 ? '−' : '+') + String(Math.abs(Math.round(value))).padStart(4, '0');
        hud.querySelector('[data-hud-scale]').textContent = `${scale.toFixed(2)}×`;
        hud.querySelector('[data-hud-x]').textContent = sign(centerX);
        hud.querySelector('[data-hud-y]').textContent = sign(-centerY);
    }

    function deterministicRandom(seed) {
        let state = seed >>> 0;
        return () => {
            state = (1664525 * state + 1013904223) >>> 0;
            return state / 4294967296;
        };
    }

    function loadEntry(entry) {
        if (!dataRequests.has(entry.key)) {
            dataRequests.set(entry.key, fetch(entry.file).then(response => {
                if (!response.ok) throw new Error(`Roadmap data HTTP ${response.status}`);
                return response.json();
            }).catch(error => { dataRequests.delete(entry.key); throw error; }));
        }
        return dataRequests.get(entry.key);
    }

    function loadGalaxy(panel) {
        return loadEntry(GALAXIES.find(galaxy => galaxy.key === panel.dataset.galaxy));
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
        for (let index = 0; index < 150; index += 1) {
            const x = random() * 1600;
            const y = random() * 1000;
            const bright = index % 31 === 0;
            addStar(x, y, bright ? 1.5 : .35 + random() * .6, bright ? 'star-bright' : '');
            if (bright) starfield.append(svgElement('path', {
                class: 'star-cross', d: `M${x - 5} ${y}h10M${x} ${y - 5}v10`
            }));
        }

        cameraGroup.replaceChildren();
        view.plotGroup = plotGraticule(view);
        cameraGroup.append(view.plotGroup);
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
            // The entrance fade is CSS (.is-plotting); GSAP only pulses active work.
            const progressing = panel.querySelectorAll('.state-progress .celestial-art');
            if (progressing.length) gsap.fromTo(progressing, { opacity: .45 }, { opacity: 1, duration: 2.6, repeat: -1, yoyo: true, ease: 'sine.inOut' });
        }, panel);
        animations.set(panel, animation);
        syncAnimationVisibility();
    }

    let mapInView = true;
    function syncAnimationVisibility() {
        app.classList.toggle('is-map-visible', mapInView && !document.hidden);
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
        renderDetail(item, parent, layout.nodes[0]);
        const hudSelection = panel.querySelector('[data-hud-sel]');
        if (hudSelection) hudSelection.textContent = nodeCode(item);
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
        const telemetry = app.querySelector('[data-roadmap-telemetry]');
        if (!telemetry) return;
        telemetry.replaceChildren();
        const cell = (className, value, caption, share) => {
            const element = document.createElement('div');
            element.className = `telemetry-cell ${className}`;
            const number = document.createElement('b');
            number.textContent = value;
            const text = document.createElement('span');
            text.textContent = caption;
            element.append(number, text);
            if (share !== undefined) {
                const bar = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                bar.setAttribute('viewBox', '0 0 100 4');
                bar.setAttribute('preserveAspectRatio', 'none');
                bar.append(svgElement('rect', { class: 'telemetry-track', width: 100, height: 4 }), svgElement('rect', { class: 'telemetry-fill', width: (share * 100).toFixed(1), height: 4 }));
                element.append(bar);
            }
            telemetry.append(element);
        };
        cell('total', String(view.nodes.length).padStart(3, '0'), t('roadmap.graph.nodes'));
        Object.entries(counts).forEach(([status, count]) => cell(`state-${status}`, String(count).padStart(3, '0'), statusLabel(status), count / view.nodes.length));
    }

    /* ---------------------------------------------------- cross-map search
       All ten maps are indexed lazily the first time the search box is used
       (their JSON is cached and reused by the map views). The current map is
       highlighted in place; results from every map are grouped, current first,
       and a result jumps straight to its map and node. */
    const searchIndex = new Map();
    let searchIndexPromise = null;
    const mapKeyOf = panel => panel?.dataset.galaxy;
    const panelForMap = key => pages.find(page => page.dataset.galaxy === key);
    const tabForPanel = panel => tabs.find(tab => tab.dataset.treePage === panel?.dataset.treePanel);
    const normalize = value => String(value || '').toLocaleLowerCase().replace(/\s+/g, ' ').trim();

    function indexGalaxy(entry, data) {
        if (searchIndex.has(entry.key)) return;
        const rows = [{ map: entry.key, id: entry.key, type: 'root', key: entry.titleKey, status: roadmapStatusOf(data.branches.flatMap(branch => branch.leaves)) }];
        data.branches.forEach((branch, index) => {
            const hub = { map: entry.key, id: `${entry.key}-${index}`, type: 'branch', en: branch.en, zh: branch.zh, status: roadmapStatusOf(branch.leaves), parent: rows[0] };
            rows.push(hub);
            branch.leaves.forEach(([en, zh, status], leafIndex) => rows.push({ map: entry.key, id: `${hub.id}-${leafIndex}`, type: 'leaf', en, zh, status, parent: hub }));
        });
        searchIndex.set(entry.key, rows);
    }

    function ensureSearchIndex() {
        if (!searchIndexPromise) {
            searchIndexPromise = Promise.allSettled(GALAXIES.map(entry => loadEntry(entry).then(data => indexGalaxy(entry, data))))
                .then(() => { if (search.value.trim()) updateSearch(); });
        }
        return searchIndexPromise;
    }

    function scoreRow(row, query) {
        const names = row.type === 'root' ? [t(row.key), row.key] : [row.en, row.zh];
        let best = 0;
        for (const name of names.map(normalize)) {
            if (!name) continue;
            if (name === query) best = Math.max(best, 4);
            else if (name.startsWith(query)) best = Math.max(best, 3);
            else if (name.split(/[\s/&·-]+/).some(word => word.startsWith(query))) best = Math.max(best, 2);
            else if (name.includes(query)) best = Math.max(best, 1);
        }
        if (normalize(nodeCode(row)).startsWith(query)) best = Math.max(best, 3);
        return best ? best + (row.type === 'root' ? 0.6 : row.type === 'branch' ? 0.3 : 0) : 0;
    }

    function searchMatches(query) {
        const groups = [];
        for (const entry of GALAXIES) {
            const rows = searchIndex.get(entry.key);
            if (!rows) continue;
            const hits = rows.map(row => ({ row, score: scoreRow(row, query) })).filter(hit => hit.score > 0)
                .sort((a, b) => b.score - a.score);
            if (hits.length) groups.push({ entry, hits });
        }
        const current = app.dataset.galaxy;
        return groups.sort((a, b) => (b.entry.key === current) - (a.entry.key === current) || b.hits[0].score - a.hits[0].score);
    }

    function resultButton(row, className = '') {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `search-result state-${row.status} ${className}`.trim();
        const code = document.createElement('span');
        code.className = 'result-code';
        code.textContent = nodeCode(row);
        const name = document.createElement('span');
        name.className = 'result-name';
        name.textContent = label(row);
        const path = document.createElement('span');
        path.className = 'result-path';
        path.textContent = row.type === 'root' ? t('roadmap.search.openMap') : label(row.parent);
        const status = document.createElement('span');
        status.className = 'result-status';
        status.textContent = statusLabel(row.status);
        button.append(code, name, path, status);
        button.addEventListener('click', () => goToResult(row));
        return button;
    }

    function renderResults(query, groups) {
        results.replaceChildren();
        const total = groups.reduce((sum, group) => sum + group.hits.length, 0);
        const complete = searchIndex.size === GALAXIES.length;
        if (!total) {
            const empty = document.createElement('p');
            empty.className = 'search-empty';
            empty.textContent = complete ? t('roadmap.graph.empty') : t('roadmap.search.loading');
            results.append(empty);
            return;
        }
        const current = app.dataset.galaxy;
        for (const { entry, hits } of groups) {
            const isCurrent = entry.key === current;
            const header = document.createElement('div');
            header.className = `search-group${isCurrent ? ' is-current' : ''}`;
            header.dataset.galaxy = entry.key;
            const title = document.createElement('span');
            title.textContent = `${t(entry.titleKey)}${isCurrent ? ` · ${t('roadmap.search.current')}` : ''}`;
            const count = document.createElement('b');
            count.textContent = String(hits.length).padStart(2, '0');
            header.append(title, count);
            results.append(header);
            const limit = isCurrent ? 8 : 3;
            hits.slice(0, limit).forEach(({ row }) => results.append(resultButton(row)));
            if (hits.length > limit) {
                const more = document.createElement('button');
                more.type = 'button';
                more.className = 'search-more';
                more.textContent = t('roadmap.search.more').replace('{n}', hits.length - limit);
                more.addEventListener('click', () => {
                    closeResults();
                    if (isCurrent) return;
                    selectPage(panelForMap(entry.key).dataset.treePanel, true);
                });
                results.append(more);
            }
        }
        if (!complete) {
            const pending = document.createElement('p');
            pending.className = 'search-empty';
            pending.textContent = t('roadmap.search.loading');
            results.append(pending);
        }
    }

    function updateSearch({ open = document.activeElement === search || results.contains(document.activeElement) } = {}) {
        const query = normalize(search.value);
        const panel = activePanel();
        const nodes = Array.from(panel.querySelectorAll('.graph-node'));
        nodes.forEach(node => {
            const item = nodeData.get(node);
            node.classList.toggle('is-match', Boolean(query) && scoreRow(item, query) > 0);
        });
        panel.classList.toggle('is-searching', Boolean(query));
        const groups = query ? searchMatches(query) : [];
        tabs.forEach(tab => {
            const key = mapKeyOf(pages.find(page => page.dataset.treePanel === tab.dataset.treePage));
            const group = groups.find(entry => entry.entry.key === key);
            if (query && searchIndex.has(key)) {
                tab.dataset.matches = t('roadmap.search.hits').replace('{n}', group ? group.hits.length : 0);
                tab.classList.toggle('has-no-match', !group);
            } else {
                delete tab.dataset.matches;
                tab.classList.remove('has-no-match');
            }
        });
        results.hidden = !query || !open;
        if (!query) { results.replaceChildren(); return; }
        if (open) renderResults(query, groups);
    }

    function closeResults() {
        results.hidden = true;
    }

    function revealNode(panel, id) {
        const node = panel.querySelector(`.graph-node[data-node-id="${CSS.escape(id)}"]`);
        if (!node) return;
        selectNode(node);
        const item = nodeData.get(node);
        if (item.type === 'leaf') focusNode(node);
        node.classList.remove('is-target');
        node.getBBox();
        node.classList.add('is-target');
        window.setTimeout(() => node.classList.remove('is-target'), 2600);
        node.focus({ preventScroll: true });
        history.replaceState(null, '', `#node-${id}`);
    }

    function goToResult(row) {
        const panel = panelForMap(row.map);
        if (!panel) return;
        search.value = '';
        closeResults();
        if (!panel.hidden && layouts.has(panel)) {
            updateSearch({ open: false });
            revealNode(panel, row.id);
        } else {
            selectPage(panel.dataset.treePanel, false, { focusId: row.id });
        }
        app.querySelector('.tree-workspace, [data-graph-canvas]')?.scrollIntoView?.({ block: 'nearest' });
    }

    const resultButtons = () => [...results.querySelectorAll('button')];
    search.addEventListener('focus', () => { ensureSearchIndex(); updateSearch({ open: true }); });
    search.addEventListener('input', () => { ensureSearchIndex(); updateSearch({ open: true }); });
    search.addEventListener('keydown', event => {
        if (event.key === 'Escape') { search.value = ''; updateSearch({ open: false }); }
        if (event.key === 'Enter') { event.preventDefault(); results.querySelector('.search-result')?.click(); }
        if (event.key === 'ArrowDown') { event.preventDefault(); resultButtons()[0]?.focus(); }
    });
    results.addEventListener('keydown', event => {
        const buttons = resultButtons();
        const index = buttons.indexOf(document.activeElement);
        if (event.key === 'ArrowDown') { event.preventDefault(); buttons[Math.min(buttons.length - 1, index + 1)]?.focus(); }
        if (event.key === 'ArrowUp') { event.preventDefault(); (index <= 0 ? search : buttons[index - 1]).focus(); }
        if (event.key === 'Escape') { event.preventDefault(); search.focus(); closeResults(); }
    });
    search.closest('.map-search')?.addEventListener('focusout', event => {
        if (!event.currentTarget.contains(event.relatedTarget)) closeResults();
    });
    app.querySelectorAll('[data-map-zoom]').forEach(button => button.addEventListener('click', () => {
        if (button.dataset.mapZoom === 'fit') { fitMap(); return; }
        const panel = activePanel();
        const view = layouts.get(panel);
        zoomMap(panel.querySelector('[data-graph-canvas]'), button.dataset.mapZoom === 'in' ? 1.35 : 1 / 1.35, { x: view.width / 2, y: view.height / 2 });
    }));

    async function selectPage(name, updateHistory, options = {}) {
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
        if (updateHistory) history.replaceState(null, "", "#tree-" + pageName);
        const activePage = pages.find((page) => page.dataset.treePanel === pageName);
        const activeShell = activePage?.querySelector("[data-graph-canvas]");
        if (!activeShell) return;
        // Cross-fade: the current map dims while the next map's data loads,
        // then the panels swap and the new plot fades in once fully drawn.
        const outgoing = activePanel();
        const fading = !reduceMotion && outgoing && outgoing !== activePage && layouts.has(outgoing);
        if (!reduceMotion) app.classList.add('is-plotting');
        const showPage = () => {
            pages.forEach((page) => {
                animations.get(page)?.revert();
                animations.delete(page);
                stopCamera(page.querySelector("[data-graph-canvas]"));
                const active = page === activePage;
                page.classList.toggle("is-active", active);
                page.hidden = !active;
            });
            app.dataset.galaxy = activePage.dataset.galaxy;
        };
        results.hidden = true;
        app.querySelectorAll('[data-map-zoom]').forEach(button => { button.disabled = true; });
        activePage.setAttribute('aria-busy', 'true');
        app.querySelector('[data-roadmap-count]').textContent = t('roadmap.graph.loading');
        detail.textContent = t('roadmap.graph.loading');
        try {
            const [data] = await Promise.all([loadGalaxy(activePage), fading ? new Promise(resolve => window.setTimeout(resolve, 200)) : null]);
            if (epoch !== selectionEpoch) return;
            showPage();
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
            showPage();
            app.classList.remove('is-plotting');
            activePage.setAttribute('aria-busy', 'false');
            app.querySelector('[data-roadmap-count]').textContent = t('roadmap.graph.loadError');
            detail.textContent = t('roadmap.graph.loadError');
            return;
        }
        activePage.setAttribute('aria-busy', 'false');
        app.querySelectorAll('[data-map-zoom]').forEach(button => { button.disabled = false; });
        cameraFor(activeShell).geometry = null;
        measureLabels(activePage);
        applyCamera(activeShell);
        animateGalaxy(activePage);
        updateSearch({ open: false });
        updateSummary();
        detail.textContent = t('roadmap.graph.select');
        const hudSelection = activeShell.querySelector('[data-hud-sel]');
        if (hudSelection) hudSelection.textContent = '—';
        if (options.focusId) revealNode(activePage, options.focusId);
        // Two frames: the first lays out the new plot at opacity 0, the second fades it in.
        requestAnimationFrame(() => requestAnimationFrame(() => {
            if (epoch === selectionEpoch) app.classList.remove('is-plotting');
        }));
    }

    tabs.forEach((tab, index) => {
        const panel = pages.find(page => page.dataset.treePanel === tab.dataset.treePage);
        const entry = GALAXIES.find(galaxy => galaxy.key === panel?.dataset.galaxy);
        if (entry) tab.dataset.nodes = String(entry.nodes);
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
        const sweep = document.createElement('div');
        sweep.className = 'map-sweep';
        sweep.setAttribute('aria-hidden', 'true');
        const hud = document.createElement('div');
        hud.className = 'map-hud';
        hud.setAttribute('aria-hidden', 'true');
        for (const [name, initial] of [['scale', '1.00×'], ['x', '+0000'], ['y', '+0000'], ['sel', '—']]) {
            const field = document.createElement('span');
            field.append(`${name.toUpperCase()} `);
            const value = document.createElement('b');
            value.setAttribute(`data-hud-${name}`, '');
            value.textContent = initial;
            field.append(value);
            hud.append(field);
        }
        shell.prepend(sweep);
        shell.append(hud);
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
            if (camera.moved) applyCamera(shell, true);
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

    window.addEventListener("hashchange", () => { const route = routeFromHash(); selectPage(route.page, false, route); });
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
    const initialRoute = routeFromHash();
    selectPage(initialRoute.page, false, initialRoute);
}());
