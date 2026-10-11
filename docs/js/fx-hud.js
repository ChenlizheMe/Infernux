/* Infernux FX layer: tape HUD, signal acquisition, text scramble, pixel
   resolve, counters, chromatic glitch and pointer telemetry. Loaded on idle by
   motion.js. Visual state is expressed through classes, attributes and text;
   every loop is demand-driven and stops while the page is hidden. */
(function () {
    if (globalThis.__infernuxFx) return;
    globalThis.__infernuxFx = true;

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const gsap = globalThis.gsap;
    const glyphs = '▚▞▙▟█▓▒░<>/\\#01';
    const pad = (value, size) => String(Math.max(0, Math.floor(value))).padStart(size, '0');

    function element(tag, className, parent) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (parent) parent.appendChild(node);
        return node;
    }

    function onVisible(targets, callback, options = {}) {
        if (!targets.length) return;
        if (!('IntersectionObserver' in window)) {
            targets.forEach(callback);
            return;
        }
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (!entry.isIntersecting) return;
                if (options.once !== false) observer.unobserve(entry.target);
                callback(entry.target);
            });
        }, { rootMargin: options.rootMargin || '0px 0px -8% 0px', threshold: options.threshold || 0 });
        targets.forEach(target => observer.observe(target));
    }

    /* ---------------------------------------------- tape timecode + HUD */
    function timecode(frames) {
        const fps = 24;
        const f = frames % fps;
        const totalSeconds = Math.floor(frames / fps);
        return `${pad(totalSeconds / 3600, 2)}:${pad((totalSeconds / 60) % 60, 2)}:${pad(totalSeconds % 60, 2)}:${pad(f, 2)}`;
    }

    function buildHud() {
        const frame = element('div', 'fx-frame', document.body);
        frame.setAttribute('aria-hidden', 'true');
        for (let i = 0; i < 4; i += 1) element('i', '', frame);

        const hud = element('div', 'fx-hud', document.body);
        hud.setAttribute('aria-hidden', 'true');
        const rec = element('span', 'fx-hud-rec', hud);
        rec.textContent = 'TAPE A';
        const tc = element('span', '', hud);
        tc.append('TC ');
        const tcValue = element('b', '', tc);
        const pos = element('span', 'fx-hud-pos', hud);
        pos.append('POS ');
        const posValue = element('b', '', pos);
        const meterWrap = element('span', '', hud);
        meterWrap.append('SIG ');
        const meter = element('span', 'fx-hud-meter', meterWrap);
        const bars = Array.from({ length: 10 }, () => element('i', '', meter));
        const pointer = element('span', 'fx-hud-pointer', hud);
        pointer.append('XY ');
        const pointerValue = element('b', '', pointer);
        pointerValue.textContent = '0000 · 0000';

        const heroCounter = document.querySelector('[data-fx-timecode]');
        let lastY = window.scrollY;
        let velocity = 0;
        let frameRequest = 0;
        let pointerX = 0;
        let pointerY = 0;
        let armed = reduceMotion;

        function render() {
            frameRequest = 0;
            const y = window.scrollY;
            const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
            hud.classList.toggle('is-live', armed && y > window.innerHeight * 0.55);
            const code = timecode(Math.round(y / 3));
            tcValue.textContent = code;
            if (heroCounter) heroCounter.textContent = code;
            posValue.textContent = `${pad((y / max) * 100, 3)}%`;
            velocity = Math.max(velocity * 0.86, Math.min(1, Math.abs(y - lastY) / 60));
            lastY = y;
            const lit = Math.round(velocity * bars.length);
            bars.forEach((bar, index) => bar.classList.toggle('on', index < lit));
            pointerValue.textContent = `${pad(pointerX, 4)} · ${pad(pointerY, 4)}`;
            if (velocity > 0.02 && !document.hidden) frameRequest = window.requestAnimationFrame(render);
        }
        const request = () => {
            if (!frameRequest) frameRequest = window.requestAnimationFrame(render);
        };
        window.addEventListener('scroll', request, { passive: true });
        window.addEventListener('pointermove', event => {
            pointerX = event.clientX;
            pointerY = event.clientY;
            request();
        }, { passive: true });
        render();
        window.setTimeout(() => {
            armed = true;
            render();
        }, reduceMotion ? 0 : 900);
    }

    /* ------------------------------------------- one-shot signal acquire */
    function bootSweep() {
        if (reduceMotion) return;
        try {
            if (sessionStorage.getItem('inx-fx-boot') === '1') return;
            sessionStorage.setItem('inx-fx-boot', '1');
        } catch {
            return;
        }
        const boot = element('div', 'fx-boot', document.body);
        boot.setAttribute('aria-hidden', 'true');
        window.setTimeout(() => boot.remove(), 1100);
    }

    /* ------------------------------------------------------ text scramble */
    function textNodes(root) {
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
            acceptNode: node => node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT
        });
        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        return nodes;
    }

    function scramble(target) {
        if (reduceMotion || target.dataset.fxScrambling === 'true') return;
        const nodes = textNodes(target).map(node => ({ node, text: node.nodeValue }));
        const length = nodes.reduce((sum, entry) => sum + entry.text.length, 0);
        if (!length || length > 90) return;
        target.dataset.fxScrambling = 'true';
        target.classList.add('fx-scrambling');
        const total = 16;
        let step = 0;
        function tick() {
            step += 1;
            const reveal = Math.floor((step / total) * length);
            let offset = 0;
            nodes.forEach(entry => {
                if (!entry.node.isConnected) return;
                let output = '';
                for (let i = 0; i < entry.text.length; i += 1) {
                    const char = entry.text[i];
                    output += offset + i < reveal || char === ' ' || char === '·' || char === '/'
                        ? char
                        : glyphs[(Math.random() * glyphs.length) | 0];
                }
                entry.node.nodeValue = output;
                offset += entry.text.length;
            });
            if (step < total) {
                window.setTimeout(tick, 28);
                return;
            }
            nodes.forEach(entry => {
                if (entry.node.isConnected) entry.node.nodeValue = entry.text;
            });
            target.classList.remove('fx-scrambling');
            delete target.dataset.fxScrambling;
        }
        tick();
    }

    const scrambleSelector = '.section-kicker, .mini-tag, .hero-index-label, .engineering-board-head span, .nn-head span, .n3-word, .reel-index, .footer-column h4, .doc-build-label, .learn-course-kicker';

    /* ------------------------------------------------------ pixel resolve */
    function pixelResolve(image) {
        if (reduceMotion || image.dataset.fxResolved === 'true') return;
        const host = image.parentElement;
        if (!host) return;
        const start = () => {
            const rect = image.getBoundingClientRect();
            if (!rect.width || !image.naturalWidth) return;
            image.dataset.fxResolved = 'true';
            const canvas = element('canvas', 'fx-pixel-canvas');
            canvas.setAttribute('aria-hidden', 'true');
            canvas.width = Math.min(1280, Math.round(rect.width));
            canvas.height = Math.round(canvas.width * (rect.height / rect.width));
            const context = canvas.getContext('2d');
            const small = document.createElement('canvas');
            const smallContext = small.getContext('2d');
            if (!context || !smallContext) return;
            host.appendChild(canvas);
            const blocks = [40, 24, 14, 8, 4, 2];
            let index = 0;
            function drawStep() {
                const block = blocks[index];
                small.width = Math.max(1, Math.round(canvas.width / block));
                small.height = Math.max(1, Math.round(canvas.height / block));
                smallContext.imageSmoothingEnabled = true;
                smallContext.drawImage(image, 0, 0, small.width, small.height);
                context.imageSmoothingEnabled = false;
                context.drawImage(small, 0, 0, canvas.width, canvas.height);
                index += 1;
                if (index < blocks.length) {
                    window.setTimeout(drawStep, 70);
                    return;
                }
                canvas.classList.add('is-done');
                window.setTimeout(() => canvas.remove(), 400);
            }
            drawStep();
        };
        if (image.complete && image.naturalWidth) start();
        else image.addEventListener('load', start, { once: true });
    }

    /* ------------------------------------------------------------ counters */
    function countUp(target) {
        const value = Number(target.dataset.fxCount);
        if (!Number.isFinite(value)) return;
        const format = number => Math.round(number).toLocaleString('en-US');
        if (reduceMotion || !gsap) {
            target.textContent = format(value);
            return;
        }
        const state = { value: 0 };
        gsap.to(state, {
            value, duration: 1.4, ease: 'power3.out',
            onUpdate: () => { target.textContent = format(state.value); },
            onComplete: () => { target.textContent = format(value); }
        });
    }

    /* ----------------------------------------------- chromatic title glitch */
    function glitchTitle() {
        const title = document.querySelector('.hero-title');
        if (!title || reduceMotion) return;
        title.classList.add('fx-glitch');
        const mirror = () => title.setAttribute('data-text', title.textContent.replace(/\s+/g, ' ').trim());
        mirror();
        document.addEventListener('site:language-changed', mirror);
        let visible = true;
        if ('IntersectionObserver' in window) {
            new IntersectionObserver(entries => { visible = entries[0].isIntersecting; }).observe(title);
        }
        function pulse() {
            if (visible && !document.hidden) {
                title.classList.add('is-glitching');
                window.setTimeout(() => title.classList.remove('is-glitching'), 460);
            }
            window.setTimeout(pulse, 4200 + Math.random() * 4200);
        }
        window.setTimeout(pulse, 1800);
    }

    /* -------------------------------------------------- pointer telemetry */
    function pointerTelemetry() {
        document.querySelectorAll('[data-fx-pointer]').forEach(surface => {
            const readout = surface.querySelector('[data-fx-coord]');
            const reticle = surface.querySelector('.hud-reticle');
            const image = surface.querySelector('img, canvas');
            if (!readout) return;
            const moveX = reticle && gsap && !reduceMotion ? gsap.quickTo(reticle, 'x', { duration: 0.35, ease: 'power3.out' }) : null;
            const moveY = reticle && gsap && !reduceMotion ? gsap.quickTo(reticle, 'y', { duration: 0.35, ease: 'power3.out' }) : null;
            surface.addEventListener('pointermove', event => {
                const rect = surface.getBoundingClientRect();
                const u = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
                const v = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
                const width = image?.naturalWidth || image?.width || rect.width;
                const height = image?.naturalHeight || image?.height || rect.height;
                readout.textContent = `X ${pad(u * width, 4)} · Y ${pad(v * height, 4)}`;
                if (moveX && moveY) {
                    moveX((u - 0.5) * rect.width);
                    moveY((v - 0.5) * rect.height);
                }
            }, { passive: true });
            surface.addEventListener('pointerleave', () => {
                if (moveX && moveY) {
                    moveX(0);
                    moveY(0);
                }
            });
        });
    }

    /* ---------------------------------------------- demo reel playback
       Clips ship as preload="none" with data-src. A clip gets its source and
       plays only while visible; it pauses offscreen or in a hidden tab, and
       under prefers-reduced-motion waits for an explicit play. */
    function reelPlayback() {
        const videos = [...document.querySelectorAll('[data-reel] video[data-src]')];
        if (!videos.length) return;
        const inView = new Set();
        const setState = (video, playing) => {
            video.closest('.reel-tape')?.classList.toggle('is-playing', playing);
            video.parentElement.querySelector('.reel-toggle')?.setAttribute('aria-pressed', String(playing));
        };
        const play = video => {
            if (!video.getAttribute('src')) video.src = video.dataset.src;
            const attempt = video.play();
            if (attempt) attempt.then(() => setState(video, true)).catch(() => setState(video, false));
        };
        const pause = video => {
            video.pause();
            setState(video, false);
        };
        videos.forEach(video => {
            video.parentElement.querySelector('.reel-toggle')?.addEventListener('click', () => {
                if (video.paused) {
                    video.dataset.userPaused = 'false';
                    play(video);
                } else {
                    video.dataset.userPaused = 'true';
                    pause(video);
                }
            });
        });
        if (reduceMotion || !('IntersectionObserver' in window)) return;
        const observer = new IntersectionObserver(entries => entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                inView.add(video);
                if (video.dataset.userPaused !== 'true' && !document.hidden) play(video);
            } else {
                inView.delete(video);
                if (!video.paused) pause(video);
            }
        }), { threshold: 0.35 });
        videos.forEach(video => observer.observe(video));
        document.addEventListener('visibilitychange', () => {
            videos.forEach(video => {
                if (document.hidden && !video.paused) pause(video);
                else if (!document.hidden && inView.has(video) && video.dataset.userPaused !== 'true') play(video);
            });
        });
    }

    function init() {
        buildHud();
        bootSweep();
        reelPlayback();
        onVisible([...document.querySelectorAll(scrambleSelector)], scramble);
        document.addEventListener('site:language-changed', () => {
            document.querySelectorAll(scrambleSelector).forEach(target => {
                const rect = target.getBoundingClientRect();
                if (rect.bottom > 0 && rect.top < window.innerHeight) scramble(target);
            });
        });
        onVisible([...document.querySelectorAll('img[data-pixel-reveal]')], pixelResolve, { rootMargin: '0px 0px -12% 0px' });
        onVisible([...document.querySelectorAll('[data-fx-count]')], countUp);
        glitchTitle();
        pointerTelemetry();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
    else init();
}());
