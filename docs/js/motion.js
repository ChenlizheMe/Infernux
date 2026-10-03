/*
 * Infernux motion layer
 *
 * GSAP and ScrollTrigger are vendored under js/vendor at a fixed version so
 * the site never depends on a CDN or an unavailable third-party request at
 * runtime. This file is intentionally defensive: the core site remains
 * usable if a browser blocks motion or a local asset is missing.
 */
(function () {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const gsap = globalThis.gsap;
    const ScrollTrigger = globalThis.ScrollTrigger;

    if (reducedMotion || !gsap || !ScrollTrigger) {
        document.documentElement.classList.toggle('motion-reduced', reducedMotion);
        return;
    }

    gsap.registerPlugin(ScrollTrigger);

    function all(selector, root = document) {
        return Array.from(root.querySelectorAll(selector));
    }

    function revealOnScroll(elements, options = {}) {
        elements.filter(Boolean).forEach((element, index) => {
            gsap.fromTo(element,
                { autoAlpha: 0, y: options.y ?? 34, rotateX: options.rotateX ?? -3 },
                {
                    autoAlpha: 1,
                    y: 0,
                    rotateX: 0,
                    duration: options.duration ?? 0.78,
                    delay: (options.delay ?? 0) + index * (options.stagger ?? 0.035),
                    ease: 'power3.out',
                    overwrite: 'auto',
                    scrollTrigger: {
                        trigger: element,
                        start: options.start ?? 'top 86%',
                        once: true
                    }
                }
            );
        });
    }

    function buildHeroTimeline(hero) {
        const timeline = gsap.timeline({ defaults: { ease: 'power3.out' } });
        const copy = hero.querySelector('.hero-copy');
        const panel = hero.querySelector('.hero-panel');
        const copyItems = copy ? all('.hero-codename, .hero-title, .hero-subtitle, .hero-description, .hero-actions, .hero-platform-note, .hero-text-links, .hero-meta', copy) : [];

        timeline.from(hero, { autoAlpha: 0, y: 48, rotateX: -3, duration: 1.05 })
            .from(copyItems, { autoAlpha: 0, y: 28, duration: 0.7, stagger: 0.075 }, '-=0.68')
            .from(panel, { autoAlpha: 0, x: 68, duration: 0.92 }, '-=0.72');

        if (panel) {
            gsap.to(panel, {
                y: -72,
                rotateZ: 1.2,
                ease: 'none',
                scrollTrigger: {
                    trigger: hero,
                    start: 'top top',
                    end: 'bottom top',
                    scrub: 0.7
                }
            });
        }
        return timeline;
    }

    function addTapeAnimations() {
        const rails = all('.hero-panel-rail-line');
        if (rails.length) {
            gsap.fromTo(rails,
                { scaleX: 0.18, transformOrigin: 'left center', autoAlpha: 0.45 },
                { scaleX: 1, autoAlpha: 1, duration: 1.8, repeat: -1, yoyo: true, ease: 'sine.inOut', stagger: 0.16 }
            );
        }
        gsap.to(all('.hero-panel'), {
            '--signal-offset': '8px',
            duration: 2.4,
            repeat: -1,
            yoyo: true,
            ease: 'sine.inOut'
        });
    }

    function addMicroSignals() {
        gsap.to(all('.github-live-dot'), {
            scale: 1.45,
            opacity: 0.55,
            duration: 1.05,
            repeat: -1,
            yoyo: true,
            ease: 'sine.inOut'
        });
        all('.code-header .dot').forEach((dot, index) => {
            gsap.to(dot, {
                y: -2,
                duration: 0.72,
                delay: index * 0.08,
                repeat: -1,
                yoyo: true,
                ease: 'sine.inOut'
            });
        });
    }

    function addLinkFeedback() {
        all('.btn, .github-star-link, .nav-links a').forEach(link => {
            link.addEventListener('pointerenter', () => gsap.to(link, { y: -2, duration: 0.2, ease: 'power2.out', overwrite: 'auto' }), { passive: true });
            link.addEventListener('pointerleave', () => gsap.to(link, { y: 0, duration: 0.26, ease: 'power2.out', overwrite: 'auto' }), { passive: true });
        });
    }

    function initMotion() {
        const hero = document.querySelector('.hero-slab, .subpage-hero, .hub-hero');
        if (hero) buildHeroTimeline(hero);

        revealOnScroll(all('.section-head, .manifesto-strip, .cta-panel, .demo-frame, .stack-board, [data-gsap-reveal]'));
        all('.grid-3, .capability-grid, .status-grid').forEach(group => {
            const items = Array.from(group.children);
            if (items.length) revealOnScroll(items, { y: 42, rotateX: -5, stagger: 0.08, start: 'top 82%' });
        });

        all('.section-title').forEach(title => {
            gsap.fromTo(title, { clipPath: 'inset(0 100% 0 0)' }, {
                clipPath: 'inset(0 0% 0 0)',
                duration: 0.9,
                ease: 'power3.inOut',
                scrollTrigger: { trigger: title, start: 'top 88%', once: true }
            });
        });

        addTapeAnimations();
        addMicroSignals();
        addLinkFeedback();
        ScrollTrigger.refresh();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initMotion, { once: true });
    } else {
        initMotion();
    }
}());
