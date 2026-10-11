/* Local GSAP motion layer: finite entrances and wipes. Continuous effects live
   in lazily loaded modules that pause offscreen and honour reduced motion. */
(function () {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const gsap = globalThis.gsap;
    const ScrollTrigger = globalThis.ScrollTrigger;
    gsap.registerPlugin(ScrollTrigger);

    function loadModule(src) {
        if (document.querySelector(`script[data-fx-module="${src}"]`)) return;
        const script = document.createElement('script');
        script.src = src;
        script.async = true;
        script.dataset.fxModule = src;
        document.head.appendChild(script);
    }

    function scheduleModules() {
        const idle = window.requestIdleCallback || (callback => window.setTimeout(callback, 350));
        idle(() => loadModule('/js/fx-hud.js?v=3'), { timeout: 1500 });
        // The hero world is first-view content: start it as soon as the main
        // thread settles, without blocking the headline or first paint.
        if (document.querySelector('[data-fx-world]')) idle(() => loadModule('/js/fx-world.js?v=2'), { timeout: 600 });
    }

    function init() {
        gsap.matchMedia().add('(prefers-reduced-motion: no-preference)', () => {
            const hero = document.querySelector('.hero-slab');
            if (hero) {
                // The LCP image and headline stay visible; only chrome enters.
                gsap.timeline({ defaults: { duration: 0.55, ease: 'power2.out' } })
                    .from('.hero-index-rule', { scaleX: 0, transformOrigin: 'left', ease: 'steps(16)', duration: 0.7 }, 0)
                    .from('.hero-codename, .hero-actions', { y: 12, opacity: 0, stagger: 0.08 }, 0.05)
                    .from('.hero-panel-rail-line', { scaleX: 0, transformOrigin: 'left' }, 0)
                    .from('.nn-inspector', { y: 18, opacity: 0, duration: 0.6 }, 0.15)
                    .from('.nn-action, .nn-telemetry > span', { x: -10, opacity: 0, stagger: 0.05, ease: 'steps(5)', duration: 0.3 }, 0.45)
                    .from('.engineering-board .metric-card, .engineering-board .github-star-deck', { y: 14, opacity: 0, stagger: 0.07 }, 0.3);
            }
            document.querySelectorAll('.section-head, .n3-type, .n3-loop-head, .reel-rack-head, .cta-panel, [data-gsap-reveal]').forEach(element => {
                gsap.from(element, {
                    y: 18, opacity: 0, duration: 0.5, ease: 'power2.out',
                    scrollTrigger: { trigger: element, start: 'top 94%', once: true }
                });
            });
            // Swiss headline wipe: a stepped reveal, like a line printer.
            document.querySelectorAll('.section-title').forEach(title => {
                gsap.fromTo(title, { clipPath: 'inset(0 100% 0 0)' }, {
                    clipPath: 'inset(0 0% 0 0)', duration: 0.7, ease: 'steps(14)',
                    scrollTrigger: { trigger: title, start: 'top 92%', once: true },
                    onComplete: () => gsap.set(title, { clearProps: 'clipPath' })
                });
            });
            ScrollTrigger.batch('[data-reveal], .reel-tape, .n3-item, .n3-stage', {
                start: 'top 92%',
                once: true,
                onEnter: batch => gsap.from(batch, { y: 26, opacity: 0, duration: 0.55, ease: 'power3.out', stagger: 0.08 })
            });
        });
        document.documentElement.classList.toggle('motion-reduced', preference.matches);
        preference.addEventListener('change', event => document.documentElement.classList.toggle('motion-reduced', event.matches));
        scheduleModules();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
    else init();
}());
