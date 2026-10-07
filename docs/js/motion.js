/* Local GSAP: finite entrances, no idle loops or scroll-driven layout work. */
(function () {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const gsap = globalThis.gsap;
    const ScrollTrigger = globalThis.ScrollTrigger;
    gsap.registerPlugin(ScrollTrigger);
    function init() {
        gsap.matchMedia().add('(prefers-reduced-motion: no-preference)', () => {
            const hero = document.querySelector('.hero-slab');
            if (hero) {
                // Keep the LCP image and headline visible throughout the entrance.
                gsap.timeline({ defaults: { duration: .55, ease: 'power2.out' } })
                    .from('.hero-codename, .hero-actions', { y: 12, opacity: 0, stagger: .08 })
                    .from('.hero-panel-rail-line', { scaleX: 0, transformOrigin: 'left' }, 0);
            }
            document.querySelectorAll('.section-head, .manifesto-strip, .cta-panel, [data-gsap-reveal]').forEach(element => {
                gsap.from(element, {
                    y: 18, opacity: 0, duration: .5, ease: 'power2.out',
                    scrollTrigger: { trigger: element, start: 'top 94%', once: true }
                });
            });
        });
        document.documentElement.classList.toggle('motion-reduced', preference.matches);
        preference.addEventListener('change', event => document.documentElement.classList.toggle('motion-reduced', event.matches));
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
    else init();
}());
