import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = relative => fs.readFileSync(path.join(root, relative), 'utf8');
const assert = (condition, message) => {
    if (!condition) throw new Error(message);
};

const gsap = read('js/vendor/gsap.min.js');
const scrollTrigger = read('js/vendor/ScrollTrigger.min.js');
const motion = read('js/motion.js');
const sharedStyle = read('css/style.css');
const main = read('js/main.js');
assert(gsap.length > 50_000, 'vendored GSAP bundle is missing or unexpectedly small');
assert(scrollTrigger.length > 25_000, 'vendored ScrollTrigger bundle is missing or unexpectedly small');
assert(motion.includes('registerPlugin(ScrollTrigger)'), 'motion layer must register ScrollTrigger');
assert(motion.includes('gsap.timeline'), 'motion layer must build a GSAP timeline');
assert(motion.includes('scrollTrigger:'), 'motion layer must define ScrollTrigger scenes');
assert(motion.includes('prefers-reduced-motion'), 'motion layer must honor reduced motion');
assert(!motion.includes('https://') && !motion.includes('http://'), 'motion layer must not load a remote runtime');
assert(motion.includes('hero-panel-rail-line'), 'motion layer must animate the tape-futurist signal rail');
assert(!motion.includes('rotationY') && !motion.includes('rotationX'), 'the starter component must stay flat');
assert(!sharedStyle.includes('tape-reel') && !sharedStyle.includes('tape-deck-'), 'the starter component must not be drawn as a cassette');
assert(!sharedStyle.includes('[data-theme="light"]') && !sharedStyle.includes('theme-toggle'), 'the shared shell must be dark-only');
assert(!main.includes('toggleTheme') && !main.includes('data-theme'), 'theme switching code must be removed from the runtime');

const htmlFiles = [];
function collect(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
        const fullPath = path.join(directory, entry.name);
        if (entry.isDirectory()) collect(fullPath);
        else if (entry.name.endsWith('.html')) htmlFiles.push(fullPath);
    }
}
collect(root);

const pages = htmlFiles.filter(file => read(path.relative(root, file)).includes('main.js?v=16'));
assert(pages.length > 100, `expected the generated site to load motion on every page (found ${pages.length})`);
for (const file of pages) {
    const html = read(path.relative(root, file));
    assert(!html.includes('theme-toggle') && !html.includes('dark light'), `${path.relative(root, file)} still advertises a light theme`);
    assert(html.includes('vendor/gsap.min.js?v=3.13.0'), `${path.relative(root, file)} is missing local GSAP`);
    assert(html.includes('vendor/ScrollTrigger.min.js?v=3.13.0'), `${path.relative(root, file)} is missing local ScrollTrigger`);
    assert(html.includes('motion.js?v=1'), `${path.relative(root, file)} is missing the motion layer`);
}

console.log(`Motion client verified: GSAP ${gsap.length} bytes, ScrollTrigger ${scrollTrigger.length} bytes, ${pages.length} HTML pages wired locally.`);
