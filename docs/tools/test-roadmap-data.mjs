import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const source = await readFile(new URL('../js/roadmap-data.js', import.meta.url), 'utf8');
const layoutSource = await readFile(new URL('../js/roadmap-layout.js', import.meta.url), 'utf8');
const { GALAXIES: index, layoutRoadmap } = vm.runInNewContext(`${source}\n${layoutSource}\n({ GALAXIES, layoutRoadmap })`);
const GALAXIES = await Promise.all(index.map(async entry => {
    const galaxy = JSON.parse(await readFile(new URL(`../${entry.file.split('?')[0]}`, import.meta.url), 'utf8'));
    assert.equal(galaxy.key, entry.key);
    assert.equal(galaxy.titleKey, entry.titleKey);
    assert.equal(1 + galaxy.branches.length + galaxy.branches.reduce((n, branch) => n + branch.leaves.length, 0), entry.nodes);
    return galaxy;
}));
const statuses = new Set(['live', 'progress', 'planned', 'future']);
assert.equal(GALAXIES.length, 10);
assert.equal(new Set(GALAXIES.map(g => g.key)).size, 10);
assert(GALAXIES.some(g => g.key === 'neural'));

function checkLayout(galaxy) {
    const view = layoutRoadmap(galaxy);
    const expected = 1 + galaxy.branches.length + galaxy.branches.reduce((sum, b) => sum + b.leaves.length, 0);
    assert.equal(view.nodes.length, expected, `${galaxy.key}: dropped nodes`);
    assert.equal(view.edges.length, expected - 1, `${galaxy.key}: disconnected tree`);
    const ids = new Set(view.nodes.map(n => n.id));
    assert.equal(ids.size, expected, `${galaxy.key}: duplicate IDs`);
    const children = new Set(view.edges.map(e => e.to));
    assert.equal(children.size, expected - 1);
    for (const node of view.nodes) {
        assert(Number.isFinite(node.x) && Number.isFinite(node.y));
        assert(node.x >= 0 && node.x <= view.width && node.y >= 0 && node.y <= view.height, `${node.id}: outside map`);
        if (node.type !== 'root') assert(children.has(node.id));
    }
    for (const edge of view.edges) assert(ids.has(edge.from) && ids.has(edge.to));
    const root = view.nodes.find(node => node.type === 'root');
    const leaves = galaxy.branches.flatMap(branch => branch.leaves);
    assert.equal(root.completion, leaves.filter(row => row[2] === 'live').length / leaves.length);
    assert.equal(root.x, view.width / 2);
    assert.equal(root.y, view.height / 2);
    for (const hub of view.nodes.filter(node => node.type === 'branch')) {
        assert(view.edges.some(edge => edge.from === root.id && edge.to === hub.id), `${hub.id}: must grow from the single center`);
        const children = view.nodes.filter(node => node.parent === hub.id);
        for (const child of children) {
            assert(Math.hypot(child.x - hub.x, child.y - hub.y) <= Math.max(600, 180 * Math.sqrt(children.length)), `${child.id}: too far from its own parent`);
            assert(root.radius / child.radius <= 3, `${child.id}: unreadably small relative to root`);
            const box = hub.subtreeBounds;
            assert(child.bounds.left >= box.left && child.bounds.right <= box.right && child.bounds.top >= box.top && child.bounds.bottom <= box.bottom, `${child.id}: branch focus bounds omit child`);
        }
    }
    // Reserved label footprints cover both languages, including uneven trees.
    for (let i = 0; i < view.nodes.length; i++) for (let j = i + 1; j < view.nodes.length; j++) {
        const a = view.nodes[i].bounds, b = view.nodes[j].bounds;
        assert(!(a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top), `${galaxy.key}: overlapping labels`);
    }
    assert.equal(JSON.stringify(view), JSON.stringify(layoutRoadmap(galaxy)), 'refresh must not rearrange the map');
    return expected;
}

let total = 0;
for (const galaxy of GALAXIES) {
    const names = new Set();
    assert(galaxy.branches.length > 4, `${galaxy.key}: overview-only data`);
    for (const branch of galaxy.branches) {
        assert(branch.en && branch.zh && branch.leaves.length);
        for (const [en, zh, status] of branch.leaves) {
            assert(en?.trim() && zh?.trim(), `${galaxy.key}: missing translation`);
            assert(statuses.has(status), `${galaxy.key}: unknown status`);
            const key = en.toLowerCase();
            assert(!names.has(key), `${galaxy.key}: duplicate goal ${en}`);
            names.add(key);
        }
    }
    total += checkLayout(galaxy);
    assert(index.find(entry => entry.key === galaxy.key).nodes >= (galaxy.key === 'rendering' ? 109 : 91) * 3, 'each map must contain at least three times its original node count');
}
assert(total >= 928 * 3);
const renderGoals = GALAXIES.find(g => g.key === 'rendering').branches.flatMap(b => b.leaves.map(row => row[0]));
for (const goal of ['Cloud ray marching', 'Multiple importance sampling', 'MIS balance heuristic', 'Shader binding tables']) assert(renderGoals.includes(goal), `missing requested rendering detail: ${goal}`);
// A growing roadmap must not depend on four/eight fixed slots per branch.
const uneven = structuredClone(GALAXIES[0]);
uneven.branches[0].leaves.push(...structuredClone(uneven.branches[1].leaves));
uneven.branches[1].leaves = uneven.branches[1].leaves.slice(0, 3);
uneven.branches.pop();
checkLayout(uneven);
const progressing = structuredClone(GALAXIES[0]);
for (const branch of progressing.branches) for (const leaf of branch.leaves) leaf[2] = 'planned';
progressing.branches[0].leaves[0][2] = 'live';
const partialView = layoutRoadmap(progressing);
assert.equal(partialView.nodes.find(node => node.type === 'root').status, 'partial');
assert.equal(partialView.nodes.find(node => node.id === `${progressing.key}-0`).status, 'partial');
assert.equal(partialView.nodes.find(node => node.id === `${progressing.key}-1`).status, 'planned');
assert(!partialView.nodes.some(node => node.status === 'progress'), 'completion must never imply active work');
progressing.branches[0].leaves[0][2] = 'progress';
const progressView = layoutRoadmap(progressing);
assert.equal(progressView.nodes.find(node => node.type === 'root').status, 'progress');
assert.equal(progressView.nodes.find(node => node.id === `${progressing.key}-0`).status, 'progress');
assert.equal(progressView.nodes.find(node => node.id === `${progressing.key}-1`).status, 'planned');
for (const branch of progressing.branches) for (const leaf of branch.leaves) leaf[2] = 'live';
assert(layoutRoadmap(progressing).nodes.every(node => node.status === 'live'));
for (const branch of progressing.branches) for (const leaf of branch.leaves) leaf[2] = 'future';
assert(layoutRoadmap(progressing).nodes.every(node => node.status === 'future'));
console.log(`Roadmap verified: ${GALAXIES.length} maps, ${total} bilingual nodes; stable radial layout, one center, non-overlapping bilingual footprints and explicit progress propagation.`);
