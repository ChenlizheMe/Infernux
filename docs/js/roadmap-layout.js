/* Seeded radial growth: irregular positions, stable across refreshes. */
function roadmapLabelLines(text, limit = 14) {
    const lines = [];
    let line = '';
    let width = 0;
    for (const char of text) {
        const units = char.charCodeAt(0) > 255 ? 1 : .56;
        if (width + units > limit && line) {
            const space = line.lastIndexOf(' ');
            if (space > line.length / 3) {
                lines.push(line.slice(0, space));
                line = line.slice(space + 1);
                width = Array.from(line).reduce((sum, c) => sum + (c.charCodeAt(0) > 255 ? 1 : .56), 0);
            } else { lines.push(line.trim()); line = ''; width = 0; }
        }
        line += char;
        width += units;
    }
    if (line.trim()) lines.push(line.trim());
    return lines;
}

/* A branch's status derives from its leaves; shared by layout and search. */
function roadmapStatusOf(leaves) {
    return leaves.some(row => row[2] === 'progress') ? 'progress'
        : leaves.every(row => row[2] === 'live') ? 'live'
        : leaves.some(row => row[2] === 'live') ? 'partial'
        : leaves.every(row => row[2] === 'future') ? 'future' : 'planned';
}

function layoutRoadmap(galaxy) {
    let seed = 2166136261;
    for (const char of galaxy.key) seed = Math.imul(seed ^ char.charCodeAt(0), 16777619) >>> 0;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    const kinds = ['gas', 'ringed', 'rock', 'ice'];
    const statusOf = roadmapStatusOf;
    const allLeaves = galaxy.branches.flatMap(branch => branch.leaves);
    const completion = leaves => leaves.filter(row => row[2] === 'live').length / leaves.length;
    const nodes = [], edges = [];
    const intersects = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
    const translate = (box, x, y) => ({ left: box.left + x, right: box.right + x, top: box.top + y, bottom: box.bottom + y });
    const union = boxes => ({ left: Math.min(...boxes.map(b => b.left)), right: Math.max(...boxes.map(b => b.right)), top: Math.min(...boxes.map(b => b.top)), bottom: Math.max(...boxes.map(b => b.bottom)) });
    const bounds = node => {
        const font = node.type === 'root' ? 30 : node.type === 'branch' ? 25 : 21;
        const strings = node.key ? ['NEURAL NETWORKS & HPC', '神经网络与高性能计算'] : [node.en, node.zh];
        const rows = strings.map(value => roadmapLabelLines(value));
        const widths = rows.flat().map(line => Array.from(line).reduce((sum, char) => sum + (char.charCodeAt(0) > 255 ? 1 : .56) * font, 0));
        const halfWidth = Math.max(node.radius * 1.9, ...widths.map(width => width / 2)) + 12;
        return { left: node.x - halfWidth, right: node.x + halfWidth, top: node.y - node.radius * 1.9 - 8,
            bottom: node.y + node.radius + 30 + Math.max(...rows.map(lines => lines.length)) * font * 1.25 + 10 };
    };
    // Pack a complete local system before placing it globally. A child's spacing
    // depends on its own siblings, never on the total number of roadmap goals.
    const root = { id: galaxy.key, type: 'root', x: 0, y: 0, radius: 46, key: galaxy.titleKey,
        status: statusOf(allLeaves), completion: completion(allLeaves), kind: 'star' };
    root.bounds = bounds(root); nodes.push(root);
    const systems = [root.bounds];
    const phase = random() * Math.PI * 2;
    const goldenAngle = 2.399963229728653;
    galaxy.branches.forEach((branch, index) => {
        const hub = { id: `${galaxy.key}-${index}`, parent: galaxy.key, type: 'branch', x: 0, y: 0,
            radius: 32 + random() * 4, en: branch.en, zh: branch.zh, status: statusOf(branch.leaves),
            completion: completion(branch.leaves), kind: kinds[index % kinds.length] };
        hub.bounds = bounds(hub);
        const local = [hub];
        const rotation = random() * Math.PI * 2;
        branch.leaves.forEach(([en, zh, status], leafIndex) => {
            const radius = 18 + random() * 4;
            const leaf = { id: `${hub.id}-${leafIndex}`, parent: hub.id, type: 'leaf', radius, en, zh, status,
                kind: ['asteroid', ...kinds][(index + leafIndex) % 5] };
            let attempt = 0;
            // The expanding disk finds the nearest available bilingual footprint.
            do {
                const angle = rotation + (leafIndex + attempt) * goldenAngle;
                const distance = 155 + 25 * Math.sqrt(attempt);
                leaf.x = Math.cos(angle) * distance * 1.35;
                leaf.y = Math.sin(angle) * distance;
                leaf.bounds = bounds(leaf);
                attempt++;
            } while (local.some(other => intersects(leaf.bounds, other.bounds)));
            local.push(leaf);
        });
        const localBounds = union(local.map(n => n.bounds));
        const reserved = { left: localBounds.left - 40, right: localBounds.right + 40, top: localBounds.top - 40, bottom: localBounds.bottom + 40 };
        let position, box, attempt = 0;
        do {
            const angle = phase + (index + attempt) * goldenAngle;
            const distance = 420 + 48 * Math.sqrt(attempt);
            position = { x: Math.cos(angle) * distance * 1.18, y: Math.sin(angle) * distance };
            box = translate(reserved, position.x, position.y);
            attempt++;
        } while (systems.some(other => intersects(box, other)));
        systems.push(box);
        for (const node of local) {
            node.x += position.x; node.y += position.y;
            node.bounds = translate(node.bounds, position.x, position.y);
            nodes.push(node);
            edges.push({ from: node.parent, to: node.id, status: node.status });
        }
        hub.subtreeBounds = translate(localBounds, position.x, position.y);
        hub.focusX = (hub.subtreeBounds.left + hub.subtreeBounds.right) / 2;
        hub.focusY = (hub.subtreeBounds.top + hub.subtreeBounds.bottom) / 2;
    });
    const allBounds = union(nodes.map(node => node.bounds));
    const halfWidth = Math.max(Math.abs(allBounds.left), Math.abs(allBounds.right)) + 70;
    const halfHeight = Math.max(Math.abs(allBounds.top), Math.abs(allBounds.bottom)) + 70;
    for (const node of nodes) {
        node.x += halfWidth; node.y += halfHeight;
        node.bounds = translate(node.bounds, halfWidth, halfHeight);
        if (node.subtreeBounds) {
            node.subtreeBounds = translate(node.subtreeBounds, halfWidth, halfHeight);
            node.focusX += halfWidth; node.focusY += halfHeight;
        }
    }
    return { width: halfWidth * 2, height: halfHeight * 2, nodes, edges };
}
