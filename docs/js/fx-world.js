/* Infernux hero world: a NASA-punk orbital skirmish rendered behind the home
   page. The INX fleet (four ship classes, four weapon types) is flown by a
   genuine 16-8-4 policy network with hand-set weights: seven sensor rays read
   hazards and targets, the outputs steer, thrust and fire. A drone swarm and
   its carrier answer with plasma, needles and orbs. The inspector panel traces
   the focused ship's activations. Canvas2D draws a low-resolution pixel frame;
   one WebGL pass adds bloom, chromatic split, Bayer-dithered quantisation,
   scanlines, grain and vignette. The loop pauses offscreen or hidden; reduced
   motion renders one settled frame. Visual state uses attributes and text only. */
(function () {
    const host = document.querySelector('[data-fx-world]');
    if (!host || host.dataset.fxReady === 'true') return;
    host.dataset.fxReady = 'true';

    const canvas = host.querySelector('canvas');
    const overlay = host.querySelector('.world-overlay');
    const panel = document.querySelector('[data-nn-panel]');
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const SVG = 'http://www.w3.org/2000/svg';
    const TAU = Math.PI * 2;
    const TICK = 1 / 30;
    const ANGLES = 32;
    const RAY_SPREAD = 0.3;
    const HAZARD_RANGE = 64;
    const TARGET_RANGE = 120;

    /* ------------------------------------------------------------ noise */
    function generator(seed) {
        let state = seed >>> 0;
        return () => {
            state = (state + 0x6D2B79F5) >>> 0;
            let t = state;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }
    let random = generator(20261010);
    function hash(x, y, seed) {
        let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(seed | 0, 2147483647)) | 0;
        h = Math.imul(h ^ (h >>> 13), 1274126177);
        return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
    }
    function valueNoise(x, y, seed) {
        const ix = Math.floor(x), iy = Math.floor(y);
        const fx = x - ix, fy = y - iy;
        const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
        const a = hash(ix, iy, seed), b = hash(ix + 1, iy, seed), c = hash(ix, iy + 1, seed), d = hash(ix + 1, iy + 1, seed);
        return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
    }
    function fbm(x, y, seed) {
        let value = 0, amplitude = 0.5, frequency = 1;
        for (let octave = 0; octave < 4; octave += 1) {
            value += amplitude * valueNoise(x * frequency, y * frequency, seed + octave * 31);
            amplitude *= 0.5;
            frequency *= 2;
        }
        return value / 0.9375;
    }
    const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
    const bayer = (x, y) => (BAYER[(y & 3) * 4 + (x & 3)] + 0.5) / 16;
    const wrapDelta = (delta, size) => delta - Math.round(delta / size) * size;
    const wrap = (value, size) => ((value % size) + size) % size;
    const angleDelta = delta => Math.atan2(Math.sin(delta), Math.cos(delta));

    /* ---------------------------------------------------------- sprites */
    const FRIEND = { W: '#d8d1c2', w: '#9d9688', o: '#ff6a3d', K: '#34373c', c: '#86e6ff', e: '#ffb15c', L: '#fff3c4' };
    const FOE = { D: '#55607a', d: '#8d9ab4', G: '#7dff9a', M: '#ff4fd8' };
    const ROCK = { r: '#6d665d', R: '#8f877b', s: '#47423c' };
    const SHAPES = {
        interceptor: ['..W......', '..WW.....', 'eKwWWWc..', 'eKoooWWWW', 'eKwWWWc..', '..WW.....', '..W......'],
        gunship: ['...WWW.....', '..KwwwW....', 'eKWWoWWWW..', 'eKWWoWWcWW.', 'eKoooooWWWW', 'eKWWoWWcWW.', 'eKWWoWWWW..', '..KwwwW....', '...WWW.....'],
        lancer: ['..WWW..........', 'eKWWWWWww......', 'eKWoooWWWWWWWw.', 'eKWoWoWWWccWWWL', 'eKWoooWWWWWWWw.', 'eKWWWWWww......', '..WWW..........'],
        missile: ['..oo.......', '..KWWW.....', 'eKWWWWWW...', '...WWWWcW..', 'eKWoWWWWWW.', '...WWWWcW..', 'eKWWWWWW...', '..KWWW.....', '..oo.......'],
        drone: ['.D...D.', 'DDD.DDD', '.DdddD.', '..dGd..', '.DdddD.', 'DDD.DDD', '.D...D.'],
        raider: ['DD.........', 'DdDD.......', '.DdddDDM...', 'MDdGGddddDM', '.DdddDDM...', 'DdDD.......', 'DD.........'],
        carrier: [
            '....DDDDDDDD.........', '...DdddddddDD........', '..DdMdMdMdMddDDD.....', 'DDdddddddddddddDD....',
            'DdGdddDDDDDdddddDDD..', 'DddddD.....DddddddDD.', 'DdGddD..G..DdddGGddDM', 'DddddD.....DddddddDD.',
            'DdGdddDDDDDdddddDDD..', 'DDdddddddddddddDD....', '..DdMdMdMdMddDDD.....', '...DdddddddDD........', '....DDDDDDDD.........'
        ],
        rock: ['..rRr..', '.rRRRr.', 'rRRrRsr', 'rRrsrsr', 'rRrrssr', '.rssss.', '..sss..']
    };
    const hexColor = hex => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];

    function makeSprite(rows, palette) {
        const h = rows.length;
        const w = rows[0].length;
        const size = Math.ceil(Math.hypot(w, h)) + 2;
        const colours = Object.fromEntries(Object.entries(palette).map(([key, value]) => [key, hexColor(value)]));
        const frames = [];
        for (let index = 0; index < ANGLES; index += 1) {
            const angle = (index / ANGLES) * TAU;
            const cos = Math.cos(angle), sin = Math.sin(angle);
            const frame = document.createElement('canvas');
            frame.width = frame.height = size;
            const context = frame.getContext('2d');
            const image = context.createImageData(size, size);
            for (let py = 0; py < size; py += 1) {
                for (let px = 0; px < size; px += 1) {
                    const dx = px + 0.5 - size / 2, dy = py + 0.5 - size / 2;
                    const sx = Math.floor(dx * cos + dy * sin + w / 2);
                    const sy = Math.floor(-dx * sin + dy * cos + h / 2);
                    if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
                    const colour = colours[rows[sy][sx]];
                    if (!colour) continue;
                    const offset = (py * size + px) * 4;
                    image.data[offset] = colour[0];
                    image.data[offset + 1] = colour[1];
                    image.data[offset + 2] = colour[2];
                    image.data[offset + 3] = 255;
                }
            }
            context.putImageData(image, 0, 0);
            frames.push(frame);
        }
        return { frames, size, length: w };
    }
    let sprites = null;
    function buildSprites() {
        sprites = {};
        for (const [name, rows] of Object.entries(SHAPES)) {
            const palette = name === 'rock' ? ROCK : ['drone', 'raider', 'carrier'].includes(name) ? FOE : FRIEND;
            sprites[name] = makeSprite(rows, palette);
        }
    }

    const CLASSES = {
        interceptor: { team: 0, speed: 1.2, turn: 0.1, hp: 3, radius: 4, weapon: 'pulse', cooldown: 16, label: 'INTERCEPTOR' },
        gunship: { team: 0, speed: 0.85, turn: 0.075, hp: 6, radius: 5, weapon: 'flak', cooldown: 34, label: 'GUNSHIP' },
        lancer: { team: 0, speed: 0.65, turn: 0.05, hp: 9, radius: 6, weapon: 'rail', cooldown: 110, label: 'LANCER' },
        missile: { team: 0, speed: 0.8, turn: 0.065, hp: 5, radius: 5, weapon: 'missile', cooldown: 70, label: 'MISSILE BOAT' },
        drone: { team: 1, speed: 1.0, turn: 0.09, hp: 3, radius: 4, weapon: 'plasma', cooldown: 34, label: 'DRONE' },
        raider: { team: 1, speed: 0.9, turn: 0.065, hp: 7, radius: 5, weapon: 'needle', cooldown: 50, label: 'RAIDER' },
        carrier: { team: 1, speed: 0.22, turn: 0.012, hp: 40, radius: 10, weapon: 'orb', cooldown: 70, label: 'CARRIER' }
    };

    /* --------------------------------------------------- policy network */
    const INPUTS = 16, HIDDEN = 8, OUTPUTS = 4;
    const W1 = Array.from({ length: HIDDEN }, () => new Float32Array(INPUTS));
    const B1 = new Float32Array([-0.2, -0.2, -0.4, -0.1, -0.1, -0.1, 0.9, 0]);
    const W2 = Array.from({ length: OUTPUTS }, () => new Float32Array(HIDDEN));
    const B2 = new Float32Array([-1.2, 0.3, -1.2, -2.0]);
    const hz = i => i, tg = i => 7 + i, HP = 14, BEARING = 15;
    [[hz(0), 0.5], [hz(1), 0.8], [hz(2), 1.1]].forEach(([i, w]) => { W1[0][i] = w; });
    [[hz(6), 0.5], [hz(5), 0.8], [hz(4), 1.1]].forEach(([i, w]) => { W1[1][i] = w; });
    [[hz(3), 1.6], [hz(2), 0.5], [hz(4), 0.5]].forEach(([i, w]) => { W1[2][i] = w; });
    [[tg(0), 0.7], [tg(1), 1.0], [tg(2), 1.2], [HP, -0.3]].forEach(([i, w]) => { W1[3][i] = w; });
    [[tg(6), 0.7], [tg(5), 1.0], [tg(4), 1.2], [HP, -0.3]].forEach(([i, w]) => { W1[4][i] = w; });
    [[tg(3), 1.8], [tg(2), 0.5], [tg(4), 0.5], [HP, -0.2]].forEach(([i, w]) => { W1[5][i] = w; });
    for (let i = 0; i < 7; i += 1) { W1[6][hz(i)] = -0.4; W1[6][tg(i)] = -0.25; }
    [[BEARING, 1.6], [HP, 0.2]].forEach(([i, w]) => { W1[7][i] = w; });
    W2[0].set([-2.0, 2.0, -0.6, 2.2, -2.2, 0, 0, -1.2]);
    W2[1].set([0, 0, -2.2, 0.3, 0.3, 1.2, 1.5, 0.2]);
    W2[2].set([2.0, -2.0, 1.4, -2.2, 2.2, 0, 0, 1.2]);
    W2[3].set([0, 0, 0, 0.6, 0.6, 3.4, -0.6, 0]);
    const sigmoid = x => 1 / (1 + Math.exp(-x));

    function forward(ship) {
        for (let j = 0; j < HIDDEN; j += 1) {
            let sum = B1[j];
            const row = W1[j];
            for (let i = 0; i < INPUTS; i += 1) sum += row[i] * ship.input[i];
            ship.hidden[j] = Math.tanh(sum);
        }
        for (let k = 0; k < OUTPUTS; k += 1) {
            let sum = B2[k];
            const row = W2[k];
            for (let j = 0; j < HIDDEN; j += 1) sum += row[j] * ship.hidden[j];
            ship.output[k] = sigmoid(sum);
        }
    }

    /* ------------------------------------------------------- the world */
    let world = null;
    let ships = [];
    let bullets = [];
    let beams = [];
    let particles = [];
    let rings = [];
    let rocks = [];
    let stars = [];
    let focus = null;
    let step = 0;
    let kills = 0;
    let losses = 0;
    let serial = 0;
    const rewardHistory = [];

    function spawnShip(type, x, y, angle) {
        const spec = CLASSES[type];
        const ship = {
            id: serial += 1, type, spec, team: spec.team, x, y, a: angle, vx: 0, vy: 0, hp: spec.hp, cool: (random() * spec.cooldown) | 0,
            charge: 0, burst: 0, hurt: 0, phase: random() * TAU,
            input: new Float32Array(INPUTS), hidden: new Float32Array(HIDDEN), output: new Float32Array(OUTPUTS),
            rays: Array.from({ length: 7 }, () => ({ d: 0, hit: 0 }))
        };
        ships.push(ship);
        return ship;
    }

    function edgePoint() {
        const side = random() < 0.5;
        return side ? { x: random() * world.W, y: random() < 0.5 ? -6 : world.H + 6 } : { x: random() < 0.5 ? -6 : world.W + 6, y: random() * world.H };
    }

    function populate() {
        ships = [];
        bullets = [];
        beams = [];
        particles = [];
        rings = [];
        const { W, H } = world;
        const fleet = ['interceptor', 'interceptor', 'interceptor', 'gunship', 'gunship', 'lancer', 'missile', 'missile'];
        if (W > 360) fleet.push('interceptor', 'gunship');
        fleet.forEach(type => spawnShip(type, W * (0.5 + random() * 0.35), H * (0.2 + random() * 0.45), Math.PI + (random() - 0.5)));
        const swarm = W > 360 ? 11 : 8;
        for (let i = 0; i < swarm; i += 1) spawnShip(i % 4 === 3 ? 'raider' : 'drone', W * (0.35 + random() * 0.45), H * (0.15 + random() * 0.6), random() * TAU);
        spawnShip('carrier', W * 0.3, H * 0.3, 0.1);
        rocks = Array.from({ length: Math.max(4, Math.round(W / 80)) }, () => ({
            x: random() * W, y: random() * H, vx: (random() - 0.5) * 0.18, vy: (random() - 0.5) * 0.12, a: random() * TAU, spin: (random() - 0.5) * 0.02, r: 3.5
        }));
        stars = Array.from({ length: Math.round((W * H) / 520) }, () => ({ x: random() * W, y: random() * H, z: random() < 0.7 ? 0.04 : 0.11, b: random() }));
        pickFocus();
    }

    function pickFocus() {
        const fleet = ships.filter(ship => ship.team === 0);
        const visible = fleet.filter(ship => ship.x > world.W * 0.5 && ship.x < world.W * 0.92 && ship.y > world.H * 0.16 && ship.y < world.H * 0.6);
        const pool = visible.length ? visible : fleet;
        focus = pool.length ? pool[(random() * pool.length) | 0] : null;
    }

    function nearestEnemy(ship, range = Infinity) {
        let best = null;
        let bestDistance = range;
        for (const other of ships) {
            if (other.team === ship.team) continue;
            const distance = Math.hypot(wrapDelta(other.x - ship.x, world.W), wrapDelta(other.y - ship.y, world.H));
            if (distance < bestDistance) { bestDistance = distance; best = other; }
        }
        return best ? { ship: best, distance: bestDistance } : null;
    }

    function burst(x, y, colours, count, speed = 1) {
        for (let i = 0; i < count; i += 1) {
            const angle = random() * TAU;
            const velocity = (0.3 + random()) * speed;
            particles.push({ x, y, vx: Math.cos(angle) * velocity, vy: Math.sin(angle) * velocity, life: 12 + random() * 16, colour: colours[(random() * colours.length) | 0], drag: 0.9 });
        }
    }

    function explode(ship) {
        const big = ship.type === 'carrier';
        const palette = ship.team === 0 ? ['#fff3c4', '#ffb15c', '#ff6a3d', '#eee8da'] : ['#7dff9a', '#ff4fd8', '#d7ffe0', '#ffffff'];
        burst(ship.x, ship.y, palette, big ? 60 : 18, big ? 1.8 : 1.1);
        burst(ship.x, ship.y, ['#5b5f66', '#8a8f96'], big ? 20 : 6, 0.6);
        rings.push({ x: ship.x, y: ship.y, r: 1, max: big ? 34 : 12, colour: ship.team === 0 ? '#ffb15c' : '#7dff9a' });
    }

    function damage(ship, amount, attacker) {
        ship.hp -= amount;
        ship.hurt = 6;
        if (ship.hp > 0) return;
        explode(ship);
        ships.splice(ships.indexOf(ship), 1);
        if (ship.team === 1) kills += attacker === 0 ? 1 : 0;
        else losses += 1;
        if (ship === focus) focus = null;
        const respawn = { type: ship.type, team: ship.team, wait: ship.type === 'carrier' ? 360 : ship.team === 0 ? 90 : 24 };
        world.queue.push(respawn);
    }

    /* ----------------------------------------------------------- sensing */
    function sense(ship) {
        ship.input.fill(0);
        const { W, H } = world;
        for (let r = 0; r < 7; r += 1) {
            const angle = ship.a + (r - 3) * RAY_SPREAD;
            const cos = Math.cos(angle), sin = Math.sin(angle);
            let hazard = 0, target = 0, reach = TARGET_RANGE, hit = 0;
            const probe = (dx, dy, width, range, kind) => {
                const along = dx * cos + dy * sin;
                if (along <= 0 || along >= range) return;
                if (Math.abs(dx * sin - dy * cos) > width) return;
                const value = 1 - along / range;
                if (kind === 'target') {
                    if (value > target) target = value;
                } else if (value > hazard) hazard = value;
                if (along < reach) { reach = along; hit = kind === 'target' ? 2 : 3; }
            };
            for (const other of ships) {
                if (other.team === ship.team) continue;
                probe(wrapDelta(other.x - ship.x, W), wrapDelta(other.y - ship.y, H), other.spec.radius + 3, TARGET_RANGE, 'target');
            }
            for (const bullet of bullets) {
                if (bullet.team === ship.team) continue;
                probe(wrapDelta(bullet.x - ship.x, W), wrapDelta(bullet.y - ship.y, H), 4, HAZARD_RANGE, 'hazard');
            }
            for (const rock of rocks) probe(wrapDelta(rock.x - ship.x, W), wrapDelta(rock.y - ship.y, H), 6, HAZARD_RANGE, 'hazard');
            ship.input[hz(r)] = hazard;
            ship.input[tg(r)] = target;
            ship.rays[r].d = hit ? reach : (r === 3 ? TARGET_RANGE : HAZARD_RANGE);
            ship.rays[r].hit = hit;
        }
        ship.input[HP] = 1 - ship.hp / ship.spec.hp;
        const enemy = nearestEnemy(ship);
        if (enemy) {
            const bearing = angleDelta(Math.atan2(wrapDelta(enemy.ship.y - ship.y, H), wrapDelta(enemy.ship.x - ship.x, W)) - ship.a);
            ship.input[BEARING] = Math.max(-1, Math.min(1, bearing / 1.2));
        }
    }

    /* ----------------------------------------------------------- weapons */
    function fire(ship) {
        const { a, x, y, spec, team } = ship;
        const nose = sprites[ship.type].length / 2;
        const nx = x + Math.cos(a) * nose, ny = y + Math.sin(a) * nose;
        const shot = (angle, speed, life, type, damageValue, extra = {}) => bullets.push({
            x: nx, y: ny, vx: Math.cos(angle) * speed + ship.vx * 0.5, vy: Math.sin(angle) * speed + ship.vy * 0.5,
            life, type, team, damage: damageValue, ...extra
        });
        switch (spec.weapon) {
        case 'pulse': {
            const px = -Math.sin(a) * 1.6, py = Math.cos(a) * 1.6;
            bullets.push({ x: nx + px, y: ny + py, vx: Math.cos(a) * 4.2, vy: Math.sin(a) * 4.2, life: 30, type: 'pulse', team, damage: 1 });
            bullets.push({ x: nx - px, y: ny - py, vx: Math.cos(a) * 4.2, vy: Math.sin(a) * 4.2, life: 30, type: 'pulse', team, damage: 1 });
            break;
        }
        case 'flak':
            [-0.2, 0, 0.2].forEach(offset => shot(a + offset + (random() - 0.5) * 0.06, 3.0, 24 + random() * 6, 'flak', 1));
            break;
        case 'rail':
            ship.charge = 26;
            return;
        case 'missile': {
            const target = nearestEnemy(ship, 200);
            shot(a + (random() - 0.5) * 0.5, 1.4, 130, 'missile', 3, { target: target?.ship || null, heading: a });
            break;
        }
        case 'plasma':
            shot(a, 2.2, 70, 'plasma', 1);
            break;
        case 'needle':
            ship.burst = 3;
            break;
        case 'orb':
            [-0.5, 0, 0.5].forEach(offset => shot(a + offset, 1.05, 170, 'orb', 2));
            break;
        default:
        }
        if (spec.weapon !== 'needle') burst(nx, ny, ship.team === 0 ? ['#fff3c4'] : ['#d7ffe0'], 2, 0.4);
    }

    function dischargeRail(ship) {
        const { W, H } = world;
        const cos = Math.cos(ship.a), sin = Math.sin(ship.a);
        let length = 170;
        let victim = null;
        for (const other of ships) {
            if (other.team === ship.team) continue;
            const dx = wrapDelta(other.x - ship.x, W), dy = wrapDelta(other.y - ship.y, H);
            const along = dx * cos + dy * sin;
            if (along <= 0 || along > length) continue;
            if (Math.abs(dx * sin - dy * cos) <= other.spec.radius + 1) { length = along; victim = other; }
        }
        beams.push({ x: ship.x, y: ship.y, a: ship.a, length, life: 9 });
        if (victim) {
            burst(ship.x + cos * length, ship.y + sin * length, ['#e9e2ff', '#c9b6ff', '#ffffff'], 10, 1.4);
            damage(victim, 4, ship.team);
        }
    }

    /* ------------------------------------------------------- simulation */
    function updateFleetShip(ship) {
        sense(ship);
        forward(ship);
        const [left, thrust, right, trigger] = ship.output;
        ship.a += (right - left) * ship.spec.turn;
        // Keep the fleet in the open right-centre theatre behind the copy.
        const homeX = world.W * 0.7 - ship.x, homeY = world.H * 0.38 - ship.y;
        const homeDistance = Math.hypot(homeX, homeY);
        if (homeDistance > world.W * 0.3) {
            ship.a += Math.max(-0.03, Math.min(0.03, angleDelta(Math.atan2(homeY, homeX) - ship.a))) * Math.min(1, (homeDistance - world.W * 0.3) / 40);
        }
        const power = (0.25 + 0.75 * thrust) * ship.spec.speed;
        ship.vx += (Math.cos(ship.a) * power - ship.vx) * 0.08;
        ship.vy += (Math.sin(ship.a) * power - ship.vy) * 0.08;
        if (ship.charge > 0) {
            ship.charge -= 1;
            ship.vx *= 0.9;
            ship.vy *= 0.9;
            if (ship.charge === 0) dischargeRail(ship);
        } else if (ship.cool <= 0 && trigger > 0.55) {
            fire(ship);
            ship.cool = ship.spec.cooldown;
        }
    }

    function updateSwarmShip(ship) {
        const enemy = nearestEnemy(ship);
        const { W, H } = world;
        if (ship.type === 'carrier') {
            ship.a += Math.sin(step * 0.004 + ship.phase) * 0.004;
            const drones = ships.filter(other => other.type === 'drone').length;
            if (step % 150 === 0 && drones < 12) {
                const launched = spawnShip('drone', ship.x + Math.cos(ship.a + Math.PI / 2) * 8, ship.y + Math.sin(ship.a + Math.PI / 2) * 8, ship.a + Math.PI / 2);
                burst(launched.x, launched.y, ['#7dff9a'], 6, 0.5);
            }
        } else if (enemy) {
            const dx = wrapDelta(enemy.ship.x - ship.x, W), dy = wrapDelta(enemy.ship.y - ship.y, H);
            const swirl = ship.type === 'drone' ? Math.sin(step * 0.05 + ship.phase) * 0.7 : 0;
            ship.a += Math.max(-ship.spec.turn, Math.min(ship.spec.turn, angleDelta(Math.atan2(dy, dx) + swirl - ship.a)));
        }
        const speed = ship.spec.speed * (ship.type === 'drone' ? 0.85 + 0.3 * Math.sin(step * 0.07 + ship.phase) : 1);
        ship.vx += (Math.cos(ship.a) * speed - ship.vx) * 0.06;
        ship.vy += (Math.sin(ship.a) * speed - ship.vy) * 0.06;
        if (ship.burst > 0 && step % 3 === 0) {
            const a = ship.a;
            bullets.push({ x: ship.x + Math.cos(a) * 5, y: ship.y + Math.sin(a) * 5, vx: Math.cos(a) * 3.6, vy: Math.sin(a) * 3.6, life: 36, type: 'needle', team: 1, damage: 1 });
            ship.burst -= 1;
        }
        if (ship.cool <= 0 && enemy && enemy.distance < (ship.type === 'carrier' ? 140 : 95)) {
            const bearing = Math.abs(angleDelta(Math.atan2(wrapDelta(enemy.ship.y - ship.y, H), wrapDelta(enemy.ship.x - ship.x, W)) - ship.a));
            if (bearing < (ship.type === 'carrier' ? 1.4 : 0.35)) {
                fire(ship);
                ship.cool = ship.spec.cooldown;
            }
        }
    }

    function updateBullets() {
        const { W, H } = world;
        for (let i = bullets.length - 1; i >= 0; i -= 1) {
            const bullet = bullets[i];
            if (bullet.type === 'missile') {
                if (!bullet.target || !ships.includes(bullet.target)) bullet.target = nearestEnemy({ x: bullet.x, y: bullet.y, team: bullet.team }, 160)?.ship || null;
                if (bullet.target) {
                    const desired = Math.atan2(wrapDelta(bullet.target.y - bullet.y, H), wrapDelta(bullet.target.x - bullet.x, W));
                    bullet.heading += Math.max(-0.09, Math.min(0.09, angleDelta(desired - bullet.heading)));
                }
                const speed = Math.min(2.8, Math.hypot(bullet.vx, bullet.vy) + 0.05);
                bullet.vx = Math.cos(bullet.heading) * speed;
                bullet.vy = Math.sin(bullet.heading) * speed;
                if (step % 2 === 0) particles.push({ x: bullet.x, y: bullet.y, vx: (random() - 0.5) * 0.2, vy: (random() - 0.5) * 0.2, life: 22, colour: '#6a6e75', drag: 0.96 });
            } else if (bullet.type === 'orb') {
                const enemy = nearestEnemy({ x: bullet.x, y: bullet.y, team: 1 }, 90);
                if (enemy) {
                    bullet.vx += wrapDelta(enemy.ship.x - bullet.x, W) * 0.0006;
                    bullet.vy += wrapDelta(enemy.ship.y - bullet.y, H) * 0.0006;
                }
            }
            bullet.x = wrap(bullet.x + bullet.vx, W);
            bullet.y = wrap(bullet.y + bullet.vy, H);
            bullet.life -= 1;
            let spent = bullet.life <= 0;
            if (!spent) {
                for (const ship of ships) {
                    if (ship.team === bullet.team) continue;
                    if (Math.abs(wrapDelta(ship.x - bullet.x, W)) > ship.spec.radius + 1 || Math.abs(wrapDelta(ship.y - bullet.y, H)) > ship.spec.radius + 1) continue;
                    burst(bullet.x, bullet.y, bullet.team === 0 ? ['#fff3c4', '#ffb15c'] : ['#d7ffe0', '#ff4fd8'], bullet.type === 'missile' ? 12 : 4, bullet.type === 'missile' ? 1.2 : 0.7);
                    if (bullet.type === 'missile') rings.push({ x: bullet.x, y: bullet.y, r: 1, max: 8, colour: '#ffb15c' });
                    damage(ship, bullet.damage, bullet.team);
                    spent = true;
                    break;
                }
            }
            if (!spent) {
                for (const rock of rocks) {
                    if (Math.hypot(wrapDelta(rock.x - bullet.x, W), wrapDelta(rock.y - bullet.y, H)) > rock.r) continue;
                    burst(bullet.x, bullet.y, ['#c9bfae', '#8f877b'], 3, 0.6);
                    spent = true;
                    break;
                }
            }
            if (spent) {
                if (bullet.type === 'flak' && bullet.life <= 0) burst(bullet.x, bullet.y, ['#ffd166', '#fff3c4'], 4, 0.6);
                bullets.splice(i, 1);
            }
        }
    }

    function tick() {
        step += 1;
        const { W, H } = world;
        for (const ship of ships.slice()) {
            if (!ships.includes(ship)) continue;
            if (ship.cool > 0) ship.cool -= 1;
            if (ship.hurt > 0) ship.hurt -= 1;
            if (ship.team === 0) updateFleetShip(ship);
            else updateSwarmShip(ship);
            ship.x = wrap(ship.x + ship.vx, W);
            ship.y = wrap(ship.y + ship.vy, H);
            if (ship.team === 0 && step % 2 === 0 && ship.output[1] > 0.5) {
                const back = ship.a + Math.PI;
                particles.push({ x: ship.x + Math.cos(back) * 4, y: ship.y + Math.sin(back) * 4, vx: Math.cos(back) * 0.5, vy: Math.sin(back) * 0.5, life: 8, colour: '#ffb15c', drag: 0.85 });
            }
        }
        updateBullets();
        for (const rock of rocks) {
            rock.x = wrap(rock.x + rock.vx, W);
            rock.y = wrap(rock.y + rock.vy, H);
            rock.a += rock.spin;
        }
        for (const star of stars) star.x = wrap(star.x - star.z, W);
        beams = beams.filter(beam => (beam.life -= 1) > 0);
        rings = rings.filter(ring => (ring.r += ring.max / 10) < ring.max);
        particles = particles.filter(p => {
            p.x += p.vx;
            p.y += p.vy;
            p.vx *= p.drag;
            p.vy *= p.drag;
            return (p.life -= 1) > 0;
        });
        if (particles.length > 900) particles.splice(0, particles.length - 900);
        for (let i = world.queue.length - 1; i >= 0; i -= 1) {
            const entry = world.queue[i];
            if ((entry.wait -= 1) > 0) continue;
            world.queue.splice(i, 1);
            if (entry.team === 0) {
                const dock = world.dock;
                const ship = spawnShip(entry.type, dock.x, dock.y, Math.PI + (random() - 0.5));
                burst(ship.x, ship.y, ['#86e6ff', '#eee8da'], 8, 0.5);
            } else {
                const point = entry.type === 'carrier' ? edgePoint() : { x: W + 4, y: H * (0.15 + random() * 0.6) };
                spawnShip(entry.type, point.x, point.y, Math.atan2(H / 2 - point.y, W / 2 - point.x));
            }
        }
        if (!focus || !ships.includes(focus) || (step % 300 === 0) || focus.x < world.W * 0.44 || focus.y > world.H * 0.72) pickFocus();
        if (step % 30 === 0) {
            rewardHistory.push(kills - losses * 0.5);
            if (rewardHistory.length > 40) rewardHistory.shift();
        }
    }

    /* ----------------------------------------------------- backdrop art */
    function paintBackdrop(W, H) {
        const image = new ImageData(W, H);
        const data = image.data;
        const planet = { x: W * 0.9, y: H * 0.06, r: H * 0.2 };
        const ring = { a: planet.r * 1.8, b: planet.r * 0.32, tilt: 0.42 };
        const moon = { x: W * 0.52, y: H * 0.16, r: Math.max(4, H * 0.028) };
        const light = [-0.62, -0.55, 0.56];
        const cosT = Math.cos(ring.tilt), sinT = Math.sin(ring.tilt);
        const put = (offset, colour) => { data[offset] = colour[0]; data[offset + 1] = colour[1]; data[offset + 2] = colour[2]; data[offset + 3] = 255; };
        for (let py = 0; py < H; py += 1) {
            for (let px = 0; px < W; px += 1) {
                const offset = (py * W + px) * 4;
                const t = py / H;
                let colour = [5 + t * 4, 7 + t * 5, 12 + t * 10];
                const n = fbm(px * 0.018, py * 0.03, 7);
                const m = fbm(px * 0.012 + 30, py * 0.02, 11);
                const nebula = Math.max(0, (n - 0.48) / 0.4);
                const level = Math.floor(nebula * 4 + bayer(px, py) - 0.5) / 4;
                if (level > 0) {
                    const tint = m > 0.5 ? [36, 92, 112] : [74, 44, 104];
                    colour = colour.map((value, index) => value + tint[index] * level * 0.55);
                }
                const star = hash(px, py, 3);
                if (star > 0.9965) colour = star > 0.9993 ? [240, 236, 226] : [130, 140, 160];

                const rx = px - planet.x, ry = py - planet.y;
                const lx = rx * cosT + ry * sinT, ly = -rx * sinT + ry * cosT;
                const ringValue = Math.hypot(lx / ring.a, ly / ring.b);
                const inRing = ringValue > 0.78 && ringValue < 1.0 && Math.abs(Math.sin(ringValue * 46)) > 0.25;
                const distance = Math.hypot(rx, ry);
                const ringColour = () => {
                    const band = Math.sin(ringValue * 60) > 0 ? [190, 168, 132] : [140, 122, 98];
                    return ringValue > 0.9 ? band.map(v => v * 0.8) : band;
                };
                if (inRing && ly < 0 && distance > planet.r) put(offset, ringColour());
                else if (distance <= planet.r) {
                    const nx = rx / planet.r, ny = ry / planet.r;
                    const nz = Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny));
                    const shade = Math.max(0, nx * light[0] + ny * light[1] + nz * light[2]);
                    const bands = fbm(px * 0.01, py * 0.11 + nx * 2, 23);
                    let base = bands > 0.62 ? [214, 164, 108] : bands > 0.46 ? [178, 112, 72] : bands > 0.34 ? [140, 82, 58] : [196, 140, 92];
                    if (Math.abs(ny + 0.32 - Math.sin(nx * 4) * 0.04) < 0.035) base = [230, 196, 140];
                    const lit = Math.floor((0.12 + shade * 1.05) * 5 + bayer(px, py) - 0.5) / 5;
                    colour = base.map(v => v * Math.max(0.06, Math.min(1.05, lit)));
                    if (distance > planet.r - 1.5 && shade > 0.15) colour = [130, 200, 226];
                    if (inRing && ly > 0) colour = ringColour();
                    put(offset, colour);
                    continue;
                } else if (inRing) put(offset, ringColour());
                else if (distance < planet.r + 3) {
                    const glow = 1 - (distance - planet.r) / 3;
                    colour = colour.map((value, index) => value + [60, 130, 160][index] * glow * 0.6);
                    put(offset, colour);
                } else {
                    const md = Math.hypot(px - moon.x, py - moon.y);
                    if (md <= moon.r) {
                        const shade = 1 - (px - moon.x + moon.r * 0.6) / (moon.r * 2);
                        const lit = Math.floor(Math.max(0, shade) * 4 + bayer(px, py) - 0.5) / 4;
                        colour = [178, 174, 166].map(v => v * Math.max(0.1, lit));
                        if (hash(px, py, 9) > 0.86) colour = colour.map(v => v * 0.7);
                    }
                    put(offset, colour);
                }
                if (data[offset + 3] === 0) put(offset, colour);
            }
        }
        const backdrop = document.createElement('canvas');
        backdrop.width = W;
        backdrop.height = H;
        const context = backdrop.getContext('2d');
        context.putImageData(image, 0, 0);

        // Orbital dock: truss, solar arrays, habitat ring, hazard collar.
        const sx = Math.round(W * 0.6), sy = Math.round(H * 0.62);
        const fill = (colour, x, y, w, h) => { context.fillStyle = colour; context.fillRect(sx + x, sy + y, w, h); };
        fill('#8a8f96', -30, 0, 60, 1);
        fill('#5d6168', -30, 2, 60, 1);
        for (let x = -30; x < 30; x += 4) { fill('#71757c', x, 1, 1, 1); fill('#71757c', x + 2, 1, 1, 1); }
        for (const side of [-1, 1]) {
            for (const offset of [18, 28]) {
                const px = side > 0 ? offset - 4 : -offset - 4;
                fill('#26446e', px, -9, 9, 7);
                fill('#26446e', px, 5, 9, 7);
                for (let gx = 0; gx < 9; gx += 3) { fill('#5f8cc4', px + gx, -9, 1, 7); fill('#5f8cc4', px + gx, 5, 1, 7); }
                fill('#5f8cc4', px, -6, 9, 1);
                fill('#5f8cc4', px, 8, 9, 1);
                fill('#8a8f96', px + 4, -2, 1, 7);
            }
        }
        for (let angle = 0; angle < TAU; angle += 0.09) {
            fill('#b8b2a4', Math.round(Math.cos(angle) * 11), Math.round(Math.sin(angle) * 6) + 1, 1, 1);
        }
        fill('#eee8da', -5, -4, 10, 10);
        fill('#c9c2b2', -5, 4, 10, 2);
        fill('#ff6a3d', -5, -1, 10, 2);
        for (let x = -5; x < 5; x += 2) fill('#1b1d20', x, 2, 1, 1);
        for (let x = -6; x < 6; x += 1) fill((x & 1) ? '#ff6a3d' : '#1b1d20', x + 1, -6, 1, 2);
        fill('#eee8da', -1, -12, 2, 6);
        fill('#8a8f96', 0, -16, 1, 4);
        fill('#eee8da', 5, -1, 6, 3);
        fill('#86e6ff', 11, 0, 1, 1);
        return { backdrop, dock: { x: sx + 12, y: sy + 1 }, station: { x: sx, y: sy }, planet };
    }

    /* ------------------------------------------------------------ frame */
    let frame = null;
    let ctx = null;
    const bulletStyle = {
        pulse: { core: '#ffffff', tail: '#9ff3ff', length: 3 },
        flak: { core: '#ffe9a8', tail: '#ffb547', length: 1 },
        missile: { core: '#fff3c4', tail: '#ff6a3d', length: 2 },
        plasma: { core: '#e4ffe9', tail: '#4fe07a', length: 0 },
        needle: { core: '#ffd9f6', tail: '#ff4fd8', length: 3 },
        orb: { core: '#ffe8fb', tail: '#c23db0', length: 0 }
    };

    function pixel(x, y, colour, alpha = 1) {
        ctx.globalAlpha = alpha;
        ctx.fillStyle = colour;
        ctx.fillRect(Math.round(x), Math.round(y), 1, 1);
    }

    function drawSprite(name, x, y, angle) {
        const sprite = sprites[name];
        const index = ((Math.round((angle / TAU) * ANGLES) % ANGLES) + ANGLES) % ANGLES;
        ctx.drawImage(sprite.frames[index], Math.round(x - sprite.size / 2), Math.round(y - sprite.size / 2));
    }

    function drawFrame() {
        const { W, H } = world;
        ctx.globalAlpha = 1;
        ctx.drawImage(world.backdrop, 0, 0);
        for (const star of stars) {
            const twinkle = 0.45 + 0.55 * Math.sin(step * 0.05 + star.b * 40);
            pixel(star.x, star.y, star.z > 0.1 ? '#f2eee4' : '#8d97ab', star.z > 0.1 ? 0.6 + 0.4 * twinkle : 0.35 * twinkle);
        }
        const blink = step % 40 < 6;
        pixel(world.station.x - 30, world.station.y, blink ? '#ff5b4f' : '#5a2724');
        pixel(world.station.x + 30, world.station.y, step % 40 > 20 && step % 40 < 26 ? '#7dff9a' : '#24502f');
        pixel(world.station.x, world.station.y - 16, step % 60 < 4 ? '#ffffff' : '#7a7d82');

        for (const rock of rocks) drawSprite('rock', rock.x, rock.y, rock.a);

        for (const p of particles) {
            pixel(p.x, p.y, p.colour, Math.min(1, p.life / 10) * (p.colour === '#6a6e75' ? 0.6 : 1));
        }

        for (const beam of beams) {
            const fade = beam.life / 9;
            const cos = Math.cos(beam.a), sin = Math.sin(beam.a);
            for (let d = 4; d < beam.length; d += 1) {
                const x = beam.x + cos * d, y = beam.y + sin * d;
                pixel(x, y, '#ffffff', fade);
                pixel(x - sin, y + cos, '#c9b6ff', fade * 0.7);
                pixel(x + sin, y - cos, '#c9b6ff', fade * 0.7);
            }
        }

        for (const bullet of bullets) {
            const style = bulletStyle[bullet.type];
            const speed = Math.hypot(bullet.vx, bullet.vy) || 1;
            const ux = bullet.vx / speed, uy = bullet.vy / speed;
            for (let i = 1; i <= style.length; i += 1) pixel(bullet.x - ux * i, bullet.y - uy * i, style.tail, 1 - i / (style.length + 1));
            if (bullet.type === 'plasma' || bullet.type === 'orb') {
                const radius = bullet.type === 'orb' ? 2 : 1;
                ctx.globalAlpha = 0.35;
                ctx.fillStyle = style.tail;
                ctx.fillRect(Math.round(bullet.x) - radius - 1, Math.round(bullet.y) - radius - 1, radius * 2 + 3, radius * 2 + 3);
                ctx.globalAlpha = 1;
                ctx.fillRect(Math.round(bullet.x) - radius, Math.round(bullet.y) - radius, radius * 2 + 1, radius * 2 + 1);
            }
            pixel(bullet.x, bullet.y, style.core);
        }

        if (focus && ships.includes(focus)) {
            const colours = ['#3a3d42', '#9aa3ad', '#7fe3ff', '#ff6e6e'];
            for (let r = 0; r < 7; r += 1) {
                const ray = focus.rays[r];
                const angle = focus.a + (r - 3) * RAY_SPREAD;
                const cos = Math.cos(angle), sin = Math.sin(angle);
                const colour = colours[ray.hit] || '#9aa3ad';
                for (let d = 6; d < ray.d; d += 3) pixel(focus.x + cos * d, focus.y + sin * d, ray.hit ? colour : '#9aa3ad', ray.hit ? 0.85 : 0.25);
                if (ray.hit) {
                    const x = Math.round(focus.x + cos * ray.d), y = Math.round(focus.y + sin * ray.d);
                    ctx.globalAlpha = 1;
                    ctx.fillStyle = colour;
                    ctx.fillRect(x - 1, y, 3, 1);
                    ctx.fillRect(x, y - 1, 1, 3);
                }
            }
        }

        for (const ship of ships) {
            ctx.globalAlpha = 1;
            if (ship.team === 0) {
                const back = ship.a + Math.PI;
                const length = sprites[ship.type].length / 2 + 1;
                const flame = 1 + Math.round(ship.output[1] * 3 * (0.7 + 0.3 * Math.sin(step * 0.9 + ship.phase)));
                for (let i = 0; i < flame; i += 1) pixel(ship.x + Math.cos(back) * (length + i), ship.y + Math.sin(back) * (length + i), i === 0 ? '#fff3c4' : '#ffb15c', 1 - i / (flame + 1));
            } else if (ship.type !== 'carrier') {
                pixel(ship.x, ship.y, '#7dff9a', 0.4 + 0.4 * Math.sin(step * 0.3 + ship.phase));
            }
            ctx.globalAlpha = 1;
            drawSprite(ship.type, ship.x, ship.y, ship.a);
            if (ship.hurt > 0 && ship.hurt % 2 === 0) pixel(ship.x, ship.y, '#ffffff');
            if (ship.charge > 0) {
                const nose = sprites[ship.type].length / 2;
                const glow = 1 - ship.charge / 26;
                const x = Math.round(ship.x + Math.cos(ship.a) * nose), y = Math.round(ship.y + Math.sin(ship.a) * nose);
                ctx.globalAlpha = 0.4 + glow * 0.6;
                ctx.fillStyle = '#c9b6ff';
                const size = 1 + Math.round(glow * 2);
                ctx.fillRect(x - size, y - size, size * 2 + 1, size * 2 + 1);
                pixel(x, y, '#ffffff');
            }
            if (ship.type === 'carrier') {
                const hp = Math.max(0, ship.hp / ship.spec.hp);
                ctx.globalAlpha = 0.9;
                ctx.fillStyle = '#2a2c2e';
                ctx.fillRect(Math.round(ship.x) - 10, Math.round(ship.y) - 14, 20, 1);
                ctx.fillStyle = '#7dff9a';
                ctx.fillRect(Math.round(ship.x) - 10, Math.round(ship.y) - 14, Math.max(1, Math.round(hp * 20)), 1);
            }
        }

        for (const ring of rings) {
            const alpha = 1 - ring.r / ring.max;
            const count = Math.max(10, Math.round(ring.r * 4));
            for (let i = 0; i < count; i += 1) {
                const angle = (i / count) * TAU;
                pixel(ring.x + Math.cos(angle) * ring.r, ring.y + Math.sin(angle) * ring.r, ring.colour, alpha);
            }
            if (ring.r < 4) {
                ctx.globalAlpha = alpha;
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(Math.round(ring.x) - 2, Math.round(ring.y) - 2, 5, 5);
            }
        }
        ctx.globalAlpha = 1;
    }

    /* ---------------------------------------------------------- overlay */
    const svgNode = (tag, attributes = {}, parent = null) => {
        const node = document.createElementNS(SVG, tag);
        for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
        if (parent) parent.appendChild(node);
        return node;
    };
    let overlayRefs = null;

    function buildOverlay() {
        if (!overlay) return;
        overlay.replaceChildren();
        overlay.setAttribute('viewBox', `0 0 ${world.W} ${world.H}`);
        const links = [0, 1].map(() => ({ line: svgNode('line', { class: 'wo-link' }, overlay), text: svgNode('text', { class: 'wo-cyan' }, overlay) }));
        const hostiles = Array.from({ length: 5 }, () => {
            const group = svgNode('g', { class: 'wo-hostile', visibility: 'hidden' }, overlay);
            svgNode('path', { d: 'M-6 -3 V-6 H-3 M3 -6 H6 V-3 M6 3 V6 H3 M-3 6 H-6 V3' }, group);
            const text = svgNode('text', { x: 8, y: 1.5, class: 'wo-accent' }, group);
            return { group, text };
        });
        const reticle = svgNode('g', { class: 'wo-reticle' }, overlay);
        ['M-9 -5 V-9 H-5', 'M5 -9 H9 V-5', 'M9 5 V9 H5', 'M-5 9 H-9 V5'].forEach(d => svgNode('path', { d }, reticle));
        const label = svgNode('text', { x: 11, y: -5 }, reticle);
        const detail = svgNode('text', { x: 11, y: -1.6, class: 'wo-accent' }, reticle);
        const dock = svgNode('text', { x: world.station.x - 30, y: world.station.y - 13, class: 'wo-dim' }, overlay);
        dock.textContent = 'INX ORBITAL DOCK · RESPAWN';
        overlayRefs = { links, hostiles, reticle, label, detail, cache: new Map() };
    }

    function attr(node, key, value) {
        let entry = overlayRefs.cache.get(node);
        if (!entry) { entry = {}; overlayRefs.cache.set(node, entry); }
        if (entry[key] === value) return;
        entry[key] = value;
        node.setAttribute(key, value);
    }
    const text = (node, value) => { if (node.textContent !== value) node.textContent = value; };

    function actionName(output) {
        const [left, thrust, right, trigger] = output;
        if (trigger > 0.55) return 'FIRE';
        if (right - left > 0.2) return 'TURN R';
        if (left - right > 0.2) return 'TURN L';
        return thrust > 0.5 ? 'BURN' : 'COAST';
    }

    function updateOverlay() {
        if (!overlayRefs) return;
        const { W, H } = world;
        const { reticle, label, detail, links, hostiles } = overlayRefs;
        const live = focus && ships.includes(focus);
        attr(reticle, 'visibility', live ? 'visible' : 'hidden');
        const enemies = ships.filter(ship => ship.team === 1);
        if (live) {
            attr(reticle, 'transform', `translate(${focus.x.toFixed(1)} ${focus.y.toFixed(1)})`);
            text(label, `INX-${String(focus.id).padStart(2, '0')} · ${focus.spec.label}`);
            text(detail, `π → ${actionName(focus.output)} ${Math.max(...focus.output).toFixed(2)}`);
            const ranked = enemies.map(ship => ({ ship, dx: wrapDelta(ship.x - focus.x, W), dy: wrapDelta(ship.y - focus.y, H) }))
                .map(entry => ({ ...entry, d: Math.hypot(entry.dx, entry.dy) }))
                .filter(entry => entry.d < TARGET_RANGE * 1.3)
                .sort((a, b) => a.d - b.d);
            const total = ranked.slice(0, 2).reduce((sum, entry) => sum + Math.exp(-entry.d / 40), 0);
            links.forEach((link, index) => {
                const entry = ranked[index];
                const visible = Boolean(entry);
                attr(link.line, 'visibility', visible ? 'visible' : 'hidden');
                attr(link.text, 'visibility', visible ? 'visible' : 'hidden');
                if (!entry) return;
                const x2 = focus.x + entry.dx, y2 = focus.y + entry.dy;
                attr(link.line, 'x1', focus.x.toFixed(1));
                attr(link.line, 'y1', focus.y.toFixed(1));
                attr(link.line, 'x2', x2.toFixed(1));
                attr(link.line, 'y2', y2.toFixed(1));
                attr(link.text, 'x', (x2 + 5).toFixed(1));
                attr(link.text, 'y', (y2 + 7).toFixed(1));
                text(link.text, `att ${(Math.exp(-entry.d / 40) / total).toFixed(2)}`);
            });
        } else {
            links.forEach(link => { attr(link.line, 'visibility', 'hidden'); attr(link.text, 'visibility', 'hidden'); });
        }
        const tagged = enemies.filter(ship => ship.x > W * 0.46 && ship.x < W * 0.68 && ship.y > H * 0.14 && ship.y < H * 0.9).sort((a, b) => b.spec.hp - a.spec.hp).slice(0, 3);
        hostiles.forEach((slot, index) => {
            const ship = tagged[index];
            attr(slot.group, 'visibility', ship ? 'visible' : 'hidden');
            if (!ship) return;
            attr(slot.group, 'transform', `translate(${ship.x.toFixed(1)} ${ship.y.toFixed(1)})`);
            text(slot.text, ship.type === 'carrier' ? `CARRIER · HP ${Math.max(0, ship.hp)}` : ship.spec.label);
        });
    }

    /* ------------------------------------------------- inspector panel */
    let panelRefs = null;

    function buildPanel() {
        if (!panel) return;
        const graph = panel.querySelector('[data-nn-graph]');
        if (!graph) return;
        graph.replaceChildren();
        const inputY = i => 14 + i * 11.6;
        const hiddenY = j => 26 + j * 21;
        const outputY = k => 40 + k * 40;
        const X = [52, 188, 304];
        const edges = [];
        const edgeLayer = svgNode('g', {}, graph);
        for (let j = 0; j < HIDDEN; j += 1) {
            for (let i = 0; i < INPUTS; i += 1) {
                const weight = W1[j][i];
                if (!weight) continue;
                edges.push({ node: svgNode('line', { class: `nn-edge ${weight > 0 ? 'pos' : 'neg'}`, x1: X[0], y1: inputY(i), x2: X[1], y2: hiddenY(j), 'stroke-opacity': 0.08 }, edgeLayer), layer: 0, from: i, weight });
            }
        }
        for (let k = 0; k < OUTPUTS; k += 1) {
            for (let j = 0; j < HIDDEN; j += 1) {
                const weight = W2[k][j];
                if (!weight) continue;
                edges.push({ node: svgNode('line', { class: `nn-edge ${weight > 0 ? 'pos' : 'neg'}`, x1: X[1], y1: hiddenY(j), x2: X[2], y2: outputY(k), 'stroke-opacity': 0.08 }, edgeLayer), layer: 1, from: j, weight });
            }
        }
        const node = (x, y, r) => {
            svgNode('circle', { class: 'nn-node-ring', cx: x, cy: y, r: r + 1.4 }, graph);
            return svgNode('circle', { class: 'nn-node pos', cx: x, cy: y, r, 'fill-opacity': 0.1 }, graph);
        };
        const names = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 't0', 't1', 't2', 't3', 't4', 't5', 't6', 'HP', 'β'];
        const inputs = names.map((name, i) => {
            const label = svgNode('text', { x: 8, y: inputY(i) + 2.6 }, graph);
            label.textContent = name;
            return node(X[0], inputY(i), 3.4);
        });
        const hidden = Array.from({ length: HIDDEN }, (_, j) => node(X[1], hiddenY(j), 4.4));
        const outputLabels = ['◀ TURN L', '▲ THRUST', 'TURN R ▶', '◆ FIRE'];
        const outputs = outputLabels.map((name, k) => node(X[2], outputY(k), 5.6));
        const outputText = outputLabels.map((name, k) => {
            const label = svgNode('text', { x: X[2] + 11, y: outputY(k) + 3, class: 'nn-out-label' }, graph);
            label.textContent = name;
            return label;
        });
        [['IN · 14 RAYS + HP + β', 8], ['HIDDEN · 8 · tanh', X[1] - 34], ['OUT · 4 · σ', X[2] - 14]].forEach(([value, x]) => {
            const label = svgNode('text', { x, y: 210, class: 'nn-layer-label' }, graph);
            label.textContent = value;
        });
        panelRefs = {
            edges, inputs, hidden, outputs, outputText,
            bars: [...panel.querySelectorAll('[data-nn-bar]')],
            values: [...panel.querySelectorAll('[data-nn-value]')],
            agent: panel.querySelector('[data-nn-agent]'),
            step: panel.querySelector('[data-nn-step]'),
            reward: panel.querySelector('[data-nn-reward]'),
            resets: panel.querySelector('[data-nn-resets]'),
            spark: panel.querySelector('[data-nn-spark]'),
            cache: new WeakMap()
        };
    }

    function panelAttr(node, key, value) {
        if (!node) return;
        let entry = panelRefs.cache.get(node);
        if (!entry) { entry = {}; panelRefs.cache.set(node, entry); }
        if (entry[key] === value) return;
        entry[key] = value;
        node.setAttribute(key, value);
    }

    function updatePanel() {
        if (!panelRefs || !focus) return;
        const ship = focus;
        const refs = panelRefs;
        refs.edges.forEach(edge => {
            const source = edge.layer === 0 ? ship.input[edge.from] : ship.hidden[edge.from];
            panelAttr(edge.node, 'stroke-opacity', (0.07 + Math.min(0.9, Math.abs(edge.weight * source) * 0.75)).toFixed(2));
        });
        const paint = (circle, value) => {
            panelAttr(circle, 'class', `nn-node ${value >= 0 ? 'pos' : 'neg'}`);
            panelAttr(circle, 'fill-opacity', (0.08 + Math.min(1, Math.abs(value)) * 0.92).toFixed(2));
        };
        refs.inputs.forEach((circle, i) => paint(circle, ship.input[i]));
        refs.hidden.forEach((circle, j) => paint(circle, ship.hidden[j]));
        refs.outputs.forEach((circle, k) => paint(circle, ship.output[k]));
        let best = 0;
        for (let k = 1; k < OUTPUTS; k += 1) if (ship.output[k] > ship.output[best]) best = k;
        refs.outputText.forEach((label, k) => panelAttr(label, 'class', `nn-out-label${k === best ? ' is-max' : ''}`));
        refs.bars.forEach((bar, k) => panelAttr(bar, 'width', (ship.output[k] * 100).toFixed(1)));
        refs.values.forEach((value, k) => text(value, ship.output[k].toFixed(2)));
        if (refs.agent) text(refs.agent, `INX-${String(ship.id).padStart(2, '0')} · HP ${(ship.hp / ship.spec.hp).toFixed(2)}`);
        if (refs.step) text(refs.step, String(step).padStart(6, '0'));
        if (refs.reward) text(refs.reward, String(kills));
        if (refs.resets) text(refs.resets, String(losses));
        if (refs.spark && rewardHistory.length > 1) {
            const min = Math.min(...rewardHistory);
            const span = Math.max(1, Math.max(...rewardHistory) - min);
            const points = rewardHistory.map((value, index) => `${((index / 39) * 120).toFixed(1)},${(22 - ((value - min) / span) * 20).toFixed(1)}`).join(' ');
            panelAttr(refs.spark, 'points', points);
        }
    }

    /* ------------------------------------------------------- GPU post */
    const postSource = `
precision mediump float;
varying vec2 vUv;
uniform sampler2D uScene;
uniform vec2 uSrc;
uniform float uTime;
float bayer2(vec2 a) { a = floor(a); return fract(dot(a, vec2(0.5, a.y * 0.75))); }
float bayer4(vec2 a) { return bayer2(0.5 * a) * 0.25 + bayer2(a); }
float hash(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
void main() {
    vec2 uv = vUv;
    vec2 dir = uv - 0.5;
    vec2 off = dir * 0.002 + vec2(0.35 / uSrc.x, 0.0);
    vec3 col = vec3(texture2D(uScene, uv + off).r, texture2D(uScene, uv).g, texture2D(uScene, uv - off).b);
    vec3 glow = vec3(0.0);
    for (int i = 0; i < 16; i++) {
        float a = float(i) * 0.3927;
        float r = 1.5 + mod(float(i), 4.0) * 1.6;
        vec3 q = texture2D(uScene, uv + vec2(cos(a), sin(a)) * r / uSrc).rgb;
        glow += q * smoothstep(0.82, 1.0, max(q.r, max(q.g, q.b)));
    }
    col += glow / 16.0 * 1.6;
    float th = bayer4(floor(uv * uSrc)) - 0.5;
    col = floor(col * 8.0 + th * 0.9 + 0.5) / 8.0;
    col *= mod(gl_FragCoord.y, 3.0) < 1.0 ? 0.72 : 1.0;
    col += (hash(gl_FragCoord.xy + fract(uTime * 13.0) * 97.0) - 0.5) * 0.05;
    col *= 1.0 - dot(dir, dir) * 0.7;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}`;
    const vertexSource = 'attribute vec2 aPos; varying vec2 vUv; void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }';

    let gl = canvas.getContext('webgl', { antialias: false, alpha: false, depth: false, stencil: false, powerPreference: 'low-power' });
    let output2d = null;
    let gpu = null;

    function setupGpu() {
        const compile = (type, source) => {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
        };
        const vertex = compile(gl.VERTEX_SHADER, vertexSource);
        const fragment = compile(gl.FRAGMENT_SHADER, postSource);
        if (!vertex || !fragment) return null;
        const program = gl.createProgram();
        gl.attachShader(program, vertex);
        gl.attachShader(program, fragment);
        gl.bindAttribLocation(program, 0, 'aPos');
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
        gl.useProgram(program);
        gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return {
            uScene: gl.getUniformLocation(program, 'uScene'),
            uSrc: gl.getUniformLocation(program, 'uSrc'),
            uTime: gl.getUniformLocation(program, 'uTime')
        };
    }
    if (gl) gpu = setupGpu();
    if (!gpu) {
        gl = null;
        output2d = canvas.getContext('2d');
    }

    function present(time) {
        if (gl && gpu) {
            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
            gl.uniform1i(gpu.uScene, 0);
            gl.uniform2f(gpu.uSrc, world.W, world.H);
            gl.uniform1f(gpu.uTime, time);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        } else if (output2d) {
            output2d.imageSmoothingEnabled = false;
            output2d.drawImage(frame, 0, 0, canvas.width, canvas.height);
        }
        host.classList.add('is-live');
    }

    /* ------------------------------------------------------ lifecycle */
    function build() {
        const rect = host.getBoundingClientRect();
        if (!rect.width || !rect.height) return false;
        const W = Math.round(Math.min(340, Math.max(140, rect.width / 4.8)));
        const H = Math.max(100, Math.round((W * rect.height) / rect.width));
        const ratio = Math.min(window.devicePixelRatio || 1, 1.25);
        canvas.width = Math.min(2000, Math.round(rect.width * ratio));
        canvas.height = Math.max(1, Math.round((canvas.width * rect.height) / rect.width));
        if (world && Math.abs(world.W - W) < 8 && Math.abs(world.H - H) < 8) return true;
        random = generator(20261010);
        if (!sprites) buildSprites();
        world = { W, H, queue: [], ...paintBackdrop(W, H) };
        frame = document.createElement('canvas');
        frame.width = W;
        frame.height = H;
        ctx = frame.getContext('2d');
        step = 0;
        kills = 0;
        losses = 0;
        rewardHistory.length = 0;
        populate();
        buildOverlay();
        buildPanel();
        return true;
    }

    let running = false;
    let visible = true;
    let request = 0;
    let last = 0;
    let accumulator = 0;
    let clock = 0;

    function render() {
        drawFrame();
        present(clock);
        updateOverlay();
        if (step % 2 === 0 || !running) updatePanel();
    }

    function loop(now) {
        request = 0;
        if (!running) return;
        const delta = last ? Math.min(0.1, (now - last) / 1000) : 0;
        last = now;
        accumulator += delta;
        clock += delta;
        let ticked = false;
        for (let i = 0; i < 3 && accumulator >= TICK; i += 1) {
            tick();
            accumulator -= TICK;
            ticked = true;
        }
        if (ticked) render();
        request = window.requestAnimationFrame(loop);
    }

    function setRunning(next) {
        const shouldRun = next && !reduceMotion && Boolean(world);
        if (shouldRun === running) return;
        running = shouldRun;
        if (running) {
            last = 0;
            request = window.requestAnimationFrame(loop);
        } else if (request) {
            window.cancelAnimationFrame(request);
            request = 0;
        }
    }

    function settle() {
        for (let i = 0; i < 420; i += 1) tick();
        render();
    }

    if (!build()) return;
    if (reduceMotion) settle();
    else render();

    let resizeTimer = 0;
    if ('ResizeObserver' in window) {
        new ResizeObserver(() => {
            window.clearTimeout(resizeTimer);
            resizeTimer = window.setTimeout(() => {
                if (!build()) return;
                if (reduceMotion) settle();
                else render();
            }, 180);
        }).observe(host);
    }
    if ('IntersectionObserver' in window) {
        new IntersectionObserver(entries => {
            visible = entries[0].isIntersecting;
            setRunning(visible && !document.hidden);
        }).observe(host);
    }
    document.addEventListener('visibilitychange', () => setRunning(visible && !document.hidden));
    canvas.addEventListener('webglcontextlost', event => {
        event.preventDefault();
        setRunning(false);
        host.classList.remove('is-live');
    });
    setRunning(visible && !document.hidden);
}());
