(() => {
    "use strict";
    // Evidence paths are kept beside every claim. Edges describe capability paths,
    // not API-level dependencies or commitments to release dates.
    const systems = {
    "render": {
        "label": {
            "en": "Render",
            "zh": "渲染"
        },
        "icon": "◉",
        "nodes": [
            {
                "id": "render",
                "label": {
                    "en": "Scene render graph",
                    "zh": "场景渲染图"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/renderer/SceneRenderGraph.cpp",
                "body": {
                    "en": "Scene rendering is organized through the scene render graph.",
                    "zh": "场景渲染由场景渲染图组织。"
                },
                "x": 145,
                "y": 85
            },
            {
                "id": "hir",
                "label": {
                    "en": "Particle graph HIR",
                    "zh": "粒子图 HIR"
                },
                "state": "shipped",
                "source": "python/Infernux/particle/hir.py",
                "body": {
                    "en": "Authored particle graphs lower into a high-level intermediate representation.",
                    "zh": "创作的粒子图降低为高级中间表示。"
                },
                "x": 545,
                "y": 85
            },
            {
                "id": "kernel",
                "label": {
                    "en": "Particle kernel IR",
                    "zh": "粒子内核 IR"
                },
                "state": "shipped",
                "source": "python/Infernux/particle/kernel_ir.py",
                "body": {
                    "en": "Particle kernels have a typed intermediate representation used by the GPU backend.",
                    "zh": "粒子内核使用带类型的中间表示供 GPU 后端处理。"
                },
                "x": 545,
                "y": 240
            },
            {
                "id": "glsl",
                "label": {
                    "en": "GPU compilation",
                    "zh": "GPU 编译"
                },
                "state": "shipped",
                "source": "python/Infernux/particle/gpu_glsl_backend.py",
                "body": {
                    "en": "The particle backend generates GLSL from kernel IR.",
                    "zh": "粒子后端从内核 IR 生成 GLSL。"
                },
                "x": 545,
                "y": 395
            },
            {
                "id": "particles",
                "label": {
                    "en": "GPU particles",
                    "zh": "GPU 粒子"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/renderer/particle/ParticleGpuRuntime.cpp",
                "body": {
                    "en": "GPU particle execution already exists in the runtime; it is not an unbuilt roadmap item.",
                    "zh": "运行时已实现 GPU 粒子执行，并非尚未开发的规划项。"
                },
                "x": 145,
                "y": 395
            }
        ],
        "edges": [
            [
                "hir",
                "kernel"
            ],
            [
                "kernel",
                "glsl"
            ],
            [
                "glsl",
                "particles"
            ],
            [
                "render",
                "particles"
            ]
        ]
    },
    "audio": {
        "label": {
            "en": "Audio",
            "zh": "音频"
        },
        "icon": "◌",
        "nodes": [
            {
                "id": "clips",
                "label": {
                    "en": "Audio clips",
                    "zh": "音频片段"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/audio/AudioClipLoader.cpp",
                "body": {
                    "en": "Clip loading supplies audio resources to playback.",
                    "zh": "音频片段加载为播放提供资源。"
                },
                "x": 145,
                "y": 85
            },
            {
                "id": "streams",
                "label": {
                    "en": "Stream decoding",
                    "zh": "流式解码"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/audio/AudioStreamDecoder.h",
                "body": {
                    "en": "Stream decoding supports streamed audio resources.",
                    "zh": "流式解码支持流式音频资源。"
                },
                "x": 545,
                "y": 85
            },
            {
                "id": "audio",
                "label": {
                    "en": "Audio engine",
                    "zh": "音频引擎"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/audio/AudioEngine.cpp",
                "body": {
                    "en": "The engine manages voices, shared output and spatial source updates.",
                    "zh": "音频引擎管理声音、共享输出及空间音源更新。"
                },
                "x": 345,
                "y": 235
            },
            {
                "id": "spatial",
                "label": {
                    "en": "Spatial sources",
                    "zh": "空间音源"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/audio/AudioSource.cpp",
                "body": {
                    "en": "Audio sources connect playback to scene position and listener behavior.",
                    "zh": "音源将播放与场景位置和监听器行为连接。"
                },
                "x": 145,
                "y": 410
            },
            {
                "id": "buses",
                "label": {
                    "en": "Bus automation",
                    "zh": "总线自动化"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/audio/AudioBusAutomation.h",
                "body": {
                    "en": "Bus automation and mixer infrastructure already exist; this map does not promise a new visual mix editor.",
                    "zh": "总线自动化和混音基础设施已存在；本图不承诺新的可视化混音编辑器。"
                },
                "x": 545,
                "y": 410
            }
        ],
        "edges": [
            [
                "clips",
                "audio"
            ],
            [
                "streams",
                "audio"
            ],
            [
                "audio",
                "spatial"
            ],
            [
                "audio",
                "buses"
            ]
        ]
    },
    "gameplay": {
        "label": {
            "en": "Gameplay",
            "zh": "游戏逻辑"
        },
        "icon": "◇",
        "nodes": [
            {
                "id": "gameplay",
                "label": {
                    "en": "Python components",
                    "zh": "Python 组件"
                },
                "state": "shipped",
                "source": "python/Infernux/components/component.py",
                "body": {
                    "en": "Python components form the gameplay authoring surface.",
                    "zh": "Python 组件构成游戏逻辑创作接口。"
                },
                "x": 145,
                "y": 85
            },
            {
                "id": "scenes",
                "label": {
                    "en": "Scene lifecycle",
                    "zh": "场景生命周期"
                },
                "state": "shipped",
                "source": "python/Infernux/engine/scene_manager.py",
                "body": {
                    "en": "The scene manager owns scene loading and runtime lifecycle.",
                    "zh": "场景管理器负责场景加载与运行时生命周期。"
                },
                "x": 545,
                "y": 85
            },
            {
                "id": "prefabs",
                "label": {
                    "en": "Prefab authoring",
                    "zh": "Prefab 创作"
                },
                "state": "shipped",
                "source": "python/Infernux/engine/interaction/prefabs.py",
                "body": {
                    "en": "Prefab authoring uses editor commands for reversible operations.",
                    "zh": "Prefab 创作使用编辑器命令实现可撤销操作。"
                },
                "x": 145,
                "y": 255
            },
            {
                "id": "player",
                "label": {
                    "en": "Player export",
                    "zh": "Player 导出"
                },
                "state": "shipped",
                "source": "python/Infernux/engine/game_builder.py",
                "body": {
                    "en": "The game builder packages authored content and runtime dependencies.",
                    "zh": "游戏构建器打包创作内容与运行时依赖。"
                },
                "x": 345,
                "y": 430
            }
        ],
        "edges": [
            [
                "gameplay",
                "prefabs"
            ],
            [
                "prefabs",
                "player"
            ],
            [
                "gameplay",
                "player"
            ],
            [
                "scenes",
                "player"
            ]
        ]
    },
    "ai": {
        "label": {
            "en": "AI / 3N",
            "zh": "AI / 3N"
        },
        "icon": "✦",
        "nodes": [
            {
                "id": "schema",
                "label": {
                    "en": "Shared schemas",
                    "zh": "共享 Schema"
                },
                "state": "planned",
                "source": "docs/roadmap.html",
                "body": {
                    "en": "Planned 0.4.2: shared definitions for world state, commands and queries.",
                    "zh": "规划 0.4.2：世界状态、命令与查询的共享定义。"
                },
                "x": 145,
                "y": 85
            },
            {
                "id": "snapshots",
                "label": {
                    "en": "Snapshots + delta",
                    "zh": "快照与增量"
                },
                "state": "planned",
                "source": "docs/roadmap.html",
                "body": {
                    "en": "Planned world snapshots and deltas build on a shared semantic description.",
                    "zh": "规划的世界快照和增量建立在共享语义描述之上。"
                },
                "x": 145,
                "y": 240
            },
            {
                "id": "replay",
                "label": {
                    "en": "World replay",
                    "zh": "世界回放"
                },
                "state": "planned",
                "source": "docs/roadmap.html",
                "body": {
                    "en": "Planned 0.4.3: repeatable world replay contracts. This does not claim deterministic audio replay.",
                    "zh": "规划 0.4.3：可重复的世界回放契约，不代表确定性音频回放已获规划。"
                },
                "x": 145,
                "y": 405
            },
            {
                "id": "batch",
                "label": {
                    "en": "Batch + tensors",
                    "zh": "批量世界与张量"
                },
                "state": "dim",
                "source": "docs/roadmap.html",
                "body": {
                    "en": "Planned 0.4.4: batch world stepping and a tensor data plane with explicit ownership.",
                    "zh": "规划 0.4.4：批量世界步进与所有权明确的张量数据平面。"
                },
                "x": 545,
                "y": 240
            },
            {
                "id": "agents",
                "label": {
                    "en": "Governed agents",
                    "zh": "受治理 Agent"
                },
                "state": "dim",
                "source": "docs/roadmap.html",
                "body": {
                    "en": "The 0.5.2 goal connects schemas, replay, batch worlds and governed tools end to end.",
                    "zh": "0.5.2 目标是端到端连接 Schema、回放、批量世界与受治理工具。"
                },
                "x": 545,
                "y": 405
            }
        ],
        "edges": [
            [
                "schema",
                "snapshots"
            ],
            [
                "snapshots",
                "replay"
            ],
            [
                "schema",
                "batch"
            ],
            [
                "replay",
                "agents"
            ],
            [
                "batch",
                "agents"
            ]
        ]
    },
    "tools": {
        "label": {
            "en": "Tools",
            "zh": "工具包"
        },
        "icon": "▱",
        "nodes": [
            {
                "id": "tools",
                "label": {
                    "en": "Editor commands",
                    "zh": "编辑器命令"
                },
                "state": "shipped",
                "source": "python/Infernux/engine/interaction/commands.py",
                "body": {
                    "en": "Editor operations provide a shared command surface.",
                    "zh": "编辑器操作提供共享命令接口。"
                },
                "x": 145,
                "y": 85
            },
            {
                "id": "prefabtools",
                "label": {
                    "en": "Prefab commands",
                    "zh": "Prefab 命令"
                },
                "state": "shipped",
                "source": "python/Infernux/engine/interaction/prefabs.py",
                "body": {
                    "en": "Prefab editing is one existing consumer of the editor command surface.",
                    "zh": "Prefab 编辑是编辑器命令接口的现有使用方。"
                },
                "x": 545,
                "y": 85
            },
            {
                "id": "contracts",
                "label": {
                    "en": "Engine Tool contract",
                    "zh": "引擎工具契约"
                },
                "state": "planned",
                "source": "docs/roadmap.html",
                "body": {
                    "en": "Planned 0.5.0: schema, permissions, tests, provenance and resource limits for Engine Tools.",
                    "zh": "规划 0.5.0：引擎工具的 Schema、权限、测试、来源与资源限制。"
                },
                "x": 145,
                "y": 250
            },
            {
                "id": "sandbox",
                "label": {
                    "en": "Sandbox + packages",
                    "zh": "沙箱与分发"
                },
                "state": "dim",
                "source": "docs/roadmap.html",
                "body": {
                    "en": "Planned 0.5.1: distribution of plugins and models with sandboxed tests.",
                    "zh": "规划 0.5.1：带沙箱测试的插件与模型分发。"
                },
                "x": 145,
                "y": 415
            },
            {
                "id": "governance",
                "label": {
                    "en": "Governed workflow",
                    "zh": "受治理工作流"
                },
                "state": "dim",
                "source": "docs/roadmap.html",
                "body": {
                    "en": "Planned 0.5.2: an end-to-end workflow that subjects agent tools to the same engineering bar.",
                    "zh": "规划 0.5.2：让 Agent 工具满足相同工程标准的端到端工作流。"
                },
                "x": 545,
                "y": 415
            }
        ],
        "edges": [
            [
                "tools",
                "prefabtools"
            ],
            [
                "tools",
                "contracts"
            ],
            [
                "contracts",
                "sandbox"
            ],
            [
                "sandbox",
                "governance"
            ]
        ]
    },
    "physics": {
        "label": {
            "en": "Physics",
            "zh": "物理"
        },
        "icon": "⌁",
        "nodes": [
            {
                "id": "physics",
                "label": {
                    "en": "Physics world",
                    "zh": "物理世界"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/scene/physics/PhysicsWorld.cpp",
                "body": {
                    "en": "The native physics world owns rigid-body simulation and queries.",
                    "zh": "原生物理世界负责刚体模拟与查询。"
                },
                "x": 345,
                "y": 75
            },
            {
                "id": "queries",
                "label": {
                    "en": "Scene queries",
                    "zh": "场景查询"
                },
                "state": "shipped",
                "source": "python/Infernux/physics/__init__.py",
                "body": {
                    "en": "Raycasts, overlaps and shape casts are already exposed through the Python Physics API.",
                    "zh": "Python Physics API 已提供射线、重叠及形状投射查询。"
                },
                "x": 145,
                "y": 245
            },
            {
                "id": "callbacks",
                "label": {
                    "en": "Contact callbacks",
                    "zh": "碰撞回调"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/scene/physics/PhysicsContactListener.cpp",
                "body": {
                    "en": "The contact listener handles collision events in the physics world.",
                    "zh": "碰撞监听器处理物理世界中的碰撞事件。"
                },
                "x": 545,
                "y": 245
            },
            {
                "id": "raybatch",
                "label": {
                    "en": "Batched raycasts",
                    "zh": "批量射线查询"
                },
                "state": "shipped",
                "source": "cpp/infernux/function/scene/physics/PhysicsWorld.h",
                "body": {
                    "en": "Batched raycasts already exist. They are distinct from the planned batch-world lifecycle.",
                    "zh": "批量射线查询已存在，与规划中的批量世界生命周期不同。"
                },
                "x": 145,
                "y": 420
            },
            {
                "id": "sim",
                "label": {
                    "en": "Batch worlds",
                    "zh": "批量世界"
                },
                "state": "dim",
                "source": "docs/roadmap.html",
                "body": {
                    "en": "Planned 0.4.4: create, step and reset many worlds through explicit repeatable contracts.",
                    "zh": "规划 0.4.4：通过明确且可重复的契约创建、推进与重置多个世界。"
                },
                "x": 545,
                "y": 420
            }
        ],
        "edges": [
            [
                "physics",
                "queries"
            ],
            [
                "physics",
                "callbacks"
            ],
            [
                "queries",
                "raybatch"
            ],
            [
                "physics",
                "sim"
            ]
        ]
    }
};
    const orbit = document.querySelector("[data-roadmap-orbit]");
    if (!orbit) return;
    const svg = orbit.querySelector(".roadmap-orbit-svg");
    const tabs = orbit.querySelector("[data-roadmap-tabs]");
    const detail = { status: orbit.querySelector("[data-roadmap-detail-status]"), title: orbit.querySelector("[data-roadmap-detail-title]"), body: orbit.querySelector("[data-roadmap-detail-body]"), meta: orbit.querySelector("[data-roadmap-detail-meta]") };
    let activeSystem = "render";
    let selectedNode = "render";
    const language = () => document.documentElement.lang?.toLowerCase().startsWith("zh") ? "zh" : "en";
    const text = value => value?.[language()] || value?.en || "";
    const statusText = state => ({ shipped: { en: "SHIPPED", zh: "已实现" }, planned: { en: "PLANNED", zh: "已规划" }, dim: { en: "HORIZON", zh: "远期" } }[state]);
    function esc(value) { return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char])); }
    function selectNode(node) {
        selectedNode = node.id;
        orbit.querySelectorAll(".roadmap-node").forEach(item => item.classList.toggle("is-selected", item.dataset.nodeId === node.id));
        detail.status.textContent = text(statusText(node.state)); detail.status.dataset.state = node.state;
        detail.title.textContent = text(node.label); detail.body.textContent = text(node.body);
        detail.meta.textContent = node.source ? `SOURCE · ${node.source}` : "";
    }
    function draw(system) {
        const data = systems[system]; if (!data || !svg) return;
        svg.innerHTML = "";
        const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
        defs.innerHTML = `<filter id="orbit-glow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`; svg.appendChild(defs);
        const byId = Object.fromEntries(data.nodes.map(node => [node.id, node]));
        data.edges.forEach(([fromId, toId]) => { const from = byId[fromId], to = byId[toId]; const line = document.createElementNS("http://www.w3.org/2000/svg", "path"); line.setAttribute("d", `M ${from.x} ${from.y} C ${from.x} ${(from.y + to.y) / 2}, ${to.x} ${(from.y + to.y) / 2}, ${to.x} ${to.y}`); line.setAttribute("class", "roadmap-edge"); svg.appendChild(line); });
        data.nodes.forEach(node => { const group = document.createElementNS("http://www.w3.org/2000/svg", "g"); group.setAttribute("class", `roadmap-node roadmap-node-${node.state}`); group.dataset.nodeId = node.id; group.setAttribute("tabindex", "0"); group.setAttribute("role", "button"); group.setAttribute("aria-label", `${text(node.label)} · ${text(statusText(node.state))}`); group.setAttribute("data-source", node.source); group.innerHTML = `<circle class="roadmap-node-halo" cx="${node.x}" cy="${node.y}" r="34"></circle><circle class="roadmap-node-dot" cx="${node.x}" cy="${node.y}" r="12"></circle><text class="roadmap-node-label" x="${node.x}" y="${node.y + 54}" text-anchor="middle">${esc(text(node.label))}</text><text class="roadmap-node-state" x="${node.x}" y="${node.y + 72}" text-anchor="middle">${esc(text(statusText(node.state)))}</text>`; group.addEventListener("pointerenter", () => selectNode(node)); group.addEventListener("focus", () => selectNode(node)); group.addEventListener("click", () => selectNode(node)); group.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(node); } }); svg.appendChild(group); });
        selectNode(data.nodes.find(node => node.id === selectedNode) || data.nodes[0]);
    }
    function renderTabs() { tabs.innerHTML = Object.entries(systems).map(([key, data]) => `<button type="button" role="tab" aria-selected="${key === activeSystem}" class="roadmap-tab ${key === activeSystem ? "is-active" : ""}" data-system="${key}"><span aria-hidden="true">${data.icon}</span>${esc(text(data.label))}</button>`).join(""); tabs.querySelectorAll("[data-system]").forEach(button => button.addEventListener("click", () => { activeSystem = button.dataset.system; selectedNode = systems[activeSystem].nodes[0].id; renderTabs(); draw(activeSystem); })); }
    document.addEventListener("site:language-changed", () => { renderTabs(); draw(activeSystem); });
    renderTabs(); draw(activeSystem);
})();
