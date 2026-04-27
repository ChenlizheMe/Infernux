const translations = {
    en: {
        /* Brand */
        "brand.ribbonKicker": "Open-source engine · 0.1.5 live",
        "brand.ribbonName": "INFER<span class=\"mission-accent\">NUX</span>",
        "brand.ribbonSub": "熔炉 · ENG-CORE",
        "brand.navShort": "熔炉 · INFERNUX",
        "brand.footerTitle": "熔炉 · INFERNUX",
        "pageTitle.index": "熔炉 · Infernux — Open game engine",
        "pageTitle.wiki": "熔炉 · Infernux — Documentation deck",
        "pageTitle.roadmap": "熔炉 · Infernux — Transit plan",

        /* Nav */
        "nav.home": "Home",
        "nav.features": "Principles",
        "nav.showcase": "Capabilities",
        "nav.roadmap": "Roadmap",
        "nav.docs": "Docs",

        /* Hero */
        "hero.subtitle": "Programmable render stack · Unity-like API shape · high-performance core",
        "hero.viewGithub": "Inspect the repository",
        "hero.roadmap": "Roadmap",
        "download.hubInstaller": "Download Hub installer",

        "home.hero.badge": "RELEASE 0.1.5 · LIVE",
        "home.hero.kicker": "Open source · MIT · Win64 (broader platforms in development)",
        "home.hero.title": "<span class=\"accent\">Infernux</span> — the forge is lit. Come stoke the fire.",
        "home.hero.description": "Infernux (熔炉) is a game engine built from a Python–C++ split: C++17 powers the high-performance core; Python 3.12 drives editor tooling and is the gameplay scripting layer for developers.",
        "home.hero.docs": "Open the docs",
        "home.hero.metric.render": "Native render core",
        "home.hero.metric.python": "Gameplay & tooling",
        "home.hero.metric.license": "No royalties",

        /* Manifesto */
        "home.manifesto.tag": "FLIGHT NOTE",
        "home.manifesto": "Infernux stands on a simple idea: commercial engine licenses keep getting heavier, and indies and small teams pay the price. When “vendor freedom” comes with a growing bill, open source stops being optional — it’s the answer. That is what 熔炉 is for.",

        /* Demo */
        "home.demo.kicker": "Performance study · 0.1.5",
        "home.demo.title": "Complex-scene rendering tests show our performance is competitive with Unity.",
        "home.demo.intro": "To stress-test Infernux we built this scene: ten thousand cubes, vertex shader–driven sine motion, lit by one directional light.",
        "home.demo.panel.title": "What the numbers show",
        "home.demo.panel.item1": "In the editor the engine averaged about 127 FPS; in play mode about 171 FPS. Unity (IL2CPP) measured 61 FPS and 187 FPS in the same setups.",
        "home.demo.panel.item2": "Further tests suggest Infernux is roughly within 10% of Unity on rendering and scene editing, while pure-compute cases can reach up to 7× Unity throughput in our measurements.",
        "home.demo.panel.item3": "See the technical report: <a href=\"https://arxiv.org/pdf/2604.10263\" target=\"_blank\" rel=\"noopener\">Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting</a>",

        /* System pillars */
        "home.system.kicker": "Design values",
        "home.system.title": "Three ideas the whole engine is built around.",
        "home.system.intro": "The design targets engineering control instead of platform lock-in. Each principle cuts hidden cost for teams shipping real games and real tools.",
        "home.system.card1.title": "Principle I · Low migration cost",
        "home.system.card1.body": "We borrow interaction patterns from familiar tools and engines to reduce switching friction. There is a Figma-like standalone UI editor and a Python runtime that mirrors Unity’s at play time.",
        "home.system.card2.title": "Principle II · Intuitive first",
        "home.system.card2.body": "Heavy features like the render path exist to be extended, not worshipped. Programmable pipelines and other advanced APIs are shaped so newcomers can follow them without fighting the engine.",
        "home.system.card3.title": "Principle III · No business trap",
        "home.system.card3.body": "The engine is and will stay as free for indies and small teams to use commercially as we can make it, so more developers can reach state-of-the-art technology.",

        /* Capabilities */
        "home.capabilities.kicker": "Status board · 0.1.5",
        "home.capabilities.title": "Where the engine stands today.",
        "home.capabilities.intro": "The engine is still in active development: we are tightening every subsystem to reach a solid technical preview. Open issues and send PRs on GitHub.",
        "home.capabilities.render.tag": "SUBSYSTEM · RENDER",
        "home.capabilities.render.title": "Vulkan render core",
        "home.capabilities.render.body": "Forward and deferred rendering, PBR materials, cascaded shadows, MSAA, shader reflection, post-processing stack, and RenderGraph-based pass scheduling.",
        "home.capabilities.physics.tag": "SUBSYSTEM · PHYSICS",
        "home.capabilities.physics.title": "Jolt rigid bodies",
        "home.capabilities.physics.body": "Rigid bodies, colliders, scene queries, collision callbacks, layer filtering, and scene-synchronized transforms backed by Jolt Physics.",
        "home.capabilities.animation.tag": "PREVIEW · ANIMATION",
        "home.capabilities.animation.title": "2D / 3D animation stack",
        "home.capabilities.animation.body": "Sprite-based <code>SpiritAnimator</code>, <code>AnimClip2D</code> and <code>AnimClip3D</code> assets, embedded FBX takes, skeletal animation playback, skinned mesh rendering, and an FSM editor for state machines.",
        "home.capabilities.editor.tag": "SUBSYSTEM · EDITOR",
        "home.capabilities.editor.title": "12-panel editor shell",
        "home.capabilities.editor.body": "Hierarchy, Inspector, Scene View, Game View, Project, Console, UI editor, Toolbar, gizmos, multi-selection, undo / redo, and play-mode scene isolation in a single shell.",
        "home.capabilities.script.tag": "SUBSYSTEM · SCRIPT",
        "home.capabilities.script.title": "Python gameplay layer",
        "home.capabilities.script.body": "Unity-style component lifecycle, serialized inspector fields, decorators, input APIs, coroutines, prefabs, hot-reload, and a built-in <code>@njit</code> decorator that opts in to Numba JIT with automatic pure-Python fallback.",
        "home.capabilities.asset.tag": "SUBSYSTEM · ASSETS",
        "home.capabilities.asset.title": "Asset & project pipeline",
        "home.capabilities.asset.body": "GUID-based AssetDatabase, <code>.meta</code> sidecar files, dependency tracking, scene serialization, asset previews, Nuitka-based standalone build, and a PySide6 Hub launcher.",
        "home.capabilities.ai.tag": "EXTENSION · ECOSYSTEM",
        "home.capabilities.ai.title": "ML & data libraries on tap",
        "home.capabilities.ai.body": "Because the gameplay and tooling surface is Python, you can pull NumPy, Numba, Taichi, OpenCV, and PyTorch directly into gameplay logic, editor tools, or offline pipelines.",

        /* Status */
        "home.status.kicker": "Project status",
        "home.status.title": "Release 0.1.5: technical preview with animation and upgraded asset workflows.",
        "home.status.intro": "The engine has moved beyond static-scene authoring into animation and content workflows, while the roadmap still calls out what needs production hardening.",
        "home.status.card1.tag": "CHECKPOINT · 0.1.5",
        "home.status.card1.title": "Runtime, editor, and animation foundation",
        "home.status.card1.body": "Rendering, physics, audio, Python scripting, prefabs, game UI, editor authoring, GUID asset workflows, 2D/3D animation previews, skinned meshes, asset previews, and standalone build are all online.",
        "home.status.card2.tag": "NEXT · TRANSIT",
        "home.status.card2.title": "Advanced UI and content scale",
        "home.status.card2.value": "v0.2 → v0.4",
        "home.status.card2.body": "Upcoming milestones focus on richer UI controls, GPU particles, terrain, onboarding material, and stronger content production paths.",

        /* CTA */
        "cta.title": "Build something that truly belongs to you.",
        "cta.desc": "Infernux (熔炉) is for teams who want source access, clear architecture, and a workflow they can reshape. Start with the docs, read the code, and push the engine forward.",
        "cta.star": "Star on GitHub",

        /* Footer */
        "footer.tagline": "Open code, explicit architecture, and a render stack you can actually reason about.",
        "footer.resources": "Resources",
        "footer.community": "Community",
        "footer.issues": "Issues",
        "footer.discussions": "Discussions",
        "footer.email": "Email",

        /* Roadmap */
        "roadmap.hero.badge": "TRANSIT PLAN · 0.1.5 → 1.0",
        "roadmap.hero.kicker": "Release roadmap",
        "roadmap.hero.title": "A roadmap that reads like engineering work, not decorative ambition.",
        "roadmap.hero.description": "熔炉 is moving from technical preview toward a more complete production pipeline. The priorities below group work by release leverage and workflow impact, not by vague marketing buckets.",
        "roadmap.hero.primary": "Track issues",
        "roadmap.hero.secondary": "Read docs first",
        "roadmap.release.current.tag": "CHECKPOINT · ONLINE",
        "roadmap.release.current.title": "Runtime, editor, and animation foundation",
        "roadmap.release.current.item1": "Vulkan renderer, Python scripting, 12-panel editor, audio, Jolt physics, prefabs, game UI, 2D/3D animation previews, skinned meshes, asset previews, and standalone build.",
        "roadmap.release.current.item2": "Usable for serious technical preview work today: scripting, physics, audio, rendering, UI, animation previews, and editor tooling all running in a single stack.",
        "roadmap.release.next.tag": "NEXT · TRANSIT",
        "roadmap.release.next.title": "Advanced UI and content scale",
        "roadmap.release.next.item1": "Advanced UI controls, GPU particles, terrain, and stronger content production paths.",
        "roadmap.release.next.item2": "Support larger projects with richer runtime UI, scene content, and production workflows.",
        "roadmap.release.mid.tag": "MID-COURSE",
        "roadmap.release.mid.title": "Particles, terrain, and content pipeline",
        "roadmap.release.mid.item1": "GPU particle system, terrain, and broader model and material pipeline.",
        "roadmap.release.mid.item2": "Expand the range of visual content the engine can produce without collapsing under content complexity.",
        "roadmap.release.long.tag": "SHIPPING TRAJECTORY",
        "roadmap.release.long.title": "Networking and project lifecycle",
        "roadmap.release.long.item1": "Networking foundations and a more complete project lifecycle from prototype to release.",
        "roadmap.release.long.item2": "Move from Windows-first technical preview toward a multi-platform production engine.",

        "roadmap.lanes.kicker": "Operations bands",
        "roadmap.lanes.title": "The roadmap is split by what each band actually unlocks.",
        "roadmap.lanes.intro": "Each band answers one practical question: can teams author faster, can content scale, can projects ship, and can the architecture stay readable while the engine grows?",
        "roadmap.lanes.card1.title": "Authoring band",
        "roadmap.lanes.card1.item1": "Animation state machines and rigged character workflows beyond the 0.1.5 preview.",
        "roadmap.lanes.card1.item2": "Advanced UI controls (ScrollView, Slider, layout groups).",
        "roadmap.lanes.card1.item3": "Safer asset rename and dependency repair paths.",
        "roadmap.lanes.card2.title": "Content band",
        "roadmap.lanes.card2.item1": "GPU particle system.",
        "roadmap.lanes.card2.item2": "Terrain system for larger production scenes.",
        "roadmap.lanes.card2.item3": "Better scene composition for reusable runtime content.",
        "roadmap.lanes.card3.title": "Shipping band",
        "roadmap.lanes.card3.item1": "Networking foundations.",
        "roadmap.lanes.card3.item2": "A more robust project lifecycle from prototype to release.",
        "roadmap.lanes.card4.title": "Architecture band",
        "roadmap.lanes.card4.item1": "Keep render and tooling internals explicit as the engine grows.",
        "roadmap.lanes.card4.item2": "Avoid accidental complexity that would erase the engine's main advantage.",
        "roadmap.lanes.card4.item3": "Preserve scriptability while strengthening native ownership boundaries.",

        "roadmap.priorities.kicker": "Immediate focus",
        "roadmap.priorities.title": "What deserves attention in the next two milestones.",
        "roadmap.priorities.intro": "These items are the ones most likely to improve the engine's day-to-day usefulness for contributors and early projects.",
        "roadmap.priorities.card1.title": "Animation system maturation",
        "roadmap.priorities.card1.body": "2D / 3D animation previews and skinned meshes shipped in 0.1.5. The next focus is hardening retargeting, blend trees, and runtime control surfaces.",
        "roadmap.priorities.card2.title": "Advanced UI controls",
        "roadmap.priorities.card2.body": "The base UI system (Canvas, Text, Image, Button) is stable. ScrollView, Slider, layout groups, and editor-level UI tools come next.",
        "roadmap.priorities.card3.title": "Documentation & onboarding",
        "roadmap.priorities.card3.body": "The API reference is auto-generated. Getting-started guides, architecture notes, and example projects are needed to lower the onboarding barrier.",

        /* Wiki */
        "wiki.hero.badge": "DOCUMENTATION DECK",
        "wiki.hero.kicker": "API · architecture · workflow",
        "wiki.hero.title": "Documentation should reduce uncertainty, not just prove that docs exist.",
        "wiki.hero.description": "Use this page as the entry chart into the scripting API, architecture notes, repository setup, and long-form guides for 熔炉.",
        "wiki.hero.primary": "Open API docs",
        "wiki.hero.secondary": "Open README",
        "wiki.library.kicker": "Written guides",
        "wiki.library.title": "Architecture notes, advanced guides, and system deep-dives.",
        "wiki.library.intro": "Detailed articles covering how 熔炉 works internally and how to scale your project.",
        "wiki.library.loading": "Loading Markdown guides...",
        "wiki.library.search": "Search guides..."
    },

    zh: {
        /* Brand */
        "brand.ribbonKicker": "开源引擎 · 0.1.5 已上线",
        "brand.ribbonName": "<span class=\"mission-accent\">熔</span>炉",
        "brand.ribbonSub": "INFERNUX · ENG-CORE",
        "brand.navShort": "熔炉 · INFERNUX",
        "brand.footerTitle": "熔炉 · INFERNUX",
        "pageTitle.index": "熔炉 · Infernux — 开源游戏引擎",
        "pageTitle.wiki": "熔炉 · Infernux — 文档中枢",
        "pageTitle.roadmap": "熔炉 · Infernux — 发布路线图",

        /* Nav */
        "nav.home": "首页",
        "nav.features": "原则",
        "nav.showcase": "能力",
        "nav.roadmap": "路线图",
        "nav.docs": "文档",

        /* Hero */
        "hero.subtitle": "可编程渲染管线 · 类Unity API架构 · 高性能内核",
        "hero.viewGithub": "查看仓库",
        "hero.roadmap": "路线图",
        "download.hubInstaller": "下载 Hub 安装器",

        "home.hero.badge": "RELEASE 0.1.5 · 已上线",
        "home.hero.kicker": "开源 · MIT · WIN64（多平台拓展正在开发中）",
        "home.hero.title": "<span class=\"accent\">熔炉</span>初燃，来一起添把火",
        "home.hero.description": "熔炉（Infernux）是一个由 Python-C++ 混合编程构造的游戏引擎。 C++17 用于构造引擎的高性能核心， Python3.12 则用于实现编辑器工具，并作为游戏逻辑脚本提供给开发者。",
        "home.hero.docs": "阅读文档",
        "home.hero.metric.render": "原生渲染内核",
        "home.hero.metric.python": "玩法与工具",
        "home.hero.metric.license": "开源免费可商用",

        /* Manifesto */
        "home.manifesto.tag": "FLIGHT NOTE",
        "home.manifesto": "熔炉的立场很简单——商业引擎的授权费用越来越高，独立开发者与小团队的负担日益增长。商业引擎把创造的自由标上价格，开源就不再是选项，是答案。而这就是熔炉引擎的价值。",

        /* Demo */
        "home.demo.kicker": "性能实验 · 0.1.5",
        "home.demo.title": "复杂场景的渲染实验表明，我们的性能不比Unity差。",
        "home.demo.intro": "为测试熔炉引擎的性能，我们构建了如下场景：一万个立方体，经由顶点着色器驱动正弦波动，受单一方向光源照射。",
        "home.demo.panel.title": "实验结果表明...",
        "home.demo.panel.item1": "在编辑器中，我们的引擎运行帧率达到了平均127FPS；在运行时模式，我们的帧率则是171FPS。Unity（IL2CPP）在这两个情景下的帧率则分别是61FPS和187FPS。",
        "home.demo.panel.item2": "更多相关实验表明，在渲染与场景编辑上，熔炉引擎大约比Unity慢10%；而在纯计算领域，我们最高能达到Unity效率的7倍。",
        "home.demo.panel.item3": "更多实验数据请参阅技术报告：<a href=\"https://arxiv.org/pdf/2604.10263\" target=\"_blank\">Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting</a>",

        /* System pillars */
        "home.system.kicker": "设计理念",
        "home.system.title": "这台引擎的基础设计围绕着三个最核心的原则。",
        "home.system.intro": "这台引擎围绕工程控制权而不是平台绑定来设计。每条原则都为做真实游戏与真实工具的团队减少隐藏成本。",
        "home.system.card1.title": "原则 I · 迁移方便",
        "home.system.card1.body": "我们从知名的工具与游戏引擎中学习引擎的交互模式，尽可能减少开发者的迁移成本。我们有一个类Figma的独立UI编辑器，也有和Unity运行时完全一致的Python运行时。",
        "home.system.card2.title": "原则 II · 符合直觉",
        "home.system.card2.body": "渲染路径等复杂功能是用来扩展的，不是用来供着的。我们尽可能将可编程渲染管线以及其它类似的高级API构造得符合入门开发者的直觉",
        "home.system.card3.title": "原则 III · 没有商业陷阱",
        "home.system.card3.body": "引擎的本体将采用且尽可能永久对独立开发者或小团队免费开源可商用，我们希望让更多的开发者能用上最先进水平的技术。",

        /* Capabilities */
        "home.capabilities.kicker": "Operations Board · 0.1.5",
        "home.capabilities.title": "引擎现状。",
        "home.capabilities.intro": "目前，引擎还处于紧张地开发阶段，我们正在一步一步完善各个功能，以期达到技术预览的标准。欢迎在Github上提Issue或是贡献自己的PR！",
        "home.capabilities.render.tag": "子系统 · RENDER",
        "home.capabilities.render.title": "Vulkan 渲染内核",
        "home.capabilities.render.body": "前向 / 延迟渲染、PBR、级联阴影、MSAA、Shader 反射、后处理栈，以及基于 RenderGraph 的 Pass 调度。",
        "home.capabilities.physics.tag": "子系统 · PHYSICS",
        "home.capabilities.physics.title": "Jolt 刚体物理",
        "home.capabilities.physics.body": "刚体、碰撞体、场景查询、碰撞回调、层过滤与场景同步变换，全部基于 Jolt Physics。",
        "home.capabilities.animation.tag": "预览 · ANIMATION",
        "home.capabilities.animation.title": "2D / 3D 动画栈",
        "home.capabilities.animation.body": "基于精灵的 <code>SpiritAnimator</code>、<code>AnimClip2D</code> 与 <code>AnimClip3D</code> 资产、FBX 内嵌动作、骨骼动画播放、蒙皮网格渲染，以及用于编辑状态机的 FSM 面板。",
        "home.capabilities.editor.tag": "子系统 · EDITOR",
        "home.capabilities.editor.title": "12 面板编辑器外壳",
        "home.capabilities.editor.body": "Hierarchy、Inspector、Scene View、Game View、Project、Console、UI 编辑、Toolbar、Gizmo、多选、撤销 / 重做与 Play 模式场景隔离全部集中在同一套外壳。",
        "home.capabilities.script.tag": "子系统 · SCRIPT",
        "home.capabilities.script.title": "Python 玩法层",
        "home.capabilities.script.body": "类 Unity 的组件生命周期、Inspector 可见的序列化字段、装饰器、输入 API、协程、预制体、热重载，以及内置 <code>@njit</code> 装饰器：自动接入 Numba JIT，并在不可用时降级为纯 Python。",
        "home.capabilities.asset.tag": "子系统 · ASSETS",
        "home.capabilities.asset.title": "资产与项目管线",
        "home.capabilities.asset.body": "基于 GUID 的 AssetDatabase、<code>.meta</code> 附属文件、依赖追踪、场景序列化、资产预览、Nuitka 独立构建，以及基于 PySide6 的 Hub 启动器。",
        "home.capabilities.ai.tag": "扩展 · ECOSYSTEM",
        "home.capabilities.ai.title": "ML 与数据库都能直接接入",
        "home.capabilities.ai.body": "因为玩法和工具都跑在 Python 上，可以把 NumPy、Numba、Taichi、OpenCV、PyTorch 直接接入玩法逻辑、编辑器工具或离线管线。",

        /* Status */
        "home.status.kicker": "项目状态",
        "home.status.title": "Release 0.1.5：加入动画与资产工作流升级的技术预览版。",
        "home.status.intro": "引擎已经从静态场景编辑继续向动画和内容工作流推进，同时路线图仍明确标出还需要生产级打磨的部分。",
        "home.status.card1.tag": "CHECKPOINT · 0.1.5",
        "home.status.card1.title": "运行时、编辑器与动画基础",
        "home.status.card1.body": "渲染、物理、音频、Python 脚本、预制体、游戏 UI、编辑器编排、GUID 资产工作流、2D/3D 动画预览、蒙皮网格、资产预览与独立构建现在都已上线。",
        "home.status.card2.tag": "下一程 · TRANSIT",
        "home.status.card2.title": "高级 UI 与内容规模化",
        "home.status.card2.value": "v0.2 → v0.4",
        "home.status.card2.body": "下一阶段集中在更丰富的 UI 控件、GPU 粒子、地形、上手资料以及更强的内容生产链路。",

        /* CTA */
        "cta.title": "去构建真正属于你自己的东西。",
        "cta.desc": "熔炉面向那些希望拥有源码访问权、清晰架构与可重塑工作流的团队。先看文档，再看代码，然后把这台引擎继续向前推。",
        "cta.star": "在 GitHub 上 Star",

        /* Footer */
        "footer.tagline": "开放代码、明确架构，以及一套你能真正讲清楚的渲染栈。",
        "footer.resources": "资源",
        "footer.community": "社区",
        "footer.issues": "问题反馈",
        "footer.discussions": "讨论区",
        "footer.email": "邮箱",

        /* Roadmap */
        "roadmap.hero.badge": "TRANSIT PLAN · 0.1.5 → 1.0",
        "roadmap.hero.kicker": "发布路线图",
        "roadmap.hero.title": "这是一份像工程进度表的路线图，不是一份装饰性愿景。",
        "roadmap.hero.description": "熔炉正在从技术预览向更完整的生产管线推进。下面的优先级按发布杠杆和工作流影响来分组，而不是按宽泛的营销主题。",
        "roadmap.hero.primary": "查看 Issues",
        "roadmap.hero.secondary": "先读文档",
        "roadmap.release.current.tag": "CHECKPOINT · 已上线",
        "roadmap.release.current.title": "运行时、编辑器与动画基础",
        "roadmap.release.current.item1": "Vulkan 渲染器、Python 脚本、12 面板编辑器、音频、Jolt 物理、预制体、游戏 UI、2D/3D 动画预览、蒙皮网格、资产预览与独立构建。",
        "roadmap.release.current.item2": "今天已经能用来做认真的技术预览：脚本、物理、音频、渲染、UI、动画预览与编辑器工具都跑在同一套栈里。",
        "roadmap.release.next.tag": "下一程 · TRANSIT",
        "roadmap.release.next.title": "高级 UI 与内容规模化",
        "roadmap.release.next.item1": "高级 UI 控件、GPU 粒子、地形与更强的内容生产链路。",
        "roadmap.release.next.item2": "支撑更大的项目，包含更丰富的运行时 UI、场景内容与生产工作流。",
        "roadmap.release.mid.tag": "中段 · MID-COURSE",
        "roadmap.release.mid.title": "粒子、地形与内容管线",
        "roadmap.release.mid.item1": "GPU 粒子系统、地形支持，以及更广的模型与材质管线。",
        "roadmap.release.mid.item2": "扩大引擎能承载的内容范围，同时不让内容复杂度压垮架构。",
        "roadmap.release.long.tag": "向发布 · TRAJECTORY",
        "roadmap.release.long.title": "网络与项目生命周期",
        "roadmap.release.long.item1": "网络基础与从原型到发布的更完整项目生命周期。",
        "roadmap.release.long.item2": "从 Windows-first 技术预览继续推进到多平台生产引擎。",

        "roadmap.lanes.kicker": "工作方向",
        "roadmap.lanes.title": "路线图按“每条带能解锁什么”来切分，而不是按主题命名。",
        "roadmap.lanes.intro": "每条带都在回答一个具体问题：团队能否更快创作、内容能否扩张、项目能否发布、架构能否在引擎成长时保持可读。",
        "roadmap.lanes.card1.title": "创作带",
        "roadmap.lanes.card1.item1": "在 0.1.5 预览之上，继续推进动画状态机与绑定角色工作流。",
        "roadmap.lanes.card1.item2": "高级 UI 控件（ScrollView、Slider、布局组件）。",
        "roadmap.lanes.card1.item3": "更安全的资产重命名与依赖修复路径。",
        "roadmap.lanes.card2.title": "内容带",
        "roadmap.lanes.card2.item1": "GPU 粒子系统。",
        "roadmap.lanes.card2.item2": "支撑更大场景的地形系统。",
        "roadmap.lanes.card2.item3": "改进场景组织方式，便于内容复用。",
        "roadmap.lanes.card3.title": "发布带",
        "roadmap.lanes.card3.item1": "网络基础。",
        "roadmap.lanes.card3.item2": "更稳健的项目生命周期，从原型到发布。",
        "roadmap.lanes.card4.title": "架构带",
        "roadmap.lanes.card4.item1": "随着引擎扩张，让内部所有权边界与结构保持清晰。",
        "roadmap.lanes.card4.item2": "避免意外复杂度抹掉项目最重要的优势。",
        "roadmap.lanes.card4.item3": "在保持原生性能的同时维护 Python 脚本能力。",

        "roadmap.priorities.kicker": "近期重点",
        "roadmap.priorities.title": "下两个里程碑里值得花精力的事情。",
        "roadmap.priorities.intro": "这些工作项最有可能直接提升贡献者与早期项目的日常可用性。",
        "roadmap.priorities.card1.title": "动画系统打磨",
        "roadmap.priorities.card1.body": "0.1.5 已经交付 2D / 3D 动画预览与蒙皮网格。下一步重点是重定向、Blend Tree 与运行时控制面打磨。",
        "roadmap.priorities.card2.title": "高级 UI 控件",
        "roadmap.priorities.card2.body": "基础 UI 系统（Canvas、Text、Image、Button）已稳定。接下来是 ScrollView、Slider、布局组件以及编辑器级 UI 工具。",
        "roadmap.priorities.card3.title": "文档与上手资料",
        "roadmap.priorities.card3.body": "API 参考已经自动生成。还需要入门指南、架构说明与示例项目来降低上手门槛。",

        /* Wiki */
        "wiki.hero.badge": "DOCUMENTATION DECK",
        "wiki.hero.kicker": "API · 架构 · 工作流",
        "wiki.hero.title": "文档应该减少不确定性，而不是只证明文档存在。",
        "wiki.hero.description": "把这页当成熔炉的入口图：从这里进入脚本 API、架构笔记、仓库级搭建说明，以及更长篇的系统文档。",
        "wiki.hero.primary": "打开 API 文档",
        "wiki.hero.secondary": "打开 README",
        "wiki.library.kicker": "手写指南",
        "wiki.library.title": "架构笔记、进阶指南与系统深挖。",
        "wiki.library.intro": "这些页面解释熔炉是怎么组织起来的，以及在更复杂的项目里应该怎么使用它。",
        "wiki.library.loading": "正在加载 Markdown 指南...",
        "wiki.library.search": "搜索指南..."
    }
};

let currentLang = localStorage.getItem('lang') || 'en';

function applyLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('lang', lang);
    document.querySelectorAll('[data-i18n]').forEach((element) => {
        const key = element.getAttribute('data-i18n');
        const value = translations[lang]?.[key];
        if (!value) {
            return;
        }
        if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
            element.placeholder = value;
        } else {
            element.innerHTML = value;
        }
    });
    document.querySelectorAll('[data-href-en][data-href-zh]').forEach((element) => {
        element.setAttribute('href', lang === 'zh'
            ? element.getAttribute('data-href-zh')
            : element.getAttribute('data-href-en'));
    });
    const titleKey = document.documentElement.getAttribute('data-title-i18n');
    if (titleKey) {
        const titleText = translations[lang]?.[titleKey];
        if (titleText) {
            document.title = titleText;
        }
    }
    const langText = document.getElementById('lang-text');
    if (langText) {
        langText.textContent = lang === 'en' ? '中文' : 'EN';
    }
    document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN';
    document.dispatchEvent(new CustomEvent('site:language-changed', { detail: { lang } }));
}

function toggleLanguage() {
    applyLanguage(currentLang === 'en' ? 'zh' : 'en');
}

document.addEventListener('DOMContentLoaded', function() {
    applyLanguage(currentLang);
});
