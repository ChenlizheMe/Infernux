const translations = {
    en: {
        /* Brand */
        "brand.ribbonKicker": "Open engine · release 0.1.5 online",
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
        "nav.comparison": "Architecture",
        "nav.roadmap": "Roadmap",
        "nav.docs": "Docs",

        /* Hero */
        "hero.subtitle": "C++ / Vulkan core · Python production layer · repository-first",
        "hero.viewGithub": "Inspect the repository",
        "hero.roadmap": "Roadmap",
        "download.hubInstaller": "Download Hub installer",

        "home.hero.badge": "RELEASE 0.1.5 · ONLINE",
        "home.hero.kicker": "OPEN-SOURCE · MIT · WIN64",
        "home.hero.title": "An engine you can <span class=\"accent\">disassemble</span>, extend, and ship.",
        "home.hero.description": "熔炉 (Infernux) is a from-scratch game engine that pairs a C++17 / Vulkan native runtime with a Python production layer for gameplay, editor tooling, and render-stack authoring. It is built for teams who want readable internals, a fast inner loop, and an MIT-licensed codebase they can actually own.",
        "home.hero.docs": "Open the docs",
        "home.hero.metric.render": "Native render core",
        "home.hero.metric.python": "Gameplay & tooling",
        "home.hero.metric.license": "No royalties",

        "home.panel.manifest.label": "Engine readout",
        "home.panel.manifest.title": "Runtime, editor, animation, and packaging on one stack.",
        "home.panel.manifest.body": "Vulkan rendering, Jolt physics, the editor shell, Python hot-reload, GUID-based asset workflows, play-mode isolation, 2D/3D animation previews, skinned meshes, and standalone packaging are wired into the same build today.",

        /* Manifesto */
        "home.manifesto.tag": "FLIGHT NOTE",
        "home.manifesto": "熔炉 is shaped around a simple position: keep the hot path native, keep the workflow scriptable, and keep the architecture readable enough that a team can actually take ownership of it instead of negotiating with a vendor.",

        /* Demo */
        "home.demo.kicker": "Runtime capture · 0.1.5",
        "home.demo.title": "Live editor session running a height-shaded voxel scene at production density.",
        "home.demo.intro": "The capture below is unedited: the editor sits in Play mode, the inspector is open on a custom <code>height_gradient</code> shader, and the runtime is keeping the entire pixel grid lit and updating without any scripted interaction.",
        "home.demo.panel.title": "What this capture proves",
        "home.demo.panel.item1": "A real editor session — Hierarchy, Scene/Game/UI tabs, Project, Console, and Inspector all attached to the live runtime.",
        "home.demo.panel.item2": "Custom material authored in Python: <code>height_gradient.frag</code> shading hundreds of cube clones in a single material slot.",
        "home.demo.panel.item3": "173 FPS on the test workstation while running play mode, build pipeline, and the inspector in parallel.",

        /* System pillars */
        "home.system.kicker": "Operating principles",
        "home.system.title": "Three rules that decide what enters the engine.",
        "home.system.intro": "The engine is shaped around engineering control rather than platform lock-in. Each principle reduces hidden cost for teams building real games and real tools.",
        "home.system.card1.title": "PRINCIPLE I · Fast inner loop",
        "home.system.card1.body": "Python scripts, hot-reload, asset previews, and editor-side tooling keep the iteration loop under a second without weakening the native runtime core.",
        "home.system.card2.title": "PRINCIPLE II · Transparent render",
        "home.system.card2.body": "RenderGraph, RenderStack, and material descriptors are inspectable and scriptable from Python. The render path is meant to be extended, not hidden.",
        "home.system.card3.title": "PRINCIPLE III · No business trap",
        "home.system.card3.body": "MIT licensing and repository-first development keep the cost model simple: the work is in the code, not in negotiating with a vendor.",

        /* Capabilities */
        "home.capabilities.kicker": "Operations board · 0.1.5",
        "home.capabilities.title": "What is online today, grouped the way real engineering teams evaluate an engine.",
        "home.capabilities.intro": "This is the current technical preview, sorted by the work areas that actually matter: rendering, simulation, content, scripting, tooling, and asset pipeline.",
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

        /* Architecture */
        "home.difference.kicker": "Stack anatomy",
        "home.difference.title": "Native ownership where it matters, Python where iteration speed matters.",
        "home.difference.intro": "熔炉 keeps performance-critical systems in C++17 and puts iteration-heavy work in Python, with pybind11 sitting between them as an explicit binding seam.",
        "home.stack.title": "Layered architecture",
        "home.stack.layer1.label": "L4 · Project & tools",
        "home.stack.layer1.body": "PySide6 Hub launcher, editor panels, project management, and content workflows.",
        "home.stack.layer2.label": "L3 · Gameplay & render",
        "home.stack.layer2.body": "Python components, RenderGraph authoring, runtime behaviors, and render-stack scripting.",
        "home.stack.layer3.label": "L2 · Binding seam",
        "home.stack.layer3.body": "pybind11 keeps the Python API direct enough to debug while preserving native ownership boundaries.",
        "home.stack.layer4.label": "L1 · Native core",
        "home.stack.layer4.body": "Scene systems, Vulkan renderer, Jolt physics, resource ownership, and platform abstraction in C++17.",
        "home.pipeline.title": "Flight plan: project to runtime",
        "home.pipeline.step1.label": "Author",
        "home.pipeline.step1.body": "Write gameplay or render logic in Python and expose parameters through serialized inspector fields.",
        "home.pipeline.step2.label": "Describe",
        "home.pipeline.step2.body": "Author passes, resources, and dependencies through RenderGraph and related pipeline APIs.",
        "home.pipeline.step3.label": "Run",
        "home.pipeline.step3.body": "The native backend handles Vulkan scheduling, barriers, memory management, and submission.",
        "home.pipeline.step4.label": "Iterate",
        "home.pipeline.step4.body": "Inspect the result in the editor, hot-reload scripts, retune the data, and repeat without restarting the project.",

        /* Workflow */
        "home.workflow.kicker": "Studio workflow",
        "home.workflow.title": "From cold-start to validated runtime in four explicit steps.",
        "home.workflow.intro": "This is the day-to-day loop the engine is trying to optimize, not a marketing-level feature list.",
        "home.workflow.card1.title": "01 · Create",
        "home.workflow.card1.body": "Use the Hub to create or open projects. Assets are tracked through stable GUIDs and <code>.meta</code> sidecar files.",
        "home.workflow.card2.title": "02 · Author",
        "home.workflow.card2.body": "Compose scenes, attach scripts, configure materials, and wire render behavior through one consistent data model.",
        "home.workflow.card3.title": "03 · Inspect",
        "home.workflow.card3.body": "Use Scene View, Inspector, console filters, and editor-side tooling to shorten the iteration distance.",
        "home.workflow.card4.title": "04 · Run",
        "home.workflow.card4.body": "Run the scene with play-mode isolation, profile the render graph, validate scripting hooks, and keep the loop tight.",

        /* Status */
        "home.status.kicker": "Project status",
        "home.status.title": "Release 0.1.5: technical preview with animation and asset workflow upgrades.",
        "home.status.intro": "The engine has moved beyond static-scene authoring while the roadmap stays clear about what still needs production hardening.",
        "home.status.card1.tag": "CHECKPOINT · 0.1.5",
        "home.status.card1.title": "Runtime, editor, and animation foundation",
        "home.status.card1.body": "Rendering, physics, audio, Python scripting, prefabs, game UI, editor authoring, GUID asset workflows, 2D/3D animation previews, skinned meshes, asset previews, and standalone build are all online.",
        "home.status.card2.tag": "NEXT · TRANSIT",
        "home.status.card2.title": "Advanced UI and content scale",
        "home.status.card2.value": "v0.2 → v0.4",
        "home.status.card2.body": "Upcoming milestones focus on richer UI controls, GPU particles, terrain, onboarding material, and stronger content production paths.",

        /* CTA */
        "cta.title": "Build something that belongs to you.",
        "cta.desc": "熔炉 exists for teams who want source access, architectural clarity, and a workflow they can actually reshape. Start with the docs, inspect the code, and push the engine further.",
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
        "nav.comparison": "架构",
        "nav.roadmap": "路线图",
        "nav.docs": "文档",

        /* Hero */
        "hero.subtitle": "C++ / Vulkan 内核 · Python 生产层 · 仓库优先",
        "hero.viewGithub": "查看仓库",
        "hero.roadmap": "路线图",
        "download.hubInstaller": "下载 Hub 安装器",

        "home.hero.badge": "RELEASE 0.1.5 · 已上线",
        "home.hero.kicker": "开源 · MIT · WIN64",
        "home.hero.title": "一台你可以<span class=\"accent\">拆开重组</span>、扩展并交付的引擎。",
        "home.hero.description": "熔炉（Infernux）是一台从零开始构建的游戏引擎，用 C++17 / Vulkan 原生运行时承载性能关键路径，再用 Python 生产层覆盖玩法、编辑器工具与渲染栈编排。它面向那些希望看懂内部、缩短迭代、且真正能接管 MIT 授权代码库的团队。",
        "home.hero.docs": "阅读文档",
        "home.hero.metric.render": "原生渲染内核",
        "home.hero.metric.python": "玩法与工具",
        "home.hero.metric.license": "无版税",

        "home.panel.manifest.label": "引擎读出",
        "home.panel.manifest.title": "运行时、编辑器、动画与打包共用一套栈。",
        "home.panel.manifest.body": "Vulkan 渲染、Jolt 物理、编辑器外壳、Python 热重载、基于 GUID 的资产工作流、Play 模式隔离、2D/3D 动画预览、蒙皮网格与独立打包，今天都跑在同一份构建里。",

        /* Manifesto */
        "home.manifesto.tag": "FLIGHT NOTE",
        "home.manifesto": "熔炉的立场很简单：性能热路径留在原生层，工作流保持可脚本化，架构保持足够可读，让团队真的能接手这台引擎，而不是和厂商谈判。",

        /* Demo */
        "home.demo.kicker": "实机捕获 · 0.1.5",
        "home.demo.title": "编辑器中实机运行的高度梯度像素化场景，密度直接逼近实际生产。",
        "home.demo.intro": "下面这张截图未经修饰：编辑器处于 Play 模式，Inspector 打开了一个自定义 <code>height_gradient</code> 着色器，运行时只是持续点亮并刷新整片像素网格，没有任何脚本化交互在掩盖负载。",
        "home.demo.panel.title": "这张实机捕获说明了什么",
        "home.demo.panel.item1": "真实编辑器会话 — Hierarchy、Scene/Game/UI 标签、Project、Console 与 Inspector 全部接在运行时上。",
        "home.demo.panel.item2": "Python 中编写的自定义材质：<code>height_gradient.frag</code> 在单一材质槽里着色上百个立方体克隆体。",
        "home.demo.panel.item3": "在测试机上 173 FPS，同时跑着 Play 模式、构建链路与 Inspector。",

        /* System pillars */
        "home.system.kicker": "工作原则",
        "home.system.title": "决定哪些东西能进入引擎的三条规则。",
        "home.system.intro": "这台引擎围绕工程控制权而不是平台绑定来设计。每条原则都为做真实游戏与真实工具的团队减少隐藏成本。",
        "home.system.card1.title": "原则 I · 紧凑迭代环",
        "home.system.card1.body": "Python 脚本、热重载、资产预览与编辑器侧工具把迭代环压在一秒以内，同时不削弱原生运行时核心。",
        "home.system.card2.title": "原则 II · 透明渲染",
        "home.system.card2.body": "RenderGraph、RenderStack 与材质描述符可在 Python 中查看、扩展、脚本化。渲染路径是用来扩展的，不是用来供着的。",
        "home.system.card3.title": "原则 III · 没有商业陷阱",
        "home.system.card3.body": "MIT 协议加上仓库优先的开发节奏，把成本模型摆在明处：工作量在代码里，而不是在和厂商谈判里。",

        /* Capabilities */
        "home.capabilities.kicker": "Operations Board · 0.1.5",
        "home.capabilities.title": "今天已经上线的能力，按真实工程团队评估引擎时关心的方向归类。",
        "home.capabilities.intro": "这是当前技术预览版的现状，按渲染、物理、内容、脚本、工具与资产管线这些真正会被评估的工作面来组织。",
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

        /* Architecture */
        "home.difference.kicker": "栈结构",
        "home.difference.title": "性能关键的部分留给原生，迭代频繁的部分交给 Python。",
        "home.difference.intro": "熔炉用 C++17 承载性能关键系统，把迭代繁重的工作放在 Python 中，pybind11 作为明确的绑定缝合层。",
        "home.stack.title": "分层架构",
        "home.stack.layer1.label": "L4 · 项目与工具",
        "home.stack.layer1.body": "PySide6 Hub 启动器、编辑器面板、项目管理与内容工作流。",
        "home.stack.layer2.label": "L3 · 玩法与渲染",
        "home.stack.layer2.body": "Python 组件、RenderGraph 编排、运行时行为，以及渲染栈脚本化。",
        "home.stack.layer3.label": "L2 · 绑定缝",
        "home.stack.layer3.body": "pybind11 让 Python API 保持可调试的直接性，同时维持原生层明确的所有权边界。",
        "home.stack.layer4.label": "L1 · 原生内核",
        "home.stack.layer4.body": "场景系统、Vulkan 渲染器、Jolt 物理、资源所有权与平台抽象，全部用 C++17 编写。",
        "home.pipeline.title": "Flight Plan：从项目到运行时",
        "home.pipeline.step1.label": "编写",
        "home.pipeline.step1.body": "在 Python 中编写玩法或渲染逻辑，并通过序列化字段暴露到 Inspector。",
        "home.pipeline.step2.label": "描述",
        "home.pipeline.step2.body": "通过 RenderGraph 及相关管线 API 描述 Pass、资源与依赖关系。",
        "home.pipeline.step3.label": "运行",
        "home.pipeline.step3.body": "原生后端负责 Vulkan 调度、屏障、内存与提交。",
        "home.pipeline.step4.label": "迭代",
        "home.pipeline.step4.body": "在编辑器中查看结果、热重载脚本、调整数据，然后继续循环，无需重启项目。",

        /* Workflow */
        "home.workflow.kicker": "实际工作流",
        "home.workflow.title": "从冷启动到运行时验证，整条路径只有四个明确步骤。",
        "home.workflow.intro": "这是引擎在尝试优化的真实日常环路，而不是营销层面的功能堆砌。",
        "home.workflow.card1.title": "01 · 创建",
        "home.workflow.card1.body": "通过 Hub 创建或打开项目，资产由稳定 GUID 与 <code>.meta</code> 文件管理。",
        "home.workflow.card2.title": "02 · 编排",
        "home.workflow.card2.body": "组装场景、挂载脚本、配置材质，并通过统一的数据模型组织渲染行为。",
        "home.workflow.card3.title": "03 · 检查",
        "home.workflow.card3.body": "用 Scene View、Inspector、控制台过滤与编辑器侧工具来缩短迭代距离。",
        "home.workflow.card4.title": "04 · 运行",
        "home.workflow.card4.body": "进入 Play 模式并保持场景隔离，分析 Render Graph、验证脚本钩子，把环路一直跑紧。",

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
