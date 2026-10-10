<!-- language:en -->

<span class="mini-tag">Gameplay Scripting · Chapter 1</span>

# Create and run your first component

Infernux gameplay code lives in Python classes derived from `InxComponent`. An instance is attached to a `GameObject`, receives lifecycle callbacks from the scene, and can reach its owner's `Transform` through `self.transform`.

<div class="learn-article-toc"><strong>In this chapter</strong><a href="#create-the-script">Create the script</a><a href="#attach-and-run">Attach and run</a><a href="#component-model">GameObject and component</a><a href="#lifecycle">Lifecycle</a><a href="#troubleshooting">Common errors</a></div>

<figure class="learn-figure">
  <img src="../assets/learn/gameplay-first-component.webp" alt="A Python component attached to a GameObject and reporting lifecycle events to the Console" loading="lazy" decoding="async">
  <figcaption>A script becomes scene behavior after an <code>InxComponent</code> instance is attached to a <code>GameObject</code>.</figcaption>
</figure>

## Create the script {#create-the-script}

Open a project with the **Project**, **Hierarchy**, **Inspector**, **Game**, and **Console** panels visible. In the Project panel, open `Assets`, right-click empty space, choose **Script (.py)**, and name the asset `HelloComponent`. Script names must be valid Python identifiers; the editor adds `.py` when needed.

Open `HelloComponent.py` and replace its contents with this complete component:

```python
import infernux as inx


class HelloComponent(inx.InxComponent):
    _elapsed_seconds = 0.0
    _reported_first_second = False

    def awake(self) -> None:
        inx.Debug.log(f"Awake: {self.game_object.name}", self)

    def on_enable(self) -> None:
        inx.Debug.log("OnEnable", self)

    def start(self) -> None:
        inx.Debug.log("Start", self)

    def update(self, delta_time: float) -> None:
        self._elapsed_seconds += delta_time
        if not self._reported_first_second and self._elapsed_seconds >= 1.0:
            self._reported_first_second = True
            inx.Debug.log("Update has run for one second", self)

    def on_disable(self) -> None:
        inx.Debug.log("OnDisable", self)

    def on_destroy(self) -> None:
        inx.Debug.log("OnDestroy", self)
```

Ordinary private attributes beginning with `_` stay out of serialization and the Inspector. They are useful for runtime bookkeeping. To save private data explicitly, declare it with `serialized_field(..., hidden=True)`, as described in the next chapter. Do not add an `__init__` method: `InxComponent` owns construction and raises `TypeError` when a subclass overrides it. Use `awake` or `start` for setup.

## Attach and run {#attach-and-run}

1. Create or select any active object in **Hierarchy** and rename it `HelloObject`.
2. In its Inspector, click **Add Component**, search for `HelloComponent`, and add it. The component appears in the object's component list.
3. Save the scene, clear the Console, and enter **Play** mode.
4. Wait one second. The Console should show `Awake: HelloObject`, `OnEnable`, `Start`, and `Update has run for one second` in that order for this component.
5. While still playing, disable the component checkbox and enable it again. `OnDisable` and another `OnEnable` should appear; `Start` remains a one-time callback.
6. Stop Play mode. Teardown can add `OnDisable` and `OnDestroy` entries.

The exercise passes when there are no import or lifecycle exceptions, the one-second message appears only once per component instance, and enable-state changes produce the matching callbacks.

Entering Play replaces the Edit-mode component instance. If that instance has already received `awake`, its teardown can log `OnDestroy` before the fresh Play instance logs `Awake: HelloObject`. These messages belong to different instance lifetimes.

During cleanup, `self.game_object` and the retiring component's fields remain available until its callback returns. If entering Play fails, the original Edit instance remains in place without running its destroy callback.

## GameObject and component {#component-model}

A `GameObject` supplies scene identity, hierarchy, active state, tag, layer, and an always-present `Transform`. Components supply behavior or data. Several components can share one owner.

Inside a live `InxComponent`:

- `self.game_object` returns the owning `GameObject`.
- `self.transform` is the owner's `Transform` shortcut.
- `self.enabled` controls this component's enabled state.
- `self.game_object.get_component(SomeType)` returns the first matching component or `None`.

These owner properties are available during normal bound lifecycle use. Accessing them on a detached or destroyed component raises a runtime error, so cleanup should retain only the data it needs.

## Imports and script reload {#imports-reload}

`infernux` is the engine's Python package. Import it once as `import infernux as inx` and reach everything through `inx`: `inx.InxComponent`, `inx.Vector3`, `inx.RenderStack`, `inx.renderstack`, `inx.jit`. It is an ordinary Python import backed by the same component registry. Installed Python libraries can be imported on first use without an engine import whitelist, and their ordinary factories, decorators, and field metadata are allowed. Older scripts that import `Infernux` should switch to `import infernux as inx`.

Ordinary standard-library declarations are allowed too: for example, `threading.Lock()`, `asyncio.iscoroutine(value)`, and `subprocess.list2cmdline(arguments)`. Reload checks diagnose explicit import-time operations such as starting processes, opening network connections, writing files, or changing interpreter-wide state. Put those operations in the appropriate lifecycle method and release resources when the component stops. The static check is not a sandbox and does not prove arbitrary library code has no side effects.

Saving a script compiles its source and stages its project dependencies before publishing the new revision to existing components. Syntax or import failure keeps the previously published revision running and reports the rejected source. Correcting the source allows the next save to publish; restarting the editor is unnecessary. The transaction protects script publication, not arbitrary external library side effects. Keep scene changes, file writes, and process or thread startup in lifecycle callbacks rather than module declarations.

## Lifecycle at a glance {#lifecycle}

When a scene starts, Infernux first runs `awake` and `on_enable` across active objects, then runs `start` on enabled components. Per-frame execution follows with `update(delta_time)` and then `late_update(delta_time)`. `fixed_update(fixed_delta_time)` runs on the fixed simulation step and belongs to physics work.

| Callback | Use it for |
| --- | --- |
| `awake()` | One-time local setup |
| `on_enable()` / `on_disable()` | Subscribe and unsubscribe active behavior |
| `start()` | Setup that expects other active components to have completed `awake` |
| `update(delta_time)` | Frame-driven gameplay |
| `fixed_update(fixed_delta_time)` | Fixed-step physics logic |
| `late_update(delta_time)` | Work that follows regular updates, such as camera follow |
| `on_destroy()` | Final cleanup |

An inactive GameObject defers `awake` until it first becomes active. `start` runs once, immediately before that component's first simulation update. Regular `update`, `fixed_update`, and `late_update` callbacks run in Play mode.

Edit-mode execution is a separate opt-in with `@inx.execute_in_edit_mode`. You can add or remove this decorator while the component is attached: a successful script reload updates its edit-mode execution setting without replacing the component or losing serialized values. If the candidate fails to publish, the previous setting remains active. Play-mode callbacks continue to run regardless of this decorator.

For opted-in Python components, the editor runs all `update` callbacks before all `late_update` callbacks in the same frame. This supports camera-follow and reflection previews without entering Play. It does not start fixed-step physics; ordinary gameplay components remain inactive.

Edit-mode preview callbacks receive the editor frame duration. Use their `delta_time` argument for preview animation: `Time.delta_time` and `Time.frame_count` belong to the Play clock, which does not advance in Edit mode.

## Common errors {#troubleshooting}

- **The script is absent from Add Component.** Save the file, confirm the class derives from `InxComponent`, and inspect the Console for an import or syntax error.
- **Creation reports an invalid script name.** Use a Python identifier such as `HelloComponent`; spaces, hyphens, and a leading digit are invalid.
- **Class creation fails around `__init__`.** Remove the override and move setup to `awake` or `start`.
- **The Console fills every frame.** Keep routine logging out of an unconditional `update`; the example uses a guard so it reports once.
- **No lifecycle messages appear.** Confirm Play mode is active, the GameObject is active in the hierarchy, and the component checkbox is enabled.
- **A callback receives the wrong arguments.** Use the exact signatures shown above. `update`, `fixed_update`, and `late_update` each receive one time value; the other callbacks shown receive none.

## Next chapter

[GameObjects, components, and authored fields](gameplay-components-fields.html) adds saved Inspector values, scene references, and component constraints to this working lifecycle baseline.

<!-- language:zh -->

<span class="mini-tag">玩法脚本 · 第 1 章</span>

# 创建并运行第一个组件

Infernux 的玩法代码写在继承 `InxComponent` 的 Python 类中。组件实例挂到 `GameObject` 后，会接收场景发出的生命周期回调，也能通过 `self.transform` 访问所属物体的 `Transform`。

<div class="learn-article-toc"><strong>本章内容</strong><a href="#create-the-script_1">创建脚本</a><a href="#attach-and-run_1">挂载并运行</a><a href="#component-model_1">GameObject 与组件</a><a href="#lifecycle_1">生命周期</a><a href="#troubleshooting_1">常见错误</a></div>

<figure class="learn-figure">
  <img src="../assets/learn/gameplay-first-component.webp" alt="Python 组件挂到 GameObject，并把生命周期事件输出到 Console" loading="lazy" decoding="async">
  <figcaption><code>InxComponent</code> 实例挂到 <code>GameObject</code> 后，脚本就会参与场景运行。</figcaption>
</figure>

## 创建脚本 {#create-the-script_1}

打开一个项目，并显示 **Project**、**Hierarchy**、**Inspector**、**Game** 与 **Console** 面板。在 Project 中进入 `Assets`，右键空白处，选择 **脚本 (.py)**，命名为 `HelloComponent`。脚本名必须是有效的 Python 标识符；需要时编辑器会自动补上 `.py`。

打开 `HelloComponent.py`，把内容替换为下面这份完整组件：

```python
import infernux as inx


class HelloComponent(inx.InxComponent):
    _elapsed_seconds = 0.0
    _reported_first_second = False

    def awake(self) -> None:
        inx.Debug.log(f"Awake: {self.game_object.name}", self)

    def on_enable(self) -> None:
        inx.Debug.log("OnEnable", self)

    def start(self) -> None:
        inx.Debug.log("Start", self)

    def update(self, delta_time: float) -> None:
        self._elapsed_seconds += delta_time
        if not self._reported_first_second and self._elapsed_seconds >= 1.0:
            self._reported_first_second = True
            inx.Debug.log("Update has run for one second", self)

    def on_disable(self) -> None:
        inx.Debug.log("OnDisable", self)

    def on_destroy(self) -> None:
        inx.Debug.log("OnDestroy", self)
```

以 `_` 开头的普通私有属性不会进入序列化和 Inspector，适合保存运行时状态。需要明确保存私有数据时，使用下一章介绍的 `serialized_field(..., hidden=True)`。请勿添加 `__init__`：组件构造由 `InxComponent` 管理，子类覆写它时会抛出 `TypeError`。初始化工作放进 `awake` 或 `start`。

## 挂载并运行 {#attach-and-run_1}

1. 在 **Hierarchy** 中创建或选择一个启用的物体，命名为 `HelloObject`。
2. 在 Inspector 底部点击 **Add Component**，搜索 `HelloComponent` 并添加。组件会出现在该物体的组件列表中。
3. 保存场景，清空 Console，然后进入 **Play** 模式。
4. 等待一秒。该组件应依次输出 `Awake: HelloObject`、`OnEnable`、`Start` 和 `Update has run for one second`。
5. 保持运行，关闭组件标题旁的启用复选框，再重新打开。Console 应新增 `OnDisable` 和一次 `OnEnable`；`Start` 只执行一次。
6. 停止 Play。销毁流程可能继续输出 `OnDisable` 与 `OnDestroy`。

没有导入或生命周期异常、一秒提示对每个组件实例只出现一次、切换启用状态能得到对应回调，就算验证通过。

进入 Play 时，引擎会替换编辑态组件实例。如果旧实例已经执行过 `awake`，销毁它时可能先输出 `OnDestroy`，随后新的 Play 实例才输出 `Awake: HelloObject`。这些日志属于不同实例的生命周期。

清理回调返回前，`self.game_object` 与旧组件自己的字段仍然可用。如果进入 Play 失败，原编辑态实例会保留，不会执行它的销毁回调。

## GameObject 与组件 {#component-model_1}

`GameObject` 提供场景身份、层级、活动状态、Tag、Layer，以及始终存在的 `Transform`。组件负责行为或数据；一个物体可以同时拥有多个组件。

在正常运行且已挂载的 `InxComponent` 中：

- `self.game_object` 返回所属 `GameObject`。
- `self.transform` 是所属物体 `Transform` 的快捷入口。
- `self.enabled` 控制当前组件的启用状态。
- `self.game_object.get_component(SomeType)` 返回第一个匹配组件；找不到时返回 `None`。

组件脱离物体或已经销毁后，所属物体属性将不可用。清理逻辑只应使用自己确实需要的数据。

## 导入与脚本热重载 {#imports-reload_1}

`infernux` 是引擎的 Python 包。只需 `import infernux as inx` 导入一次，之后都通过 `inx` 访问：`inx.InxComponent`、`inx.Vector3`、`inx.RenderStack`、`inx.renderstack`、`inx.jit`。这是普通的 Python 导入，背后是同一套组件注册表。已安装的 Python 库可以在首次使用时导入，不需要引擎导入白名单；普通工厂、装饰器和字段元数据也可以使用。旧脚本中的 `Infernux` 导入请改为 `import infernux as inx`。

普通标准库声明也允许使用，例如 `threading.Lock()`、`asyncio.iscoroutine(value)` 和 `subprocess.list2cmdline(arguments)`。重载检查会诊断导入期间明确发生的启动进程、建立网络连接、写文件或修改解释器全局状态等操作。应把这些操作放到合适的生命周期方法里，并在组件停止时释放资源。静态检查不是沙箱，不能证明任意第三方库都没有副作用。

保存脚本后，引擎会先编译源码并准备项目依赖，再把新修订发布给已有组件。语法或导入失败时，上一份已发布修订继续运行，并报告被拒绝的源码；修正后再次保存即可发布，无需重启编辑器。事务保护的是脚本发布，不能回滚任意外部库的副作用。场景修改、文件写入、进程或线程启动应放在生命周期回调中，而不是模块声明阶段。

## 生命周期速览 {#lifecycle_1}

场景启动时，Infernux 会先在活动物体上完成 `awake` 与 `on_enable`，再为启用的组件调用 `start`。随后进入逐帧执行：`update(delta_time)` 完成后运行 `late_update(delta_time)`。`fixed_update(fixed_delta_time)` 按固定模拟步长运行，适合物理逻辑。

| 回调 | 适合处理 |
| --- | --- |
| `awake()` | 一次性的本地初始化 |
| `on_enable()` / `on_disable()` | 订阅与取消订阅活动行为 |
| `start()` | 依赖其他活动组件已完成 `awake` 的初始化 |
| `update(delta_time)` | 按帧推进的玩法逻辑 |
| `fixed_update(fixed_delta_time)` | 固定步长物理逻辑 |
| `late_update(delta_time)` | 跟随常规更新的工作，例如相机跟随 |
| `on_destroy()` | 最终清理 |

非活动 GameObject 会把 `awake` 推迟到第一次激活。`start` 只运行一次，位置在该组件第一次模拟更新之前。普通的 `update`、`fixed_update` 与 `late_update` 只在 Play 模式运行。

编辑模式执行需要通过 `@inx.execute_in_edit_mode` 单独选择加入。组件挂载后仍可以添加或移除这个装饰器：脚本成功热重载时会更新编辑模式执行设置，不替换组件，也不丢失序列化值。如果候选脚本发布失败，之前的设置仍然生效。Play 模式下的回调不受这个装饰器限制。

对选择加入的 Python 组件，编辑器在同一帧先执行所有 `update`，再执行所有 `late_update`。跟随相机和反射预览因此不必进入 Play 才能更新。这不会启动固定步物理，普通游戏逻辑也不会跟着执行。

编辑模式预览回调收到的是编辑器帧时长。预览动画应使用回调的 `delta_time` 参数；`Time.delta_time` 和 `Time.frame_count` 属于 Play 时钟，在编辑模式下不会推进。

## 常见错误 {#troubleshooting_1}

- **Add Component 找不到脚本。** 保存文件，确认类继承 `InxComponent`，再查看 Console 中的导入或语法错误。
- **创建脚本时提示名称无效。** 使用 `HelloComponent` 这类 Python 标识符；空格、连字符和开头数字都不合法。
- **类创建在 `__init__` 附近失败。** 删除该覆写，把初始化移到 `awake` 或 `start`。
- **Console 每帧刷屏。** 避免在无条件 `update` 中持续输出；示例用布尔标记确保只报告一次。
- **看不到生命周期消息。** 检查 Play 模式、GameObject 的层级活动状态和组件启用复选框。
- **回调参数错误。** 使用示例中的准确签名。`update`、`fixed_update`、`late_update` 各接收一个时间值；本章列出的其余回调不接收参数。

## 下一章

[GameObject、组件与可编辑字段](gameplay-components-fields.html)会在这条已跑通的生命周期基线上加入 Inspector 保存值、场景引用和组件约束。
