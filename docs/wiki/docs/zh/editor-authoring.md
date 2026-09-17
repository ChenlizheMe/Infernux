# 编写编辑器工具

`inx.editor` 是项目自定义编辑器工具的公开入口。创建物体、Prefab、保存和撤销都沿用编辑器现有服务，不需要脚本自行拼装 Scene JSON。它要求已打开的编辑器会话，Play 模式不可用；不要把这些调用放进游戏运行脚本。

把下面的脚本保存为 `Assets/Editor/CreateObstacle.py`。加载后按 F9，或在命令面板中搜索“创建机关”，即可创建机关层级、保存为 Prefab 并放置一个实例。

## 创建与撤销

```python
import infernux as inx

class ObstacleTools(inx.InxPreload):
    def preload(self, context):
        inx.editor.EditorCommandRegistry.instance().register(
            inx.editor.EditorCommand(
                "my_game.create_obstacle", self.create_obstacle,
                display_name="创建机关",
            )
        )
        inx.editor.ShortcutRouter.instance().register(
            inx.editor.ShortcutBinding(
                "my_game.create_obstacle", inx.editor.KeyChord("F9"),
                binding_id="my_game.create_obstacle.f9",
            )
        )

    def create_obstacle(self, context):
        with inx.editor.edit_scene("创建机关"):
            root = inx.editor.create_game_object("机关")
            inx.editor.create_game_object("主体", kind="primitive.cube", parent=root)
            path = inx.editor.create_prefab(root, "Assets")
            instance = inx.editor.instantiate_prefab(path)
        return instance.id
```

一次撤销会移除这组创建的物体、资产和实例，重做会恢复它们。注册操作放在 `preload` 中：热重载、卸载及关闭项目时，注册项随所属脚本清理，不要每帧注册，也不要在模块导入时直接创建场景物体。

- `create_game_object` 返回新物体；`kind` 与 Hierarchy 创建类型共用，例如 `empty`、`primitive.sphere`、`ui.button`。
- 初始组件和 Transform 值可以放进 `configure(obj)` 回调，在创建快照记录前完成，撤销/重做能恢复这些初值。回调抛出异常会取消该物体的创建。
- `edit_scene` 只合并已经走编辑器历史服务的操作，不会自动拦截任意 Python 赋值，也不是整个脚本失败时的全量回滚事务。
- `create_prefab(obj, directory)` 的第二个参数是目录，不是文件名；它创建唯一名称的资产并建立源链接，不覆盖已有 Prefab。
- `instantiate_prefab` 返回新实例；`apply_prefab(obj)` 应用覆盖，`revert_prefab(obj)` 恢复源值，均进入全局撤销历史。

## 保存和切换场景

`save_scene("Assets/Scenes/关卡.scene")` 通过正常保存事务另存当前场景；省略路径则保存当前文档，无路径时请求另存对话框。返回结果不能一律当作“保存完成”：

- `APPLIED` / `NO_OP`：操作完成 / 没有需要保存的更改。
- `PENDING`：仍在等待对话框或异步操作。
- `REJECTED` / `FAILED`：查看 `message`，不要继续报告保存成功。

路径相对于项目目录，也可传绝对路径，最终仍受项目保存边界约束。

`new_scene()` 和 `open_scene(path)` 保留未保存更改确认和延迟切换规则。函数返回不代表新场景已经加载，不能紧接着往“目标场景”创建物体。

`undo()` / `redo()` 默认等待编辑器安全时点再执行。注册这类命令时设 `creates_user_action=False`，避免把历史重放又当作新编辑。`defer=False` 只用于调用者已经处于安全时点的非渲染宿主，不用于界面回调。

Prefab 的组件源身份与运行时 ID 分开保存。删除或重排同类型组件后，Apply/Revert、Undo/Redo 和场景同步仍按身份找到原组件；实例新增的组件在应用前保持私有，已删除的源 ID 不会分配给之后的新组件。

没有组件源身份的旧场景只迁移一次，按保存的基线和同类型组件顺序建立链接。如果旧场景在记录身份之前就删除了某个同类型组件，缺失的历史无法凭空恢复。

组件引用同时保存所属物体、组件类型和精确组件 ID。脚本直接赋值组件实例即可；Inspector 选择器会逐个列出组件，拖入 GameObject 则绑定第一个符合类型的组件。删除目标后，引用返回 `None`，不会转而指向另一个同类型组件；保存 Prefab 资产前需要清除失效引用。

旧的仅按类型记录的引用仍可读取，下次保存时写入精确 ID。复制、Prefab 操作和场景载入会同步重映射内部引用；指向复制范围之外的场景对象时，保留外部引用。

这些入口不是完整的 Unity Editor SDK。嵌套 Prefab 仍是待完成的独立工作。
