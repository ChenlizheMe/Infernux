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

## 创建数据资产和构建列表

运行时数据类型放在 `Assets/Scripts/LevelConfig.py`，用 `DataAsset` 和 `serialized_field` 声明。编辑器工具可以用 `from Scripts.LevelConfig import LevelConfig` 引用它；`Assets` 是项目模块根，不是 Python 包名。

```python
from Scripts.LevelConfig import LevelConfig

with inx.editor.edit_scene("创建关卡配置"):
    inx.editor.create_folder("Assets/NewLevel")
    path = inx.editor.create_data_asset(
        LevelConfig(title="Garden"), "Assets/NewLevel/Level.inxdata",
    )
config = inx.DataAsset.load(path)
```

`create_data_asset` 保存传入值的独立副本，由资产系统创建 GUID 和导入结果，不改变原对象的持久身份。目标文件必须不存在，目录必须已建立；撤销移除新资产，重做保留原 GUID 和字段值。它不覆盖已有配置；已有资产应通过 Inspector 的文档编辑流程修改。

作者脚本可以操作同一份文档，无需打开 Inspector：

```python
level = inx.editor.load_data_asset("Assets/Data/Level01.inxdata")
with inx.editor.edit_scene("更新关卡配置"):
    inx.editor.set_data_asset_fields(level, title="第一关", difficulty=3)
result = inx.editor.save_data_asset(level)
```

`load_data_asset` 读取编辑器当前文档，包括尚未自动保存的修改；不要用直接读取磁盘的 `DataAsset.load` 来检查刚完成的撤销。`set_data_asset_fields` 先在独立副本中验证全部声明字段，再形成一条可撤销修改；只读或未知字段会拒绝整个调用，相同值返回 `False`。多份资产的修改可以放在同一个编辑分组内。`save_data_asset` 使用现有保存事务，仍需检查 `APPLIED/PENDING/FAILED`，不绕过自动保存或外部冲突处理。

关卡目录等资产列表可声明为 `inx.list_field(element_type=inx.FieldType.ASSET, asset_type="DataAsset")`，列表中保存资产引用，而不是复制每份资产的内容。

作者工具添加组件时，使用 `inx.editor.add_component(obj, "MyComponent", configure=initialize)`。它与 Inspector 共用组件添加服务，解析已发布的脚本和资产身份，并把初始化值纳入撤销/重做。直接从旧的导入类构造实例再调用 `add_py_component`，会绕过这条作者工具链路。

普通的 `obj.add_component(MyComponent)` 与 `obj.get_component(MyComponent)` 也会把保留的类句柄解析到当前已发布脚本类型，包括 preload 在首次资产 GUID 发布前导入的类。匹配按所属模块和限定类名，不会混淆不同脚本中的同名类。直接调用 `MyComponent()` 再传给低层 `add_py_component` 不执行这一步身份解析；编辑已有物体且需要撤销时，仍使用 `inx.editor.add_component`。

工具需要加载已有材质时，先用 `inx.Application.asset_path("Assets/Materials/Example.mat")` 解析项目路径，再传给 `inx.AssetManager.load`。不要依赖启动编辑器时的工作目录，也不需要在脚本里手写 GUID。

`get_build_scenes()` 返回独立的、有序的项目相对路径列表。`set_build_scenes(paths)` 用一条可撤销操作更新同一份 Build Settings；路径必须是 `Assets` 下已存在的 `.scene`。传空列表表示清空，相同列表不新增历史。编辑器的自动保存仍生效，也可显式调用 `save_project_settings()`，其返回状态与场景保存相同，不要把 `PENDING` 当成写盘完成。

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

在 Prefab 内改变物体的父级不会改变其身份。源资产的层级修改可以与实例的属性修改合并，实例自己的父级覆盖也会保留。如果两端的修改合成了父子循环，Apply 会在写入资产前拒绝该冲突；调整层级后再应用。

## 还原单个属性覆盖

```python
inx.editor.revert_property_override(component, "amount")
```

从最近一层 Prefab 源还原指定的序列化字段，形成一条可撤销操作。其它字段的修改、物体身份和组件引用保持不变，也不会修改源资产。返回 `False` 表示没有需要记录的变化。

字段名使用公开的 Python 名称，例如碰撞体的 `size`、Transform 的 `local_position`。显式还原根 Transform 字段会恢复源值；整体 Revert 则保留根物体在场景中的摆放。集合字段按整个字段还原，目前不接受嵌套属性路径、实例新增组件或只读字段。如果源引用指向实例中已删除的成员，应先恢复该成员，不会借用其它物体的 ID 补上引用。

## 查询属性覆盖

```python
if inx.editor.is_property_override(component, "amount"):
    inx.editor.revert_property_override(component, "amount")

for change in inx.editor.get_property_modifications(instance):
    print(change.object_id, change.component_id, change.property_path,
          change.source_value, change.instance_value)
```

查询与 Apply/Revert 共用源身份和引用比较规则，同名物体、同类型组件不会混在一起。
`is_property_override` 接受公开 Python 字段名，也可以查询只读序列化字段。未声明的字段会报错；普通物体和实例新增组件没有对应的源属性覆盖。

`get_property_modifications` 返回最近一层 Prefab 及其子树的独立数据快照。记录包含场景的 `object_id/component_id` 和资产内的 `source_object_id/source_component_id`，不依靠名称或排列下标定位。组件字段路径为 `data.<序列化字段名>`，Transform 使用 `position/rotation/scale`。值采用现有序列化格式：源引用使用资产局部 ID，实例引用使用场景 ID；集合按整个字段比较，不拆成单独数组元素。

根物体摆放和组织属性的差异也会返回，并标记 `is_default_override`，对应整体 Apply/Revert 原有的保留规则。新增、删除物体或组件以及排序属于结构覆盖，不混入属性记录。这里返回当前值与源资产的差异，不是历史修改日志，也不模拟 Unity 已失效的历史 override。查询不会写盘、增加 Undo 或推进合并基线；修改返回数据不会改变资产。

## 嵌套 Prefab

把 Prefab 实例放在另一个待保存的层级下，再保存外层 Prefab，内层实例会保留自己的资产链接。同一内层资产可以出现多次，引用仍指向各自实例中的物体和组件，不会因为源 ID 相同而串到另一份实例。

- 对内层实例调用 `apply_prefab`，修改的是内层源；对外层根调用，则把该修改记录在外层资产中。
- 外层 Apply/Revert、Undo/Redo 和 Prefab Mode 保存保留内层链接。复制内层实例会创建新的外层成员，不会复用原成员的身份。
- 解包外层实例保留内层 Prefab 链接；这不是递归解包所有层。
- 把内层实例移到另一份外层实例下，会解除旧外层的成员身份，保留内层自身的资产链接。Undo 恢复原归属，Redo 再次执行移动。同一外层实例内部换父节点不会改变成员身份。
- 普通源节点移出所属实例后，不再保留原实例的成员身份。调整子节点顺序应使用兄弟排序，不要用先脱离再挂回的方式；需要恢复脱离前的归属时使用 Undo。
- 场景重开、重新实例化和构建解析内层源的最新内容，即使外层文件没有变化。循环嵌套会报错，不会展开成无限层级。

这些入口不是完整的 Unity Editor SDK。项目作者工具的完整迁移仍是单独的验收任务。
