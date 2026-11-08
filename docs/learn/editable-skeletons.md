# 可编辑骨骼与蒙皮

导入带骨骼的模型后，角色的 SkeletonInstance 会展开完整节点层级，包括没有蒙皮权重的辅助节点。每个节点都是普通 GameObject/Transform。动画不是访问骨骼的前提；没有 SkeletalAnimator 也可以修改姿态，或者把普通物理组件挂在骨骼对象上。

```python
from infernux import InxComponent
from infernux.lib import Vector3

class AimHead(InxComponent):
    def start(self):
        self.skeleton = self.game_object.get_component('SkeletonInstance')
        self.head = self.skeleton.get_bone_transform('Head')

    def late_update(self, dt):
        # 动画已经完成采样；这里的修正会进入这一帧最终的蒙皮。
        self.head.local_euler_angles = Vector3(0, 25, 0)
```

节点名称来自模型；上例中的 Head 需要换成实际名称。骨架提供 bone_names、parent_indices、bones 和 root。get_bone_transform(name) 可以访问普通节点；get_human_bone_transform(slot) 依赖已经配置好的 Humanoid 定义。get_local_matrices、get_root_space_matrices、get_world_matrices 分别读取局部、角色根空间和世界空间姿态，返回 float32[N,4,4]，按 [节点,行,列] 索引。get_bind_local_matrices 读取导入绑定姿态；reset_to_bind_pose 恢复绑定姿态。

执行顺序统一为游戏 update、animation_update、late_update、最终蒙皮发布。SkeletalAnimator 在 animation_update 中采样，同一个 SkeletonInstance 的多个蒙皮只采样一次，嵌套 Animator 各自控制自己的角色。游戏逻辑可以在 update 中更新动画参数，在 late_update 中修正骨骼。暂停时的单步也遵循相同顺序。

SkinnedMeshRenderer.skeleton 指向姿态所属的 SkeletonInstance，通常会自动绑定同一角色的骨架；bones 列表对应这个网格的蒙皮调色板条目，而 SkeletonInstance.bones 对应完整节点层级。两个列表的数量和顺序不必相同，同一骨骼也可能因不同 inverse bind 在蒙皮列表中出现多次。get_skin_matrices(previous=False) 可复制最终 CPU 调色板用于检查；previous=True 读取上一渲染帧的调色板。这不是 GPU 回读。

角色整体的位置、旋转和缩放放在 SkeletonInstance 所属的角色根上。蒙皮结果已经包含骨骼到角色根的变换，渲染器所在的子对象不会再额外叠加一个模型节点偏移。包围盒、子网格、阴影与粒子网格消费使用相同空间约定。

保存、克隆和 Prefab 实例化保留姿态并重映射骨架引用，两个角色不会共享可变骨骼。资源重导入更新未被作者覆盖的绑定姿态。要恢复被删除的源节点或源节点的父子关系，调用 rebuild_hierarchy。缺失骨骼会使对应角色的蒙皮失效，严格骨骼查询会给出原因；其他角色继续工作。给渲染器显式赋不兼容的骨架会在赋值处报告错误并保留原绑定。
