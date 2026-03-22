# 物理系统 — 让东西掉下来（然后撞在一起）

<div class="class-info">
教程 &nbsp;|&nbsp; <a href="../../en/tutorials/physics.html">English</a>
</div>

## 概述

InfEngine 底层使用 **Jolt Physics** —— 一个连《地平线：西之绝境》都在用的物理引擎。你可以用刚体、五种碰撞体、射线检测、触发器，模拟从台球到城市拆迁的一切场景。

## 你的第一个物理对象

每个物理对象需要两样东西：**碰撞体**（形状）和**刚体**（物理行为）。就像给物体一个身体，然后告诉它"欢迎来到有重力的世界"。

### 第一步：添加碰撞体

在编辑器中选中一个 GameObject，添加碰撞体组件：

```python
from InfEngine import *

class PhysicsSetup(InfComponent):
    def start(self):
        # 给这个对象加上物理
        box = self.game_object.add_component("BoxCollider")
        rb = self.game_object.add_component("Rigidbody")
        
        Debug.log("物理引擎接管了。祝你好运。")
```

或者直接在 Inspector 面板操作 —— 点击 **Add Component → Physics → BoxCollider** 和 **Rigidbody**。

### 第二步：看它掉下来

按下 Play。你的对象掉了。这就是重力。恭喜，牛顿会为你骄傲的。

## 碰撞体类型

| 碰撞体 | 适用场景 | 性能 |
|--------|---------|------|
| `BoxCollider` | 箱子、墙壁、地板、建筑 | ⚡ 快 |
| `SphereCollider` | 球、子弹、触发区域 | ⚡ 最快 |
| `CapsuleCollider` | 角色、人形物体 | ⚡ 快 |
| `MeshCollider` | 复杂的静态几何体 | 🐢 费性能 |

> **小贴士：** `MeshCollider` 只用于静态物体。运动物体请用基本碰撞体组合。你的帧率会感谢你的。

### 配置碰撞体

```python
class ColliderDemo(InfComponent):
    def start(self):
        # 自定义盒体大小
        box = self.game_object.get_component(BoxCollider)
        box.size = vector3(2, 1, 3)
        box.center = vector3(0, 0.5, 0)
        
        # 球形碰撞体
        sphere = self.game_object.get_component(SphereCollider)
        sphere.radius = 1.5
        
        # 角色用胶囊体
        cap = self.game_object.get_component(CapsuleCollider)
        cap.radius = 0.5
        cap.height = 2.0
```

## 刚体 (Rigidbody)

`Rigidbody` 组件让物体服从物理规律。没有它，碰撞体就只是隐形的墙。

### 关键属性

```python
class RigidbodyDemo(InfComponent):
    def start(self):
        rb = self.game_object.get_component(Rigidbody)
        
        rb.mass = 10.0              # 重量级选手（千克）
        rb.drag = 0.5               # 空气阻力
        rb.angular_drag = 0.05      # 旋转阻力
        rb.use_gravity = True       # 会掉下来（通常你都想要这个）
        rb.is_kinematic = False     # 服从物理力
```

### 施加力

```python
class Rocket(InfComponent):
    thrust: float = serialized_field(default=100.0)
    
    def fixed_update(self):
        rb = self.game_object.get_component(Rigidbody)
        
        # 持续推力 — 使用 ForceMode.Force
        if Input.get_key(KeyCode.SPACE):
            rb.add_force(vector3(0, self.thrust, 0))
        
        # 一次性冲量 — 像跳跃
        if Input.get_key_down(KeyCode.W):
            rb.add_force(vector3(0, 500, 0), ForceMode.Impulse)
        
        # 在某个点施加力 — 会产生旋转！
        if Input.get_key_down(KeyCode.E):
            hit_point = self.transform.position + vector3(1, 0, 0)
            rb.add_force_at_position(vector3(0, 200, 0), hit_point)
```

> **重要：** 施加力永远写在 `fixed_update()` 里，不要写在 `update()` 里。物理引擎按固定时间步运行，混用只会得到"神秘抖动的物体"™。

### 运动学刚体

运动学刚体不受力影响，但能推动其他物体。非常适合做移动平台、门、以及每个游戏都有的那种电梯谜题。

```python
class MovingPlatform(InfComponent):
    speed: float = serialized_field(default=2.0)
    
    def start(self):
        rb = self.game_object.get_component(Rigidbody)
        rb.is_kinematic = True
    
    def fixed_update(self):
        rb = self.game_object.get_component(Rigidbody)
        new_y = Mathf.sin(Time.time * self.speed) * 3.0
        rb.move_position(vector3(0, new_y, 0))
```

## 触发器

给碰撞体设置 `is_trigger = True` 就可以检测重叠而不产生物理碰撞。适合拾取物、区域检测、隐形死亡墙。

```python
class PickupZone(InfComponent):
    def start(self):
        collider = self.game_object.get_component(BoxCollider)
        collider.is_trigger = True
    
    def on_trigger_enter(self, other):
        Debug.log(f"{other.game_object.name} 进入了区域！")
    
    def on_trigger_exit(self, other):
        Debug.log(f"{other.game_object.name} 离开了。我们很想念它。")
```

## 射线检测

向场景中发射一条隐形射线，看看它击中了什么。用于视线检测、射击、鼠标拾取、地面检测。

```python
class RaycastDemo(InfComponent):
    def update(self):
        origin = self.transform.position
        direction = self.transform.forward
        
        hit = Physics.raycast(origin, direction, max_distance=100.0)
        if hit:
            Debug.log(f"击中了 {hit.collider.game_object.name}，位置 {hit.point}")
            Debug.draw_line(origin, hit.point, vector3(1, 0, 0))
```

## 冻结轴

有时你不想让物理引擎旋转你的物体（2D 游戏：说的就是你）。用约束：

```python
class FreezeRotation(InfComponent):
    def start(self):
        rb = self.game_object.get_component(Rigidbody)
        rb.constraints = (
            RigidbodyConstraints.FreezeRotationX |
            RigidbodyConstraints.FreezeRotationY |
            RigidbodyConstraints.FreezeRotationZ
        )
```

## 常用模式

### 地面检测

```python
class GroundCheck(InfComponent):
    is_grounded: bool = False
    
    def fixed_update(self):
        origin = self.transform.position
        self.is_grounded = Physics.raycast(
            origin, vector3(0, -1, 0), max_distance=1.1
        ) is not None
```

### 简单角色控制器

```python
class SimpleCharacter(InfComponent):
    speed: float = serialized_field(default=5.0)
    jump_force: float = serialized_field(default=8.0)
    
    def fixed_update(self):
        rb = self.game_object.get_component(Rigidbody)
        
        # 移动
        h = Input.get_axis("Horizontal")
        v = Input.get_axis("Vertical")
        move = vector3(h, 0, v) * self.speed
        rb.velocity = vector3(move.x, rb.velocity.y, move.z)
        
        # 跳跃
        if Input.get_key_down(KeyCode.SPACE):
            hit = Physics.raycast(
                self.transform.position, vector3(0, -1, 0), 1.1
            )
            if hit:
                rb.add_force(vector3(0, self.jump_force, 0), ForceMode.Impulse)
```

## 另请参阅

- [Rigidbody API](../api/Rigidbody.md)
- [BoxCollider API](../api/BoxCollider.md)
- [Physics API](../api/Physics.md)
- [协程](coroutines.md) — 延迟执行物理操作
