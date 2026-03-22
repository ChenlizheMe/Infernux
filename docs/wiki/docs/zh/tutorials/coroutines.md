# 协程与时间 — 优雅地"等一下"

<div class="class-info">
教程 &nbsp;|&nbsp; <a href="../../en/tutorials/coroutines.html">English</a>
</div>

## 概述

有时你需要等待。等 2 秒后重生。等玩家走到足够近。等爆炸动画播完。你*可以*用计时器和状态机，但协程让你把这种顺序逻辑写成直白的 Python 生成器。代码读起来像剧本，而不是流程图。

## Time 类

在讲协程之前，先认识 `Time` —— 你通往帧计时的大门。

```python
from InfEngine import *

class TimeDemo(InfComponent):
    def update(self):
        # 游戏启动以来的时间（受 time_scale 影响）
        Debug.log(f"时间: {Time.time:.2f}")
        
        # 上一帧耗时
        Debug.log(f"帧耗时: {Time.delta_time:.4f}")
        
        # 慢动作！
        if Input.get_key_down(KeyCode.T):
            Time.time_scale = 0.25  # 四分之一速度
        
        # 恢复正常
        if Input.get_key_down(KeyCode.Y):
            Time.time_scale = 1.0
```

### 关键属性

| 属性 | 说明 |
|------|------|
| `Time.time` | 启动以来的秒数（受缩放） |
| `Time.delta_time` | 上一帧耗时（受缩放）。**移动必用！** |
| `Time.unscaled_time` | 真实秒数（不受 time_scale 影响） |
| `Time.unscaled_delta_time` | 真实帧耗时 |
| `Time.fixed_delta_time` | 物理时间步（默认 0.02 = 50Hz） |
| `Time.time_scale` | 速度倍率。0=暂停，1=正常，2=双倍速 |
| `Time.frame_count` | 累计渲染帧数 |

> **黄金法则：** 任何速度都要乘以 `Time.delta_time`。否则你的游戏在 120 FPS 时快两倍，30 FPS 时慢一半。没人想要这样。

## 你的第一个协程

协程就是一个 `yield` 特殊等待对象的 Python 生成器：

```python
class FirstCoroutine(InfComponent):
    def start(self):
        self.start_coroutine(self.count_down())
    
    def count_down(self):
        Debug.log("3...")
        yield WaitForSeconds(1)
        Debug.log("2...")
        yield WaitForSeconds(1)
        Debug.log("1...")
        yield WaitForSeconds(1)
        Debug.log("出发！🚀")
```

搞定。不需要状态机。不需要计时器变量。只要 `yield` 然后等就好。

## 等待类型

| 类 | 等什么 | 受 time_scale 影响？ |
|----|--------|---------------------|
| `WaitForSeconds(n)` | `n` 秒游戏时间 | ✅ 是 |
| `WaitForSecondsRealtime(n)` | `n` 秒真实时间 | ❌ 否 |
| `WaitForEndOfFrame()` | 当前帧结束 | — |
| `WaitForFixedUpdate()` | 下一个物理步 | — |
| `WaitUntil(predicate)` | 直到 `predicate()` 返回 `True` | — |
| `WaitWhile(predicate)` | 只要 `predicate()` 返回 `True` 就继续等 | — |

### 示例

```python
class WaitExamples(InfComponent):
    def start(self):
        self.start_coroutine(self.various_waits())
    
    def various_waits(self):
        # 等 2 秒游戏时间（time_scale=0 时会暂停）
        yield WaitForSeconds(2)
        
        # 等 2 秒真实时间（暂停时也在走 — 适合暂停菜单）
        yield WaitForSecondsRealtime(2)
        
        # 等玩家按空格
        yield WaitUntil(lambda: Input.get_key_down(KeyCode.SPACE))
        
        # 玩家按住 Shift 就一直等
        yield WaitWhile(lambda: Input.get_key(KeyCode.LEFT_SHIFT))
        
        # 等一个物理步
        yield WaitForFixedUpdate()
        
        # 等帧结束（适合截图）
        yield WaitForEndOfFrame()
```

## 管理协程

### 启动

```python
self.my_routine = self.start_coroutine(self.do_stuff())
```

### 停止

```python
# 停止特定协程
self.stop_coroutine(self.my_routine)

# 停止此组件上的所有协程
self.stop_all_coroutines()
```

### 串联（协程中调用协程）

```python
def main_sequence(self):
    Debug.log("阶段 1：热身")
    yield from self.warm_up()
    Debug.log("阶段 2：战斗")
    yield from self.battle_phase()
    Debug.log("阶段 3：胜利")

def warm_up(self):
    yield WaitForSeconds(3)

def battle_phase(self):
    yield WaitUntil(lambda: self.enemies_dead())
```

## 常用模式

### 延迟动作

```python
class Grenade(InfComponent):
    fuse_time: float = serialized_field(default=3.0)
    
    def start(self):
        self.start_coroutine(self.explode_after_delay())
    
    def explode_after_delay(self):
        yield WaitForSeconds(self.fuse_time)
        Debug.log("BOOM！💥")
        self.destroy(self.game_object)
```

### 循环生成

```python
class Spawner(InfComponent):
    interval: float = serialized_field(default=2.0)
    
    def start(self):
        self.start_coroutine(self.spawn_loop())
    
    def spawn_loop(self):
        while True:
            self.spawn_enemy()
            yield WaitForSeconds(self.interval)
    
    def spawn_enemy(self):
        Debug.log("生成敌人...")
```

### 平滑移动

```python
class SmoothMove(InfComponent):
    def move_to(self, target, duration):
        self.start_coroutine(self._do_move(target, duration))
    
    def _do_move(self, target, duration):
        start = self.transform.position
        elapsed = 0.0
        while elapsed < duration:
            t = elapsed / duration
            t = t * t * (3 - 2 * t)  # SmoothStep
            self.transform.position = vector3.lerp(start, target, t)
            elapsed += Time.delta_time
            yield WaitForEndOfFrame()
        self.transform.position = target
```

### 打字机效果

```python
class Typewriter(InfComponent):
    chars_per_second: float = serialized_field(default=30.0)
    
    def show_text(self, full_text):
        self.start_coroutine(self._type(full_text))
    
    def _type(self, full_text):
        text_comp = self.game_object.get_py_component(UIText)
        for i in range(len(full_text) + 1):
            text_comp.text = full_text[:i]
            yield WaitForSeconds(1.0 / self.chars_per_second)
```

## Mathf 工具类

协程里经常需要数学辅助函数。`Mathf` 全都有：

```python
# 限制范围
hp = Mathf.clamp(hp - damage, 0, max_hp)

# 平滑插值
value = Mathf.lerp(start, end, t)
value = Mathf.smooth_step(0, 1, t)

# 匀速趋近目标
pos = Mathf.move_towards(current, target, speed * Time.delta_time)

# 三角函数
y = Mathf.sin(Time.time * frequency) * amplitude
```

## 另请参阅

- [Time API](../api/Time.md)
- [Mathf API](../api/Mathf.md)
- [Coroutine API](../api/Coroutine.md)
- [WaitForSeconds API](../api/WaitForSeconds.md)
- [UI 教程](ui.md) — 协程驱动的 UI 动画
