# 音频系统 — 让你的游戏不再寂静

<div class="class-info">
教程 &nbsp;|&nbsp; <a href="../../en/tutorials/audio.html">English</a>
</div>

## 概述

没有声音的游戏就是个带鼠标交互的屏保。InfEngine 的音频系统提供 **3D 空间化音频**、多轨播放、一次性音效，以及至关重要的音量滑块。底层走的是 C++ 音频引擎 —— 低延迟，不卡顿。

## 核心概念

| 组件 | 作用 |
|------|------|
| `AudioSource` | 扬声器 — 在场景中的某个位置播放音频 |
| `AudioListener` | 耳朵 — 通常挂在主摄像机上，场景中只能有一个！ |
| `AudioClip` | 音频文件 — WAV, OGG, MP3 |

> **经验法则：** 主摄像机上挂一个 `AudioListener`，场景中需要多少 `AudioSource` 就放多少。

## 快速开始：播放一个声音

```python
from InfEngine import *

class PlaySound(InfComponent):
    def start(self):
        source = self.game_object.get_component(AudioSource)
        source.play()
        Debug.log("🔊 声音正在播放！")
```

就这么简单。前提是你在 Inspector 里分配了音频剪辑。

## 在编辑器中设置音频

1. 选中一个 GameObject（或者创建一个空物体用于环境音）
2. **Add Component → Audio → AudioSource**
3. 把音频文件拖到 Inspector 的 **Clip** 字段
4. 如果需要自动播放，勾选 **Play On Awake**
5. 给主摄像机添加 **AudioListener**（编辑器通常已经帮你做了）

## 通过脚本播放音频

### 基本播放

```python
class MusicPlayer(InfComponent):
    def start(self):
        source = self.game_object.get_component(AudioSource)
        source.volume = 0.7
        source.loop = True
        source.play()
    
    def update(self):
        source = self.game_object.get_component(AudioSource)
        
        # 暂停 / 继续
        if Input.get_key_down(KeyCode.P):
            if source.is_playing:
                source.pause()
            else:
                source.unpause()
        
        # 停止
        if Input.get_key_down(KeyCode.S):
            source.stop()
```

### 一次性音效

用 `play_one_shot()` 播放临时音效（枪声、脚步声、UI 点击声）。不会打断正在播放的音频。

```python
class Weapon(InfComponent):
    fire_sound: AudioClip = serialized_field()
    
    def update(self):
        if Input.get_mouse_button_down(0):
            source = self.game_object.get_component(AudioSource)
            source.play_one_shot(self.fire_sound)
            Debug.log("砰砰！")
```

## 3D 空间音频

用 `spatial_blend` 控制音频受听者距离和方向的影响程度：

- `0.0` = 纯 2D（哪里都一样响 — 适合背景音乐/UI 音效）
- `1.0` = 纯 3D（越远越小声 — 适合世界里的声音）

```python
class SpatialAudio(InfComponent):
    def start(self):
        source = self.game_object.get_component(AudioSource)
        source.spatial_blend = 1.0      # 完全 3D
        source.min_distance = 1.0       # 1 米内满音量
        source.max_distance = 50.0      # 50 米外静音
        source.loop = True
        source.play()
```

走近或远离音源，你会听到音量变化。不是魔法，是反平方定律。

## 常用模式

### 脚步声系统

```python
class Footsteps(InfComponent):
    step_interval: float = serialized_field(default=0.4)
    step_sound: AudioClip = serialized_field()
    _timer: float = 0
    
    def update(self):
        h = Input.get_axis("Horizontal")
        v = Input.get_axis("Vertical")
        if abs(h) > 0.1 or abs(v) > 0.1:
            self._timer += Time.delta_time
            if self._timer >= self.step_interval:
                self._timer = 0
                source = self.game_object.get_component(AudioSource)
                source.play_one_shot(self.step_sound)
```

### 背景音乐管理器

```python
class BGMManager(InfComponent):
    """跨场景存活，渐入渐出切换。"""
    fade_speed: float = serialized_field(default=1.0)
    _target_volume: float = 1.0
    
    def start(self):
        source = self.game_object.get_component(AudioSource)
        source.loop = True
        source.spatial_blend = 0.0  # 2D — 到处都能听到
        source.play()
    
    def update(self):
        source = self.game_object.get_component(AudioSource)
        source.volume = Mathf.move_towards(
            source.volume, self._target_volume,
            self.fade_speed * Time.delta_time
        )
    
    def fade_out(self):
        self._target_volume = 0.0
    
    def fade_in(self):
        self._target_volume = 1.0
```

## 支持的格式

| 格式 | 推荐用途 |
|------|---------|
| WAV | 短音效（无压缩，加载快） |
| OGG | 音乐和较长片段（压缩，文件小） |
| MP3 | 也支持（但 OGG 通常更好） |

## 另请参阅

- [AudioSource API](../api/AudioSource.md)
- [AudioListener API](../api/AudioListener.md)
- [AudioClip API](../api/AudioClip.md)
- [物理教程](physics.md) — 碰撞时触发声音
