# Audio — Making Your Game Not Silent

<div class="class-info">
Tutorial &nbsp;|&nbsp; <a href="../../zh/tutorials/audio.html">中文</a>
</div>

## Overview

A game without sound is just a screensaver with extra steps. InfEngine's audio system gives you **3D spatialized audio**, multi-track playback, one-shot sound effects, and the all-important volume slider. Under the hood, everything routes through the C++ audio engine — meaning low latency and no Python-side stuttering.

## Core Concepts

| Component | Role |
|-----------|------|
| `AudioSource` | The speaker — plays audio clips at a position in the scene |
| `AudioListener` | The ear — usually on the main camera. Only one active at a time! |
| `AudioClip` | The audio file — WAV, OGG, MP3 |

> **Rule of thumb:** One `AudioListener` on your camera, as many `AudioSource` components as you need scattered across the scene.

## Quick Start: Play a Sound

```python
from InfEngine import *

class PlaySound(InfComponent):
    def start(self):
        source = self.game_object.get_component(AudioSource)
        source.play()
        Debug.log("🔊 Sound is playing!")
```

That's it. Assuming you've assigned an audio clip in the Inspector, you're golden.

## Setting Up Audio in the Editor

1. Select a GameObject (or create an empty one for ambient sound)
2. **Add Component → Audio → AudioSource**
3. Drag an audio file into the **Clip** field in the Inspector
4. Toggle **Play On Awake** if you want it to start automatically
5. Add **AudioListener** to your main camera (the editor usually does this for you)

## Playing Audio from Script

### Basic Playback

```python
class MusicPlayer(InfComponent):
    def start(self):
        source = self.game_object.get_component(AudioSource)
        source.volume = 0.7
        source.loop = True
        source.play()
    
    def update(self):
        source = self.game_object.get_component(AudioSource)
        
        # Pause / unpause
        if Input.get_key_down(KeyCode.P):
            if source.is_playing:
                source.pause()
            else:
                source.unpause()
        
        # Stop
        if Input.get_key_down(KeyCode.S):
            source.stop()
```

### One-Shot Sound Effects

Use `play_one_shot()` for quick fire-and-forget sounds (gunshots, footsteps, UI clicks). It plays the clip once without interrupting whatever is already playing.

```python
class Weapon(InfComponent):
    fire_sound: AudioClip = serialized_field()
    
    def update(self):
        if Input.get_mouse_button_down(0):
            source = self.game_object.get_component(AudioSource)
            source.play_one_shot(self.fire_sound)
            Debug.log("Pew pew!")
```

## 3D Spatial Audio

Set `spatial_blend` to control how much the audio is affected by listener distance and direction:

- `0.0` = fully 2D (same volume everywhere — good for music/UI)
- `1.0` = fully 3D (gets quieter with distance — good for world sounds)

```python
class SpatialAudio(InfComponent):
    def start(self):
        source = self.game_object.get_component(AudioSource)
        source.spatial_blend = 1.0      # Full 3D
        source.min_distance = 1.0       # Full volume within 1 meter
        source.max_distance = 50.0      # Silent beyond 50 meters
        source.loop = True
        source.play()
```

Walk your AudioListener (camera) towards and away from the source. You'll hear the volume change. Magic? No, inverse-square law.

## Common Patterns

### Footstep System

```python
class Footsteps(InfComponent):
    step_interval: float = serialized_field(default=0.4)
    step_sound: AudioClip = serialized_field()
    _timer: float = 0
    
    def update(self):
        # Only play when moving
        h = Input.get_axis("Horizontal")
        v = Input.get_axis("Vertical")
        if abs(h) > 0.1 or abs(v) > 0.1:
            self._timer += Time.delta_time
            if self._timer >= self.step_interval:
                self._timer = 0
                source = self.game_object.get_component(AudioSource)
                source.play_one_shot(self.step_sound)
```

### Background Music Manager

```python
class BGMManager(InfComponent):
    """Survives scene changes, fades between tracks."""
    fade_speed: float = serialized_field(default=1.0)
    _target_volume: float = 1.0
    
    def start(self):
        source = self.game_object.get_component(AudioSource)
        source.loop = True
        source.spatial_blend = 0.0  # 2D — music everywhere
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

## Supported Formats

| Format | Recommended For |
|--------|----------------|
| WAV | Short sound effects (uncompressed, fast load) |
| OGG | Music and longer clips (compressed, smaller files) |
| MP3 | Also supported (but OGG is usually better) |

## See Also

- [AudioSource API](../api/AudioSource.md)
- [AudioListener API](../api/AudioListener.md)
- [AudioClip API](../api/AudioClip.md)
- [Physics Tutorial](physics.md) — trigger sounds on collision
