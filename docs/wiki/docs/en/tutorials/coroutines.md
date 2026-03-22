# Coroutines & Time — Doing Things Later (Without Losing Your Mind)

<div class="class-info">
Tutorial &nbsp;|&nbsp; <a href="../../zh/tutorials/coroutines.html">中文</a>
</div>

## Overview

Sometimes you need to wait. Wait 2 seconds before respawning. Wait until the player is close enough. Wait for the explosion animation to finish. You *could* use timers and state machines, but coroutines let you write this kind of sequential logic as straightforward Python generators. The code reads like a script instead of a flowchart.

## The Time Class

Before we get to coroutines, meet `Time` — your gateway to frame timing.

```python
from InfEngine import *

class TimeDemo(InfComponent):
    def update(self):
        # Time since game started (affected by time_scale)
        Debug.log(f"Time: {Time.time:.2f}")
        
        # How long the last frame took
        Debug.log(f"Delta: {Time.delta_time:.4f}")
        
        # Slow motion!
        if Input.get_key_down(KeyCode.T):
            Time.time_scale = 0.25  # Quarter speed
        
        # Back to normal
        if Input.get_key_down(KeyCode.Y):
            Time.time_scale = 1.0
```

### Key Properties

| Property | Description |
|----------|-------------|
| `Time.time` | Seconds since start (scaled) |
| `Time.delta_time` | Last frame duration (scaled). **Use this for movement!** |
| `Time.unscaled_time` | Real seconds (ignores time_scale) |
| `Time.unscaled_delta_time` | Real frame duration |
| `Time.fixed_delta_time` | Physics timestep (default 0.02 = 50 Hz) |
| `Time.time_scale` | Speed multiplier. 0 = paused, 1 = normal, 2 = double speed |
| `Time.frame_count` | Total frames rendered |

> **Golden rule:** Multiply any speed by `Time.delta_time`. Otherwise your game runs twice as fast at 120 FPS and half as fast at 30 FPS. Nobody wants that.

## Your First Coroutine

A coroutine is a Python generator that `yield`s special wait objects:

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
        Debug.log("Go! 🚀")
```

That's it. No state machine. No timer variables. Just `yield` and wait.

## Wait Types

| Class | Waits For | Respects time_scale? |
|-------|-----------|---------------------|
| `WaitForSeconds(n)` | `n` seconds of game time | ✅ Yes |
| `WaitForSecondsRealtime(n)` | `n` real seconds | ❌ No |
| `WaitForEndOfFrame()` | End of current frame | — |
| `WaitForFixedUpdate()` | Next physics step | — |
| `WaitUntil(predicate)` | Until `predicate()` returns `True` | — |
| `WaitWhile(predicate)` | While `predicate()` returns `True` | — |

### Examples

```python
class WaitExamples(InfComponent):
    def start(self):
        self.start_coroutine(self.various_waits())
    
    def various_waits(self):
        # Wait 2 game seconds (pauses if time_scale = 0)
        yield WaitForSeconds(2)
        
        # Wait 2 real seconds (even when paused — good for pause menus)
        yield WaitForSecondsRealtime(2)
        
        # Wait until the player presses Space
        yield WaitUntil(lambda: Input.get_key_down(KeyCode.SPACE))
        
        # Wait as long as the player holds Shift
        yield WaitWhile(lambda: Input.get_key(KeyCode.LEFT_SHIFT))
        
        # Wait one physics step
        yield WaitForFixedUpdate()
        
        # Wait until end of frame (good for screenshot capture)
        yield WaitForEndOfFrame()
```

## Managing Coroutines

### Starting

```python
# Start and keep a handle
self.my_routine = self.start_coroutine(self.do_stuff())
```

### Stopping

```python
# Stop a specific coroutine
self.stop_coroutine(self.my_routine)

# Stop ALL coroutines on this component
self.stop_all_coroutines()
```

### Chaining (Coroutine from Coroutine)

```python
def main_sequence(self):
    Debug.log("Phase 1: Warm up")
    yield from self.warm_up()          # yield from another generator
    Debug.log("Phase 2: Battle")
    yield from self.battle_phase()
    Debug.log("Phase 3: Victory")

def warm_up(self):
    yield WaitForSeconds(3)

def battle_phase(self):
    yield WaitUntil(lambda: self.enemies_dead())
```

## Common Patterns

### Delayed Action

```python
class Grenade(InfComponent):
    fuse_time: float = serialized_field(default=3.0)
    
    def start(self):
        self.start_coroutine(self.explode_after_delay())
    
    def explode_after_delay(self):
        yield WaitForSeconds(self.fuse_time)
        Debug.log("BOOM! 💥")
        self.destroy(self.game_object)
```

### Repeating Action

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
        Debug.log("Spawning enemy...")
```

### Smooth Value Transition

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

### Typewriter Text Effect

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

## The Mathf Utility Class

Coroutines often need math helpers. `Mathf` has you covered:

```python
# Clamp a value
hp = Mathf.clamp(hp - damage, 0, max_hp)

# Smooth interpolation
value = Mathf.lerp(start, end, t)
value = Mathf.smooth_step(0, 1, t)

# Move toward a target at fixed speed
pos = Mathf.move_towards(current, target, speed * Time.delta_time)

# Trigonometry
y = Mathf.sin(Time.time * frequency) * amplitude
```

## See Also

- [Time API](../api/Time.md)
- [Mathf API](../api/Mathf.md)
- [Coroutine API](../api/Coroutine.md)
- [WaitForSeconds API](../api/WaitForSeconds.md)
- [UI Tutorial](ui.md) — coroutine-driven UI animations
