# Physics — Making Things Fall (And Crash Into Each Other)

<div class="class-info">
Tutorial &nbsp;|&nbsp; <a href="../../zh/tutorials/physics.html">中文</a>
</div>

## Overview

InfEngine uses **Jolt Physics** under the hood — an engine so good that Horizon Forbidden West shipped with it. You get rigidbodies, five kinds of colliders, raycasting, triggers, and enough control to simulate a game of billiards or a city-scale demolition derby.

## Your First Physics Object

Every physics object needs two things: a **Collider** (shape) and a **Rigidbody** (physics behavior). Think of it as giving your object a body and teaching it about gravity.

### Step 1: Add a Collider

In the editor, select a GameObject and add a collider component:

```python
from InfEngine import *

class PhysicsSetup(InfComponent):
    def start(self):
        # Add physics to this object
        box = self.game_object.add_component("BoxCollider")
        rb = self.game_object.add_component("Rigidbody")
        
        Debug.log("Physics is now in charge. Good luck.")
```

Or just add them from the Inspector panel — click **Add Component → Physics → BoxCollider** and **Rigidbody**.

### Step 2: Watch It Fall

Press Play. Your object falls. That's gravity at work. Congratulations, Newton would be proud.

## Collider Types

| Collider | Best For | Performance |
|----------|----------|-------------|
| `BoxCollider` | Crates, walls, floors, buildings | ⚡ Fast |
| `SphereCollider` | Balls, projectiles, trigger zones | ⚡ Fastest |
| `CapsuleCollider` | Characters, humanoid shapes | ⚡ Fast |
| `MeshCollider` | Complex static geometry | 🐢 Expensive |

> **Pro tip:** Use `MeshCollider` only for static objects. For moving things, combine primitive colliders. Your frame rate will thank you.

### Configuring Colliders

```python
class ColliderDemo(InfComponent):
    def start(self):
        # Box collider with custom size
        box = self.game_object.get_component(BoxCollider)
        box.size = vector3(2, 1, 3)
        box.center = vector3(0, 0.5, 0)
        
        # Sphere collider
        sphere = self.game_object.get_component(SphereCollider)
        sphere.radius = 1.5
        
        # Capsule for a character
        cap = self.game_object.get_component(CapsuleCollider)
        cap.radius = 0.5
        cap.height = 2.0
```

## Rigidbody

The `Rigidbody` component makes an object obey physics. Without it, colliders are just invisible walls.

### Key Properties

```python
class RigidbodyDemo(InfComponent):
    def start(self):
        rb = self.game_object.get_component(Rigidbody)
        
        rb.mass = 10.0              # Heavy boi (kg)
        rb.drag = 0.5               # Air resistance
        rb.angular_drag = 0.05      # Rotational resistance
        rb.use_gravity = True       # Falls down (usually what you want)
        rb.is_kinematic = False     # Obeys physics forces
```

### Applying Forces

```python
class Rocket(InfComponent):
    thrust: float = serialized_field(default=100.0)
    
    def fixed_update(self):
        rb = self.game_object.get_component(Rigidbody)
        
        # Continuous thrust — use ForceMode.Force
        if Input.get_key(KeyCode.SPACE):
            rb.add_force(vector3(0, self.thrust, 0))
        
        # One-time impulse — like a jump
        if Input.get_key_down(KeyCode.W):
            rb.add_force(vector3(0, 500, 0), ForceMode.Impulse)
        
        # Force at a point — causes rotation too!
        if Input.get_key_down(KeyCode.E):
            hit_point = self.transform.position + vector3(1, 0, 0)
            rb.add_force_at_position(vector3(0, 200, 0), hit_point)
```

> **Important:** Always apply forces in `fixed_update()`, not `update()`. Physics runs at a fixed timestep — mixing the two is how you get Objects That Vibrate Ominously™.

### Kinematic Bodies

Kinematic rigidbodies don't respond to forces but can push dynamic objects around. Perfect for moving platforms, doors, or that one elevator puzzle every game has.

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

## Triggers

Set `is_trigger = True` on a collider to make it detect overlap without physical collision. Great for pickups, zone detection, and invisible death walls.

```python
class PickupZone(InfComponent):
    def start(self):
        collider = self.game_object.get_component(BoxCollider)
        collider.is_trigger = True
    
    def on_trigger_enter(self, other):
        Debug.log(f"{other.game_object.name} entered the zone!")
    
    def on_trigger_exit(self, other):
        Debug.log(f"{other.game_object.name} left. We miss them already.")
```

## Raycasting

Shoot an invisible ray into the scene and see what it hits. Used for line-of-sight checks, shooting mechanics, mouse picking, and ground detection.

```python
class RaycastDemo(InfComponent):
    def update(self):
        origin = self.transform.position
        direction = self.transform.forward
        
        # Single raycast
        hit = Physics.raycast(origin, direction, max_distance=100.0)
        if hit:
            Debug.log(f"Hit {hit.collider.game_object.name} at {hit.point}")
            Debug.draw_line(origin, hit.point, vector3(1, 0, 0))
```

## Freezing Axes

Sometimes you don't want physics to rotate your object (looking at you, 2D games). Use constraints:

```python
class FreezeRotation(InfComponent):
    def start(self):
        rb = self.game_object.get_component(Rigidbody)
        # Freeze all rotation — object won't tip over
        rb.constraints = (
            RigidbodyConstraints.FreezeRotationX |
            RigidbodyConstraints.FreezeRotationY |
            RigidbodyConstraints.FreezeRotationZ
        )
```

## Common Patterns

### Ground Check

```python
class GroundCheck(InfComponent):
    is_grounded: bool = False
    
    def fixed_update(self):
        origin = self.transform.position
        self.is_grounded = Physics.raycast(
            origin, vector3(0, -1, 0), max_distance=1.1
        ) is not None
```

### Simple Character Controller

```python
class SimpleCharacter(InfComponent):
    speed: float = serialized_field(default=5.0)
    jump_force: float = serialized_field(default=8.0)
    
    def fixed_update(self):
        rb = self.game_object.get_component(Rigidbody)
        
        # Movement
        h = Input.get_axis("Horizontal")
        v = Input.get_axis("Vertical")
        move = vector3(h, 0, v) * self.speed
        rb.velocity = vector3(move.x, rb.velocity.y, move.z)
        
        # Jump
        if Input.get_key_down(KeyCode.SPACE):
            hit = Physics.raycast(
                self.transform.position, vector3(0, -1, 0), 1.1
            )
            if hit:
                rb.add_force(vector3(0, self.jump_force, 0), ForceMode.Impulse)
```

## See Also

- [Rigidbody API](../api/Rigidbody.md)
- [BoxCollider API](../api/BoxCollider.md)
- [Physics API](../api/Physics.md)
- [Coroutines](coroutines.md) — for delayed physics actions
