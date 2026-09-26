# Camera

<div class="class-info">
class in <b>Infernux.components.builtin</b>
</div>

**Inherits from:** [BuiltinComponent](Component.md)

## Description

A Camera component that renders a view of the scene.

<!-- USER CONTENT START --> description
**Status:** Preview · **Verified with:** 0.4.0

Use a perspective Camera for normal 3D depth and an orthographic Camera for scale-stable 2D framing. Keep near/far clipping proportional to scene scale.
<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| target_texture | `RenderTexture | RenderTextureRef | None` | Output owner, unresolved asset reference, or None for the screen. |
| projection_matrix | `NDArray[np.float32]` |  |
| view_matrix | `NDArray[np.float32]` |  |
| camera_to_world_matrix | `NDArray[np.float32]` |  *(read-only)* |
| has_custom_view_matrix | `bool` |  *(read-only)* |
| invert_culling | `bool` |  |
| has_custom_projection_matrix | `bool` |  *(read-only)* |
| projection_mode | `int` | The projection mode (Perspective or Orthographic). |
| field_of_view | `float` | The vertical field of view in degrees. |
| use_physical_properties | `bool` | Use physical lens and sensor properties for perspective projection. |
| iso | `int` | Physical camera sensor sensitivity. |
| shutter_speed | `float` | Physical camera exposure time in seconds. |
| aperture | `float` | Physical camera aperture in f-stops. |
| focus_distance | `float` | Physical camera focus-plane distance. |
| blade_count | `int` | Physical camera diaphragm blade count. |
| curvature | `Vector2` | Minimum and maximum aperture used for diaphragm curvature. |
| barrel_clipping | `float` | Optical-vignetting cat-eye strength. |
| anamorphism | `float` | Physical camera sensor stretch. |
| focal_length | `float` | Physical camera focal length in millimetres. |
| sensor_type | `CameraSensorType` | Derived sensor-size preset; Custom preserves the authored sensor size. |
| sensor_size | `Any` | Physical camera sensor size in millimetres (width, height). |
| lens_shift | `Any` | Physical camera lens shift in normalized sensor units. |
| gate_fit | `Any` |  |
| orthographic_size | `float` | Half-size of the camera in orthographic mode. |
| aspect_ratio | `float` | The aspect ratio of the camera (width / height). *(read-only)* |
| near_clip | `float` | The near clipping plane distance. |
| far_clip | `float` | The far clipping plane distance. |
| depth | `float` | The rendering order of the camera. |
| culling_mask | `int` | The layer mask used for culling objects. |
| clear_flags | `int` | How the camera clears the background before rendering. |
| background_color | `List[float]` | The background color used when clear flags is set to solid color. |
| stop_nans | `bool` | Whether invalid final-output pixels are replaced before display encoding. |
| dithering | `bool` | Whether display-space dithering is applied before output quantization. |
| pixel_width | `int` | The width of the camera's render target in pixels. *(read-only)* |
| pixel_height | `int` | The height of the camera's render target in pixels. *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `set_clip_planes(near_clip: float, far_clip: float) → None` |  |
| `reset_view_matrix() → None` |  |
| `reset_history() → None` | Discard this camera's accumulated history before its next render. |
| `reset_projection_matrix() → None` |  |
| `calculate_oblique_matrix(clip_plane: ArrayLike) → NDArray[np.float32]` |  |
| `screen_to_world_point(x: float, y: float, depth: float = ...) → Optional[Tuple[float, float, float]]` | Convert a screen-space point to world coordinates. |
| `world_to_screen_point(x: float, y: float, z: float) → Optional[Tuple[float, float]]` | Convert a world-space point to screen coordinates. |
| `screen_point_to_ray(x: float, y: float, viewport_width: Optional[float] = ..., viewport_height: Optional[float] = ...) → Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]` | Cast a ray from a screen-space point into the scene. |
| `serialize() → str` | Serialize the component to a JSON string. |
| `deserialize(json_str: str) → bool` | Deserialize the component from a JSON string. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Lifecycle Methods

| Method | Description |
|------|------|
| `on_draw_gizmos_selected() → None` | Draw the camera frustum gizmo when selected in the editor. |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example
```python
import infernux as inx

camera_object = inx.GameObject.find("Main Camera")
if camera_object is not None:
    camera = camera_object.get_component(inx.Camera)
    if camera is not None:
        camera.projection_mode = 1  # Orthographic
        camera.orthographic_size = 5.0
```
<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also
- [CameraProjection](CameraProjection.md)
<!-- USER CONTENT END -->
