# Component Icons (20×20 px)

Place 20×20 PNG icons here. The filename convention is
`component_<type_name_lowercase>.png`, where `<type_name_lowercase>` is the
component's `type_name` in lowercase.

## Built-in component icons

| Filename                        | Component       |
|---------------------------------|-----------------|
| `component_transform.png`      | Transform       |
| `component_camera.png`         | Camera          |
| `component_light.png`          | Light           |
| `component_meshrenderer.png`   | MeshRenderer    |
| `component_script.png`         | Generic script fallback |

## Fallback

If a Python script component has no component-specific icon, the inspector
automatically falls back to `component_script.png`.

If no icon file is found at all, the Inspector header will render without
an icon (text only).

## Python script components

Python components can also have per-class icons. For a component with
`type_name = "PlayerController"`, create `component_playercontroller.png`.
If that file doesn't exist, `component_script.png` is used as the fallback.
