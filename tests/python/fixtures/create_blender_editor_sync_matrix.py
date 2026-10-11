"""Author the real Blender source used by Editor model-instance sync acceptance.

Run with Blender::

    blender --background --python create_blender_editor_sync_matrix.py -- OUTPUT.blend initial
    blender --background --python create_blender_editor_sync_matrix.py -- OUTPUT.blend changed

The two revisions deliberately keep the geometry of ``RenameMe`` identical so
the importer can retain its stable subresource identity while the DCC path is
renamed and moved from ``AuthoringPivot`` to ``AuthoringRoot``.  ``DeleteMe``
disappears and ``AddedFromBlender`` appears in the changed revision.
Cameras/lights remain in the source file only to prove that the current model
contract excludes them without closing a future extension.
"""
from __future__ import annotations

import sys

import bpy


def _link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _mesh_object(name: str, vertices, faces, parent, location):
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = _link(bpy.data.objects.new(name, mesh))
    obj.parent = parent
    obj.location = location
    return obj


def main() -> None:
    output, revision = sys.argv[sys.argv.index("--") + 1 :]
    if revision not in {"initial", "changed"}:
        raise ValueError("revision must be 'initial' or 'changed'")

    bpy.ops.wm.read_factory_settings(use_empty=True)

    material = bpy.data.materials.new("StableAuthoringMaterial")
    material.use_nodes = True
    material.diffuse_color = (0.12, 0.48, 0.82, 1.0)

    root = _link(bpy.data.objects.new("AuthoringRoot", None))
    root.location = (1.0, 2.0, 3.0)
    pivot = _link(bpy.data.objects.new("AuthoringPivot", None))
    pivot.parent = root
    pivot.location = (-0.5, 0.25, 1.5)

    stable_name = "RenameMe" if revision == "initial" else "RenamedStable"
    stable_parent = pivot if revision == "initial" else root
    stable = _mesh_object(
        stable_name,
        [(0, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(0, 1, 2), (0, 3, 1)],
        stable_parent,
        (2.0, 0.5, -1.0),
    )
    stable.data.materials.append(material)
    stable.scale = (-1.0, 1.5, 0.75)
    bevel = stable.modifiers.new("BakedBevel", "BEVEL")
    bevel.width = 0.08
    bevel.segments = 2
    stable["unsupported_driver_probe"] = 1.0
    driver = stable.driver_add('["unsupported_driver_probe"]')
    driver.driver.expression = "1.0"

    _mesh_object(
        "AlwaysHere",
        [(0, 0, 0), (1.25, 0, 0), (1.25, 1, 0), (0, 1, 0)],
        [(0, 1, 2), (0, 2, 3)],
        root,
        (-2.0, 0.0, 0.5),
    )

    if revision == "initial":
        _mesh_object(
            "DeleteMe",
            [(0, 0, 0), (0.75, 0, 0), (0, 0.5, 0)],
            [(0, 1, 2)],
            root,
            (0.0, -1.5, 0.0),
        )
    else:
        _mesh_object(
            "AddedFromBlender",
            [(0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0), (0.25, 0.25, 1)],
            [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
            root,
            (0.0, 1.5, 0.0),
        )

    camera_data = bpy.data.cameras.new("FutureCameraData")
    camera = _link(bpy.data.objects.new("FutureCamera", camera_data))
    camera.parent = root
    light_data = bpy.data.lights.new("FutureLightData", "POINT")
    light = _link(bpy.data.objects.new("FutureLight", light_data))
    light.parent = root

    bpy.ops.wm.save_as_mainfile(filepath=output)


if __name__ == "__main__":
    main()
