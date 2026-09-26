"""Run only in the managed Blender process; do not import from the engine."""
import json
import sys
from pathlib import Path


def main():
    import bpy

    if bpy.app.version[:2] != (5, 2):
        raise RuntimeError(f"Infernux model import requires Blender 5.2, got {bpy.app.version_string}")
    source, destination, report_path = sys.argv[sys.argv.index("--") + 1:]
    # Never execute embedded project scripts or replace the importer's UI.
    bpy.ops.wm.open_mainfile(filepath=source, load_ui=False, use_scripts=False)
    # glTF has no core object-visibility property. Carry the authored Blender
    # render/view visibility through the disposable conversion payload as one
    # importer-owned extra; the source .blend is never saved or modified.
    for obj in bpy.data.objects:
        obj["infernux_source_visible"] = not obj.hide_viewport and not obj.hide_render
    diagnostics = []
    simulation_modifiers = {"CLOTH", "FLUID", "SOFT_BODY", "DYNAMIC_PAINT", "PARTICLE_SYSTEM"}

    # Cameras and lights are valid Blender scene authoring objects, but they
    # are outside the current composite-model payload.  Report that boundary
    # instead of silently dropping them.  Keeping this as source inventory
    # (rather than an "unsupported" error) leaves one clear extension point
    # when scene-object import is introduced later.
    for obj in bpy.data.objects:
        if obj.type in {"CAMERA", "LIGHT"}:
            diagnostics.append({
                "code": "ignored_scene_object",
                "owner": obj.name,
                "property": "object/type",
                "detail": obj.type,
            })

    # Object constraints are authoring controls, not Player components.  Bake
    # their evaluated frame into the disposable copy before exporting, then
    # remove the Blender-only controls.  This preserves the visible authored
    # pose without carrying a second constraint runtime into Infernux.
    dependency_graph = bpy.context.evaluated_depsgraph_get()
    constrained = [obj for obj in bpy.data.objects if obj.constraints]
    constrained_world = {
        obj.name: obj.evaluated_get(dependency_graph).matrix_world.copy()
        for obj in constrained
    }
    for obj in constrained:
        constraint_names = sorted(constraint.name for constraint in obj.constraints)
        for constraint in list(obj.constraints):
            obj.constraints.remove(constraint)
        obj.matrix_world = constrained_world[obj.name]
        diagnostics.append({
            "code": "baked_object_constraints",
            "owner": obj.name,
            "property": "object/constraints",
            "detail": ",".join(constraint_names),
        })

    # Curves and text are source authoring geometry.  Freeze their evaluated
    # surface into the same Mesh pipeline as FBX/glTF rather than teaching the
    # Player Blender curve semantics.
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.data.objects):
        if obj.type not in {"CURVE", "FONT"} or obj.name not in bpy.context.view_layer.objects:
            continue
        source_type = obj.type
        bpy.ops.object.select_all(action="DESELECT")
        obj.hide_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        result = bpy.ops.object.convert(target="MESH")
        if result != {"FINISHED"}:
            raise RuntimeError(f"Could not bake {source_type} object '{obj.name}' to a mesh")
        diagnostics.append({
            "code": "baked_authoring_geometry",
            "owner": obj.name,
            "property": "object/type",
            "detail": source_type,
        })

    for obj in bpy.data.objects:
        animation = obj.animation_data
        if animation and animation.drivers:
            diagnostics.append({
                "code": "unsupported_driver",
                "owner": obj.name,
                "property": "object/driver",
                "detail": ",".join(sorted({curve.data_path for curve in animation.drivers})),
            })
        for modifier in obj.modifiers:
            if modifier.type in simulation_modifiers:
                diagnostics.append({
                    "code": "unsupported_simulation",
                    "owner": obj.name,
                    "property": "modifier/" + modifier.name,
                    "detail": modifier.type,
                })
            elif obj.type == "MESH" and obj.data and obj.data.shape_keys and modifier.type != "ARMATURE":
                diagnostics.append({
                    "code": "unsupported_modifier_with_shape_keys",
                    "owner": obj.name,
                    "property": "modifier/" + modifier.name,
                    "detail": modifier.type,
                })
    for material in bpy.data.materials:
        nodes = material.node_tree
        animation = nodes.animation_data if nodes else None
        if animation and animation.drivers:
            diagnostics.append({
                "code": "unsupported_driver",
                "owner": material.name,
                "property": "material/driver",
                "detail": ",".join(sorted({curve.data_path for curve in animation.drivers})),
            })
    # Mutate only Blender's disposable in-memory copy. Ordinary geometry
    # modifiers are applied per object, so a morph mesh does not disable
    # baking on unrelated meshes in the same source. Armature modifiers remain
    # live for the glTF skin exporter and simulations are reported, not sampled.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in list(bpy.data.objects):
        if obj.type != "MESH" or obj.data.shape_keys or obj.name not in bpy.context.view_layer.objects:
            continue
        candidates = [(modifier.name, modifier.type) for modifier in obj.modifiers
                      if modifier.show_render and modifier.type != "ARMATURE" and
                      modifier.type not in simulation_modifiers]
        if not candidates:
            continue
        obj.hide_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        for modifier_name, modifier_type in candidates:
            result = bpy.ops.object.modifier_apply(modifier=modifier_name)
            if result != {"FINISHED"}:
                raise RuntimeError(f"Could not bake modifier '{modifier_name}' on '{obj.name}'")
            diagnostics.append({
                "code": "baked_geometry_modifier",
                "owner": obj.name,
                "property": "modifier/" + modifier_name,
                "detail": modifier_type,
            })
        obj.select_set(False)

    diagnostics.sort(key=lambda item: (item["code"], item["owner"], item["property"], item["detail"]))
    external_image_paths = {}
    for image in bpy.data.images:
        if image.source != "FILE" or image.packed_file is not None or not image.filepath:
            continue
        path = bpy.path.abspath(image.filepath, library=image.library)
        # Blender's GLB exporter names an external image from its file stem,
        # not from the editable Blender image datablock name.  The native
        # importer joins this report to the exported GLB by image name.
        export_name = Path(path).stem
        previous_path = external_image_paths.get(export_name)
        if previous_path is not None and previous_path != path:
            raise RuntimeError(
                f"External Blender images export as the same GLB image '{export_name}': "
                f"'{previous_path}' and '{path}'"
            )
        external_image_paths[export_name] = path
    external_images = [
        {"name": name, "path": path}
        for name, path in sorted(external_image_paths.items())
    ]
    with open(report_path, "w", encoding="utf-8") as report:
        json.dump({"diagnostics": diagnostics, "external_images": external_images}, report,
                  ensure_ascii=False, separators=(",", ":"))
    result = bpy.ops.export_scene.gltf(
        filepath=destination,
        export_format="GLB",
        # Blender authors in a right-handed Z-up space; glTF is the single
        # right-handed Y-up, metre-based interchange boundary.  Do not add an
        # importer-side pre-rotation or pre-scale: Scale Factor and optional
        # source-root baking belong to the common Infernux model importer.
        export_yup=True,
        # Blender is an artwork source for model geometry.  Scene cameras and
        # lights belong to the Infernux scene and are intentionally ignored.
        export_cameras=False,
        export_lights=False,
        export_animations=True,
        export_skins=True,
        export_morph=True,
        # Preserve object/Empty/Armature/collection-instance transforms.  The
        # common importer owns the one-time hierarchy/geometry conversion.
        export_apply=False,
        export_gpu_instances=True,
        export_extras=True,
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"Blender glTF export did not finish: {result}")


if __name__ == "__main__":
    main()
