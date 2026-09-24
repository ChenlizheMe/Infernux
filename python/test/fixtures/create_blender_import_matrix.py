"""Author the real Blender fixture used by the model-import acceptance test."""
import sys

import bpy


def link_object(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def add_uvs(mesh):
    layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
            layer.data[loop_index].uv = (vertex.co.x, vertex.co.y)


def main():
    output, external_texture = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)

    image = bpy.data.images.new("External Albedo", width=4, height=4)
    image.pixels = [0.05, 0.25, 0.9, 0.55] * 16
    image.filepath_raw = external_texture
    image.file_format = "PNG"
    image.save()

    material = bpy.data.materials.new("TransparentExternal")
    material.use_nodes = True
    material.diffuse_color = (0.05, 0.25, 0.9, 0.55)
    material.surface_render_method = "DITHERED"
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Alpha"].default_value = 0.55
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(texture.outputs["Alpha"], bsdf.inputs["Alpha"])

    embedded_image = bpy.data.images.new("Embedded Accent", width=4, height=4)
    embedded_image.pixels = [0.9, 0.15, 0.05, 1.0] * 16
    embedded_material = bpy.data.materials.new("OpaqueEmbedded")
    embedded_material.use_nodes = True
    embedded_material.diffuse_color = (0.9, 0.15, 0.05, 1.0)
    embedded_nodes = embedded_material.node_tree.nodes
    embedded_links = embedded_material.node_tree.links
    embedded_bsdf = embedded_nodes.get("Principled BSDF")
    embedded_texture = embedded_nodes.new("ShaderNodeTexImage")
    embedded_texture.image = embedded_image
    embedded_links.new(embedded_texture.outputs["Color"], embedded_bsdf.inputs["Base Color"])

    root = link_object(bpy.data.objects.new("Assembly", None))
    root.rotation_euler = (0.15, -0.25, 0.4)

    # A deliberately asymmetric coordinate probe.  Blender authors in a
    # right-handed Z-up space; the managed glTF handoff is right-handed Y-up.
    # Keeping this object outside the rotated Assembly makes the expected
    # engine coordinates exact and catches accidental double axis/scale
    # conversion in the real .blend path.
    axis_mesh = bpy.data.meshes.new("AxisContractMesh")
    axis_mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 0, 2)], [], [(0, 1, 2)])
    add_uvs(axis_mesh)
    axis_probe = link_object(bpy.data.objects.new("AxisContract", axis_mesh))
    axis_probe.location = (3.0, 4.0, 5.0)

    hard_mesh = bpy.data.meshes.new("HardEdgeMesh")
    hard_mesh.from_pydata(
        [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)],
        [],
        [(0, 1, 2), (0, 3, 1)],
    )
    add_uvs(hard_mesh)
    hard_mesh.materials.append(material)
    hard_mesh.materials.append(embedded_material)
    for polygon in hard_mesh.polygons:
        polygon.use_smooth = False
    hard_mesh.polygons[1].material_index = 1
    hard = link_object(bpy.data.objects.new("MirroredHardEdge", hard_mesh))
    hard.parent = root
    hard.location = (1.0, 0.5, -0.25)
    hard.scale = (-1.0, 1.5, 0.75)
    bevel = hard.modifiers.new("BakedBevel", "BEVEL")
    bevel.width = 0.12
    bevel.segments = 2
    geometry_nodes = bpy.data.node_groups.new("BakedGeometryNodes", "GeometryNodeTree")
    geometry_nodes.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    geometry_nodes.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    group_input = geometry_nodes.nodes.new("NodeGroupInput")
    group_output = geometry_nodes.nodes.new("NodeGroupOutput")
    geometry_nodes.links.new(group_input.outputs["Geometry"], group_output.inputs["Geometry"])
    geometry_modifier = hard.modifiers.new("BakedGeometryNodes", "NODES")
    geometry_modifier.node_group = geometry_nodes
    hard["driver_probe"] = 1.0
    driver = hard.driver_add('["driver_probe"]')
    driver.driver.expression = "1.0"

    morph_mesh = bpy.data.meshes.new("MorphMesh")
    morph_mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    add_uvs(morph_mesh)
    morph = link_object(bpy.data.objects.new("MorphWithModifier", morph_mesh))
    morph.parent = root
    morph.location = (3.0, 0.0, 0.0)
    morph.shape_key_add(name="Basis")
    inflate = morph.shape_key_add(name="Inflate")
    inflate.data[2].co.z = 0.5
    morph_bevel = morph.modifiers.new("UnbakedMorphBevel", "BEVEL")
    morph_bevel.width = 0.1

    source_collection = bpy.data.collections.new("ReusableCollection")
    source_mesh = bpy.data.meshes.new("ReusableTriangleMesh")
    source_mesh.from_pydata([(0, 0, 0), (0.75, 0, 0), (0, 0.75, 0)], [], [(0, 1, 2)])
    add_uvs(source_mesh)
    source_mesh.materials.append(material)
    reusable = link_object(bpy.data.objects.new("ReusableTriangle", source_mesh), source_collection)
    reusable.location = (0.5, 0.0, 0.0)
    instance = link_object(bpy.data.objects.new("CollectionInstance", None))
    instance.parent = root
    instance.instance_type = "COLLECTION"
    instance.instance_collection = source_collection
    instance.location = (-2.0, 0.0, 0.0)

    constraint_target = link_object(bpy.data.objects.new("ConstraintTarget", None))
    constraint_target.parent = root
    constraint_target.location = (0.75, -1.25, 0.5)
    constrained_mesh = bpy.data.meshes.new("ConstrainedTriangleMesh")
    constrained_mesh.from_pydata([(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0)], [], [(0, 1, 2)])
    add_uvs(constrained_mesh)
    constrained = link_object(bpy.data.objects.new("ConstrainedTriangle", constrained_mesh))
    constrained.parent = root
    copy_location = constrained.constraints.new("COPY_LOCATION")
    copy_location.name = "BakedCopyLocation"
    copy_location.target = constraint_target

    curve_data = bpy.data.curves.new("BakedCurveData", "CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.06
    curve_data.bevel_resolution = 1
    spline = curve_data.splines.new("POLY")
    spline.points.add(2)
    spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
    spline.points[1].co = (0.5, 0.25, 0.5, 1.0)
    spline.points[2].co = (1.0, 0.0, 0.75, 1.0)
    curve = link_object(bpy.data.objects.new("BakedCurve", curve_data))
    curve.parent = root
    curve.location = (-1.0, 1.5, 0.0)

    simulation_mesh = bpy.data.meshes.new("SimulationSourceMesh")
    simulation_mesh.from_pydata(
        [(-0.5, 0, 0), (0.5, 0, 0), (0.5, 1, 0), (-0.5, 1, 0)],
        [],
        [(0, 1, 2), (0, 2, 3)],
    )
    add_uvs(simulation_mesh)
    simulation = link_object(bpy.data.objects.new("UnsupportedCloth", simulation_mesh))
    simulation.parent = root
    simulation.location = (2.0, 1.5, 0.0)
    simulation.modifiers.new("AuthoringCloth", "CLOTH")

    armature_data = bpy.data.armatures.new("RigData")
    armature = link_object(bpy.data.objects.new("Rig", armature_data))
    armature.parent = root
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bone = armature_data.edit_bones.new("RootBone")
    bone.head = (0.0, 0.0, 0.0)
    bone.tail = (0.0, 1.0, 0.0)
    tip_bone = armature_data.edit_bones.new("TipBone")
    tip_bone.parent = bone
    tip_bone.use_connect = True
    tip_bone.head = bone.tail
    tip_bone.tail = (0.0, 2.0, 0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.select_set(False)

    skin_mesh = bpy.data.meshes.new("SkinnedQuadMesh")
    skin_mesh.from_pydata(
        [(-0.5, 0, 0), (0.5, 0, 0), (0.5, 1, 0), (-0.5, 1, 0)],
        [],
        [(0, 1, 2), (0, 2, 3)],
    )
    add_uvs(skin_mesh)
    skin_mesh.materials.append(material)
    skin = link_object(bpy.data.objects.new("SkinnedQuad", skin_mesh))
    skin.parent = armature
    skin.location = (0.0, 0.0, 2.0)
    root_group = skin.vertex_groups.new(name="RootBone")
    root_group.add([0, 1], 1.0, "REPLACE")
    tip_group = skin.vertex_groups.new(name="TipBone")
    tip_group.add([2, 3], 1.0, "REPLACE")
    modifier = skin.modifiers.new("Armature", "ARMATURE")
    modifier.object = armature
    pose_bone = armature.pose.bones["RootBone"]
    pose_bone.rotation_mode = "XYZ"
    pose_bone.keyframe_insert(data_path="rotation_euler", frame=1)
    pose_bone.rotation_euler.z = 0.5
    pose_bone.keyframe_insert(data_path="rotation_euler", frame=20)
    pose_tip = armature.pose.bones["TipBone"]
    pose_tip.rotation_mode = "XYZ"
    pose_tip.keyframe_insert(data_path="rotation_euler", frame=1)
    pose_tip.rotation_euler.x = -0.35
    pose_tip.keyframe_insert(data_path="rotation_euler", frame=20)
    bpy.context.scene.frame_end = 20

    camera_data = bpy.data.cameras.new("IgnoredCameraData")
    camera = link_object(bpy.data.objects.new("IgnoredCamera", camera_data))
    camera.parent = root
    light_data = bpy.data.lights.new("IgnoredLightData", "POINT")
    light = link_object(bpy.data.objects.new("IgnoredLight", light_data))
    light.parent = root

    bpy.ops.wm.save_as_mainfile(filepath=output)


if __name__ == "__main__":
    main()
