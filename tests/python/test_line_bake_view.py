"""Baked ribbons consume the same view plane as their realtime geometry."""
import numpy as np
import pytest

from infernux.lib import CameraProjection, LineAlignment, Vector3


@pytest.mark.parametrize('alignment', [LineAlignment.View, LineAlignment.TransformZ])
@pytest.mark.parametrize('projection', [CameraProjection.Perspective, CameraProjection.Orthographic])
@pytest.mark.parametrize('rotation', [(0.,0.,0.), (45.,0.,0.), (30.,40.,65.)])
@pytest.mark.parametrize('custom_view', [False, True])
def test_baked_ribbon_width_lies_in_the_rendered_plane(scene, alignment, projection, rotation, custom_view):
    line = scene.create_game_object('Ribbon').add_component('LineRenderer')
    line.set_positions([(-2.,0.,0.), (2.,0.,0.)])
    line.start_width = line.end_width = .8
    line.alignment = alignment
    camera = scene.create_game_object('BakeCamera').add_component('Camera')
    camera.transform.position = Vector3(0.,0.,-5.)
    camera.transform.euler_angles = Vector3(*rotation)
    camera.projection_mode = projection
    if custom_view:
        rendered_view = camera.view_matrix.copy()
        camera.transform.euler_angles = Vector3(80.,20.,30.)
        camera.transform.position = Vector3(5.,6.,7.)
        camera.view_matrix = rendered_view
    normal = -np.linalg.inv(camera.view_matrix)[:3,2] if alignment == LineAlignment.View else np.array([0.,0.,1.])
    target = scene.create_game_object('Snapshot').add_component('MeshRenderer')
    line.bake_mesh(target,camera=camera)
    positions = np.asarray(target._cpp_component.get_positions())
    sides = positions[1::2]-positions[::2]
    np.testing.assert_allclose(np.linalg.norm(sides,axis=1), .8, atol=1.e-5)
    np.testing.assert_allclose(sides @ normal, 0., atol=1.e-5)


@pytest.mark.parametrize('projection', [CameraProjection.Perspective, CameraProjection.Orthographic])
def test_translating_camera_without_rotating_does_not_twist_a_baked_ribbon(scene, projection):
    line = scene.create_game_object('Ribbon').add_component('LineRenderer')
    line.alignment = LineAlignment.View
    line.set_positions([(-2.,0.,0.),(0.,1.,.2),(2.,.3,0.)])
    line.start_width = line.end_width = .8
    camera = scene.create_game_object('BakeCamera').add_component('Camera')
    camera.projection_mode = projection
    camera.transform.euler_angles = Vector3(30.,15.,25.)
    target = scene.create_game_object('Snapshot').add_component('MeshRenderer')
    camera.transform.position = Vector3(0.,0.,-5.)
    line.bake_mesh(target,camera=camera)
    before=np.asarray(target._cpp_component.get_positions())
    camera.transform.position = Vector3(8.,9.,-2.)
    line.bake_mesh(target,camera=camera)
    np.testing.assert_allclose(target._cpp_component.get_positions(),before,atol=1.e-5)


@pytest.mark.parametrize('world_space', [False,True])
@pytest.mark.parametrize('lighting', [False,True])
def test_baked_normal_and_handedness_preserve_the_realtime_lighting_contract(scene,world_space,lighting):
    line=scene.create_game_object('LitRibbon').add_component('LineRenderer')
    line.alignment=LineAlignment.View
    line.use_world_space=world_space
    line.generate_lighting_data=lighting
    line.transform.euler_angles=Vector3(10.,20.,30.)
    line.transform.local_scale=Vector3(-2.,.6,1.3)
    line.set_positions([(-1.,0.,0.),(1.,0.,0.)])
    camera=scene.create_game_object('BakeCamera').add_component('Camera')
    camera.transform.euler_angles=Vector3(35.,40.,5.)
    target=scene.create_game_object('Snapshot').add_component('MeshRenderer')
    line.bake_mesh(target,camera=camera)
    # Compare in world coordinates, the space consumed by material lighting.
    linear=np.eye(3) if world_space else np.array(line.transform.local_to_world_matrix()).reshape(4,4,order='F')[:3,:3]
    normals=np.asarray(target._cpp_component.get_normals()) @ np.linalg.inv(linear)
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    expected=-np.linalg.inv(camera.view_matrix)[:3,2] if lighting else np.linalg.inv(linear).T @ [0.,0.,1.]
    expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(normals,np.tile(expected,(len(normals),1)),atol=2.e-5)
    tangents=np.asarray(target._cpp_component.get_tangents())
    np.testing.assert_allclose(tangents[:,3]*np.sign(np.linalg.det(linear)),1.)
