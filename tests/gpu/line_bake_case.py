"""Compare actual realtime ribbon pixels with the public BakeMesh result."""
import numpy as np

from infernux.lib import CameraClearFlags, CameraProjection, LineAlignment, SceneManager, Vector3, vec4f
from infernux.renderstack import RenderStackPipeline


class LineBakeCase:
    scenarios = ('perspective','orthographic','custom','near_parallel','local','mirrored','transform_z')

    def __init__(self,native,scene,scenario):
        assert scenario in self.scenarios
        self.native,self.scene,self.scenario=native,scene,scenario
        self.previous_camera=scene.main_camera
        manager=SceneManager.instance()
        self.roots=[(obj,obj.active_self) for i in range(manager.scene_count) for obj in manager.get_scene_at(i).get_root_objects()]
        for obj,_ in self.roots:
            obj.active=False
        self.camera_object=scene.create_game_object('LineBakeCamera')
        self.camera=self.camera_object.add_component('Camera')
        self.camera.transform.euler_angles=Vector3(35.,20.,25.)
        forward=np.array(tuple(self.camera.transform.forward))
        self.camera.transform.position=Vector3(*(-6.*forward))
        self.camera.set_clip_planes(.1,20.)
        if scenario=='orthographic':
            self.camera.projection_mode=CameraProjection.Orthographic
            self.camera.orthographic_size=2.5
        if scenario=='custom':
            view=self.camera.view_matrix.copy()
            self.camera.transform.position=Vector3(25.,30.,40.)
            self.camera.transform.euler_angles=Vector3(5.,70.,30.)
            self.camera.view_matrix=view
        self.camera.clear_flags=CameraClearFlags.SolidColor
        self.camera.background_color=vec4f(0.,0.,0.,1.)
        scene.main_camera=self.camera
        self.line_object=scene.create_game_object('RealtimeRibbon')
        self.line=self.line_object.add_component('LineRenderer')
        self.line.alignment=LineAlignment.TransformZ if scenario=='transform_z' else LineAlignment.View
        self.line.start_color=self.line.end_color=[1.,0.,0.,1.]
        self.line.start_width=self.line.end_width=.5
        points=[(-1.5,0.,0.),(0.,.6,.2),(1.5,0.,0.)]
        if scenario=='near_parallel':
            world=np.linalg.inv(self.camera.view_matrix)
            direction=-world[:3,2]+.1*world[:3,1]
            direction/=np.linalg.norm(direction)
            points=[tuple(-1.5*direction),tuple(1.5*direction)]
        self.line.set_positions(points)
        self.baked_object=scene.create_game_object('BakedRibbon')
        self.baked=self.baked_object.add_component('MeshRenderer')
        if scenario in ('local','mirrored'):
            self.line.use_world_space=False
            rotation=Vector3(10.,30.,15.)
            scale=Vector3(-1.3 if scenario=='mirrored' else 1.3,.7,1.2)
            for obj in (self.line_object,self.baked_object):
                obj.transform.euler_angles=rotation
                obj.transform.local_scale=scale
        self.line.bake_mesh(self.baked,camera=self.camera)
        self.material=self.line.get_effective_material()
        self.baked.material=self.material
        self.baked_object.active=False
        self.pipeline=RenderStackPipeline()
        self.realtime=None

    def observe(self,pixels):
        pixels=np.asarray(pixels).copy()
        if self.realtime is None:
            self.realtime=pixels
            self.line_object.active=False
            self.baked_object.active=True
            return dict(done=False)
        def red(image):
            return (image[...,0]>.25)&(image[...,0]>3*image[...,1])
        original,current=red(self.realtime),red(pixels)
        union=np.count_nonzero(original|current)
        difference=np.count_nonzero(original^current)
        error=float(np.abs(self.realtime[...,:3].astype(float)-pixels[...,:3]).mean())
        result=dict(done=True,scenario=self.scenario,realtime_pixels=int(original.sum()),baked_pixels=int(current.sum()),
                    differing_pixels=int(difference),mean_error=error)
        assert original.sum()>80 and current.sum()>80,result
        assert difference<=max(4,union*.01) and error<.003,result
        return result

    def close(self):
        self.scene.main_camera=self.previous_camera
        for obj in (self.line_object,self.baked_object,self.camera_object):
            self.scene.destroy_game_object(obj)
        self.scene.process_pending_destroys()
        for obj,active in self.roots:
            obj.active=active
        self.pipeline.dispose()
