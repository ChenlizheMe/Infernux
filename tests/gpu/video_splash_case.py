"""Production SplashPlayer draws real async decoded textures through native GUI."""
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from infernux.engine import splash_player as module
from infernux.engine._build_splash import BuildSplashMixin
from infernux.lib import InxGUIRenderable


class VideoSplashCase:
    phases = 6
    panel_id = 'repair.video_splash'

    def __init__(self,native,directory):
        self.native = native
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        for name,size,color in [('First',(16,8),'red'),('Second',(8,16),'blue')]:
            output = io.BytesIO()
            Image.new('RGB',size,color).save(output,format='JPEG')
            BuildSplashMixin._write_infsplash(str(self.directory/(name+'.infsplash')),
                                            [output.getvalue()]*2,1.,*size)
        self.player = module.SplashPlayer([dict(type='video',path=name+'.infsplash',layout='cover',
            duration=2.,fade_in=0.,fade_out=0.) for name in ('First','Second','First')],str(self.directory))
        self.clock, self.phase, self.frames = 10., 0, 0
        self.original_clock = module._time
        module._time = SimpleNamespace(monotonic=lambda:self.clock)
        self.errors = []
        owner = self

        class Renderable(InxGUIRenderable):
            def on_render(self,ctx):
                try:
                    x,y,w,h = ctx.get_main_viewport_bounds()
                    ctx.set_next_window_pos(x,y,0,0.,0.)
                    ctx.set_next_window_size(w,h,0)
                    flags = (1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<5)|(1<<8)|(1<<16)|(1<<17)|(1<<19)
                    ctx.push_style_var_vec2(2,0.,0.)
                    ctx.push_style_var_float(4,0.)
                    try:
                        visible = ctx.begin_window('##VideoSplashProof',True,flags)
                        try:
                            if visible:
                                owner.player.update(ctx,native,x,y,w,h)
                                owner.frames += 1
                        finally:
                            ctx.end_window()
                    finally:
                        ctx.pop_style_var(2)
                except BaseException as error:
                    owner.errors.append(error)

        self.renderable = Renderable()
        native.register_gui_renderable(self.panel_id,self.renderable)

    def change(self,phase):
        assert 0 <= phase < self.phases
        self.phase = phase
        self.clock = 10. if phase == 0 else 10. + phase + .1

    def poll(self):
        if self.errors:
            raise self.errors[0]
        player = self.player
        if (player._idx != self.phase//2 or player._vlast_frame != self.phase%2
                or not player._tex_id or player._vpending_key):
            return dict(done=False)
        size = tuple(self.native.get_texture_preview_size(player._vdisplay_key))
        expected = (8,16) if self.phase//2 == 1 else (16,8)
        assert size == expected, (self.phase,size,expected)
        assert self.native.get_texture_preview_texture_id(player._vdisplay_key) == player._tex_id
        records = [r for r in self.native.preview_task_snapshots if r['resource_key'] in player._vkeys]
        assert len(records) <= 2 and sum(r['texture_id'] != 0 for r in records) <= 2
        return dict(done=True,phase=self.phase,size=size,texture=player._tex_id,frames=self.frames,
                    snapshots=[{k:v for k,v in r.items() if k != 'pixel_hash'} for r in records])

    def observe_capture(self,path):
        pixels = np.asarray(Image.open(path).convert('RGB'))
        h,w = pixels.shape[:2]
        center = pixels[h//4:3*h//4,w//4:3*w//4].astype(np.int16)
        channel = 2 if self.phase//2 == 1 else 0
        expected = center[...,channel] > 180
        others = np.delete(center,channel,axis=-1).max(axis=-1) < 70
        assert (expected & others).mean() > .98, (self.phase,center.mean(axis=(0,1)))
        return dict(phase=self.phase,color='blue' if channel == 2 else 'red',
                    mean=center.mean(axis=(0,1)).tolist(),pixels=int(expected.size))

    def close(self):
        self.player.cleanup(self.native)
        self.native.unregister_gui_renderable(self.panel_id)
        module._time = self.original_clock
        for key in self.player._vkeys:
            assert self.native.get_texture_preview_texture_id(key) == 0
            assert tuple(self.native.get_texture_preview_size(key)) == (0,0)
