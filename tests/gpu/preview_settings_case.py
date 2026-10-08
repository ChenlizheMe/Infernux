"""Public image-preview settings survive load boundaries and reach the GUI."""
from pathlib import Path

import numpy as np
from PIL import Image

from infernux.lib import InxGUIRenderable


class PreviewSettingsCase:
    phases = ('before_load', 'switch_image', 'same_image', 'unload_reload',
              'switch_previewer', 'after_load')
    panel_id = 'repair.preview_settings'

    def __init__(self, native, directory):
        self.native = native
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.paths = [self.directory / 'First.png', self.directory / 'Second.png']
        for path in self.paths:
            Image.new('RGBA', (8, 4), (80, 120, 160, 255)).save(path)
        self.text = self.directory / 'Note.txt'
        self.text.write_text('Preview type switch', encoding='utf-8')
        self.manager = native.get_resource_preview_manager()
        self.manager.unload_preview()
        self.errors = []
        self.phase = 0
        self.change(0)
        owner = self

        class Renderable(InxGUIRenderable):
            def on_render(self, ctx):
                try:
                    x, y, w, h = ctx.get_main_viewport_bounds()
                    ctx.set_next_window_pos(x, y, 0, 0., 0.)
                    ctx.set_next_window_size(w, h, 0)
                    flags = (1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<5)|(1<<8)|(1<<16)|(1<<17)|(1<<19)
                    ctx.push_style_var_vec2(2, 0., 0.)
                    ctx.push_style_var_float(4, 0.)
                    try:
                        visible = ctx.begin_window('##PreviewSettingsProof', True, flags)
                        try:
                            if visible:
                                owner.manager.render_preview(ctx, w, h)
                        finally:
                            ctx.end_window()
                    finally:
                        ctx.pop_style_var(2)
                except BaseException as error:
                    owner.errors.append(error)

        self.renderable = Renderable()
        native.register_gui_renderable(self.panel_id, self.renderable)

    def change(self, phase):
        assert 0 <= phase < len(self.phases)
        self.phase = phase
        manager = self.manager
        current = self.paths[0] if phase == 0 else self.paths[1]
        pending_bytes = self.native.pending_imgui_texture_upload_bytes
        if phase == 0:
            manager.set_preview_settings(0, 2, False)
        elif phase == 3:
            manager.unload_preview()
        elif phase == 4:
            assert manager.load_preview(str(self.text))
        elif phase == 5:
            manager.unload_preview()
            manager.set_preview_settings(0, 0, True)
        assert manager.load_preview(str(current))
        if phase == 5:
            manager.set_preview_settings(0, 2, False)
        self.submitted_bytes = self.native.pending_imgui_texture_upload_bytes - pending_bytes
        if phase != 5:
            # A new 2x1 RGBA preview contains 8 bytes; reopening the same
            # preview must not schedule a redundant upload.
            assert self.submitted_bytes == (0 if phase == 2 else 8), (phase, self.submitted_bytes)
        assert manager.get_loaded_path() == str(current)
        self.key = str(current) + '::prev_0_2_0'
        self.default_key = str(current) + '::prev_0_0_1'
        self.ready_frames = 0

    def poll(self):
        if self.errors:
            raise self.errors[0]
        self.native.request_full_speed_frame()
        self.ready_frames = self.ready_frames + 1 if self.native.get_imgui_texture_id(self.key) else 0
        if self.ready_frames < 3:
            return dict(done=False, phase=self.phases[self.phase])
        assert not self.native.get_imgui_texture_id(self.default_key)
        return dict(done=True, phase=self.phases[self.phase], texture=self.native.get_imgui_texture_id(self.key),
                    version=self.native.get_imgui_texture_version(self.key), submitted_bytes=self.submitted_bytes)

    def observe_capture(self, path):
        pixels = np.asarray(Image.open(path).convert('RGB'))
        h, w = pixels.shape[:2]
        sample = pixels[2*h//5:3*h//5, 2*w//5:3*w//5].astype(float)
        linear = np.array([80, 120, 160]) / 255.
        expected = np.floor((1.055 * linear ** (1/2.4) - .055) * 255)
        difference = np.abs(sample - expected)
        assert (difference <= 2).all(axis=-1).mean() > .98, (sample.mean(axis=(0,1)), expected)
        return dict(phase=self.phases[self.phase], mean=sample.mean(axis=(0,1)).tolist(), expected=expected.tolist())

    def close(self):
        self.manager.unload_preview()
        self.manager.set_preview_settings(0, 0, True)
        self.native.unregister_gui_renderable(self.panel_id)
        for path in self.paths:
            for suffix in ('::prev_0_2_0', '::prev_0_0_1'):
                assert not self.native.get_imgui_texture_id(str(path) + suffix)
            path.unlink()
        self.text.unlink()
        self.directory.rmdir()
