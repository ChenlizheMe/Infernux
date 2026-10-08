"""Video playback owns asynchronous frames and bounded texture lifetimes."""
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from infernux.engine import splash_player as module
from infernux.engine._build_splash import BuildSplashMixin


def write_video(path, size=(16,8), color='red', count=3, fps=1.):
    output = io.BytesIO()
    Image.new('RGB',size,color).save(output,format='JPEG')
    frames = [output.getvalue()] * count
    BuildSplashMixin._write_infsplash(str(path),frames,fps,*size)
    return frames


def item(path):
    return dict(type='video',path=path,layout='cover',duration=3.,fade_in=0.,fade_out=0.)


class DrawRecorder:
    def __init__(self):
        self.draws = []

    def draw_image_rect(self,*arguments):
        self.draws.append(arguments)


class ControlledQueue:
    """Control only task completion timing; Splash and binary IO remain real."""
    def __init__(self,immediate=False):
        self.immediate = immediate
        self.pending, self.ready, self.payloads = {}, {}, {}
        self.scheduled, self.released = [], []
        self.next_texture = 100

    def request_full_speed_frame(self):
        pass

    def pump_preview_tasks(self):
        pass

    def schedule_texture_preview_from_memory(self,key,data,stamp,nearest):
        token = (key,bytes(data),stamp)
        self.scheduled.append(token)
        if key not in self.ready:
            self.pending[key] = token
            if self.immediate:
                self.complete(token)
        return True

    def complete(self,token):
        key,data,_ = token
        if self.pending.get(key) is not token:
            return
        self.pending.pop(key)
        self.next_texture += 1
        self.ready[key] = self.next_texture
        self.payloads[self.next_texture] = data

    def get_texture_preview_texture_id(self,key):
        return self.ready.get(key,0)

    def invalidate_texture_preview_task(self,key):
        # Existing preview invalidation deliberately retains the displayed texture.
        pass

    def release_texture_preview_task(self,key):
        self.pending.pop(key,None)
        texture = self.ready.pop(key,None)
        if texture is not None:
            self.payloads.pop(texture)
        self.released.append(key)


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(value=10.)
    monkeypatch.setattr(module,'_time',SimpleNamespace(monotonic=lambda:state.value))
    return state


@pytest.mark.parametrize('immediate',[False,True])
def test_video_adopts_current_frame_without_resubmitting(tmp_path,clock,immediate):
    write_video(tmp_path/'Intro.infsplash')
    player = module.SplashPlayer([item('Intro.infsplash')],str(tmp_path))
    queue,draw = ControlledQueue(immediate),DrawRecorder()
    try:
        player.update(draw,queue,0,0,100,100)
        token = queue.scheduled[0]
        queue.complete(token)
        for _ in range(4):
            clock.value += .1
            player.update(draw,queue,0,0,100,100)
        assert draw.draws and draw.draws[-1][0] == queue.ready[token[0]]
        assert len(queue.scheduled) == 1
    finally:
        player.cleanup(queue)


@pytest.mark.parametrize('same_clip',[False,True])
def test_consecutive_videos_own_their_frame_resources(tmp_path,clock,same_clip):
    first = write_video(tmp_path/'First.infsplash',count=2)
    second = first if same_clip else write_video(tmp_path/'Second.infsplash',size=(8,16),color='blue',count=2)
    player = module.SplashPlayer([item('First.infsplash'),item('First.infsplash' if same_clip else 'Second.infsplash')],str(tmp_path))
    queue,draw = ControlledQueue(True),DrawRecorder()
    try:
        for when in (10.,11.1):
            clock.value = when
            player.update(draw,queue,0,0,100,100)
        old_textures = set(queue.ready.values())
        clock.value = 12.1
        player.update(draw,queue,0,0,100,100)
        clock.value = 12.2
        player.update(draw,queue,0,0,100,100)
        assert player._idx == 1
        assert draw.draws[-1][0] not in old_textures
        assert queue.payloads[draw.draws[-1][0]] == second[0]
        assert not old_textures.intersection(queue.payloads)
    finally:
        player.cleanup(queue)
    assert not queue.pending and not queue.ready


@pytest.mark.parametrize('delay', [1.1, 2.1])
def test_slow_decode_publishes_then_skips_to_current_frame(tmp_path,clock,delay):
    write_video(tmp_path/'Intro.infsplash', count=5)
    player = module.SplashPlayer([item('Intro.infsplash')],str(tmp_path))
    queue,draw = ControlledQueue(),DrawRecorder()
    try:
        player.update(draw,queue,0,0,100,100)
        old = queue.scheduled[-1]
        clock.value = 10. + delay
        player.update(draw,queue,0,0,100,100)
        assert queue.scheduled == [old], 'Do not cancel every frame and starve playback'
        queue.complete(old)
        player.update(draw,queue,0,0,100,100)
        assert draw.draws[-1][0] == queue.ready[old[0]]
        latest = queue.scheduled[-1]
        assert latest[2] == int(delay) + 1, 'Skip missed frames and decode only the current frame'
        queue.complete(latest)
        player.update(draw,queue,0,0,100,100)
        assert draw.draws[-1][0] == queue.ready[latest[0]]
        assert old[0] not in queue.ready
    finally:
        player.cleanup(queue)


def test_cleanup_retires_a_video_that_has_not_uploaded(tmp_path,clock):
    write_video(tmp_path/'Intro.infsplash')
    player = module.SplashPlayer([item('Intro.infsplash')],str(tmp_path))
    queue = ControlledQueue()
    player.update(DrawRecorder(),queue,0,0,100,100)
    token = queue.scheduled[-1]
    player.cleanup(queue)
    queue.complete(token)
    assert not queue.pending and not queue.ready
    assert player._vfile is None


def test_long_video_keeps_texture_residency_bounded(tmp_path,clock):
    write_video(tmp_path/'Long.infsplash',count=100,fps=10.)
    player = module.SplashPlayer([item('Long.infsplash')],str(tmp_path))
    queue,draw = ControlledQueue(True),DrawRecorder()
    try:
        for frame in range(100):
            clock.value = 10. if frame == 0 else 10. + frame/10. + .025
            player.update(draw,queue,0,0,100,100)
            assert len(queue.ready) <= 2
        assert len(queue.scheduled) == 100
    finally:
        player.cleanup(queue)
    assert not queue.ready
