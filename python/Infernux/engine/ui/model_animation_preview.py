"""Editor-only transport for published model clips, independent of scene playback."""
from dataclasses import dataclass
import math
import time


@dataclass
class AnimationPreviewTransport:
    clip_id: str = ""
    seconds: float = 0.0
    speed: float = 1.0
    frame_rate: int = 30
    playing: bool = False
    last_time: float | None = None

    def select(self, clip_id: str) -> None:
        if clip_id != self.clip_id:
            self.clip_id = clip_id
            self.seconds = 0.0
            self.playing = False
            self.last_time = None

    def advance(self, now: float, duration: float) -> None:
        if self.playing and self.last_time is not None and duration > 0.0:
            self.seconds = (self.seconds + max(0.0, now - self.last_time) * self.speed) % duration
        self.seconds = min(max(self.seconds, 0.0), max(duration, 0.0))
        if duration <= 0.0:
            self.playing = False
        self.last_time = now

    def seek(self, seconds: float, duration: float) -> None:
        self.seconds = min(max(seconds, 0.0), max(duration, 0.0))
        self.playing = False
        self.last_time = None

    def frame(self, duration: float) -> tuple[int, int]:
        # This is an editor timeline grid, not the source format's ticks/second.
        last = math.ceil(duration * self.frame_rate)
        current = last if self.seconds >= duration else math.floor(self.seconds * self.frame_rate + 1e-7)
        return min(last, current), last

    def seek_frame(self, frame: int, duration: float) -> None:
        self.seek(frame / self.frame_rate, duration)


def render_animation_transport(ctx, transport, duration):
    """Only authored edits seek: float32 widget round trips are not user input."""
    from Infernux.engine.i18n import t
    transport.advance(time.perf_counter(), duration)

    label = t("animclip_editor.pause" if transport.playing else "animclip_editor.play")
    if ctx.button(label + "###model_preview_play"):
        transport.playing = not transport.playing and duration > 0.0
    ctx.record_semantic_item("button", label, duration > 0.0, "asset.mesh.preview.play",
                             bool_value=transport.playing)
    ctx.same_line()
    transport.speed = ctx.drag_float(t("asset.animation_preview_speed") + "##model_preview_speed",
                                     transport.speed, .05, .05, 4.0)
    ctx.record_semantic_item("drag_float", t("asset.animation_preview_speed"), True, "asset.mesh.preview.speed",
                             numeric_value=transport.speed)
    seconds = ctx.drag_float(t("asset.animation_preview_time") + "##model_preview_time",
                             transport.seconds, max(duration / 300.0, .001), 0.0, max(duration, .001))
    edited = ctx.is_item_edited()
    if edited:
        transport.seek(seconds, duration)
    ctx.record_semantic_item("drag_float", t("asset.animation_preview_time"), True, "asset.mesh.preview.time",
                             numeric_value=transport.seconds)
    ctx.label(f"{transport.seconds:.3f} / {duration:.3f} s")
    transport.frame_rate = ctx.drag_int(t("asset.animation_preview_fps") + "##model_preview_fps",
                                        transport.frame_rate, 1.0, 1, 240)
    ctx.record_semantic_item("drag_int", t("asset.animation_preview_fps"), True, "asset.mesh.preview.fps",
                             numeric_value=transport.frame_rate)
    frame, last_frame = transport.frame(duration)
    frame = ctx.drag_int(t("asset.animation_preview_frame") + "##model_preview_frame",
                          frame, 1.0, 0, max(last_frame, 1))
    if ctx.is_item_edited():
        transport.seek_frame(frame, duration)
    ctx.record_semantic_item("drag_int", t("asset.animation_preview_frame"), True, "asset.mesh.preview.frame",
                             numeric_value=transport.frame(duration)[0])


def render_model_animation_preview(ctx, panel, state, clips, *, source_path=None):
    """Read published clips only; controls never modify import drafts or the scene."""
    from Infernux.core.assets import AssetManager
    from Infernux.engine.i18n import t
    from Infernux.lib import AssetRegistry
    from .asset_resource_preview import _resolve_native_engine

    if not clips:
        ctx.text_wrapped(t("asset.animation_preview_empty"))
        return
    source_path = source_path or state.file_path
    mesh = AssetRegistry.instance().load_mesh(source_path)
    if mesh is None or mesh.vertex_count == 0:
        ctx.text_wrapped(t("asset.animation_preview_needs_mesh"))
        return
    transport = state.extra.setdefault("model_animation_transport", AnimationPreviewTransport())
    ids = [clip["id"] for clip in clips]
    transport.select(transport.clip_id if transport.clip_id in ids else ids[0])
    index = ctx.combo(t("asset.animation_preview_clip") + "##model_preview_clip",
                      ids.index(transport.clip_id), [clip["name"] for clip in clips])
    ctx.record_semantic_item("combo", t("asset.animation_preview_clip"), True, "asset.mesh.preview.clip")
    transport.select(ids[index])
    render_animation_transport(ctx, transport, float(clips[index]["duration"]))

    native = _resolve_native_engine(panel)
    width = max(32.0, ctx.get_content_region_avail_width() - 8.0)
    size = int(min(width, 320.0))
    texture = native.render_model_animation_preview(
        mesh, transport.clip_id, transport.seconds, size,
        AssetManager.preview_dependency_signature(source_path))
    if texture:
        ctx.set_cursor_pos_x(ctx.get_cursor_pos_x() + (width - size) * .5)
        ctx.image(texture, float(size), float(size))
        ctx.record_semantic_item("image", t("asset.animation_preview_clip"), True, "asset.mesh.preview.image")
    else:
        ctx.dummy(float(size), float(size))
    ctx.text_wrapped(t("asset.animation_preview_published"))
    ctx.separator()


def render_clip_animation_preview(ctx, panel, state, model_path, take):
    """A clip asset previews its published source, never the virtual child path."""
    import json
    from Infernux.core.asset_types import read_asset_metadata
    from Infernux.core.assets import AssetManager
    from Infernux.engine.i18n import t

    signature = (model_path, AssetManager.preview_dependency_signature(model_path)) if model_path else ("", 0)
    cached = state.extra.get("clip_preview_source")
    if cached is None or cached[0] != signature:
        metadata = read_asset_metadata(model_path) if model_path else None
        cached = (signature, json.loads((metadata or {}).get("model_animations", "[]")))
        state.extra["clip_preview_source"] = cached
    clips = cached[1]
    # Existing standalone clips may name a take; imported children use stable IDs.
    clip = next((item for item in clips if item["id"] == take), None)
    if clip is None:
        clip = next((item for item in clips if item["name"] == take), None)
    if clip is None:
        ctx.text_wrapped(t("asset.animation_preview_missing_take"))
        return
    render_model_animation_preview(ctx, panel, state, [clip], source_path=model_path)
