"""
SplashPlayer — renders a sequence of splash images / videos at game startup.

Supports two item types:

* **image** — loaded via the engine's TextureLoader (stb_image), rendered
  fullscreen with configurable fade-in / hold / fade-out timing.
* **video** — pre-extracted at build time into ``.infsplash`` binary blobs
  (JPEG frame sequences).  Frames are decoded at runtime via
  ``TextureLoader.load_from_memory()`` and rendered at the original FPS.

Auto-scales each item to fill the viewport.
"""

from __future__ import annotations

import os
import struct
import time as _time
from typing import Dict, List, Optional
from Infernux.debug import Debug


class SplashPlayer:
    """Plays a list of splash items sequentially, then reports completion."""

    _TEX_NAME = "_inf_splash"

    def __init__(self, splash_items: List[Dict], data_root: str):
        self._items = splash_items or []
        self._data_root = data_root
        self._idx = 0
        self._start: Optional[float] = None
        self._finished = len(self._items) == 0

        # GPU texture currently displayed
        self._tex_id: int = 0
        self._img_w: int = 0
        self._img_h: int = 0

        # Video state
        self._vfile = None
        self._vindex: list = []  # [(offset, size), ...]
        self._vfps: float = 30.0
        self._vdata_offset: int = 0
        self._vlast_frame: int = -1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_finished(self) -> bool:
        return self._finished

    def update(self, ctx, native_engine, x0: float, y0: float,
               vp_w: float, vp_h: float):
        """Call once per frame to advance & render the splash sequence."""
        if self._finished:
            return

        now = _time.time()

        if self._start is None:
            self._start = now
            self._load_item(native_engine)

        item = self._items[self._idx]
        elapsed = now - self._start
        duration = self._item_duration(item)
        alpha = self._alpha(elapsed, duration,
                            item.get("fade_in", 0.5),
                            item.get("fade_out", 0.5))

        # Video: advance frame
        if item.get("type") == "video" and self._vindex:
            target = min(int(elapsed * self._vfps), len(self._vindex) - 1)
            if target != self._vlast_frame:
                self._vlast_frame = target
                self._upload_video_frame(native_engine, target)

        # Render (scale-to-fill, preserving aspect ratio)
        if self._tex_id:
            rx, ry, rw, rh = self._cover_rect(
                vp_w, vp_h, self._img_w, self._img_h)
            ctx.draw_image_rect(
                self._tex_id,
                x0 + rx, y0 + ry, x0 + rx + rw, y0 + ry + rh,
                0.0, 0.0, 1.0, 1.0,        # uv
                1.0, 1.0, 1.0, alpha,       # tint RGBA
            )

        # Advance?
        if elapsed >= duration:
            self._unload_item(native_engine)
            self._idx += 1
            if self._idx >= len(self._items):
                self._finished = True
            else:
                self._start = now
                self._load_item(native_engine)

    def cleanup(self, native_engine):
        """Release any held resources."""
        self._unload_item(native_engine)

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    def _item_duration(self, item: Dict) -> float:
        if item.get("type") == "video" and self._vindex:
            return len(self._vindex) / max(self._vfps, 1.0)
        return item.get("duration", 3.0)

    @staticmethod
    def _cover_rect(vp_w: float, vp_h: float,
                    img_w: int, img_h: int):
        """Return (x, y, w, h) that scales the image to *cover* the viewport
        while preserving aspect ratio (like CSS background-size: cover)."""
        if img_w <= 0 or img_h <= 0 or vp_w <= 0 or vp_h <= 0:
            return (0.0, 0.0, vp_w, vp_h)
        scale = max(vp_w / img_w, vp_h / img_h)
        w = img_w * scale
        h = img_h * scale
        x = (vp_w - w) * 0.5
        y = (vp_h - h) * 0.5
        return (x, y, w, h)

    @staticmethod
    def _alpha(elapsed: float, duration: float,
               fade_in: float, fade_out: float) -> float:
        if elapsed < fade_in:
            return elapsed / fade_in if fade_in > 0 else 1.0
        if elapsed > duration - fade_out:
            remaining = max(0.0, duration - elapsed)
            return remaining / fade_out if fade_out > 0 else 0.0
        return 1.0

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def _load_item(self, native_engine):
        item = self._items[self._idx]
        path = os.path.join(self._data_root, item["path"])
        if item.get("type") == "video":
            self._load_video(native_engine, path)
        else:
            self._load_image(native_engine, path)

    def _load_image(self, native_engine, path: str):
        from Infernux.lib import TextureLoader
        td = TextureLoader.load_from_file(path)
        if not td.is_valid():
            return
        self._img_w = td.width
        self._img_h = td.height
        self._tex_id = native_engine.upload_texture_for_imgui(
            self._TEX_NAME, td.get_pixels_list(), td.width, td.height,
        )

    def _load_video(self, native_engine, path: str):
        try:
            f = open(path, "rb")
        except OSError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return
        magic = f.read(8)
        if magic != b"INFSPLSH":
            f.close()
            return

        hdr = f.read(16)
        count, self._vfps, _w, _h = struct.unpack("<IfII", hdr)
        self._img_w = _w
        self._img_h = _h

        idx_data = f.read(count * 8)
        self._vindex = [
            struct.unpack_from("<II", idx_data, i * 8)
            for i in range(count)
        ]
        self._vdata_offset = f.tell()
        self._vfile = f
        self._vlast_frame = -1

        # Upload first frame immediately
        if self._vindex:
            self._upload_video_frame(native_engine, 0)
            self._vlast_frame = 0

    def _upload_video_frame(self, native_engine, frame_idx: int):
        if not self._vfile or frame_idx >= len(self._vindex):
            return
        offset, size = self._vindex[frame_idx]
        self._vfile.seek(self._vdata_offset + offset)
        jpeg = self._vfile.read(size)

        from Infernux.lib import TextureLoader
        td = TextureLoader.load_from_memory(jpeg)
        if not td.is_valid():
            return

        # Replace previous texture
        if self._tex_id:
            native_engine.remove_imgui_texture(self._TEX_NAME)
        self._tex_id = native_engine.upload_texture_for_imgui(
            self._TEX_NAME, td.get_pixels_list(), td.width, td.height,
        )

    def _unload_item(self, native_engine):
        if self._tex_id:
            try:
                native_engine.remove_imgui_texture(self._TEX_NAME)
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass
            self._tex_id = 0
        if self._vfile:
            self._vfile.close()
            self._vfile = None
        self._vindex = []
        self._vlast_frame = -1
