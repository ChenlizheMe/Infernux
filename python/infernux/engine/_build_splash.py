"""BuildSplashMixin — extracted from GameBuilder."""
from __future__ import annotations



import os
import shutil
import struct



class BuildSplashMixin:
    """BuildSplashMixin method group for GameBuilder."""

    def _process_splash_items(self, final_dir: str):
        """Copy/convert splash items into Data/Splash/."""
        if not self.splash_items:
            return

        splash_dir = os.path.join(final_dir, "Data", "Splash")
        os.makedirs(splash_dir, exist_ok=True)
        staged_by_guid: dict[str, str] = {}
        if self._built_icon_path and self.icon_guid:
            staged_by_guid[self.icon_guid] = self._built_icon_path

        for item in self.splash_items:
            asset_guid = item["asset_guid"]
            src_path = self._asset_source_for_guid(asset_guid, "Splash item")
            item_type = item.get("type", "image")
            if item_type not in {"image", "video"}:
                raise ValueError(f"Unsupported splash item type: {item_type}")
            item["_layout"] = (
                "logo"
                if item_type == "image" and asset_guid == self.icon_guid
                else "contain" if item_type == "image" else "cover"
            )
            existing = staged_by_guid.get(asset_guid)
            if existing:
                item["_built_path"] = existing
                continue
            base_name = os.path.splitext(os.path.basename(src_path))[0]

            if item_type == "video":
                out_name = base_name + ".infsplash"
                out_path = os.path.join(splash_dir, out_name)
                if os.path.exists(out_path):
                    raise RuntimeError(f"Splash destination collision: {out_path}")
                self._extract_video_frames(src_path, out_path)
                item["_built_path"] = f"Splash/{out_name}"
            else:
                ext = os.path.splitext(src_path)[1]
                out_name = base_name + ext
                out_path = os.path.join(splash_dir, out_name)
                if os.path.exists(out_path):
                    raise RuntimeError(f"Splash destination collision: {out_path}")
                shutil.copy2(src_path, out_path)
                item["_built_path"] = f"Splash/{out_name}"
            staged_by_guid[asset_guid] = item["_built_path"]

    def _extract_video_frames(self, video_path: str, output_path: str):
        """Extract video frames to .infsplash binary blob."""
        from infernux.engine.game_builder import _ensure_video_splash_packages
        _ensure_video_splash_packages()
        self._extract_with_imageio(video_path, output_path)

    def _extract_with_imageio(self, video_path: str, output_path: str):
        """Extract video frames using imageio+av."""
        import imageio.v3 as iio

        frames_data: list[bytes] = []
        width = height = 0
        for frame in iio.imiter(video_path, plugin="pyav"):
            height, width = frame.shape[:2]
            jpeg_bytes = iio.imwrite(
                "<bytes>", frame, extension=".jpg", quality=85
            )
            frames_data.append(jpeg_bytes)

        meta = iio.immeta(video_path, plugin="pyav")
        fps = meta.get("fps", 30.0) or 30.0
        self._write_infsplash(output_path, frames_data, fps, width, height)

    @staticmethod
    def _write_infsplash(
        path: str, frames: list, fps: float, width: int, height: int
    ):
        """Write .infsplash binary (magic + header + index + JPEG data)."""
        with open(path, "wb") as f:
            f.write(b"INFSPLSH")
            f.write(struct.pack("<IfII", len(frames), fps, width, height))
            offset = 0
            for data in frames:
                f.write(struct.pack("<II", offset, len(data)))
                offset += len(data)
            for data in frames:
                f.write(data)

