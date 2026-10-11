"""Shared GUID-based content cook for platform-owned Player hosts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from infernux.engine.build import BuildConfiguration, BuildRequest
from infernux.engine.build_settings import load_build_settings_for_build
from infernux.engine.game_builder import GameBuilder
from infernux.engine.player_build_preflight import (
    publish_player_asset_catalog_for_host,
)
from infernux.engine.path_utils import resolved_path


@dataclass(frozen=True, slots=True)
class PlatformContentCookResult:
    game_name: str
    data_directory: Path
    settings: Mapping[str, object]
    python_sources: tuple[Path, ...] = ()


def read_cooked_player_icon(
    data_directory: str | Path,
    *,
    default_icon: str | Path,
) -> bytes:
    """Read the icon already sealed by the shared Player content cook.

    Platform exporters consume the cooked ``BuildManifest`` instead of
    re-reading editor settings.  An empty project icon selects the explicit
    engine default supplied by the host distribution.  A configured icon is
    read from the validated ``Content.inxpkg`` produced by the cook; it is not
    looked up again through its authoring path.
    """

    data_root = Path(resolved_path(data_directory))
    try:
        from infernux.engine.platform_player_bootstrap import (
            read_player_build_manifest,
        )

        manifest = read_player_build_manifest(data_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"Player branding manifest is unreadable in sealed catalog: {data_root}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("Player branding manifest must be a JSON object")

    relative = str(manifest.get("icon_path", "") or "").strip()
    if relative:
        normalized = Path(relative.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"Player icon path escapes the cooked data root: {relative}")
        from infernux.engine.player_package_native import read_entry

        archive = data_root / "Content.inxpkg"
        if not archive.is_file():
            raise ValueError(f"Cooked Player content package is missing: {archive}")
        try:
            payload = read_entry(archive, normalized.as_posix())
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"Cooked Player icon is missing: {relative}") from error
        if not payload:
            raise ValueError(f"Cooked Player icon is empty: {relative}")
        return payload

    icon = Path(resolved_path(default_icon))
    if not icon.is_file():
        raise ValueError(f"Default Player icon is missing: {icon}")
    payload = icon.read_bytes()
    if not payload:
        raise ValueError(f"Default Player icon is empty: {icon}")
    return payload


def build_settings_for_request(request: BuildRequest) -> dict[str, object]:
    from infernux.engine.interaction.project_settings import (
        normalize_build_settings,
    )

    if "build_settings" in request.profile.options:
        configured = request.profile.options["build_settings"]
        if not isinstance(configured, Mapping):
            raise TypeError(
                "BuildProfile.options['build_settings'] must be a mapping"
            )
        settings = normalize_build_settings(dict(configured))
    else:
        settings = load_build_settings_for_build(request.project_root)

    target_options = settings["platform_options"].get(str(request.target), {})
    resolved_options = dict(target_options)
    resolved_options.update(
        {
            key: value
            for key, value in request.profile.options.items()
            if key != "build_settings"
        }
    )
    required_presentation = (
        "display_mode",
        "window_width",
        "window_height",
        "window_resizable",
    )
    missing = [key for key in required_presentation if key not in resolved_options]
    if missing:
        raise ValueError(
            f"Build target {request.target} did not resolve Player presentation "
            "options: " + ", ".join(missing)
        )
    display_mode = resolved_options["display_mode"]
    if not isinstance(display_mode, str) or display_mode not in {
        "fullscreen_borderless", "windowed"
    }:
        raise ValueError("Player presentation display_mode is invalid")
    for key in ("window_width", "window_height"):
        value = resolved_options[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Player presentation {key} must be an integer")
        if value <= 0:
            raise ValueError(f"Player presentation {key} must be positive")
    if not isinstance(resolved_options["window_resizable"], bool):
        raise TypeError("Player presentation window_resizable must be a boolean")
    settings.update(
        {key: resolved_options[key] for key in required_presentation}
    )
    return settings


def cook_platform_content(
    request: BuildRequest,
    output_root: str | Path,
    *,
    platform_host: Mapping[str, object],
    gpu_compute_aot: bool = False,
) -> PlatformContentCookResult:
    """Cook one immutable Player content closure for a native platform host."""

    settings = build_settings_for_request(request)
    game_name = str(settings["game_name"]).strip() or Path(
        resolved_path(request.project_root)
    ).name
    root = Path(resolved_path(output_root))
    root.mkdir(parents=True, exist_ok=True)
    request.report("cook", 0, 1000, "Publishing current project asset catalog")
    if request.asset_catalog_entries:
        catalog_entries = [dict(item) for item in request.asset_catalog_entries]
    else:
        # Publish and fully release the temporary headless host before the
        # content builder creates runtime-facing caches.
        catalog = publish_player_asset_catalog_for_host(request.project_root)
        catalog_entries = list(catalog["entries"])
    builder = GameBuilder(
        request.project_root,
        str(root),
        game_name=game_name,
        icon_guid=str(settings["icon_guid"]),
        display_mode=str(settings["display_mode"]),
        window_width=int(settings["window_width"]),
        window_height=int(settings["window_height"]),
        window_resizable=bool(settings["window_resizable"]),
        splash_items=list(settings["splash_items"]),
        debug_mode=request.profile.configuration
        is BuildConfiguration.DEVELOPMENT,
        lto=False,
        include_jit_runtime=False,
        build_scene_guids=list(settings["scene_guids"]),
    )
    builder.freeze_asset_index_entries(catalog_entries)
    def report(message: str, fraction: float) -> None:
        request.report(
            "cook",
            max(0, min(1000, int(float(fraction) * 1000))),
            1000,
            message,
        )

    cooked = Path(
        builder.cook_platform_content(
            str(root),
            platform_host=dict(platform_host),
            gpu_compute_aot=gpu_compute_aot,
            on_progress=report,
        )
    )
    if not cooked.is_dir():
        raise RuntimeError(
            f"Platform Player cook did not produce its data directory: {cooked}"
        )
    python_sources = tuple(
        Path(path) for path in builder.cooked_python_source_paths()
    )
    builder._runtime_platform = str(platform_host.get("platform", "")).strip().casefold()
    return PlatformContentCookResult(
        game_name,
        cooked,
        dict(settings),
        python_sources,
    )


__all__ = [
    "PlatformContentCookResult",
    "build_settings_for_request",
    "cook_platform_content",
    "read_cooked_player_icon",
]
