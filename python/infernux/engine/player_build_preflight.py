"""Shared publication barrier for standalone Player builds.

Every build entry point must hand :class:`GameBuilder` an immutable snapshot
of an AssetIndex produced *after* editor document writes have reached disk.
The docked Build Settings UI has an incremental variant of this barrier; this
module provides the synchronous form used by automation and headless hosts.
"""

from __future__ import annotations

import os
from infernux._source_kinds import is_host_python_source
from typing import Any

from infernux.engine.path_utils import resolved_path, same_path


def publish_player_asset_catalog(project_root: str, asset_database: Any) -> dict[str, Any]:
    """Flush authoring writes, rebuild derived products, and snapshot AssetIndex."""

    requested_root = str(project_root or "").strip()
    if not requested_root:
        raise RuntimeError("No project root found")
    root = resolved_path(requested_root)
    if asset_database is None:
        raise RuntimeError("The editor asset database is unavailable")

    from infernux.core.assets import AssetManager
    from infernux.renderstack.discovery import discover_effect_features

    applied = AssetManager.flush_all_asset_writes(database=asset_database)
    if applied is not None and not applied:
        raise RuntimeError(f"Model Apply failed before export for {applied.path!r}: {applied.error}")
    discover_effect_features()
    # A headless/editor worker can arrive here while the initial scan is still
    # pending. Commit that owner-side scan before requesting the build refresh;
    # AssetDatabase intentionally rejects overlapping refresh transactions.
    if bool(getattr(asset_database, "refresh_pending", False)):
        complete = getattr(asset_database, "complete_pending_refresh", None)
        if callable(complete):
            complete()
    asset_database.refresh()

    from infernux.particle.artifact import ParticleArtifactRegistry

    try:
        ParticleArtifactRegistry.ensure_project_compiled(root, raise_on_error=True)
    except Exception as exc:
        raise RuntimeError(f"Particle artifact compile failed: {exc}") from exc

    asset_database.flush_derived_index()
    index_path = str(getattr(asset_database, "asset_index_path", "") or "")
    if not index_path or not os.path.isfile(index_path):
        raise RuntimeError(
            "The editor could not publish the current Library/AssetIndex.json"
        )

    from infernux.engine.runtime_artifact_catalog import load_asset_index

    return capture_player_build_inputs(root, load_asset_index(root), path=index_path)


def capture_player_build_inputs(project_root, entries, *, path="") -> dict[str, Any]:
    """Freeze asset revisions and their owner-published type semantics together."""
    from infernux.application import Application
    from infernux.engine.path_utils import relative_path, resolve_project_path
    from infernux.engine.project_context import (
        is_editor_asset_path, is_project_script, is_preload_python_library,
    )
    from infernux.engine.resources_manager import ResourceChangeHandler, ResourcesManager
    from infernux.engine.runtime_artifact_catalog import source_fingerprint, source_path_for_entry
    from infernux.engine.player_script_catalog import capture_runtime_type_records
    from infernux.engine.script_source import validate_loaded_preload_sources

    root = resolved_path(project_root)
    scripts = []
    for entry in entries:
        raw = str(entry.get("normalized_path", ""))
        if not is_host_python_source(raw):
            continue
        candidate = resolve_project_path(raw, root)
        if (is_project_script(candidate, root)
                and not is_editor_asset_path(relative_path(candidate, root))):
            source = source_path_for_entry(root, entry)
            source_fingerprint(root, entry)
            scripts.append((source, entry))
    validate_loaded_preload_sources(source for source, _ in scripts)
    sources = {}
    reload_paths = [source for source, _ in scripts if not is_preload_python_library(source)]
    if reload_paths:
        manager = ResourcesManager.instance()
        local = manager is None
        if local:
            engine = Application._current_engine()
            handler = ResourceChangeHandler(engine._engine, project_path=root)
        else:
            manager._wait_for_startup_idle()
            handler = manager._ensure_event_handler()
        try:
            sources = handler.publish_script_sources(reload_paths)
            types = capture_runtime_type_records(root, [entry for _, entry in scripts], sources)
            for _, entry in scripts:
                source_fingerprint(root, entry)
        finally:
            if local:
                handler.cleanup()
    else:
        types = capture_runtime_type_records(root, [entry for _, entry in scripts], sources)
        for _, entry in scripts:
            source_fingerprint(root, entry)
    validate_loaded_preload_sources(source for source, _ in scripts)
    return {"path": str(path), "entries": entries, "runtime_type_records": types}


def publish_player_asset_catalog_for_host(project_root: str) -> dict[str, Any]:
    """Publish a current catalog from either the active Editor or a headless host."""

    root = resolved_path(project_root)
    from infernux.lib import AssetRegistry

    active_database = AssetRegistry.instance().get_asset_database()
    if active_database is not None:
        active_root = str(getattr(active_database, "project_root", "") or "")
        if active_root and same_path(active_root, root):
            if active_database.is_owner_thread():
                return publish_player_asset_catalog(root, active_database)
            from infernux.host.commands import MainThreadCommandQueue

            # Background builds consume a snapshot; only its authoring
            # publication runs on the engine owner. Compilation stays on the
            # calling worker and never borrows the mutable AssetDatabase.
            return MainThreadCommandQueue.instance().run_sync(
                "player.build.publish_asset_catalog",
                lambda: publish_player_asset_catalog(root, active_database),
                timeout_ms=600000,
            )
        if active_root:
            raise RuntimeError(
                "The active AssetDatabase belongs to another project; run this build "
                "in an isolated headless process"
            )

    from infernux.engine.engine import Engine
    from infernux.engine.library_sync import sync_resources
    from infernux.lib import LogLevel, RuntimeMode
    from infernux.resources import activate_library

    # Use the same built-in source identities as graphical Editor startup.
    sync_resources(root)
    activate_library(root)
    engine = Engine(LogLevel.Info, mode=RuntimeMode.Headless)
    try:
        engine.init_headless(root)
        return publish_player_asset_catalog(root, engine.get_asset_database())
    finally:
        engine.exit()


__all__ = [
    "publish_player_asset_catalog",
    "publish_player_asset_catalog_for_host",
    "capture_player_build_inputs",
]
