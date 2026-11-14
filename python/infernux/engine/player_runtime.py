"""Minimal runtime session used by the packaged Player."""

from __future__ import annotations

import sys
import tempfile
from typing import TYPE_CHECKING, Any, Optional

from infernux.application import ContentPublication
from infernux.debug import Debug
from infernux.engine.player_scene import PlayerSceneService
from infernux.engine.work_runtime import WorkRuntime

if TYPE_CHECKING:
    from infernux.engine.player_service_graph import (
        PlayerRuntimeAssetCatalog,
        RuntimeProductManifest,
    )


class PlayerRuntimeSession(WorkRuntime):
    """Own the standalone Player scene without editor document services."""

    def __init__(
        self,
        *,
        asset_database: Any = None,
        native_engine: Any = None,
        scheduler: Any = None,
        scene_service: Any = None,
        frame_work: Any = None,
    ):
        from infernux.engine.runtime_dispatch import bind_runtime_dispatch_owner

        bind_runtime_dispatch_owner()
        self._asset_database = asset_database
        self._native_engine = native_engine
        if frame_work is None:
            from infernux.lib import _Infernux

            frame_work = native_engine.frame_work if native_engine is not None else _Infernux._FrameWorkOwner()
        self._initialize_work_runtime(frame_work, accepting=True)
        if scheduler is None:
            from infernux.components._component_lifecycle import RuntimeExecutionScheduler

            scheduler = RuntimeExecutionScheduler(name="player", native_bridge=True)
        self._execution_scheduler = scheduler
        self._scene_service = scene_service or PlayerSceneService(
            asset_database=asset_database,
            native_engine=native_engine,
        )
        self._scene_service_installed = False
        self._runtime_world_session = None
        self._runtime_world_id = None
        self._runtime_manifest: Optional[RuntimeProductManifest] = None
        self._runtime_catalog: Optional[PlayerRuntimeAssetCatalog] = None
        self._content_workspace: Optional[tempfile.TemporaryDirectory] = None
        self._pending_content: Optional[tuple[ContentPublication, str]] = None
        self._content_roots: tuple[str, ...] = ()
        self._state = "stopped"
        self._membership_warmup_frames = 0
        self._startup_warmup = None

    def _request_full_speed_frame(self):
        # Browser frames already run continuously; desktop sessions borrow the
        # native loop's wake operation instead of creating a second driver.
        if self._native_engine is not None:
            self._native_engine.request_full_speed_frame()

    def schedule_startup_warmup(self, warmup: Any) -> None:
        """Install the post-present warmup queue before the native loop starts."""
        if self._startup_warmup is not None:
            raise RuntimeError("Player startup warmup is already scheduled")
        self._startup_warmup = warmup

    def pump_startup_warmup(self) -> bool:
        """Advance the bounded startup queue and report when it is drained."""
        warmup = self._startup_warmup
        if warmup is None:
            return True
        if warmup.pump():
            self._startup_warmup = None
            return True
        return False

    def _refresh_execution_membership(self) -> None:
        # Defer the registry scan to the scheduler's next safe point.  Calling
        # the scan synchronously from native scene callbacks opens a runtime
        # transaction while a frame is active; marking it pending preserves the
        # single-owner boundary without dropping freshly published components.
        # The native pre-scene callback may run while a frame transaction is
        # active.  Only mark the registry scan here; the scheduler consumes it
        # at its next owner safe point and publishes the immutable phase plan.
        scheduler = self._execution_scheduler
        scheduler._registry_scan_pending = True
        # Advertise that Python work exists immediately.  This only changes
        # the cheap native gate; the phase counts are published by
        # ``prepare_frame`` at the next owner safe point.
        scheduler._sync_native_work_availability()

    @property
    def is_playing(self) -> bool:
        return self._state == "playing"

    @property
    def is_paused(self) -> bool:
        from infernux.lib import SceneManager
        return self.is_playing and SceneManager.instance().is_paused()

    @property
    def state(self) -> str:
        return self._state

    @property
    def active_scene_path(self) -> Optional[str]:
        return self._scene_service.active_scene_path

    @property
    def last_scene_error(self) -> str:
        return str(getattr(self._scene_service, "last_error", ""))

    @property
    def execution_scheduler(self) -> Any:
        """Return the shared on-demand phase-plan service for diagnostics."""
        return self._execution_scheduler

    def get_asset_database(self) -> Any:
        """Return the Player's already-published runtime asset database.

        ``AssetManager.initialize`` consumes this narrow host contract in both
        desktop and Web Players. The returned database is the immutable
        runtime catalog view installed by the platform host; it is never an
        Editor source database and cannot scan ``Assets``.
        """
        if self._asset_database is None:
            raise RuntimeError("Player runtime asset database is not configured")
        return self._asset_database

    def get_native_engine(self) -> Any:
        """Return the desktop host, or None for a platform-owned frame loop."""
        return self._native_engine

    @property
    def runtime_manifest(self) -> Optional[RuntimeProductManifest]:
        return self._runtime_manifest

    def configure_runtime_contract(
        self,
        runtime_manifest: RuntimeProductManifest,
        runtime_catalog: PlayerRuntimeAssetCatalog,
    ) -> None:
        """Bind the build-authored Player product before loading a scene."""
        from infernux.engine.player_service_graph import (
            PlayerRuntimeAssetCatalog,
            RuntimeProductManifest,
        )

        if not isinstance(runtime_manifest, RuntimeProductManifest):
            raise TypeError("PlayerRuntimeSession requires a RuntimeProductManifest")
        if not isinstance(runtime_catalog, PlayerRuntimeAssetCatalog):
            raise TypeError("PlayerRuntimeSession requires a RuntimeAssetCatalog")
        if not runtime_manifest.flavor.is_player:
            raise RuntimeError("EditorDevelopment cannot configure PlayerRuntimeSession")
        for service_id in (
            "runtime_asset_catalog",
            "runtime_type_registry",
            "player_scene_service",
            "player_runtime_session",
        ):
            runtime_manifest.require_service(service_id)
        if self._runtime_manifest is not None:
            if (
                self._runtime_manifest is runtime_manifest
                and self._runtime_catalog is runtime_catalog
            ):
                return
            raise RuntimeError("PlayerRuntimeSession runtime contract is already configured")
        if self._state != "stopped":
            raise RuntimeError("PlayerRuntimeSession cannot be configured after activation")
        bind_catalog = getattr(self._scene_service, "bind_runtime_catalog", None)
        if not callable(bind_catalog):
            raise RuntimeError("Player scene service cannot bind RuntimeAssetCatalog")
        bind_catalog(runtime_catalog)
        self._set_runtime_catalog(runtime_catalog)
        self._runtime_manifest = runtime_manifest
        self._validate_player_boundary()

    def _set_runtime_catalog(self, runtime_catalog: PlayerRuntimeAssetCatalog) -> None:
        from infernux.engine.project_context import (
            set_runtime_asset_extension_resolver,
            set_runtime_asset_query,
            set_runtime_asset_resolver,
            set_runtime_package_resolver,
        )

        set_runtime_asset_resolver(runtime_catalog.resolve_guid)
        set_runtime_asset_extension_resolver(runtime_catalog.source_extension_for_guid)
        set_runtime_package_resolver(runtime_catalog.resolve_package)
        set_runtime_asset_query(runtime_catalog.query_asset_guids)
        self._runtime_catalog = runtime_catalog

    def publish_content(self, data_directory: str) -> ContentPublication:
        from infernux.engine.path_utils import resolved_path
        from infernux.engine.platform_player_bootstrap import _CONTENT_ROOTS
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread

        assert_runtime_owner_thread()
        if sys.platform in {"android", "emscripten"} or not self._scene_service_installed or self._state == "stopping":
            raise RuntimeError("Content publication requires a running desktop Player")
        if self._pending_content is not None:
            raise RuntimeError("A Player content publication is already preparing")
        if self._content_workspace is None:
            self._content_workspace = tempfile.TemporaryDirectory(prefix="infernux-player-content-")
        destination = tempfile.mkdtemp(dir=self._content_workspace.name)
        database = self.get_asset_database()
        database._begin_runtime_asset_catalog(
            resolved_path(data_directory), resolved_path(destination), sorted(_CONTENT_ROOTS)
        )
        publication = ContentPublication()
        self._pending_content = publication, destination
        return publication

    def _publish_ready_content(self) -> None:
        if self._pending_content is None:
            return
        from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
        from infernux.engine.platform_player_bootstrap import _parse_package_index
        from infernux.engine.project_context import (
            get_script_module_name,
            set_runtime_content_roots,
        )
        from infernux.engine.runtime_artifact_catalog import CATALOG_SCHEMA
        from infernux.engine.runtime_type_registry import _prepare_runtime_type_registry

        publication, destination = self._pending_content
        database = self.get_asset_database()
        try:
            documents = database._prepared_runtime_catalog_documents()
            if documents is None:
                return
            index = _parse_package_index(documents["package_index"])
            for kind in ("content", "catalog"):
                if index.get(kind) != tuple(documents[f"{kind}_identity"]):
                    raise RuntimeError(f"Player {kind} archive disagrees with its package index")
            catalog = documents["catalog"]
            if catalog.get("$schema") != CATALOG_SCHEMA:
                raise RuntimeError("Unsupported runtime asset catalog schema")
            incoming = PlayerRuntimeAssetCatalog.from_documents(
                destination, catalog, documents["records"], content_files=documents["content_files"]
            )
            guids = set(documents["asset_guids"])
            merged = self._runtime_catalog.with_additional_content(incoming, guids)
            occupied_modules = set(sys.modules) | set(sys.builtin_module_names)
            occupied_modules.update(get_script_module_name(path) for path in self._runtime_catalog.script_payloads())
            scripts = {}
            modules = {}
            for guid in guids:
                if incoming.source_extension_for_guid(guid) != ".pyc":
                    continue
                path = incoming.resolve_guid(guid)
                module = get_script_module_name(path, incoming.project_root)
                if module is None or module in occupied_modules:
                    raise RuntimeError(f"Cooked content uses an unavailable Python module identity: {module}")
                scripts[guid] = module
                modules[module] = path
                occupied_modules.add(module)
            types = _prepare_runtime_type_registry(documents["types"], new_scripts=scripts)
            asset_guids = tuple(sorted(guids))
            scene_guids = tuple(guid for guid in asset_guids if incoming.resolve_scene(guid) is not None)
            roots = (*self._content_roots, incoming.project_root)

            # Both candidates have been validated. Commit the catalogs on the
            # same owner boundary before dispatching any scene or script work.
            types.publish()
            database._publish_runtime_asset_catalog()
            self._scene_service.bind_runtime_catalog(merged)
            self._set_runtime_catalog(merged)
            set_runtime_content_roots(roots, modules)
            self._content_roots = roots
            publication.asset_guids = asset_guids
            publication.scene_guids = scene_guids
            publication.status = "published"
        except Exception as exc:
            database._discard_runtime_asset_catalog()
            publication.error = str(exc) or type(exc).__name__
            publication.status = "failed"
        self._pending_content = None

    def _validate_player_boundary(self) -> None:
        forbidden = (
            "_scene_backup",
            "_scene_dirty_backup",
            "_resources_manager",
            "_selection_manager",
            "_undo_manager",
            "_preview_service",
            "_import_coordinator",
        )
        leaked = [name for name in forbidden if hasattr(self, name)]
        if leaked:
            raise RuntimeError(
                "PlayerRuntimeSession contains Editor state: " + ", ".join(leaked)
            )

    def load_scene(self, path: str) -> bool:
        """Load a scene through the runtime transaction path."""
        if self._runtime_manifest is None or self._runtime_catalog is None:
            Debug.log_error("Player scene load rejected: runtime contract is not configured")
            return False
        loaded = bool(
            self._scene_service.load_initial(path, on_tick=self._pump_startup_events)
        )
        if loaded:
            self._refresh_execution_membership()
        return loaded

    def _pump_startup_events(self) -> None:
        native = self._native_engine
        if native is None:
            return
        pump = getattr(native, "pump_events", None)
        if callable(pump) and pump() is False:
            raise RuntimeError("Player startup cancelled")

    def activate(self) -> bool:
        """Start the loaded scene without creating an editor play snapshot."""
        if self._runtime_manifest is None or self._runtime_catalog is None:
            Debug.log_error("Player activation rejected: runtime contract is not configured")
            return False
        from infernux.lib import SceneManager

        scene = SceneManager.instance().get_active_scene()
        if scene is None:
            Debug.log_warning("Player cannot start without an active scene")
            return False
        from infernux.lib import _Infernux as native_module

        if getattr(native_module, "__runtime_profile__", "desktop") != "web-player":
            from infernux.renderstack.render_stack import RenderStack

            RenderStack.clear_active_instance()
        from infernux.scene import SceneManager as RuntimeSceneManager

        from infernux.application import Application
        from infernux.engine.runtime_worlds import RuntimeWorldSession, capture_runtime_sources, component_world_id
        engine = Application._current_engine()
        self._runtime_world_session = RuntimeWorldSession(engine, capture_runtime_sources(),
                                                         scene_service=self._scene_service)
        try:
            identity = self._runtime_world_session.create()
            self._runtime_world_id = identity
            domain = engine._world_runtimes[identity]
            self._execution_scheduler = domain.scheduler
            self._scene_service = domain.scene_loads._runtime_scene_service
            self._scene_service_installed = True
            with native_module._RuntimeWorldScope(identity):
                scene = SceneManager.instance().get_active_scene()
                self._refresh_loaded_scene(scene)
                self._refresh_execution_membership()
                SceneManager.instance().play()
                self._refresh_execution_membership()
                from infernux.components.component import InxComponent
                from infernux.engine.startup_warmup import run_component_warmups
                from infernux.engine.project_context import get_project_root
                run_component_warmups(
                    (component for values in InxComponent._active_instances.values() for component in values
                     if component_world_id(component) == identity),
                    scope="player", project_path=get_project_root())
        except BaseException:
            if self._runtime_world_id is not None:
                from infernux.lib import _Infernux
                with _Infernux._RuntimeWorldScope(self._runtime_world_id):
                    SceneManager.instance().stop()
            self._runtime_world_session.close()
            self._runtime_world_session = None
            self._runtime_world_id = None
            self._scene_service_installed = False
            raise
        self._state = "playing"
        # Native Start may publish Python component mirrors one frame later
        # on a packaged scene. Keep a short deterministic warm-up window so
        # the first fixed step cannot observe an empty runtime phase plan.
        self._membership_warmup_frames = 8
        return True

    def prepare_frame(self) -> None:
        if not self.is_playing:
            return
        from infernux.lib import _Infernux
        with _Infernux._RuntimeWorldScope(self._runtime_world_id):
            self._prepare_runtime_world()

    def _prepare_runtime_world(self) -> None:
        """Prepare one Player frame at the safe pre-scene boundary.

        Scene callbacks can queue a runtime load while the native scene is
        being iterated. The editor advances that transaction before the next
        native scene tick; Player must do the same without constructing any
        editor-only scene services.
        """
        if not self.is_playing:
            return
        self.pump_startup_warmup()
        self._publish_ready_content()
        if self._membership_warmup_frames > 0:
            self._refresh_execution_membership()
            self._membership_warmup_frames -= 1
        previous_scene_path = self._scene_service.active_scene_path
        self._scene_service.process_pending_load()
        if self._scene_service.active_scene_path != previous_scene_path:
            self._refresh_execution_membership()

    def shutdown(self) -> None:
        """Retire Player scenes while their resource catalog is still bound."""
        from infernux.lib import SceneManager

        self._state = "stopping"
        self._shutdown_work_runtime()
        if self._pending_content is not None:
            publication, _ = self._pending_content
            self.get_asset_database()._discard_runtime_asset_catalog()
            publication.status = "cancelled"
            self._pending_content = None
        native = SceneManager.instance()
        if self._runtime_world_id is None:
            from infernux.lib import _Infernux
            with _Infernux._RuntimeWorldScope(native._authoring_runtime_world_id):
                self._scene_service.cancel_pending_load()
        native.stop()
        if self._runtime_world_session is not None:
            self._runtime_world_session.close()
            self._runtime_world_session = None
        # The Engine companion and shared service retain pending retired domains.
        self._runtime_world_id = None
        from infernux.application import Application

        if Application._current_engine() is self:
            from infernux.compute import _release_engine_resources

            _release_engine_resources()
            self._frame_work.stop()
        self._graph_event_signals = None
        self._scene_service_installed = False
        self._state = "stopped"

    def release_content(self) -> None:
        """Release the extraction tree after the native asset worker drains."""
        from infernux.engine.project_context import (
            set_runtime_asset_extension_resolver,
            set_runtime_asset_query,
            set_runtime_asset_resolver,
            set_runtime_content_roots,
            set_runtime_package_resolver,
        )

        set_runtime_asset_resolver(None)
        set_runtime_asset_extension_resolver(None)
        set_runtime_package_resolver(None)
        set_runtime_asset_query(None)
        set_runtime_content_roots(())
        self._content_roots = ()
        if self._content_workspace is not None:
            self._content_workspace.cleanup()
            self._content_workspace = None

    @staticmethod
    def _refresh_loaded_scene(scene: Any) -> None:
        try:
            from infernux.components.builtin.sprite_renderer import SpriteRenderer

            SpriteRenderer.init_all_in_scene(scene)
        except Exception as exc:
            Debug.log_internal(f"Player SpriteRenderer init: {exc}")
