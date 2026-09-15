"""Player-only scene ownership and deferred runtime loading."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from Infernux.debug import Debug
from Infernux.engine.path_utils import resolved_path
from Infernux.engine.player_log import write_player_log
from Infernux.engine.runtime_scene_transaction import SceneDocumentTransaction

if TYPE_CHECKING:
    from Infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog


def _player_log(message: str) -> None:
    """Append packaged scene diagnostics without enabling release console spam."""
    try:
        write_player_log(message)
    except Exception as exc:
        Debug.log_suppressed("player_scene.write_player_log", exc)


class PlayerSceneService:
    """Own Player scene transactions without editor document services."""

    def __init__(self, *, asset_database: Any = None, native_engine: Any = None) -> None:
        self._asset_database = asset_database
        self._native_engine = native_engine
        self._runtime_catalog: Any = None
        self._active_scene_path: Optional[str] = None
        self._pending_scene_path: Optional[str] = None
        self._pending_scene_mode = "single"
        self._transaction: Optional[SceneDocumentTransaction] = None
        self._transaction_path: Optional[str] = None
        self._transaction_mode = "single"
        self._transaction_target: Any = None
        self._wait_for_ready = False
        self._hold_for_activation = False
        self._request_generation = 0
        self._transaction_generation = 0
        self._last_error = ""

    @property
    def active_scene_path(self) -> Optional[str]:
        return self._active_scene_path

    @property
    def is_load_pending(self) -> bool:
        return self._pending_scene_path is not None or self._transaction is not None

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def is_prepared(self) -> bool:
        return bool(
            self._hold_for_activation
            and self._transaction is not None
            and self._transaction.status == "ready_to_commit"
        )

    def bind_runtime_catalog(self, runtime_catalog: PlayerRuntimeAssetCatalog) -> None:
        """Bind the immutable catalog before any Player scene is loaded."""
        from Infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog

        if not isinstance(runtime_catalog, PlayerRuntimeAssetCatalog):
            raise TypeError("PlayerSceneService requires a RuntimeAssetCatalog")
        if self._runtime_catalog is not None and self._runtime_catalog is not runtime_catalog:
            raise RuntimeError("PlayerSceneService runtime catalog is already bound")
        if self.is_load_pending or self._active_scene_path is not None:
            raise RuntimeError("PlayerSceneService catalog cannot change after scene loading")
        self._runtime_catalog = runtime_catalog

    @staticmethod
    def _normalize_mode(mode: Any) -> str:
        value = str(getattr(mode, "value", mode)).strip().lower()
        if value not in {"single", "additive"}:
            raise ValueError("scene load mode must be 'single' or 'additive'")
        return value

    def load_initial(self, path: str, *, on_tick=None) -> bool:
        """Synchronously load the first Player scene before activation."""
        target = self._validated_scene_path(path)
        if target is None:
            self._last_error = f"scene file is unavailable: {path}"
            return False
        self._last_error = ""
        transaction = self._new_transaction(target)
        if not transaction.run_to_completion(raise_on_failure=False, on_tick=on_tick):
            self._last_error = transaction.error or "scene transaction failed"
            Debug.log_error(
                f"Player scene load failed for '{target}': {self._last_error}"
            )
            return False
        self._log_transaction_timings(target, transaction, initial=True)
        self._publish_completed_scene(target, start_for_play=False)
        return True

    def request_load(self, path: str, *, mode: Any = "single") -> bool:
        """Queue a scene replacement for the next safe Player frame boundary."""
        load_mode = self._normalize_mode(mode)
        target = self._validated_scene_path(path)
        if target is None:
            self._last_error = f"scene file is unavailable: {path}"
            return False
        generation = self._begin_load_request()
        if generation is None:
            return False
        self._last_error = ""
        self._pending_scene_path = target
        self._pending_scene_mode = load_mode
        return True

    def request_prepared_load(
        self, path: str, *, hold_for_activation: bool = False, mode: Any = "single"
    ) -> bool:
        """Start preparing a replacement now and publish it when ready."""
        load_mode = self._normalize_mode(mode)
        target = self._validated_scene_path(path)
        if target is None:
            self._last_error = f"scene file is unavailable: {path}"
            return False
        generation = self._begin_load_request()
        if generation is None:
            return False
        self._last_error = ""
        transaction, additive_target = self._transaction_for_mode(target, load_mode)
        try:
            transaction.start()
        except Exception as exc:
            if additive_target is not None:
                from Infernux.lib import SceneManager
                SceneManager.instance().unload_scene(additive_target)
            self._last_error = str(exc) or type(exc).__name__
            Debug.log_error(f"Player scene preparation could not start: {exc}")
            return False
        self._transaction = transaction
        self._transaction_path = target
        self._transaction_mode = load_mode
        self._transaction_target = additive_target
        self._transaction_generation = generation
        self._wait_for_ready = True
        self._hold_for_activation = bool(hold_for_activation)
        return True

    def activate_prepared_load(self) -> bool:
        """Release a held prepared transaction for its owner-thread commit."""
        if not self.is_prepared:
            return False
        self._hold_for_activation = False
        return True

    def process_pending_load(self) -> None:
        """Advance deferred loading, committing a prepared scene when ready."""
        if self._transaction is None:
            target = self._pending_scene_path
            if target is None:
                return
            load_mode = self._pending_scene_mode
            self._pending_scene_path = None
            self._pending_scene_mode = "single"
            transaction, additive_target = self._transaction_for_mode(target, load_mode)
            try:
                transaction.start()
            except Exception as exc:
                if additive_target is not None:
                    from Infernux.lib import SceneManager
                    SceneManager.instance().unload_scene(additive_target)
                self._last_error = str(exc) or type(exc).__name__
                Debug.log_error(f"Player scene load could not start: {exc}")
                return
            self._transaction = transaction
            self._transaction_path = target
            self._transaction_mode = load_mode
            self._transaction_target = additive_target
            self._transaction_generation = self._request_generation
            self._wait_for_ready = False
            return

        transaction = self._transaction
        if self._transaction_generation != self._request_generation:
            if not transaction.is_complete:
                transaction.cancel()
            self._discard_additive_target()
            self._clear_pending_load()
            return
        if self._hold_for_activation and transaction.status == "ready_to_commit":
            return
        # Keep Player loading genuinely asynchronous too.  A prepared-load
        # request may spend many frames in worker-backed resource preflight;
        # spinning here would freeze the Player until those jobs finish.
        if not transaction.poll():
            return

        target = self._transaction_path
        load_mode = self._transaction_mode
        additive_target = self._transaction_target
        # Phase timings are optional diagnostics, not part of the scene
        # transaction protocol.  Alternate/test transactions must remain
        # valid without implementing profiling state.
        phase_timings = getattr(transaction, "phase_timings_ms", None)
        succeeded = bool(transaction.succeeded)
        error = transaction.error
        if not succeeded:
            self._discard_additive_target()
        self._clear_pending_load()
        if not succeeded:
            self._last_error = error or "scene transaction failed"
            Debug.log_error(
                f"Player scene load failed for '{target}': {self._last_error}"
            )
            return
        assert target is not None
        if phase_timings:
            self._log_transaction_timings(target, transaction, initial=False)
        if load_mode == "additive":
            self._publish_completed_additive_scene(target, additive_target)
        else:
            self._publish_completed_scene(target, start_for_play=True)

    @staticmethod
    def _log_transaction_timings(
        target: str, transaction: SceneDocumentTransaction, *, initial: bool
    ) -> None:
        phase_timings = getattr(transaction, "phase_timings_ms", None) or {}
        if not phase_timings:
            return
        details = ", ".join(
            f"{name}={float(duration):.1f}ms"
            for name, duration in phase_timings.items()
        )
        message = (
            f"[SceneLoad] {'initial' if initial else 'switch'} "
            f"scene={os.path.basename(target)!r}, {details}"
        )
        _player_log(message)

    def cancel_pending_load(self) -> None:
        self._request_generation += 1
        if self._transaction is not None:
            self._transaction.cancel()
        self._discard_additive_target()
        self._clear_pending_load()

    def _clear_pending_load(self) -> None:
        self._pending_scene_path = None
        self._pending_scene_mode = "single"
        self._transaction = None
        self._transaction_path = None
        self._transaction_mode = "single"
        self._transaction_target = None
        self._transaction_generation = 0
        self._wait_for_ready = False
        self._hold_for_activation = False

    def _begin_load_request(self) -> Optional[int]:
        transaction = self._transaction
        if transaction is not None and not transaction.is_complete:
            try:
                if not transaction.cancel():
                    self._last_error = "the active scene transaction is already committing"
                    return None
            except Exception as exc:
                self._last_error = str(exc) or type(exc).__name__
                return None
            self._discard_additive_target()
        self._clear_pending_load()
        self._request_generation += 1
        return self._request_generation

    def _new_transaction(self, path: str) -> SceneDocumentTransaction:
        from Infernux.lib import SceneManager

        scene_manager = SceneManager.instance()
        scene = scene_manager.get_active_scene()
        if scene is None:
            scene = scene_manager.create_scene("PlayerScene")
        return SceneDocumentTransaction(
            scene,
            path=path,
            asset_database=self._asset_database,
            native_engine=self._native_engine,
            clear_registries=True,
            before_commit=getattr(
                scene_manager, "prepare_active_scene_replacement", None
            ),
        )

    def _new_additive_transaction(self, path: str):
        from Infernux.lib import SceneManager

        scene_manager = SceneManager.instance()
        scene = scene_manager.create_scene(os.path.splitext(os.path.basename(path))[0])
        return (
            SceneDocumentTransaction(
                scene,
                path=path,
                asset_database=self._asset_database,
                native_engine=self._native_engine,
                clear_registries=False,
            ),
            scene,
        )

    def _transaction_for_mode(self, path: str, mode: str):
        if mode == "additive":
            return self._new_additive_transaction(path)
        return self._new_transaction(path), None

    def _discard_additive_target(self) -> None:
        target = self._transaction_target
        if self._transaction_mode != "additive" or target is None:
            return
        from Infernux.lib import SceneManager

        SceneManager.instance().unload_scene(target)
        self._transaction_target = None

    def _validated_scene_path(self, path: str) -> Optional[str]:
        if self._runtime_catalog is None:
            Debug.log_error("Player scene load rejected: RuntimeAssetCatalog is not bound")
            return None
        target = self._runtime_catalog.resolve_scene(path)
        if target is None:
            Debug.log_warning(f"Player scene is absent from RuntimeAssetCatalog: {path}")
            return None
        return resolved_path(target)

    def _publish_completed_scene(self, path: str, *, start_for_play: bool) -> None:
        from Infernux.lib import SceneManager

        scene_manager = SceneManager.instance()
        scene = scene_manager.get_active_scene()
        if scene is not None:
            kept_world = int(scene.world_id)
            for index in range(int(scene_manager.scene_count) - 1, -1, -1):
                loaded = scene_manager.get_scene_at(index)
                if loaded is not None and int(loaded.world_id) != kept_world:
                    scene_manager.unload_scene(loaded)
            scene_manager.set_active_scene(scene)
        self._active_scene_path = path
        self._last_error = ""
        from Infernux.timing import Time
        Time._reset_frame_delta()
        if start_for_play:
            scene_manager._start_active_scene_for_play()
        message = (
            f"Player loaded scene: {os.path.basename(path)} "
            f"(objects={len(scene.get_all_objects()) if scene is not None else 0}, "
            f"camera={scene is not None and scene.main_camera is not None})"
        )
        _player_log(f"[SceneLoad] {message}")

    def _publish_completed_additive_scene(self, path: str, scene) -> None:
        if scene is None:
            raise RuntimeError("additive Player scene transaction lost its target Scene")
        from Infernux.lib import SceneManager

        SceneManager.instance()._start_scene_for_play(scene)
        self._last_error = ""
        _player_log(
            f"[SceneLoad] Player added scene: {os.path.basename(path)} "
            f"(objects={len(scene.get_all_objects())}, camera={scene.main_camera is not None})"
        )


__all__ = ["PlayerSceneService"]
