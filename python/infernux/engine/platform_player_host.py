"""Player companion for a platform-owned native event loop."""

from .runtime_world_host import RuntimeWorldHost


class PlatformPlayerHost(RuntimeWorldHost):
    def __init__(self, *, asset_database, frame_work, compute_host, initial_world_count=1,
                 process_input, native_services):
        from .player_runtime import PlayerRuntimeSession

        self._closing = False
        self._native_services = native_services
        self._asset_database = asset_database
        self._process_input = process_input
        self._infernux_compute_host = compute_host
        self._play_mode_manager = None
        self._initialize_work_runtime(frame_work, accepting=True)
        self._initialize_world_runtime('player')
        self._player_runtime = PlayerRuntimeSession(
            asset_database=asset_database, frame_work=frame_work,
            scheduler=self._authoring_scheduler, initial_world_count=initial_world_count)
        self._install_runtime_world_callbacks()

    def get_asset_database(self):
        return self._asset_database

    def get_native_engine(self):
        """The platform owns its device and event loop outside the desktop application."""
        return self._native_services

    def get_player_runtime(self):
        return self._player_runtime

    def _runtime_is_closing(self):
        return self._closing

    def request_exit(self):
        self._closing = True

    def _request_full_speed_frame(self):
        """The browser already services every available animation frame."""

    def _dispatch_runtime_input(self):
        self._process_input(self._runtime_scene_manager.get_frame_delta_time())

    def prepare_frame(self, delta_time):
        from infernux.lib import SceneManager, _Infernux
        from infernux.host.commands import MainThreadCommandQueue
        from infernux.runtime_services import _update_snapshot, _dispatch_updates

        self._retire_result_consumers()
        work = self._frame_work
        work.begin_scope()
        try:
            with _Infernux._RuntimeWorldScope(SceneManager.instance()._authoring_runtime_world_id):
                with self._authoring_scheduler.change_journal.transaction():
                    MainThreadCommandQueue.instance().drain(
                        max_commands=None,
                        can_continue=lambda: work.active and work.accepting and work.remaining(background=True) > 0)
            if self._cpu_range_tasks or self._infernux_compute_host is not None:
                self._cpu_work()
        finally:
            work.end_scope()
        if self._runtime_world_session is not None:
            self._runtime_world_session.pump()
        self._player_runtime.prepare_frame()
        self._prepare_runtime_world_schedulers()
        self._clock_callbacks = _update_snapshot()
        self._clock_delta = float(delta_time)
        if self._clock_callbacks['service_update']:
            _dispatch_updates(self._clock_callbacks['service_update'], delta_time, delta_time, gameplay_phase=False)
