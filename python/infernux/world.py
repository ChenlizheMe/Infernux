"""Public management of independent runtime worlds in the current Play session.

Component object operations keep their implicit owner. Only this explicit API
manages other worlds; native SceneManager remains the sole collection owner.
"""
from __future__ import annotations

from dataclasses import dataclass
import weakref


def _owner_session():
    from infernux.application import Application
    from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
    assert_runtime_owner_thread()
    engine = Application._current_engine()
    session = engine._runtime_world_session if engine is not None else None
    if session is None or session.closed:
        raise RuntimeError('WorldManager requires an active Play session')
    from infernux.lib import SceneManager
    manager = SceneManager.instance()
    owner = int(manager._current_runtime_world_id)
    if owner != int(manager._authoring_runtime_world_id):
        if int(manager._get_runtime_world_handle(owner).management_session) != session.identity:
            raise RuntimeError('WorldManager callback belongs to a retired Play session')
    return session


def _post(session, name, action):
    from infernux.host.commands import MainThreadCommandQueue
    # A prior Play's accepted command can never mutate the next Play session.
    MainThreadCommandQueue.instance().post(name, lambda: action() if not session.closed else None)


@dataclass(eq=False, init=False)
class World:
    """Stable world handle returned immediately, including during preparation.

    state is preparing, ready, retiring, failed, cancelled or destroyed. Graph
    access requires ready. error remains available after failed/cancelled;
    handles remain inspectable after Stop without keeping native worlds alive.
    """
    _session_ref: object
    _handle: object

    def __init__(self, session, handle):
        # A retained terminal handle must not retain the initial template,
        # Engine, native graph, Python realms or shared GPU arena through it.
        self._session_ref = weakref.ref(session)
        self._handle = handle

    @property
    def _session(self):
        return self._session_ref()

    @property
    def id(self) -> int:
        return int(self._handle.id)

    @property
    def state(self) -> str:
        return self._handle.state

    @property
    def is_ready(self) -> bool:
        return self.state == 'ready'

    @property
    def error(self) -> str:
        return self._handle.error

    @property
    def pending_primary(self) -> bool:
        return self._handle.pending_primary

    @property
    def destroy_requested(self) -> bool:
        return self._handle.destroy_requested

    @property
    def display_in_game(self) -> bool:
        return self._handle.display_in_game

    @display_in_game.setter
    def display_in_game(self, value: bool):
        self._require_current()
        if type(value) is not bool:
            raise TypeError('display_in_game must be bool')
        if self.state not in {'preparing', 'ready'} or self.destroy_requested:
            raise RuntimeError('the world is no longer available for display')
        # Presentation metadata only: this never changes simulation scheduling.
        self._handle.display_in_game = value

    def _require_current(self):
        if _owner_session() is not self._session:
            raise ValueError('the world belongs to a different Play session')

    def get_scenes(self) -> tuple:
        """Return fully published loaded Scenes in this world's own order."""
        self._require_current()
        if not self.is_ready or self.destroy_requested:
            raise RuntimeError('the world scene graph is not available')
        from infernux.lib import SceneManager, _Infernux
        manager = SceneManager.instance()
        with _Infernux._RuntimeWorldScope(self.id):
            return tuple(manager.get_scene_at(index) for index in range(manager.scene_count))

    def get_active_scene(self):
        self._require_current()
        if not self.is_ready or self.destroy_requested:
            raise RuntimeError('the world scene graph is not available')
        from infernux.lib import SceneManager, _Infernux
        with _Infernux._RuntimeWorldScope(self.id):
            return SceneManager.instance().get_active_scene()

    def destroy(self) -> bool:
        return WorldManager.destroy_world(self)

    def __eq__(self, other):
        return isinstance(other, World) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class WorldManager:
    """One native runtime collection, shared by scripts, GUI and Host commands.

    Mutations are accepted in call order and committed at the next safe owner
    boundary. Background transports enter via MainThreadCommandQueue.run_sync;
    ordinary component calls receive World/tuple/bool, never CommandFuture.
    """

    @staticmethod
    def create_world() -> World:
        return WorldManager.create_worlds(1)[0]

    @staticmethod
    def create_worlds(count: int) -> tuple[World, ...]:
        if type(count) is not int or count < 0:
            raise ValueError('world count must be a non-negative integer')
        session = _owner_session()
        worlds = []
        for _ in range(count):
            handle = session.reserve()
            world = World(session, handle)
            worlds.append(world)
        if worlds:
            handles = tuple(world._handle for world in worlds)
            _post(session, 'WorldManager.create_worlds', lambda: session.request_create(handles))
        return tuple(worlds)

    @staticmethod
    def destroy_world(world: World) -> bool:
        if not isinstance(world, World):
            raise TypeError('destroy_world requires a World')
        world._require_current()
        if world.state in {'retiring', 'failed', 'cancelled', 'destroyed'} or world.destroy_requested:
            return False
        world._handle.destroy_requested = True
        _post(world._session, 'WorldManager.destroy_world', lambda: world._session.destroy(world._handle))
        return True

    @staticmethod
    def set_primary_world(world: World) -> None:
        if not isinstance(world, World):
            raise TypeError('set_primary_world requires a World')
        world._require_current()
        from infernux.lib import SceneManager
        SceneManager.instance()._request_runtime_world_primary(world._handle)
        _post(world._session, 'WorldManager.set_primary_world',
              lambda: world._session.apply_primary(world._handle))

    @staticmethod
    def get_worlds() -> tuple[World, ...]:
        session = _owner_session()
        from infernux.lib import SceneManager
        manager = SceneManager.instance()
        return tuple(World(session, manager._get_runtime_world_handle(identity))
                     for identity in manager._runtime_world_ids)

    @staticmethod
    def get_primary_world() -> World | None:
        worlds = WorldManager.get_worlds()
        return worlds[0] if worlds else None

    @staticmethod
    def get_current_world() -> World | None:
        """Return the runtime owner; an explicit authoring scope returns None.

        Native SceneManager already selects primary when no scope is present.
        """
        session = _owner_session()
        from infernux.lib import SceneManager
        manager = SceneManager.instance()
        identity = int(manager._current_runtime_world_id)
        if identity == int(manager._authoring_runtime_world_id):
            return None
        return World(session, manager._get_runtime_world_handle(identity))

    @staticmethod
    def set_world_count(count: int) -> tuple[World, ...]:
        """Idempotently request the total, including pending initial instances."""
        if type(count) is not int or count < 0:
            raise ValueError('world count must be a non-negative integer')
        session = _owner_session()
        from infernux.lib import SceneManager
        handles = SceneManager.instance()._managed_runtime_world_handles
        published = [handle for handle in handles if handle.state == 'ready' and not handle.destroy_requested]
        preparing = sorted((handle for handle in handles if handle.state == 'preparing'
                            and not handle.destroy_requested), key=lambda handle: handle.id)
        worlds = tuple(World(session, handle) for handle in (*published, *preparing))
        if len(worlds) < count:
            return worlds + WorldManager.create_worlds(count - len(worlds))
        for world in reversed(worlds[count:]):
            WorldManager.destroy_world(world)
        return worlds[:count]

    @staticmethod
    def get_world_statuses() -> tuple[World, ...]:
        """Return native current-Play management state, including lightweight terminals.

        Published worlds lead in actual primary order; preparing, retiring and
        terminal records follow. Terminal history ends at Stop; retaining a World
        keeps its metadata inspectable without keeping its graph or domain alive.
        """
        session = _owner_session()
        from infernux.lib import SceneManager
        return tuple(World(session, handle) for handle in SceneManager.instance()._managed_runtime_world_handles)


__all__ = ['World', 'WorldManager']
