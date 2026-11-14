"""Python state accompanying actual native RuntimeWorld identities.

The scene transaction owns creation. This module owns no clock, GPU host,
device, scene collection or alternate simulation loop.
"""
from __future__ import annotations

import builtins
import importlib
import importlib.util
from pathlib import Path
import sys
import types
from collections import ChainMap
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSceneSource:
    name: str
    snapshot: object
    was_active: bool
    scene: object


def capture_runtime_sources():
    from infernux.lib import SceneManager
    manager = SceneManager.instance()
    active = manager.get_active_scene()
    return tuple(RuntimeSceneSource(scene.name, scene._capture_play_mode_snapshot(), scene is active, scene)
                 for scene in (manager.get_scene_at(index) for index in range(manager.scene_count)))


def current_world_runtime():
    from infernux.application import Application
    engine = Application._current_engine()
    if engine is None:
        return None
    from infernux.lib import SceneManager
    manager = SceneManager.instance()
    identity = int(manager._current_runtime_world_id)
    if identity == int(manager._authoring_runtime_world_id):
        return None
    return engine._world_runtimes[identity]


def component_world_id(component):
    native = getattr(component, '_cpp_component', None)
    if native is None:
        return 0
    owner = native.game_object
    scene = owner.scene if owner is not None else None
    return int(scene._runtime_world_id) if scene is not None else 0


def source_module_name(name):
    module = sys.modules.get(name)
    marker = getattr(module, '__infernux_source_name__', None)
    if marker is not None:
        return marker
    return name


def runtime_module_name(path, default):
    domain = current_world_runtime()
    return domain.modules.prefix + '.' + default if domain is not None else default


class HostModuleView(Mapping):
    """Shared interpreter modules, excluding source and foreign runtime realms."""

    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, name):
        module = sys.modules[name]
        if name == self.owner.prefix or name.startswith(self.owner.prefix + '.'):
            return module
        if name.startswith('_infernux_runtime_') or self.owner._path(name) is not None:
            raise KeyError(name)
        source = getattr(module, '__file__', None)
        if source and any(Path(source).resolve().is_relative_to(root) for root in self.owner.roots):
            raise KeyError(name)
        return module

    def __iter__(self):
        for name in tuple(sys.modules):
            try:
                self[name]
            except KeyError:
                continue
            yield name

    def __len__(self):
        return sum(1 for _name in self)


class ProjectModules:
    """Persistent private project import table, including helper dependency cycles."""

    def __init__(self, identity, serializable_catalog):
        from .project_context import get_project_script_roots
        self.identity = int(identity)
        self.serializable_catalog = serializable_catalog
        self.prefix = f'_infernux_runtime_{identity}'
        self.roots = tuple(Path(root).resolve() for root in get_project_script_roots())
        self.modules = {}
        self.specifications = {}
        self.published = []
        package = types.ModuleType(self.prefix)
        package.__path__ = []
        sys.modules[self.prefix] = package
        self.published.append(self.prefix)
        self.sys = types.ModuleType('sys')
        self.sys.__dict__.update(vars(sys))
        self.sys.modules = ChainMap(self.modules, HostModuleView(self))
        self.importlib = types.ModuleType('importlib')
        self.importlib.__dict__.update(vars(importlib))
        self.importlib.import_module = self.import_module

    def _path(self, name):
        if name in self.specifications:
            return self.specifications[name]
        relative = Path(*name.split('.'))
        roots = [(root, relative) for root in self.roots]
        parts = name.split('.')
        for count in range(len(parts) - 1, 0, -1):
            parent = self.specifications.get('.'.join(parts[:count]))
            if parent is not None:
                root = parent if parent.is_dir() else parent.parent
                roots.insert(0, (root, Path(*parts[count:])))
                break
        for root, relative in roots:
            for extension in ('.py', '.pyc'):
                package = root / relative / ('__init__' + extension)
                if package.is_file():
                    return package
            for extension in ('.py', '.pyc'):
                path = root / relative.with_suffix(extension)
                if path.is_file():
                    return path
            namespace = root / relative
            if namespace.is_dir():
                return namespace
        if name == 'random':
            return Path(importlib.util.find_spec(name).origin)
        return None

    def load_file(self, name, path):
        path = Path(path).resolve()
        self.specifications[name] = path
        parent = name.rpartition('.')[0]
        directory = path.parent
        if path.stem == '__init__':
            directory = directory.parent
        while parent:
            package = next((directory / ('__init__' + extension) for extension in ('.py', '.pyc')
                            if (directory / ('__init__' + extension)).is_file()), directory)
            self.specifications.setdefault(parent, package)
            parent = parent.rpartition('.')[0]
            directory = directory.parent
        return self.load(name)

    def load(self, name):
        published = sys.modules.get(self.prefix + '.' + name)
        if published is not None:
            self.modules[name] = published
            return published
        if name in self.modules:
            return self.modules[name]
        path = self._path(name)
        if path is None:
            return importlib.import_module(name)
        parent, _, child = name.rpartition('.')
        if parent:
            self.load(parent)
        qualified = self.prefix + '.' + name
        if path.is_dir():
            module = types.ModuleType(qualified)
            module.__package__ = qualified
            module.__path__ = [str(path)]
            module.__file__ = None
            code = None
            accepted_code = None
        else:
            package = path.stem == '__init__'
            spec = importlib.util.spec_from_file_location(qualified, str(path),
                submodule_search_locations=[str(path.parent)] if package else None)
            if spec is None or spec.loader is None:
                raise ImportError(f'cannot load project module: {path}')
            module = importlib.util.module_from_spec(spec)
            # The normal first-import/stage authority owns compilation and acceptance.
            # An existing accepted value never reads a changed disk file or its pyc.
            from .script_source import prepare_module_code
            accepted_code = prepare_module_code(name, str(path))
            code = accepted_code.code
        module.__infernux_source_name__ = name
        module.__infernux_serializable_catalog__ = self.serializable_catalog
        self.modules[name] = module
        sys.modules[qualified] = module
        self.published.append(qualified)
        if parent:
            setattr(self.modules[parent], child, module)
        namespace = dict(vars(builtins))
        namespace['__import__'] = self._import
        module.__dict__['__builtins__'] = namespace
        try:
            if code is not None:
                from infernux.components.value_ownership import value_world
                with value_world(self.identity):
                    exec(code, vars(module))
                if any(path.is_relative_to(root) for root in self.roots):
                    from infernux.components.script_loader import (
                        _class_literal_defaults, _declared_class_literals, _remember_module_literals,
                    )
                    from .script_closure_reload import remember_closure_code
                    _class_literal_defaults.update(_declared_class_literals((module,)))
                    _remember_module_literals(module)
                    remember_closure_code(module, code)
                if accepted_code is not None:
                    from .script_source import publish_module_code
                    module.__infernux_accepted_code__ = accepted_code
                    publish_module_code(str(path), accepted_code)
        except BaseException:
            del self.modules[name]
            sys.modules.pop(qualified, None)
            raise
        return module

    def import_module(self, name, package=None):
        if name.startswith('.'):
            package = self._canonical(package)
            name = importlib.util.resolve_name(name, package)
        return self.load(self._canonical(name))

    def _canonical(self, name):
        prefix = self.prefix + '.'
        return name[len(prefix):] if name.startswith(prefix) else name

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            package = self._canonical(globals['__package__'])
            name = importlib.util.resolve_name('.' * level + name, package)
        else:
            name = self._canonical(name)
        if name == 'sys':
            return self.sys
        if name == 'importlib' or name.startswith('importlib.'):
            if name == 'importlib':
                return self.importlib
            module = importlib.import_module(name)
            root_child = name.split('.')[1]
            setattr(self.importlib, root_child, getattr(importlib, root_child))
            return module if fromlist else self.importlib
        if self._path(name) is None and name not in self.modules:
            return builtins.__import__(name, globals, locals, fromlist, 0)
        module = self.load(name)
        if fromlist:
            for child in fromlist:
                if child != '*' and not hasattr(module, child) and self._path(name + '.' + child) is not None:
                    self.load(name + '.' + child)
            return module
        return self.load(name.split('.')[0])

    def close(self):
        for name in tuple(sys.modules):
            if name == self.prefix or name.startswith(self.prefix + '.'):
                sys.modules.pop(name, None)
        self.modules.clear()
        self.specifications.clear()
        self.published.clear()


class WorldRuntime:
    def __init__(self, identity):
        from infernux.components._component_lifecycle import RuntimeExecutionScheduler
        from .runtime_change_journal import RuntimeChangeJournal
        from infernux.components.registry import _author_type_registrations
        from infernux.components.serializable_object import _SHARED_SERIALIZABLE_REGISTRY
        self.identity = int(identity)
        self.initializing = True
        self.type_registrations = {key: value for key, value in _author_type_registrations.items()
                                   if not value.project_script}
        self.script_registrations = {}
        self.serializable_types = ChainMap({}, _SHARED_SERIALIZABLE_REGISTRY)
        self.journal = RuntimeChangeJournal()
        from infernux.scene import RuntimeSceneLoadState
        self.scene_loads = RuntimeSceneLoadState()
        self.data_assets = {}
        from .runtime_dispatch import initial_runtime_world_epoch
        self.epoch = initial_runtime_world_epoch(
            registration.component_type for registration in self.type_registrations.values()
            if registration.component_type is not None)
        from infernux.runtime_services import _author_update_callbacks
        self.updates = {phase: () for phase in _author_update_callbacks}
        from infernux.execution import ExecutionScope
        self.scope = ExecutionScope()
        self.scope._world_id = self.identity
        self.scheduler = RuntimeExecutionScheduler(name=f'world:{identity}', change_journal=self.journal,
                                                   native_bridge=True, world_id=self.identity,
                                                   runtime_updates=self.updates)
        self.modules = ProjectModules(self.identity, self.serializable_types.maps[0])

    def close(self):
        self.begin_retirement()
        from infernux.components._cds_bridge import retire_runtime_realm
        retire_runtime_realm(self.modules.prefix, self.epoch.descriptors)
        self.scheduler.unbind_native_bridge()
        self.scheduler.clear()
        self.modules.close()
        self.type_registrations.clear()
        self.script_registrations.clear()
        self.serializable_types.maps[0].clear()
        self.data_assets.clear()
        self.updates.clear()
        from .runtime_dispatch import RuntimeRevisionEpoch
        self.epoch = RuntimeRevisionEpoch(0, {})

    def begin_retirement(self):
        self.initializing = False
        transaction = self.scene_loads._active_scene_transaction
        if transaction is not None and not transaction.is_complete:
            transaction.cancel()
        service = self.scene_loads._runtime_scene_service
        if service is not None:
            service.cancel_pending_load()
        self.scope.close()
        for registrations in tuple(self.updates.values()):
            for registration in registrations:
                registration.close()


class RuntimeWorldInitializationCancelled(Exception):
    """A lifecycle callback cancelled initialization without a preparation failure."""


class RuntimeWorldSession:
    """One immutable initial template; native owns identity, order and membership."""

    def __init__(self, engine, snapshots, *, scene_service=None):
        from collections import deque
        from infernux.lib import SceneManager
        self.engine = engine
        self.snapshots = tuple(RuntimeSceneSource(item.name, item.snapshot, item.was_active, item.scene)
                               for item in snapshots)
        persistent = SceneManager.instance().get_runtime_persistent_scene()
        self.persistent_source = (RuntimeSceneSource(persistent.name, persistent._capture_play_mode_snapshot(),
                                                     False, persistent) if persistent is not None else None)
        self.sources = self.snapshots + ((self.persistent_source,) if self.persistent_source else ())
        # Template descriptors are gathered once for the entire Play, never from
        # live worlds. Each materialization still creates its own Python realm.
        self.records = tuple(item.snapshot._python_component_records() for item in self.sources)
        self.scene_service = scene_service
        self._preparations = deque()
        self.closed = False
        if engine._runtime_world_session is not None:
            raise RuntimeError('an initial-world template is already installed')
        self.identity = int(SceneManager.instance()._begin_runtime_world_management_session())
        engine._runtime_world_session = self
        # The existing owner selects this ordinary background source between
        # peer sources, repeats ready work while credit remains and retains its
        # foreground reserve. Callbacks never keep a closed session alive.
        import weakref
        reference = weakref.ref(self)
        from infernux.lib import _Infernux

        def describe():
            session = reference()
            return session._describe_preparations() if session is not None else _Infernux._FrameWorkFacts()

        def advance(gpu, token):
            session = reference()
            return session._advance_preparation(gpu, token) if session is not None else False

        self._source = engine._frame_work.add_source(None, describe, advance)

    def reserve(self):
        from infernux.lib import SceneManager
        if self.closed:
            raise RuntimeError('the Play session is closing')
        return SceneManager.instance()._reserve_runtime_world()

    def create(self, count=1):
        """Reserve the initial total before the first publication can call Awake."""
        if type(count) is not int or count < 1:
            raise ValueError('initial world count must be a positive integer')
        handles = tuple(self.reserve() for _ in range(count))
        self.request_create(handles[1:])
        handle = handles[0]
        for _ in self._prepare(handle, start_play=False):
            pass
        if self.closed or handle.destroy_requested or handle.retirement_outcome == 'cancelled':
            raise RuntimeWorldInitializationCancelled(handle.error)
        if handle.state != 'ready':
            raise RuntimeError(handle.error or 'initial world preparation failed')
        return int(handle.id)

    def initialization_current(self, handle, *, started=False):
        """Check the captured owner after a user lifecycle callback returns."""
        if (self.closed or self.engine._runtime_world_session is not self
                or handle.state != 'ready' or handle.destroy_requested
                or self.engine.get_native_engine().exit_requested):
            return False
        if started:
            from infernux.lib import SceneManager, _Infernux
            with _Infernux._RuntimeWorldScope(int(handle.id)):
                return SceneManager.instance().is_playing()
        return True

    def cancel_initialization(self, handle):
        from infernux.lib import SceneManager
        if handle.state not in {'preparing', 'ready', 'retiring'}:
            return
        handle.error = 'RuntimeWorld initialization was cancelled'
        handle.retirement_outcome = 'cancelled'
        if handle.state in {'preparing', 'ready'}:
            SceneManager.instance()._begin_runtime_world_retirement(handle.id)

    def request_create(self, handles):
        """Register an accepted batch in order, without materializing its graphs."""
        for handle in handles:
            if self.closed or handle.destroy_requested:
                self._cancel_reservation(handle, 'Play session closed' if self.closed else 'creation cancelled')
            else:
                self._preparations.append((handle, self._prepare(handle, start_play=True)))
        self.pump()

    def _cancel_reservation(self, handle, reason):
        from infernux.lib import SceneManager
        if handle.state == 'preparing':
            SceneManager.instance()._cancel_reserved_runtime_world(handle, 'cancelled', reason)

    def _prepare(self, handle, *, start_play):
        from infernux.lib import SceneManager, _Infernux
        from infernux.engine.component_restore import (
            _prepare_python_component_records, PreparedPythonComponentGraph,
            publish_prepared_scene_python_components,
        )
        manager = SceneManager.instance()
        identity = int(handle.id)
        tokens = []
        prepared = None
        materialized = False
        try:
            if self.closed or handle.destroy_requested:
                self._cancel_reservation(handle, 'creation cancelled')
                return
            manager._begin_reserved_runtime_world(handle)
            materialized = True
            domain = WorldRuntime(identity)
            self.engine._world_runtimes[identity] = domain
            yield
            with _Infernux._RuntimeWorldScope(identity):
                object_ids = set().union(*(set(record[0]) for record in self.records))
                native_types = set().union(*(set(record[1]) for record in self.records))
                descriptors = [item for record in self.records for item in record[2]]
                from infernux.components.value_ownership import value_world
                with value_world(identity):
                    prepared = _prepare_python_component_records(object_ids, native_types, descriptors,
                        self.engine.get_asset_database(), allow_missing_scripts=False)
            yield
            scenes = []
            object_map, component_map = {}, {}
            for item in self.sources:
                with _Infernux._RuntimeWorldScope(identity):
                    scene = (manager._create_preparing_persistent_scene() if item is self.persistent_source
                             else manager.create_scene(item.name))
                    token = scene._commit_play_mode_snapshot_retaining_world(item.snapshot, new_instance_ids=True)
                    if token is None:
                        raise RuntimeError(f'cannot clone Scene {item.name}')
                    tokens.append(token)
                    object_map.update(token.object_id_remap)
                    component_map.update(token.component_id_remap)
                    scenes.append(scene)
                    if item.was_active:
                        manager.set_active_scene(scene)
                yield
            with _Infernux._RuntimeWorldScope(identity):
                manager._remap_preparing_world_references(identity, component_map)
            yield
            attached = []
            cursor = 0
            for scene, record in zip(scenes, self.records):
                with _Infernux._RuntimeWorldScope(identity):
                    count = len(record[2])
                    graph = PreparedPythonComponentGraph(prepared.components[cursor:cursor + count])
                    cursor += count
                    attached.extend(publish_prepared_scene_python_components(scene, graph,
                        clear_registries=False, object_id_map=object_map, component_id_map=component_map,
                        defer_lifecycle=True))
                yield
            with _Infernux._RuntimeWorldScope(identity):
                prepared.consume()
            # No yield retains a native scope. No partially attached graph is
            # publicly accessible; all proxies exist before deserialization hooks.
            for _target, instance, _native in attached:
                with _Infernux._RuntimeWorldScope(identity):
                    instance._call_on_after_deserialize()
                yield
            with _Infernux._RuntimeWorldScope(identity):
                if self.scene_service is not None:
                    domain.scene_loads._runtime_scene_service = self.scene_service.clone_for_runtime(
                        {int(item.scene.world_id): scene for item, scene in zip(self.sources, scenes)})
                for token in tokens:
                    token.finalize()
                tokens.clear()
                domain.scheduler.bind_native_bridge(manager)
            yield
            # Publication, Awake, and initial Start form one existing synchronous
            # lifecycle quantum. Commands issued inside them are always posted.
            if self.closed or handle.destroy_requested:
                raise GeneratorExit
            paused = self._is_paused()
            with _Infernux._RuntimeWorldScope(identity):
                if not manager._publish_runtime_world(identity):
                    handle.error = 'RuntimeWorld publication was cancelled'
                    handle.retirement_outcome = 'cancelled'
                    manager._begin_runtime_world_retirement(identity)
                    return
                if not self.initialization_current(handle):
                    self.cancel_initialization(handle)
                    return
                if start_play:
                    manager.play()
                    if not self.initialization_current(handle, started=True):
                        self.cancel_initialization(handle)
                        return
                    if paused:
                        manager.pause()
                    from infernux.components.component import InxComponent
                    from infernux.engine.startup_warmup import run_component_warmups
                    from infernux.engine.project_context import get_project_root
                    run_component_warmups(
                        (component for values in InxComponent._active_instances.values() for component in values
                         if component_world_id(component) == identity),
                        scope='runtime-world', project_path=get_project_root(),
                        can_continue=lambda: self.initialization_current(handle, started=True))
                    if not self.initialization_current(handle, started=True):
                        self.cancel_initialization(handle)
                        return
            if handle.pending_primary:
                self.apply_primary(handle)
            if start_play:
                domain.initializing = False
        except BaseException as error:
            if materialized:
                with _Infernux._RuntimeWorldScope(identity):
                    for token in reversed(tokens):
                        token.rollback()
                    if prepared is not None:
                        prepared.discard()
                handle.error = ('creation cancelled' if isinstance(error, GeneratorExit) else str(error))
                handle.retirement_outcome = 'cancelled' if isinstance(error, GeneratorExit) else 'failed'
                manager._begin_runtime_world_retirement(identity)
            else:
                manager._cancel_reserved_runtime_world(handle, 'failed', str(error))
            if not isinstance(error, GeneratorExit):
                from infernux.debug import Debug
                Debug.log_error(f'World {identity} preparation failed: {handle.error}')
            if isinstance(error, GeneratorExit):
                raise

    def _is_paused(self):
        player = self.engine._player_runtime
        if player is not None:
            return player.is_paused
        manager = self.engine._play_mode_manager
        return manager is not None and manager.is_paused

    def finish_initialization(self):
        """Seal the cold world's first Awake/Start/warmup declaration window."""
        from infernux.lib import SceneManager
        for identity in SceneManager.instance()._runtime_world_ids:
            self.engine._world_runtimes[int(identity)].initializing = False

    def apply_primary(self, handle):
        from infernux.lib import SceneManager
        if self.closed or handle.destroy_requested:
            return
        if handle.state == 'ready':
            SceneManager.instance()._set_primary_runtime_world(handle.id)

    def destroy(self, handle):
        from infernux.lib import SceneManager
        if self.closed or handle.state in {'failed', 'cancelled', 'destroyed', 'retiring'}:
            return
        for candidate, preparation in tuple(self._preparations):
            if candidate.id == handle.id:
                self._preparations.remove((candidate, preparation))
                # Closing a suspended generator rolls back that world's staging
                # tokens and transitions it to retirement without touching peers.
                if handle.id in self.engine._world_runtimes:
                    preparation.close()
                else:
                    self._cancel_reservation(handle, 'creation cancelled')
                return
        if handle.state == 'preparing':
            self._cancel_reservation(handle, 'creation cancelled')
        else:
            SceneManager.instance()._begin_runtime_world_retirement(handle.id)

    def _describe_preparations(self):
        from infernux.lib import _Infernux
        pending = not self.closed and bool(self._preparations)
        candidate = (_Infernux._FrameWorkCandidate(self._preparations[0][0].id, 0, True)
                     if pending else _Infernux._FrameWorkCandidate())
        return _Infernux._FrameWorkFacts(host=candidate, pending=pending)

    def _advance_preparation(self, gpu, token):
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread, _assert_world_dispatch_safe_point
        assert_runtime_owner_thread()
        work = self.engine._frame_work
        if (self.closed or gpu or not self._preparations or not work.active or not work.accepting
                or work.remaining(background=True) <= 0 or self._preparations[0][0].id != token):
            return False
        handle, preparation = self._preparations[0]
        domain = self.engine._world_runtimes.get(int(handle.id))
        if domain is not None:
            _assert_world_dispatch_safe_point(domain)
        self._preparations.popleft()
        if handle.destroy_requested:
            if handle.id in self.engine._world_runtimes:
                preparation.close()
            else:
                self._cancel_reservation(handle, 'creation cancelled')
            return True
        # FrameWorkOwner accounts this complete synchronous quantum, then
        # chooses a peer source before it can choose our next world again.
        try:
            next(preparation)
        except StopIteration:
            pass
        else:
            self._preparations.append((handle, preparation))
        return True

    def pump(self):
        """Advertise ready preparation to the existing Host work driver."""
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        assert_runtime_owner_thread()
        if not self.closed and self._preparations:
            self.engine._frame_work.wake(self._source)

    def close(self):
        from infernux.lib import SceneManager
        if self.closed:
            return
        self.closed = True
        self.engine._frame_work.remove_source(self._source)
        manager = SceneManager.instance()
        for handle, preparation in self._preparations:
            if handle.id in self.engine._world_runtimes:
                preparation.close()
            else:
                self._cancel_reservation(handle, 'Play session closed')
        self._preparations.clear()
        for handle in manager._managed_runtime_world_handles:
            if handle.state == 'preparing':
                self._cancel_reservation(handle, 'Play session closed')
            elif handle.state == 'ready':
                manager._begin_runtime_world_retirement(handle.id)
        manager._collect_retired_runtime_worlds()
        manager._end_runtime_world_management_session()
        if self.engine._runtime_world_session is self:
            self.engine._runtime_world_session = None
        self.snapshots = self.sources = self.records = ()
        self.persistent_source = self.scene_service = None

class WorldScriptPublication:
    """Compose existing per-owner reload transactions under one source publication."""

    def __init__(self, transactions, *, deletion=False):
        self.transactions = tuple(transactions)
        self.deletion = deletion
        self.committed = False
        self.rolled_back = False
        self._result = 0 if deletion else {}

    @property
    def had_live_targets(self):
        return any(transaction.had_live_targets for _identity, transaction in self.transactions)

    @property
    def recovered_count(self):
        return sum(getattr(transaction, 'recovered_count', 0) for _identity, transaction in self.transactions)

    def commit(self):
        from infernux.lib import _Infernux
        if self.rolled_back:
            raise RuntimeError('script publication has been rolled back')
        if self.committed:
            return self._result
        try:
            for identity, transaction in self.transactions:
                with _Infernux._RuntimeWorldScope(identity):
                    result = transaction.commit()
                    if self.deletion:
                        self._result += result
                    else:
                        self._result.update(result)
            self.committed = True
            return self._result
        except BaseException:
            self.rollback()
            raise

    def rollback(self):
        from infernux.lib import _Infernux
        if self.rolled_back:
            return
        errors = []
        for identity, transaction in reversed(self.transactions):
            try:
                with _Infernux._RuntimeWorldScope(identity):
                    transaction.rollback()
            except Exception as error:
                errors.append(error)
        self.committed = False
        self.rolled_back = True
        if errors:
            raise RuntimeError('failed to roll back World script publication') from errors[0]

    def finalize(self):
        from infernux.lib import _Infernux
        for identity, transaction in self.transactions:
            with _Infernux._RuntimeWorldScope(identity):
                finalize = getattr(transaction, 'finalize', None)
                if finalize is not None:
                    finalize()
