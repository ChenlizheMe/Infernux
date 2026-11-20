"""The single owner of script invalidation, construction and activation."""

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import sys
import traceback

from infernux.engine.path_utils import is_path_within, path_key, resolved_path
from infernux.engine.project_context import get_script_module_name
from infernux.engine.script_dependency_graph import ScriptDependencyGraph
from .modules import (
    ModuleBatch, RuntimeModuleBatch, remove_module,
    observe_runtime_imports, stop_observing_runtime_imports, is_native_module,
)
from .owners import ReloadOwner
from .source import SourceFrontend
from .adapters import ReloadAdapters


@dataclass(frozen=True, slots=True)
class ReloadState:
    path: str
    status: str
    error: str = ""


@dataclass(eq=False)
class _Publication:
    owner: ReloadOwner
    batch: ModuleBatch
    prepared: list = field(default_factory=list)
    prepared_names: set = field(default_factory=set)
    initialized: int = 0
    error: Exception | None = None
    retired: bool = False


class ReloadCoordinator:
    def __init__(self, project_path, *, adapter, on_publish=None):
        self.graph = ScriptDependencyGraph(project_path)
        self.frontend = SourceFrontend()
        self.adapter = ReloadAdapters(adapter)
        self.on_publish = on_publish
        self.states = {}
        self._pending = {}
        self._ready = {}
        self._owners = {}
        self._unavailable = set()
        self._restart_only = set()
        self._native_sources = {}
        self._native_checks = None
        self._source_inputs = set()
        self._initialization_failures = {}
        self._wave = None
        self._closed = False
        sys.meta_path.insert(0, self)
        observe_runtime_imports(self)

    def register_source_scope(self, path, module_name, *, restart_only=False, loaded_module=None):
        """Declare one exact engine source and its real module identity.

        Reconstructible scopes use ordinary adapters and this coordinator's
        loader. Core/native scopes only retire and report restart_required.
        An already loaded scope may be adopted at startup; its adapter must
        attach the returned owner to its existing entry points.
        """
        path = resolved_path(path)
        if loaded_module is not None and (
            sys.modules.get(module_name) is not loaded_module
            or path_key(getattr(loaded_module, "__file__", "")) != path_key(path)
        ):
            raise ValueError(f"Loaded module does not belong to the registered source: {module_name}")
        if not restart_only and Path(path).suffix != ".py":
            raise ValueError("Native source scopes require restart_only=True")
        self.graph.register_source(path, module_name)
        if restart_only:
            self._restart_only.add(path)
            if loaded_module is not None and is_native_module(loaded_module):
                self._native_sources[path] = hashlib.sha256(Path(path).read_bytes()).digest()
                if self._native_checks is not None:
                    self._native_checks[path] = False
        if loaded_module is None:
            return None
        self.graph.upsert(path, b"" if restart_only else Path(path).read_bytes())
        self.states[path] = ReloadState(path, "active")
        owner = ReloadOwner((path,), coordinator=self)
        owner.active = True
        owner.bind_modules((module_name,))
        self._owners[owner] = None
        return owner

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self._unavailable:
            raise ImportError(f"Current script is unavailable: {fullname}")
        from .owners import current_owner
        owner = current_owner()
        if owner is not None and owner.active and owner.coordinator is self:
            import importlib.machinery
            from .source import SourceResult
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if (spec is not None and spec.origin and spec.origin.endswith(".py")
                    and self.graph.owns_path(spec.origin)):
                source_path = resolved_path(spec.origin)
                source = Path(source_path).read_bytes()
                result = SourceResult(source_path, 0, source, compile(source, source_path, "exec"), None)
                self.graph.set_module_name(source_path, fullname)
                batch = RuntimeModuleBatch({fullname: result}, coordinator=self, owner=owner)
                return batch.find_spec(fullname, path, target)
        return None

    def record_import(self, importer, module, owner, *, verify_native=False):
        path = getattr(module, "__file__", None)
        if not path or not self.graph.owns_path(path):
            return
        if is_native_module(module):
            path = resolved_path(path)
            state = self.states.get(path)
            if path not in self._restart_only:
                self.register_source_scope(path, module.__name__, restart_only=True, loaded_module=module)
            elif state is None or state.status != "active":
                raise ImportError(f"Loaded native extension changed; restart the editor: {path}")
            if verify_native and self._native_source_changed(path):
                error = ImportError(f"Loaded native extension changed; restart the editor: {path}")
                self.queue_initialization_failure(path, error)
                raise error
        sources = (importer,) if self.graph.module_for_name(importer) is not None else tuple(
            record.id.module_name for path in owner.paths
            if (record := self.graph.module_for_path(path)) is not None
        )
        self.graph.record_runtime_imports((name, module.__name__) for name in sources)

    def _native_source_changed(self, path):
        if self._native_checks is not None and path in self._native_checks:
            return self._native_checks[path]
        try:
            changed = hashlib.sha256(Path(path).read_bytes()).digest() != self._native_sources[path]
        except OSError:
            changed = True
        if self._native_checks is not None:
            self._native_checks[path] = changed
        return changed

    def retire_native_sources(self, paths, *, force=False):
        paths = tuple(resolved_path(path) for path in paths
                      if resolved_path(path) in self._native_sources
                      and (force or self._native_source_changed(resolved_path(path))))
        if paths:
            self.request((), removed=paths)
        return bool(paths)

    def record_owner_dependency(self, consumer, provider):
        consumer, provider = consumer.source_owner, provider.source_owner
        if consumer is provider or consumer.closed or provider.closed:
            return
        identities = (consumer.paths, provider.paths)
        if consumer._dependencies.get(provider) == identities:
            return
        sources = [record.id.module_name for path in consumer.paths
                   if (record := self.graph.module_for_path(path)) is not None]
        targets = [record.id.module_name for path in provider.paths
                   if (record := self.graph.module_for_path(path)) is not None]
        self.graph.record_runtime_imports((source, target) for source in sources for target in targets)
        consumer._dependencies[provider] = identities

    def is_current_candidate(self, name):
        candidate = self._wave.get(name) if self._wave is not None else None
        return bool(candidate is not None and candidate.error is None
                    and candidate.batch.error is None and not candidate.owner.closed
                    and name in candidate.batch.modules)

    def admits_candidate_service(self, consumer, provider):
        """Only controlled current publication code can use a staged provider."""
        if self._wave is None or consumer is None:
            return False
        owners = {item.owner for item in self._wave.values()
                  if item.error is None and item.batch.error is None and not item.owner.closed}
        return consumer.source_owner in owners and provider.source_owner in owners

    def candidate_batch(self, name, *, consumer):
        candidate = self._wave.get(name) if self._wave is not None else None
        if candidate is None:
            return None
        consumer._remember_attempt(name)
        if consumer.owner is not None:
            # Record attempted dynamic imports too: a broken provider must
            # recover its consumer when that provider's source is repaired.
            self.record_owner_dependency(consumer.owner, candidate.owner)
        if candidate.error is not None or candidate.batch.error is not None:
            raise ImportError(f"Current dependency is unavailable: {name}") from (
                candidate.error or candidate.batch.error)
        return candidate.batch

    def register_owned_library(self, owner, root):
        """A preload owns all Python sources it hands to its services/processes."""
        from .modules import _constructing
        from infernux.components.script_loader import set_script_error, _clear_script_error

        owner = owner.source_owner
        owner.library_roots.add(root)
        batch = _constructing.get()
        batches = ({item.batch for item in self._wave.values()} if self._wave is not None
                   else {batch} if batch is not None else set())
        captured = {item.path: item.source for current in batches for item in current.sources.values()}
        executing = {current.sources[name].path for current in batches for name in current.modules}
        consumers = tuple(record.id.module_name for path in owner.paths
                          if (record := self.graph.module_for_path(path)) is not None)
        sources, errors = {}, []
        for file in sorted(Path(root).rglob("*.py")):
            if "__pycache__" in file.parts or not is_path_within(file, root):
                continue
            path = resolved_path(file)
            state = self.states.get(path)
            source = b""
            try:
                source = captured[path] if path in captured else file.read_bytes()
                if (path not in executing and (self.graph.published_source(path) != source
                        or state is None or state.status != "active")):
                    compile(source, path, "exec", dont_inherit=True)
                if path not in executing and not (state is not None and state.status == "active" and path not in self._source_inputs):
                    self._source_inputs.add(path)
                    self.states[path] = ReloadState(path, "active")
                    _clear_script_error(path)
                sources[path] = source
            except Exception as exc:
                sources[path] = source
                self._source_inputs.add(path)
                message = "".join(traceback.format_exception(exc))
                self.states[path] = ReloadState(path, "failed", message)
                set_script_error(path, message)
                errors.append(exc)
        if sources:
            changed = {path: source for path, source in sources.items()
                       if self.graph.published_source(path) != source}
            if changed:
                self.graph.commit_transaction(self.graph.stage_transaction(changed))
            self.adapter.record_dependencies(self.graph)
            targets = [self.graph.module_for_path(path).id.module_name for path in sources]
            self.graph.record_runtime_imports((source, target) for source in consumers for target in targets)
            owner.paths = owner.paths.union(sources)
        if errors:
            raise ExceptionGroup("Owned Python library is unavailable", errors)

    def adopt_runtime_imports(self, batch, owner):
        """A normal dynamic import extends the admitting source's lifetime."""
        self.graph.commit_transaction(self.graph.stage_transaction(
            {result.path: result.source for result in batch.sources.values()}
        ))
        self.adapter.record_dependencies(self.graph)
        self.graph.record_runtime_imports(batch.runtime_imports)
        owner.source_owner.paths = owner.source_owner.paths.union(result.path for result in batch.sources.values())
        owner.bind_modules(batch.modules)
        for result in batch.sources.values():
            self.states[result.path] = ReloadState(result.path, "active")

    @property
    def pending_count(self):
        return len(self._pending.keys() | self._initialization_failures.keys())

    def queue_initialization_failure(self, path, exception):
        """Close callback admission now; retire the source at the next safe point.

        Native lifecycle callbacks run inside a dispatch epoch, so they cannot
        replace components or publish types. The owner is the identity of the
        failing implementation; an accepted source request supersedes its queue.
        """
        path = resolved_path(path)
        affected = {path}
        while True:
            previous = set(affected)
            affected.update(self.graph.module_for_path(item.path_key).source_path
                            for item in self.graph.affected_closure(affected))
            owners = tuple(self._owners)
            if self._wave is not None:
                owners += tuple({item.owner for item in self._wave.values()})
            for owner in owners:
                if owner.paths.intersection(affected):
                    affected.update(owner.paths)
                    owner.stop_admission()
            if affected == previous:
                break
        from infernux.components.script_loader import set_script_error
        message = "".join(traceback.format_exception(exception))
        for source in affected:
            self._initialization_failures[source] = exception
            self.states[source] = ReloadState(source, "restart_required" if source in self._restart_only else "failed", message)
            set_script_error(source, message)
        return tuple(sorted(affected))

    def _drain_initialization_failures(self):
        if not self._initialization_failures:
            return 0
        failures = dict(self._initialization_failures)
        # Existing retirement owns every module/type/domain cleanup. Files still
        # present on disk remain graph inputs, and are never recompiled here.
        affected = self.request((), removed=tuple(failures))
        from infernux.components.script_loader import set_script_error
        for path in affected:
            exception = failures.get(path, next(iter(failures.values())))
            message = "".join(traceback.format_exception(exception))
            self.states[path] = ReloadState(path, "restart_required" if path in self._restart_only else "failed", message)
            set_script_error(path, message)
            self.adapter.fail((path,), message)
        return len(affected)

    def request(self, paths, *, snapshots=None, removed=()):
        """Owner safe point: invalidate once, then queue exact current bytes."""
        from infernux.engine.runtime_dispatch import assert_runtime_dispatch_safe_point
        from infernux.components.registry import unregister_component_script
        from infernux.components.script_loader import set_script_error

        assert_runtime_dispatch_safe_point()
        paths = set(resolved_path(path) for path in paths)
        # Package initializers are real dependencies, not fabricated namespaces.
        for path in tuple(paths):
            folder = Path(path).parent
            if Path(path).name == "__init__.py":
                folder = folder.parent
            while get_script_module_name(str(folder / "__init__.py"), self.graph.project_root):
                parent = folder / "__init__.py"
                if parent.is_file():
                    parent_path = resolved_path(parent)
                    state = self.states.get(parent_path)
                    try:
                        changed_parent = self.graph.published_source(parent_path) != parent.read_bytes()
                    except OSError:
                        changed_parent = True
                    if state is None or state.status != "active" or changed_parent:
                        paths.add(parent_path)
                folder = folder.parent
        paths = tuple(sorted(paths))
        removed = tuple(resolved_path(path) for path in removed)
        affected = {
            item.path_key: self.graph.module_for_path(item.path_key).source_path
            for item in self.graph.affected_closure((*paths, *removed))
        }
        sources = {}
        for path in paths:
            try:
                source = (b"" if path in self._restart_only else
                          snapshots[path] if snapshots is not None and path in snapshots else Path(path).read_bytes())
                if isinstance(source, Exception):
                    raise source
                if isinstance(source, str):
                    source = source.encode("utf-8")
                sources[path] = source
            except OSError as exc:
                sources[path] = exc
                self.states[path] = ReloadState(path, "failed", str(exc))
                set_script_error(path, str(exc))
            affected[path_key(path)] = path
        for path in removed:
            affected[path_key(path)] = path
        for path in (*paths, *removed):
            name = self.graph.module_name_for_path(path)
            existing = self.graph.module_for_name(name) if name else None
            if existing is not None and path_key(existing.source_path) != path_key(path):
                affected[path_key(existing.source_path)] = existing.source_path
        for owner in self._owners:
            if any(is_path_within(path, root) for path in (*paths, *removed) for root in owner.library_roots):
                for path in owner.paths:
                    affected[path_key(path)] = path
                self._source_inputs.update(path for path in (*paths, *removed)
                                           if self.graph.module_for_path(path) is None)
        upserts = {path: source for path, source in sources.items() if isinstance(source, bytes)}
        graph_error = None
        if upserts or removed:
            try:
                self.graph.commit_transaction(self.graph.stage_transaction(upserts, removals=tuple(path for path in removed if not Path(path).is_file())))
                self.adapter.record_dependencies(self.graph)
            except Exception as exc:
                graph_error = "".join(traceback.format_exception(exc))
        for item in self.graph.affected_closure(paths):
            affected[item.path_key] = self.graph.module_for_path(item.path_key).source_path
        # Imports constructed together share a lifetime (including libraries
        # imported by preload()). Retire their entire source group together.
        expanded = True
        while expanded:
            expanded = False
            for owner in self._owners:
                if owner.paths.intersection(affected.values()):
                    for path in owner.paths:
                        if path_key(path) not in affected:
                            affected[path_key(path)] = path
                            expanded = True
            for item in self.graph.affected_closure(tuple(affected.values())):
                if item.path_key not in affected:
                    affected[item.path_key] = self.graph.module_for_path(item.path_key).source_path
                    expanded = True
        affected_paths = tuple(affected.values())
        for path in affected_paths:
            self._initialization_failures.pop(path, None)
        for owner in reversed(tuple(self._owners)):
            if owner.paths.intersection(affected_paths):
                try:
                    owner.close()
                except Exception as exc:
                    from infernux.debug import Debug

                    Debug.log_exception(exc)
                self._owners.pop(owner, None)
        retirement_error = graph_error
        try:
            self.adapter.retire(affected_paths)
        except Exception as exc:
            retirement_error = (retirement_error or "") + "".join(traceback.format_exception(exc))
            from infernux.debug import Debug

            Debug.log_error(retirement_error)
        for path in affected_paths:
            record = self.graph.module_for_path(path)
            name = record.id.module_name if record is not None else self.graph.module_name_for_path(path)
            if name:
                self._unavailable.add(name)
                if path not in self._restart_only:
                    remove_module(name)
                else:
                    from infernux.components.serializable_object import _publish_serializable_types
                    _publish_serializable_types({}, {name})
            unregister_component_script(path)
            from infernux.renderstack.render_effect_compiler import _publish_effect_features

            _publish_effect_features((), source_key=path_key(path))
            self._ready.pop(path, None)
            if retirement_error is not None or path in removed or path in self._restart_only:
                self.frontend.cancel(path)
                self._pending.pop(path, None)
                restart = path in self._restart_only
                message = retirement_error or ("Loaded core/native source changed; restart the editor" if restart else "Script source was deleted")
                self.states[path] = ReloadState(path, "restart_required" if restart else "failed", message)
                set_script_error(path, message)
            else:
                self.states[path] = ReloadState(path, "pending")
                source = sources.get(path, snapshots.get(path) if snapshots is not None else None)
                self._pending[path] = self.frontend.submit(path, source)
        return affected_paths

    def compile_pending(self):
        return self.frontend.compile_pending()

    def drain(self):
        if self._wave is not None:
            raise RuntimeError("Cannot recursively publish a source wave")
        failed_initializations = self._drain_initialization_failures()
        if not self._pending:
            return failed_initializations
        self._ready.update(self.frontend.take_ready())
        # Bindings form one safe-point barrier. Compilation failures remain
        # isolated by dependency; they do not make unrelated candidates fail.
        if any(path not in self._ready for path in self._pending):
            return failed_initializations
        accepted = tuple(self._pending)
        publications = []
        wave = {}
        for records in self.graph.ordered_batches(self._pending):
            paths = tuple(record.source_path for record in records)
            owner = ReloadOwner(paths, coordinator=self)
            batch = ModuleBatch(
                {record.id.module_name: self._ready[record.source_path] for record in records},
                unavailable=self._unavailable, coordinator=self, owner=owner,
            )
            candidate = _Publication(owner, batch)
            candidate.error = next((result.error for result in batch.sources.values()
                                    if result.error is not None), None)
            publications.append(candidate)
            wave.update((name, candidate) for name in batch.sources)
        self._wave = wave
        self._native_checks = {}
        try:
            for candidate in publications:
                self._publication_call(candidate, lambda c=candidate: self.adapter.validate_sources(c.owner.paths))
                for path in candidate.owner.paths:
                    for edge in self.graph.dependencies_of(path):
                        if edge.target is None or edge.target.module_name in wave:
                            continue
                        dependency = self.graph.module_for_name(edge.target.module_name)
                        state = self.states.get(dependency.source_path)
                        if state is not None and state.status != "active":
                            candidate.error = candidate.error or ImportError(
                                f"Dependency {dependency.source_path} is unavailable: {state.error}")
            self._retire_failed_publications(publications)
            for candidate in publications:
                names = tuple(name for name, result in candidate.batch.sources.items()
                              if result.path not in self._source_inputs)
                self._publication_call(candidate, lambda c=candidate, names=names: c.batch.load(names))
                self._retire_failed_publications(publications)

            from infernux.components.script_loader import _clear_script_error
            activated, notified = {}, set()
            while True:
                self._activate_publications(publications, activated)
                # Lifecycle hooks can import another source after preparation.
                # That source must activate before any completion notification.
                if any(candidate.error is None and candidate.batch.modules.keys() - candidate.prepared_names
                       for candidate in publications):
                    continue
                for candidate in publications:
                    if candidate.error is not None:
                        continue
                    sources = tuple((name, result) for name, result in candidate.batch.sources.items()
                                    if (candidate, result.path) not in notified)
                    if not sources:
                        continue
                    self._publication_call(candidate, candidate.batch.publish)
                    self._retire_failed_publications(publications)
                    if (candidate.error is not None
                            or candidate.batch.modules.keys() - candidate.prepared_names):
                        continue
                    if candidate.batch.modules:
                        self._owners[candidate.owner] = None
                    else:
                        candidate.owner.close()
                    def complete(c=candidate, sources=sources):
                        for name, result in sources:
                            if name in c.batch.modules:
                                self._source_inputs.discard(result.path)
                            self._unavailable.discard(name)
                            self.states[result.path] = ReloadState(result.path, "active")
                            _clear_script_error(result.path)
                        # One owner/import scope and reconciliation per known
                        # source group, even when the group has many files.
                        for _name, result in sources:
                            notified.add((c, result.path))
                            if self.on_publish is not None:
                                self.on_publish(result.path)
                    self._publication_call(candidate, complete)
                    self._retire_failed_publications(publications)
                # Notifications may import new current sources. Complete their
                # normal lifecycle before notifying them, without replaying peers.
                if any(candidate.error is None and (
                           candidate.batch.modules.keys() - candidate.prepared_names
                           or any((candidate, result.path) not in notified
                                  for result in candidate.batch.sources.values()))
                       for candidate in publications):
                    continue
                break
            from infernux.host.queries import emit_signal
            emit_signal("operations.changed")
        finally:
            self._wave = None
            self._native_checks = None
            # Unreadable sources have no graph record, but still finish this
            # accepted request with a failed current-source result.
            for path in accepted:
                if self.graph.module_for_path(path) is None:
                    error = self._ready[path].error or ImportError(f"No current script identity: {path}")
                    self._report_failed_sources((path,), error)
                self._pending.pop(path, None)
                self._ready.pop(path, None)
        return failed_initializations + len(accepted)

    def _activate_publications(self, publications, activated):
        # Bind all current declarations before initialization. A later import
        # joins the same barrier; only newly prepared groups repeat these phases.
        self._prepare_publications(publications)
        while True:
            initialized = False
            for candidate in self._ordered_publications(publications):
                while candidate.error is None and candidate.initialized < len(candidate.prepared):
                    prepared = candidate.prepared[candidate.initialized]
                    candidate.initialized += 1
                    self._publication_call(candidate, lambda c=candidate, p=prepared:
                                           self.adapter.initialize(p, c.owner))
                    self._retire_failed_publications(publications)
                    initialized = True
            added = self._prepare_publications(publications)
            if not initialized and not added:
                break
        for phase in ("after_deserialize", "awake", "start"):
            for candidate in self._ordered_publications(publications):
                for prepared in candidate.prepared[activated.get(candidate, 0):]:
                    self._publication_call(candidate, lambda c=candidate, p=prepared:
                                           self.adapter.phase(p, c.owner, phase))
                    self._retire_failed_publications(publications)
        # All owners receive admission before enable/view hooks observe peers.
        for candidate in publications:
            if candidate.error is None and not candidate.owner.closed:
                candidate.owner.start_admission()
        for candidate in publications:
            for prepared in candidate.prepared[activated.get(candidate, 0):]:
                self._publication_call(candidate, lambda c=candidate, p=prepared:
                                       self.adapter.phase(p, c.owner, "admit"))
        for candidate in self._ordered_publications(publications):
            for prepared in candidate.prepared[activated.get(candidate, 0):]:
                self._publication_call(candidate, lambda p=prepared: self.adapter.activate(p))
                self._retire_failed_publications(publications)
            activated[candidate] = len(candidate.prepared)

    def _ordered_publications(self, publications):
        by_path = {path: candidate for candidate in publications for path in candidate.owner.paths}
        ordered = []
        for records in self.graph.ordered_batches(by_path):
            for record in records:
                candidate = by_path[record.source_path]
                if candidate not in ordered:
                    ordered.append(candidate)
        return ordered

    def _publication_call(self, candidate, callback):
        if candidate.error is not None:
            return None
        try:
            with candidate.owner.enter(), candidate.batch.imports():
                result = callback()
        except Exception as exc:
            candidate.error = exc
            result = None
        try:
            batch = candidate.batch
            candidate.owner.paths = candidate.owner.paths.union(item.path for item in batch.sources.values())
            candidate.owner.bind_modules(batch.modules)
            imported = {item.path: item.source for item in batch.sources.values()
                        if self.graph.published_source(item.path) != item.source and item.source is not None}
            if imported:
                self.graph.commit_transaction(self.graph.stage_transaction(imported))
                self.adapter.record_dependencies(self.graph)
            self.graph.record_runtime_imports(batch.runtime_imports)
            for name in batch.sources:
                self._wave[name] = candidate
            candidate.error = candidate.error or batch.error
            candidate.error = candidate.error or next((self._initialization_failures[path]
                for path in candidate.owner.paths if path in self._initialization_failures), None)
        except Exception as exc:
            candidate.error = candidate.error or exc
        return result

    def _prepare_publications(self, publications):
        pending = [(candidate, {name: module for name, module in candidate.batch.modules.items()
                               if name not in candidate.prepared_names})
                   for candidate in publications if candidate.error is None]
        pending = [(candidate, modules) for candidate, modules in pending if modules]
        for candidate, _modules in pending:
            def publish(c=candidate):
                c.batch.publish_components(self.adapter.guid_for_path)
                c.batch.publish()
            self._publication_call(candidate, publish)
        self._retire_failed_publications(publications)
        bindings = []
        for candidate, modules in pending:
            prepared = self._publication_call(candidate, lambda c=candidate, m=modules:
                                              self.adapter.prepare(m, c.owner))
            candidate.prepared_names.update(modules)
            if candidate.error is None:
                candidate.prepared.append(prepared)
                bindings.append((candidate, prepared))
        self._retire_failed_publications(publications)
        for candidate, prepared in bindings:
            self._publication_call(candidate, lambda p=prepared: self.adapter.bind(p))
            self._retire_failed_publications(publications)
        return bool(pending)

    def _retire_failed_publications(self, publications):
        for candidate in publications:
            candidate.error = candidate.error or candidate.batch.error
        changed = True
        while changed:
            changed = False
            for candidate in publications:
                if candidate.error is None:
                    continue
                affected = {self.graph.module_for_path(item.path_key).source_path
                            for item in self.graph.affected_closure(candidate.owner.paths)}
                for dependent in publications:
                    if dependent.error is None and dependent.owner.paths.intersection(affected):
                        dependent.error = ImportError("Current dependency failed to initialize")
                        dependent.error.__cause__ = candidate.error
                        dependent.owner.stop_admission()
                        changed = True
        for candidate in publications:
            if candidate.error is None or candidate.retired:
                continue
            candidate.retired = True
            errors = [candidate.error]
            try:
                candidate.owner.close()
            except Exception as exc:
                errors.append(exc)
            self._owners.pop(candidate.owner, None)
            candidate.batch.discard()
            try:
                self.adapter.retire(candidate.owner.paths)
            except Exception as exc:
                errors.append(exc)
            from infernux.components.registry import unregister_component_script
            from infernux.renderstack.render_effect_compiler import _publish_effect_features
            for name, result in candidate.batch.sources.items():
                try:
                    unregister_component_script(result.path)
                    _publish_effect_features((), source_key=path_key(result.path))
                except Exception as exc:
                    errors.append(exc)
                self._unavailable.add(name)
                self._initialization_failures.pop(result.path, None)
            error = errors[0] if len(errors) == 1 else ExceptionGroup("Current source publication failed", errors)
            self._report_failed_sources(candidate.owner.paths, error)

    def _report_failed_sources(self, paths, error):
        from infernux.components.script_loader import set_script_error
        from infernux.debug import Debug

        message = "".join(traceback.format_exception(error))
        self.adapter.fail(paths, message)
        for path in paths:
            self.states[path] = ReloadState(path, "failed", message)
            set_script_error(path, message)
        Debug.log_error(message, source_file=next(iter(paths)))

    def ensure_ready(self, paths, *, snapshots=None):
        paths = tuple(resolved_path(path) for path in paths)
        if self._wave is not None:
            # Type resolution during deserialization consumes the exact staged
            # modules. It cannot retire/recompile a second time inside this wave.
            for path in paths:
                name = self.graph.module_name_for_path(path)
                candidate = self._wave.get(name)
                if candidate is None:
                    state = self.states.get(path)
                    if state is not None and state.status == "active":
                        continue
                    raise RuntimeError(f"Source was not prepared in this publication: {path}")
                result = candidate.batch.sources[name]
                source = snapshots.get(path, result.source) if snapshots is not None else result.source
                if isinstance(source, str):
                    source = source.encode("utf-8")
                if candidate.error is not None or result.error is not None or source != result.source:
                    raise RuntimeError(f"Current source is unavailable in this publication: {path}") from (
                        candidate.error or result.error)
                if name not in candidate.batch.modules:
                    with candidate.owner.enter(), candidate.batch.imports():
                        candidate.batch.load((name,))
            return
        changed = []
        captured = {}
        for path in paths:
            try:
                source = (b"" if path in self._restart_only else
                          snapshots[path] if snapshots is not None and path in snapshots else Path(path).read_bytes())
            except OSError as exc:
                source = exc
            captured[path] = source
            state = self.states.get(path)
            if self.graph.published_source(path) != source or state is None or state.status != "active":
                changed.append(path)
        if changed:
            self.request(changed, snapshots=captured)
        dependencies = tuple(path for path in self.graph.dependency_paths(paths) if path not in captured)
        changed_dependencies = []
        for path in dependencies:
            try:
                source = b"" if path in self._restart_only else Path(path).read_bytes()
            except OSError as exc:
                source = exc
            captured[path] = source
            state = self.states.get(path)
            if self.graph.published_source(path) != source or state is None or state.status != "active":
                changed_dependencies.append(path)
        if changed_dependencies:
            self.request(changed_dependencies, snapshots=captured)
        self.compile_pending()
        self.frontend.wait_for(tuple(request for path, request in self._pending.items() if path not in self._ready))
        self.drain()
        failed = [
            state
            for path in (*paths, *dependencies)
            if (state := self.states.get(path)) is None or state.status != "active"
        ]
        if failed:
            raise RuntimeError(f"Current scripts are not ready: {failed}")

    def close(self):
        if self._closed:
            return
        self._closed = True
        for owner in reversed(tuple(self._owners)):
            try:
                owner.close()
            except Exception as exc:
                from infernux.debug import Debug

                Debug.log_exception(exc)
        self._owners.clear()
        self.frontend.clear()
        self._pending.clear()
        self._ready.clear()
        self._initialization_failures.clear()
        sys.meta_path.remove(self)
        stop_observing_runtime_imports(self)
