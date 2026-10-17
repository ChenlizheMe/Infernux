"""ComponentLifecycleMixin — extracted from InxComponent."""
from __future__ import annotations


import weakref
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple, Optional
import sys



_RUNTIME_PHASE_NAMES = (
    "update",
    "fixed_update",
    "late_update",
    "physics_pre_step",
    "physics_post_step",
)
_RUNTIME_SCHEDULER_PHASES = (
    "fixed_update",
    "physics_pre_step",
    "physics_post_step",
    "update",
    "late_update",
)
_MISSING = object()
_runtime_phase_presence_fn = None


def _runtime_phase_presence(component_type: type) -> tuple[bool, ...]:
    global _runtime_phase_presence_fn
    resolver = _runtime_phase_presence_fn
    if resolver is None:
        from infernux.engine.runtime_dispatch import runtime_phase_presence

        resolver = _runtime_phase_presence_fn = runtime_phase_presence
    return resolver(component_type)
# The scheduler only derives plan invalidation from structural domains. Keeping
# field/value domains out lets the journal skip per-field bookkeeping in the
# Player, where no Inspector cursor subscribes to them.
_SCHEDULER_CHANGE_DOMAINS = (
    "scene_topology",
    "component_structure",
    "component_enabled",
    "script_schema",
)


def _missing_runtime_phase(*_args):
    """No-op phase for lightweight mixins that omit optional callbacks."""
    return None


class RuntimeExecutionScheduler:
    """Shared on-demand runtime plan/cache used by Editor and Player.

    Native ``SceneManager`` owns the phase boundaries. Python-side execution
    uses :class:`RuntimeExecutionFrame` so a native frame has one immutable
    component/dispatch snapshot, including frames with multiple fixed steps.
    Structural events keep the revision and component set current; steady
    native frames consume the published snapshot without rebuilding it.
    """

    _live_schedulers = weakref.WeakSet()
    _primary_scheduler: Any = None
    _runtime_service_phases: dict[str, set[str]] = {}

    def __init__(
        self,
        *,
        name: str = "runtime",
        change_journal: Any = None,
        native_bridge: bool = False,
    ) -> None:
        if change_journal is None:
            from infernux.engine.runtime_change_journal import runtime_change_journal

            change_journal = runtime_change_journal()
        self.name = str(name)
        self._native_bridge = bool(native_bridge)
        self._native_scene_manager = None
        self._change_journal = change_journal
        self._change_cursor = change_journal.create_cursor(
            f"runtime-execution:{self.name}:{id(self):x}",
            domains=_SCHEDULER_CHANGE_DOMAINS,
        )
        # Scene replacement publishes the new graph before the retained old
        # graph is finalized. Component IDs are scene-local, and freshly bound
        # Python mirrors both start at native generation 1, so the old and new
        # instances may temporarily have the same stable token. Keep instance
        # identity in the scheduler membership key; the two-field token remains
        # the durable identity used by the change journal and diagnostics.
        self._components: dict[tuple[int, int, int], Any] = {}
        self._component_tokens: dict[int, tuple[int, int, int]] = {}
        self._phase_plan: dict[str, tuple[Any, ...]] = {
            phase: () for phase in _RUNTIME_SCHEDULER_PHASES
        }
        self._structure_revision = 0
        self._plan_revision = -1
        self._registry_scan_pending = True
        self._dispatch_types_dirty = True
        self._counters = defaultdict(int)
        self._native_frame: RuntimeExecutionFrame | None = None
        self._cpu_mesh_frame_open = False
        self._execution_snapshot: _RuntimeExecutionSnapshot | None = None
        # Native work availability is maintained incrementally: registration is
        # O(1) and only a false<->true transition crosses to the SceneManager.
        self._work_components: set[int] = set()
        self._native_work_available = False
        self._native_work_dirty = False
        self._last_barrier_changes: dict[Any, Any] = {}
        self._last_completed_frame_changes: dict[Any, Any] = {}
        self._last_completed_barrier_sequence: tuple[tuple[Any, Any], ...] = ()
        self._live_schedulers.add(self)
        primary = RuntimeExecutionScheduler._primary_scheduler
        if primary is None or primary() is None:
            RuntimeExecutionScheduler._primary_scheduler = weakref.ref(self)

    def _component_has_runtime_work(self, component: Any) -> bool:
        if self._active_coroutine_scheduler(component) is not None:
            return True
        return any(_runtime_phase_presence(type(component)))

    def _note_component_work(self, component: Any, *, present: bool) -> None:
        """Update one component's work contribution and publish transitions.

        Scene loading registers thousands of components; each registration must
        stay O(1). The native lifecycle fast path only needs to know whether any
        work exists, so a set membership plus a level check is sufficient.
        """
        key = id(component)
        if present and self._component_has_runtime_work(component):
            self._work_components.add(key)
        else:
            self._work_components.discard(key)
        self._recompute_work_availability()

    def _recompute_work_availability(self) -> None:
        if not self._native_bridge or self._native_scene_manager is None:
            return
        has_work = bool(self._work_components) or any(self._runtime_service_phases.values())
        if has_work == self._native_work_available and not self._native_work_dirty:
            return
        self._native_work_dirty = False
        if has_work != self._native_work_available:
            self._native_work_available = has_work
            self._native_scene_manager.set_runtime_lifecycle_work_available(has_work)

    def _sync_native_work_availability(self, *, force: bool = False) -> None:
        """Recompute the work mirror; force rebuilds it from live membership."""
        if not self._native_bridge or self._native_scene_manager is None:
            return
        if force:
            self._work_components = {
                id(component)
                for component in self._components.values()
                if self._component_has_runtime_work(component)
            }
        self._recompute_work_availability()

    @classmethod
    def set_runtime_service_phase_active(
        cls,
        service: str,
        phase: str,
        active: bool,
    ) -> None:
        """Publish phase demand from one process-wide runtime service."""
        if phase not in _RUNTIME_SCHEDULER_PHASES:
            raise ValueError(f"unknown runtime phase: {phase}")
        name = str(service).strip()
        if not name:
            raise ValueError("runtime service name must not be empty")
        services = cls._runtime_service_phases.setdefault(phase, set())
        changed = False
        if active and name not in services:
            services.add(name)
            changed = True
        elif not active and name in services:
            services.remove(name)
            changed = True
        if not changed:
            return
        for scheduler in tuple(cls._live_schedulers):
            scheduler._recompute_work_availability()
            scheduler._publish_native_plan()

    @classmethod
    def _runtime_service_count(cls, phase: str) -> int:
        return len(cls._runtime_service_phases.get(phase, ()))

    def _publish_native_plan(self) -> None:
        """Publish one structural plan summary; never crosses per component/frame."""
        if not self._native_bridge or self._native_scene_manager is None:
            return
        self._native_scene_manager.set_runtime_lifecycle_plan(
            max(0, int(self._plan_revision)),
            len(self._phase_plan["fixed_update"])
            + self._runtime_service_count("fixed_update"),
            len(self._phase_plan["update"])
            + self._runtime_service_count("update"),
            len(self._phase_plan["late_update"])
            + self._runtime_service_count("late_update"),
        )

    def bind_native_bridge(self, scene_manager: Any) -> None:
        """Bind the one native SceneManager that owns runtime phase boundaries."""
        if not self._native_bridge:
            raise RuntimeError("scheduler was not configured for a native bridge")
        if scene_manager is None:
            raise TypeError("scene_manager is required")
        if self._native_scene_manager is not None and self._native_scene_manager is not scene_manager:
            raise RuntimeError("runtime scheduler is already bound to another SceneManager")
        self._native_scene_manager = scene_manager
        self.sync_native_work_availability()

    def unbind_native_bridge(self) -> None:
        self._native_scene_manager = None

    def sync_native_work_availability(self) -> None:
        """Synchronize the native fast-path after lifecycle bridge installation."""
        self._sync_active_registry_once()
        self._sync_native_work_availability(force=True)
        self._publish_native_plan()

    def refresh_scene_membership(self) -> None:
        """Reconcile the plan after one complete scene graph publication.

        Component bind/unbind notifications remain the cheap steady-state
        path.  Scene deserialization is different: it publishes a whole
        Python mirror while the retained previous graph can still be alive.
        Treat that publication boundary as authoritative and reconcile once,
        so a missed or reordered incremental notification cannot leave the
        native lifecycle fast path disabled for the entire scene.
        """
        self._registry_scan_pending = True
        self.prepare_frame()
        self._sync_native_work_availability(force=True)

    @classmethod
    def _notify_component_structure(cls, component: Any, *, present: bool) -> None:
        from infernux.engine.runtime_change_journal import RuntimeChangeDomain

        primary_ref = cls._primary_scheduler
        primary = primary_ref() if primary_ref is not None else None
        if primary is not None and len(cls._live_schedulers) == 1:
            # Scene load registers one component at a time; the single-scheduler
            # case must not iterate the WeakSet or dedupe journals per call.
            primary._change_journal.publish(
                RuntimeChangeDomain.COMPONENT_STRUCTURE,
                stable_id=cls._component_token(component),
            )
            primary._on_component_structure(component, present=present)
            return
        schedulers = tuple(cls._live_schedulers)
        cls._publish_to_scheduler_journals(
            schedulers,
            RuntimeChangeDomain.COMPONENT_STRUCTURE,
            stable_id=cls._component_token(component),
        )
        for scheduler in schedulers:
            scheduler._on_component_structure(component, present=present)

    @classmethod
    def _notify_component_revision(cls, component: Any, domain: Any) -> None:
        schedulers = tuple(cls._live_schedulers)
        cls._publish_to_scheduler_journals(
            schedulers,
            domain,
            stable_id=cls._component_token(component),
        )
        for scheduler in schedulers:
            scheduler._refresh_owner_active_mirror(component)

    @classmethod
    def _notify_component_runtime_work(cls, component: Any) -> None:
        for scheduler in tuple(cls._live_schedulers):
            scheduler._on_component_runtime_work(component)

    def _on_component_runtime_work(self, component: Any) -> None:
        """Invalidate phase membership when coroutine work starts or ends."""
        token = self._component_tokens.get(id(component))
        if token is not None and self._components.get(token) is component:
            self._structure_revision += 1
            self._execution_snapshot = None
            self._counters["coroutine_plan_invalidations"] += 1
        self._note_component_work(component, present=True)

    @staticmethod
    def _journal_type_id(component_type: type) -> str:
        type_id = component_type.__dict__.get("_runtime_journal_type_id")
        if type_id is None:
            type_id = sys.intern(
                str(
                    getattr(component_type, "_type_guid_", None)
                    or f"{component_type.__module__}.{component_type.__qualname__}"
                )
            )
            try:
                component_type._runtime_journal_type_id = type_id
            except (AttributeError, TypeError):
                pass
        return type_id

    @classmethod
    def _notify_component_value(
        cls,
        component: Any,
        field_id: Any,
    ) -> None:
        from infernux.engine.runtime_change_journal import RuntimeChangeDomain

        type_id = cls._journal_type_id(type(component))
        component_id = component._component_id or id(component)
        primary_ref = cls._primary_scheduler
        primary = primary_ref() if primary_ref is not None else None
        if primary is not None and len(cls._live_schedulers) == 1:
            # Steady state: one scheduler owns one journal. Avoid iterating the
            # WeakSet and the generic per-publish validation below. When no
            # consumer subscribes to field domains the write is journal-free.
            journal = primary._change_journal
            if journal._field_tracking:
                journal.publish_component_field_value(
                    type_id, component_id, field_id
                )
            return
        journals = cls._scheduler_journals(tuple(cls._live_schedulers))
        for journal in journals:
            if not journal._field_tracking:
                continue
            journal.publish_component_field(type_id, component_id, field_id)
            journal.publish(
                RuntimeChangeDomain.SCRIPT_VALUE,
                stable_id=component_id,
            )

    @classmethod
    def _notify_component_values(
        cls,
        components: tuple[Any, ...],
        field_id: Any,
    ) -> None:
        """Publish one homogeneous batch write without per-component calls."""
        if not components:
            return
        type_id = cls._journal_type_id(type(components[0]))
        # Unregistered candidates are private: their values must not invalidate
        # runtime consumers even though the native scatter already succeeded.
        component_ids = [
            component._component_id or id(component)
            for component in components
            if component._registered_go_id is not None
        ]
        if not component_ids:
            return
        primary_ref = cls._primary_scheduler
        primary = primary_ref() if primary_ref is not None else None
        if primary is not None and len(cls._live_schedulers) == 1:
            journal = primary._change_journal
            if journal._field_tracking:
                journal.publish_component_fields(type_id, component_ids, field_id)
            return
        for journal in cls._scheduler_journals(tuple(cls._live_schedulers)):
            if journal._field_tracking:
                journal.publish_component_fields(type_id, component_ids, field_id)

    @staticmethod
    def _scheduler_journals(schedulers: tuple[Any, ...]) -> tuple[Any, ...]:
        journals = []
        seen = set()
        for scheduler in schedulers:
            journal = scheduler._change_journal
            identity = id(journal)
            if identity in seen:
                continue
            seen.add(identity)
            journals.append(journal)
        if not journals:
            from infernux.engine.runtime_change_journal import runtime_change_journal

            journals.append(runtime_change_journal())
        return tuple(journals)

    @classmethod
    def _publish_to_scheduler_journals(
        cls,
        schedulers: tuple[Any, ...],
        domain: Any,
        **change: Any,
    ) -> None:
        for journal in cls._scheduler_journals(schedulers):
            journal.publish(domain, **change)

    @staticmethod
    def _component_token(component: Any) -> tuple[int, int]:
        component_id = int(getattr(component, "component_id", 0) or getattr(component, "_component_id", 0) or id(component))
        generation = int(getattr(component, "_native_generation", 0) or 0)
        return component_id, generation

    @classmethod
    def _membership_token(cls, component: Any) -> tuple[int, int, int]:
        component_id, generation = cls._component_token(component)
        return component_id, generation, id(component)

    @staticmethod
    def _sort_key(component: Any) -> tuple[int, int, int]:
        return (
            int(getattr(component, "execution_order", getattr(component, "_execution_order", 0)) or 0),
            int(getattr(component, "component_id", getattr(component, "_component_id", 0)) or 0),
            int(getattr(component, "_native_generation", 0) or 0),
        )

    def _invalidate(self, *, revision: int | None = None) -> None:
        next_revision = self._structure_revision + 1
        if revision is not None:
            next_revision = max(next_revision, int(revision))
        self._structure_revision = next_revision
        self._counters["plan_invalidations"] += 1

    def _on_component_structure(self, component: Any, *, present: bool) -> None:
        self._registry_scan_pending = False
        self._dispatch_types_dirty = True
        if present:
            # A present event may also represent a rebind.  Refresh from the
            # current owner instead of trusting the previous component mirror.
            self._refresh_owner_active_mirror(component)
            token = self._membership_token(component)
            previous = self._component_tokens.get(id(component))
            self._component_tokens[id(component)] = token
            self._components[token] = component
            if previous is not None and previous != token:
                if self._components.get(previous) is component:
                    self._components.pop(previous, None)
        else:
            previous = self._component_tokens.pop(id(component), None)
            if previous is not None and self._components.get(previous) is component:
                self._components.pop(previous, None)
        # Registration happens per component during scene load; the work mirror
        # is updated incrementally so loading stays O(1) per component.
        self._note_component_work(component, present=present)
        # The typed journal advances the plan revision at the next explicit
        # runtime barrier.  The object map is updated immediately so that the
        # next frame can build from authoritative membership without a scan.

    def register_component(self, component: Any) -> None:
        """Register a component for direct scheduler users and tests."""
        from infernux.engine.runtime_change_journal import RuntimeChangeDomain

        self._change_journal.publish(
            RuntimeChangeDomain.COMPONENT_STRUCTURE,
            stable_id=self._component_token(component),
        )
        self._on_component_structure(component, present=True)

    def unregister_component(self, component: Any) -> None:
        from infernux.engine.runtime_change_journal import RuntimeChangeDomain

        self._change_journal.publish(
            RuntimeChangeDomain.COMPONENT_STRUCTURE,
            stable_id=self._component_token(component),
        )
        self._on_component_structure(component, present=False)

    def mark_structure_changed(self, reason: str = "") -> None:
        """Invalidate the plan for topology/enabled/order changes."""
        del reason
        self._dispatch_types_dirty = True
        from infernux.engine.runtime_change_journal import RuntimeChangeDomain

        self._change_journal.publish(
            RuntimeChangeDomain.COMPONENT_STRUCTURE,
            broad=True,
        )

    def mark_type_body_reload(
        self,
        component_type: type,
        *,
        phase_presence_changed: bool = False,
    ) -> None:
        """Publish a new type dispatch generation without rebuilding the plan."""
        self._counters["body_reload_updates"] += 1
        if phase_presence_changed:
            # Replacing a helper does not affect the phase plan.  Adding or
            # removing a runtime phase does.
            from infernux.engine.runtime_change_journal import RuntimeChangeDomain

            type_id = (
                getattr(component_type, "_type_guid_", None)
                or f"{component_type.__module__}.{component_type.__qualname__}"
            )
            self._change_journal.publish(
                RuntimeChangeDomain.SCRIPT_SCHEMA,
                stable_id=type_id,
            )
            self._sync_native_work_availability(force=True)

    def mark_runtime_revision(self, revision: int) -> None:
        """Apply an external structural revision monotonically."""
        revision = int(revision)
        if revision > self._structure_revision:
            self._structure_revision = revision
            self._counters["plan_invalidations"] += 1

    def _sync_active_registry_once(self) -> None:
        if not self._registry_scan_pending:
            return
        self._registry_scan_pending = False
        try:
            from .component import InxComponent

            active = {}
            for values in tuple(getattr(InxComponent, "_active_instances", {}).values()):
                for component in tuple(values):
                    active[self._membership_token(component)] = component
        except (ImportError, AttributeError, RuntimeError):
            active = {}
        if set(active) != set(self._components):
            self._components = active
            self._component_tokens = {id(component): token for token, component in active.items()}
            self._dispatch_types_dirty = True
            for component in active.values():
                self._ensure_owner_active_mirror(component)
            from infernux.engine.runtime_change_journal import RuntimeChangeDomain

            self._change_journal.publish(
                RuntimeChangeDomain.COMPONENT_STRUCTURE,
                broad=True,
            )

    def consume_runtime_changes(self, barrier: Any, *, frame: Any = None) -> Any:
        """Consume one deterministic barrier and update derived revisions."""
        from infernux.engine.runtime_change_journal import (
            RuntimeChangeDomain,
            RuntimeFrameBarrier,
        )

        typed_barrier = RuntimeFrameBarrier(barrier)
        changes = self._change_journal.consume(self._change_cursor)
        if changes.affects(
            RuntimeChangeDomain.SCENE_TOPOLOGY,
            RuntimeChangeDomain.COMPONENT_STRUCTURE,
            RuntimeChangeDomain.COMPONENT_ENABLED,
            RuntimeChangeDomain.SCRIPT_SCHEMA,
        ):
            self._invalidate(revision=changes.revision)
        self._last_barrier_changes[typed_barrier] = changes
        if frame is not None:
            frame.barrier_changes[typed_barrier] = changes
            frame.barrier_sequence.append((typed_barrier, changes))
        self._counters["change_barriers"] += 1
        if changes.changed:
            self._counters["change_barriers_dirty"] += 1
            self._counters["change_domains_consumed"] += len(changes.changes)
        return changes

    @property
    def change_journal(self) -> Any:
        return self._change_journal

    @property
    def change_cursor_revision(self) -> int:
        return int(self._change_cursor.revision)

    def consumed_domain_revision(self, domain: Any) -> int:
        from infernux.engine.runtime_change_journal import RuntimeChangeDomain

        return int(
            self._change_cursor.domain_revisions.get(RuntimeChangeDomain(domain), 0)
        )

    def last_completed_frame_changes(self) -> dict[Any, Any]:
        return dict(self._last_completed_frame_changes)

    def last_completed_barrier_sequence(self) -> tuple[tuple[Any, Any], ...]:
        return tuple(self._last_completed_barrier_sequence)

    @staticmethod
    def _owner_is_active_in_hierarchy(component: Any) -> bool:
        """Read the cached owner only when a plan is being changed/built.

        Real components keep their bound GameObject in ``_game_object``.  A
        lightweight scheduler double may omit that attribute entirely, which
        is intentionally treated as an active owner.  ``None`` means an
        explicitly unbound component and is therefore not executable.
        """
        owner = getattr(component, "_game_object", _MISSING)
        if owner is _MISSING:
            # Lightweight doubles may expose only the public owner slot.
            owner = getattr(component, "game_object", _MISSING)
        if owner is _MISSING:
            return True
        if owner is None:
            return False
        try:
            scene = getattr(owner, "scene", None)
            if scene is not None and scene.is_preview:
                return False
            active = getattr(owner, "active_in_hierarchy", _MISSING)
            if active is _MISSING:
                return True
            return bool(active() if callable(active) else active)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            # A native owner can be invalidated while a scene transaction is
            # draining.  Such a component must not enter a runtime plan.
            return False

    @classmethod
    def _refresh_owner_active_mirror(cls, component: Any) -> bool:
        active = cls._owner_is_active_in_hierarchy(component)
        try:
            component.__dict__["_runtime_active_in_hierarchy"] = active
        except (AttributeError, TypeError):
            # A slotted test/runtime proxy can still be handled by the direct
            # plan result; normal InxComponent instances have a __dict__.
            pass
        return active

    @classmethod
    def _ensure_owner_active_mirror(cls, component: Any) -> bool:
        try:
            mirror = component.__dict__.get("_runtime_active_in_hierarchy", _MISSING)
        except AttributeError:
            mirror = getattr(component, "_runtime_active_in_hierarchy", _MISSING)
        if mirror is not _MISSING:
            return bool(mirror)
        return cls._refresh_owner_active_mirror(component)

    @classmethod
    def _eligible(cls, component: Any) -> bool:
        return cls._ensure_owner_active_mirror(component) and not bool(
            getattr(component, "_is_destroyed", False)
        ) and bool(getattr(component, "_enabled", True))

    @staticmethod
    def _active_coroutine_scheduler(component: Any) -> Any:
        scheduler = component.__dict__.get("_runtime_coroutine_scheduler")
        if scheduler is None:
            return None
        return scheduler if scheduler.count > 0 else None

    def _build_plan(self) -> None:
        entries: list[tuple[Any, bool]] = []
        for component in self._components.values():
            # Re-read the native owner's active state when rebuilding a plan.
            # A component can be deserialized before its GameObject is
            # activated; retaining the first mirror would silently exclude it
            # from fixed/update dispatch in a Player.
            owner_active = self._refresh_owner_active_mirror(component)
            if not owner_active:
                continue
            has_coroutines = self._active_coroutine_scheduler(component) is not None
            if has_coroutines or self._eligible(component):
                entries.append((component, has_coroutines))
        entries.sort(key=lambda item: self._sort_key(item[0]))
        presence_by_type = {
            component_type: _runtime_phase_presence(component_type)
            for component_type in {type(component) for component, _ in entries}
        }
        plan: dict[str, tuple[Any, ...]] = {}
        for phase in _RUNTIME_SCHEDULER_PHASES:
            index = _RUNTIME_PHASE_NAMES.index(phase)
            plan[phase] = tuple(
                component
                for component, has_coroutines in entries
                if has_coroutines
                or presence_by_type[type(component)][index]
            )
        self._phase_plan = plan
        self._plan_revision = self._structure_revision
        self._execution_snapshot = None
        self._counters["plan_builds"] += 1
        self._publish_native_plan()

    def _execution_snapshot_for_frame(self) -> "_RuntimeExecutionSnapshot":
        """Return the immutable dispatch snapshot for the current epoch.

        Stable frames share this object. A topology change rebuilds the phase
        plan, while a body reload publishes a different epoch object; either
        transition creates a fresh snapshot at the native frame boundary.
        """
        from infernux.engine.runtime_dispatch import current_runtime_epoch

        epoch = current_runtime_epoch()
        snapshot = self._execution_snapshot
        if (
            snapshot is not None
            and snapshot.structure_revision == self._structure_revision
            and snapshot.epoch is epoch
        ):
            self._counters["execution_snapshot_hits"] += 1
            return snapshot

        components: dict[int, tuple[Any, tuple[Any, ...]]] = {}
        for phase_components in self._phase_plan.values():
            for component in phase_components:
                component_key = id(component)
                if component_key in components:
                    continue
                descriptor = epoch.require_descriptor(type(component))
                components[component_key] = (component, descriptor.phase_invokers)

        dispatch_plan: dict[str, tuple[_DispatchEntry, ...]] = {}
        for phase, phase_components in self._phase_plan.items():
            invoker_index = _RUNTIME_PHASE_NAMES.index(phase)
            dispatch_plan[phase] = tuple(
                _DispatchEntry(component, components[id(component)][1][invoker_index])
                for component in phase_components
            )

        snapshot = _RuntimeExecutionSnapshot(
            structure_revision=self._structure_revision,
            epoch=epoch,
            phase_plan=MappingProxyType(
                {phase: tuple(values) for phase, values in self._phase_plan.items()}
            ),
            component_snapshots=MappingProxyType(components),
            dispatch_plan=MappingProxyType(dispatch_plan),
        )
        self._execution_snapshot = snapshot
        self._counters["execution_snapshot_builds"] += 1
        return snapshot

    def begin_frame(self) -> "RuntimeExecutionFrame":
        """Capture the executable plan at the only Python frame boundary.

        A body-only script reload is rejected while a frame is active.  The
        frame still owns the invoker tuples and immutable epoch it captured,
        so fixed/update/late phases share one revision.  The capture is
        deliberately separate from component execution:
        no Python lifecycle callback can mutate the snapshot while it is being
        assembled.
        """
        self.prepare_frame()
        frame = RuntimeExecutionFrame(self, self._execution_snapshot_for_frame())
        from infernux.engine.runtime_change_journal import RuntimeFrameBarrier

        safe_point = self._last_barrier_changes.get(RuntimeFrameBarrier.SAFE_POINT)
        if safe_point is not None:
            frame.barrier_changes[RuntimeFrameBarrier.SAFE_POINT] = safe_point
            frame.barrier_sequence.append((RuntimeFrameBarrier.SAFE_POINT, safe_point))
        self._counters["frame_begins"] += 1
        return frame

    def _execute_frame_phase(
        self,
        frame: "RuntimeExecutionFrame",
        phase: str,
        delta_time: float,
        *,
        editor_only: bool = False,
    ) -> None:
        from infernux.compute import recording

        counters = self._counters
        tick_coroutines = (
            phase == "update" or phase == "fixed_update" or phase == "late_update"
        )
        with recording():
            for component, invoke in frame.dispatch_plan[phase]:
                # ComponentLifecycleMixin declares these mirrors as class
                # defaults, so the steady-state path is a direct LOAD_ATTR.
                if component._is_destroyed:
                    counters["phase_skips"] += 1
                    continue
                if not component._runtime_active_in_hierarchy:
                    counters["phase_skips"] += 1
                    continue
                if editor_only and not component._execute_in_edit_mode:
                    counters["phase_skips"] += 1
                    continue

                if component._enabled:
                    try:
                        invoke(component, delta_time)
                    except Exception as exc:
                        self._report_phase_error(component, exc)

                # Keep the component/invoker tuple stable for the frame, but make
                # enabled transitions visible between phases just like the legacy
                # Scene traversal. A disable in fixed_update therefore suppresses
                # the following update/late_update callbacks immediately.
                # Component ``enabled`` gates the user callback, while the owning
                # hierarchy independently gates coroutine work.
                if not component._runtime_active_in_hierarchy:
                    continue
                if tick_coroutines:
                    scheduler = component.__dict__.get("_runtime_coroutine_scheduler")
                    if scheduler is not None and scheduler.count > 0:
                        try:
                            if phase == "update":
                                scheduler.tick_update(delta_time, epoch=frame.epoch)
                            elif phase == "fixed_update":
                                scheduler.tick_fixed_update(delta_time, epoch=frame.epoch)
                            else:
                                scheduler.tick_late_update(delta_time, epoch=frame.epoch)
                        except Exception as exc:
                            self._report_phase_error(component, exc)
                counters["phase_dispatches"] += 1

    def _report_phase_error(self, component: Any, exc: Exception) -> None:
        """Route one phase failure; only the error path pays for the fallback."""
        reporter = getattr(component, "_report_lifecycle_exception", None)
        if reporter is None:
            self._counters["phase_errors"] += 1
        else:
            reporter(exc)

    def execute_frame(
        self,
        fixed_delta_time: float,
        delta_time: float,
        late_delta_time: Optional[float] = None,
    ) -> None:
        """Execute all lifecycle phases against one dispatch snapshot.

        This is the explicit Python runtime entry point used by headless and
        test compositions.  The graphical editor and packaged Player still
        use the native Scene callback path, but they can share the same frame
        boundary when they opt into Python lifecycle execution.
        """
        frame = self.begin_frame()
        try:
            frame.execute_phase("fixed_update", fixed_delta_time)
            frame.execute_phase("physics_pre_step", fixed_delta_time)
            frame.execute_phase("physics_post_step", fixed_delta_time)
            frame.execute_phase("update", delta_time)
            frame.execute_phase(
                "late_update",
                delta_time if late_delta_time is None else late_delta_time,
            )
        finally:
            from infernux.engine.runtime_change_journal import RuntimeFrameBarrier

            self.consume_runtime_changes(RuntimeFrameBarrier.RETIREMENT, frame=frame)
            self._last_completed_frame_changes = dict(frame.barrier_changes)
            self._last_completed_barrier_sequence = tuple(frame.barrier_sequence)
            frame.close()

    def prepare_frame(self) -> dict[str, tuple[Any, ...]]:
        """Explicitly build or return the plan for the current revision."""
        self._sync_active_registry_once()
        if self._dispatch_types_dirty:
            from infernux.engine.runtime_dispatch import ensure_runtime_dispatch_types

            ensure_runtime_dispatch_types(
                tuple({type(component) for component in self._components.values()})
            )
            self._dispatch_types_dirty = False
        # Descriptor publication can emit ScriptSchema while establishing a
        # previously unseen type.  Consume after that publication so the same
        # safe point builds one plan, rather than rebuilding on the next call.
        from infernux.engine.runtime_change_journal import RuntimeFrameBarrier

        self.consume_runtime_changes(RuntimeFrameBarrier.SAFE_POINT)
        # Registration marks the native work mirror dirty; flush it once at the
        # frame boundary instead of once per component during scene load.
        self._sync_native_work_availability()
        if self._plan_revision == self._structure_revision:
            self._counters["plan_hits"] += 1
        else:
            self._build_plan()
        self._counters["plan_prepare_calls"] += 1
        return self._phase_plan

    def phase_plan(self, phase: str) -> tuple[Any, ...]:
        if phase not in _RUNTIME_SCHEDULER_PHASES:
            raise ValueError(f"unknown runtime phase: {phase}")
        return self.prepare_frame()[phase]

    def phase_plan_snapshot(self) -> dict[str, tuple[Any, ...]]:
        """Return the last published plan without entering a scheduler safe point."""
        return {phase: tuple(plan) for phase, plan in self._phase_plan.items()}

    def execute_phase(self, phase: str, delta_time: float) -> None:
        """Execute one standalone phase for tests and headless composition."""
        if phase not in _RUNTIME_SCHEDULER_PHASES:
            raise ValueError(f"unknown runtime phase: {phase}")
        frame = self.begin_frame()
        try:
            frame.execute_phase(phase, delta_time)
        finally:
            frame.close()

    # Native SceneManager bridge -------------------------------------------------
    # These methods deliberately keep one RuntimeExecutionFrame alive across
    # the native frame. SceneManager may consume fixed_update more than once,
    # but it never asks Python to rebuild the component/invoker snapshot for
    # those steps.

    def begin_native_frame(self) -> None:
        self.end_native_frame()
        self._native_frame = self.begin_frame()
        from infernux.compute import _begin_cpu_mesh_frame

        _begin_cpu_mesh_frame()
        self._cpu_mesh_frame_open = True
        self._counters["native_frame_begins"] += 1

    def execute_native_phase(self, phase: str, delta_time: float) -> None:
        if self._native_frame is None:
            raise RuntimeError("native lifecycle phase requires SceneManager.begin_native_frame")
        self._native_frame.execute_phase(phase, float(delta_time))
        self._last_native_phase = (phase, float(delta_time))
        if phase == "fixed_update":
            self._counters["native_fixed_callbacks"] += 1
            self._counters["native_fixed_delta_sum"] += float(delta_time)
        self._counters["native_phase_dispatches"] += 1

    def execute_native_editor_update(self, delta_time: float) -> None:
        if self._native_frame is None:
            raise RuntimeError("native editor phase requires SceneManager.begin_native_frame")
        from infernux.engine.runtime_change_journal import RuntimeFrameBarrier

        for phase, barrier in (
            ("update", RuntimeFrameBarrier.UPDATE_SCRIPT),
            ("late_update", RuntimeFrameBarrier.LATE_SCRIPT),
        ):
            self.consume_runtime_changes(barrier, frame=self._native_frame)
            with self._change_journal.transaction():
                self._execute_frame_phase(
                    self._native_frame, phase, float(delta_time), editor_only=True,
                )
        self._counters["native_editor_dispatches"] += 1

    def consume_native_barrier(self, barrier: Any) -> Any:
        """Consume a barrier emitted by the native production frame flow."""
        if self._native_frame is None:
            self._counters["native_barriers_without_frame"] += 1
            return None
        changes = self.consume_runtime_changes(barrier, frame=self._native_frame)
        self._counters["native_barriers"] += 1
        return changes

    def end_native_frame(self) -> None:
        if self._native_frame is None and not self._cpu_mesh_frame_open:
            return
        try:
            if self._native_frame is not None:
                from infernux.engine.runtime_change_journal import RuntimeFrameBarrier

                self.consume_runtime_changes(
                    RuntimeFrameBarrier.RETIREMENT,
                    frame=self._native_frame,
                )
                self._last_completed_frame_changes = dict(self._native_frame.barrier_changes)
                self._last_completed_barrier_sequence = tuple(
                    self._native_frame.barrier_sequence
                )
                self._native_frame.close()
                self._native_frame = None
        finally:
            if self._cpu_mesh_frame_open:
                from infernux.compute import _end_cpu_mesh_frame

                self._cpu_mesh_frame_open = False
                _end_cpu_mesh_frame()
        self._counters["native_frame_ends"] += 1

    def profiler_snapshot(self) -> dict[str, int]:
        """Return cheap integer counters; no formatting or logging occurs."""
        return dict(self._counters)

    def reset_profiler(self) -> None:
        self._counters.clear()

    def clear(self) -> None:
        """Release plan references at runtime shutdown without a new service."""
        self.end_native_frame()
        for frame in tuple(getattr(self, "_active_frames", ())):
            frame.close()
        self._components.clear()
        self._component_tokens.clear()
        self._sync_native_work_availability(force=True)
        self._invalidate()
        self._phase_plan = {phase: () for phase in _RUNTIME_SCHEDULER_PHASES}
        self._plan_revision = self._structure_revision
        self._publish_native_plan()
        self._execution_snapshot = None
        self._registry_scan_pending = True
        self._dispatch_types_dirty = True
        self._last_barrier_changes.clear()
        self._last_completed_frame_changes.clear()
        self._last_completed_barrier_sequence = ()
        self._change_cursor = self._change_journal.create_cursor(
            f"runtime-execution:{self.name}:{id(self):x}",
            domains=_SCHEDULER_CHANGE_DOMAINS,
        )


@dataclass(frozen=True, slots=True)
class _RuntimeExecutionSnapshot:
    structure_revision: int
    epoch: Any
    phase_plan: Mapping[str, tuple[Any, ...]]
    component_snapshots: Mapping[int, tuple[Any, tuple[Any, ...]]]
    dispatch_plan: Mapping[str, tuple["_DispatchEntry", ...]]


class _DispatchEntry(NamedTuple):
    """One component and the phase invoker captured for this frame epoch.

    The entry removes two steady-state costs from the dispatch loop: the
    ``id(component)`` snapshot lookup and the per-phase invoker tuple index.
    Phase membership and the invoker are both immutable for the frame; only
    the runtime state flags are read per phase.
    """

    component: Any
    invoke: Any


class RuntimeExecutionFrame:
    """One immutable lifecycle execution view.

    The scheduler owns topology revisions; this object additionally owns the
    invoker tuples and the immutable dispatch epoch.  Safe-point publication
    prevents a Play hot reload from changing ``update`` while a caller is
    still finishing ``late_update``.
    """

    def __init__(
        self,
        scheduler: RuntimeExecutionScheduler,
        snapshot: _RuntimeExecutionSnapshot,
    ) -> None:
        self._scheduler = scheduler
        self.phase_plan = snapshot.phase_plan
        self.component_snapshots = snapshot.component_snapshots
        self.dispatch_plan = snapshot.dispatch_plan
        self.structure_revision = snapshot.structure_revision
        self.epoch = snapshot.epoch
        self.barrier_changes: dict[Any, Any] = {}
        self.barrier_sequence: list[tuple[Any, Any]] = []
        self._closed = False
        active_frames = getattr(scheduler, "_active_frames", None)
        if active_frames is None:
            active_frames = weakref.WeakSet()
            scheduler._active_frames = active_frames
        active_frames.add(self)

    def execute_phase(self, phase: str, delta_time: float) -> None:
        if self._closed:
            raise RuntimeError("runtime execution frame is already closed")
        if phase not in _RUNTIME_SCHEDULER_PHASES:
            raise ValueError(f"unknown runtime phase: {phase}")
        from infernux.engine.runtime_change_journal import RuntimeFrameBarrier

        barriers = {
            "fixed_update": RuntimeFrameBarrier.FIXED_SCRIPT,
            "physics_pre_step": RuntimeFrameBarrier.PHYSICS_PRE_SCRIPT,
            "physics_post_step": RuntimeFrameBarrier.PHYSICS_POST_SCRIPT,
            "update": RuntimeFrameBarrier.UPDATE_SCRIPT,
            "late_update": RuntimeFrameBarrier.LATE_SCRIPT,
        }
        self._scheduler.consume_runtime_changes(barriers[phase], frame=self)
        # A phase can author thousands of field/CDS writes. Their setters call
        # notify_runtime_component_value_changed(), which accumulates here
        # without taking the journal lock per write. The next barrier publishes
        # one coalesced revision for the whole phase.
        with self._scheduler.change_journal.transaction():
            self._scheduler._execute_frame_phase(self, phase, delta_time)

    def resolve_method(self, component: Any, name: str) -> Any:
        """Resolve a helper against this frame's captured dispatch epoch."""
        if self._closed:
            raise RuntimeError("runtime execution frame is already closed")
        return self.epoch.resolve_method(component, name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        active_frames = getattr(self._scheduler, "_active_frames", None)
        if active_frames is not None:
            active_frames.discard(self)


def _runtime_phase_invokers(component, *, epoch=None):
    """Return phase invokers from the selected published epoch."""
    if epoch is None:
        from infernux.engine.runtime_dispatch import current_runtime_epoch

        epoch = current_runtime_epoch()
    return epoch.require_descriptor(type(component)).phase_invokers


def resolve_runtime_method(component: Any, name: str, *, epoch: Any = None) -> Any:
    """Resolve a helper using the current epoch or an execution-frame epoch."""
    from infernux.engine.runtime_dispatch import resolve_runtime_method as _resolve

    return _resolve(component, name, epoch=epoch)


def notify_runtime_component_added(component: Any) -> None:
    """Notify shared runtime plans about a newly bound component."""
    RuntimeExecutionScheduler._notify_component_structure(component, present=True)


def notify_runtime_component_removed(component: Any) -> None:
    """Notify shared runtime plans about a detached/destroyed component."""
    RuntimeExecutionScheduler._notify_component_structure(component, present=False)


def notify_runtime_component_changed(component: Any) -> None:
    """Compatibility notification for component enabled/activity changes."""
    from infernux.engine.runtime_change_journal import RuntimeChangeDomain

    RuntimeExecutionScheduler._notify_component_revision(
        component,
        RuntimeChangeDomain.COMPONENT_ENABLED,
    )


def notify_runtime_component_structure_changed(component: Any) -> None:
    from infernux.engine.runtime_change_journal import RuntimeChangeDomain

    RuntimeExecutionScheduler._notify_component_revision(
        component,
        RuntimeChangeDomain.COMPONENT_STRUCTURE,
    )


def notify_runtime_component_owner_activity_changed(component: Any) -> None:
    from infernux.engine.runtime_change_journal import RuntimeChangeDomain

    RuntimeExecutionScheduler._notify_component_revision(
        component,
        RuntimeChangeDomain.SCENE_TOPOLOGY,
    )


def notify_runtime_component_value_changed(component: Any, field_id: Any) -> None:
    """Publish one successful field/CDS write without invalidating phase plans.

    ``SerializedFieldDescriptor`` and direct CDS setters share this single
    integration point. Call it only after the authoritative write succeeds;
    the active phase/safe-point transaction coalesces repeated field keys.
    """
    RuntimeExecutionScheduler._notify_component_value(component, field_id)


def notify_runtime_component_values(components: tuple[Any, ...], field_id: Any) -> None:
    """Publish one homogeneous batch write (CDS scatter) in a single pass."""
    RuntimeExecutionScheduler._notify_component_values(components, field_id)


class ComponentLifecycleMixin:
    """ComponentLifecycleMixin method group for InxComponent."""

    # Runtime state mirrors used by the dispatch hot loop. Declaring them on the
    # class makes ``component._enabled`` a direct attribute load for every real
    # component and test double, instead of a per-phase ``getattr(..., default)``.
    # Instances that need another value assign the instance attribute as before.
    _is_destroyed = False
    _enabled = True
    _runtime_active_in_hierarchy = True
    _execute_in_edit_mode = False
    _cds_key = None
    _component_id = 0
    _registered_go_id = None

    def _resolve_runtime_method(self, name: str, *, epoch: Any = None) -> Any:
        """Resolve a helper through the owner-published runtime epoch."""
        return resolve_runtime_method(self, name, epoch=epoch)

    def _safe_lifecycle_call(
        self,
        method_name: str,
        *args,
        epoch: Any = None,
    ) -> bool:
        """Call a lifecycle method from the selected published epoch.

        Lifecycle entry points are not cached on the instance.  Resolving the
        immutable descriptor creates only the short-lived bound method needed
        for this call, so a body publication is visible to the next safe-point
        callback without retaining the previous Python function.
        """
        try:
            if epoch is None:
                from infernux.engine.runtime_dispatch import current_runtime_epoch

                epoch = current_runtime_epoch()
            callback = self._resolve_runtime_method(method_name, epoch=epoch)
            epoch.require_descriptor(type(self))
            if callback is None:
                raise AttributeError(
                    f"{type(self).__name__} has no lifecycle method '{method_name}'"
                )
            callback(*args)
            return True
        except Exception as exc:
            try:
                from infernux.debug import Debug
                Debug.report_script_error(
                    exc,
                    f"{type(self).__name__}.{method_name}",
                    self,
                )
            except (ImportError, RuntimeError):
                # Absolute fallback if debug itself cannot be imported.
                import traceback
                traceback.print_exc()
            return False

    def _report_lifecycle_exception(self, exc: Exception) -> None:
        """Route a phase exception without adding work to the success path."""
        try:
            from infernux.debug import Debug
            Debug.report_script_error(exc, f"{type(self).__name__}.lifecycle", self)
        except (ImportError, RuntimeError):
            import traceback
            traceback.print_exc()

    def _disable_after_awake_exception(self):
        """Unity disables a script component if its Awake throws."""
        self._enabled = False

        cpp_component = getattr(self, '_cpp_component', None)
        if cpp_component is None:
            return

        try:
            cpp_component.enabled = False
        except RuntimeError:
            self._invalidate_native_binding()

    def _call_awake(self):
        """Internal: Trigger awake lifecycle."""
        if self._awake_called:
            return
        self._awake_called = True
        if not self._safe_lifecycle_call("awake"):
            self._disable_after_awake_exception()

    def _call_start(self):
        """Internal: Trigger start lifecycle if not already called."""
        if self._has_started:
            return
        self._has_started = True
        self._safe_lifecycle_call("start")

    def _call_update(self, delta_time: float):
        """Internal: Trigger update lifecycle."""
        # The native proxy already checked its authoritative enabled bit before
        # entering Python.  Reading ``self.enabled`` here performed another
        # scene/handle lookup for every component and every frame.  The mirror
        # is updated by bind/enable/disable transitions, so this is sufficient
        # for direct Python callers as well.
        if not self._enabled:
            return
        try:
            invokers = _runtime_phase_invokers(self)
            invokers[0](self, delta_time)
        except Exception as exc:
            self._report_lifecycle_exception(exc)
        # Tick coroutines after user update (matching Unity order).  Keep the
        # scheduler local so the phase does not perform another attribute
        # lookup or call a second wrapper method.
        scheduler = self.__dict__.get("_runtime_coroutine_scheduler")
        if scheduler is not None:
            scheduler.tick_update(delta_time)

    def _call_fixed_update(self, fixed_delta_time: float):
        """Internal: Trigger fixed_update lifecycle."""
        if not self._enabled:
            return
        try:
            invokers = _runtime_phase_invokers(self)
            invokers[1](self, fixed_delta_time)
        except Exception as exc:
            self._report_lifecycle_exception(exc)
        scheduler = self.__dict__.get("_runtime_coroutine_scheduler")
        if scheduler is not None:
            scheduler.tick_fixed_update(fixed_delta_time)

    def _call_late_update(self, delta_time: float):
        """Internal: Trigger late_update lifecycle."""
        if not self._enabled:
            return
        try:
            invokers = _runtime_phase_invokers(self)
            invokers[2](self, delta_time)
        except Exception as exc:
            self._report_lifecycle_exception(exc)
        scheduler = self.__dict__.get("_runtime_coroutine_scheduler")
        if scheduler is not None:
            scheduler.tick_late_update(delta_time)

    def _call_on_destroy(self):
        """Internal: Trigger on_destroy lifecycle."""
        if self._is_destroyed or self.__dict__.get("_is_destroying", False):
            return  # Already destroyed, don't call again
        # Keep the native owner available until cleanup returns. The in-progress
        # flag prevents reentrant destruction while disabled/removed membership
        # prevents any further runtime phase from scheduling this component.
        self.__dict__["_is_destroying"] = True
        self._enabled = False
        try:
            self._remove_from_active_registry()
            try:
                # Coroutine cleanup also runs while the owner is still bound.
                if self._coroutine_scheduler is not None:
                    self._coroutine_scheduler.stop_all()
            finally:
                self._coroutine_scheduler = None
                self.__dict__["_runtime_coroutine_scheduler"] = None
                if self._awake_called:
                    self._safe_lifecycle_call("on_destroy")
        finally:
            self.__dict__["_runtime_coroutine_scheduler"] = None
            self._invalidate_native_binding()
            self.__dict__["_is_destroying"] = False

    def _finalize_play_domain_replacement(self, *, was_awake: bool) -> None:
        """Finish an edit-domain instance after transactional Play replacement.

        Native identity-preserving replacement detaches the old Python object
        without invoking ``on_destroy`` because ordinary script hot reload must
        keep the lifecycle uninterrupted. Entering Play is different: the edit
        scripting domain is retired and class-level registrations owned by the
        old instance must be released before the fresh Play instance runs Awake.

        This hook is deliberately called only after the complete replacement
        batch has committed, so a failed transaction can still roll back to the
        untouched edit-domain instances.
        """
        if self.__dict__.get("_play_domain_destroy_finalized", False):
            return
        self.__dict__["_play_domain_destroy_finalized"] = True
        owner = self.__dict__["_play_domain_retirement_owner"]

        def cleanup():
            try:
                if self._coroutine_scheduler is not None:
                    self._coroutine_scheduler.stop_all()
            finally:
                self._coroutine_scheduler = None
                self.__dict__["_runtime_coroutine_scheduler"] = None
                if was_awake:
                    # Resolve against the retired class body; the replacement
                    # has its own native proxy, fields and dispatch descriptor.
                    self.on_destroy()

        try:
            self._invoke_native_cleanup(cleanup, None, owner, True)
        except Exception as exc:
            self._report_lifecycle_exception(exc)
        finally:
            self._invalidate_native_binding()
            self.__dict__.pop("_play_domain_retirement_owner", None)

    def _set_component_data_slot(self, slot, class_id):
        """Bind slot ownership; transactions retain responsibility for replaced slots."""
        if self._cds_finalizer is not None:
            self._cds_finalizer.detach()
        self._cds_slot = slot
        self._cds_class_id = class_id
        # One tuple carries both identity words for the descriptor fast path.
        self._cds_key = (class_id, slot) if slot is not None and class_id is not None else None
        self._cds_finalizer = None
        if slot is not None and class_id is not None:
            from ._cds_bridge import release_slot

            self._cds_finalizer = weakref.finalize(self, release_slot, type(self), slot, class_id)
            self._cds_finalizer.atexit = False

    def _release_component_data_slot(self):
        """Relinquish the numeric-field slot exactly once."""
        cds_slot = getattr(self, '_cds_slot', None)
        cds_class_id = getattr(self, '_cds_class_id', None)
        self._set_component_data_slot(None, None)
        if cds_slot is None or cds_class_id is None:
            return
        from ._cds_bridge import release_slot as _cds_free
        _cds_free(self.__class__, cds_slot, cds_class_id)

    def _call_on_enable(self):
        """Internal: Trigger on_enable lifecycle."""
        # A hierarchy activation is independent from the component's own
        # enabled flag.  Native state synchronization owns ``_enabled``;
        # this callback only publishes the cheap runtime activity mirror.
        self.__dict__["_runtime_active_in_hierarchy"] = True
        notify_runtime_component_owner_activity_changed(self)
        self._safe_lifecycle_call("on_enable")

    def _call_on_disable(self):
        """Internal: Trigger on_disable lifecycle."""
        # Do not turn a still-enabled component into a disabled component when
        # only its GameObject or one of its parents became inactive.
        self.__dict__["_runtime_active_in_hierarchy"] = False
        notify_runtime_component_owner_activity_changed(self)
        self._safe_lifecycle_call("on_disable")

    def _call_on_validate(self):
        """Internal: Trigger on_validate lifecycle (editor only)."""
        self._safe_lifecycle_call("on_validate")

    def _call_reset(self) -> bool:
        """Trigger editor Reset and report whether the callback succeeded."""
        return self._safe_lifecycle_call("reset")

    def _call_on_after_deserialize(self):
        """Trigger the transactional post-deserialize hook and propagate failure."""
        from infernux.engine.runtime_dispatch import current_runtime_epoch

        epoch = current_runtime_epoch()
        callback = self._resolve_runtime_method("on_after_deserialize", epoch=epoch)
        epoch.require_descriptor(type(self))
        if callback is None:
            raise AttributeError(
                f"{type(self).__name__} has no lifecycle method 'on_after_deserialize'"
            )
        callback()

    def _call_on_before_serialize(self):
        """Internal: Trigger on_before_serialize lifecycle."""
        self._safe_lifecycle_call("on_before_serialize")

