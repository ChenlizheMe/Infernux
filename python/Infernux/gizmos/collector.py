"""
GizmosCollector — per-frame scene walk that invokes gizmo callbacks and uploads.

Called once per frame (before ``SubmitCulling()``) to:
1. Reset Gizmos per-frame state.
2. [OPTIMISED] Iterate Python components via ``InxComponent._active_instances``
   (populated by ``_set_game_object()``); zero pybind11 calls when no Python
   components exist in the scene.
3. [OPTIMISED] For built-in C++ components that define gizmo methods or icons,
   use a per-type cached GO list rebuilt lazily after scene changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, TYPE_CHECKING

from Infernux.gizmos.gizmos import Gizmos, ICON_KIND_DEFAULT
from Infernux.components.fields import SerializedFieldDescriptor
from Infernux.debug import Debug
from Infernux.engine.editor_visibility import (
    component_owner_is_active_in_hierarchy,
    game_object_is_active_in_hierarchy,
)

if TYPE_CHECKING:
    from Infernux.engine.engine import Engine


# ---------------------------------------------------------------------------
# Module-level scene-change notification flag
#
# Call ``notify_scene_changed()`` from play_mode.py / scene_manager.py when
# objects are added or removed from the scene.  The next collect() call will
# rebuild the per-type icon cache.
# ---------------------------------------------------------------------------
_scene_dirty: bool = True   # start dirty so the cache is populated on first frame

# Rate-limited warning logger for per-frame gizmo issues (avoids console spam)
_gizmo_warn_count: int = 0
_GIZMO_WARN_LIMIT: int = 5


@dataclass(frozen=True, slots=True)
class GizmoCollectionObservation:
    """Low-overhead phase marker for one authoritative Gizmo collection.

    The values deliberately stop at the native upload boundary. GPU execution
    remains owned by renderer profiling, so this marker cannot accidentally
    report CPU submission time as GPU draw time.
    """

    scene_count: int = 0
    python_callbacks: int = 0
    builtin_callbacks: int = 0
    skipped_inactive: int = 0
    skipped_disabled: int = 0
    cpu_vertices: int = 0
    cpu_line_indices: int = 0
    cpu_draws: int = 0
    resident_vertices: int = 0
    resident_draws: int = 0
    icons: int = 0
    collect_ms: float = 0.0
    pack_ms: float = 0.0
    upload_ms: float = 0.0
    total_ms: float = 0.0


def _log_gizmo_warning(msg: str) -> None:
    """Log a gizmo-related warning, suppressed after a per-session limit."""
    global _gizmo_warn_count
    if _gizmo_warn_count >= _GIZMO_WARN_LIMIT:
        return
    _gizmo_warn_count += 1
    suffix = "" if _gizmo_warn_count < _GIZMO_WARN_LIMIT else " (further warnings suppressed)"
    Debug.log_warning(f"{msg}{suffix}")


def notify_scene_changed() -> None:
    """Signal that the scene has changed; next frame rebuilds the icon cache."""
    global _scene_dirty
    _scene_dirty = True


class GizmosCollector:
    """Stateful per-frame collector.

    Keep a single long-lived instance per Engine; call ``invalidate_cache()``
    whenever objects are added/removed (scene reload, play-mode transitions).

    Performance characteristics for a scene with N purely-C++ objects and K
    Python-component objects:
      * Pass 1: O(K) — zero pybind11 calls when K=0
      * Pass 2: O(icon_instances) per frame after cache warm-up; O(N) once on
        cache miss (per icon type, e.g. first frame or after scene change)

    Usage::

        collector = GizmosCollector()
        # on scene change / play-mode exit:
        collector.invalidate_cache()
        # each frame:
        collector.collect_and_upload(engine)
    """

    def __init__(self):
        # per-type GO cache: type_name -> list of live GO pybind11 objects
        # that carry that component.  Rebuilt lazily when _cache_built doesn't
        # contain the type_name.
        self._icon_cache: Dict[str, list] = {}
        self._cache_built: set = set()   # type_names whose cache entry is filled
        # Loaded scenes form one editor world.  Cache invalidation therefore
        # follows the complete resident scene set, not whichever scene happens
        # to be active for object creation.
        self._last_structure_signature = ()
        self._builtin_registry = None
        # The selected-object subtree is stable until the scene structure or
        # selection changes.  Keep only its IDs: component state and Gizmo
        # geometry are still evaluated every frame.
        self._selection_cache_key = None
        self._selected_ancestor_ids = frozenset()
        self._cache_generation = 0
        # Native clears are transition based. Empty frames must not cross the
        # Python/C++ boundary merely to clear buffers that are already empty.
        self._cpu_uploaded = False
        self._resident_uploaded = False
        self._icons_uploaded = False
        self._last_observation = GizmoCollectionObservation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Invalidate the GO-to-type cache (call on scene change / play mode exit)."""
        self._icon_cache.clear()
        self._cache_built.clear()
        self._cache_generation += 1
        self._selection_cache_key = None
        self._selected_ancestor_ids = frozenset()

    @property
    def last_observation(self) -> GizmoCollectionObservation:
        """Return the most recently completed per-frame phase observation."""
        return self._last_observation

    def retire_uploaded(self, native) -> None:
        """Retire every CPU/GPU Gizmo publication and transient Python frame.

        Used by hidden Scene Views, the global Gizmos switch, and empty loaded
        worlds. Resident compute storage remains owned by its source buffer;
        native draw leases and per-frame Python descriptor references are
        released immediately.
        """
        Gizmos._begin_frame()
        from Infernux.engine.interaction.handles import EditorHandleRegistry
        if EditorHandleRegistry._instance is not None:
            EditorHandleRegistry._instance.retire_frame()
        if native is not None and (
            self._cpu_uploaded or self._resident_uploaded or self._icons_uploaded
        ):
            native.clear_component_gizmos()
        self._cpu_uploaded = False
        self._resident_uploaded = False
        self._icons_uploaded = False
        self._last_observation = GizmoCollectionObservation()

    def collect_and_upload(self, engine: 'Engine') -> None:
        """Walk the scene, invoke callbacks, upload geometry to C++."""
        global _scene_dirty

        frame_started = time.perf_counter()
        python_callbacks = 0
        builtin_callbacks = 0
        skipped_inactive = 0
        skipped_disabled = 0

        from Infernux.lib import SceneManager as _SM
        # Ensure built-in wrapper classes (Camera, Light, etc.) have run their
        # BuiltinComponent.__init_subclass__ registration before we snapshot
        # _builtin_registry. Without this prewarm, icon-only gizmos can appear
        # to "do nothing" if no earlier code path imported the wrappers yet.
        import Infernux.components.builtin  # noqa: F401
        from Infernux.components.builtin_component import BuiltinComponent
        from Infernux.components.component import InxComponent

        native = engine.get_native_engine()
        if native is None:
            return

        # Drop the previous frame's transient descriptors before any early
        # return. This is also the exact start of the new accumulation frame.
        Gizmos._begin_frame()

        scene_manager = _SM.instance()
        scenes = tuple(
            scene
            for index in range(int(scene_manager.scene_count))
            if (scene := scene_manager.get_scene_at(index)) is not None
        )
        if not scenes:
            self.retire_uploaded(native)
            return

        # One signature covers both scene residency and hierarchy mutations.
        # World identity keeps equal structure revisions from aliasing after an
        # additive load/unload.
        signature = tuple(
            (int(scene.world_id), int(scene.structure_version))
            for scene in scenes
        )
        if signature != self._last_structure_signature:
            self.invalidate_cache()
            self._last_structure_signature = signature
            _scene_dirty = False
        elif _scene_dirty:
            self.invalidate_cache()
            _scene_dirty = False

        selected_id: int = engine.get_selected_object_id()

        # Build the selected subtree only when its inputs changed.  The set is
        # used for membership tests below, while all component enabled checks,
        # callbacks, transforms, and uploads remain per-frame.
        selected_scene = next(
            (scene for scene in scenes if scene.find_by_id(selected_id) is not None),
            None,
        ) if selected_id else None
        selected_ancestors = self._get_selected_ancestor_ids(
            selected_scene, selected_id
        )

        # Builtin wrapper registration is process-stable after the import above;
        # copying this dictionary for every scene frame was needless work.
        if self._builtin_registry is None:
            self._builtin_registry = dict(BuiltinComponent._builtin_registry)
        builtin_registry = self._builtin_registry

        # ====================================================================
        # Pass 1: Python component gizmos
        #
        # ``InxComponent._active_instances`` is a {go_id: [InxComponent, ...]}
        # dict populated by ``_set_game_object()`` — no pybind11 calls needed
        # to discover which objects have Python components.
        # For a scene with zero Python components this pass costs nothing.
        # ====================================================================
        active = InxComponent._active_instances
        if active:
            # Snapshot to avoid issues if the dict mutates during iteration
            for go_id, components in list(active.items()):
                is_selected = go_id in selected_ancestors
                for comp in components:
                    if isinstance(comp, BuiltinComponent):
                        continue
                    if not component_owner_is_active_in_hierarchy(comp):
                        skipped_inactive += 1
                        continue
                    if not getattr(comp, 'enabled', True):
                        skipped_disabled += 1
                        continue

                    # Scene icon — Python components opt in the same way the
                    # C++ wrappers do, by declaring _gizmo_icon_color.
                    icon_color = getattr(comp, '_gizmo_icon_color', None)
                    if icon_color is not None:
                        self._register_python_component_icon(comp, go_id, icon_color)

                    always_show = getattr(comp, '_always_show', True)
                    should_draw = always_show or is_selected
                    has_gizmos = (
                        type(comp).on_draw_gizmos is not InxComponent.on_draw_gizmos
                        or type(comp).on_draw_gizmos_selected is not InxComponent.on_draw_gizmos_selected
                    )
                    if should_draw and has_gizmos:
                        comp._call_on_draw_gizmos()
                        python_callbacks += 1

                    if is_selected and has_gizmos:
                        comp._call_on_draw_gizmos_selected()
                        python_callbacks += 1

        # ====================================================================
        # Pass 2: Built-in C++ component gizmos (Camera, Light, BoxCollider…)
        #
        # Optimisations applied:
        #   a) Skip types that have neither icon colour nor custom gizmo methods
        #      (e.g. Rigidbody, MeshRenderer) — they contribute nothing.
        #   b) Skip selection-only types (always_show=False, no icon) when
        #      nothing is selected; only visit selected descendants for these.
        #   c) For icon types (Camera, Light): use _get_icon_instances() which
        #      builds a per-type GO list once and reuses it every subsequent
        #      frame (~O(1) after first build, O(N) on cache miss).
        # ====================================================================
        for type_name, wrapper_cls in builtin_registry.items():

            # --- a) Pre-filter: does this type contribute anything visible? ---
            icon_color = self._resolve_class_value(wrapper_cls, '_gizmo_icon_color', None)
            has_gizmos = (
                wrapper_cls.on_draw_gizmos is not InxComponent.on_draw_gizmos
                or wrapper_cls.on_draw_gizmos_selected is not InxComponent.on_draw_gizmos_selected
            )
            if icon_color is None and not has_gizmos:
                continue  # Nothing to draw (Rigidbody, MeshRenderer, …)

            always_show_cls = bool(self._resolve_class_value(wrapper_cls, '_always_show', True))

            # --- b) Skip selection-only types when nothing is selected ---
            if icon_color is None and not always_show_cls and not selected_ancestors:
                continue  # Colliders etc. — nothing to show, nothing selected

            # --- c) Determine which GOs to iterate ---
            if icon_color is not None:
                # Icon types: use the per-type GO cache (O(1) after first frame)
                matching = self._get_icon_instances(scenes, type_name)
            elif not always_show_cls and selected_ancestors:
                # Selection-only gizmos: only visit selected + descendant objects
                matching = [
                    selected_scene.find_by_id(gid)
                    for gid in selected_ancestors if gid
                ]
                matching = [go for go in matching if go is not None]
            else:
                # Always-visible gizmos span the complete loaded editor world.
                matching = [
                    go
                    for loaded_scene in scenes
                    for go in loaded_scene.get_all_objects()
                ]

            for go in matching:
                # Gizmos follow GameObject.activeInHierarchy, not activeSelf.
                # Keep this gate before component lookup, icon registration,
                # wrapper creation, and callbacks so every built-in component
                # shares the same hierarchy visibility contract.
                if not game_object_is_active_in_hierarchy(go):
                    skipped_inactive += 1
                    continue
                try:
                    go_id = go.id
                    is_selected = go_id in selected_ancestors
                    # Always use C++ lookup — _type_map stores Python wrappers
                    # which can become stale after component removal.
                    cpp_comp = go.get_cpp_component(type_name)
                except Exception as exc:
                    _log_gizmo_warning(f"Gizmo: failed to query component '{type_name}': {exc}")
                    continue
                if cpp_comp is None:
                    continue

                try:
                    enabled = cpp_comp.enabled
                except Exception as exc:
                    _log_gizmo_warning(f"Gizmo: failed to read enabled on '{type_name}': {exc}")
                    continue
                if not enabled:
                    skipped_disabled += 1
                    continue

                # ---- Icon registration (always, regardless of selection) ----
                if icon_color is not None:
                    transform = go.get_transform()
                    if transform is not None:
                        pos = transform.position
                        icon_kind = self._resolve_class_value(wrapper_cls, '_gizmo_icon_kind', ICON_KIND_DEFAULT)
                        Gizmos.draw_icon(
                            (pos.x, pos.y, pos.z), go_id, icon_color, icon_kind=icon_kind)

                # ---- Gizmo lifecycle ----
                if not has_gizmos:
                    continue

                try:
                    wrapper = wrapper_cls._get_or_create_wrapper(cpp_comp, go)
                except Exception as exc:
                    _log_gizmo_warning(f"Gizmo: failed to create wrapper for '{type_name}': {exc}")
                    continue
                if wrapper is None:
                    continue

                try:
                    always_show_inst = getattr(wrapper, '_always_show', True)
                    should_draw = always_show_inst or is_selected
                    if should_draw:
                        wrapper._call_on_draw_gizmos()
                        builtin_callbacks += 1

                    if is_selected:
                        wrapper._call_on_draw_gizmos_selected()
                        builtin_callbacks += 1
                except Exception as exc:
                    _log_gizmo_warning(f"Gizmo callback failed for '{type_name}': {exc}")
                    wrapper._invalidate_native_binding()

        # Project and plugin handles share the exact Gizmo publication frame.
        # Their registry owns interaction/callback lifetimes; this collector
        # only appends the submitted geometry to the existing batch.
        from Infernux.engine.interaction.handles import EditorHandleRegistry
        if EditorHandleRegistry._instance is not None:
            EditorHandleRegistry._instance.collect(engine)

        collect_finished = time.perf_counter()

        # ---- Pack and upload line gizmo data ----
        packed = Gizmos._get_packed_data()
        resident = Gizmos._get_resident_data()
        icon_packed = Gizmos._get_packed_icon_data()
        pack_finished = time.perf_counter()

        cpu_vertices = 0
        cpu_line_indices = 0
        cpu_draws = 0
        if packed is not None:
            vert_buf, cpu_vertices, idx_buf, desc_buf, cpu_draws = packed
            cpu_line_indices = len(idx_buf)
            native.upload_component_gizmos(
                vert_buf, cpu_vertices, idx_buf, desc_buf, cpu_draws)
            self._cpu_uploaded = True
        elif self._cpu_uploaded:
            native.clear_component_cpu_gizmos()
            self._cpu_uploaded = False

        # Resident lines retain positions on the GPU. Their compute expansion
        # has already been recorded in the current Gizmo phase; this call only
        # publishes stable topology and native buffer identities to rendering.
        if resident:
            native.upload_component_resident_gizmos(resident)
            self._resident_uploaded = True
        elif self._resident_uploaded:
            native.upload_component_resident_gizmos([])
            self._resident_uploaded = False

        # ---- Pack and upload icon data ----
        icon_count = 0
        if icon_packed is not None:
            pos_color_buf, id_buf, kind_buf, icon_count = icon_packed
            native.upload_component_gizmo_icons(
                pos_color_buf, id_buf, kind_buf, icon_count)
            self._icons_uploaded = True
        elif self._icons_uploaded:
            native.clear_component_gizmo_icons()
            self._icons_uploaded = False

        upload_finished = time.perf_counter()
        self._last_observation = GizmoCollectionObservation(
            scene_count=len(scenes),
            python_callbacks=python_callbacks,
            builtin_callbacks=builtin_callbacks,
            skipped_inactive=skipped_inactive,
            skipped_disabled=skipped_disabled,
            cpu_vertices=cpu_vertices,
            cpu_line_indices=cpu_line_indices,
            cpu_draws=cpu_draws,
            resident_vertices=sum(int(item[2]) for item in resident),
            resident_draws=len(resident),
            icons=icon_count,
            collect_ms=(collect_finished - frame_started) * 1000.0,
            pack_ms=(pack_finished - collect_finished) * 1000.0,
            upload_ms=(upload_finished - pack_finished) * 1000.0,
            total_ms=(upload_finished - frame_started) * 1000.0,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_selected_ancestor_ids(self, scene, selected_id: int):
        """Return selected + descendant IDs, reusing the stable scene relation."""
        selected_id = int(selected_id or 0)
        key = (self._cache_generation, selected_id)
        if key != self._selection_cache_key:
            self._selected_ancestor_ids = (
                frozenset(self._build_ancestor_set(scene, selected_id))
                if selected_id
                else frozenset()
            )
            self._selection_cache_key = key
        return self._selected_ancestor_ids

    @staticmethod
    def _register_python_component_icon(comp, go_id: int, icon_color) -> None:
        """Queue the scene billboard icon for a Python component instance."""
        try:
            transform = comp.game_object.get_transform()
        except Exception as exc:
            _log_gizmo_warning(f"Gizmo: failed to read transform for icon: {exc}")
            return
        if transform is None:
            return
        pos = transform.position
        Gizmos.draw_icon(
            (pos.x, pos.y, pos.z), go_id, icon_color,
            icon_kind=getattr(comp, '_gizmo_icon_kind', ICON_KIND_DEFAULT),
        )

    def _get_icon_instances(self, scenes, type_name: str) -> list:
        """Return the cached list of GOs that carry *type_name*.

        On the first call for a given type (or after ``invalidate_cache()``),
        walks all scene objects once to find matching GOs.  Subsequent calls
        return the cached list in O(1).
        """
        if type_name in self._cache_built:
            return self._icon_cache.get(type_name, [])

        if not isinstance(scenes, (tuple, list)):
            scenes = (scenes,)

        # Build this type's cache entry via a one-time loaded-world walk.
        result: List = []
        for scene in scenes:
            for go in scene.get_all_objects():
                if go.get_cpp_component(type_name) is not None:
                    result.append(go)

        self._icon_cache[type_name] = result
        self._cache_built.add(type_name)
        return result

    @staticmethod
    def _build_ancestor_set(scene, selected_id: int) -> set:
        """Build a set containing the selected object ID and all its descendant IDs."""
        result = set()
        if not selected_id or scene is None:
            return result

        selected_go = scene.find_by_id(selected_id)
        if selected_go is None:
            result.add(selected_id)
            return result

        # Iterative DFS to collect selected GO + all descendants
        stack = [selected_go]
        while stack:
            go = stack.pop()
            gid = go.id
            result.add(gid)
            children = go.get_children()
            if children:
                stack.extend(children)

        return result

    @staticmethod
    def _resolve_class_value(cls_obj, name: str, default):
        value = getattr(cls_obj, name, default)
        if isinstance(value, SerializedFieldDescriptor):
            return value.metadata.default
        return value
