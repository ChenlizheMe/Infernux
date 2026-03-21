"""
RenderStack — Scene-level rendering configuration component.

RenderStack is a scene-singleton InfComponent that manages:
- The active RenderPipeline (topology skeleton + injection points)
- All mounted RenderPass instances (user effects + built-in passes)
- Graph construction: combines pipeline topology with injected passes

Architecture::

    RenderStack (InfComponent, scene singleton)
      ├── selected_pipeline: RenderPipeline  (defines topology skeleton)
      └── pass_entries: List[PassEntry]      (user-mounted passes)

    Each frame:
      1. RenderStack.render(context, camera)
      2. Lazy-build graph if invalidated
      3. context.apply_graph(desc) + context.submit_culling(culling)

Build flow (Section 7.1)::

    graph = RenderGraph("Pipeline+Stack")
    bus = ResourceBus()
    pipeline.define_topology(graph, bus, callback)
      └── callback triggers _inject_passes_at for each injection point
    graph.set_output(bus.get("color"))
    graph.build() → RenderGraphDescription

Usage::

    # In a scene setup script
    stack = game_object.add_component(RenderStack)
    stack.set_pipeline("Default Forward")
    stack.add_pass(BloomPass())
"""

from __future__ import annotations

import json as _json
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from InfEngine.components.component import InfComponent
from InfEngine.components.decorators import disallow_multiple, add_component_menu
from InfEngine.renderstack.injection_point import InjectionPoint
from InfEngine.renderstack.resource_bus import ResourceBus

if TYPE_CHECKING:
    from InfEngine.rendergraph.graph import RenderGraph
    from InfEngine.renderstack.render_pass import RenderPass


@dataclass
class PassEntry:
    """RenderStack 中一个 Pass 槽位的持久化数据。"""

    render_pass: "RenderPass"
    enabled: bool = True
    order: int = 0
    # injection_point 从 render_pass.injection_point 读取


@disallow_multiple
@add_component_menu("Rendering/RenderStack")
class RenderStack(InfComponent):
    """场景级渲染配置组件。

    管理当前场景使用的 RenderPipeline 和所有挂载的 RenderPass。
    每个场景最多一个 RenderStack 实例。

    Attributes:
        pipeline_class_name: 选中的 Pipeline 类名（空 = 默认管线）。
        pass_entries: 挂载的 Pass 列表。
    """

    _component_category_ = "Rendering"

    # ---- Class-level singleton (scene-global) ----
    _active_instance: Optional["RenderStack"] = None

    @classmethod
    def instance(cls) -> Optional["RenderStack"]:
        """Return the active scene-scoped RenderStack, or None."""
        return cls._active_instance

    # ---- 序列化字段 ----
    pipeline_class_name: str = ""
    mounted_passes_json: str = ""   # 持久化 pass 配置
    pipeline_params_json: str = ""  # 按管线持久化参数快照

    # ---- 运行时（不序列化） ----
    _pipeline = None  # Optional[RenderPipeline]
    _graph_desc = None  # cached RenderGraphDescription
    _resource_bus: Optional[ResourceBus] = None
    _build_failed: bool = False  # True after a build error; cleared by invalidate_graph()
    _pipeline_module = None  # module object for watchdog hot-reload subscription
    _pass_entries: List[PassEntry] = None  # initialized properly in awake()
    _pipeline_param_store: Dict[str, Dict[str, object]] = None
    _pipeline_catalog_signature: tuple = ()
    _topology_probe_cache = None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def awake(self) -> None:
        """组件初始化。验证场景单例约束。

        如果当前场景已存在另一个 RenderStack，输出错误日志
        并尝试移除自身（只允许一个 RenderStack per scene）。
        """
        # Initialize instance-level fields (not serialized), but do NOT
        # stomp values already restored by on_after_deserialize().
        if self._pass_entries is None:
            self._pass_entries = []
        if self._pipeline_param_store is None:
            self._pipeline_param_store = {}
        self._pipeline_catalog_signature = ()
        self._register_pipeline_catalog_reload()
        self._sync_pipeline_catalog()

        if RenderStack._active_instance is not None and RenderStack._active_instance is not self:
            go_name = getattr(self.game_object, "name", "???")
            ex_go = RenderStack._active_instance.game_object
            ex_go_name = getattr(ex_go, "name", "???") if ex_go else "???"
            print(
                f"[RenderStack] Scene already has a RenderStack on "
                f"'{ex_go_name}'. Only one per scene is allowed. "
                f"Removing duplicate from '{go_name}'.",
                file=sys.stderr,
            )
            self.game_object.remove_component(self)
            return
        RenderStack._active_instance = self

    def on_destroy(self) -> None:
        """清理 pipeline 和 pass 资源。"""
        self._unregister_pipeline_catalog_reload()
        if RenderStack._active_instance is self:
            RenderStack._active_instance = None
        if self._pipeline is not None and hasattr(self._pipeline, "dispose"):
            self._pipeline.dispose()
        self._pipeline = None
        self._graph_desc = None
        self._resource_bus = None

    # ------------------------------------------------------------------
    # Serialization hooks
    # ------------------------------------------------------------------

    def on_before_serialize(self) -> None:
        """Save pass_entries into mounted_passes_json."""
        self._save_current_pipeline_params()
        entries = []
        for e in self._pass_entries:
            entry_data = {
                "class": type(e.render_pass).__name__,
                "enabled": e.enabled,
                "order": e.order,
            }
            # FullScreenEffect: also persist tuneable parameters
            from InfEngine.renderstack.fullscreen_effect import FullScreenEffect
            if isinstance(e.render_pass, FullScreenEffect):
                entry_data["params"] = e.render_pass.get_params_dict()
            entries.append(entry_data)
        # Only overwrite when we actually have runtime entries; when
        # _pass_entries is empty (e.g. discover_passes() failed) preserve the
        # existing serialised data so the play-mode snapshot keeps the values.
        if entries:
            self.mounted_passes_json = _json.dumps(entries)
        self.pipeline_params_json = _json.dumps(self._pipeline_param_store) if self._pipeline_param_store else ""

    def on_after_deserialize(self) -> None:
        """Recreate pass_entries from mounted_passes_json."""
        # Register as the active instance so that the fast-path in
        # RenderStackPipeline._find_render_stack works even in edit mode
        # (where awake() is not called).
        if RenderStack._active_instance is None:
            RenderStack._active_instance = self

        # Ensure _pass_entries is initialized (may be called before awake())
        if self._pass_entries is None:
            self._pass_entries = []
        else:
            # Scene / project reopen can invoke deserialization multiple times
            # during object reconstruction. Always rebuild from JSON instead of
            # appending onto previously restored runtime state.
            self._pass_entries.clear()
        if self._pipeline_param_store is None:
            self._pipeline_param_store = {}

        if self.pipeline_params_json:
            try:
                data = _json.loads(self.pipeline_params_json)
                if isinstance(data, dict):
                    self._pipeline_param_store = data
            except (ValueError, _json.JSONDecodeError):
                self._pipeline_param_store = {}

        if not self.mounted_passes_json:
            return
        from InfEngine.renderstack.discovery import discover_passes

        all_passes = discover_passes()
        items = _json.loads(self.mounted_passes_json)
        restored_keys = set()
        for item in items:
            cls_name = item.get("class", "")
            cls = all_passes.get(cls_name)
            if cls is None:
                # Also try name→class mapping by class __name__
                for pcls in all_passes.values():
                    if pcls.__name__ == cls_name:
                        cls = pcls
                        break
            if cls is None:
                print(
                    f"[RenderStack] Cannot restore pass '{cls_name}' "
                    f"— class not found.",
                    file=sys.stderr,
                )
                continue
            inst = cls()
            # FullScreenEffect: restore tuneable parameters
            from InfEngine.renderstack.fullscreen_effect import FullScreenEffect
            if isinstance(inst, FullScreenEffect) and "params" in item:
                inst.set_params_dict(item["params"])
            entry = PassEntry(
                render_pass=inst,
                enabled=item.get("enabled", True),
                order=item.get("order", 0),
            )
            inst.enabled = entry.enabled
            key = (inst.injection_point, inst.name)
            if key in restored_keys:
                continue
            restored_keys.add(key)
            self._pass_entries.append(entry)

        # Validate injection points (warn only — don't drop entries, because
        # the pipeline might not be loaded yet or may change later).
        if self._pass_entries:
            try:
                valid_points = {p.name for p in self.injection_points}
                for entry in self._pass_entries:
                    ip = entry.render_pass.injection_point
                    if ip not in valid_points:
                        print(
                            f"[RenderStack] Restored pass '{entry.render_pass.name}' "
                            f"has unknown injection_point '{ip}'. "
                            f"Valid: {sorted(valid_points)}",
                            file=sys.stderr,
                        )
            except (RuntimeError, AttributeError):
                pass  # pipeline not available yet — skip validation
            self.invalidate_graph()

    # ==================================================================
    # Pipeline management
    # ==================================================================

    @staticmethod
    def discover_pipelines() -> Dict[str, type]:
        """扫描项目中所有 RenderPipeline 子类。

        Returns:
            ``{display_name: class}`` 字典。
        """
        from InfEngine.renderstack.discovery import discover_pipelines

        return discover_pipelines()

    def set_pipeline(self, pipeline_class_name: str) -> None:
        """切换渲染管线。传空字符串使用默认管线。"""
        if self.pipeline_class_name == pipeline_class_name:
            return
        self._save_current_pipeline_params()
        self.pipeline_class_name = pipeline_class_name
        self._pipeline = None
        self._cached_ips = None
        self.invalidate_graph()

    @property
    def pipeline(self):  # -> RenderPipeline
        """当前使用的管线实例（懒创建）。"""
        if self._pipeline is None:
            self._pipeline = self._create_pipeline()
            self._restore_pipeline_params(self._pipeline)
            # Wire back-reference so pipeline param changes can
            # invalidate the graph via self._render_stack.
            if hasattr(self._pipeline, '_render_stack'):
                self._pipeline._render_stack = self
        return self._pipeline

    @property
    def injection_points(self) -> List[InjectionPoint]:
        """当前管线定义的所有注入点（只读，供 Editor 显示）。

        注入点在 ``build_graph()`` 中由 ``define_topology()``
        产生。首次调用时会做一次 dry-run 构建。
        """
        if not hasattr(self, "_cached_ips") or self._cached_ips is None:
            g = self._build_full_topology_probe()
            self._cached_ips = g.injection_points
        return self._cached_ips

    @property
    def pass_entries(self) -> List[PassEntry]:
        """Mounted pass entries (read-only view for UI/integration code)."""
        if self._pass_entries is None:
            self._pass_entries = []
        return self._pass_entries

    # ==================================================================
    # Pass management
    # ==================================================================

    def add_pass(self, render_pass: "RenderPass") -> bool:
        """将一个 RenderPass 挂载到 RenderStack。

        自动根据 ``render_pass.injection_point`` 分配到对应注入点。

        Returns:
            ``False`` 如果注入点不存在。
        """
        valid_points = {p.name for p in self.injection_points}
        if render_pass.injection_point not in valid_points:
            import logging
            logging.getLogger("InfEngine.RenderStack").warning(
                "RenderPass '%s' has unknown injection_point '%s'. "
                "Valid points: %s",
                render_pass.name,
                render_pass.injection_point,
                ", ".join(sorted(valid_points)),
            )
            return False
        for entry in self._pass_entries:
            if (entry.render_pass.injection_point == render_pass.injection_point and
                    entry.render_pass.name == render_pass.name):
                return False
        entry = PassEntry(
            render_pass=render_pass,
            enabled=render_pass.enabled,
            order=render_pass.default_order,
        )
        self._pass_entries.append(entry)
        self.invalidate_graph()
        return True

    def remove_pass(self, pass_name: str) -> bool:
        """移除一个已挂载的 RenderPass（按 name 匹配）。"""
        for i, entry in enumerate(self._pass_entries):
            if entry.render_pass.name == pass_name:
                self._pass_entries.pop(i)
                self.invalidate_graph()
                return True
        return False

    def set_pass_enabled(self, pass_name: str, enabled: bool) -> None:
        """启用 / 禁用一个 Pass。"""
        for entry in self._pass_entries:
            if entry.render_pass.name == pass_name:
                entry.enabled = enabled
                entry.render_pass.enabled = enabled
                self.invalidate_graph()
                return

    def reorder_pass(self, pass_name: str, new_order: int) -> None:
        """调整 Pass 在同一注入点内的 order。"""
        for entry in self._pass_entries:
            if entry.render_pass.name == pass_name:
                entry.order = new_order
                self.invalidate_graph()
                return

    def move_pass_before(self, dragged_name: str, target_name: str) -> None:
        """将 dragged_name 移动到 target_name 之前（同注入点内）。

        通过重新分配 order 值实现稳定排序，不改变其他 pass 的相对顺序。
        """
        dragged_entry = None
        target_entry = None
        for e in self._pass_entries:
            if e.render_pass.name == dragged_name:
                dragged_entry = e
            if e.render_pass.name == target_name:
                target_entry = e
        if dragged_entry is None or target_entry is None:
            return
        if dragged_entry.render_pass.injection_point != target_entry.render_pass.injection_point:
            return

        ip = dragged_entry.render_pass.injection_point
        entries = self.get_passes_at(ip)

        # Remove dragged, insert before target, reassign orders
        ordered_names = [e.render_pass.name for e in entries if e.render_pass.name != dragged_name]
        try:
            idx = ordered_names.index(target_name)
        except ValueError:
            return
        ordered_names.insert(idx, dragged_name)

        # Reassign orders with stable spacing
        name_to_entry = {e.render_pass.name: e for e in self._pass_entries}
        for i, name in enumerate(ordered_names):
            entry = name_to_entry.get(name)
            if entry is not None:
                entry.order = (i + 1) * 10

        self.invalidate_graph()

    def get_passes_at(self, injection_point: str) -> List[PassEntry]:
        """获取某个注入点下的所有 Pass（按 order 排序）。"""
        match_names = {injection_point}

        entries = [
            e
            for e in self._pass_entries
            if e.render_pass.injection_point in match_names
        ]
        entries.sort(key=lambda e: e.order)
        return entries

    # ==================================================================
    # Graph construction
    # ==================================================================

    def _build_full_topology_probe(self):
        """Return a RenderGraph with the pipeline-defined topology.

        Used by ``injection_points`` and the inspector renderer to display
        the same sequence the pipeline explicitly defines.
        """
        if self._topology_probe_cache is not None:
            return self._topology_probe_cache

        from InfEngine.rendergraph.graph import RenderGraph
        g = RenderGraph("_FullTopologyProbe")
        self.pipeline.define_topology(g)
        # Keep the inspector probe consistent with build(): post-process
        # injection points are guaranteed to exist even when Screen UI is off.
        _auto_res = {"color"}
        if not g.has_injection_point("before_post_process"):
            g.injection_point("before_post_process", resources=_auto_res)
        if not g.has_injection_point("after_post_process"):
            g.injection_point("after_post_process", resources=_auto_res)
        self._topology_probe_cache = g
        return g

    def invalidate_graph(self) -> None:
        """标记 graph 需要重建（Pass 变更时自动调用）。

        以下操作会自动调用此方法：
        - ``add_pass()`` / ``remove_pass()``
        - ``set_pass_enabled()`` / ``reorder_pass()``
        - ``set_pipeline()``
        """
        self._graph_desc = None
        self._build_failed = False  # allow retry after explicit invalidation
        self._topology_probe_cache = None

    def build_graph(self):  # -> RenderGraphDescription
        """构建完整的 RenderGraph。

        Steps:
            1. ``graph = RenderGraph("Pipeline+Stack")``
            2. 设置 injection callback → 在每个注入点处触发 pass 注入
            3. ``pipeline.define_topology(graph)``
            4. 验证：注入点不得出现在第一个 pass 之前
            5. ``graph.set_output(bus.get("color") or "color")``
            6. ``graph.build() → RenderGraphDescription``

        Returns:
            Compiled ``RenderGraphDescription`` ready for
            ``context.apply_graph()``.
        """
        from InfEngine.rendergraph.graph import RenderGraph

        # Guard: ensure pass_entries is initialized even if awake() hasn't run yet
        if self._pass_entries is None:
            self._pass_entries = []

        graph = RenderGraph("Pipeline+Stack")
        bus = ResourceBus()
        self._resource_bus = bus

        # Callback: invoked every time pipeline calls graph.injection_point()
        def on_injection_point(point_name: str) -> None:
            # Sync bus with all graph textures (add any new ones)
            for tex in graph._textures:
                if not bus.has(tex.name):
                    bus.set(tex.name, tex)
            self._inject_passes_at(point_name, graph, bus)

        graph._injection_callback = on_injection_point

        # Pipeline populates graph with passes + injection points
        self.pipeline.define_topology(graph)

        # Ensure before/after_post_process injection points exist WHILE the
        # callback is still active. graph.build() also auto-injects these,
        # but that happens after the callback is detached — effects targeting
        # these points would never be injected.  Calling injection_point()
        # here triggers the callback so mounted effects are properly inserted.
        _auto_res = {"color"}
        if not graph.has_injection_point("before_post_process"):
            graph.injection_point("before_post_process", resources=_auto_res)
        if not graph.has_injection_point("after_post_process"):
            graph.injection_point("after_post_process", resources=_auto_res)

        # Validate: no injection point before first pass
        graph.validate_no_ip_before_first_pass()

        # If post-processing effects redirected "color" to a different
        # texture, blit the result back to the original camera target
        # (backbuffer) so it gets presented to the screen.
        original_color = graph.get_texture("color")
        final_color = bus.get("color")
        if (final_color is not None
                and original_color is not None
                and final_color is not original_color):
            # Move _ScreenUI_Overlay (if present) so it renders AFTER the
            # final blit — otherwise the blit overwrites the overlay UI.
            overlay_pass = graph.remove_pass("_ScreenUI_Overlay")

            with graph.add_pass("_FinalCompositeBlit") as p:
                p.set_texture("_SourceTex", final_color)
                p.write_color(original_color)
                p.fullscreen_quad("fullscreen_blit")

            # Re-append overlay after the blit
            if overlay_pass is not None:
                graph.append_pass(overlay_pass)

            graph.set_output(original_color)
        elif final_color is not None:
            graph.set_output(final_color)
        else:
            graph.set_output("color")

        return graph.build()

    def render(self, context, camera) -> None:
        """每帧渲染入口。由 RenderStackPipeline 调用。

        Lazy-builds the graph on first call or after invalidation,
        then applies the compiled graph and submits culling results.

        Args:
            context: The render context provided by the engine.
            camera: The camera to render from.
        """
        # Guard: ensure pass_entries is initialized even if awake() hasn't run yet
        if self._pass_entries is None:
            self._pass_entries = []

        context.setup_camera_properties(camera)
        culling = context.cull(camera)

        # Lazy build graph topology (skip if last build failed)
        if self._graph_desc is None and not self._build_failed:
            self._graph_desc = self.build_graph()

        if self._graph_desc is None:
            # Build previously failed; skip rendering until hot-reload fixes it
            context.submit_culling(culling)
            return

        context.apply_graph(self._graph_desc)
        context.submit_culling(culling)

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _create_pipeline(self):  # -> RenderPipeline
        """根据 ``pipeline_class_name`` 实例化管线。

        空名称或名称未找到时，返回 ``DefaultForwardPipeline``。
        订阅 watchdog 回调以支持文件保存后的 hot-reload。
        """
        import inspect, os
        from InfEngine.renderstack.default_forward_pipeline import (
            DefaultForwardPipeline,
        )

        if not self.pipeline_class_name:
            self._unregister_pipeline_reload()
            return DefaultForwardPipeline()

        pipelines = self.discover_pipelines()
        cls = pipelines.get(self.pipeline_class_name)
        if cls is None:
            warnings.warn(
                f"[RenderStack] Pipeline '{self.pipeline_class_name}' "
                f"not found. Available: {list(pipelines.keys())}. "
                f"Falling back to DefaultForwardPipeline.",
                stacklevel=2,
            )
            self.pipeline_class_name = ""
            self._unregister_pipeline_reload()
            return DefaultForwardPipeline()

        pipeline = cls()
        # Register watchdog callback for hot-reload
        self._register_pipeline_reload(cls)
        return pipeline

    def _register_pipeline_reload(self, pipeline_cls) -> None:
        """Subscribe to watchdog file-change events for the pipeline's source file."""
        import sys as _sys
        mod = _sys.modules.get(pipeline_cls.__module__)
        if mod is None:
            return
        src = getattr(mod, '__file__', None)
        if not src:
            return
        self._pipeline_module = mod
        from InfEngine.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.register_script_reload_callback(src, self._on_pipeline_file_changed)

    def _unregister_pipeline_reload(self) -> None:
        """Unsubscribe from watchdog callbacks."""
        from InfEngine.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.unregister_script_reload_callback(self._on_pipeline_file_changed)
        self._pipeline_module = None

    def _on_pipeline_file_changed(self, file_path: str) -> None:
        """Watchdog callback — called on main thread when pipeline source is saved."""
        import importlib
        from InfEngine.renderstack.discovery import invalidate_discovery_cache
        mod = self._pipeline_module
        if mod is None:
            return
        print(f"[RenderStack] Pipeline file changed, reloading...", file=sys.stderr)
        self._save_current_pipeline_params()
        invalidate_discovery_cache()
        importlib.reload(mod)
        self._pipeline = None   # re-instantiate on next .pipeline access
        self.invalidate_graph() # clears _build_failed + _graph_desc
        print(f"[RenderStack] Pipeline reloaded.", file=sys.stderr)

    def _sync_pipeline_catalog(self) -> None:
        """Refresh available pipeline catalog and enforce fallback policy."""
        names = set(self.discover_pipelines().keys())
        signature = tuple(sorted(names))
        if signature == self._pipeline_catalog_signature:
            return

        self._pipeline_catalog_signature = signature

        current = self.pipeline_class_name
        if current and current not in names:
            warnings.warn(
                f"[RenderStack] Pipeline '{current}' was removed. Falling back to DefaultForwardPipeline.",
                stacklevel=2,
            )
            self.set_pipeline("")
            return

        # Refresh pipeline type on catalog changes so newly edited classes can be re-instantiated.
        if self._pipeline is not None:
            self._save_current_pipeline_params()
            self._pipeline = None
            self._cached_ips = None
            self.invalidate_graph()

    def _register_pipeline_catalog_reload(self) -> None:
        """Subscribe to watchdog-driven script catalog changes."""
        from InfEngine.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.register_script_catalog_callback(self._on_script_catalog_changed)

    def _unregister_pipeline_catalog_reload(self) -> None:
        """Unsubscribe from watchdog-driven script catalog changes."""
        from InfEngine.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.unregister_script_catalog_callback(self._on_script_catalog_changed)

    def _on_script_catalog_changed(self, file_path: str, event_type: str) -> None:
        """ResourcesManager callback for create/delete/move/modify of python scripts."""
        from InfEngine.renderstack.discovery import invalidate_discovery_cache
        invalidate_discovery_cache()
        self._sync_pipeline_catalog()

    def _pipeline_key(self, pipeline_name: str) -> str:
        return pipeline_name if pipeline_name else "__default__"

    def _save_current_pipeline_params(self) -> None:
        if self._pipeline_param_store is None:
            self._pipeline_param_store = {}
        if self._pipeline is None:
            return
        try:
            from InfEngine.components.serialized_field import get_serialized_fields
            from enum import Enum

            key = self._pipeline_key(self.pipeline_class_name)
            fields = get_serialized_fields(self._pipeline.__class__)
            params = {}
            for field_name in fields.keys():
                value = getattr(self._pipeline, field_name, None)
                if isinstance(value, Enum):
                    params[field_name] = {"__enum_name__": value.name}
                else:
                    params[field_name] = value
            self._pipeline_param_store[key] = params
        except (ImportError, RuntimeError, AttributeError):
            return

    def _restore_pipeline_params(self, pipeline) -> None:
        if self._pipeline_param_store is None:
            self._pipeline_param_store = {}
        try:
            from InfEngine.components.serialized_field import get_serialized_fields, FieldType
        except ImportError:
            return

        key = self._pipeline_key(self.pipeline_class_name)
        saved = self._pipeline_param_store.get(key)
        if not isinstance(saved, dict):
            return

        fields = get_serialized_fields(pipeline.__class__)
        pipeline._inf_deserializing = True
        try:
            for field_name, meta in fields.items():
                if field_name not in saved:
                    continue
                value = saved[field_name]
                try:
                    if meta.field_type == FieldType.ENUM and isinstance(value, dict) and "__enum_name__" in value:
                        enum_name = value.get("__enum_name__", "")
                        enum_cls = meta.enum_type
                        if enum_cls is not None and enum_name in enum_cls.__members__:
                            setattr(pipeline, field_name, enum_cls[enum_name])
                            continue
                    setattr(pipeline, field_name, value)
                except (AttributeError, TypeError, ValueError):
                    continue
        finally:
            pipeline._inf_deserializing = False

    def _inject_passes_at(
        self,
        point_name: str,
        graph: "RenderGraph",
        bus: ResourceBus,
    ) -> None:
        """在指定注入点按 order 注入所有 enabled Pass。

        Responsibilities:
            1. 获取此注入点的 PassEntry 列表（按 order 排序）
            2. 验证每个 Pass 的资源需求（validate + warn on failure）
            3. 调用 ``pass.inject(graph, bus)``

        Args:
            point_name: 注入点名称。
            graph: 当前构建中的 RenderGraph。
            bus: 资源总线。
        """
        entries = self.get_passes_at(point_name)
        enabled = [e for e in entries if e.enabled]

        if not enabled:
            return

        for entry in enabled:
            rp = entry.render_pass

            # Validate resource requirements before injection
            errors = rp.validate(bus.available_resources)
            if errors:
                for err in errors:
                    print(f"[RenderStack] {err}", file=sys.stderr)
                print(
                    f"[RenderStack] Skipping pass '{rp.name}' at "
                    f"'{point_name}' due to validation errors.",
                    file=sys.stderr,
                )
                continue

            # Warn on creates collision
            for res_name in rp.creates:
                if bus.has(res_name):
                    warnings.warn(
                        f"[RenderStack] Pass '{rp.name}' creates "
                        f"resource '{res_name}' which already exists "
                        f"in bus. It will be overwritten.",
                        stacklevel=2,
                    )

            rp.inject(graph, bus)


