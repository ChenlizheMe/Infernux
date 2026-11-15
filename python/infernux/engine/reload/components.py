"""Replace Python instances while the native world retains its own state."""

from infernux.engine.path_utils import path_key
from infernux.engine._play_mode_serialization import PlayModeSerializationMixin


class ComponentAdapter(PlayModeSerializationMixin):
    def __init__(self, scene_provider):
        self._scenes = scene_provider
        self._guids = {}

    def guid_for_path(self, path):
        from infernux.core.assets import AssetManager

        return self._guids.get(path_key(path)) or AssetManager._get_guid_from_path(path)

    def retire(self, paths):
        from infernux.components.missing_script import MissingScript, create_missing_script_component
        from infernux.components.registry import component_types_for_script_path
        from infernux.execution import close_component_execution

        keys = {path_key(path) for path in paths}
        declarations = {
            path_key(path): tuple(cls.__name__ for cls in component_types_for_script_path(path))
            for path in paths
        }
        retiring = []
        for scene in self._scenes():
            for obj in scene.get_all_objects():
                for component in tuple(obj.get_py_components()):
                    path = getattr(component, "_script_path", "")
                    if not path or path_key(path) not in keys or isinstance(component, MissingScript):
                        continue
                    retiring.append((obj, component, path))
        # Disable the whole affected set before serialization or user cleanup can fail.
        for _obj, component, _path in retiring:
            component.__dict__.setdefault("_reload_enabled", component.enabled)
            component.enabled = False
        for obj, component, path in retiring:
            state = component.__dict__.get("_reload_state")
            if state is None:
                state = self._serialize_py_component(component)
                state["enabled"] = component._reload_enabled
                state["awake_called"] = component._awake_called
                state["has_started"] = component._has_started
                state["execution_order"] = component.execution_order
                state["declaration_names"] = declarations[path_key(path)]
            self._guids[path_key(path)] = state["script_guid"]
            component.enabled = False
            component.stop_all_coroutines()
            close_component_execution(component)
            missing = create_missing_script_component(
                type_name=state["type_name"],
                script_guid=state["script_guid"],
                type_guid=state["type_guid"],
                module_name=state["module_name"],
                qualified_name=state["qualified_name"],
                fields=state["fields"],
                error="Current script is being reloaded",
            )
            missing._reload_state = state
            missing._script_path = path
            missing._component_id = state["component_id"]
            missing._enabled = False
            if obj.replace_py_component(component, missing) is not missing:
                raise RuntimeError(f"Cannot retire script on GameObject {obj.id}")

    def prepare(self, modules, owner):
        from infernux.components.missing_script import MissingScript
        from infernux.components.component import InxComponent

        prepared = []
        by_name, by_guid = {}, {}
        for name, module in modules.items():
            declarations = {
                value.__name__: value for value in vars(module).values()
                if isinstance(value, type) and issubclass(value, InxComponent)
                and value is not InxComponent and value.__module__ == name
            }
            if declarations:
                entry = (module.__file__, declarations)
                by_name[name] = entry
                guid = self.guid_for_path(module.__file__)
                if guid:
                    by_guid[guid] = entry
        for scene in self._scenes():
            for obj in scene.get_all_objects():
                for missing in tuple(obj.get_py_components()):
                    if not isinstance(missing, MissingScript):
                        continue
                    state = getattr(missing, "_reload_state", None)
                    if state is None:
                        state = self._serialize_py_component(missing)
                        state["enabled"] = missing.__dict__.get("_reload_enabled", state["enabled"])
                    entry = by_name.get(type(missing).__module__) or by_guid.get(state["script_guid"])
                    if entry is None:
                        continue
                    script_path, declarations = entry
                    candidate = declarations.get(state["type_name"])
                    if candidate is None:
                        # A rename is unambiguous only across the complete file,
                        # including declarations that are not mounted in a Scene.
                        previous = set(state.get("declaration_names", (state["type_name"],)))
                        removed = previous - declarations.keys()
                        added = declarations.keys() - previous
                        if removed == {state["type_name"]} and len(added) == 1:
                            candidate = declarations[next(iter(added))]
                    if candidate is None:
                        missing._broken_error = f"Current source does not declare {state['type_name']}"
                        continue
                    instance = candidate()
                    instance._script_path = script_path
                    instance._reload_state = state
                    self._apply_py_component_state(instance, state)
                    instance._enabled = False
                    prepared.append((obj, missing, instance, state))
        return prepared

    def bind(self, prepared):
        # Complete every native binding before any user initialization runs.
        for obj, previous, instance, state in prepared:
            if obj.replace_py_component(previous, instance) is not instance:
                raise RuntimeError(f"Cannot bind current script on GameObject {obj.id}")
            instance._awake_called = False
            instance._has_started = False
            instance.execution_order = state.get("execution_order", 0)

    def initialize(self, prepared, owner):
        # Domain initialization (such as preload services) finishes for the
        # whole publication before any component lifecycle callback runs.
        pass

    def after_deserialize(self, prepared, owner):
        from infernux.execution import component_execution, execution_phase

        with owner.enter(), execution_phase():
            for _obj, _previous, instance, _state in prepared:
                instance._call_on_after_deserialize()

    def awake(self, prepared, owner):
        from infernux.lib import SceneManager
        from infernux.execution import component_execution, execution_phase

        playing = SceneManager.instance().is_playing()
        with owner.enter(), execution_phase():
            for _obj, _previous, instance, state in prepared:
                if state.get("awake_called") or (playing and instance.game_object.active_in_hierarchy):
                    with component_execution(instance):
                        instance.awake()
                    instance._awake_called = True

    def start(self, prepared, owner):
        from infernux.lib import SceneManager
        from infernux.execution import component_execution, execution_phase

        playing = SceneManager.instance().is_playing()
        with owner.enter(), execution_phase():
            for _obj, _previous, instance, state in prepared:
                if state.get("has_started") or (playing and instance.game_object.active_in_hierarchy and state["enabled"]):
                    with component_execution(instance):
                        instance.start()
                    instance._has_started = True

    def admit(self, prepared, owner):
        for _obj, _previous, instance, _state in prepared:
            instance._reload_owner.active = True

    def activate(self, prepared):
        for _obj, _previous, instance, state in prepared:
            instance.enabled = state["enabled"]
            error = instance.__dict__.get("_initialization_error")
            if error is not None:
                raise error
            instance.__dict__.pop("_reload_state", None)

    def fail(self, paths, error):
        from infernux.components.missing_script import MissingScript

        keys = {path_key(path) for path in paths}
        for scene in self._scenes():
            for obj in scene.get_all_objects():
                for component in obj.get_py_components():
                    if path_key(getattr(component, "_script_path", "")) in keys:
                        component.__dict__.setdefault("_reload_enabled", component.enabled)
                        component.enabled = False
                        if isinstance(component, MissingScript):
                            component._broken_error = error
