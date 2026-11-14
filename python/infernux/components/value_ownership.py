"""Internal immutable scene-reference provenance and explicit decode boundaries."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

UNSET = object()
_decode_world = ContextVar('infernux_reference_decode_world', default=UNSET)


def scene_owner(scene):
    owner = int(scene._runtime_world_id)
    if owner:
        return owner
    if scene.is_preview:
        from infernux.lib import SceneManager
        return int(SceneManager.instance()._authoring_runtime_world_id)
    raise RuntimeError('Scene has no reference owner')


def object_owner(game_object):
    return scene_owner(game_object.scene)


def capture_id_owner():
    """Concrete nonzero raw-ID construction has explicit current-World semantics."""
    from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
    from infernux.lib import SceneManager
    assert_runtime_owner_thread()
    identity = int(SceneManager.instance()._current_runtime_world_id)
    if not identity:
        raise RuntimeError('A scene reference requires an active World')
    return identity


def reference_owner(target, identifier, explicit):
    if target is not None:
        return object_owner(target)
    if not identifier:
        return None
    return capture_id_owner() if explicit is UNSET else explicit


def typed_creation_owner():
    owner = _decode_world.get()
    if owner is not UNSET:
        return owner
    from infernux.application import Application
    if Application._current_engine() is None:
        return None  # Pure source data remains valid before startup/after exit.
    return capture_id_owner()


@contextmanager
def value_world(owner):
    token = _decode_world.set(owner)
    try:
        if owner is None:
            yield  # Explicit source validation uses its type catalog, not a graph.
        else:
            from infernux.lib import _Infernux
            with _Infernux._RuntimeWorldScope(owner):
                yield
    finally:
        _decode_world.reset(token)


def source_values():
    return _decode_world.get() is None


def resolve_in_value_world(owner, operation):
    context = value_world(owner)
    try:
        context.__enter__()
    except RuntimeError as error:
        # PythonWorldScope::Enter has one explicit retired-owner error. Other
        # scope/thread errors and resolver errors must remain visible.
        if str(error) != 'RuntimeWorld has retired':
            raise
        return None
    try:
        return operation()  # Resolver exceptions are not swallowed.
    finally:
        context.__exit__(None, None, None)


def receiver_owner(instance):
    cleanup = instance.__dict__.get('_native_cleanup_binding')
    if cleanup is not None:
        return object_owner(cleanup[1])
    native = instance.__dict__.get('_cpp_component')
    if native is not None:
        return object_owner(native.game_object)
    attached_object = instance.__dict__.get('_game_object')
    if attached_object is not None:
        return object_owner(attached_object)
    return instance.__dict__.get('_value_owner_world_id')


def owned_operation(function):
    @wraps(function)
    def invoke(instance, *args, **kwargs):
        if source_values():
            return function(instance, *args, **kwargs)
        owner = receiver_owner(instance)
        if owner is None:
            return function(instance, *args, **kwargs)
        with value_world(owner):
            return function(instance, *args, **kwargs)
    return invoke


def is_owned_reference_field(metadata):
    """Declaration-time/set/attach classification; never a field-read walk."""
    from .fields import FieldType
    field_type = metadata.element_type if metadata.field_type == FieldType.LIST else metadata.field_type
    if field_type in (FieldType.GAME_OBJECT, FieldType.COMPONENT):
        return True
    if field_type == FieldType.ASSET and metadata.asset_type:
        from infernux.core.asset_reference_types import asset_type_registry
        descriptor = asset_type_registry.require(metadata.asset_type)
        return descriptor.type_id == 'DataAsset' or 'DataAsset' in descriptor.compatible_types
    return False


def bind_scene_value(value, metadata, owner):
    """Setter/attach only. Bind source/selector refs by replacement, not mutation."""
    from .fields import FieldType
    leaf_type = metadata.element_type if metadata.field_type == FieldType.LIST else metadata.field_type
    if owner is None and leaf_type != FieldType.ASSET:
        return value
    from .ref_wrappers import GameObjectRef, ComponentRef
    from infernux.core.asset_ref import DataAssetRef
    if metadata.field_type == FieldType.LIST:
        return [_bind_scene_ref(item, owner, GameObjectRef, ComponentRef, DataAssetRef) for item in value]
    return _bind_scene_ref(value, owner, GameObjectRef, ComponentRef, DataAssetRef)


def _bind_scene_ref(value, owner, object_type, component_type, data_asset_type):
    if isinstance(value, data_asset_type):
        current = value._value_owner_world_id
        if current is UNSET or (current is None and owner is not None):
            return data_asset_type(guid=value.guid, path_hint=value._path_hint, _owner_world_id=owner)
        return value
    if owner is None:
        return value
    if isinstance(value, object_type) and value.persistent_id and value._value_owner_world_id is None:
        return object_type(persistent_id=value.persistent_id, _owner_world_id=owner)
    if isinstance(value, component_type) and value.go_id and value._value_owner_world_id is None:
        return component_type(go_id=value.go_id, component_type=value._component_type,
                              component_id=value.component_id, _owner_world_id=owner)
    return value
