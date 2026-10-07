"""
Script loader for dynamically importing InxComponent subclasses from .py files.

This module provides utilities to load Python scripts and extract component classes
for use in the Infernux editor. Used for drag-and-drop script attachment.
"""

from __future__ import annotations

import os
import sys
import importlib
import importlib.util
import inspect
import tokenize
import types
from dataclasses import dataclass, fields as dataclass_fields
from typing import TYPE_CHECKING, Iterable, Type, List, Optional

from infernux.engine.path_utils import path_key, resolved_path
from infernux.debug import Debug
from infernux.engine.project_context import (
    get_script_module_name,
    resolve_script_path,
    temporary_script_import_paths,
)
if TYPE_CHECKING:
    from infernux.engine.candidate_import import CandidateImportTransaction

from .component import InxComponent
from ._cds_migration import (
    ClassSchemaMigration,
    build_class_schema_migration,
    collect_live_instances,
    prepare_instance_values,
)


class ScriptLoadError(Exception):
    """Raised when a script cannot be loaded or doesn't contain valid components."""

    def __init__(self, message: str, *, source_file: str = "", source_line: int = 0):
        super().__init__(message)
        self.source_file = source_file
        self.source_line = source_line


class ScriptReloadRejected(ScriptLoadError):
    """Raised when a saved script is not a body-only compatible reload."""


_MODULE_ABSENT = object()


def _new_candidate_import_transaction():
    """Load the Editor-only candidate importer only when hot reload uses it."""

    from infernux.engine.candidate_import import CandidateImportTransaction

    return CandidateImportTransaction()


@dataclass(frozen=True)
class ComponentBodyReloadRequest:
    """One Play/Pause body candidate and its live targets."""

    file_path: str
    target_types: tuple[type, ...] = ()
    instances_by_type: Optional[dict[type, tuple[object, ...]]] = None
    script_guid: str = ""
    source: bytes | str | None = None
    code: types.CodeType | None = None
    retire_script_paths: tuple[str, ...] = ()


@dataclass
class _SchemaInstanceBeforeImage:
    target_type: type
    candidate_type: type
    instance: object
    old_slot: object
    old_class_id: object
    old_values: dict[str, object]
    instance_dict: dict[str, tuple[bool, object]]
    new_slot: object = None
    new_class_id: object = None


class ComponentBodyReloadTransaction:
    """All-or-nothing class-body publication for one or more scripts.

    Candidate modules live in a transaction-private import table while
    staging. No candidate module is published to ``sys.modules`` until
    ``commit`` succeeds. A precise before-image is retained for the component
    registry, class bodies, epoch publication and script diagnostics, so a
    later owner-side rollback can restore the
    exact pre-commit state without touching unrelated modules.
    """

    def __init__(
        self,
        requests: tuple[ComponentBodyReloadRequest, ...],
        plans: tuple[tuple[type, tuple[tuple[str, bool, object], ...]], ...],
        *,
        registry_entries: tuple[tuple[str, tuple[type, ...]], ...],
        registry_snapshot: object,
        module_snapshot: dict[str, object],
        diagnostic_snapshot: tuple[dict[str, str], int],
        had_live_targets: bool,
        member_status: tuple[tuple[str, bool, int], ...],
        candidate_import: CandidateImportTransaction | None = None,
        schema_migrations: tuple[ClassSchemaMigration, ...] = (),
        cds_publish_types: tuple[type, ...] = (),
        registration_publish_types: tuple[type, ...] = (),
    ) -> None:
        self.requests = requests
        self.plans = plans
        # A live target keeps the stable old class (its body is patched); a
        # component with no live instance publishes the candidate class.  The
        # distinction is resolved during staging and committed atomically.
        self.registry_entries = registry_entries
        self.registry_snapshot = registry_snapshot
        self.module_snapshot = module_snapshot
        self.diagnostic_snapshot = diagnostic_snapshot
        self.had_live_targets = bool(had_live_targets)
        self.member_status = member_status
        self.schema_migrations = tuple(schema_migrations)
        self.cds_publish_types = tuple(dict.fromkeys(cds_publish_types))
        self.registration_publish_types = tuple(
            dict.fromkeys(registration_publish_types or self.cds_publish_types)
        )
        self._candidate_import = candidate_import or _new_candidate_import_transaction()
        self._class_snapshots = tuple(
            (
                target_type,
                _snapshot_object_attributes(target_type, (name for name, _, _ in operations)),
            )
            for target_type, operations in plans
        )
        self._committed = False
        self._rolled_back = False
        self._result: dict[type, tuple[str, ...]] = {}
        self._dispatch_publication = None
        self._cds_publication = None
        self._registration_pending_snapshots = tuple(
            (
                component_type,
                _snapshot_object_attributes(
                    component_type,
                    ("_component_registration_pending_",),
                ),
            )
            for component_type in self.registration_publish_types
        )
        self._schema_instance_images: list[_SchemaInstanceBeforeImage] = []
        self._old_slots_retired = False
        self._finalized = False
        self._retired_types: tuple[type, ...] = ()

    @property
    def committed(self) -> bool:
        return self._committed and not self._rolled_back

    @property
    def rolled_back(self) -> bool:
        return self._rolled_back

    @property
    def finalized(self) -> bool:
        return self._finalized

    def commit(self) -> dict[type, tuple[str, ...]]:
        """Publish all registry entries and class body plans exactly once."""
        if self._rolled_back:
            raise ScriptReloadRejected("reload transaction has already been rolled back")
        if self._committed:
            return dict(self._result)
        try:
            from infernux.engine.runtime_dispatch import assert_runtime_dispatch_safe_point

            # Guard before registry/class mutation.  A later publication
            # failure must never leave a body patch visible in an active frame.
            assert_runtime_dispatch_safe_point()
            with self._candidate_import.serializable_type_scope():
                _prepare_transaction_schema_state(self)
            self._result = _apply_component_body_patch_plans(self.plans)
            _refresh_transaction_component_metadata(self)
            if self._cds_publication is not None:
                self._cds_publication.commit()
            _publish_transaction_instance_schemas(self)
            if self.registry_entries:
                from .registry import publish_component_script_types_batch

                self._retired_types = publish_component_script_types_batch(
                    self.registry_entries,
                    remove_paths=tuple(
                        path
                        for request in self.requests
                        for path in request.retire_script_paths
                    ),
                    cds_prepared=self._cds_publication is not None,
                )
            from infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch

            # The immutable epoch is published only after registry and every
            # body operation in this transaction has succeeded.
            dispatch_types = tuple(dict.fromkeys(
                [target_type for target_type, _operations in self.plans]
                + [
                    component_type
                    for _path, component_types in self.registry_entries
                    for component_type in component_types
                ]
            ))
            self._dispatch_publication = publish_runtime_dispatch_epoch(
                dispatch_types,
                retired_types=self._retired_types,
                defer_commit=True,
            )
            self._dispatch_publication.commit()
            # Module publication is the final durable side effect of the
            # transaction. Candidate execution itself never touched sys.modules.
            self._candidate_import.commit()
            from ._component_registration import mark_component_registration_published

            for component_type in self.registration_publish_types:
                mark_component_registration_published(component_type)
            self._committed = True
            return dict(self._result)
        except Exception:
            self.rollback()
            raise

    def finalize(self) -> None:
        """Close the native CDS rollback window after the outer durable commit."""
        if self._rolled_back:
            raise ScriptReloadRejected("reload transaction has already been rolled back")
        if not self._committed:
            raise ScriptReloadRejected("reload transaction must commit before finalization")
        if self._finalized:
            return
        from infernux.engine.runtime_dispatch import assert_runtime_dispatch_safe_point

        assert_runtime_dispatch_safe_point()
        if self._cds_publication is not None:
            self._cds_publication.finalize()
        # The outer dependency graph and LKG transaction may still reject a
        # successful body/CDS publication.  Preserve the exact old handles
        # until that durable edge has finalized the native schema.
        _retire_transaction_old_slots(self)
        self._finalized = True

    def rollback(self) -> None:
        """Restore the complete pre-stage/pre-commit state; idempotent."""
        if self._rolled_back:
            return
        if self._finalized:
            raise ScriptReloadRejected(
                "reload transaction is durable and can no longer be rolled back"
            )
        if (
            self._cds_publication is not None
            and self._cds_publication.finalized
        ):
            raise ScriptReloadRejected(
                "reload transaction has finalized its native schema and cannot be rolled back"
            )
        from .registry import restore_component_registry_state

        try:
            self._candidate_import.rollback()
            for target_type, snapshot in self._class_snapshots:
                _restore_object_attributes(target_type, snapshot)
            _refresh_transaction_component_metadata(self)
            if self._dispatch_publication is not None:
                self._dispatch_publication.rollback()
            _invalidate_serialized_field_caches(
                target_type for target_type, _snapshot in self._class_snapshots
            )
            for component_type, snapshot in self._registration_pending_snapshots:
                _restore_object_attributes(component_type, snapshot)
            _restore_transaction_instance_schemas(self)
            if self._cds_publication is not None:
                self._cds_publication.rollback()
        finally:
            try:
                restore_component_registry_state(self.registry_snapshot)
            finally:
                try:
                    _restore_sys_modules(self.module_snapshot)
                finally:
                    _restore_script_diagnostics(self.diagnostic_snapshot)
        self._result = {}
        self._committed = False
        self._rolled_back = True


def _snapshot_object_attributes(owner: object, names: Iterable[str]) -> dict[str, tuple[bool, object]]:
    values = getattr(owner, "__dict__", {})
    return {
        name: (True, getattr(owner, name))
        if isinstance(owner, type) and name in {"__name__", "__qualname__", "__bases__"}
        else (name in values, values.get(name))
        for name in names
    }


def _restore_object_attributes(owner: object, snapshot: dict[str, tuple[bool, object]]) -> None:
    values = getattr(owner, "__dict__", {})
    for name, (present, value) in snapshot.items():
        if present:
            setattr(owner, name, value)
        elif name in values:
            delattr(owner, name)


def _invalidate_serialized_field_caches(component_types: Iterable[type]) -> None:
    """Discard metadata views after publishing or restoring stable classes."""
    from .fields import clear_serialized_fields_cache

    for component_type in component_types:
        clear_serialized_fields_cache(component_type)


def _serialized_descriptor(owner: type, name: str):
    from .fields import SerializedFieldDescriptor

    for base in owner.__mro__:
        value = base.__dict__.get(name)
        if isinstance(value, SerializedFieldDescriptor):
            return value
    return None


def _transaction_instances(
    transaction: ComponentBodyReloadTransaction,
    target_type: type,
) -> tuple[object, ...]:
    values: list[object] = []
    seen: set[int] = set()
    for request in transaction.requests:
        for instance in (request.instances_by_type or {}).get(target_type, ()):
            if id(instance) not in seen and type(instance) is target_type:
                seen.add(id(instance))
                values.append(instance)
    for instance in collect_live_instances(target_type):
        if id(instance) not in seen:
            seen.add(id(instance))
            values.append(instance)
    return tuple(values)


_BODY_PATCH_NATIVE_METADATA_KEYS = frozenset({
    "__name__", "__qualname__", "_type_guid_", "_execute_in_edit_mode_",
})


def _refresh_transaction_component_metadata(transaction: ComponentBodyReloadTransaction) -> None:
    for target_type, operations in transaction.plans:
        if not any(name in _BODY_PATCH_NATIVE_METADATA_KEYS for name, _, _ in operations):
            continue
        _refresh_component_metadata(target_type, _transaction_instances(transaction, target_type))


def _refresh_component_metadata(target_type: type, instances: Iterable[object]) -> None:
    for instance in instances:
        instance._component_name = target_type.__name__
        instance._execute_in_edit_mode = bool(getattr(target_type, "_execute_in_edit_mode_", False))
        native = instance._get_bound_native_component()
        if native is not None:
            native.refresh_python_lifecycle_dispatch()


def _write_private_descriptor_value(descriptor, instance: object, value: object) -> None:
    values = getattr(instance, "__dict__", {})
    had_flag = "_inf_deserializing" in values
    old_flag = values.get("_inf_deserializing")
    setattr(instance, "_inf_deserializing", True)
    try:
        descriptor.__set__(instance, value)
    finally:
        if had_flag:
            setattr(instance, "_inf_deserializing", old_flag)
        else:
            values.pop("_inf_deserializing", None)


def _prepare_transaction_schema_state(
    transaction: ComponentBodyReloadTransaction,
) -> None:
    if transaction._cds_publication is not None:
        return

    from . import _cds_bridge
    from .fields import get_serialized_fields

    # Capture the authored instance state before preparing or mutating any
    # native schema.  Preparing a replacement layout is intentionally an
    # isolated transaction, but the old slot must never be the only source of
    # truth for a live component during that transition.
    migration_instances: dict[
        int,
        tuple[tuple[object, ...], dict[object, dict[str, object]]],
    ] = {}
    for migration in transaction.schema_migrations:
        if (
            _serialized_schema_signature(migration.target_type)
            == _serialized_schema_signature(migration.candidate_type)
            and migration.target_type._get_type_guid() == migration.candidate_type._get_type_guid()
        ):
            continue
        instances = _transaction_instances(transaction, migration.target_type)
        migration_instances[id(migration)] = (
            instances,
            dict(prepare_instance_values(migration, instances)),
        )

    if not migration_instances and not transaction.cds_publish_types:
        return

    publication = (
        _cds_bridge.prepare_schema_publication(
            transaction.cds_publish_types,
            # Constraints can change stored values without changing numeric
            # shape. Stage every changed live schema in private storage.
            private_layout_types=(
                migration.candidate_type
                for migration in transaction.schema_migrations
                if id(migration) in migration_instances
                and migration_instances[id(migration)][0]
            ),
        )
        if transaction.cds_publish_types
        else None
    )
    transaction._cds_publication = publication
    images: list[_SchemaInstanceBeforeImage] = []
    try:
        for migration in transaction.schema_migrations:
            if id(migration) not in migration_instances:
                continue

            instances, prepared_values = migration_instances[id(migration)]
            target_fields = get_serialized_fields(migration.target_type)
            candidate_fields = get_serialized_fields(migration.candidate_type)
            uses_cds = bool(
                getattr(migration.candidate_type, "_uses_component_data_store", True)
            )
            entry = None
            if uses_cds:
                if publication is None:
                    raise RuntimeError(
                        f"CDS publication is unavailable for "
                        f"{migration.candidate_type.__qualname__}"
                    )
                publication.retire_published_layout(migration.target_type)
                entry = publication.entry(migration.candidate_type)
                if entry.candidate_id is not None:
                    publication.reserve(migration.candidate_type, len(instances))

            field_names = set(target_fields) | set(candidate_fields)
            for instance in instances:
                old_values: dict[str, object] = {}
                for name in target_fields:
                    descriptor = _serialized_descriptor(migration.target_type, name)
                    if descriptor is None:
                        raise RuntimeError(
                            f"live serialized descriptor is unavailable: "
                            f"{migration.target_type.__qualname__}.{name}"
                        )
                    old_values[name] = descriptor.get_raw(instance)

                instance_values = getattr(instance, "__dict__", {})
                image = _SchemaInstanceBeforeImage(
                    migration.target_type,
                    migration.candidate_type,
                    instance,
                    getattr(instance, "_cds_slot", None),
                    getattr(instance, "_cds_class_id", None),
                    old_values,
                    {
                        name: (name in instance_values, instance_values.get(name))
                        for name in field_names
                    },
                )

                if entry is not None and entry.candidate_id is not None:
                    image.new_slot = publication.allocate_slot(migration.candidate_type)
                    # The prepared semantic values already include renames,
                    # defaults and conversions. Write them once; copying raw
                    # native fields first would immediately be overwritten.
                    for field in migration.fields:
                        if not _cds_bridge.is_cds_backed(field.target.field_type):
                            continue
                        publication.set_value(
                            migration.candidate_type,
                            field.target_name,
                            image.new_slot,
                            prepared_values[instance][field.target_name],
                        )
                elif entry is not None and entry.class_id is not None:
                    if image.old_class_id != entry.class_id:
                        raise RuntimeError(
                            f"CDS layout reuse disagrees with live class ID for "
                            f"{migration.target_type.__qualname__}"
                        )
                    image.new_slot = image.old_slot

                for field in migration.fields:
                    if uses_cds and _cds_bridge.is_cds_backed(field.target.field_type):
                        continue
                    candidate_descriptor = _serialized_descriptor(
                        migration.candidate_type,
                        field.target_name,
                    )
                    target_descriptor = (
                        _serialized_descriptor(migration.target_type, field.source_name)
                        if field.source_name is not None
                        else None
                    )
                    if candidate_descriptor is None:
                        raise RuntimeError(
                            f"candidate serialized descriptor is unavailable: "
                            f"{migration.candidate_type.__qualname__}.{field.target_name}"
                        )
                    if candidate_descriptor is target_descriptor:
                        continue
                    _write_private_descriptor_value(
                        candidate_descriptor,
                        instance,
                        prepared_values[instance][field.target_name],
                    )
                images.append(image)

        if publication is not None:
            publication.seal()
        for image in images:
            if publication is not None and bool(
                getattr(image.candidate_type, "_uses_component_data_store", True)
            ):
                image.new_class_id = publication.entry(image.candidate_type).class_id
        transaction._schema_instance_images = images
    except Exception:
        if publication is not None:
            publication.rollback()
        transaction._schema_instance_images = []
        raise


def _publish_transaction_instance_schemas(
    transaction: ComponentBodyReloadTransaction,
) -> None:
    for image in transaction._schema_instance_images:
        image.instance._cds_slot = image.new_slot
        image.instance._cds_class_id = image.new_class_id
        values = getattr(image.instance, "__dict__", {})
        for name in image.instance_dict:
            values.pop(name, None)


def _retire_transaction_old_slots(
    transaction: ComponentBodyReloadTransaction,
) -> None:
    if transaction._old_slots_retired:
        return
    from ._cds_bridge import release_slot

    for image in transaction._schema_instance_images:
        if image.old_slot is None or image.old_class_id is None:
            continue
        if (
            image.old_slot == image.new_slot
            and image.old_class_id == image.new_class_id
        ):
            continue
        release_slot(
            image.target_type,
            image.old_slot,
            class_id=int(image.old_class_id),
        )
    transaction._old_slots_retired = True


def _restore_transaction_instance_schemas(
    transaction: ComponentBodyReloadTransaction,
) -> None:
    if not transaction._schema_instance_images:
        return
    from . import _cds_bridge
    from .fields import get_serialized_fields

    for image in transaction._schema_instance_images:
        restored_slot = image.old_slot
        if image.old_class_id is not None and image.old_slot is not None:
            if not _cds_bridge.published_slot_is_alive(
                int(image.old_class_id),
                image.old_slot,
            ):
                restored_slot = _cds_bridge.allocate_published_slot(
                    int(image.old_class_id)
                )
                for name in get_serialized_fields(image.target_type):
                    descriptor = _serialized_descriptor(image.target_type, name)
                    if descriptor is None or descriptor._cds_class_id is None:
                        continue
                    _cds_bridge.set_published_value(
                        int(image.old_class_id),
                        int(descriptor._cds_field_id),
                        int(descriptor._cds_type_code),
                        restored_slot,
                        image.old_values[name],
                    )
        image.instance._cds_slot = restored_slot
        image.instance._cds_class_id = image.old_class_id
        values = getattr(image.instance, "__dict__", {})
        for name, (present, value) in image.instance_dict.items():
            if present:
                values[name] = value
            else:
                values.pop(name, None)


def _restore_sys_modules(before_image: dict[str, object]) -> None:
    """Restore only module entries captured as touched by this transaction."""
    for name, previous in before_image.items():
        if previous is _MODULE_ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _snapshot_script_diagnostics() -> tuple[dict[str, str], int]:
    """Capture diagnostics without changing their public revision."""
    return dict(_script_errors), _script_error_revision


def _restore_script_diagnostics(snapshot: tuple[dict[str, str], int]) -> None:
    """Restore the exact diagnostics visible before candidate execution."""
    global _script_error_revision
    errors, revision = snapshot
    _script_errors.clear()
    _script_errors.update(errors)
    _script_error_revision = revision


def _component_classes_from_module(
    module: types.ModuleType,
    module_name: str,
) -> list[Type[InxComponent]]:
    """Extract component declarations in source execution order.

    ``inspect.getmembers`` sorts names alphabetically.  That is the wrong
    ordering for a script asset: the first component declaration is the
    deterministic attachment target when the authored class name changes.
    Module dictionaries retain execution order, so use that order directly.
    """
    components = []
    for _name, obj in vars(module).items():
        if (
            inspect.isclass(obj)
            and obj is not InxComponent
            and issubclass(obj, InxComponent)
            and obj.__module__ == module_name
        ):
            components.append(obj)
    return components


def _replace_candidate_type_value(
    value: object,
    replacements: dict[type, type],
    memo: dict[int, object],
) -> object:
    """Replace candidate class identities inside private candidate values."""
    if isinstance(value, type) and value in replacements:
        return replacements[value]
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, (staticmethod, classmethod)):
        memo[identity] = value
        _replace_candidate_type_value(value.__func__, replacements, memo)
        return value
    if isinstance(value, property):
        memo[identity] = value
        for accessor in (value.fget, value.fset, value.fdel):
            if accessor is not None:
                _replace_candidate_type_value(accessor, replacements, memo)
        return value
    if isinstance(value, types.FunctionType):
        memo[identity] = value
        defaults = value.__defaults__
        if defaults:
            value.__defaults__ = tuple(
                _replace_candidate_type_value(item, replacements, memo)
                for item in defaults
            )
        kwdefaults = value.__kwdefaults__
        if kwdefaults:
            value.__kwdefaults__ = {
                key: _replace_candidate_type_value(item, replacements, memo)
                for key, item in kwdefaults.items()
            }
        annotations = value.__annotations__
        if annotations:
            value.__annotations__ = {
                key: _replace_candidate_type_value(item, replacements, memo)
                for key, item in annotations.items()
            }
        for cell in value.__closure__ or ():
            try:
                captured = cell.cell_contents
            except ValueError:
                continue
            rebound = _replace_candidate_type_value(captured, replacements, memo)
            if rebound is not captured:
                cell.cell_contents = rebound
        return value
    if isinstance(value, tuple):
        memo[identity] = value
        result = tuple(
            _replace_candidate_type_value(item, replacements, memo) for item in value
        )
        memo[identity] = result
        return result
    if isinstance(value, list):
        memo[identity] = value
        for index, item in enumerate(tuple(value)):
            value[index] = _replace_candidate_type_value(item, replacements, memo)
        return value
    if isinstance(value, dict):
        memo[identity] = value
        items = tuple(value.items())
        value.clear()
        for key, item in items:
            rebound_key = _replace_candidate_type_value(key, replacements, memo)
            value[rebound_key] = _replace_candidate_type_value(item, replacements, memo)
        return value
    if isinstance(value, set):
        memo[identity] = value
        items = tuple(value)
        value.clear()
        value.update(
            _replace_candidate_type_value(item, replacements, memo) for item in items
        )
        return value
    if isinstance(value, frozenset):
        memo[identity] = value
        result = frozenset(
            _replace_candidate_type_value(item, replacements, memo) for item in value
        )
        memo[identity] = result
        return result
    return value


def _rebind_candidate_type_references(
    modules: Iterable[types.ModuleType],
    replacements: dict[type, type],
) -> None:
    """Make every committed candidate module refer to stable live classes."""
    if not replacements:
        return
    memo: dict[int, object] = {}
    module_values = tuple(dict.fromkeys(modules))
    for module in module_values:
        for name, value in tuple(vars(module).items()):
            if name == "__builtins__" or isinstance(value, types.ModuleType):
                continue
            # Star imports expose a large part of the engine API in the
            # candidate namespace. Those functions and classes belong to
            # their original modules and cannot capture this transaction's
            # candidate types. Recursing through them is both incorrect and
            # extremely expensive for ordinary ``from infernux import *``
            # component scripts.
            direct_candidate_type = isinstance(value, type) and value in replacements
            if (
                isinstance(value, (types.FunctionType, type))
                and not direct_candidate_type
                and getattr(value, "__module__", "") != module.__name__
            ):
                continue
            rebound = _replace_candidate_type_value(value, replacements, memo)
            if rebound is not value:
                setattr(module, name, rebound)
    for candidate_type in replacements:
        for name, value in tuple(candidate_type.__dict__.items()):
            if name.startswith("__"):
                continue
            rebound = _replace_candidate_type_value(value, replacements, memo)
            if rebound is not value:
                setattr(candidate_type, name, rebound)


def stage_component_body_reload_batch(
    requests: Iterable[ComponentBodyReloadRequest],
) -> ComponentBodyReloadTransaction:
    """Stage all script candidates before touching any live class body."""
    incoming_requests = tuple(requests)
    if not incoming_requests:
        raise ValueError("at least one reload request is required")

    normalized_requests = []
    request_modules = []
    seen_module_names: set[str] = set()
    for request in incoming_requests:
        if not isinstance(request, ComponentBodyReloadRequest):
            raise TypeError("requests must contain ComponentBodyReloadRequest values")
        file_path = resolve_script_path(request.file_path)
        if not file_path or not os.path.exists(file_path):
            raise ScriptLoadError(f"Script file not found: {request.file_path}")
        module_name = get_script_module_name(file_path) or _unique_module_name_for_path(file_path)
        if module_name in seen_module_names:
            raise ScriptReloadRejected(
                f"script module appears more than once in reload batch: {module_name}"
            )
        seen_module_names.add(module_name)
        normalized_requests.append(
            ComponentBodyReloadRequest(
                file_path=file_path,
                target_types=tuple(dict.fromkeys(request.target_types)),
                instances_by_type={
                    target_type: tuple(values)
                    for target_type, values in (request.instances_by_type or {}).items()
                },
                script_guid=request.script_guid,
                source=request.source,
                code=request.code,
                retire_script_paths=tuple(request.retire_script_paths),
            )
        )
        request_modules.append(module_name)
    normalized_requests = tuple(normalized_requests)
    request_modules = tuple(request_modules)

    from .registry import (
        component_types_for_script_path,
        restore_component_registry_state,
        snapshot_component_registry_state,
    )

    registry_snapshot = snapshot_component_registry_state()
    # Capture all published declarations before candidate imports, including
    # declarations without Scene instances and paths retired by an asset move.
    published_types_by_module = {
        module_name: tuple(dict.fromkeys(
            component_type
            for script_path in (request.file_path, *request.retire_script_paths)
            for component_type in component_types_for_script_path(script_path)
        ))
        for request, module_name in zip(normalized_requests, request_modules)
    }
    live_targets = {
        component_type
        for request, module_name in zip(normalized_requests, request_modules)
        for component_type in (*request.target_types, *published_types_by_module[module_name])
        if component_type in request.target_types or collect_live_instances(component_type)
    }
    # A live subclass also retains its base class objects, even when the bases
    # have no instances of their own. Preserve those identities across files
    # before deciding which declarations may publish as fresh candidates.
    retained_types = live_targets | {
        base for component_type in live_targets for base in component_type.__mro__[1:]
    }
    diagnostic_snapshot = _snapshot_script_diagnostics()
    module_snapshot: dict[str, object] = {}
    plans: list[tuple[type, tuple[tuple[str, bool, object], ...]]] = []
    matched_types: list[tuple[type, type]] = []
    schema_migrations: list[ClassSchemaMigration] = []
    cds_publish_types: list[type] = []
    registration_publish_types: list[type] = []
    registry_entries: list[tuple[str, tuple[type, ...]]] = []
    seen_types: set[type] = set()
    member_status: list[tuple[str, bool, int]] = []
    had_live_targets = False
    candidate_import = _new_candidate_import_transaction()
    for request, module_name in zip(normalized_requests, request_modules):
        candidate_import.register(
            module_name,
            request.file_path,
            source=request.source,
            code=request.code,
        )

    try:
        for request, module_name in zip(normalized_requests, request_modules):
            file_path = request.file_path
            published_types = published_types_by_module[module_name]
            targets = tuple(dict.fromkeys((
                *request.target_types,
                *(component_type for component_type in published_types
                  if component_type in retained_types),
            )))
            instances_by_type = {
                target_type: tuple(values)
                for target_type, values in (request.instances_by_type or {}).items()
            }
            if targets:
                had_live_targets = True
                if seen_types.intersection(targets):
                    raise ScriptReloadRejected(
                        f"component type appears in more than one reload request: "
                        f"{next(iter(seen_types.intersection(targets))).__qualname__}"
                    )
                seen_types.update(targets)

            candidate_module = candidate_import.module_for(module_name)
            diagnostic = None
            if candidate_module is not None:
                candidate_path = resolved_path(
                    getattr(candidate_module, "__file__", "") or ""
                )
                if candidate_path and path_key(candidate_path) != path_key(file_path):
                    raise ScriptLoadError(
                        f"reload batch module '{module_name}' resolved to unexpected path: "
                        f"{candidate_path}"
                    )
                candidates = _component_classes_from_module(candidate_module, module_name)
            else:
                try:
                    from ._component_registration import (
                        candidate_component_registration_scope,
                    )

                    with candidate_component_registration_scope():
                        # Match the ordinary loader: a successful candidate
                        # clears only its staged diagnostic. The before-image
                        # is restored below until the owner commits.
                        _clear_script_error(file_path)
                        candidate_module = candidate_import.load(module_name)
                        candidates = _component_classes_from_module(
                            candidate_module,
                            module_name,
                        )
                    diagnostic = get_script_error_by_path(file_path)
                except Exception as exc:
                    import traceback

                    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    frames = traceback.extract_tb(exc.__traceback__)
                    source_line = next(
                        (frame.lineno for frame in reversed(frames)
                         if _normalize_script_path(frame.filename) == _normalize_script_path(file_path)),
                        0,
                    )
                    raise ScriptLoadError(details, source_file=file_path, source_line=source_line) from exc
                finally:
                    # __init_subclass__ may have touched the registry while
                    # the candidate module was executed. Keep it staged.
                    restore_component_registry_state(registry_snapshot)
                    # Loading a candidate must not publish diagnostics. The
                    # outer collector clears errors after durable commit.
                    _restore_script_diagnostics(diagnostic_snapshot)

            if diagnostic:
                raise ScriptLoadError(
                    f"component candidate import failed; keeping last-known-good: {diagnostic}"
                )
            if not candidates:
                if not targets:
                    # A helper or other dependency has no component registry
                    # entry, but it still participates in the ordered
                    # candidate module batch.
                    member_status.append((file_path, False, 0))
                    continue
                raise ScriptReloadRejected(
                    f"no component classes were loaded from {os.path.basename(file_path)}"
                )

            candidate_by_identity = {
                (candidate.__name__, candidate.__qualname__): candidate
                for candidate in candidates
            }
            if request.script_guid:
                from .component_identity import bind_asset_script_guid

                for candidate in candidates:
                    # A no-live script still needs a complete identity for
                    # Add Component and later scene attachment.
                    bind_asset_script_guid(candidate, request.script_guid)
            target_by_identity = {
                (target_type.__name__, target_type.__qualname__): target_type
                for target_type in targets
            }
            if len(target_by_identity) != len(targets):
                raise ScriptReloadRejected(
                    "multiple live component types claim the same declaration; reload rejected"
                )
            target_by_candidate: dict[type, type] = {}
            for candidate_type in candidates:
                target_type = target_by_identity.get(
                    (candidate_type.__name__, candidate_type.__qualname__)
                )
                if target_type is not None:
                    target_by_candidate[candidate_type] = target_type

            # Exact declarations keep their identity regardless of Scene or
            # source order. A rename is unique only when exactly one old and
            # one new declaration remain across the whole script, including
            # unmounted types. Never pair multiple renames by traversal order.
            published_identities = {
                (component_type.__name__, component_type.__qualname__)
                for component_type in (*published_types, *targets)
            }
            removed_identities = published_identities.difference(candidate_by_identity)
            unmatched_targets = [
                target_type
                for target_type in targets
                if target_type not in target_by_candidate.values()
            ]
            unmatched_candidates = [
                candidate_type
                for candidate_type in candidates
                if (candidate_type.__name__, candidate_type.__qualname__) not in published_identities
            ]
            if unmatched_targets and unmatched_candidates:
                if not (len(unmatched_targets) == len(removed_identities) == len(unmatched_candidates) == 1):
                    raise ScriptReloadRejected(
                        "component rename is ambiguous; rename one declaration at a time "
                        "without adding or removing other declarations"
                    )
                target_type = unmatched_targets[0]
                candidate_type = unmatched_candidates[0]
                target_by_candidate[candidate_type] = target_type
                target_by_identity[
                    (candidate_type.__name__, candidate_type.__qualname__)
                ] = target_type
            effective_targets = list(targets)
            if request.script_guid:
                from .registry import get_type_by_identity

                for candidate_type in candidates:
                    identity = (candidate_type.__name__, candidate_type.__qualname__)
                    if identity in target_by_identity:
                        continue
                    existing_type = get_type_by_identity(
                        candidate_type.__name__,
                        request.script_guid,
                        candidate_type._get_type_guid(),
                    )
                    if existing_type is None or not collect_live_instances(existing_type):
                        continue
                    if existing_type in seen_types:
                        raise ScriptReloadRejected(
                            f"component type appears in more than one reload request: "
                            f"{existing_type.__qualname__}"
                        )
                    seen_types.add(existing_type)
                    effective_targets.append(existing_type)
                    target_by_identity[identity] = existing_type
                    had_live_targets = True
            published_types = []
            for candidate_type in candidates:
                registration_publish_types.append(candidate_type)
                target_type = target_by_identity.get(
                    (candidate_type.__name__, candidate_type.__qualname__)
                )
                # Existing live objects continue to use the old class.  A
                # type without live objects must point at the candidate so it
                # can be selected from Add Component immediately after commit.
                published_types.append(target_type or candidate_type)
                if bool(getattr(candidate_type, "_uses_component_data_store", True)):
                    cds_publish_types.append(candidate_type)

            for target_type in effective_targets:
                if request.script_guid and getattr(target_type, "_asset_script_guid_", "") != request.script_guid:
                    raise ScriptReloadRejected(
                        f"component type '{target_type.__qualname__}' script identity changed; reload rejected"
                    )
                candidate_type = next(
                    (
                        candidate
                        for candidate, mapped_target in target_by_candidate.items()
                        if mapped_target is target_type
                    ),
                    candidate_by_identity.get(
                        (target_type.__name__, target_type.__qualname__)
                    ),
                )
                if candidate_type is None:
                    raise ScriptReloadRejected(
                        f"component type '{target_type.__qualname__}' was removed or renamed; reload rejected"
                    )
                matched_types.append((target_type, candidate_type))

            registry_entries.append((file_path, tuple(published_types)))

            member_status.append(
                (
                    file_path,
                    True,
                    sum(
                        len(instances_by_type.get(target_type, ()))
                        or len(collect_live_instances(target_type))
                        for target_type in effective_targets
                    ),
                )
            )

        replacements = {
            candidate_type: target_type
            for target_type, candidate_type in matched_types
        }
        _rebind_candidate_type_references(
            candidate_import.publishable_modules,
            replacements,
        )
        schema_migrations.extend(
            build_class_schema_migration(target_type, candidate_type)
            for target_type, candidate_type in matched_types
        )
        plans.extend(
            (
                target_type,
                _plan_component_class_body_patch(
                    target_type, candidate_type, publish_identity=True,
                    type_replacements=replacements,
                ),
            )
            for target_type, candidate_type in matched_types
        )
        # Newly published or unmounted subclasses must inherit the same live
        # bases as the patched classes. Apply this only at commit, after schema
        # preparation has read the complete candidate declarations. The normal
        # class snapshots cover __bases__ for rollback as well.
        for candidate_type in registration_publish_types:
            if candidate_type in replacements:
                continue
            bases = tuple(replacements.get(base, base) for base in candidate_type.__bases__)
            if bases != candidate_type.__bases__:
                plans.append((candidate_type, (("__bases__", True, bases),)))
        transaction = ComponentBodyReloadTransaction(
            normalized_requests,
            tuple(plans),
            registry_snapshot=registry_snapshot,
            registry_entries=tuple(registry_entries),
            module_snapshot=module_snapshot,
            diagnostic_snapshot=diagnostic_snapshot,
            had_live_targets=had_live_targets,
            member_status=tuple(member_status),
            candidate_import=candidate_import,
            schema_migrations=tuple(schema_migrations),
            cds_publish_types=tuple(cds_publish_types),
            registration_publish_types=tuple(registration_publish_types),
        )
        return transaction
    except Exception:
        candidate_import.rollback()
        try:
            restore_component_registry_state(registry_snapshot)
        finally:
            try:
                _restore_sys_modules(module_snapshot)
            finally:
                _restore_script_diagnostics(diagnostic_snapshot)
        raise


def rollback_component_body_reload(transaction: ComponentBodyReloadTransaction) -> None:
    """Explicit owner-side rollback hook for a committed batch."""
    if not isinstance(transaction, ComponentBodyReloadTransaction):
        raise TypeError("transaction must be a ComponentBodyReloadTransaction")
    transaction.rollback()


def retire_script_module(file_path: str) -> object | None:
    """Remove the canonical project module for a deleted/moved script.

    This is deliberately a small owner-side operation.  Candidate imports
    remain staged in ``sys.modules`` until their transaction has committed;
    callers invoke this only after the durable asset mutation succeeds.
    """
    module_name = get_script_module_name(file_path)
    if not module_name:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        return None
    module_file = resolved_path(getattr(module, "__file__", "") or "")
    if module_file and path_key(module_file) != path_key(file_path):
        # A move can keep the same canonical module name. The candidate at
        # the destination must survive; only the old module may be retired.
        return None
    return sys.modules.pop(module_name, None)


_BODY_PATCH_GENERATED_KEYS = frozenset({
    "_serialized_fields_",
    "_field_schemas_",
    "_intrinsic_script_guid_",
    "_type_guid_",
    "_asset_script_guid_",
})

_BODY_PATCH_CONTRACT_KEYS = frozenset({
    "_require_components_",
    "_incompatible_components_",
    "_component_exclusive_groups_",
    "_component_satisfied_types_",
    "_disallow_multiple_",
    "_component_user_addable_",
    "_component_removable_",
    "_component_intrinsic_",
    "_component_menu_path_",
    "_component_category_",
    "_uses_component_data_store",
    "_uses_component_data_store_",
})


def _reload_value_signature(value):
    """Make class-contract values comparable without invoking descriptors."""
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, (list, tuple)):
        return tuple(_reload_value_signature(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_reload_value_signature(item) for item in value))
    if isinstance(value, dict):
        return tuple(sorted(
            (_reload_value_signature(key), _reload_value_signature(item))
            for key, item in value.items()
        ))
    if callable(value):
        return (
            "callable",
            getattr(value, "__module__", ""),
            getattr(value, "__qualname__", repr(value)),
        )
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _serialized_schema_signature(component_type):
    """Compare declarations including the concrete runtime type bindings.

    Reloaded classes can share a module/qualname and storage layout while being
    distinct Python types. Their descriptors must publish with the new method
    globals, even for an identical-source load or a callback-only edit.
    """
    from .fields import get_serialized_fields

    signature = []
    for name, metadata in sorted(get_serialized_fields(component_type).items()):
        values = []
        for field in dataclass_fields(metadata):
            # UI callbacks are implementation details and are not field schema.
            if field.name in {"getter", "setter", "visible_when"}:
                continue
            value = getattr(metadata, field.name)
            values.append((field.name, value if isinstance(value, type) else _reload_value_signature(value)))
        signature.append((name, tuple(values)))
    return tuple(signature)


def _plan_component_class_body_patch(
    target_type: type,
    candidate_type: type,
    *,
    publish_identity: bool = False,
    type_replacements: Optional[dict[type, type]] = None,
) -> tuple[tuple[str, bool, object], ...]:
    """Validate one candidate and return its mutation-free body patch plan."""
    # The script GUID and declaration order identify the authored component;
    # the Python class name is editable source, not a scene identity.  Keep
    # the stable live class object and patch its body even when the candidate
    # was renamed.  This preserves native slots, serialized references, and
    # inspector selection across the edit.
    candidate_bases = tuple(
        (type_replacements or {}).get(base, base) for base in candidate_type.__bases__
    )
    if target_type.__bases__ != candidate_bases:
        raise ScriptReloadRejected(
            f"component '{target_type.__name__}' base classes changed; reload rejected"
        )
    target_schema = _serialized_schema_signature(target_type)
    candidate_schema = _serialized_schema_signature(candidate_type)
    identity_changed = publish_identity and target_type._get_type_guid() != candidate_type._get_type_guid()
    schema_changed = target_schema != candidate_schema or identity_changed

    for key in _BODY_PATCH_CONTRACT_KEYS:
        if _reload_value_signature(getattr(target_type, key, None)) != _reload_value_signature(
            getattr(candidate_type, key, None)
        ):
            raise ScriptReloadRejected(
                f"component '{target_type.__name__}' contract '{key}' changed; reload rejected"
            )

    field_names = {
        name for name, _ in target_schema
    } | {
        name for name, _ in candidate_schema
    }
    target_body = target_type.__dict__
    candidate_body = candidate_type.__dict__
    operations = []
    # Retain the live class object, but publish the declaration's canonical
    # identity. Player cooking derives this same GUID from the source; the
    # schema transaction migrates live slots before retiring the old identity.
    for name in (("__name__", "__qualname__") if publish_identity else ()):
        if getattr(target_type, name) != getattr(candidate_type, name):
            operations.append((name, True, getattr(candidate_type, name)))
    if identity_changed:
        operations.append(("_type_guid_", True, candidate_type._get_type_guid()))
    keys = set(target_body) | set(candidate_body)
    for name in sorted(keys):
        if (
            name in _BODY_PATCH_GENERATED_KEYS
            or name in field_names
            or name in _BODY_PATCH_CONTRACT_KEYS
            or name.startswith("__")
        ):
            continue
        target_has = name in target_body
        candidate_has = name in candidate_body
        if candidate_has:
            value = candidate_body[name]
            if not target_has or target_body[name] is not value:
                operations.append((name, True, value))
        elif target_has:
            operations.append((name, False, None))

    if schema_changed:
        from .fields import SerializedFieldDescriptor

        operations.append(("_field_schemas_", True, candidate_type.__dict__["_field_schemas_"]))
        operations.append(
            (
                "_serialized_fields_",
                True,
                dict(getattr(candidate_type, "_serialized_fields_", {})),
            )
        )
        operations.append(
            (
                "__annotations__",
                True,
                dict(candidate_type.__dict__.get("__annotations__", {})),
            )
        )
        for name in sorted(field_names):
            target_value = target_body.get(name)
            candidate_value = candidate_body.get(name)
            target_has_field = isinstance(target_value, SerializedFieldDescriptor)
            candidate_has_field = isinstance(candidate_value, SerializedFieldDescriptor)
            if candidate_has_field:
                operations.append((name, True, candidate_value))
            elif target_has_field:
                operations.append((name, False, None))
    return tuple(operations)


def _apply_component_body_patch_plans(
    plans: tuple[tuple[type, tuple[tuple[str, bool, object], ...]], ...],
) -> dict[type, tuple[str, ...]]:
    """Publish all validated class patches, rolling back on mutation failure."""
    from .fields import SerializedFieldDescriptor

    snapshots = {}
    for target_type, operations in plans:
        snapshots[target_type] = _snapshot_object_attributes(
            target_type, (name for name, _, _ in operations)
        )

    try:
        for target_type, operations in plans:
            for name, candidate_has, value in operations:
                if candidate_has:
                    setattr(target_type, name, value)
                    # Serialized declarations were compiled on the candidate.
                    # Re-running registration here turns inherited storage
                    # bindings into new declarations on the live subclass.
                    if not isinstance(value, SerializedFieldDescriptor):
                        set_name = getattr(value, "__set_name__", None)
                        if callable(set_name):
                            set_name(target_type, name)
                else:
                    delattr(target_type, name)
        _invalidate_serialized_field_caches(
            target_type for target_type, _operations in plans
        )
        result = {}
        for target_type, operations in plans:
            result[target_type] = tuple(name for name, _candidate_has, _value in operations)
        return result
    except Exception:
        for target_type, snapshot in snapshots.items():
            for name, (previous_has, previous_value) in snapshot.items():
                if previous_has:
                    setattr(target_type, name, previous_value)
                elif name in target_type.__dict__:
                    delattr(target_type, name)
        _invalidate_serialized_field_caches(snapshots)
        raise


def patch_component_class_body(target_type: type, candidate_type: type) -> tuple[str, ...]:
    """Apply one validated body patch while retaining class/instance identity."""
    from infernux.engine.runtime_dispatch import assert_runtime_dispatch_safe_point

    assert_runtime_dispatch_safe_point()
    # This low-level callback patch has no source/registry/schema owner.
    # Declaration identities are published only by a complete file transaction.
    plan = _plan_component_class_body_patch(target_type, candidate_type, publish_identity=False)
    body_snapshot = _snapshot_object_attributes(target_type, (name for name, _, _ in plan))
    publication = None
    try:
        result = _apply_component_body_patch_plans(((target_type, plan),))
        if any(name in _BODY_PATCH_NATIVE_METADATA_KEYS for name, _, _ in plan):
            _refresh_component_metadata(target_type, collect_live_instances(target_type))
        from infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch

        publication = publish_runtime_dispatch_epoch(
            (target_type,),
            defer_commit=True,
        )
        publication.commit()
        return result[target_type]
    except Exception:
        if publication is not None:
            publication.rollback()
        _restore_object_attributes(target_type, body_snapshot)
        if any(name in _BODY_PATCH_NATIVE_METADATA_KEYS for name, _, _ in plan):
            _refresh_component_metadata(target_type, collect_live_instances(target_type))
        _invalidate_serialized_field_caches((target_type,))
        raise


def reload_component_bodies(
    file_path: str,
    target_types: Iterable[type],
    *,
    script_guid: str = "",
    instances_by_type: Optional[dict[type, tuple[object, ...]]] = None,
    source: bytes | str | None = None,
    code: types.CodeType | None = None,
) -> dict[type, tuple[str, ...]]:
    """Import, validate, and atomically publish one script through batch APIs."""
    targets = tuple(dict.fromkeys(target_types))
    if not targets:
        return {}
    transaction = stage_component_body_reload_batch((
        ComponentBodyReloadRequest(
            file_path=file_path,
            target_types=targets,
            instances_by_type=instances_by_type,
            script_guid=script_guid,
            source=source,
            code=code,
        ),
    ))
    result = transaction.commit()
    transaction.finalize()
    return result


# ---------------------------------------------------------------------------
# Script error tracking — allows the editor to know which scripts are broken
# without crashing.  Components backed by broken scripts can still be
# *attached* to GameObjects (they keep their serialized data), but Play
# mode is blocked until every script compiles cleanly.
# ---------------------------------------------------------------------------

# Maps normalised absolute path → error message string
_script_errors: dict[str, str] = {}
_script_error_revision = 0


def _set_script_error_key(key: str, message: str | None) -> None:
    """Update one normalized error entry and advance the change revision."""
    global _script_error_revision
    previous = _script_errors.get(key)
    if message is None:
        if key not in _script_errors:
            return
        _script_errors.pop(key, None)
    else:
        if previous == message:
            return
        _script_errors[key] = message
    _script_error_revision += 1


def _normalize_script_path(file_path: str) -> str:
    """Return a stable absolute key for script-error bookkeeping."""
    return path_key(file_path)


def _unique_module_name_for_path(file_path: str) -> str:
    """Build a fallback module name for scripts without a valid import path."""
    import hashlib

    normalized_path = path_key(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    path_hash = hashlib.md5(normalized_path.encode()).hexdigest()[:8]
    return f"infernux_script_{module_name}_{path_hash}"


def _clear_loaded_script_modules(
    module_names: List[str],
    *,
    preserve_classes: Iterable[type] = (),
) -> None:
    """Drop cached script modules without invalidating surviving instances."""
    if not module_names:
        return

    from .fields import clear_serialized_fields_cache

    preserved_ids = {id(component_type) for component_type in preserve_classes}
    seen_module_ids: set[int] = set()
    for module_name in module_names:
        old_module = sys.modules.get(module_name)
        if old_module is None or id(old_module) in seen_module_ids:
            continue
        seen_module_ids.add(id(old_module))

        old_module_name = getattr(old_module, "__name__", "")
        for _, obj in inspect.getmembers(old_module, inspect.isclass):
            if getattr(obj, '__module__', None) != old_module_name or id(obj) in preserved_ids:
                continue
            if '_serialized_fields_' in obj.__dict__:
                # Old class objects can outlive their module through authored
                # documents, DataAsset caches, Undo history, or a retiring
                # runtime epoch.  Their immutable schema remains authoritative
                # for those instances; the replacement class receives its own
                # independently compiled schema.
                clear_serialized_fields_cache(obj)

    for module_name in module_names:
        sys.modules.pop(module_name, None)


def _record_script_error(file_path: str, exc: Exception) -> None:
    """Record that *file_path* failed to load with *exc*."""
    import traceback
    norm = _normalize_script_path(file_path)
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _set_script_error_key(norm, tb_str)
    # Also log to Console so the user sees it
    try:
        from infernux.debug import Debug
        Debug.log_error(tb_str, source_file=file_path, source_line=0)
    except ImportError:
        import sys
        print(tb_str, file=sys.stderr)


def _load_script_module(
    file_path: str,
    module_name: str,
    *,
    source_only: bool = False,
    source: bytes | str | None = None,
    code: types.CodeType | None = None,
):
    """Execute the exact script artifact resolved from its asset GUID.

    ``import_module`` performs a second path search. That is unnecessary for
    editor sources and unreliable for external sourceless modules in a Nuitka
    standalone Player. Register the canonical name before execution so cyclic
    imports resolve to the same module object.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ScriptLoadError(f"Failed to create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    from .registry import (
        snapshot_component_script_registry, restore_component_script_registry,
        component_types_for_script_path,
    )
    registry_before = snapshot_component_script_registry(file_path)
    previous_types = component_types_for_script_path(file_path)
    serializable_before = None
    try:
        from infernux.renderstack.render_effect_compiler import _stage_source_effect_features
        from .serializable_object import (
            _candidate_serializable_scope, _publish_serializable_types,
        )

        serializable_types: dict[str, type] = {}
        with (
            _stage_source_effect_features(file_path),
            _candidate_serializable_scope(serializable_types, module_name),
        ):
            if code is not None:
                exec(code, module.__dict__)
            elif source_only or source is not None:
                if not file_path.endswith(".py"):
                    raise ScriptLoadError(
                        f"Source-only script reload requires a .py file: {file_path}"
                    )
                # Play-mode candidates must reflect the bytes just saved to disk.
                # SourceFileLoader.exec_module() may reuse a timestamp-based .pyc
                # when a same-second edit also preserves the file size.
                if source is None:
                    with tokenize.open(file_path) as source_file:
                        source = source_file.read()
                code = compile(source, file_path, "exec", dont_inherit=True)
                exec(code, module.__dict__)
            else:
                spec.loader.exec_module(module)
            # First loads and reload candidates share the same data identity
            # ownership boundary. Failed source prefixes cannot replace the
            # published data catalog. Ordinary component declarations retain
            # immediate numeric storage and callback registration semantics.
            serializable_before = _publish_serializable_types(serializable_types, {module_name})
    except Exception:
        if serializable_before is not None:
            from .serializable_object import _restore_serializable_types
            _restore_serializable_types(serializable_before)
        failed_types = tuple(
            cls for cls in _component_classes_from_module(module, module_name)
            if cls not in previous_types
        )
        if failed_types or snapshot_component_script_registry(file_path) != registry_before:
            restore_component_script_registry(file_path, registry_before)
            from infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch
            publication = publish_runtime_dispatch_epoch(previous_types, retired_types=failed_types)
            publication.commit()
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
    return module


def set_script_error(file_path: str, message: str) -> None:
    """Record an error message for a script (no exception object needed)."""
    _set_script_error_key(_normalize_script_path(file_path), message)


def _clear_script_error(file_path: str) -> None:
    """Clear any previously recorded error for *file_path*."""
    _set_script_error_key(_normalize_script_path(file_path), None)


def clear_deleted_script_errors(path: str) -> list[str]:
    """Forget tracked script errors for a deleted script path.

    Accepts either a single script file or a directory path. Directory cleanup is
    useful for editor-side recursive deletes where every nested broken script
    should stop blocking Play Mode immediately instead of waiting for a restart.
    """
    if not path:
        return []

    normalized = _normalize_script_path(path)
    removed: list[str] = []

    if os.path.isdir(path):
        prefix = normalized.rstrip("\\/") + os.sep
        for key in list(_script_errors.keys()):
            if key == normalized or key.startswith(prefix):
                _set_script_error_key(key, None)
                removed.append(key)
        return removed

    if normalized in _script_errors:
        _set_script_error_key(normalized, None)
        removed.append(normalized)
    return removed


def get_script_errors() -> dict[str, str]:
    """Return a snapshot of all currently broken scripts {path: traceback}."""
    return dict(_script_errors)


def has_script_errors() -> bool:
    """Return True if any loaded script has unresolved errors."""
    return bool(_script_errors)


def get_script_error_revision() -> int:
    """Return a monotonic revision changed only when diagnostics change."""
    return _script_error_revision


def get_script_error_by_path(file_path: str) -> Optional[str]:
    """Return the error string for *file_path*, or ``None`` if it loaded OK."""
    return _script_errors.get(_normalize_script_path(file_path))


def load_component_from_file(file_path: str) -> Type[InxComponent]:
    """
    Load the first InxComponent subclass from a Python file.
    
    Args:
        file_path: Absolute path to the .py file
        
    Returns:
        The first InxComponent subclass found in the file
        
    Raises:
        ScriptLoadError: If file doesn't exist, can't be imported, or contains no components
    """
    components = load_all_components_from_file(file_path)
    if not components:
        raise ScriptLoadError(f"No InxComponent subclasses found in {file_path}")
    if len(components) > 1:
        names = ", ".join(cls.__name__ for cls in components)
        raise ScriptLoadError(
            f"Script '{file_path}' defines multiple InxComponent classes ({names}). "
            "Dragging or attaching by script file requires exactly one component class."
        )
    return components[0]


def load_all_components_from_file(
    file_path: str,
    *,
    preserve_classes: Iterable[type] = (),
    register: bool = True,
    source_only: bool = False,
    source: bytes | str | None = None,
    code: types.CodeType | None = None,
) -> List[Type[InxComponent]]:
    """
    Load all InxComponent subclasses from a Python file.
    
    Args:
        file_path: Absolute path to the .py file
        
    Returns:
        List of InxComponent subclasses found in the file (may be empty)
        
    Raises:
        ScriptLoadError: If file doesn't exist or can't be imported
    """
    # Resolve path (project-relative allowed)
    file_path = resolve_script_path(file_path)

    # Validate file exists
    if not os.path.exists(file_path):
        raise ScriptLoadError(f"Script file not found: {file_path}")
    
    if not file_path.endswith(('.py', '.pyc')):
        raise ScriptLoadError(f"Not a Python file: {file_path}")
    
    module_name = get_script_module_name(file_path) or _unique_module_name_for_path(file_path)
    previous_module = sys.modules.get(module_name, _MODULE_ABSENT)
    _clear_loaded_script_modules([module_name], preserve_classes=preserve_classes)

    importlib.invalidate_caches()

    # Execute the module — catch errors so a broken script never crashes the editor
    try:
        with temporary_script_import_paths(file_path):
            module = _load_script_module(
                file_path,
                module_name,
                source_only=source_only,
                source=source,
                code=code,
            )
    except Exception as exc:
        if previous_module is not _MODULE_ABSENT:
            sys.modules[module_name] = previous_module
        # Track this script as having a load error
        _record_script_error(file_path, exc)
        # Return empty list — the component can still be referenced by GUID/type
        # but will not be instantiable until the script is fixed.
        return []

    # If we get here the script loaded successfully — clear any prior error
    _clear_script_error(file_path)

    # Direct imports and engine-created component instances share class identity.
    sys.modules[module_name] = module

    # Find all InxComponent subclasses in the module.
    components = _component_classes_from_module(module, module_name)

    if register:
        from .registry import publish_component_script_types
        publish_component_script_types(file_path, tuple(components))

    return components


def load_component_class_from_file(file_path: str, type_name: str = "") -> Optional[Type[InxComponent]]:
    """Load a specific component class from a Python file.

    When ``type_name`` is provided, prefer an exact class-name match. If the
    authored name is missing but the file still defines exactly one
    ``InxComponent`` subclass, return that class so a pure class rename
    (same script GUID / one-component-per-file) keeps scene references alive.
    """
    components = load_all_components_from_file(file_path)
    if not components:
        return None

    if type_name:
        for component_class in components:
            if component_class.__name__ == type_name:
                return component_class

    # A script asset is attached by file/GUID, not by a fragile class-name
    # convention.  When the authored name differs (or the file declares more
    # than one component), the first declaration is the stable editor target.
    return components[0]



def create_component_instance(component_class: Type[InxComponent]) -> InxComponent:
    """
    Create an instance of a component class.
    
    Args:
        component_class: The InxComponent subclass to instantiate
        
    Returns:
        New instance of the component
        
    Raises:
        ScriptLoadError: If instantiation fails
    """
    return component_class()


def load_and_create_component(
    file_path: str,
    asset_database=None,
    type_name: str = "",
    *,
    script_guid: str = "",
) -> Optional[InxComponent]:
    """
    Convenience function: Load first component from file and create instance.
    
    Args:
        file_path: Absolute path to the .py file
        
    Returns:
        New instance of the first component found, or None if the script
        has errors (the error is already logged to Console).
        
    Note:
        ``script_guid`` should be supplied when the caller already resolved
        ``file_path`` from a stable component identity. Packaged ``.pyc``
        artifacts do not necessarily support a reverse path-to-GUID lookup.

    Raises:
        ScriptLoadError: If AssetDatabase is missing or GUID cannot be resolved.
    """
    if asset_database is None and not script_guid:
        raise ScriptLoadError("AssetDatabase is required for script components (GUID-only mode)")

    # Resolve and store script GUID
    guid = script_guid or asset_database.get_guid_from_path(file_path)
    if not guid and asset_database is not None:
        from infernux.core.assets import AssetManager
        mutation = AssetManager.import_asset(
            file_path,
            database=asset_database,
            suppress_watcher_echo=False,
        )
        guid = mutation.guid
    if not guid:
        raise ScriptLoadError(f"Failed to resolve GUID for script: {file_path}")

    # A script file is one identity/publication unit even when it declares
    # several components. Methods on one component may refer to sibling types
    # through their module globals, so binding only the requested class leaves
    # those references on provisional module GUIDs and breaks component lookup.
    from infernux.components.registry import (
        component_types_for_script_path,
        publish_component_script_types,
    )
    # Instantiation consumes the published revision; only the script-refresh
    # transaction replaces it. Re-executing here strands earlier instances.
    component_types = component_types_for_script_path(file_path)
    already_published = bool(component_types) and all(
        candidate._asset_script_guid_ == guid for candidate in component_types
    )
    if not already_published:
        component_types = tuple(load_all_components_from_file(file_path, register=False))
    if not component_types:
        return None
    component_class = next(
        (candidate for candidate in component_types if type_name and candidate.__name__ == type_name),
        component_types[0],
    )

    if not already_published:
        from infernux.components.component_identity import bind_asset_script_guid
        for candidate in component_types:
            bind_asset_script_guid(candidate, guid, register=False)
        publish_component_script_types(file_path, component_types)

    instance = create_component_instance(component_class)
    instance._script_guid = guid
    instance._script_path = resolved_path(file_path)
    return instance


def get_component_info(component_class: Type[InxComponent]) -> dict:
    """
    Extract metadata from a component class.
    
    Args:
        component_class: The InxComponent subclass
        
    Returns:
        Dictionary with component metadata (name, docstring, fields)
    """
    from .fields import get_serialized_fields
    
    return {
        'name': component_class.__name__,
        'module': component_class.__module__,
        'docstring': inspect.getdoc(component_class) or "",
        'fields': list(get_serialized_fields(component_class).keys()),
    }


# Example usage (for testing):
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
        print(f"Loading components from: {script_path}")

        components = load_all_components_from_file(script_path)
        print(f"Found {len(components)} component(s):")

        for comp_class in components:
            info = get_component_info(comp_class)
            print(f"\n  - {info['name']}")
            print(f"    Doc: {info['docstring'][:50]}...")
            print(f"    Fields: {info['fields']}")

            # Try to instantiate
            instance = create_component_instance(comp_class)
            print(f"    [OK] Instantiation successful")

    else:
        print("Usage: python script_loader.py <path_to_script.py>")
