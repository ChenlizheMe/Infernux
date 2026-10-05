"""Python-facing wrapper for shared native PhysicMaterial assets."""

from __future__ import annotations

from typing import Optional

from infernux.lib import AssetRegistry, InxPhysicMaterial as NativePhysicMaterial
from infernux.core._resource_proxy import ResourceProxy


class PhysicMaterial(ResourceProxy):
    def __new__(cls, native: Optional[NativePhysicMaterial] = None):
        return ResourceProxy.__new__(cls, native if native is not None else NativePhysicMaterial())

    def __init__(self, native: Optional[NativePhysicMaterial] = None):
        """Native creation and proxy identity are established by __new__."""

    @staticmethod
    def load(path: str) -> Optional["PhysicMaterial"]:
        native = AssetRegistry.instance().load_physic_material(path)
        return PhysicMaterial(native) if native is not None else None

    @staticmethod
    def load_by_guid(guid: str) -> Optional["PhysicMaterial"]:
        if not guid:
            return None
        native = AssetRegistry.instance().load_physic_material_by_guid(guid)
        return PhysicMaterial(native) if native is not None else None

    @property
    def native(self) -> NativePhysicMaterial:
        return self._native

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def guid(self) -> str:
        return self._native.guid

    @property
    def file_path(self) -> str:
        return self._native.file_path

    @property
    def friction(self) -> float:
        return self._native.friction

    @friction.setter
    def friction(self, value: float) -> None:
        self._native.friction = value

    @property
    def bounciness(self) -> float:
        return self._native.bounciness

    @bounciness.setter
    def bounciness(self, value: float) -> None:
        self._native.bounciness = value

    @property
    def friction_combine(self) -> int:
        return self._native.friction_combine

    @friction_combine.setter
    def friction_combine(self, value: int) -> None:
        self._native.friction_combine = int(value)

    @property
    def bounce_combine(self) -> int:
        return self._native.bounce_combine

    @bounce_combine.setter
    def bounce_combine(self, value: int) -> None:
        self._native.bounce_combine = int(value)

    def serialize_document(self) -> dict:
        return self._native.serialize_document()

    def deserialize_document(self, document: dict) -> None:
        self._native.deserialize_document(document)

    def save(self, path: str = "") -> None:
        if path:
            self._native.save_to(path)
        else:
            self._native.save()
