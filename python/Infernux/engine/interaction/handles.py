"""Project-extensible Scene View handles with preload-owned lifetimes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ._contributions import current_owner


Vec3 = tuple[float, float, float]
HandleChanged = Callable[[Any], object]
HandleCallback = Callable[[Any], object]


class EditorHandleKind(str, Enum):
    POSITION = "position"
    DIRECTION = "direction"
    RADIUS = "radius"
    LIMIT = "limit"


@dataclass(slots=True)
class EditorHandleProvider:
    """One named source of immediate-mode Scene View handles.

    ``draw`` is called once per Gizmo collection with an
    :class:`EditorHandleContext`.  ``on_retire`` releases provider-owned GPU or
    native resources when the registration, preload, or plugin is removed.
    """

    provider_id: str
    draw: Callable[["EditorHandleContext"], object]
    enabled: Optional[Callable[[], bool]] = None
    on_retire: Optional[Callable[[], object]] = None

    def __post_init__(self) -> None:
        self.provider_id = _identifier(self.provider_id, "handle provider_id")
        if not callable(self.draw):
            raise TypeError("handle provider draw callback must be callable")
        if self.enabled is not None and not callable(self.enabled):
            raise TypeError("handle provider enabled predicate must be callable")
        if self.on_retire is not None and not callable(self.on_retire):
            raise TypeError("handle provider on_retire callback must be callable")


@dataclass(frozen=True, slots=True)
class EditorHandleSnapshot:
    handle_id: str
    provider_id: str
    kind: EditorHandleKind
    value: Any
    position: Vec3
    direction: Vec3
    size: float


@dataclass(slots=True)
class _FrameHandle:
    handle_id: str
    provider_id: str
    owner: str
    kind: EditorHandleKind
    value: Any
    position: Vec3
    direction: Vec3
    size: float
    on_changed: HandleChanged
    on_begin: Optional[HandleCallback]
    on_commit: Optional[HandleCallback]
    on_cancel: Optional[HandleCallback]
    extent: float = 1.0

    def snapshot(self) -> EditorHandleSnapshot:
        return EditorHandleSnapshot(
            self.handle_id,
            self.provider_id,
            self.kind,
            self.value,
            self.position,
            self.direction,
            self.size,
        )


@dataclass(slots=True)
class _Capture:
    descriptor: _FrameHandle
    initial_value: Any
    initial_hit: Vec3
    initial_parameter: float
    plane_normal: Vec3
    endpoint: int = 0
    changed: bool = False
    transient_token: str = ""


@dataclass(slots=True)
class _RegistrationState:
    provider: EditorHandleProvider
    owner: str
    serial: int


class HandleRegistration:
    """Explicit registration lease. ``close`` is idempotent."""

    __slots__ = ("_registry", "provider_id", "_serial", "_closed")

    def __init__(self, registry: "EditorHandleRegistry", provider_id: str, serial: int) -> None:
        self._registry = registry
        self.provider_id = provider_id
        self._serial = serial
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        return self._registry._unregister_serial(self.provider_id, self._serial)

    def __enter__(self) -> "HandleRegistration":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


class EditorHandleContext:
    """Immediate-mode submission surface passed to a handle provider."""

    __slots__ = (
        "engine", "selection", "provider_id", "_registry", "_owner",
        "_staged", "_ids", "_sealed",
    )

    def __init__(self, registry, provider_id: str, owner: str, engine, selection) -> None:
        self.engine = engine
        self.selection = selection
        self.provider_id = provider_id
        self._registry = registry
        self._owner = owner
        self._staged: list[_FrameHandle] = []
        self._ids: set[str] = set()
        self._sealed = False

    def position(
        self,
        handle_id: str,
        value,
        on_changed: HandleChanged,
        *,
        size: float = 0.15,
        color=(0.98, 0.72, 0.24),
        on_begin: HandleCallback | None = None,
        on_commit: HandleCallback | None = None,
        on_cancel: HandleCallback | None = None,
    ) -> None:
        """Submit a camera-plane position handle in world coordinates."""
        point = _vec3(value, "position handle value")
        descriptor = self._descriptor(
            handle_id, EditorHandleKind.POSITION, point, point, (0.0, 0.0, 0.0),
            size, on_changed, on_begin, on_commit, on_cancel,
        )
        self._draw_point(descriptor, color)

    def direction(
        self,
        handle_id: str,
        origin,
        value,
        on_changed: HandleChanged,
        *,
        length: float = 1.0,
        size: float = 0.15,
        color=(0.34, 0.78, 1.0),
        on_begin: HandleCallback | None = None,
        on_commit: HandleCallback | None = None,
        on_cancel: HandleCallback | None = None,
    ) -> None:
        """Submit a direction endpoint; changes are normalized world vectors."""
        base = _vec3(origin, "direction handle origin")
        direction = _normalized(_vec3(value, "direction handle value"), "direction handle value")
        extent = _positive(length, "direction handle length")
        descriptor = self._descriptor(
            handle_id, EditorHandleKind.DIRECTION, direction, base, direction,
            size, on_changed, on_begin, on_commit, on_cancel,
        )
        descriptor.extent = extent
        endpoint = _add(base, _scale(direction, extent))
        descriptor.size = max(descriptor.size, extent * 0.04)
        self._draw_line_point(descriptor, base, endpoint, color)

    def radius(
        self,
        handle_id: str,
        center,
        value: float,
        on_changed: HandleChanged,
        *,
        size: float = 0.12,
        color=(0.63, 0.92, 0.35),
        on_begin: HandleCallback | None = None,
        on_commit: HandleCallback | None = None,
        on_cancel: HandleCallback | None = None,
    ) -> None:
        """Submit a non-negative world-space radius handle."""
        point = _vec3(center, "radius handle center")
        radius = float(value)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("radius handle value must be finite and non-negative")
        descriptor = self._descriptor(
            handle_id, EditorHandleKind.RADIUS, radius, point, (0.0, 0.0, 0.0),
            size, on_changed, on_begin, on_commit, on_cancel,
        )
        from Infernux.gizmos import Gizmos
        old = Gizmos.color
        Gizmos.color = self._color(descriptor, color)
        try:
            Gizmos.draw_wire_sphere(point, radius)
        finally:
            Gizmos.color = old

    def limit(
        self,
        handle_id: str,
        origin,
        axis,
        value,
        on_changed: HandleChanged,
        *,
        size: float = 0.13,
        color=(0.94, 0.48, 0.37),
        on_begin: HandleCallback | None = None,
        on_commit: HandleCallback | None = None,
        on_cancel: HandleCallback | None = None,
    ) -> None:
        """Submit two scalar endpoints along a normalized world-space axis."""
        base = _vec3(origin, "limit handle origin")
        direction = _normalized(_vec3(axis, "limit handle axis"), "limit handle axis")
        try:
            lower, upper = (float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise TypeError("limit handle value must contain (minimum, maximum)") from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("limit handle requires finite minimum <= maximum")
        limits = (lower, upper)
        descriptor = self._descriptor(
            handle_id, EditorHandleKind.LIMIT, limits, base, direction,
            size, on_changed, on_begin, on_commit, on_cancel,
        )
        self._draw_line_points(
            descriptor,
            _add(base, _scale(direction, lower)),
            _add(base, _scale(direction, upper)),
            color,
        )

    def _descriptor(
        self, local_id, kind, value, position, direction, size, on_changed,
        on_begin, on_commit, on_cancel,
    ) -> _FrameHandle:
        if self._sealed:
            raise RuntimeError("EditorHandleContext is valid only during its provider callback")
        local = _identifier(local_id, "handle_id")
        if local in self._ids:
            raise ValueError(f"handle provider submitted duplicate handle_id: {local}")
        self._ids.add(local)
        if not callable(on_changed):
            raise TypeError("handle on_changed callback must be callable")
        for name, callback in (
            ("on_begin", on_begin), ("on_commit", on_commit), ("on_cancel", on_cancel),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"handle {name} callback must be callable")
        descriptor = _FrameHandle(
            f"{self.provider_id}:{local}", self.provider_id, self._owner, kind,
            value, position, direction, _positive(size, "handle size"),
            on_changed, on_begin, on_commit, on_cancel,
        )
        self._staged.append(descriptor)
        return descriptor

    def _color(self, descriptor: _FrameHandle, value) -> tuple[float, float, float]:
        if descriptor.handle_id in {
            self._registry.hovered_handle_id, self._registry.active_capture_id,
        }:
            return (1.0, 0.92, 0.18)
        return _vec3(value, "handle color")

    def _draw_point(self, descriptor, color) -> None:
        point = descriptor.position
        extent = descriptor.size
        from Infernux.gizmos import Gizmos
        old = Gizmos.color
        Gizmos.color = self._color(descriptor, color)
        try:
            Gizmos.draw_line(_add(point, (-extent, 0.0, 0.0)), _add(point, (extent, 0.0, 0.0)))
            Gizmos.draw_line(_add(point, (0.0, -extent, 0.0)), _add(point, (0.0, extent, 0.0)))
            Gizmos.draw_line(_add(point, (0.0, 0.0, -extent)), _add(point, (0.0, 0.0, extent)))
            Gizmos.draw_wire_sphere(point, extent * 0.28, segments=12)
        finally:
            Gizmos.color = old

    def _draw_line_point(self, descriptor, start, end, color) -> None:
        from Infernux.gizmos import Gizmos
        old = Gizmos.color
        Gizmos.color = self._color(descriptor, color)
        try:
            Gizmos.draw_line(start, end)
            Gizmos.draw_wire_sphere(end, descriptor.size * 0.35, segments=12)
        finally:
            Gizmos.color = old

    def _draw_line_points(self, descriptor, start, end, color) -> None:
        from Infernux.gizmos import Gizmos
        old = Gizmos.color
        Gizmos.color = self._color(descriptor, color)
        try:
            Gizmos.draw_line(start, end)
            radius = descriptor.size * 0.35
            Gizmos.draw_wire_sphere(start, radius, segments=12)
            Gizmos.draw_wire_sphere(end, radius, segments=12)
        finally:
            Gizmos.color = old


class EditorHandleRegistry:
    """Authoritative provider, frame publication, and pointer-capture owner."""

    _instance: Optional["EditorHandleRegistry"] = None

    def __init__(self) -> None:
        self._providers: dict[str, _RegistrationState] = {}
        self._frame: dict[str, _FrameHandle] = {}
        self._serial = 0
        self._capture: _Capture | None = None
        self._hovered = ""
        self._revision = 0
        EditorHandleRegistry._instance = self

    @classmethod
    def instance(cls) -> "EditorHandleRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def providers(self) -> tuple[EditorHandleProvider, ...]:
        return tuple(state.provider for state in self._providers.values())

    @property
    def frame_handles(self) -> tuple[EditorHandleSnapshot, ...]:
        return tuple(item.snapshot() for item in self._frame.values())

    @property
    def active_capture_id(self) -> str:
        return self._capture.descriptor.handle_id if self._capture is not None else ""

    @property
    def hovered_handle_id(self) -> str:
        return self._hovered

    def register(self, provider: EditorHandleProvider, *, replace: bool = False) -> HandleRegistration:
        existing = self._providers.get(provider.provider_id)
        owner = current_owner.get()
        if existing is not None and not replace:
            raise ValueError(f"editor handle provider already registered: {provider.provider_id}")
        if existing is not None and existing.owner != owner:
            raise ValueError(
                f"editor handle provider belongs to another contributor: {provider.provider_id}"
            )
        if existing is not None:
            self._providers.pop(provider.provider_id)
            self._revision += 1
            self._retire_state(existing)
        self._serial += 1
        state = _RegistrationState(provider, owner, self._serial)
        self._providers[provider.provider_id] = state
        self._revision += 1
        return HandleRegistration(self, provider.provider_id, state.serial)

    def unregister(self, provider_id: str) -> bool:
        identifier = _identifier(provider_id, "handle provider_id")
        state = self._providers.pop(identifier, None)
        if state is None:
            return False
        self._revision += 1
        self._retire_state(state)
        return True

    def _unregister_serial(self, provider_id: str, serial: int) -> bool:
        state = self._providers.get(provider_id)
        if state is None or state.serial != serial:
            return False
        return self.unregister(provider_id)

    def unregister_owner(self, owner: str) -> int:
        contributor = str(owner or "").strip()
        if not contributor:
            raise ValueError("contribution owner must not be empty")
        identifiers = tuple(
            key for key, state in self._providers.items() if state.owner == contributor
        )
        failures: list[Exception] = []
        for identifier in identifiers:
            try:
                self.unregister(identifier)
            except Exception as exc:
                failures.append(exc)
        _raise_callback_failures(
            f"editor handle owner retirement failed for {contributor}", failures,
        )
        return len(identifiers)

    def clear(self) -> None:
        failures: list[Exception] = []
        for identifier in tuple(self._providers):
            try:
                self.unregister(identifier)
            except Exception as exc:
                failures.append(exc)
        self._frame.clear()
        self._hovered = ""
        _raise_callback_failures("editor handle registry retirement failed", failures)

    def collect(self, engine) -> None:
        """Publish one exact frame of custom handles and draw their Gizmos."""
        from .selection import SelectionService

        published: dict[str, _FrameHandle] = {}
        selection = SelectionService.instance().snapshot
        try:
            for state in tuple(self._providers.values()):
                provider = state.provider
                if provider.enabled is not None and not bool(provider.enabled()):
                    if self._capture is not None and self._capture.descriptor.provider_id == provider.provider_id:
                        self.cancel_capture()
                    continue
                context = EditorHandleContext(
                    self, provider.provider_id, state.owner, engine, selection,
                )
                try:
                    provider.draw(context)
                finally:
                    context._sealed = True
                for descriptor in context._staged:
                    if descriptor.handle_id in published:
                        raise ValueError(f"duplicate editor handle identity: {descriptor.handle_id}")
                    published[descriptor.handle_id] = descriptor
        except BaseException as collection_error:
            # A failed provider cannot leave the previous frame interactive.
            # Retire its callbacks and active gesture at the same failed
            # publication boundary instead of exposing stale handles.
            try:
                self.retire_frame()
            except BaseException as retirement_error:
                raise BaseExceptionGroup(
                    "editor handle collection and retirement failed",
                    [collection_error, retirement_error],
                ) from None
            raise
        self._frame = published
        if self._capture is not None and self._capture.descriptor.handle_id not in published:
            self.cancel_capture()
        if self._hovered not in published:
            self._hovered = ""

    def retire_frame(self) -> None:
        """Cancel live capture and release all frame-scoped callback references."""
        try:
            self.cancel_capture()
        finally:
            self._frame.clear()
            self._hovered = ""

    def hit_test(self, ray_origin, ray_direction) -> str:
        origin = _vec3(ray_origin, "handle hit-test ray origin")
        direction = _normalized(
            _vec3(ray_direction, "handle hit-test ray direction"),
            "handle hit-test ray direction",
        )
        best_id = ""
        best_score = math.inf
        for descriptor in self._frame.values():
            score = self._hit_score(descriptor, origin, direction)
            if score is not None and score < best_score:
                best_score = score
                best_id = descriptor.handle_id
        return best_id

    def set_hovered(self, handle_id: str) -> None:
        identifier = str(handle_id or "")
        if identifier and identifier not in self._frame:
            raise KeyError(f"editor handle is not in the current frame: {identifier}")
        self._hovered = identifier

    def begin_capture(
        self, handle_id: str, ray_origin, ray_direction, *, interaction_owner: str = "",
    ) -> bool:
        identifier = str(handle_id or "")
        descriptor = self._frame.get(identifier)
        if descriptor is None:
            return False
        if self._capture is not None:
            raise RuntimeError("another editor handle already owns pointer capture")
        origin = _vec3(ray_origin, "handle capture ray origin")
        direction = _normalized(_vec3(ray_direction, "handle capture ray direction"), "handle capture ray direction")
        endpoint = self._picked_endpoint(descriptor, origin, direction)
        anchor = self._anchor(descriptor, endpoint)
        plane_normal = _normalized(_sub(origin, anchor), "handle camera plane")
        hit = _plane_hit(origin, direction, anchor, plane_normal)
        if hit is None:
            hit = anchor
        parameter = _closest_axis_parameter(origin, direction, descriptor.position, descriptor.direction)
        self._capture = _Capture(
            descriptor, descriptor.value, hit, parameter, plane_normal, endpoint,
        )
        self._hovered = identifier
        if descriptor.on_begin is not None:
            try:
                descriptor.on_begin(descriptor.value)
            except BaseException:
                self._capture = None
                raise
        owner = str(interaction_owner or "").strip()
        if owner:
            from .transient_interactions import TransientInteractionService

            try:
                self._capture.transient_token = TransientInteractionService.instance().begin(
                    owner,
                    self.cancel_capture,
                    kind="custom_scene_handle_drag",
                    priority=110,
                    token_id=f"{owner}:custom_handle_drag",
                )
            except BaseException:
                self.cancel_capture()
                raise
        return True

    def update_capture(self, ray_origin, ray_direction) -> bool:
        capture = self._capture
        if capture is None:
            return False
        descriptor = capture.descriptor
        origin = _vec3(ray_origin, "handle capture ray origin")
        direction = _normalized(_vec3(ray_direction, "handle capture ray direction"), "handle capture ray direction")
        value = self._capture_value(capture, origin, direction)
        descriptor.on_changed(value)
        descriptor.value = value
        capture.changed = capture.changed or value != capture.initial_value
        return True

    def finish_capture(self, *, commit: bool = True) -> bool:
        capture = self._capture
        if capture is None:
            return False
        self._capture = None
        descriptor = capture.descriptor
        if capture.transient_token:
            from .transient_interactions import TransientInteractionService

            TransientInteractionService.instance().end(capture.transient_token)
        if not commit:
            failures: list[Exception] = []
            try:
                descriptor.on_changed(capture.initial_value)
            except Exception as exc:
                failures.append(exc)
            if descriptor.on_cancel is not None:
                try:
                    descriptor.on_cancel(capture.initial_value)
                except Exception as exc:
                    failures.append(exc)
            _raise_callback_failures(
                f"editor handle cancellation failed for {descriptor.handle_id}",
                failures,
            )
            return True
        if descriptor.on_commit is not None:
            descriptor.on_commit(descriptor.value)
        return True

    def cancel_capture(self) -> bool:
        return self.finish_capture(commit=False)

    def process_pointer(
        self, engine, mouse_x: float, mouse_y: float, width: float, height: float,
        *, left_down: bool, left_clicked: bool, hovered: bool,
    ) -> bool:
        """Drive custom handles from the Scene View's authoritative pointer edge."""
        if self._capture is not None:
            if not left_down:
                self.finish_capture(commit=True)
                return True
            ray = engine.screen_to_world_ray(mouse_x, mouse_y, width, height)
            self.update_capture(ray[:3], ray[3:])
            return True
        if not hovered:
            self._hovered = ""
            return False
        ray = engine.screen_to_world_ray(mouse_x, mouse_y, width, height)
        identifier = self.hit_test(ray[:3], ray[3:])
        self._hovered = identifier
        if not identifier:
            return False
        if left_clicked:
            self.begin_capture(
                identifier, ray[:3], ray[3:], interaction_owner="scene_view",
            )
        return True

    def _retire_state(self, state: _RegistrationState) -> None:
        provider_id = state.provider.provider_id
        failures: list[Exception] = []
        if self._capture is not None and self._capture.descriptor.provider_id == provider_id:
            try:
                self.cancel_capture()
            except Exception as exc:
                failures.append(exc)
        self._frame = {
            key: value for key, value in self._frame.items()
            if value.provider_id != provider_id
        }
        if self._hovered.startswith(provider_id + ":"):
            self._hovered = ""
        callback = state.provider.on_retire
        if callback is not None:
            try:
                callback()
            except Exception as exc:
                failures.append(exc)
        _raise_callback_failures(
            f"editor handle provider retirement failed for {provider_id}", failures,
        )

    @staticmethod
    def _anchor(descriptor: _FrameHandle, endpoint: int) -> Vec3:
        if descriptor.kind is EditorHandleKind.DIRECTION:
            return _add(
                descriptor.position,
                _scale(descriptor.direction, descriptor.extent),
            )
        if descriptor.kind is EditorHandleKind.LIMIT:
            value = descriptor.value[endpoint]
            return _add(descriptor.position, _scale(descriptor.direction, value))
        return descriptor.position

    @staticmethod
    def _picked_endpoint(descriptor: _FrameHandle, ray_origin: Vec3, ray_direction: Vec3) -> int:
        if descriptor.kind is not EditorHandleKind.LIMIT:
            return 0
        lower = _add(descriptor.position, _scale(descriptor.direction, descriptor.value[0]))
        upper = _add(descriptor.position, _scale(descriptor.direction, descriptor.value[1]))
        return int(_ray_point_distance(ray_origin, ray_direction, upper) < _ray_point_distance(ray_origin, ray_direction, lower))

    @staticmethod
    def _hit_score(descriptor: _FrameHandle, ray_origin: Vec3, ray_direction: Vec3) -> float | None:
        threshold = descriptor.size
        if descriptor.kind is EditorHandleKind.POSITION:
            distance = _ray_point_distance(ray_origin, ray_direction, descriptor.position)
        elif descriptor.kind is EditorHandleKind.DIRECTION:
            endpoint = _add(
                descriptor.position,
                _scale(descriptor.direction, descriptor.extent),
            )
            distance = _ray_point_distance(ray_origin, ray_direction, endpoint)
        elif descriptor.kind is EditorHandleKind.RADIUS:
            normal = _normalized(_sub(ray_origin, descriptor.position), "radius handle camera plane")
            hit = _plane_hit(ray_origin, ray_direction, descriptor.position, normal)
            if hit is None:
                return None
            distance = abs(_length(_sub(hit, descriptor.position)) - float(descriptor.value))
        else:
            lower, upper = descriptor.value
            a = _add(descriptor.position, _scale(descriptor.direction, lower))
            b = _add(descriptor.position, _scale(descriptor.direction, upper))
            distance = min(
                _ray_point_distance(ray_origin, ray_direction, a),
                _ray_point_distance(ray_origin, ray_direction, b),
            )
        return distance / threshold if distance <= threshold else None

    @staticmethod
    def _capture_value(capture: _Capture, ray_origin: Vec3, ray_direction: Vec3):
        descriptor = capture.descriptor
        if descriptor.kind is EditorHandleKind.LIMIT:
            parameter = _closest_axis_parameter(
                ray_origin, ray_direction, descriptor.position, descriptor.direction,
            )
            delta = parameter - capture.initial_parameter
            values = list(capture.initial_value)
            values[capture.endpoint] += delta
            if capture.endpoint == 0:
                values[0] = min(values[0], values[1])
            else:
                values[1] = max(values[1], values[0])
            return tuple(values)
        hit = _plane_hit(
            ray_origin, ray_direction, capture.initial_hit, capture.plane_normal,
        )
        if hit is None:
            return descriptor.value
        delta = _sub(hit, capture.initial_hit)
        if descriptor.kind is EditorHandleKind.POSITION:
            return _add(capture.initial_value, delta)
        if descriptor.kind is EditorHandleKind.DIRECTION:
            endpoint = _add(
                descriptor.position,
                _scale(descriptor.direction, descriptor.extent),
            )
            return _normalized(_sub(_add(endpoint, delta), descriptor.position), "direction handle value")
        return max(0.0, _length(_sub(hit, descriptor.position)))


def register_handle_provider(
    provider_id: str,
    draw: Callable[[EditorHandleContext], object],
    *,
    enabled: Callable[[], bool] | None = None,
    on_retire: Callable[[], object] | None = None,
    replace: bool = False,
) -> HandleRegistration:
    """Register a project/plugin Scene View handle provider.

    Registrations made during :class:`~Infernux.lifecycle.InxPreload` import or
    ``preload`` are automatically owned by that lifecycle and removed on
    reload, disable, uninstall, and editor shutdown.
    """
    return EditorHandleRegistry.instance().register(
        EditorHandleProvider(provider_id, draw, enabled, on_retire),
        replace=replace,
    )


def _identifier(value, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    if ":" in result:
        raise ValueError(f"{label} must not contain ':'")
    return result


def _positive(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _vec3(value, label: str) -> Vec3:
    try:
        result = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise TypeError(f"{label} must contain three numeric values") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _scale(value: Vec3, scalar: float) -> Vec3:
    return value[0] * scalar, value[1] * scalar, value[2] * scalar


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(value: Vec3) -> float:
    return math.sqrt(max(0.0, _dot(value, value)))


def _normalized(value: Vec3, label: str) -> Vec3:
    length = _length(value)
    if length <= 1e-8:
        raise ValueError(f"{label} must not be zero")
    return _scale(value, 1.0 / length)


def _plane_hit(ray_origin: Vec3, ray_direction: Vec3, origin: Vec3, normal: Vec3) -> Vec3 | None:
    denominator = _dot(ray_direction, normal)
    if abs(denominator) <= 1e-8:
        return None
    parameter = _dot(_sub(origin, ray_origin), normal) / denominator
    if parameter < 0.0:
        return None
    return _add(ray_origin, _scale(ray_direction, parameter))


def _ray_point_distance(ray_origin: Vec3, ray_direction: Vec3, point: Vec3) -> float:
    parameter = max(0.0, _dot(_sub(point, ray_origin), ray_direction))
    closest = _add(ray_origin, _scale(ray_direction, parameter))
    return _length(_sub(point, closest))


def _closest_axis_parameter(
    ray_origin: Vec3, ray_direction: Vec3, axis_origin: Vec3, axis_direction: Vec3,
) -> float:
    # Closest points between two infinite lines. Near-parallel views retain the
    # axis projection instead of inventing a screen-space fallback.
    offset = _sub(ray_origin, axis_origin)
    cross_term = _dot(ray_direction, axis_direction)
    denominator = 1.0 - cross_term * cross_term
    if abs(denominator) <= 1e-8:
        return _dot(offset, axis_direction)
    return (_dot(offset, axis_direction) - _dot(offset, ray_direction) * cross_term) / denominator


def _raise_callback_failures(label: str, failures: list[Exception]) -> None:
    if not failures:
        return
    detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures)
    raise RuntimeError(f"{label}: {detail}") from failures[0]


__all__ = [
    "EditorHandleContext",
    "EditorHandleKind",
    "EditorHandleProvider",
    "EditorHandleRegistry",
    "EditorHandleSnapshot",
    "HandleRegistration",
    "register_handle_provider",
]
