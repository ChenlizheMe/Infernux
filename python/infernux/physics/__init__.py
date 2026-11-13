"""
infernux.physics — Physics query API (Unity: UnityEngine.Physics).

Provides ``Physics.raycast()``, ``Physics.raycast_all()``,
``Physics.overlap_sphere()``, ``Physics.overlap_box()``, ``Physics.overlap_capsule()``,
``Physics.sphere_cast()``, ``Physics.box_cast()``, ``Physics.capsule_cast()``,
and gravity / layer-collision control.

Example::

    from infernux.physics import Physics
    from infernux.math import Vector3

    hit = Physics.raycast(Vector3(0, 10, 0), Vector3.down)
    if hit is not None:
        print(f"Hit {hit.game_object.name} at distance {hit.distance}")
"""

from __future__ import annotations

from typing import Optional, List

from infernux.math.coerce import coerce_quat, coerce_vec3
from infernux.lib import Physics as _CppPhysics
from infernux.lib import _wrap_native_builtin_component


def _query_layer_mask(value: Optional[int]) -> Optional[int]:
    return None if value is None else int(value)


def _gpu_state_output(out, names, *, flush: bool = True):
    from infernux.compute import Buffer, _flush_commands

    values = [out.get(name) for name in names]
    if not any(isinstance(value, Buffer) and value.device == "gpu" for value in values):
        return None
    if not all(isinstance(value, Buffer) and value.device == "gpu" for value in values):
        raise TypeError("GPU physics state outputs must all be resident GPU inx.buffer values")
    for value in values:
        value._require_open()
    host = values[0]._host
    if any(int(value._host.identity) != int(host.identity) for value in values[1:]):
        raise ValueError("GPU physics state outputs must use the same compute host")
    if flush:
        _flush_commands(host)
    return {name: value._native for name, value in zip(names, values)}


class _PhysicsMeta(type):
    @property
    def body_count(cls) -> int:
        """Number of native physics bodies currently owned by the world."""
        return int(_CppPhysics.body_count)

    @property
    def query_generation(cls) -> int:
        """Monotonic token for the currently published physics query world."""
        return int(_CppPhysics.query_generation)

    @property
    def gravity(cls):
        return _CppPhysics.get_gravity()

    @gravity.setter
    def gravity(cls, value):
        _CppPhysics.set_gravity(coerce_vec3(value))


class Physics(metaclass=_PhysicsMeta):
    """Static physics query interface (mirrors Unity's Physics class).

    All methods delegate to the C++ ``PhysicsWorld`` singleton via pybind11.
    Omitted or ``None`` query masks use ``EngineConfig.default_query_layer_mask``
    at the query boundary. Explicit masks are literal: zero selects no layers,
    and ``0xFFFFFFFF`` selects all 32 layers.
    """

    @staticmethod
    def apply_result(request, targets, apply, *, required=False):
        """Apply a managed result once at a physics boundary.

        ``apply(resolved_targets, result)`` receives the current component wrappers.
        Pending work leaves physics advancing by default. Set ``required=True``
        when the next integration depends on this result.
        A Rigidbody binding starts blocking at the first eligible fixed step
        with a published body. It retires if the Rigidbody is disabled, its
        actor or published shape changes, or its motion state is reset.
        The returned consumer exposes status/reason and can be explicitly cancelled.
        """
        from infernux.application import Application
        engine = Application._current_engine()
        if engine is None:
            raise RuntimeError('physics result application requires an active Engine')
        return engine._apply_result(request, targets, apply, boundary='physics', required=required)

    @staticmethod
    def sync_transforms():
        """Publish pending Transform edits to physics before same-frame queries."""
        _CppPhysics.sync_transforms()

    @staticmethod
    def get_rigidbody_states(rigidbodies, out=None, *, fields=None):
        """Read current solver state for a sequence of Rigidbody components.

        Returns owned, C-contiguous float32 NumPy arrays in input order by
        default. ``out`` may provide reusable CPU/GPU ``inx.buffer`` values or
        matching NumPy arrays under the returned keys. A complete GPU output
        set is filled directly by C++ in one asynchronous transfer submission:
        position, center_of_mass, linear_velocity, angular_velocity and
        inverse_mass have shape (N, 3); rotation is (N, 4), xyzw;
        inverse_inertia is (N, 3, 3), row-major and world-space about the COM.
        Reusable outputs may have a first-axis capacity greater than N; only
        the first N rows are written, so broad-phase candidate-count changes
        do not require new arrays.
        Field overlap checks cover only these written prefixes. Capacity tails
        are untouched and may share an arena with another field's written rows.
        ``fields`` selects CPU fields instead of allocating/copying all seven.
        Names follow the keys above; ``body_id`` adds uint32 (N,) Jolt index+
        sequence, and ``shape_revision`` adds uint64 (N,) shared actor shape
        identity. Selected CPU outputs contain exactly those keys. Selection
        skips unrequested solver values, including inertia calculations.

        inverse_mass contains the three diagonal entries after translation
        locks. Kinematic/static bodies have zero inverse mass and inertia.

        Call from the engine thread at the intended fixed-step phase. This
        reads physical poses, not display interpolation, and does not step or
        synchronize authored transforms. A body without a resident Collider
        raises instead of returning synthetic state. Arrays are snapshots;
        modifying them does not modify the physics world.
        """
        native_bodies = [body._require_cpp_component() for body in rigidbodies]
        if out is None:
            return _CppPhysics.get_rigidbody_states(native_bodies, fields=fields)
        if not isinstance(out, dict):
            raise TypeError("Rigidbody state output must be a dictionary")
        from infernux.compute import Buffer
        names = ("position", "center_of_mass", "linear_velocity", "angular_velocity",
                 "inverse_mass", "rotation", "inverse_inertia")
        if fields is None:
            gpu_output = _gpu_state_output(out, names)
            if gpu_output is not None:
                _CppPhysics._write_rigidbody_state_buffers(native_bodies, gpu_output)
                return out
        elif any(isinstance(value, Buffer) and value.device != "cpu" for value in out.values()):
            raise TypeError("Selected Rigidbody state fields require CPU outputs")
        native_output = {
            name: value.numpy(copy=False) if isinstance(value, Buffer) else value
            for name, value in out.items()
        }
        _CppPhysics.get_rigidbody_states(native_bodies, native_output, fields=fields)
        return out

    @staticmethod
    def reset_rigidbodies(rigidbodies, state):
        """Reset live bodies between fixed steps, outside collision callbacks.

        ``state`` has exactly position[N,3], rotation[N,4] (xyzw),
        linear_velocity[N,3] and angular_velocity[N,3], contiguous float32 CPU
        arrays or CPU Buffers. Read them with ``get_rigidbody_states(fields=...)``
        or author them directly. Input is validated before any body changes.

        Keeps body/shape/joint identities and collision rules; clears forces,
        movement targets, joint warm starts and contacts for these bodies.
        Publishes physical/display poses together without advancing time.
        Pending ``apply_result`` bindings to these bodies retire. The caller
        controls the lifetime of the producing request and its policy state.
        """
        from infernux.compute import Buffer
        arrays = {}
        for name, value in state.items():
            if isinstance(value, Buffer):
                if value.device != 'cpu':
                    raise TypeError('body reset requires CPU state; transfer GPU results explicitly')
                value = value.numpy(copy=False)
            arrays[name] = value
        _CppPhysics.reset_rigidbodies([body._require_cpp_component() for body in rigidbodies], arrays)

    @staticmethod
    def get_rigidbody_states_and_box_states(rigidbodies, state_out, box_out, *, query_triggers: bool = False):
        """Upload solver state and BoxCollider descriptors in one GPU submission.

        This is the combined hot-path contract for GPU physics consumers.  The
        two output dictionaries use the same fields as
        :meth:`get_rigidbody_states` and :meth:`get_rigidbody_box_states`; all
        fields must be resident ``inx.buffer(device="gpu")`` values on the same
        host.  The call does not step physics or perform a Python per-body
        rewalk.  It is useful when a solver needs authoritative rigidbody
        motion and shape/material data in the same fixed-step snapshot.
        """
        if not isinstance(state_out, dict) or not isinstance(box_out, dict):
            raise TypeError("Rigidbody and BoxCollider state outputs must be dictionaries")
        state_names = ("position", "center_of_mass", "linear_velocity", "angular_velocity",
                       "inverse_mass", "rotation", "inverse_inertia")
        box_names = ("body_index", "center", "rotation", "half_extents", "friction", "bounciness")
        state_gpu = _gpu_state_output(state_out, state_names)
        box_gpu = _gpu_state_output(box_out, box_names, flush=False)
        if state_gpu is None or box_gpu is None:
            raise TypeError("Combined rigidbody state upload requires resident GPU inx.buffer outputs")
        state_host = next(iter(state_out[name]._host for name in state_names))
        box_host = next(iter(box_out[name]._host for name in box_names))
        if int(state_host.identity) != int(box_host.identity):
            raise ValueError("Combined rigidbody state outputs must use the same compute host")
        native_bodies = [body._require_cpp_component() for body in rigidbodies]
        box_count = _CppPhysics._write_rigidbody_state_and_box_state_buffers(
            native_bodies, state_gpu, box_gpu, bool(query_triggers)
        )
        box_out["count"] = int(box_count)
        return state_out, box_out

    @staticmethod
    def query_rigidbodies_in_bounds(minimum, maximum,
                                    layer_mask: Optional[int] = None,
                                    query_triggers: bool = False):
        """Return Rigidbody broad-phase candidates inside a world AABB.

        This query traverses Jolt's broad-phase tree, so distant bodies do not
        add Python work. Results are candidates whose body bounds overlap the
        box; custom solvers must still perform their exact shape/contact test.
        A compound body appears once. Static collider-only objects are omitted.
        """
        from infernux.components.builtin import Rigidbody

        native = _CppPhysics.query_rigidbodies_in_bounds(
            coerce_vec3(minimum), coerce_vec3(maximum), _query_layer_mask(layer_mask), bool(query_triggers)
        )
        return [Rigidbody._get_or_create_wrapper(body, body.game_object) for body in native]

    @staticmethod
    def get_rigidbody_box_states(rigidbodies, out=None, *, query_triggers: bool = False):
        """Batch world-space BoxCollider descriptors for candidate rigidbodies.

        The returned structure is SoA data suitable for a compute kernel:
        ``body_index`` maps each box to the input body; ``center``, ``rotation``
        (xyzw), and ``half_extents`` match the shape submitted to Jolt;
        ``friction`` and ``bounciness`` carry material values. ``count`` is the
        valid prefix length. An ``out`` dictionary may use larger reusable CPU
        or GPU buffers/NumPy arrays, avoiding allocation when the candidate
        count moves. A complete GPU output set is uploaded directly from C++ in
        one asynchronous transfer submission.

        This deliberately describes boxes only. Other collider families receive
        their own typed batches instead of an untyped union with guessed fields.
        Disabled boxes and, by default, triggers are omitted.
        """
        from infernux.compute import Buffer

        native_bodies = [body._require_cpp_component() for body in rigidbodies]
        if out is None:
            return _CppPhysics.get_rigidbody_box_states(
                native_bodies, query_triggers=bool(query_triggers)
            )
        if not isinstance(out, dict):
            raise TypeError("Box collider state output must be a dictionary")
        names = ("body_index", "center", "rotation", "half_extents", "friction", "bounciness")
        gpu_output = _gpu_state_output(out, names)
        if gpu_output is not None:
            out["count"] = int(_CppPhysics._write_rigidbody_box_state_buffers(
                native_bodies, gpu_output, bool(query_triggers)
            ))
            return out
        native_output = {
            name: value.numpy(copy=False) if isinstance(value, Buffer) else value
            for name, value in out.items() if name != "count"
        }
        result = _CppPhysics.get_rigidbody_box_states(
            native_bodies, native_output, bool(query_triggers)
        )
        out["count"] = int(result["count"])
        return out

    @staticmethod
    def query_rigidbody_box_states_in_bounds(minimum, maximum, out, layer_mask: Optional[int] = None,
                                             *, query_triggers: bool = False):
        """Query nearby rigidbodies and write BoxCollider descriptors in one native pass.

        This is the resident-buffer path for custom GPU solvers.  Jolt's broad
        phase, collider filtering, transform projection, and SoA upload happen
        inside one engine call; Python receives only the candidate wrappers and
        the valid prefix count.  ``out`` must contain the six reusable
        ``inx.buffer`` fields accepted by :meth:`get_rigidbody_box_states`.
        """
        from infernux.compute import Buffer
        if not isinstance(out, dict):
            raise TypeError("Box collider state output must be a dictionary")
        names = ("body_index", "center", "rotation", "half_extents", "friction", "bounciness")
        gpu_output = _gpu_state_output(out, names)
        if gpu_output is None:
            raise TypeError("query_rigidbody_box_states_in_bounds requires inx.buffer outputs")
        native_bodies, count = _CppPhysics._query_rigidbody_box_state_buffers(
            coerce_vec3(minimum), coerce_vec3(maximum), _query_layer_mask(layer_mask), bool(query_triggers), gpu_output
        )
        out["count"] = int(count)
        from infernux.components.builtin import Rigidbody
        out["rigidbodies"] = [Rigidbody._get_or_create_wrapper(body, body.game_object) for body in native_bodies]
        return out

    @staticmethod
    def query_rigidbody_states_and_box_states_in_bounds(
        minimum, maximum, state_out, box_out,
        layer_mask: Optional[int] = None, *, query_triggers: bool = False,
    ):
        """Query Jolt broad-phase candidates and upload both GPU state sets.

        This is the single-call fixed-step path for custom GPU physics.  The
        broad-phase query, authoritative solver snapshot, flattened BoxCollider
        projection, and one compute transfer submission all remain inside the
        engine; Python receives only the candidate wrappers and valid Box
        prefix after the native call.
        """
        if not isinstance(state_out, dict) or not isinstance(box_out, dict):
            raise TypeError("Rigidbody and BoxCollider state outputs must be dictionaries")
        state_names = ("position", "center_of_mass", "linear_velocity", "angular_velocity",
                       "inverse_mass", "rotation", "inverse_inertia")
        box_names = ("body_index", "center", "rotation", "half_extents", "friction", "bounciness")
        state_gpu = _gpu_state_output(state_out, state_names)
        box_gpu = _gpu_state_output(box_out, box_names, flush=False)
        if state_gpu is None or box_gpu is None:
            raise TypeError("Combined rigidbody query requires resident GPU inx.buffer outputs")
        state_host = state_out[state_names[0]]._host
        box_host = box_out[box_names[0]]._host
        if int(state_host.identity) != int(box_host.identity):
            raise ValueError("Combined rigidbody query outputs must use the same compute host")
        native_bodies, count = _CppPhysics._query_rigidbody_state_and_box_state_buffers(
            coerce_vec3(minimum), coerce_vec3(maximum), _query_layer_mask(layer_mask), bool(query_triggers),
            state_gpu, box_gpu,
        )
        box_out["count"] = int(count)
        from infernux.components.builtin import Rigidbody
        box_out["rigidbodies"] = [Rigidbody._get_or_create_wrapper(body, body.game_object) for body in native_bodies]
        return state_out, box_out

    @staticmethod
    def apply_rigidbody_impulses(rigidbodies, linear_impulses, angular_impulses, *,
                                  expected_body_ids=None, expected_shape_revisions=None):
        """Submit aggregated feedback using CPU arrays or resident GPU buffers.

        Rows follow the input body order. Linear impulse is in world-space
        kg*m/s; angular impulse is about the body's COM in kg*m^2/s. For a
        contact impulse J at point p, aggregate J and cross(p - COM, J).
        CPU arrays must be C-contiguous. Two GPU ``inx.buffer`` values are read
        together as float32 vector3 valid prefixes in the pending compute
        submission, with one ordered queue wait. Input validation finishes
        before mutation. Optional ``expected_body_ids`` must be exact uint32
        (N,) snapshots; ``expected_shape_revisions`` is uint64 (N,) and requires
        body IDs. All identities are checked before any impulse is applied.
        Callers retaining components must also resolve their original strict
        ObjectHandles and simulation epoch before submitting on the owner.

        Uses the existing Impulse path: feedback is immediate, does not step
        physics, and is not multiplied by delta time. Submit on the engine
        thread at the intended fixed-step boundary. Jolt applies motion locks
        and velocity limits; static/kinematic bodies do not gain velocity.
        Zero feedback does not wake sleeping bodies.
        """
        import numpy as np

        from infernux.compute import Buffer
        native_bodies = [body._require_cpp_component() for body in rigidbodies]
        buffers = isinstance(linear_impulses, Buffer), isinstance(angular_impulses, Buffer)
        if any(buffers):
            if not all(buffers) or linear_impulses.device != angular_impulses.device:
                raise TypeError("Rigidbody impulse inputs must use the same CPU or GPU storage kind")
            count = len(native_bodies)
            for value in (linear_impulses, angular_impulses):
                value._require_open()
                if value.dtype != "vector3" or value.element_count < count:
                    raise ValueError(
                        "Rigidbody impulse buffers must be vector3 with body-count capacity"
                    )
            if count == 0:
                # Keep exact identity validation, without a zero-byte GPU read.
                linear_impulses = angular_impulses = np.empty((0, 3), np.float32)
            elif linear_impulses.device == "gpu":
                from infernux.compute import _submit_and_read

                if int(linear_impulses._host.identity) != int(angular_impulses._host.identity):
                    raise ValueError("Rigidbody impulse GPU buffers must use the same compute host")
                byte_size = count * 3 * np.dtype(np.float32).itemsize
                linear_bytes, angular_bytes = _submit_and_read(
                    linear_impulses._host,
                    (
                        (linear_impulses, linear_impulses._byte_offset, byte_size),
                        (angular_impulses, angular_impulses._byte_offset, byte_size),
                    ),
                )
                linear_impulses = np.frombuffer(linear_bytes, dtype=np.float32).reshape(count, 3)
                angular_impulses = np.frombuffer(angular_bytes, dtype=np.float32).reshape(count, 3)
            else:
                linear_impulses = linear_impulses.numpy(copy=False)[:count]
                angular_impulses = angular_impulses.numpy(copy=False)[:count]
        _CppPhysics.apply_rigidbody_impulses(
            native_bodies, linear_impulses, angular_impulses,
            expected_body_ids=expected_body_ids, expected_shape_revisions=expected_shape_revisions,
        )

    @staticmethod
    def set_contact_event_stream_enabled(enabled: bool = True, *, include_triggers: bool = False):
        """Enable the resolved fixed-step contact stream for custom solvers.

        When enabled, :meth:`get_contact_events` exposes contact points,
        normals, relative velocities, body IDs and sub-shape IDs from the
        most recent completed physics step. The stream is opt-in so ordinary
        projects pay no contact-event buffering cost. Call from setup before
        the first fixed step; the event arrays are replaced on each step.
        """
        _CppPhysics.set_contact_event_stream_enabled(bool(enabled), bool(include_triggers))

    @staticmethod
    def get_contact_events():
        """Return resolved contacts from the most recent completed fixed step.

        The result is a NumPy structure-of-arrays dictionary. ``type`` uses
        the native event order: collision enter/stay/exit (0/1/2), then
        trigger enter/stay/exit (3/4/5). This is a snapshot and contains no
        solver impulse; custom solvers should aggregate their own feedback
        and submit it through :meth:`apply_rigidbody_impulses`.
        """
        return _CppPhysics.get_contact_events()

    @staticmethod
    def set_contact_impulse_stream_enabled(enabled: bool = True):
        """Enable the actual Jolt velocity-solver impulse snapshot.

        Each solver midpoint reports its normal impulse and an equal share
        of the manifold's tangential friction impulse. These shares preserve
        the net linear impulse and its angular impulse, but are not separately
        solved pointwise friction forces. Independent torsional friction is
        not included. This snapshot is opt-in and separate from callbacks.
        """
        _CppPhysics.set_contact_impulse_stream_enabled(bool(enabled))

    @staticmethod
    def get_contact_impulses():
        """Return actual solved contact impulses from the latest fixed step.

        The result is a NumPy structure-of-arrays dictionary containing body
        and sub-shape IDs, solver world midpoints/normals, and the impulse on
        body B (its negative acts on A). Positions are frozen when velocity
        constraints are built, before integration and position correction.
        The snapshot is replaced by the next fixed step.
        """
        return _CppPhysics.get_contact_impulses()

    # ------------------------------------------------------------------
    # Raycast
    # ------------------------------------------------------------------

    @staticmethod
    def compute_penetration(collider_a, position_a, rotation_a, collider_b, position_b, rotation_b):
        """Query collider geometry at predicted world origins and rotations.

        Returns None for no penetration; otherwise direction/distance move A
        out of B, and point_a/point_b are the two world-space surface points.
        Poses refer to GameObject origins. Primitive colliders use their authored
        center, scale and axis. Mesh colliders use their published convex shape
        and center, including while a replacement shape is being prepared.

        Supports Box, Sphere, Capsule, Cylinder and a published convex MeshCollider.
        Non-convex or unpublished meshes raise explicitly. No Transform
        changes, body registration, broadphase synchronization, layer/trigger
        filtering or physics stepping.
        Disabled primitive colliders can be queried; only the specified member
        of a compound is considered.
        """
        return _CppPhysics.compute_penetration(
            collider_a._require_cpp_component(), coerce_vec3(position_a), coerce_quat(rotation_a),
            collider_b._require_cpp_component(), coerce_vec3(position_b), coerce_quat(rotation_b),
        )

    @staticmethod
    def raycast(origin, direction, max_distance: float = 1000.0, layer_mask: Optional[int] = None,
                query_triggers: bool = True):
        """Cast a ray and return the closest RaycastHit, or None.

        Args:
            origin: Ray origin as ``Vector3`` or ``(x, y, z)`` tuple.
            direction: Ray direction as ``Vector3`` or ``(x, y, z)`` tuple.
            max_distance: Maximum ray distance (default 1000).
            layer_mask: 32-bit layer mask used to filter hits.
            query_triggers: Whether trigger colliders should be returned.

        Returns:
            A ``RaycastHit`` object with ``point``, ``normal``, ``distance``,
            ``game_object``, and ``collider`` attributes — or ``None``.
        """
        o = coerce_vec3(origin)
        d = coerce_vec3(direction)
        return _CppPhysics.raycast(o, d, max_distance, _query_layer_mask(layer_mask), bool(query_triggers))

    @staticmethod
    def raycast_screen(camera, screen_position, viewport_size, max_distance: float = 1000.0,
                       layer_mask: Optional[int] = None, query_triggers: bool = True):
        """Raycast from a camera viewport pixel position.

        ``screen_position`` and ``viewport_size`` use top-left-origin pixels,
        matching ``Camera.screen_point_to_ray`` and the Game View input API.
        The conversion is kept here so world-object hover, picking and gameplay
        queries share one coordinate contract instead of each reimplementing it.
        """
        if camera is None:
            raise ValueError("camera is required")
        try:
            x, y = screen_position
            width, height = viewport_size
        except (TypeError, ValueError) as exc:
            raise TypeError("screen_position and viewport_size must be 2-item sequences") from exc
        ray = camera.screen_point_to_ray(float(x), float(y), float(width), float(height))
        if ray is None:
            return None
        origin, direction = ray
        return Physics.raycast(origin, direction, max_distance, layer_mask, query_triggers)

    @staticmethod
    def raycast_batch(origins, directions, out, max_distance: float = 1000.0,
                      layer_mask: Optional[int] = None, query_triggers: bool = True,
                      profile: bool = False):
        """Cast many rays into reusable NumPy result storage.

        ``origins`` and ``directions`` are C-contiguous float32 ``(N, 3)``
        arrays. ``out`` must provide writable arrays named ``hit`` (uint8),
        ``point``/``normal`` (float32 ``(capacity, 3)``), ``distance``
        (float32), ``body_id``/``sub_shape_id``/``triangle_index`` (uint32),
        and ``collider_id``/``game_object_id`` (uint64). Capacity may exceed N;
        only the first N rows are written. ``triangle_index`` is UINT32_MAX
        unless the ray hit a non-convex MeshCollider triangle.

        Call on the engine owner thread. Native execution publishes pending
        authored collider state and runs the whole batch on the shared GPU
        compute pipeline against one retained publication. No per-ray Python
        calls or CPU collision queries run. The same dictionary and arrays are
        returned unchanged. Misses use
        ``hit=0``, infinite distance and zero object/component identities.
        ``Physics.query_generation`` exposes the monotonic published-world
        token for associating retained results with the snapshot they read;
        the returned dictionary also contains that scalar under
        ``query_generation``.

        ``profile=True`` adds the native profile dictionary. Fields without a
        GPU measurement remain zero; they must not be interpreted as summed
        CPU worker timings.
        """
        return _CppPhysics.raycast_batch(
            origins, directions, out, float(max_distance), _query_layer_mask(layer_mask), bool(query_triggers), bool(profile)
        )

    @staticmethod
    def raycast_all(origin, direction, max_distance: float = 1000.0, layer_mask: Optional[int] = None,
                    query_triggers: bool = True):
        """Cast a ray and return all hits.

        Returns:
            A list of ``RaycastHit`` objects.
        """
        o = coerce_vec3(origin)
        d = coerce_vec3(direction)
        return _CppPhysics.raycast_all(o, d, max_distance, _query_layer_mask(layer_mask), bool(query_triggers))

    # ------------------------------------------------------------------
    # Overlap queries
    # ------------------------------------------------------------------

    @staticmethod
    def overlap_sphere(center, radius: float, layer_mask: Optional[int] = None,
                       query_triggers: bool = True):
        """Find all colliders within a sphere.

        Args:
            center: World-space center of the sphere.
            radius: Radius of the sphere.
            layer_mask: 32-bit layer mask.
            query_triggers: Whether to include trigger colliders.

        Returns:
            A list of ``Collider`` objects overlapping the sphere.
        """
        c = coerce_vec3(center)
        return [_wrap_native_builtin_component(collider) for collider in
                _CppPhysics.overlap_sphere(c, float(radius), _query_layer_mask(layer_mask), bool(query_triggers))]

    @staticmethod
    def overlap_box(center, half_extents, orientation=None, layer_mask: Optional[int] = None,
                    query_triggers: bool = True):
        """Find all colliders within an oriented box.

        Args:
            center: World-space center of the box.
            half_extents: Half-extents ``(hx, hy, hz)`` of the box.
            layer_mask: 32-bit layer mask.
            query_triggers: Whether to include trigger colliders.

        Returns:
            A list of ``Collider`` objects overlapping the box.
        """
        c = coerce_vec3(center)
        he = coerce_vec3(half_extents)
        return [_wrap_native_builtin_component(collider) for collider in
                _CppPhysics.overlap_box(c, he, coerce_quat(orientation), _query_layer_mask(layer_mask), bool(query_triggers))]

    @staticmethod
    def overlap_capsule(point0, point1, radius: float, layer_mask: Optional[int] = None,
                        query_triggers: bool = True):
        """Find all colliders within a capsule defined by two segment endpoints."""
        return [_wrap_native_builtin_component(collider) for collider in _CppPhysics.overlap_capsule(
            coerce_vec3(point0), coerce_vec3(point1), float(radius), _query_layer_mask(layer_mask), bool(query_triggers)
        )]

    # ------------------------------------------------------------------
    # Shape casts
    # ------------------------------------------------------------------

    @staticmethod
    def sphere_cast(origin, radius: float, direction, max_distance: float = 1000.0,
                    layer_mask: Optional[int] = None, query_triggers: bool = True):
        """Cast a sphere along a direction and return closest RaycastHit, or None.

        Args:
            origin: Start center of the sphere.
            radius: Radius of the sphere.
            direction: Direction to cast.
            max_distance: Maximum cast distance.

        Returns:
            A ``RaycastHit`` or ``None``.
        """
        o = coerce_vec3(origin)
        d = coerce_vec3(direction)
        return _CppPhysics.sphere_cast(o, float(radius), d, max_distance, _query_layer_mask(layer_mask), bool(query_triggers))

    @staticmethod
    def box_cast(center, half_extents, direction, orientation=None, max_distance: float = 1000.0,
                 layer_mask: Optional[int] = None, query_triggers: bool = True):
        """Cast a box along a direction and return closest RaycastHit, or None.

        Args:
            center: Start center of the box.
            half_extents: Half-extents ``(hx, hy, hz)`` of the box.
            direction: Direction to cast.
            max_distance: Maximum cast distance.

        Returns:
            A ``RaycastHit`` or ``None``.
        """
        c = coerce_vec3(center)
        he = coerce_vec3(half_extents)
        d = coerce_vec3(direction)
        return _CppPhysics.box_cast(
            c, he, d, coerce_quat(orientation), max_distance, _query_layer_mask(layer_mask), bool(query_triggers)
        )

    @staticmethod
    def sweep_box(collider, position, rotation, displacement):
        """Sweep a BoxCollider from a world pose and return the first blocking hit.

        The query uses the collider's current size, scale and collision filtering.
        It ignores the owning body and contacts the box can move away from.
        """
        return _CppPhysics.sweep_box(
            collider._require_cpp_component(), coerce_vec3(position),
            coerce_quat(rotation), coerce_vec3(displacement),
        )

    @staticmethod
    def capsule_cast(point0, point1, radius: float, direction, max_distance: float = 1000.0,
                     layer_mask: Optional[int] = None, query_triggers: bool = True):
        """Cast a capsule and return the closest RaycastHit, or None."""
        return _CppPhysics.capsule_cast(
            coerce_vec3(point0), coerce_vec3(point1), float(radius), coerce_vec3(direction), max_distance,
            _query_layer_mask(layer_mask), bool(query_triggers)
        )

    # ------------------------------------------------------------------
    # Layer collision control
    # ------------------------------------------------------------------

    @staticmethod
    def ignore_layer_collision(layer1: int, layer2: int, ignore: bool = True):
        """Set whether two layers should ignore collisions with each other.

        Args:
            layer1: First layer index (0-31).
            layer2: Second layer index (0-31).
            ignore: If True, disable collisions; if False, enable them.
        """
        _CppPhysics.ignore_layer_collision(int(layer1), int(layer2), bool(ignore))

    @staticmethod
    def get_ignore_layer_collision(layer1: int, layer2: int) -> bool:
        """Check if two layers are set to ignore collisions.

        Returns:
            True if the layers ignore collisions with each other.
        """
        return _CppPhysics.get_ignore_layer_collision(int(layer1), int(layer2))

    @staticmethod
    def ignore_collision(collider1, collider2, ignore: bool = True):
        """Enable or disable contacts for one exact Collider pair."""
        first = collider1._require_cpp_component() if hasattr(collider1, "_require_cpp_component") else collider1
        second = collider2._require_cpp_component() if hasattr(collider2, "_require_cpp_component") else collider2
        _CppPhysics.ignore_collision(first, second, bool(ignore))

    @staticmethod
    def get_ignore_collision(collider1, collider2) -> bool:
        """Return whether one exact Collider pair currently ignores contacts."""
        first = collider1._require_cpp_component() if hasattr(collider1, "_require_cpp_component") else collider1
        second = collider2._require_cpp_component() if hasattr(collider2, "_require_cpp_component") else collider2
        return bool(_CppPhysics.get_ignore_collision(first, second))


from .joint_batch import HingeJointBatch

__all__ = ["Physics", "HingeJointBatch"]
