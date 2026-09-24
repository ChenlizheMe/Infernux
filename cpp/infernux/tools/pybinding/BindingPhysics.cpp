/**
 * @file BindingPhysics.cpp
 * @brief Python bindings for physics collider components, PhysicsWorld, and raycast API.
 *
 * Registers BoxCollider, SphereCollider, CapsuleCollider, CylinderCollider,
 * RaycastHit, and
 * a "Physics" static
 * class with Raycast/RaycastAll methods.
 */

// Jolt types are no longer exposed in collider headers — no Jolt include needed here
// Except for gravity API which accesses PhysicsSystem directly.
#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include "function/renderer/rhi/RhiComputeBuffer.h"
#include "function/renderer/rhi/RhiComputeKernel.h"
#include "function/scene/BoxCollider.h"
#include "function/scene/CapsuleCollider.h"
#include "function/scene/Collider.h"
#include "function/scene/Component.h"
#include "function/scene/CylinderCollider.h"
#include "function/scene/GameObject.h"
#include "function/scene/HingeJoint.h"
#include "function/scene/MeshCollider.h"
#include "function/scene/Rigidbody.h"
#include "function/scene/SceneManager.h"
#include "function/scene/SliderJoint.h"
#include "function/scene/SphereCollider.h"
#include "function/scene/TagLayerManager.h"
#include "function/scene/physics/PhysicsContactListener.h"
#include "function/scene/physics/PhysicsWorld.h"
#include <core/config/EngineConfig.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace infernux
{

// Forward-declare the registry (defined in BindingScene.cpp)
class ComponentBindingRegistry;

namespace
{
void RequireFinite(const glm::vec3 &value, const char *name)
{
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
        throw std::invalid_argument(std::string(name) + " must contain finite values");
}

py::array GetPointVelocities(const Rigidbody &body, py::array points, py::array output)
{
    for (const auto &array : {points, output}) {
        if (!array.dtype().is(py::dtype::of<float>()))
            throw py::type_error("Point velocities require float32 NumPy arrays");
        if (array.ndim() != 2 || array.shape(1) != 3 || !(array.flags() & py::array::c_style))
            throw py::value_error("Point velocities require C-contiguous arrays of shape (N, 3)");
    }
    if (points.shape(0) != output.shape(0) || !output.writeable())
        throw py::value_error("Output must be writable and have the same shape as points");

    const auto inputStart = reinterpret_cast<std::uintptr_t>(points.data());
    const auto outputStart = reinterpret_cast<std::uintptr_t>(output.data());
    const auto bytes = static_cast<std::uintptr_t>(points.nbytes());
    if (inputStart != outputStart && inputStart < outputStart + bytes && outputStart < inputStart + bytes)
        throw py::value_error("Points and output may be identical but must not partially overlap");

    const auto *input = static_cast<const float *>(points.data());
    // Validate the public array boundary before writing caller-owned storage.
    for (py::ssize_t i = 0; i < points.size(); ++i)
        if (!std::isfinite(input[i]))
            throw py::value_error("Points must contain finite values");

    const auto linear = body.GetVelocity();
    const auto angular = body.GetAngularVelocity();
    const auto center = body.GetWorldCenterOfMass();
    auto *result = static_cast<float *>(output.mutable_data());
    for (py::ssize_t i = 0; i < points.size(); i += 3) {
        const glm::vec3 point(input[i], input[i + 1], input[i + 2]);
        const auto velocity = linear + glm::cross(angular, point - center);
        result[i] = velocity.x;
        result[i + 1] = velocity.y;
        result[i + 2] = velocity.z;
    }
    return output;
}

void RequireDirectionAndDistance(const glm::vec3 &direction, float maxDistance)
{
    RequireFinite(direction, "direction");
    if (glm::dot(direction, direction) <= 1e-12f)
        throw std::invalid_argument("direction must be non-zero");
    if (!std::isfinite(maxDistance) || maxDistance <= 0.0f)
        throw std::invalid_argument("max_distance must be finite and greater than zero");
}

void RequireRaycastBatchInput(const py::array &value, const char *name)
{
    if (!value.dtype().is(py::dtype::of<float>()))
        throw py::type_error(std::string(name) + " must be a float32 NumPy array");
    if (value.ndim() != 2 || value.shape(1) != 3 || !(value.flags() & py::array::c_style))
        throw py::value_error(std::string(name) + " must be a C-contiguous array of shape (N, 3)");
}

template <typename T>
py::array_t<T> RaycastBatchOutput(py::dict &output, const char *name, py::ssize_t count, py::ssize_t lanes = 0)
{
    const auto key = py::str(name);
    if (!output.contains(key))
        throw py::value_error(std::string("Raycast batch output is missing '") + name + "'");
    py::array value = py::cast<py::array>(output[key]);
    const bool correctShape = lanes == 0 ? value.ndim() == 1 : value.ndim() == 2 && value.shape(1) == lanes;
    if (!value.dtype().is(py::dtype::of<T>()) || !(value.flags() & py::array::c_style) || !value.writeable() ||
        !correctShape)
        throw py::value_error(std::string("Raycast batch output '") + name + "' has the wrong writable layout");
    if (value.shape(0) < count)
        throw py::value_error(std::string("Raycast batch output '") + name + "' has insufficient capacity");
    return py::reinterpret_borrow<py::array_t<T>>(value);
}

py::dict RaycastBatch(py::array origins, py::array directions, py::dict output, float maxDistance, uint32_t layerMask,
                      bool queryTriggers, bool profileEnabled)
{
    const auto profileNowNs = []() noexcept {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    };
    const uint64_t validationStartNs = profileEnabled ? profileNowNs() : 0;
    RequireRaycastBatchInput(origins, "origins");
    RequireRaycastBatchInput(directions, "directions");
    if (origins.shape(0) != directions.shape(0))
        throw py::value_error("origins and directions must have the same row count");
    if (!std::isfinite(maxDistance) || maxDistance <= 0.0f)
        throw py::value_error("max_distance must be finite and greater than zero");

    const py::ssize_t count = origins.shape(0);
    const auto *originData = static_cast<const float *>(origins.data());
    const auto *directionData = static_cast<const float *>(directions.data());
    static thread_local std::vector<float> normalizedDirections;
    bool directionsAlreadyNormalized = true;
    for (py::ssize_t index = 0; index < count; ++index) {
        const py::ssize_t offset = index * 3;
        const glm::vec3 origin(originData[offset], originData[offset + 1], originData[offset + 2]);
        const glm::vec3 direction(directionData[offset], directionData[offset + 1], directionData[offset + 2]);
        RequireFinite(origin, "origins");
        RequireFinite(direction, "direction");
        const double lengthSq = static_cast<double>(direction.x) * direction.x +
                                static_cast<double>(direction.y) * direction.y +
                                static_cast<double>(direction.z) * direction.z;
        if (!std::isfinite(lengthSq) || lengthSq <= 1e-12)
            throw std::invalid_argument("direction must be non-zero");
        directionsAlreadyNormalized = directionsAlreadyNormalized && lengthSq == 1.0;
    }
    const float *normalizedDirectionData = directionData;
    if (!directionsAlreadyNormalized) {
        normalizedDirections.resize(static_cast<size_t>(count) * 3u);
        for (py::ssize_t index = 0; index < count; ++index) {
            const py::ssize_t offset = index * 3;
            const glm::vec3 direction(directionData[offset], directionData[offset + 1], directionData[offset + 2]);
            const double lengthSq = static_cast<double>(direction.x) * direction.x +
                                    static_cast<double>(direction.y) * direction.y +
                                    static_cast<double>(direction.z) * direction.z;
            const double inverseLength = 1.0 / std::sqrt(lengthSq);
            normalizedDirections[static_cast<size_t>(offset)] = direction.x * inverseLength;
            normalizedDirections[static_cast<size_t>(offset + 1)] = direction.y * inverseLength;
            normalizedDirections[static_cast<size_t>(offset + 2)] = direction.z * inverseLength;
        }
        normalizedDirectionData = normalizedDirections.data();
    }

    auto hit = RaycastBatchOutput<uint8_t>(output, "hit", count);
    auto point = RaycastBatchOutput<float>(output, "point", count, 3);
    auto normal = RaycastBatchOutput<float>(output, "normal", count, 3);
    auto distance = RaycastBatchOutput<float>(output, "distance", count);
    auto bodyId = RaycastBatchOutput<uint32_t>(output, "body_id", count);
    auto subShapeId = RaycastBatchOutput<uint32_t>(output, "sub_shape_id", count);
    auto triangleIndex = RaycastBatchOutput<uint32_t>(output, "triangle_index", count);
    auto colliderId = RaycastBatchOutput<uint64_t>(output, "collider_id", count);
    auto gameObjectId = RaycastBatchOutput<uint64_t>(output, "game_object_id", count);
    const uint64_t validationEndNs = profileEnabled ? profileNowNs() : 0;

    static thread_local std::vector<RaycastHit> nativeHits;
    static thread_local std::vector<uint8_t> nativeMask;
    nativeHits.resize(static_cast<size_t>(count));
    nativeMask.resize(static_cast<size_t>(count));
    uint64_t queryGeneration = 0;
    RaycastBatchProfile nativeProfile;
    PhysicsWorld &physics = PhysicsWorld::Instance();
    const uint64_t syncStartNs = profileEnabled ? profileNowNs() : 0;
    const bool synchronizedOnOwner = physics.PrepareRaycastBatchQuery();
    const uint64_t syncNs = profileEnabled && synchronizedOnOwner ? profileNowNs() - syncStartNs : 0;
    // Owner-thread authored state was published above while the GIL and engine
    // owner context were intact. Background Python threads deliberately skip
    // that step and consume the last complete epoch. The native batch below
    // touches neither Python nor SceneManager, so releasing the GIL is safe.
    const auto dispatchNative = [&] {
        physics.RaycastBatchPublished(originData, normalizedDirectionData, static_cast<size_t>(count), maxDistance,
                                      nativeHits.data(), nativeMask.data(), layerMask, queryTriggers, &queryGeneration,
                                      profileEnabled ? &nativeProfile : nullptr, true);
    };
    // Short/medium batches already fan out through the native JobSystem.
    // Retaining the caller's GIL avoids a second, scheduler-dependent wait to
    // reacquire it after sub-millisecond native work. Very large calls still
    // release it so unrelated Python services remain responsive.
    constexpr py::ssize_t kReleaseGilRayCount = 65'536;
    if (count >= kReleaseGilRayCount) {
        py::gil_scoped_release release;
        dispatchNative();
    } else {
        dispatchNative();
    }
    if (profileEnabled)
        nativeProfile.snapshotSyncNs = syncNs;

    const uint64_t publishStartNs = profileEnabled ? profileNowNs() : 0;
    auto *hitData = hit.mutable_data();
    auto *pointData = point.mutable_data();
    auto *normalData = normal.mutable_data();
    auto *distanceData = distance.mutable_data();
    auto *bodyIdData = bodyId.mutable_data();
    auto *subShapeIdData = subShapeId.mutable_data();
    auto *triangleIndexData = triangleIndex.mutable_data();
    auto *colliderIdData = colliderId.mutable_data();
    auto *gameObjectIdData = gameObjectId.mutable_data();
    for (py::ssize_t index = 0; index < count; ++index) {
        const RaycastHit &native = nativeHits[static_cast<size_t>(index)];
        const bool didHit = nativeMask[static_cast<size_t>(index)] != 0;
        const py::ssize_t offset = index * 3;
        hitData[index] = didHit ? uint8_t{1} : uint8_t{0};
        pointData[offset] = native.point.x;
        pointData[offset + 1] = native.point.y;
        pointData[offset + 2] = native.point.z;
        normalData[offset] = native.normal.x;
        normalData[offset + 1] = native.normal.y;
        normalData[offset + 2] = native.normal.z;
        distanceData[index] = didHit ? native.distance : std::numeric_limits<float>::infinity();
        bodyIdData[index] = native.bodyId;
        subShapeIdData[index] = native.subShapeId;
        triangleIndexData[index] = native.triangleIndex;
        // The native query epoch has ended before Python publication.  Raw
        // pointers are convenience handles for synchronous C++ callers and
        // may already be retired by another thread; publish identities that
        // were frozen while the synchronized query epoch was held.
        colliderIdData[index] = native.colliderId;
        gameObjectIdData[index] = native.gameObjectId;
    }
    output[py::str("query_generation")] = py::int_(queryGeneration);
    if (profileEnabled) {
        const uint64_t publishNs = profileNowNs() - publishStartNs;
        const uint64_t broadphaseNs = nativeProfile.joltQueryCpuNs > nativeProfile.narrowPhaseCpuNs
                                          ? nativeProfile.joltQueryCpuNs - nativeProfile.narrowPhaseCpuNs
                                          : 0;
        constexpr double kNsToMs = 1.0 / 1'000'000.0;
        py::dict profile;
        profile["input_validation_ms"] = py::float_((validationEndNs - validationStartNs) * kNsToMs);
        profile["snapshot_sync_ms"] = py::float_(nativeProfile.snapshotSyncNs * kNsToMs);
        profile["snapshot_lock_wait_ms"] = py::float_(nativeProfile.snapshotLockWaitNs * kNsToMs);
        profile["dispatch_wall_ms"] = py::float_(nativeProfile.dispatchWallNs * kNsToMs);
        profile["jolt_query_cpu_ms"] = py::float_(nativeProfile.joltQueryCpuNs * kNsToMs);
        profile["broadphase_filter_lock_cpu_ms"] = py::float_(broadphaseNs * kNsToMs);
        profile["narrowphase_cpu_ms"] = py::float_(nativeProfile.narrowPhaseCpuNs * kNsToMs);
        profile["result_sort_filter_cpu_ms"] = py::float_(nativeProfile.sortFilterCpuNs * kNsToMs);
        profile["hit_publish_cpu_ms"] = py::float_(nativeProfile.hitPublishCpuNs * kNsToMs);
        profile["output_publish_ms"] = py::float_(publishNs * kNsToMs);
        profile["broadphase_candidates"] = py::int_(nativeProfile.broadphaseCandidates);
        profile["narrowphase_hits"] = py::int_(nativeProfile.narrowphaseHits);
        profile["published_hits"] = py::int_(nativeProfile.publishedHits);
        output[py::str("profile")] = std::move(profile);
    } else {
        output.attr("pop")(py::str("profile"), py::none());
    }
    return output;
}

py::array_t<float> RigidbodyStateArray(py::dict &result, const char *name, const std::vector<py::ssize_t> &shape,
                                       bool create)
{
    py::array value;
    if (create) {
        value = py::array_t<float>(shape);
        result[py::str(name)] = value;
    } else {
        const auto key = py::str(name);
        if (!result.contains(key))
            throw py::value_error(std::string("Rigidbody state output is missing '") + name + "'");
        value = py::cast<py::array>(result[key]);
    }
    if (!value.dtype().is(py::dtype::of<float>()) || !(value.flags() & py::array::c_style) || !value.writeable() ||
        value.ndim() != static_cast<py::ssize_t>(shape.size()))
        throw py::value_error(std::string("Rigidbody state '") + name + "' has the wrong float32 layout");
    if (value.shape(0) < shape[0])
        throw py::value_error(std::string("Rigidbody state '") + name + "' has insufficient capacity");
    for (py::ssize_t axis = 1; axis < value.ndim(); ++axis)
        if (value.shape(axis) != shape[static_cast<size_t>(axis)])
            throw py::value_error(std::string("Rigidbody state '") + name + "' has the wrong shape");
    return py::reinterpret_borrow<py::array_t<float>>(value);
}

py::dict GetRigidbodyStates(const std::vector<Rigidbody *> &bodies, py::object output)
{
    const auto count = static_cast<py::ssize_t>(bodies.size());
    const bool create = output.is_none();
    if (!create && !py::isinstance<py::dict>(output))
        throw py::type_error("Rigidbody state output must be a dictionary of writable arrays");
    py::dict result = create ? py::dict() : py::reinterpret_borrow<py::dict>(output);
    const std::vector<py::ssize_t> vectorShape{count, 3};
    auto position = RigidbodyStateArray(result, "position", vectorShape, create);
    auto center = RigidbodyStateArray(result, "center_of_mass", vectorShape, create);
    auto linear = RigidbodyStateArray(result, "linear_velocity", vectorShape, create);
    auto angular = RigidbodyStateArray(result, "angular_velocity", vectorShape, create);
    auto mass = RigidbodyStateArray(result, "inverse_mass", vectorShape, create);
    auto rotation = RigidbodyStateArray(result, "rotation", {count, 4}, create);
    auto inertia = RigidbodyStateArray(result, "inverse_inertia", {count, 3, 3}, create);
    auto p = position.mutable_unchecked<2>(), c = center.mutable_unchecked<2>(), v = linear.mutable_unchecked<2>();
    auto w = angular.mutable_unchecked<2>(), m = mass.mutable_unchecked<2>(), q = rotation.mutable_unchecked<2>();
    auto tensor = inertia.mutable_unchecked<3>();
    for (py::ssize_t row = 0; row < count; ++row) {
        if (!bodies[row])
            throw py::value_error("rigidbodies must not contain None");
        const auto state = bodies[row]->GetMotionState();
        for (int axis = 0; axis < 3; ++axis) {
            p(row, axis) = state.position[axis];
            c(row, axis) = state.centerOfMass[axis];
            v(row, axis) = state.linearVelocity[axis];
            w(row, axis) = state.angularVelocity[axis];
            m(row, axis) = state.inverseMass[axis];
            for (int column = 0; column < 3; ++column)
                tensor(row, axis, column) = state.inverseInertia[column][axis];
        }
        for (int axis = 0; axis < 4; ++axis)
            q(row, axis) = state.rotation[axis];
    }
    return result;
}

template <typename T> std::vector<uint8_t> ComputeBytes(const std::vector<T> &values)
{
    std::vector<uint8_t> result(values.size() * sizeof(T));
    if (!result.empty())
        std::memcpy(result.data(), values.data(), result.size());
    return result;
}

std::shared_ptr<rhi::ComputeBuffer> ComputeOutput(py::dict &output, const char *name, rhi::ComputeScalarType scalarType,
                                                  uint8_t lanes, uint64_t elementCount, rhi::ComputeHost *&host)
{
    const auto key = py::str(name);
    if (!output.contains(key))
        throw py::value_error(std::string("GPU state output is missing '") + name + "'");
    auto buffer = py::cast<std::shared_ptr<rhi::ComputeBuffer>>(output[key]);
    if (!buffer)
        throw py::value_error(std::string("GPU state output '") + name + "' must not be None");
    const auto &desc = buffer->GetDesc();
    if (desc.scalarType != scalarType || desc.lanes != lanes || desc.elementCount < elementCount)
        throw py::value_error(std::string("GPU state output '") + name + "' has the wrong type or capacity");
    if (host && !host->SharesServicesWith(buffer->GetHost()))
        throw py::value_error("GPU state outputs must belong to the same compute host");
    host = &buffer->GetHost();
    return buffer;
}

void AppendRigidbodyStateBufferUpdates(const std::vector<Rigidbody *> &bodies, py::dict output,
                                       std::vector<rhi::ComputeBufferUpdate> &updates, rhi::ComputeHost *&host)
{
    const auto count = static_cast<uint64_t>(bodies.size());
    auto position = ComputeOutput(output, "position", rhi::ComputeScalarType::Float32, 3, count, host);
    auto center = ComputeOutput(output, "center_of_mass", rhi::ComputeScalarType::Float32, 3, count, host);
    auto linear = ComputeOutput(output, "linear_velocity", rhi::ComputeScalarType::Float32, 3, count, host);
    auto angular = ComputeOutput(output, "angular_velocity", rhi::ComputeScalarType::Float32, 3, count, host);
    auto mass = ComputeOutput(output, "inverse_mass", rhi::ComputeScalarType::Float32, 3, count, host);
    auto rotation = ComputeOutput(output, "rotation", rhi::ComputeScalarType::Float32, 4, count, host);
    auto inertia = ComputeOutput(output, "inverse_inertia", rhi::ComputeScalarType::Float32, 1, count * 9, host);
    if (count == 0)
        return;

    std::vector<float> positions(count * 3), centers(count * 3), linears(count * 3), angulars(count * 3);
    std::vector<float> masses(count * 3), rotations(count * 4), inertias(count * 9);
    for (uint64_t row = 0; row < count; ++row) {
        if (!bodies[row])
            throw py::value_error("rigidbodies must not contain None");
        const auto state = bodies[row]->GetMotionState();
        for (uint64_t axis = 0; axis < 3; ++axis) {
            positions[row * 3 + axis] = state.position[axis];
            centers[row * 3 + axis] = state.centerOfMass[axis];
            linears[row * 3 + axis] = state.linearVelocity[axis];
            angulars[row * 3 + axis] = state.angularVelocity[axis];
            masses[row * 3 + axis] = state.inverseMass[axis];
            for (uint64_t column = 0; column < 3; ++column)
                inertias[row * 9 + axis * 3 + column] = state.inverseInertia[column][axis];
        }
        for (uint64_t axis = 0; axis < 4; ++axis)
            rotations[row * 4 + axis] = state.rotation[axis];
    }
    updates.push_back({position, 0, ComputeBytes(positions)});
    updates.push_back({center, 0, ComputeBytes(centers)});
    updates.push_back({linear, 0, ComputeBytes(linears)});
    updates.push_back({angular, 0, ComputeBytes(angulars)});
    updates.push_back({mass, 0, ComputeBytes(masses)});
    updates.push_back({rotation, 0, ComputeBytes(rotations)});
    updates.push_back({inertia, 0, ComputeBytes(inertias)});
}

void WriteRigidbodyStateBuffers(const std::vector<Rigidbody *> &bodies, py::dict output)
{
    rhi::ComputeHost *host = nullptr;
    std::vector<rhi::ComputeBufferUpdate> updates;
    AppendRigidbodyStateBufferUpdates(bodies, output, updates, host);
    if (!updates.empty())
        rhi::SubmitComputeBatch(*host, std::move(updates), {});
}

py::dict GetContactEvents()
{
    const auto &events = PhysicsWorld::Instance().GetContactEvents();
    const py::ssize_t count = static_cast<py::ssize_t>(events.size());
    py::array_t<uint8_t> type(count);
    py::array_t<uint32_t> bodyIds({count, py::ssize_t(2)});
    py::array_t<uint32_t> subShapeIds({count, py::ssize_t(2)});
    py::array_t<float> contactPoints({count, py::ssize_t(3)});
    py::array_t<float> contactNormals({count, py::ssize_t(3)});
    py::array_t<float> relativeVelocities({count, py::ssize_t(3)});

    auto eventTypes = type.mutable_unchecked<1>();
    auto bodies = bodyIds.mutable_unchecked<2>();
    auto subShapes = subShapeIds.mutable_unchecked<2>();
    auto points = contactPoints.mutable_unchecked<2>();
    auto normals = contactNormals.mutable_unchecked<2>();
    auto velocities = relativeVelocities.mutable_unchecked<2>();
    for (py::ssize_t index = 0; index < count; ++index) {
        const auto &event = events[static_cast<size_t>(index)];
        eventTypes(index) = static_cast<uint8_t>(event.type);
        bodies(index, 0) = event.bodyIdA;
        bodies(index, 1) = event.bodyIdB;
        subShapes(index, 0) = event.subShapeIdA;
        subShapes(index, 1) = event.subShapeIdB;
        for (int axis = 0; axis < 3; ++axis) {
            points(index, axis) = event.contactPoint[axis];
            normals(index, axis) = event.contactNormal[axis];
            velocities(index, axis) = event.relativeVelocity[axis];
        }
    }

    py::dict result;
    result["type"] = std::move(type);
    result["body_ids"] = std::move(bodyIds);
    result["sub_shape_ids"] = std::move(subShapeIds);
    result["contact_point"] = std::move(contactPoints);
    result["contact_normal"] = std::move(contactNormals);
    result["relative_velocity"] = std::move(relativeVelocities);
    return result;
}

py::dict GetContactImpulses()
{
    const auto &impulses = PhysicsWorld::Instance().GetContactImpulses();
    const py::ssize_t count = static_cast<py::ssize_t>(impulses.size());
    py::array_t<uint32_t> bodyIds({count, py::ssize_t(2)});
    py::array_t<uint32_t> subShapeIds({count, py::ssize_t(2)});
    py::array_t<float> points({count, py::ssize_t(3)});
    py::array_t<float> normals({count, py::ssize_t(3)});
    py::array_t<float> values({count, py::ssize_t(3)});
    auto bodies = bodyIds.mutable_unchecked<2>();
    auto subShapes = subShapeIds.mutable_unchecked<2>();
    auto pointValues = points.mutable_unchecked<2>();
    auto normalValues = normals.mutable_unchecked<2>();
    auto impulseValues = values.mutable_unchecked<2>();
    for (py::ssize_t index = 0; index < count; ++index) {
        const auto &impulse = impulses[static_cast<size_t>(index)];
        bodies(index, 0) = impulse.bodyIdA;
        bodies(index, 1) = impulse.bodyIdB;
        subShapes(index, 0) = impulse.subShapeIdA;
        subShapes(index, 1) = impulse.subShapeIdB;
        for (int axis = 0; axis < 3; ++axis) {
            pointValues(index, axis) = impulse.contactPoint[axis];
            normalValues(index, axis) = impulse.contactNormal[axis];
            impulseValues(index, axis) = impulse.impulse[axis];
        }
    }
    py::dict result;
    result["body_ids"] = std::move(bodyIds);
    result["sub_shape_ids"] = std::move(subShapeIds);
    result["contact_point"] = std::move(points);
    result["contact_normal"] = std::move(normals);
    result["impulse"] = std::move(values);
    return result;
}

py::array BoxStateArray(py::dict &result, const char *name, const py::dtype &dtype, py::ssize_t count,
                        const std::vector<py::ssize_t> &tail, bool create)
{
    std::vector<py::ssize_t> shape{count};
    shape.insert(shape.end(), tail.begin(), tail.end());
    py::array value;
    const auto key = py::str(name);
    if (create) {
        value = py::array(dtype, shape);
        result[key] = value;
    } else {
        if (!result.contains(key))
            throw py::value_error(std::string("Box collider state output is missing '") + name + "'");
        value = py::cast<py::array>(result[key]);
        if (value.ndim() != static_cast<py::ssize_t>(shape.size()) || value.shape(0) < count)
            throw py::value_error(std::string("Box collider state '") + name + "' has insufficient capacity");
        for (py::ssize_t axis = 1; axis < value.ndim(); ++axis)
            if (value.shape(axis) != shape[static_cast<size_t>(axis)])
                throw py::value_error(std::string("Box collider state '") + name + "' has the wrong shape");
    }
    if (!value.dtype().is(dtype) || !(value.flags() & py::array::c_style) || !value.writeable())
        throw py::value_error(std::string("Box collider state '") + name + "' has the wrong writable layout");
    return value;
}

py::dict GetRigidbodyBoxStates(const std::vector<Rigidbody *> &bodies, py::object output, bool queryTriggers)
{
    struct Entry
    {
        py::ssize_t bodyIndex;
        BoxCollider *collider;
    };
    std::vector<Entry> entries;
    for (py::ssize_t bodyIndex = 0; bodyIndex < static_cast<py::ssize_t>(bodies.size()); ++bodyIndex) {
        auto *body = bodies[bodyIndex];
        if (!body)
            throw py::value_error("rigidbodies must not contain None");
        (void)body->GetMotionState();
        auto *gameObject = body->GetGameObject();
        if (!gameObject)
            throw py::value_error("Rigidbody has no GameObject");
        for (auto *collider : gameObject->GetComponents<BoxCollider>())
            if (collider && collider->IsEnabled() && (queryTriggers || !collider->IsTrigger()))
                entries.push_back({bodyIndex, collider});
    }

    const auto count = static_cast<py::ssize_t>(entries.size());
    const bool create = output.is_none();
    if (!create && !py::isinstance<py::dict>(output))
        throw py::type_error("Box collider state output must be a dictionary of writable arrays");
    py::dict result = create ? py::dict() : py::reinterpret_borrow<py::dict>(output);
    auto bodyIndexArray = BoxStateArray(result, "body_index", py::dtype::of<int32_t>(), count, {}, create);
    auto centerArray = BoxStateArray(result, "center", py::dtype::of<float>(), count, {3}, create);
    auto rotationArray = BoxStateArray(result, "rotation", py::dtype::of<float>(), count, {4}, create);
    auto extentsArray = BoxStateArray(result, "half_extents", py::dtype::of<float>(), count, {3}, create);
    auto frictionArray = BoxStateArray(result, "friction", py::dtype::of<float>(), count, {}, create);
    auto bounceArray = BoxStateArray(result, "bounciness", py::dtype::of<float>(), count, {}, create);

    auto indices = bodyIndexArray.mutable_unchecked<int32_t, 1>();
    auto centers = centerArray.mutable_unchecked<float, 2>();
    auto rotations = rotationArray.mutable_unchecked<float, 2>();
    auto extents = extentsArray.mutable_unchecked<float, 2>();
    auto frictions = frictionArray.mutable_unchecked<float, 1>();
    auto bounces = bounceArray.mutable_unchecked<float, 1>();
    constexpr float kMinHalfExtent = 0.01f;
    for (py::ssize_t row = 0; row < count; ++row) {
        const auto &entry = entries[static_cast<size_t>(row)];
        auto *transform = entry.collider->GetGameObject()->GetTransform();
        const auto worldCenter = transform->TransformPoint(entry.collider->GetCenter());
        const auto worldRotation = glm::normalize(transform->GetWorldRotation());
        const auto halfExtents = glm::max(entry.collider->GetSize() * 0.5f * glm::abs(transform->GetWorldScale()),
                                          glm::vec3(kMinHalfExtent));
        indices(row) = static_cast<int32_t>(entry.bodyIndex);
        for (int axis = 0; axis < 3; ++axis) {
            centers(row, axis) = worldCenter[axis];
            extents(row, axis) = halfExtents[axis];
        }
        rotations(row, 0) = worldRotation.x;
        rotations(row, 1) = worldRotation.y;
        rotations(row, 2) = worldRotation.z;
        rotations(row, 3) = worldRotation.w;
        frictions(row) = entry.collider->GetFriction();
        bounces(row) = entry.collider->GetBounciness();
    }
    result[py::str("count")] = count;
    return result;
}

uint64_t AppendRigidbodyBoxStateBufferUpdates(const std::vector<Rigidbody *> &bodies, py::dict output,
                                              bool queryTriggers, std::vector<rhi::ComputeBufferUpdate> &updates,
                                              rhi::ComputeHost *&host)
{
    struct Entry
    {
        uint64_t bodyIndex;
        BoxCollider *collider;
    };
    std::vector<Entry> entries;
    for (uint64_t bodyIndex = 0; bodyIndex < bodies.size(); ++bodyIndex) {
        auto *body = bodies[bodyIndex];
        if (!body)
            throw py::value_error("rigidbodies must not contain None");
        (void)body->GetMotionState();
        auto *gameObject = body->GetGameObject();
        if (!gameObject)
            throw py::value_error("Rigidbody has no GameObject");
        for (auto *collider : gameObject->GetComponents<BoxCollider>())
            if (collider && collider->IsEnabled() && (queryTriggers || !collider->IsTrigger()))
                entries.push_back({bodyIndex, collider});
    }

    const auto count = static_cast<uint64_t>(entries.size());
    auto bodyIndex = ComputeOutput(output, "body_index", rhi::ComputeScalarType::Int32, 1, count, host);
    auto center = ComputeOutput(output, "center", rhi::ComputeScalarType::Float32, 3, count, host);
    auto rotation = ComputeOutput(output, "rotation", rhi::ComputeScalarType::Float32, 4, count, host);
    auto extents = ComputeOutput(output, "half_extents", rhi::ComputeScalarType::Float32, 3, count, host);
    auto friction = ComputeOutput(output, "friction", rhi::ComputeScalarType::Float32, 1, count, host);
    auto bounce = ComputeOutput(output, "bounciness", rhi::ComputeScalarType::Float32, 1, count, host);
    if (count == 0)
        return 0;

    std::vector<int32_t> indices(count);
    std::vector<float> centers(count * 3), rotations(count * 4), halfExtents(count * 3), frictions(count),
        bounces(count);
    constexpr float kMinHalfExtent = 0.01f;
    for (uint64_t row = 0; row < count; ++row) {
        const auto &entry = entries[row];
        auto *transform = entry.collider->GetGameObject()->GetTransform();
        const auto worldCenter = transform->TransformPoint(entry.collider->GetCenter());
        const auto worldRotation = glm::normalize(transform->GetWorldRotation());
        const auto halfExtent = glm::max(entry.collider->GetSize() * 0.5f * glm::abs(transform->GetWorldScale()),
                                         glm::vec3(kMinHalfExtent));
        indices[row] = static_cast<int32_t>(entry.bodyIndex);
        for (uint64_t axis = 0; axis < 3; ++axis) {
            centers[row * 3 + axis] = worldCenter[axis];
            halfExtents[row * 3 + axis] = halfExtent[axis];
        }
        rotations[row * 4] = worldRotation.x;
        rotations[row * 4 + 1] = worldRotation.y;
        rotations[row * 4 + 2] = worldRotation.z;
        rotations[row * 4 + 3] = worldRotation.w;
        frictions[row] = entry.collider->GetFriction();
        bounces[row] = entry.collider->GetBounciness();
    }
    updates.push_back({bodyIndex, 0, ComputeBytes(indices)});
    updates.push_back({center, 0, ComputeBytes(centers)});
    updates.push_back({rotation, 0, ComputeBytes(rotations)});
    updates.push_back({extents, 0, ComputeBytes(halfExtents)});
    updates.push_back({friction, 0, ComputeBytes(frictions)});
    updates.push_back({bounce, 0, ComputeBytes(bounces)});
    return count;
}

uint64_t WriteRigidbodyBoxStateBuffers(const std::vector<Rigidbody *> &bodies, py::dict output, bool queryTriggers)
{
    rhi::ComputeHost *host = nullptr;
    std::vector<rhi::ComputeBufferUpdate> updates;
    const auto count = AppendRigidbodyBoxStateBufferUpdates(bodies, output, queryTriggers, updates, host);
    if (!updates.empty())
        rhi::SubmitComputeBatch(*host, std::move(updates), {});
    return count;
}

uint64_t WriteRigidbodyStateAndBoxStateBuffers(const std::vector<Rigidbody *> &bodies, py::dict stateOutput,
                                               py::dict boxOutput, bool queryTriggers)
{
    rhi::ComputeHost *host = nullptr;
    std::vector<rhi::ComputeBufferUpdate> updates;
    AppendRigidbodyStateBufferUpdates(bodies, stateOutput, updates, host);
    const auto boxCount = AppendRigidbodyBoxStateBufferUpdates(bodies, boxOutput, queryTriggers, updates, host);
    if (!updates.empty())
        rhi::SubmitComputeBatch(*host, std::move(updates), {});
    return boxCount;
}

py::tuple QueryRigidbodyBoxStateBuffers(const glm::vec3 &minimum, const glm::vec3 &maximum, uint32_t layerMask,
                                        bool queryTriggers, py::dict output)
{
    RequireFinite(minimum, "minimum");
    RequireFinite(maximum, "maximum");
    if (minimum.x > maximum.x || minimum.y > maximum.y || minimum.z > maximum.z)
        throw py::value_error("minimum must not exceed maximum on any axis");
    auto bodies = PhysicsWorld::Instance().QueryRigidbodiesInBounds(minimum, maximum, layerMask, queryTriggers);
    const auto count = WriteRigidbodyBoxStateBuffers(bodies, output, queryTriggers);
    py::list result;
    for (auto *body : bodies)
        result.append(py::cast(body, py::return_value_policy::reference));
    return py::make_tuple(result, count);
}

py::tuple QueryRigidbodyStateAndBoxStateBuffers(const glm::vec3 &minimum, const glm::vec3 &maximum, uint32_t layerMask,
                                                bool queryTriggers, py::dict stateOutput, py::dict boxOutput)
{
    RequireFinite(minimum, "minimum");
    RequireFinite(maximum, "maximum");
    if (minimum.x > maximum.x || minimum.y > maximum.y || minimum.z > maximum.z)
        throw py::value_error("minimum must not exceed maximum on any axis");
    const auto bodies = PhysicsWorld::Instance().QueryRigidbodiesInBounds(minimum, maximum, layerMask, queryTriggers);
    rhi::ComputeHost *host = nullptr;
    std::vector<rhi::ComputeBufferUpdate> updates;
    AppendRigidbodyStateBufferUpdates(bodies, stateOutput, updates, host);
    const auto boxCount = AppendRigidbodyBoxStateBufferUpdates(bodies, boxOutput, queryTriggers, updates, host);
    if (!updates.empty())
        rhi::SubmitComputeBatch(*host, std::move(updates), {});
    py::list result;
    for (auto *body : bodies)
        result.append(py::cast(body, py::return_value_policy::reference));
    return py::make_tuple(result, boxCount);
}

void ApplyRigidbodyImpulseValues(const std::vector<Rigidbody *> &bodies, const float *linear, const float *angular)
{
    const auto count = static_cast<py::ssize_t>(bodies.size());
    for (py::ssize_t i = 0; i < count * 3; ++i)
        if (!std::isfinite(linear[i]) || !std::isfinite(angular[i]))
            throw py::value_error("Impulse buffers must contain finite values");
    // Reject invalid public input before any body receives feedback.
    for (const auto *body : bodies) {
        if (!body)
            throw py::value_error("rigidbodies must not contain None");
        (void)body->GetMotionState();
    }
    for (py::ssize_t row = 0; row < count; ++row) {
        const auto offset = static_cast<size_t>(row) * 3;
        const glm::vec3 impulse(linear[offset], linear[offset + 1], linear[offset + 2]);
        const glm::vec3 torque(angular[offset], angular[offset + 1], angular[offset + 2]);
        if (impulse != glm::vec3(0.0f))
            bodies[row]->AddForce(impulse, ForceMode::Impulse);
        if (torque != glm::vec3(0.0f))
            bodies[row]->AddTorque(torque, ForceMode::Impulse);
    }
}

void ApplyRigidbodyImpulses(const std::vector<Rigidbody *> &bodies,
                            const py::array_t<float, py::array::c_style> &linear,
                            const py::array_t<float, py::array::c_style> &angular)
{
    const auto count = static_cast<py::ssize_t>(bodies.size());
    for (const auto &array : {linear, angular})
        if (array.ndim() != 2 || array.shape(0) != count || array.shape(1) != 3)
            throw py::value_error("Impulse arrays must have shape (body_count, 3)");
    ApplyRigidbodyImpulseValues(bodies, linear.data(), angular.data());
}

void ApplyRigidbodyImpulseBuffers(const std::vector<Rigidbody *> &bodies,
                                  const std::shared_ptr<rhi::ComputeBuffer> &linear,
                                  const std::shared_ptr<rhi::ComputeBuffer> &angular)
{
    if (!linear || !angular)
        throw py::value_error("Impulse buffers must not be None");
    const auto count = static_cast<uint64_t>(bodies.size());
    for (const auto &buffer : {linear, angular}) {
        const auto &desc = buffer->GetDesc();
        if (desc.scalarType != rhi::ComputeScalarType::Float32 || desc.lanes != 3 || desc.elementCount < count)
            throw py::value_error("Impulse GPU buffers must be float32 vector3 with body-count capacity");
    }
    if (!linear->GetHost().SharesServicesWith(angular->GetHost()))
        throw py::value_error("Impulse GPU buffers must belong to the same compute host");
    if (count == 0)
        return;
    const auto byteSize = count * sizeof(float) * 3;
    const auto values = rhi::ReadComputeBatch(linear->GetHost(), {{linear, 0, byteSize}, {angular, 0, byteSize}});
    std::vector<float> linearValues(static_cast<size_t>(count) * 3);
    std::vector<float> angularValues(static_cast<size_t>(count) * 3);
    std::memcpy(linearValues.data(), values[0].data(), static_cast<size_t>(byteSize));
    std::memcpy(angularValues.data(), values[1].data(), static_cast<size_t>(byteSize));
    ApplyRigidbodyImpulseValues(bodies, linearValues.data(), angularValues.data());
}

void RequirePositive(float value, const char *name)
{
    if (!std::isfinite(value) || value <= 0.0f)
        throw std::invalid_argument(std::string(name) + " must be finite and greater than zero");
}

void RequirePositiveExtents(const glm::vec3 &value)
{
    RequireFinite(value, "half_extents");
    if (value.x <= 0.0f || value.y <= 0.0f || value.z <= 0.0f)
        throw std::invalid_argument("half_extents must be greater than zero on every axis");
}

void RequireOrientation(const glm::quat &orientation)
{
    if (!std::isfinite(orientation.w) || !std::isfinite(orientation.x) || !std::isfinite(orientation.y) ||
        !std::isfinite(orientation.z) || glm::dot(orientation, orientation) <= 1e-12f)
        throw std::invalid_argument("orientation must be a finite, non-zero quaternion");
}

py::list BorrowedColliderList(const std::vector<Collider *> &colliders)
{
    py::list result;
    for (Collider *collider : colliders) {
        if (collider)
            result.append(py::cast(collider, py::return_value_policy::reference));
    }
    return result;
}
} // namespace

void RegisterPhysicsBindings(py::module_ &m)
{
    using namespace pybind11::literals;

    // ====================================================================
    // CollisionInfo struct (Unity: Collision)
    // ====================================================================
    py::class_<CollisionInfo>(m, "CollisionInfo")
        .def(py::init<>())
        .def_property_readonly(
            "collider", [](const CollisionInfo &c) { return c.collider; }, py::return_value_policy::reference,
            "The other Collider involved in the collision")
        .def_property_readonly(
            "game_object", [](const CollisionInfo &c) { return c.gameObject; }, py::return_value_policy::reference,
            "The other GameObject involved in the collision")
        .def_property_readonly(
            "contact_point", [](const CollisionInfo &c) { return c.contactPoint; }, "World-space contact point")
        .def_property_readonly(
            "contact_normal", [](const CollisionInfo &c) { return c.contactNormal; },
            "Contact normal (points from other towards this)")
        .def_property_readonly(
            "relative_velocity", [](const CollisionInfo &c) { return c.relativeVelocity; },
            "Relative velocity between the two bodies")
        .def("__repr__", [](const CollisionInfo &c) {
            std::string goName = c.gameObject ? c.gameObject->GetName() : "null";
            return "<CollisionInfo other='" + goName + "'>";
        });

    // ====================================================================
    // RaycastHit struct
    // ====================================================================
    py::class_<PhysicsPenetrationResult>(m, "PenetrationResult")
        .def_readonly("direction", &PhysicsPenetrationResult::direction,
                      "World-space unit direction to push A out of B")
        .def_readonly("distance", &PhysicsPenetrationResult::distance, "Positive penetration depth in meters")
        .def_readonly("point_a", &PhysicsPenetrationResult::pointA, "World-space point on A")
        .def_readonly("point_b", &PhysicsPenetrationResult::pointB, "World-space point on B");

    py::class_<RaycastHit>(m, "RaycastHit")
        .def(py::init<>())
        .def_property_readonly(
            "point", [](const RaycastHit &h) { return h.point; }, "World-space hit point")
        .def_property_readonly(
            "normal", [](const RaycastHit &h) { return h.normal; }, "Surface normal at hit point")
        .def_property_readonly(
            "distance", [](const RaycastHit &h) { return h.distance; }, "Distance from ray origin to hit")
        .def_property_readonly(
            "body_id", [](const RaycastHit &h) { return h.bodyId; }, "Native physics body identity")
        .def_property_readonly(
            "sub_shape_id", [](const RaycastHit &h) { return h.subShapeId; }, "Sub-shape identity inside the body")
        .def_property_readonly(
            "triangle_index",
            [](const RaycastHit &h) -> py::object {
                return h.triangleIndex == 0xFFFFFFFF ? py::none() : py::cast(h.triangleIndex);
            },
            "Cooked triangle index for a non-convex MeshCollider hit, otherwise None")
        .def_property_readonly(
            "game_object",
            [](const RaycastHit &h) -> GameObject * {
                if (h.gameObject)
                    return h.gameObject;
                if (h.collider)
                    return h.collider->GetGameObject();
                if (h.bodyId == 0xFFFFFFFF)
                    return nullptr;
                if (Collider *col = PhysicsWorld::Instance().FindColliderByBodyId(h.bodyId))
                    return col->GetGameObject();
                return nullptr;
            },
            py::return_value_policy::reference, "Hit GameObject")
        .def_property_readonly(
            "collider",
            [](const RaycastHit &h) -> Collider * {
                if (h.collider)
                    return h.collider;
                if (h.bodyId == 0xFFFFFFFF)
                    return nullptr;
                return PhysicsWorld::Instance().FindColliderByBodyId(h.bodyId);
            },
            py::return_value_policy::reference, "Hit Collider component")
        .def("__repr__", [](const RaycastHit &h) { return "<RaycastHit dist=" + std::to_string(h.distance) + ">"; });

    // ====================================================================
    py::enum_<PhysicsMaterialCombine>(m, "PhysicsMaterialCombine")
        .value("Average", PhysicsMaterialCombine::Average)
        .value("Minimum", PhysicsMaterialCombine::Minimum)
        .value("Multiply", PhysicsMaterialCombine::Multiply)
        .value("Maximum", PhysicsMaterialCombine::Maximum);

    // Collider base (abstract — not directly constructible)
    // ====================================================================
    py::class_<Collider, Component>(m, "Collider")
        .def_property("is_trigger", &Collider::IsTrigger, &Collider::SetIsTrigger, "Is this collider a trigger volume?")
        .def_property(
            "center", [](Collider *c) { return c->GetCenter(); },
            [](Collider *c, const glm::vec3 &v) { c->SetCenter(v); }, "Center offset in local space")
        .def_property("physic_material", &Collider::GetPhysicMaterial, &Collider::SetPhysicMaterial,
                      "Shared PhysicMaterial; None uses engine defaults")
        .def_property("physic_material_guid", &Collider::GetPhysicMaterialGuid, &Collider::SetPhysicMaterialGuid,
                      "GUID of the persistent PhysicMaterial asset")
        .def(
            "raycast",
            [](const Collider &collider, const glm::vec3 &origin, const glm::vec3 &direction,
               float maxDistance) -> py::object {
                RequireFinite(origin, "origin");
                RequireDirectionAndDistance(direction, maxDistance);
                RaycastHit hit;
                if (PhysicsWorld::Instance().RaycastCollider(collider, origin, direction, maxDistance, hit))
                    return py::cast(hit);
                return py::none();
            },
            "origin"_a, "direction"_a, "max_distance"_a = 1000.0f,
            "Cast a ray against this collider only; layer and trigger filters do not apply")
        .def(
            "closest_point",
            [](const Collider &collider, const glm::vec3 &point) {
                RequireFinite(point, "point");
                return PhysicsWorld::Instance().ClosestPointOnCollider(collider, point);
            },
            "point"_a, "Return the closest world-space point on this collider; an interior point is unchanged")
        .def("serialize", &Collider::Serialize)
        .def("deserialize", &Collider::Deserialize, "json_str"_a);

    // ====================================================================
    // BoxCollider
    // ====================================================================
    py::class_<BoxCollider, Collider>(m, "BoxCollider")
        .def(py::init<>())
        .def_property(
            "size", [](BoxCollider *c) { return c->GetSize(); },
            [](BoxCollider *c, const glm::vec3 &v) { c->SetSize(v); }, "Size of the box collider (full extents)")
        .def("serialize", &BoxCollider::Serialize)
        .def("deserialize", &BoxCollider::Deserialize, "json_str"_a);

    // ====================================================================
    // SphereCollider
    // ====================================================================
    py::class_<SphereCollider, Collider>(m, "SphereCollider")
        .def(py::init<>())
        .def_property("radius", &SphereCollider::GetRadius, &SphereCollider::SetRadius, "Radius of the sphere collider")
        .def("serialize", &SphereCollider::Serialize)
        .def("deserialize", &SphereCollider::Deserialize, "json_str"_a);

    // ====================================================================
    // CapsuleCollider
    // ====================================================================
    py::class_<CapsuleCollider, Collider>(m, "CapsuleCollider")
        .def(py::init<>())
        .def_property("radius", &CapsuleCollider::GetRadius, &CapsuleCollider::SetRadius,
                      "Radius of the capsule collider")
        .def_property("height", &CapsuleCollider::GetHeight, &CapsuleCollider::SetHeight,
                      "Total height of the capsule (including caps)")
        .def_property("direction", &CapsuleCollider::GetDirection, &CapsuleCollider::SetDirection,
                      "Direction axis: 0=X, 1=Y, 2=Z")
        .def("serialize", &CapsuleCollider::Serialize)
        .def("deserialize", &CapsuleCollider::Deserialize, "json_str"_a);

    // ====================================================================
    // CylinderCollider
    // ====================================================================
    py::class_<CylinderCollider, Collider>(m, "CylinderCollider")
        .def(py::init<>())
        .def_property("radius", &CylinderCollider::GetRadius, &CylinderCollider::SetRadius,
                      "Radius of the cylinder collider")
        .def_property("height", &CylinderCollider::GetHeight, &CylinderCollider::SetHeight,
                      "Total height of the cylinder")
        .def_property("direction", &CylinderCollider::GetDirection, &CylinderCollider::SetDirection,
                      "Direction axis: 0=X, 1=Y, 2=Z")
        .def("serialize", &CylinderCollider::Serialize)
        .def("deserialize", &CylinderCollider::Deserialize, "json_str"_a);

    // ====================================================================
    // MeshCollider
    // ====================================================================
    py::class_<MeshCollider, Collider>(m, "MeshCollider")
        .def(py::init<>())
        .def("recook", &MeshCollider::OnMeshGeometryChanged,
             "Request collision rebuilding from current sibling MeshRenderer geometry")
        .def_property("convex", &MeshCollider::IsConvex, &MeshCollider::SetConvex,
                      "Use convex hull collision. Dynamic rigidbodies set this property to true.")
        .def_property_readonly("shape_error", &MeshCollider::GetShapeError,
                               "Last mesh cooking error; empty after successful shape creation")
        .def_property_readonly("is_cooking", &MeshCollider::IsCooking,
                               "Whether immutable collision geometry is currently cooking on a worker")
        .def_property_readonly(
            "collision_geometry_revision", &MeshCollider::GetCollisionGeometryRevision,
            "Monotonic revision of collision geometry that has been atomically published to the physics world")
        .def_static("clear_cooking_cache", &MeshCollider::ClearCookingCache,
                    "Clear cached CPU mesh-cooking payloads and reset hit/miss counters")
        .def_static(
            "get_cooking_cache_stats",
            []() {
                const auto [hits, misses] = MeshCollider::GetCookingCacheStats();
                py::dict stats;
                stats["hits"] = hits;
                stats["misses"] = misses;
                stats["pending"] = MeshCollider::GetPendingCookingCount();
                stats["async_submissions"] = MeshCollider::GetAsyncCookingSubmissionCount();
                return stats;
            },
            "Return CPU mesh-cooking cache and worker counters")
        .def(
            "get_convex_hull_positions",
            [](const MeshCollider &mc) -> py::list {
                py::list result;
                for (const auto &v : mc.GetConvexHullPositions()) {
                    result.append(py::make_tuple(v.x, v.y, v.z));
                }
                return result;
            },
            "Convex hull vertex positions in local space")
        .def(
            "get_convex_hull_edges",
            [](const MeshCollider &mc) -> py::list {
                py::list result;
                for (auto idx : mc.GetConvexHullEdges()) {
                    result.append(idx);
                }
                return result;
            },
            "Convex hull edge index pairs [a0,b0, a1,b1, ...]")
        .def("serialize", &MeshCollider::Serialize)
        .def("deserialize", &MeshCollider::Deserialize, "json_str"_a);

    // ====================================================================
    // ForceMode enum (Unity: ForceMode)
    // ====================================================================
    py::enum_<ForceMode>(m, "ForceMode")
        .value("Force", ForceMode::Force, "Continuous force (mass-dependent)")
        .value("Acceleration", ForceMode::Acceleration, "Continuous acceleration (mass-independent)")
        .value("Impulse", ForceMode::Impulse, "Instant force impulse (mass-dependent)")
        .value("VelocityChange", ForceMode::VelocityChange, "Instant velocity change (mass-independent)")
        .export_values();

    // ====================================================================
    // RigidbodyConstraints enum (Unity: RigidbodyConstraints)
    // ====================================================================
    py::enum_<RigidbodyConstraints>(m, "RigidbodyConstraints")
        .value("None", RigidbodyConstraints::None, "No constraints")
        .value("FreezePositionX", RigidbodyConstraints::FreezePositionX)
        .value("FreezePositionY", RigidbodyConstraints::FreezePositionY)
        .value("FreezePositionZ", RigidbodyConstraints::FreezePositionZ)
        .value("FreezeRotationX", RigidbodyConstraints::FreezeRotationX)
        .value("FreezeRotationY", RigidbodyConstraints::FreezeRotationY)
        .value("FreezeRotationZ", RigidbodyConstraints::FreezeRotationZ)
        .value("FreezePosition", RigidbodyConstraints::FreezePosition, "Freeze all position axes")
        .value("FreezeRotation", RigidbodyConstraints::FreezeRotation, "Freeze all rotation axes")
        .value("FreezeAll", RigidbodyConstraints::FreezeAll, "Freeze all position and rotation axes")
        .export_values();

    // ====================================================================
    // CollisionDetectionMode enum (Unity: CollisionDetectionMode)
    // ====================================================================
    py::enum_<CollisionDetectionMode>(m, "CollisionDetectionMode")
        .value("Discrete", CollisionDetectionMode::Discrete)
        .value("Continuous", CollisionDetectionMode::Continuous)
        .value("ContinuousDynamic", CollisionDetectionMode::ContinuousDynamic)
        .export_values();

    py::enum_<RigidbodyInterpolation>(m, "RigidbodyInterpolation")
        .value("None", RigidbodyInterpolation::None)
        .value("Interpolate", RigidbodyInterpolation::Interpolate)
        .export_values();

    // ====================================================================
    // Rigidbody component (Unity: Rigidbody)
    // ====================================================================
    py::class_<Rigidbody, Component>(m, "Rigidbody")
        .def(py::init<>())
        // ---- Serialized properties ----
        .def_property("mass", &Rigidbody::GetMass, &Rigidbody::SetMass, "Mass in kilograms (default 1)")
        .def_property("drag", &Rigidbody::GetDrag, &Rigidbody::SetDrag, "Linear drag (default 0)")
        .def_property("angular_drag", &Rigidbody::GetAngularDrag, &Rigidbody::SetAngularDrag,
                      "Angular drag (default 0.05)")
        .def_property("use_gravity", &Rigidbody::GetUseGravity, &Rigidbody::SetUseGravity,
                      "Use gravity? (default true)")
        .def_property("is_kinematic", &Rigidbody::IsKinematic, &Rigidbody::SetIsKinematic,
                      "Is kinematic? (default false)")
        .def_property("constraints", &Rigidbody::GetConstraints, &Rigidbody::SetConstraints,
                      "Constraints bitmask (RigidbodyConstraints)")
        .def_property("freeze_rotation", &Rigidbody::GetFreezeRotation, &Rigidbody::SetFreezeRotation,
                      "Shortcut to freeze all rotation axes")
        .def_property(
            "freeze_position_x", [](Rigidbody *rb) { return (static_cast<int>(rb->GetConstraints()) & 2) != 0; },
            [](Rigidbody *rb, bool v) {
                int c = static_cast<int>(rb->GetConstraints());
                rb->SetConstraints(v ? (c | 2) : (c & ~2));
            },
            "Freeze position X axis")
        .def_property(
            "freeze_position_y", [](Rigidbody *rb) { return (static_cast<int>(rb->GetConstraints()) & 4) != 0; },
            [](Rigidbody *rb, bool v) {
                int c = static_cast<int>(rb->GetConstraints());
                rb->SetConstraints(v ? (c | 4) : (c & ~4));
            },
            "Freeze position Y axis")
        .def_property(
            "freeze_position_z", [](Rigidbody *rb) { return (static_cast<int>(rb->GetConstraints()) & 8) != 0; },
            [](Rigidbody *rb, bool v) {
                int c = static_cast<int>(rb->GetConstraints());
                rb->SetConstraints(v ? (c | 8) : (c & ~8));
            },
            "Freeze position Z axis")
        .def_property(
            "freeze_rotation_x", [](Rigidbody *rb) { return (static_cast<int>(rb->GetConstraints()) & 16) != 0; },
            [](Rigidbody *rb, bool v) {
                int c = static_cast<int>(rb->GetConstraints());
                rb->SetConstraints(v ? (c | 16) : (c & ~16));
            },
            "Freeze rotation X axis")
        .def_property(
            "freeze_rotation_y", [](Rigidbody *rb) { return (static_cast<int>(rb->GetConstraints()) & 32) != 0; },
            [](Rigidbody *rb, bool v) {
                int c = static_cast<int>(rb->GetConstraints());
                rb->SetConstraints(v ? (c | 32) : (c & ~32));
            },
            "Freeze rotation Y axis")
        .def_property(
            "freeze_rotation_z", [](Rigidbody *rb) { return (static_cast<int>(rb->GetConstraints()) & 64) != 0; },
            [](Rigidbody *rb, bool v) {
                int c = static_cast<int>(rb->GetConstraints());
                rb->SetConstraints(v ? (c | 64) : (c & ~64));
            },
            "Freeze rotation Z axis")
        .def_property("collision_detection_mode", &Rigidbody::GetCollisionDetectionMode,
                      &Rigidbody::SetCollisionDetectionMode,
                      "Collision detection mode. Dynamic Continuous uses Jolt LinearCast sweep CCD.")
        .def_property("interpolation", &Rigidbody::GetInterpolation, &Rigidbody::SetInterpolation,
                      "Presentation interpolation mode (0=None, 1=Interpolate)")
        .def_property("max_angular_velocity", &Rigidbody::GetMaxAngularVelocity, &Rigidbody::SetMaxAngularVelocity,
                      "Maximum angular velocity in rad/s (default 7)")
        .def_property("max_linear_velocity", &Rigidbody::GetMaxLinearVelocity, &Rigidbody::SetMaxLinearVelocity,
                      "Maximum linear velocity in m/s")
        // ---- Velocity ----
        .def_property(
            "velocity", [](Rigidbody *rb) { return rb->GetVelocity(); },
            [](Rigidbody *rb, const glm::vec3 &v) { rb->SetVelocity(v); }, "Linear velocity in world space")
        .def_property(
            "angular_velocity", [](Rigidbody *rb) { return rb->GetAngularVelocity(); },
            [](Rigidbody *rb, const glm::vec3 &v) { rb->SetAngularVelocity(v); }, "Angular velocity in world space")
        // ---- Read-only world info ----
        .def(
            "get_point_velocity",
            [](const Rigidbody &body, const glm::vec3 &point) {
                RequireFinite(point, "point");
                return body.GetVelocity() + glm::cross(body.GetAngularVelocity(), point - body.GetWorldCenterOfMass());
            },
            "point"_a, "World-space velocity at a point, including rotation about the physical center of mass")
        .def("get_point_velocities", &GetPointVelocities, py::arg("points").noconvert(), py::arg("output").noconvert(),
             "Write world-space point velocities into caller-owned float32 (N, 3) output")
        .def_property_readonly(
            "world_center_of_mass", [](Rigidbody *rb) { return rb->GetWorldCenterOfMass(); },
            "World-space center of mass (read-only)")
        .def_property("position", &Rigidbody::GetPosition, &Rigidbody::SetPosition,
                      "World-space position. Assignment teleports while preserving velocity.")
        .def_property("rotation", &Rigidbody::GetRotation, &Rigidbody::SetRotation,
                      "World-space rotation. Assignment teleports while preserving velocity.")
        // ---- Force / Torque ----
        .def(
            "add_force", [](Rigidbody *rb, const glm::vec3 &f, ForceMode mode) { rb->AddForce(f, mode); }, "force"_a,
            "mode"_a = ForceMode::Force, "Add a force to the rigidbody")
        .def(
            "add_torque", [](Rigidbody *rb, const glm::vec3 &t, ForceMode mode) { rb->AddTorque(t, mode); }, "torque"_a,
            "mode"_a = ForceMode::Force, "Add a torque to the rigidbody")
        .def(
            "add_force_at_position",
            [](Rigidbody *rb, const glm::vec3 &f, const glm::vec3 &p, ForceMode mode) {
                rb->AddForceAtPosition(f, p, mode);
            },
            "force"_a, "position"_a, "mode"_a = ForceMode::Force, "Add a force at a world-space position")
        // ---- Kinematic movement ----
        .def(
            "move_position", [](Rigidbody *rb, const glm::vec3 &p) { rb->MovePosition(p); }, "position"_a,
            "Move kinematic body to target position")
        .def(
            "move_rotation", [](Rigidbody *rb, const glm::quat &q) { rb->MoveRotation(q); }, "rotation"_a,
            "Rotate kinematic body to target rotation")
        // ---- Sleep ----
        .def("is_sleeping", &Rigidbody::IsSleeping, "Is the rigidbody sleeping?")
        .def("wake_up", &Rigidbody::WakeUp, "Wake the rigidbody up")
        .def("sleep", &Rigidbody::Sleep, "Put the rigidbody to sleep")
        .def("serialize", &Rigidbody::Serialize)
        .def("deserialize", &Rigidbody::Deserialize, "json_str"_a);

    py::class_<HingeJoint, Component>(m, "HingeJoint")
        .def(py::init<>())
        .def_property("anchor", &HingeJoint::GetAnchor, &HingeJoint::SetAnchor)
        .def_property("axis", &HingeJoint::GetAxis, &HingeJoint::SetAxis)
        .def_property("connected_body", &HingeJoint::GetConnectedBody, &HingeJoint::SetConnectedBody,
                      py::return_value_policy::reference)
        .def_property("use_limits", &HingeJoint::GetUseLimits, &HingeJoint::SetUseLimits)
        .def_property("minimum_angle", &HingeJoint::GetMinimumAngle, &HingeJoint::SetMinimumAngle)
        .def_property("maximum_angle", &HingeJoint::GetMaximumAngle, &HingeJoint::SetMaximumAngle)
        .def_property("enable_collision", &HingeJoint::GetEnableCollision, &HingeJoint::SetEnableCollision)
        .def_property_readonly("current_angle", &HingeJoint::GetCurrentAngle)
        .def("serialize", &HingeJoint::Serialize)
        .def("deserialize", &HingeJoint::Deserialize, "json_str"_a);

    py::class_<SliderJoint, Component>(m, "SliderJoint")
        .def(py::init<>())
        .def_property("anchor", &SliderJoint::GetAnchor, &SliderJoint::SetAnchor)
        .def_property("axis", &SliderJoint::GetAxis, &SliderJoint::SetAxis)
        .def_property("connected_body", &SliderJoint::GetConnectedBody, &SliderJoint::SetConnectedBody,
                      py::return_value_policy::reference)
        .def_property("use_limits", &SliderJoint::GetUseLimits, &SliderJoint::SetUseLimits)
        .def_property("minimum_distance", &SliderJoint::GetMinimumDistance, &SliderJoint::SetMinimumDistance)
        .def_property("maximum_distance", &SliderJoint::GetMaximumDistance, &SliderJoint::SetMaximumDistance)
        .def_property("enable_collision", &SliderJoint::GetEnableCollision, &SliderJoint::SetEnableCollision)
        .def_property_readonly("current_position", &SliderJoint::GetCurrentPosition)
        .def("serialize", &SliderJoint::Serialize)
        .def("deserialize", &SliderJoint::Deserialize, "json_str"_a);

    // ====================================================================
    // Physics static class (Unity: Physics.Raycast)
    // ====================================================================
    py::class_<PhysicsWorld, std::unique_ptr<PhysicsWorld, py::nodelete>>(m, "Physics")
        .def_static(
            "compute_penetration",
            [](const Collider &a, const glm::vec3 &pa, const glm::quat &qa, const Collider &b, const glm::vec3 &pb,
               const glm::quat &qb) {
                RequireFinite(pa, "position_a");
                RequireFinite(pb, "position_b");
                RequireOrientation(qa);
                RequireOrientation(qb);
                return PhysicsWorld::Instance().ComputePenetration(a, pa, qa, b, pb, qb);
            },
            "collider_a"_a, "position_a"_a, "rotation_a"_a, "collider_b"_a, "position_b"_a, "rotation_b"_a,
            "Pure primitive geometry query at supplied world poses; returns None or separation/contact data")
        .def_static("get_rigidbody_states", &GetRigidbodyStates, "rigidbodies"_a, "output"_a = py::none(),
                    "Write current solver state in input body order; output may reuse writable float32 arrays")
        .def_static("_write_rigidbody_state_buffers", &WriteRigidbodyStateBuffers, "rigidbodies"_a, "output"_a,
                    "Upload current solver state directly into resident GPU buffers")
        .def_static("_write_rigidbody_state_and_box_state_buffers", &WriteRigidbodyStateAndBoxStateBuffers,
                    "rigidbodies"_a, "state_output"_a, "box_output"_a, "query_triggers"_a = false,
                    "Upload rigidbody solver state and flattened BoxCollider descriptors in one compute submission")
        .def_static("get_rigidbody_box_states", &GetRigidbodyBoxStates, "rigidbodies"_a, "output"_a = py::none(),
                    "query_triggers"_a = false,
                    "Write flattened world-space BoxCollider descriptors and source body indices")
        .def_static("_write_rigidbody_box_state_buffers", &WriteRigidbodyBoxStateBuffers, "rigidbodies"_a, "output"_a,
                    "query_triggers"_a = false,
                    "Upload flattened BoxCollider descriptors directly into resident GPU buffers")
        .def_static(
            "_query_rigidbody_box_state_buffers", &QueryRigidbodyBoxStateBuffers, "minimum"_a, "maximum"_a,
            "layer_mask"_a, "query_triggers"_a, "output"_a,
            "Query broad-phase candidates and upload their flattened BoxCollider descriptors in one native pass")
        .def_static(
            "_query_rigidbody_state_and_box_state_buffers", &QueryRigidbodyStateAndBoxStateBuffers, "minimum"_a,
            "maximum"_a, "layer_mask"_a, "query_triggers"_a, "state_output"_a, "box_output"_a,
            "Query candidates and upload authoritative rigidbody plus BoxCollider state in one compute submission")
        .def_static("apply_rigidbody_impulses", &ApplyRigidbodyImpulses, "rigidbodies"_a,
                    py::arg("linear_impulses").noconvert(), py::arg("angular_impulses").noconvert(),
                    "Apply aggregated world-space impulses; angular impulses are about each body's COM")
        .def_static("_apply_rigidbody_impulse_buffers", &ApplyRigidbodyImpulseBuffers, "rigidbodies"_a,
                    "linear_impulses"_a, "angular_impulses"_a,
                    "Read resident GPU feedback buffers together and apply their valid body prefix")
        .def_static(
            "set_contact_event_stream_enabled",
            [](bool enabled, bool includeTriggers) {
                PhysicsWorld::Instance().SetContactEventStreamEnabled(enabled, includeTriggers);
            },
            "enabled"_a, "include_triggers"_a = false,
            "Enable the resolved fixed-step contact stream for custom solvers")
        .def_static("get_contact_events", &GetContactEvents,
                    "Return resolved contact events from the most recent completed fixed step")
        .def_static(
            "set_contact_impulse_stream_enabled",
            [](bool enabled) { PhysicsWorld::Instance().SetContactImpulseStreamEnabled(enabled); }, "enabled"_a,
            "Enable the actual solved contact impulse stream for custom solvers")
        .def_static("get_contact_impulses", &GetContactImpulses,
                    "Return actual velocity-solver contact impulses from the most recent fixed step")
        .def_static(
            "query_rigidbodies_in_bounds",
            [](const glm::vec3 &minimum, const glm::vec3 &maximum, uint32_t layerMask, bool queryTriggers) {
                RequireFinite(minimum, "minimum");
                RequireFinite(maximum, "maximum");
                if (minimum.x > maximum.x || minimum.y > maximum.y || minimum.z > maximum.z)
                    throw py::value_error("minimum must not exceed maximum on any axis");
                py::list result;
                for (Rigidbody *body :
                     PhysicsWorld::Instance().QueryRigidbodiesInBounds(minimum, maximum, layerMask, queryTriggers))
                    result.append(py::cast(body, py::return_value_policy::reference));
                return result;
            },
            "minimum"_a, "maximum"_a, "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask,
            "query_triggers"_a = false,
            "Return Rigidbody broad-phase candidates whose body bounds intersect a world AABB")
        .def_property_readonly_static("body_count", [](py::object) { return PhysicsWorld::Instance().GetBodyCount(); })
        .def_property_readonly_static(
            "query_generation", [](py::object) { return PhysicsWorld::Instance().GetQueryGeneration(); },
            "Monotonic token for the currently published physics query world")
        .def_static(
            "raycast",
            [](const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance, uint32_t layerMask,
               bool queryTriggers) -> py::object {
                RequireFinite(origin, "origin");
                RequireDirectionAndDistance(direction, maxDistance);
                RaycastHit hit;
                if (PhysicsWorld::Instance().Raycast(origin, direction, maxDistance, hit, layerMask, queryTriggers)) {
                    return py::cast(hit);
                }
                return py::none();
            },
            "origin"_a, "direction"_a, "max_distance"_a = 1000.0f,
            "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask, "query_triggers"_a = true,
            "Cast a ray. Returns RaycastHit or None.")
        .def_static("raycast_batch", &RaycastBatch, "origins"_a.noconvert(), "directions"_a.noconvert(), "output"_a,
                    "max_distance"_a = 1000.0f, "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask,
                    "query_triggers"_a = true, "profile"_a = false,
                    "Cast float32 (N, 3) rays into caller-owned numeric result arrays without per-hit Python objects.")
        .def_static(
            "raycast_all",
            [](const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance, uint32_t layerMask,
               bool queryTriggers) {
                RequireFinite(origin, "origin");
                RequireDirectionAndDistance(direction, maxDistance);
                return PhysicsWorld::Instance().RaycastAll(origin, direction, maxDistance, layerMask, queryTriggers);
            },
            "origin"_a, "direction"_a, "max_distance"_a = 1000.0f,
            "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask, "query_triggers"_a = true,
            "Cast a ray and return all hits.")
        // ---- Overlap queries ----
        .def_static(
            "overlap_sphere",
            [](const glm::vec3 &center, float radius, uint32_t layerMask, bool queryTriggers) {
                RequireFinite(center, "center");
                RequirePositive(radius, "radius");
                return BorrowedColliderList(
                    PhysicsWorld::Instance().OverlapSphere(center, radius, layerMask, queryTriggers));
            },
            "center"_a, "radius"_a, "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask,
            "query_triggers"_a = true, "Find all colliders within a sphere. Returns list of Collider.")
        .def_static(
            "overlap_box",
            [](const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::quat &orientation, uint32_t layerMask,
               bool queryTriggers) {
                RequireFinite(center, "center");
                RequirePositiveExtents(halfExtents);
                RequireOrientation(orientation);
                return BorrowedColliderList(
                    PhysicsWorld::Instance().OverlapBox(center, halfExtents, orientation, layerMask, queryTriggers));
            },
            "center"_a, "half_extents"_a, "orientation"_a = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask, "query_triggers"_a = true,
            "Find all colliders within an oriented box. Returns list of Collider.")
        .def_static(
            "overlap_capsule",
            [](const glm::vec3 &point0, const glm::vec3 &point1, float radius, uint32_t layerMask, bool queryTriggers) {
                RequireFinite(point0, "point0");
                RequireFinite(point1, "point1");
                RequirePositive(radius, "radius");
                return BorrowedColliderList(
                    PhysicsWorld::Instance().OverlapCapsule(point0, point1, radius, layerMask, queryTriggers));
            },
            "point0"_a, "point1"_a, "radius"_a, "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask,
            "query_triggers"_a = true, "Find all colliders within a capsule. Returns list of Collider.")
        // ---- Shape casts ----
        .def_static(
            "sphere_cast",
            [](const glm::vec3 &origin, float radius, const glm::vec3 &direction, float maxDistance, uint32_t layerMask,
               bool queryTriggers) -> py::object {
                RequireFinite(origin, "origin");
                RequirePositive(radius, "radius");
                RequireDirectionAndDistance(direction, maxDistance);
                RaycastHit hit;
                if (PhysicsWorld::Instance().SphereCast(origin, radius, direction, maxDistance, hit, layerMask,
                                                        queryTriggers))
                    return py::cast(hit);
                return py::none();
            },
            "origin"_a, "radius"_a, "direction"_a, "max_distance"_a = 1000.0f,
            "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask, "query_triggers"_a = true,
            "Cast a sphere and return closest RaycastHit or None.")
        .def_static(
            "box_cast",
            [](const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction,
               const glm::quat &orientation, float maxDistance, uint32_t layerMask, bool queryTriggers) -> py::object {
                RequireFinite(center, "center");
                RequirePositiveExtents(halfExtents);
                RequireDirectionAndDistance(direction, maxDistance);
                RequireOrientation(orientation);
                RaycastHit hit;
                if (PhysicsWorld::Instance().BoxCast(center, halfExtents, direction, orientation, maxDistance, hit,
                                                     layerMask, queryTriggers))
                    return py::cast(hit);
                return py::none();
            },
            "center"_a, "half_extents"_a, "direction"_a, "orientation"_a = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            "max_distance"_a = 1000.0f, "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask,
            "query_triggers"_a = true, "Cast a box and return closest RaycastHit or None.")
        .def_static(
            "capsule_cast",
            [](const glm::vec3 &point0, const glm::vec3 &point1, float radius, const glm::vec3 &direction,
               float maxDistance, uint32_t layerMask, bool queryTriggers) -> py::object {
                RequireFinite(point0, "point0");
                RequireFinite(point1, "point1");
                RequirePositive(radius, "radius");
                RequireDirectionAndDistance(direction, maxDistance);
                RaycastHit hit;
                if (PhysicsWorld::Instance().CapsuleCast(point0, point1, radius, direction, maxDistance, hit, layerMask,
                                                         queryTriggers))
                    return py::cast(hit);
                return py::none();
            },
            "point0"_a, "point1"_a, "radius"_a, "direction"_a, "max_distance"_a = 1000.0f,
            "layer_mask"_a = EngineConfig::Get().defaultQueryLayerMask, "query_triggers"_a = true,
            "Cast a capsule and return closest RaycastHit or None.")
        // ---- Gravity ----
        .def_static(
            "get_gravity",
            []() -> glm::vec3 {
                auto *sys = PhysicsWorld::Instance().GetJoltSystem();
                if (!sys)
                    return EngineConfig::Get().physicsGravity;
                JPH::Vec3 g = sys->GetGravity();
                return glm::vec3(g.GetX(), g.GetY(), g.GetZ());
            },
            "Get the global gravity vector.")
        .def_static(
            "set_gravity",
            [](const glm::vec3 &g) {
                if (!std::isfinite(g.x) || !std::isfinite(g.y) || !std::isfinite(g.z))
                    throw std::invalid_argument("gravity components must be finite");
                EngineConfig::Get().physicsGravity = g;
                auto *sys = PhysicsWorld::Instance().GetJoltSystem();
                if (sys)
                    sys->SetGravity(JPH::Vec3(g.x, g.y, g.z));
            },
            "gravity"_a, "Set the global gravity vector.")
        // ---- Ignore layer collision ----
        .def_static(
            "ignore_layer_collision",
            [](int layer1, int layer2, bool ignore) {
                TagLayerManager::Instance().SetLayersCollide(layer1, layer2, !ignore);
            },
            "layer1"_a, "layer2"_a, "ignore"_a = true, "Set whether two layers should ignore collisions.")
        .def_static(
            "get_ignore_layer_collision",
            [](int layer1, int layer2) -> bool {
                return !TagLayerManager::Instance().GetLayersCollide(layer1, layer2);
            },
            "layer1"_a, "layer2"_a, "Check if two layers ignore collisions.")
        .def_static(
            "ignore_collision",
            [](Collider *collider1, Collider *collider2, bool ignore) {
                PhysicsWorld::Instance().SetColliderPairIgnored(collider1, collider2, ignore);
            },
            "collider1"_a, "collider2"_a, "ignore"_a = true,
            "Set whether one exact Collider pair should ignore contacts.")
        .def_static(
            "get_ignore_collision",
            [](Collider *collider1, Collider *collider2) {
                return PhysicsWorld::Instance().GetColliderPairIgnored(collider1, collider2);
            },
            "collider1"_a, "collider2"_a, "Check whether one exact Collider pair ignores contacts.")
        // ---- Transform sync (Unity: Physics.SyncTransforms) ----
        .def_static(
            "sync_transforms", []() { SceneManager::Instance().SyncTransforms(); },
            "Apply all pending Transform changes to the physics engine.\n"
            "Call before same-frame physics queries (raycast, overlap) when you have\n"
            "moved objects in Update and need up-to-date collision geometry.\n"
            "Unity equivalent: Physics.SyncTransforms()")
        .def_static(
            "get_actor_count", []() { return PhysicsECSStore::Instance().GetAliveActorCount(); },
            "Get the number of live PhysicsActor slots.");

    // ====================================================================
    // Register component type casters in ComponentBindingRegistry
    // ====================================================================
    // Access the singleton (defined in BindingScene.cpp, same TU linkage via static)
    // We use an extern-style approach: call the lambdas that do dynamic_cast
    // The registry is populated after RegisterSceneBindings runs.
    // To avoid header coupling, we register via a post-init hook.
}

} // namespace infernux
