/**
 * @file PhysicsWorld.cpp
 * @brief Jolt Physics integration — singleton world, body management, raycasts.
 */

// Jolt requires specific defines before including its headers
#include <Jolt/Jolt.h>

// Jolt includes (order matters)
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLockMulti.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseQuery.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/EmptyShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <limits>
#include <mutex>

#include "InfernuxJoltJobSystemAdapter.h"
#include "PhysicsContactListener.h"
#include "PhysicsLayers.h"
#include "PhysicsWorld.h"

#include "core/threading/JobSystem.h"

#include "../BoxCollider.h"
#include "../CapsuleCollider.h"
#include "../Collider.h"
#include "../Component.h"
#include "../CylinderCollider.h"
#include "../GameObject.h"
#include "../MeshCollider.h"
#include "../Rigidbody.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../SphereCollider.h"
#include "../Transform.h"
#include <core/config/EngineConfig.h>
#include <core/config/MathConstants.h>
#include <core/log/InxLog.h>
#include <core/threading/JobSystem.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <glm/gtc/constants.hpp>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace infernux
{

namespace
{

constexpr float kMinQueryDirectionLengthSq = 1e-12f;

static bool IsFinite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

static bool IsFinite(const glm::quat &value)
{
    return std::isfinite(value.w) && std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           glm::dot(value, value) > kMinQueryDirectionLengthSq;
}

static bool HasPositiveFiniteExtents(const glm::vec3 &value)
{
    return IsFinite(value) && value.x > 0.0f && value.y > 0.0f && value.z > 0.0f;
}

static JPH::Quat ToJoltQuat(const glm::quat &rotation)
{
    const glm::quat normalized = glm::normalize(rotation);
    return JPH::Quat(normalized.x, normalized.y, normalized.z, normalized.w);
}

static glm::quat CapsuleOrientation(const glm::vec3 &point0, const glm::vec3 &point1)
{
    const glm::vec3 axis = glm::normalize(point1 - point0);
    constexpr glm::vec3 up(0.0f, 1.0f, 0.0f);
    const float cosine = glm::dot(up, axis);
    if (cosine > 1.0f - 1e-6f)
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    if (cosine < -1.0f + 1e-6f)
        return glm::quat(0.0f, 1.0f, 0.0f, 0.0f);
    const glm::vec3 rotationAxis = glm::cross(up, axis);
    return glm::normalize(glm::quat(1.0f + cosine, rotationAxis.x, rotationAxis.y, rotationAxis.z));
}

static bool NormalizeQueryDirection(const glm::vec3 &direction, float maxDistance, glm::vec3 &outDirection)
{
    if (!IsFinite(direction) || !std::isfinite(maxDistance) || maxDistance <= 0.0f) {
        return false;
    }

    const float lengthSq = glm::dot(direction, direction);
    if (lengthSq <= kMinQueryDirectionLengthSq) {
        return false;
    }

    outDirection = direction / std::sqrt(lengthSq);
    return true;
}

static int MapMotionQualityMode(int quality)
{
    switch (quality) {
    case 1:
    case 2:
        return 1;
    case 0:
    case 3:
    default:
        return 0;
    }
}

static glm::mat3 InvertAllowedAngularSubspace(const glm::mat3 &inverseTensor, uint8_t allowedDofs)
{
    std::array<int, 3> axes{};
    int axisCount = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if ((allowedDofs & (1u << (axis + 3))) != 0)
            axes[axisCount++] = axis;
    }

    glm::mat3 result(0.0f);
    if (axisCount == 0)
        return result;

    float augmented[3][6]{};
    for (int row = 0; row < axisCount; ++row) {
        for (int column = 0; column < axisCount; ++column)
            augmented[row][column] = inverseTensor[axes[column]][axes[row]];
        augmented[row][axisCount + row] = 1.0f;
    }

    for (int pivotColumn = 0; pivotColumn < axisCount; ++pivotColumn) {
        int pivotRow = pivotColumn;
        for (int row = pivotColumn + 1; row < axisCount; ++row) {
            if (std::abs(augmented[row][pivotColumn]) > std::abs(augmented[pivotRow][pivotColumn]))
                pivotRow = row;
        }
        if (std::abs(augmented[pivotRow][pivotColumn]) < 1e-8f)
            throw std::logic_error("dynamic body has singular inertia on an allowed rotation axis");

        if (pivotRow != pivotColumn) {
            for (int column = 0; column < axisCount * 2; ++column)
                std::swap(augmented[pivotRow][column], augmented[pivotColumn][column]);
        }

        const float inversePivot = 1.0f / augmented[pivotColumn][pivotColumn];
        for (int column = 0; column < axisCount * 2; ++column)
            augmented[pivotColumn][column] *= inversePivot;

        for (int row = 0; row < axisCount; ++row) {
            if (row == pivotColumn)
                continue;
            const float factor = augmented[row][pivotColumn];
            for (int column = 0; column < axisCount * 2; ++column)
                augmented[row][column] -= factor * augmented[pivotColumn][column];
        }
    }

    for (int row = 0; row < axisCount; ++row) {
        for (int column = 0; column < axisCount; ++column)
            result[axes[column]][axes[row]] = augmented[row][axisCount + column];
    }
    return result;
}

template <typename Visitor>
void VisitPhysicsReceiverObjects(const Collider &collider, bool triggerEvent, Visitor &&visit)
{
    auto *leaf = collider.GetGameObject();
    auto *body = collider.GetAttachedRigidbody();
    auto *owner = body ? body->GetGameObject() : leaf;
    visit(owner);
    // Unity collision messages target the body. A trigger shape additionally
    // receives its own trigger messages on the child object.
    if (triggerEvent && collider.IsTrigger() && leaf != owner)
        visit(leaf);
}

JPH::ObjectLayer BodyLayerForMembers(const std::vector<Collider *> &members, const GameObject &owner, bool moving)
{
    int layer = -1;
    for (const auto *member : members) {
        if (!member)
            continue;
        const int memberLayer = member->GetGameObject()->GetLayer();
        if (layer >= 0 && layer != memberLayer)
            return moving ? PhysicsObjectLayers::MIXED_MOVING : PhysicsObjectLayers::MIXED_NON_MOVING;
        layer = memberLayer;
    }
    return PhysicsObjectLayers::Encode(layer < 0 ? owner.GetLayer() : layer, moving);
}

class LayerMaskObjectFilter final : public JPH::ObjectLayerFilter
{
  public:
    explicit LayerMaskObjectFilter(uint32_t layerMask) : m_layerMask(layerMask)
    {
    }

    bool ShouldCollide(JPH::ObjectLayer inLayer) const override
    {
        if (PhysicsObjectLayers::IsMixed(inLayer))
            return true;
        const int gameLayer = PhysicsObjectLayers::DecodeGameLayer(inLayer);
        return (m_layerMask & (1u << static_cast<uint32_t>(gameLayer))) != 0;
    }

  private:
    uint32_t m_layerMask;
};

class DeterministicClosestShapeRayCollector final : public JPH::CastRayCollector
{
  public:
    DeterministicClosestShapeRayCollector(const JPH::Body &body, float parentEarlyOut, bool filterRetired,
                                          uint32_t layerMask)
        : m_body(body), m_filterRetired(filterRetired), m_layerMask(layerMask)
    {
        ResetEarlyOutFraction(parentEarlyOut);
    }

    void AddHit(const JPH::RayCastResult &candidate) override
    {
        if (m_filterRetired &&
            !PhysicsWorld::Instance().AcceptsSubShapeQuery(m_body, candidate.mSubShapeID2.GetValue(), m_layerMask))
            return;
        JPH::RayCastResult result = candidate;
        result.mBodyID = m_body.GetID();
        const auto key = std::tuple(result.mFraction, result.mSubShapeID2.GetValue());
        if (!m_hadHit || key < std::tuple(m_hit.mFraction, m_hit.mSubShapeID2.GetValue())) {
            m_hit = result;
            m_hadHit = true;
        }
        // Jolt's BVHs compare with strict '<'. Keep exactly equal fractions
        // visible so sub-shape identity, rather than traversal order, resolves
        // a tie without opening the search to a meaningfully farther hit.
        UpdateEarlyOutFraction(
            std::min(GetEarlyOutFraction(), std::nextafter(result.mFraction, std::numeric_limits<float>::infinity())));
    }

    [[nodiscard]] bool HadHit() const noexcept
    {
        return m_hadHit;
    }

    [[nodiscard]] const JPH::RayCastResult &Hit() const noexcept
    {
        return m_hit;
    }

  private:
    const JPH::Body &m_body;
    bool m_filterRetired;
    uint32_t m_layerMask;
    JPH::RayCastResult m_hit{};
    bool m_hadHit = false;
};

struct RaycastPublishedBodyCache
{
    uint64_t generation = std::numeric_limits<uint64_t>::max();
    uint32_t bodyId = 0xFFFFFFFFu;
    const JPH::Body *body = nullptr;
    std::optional<JPH::TransformedShape> transformedShape;
};

// Closest-hit batches do not need Jolt's general multi-body hit collector.
// Drive the broad phase directly and use each shape's optimized single-hit
// path instead. The published query epoch makes the no-lock body view stable;
// the one-ULP early-out margin keeps equal-distance bodies visible so the
// numeric body/sub-shape tie break remains deterministic.
class DeterministicClosestRayBodyCollector final : public JPH::RayCastBodyCollector
{
  public:
    DeterministicClosestRayBodyCollector(const JPH::RRayCast &ray, const JPH::BodyLockInterface &bodyLocks,
                                         RaycastBatchProfile *profile = nullptr,
                                         RaycastPublishedBodyCache *publishedCache = nullptr,
                                         uint64_t queryGeneration = 0, uint32_t layerMask = 0xFFFFFFFFu)
        : m_ray(ray), m_bodyLocks(bodyLocks), m_profile(profile), m_publishedCache(publishedCache),
          m_queryGeneration(queryGeneration), m_layerMask(layerMask)
    {
    }

    void Reset() override
    {
        JPH::RayCastBodyCollector::Reset();
        m_hit = {};
        m_hitBody = nullptr;
        m_hadHit = false;
    }

    void AddHit(const JPH::BroadPhaseCastResult &broadphaseHit) override
    {
        if (m_profile)
            ++m_profile->broadphaseCandidates;

        const uint32_t bodyId = broadphaseHit.mBodyID.GetIndexAndSequenceNumber();
        if (m_publishedCache && m_publishedCache->generation == m_queryGeneration &&
            m_publishedCache->bodyId == bodyId && m_publishedCache->body &&
            m_publishedCache->transformedShape.has_value()) {
            AddBody(*m_publishedCache->body, *m_publishedCache->transformedShape);
            return;
        }

        JPH::BodyLockRead lock(m_bodyLocks, broadphaseHit.mBodyID);
        if (!lock.SucceededAndIsInBroadPhase())
            return;
        const JPH::Body &body = lock.GetBody();
        const JPH::TransformedShape transformedShape = body.GetTransformedShape();
        if (m_publishedCache) {
            m_publishedCache->generation = m_queryGeneration;
            m_publishedCache->bodyId = bodyId;
            m_publishedCache->body = &body;
            m_publishedCache->transformedShape = transformedShape;
        }
        AddBody(body, transformedShape);
    }

  private:
    void AddBody(const JPH::Body &body, const JPH::TransformedShape &transformedShape)
    {
        const bool filterRetired =
            PhysicsWorld::Instance().HasRetiredShapeMembers() || PhysicsObjectLayers::IsMixed(body.GetObjectLayer());
        if (filterRetired && !PhysicsWorld::Instance().FindColliderByBodyId(body.GetID().GetIndexAndSequenceNumber()))
            return;
        JPH::RayCastResult result;
        result.mFraction = GetEarlyOutFraction();
        const uint64_t narrowStartNs = m_profile ? ProfileNowNs() : 0;
        bool hit = false;
        const JPH::EShapeType shapeType = body.GetShape()->GetType();
        if (shapeType == JPH::EShapeType::Convex && body.GetShape()->GetSubType() != JPH::EShapeSubType::Triangle) {
            // The single-result path is substantially cheaper for ordinary
            // solid colliders and its inside-volume behavior is the public
            // contract expected for convex shapes.
            hit = transformedShape.CastRay(m_ray, result);
        } else if (shapeType == JPH::EShapeType::Mesh && transformedShape.GetShapeScale().GetX() > 0.0f &&
                   transformedShape.GetShapeScale().GetY() > 0.0f && transformedShape.GetShapeScale().GetZ() > 0.0f) {
            JPH::RayCast localRay(m_ray.Transformed(transformedShape.GetInverseCenterOfMassTransform()));
            const JPH::Vec3 inverseScale = transformedShape.GetShapeScale().Reciprocal();
            localRay.mOrigin *= inverseScale;
            localRay.mDirection *= inverseScale;
            hit = static_cast<const JPH::MeshShape *>(body.GetShape())
                      ->CastRayIgnoreBackFaces(localRay, JPH::SubShapeIDCreator(), result);
            if (hit)
                result.mBodyID = body.GetID();
        } else {
            // Jolt's single-result shape path reports triangle back faces and
            // cannot continue to a later valid front face. Non-convex,
            // compound and decorated shapes therefore keep RayCastSettings'
            // one-sided semantics while still collecting only the closest
            // valid hit for this broadphase candidate.
            DeterministicClosestShapeRayCollector closest(body, GetEarlyOutFraction(), filterRetired, m_layerMask);
            transformedShape.CastRay(m_ray, JPH::RayCastSettings(), closest);
            if (closest.HadHit()) {
                result = closest.Hit();
                hit = true;
            }
        }
        if (m_profile) {
            m_profile->narrowPhaseCpuNs += ProfileNowNs() - narrowStartNs;
            if (hit)
                ++m_profile->narrowphaseHits;
        }
        if (!hit || (filterRetired &&
                     !PhysicsWorld::Instance().AcceptsSubShapeQuery(body, result.mSubShapeID2.GetValue(), m_layerMask)))
            return;

        const auto key =
            std::tuple(result.mFraction, result.mBodyID.GetIndexAndSequenceNumber(), result.mSubShapeID2.GetValue());
        if (!m_hadHit || key < std::tuple(m_hit.mFraction, m_hit.mBodyID.GetIndexAndSequenceNumber(),
                                          m_hit.mSubShapeID2.GetValue())) {
            m_hit = result;
            m_hitBody = &body;
            m_hadHit = true;
            UpdateEarlyOutFraction(std::min(GetEarlyOutFraction(),
                                            std::nextafter(result.mFraction, std::numeric_limits<float>::infinity())));
        }
    }

  public:
    [[nodiscard]] bool HadHit() const noexcept
    {
        return m_hadHit;
    }

    [[nodiscard]] const JPH::RayCastResult &Hit() const noexcept
    {
        return m_hit;
    }

    [[nodiscard]] const JPH::Body &HitBody() const noexcept
    {
        return *m_hitBody;
    }

    static uint64_t ProfileNowNs() noexcept
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }

    JPH::RRayCast m_ray;
    const JPH::BodyLockInterface &m_bodyLocks;
    JPH::RayCastResult m_hit{};
    const JPH::Body *m_hitBody = nullptr;
    RaycastBatchProfile *m_profile = nullptr;
    RaycastPublishedBodyCache *m_publishedCache = nullptr;
    uint64_t m_queryGeneration = 0;
    uint32_t m_layerMask;
    bool m_hadHit = false;
};

class FilteredCCDCollector final : public JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector>
{
  public:
    FilteredCCDCollector(const JPH::Body &source, const JPH::Body &target) : m_source(source), m_target(target)
    {
    }

    void AddHit(const JPH::ShapeCastResult &result) override
    {
        if (PhysicsWorld::Instance().AcceptsCCDCandidate(m_source, m_target, result.mSubShapeID1.GetValue(),
                                                         result.mSubShapeID2.GetValue()))
            JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector>::AddHit(result);
    }

  private:
    const JPH::Body &m_source;
    const JPH::Body &m_target;
};

class ProfiledAllHitRayCollector final : public JPH::AllHitCollisionCollector<JPH::CastRayCollector>
{
  public:
    explicit ProfiledAllHitRayCollector(RaycastBatchProfile *profile) : m_profile(profile)
    {
    }

    void OnBody(const JPH::Body &) override
    {
        ++m_profile->broadphaseCandidates;
        m_narrowStartNs = ProfileNowNs();
    }

    void OnBodyEnd() override
    {
        if (m_narrowStartNs != 0) {
            m_profile->narrowPhaseCpuNs += ProfileNowNs() - m_narrowStartNs;
            m_narrowStartNs = 0;
        }
    }

    void AddHit(const JPH::RayCastResult &result) override
    {
        ++m_profile->narrowphaseHits;
        JPH::AllHitCollisionCollector<JPH::CastRayCollector>::AddHit(result);
    }

  private:
    static uint64_t ProfileNowNs() noexcept
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }

    RaycastBatchProfile *m_profile;
    uint64_t m_narrowStartNs = 0;
};

static JPH::RefConst<JPH::Shape> BuildShapeForColliderSet(const PhysicsActorData &actor, const Collider *exclude,
                                                          size_t *outShapeCount = nullptr,
                                                          std::vector<Collider *> *outMembers = nullptr)
{
    auto *go = actor.owner;
    std::vector<std::pair<Collider *, JPH::RefConst<JPH::Shape>>> childShapes;
    const auto &colliders = actor.colliders;
    childShapes.reserve(colliders.size());
    bool complete = true;

    for (auto *col : colliders) {
        if (!col || col == exclude || col->IsBeingDestroyed() || !col->IsEnabled() ||
            !col->GetGameObject()->IsActiveInHierarchy()) {
            continue;
        }

        auto child = col->CreateJoltShape();
        if (child) {
            if (child->GetSubType() == JPH::EShapeSubType::Empty)
                continue; // Empty children have no geometry or compound mass.
            if (col->GetGameObject() != go) {
                const auto *actorTransform = go->GetTransform();
                const auto *childTransform = col->GetGameObject()->GetTransform();
                const auto inverseRotation = glm::inverse(actorTransform->GetWorldRotation());
                const auto position = inverseRotation * (childTransform->GetPosition() - actorTransform->GetPosition());
                const auto rotation = glm::normalize(inverseRotation * childTransform->GetWorldRotation());
                // Each primitive already bakes world scale. Only the rigid
                // relative pose belongs here; a full inverse matrix scales twice.
                child =
                    new JPH::RotatedTranslatedShape(JPH::Vec3(position.x, position.y, position.z),
                                                    JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), child);
            }
            childShapes.emplace_back(col, std::move(child));
        } else {
            // A pending or rejected child must not silently disappear from
            // a replacement compound. Visit every child to start its cook.
            complete = false;
        }
    }

    if (!complete) {
        if (outShapeCount)
            *outShapeCount = 0;
        return nullptr;
    }

    if (outShapeCount)
        *outShapeCount = childShapes.size();
    if (outMembers) {
        outMembers->reserve(childShapes.size());
        for (const auto &child : childShapes)
            outMembers->push_back(child.first);
    }

    JPH::RefConst<JPH::Shape> shape;
    if (childShapes.empty()) {
        shape = new JPH::EmptyShape();
    } else if (childShapes.size() == 1) {
        shape = childShapes.front().second;
    } else {
        JPH::StaticCompoundShapeSettings compoundSettings;
        for (size_t i = 0; i < childShapes.size(); ++i) {
            compoundSettings.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), childShapes[i].second,
                                      static_cast<uint32_t>(i));
        }
        auto result = compoundSettings.Create();
        if (result.HasError())
            return nullptr;
        shape = result.Get();
    }
    if (auto *body = actor.rigidbody; body && body->IsEnabled()) {
        if (const auto &properties = body->GetAuthoredMassProperties()) {
            const auto center = properties->center * go->GetTransform()->GetWorldScale();
            const JPH::Vec3 offset = JPH::Vec3(center.x, center.y, center.z) - shape->GetCenterOfMass();
            shape = new JPH::OffsetCenterOfMassShape(shape, offset);
        }
    }
    return shape;
}

static void PublishColliderCenters(const PhysicsActorData &actor)
{
    auto &store = PhysicsECSStore::Instance();
    for (auto *collider : actor.colliders)
        store.GetCollider(collider->GetECSHandle()).publishedRigidCenter = collider->GetCenterInRigidSpace();
}

static std::optional<uint32_t> CompoundMemberIndex(const JPH::Shape *shape, uint32_t subShapeIdValue)
{
    // Decorators (including explicit COM) preserve the child's subshape bits.
    // Queries and contacts must resolve the same published compound member.
    while (shape && shape->GetType() == JPH::EShapeType::Decorated)
        shape = static_cast<const JPH::DecoratedShape *>(shape)->GetInnerShape();
    if (!shape || shape->GetType() != JPH::EShapeType::Compound)
        return std::nullopt;
    const auto *compound = static_cast<const JPH::CompoundShape *>(shape);
    JPH::SubShapeID subShapeId;
    subShapeId.SetValue(subShapeIdValue);
    if (!compound->IsSubShapeIDValid(subShapeId))
        return std::nullopt;
    JPH::SubShapeID remainder;
    return compound->GetCompoundUserData(compound->GetSubShapeIndexFromID(subShapeId, remainder));
}

static JPH::Mat44 ScaledAuthoredInertia(const RigidbodyMassProperties &properties, const glm::vec3 &scale)
{
    const glm::dmat3 rotation = glm::mat3_cast(glm::dquat(properties.rotation));
    glm::dmat3 diagonal(0.0);
    for (int i = 0; i < 3; ++i)
        diagonal[i][i] = properties.moments[i];
    const glm::dmat3 inertia = rotation * diagonal * glm::transpose(rotation);
    const double halfTrace = .5 * (inertia[0][0] + inertia[1][1] + inertia[2][2]);
    glm::dmat3 secondMoment(0.0);
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            secondMoment[column][row] =
                ((column == row ? halfTrace : 0.0) - inertia[column][row]) * double(scale[column]) * double(scale[row]);
    const double trace = secondMoment[0][0] + secondMoment[1][1] + secondMoment[2][2];
    JPH::Mat44 result = JPH::Mat44::sIdentity();
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            result(row, column) = float((column == row ? trace : 0.0) - secondMoment[column][row]);
    // Constant authored mass: do not multiply by volume and divide it out.
    // This also retains valid planar inertia at a single zero scale axis.
    return result;
}

struct MassConfiguration
{
    float mass;
    JPH::Vec3 inverse;
    JPH::Quat axes;
};

static MassConfiguration PrepareMassConfiguration(const JPH::MassProperties &properties)
{
    const auto diagonal = properties.mInertia.GetDiagonal3();
    const float magnitude = std::max({diagonal.GetX(), diagonal.GetY(), diagonal.GetZ()});
    if (!std::isfinite(magnitude) || magnitude <= 0.0f)
        throw std::invalid_argument("rigidbody inertia is not finite positive at the current scale");
    auto normalized = properties;
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            normalized.mInertia(row, column) = float(double(properties.mInertia(row, column)) / magnitude);
    JPH::Mat44 axes;
    JPH::Vec3 moments;
    if (!normalized.DecomposePrincipalMomentsOfInertia(axes, moments))
        throw std::invalid_argument("rigidbody inertia cannot be decomposed at the current scale");
    JPH::Vec3 inverse = JPH::Vec3::sZero();
    for (int i = 0; i < 3; ++i) {
        const double moment = double(moments[i]) * magnitude;
        const float value = float(1.0 / moment);
        if (moment <= 0.0 || !std::isfinite(value) || value <= 0.0f)
            throw std::invalid_argument("rigidbody inverse inertia is not representable at the current scale");
        inverse.SetComponent(i, value);
    }
    return {properties.mMass, inverse, axes.GetQuaternion()};
}

static void ApplyMassConfiguration(JPH::MotionProperties &motion, JPH::EAllowedDOFs allowed,
                                   const MassConfiguration &prepared)
{
    JPH::MassProperties configuration;
    configuration.mMass = prepared.mass;
    configuration.mInertia = JPH::Mat44::sIdentity();
    // Jolt's generic setter substitutes a unit sphere for small inertia.
    // Configure mass/DOFs, then install the validated physical eigensystem
    // under the same body lock. No synthetic tensor is externally published.
    motion.SetMassProperties(allowed, configuration);
    if ((uint32_t(allowed) & 0x38U) != 0)
        motion.SetInverseInertia(prepared.inverse, prepared.axes);
}

static MassConfiguration BodyMassConfiguration(const JPH::Shape *shape, float mass,
                                               const RigidbodyMassProperties *properties, const glm::vec3 &scale)
{
    JPH::MassProperties value;
    if (properties) {
        value.mMass = mass;
        value.mInertia = ScaledAuthoredInertia(*properties, scale);
    } else {
        value = shape->GetMassProperties();
        value.ScaleToMass(mass);
    }
    return PrepareMassConfiguration(value);
}

} // namespace

// ============================================================================
// Jolt trace / assert callbacks (required by Jolt)
// ============================================================================

static void JoltTraceImpl(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    INXLOG_DEBUG("[Jolt] ", buffer);
}

#ifdef JPH_ENABLE_ASSERTS
static bool JoltAssertFailed(const char *expression, const char *message, const char *file, unsigned int line)
{
    INXLOG_ERROR("[Jolt Assert] ", file, ":", line, " – ", expression, " – ", message ? message : "");
    // Returning true triggers __debugbreak() inside Jolt's assert macro.
    // In debug builds (debugger attached), this is desirable so we can inspect the state.
    // In release / no-debugger builds, return false to log-and-continue.
#ifdef NDEBUG
    return false;
#else
    return true;
#endif
}
#endif

// ============================================================================
// Layer helpers (must outlive PhysicsSystem)
// ============================================================================

struct PhysicsWorld::LayerInterfaces
{
    BPLayerInterface bpInterface;
    ObjectVsBPLayerFilter objVsBpFilter;
    ObjectLayerPairFilter objPairFilter;
};

// ============================================================================
// Singleton
// ============================================================================

uint32_t PhysicsWorld::GetDefaultQueryLayerMask()
{
    return EngineConfig::Get().defaultQueryLayerMask;
}

PhysicsWorld &PhysicsWorld::Instance()
{
    // Intentionally leaked: Shutdown() is called explicitly from
    // Infernux::Cleanup(); a static destructor would re-run it after
    // dependent singletons may already be gone.
    static PhysicsWorld *instance = new PhysicsWorld();
    return *instance;
}

PhysicsWorld::~PhysicsWorld()
{
    Shutdown();
}

// ============================================================================
// Init / Shutdown
// ============================================================================

void PhysicsWorld::RequireOwnerThread() const
{
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    if (!m_initialized || std::this_thread::get_id() != m_ownerThreadId)
        throw std::logic_error("joint batch IO requires the active physics owner thread");
}

void PhysicsWorld::Initialize()
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (m_initialized)
        return;

    // Initialize is an engine-owner operation. Batch queries issued from any
    // other thread must never walk SceneManager authoring state.
    m_ownerThreadId = std::this_thread::get_id();

    // Register Jolt allocation hooks (use default malloc)
    JPH::RegisterDefaultAllocator();

    // Install trace / assert callbacks
    JPH::Trace = JoltTraceImpl;
#ifdef JPH_ENABLE_ASSERTS
    JPH::AssertFailed = JoltAssertFailed;
#endif

    // Create factory & register types
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    // -------------------------------------------------------------------------
    // TempAllocatorImpl — shared stack pool for ALL Jolt worker threads.
    //
    // IMPORTANT: this is NOT a per-thread allocator.  All threads push/pop
    // from the same pool concurrently during broadphase + narrowphase +
    // constraint solver.  Running out of space calls std::abort() immediately,
    // so the pool must be sized for PEAK simultaneous usage across all threads.
    //
    // Sizing guide (bodies in scene → recommended pool):
    //   < 256  bodies : 32 MB  is comfortable
    //   < 1024 bodies : 64 MB  recommended
    //   < 4096 bodies : 128 MB recommended
    //   ≥ 4096 bodies : 256 MB or more
    //
    // Increasing this value has negligible real memory cost because the OS
    // only commits pages that are actually touched (virtual memory).
    // -------------------------------------------------------------------------
    auto &cfg = EngineConfig::Get();

    if (!JobSystem::IsAvailable()) {
        throw std::logic_error("PhysicsWorld requires an initialized Infernux JobSystem");
    }
    JobSystem::Get().SetDomainConcurrency(JobDomain::Physics, cfg.physicsMaxConcurrency);

    m_tempAllocator = std::make_unique<JPH::TempAllocatorImpl>(cfg.physicsTempAllocatorSize);

    m_jobSystem = std::make_unique<InfernuxJoltJobSystemAdapter>(cfg.physicsMaxJobs, cfg.physicsMaxBarriers,
                                                                 cfg.physicsMaxConcurrency);

    // Layer interfaces
    m_layers = std::make_unique<LayerInterfaces>();

    // Physics system
    m_physicsSystem = std::make_unique<JPH::PhysicsSystem>();
    m_physicsSystem->Init(cfg.physicsMaxBodies, 0, cfg.physicsMaxBodyPairs, cfg.physicsMaxContactConstraints,
                          m_layers->bpInterface, m_layers->objVsBpFilter, m_layers->objPairFilter);
    // Preserve a stream request made before the world was initialized.
    m_physicsSystem->SetRecordAppliedContactImpulses(m_contactImpulseStreamEnabled);

    // Tune physics settings for thin-body stability and precision.
    //  - Penetration slop: 2 mm (default 20 mm). Min BoxShape thickness is
    //    2 × kMinHalfExtent = 0.02 m; 2 mm slop keeps sinking at 10 %.
    //  - Speculative contact: 10 mm (default 20 mm).  Tighter for thin shells.
    //  - Position solver: 3 iterations (default 2). Better stacking stability.
    //  - LinearCast max penetration: 10 % of inner radius (default 25 %).
    //  - Baumgarte: 0.15 (default 0.2). Softer correction prevents violent
    //    pop-out when thin bodies briefly penetrate a surface.
    //  - Max penetration distance: 50 mm (default 200 mm). Limits per-step
    //    correction so thin-body overlaps resolve gradually.
    //  - LinearCast threshold: 50 % (default 75 %). Triggers CCD earlier,
    //    critical for thin bodies whose inner radius is very small.
    JPH::PhysicsSettings settings = m_physicsSystem->GetPhysicsSettings();
    settings.mPenetrationSlop = cfg.physicsPenetrationSlop;
    settings.mSpeculativeContactDistance = cfg.physicsSpeculativeContactDistance;
    settings.mNumVelocitySteps = cfg.physicsVelocitySteps;
    settings.mNumPositionSteps = cfg.physicsPositionSteps;
    settings.mLinearCastMaxPenetration = cfg.physicsLinearCastMaxPenetration;
    settings.mBaumgarte = cfg.physicsBaumgarte;
    settings.mMaxPenetrationDistance = cfg.physicsMaxPenetrationDistance;
    settings.mLinearCastThreshold = cfg.physicsLinearCastThreshold;
    settings.mMinVelocityForRestitution = cfg.physicsMinVelocityForRestitution;
    settings.mTimeBeforeSleep = cfg.physicsTimeBeforeSleep;
    settings.mPointVelocitySleepThreshold = cfg.physicsPointVelocitySleepThreshold;
    m_physicsSystem->SetPhysicsSettings(settings);

    // Gravity
    m_physicsSystem->SetGravity(JPH::Vec3(cfg.physicsGravity.x, cfg.physicsGravity.y, cfg.physicsGravity.z));

    // Install contact listener for collision/trigger callbacks
    m_contactListener = std::make_unique<InxContactListener>();
    m_physicsSystem->SetContactListener(m_contactListener.get());

    // Warm up the broadphase through the same Physics domain used by real
    // simulation steps. The adapter never owns a worker thread.
    {
        auto warmupGroup = JobSystem::Get().CreateTaskGroup(JobDomain::Physics, JobPriority::Critical);
        m_jobSystem->BeginFrame(warmupGroup);
        m_physicsSystem->OptimizeBroadPhase();
        m_jobSystem->EndFrame();
        warmupGroup.Close();
        JobSystem::Get().Wait(warmupGroup);
    }

    m_initialized = true;
}

void PhysicsWorld::Shutdown()
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized)
        return;

    while (!m_ragdolls.empty())
        DestroyRagdollAssembly(m_ragdolls.begin()->first);

    // Constraints must leave the solver before either referenced body dies.
    while (!m_constraints.empty())
        DestroyConstraint(m_constraints.begin()->first);

    // Step 1: drop contact pair tracking before any body dies, so callbacks
    // racing the teardown can't dereference freed Collider* pointers.
    if (m_contactListener)
        m_contactListener->ClearAll();

    // Step 1b: destroy any bodies that still exist. Scenes should already be
    // gone (SceneManager::Shutdown runs first in Infernux::Cleanup), so a
    // non-empty map here indicates a teardown-ordering bug — sweep it anyway
    // so Jolt's PhysicsSystem is destroyed with zero live bodies.
    if (m_physicsSystem && !m_bodies.empty()) {
        INXLOG_WARN("PhysicsWorld: ", m_bodies.size(),
                    " bodies still alive at shutdown — destroying them (check teardown order).");
        JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
        for (const auto &[id, collider] : m_bodies) {
            (void)collider;
            JPH::BodyID bodyId(id);
            if (bi.IsAdded(bodyId))
                bi.RemoveBody(bodyId);
            bi.DestroyBody(bodyId);
        }
    }
    m_bodies.clear();
    m_retiredShapeBodyCount = 0;
    m_poseReadbackBodyIds.clear();
    m_staticContinuousBodyIds.clear();
    m_continuousBodyIds.clear();
    m_kinematicMoveStates.clear();
    m_nextConstraintId = 1;
    m_lastDynamicCCDSplitCount = 0;
    m_previousSolverStepDuration = 0.0f;
    m_ownerThreadId = {};

    // Step 2: tear down subsystems in dependency order (newest first).
    m_contactListener.reset();
    m_physicsSystem.reset();
    if (m_jobSystem) {
        m_jobSystem->Shutdown();
        m_jobSystem.reset();
    }
    if (JobSystem::IsAvailable()) {
        JobSystem::Get().SetDomainConcurrency(JobDomain::Physics, 0);
    }
    m_tempAllocator.reset();
    m_layers.reset();

    // Step 3: drop Jolt's process-global state. Safe because every owned
    // unique_ptr above has been released, so no Jolt object outlives the
    // factory.
    JPH::UnregisterTypes();
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;

    m_initialized = false;
}

// ============================================================================
// Step
// ============================================================================

float PhysicsWorld::FindEarliestDynamicCCDFraction(float deltaTime) const
{
    if (m_continuousBodyIds.empty())
        return 1.0f;

    struct BodySnapshot
    {
        JPH::BodyID id;
        const JPH::Body *body;
        JPH::ObjectLayer layer;
        JPH::CollisionGroup collisionGroup;
        JPH::RefConst<JPH::Shape> shape;
        JPH::RMat44 centerOfMassTransform;
        JPH::TransformedShape transformedShape;
        JPH::AABox sweptBounds;
        JPH::Vec3 deltaPosition;
        bool continuousDynamic = false;
    };

    JPH::BodyIDVector activeBodyIds;
    m_physicsSystem->GetActiveBodies(JPH::EBodyType::RigidBody, activeBodyIds);

    std::vector<BodySnapshot> bodies;
    bodies.reserve(activeBodyIds.size());
    for (const JPH::BodyID id : activeBodyIds) {
        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), id);
        if (!lock.Succeeded())
            continue;

        const JPH::Body &body = lock.GetBody();
        if (body.IsStatic() || body.IsSensor() || body.GetMotionPropertiesUnchecked() == nullptr)
            continue;

        BodySnapshot snapshot;
        snapshot.id = id;
        // The owner holds the query write epoch across this complete scan.
        snapshot.body = &body;
        snapshot.layer = body.GetObjectLayer();
        snapshot.collisionGroup = body.GetCollisionGroup();
        snapshot.shape = body.GetShape();
        snapshot.centerOfMassTransform = body.GetCenterOfMassTransform();
        snapshot.transformedShape = body.GetTransformedShape();
        snapshot.deltaPosition = deltaTime * body.GetLinearVelocity();
        snapshot.sweptBounds = body.GetWorldSpaceBounds();
        JPH::AABox endBounds = snapshot.sweptBounds;
        endBounds.Translate(snapshot.deltaPosition);
        snapshot.sweptBounds.Encapsulate(endBounds);
        snapshot.continuousDynamic =
            body.IsDynamic() && body.GetMotionProperties()->GetMotionQuality() == JPH::EMotionQuality::LinearCast;
        bodies.push_back(std::move(snapshot));
    }

    JPH::ShapeCastSettings castSettings;
    castSettings.mUseShrunkenShapeAndConvexRadius = true;
    castSettings.mBackFaceModeTriangles = JPH::EBackFaceMode::IgnoreBackFaces;
    castSettings.mBackFaceModeConvex = JPH::EBackFaceMode::IgnoreBackFaces;

    float earliestFraction = 1.0f;
    for (size_t sourceIndex = 0; sourceIndex < bodies.size(); ++sourceIndex) {
        const BodySnapshot &source = bodies[sourceIndex];
        if (!source.continuousDynamic)
            continue;

        for (size_t targetIndex = 0; targetIndex < bodies.size(); ++targetIndex) {
            if (sourceIndex == targetIndex)
                continue;

            const BodySnapshot &target = bodies[targetIndex];
            if (target.continuousDynamic && source.id > target.id)
                continue;
            if (!source.sweptBounds.Overlaps(target.sweptBounds))
                continue;
            if (!m_layers->objPairFilter.ShouldCollide(source.layer, target.layer) ||
                !source.collisionGroup.CanCollide(target.collisionGroup))
                continue;

            const JPH::Vec3 relativeMotion = source.deltaPosition - target.deltaPosition;
            if (relativeMotion.LengthSq() <= 1e-12f)
                continue;

            JPH::RShapeCast cast(source.shape, JPH::Vec3::sOne(), source.centerOfMassTransform, relativeMotion);
            FilteredCCDCollector collector(*source.body, *target.body);
            JPH::ShapeFilter shapeFilter;
            shapeFilter.mBodyID2 = target.id;
            target.transformedShape.CastShape(cast, castSettings, source.centerOfMassTransform.GetTranslation(),
                                              collector, shapeFilter);
            if (!collector.HadHit())
                continue;

            const float fraction = collector.mHit.mFraction;
            if (fraction > 1e-4f && fraction < earliestFraction)
                earliestFraction = fraction;
        }
    }

    return earliestFraction;
}

float PhysicsWorld::FindEarliestStaticCCDFraction(float deltaTime) const
{
    if (m_staticContinuousBodyIds.empty())
        return 1.0f;

    JPH::ShapeCastSettings castSettings;
    castSettings.mUseShrunkenShapeAndConvexRadius = true;
    castSettings.mBackFaceModeTriangles = JPH::EBackFaceMode::IgnoreBackFaces;
    castSettings.mBackFaceModeConvex = JPH::EBackFaceMode::IgnoreBackFaces;

    float earliestFraction = 1.0f;
    for (const uint32_t sourceRawId : m_staticContinuousBodyIds) {
        const JPH::BodyID sourceId(sourceRawId);
        JPH::BodyLockRead sourceLock(m_physicsSystem->GetBodyLockInterface(), sourceId);
        if (!sourceLock.Succeeded())
            continue;
        const JPH::Body &source = sourceLock.GetBody();
        if (!source.IsDynamic() || source.IsSensor() || !source.IsActive())
            continue;

        const JPH::Vec3 motion = deltaTime * source.GetLinearVelocity();
        if (motion.LengthSq() <= 1e-12f)
            continue;

        JPH::AABox sweptBounds = source.GetWorldSpaceBounds();
        JPH::AABox endBounds = sweptBounds;
        endBounds.Translate(motion);
        sweptBounds.Encapsulate(endBounds);
        const JPH::RMat44 sourceTransform = source.GetCenterOfMassTransform();
        const JPH::RShapeCast cast(source.GetShape(), JPH::Vec3::sOne(), sourceTransform, motion);

        for (const auto &[targetRawId, collider] : m_bodies) {
            (void)collider;
            if (targetRawId == sourceRawId)
                continue;
            const JPH::BodyID targetId(targetRawId);
            JPH::BodyLockRead targetLock(m_physicsSystem->GetBodyLockInterface(), targetId);
            if (!targetLock.Succeeded())
                continue;
            const JPH::Body &target = targetLock.GetBody();
            if (!target.IsStatic() || target.IsSensor() || !sweptBounds.Overlaps(target.GetWorldSpaceBounds()))
                continue;
            if (!m_layers->objPairFilter.ShouldCollide(source.GetObjectLayer(), target.GetObjectLayer()) ||
                !source.GetCollisionGroup().CanCollide(target.GetCollisionGroup()))
                continue;

            FilteredCCDCollector collector(source, target);
            JPH::ShapeFilter shapeFilter;
            shapeFilter.mBodyID2 = targetId;
            target.GetTransformedShape().CastShape(cast, castSettings, sourceTransform.GetTranslation(), collector,
                                                   shapeFilter);
            if (collector.HadHit() && collector.mHit.mFraction > 1e-4f)
                earliestFraction = std::min(earliestFraction, collector.mHit.mFraction);
        }
    }
    return earliestFraction;
}

void PhysicsWorld::SettleKinematicMoves()
{
    if (m_kinematicMoveStates.empty())
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    for (auto it = m_kinematicMoveStates.begin(); it != m_kinematicMoveStates.end();) {
        KinematicMoveState &state = it->second;
        if (state.movedThisStep) {
            // A target arrived since the previous step — let this step
            // integrate it, then re-evaluate.
            state.movedThisStep = false;
            ++it;
            continue;
        }

        const JPH::BodyID id(it->first);

        // The body arrived at its last target and no new target came in.
        // Without this, the MoveKinematic velocity persists and the body
        // glides away from its Transform forever.
        if (state.idleSteps == 0)
            bi.SetLinearAndAngularVelocity(id, JPH::Vec3::sZero(), JPH::Vec3::sZero());
        ++state.idleSteps;

        if (!state.restoreStatic) {
            // Genuine kinematic Rigidbody — stopping it is all we owe it.
            it = m_kinematicMoveStates.erase(it);
            continue;
        }

        // Temporarily-kinematic static: keep it kinematic for a few idle
        // steps, since a gizmo drag does not deliver a new pose on every
        // fixed step, then restore. SetMotionType(Static) deactivates the
        // body and zeroes its velocity inside Jolt.
        constexpr int kRestoreAfterIdleSteps = 3;
        if (state.idleSteps >= kRestoreAfterIdleSteps) {
            bool restore = true;
            if (auto found = m_bodies.find(it->first); found != m_bodies.end()) {
                const Rigidbody *rb = found->second.rigidbody;
                restore = (rb == nullptr || !rb->IsEnabled());
            }
            if (restore)
                bi.SetMotionType(id, JPH::EMotionType::Static, JPH::EActivation::DontActivate);
            it = m_kinematicMoveStates.erase(it);
            continue;
        }
        ++it;
    }
}

void PhysicsWorld::Step(float deltaTime)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    m_poseReadbackBodyIds.clear();
    m_contactImpulses.clear();
    m_lastDynamicCCDSplitCount = 0;
    if (!m_initialized)
        return;

    SettleKinematicMoves();

    JPH::BodyIDVector activeBefore;
    m_physicsSystem->GetActiveBodies(JPH::EBodyType::RigidBody, activeBefore);

    if (m_contactListener) {
        uint8_t eventInterestMask = 0;
        std::unordered_set<GameObject *> visitedObjects;
        visitedObjects.reserve(m_bodies.size());
        const auto inspectReceiver = [&](GameObject *gameObject) {
            if (!gameObject || !gameObject->IsActiveInHierarchy() || !visitedObjects.insert(gameObject).second)
                return;
            for (const auto &component : gameObject->GetAllComponents()) {
                if (!component || !component->WantsPhysicsCallbacks())
                    continue;
                if (component->WantsCollisionEnterCallbacks())
                    eventInterestMask |= CollisionEnterInterest;
                if (component->WantsCollisionStayCallbacks())
                    eventInterestMask |= CollisionStayInterest;
                if (component->WantsCollisionExitCallbacks())
                    eventInterestMask |= CollisionExitInterest;
                if (component->WantsTriggerEnterCallbacks())
                    eventInterestMask |= TriggerEnterInterest;
                if (component->WantsTriggerStayCallbacks())
                    eventInterestMask |= TriggerStayInterest;
                if (component->WantsTriggerExitCallbacks())
                    eventInterestMask |= TriggerExitInterest;
            }
        };
        for (const auto &[bodyId, record] : m_bodies) {
            (void)bodyId;
            for (auto *collider : record.members)
                if (collider)
                    VisitPhysicsReceiverObjects(*collider, true, inspectReceiver);
        }
        if (m_contactEventStreamEnabled) {
            eventInterestMask |= CollisionEnterInterest | CollisionStayInterest | CollisionExitInterest;
            if (m_contactEventStreamIncludeTriggers)
                eventInterestMask |= TriggerEnterInterest | TriggerStayInterest | TriggerExitInterest;
        }
        m_contactListener->SetEventInterestMask(eventInterestMask);
        m_contactListener->PreStep();
    }

    // Jolt's LinearCast narrow phase handles relative target motion, but its
    // broad-phase cast only spans the source body's own displacement. Two fast
    // dynamic bodies can therefore meet halfway without either source cast
    // reaching the other's starting AABB. Split at the earliest relative TOI
    // so the following Jolt update starts with the pair touching and lets the
    // regular contact solver own the response.
    constexpr int kMaxDynamicCCDSplits = 8;
    constexpr float kTOIPadding = 1e-4f;
    constexpr float kMinStepDuration = 1e-6f;
    auto physicsGroup = JobSystem::Get().CreateTaskGroup(JobDomain::Physics, JobPriority::Critical);
    m_jobSystem->BeginFrame(physicsGroup);
    float remainingTime = deltaTime;
    int dynamicCCDSplits = 0;
    auto appendContactImpulses = [&]() {
        if (!m_contactImpulseStreamEnabled)
            return;
        const auto &solved = m_physicsSystem->GetAppliedContactImpulses();
        m_contactImpulses.reserve(m_contactImpulses.size() + solved.size());
        for (const auto &value : solved) {
            ContactImpulse impulse;
            impulse.bodyIdA = value.body1ID;
            impulse.bodyIdB = value.body2ID;
            impulse.subShapeIdA = value.subShapeID1;
            impulse.subShapeIdB = value.subShapeID2;
            impulse.contactPoint = {value.contactPoint.x, value.contactPoint.y, value.contactPoint.z};
            impulse.contactNormal = {value.normal.x, value.normal.y, value.normal.z};
            impulse.impulse = {value.impulse.x, value.impulse.y, value.impulse.z};
            m_contactImpulses.push_back(impulse);
        }
    };
    const auto updateSolver = [&](float duration, int collisionSteps) {
        const float stepDuration = duration / collisionSteps;
        const JPH::PhysicsSettings settings = m_physicsSystem->GetPhysicsSettings();
        // Jolt scales cached impulses by the ratio of the new and old time
        // steps. After a short CCD slice this can magnify an impact impulse.
        const bool resetWarmStart = settings.mConstraintWarmStart && m_previousSolverStepDuration > 0.0f &&
                                    stepDuration > 2.0f * m_previousSolverStepDuration;
        if (resetWarmStart) {
            auto coldSettings = settings;
            coldSettings.mConstraintWarmStart = false;
            m_physicsSystem->SetPhysicsSettings(coldSettings);
        }
        const auto error = m_physicsSystem->Update(duration, collisionSteps, m_tempAllocator.get(), m_jobSystem.get());
        if (resetWarmStart)
            m_physicsSystem->SetPhysicsSettings(settings);
        m_previousSolverStepDuration = stepDuration;
        if (error != JPH::EPhysicsUpdateError::None)
            throw std::runtime_error("physics update failed (error flags " +
                                     std::to_string(static_cast<uint32_t>(error)) + ")");
        appendContactImpulses();
    };
    try {
        while (remainingTime > kMinStepDuration) {
            const float hitFraction =
                std::min(FindEarliestStaticCCDFraction(remainingTime), FindEarliestDynamicCCDFraction(remainingTime));
            if (hitFraction >= 1.0f || dynamicCCDSplits >= kMaxDynamicCCDSplits) {
                updateSolver(remainingTime, EngineConfig::Get().physicsCollisionSteps);
                remainingTime = 0.0f;
                break;
            }

            const float segmentFraction = std::clamp(hitFraction + kTOIPadding, kTOIPadding, 0.999f);
            const float segmentTime = remainingTime * segmentFraction;
            updateSolver(segmentTime, 1);
            remainingTime -= segmentTime;
            ++dynamicCCDSplits;
        }
    } catch (...) {
        m_jobSystem->EndFrame();
        physicsGroup.Close();
        JobSystem::Get().Wait(physicsGroup);
        throw;
    }
    m_jobSystem->EndFrame();
    physicsGroup.Close();
    JobSystem::Get().Wait(physicsGroup);
    m_lastDynamicCCDSplitCount = static_cast<size_t>(dynamicCCDSplits);

    JPH::BodyIDVector activeAfter;
    m_physicsSystem->GetActiveBodies(JPH::EBodyType::RigidBody, activeAfter);
    m_poseReadbackBodyIds.reserve(activeBefore.size() + activeAfter.size());
    for (const JPH::BodyID id : activeBefore)
        m_poseReadbackBodyIds.push_back(id.GetIndexAndSequenceNumber());
    for (const JPH::BodyID id : activeAfter)
        m_poseReadbackBodyIds.push_back(id.GetIndexAndSequenceNumber());
    std::sort(m_poseReadbackBodyIds.begin(), m_poseReadbackBodyIds.end());
    m_poseReadbackBodyIds.erase(std::unique(m_poseReadbackBodyIds.begin(), m_poseReadbackBodyIds.end()),
                                m_poseReadbackBodyIds.end());

    // Resolve raw contact events through pair tracking — suppresses spurious
    // Enter/Exit caused by Jolt's body sleep/wake cycles.
    if (m_contactListener) {
        JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
        m_contactListener->ResolveEvents(bi);
    }

    // A completed fixed step publishes one new synchronized query epoch. Body
    // poses may have changed even when membership and authored transforms did
    // not, so generation changes at this publication boundary rather than per
    // solver write.
    if (!activeBefore.empty() || !activeAfter.empty())
        m_queryGeneration.fetch_add(1, std::memory_order_release);
}

static void PublishRaycastSubShape(const JPH::Body &body, const JPH::SubShapeID &subShapeId, RaycastHit &outHit)
{
    outHit.subShapeId = subShapeId.GetValue();
    JPH::SubShapeID remainder;
    const JPH::Shape *leaf = body.GetShape()->GetLeafShape(subShapeId, remainder);
    if (leaf && leaf->GetSubType() == JPH::EShapeSubType::Mesh) {
        outHit.triangleIndex = static_cast<const JPH::MeshShape *>(leaf)->GetTriangleUserData(remainder);
    }
}

void PhysicsWorld::SetContactEventStreamEnabled(bool enabled, bool includeTriggers)
{
    m_contactEventStreamEnabled = enabled;
    m_contactEventStreamIncludeTriggers = enabled && includeTriggers;
}

void PhysicsWorld::SetContactImpulseStreamEnabled(bool enabled)
{
    m_contactImpulseStreamEnabled = enabled;
    if (m_physicsSystem)
        m_physicsSystem->SetRecordAppliedContactImpulses(enabled);
    if (!enabled)
        m_contactImpulses.clear();
}

const std::vector<ContactEvent> &PhysicsWorld::GetContactEvents() const
{
    static const std::vector<ContactEvent> empty;
    return m_contactListener ? m_contactListener->GetEvents() : empty;
}

// ============================================================================
// Contact event dispatch (Unity-style collision/trigger callbacks)
// ============================================================================

size_t PhysicsWorld::DispatchContactEvents()
{
    const auto admittedState = SceneManager::Instance().GetSimulationState();
    if (!m_contactListener)
        return 0;

    const auto &events = m_contactListener->GetEvents();
    if (events.empty())
        return 0;
    const size_t eventCount = events.size();
    struct DispatchScope
    {
        bool &active;
        bool previous;
        ~DispatchScope()
        {
            active = previous;
        }
    } dispatchScope{m_dispatchingContacts, m_dispatchingContacts};
    m_dispatchingContacts = true;

    // Resolve the complete batch before any user code can remove a collider,
    // rebuild a compound shape or clear the listener's event storage. Subshape
    // slots and serialized IDs alone do not identify a component lifetime.
    struct PendingContact
    {
        ContactEvent event;
        ObjectHandle colliderA;
        ObjectHandle colliderB;
        PhysicsTargetReference bodyA;
        PhysicsTargetReference bodyB;
    };
    std::vector<PendingContact> pending;
    pending.reserve(eventCount);
    for (const auto &event : events) {
        Collider *a = ResolveColliderForSubShape(event.bodyIdA, event.subShapeIdA);
        Collider *b = ResolveColliderForSubShape(event.bodyIdB, event.subShapeIdB);
        if (a && b) {
            const auto captureBody = [](Collider *collider) {
                auto *body = collider->GetAttachedRigidbody();
                return PhysicsTargetReference(body);
            };
            pending.push_back({event, a->GetHandle(), b->GetHandle(), captureBody(a), captureBody(b)});
        }
    }

    std::vector<ObjectHandle> receiversA;
    std::vector<ObjectHandle> receiversB;
    receiversA.reserve(8);
    receiversB.reserve(8);

    auto resolveActiveComponent = [](const ObjectHandle &handle) -> Component * {
        if (!handle.IsValid())
            return nullptr;
        Component *comp = Component::FindByComponentId(handle.id);
        if (!comp || comp->GetHandle() != handle || comp->IsDestroyed() || comp->IsBeingDestroyed())
            return nullptr;
        GameObject *owner = comp->GetGameObject();
        if (!owner || owner->IsDestroying() || !owner->IsActiveInHierarchy())
            return nullptr;
        return comp;
    };
    auto resolveCollider = [&](const ObjectHandle &handle, uint32_t bodyId) -> Collider * {
        // Only handles captured from Collider instances enter this path. A
        // matching generation cannot resolve to a replacement component type.
        auto *collider = static_cast<Collider *>(resolveActiveComponent(handle));
        return collider && collider->IsEnabled() && collider->GetBodyId() == bodyId ? collider : nullptr;
    };

    auto wantsEvent = [](const Component &comp, ContactEventType type) {
        if (!comp.IsEnabled())
            return false;
        switch (type) {
        case ContactEventType::CollisionEnter:
            return comp.WantsCollisionEnterCallbacks();
        case ContactEventType::CollisionStay:
            return comp.WantsCollisionStayCallbacks();
        case ContactEventType::CollisionExit:
            return comp.WantsCollisionExitCallbacks();
        case ContactEventType::TriggerEnter:
            return comp.WantsTriggerEnterCallbacks();
        case ContactEventType::TriggerStay:
            return comp.WantsTriggerStayCallbacks();
        case ContactEventType::TriggerExit:
            return comp.WantsTriggerExitCallbacks();
        }
        return false;
    };

    for (const auto &contact : pending) {
        if (SceneManager::Instance().GetSimulationState() != admittedState)
            break;
        const auto &evt = contact.event;
        Collider *colA = resolveCollider(contact.colliderA, evt.bodyIdA);
        Collider *colB = resolveCollider(contact.colliderB, evt.bodyIdB);
        if (!colA || !colB)
            continue;

        GameObject *goA = colA->GetGameObject();
        GameObject *goB = colB->GetGameObject();
        if (!goA || !goB)
            continue;

        ContactEventType type = evt.type;
        const bool triggerPair = colA->IsTrigger() || colB->IsTrigger();
        if (triggerPair) {
            if (type == ContactEventType::CollisionEnter)
                type = ContactEventType::TriggerEnter;
            else if (type == ContactEventType::CollisionStay)
                type = ContactEventType::TriggerStay;
            else if (type == ContactEventType::CollisionExit)
                type = ContactEventType::TriggerExit;
        } else {
            if (type == ContactEventType::TriggerEnter)
                type = ContactEventType::CollisionEnter;
            else if (type == ContactEventType::TriggerStay)
                type = ContactEventType::CollisionStay;
            else if (type == ContactEventType::TriggerExit)
                type = ContactEventType::CollisionExit;
        }

        // Layer filtering for trigger events — sensors bypass Jolt's object layer
        // pair filter, so enforce Infernux's layer collision matrix here.
        bool isTrigger = (type == ContactEventType::TriggerEnter || type == ContactEventType::TriggerStay ||
                          type == ContactEventType::TriggerExit);
        if (isTrigger) {
            int layerA = goA->GetLayer();
            int layerB = goB->GetLayer();
            if (!TagLayerManager::Instance().GetLayersCollide(layerA, layerB))
                continue;
        }

        receiversA.clear();
        receiversB.clear();

        const auto collectReceivers = [&](Collider &collider, std::vector<ObjectHandle> &receivers) {
            VisitPhysicsReceiverObjects(collider, isTrigger, [&](GameObject *object) {
                for (const auto &component : object->GetAllComponents())
                    if (wantsEvent(*component, type))
                        receivers.push_back(component->GetHandle());
            });
        };
        collectReceivers(*colA, receiversA);
        collectReceivers(*colB, receiversB);

        if (receiversA.empty() && receiversB.empty())
            continue;

        // Snapshot both receiver lists before invoking either side. Newly added
        // components do not inherit this contact's already queued delivery.
        auto dispatchToReceivers = [&](const std::vector<ObjectHandle> &receivers, bool sideA) {
            for (const auto &handle : receivers) {
                if (SceneManager::Instance().GetSimulationState() != admittedState)
                    return;
                Collider *a = resolveCollider(contact.colliderA, evt.bodyIdA);
                Collider *b = resolveCollider(contact.colliderB, evt.bodyIdB);
                if (!a || !b)
                    break;
                Component *comp = resolveActiveComponent(handle);
                if (!comp || !wantsEvent(*comp, type))
                    continue;
                bool stillReceives = false;
                VisitPhysicsReceiverObjects(*(sideA ? a : b), isTrigger, [&](GameObject *object) {
                    stillReceives |= comp->GetGameObject() == object;
                });
                if (!stillReceives)
                    continue;

                CollisionInfo info;
                info.collider = sideA ? b : a;
                info.bodyTarget = sideA ? contact.bodyB : contact.bodyA;
                auto *otherBody = dynamic_cast<Rigidbody *>(info.bodyTarget.GetComponent());
                info.gameObject = otherBody ? otherBody->GetGameObject() : info.collider->GetGameObject();
                info.target = PhysicsTargetReference(info.collider);
                info.contactPoint = evt.contactPoint;
                // Jolt's normal points A -> B; the public normal points from
                // the other collider towards the receiving component.
                info.contactNormal = sideA ? -evt.contactNormal : evt.contactNormal;
                info.relativeVelocity = sideA ? evt.relativeVelocity : -evt.relativeVelocity;
                switch (type) {
                case ContactEventType::CollisionEnter:
                    comp->OnCollisionEnter(info);
                    break;
                case ContactEventType::CollisionStay:
                    comp->OnCollisionStay(info);
                    break;
                case ContactEventType::CollisionExit:
                    comp->OnCollisionExit(info);
                    break;
                case ContactEventType::TriggerEnter:
                    comp->OnTriggerEnter(info.collider);
                    break;
                case ContactEventType::TriggerStay:
                    comp->OnTriggerStay(info.collider);
                    break;
                case ContactEventType::TriggerExit:
                    comp->OnTriggerExit(info.collider);
                    break;
                }
            }
        };

        dispatchToReceivers(receiversA, true);
        dispatchToReceivers(receiversB, false);
    }
    return eventCount;
}

// ============================================================================
// Body management
// ============================================================================

uint32_t PhysicsWorld::CreateBody(PhysicsActorData &actor)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized)
        return 0xFFFFFFFF;
    auto *go = actor.owner;
    const bool isStatic = !actor.rigidbody || !actor.rigidbody->IsEnabled();

    size_t shapeCount = 0;
    std::vector<Collider *> members;
    auto shape = BuildShapeForColliderSet(actor, nullptr, &shapeCount, &members);
    if (!shape)
        return 0xFFFFFFFF;
    const auto *collider = members.empty() ? nullptr : members.front();

    Transform *tf = go->GetTransform();
    glm::quat rot = tf->GetWorldRotation();
    glm::vec3 pos = tf->GetPosition();

    JPH::EMotionType motionType = isStatic ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic;
    JPH::ObjectLayer objLayer = BodyLayerForMembers(members, *go, !isStatic);

    JPH::BodyCreationSettings settings(shape, JPH::RVec3(pos.x, pos.y, pos.z), JPH::Quat(rot.x, rot.y, rot.z, rot.w),
                                       motionType, objLayer);

    // Allow static bodies to later be switched to dynamic/kinematic
    // (e.g. when a Rigidbody component is added). Without this flag,
    // Jolt does not create MotionProperties for static bodies and
    // SetMotionType() will crash.
    settings.mAllowDynamicOrKinematic = true;
    settings.mApplyGyroscopicForce = true;
    settings.mIsSensor = !members.empty() && std::all_of(members.begin(), members.end(),
                                                         [](const Collider *member) { return member->IsTrigger(); });
    settings.mUseManifoldReduction = shapeCount <= 1;

    // Body-level fallback. Contact callbacks replace these values using the
    // actual compound subshapes and their material combine modes.
    settings.mFriction = collider ? collider->GetFriction() : 0.0f;
    settings.mRestitution = collider ? collider->GetBounciness() : 0.0f;

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    JPH::Body *body = bodyInterface.CreateBody(settings);
    if (!body) {
        INXLOG_ERROR("PhysicsWorld: Failed to create body.");
        return 0xFFFFFFFF;
    }

    JPH::BodyID bodyId = body->GetID();
    // NOTE: Body is created but NOT added to broadphase here.
    // Collider::OnEnable() queues broadphase publication separately.

    uint32_t id = bodyId.GetIndexAndSequenceNumber();
    SetBodyActorOwner(id, actor);
    m_bodies.at(id).members = std::move(members);
    auto &publishedRecord = m_bodies.at(id);
    m_physicsSystem->GetBodyInterface().SetObjectLayer(
        JPH::BodyID(id), BodyLayerForMembers(publishedRecord.members, *publishedRecord.gameObject,
                                             m_physicsSystem->GetBodyInterface().GetMotionType(JPH::BodyID(id)) !=
                                                 JPH::EMotionType::Static));
    PublishBodyQueryIdentitiesUnlocked(id);
    PublishBodyShapeGeometryUnlocked(id);
    PublishColliderCenters(actor);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
    return id;
}

void PhysicsWorld::DestroyBody(uint32_t id)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized || id == 0xFFFFFFFF)
        return;
    InvalidateRagdollsForBody(id);
    DestroyConstraintsForBody(id);

    // Contract: caller (PhysicsECSStore::RetireActor) must have already removed
    // this body from the broadphase. See PhysicsWorld.h::DestroyBody for the
    // full ordering invariant.
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.DestroyBody(JPH::BodyID(id));

    if (m_bodies.at(id).hasRetiredShapeMembers)
        --m_retiredShapeBodyCount;
    m_bodies.erase(id);
    m_staticContinuousBodyIds.erase(id);
    m_continuousBodyIds.erase(id);
    m_kinematicMoveStates.erase(id);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
    if (m_contactListener)
        m_contactListener->RemoveIgnoredPairsForBody(id);
}

void PhysicsWorld::SetBodyPosition(uint32_t bodyId, const glm::vec3 &pos, const glm::quat &rot,
                                   bool cancelKinematicMove)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    SetBodyPositionUnlocked(bodyId, pos, rot, cancelKinematicMove);
}

void PhysicsWorld::SetBodyPositionUnlocked(uint32_t bodyId, const glm::vec3 &pos, const glm::quat &rot,
                                           bool cancelKinematicMove)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    // Transform sync runs serially on the main thread before PhysicsSystem::Update.
    // Avoid taking Jolt's striped body lock once per moved static/kinematic body.
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterfaceNoLock();
    if (cancelKinematicMove && m_kinematicMoveStates.erase(bodyId))
        bodyInterface.SetLinearAndAngularVelocity(JPH::BodyID(bodyId), JPH::Vec3::sZero(), JPH::Vec3::sZero());
    bodyInterface.SetPositionAndRotation(JPH::BodyID(bodyId), JPH::RVec3(pos.x, pos.y, pos.z),
                                         JPH::Quat(rot.x, rot.y, rot.z, rot.w), JPH::EActivation::DontActivate);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::SetBodyPositionsBatch(const std::vector<PhysicsBodyPoseUpdate> &updates)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized || updates.empty())
        return;

    static thread_local JPH::BodyIDVector bodyIds;
    static thread_local std::vector<JPH::RVec3> positions;
    static thread_local std::vector<JPH::Quat> rotations;
    bodyIds.clear();
    positions.clear();
    rotations.clear();
    bodyIds.reserve(updates.size());
    positions.reserve(updates.size());
    rotations.reserve(updates.size());
    for (const auto &update : updates) {
        if (update.bodyId == 0xFFFFFFFF)
            continue;
        bodyIds.emplace_back(update.bodyId);
        positions.emplace_back(update.position.x, update.position.y, update.position.z);
        rotations.emplace_back(update.rotation.x, update.rotation.y, update.rotation.z, update.rotation.w);
    }
    if (bodyIds.empty())
        return;

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterfaceNoLock();
    // Jolt 5.6 removed the legacy batch setter. Keep the update path explicit
    // and use the no-lock interface; the caller already owns the serial sync
    // phase, so this preserves the same ordering without a compatibility shim.
    for (size_t i = 0; i < bodyIds.size(); ++i)
        bodyInterface.SetPositionAndRotationWhenChanged(bodyIds[i], positions[i], rotations[i],
                                                        JPH::EActivation::DontActivate);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::RequireResetPhase() const
{
    RequireOwnerThread();
    const auto &manager = SceneManager::Instance();
    if (!m_initialized || !manager.IsPlaying() || m_dispatchingContacts ||
        manager.GetFixedStepIndex() != manager.GetCompletedFixedStepIndex())
        throw std::runtime_error("body reset requires Play between completed steps, outside contact dispatch");
}

void PhysicsWorld::ResetBodyMotionStates(const std::vector<PhysicsBodyMotionState> &states)
{
    RequireResetPhase();
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    auto &interface = m_physicsSystem->GetBodyInterfaceNoLock();
    std::unordered_set<uint32_t> ids;
    JPH::BodyIDVector contactResetIds;
    contactResetIds.reserve(states.size());
    for (const auto &state : states) {
        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterfaceNoLock(), JPH::BodyID(state.bodyId));
        if (!lock.Succeeded() || lock.GetBody().IsStatic() || !lock.GetBody().IsInBroadPhase())
            throw std::invalid_argument("reset requires resident dynamic or kinematic bodies");
        ids.insert(state.bodyId);
        contactResetIds.emplace_back(state.bodyId);
    }
    for (const auto &state : states) {
        const JPH::BodyID id(state.bodyId);
        m_kinematicMoveStates.erase(state.bodyId);
        interface.SetPositionAndRotation(
            id, JPH::RVec3(state.position.x, state.position.y, state.position.z),
            JPH::Quat(state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w),
            JPH::EActivation::DontActivate);
        {
            JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterfaceNoLock(), id);
            lock.GetBody().ResetMotion();
        }
        interface.SetLinearAndAngularVelocity(
            id, JPH::Vec3(state.linearVelocity.x, state.linearVelocity.y, state.linearVelocity.z),
            JPH::Vec3(state.angularVelocity.x, state.angularVelocity.y, state.angularVelocity.z));
        interface.ActivateBody(id);
    }
    for (auto &[id, record] : m_constraints) {
        if (ids.count(record.bodyIdA) || ids.count(record.bodyIdB)) {
            record.constraint->ResetWarmStart();
            if (record.constraint->GetSubType() == JPH::EConstraintSubType::Hinge)
                static_cast<JPH::HingeConstraint *>(record.constraint)->ResetRotorState();
            ActivateConstraintBodies(record);
        }
    }
    m_physicsSystem->ForgetBodyContacts(contactResetIds.data(), static_cast<int>(contactResetIds.size()));
    if (m_contactListener)
        m_contactListener->ResetBodies(ids);
    m_contactImpulses.erase(std::remove_if(m_contactImpulses.begin(), m_contactImpulses.end(),
                                           [&](const ContactImpulse &event) {
                                               return ids.count(event.bodyIdA) || ids.count(event.bodyIdB);
                                           }),
                            m_contactImpulses.end());
    if (!states.empty())
        m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::UpdateBodyShape(Collider *collider, const Collider *exclude)
{
    auto &store = PhysicsECSStore::Instance();
    const auto handle = store.GetCollider(collider->GetECSHandle()).actorHandle;
    if (store.IsValid(handle))
        UpdateBodyShape(store.GetActor(handle), exclude);
}

void PhysicsWorld::UpdateBodyShape(PhysicsActorData &actor, const Collider *exclude)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    const uint32_t id = actor.bodyId;
    if (!m_initialized || id == 0xFFFFFFFF)
        return;
    // Component lifetime ends immediately, independently of an asynchronous
    // replacement mesh cook. Keep old subshape indices, retire their identities.
    auto &record = m_bodies.at(id);
    SetBodyActorOwner(id, actor);
    for (auto &member : record.members) {
        if (member &&
            (member == exclude ||
             std::find(actor.colliders.begin(), actor.colliders.end(), member) == actor.colliders.end() ||
             member->IsBeingDestroyed() || !member->IsEnabled() || !member->GetGameObject()->IsActiveInHierarchy()))
            member = nullptr;
    }
    const bool hasRetiredMembers = std::any_of(record.members.begin(), record.members.end(),
                                               [](const Collider *member) { return member == nullptr; });
    if (hasRetiredMembers && !record.hasRetiredShapeMembers) {
        record.hasRetiredShapeMembers = true;
        ++m_retiredShapeBodyCount;
        m_physicsSystem->GetBodyInterface().InvalidateContactCache(JPH::BodyID(id));
    }
    PublishBodyQueryIdentitiesUnlocked(id);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
    size_t shapeCount = 0;
    std::vector<Collider *> members;
    auto newShape = BuildShapeForColliderSet(actor, exclude, &shapeCount, &members);
    if (!newShape)
        return;

    auto *rigidbody = actor.rigidbody;
    std::optional<MassConfiguration> massConfiguration;
    if (rigidbody && rigidbody->IsEnabled() && !rigidbody->IsKinematic()) {
        const auto &properties = rigidbody->GetAuthoredMassProperties();
        massConfiguration = BodyMassConfiguration(newShape, rigidbody->GetMass(), properties ? &*properties : nullptr,
                                                  actor.owner->GetTransform()->GetWorldScale());
    }
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.SetUseManifoldReduction(JPH::BodyID(id), shapeCount <= 1);
    // The old bounds contain sleepers whose support is about to disappear.
    // Jolt activates the changed body itself, not those neighbouring bodies.
    WakeBodiesTouchingStatic(id);
    // SetShape takes Jolt's body write lock and Body owns a RefConst<Shape>.
    // Queries that already hold the read side therefore finish against the
    // previously published immutable shape; its BVH is retired only after
    // the last reference is released.  Never mutate a published mesh shape.
    // Geometry changes must retain authored mass and allowed axes. Recompute
    // inertia from the new shape through the same path as body configuration,
    // without first replacing it with the shape's default density mass.
    const JPH::Vec3 previousCenterOfMass = bodyInterface.GetShape(JPH::BodyID(id))->GetCenterOfMass();
    bodyInterface.SetShape(JPH::BodyID(id), newShape, false, JPH::EActivation::Activate);
    // All shape-publication paths share this actor revision, including sibling
    // additions/removals. Publish immediately after the real SetShape commit.
    ++actor.shapeRevision;
    const JPH::Vec3 deltaCenterOfMass = newShape->GetCenterOfMass() - previousCenterOfMass;
    for (auto &[constraintId, record] : m_constraints) {
        if (record.bodyIdA == id || record.bodyIdB == id) {
            // Jolt stores attachment points relative to the body's COM.
            // SetShape preserves the ID but does not notify constraints.
            if (deltaCenterOfMass != JPH::Vec3::sZero())
                record.constraint->NotifyShapeChanged(JPH::BodyID(id), deltaCenterOfMass);
            UpdateConstraintScale(record, id, actor.owner->GetTransform()->GetWorldScale());
        }
    }
    if (massConfiguration) {
        JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(id));
        if (lock.Succeeded() && lock.GetBody().IsDynamic())
            ApplyMassConfiguration(*lock.GetBody().GetMotionProperties(),
                                   static_cast<JPH::EAllowedDOFs>(0x3F & ~(rigidbody->GetConstraints() >> 1)),
                                   *massConfiguration);
    }
    // Publish geometry, aggregate sensor state and its exact member identities
    // under the same query snapshot lock. A single remaining shape has no
    // compound subshape ID, so its primary identity must follow that member.
    const bool isSensor = !members.empty() && std::all_of(members.begin(), members.end(),
                                                          [](const Collider *member) { return member->IsTrigger(); });
    const bool sensorChanged = bodyInterface.IsSensor(JPH::BodyID(id)) != isSensor;
    bodyInterface.SetIsSensor(JPH::BodyID(id), isSensor);
    SetBodyActorOwner(id, actor);
    if (record.hasRetiredShapeMembers) {
        record.hasRetiredShapeMembers = false;
        --m_retiredShapeBodyCount;
    }
    m_bodies.at(id).members = std::move(members);
    auto &publishedRecord = m_bodies.at(id);
    m_physicsSystem->GetBodyInterface().SetObjectLayer(
        JPH::BodyID(id), BodyLayerForMembers(publishedRecord.members, *publishedRecord.gameObject,
                                             m_physicsSystem->GetBodyInterface().GetMotionType(JPH::BodyID(id)) !=
                                                 JPH::EMotionType::Static));
    PublishBodyQueryIdentitiesUnlocked(id);
    PublishBodyShapeGeometryUnlocked(id);
    PublishColliderCenters(actor);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
    // Expanded or newly solid geometry can also reach sleeping bodies outside
    // the old bounds. These queries run only on authored shape changes.
    WakeBodiesTouchingStatic(id);
    if (sensorChanged) {
        InvalidateContactPairsForBody(id);
    }
}

void PhysicsWorld::RefreshBodyTriggerState(uint32_t bodyId)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    const auto &members = m_bodies.at(bodyId).members;
    const bool isSensor = !members.empty() && std::all_of(members.begin(), members.end(), [](const Collider *member) {
        return !member || member->IsTrigger();
    });
    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        lock.GetBody().SetIsSensor(isSensor);
        PublishBodyQueryIdentitiesUnlocked(bodyId);
        m_queryGeneration.fetch_add(1, std::memory_order_release);
    }
}

void PhysicsWorld::InvalidateContactPairsForBody(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;
    if (m_contactListener)
        m_contactListener->InvalidatePairsForBody(bodyId);
    auto &bodies = m_physicsSystem->GetBodyInterface();
    const JPH::BodyID id(bodyId);
    m_physicsSystem->ForgetBodyContacts(&id, 1);
    bodies.ActivateBody(JPH::BodyID(bodyId));
}

void PhysicsWorld::AddBodiesBatch(const std::vector<uint32_t> &bodies)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized || bodies.empty())
        return;

    // Separate static and dynamic bodies since they need different activation modes.
    std::vector<JPH::BodyID> staticIds;
    std::vector<JPH::BodyID> dynamicIds;
    staticIds.reserve(bodies.size());
    dynamicIds.reserve(bodies.size() / 4); // most spawned bodies are static

    // The owner thread publishes bodies outside Step, under the query write
    // boundary. Read their final motion types, not stale enqueue-time flags.
    const auto &bodyRead = m_physicsSystem->GetBodyInterfaceNoLock();
    for (const uint32_t id : bodies) {
        if (id == 0xFFFFFFFF)
            continue;
        const bool isStatic = bodyRead.GetMotionType(JPH::BodyID(id)) == JPH::EMotionType::Static;
        if (isStatic)
            staticIds.push_back(JPH::BodyID(id));
        else
            dynamicIds.push_back(JPH::BodyID(id));
    }

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();

    if (!staticIds.empty()) {
        JPH::BodyInterface::AddState state = bi.AddBodiesPrepare(staticIds.data(), static_cast<int>(staticIds.size()));
        bi.AddBodiesFinalize(staticIds.data(), static_cast<int>(staticIds.size()), state,
                             JPH::EActivation::DontActivate);
    }
    if (!dynamicIds.empty()) {
        JPH::BodyInterface::AddState state =
            bi.AddBodiesPrepare(dynamicIds.data(), static_cast<int>(dynamicIds.size()));
        bi.AddBodiesFinalize(dynamicIds.data(), static_cast<int>(dynamicIds.size()), state, JPH::EActivation::Activate);
    }
    if (!staticIds.empty() || !dynamicIds.empty())
        m_queryGeneration.fetch_add(1, std::memory_order_release);
    // Start and disabled-body authoring can queue forces before residency.
    // Submit them only after Jolt has accepted the unique final body set,
    // for initial creation and reactivation through the same publication path.
    for (const uint32_t id : bodies) {
        auto *rigidbody = FindRigidbodyByBodyId(id);
        if (rigidbody && rigidbody->IsEnabled())
            rigidbody->FlushPendingForceCommands();
    }
}

bool PhysicsWorld::IsBodyInBroadphase(uint32_t bodyId) const
{
    return m_initialized && bodyId != 0xFFFFFFFF && m_physicsSystem->GetBodyInterface().IsAdded(JPH::BodyID(bodyId));
}

void PhysicsWorld::RemoveBodyFromBroadphase(uint32_t bodyId, bool preserveVelocity)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    InvalidateRagdollsForBody(bodyId);

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    WakeBodiesTouchingStatic(bodyId);
    JPH::Vec3 linear = JPH::Vec3::sZero(), angular = JPH::Vec3::sZero();
    if (preserveVelocity)
        bodyInterface.GetLinearAndAngularVelocity(JPH::BodyID(bodyId), linear, angular);
    bodyInterface.RemoveBody(JPH::BodyID(bodyId));
    // Jolt deactivation clears velocity. A retained scene suspends residency,
    // not motion state: retain the values on the same allocated body.
    if (preserveVelocity)
        bodyInterface.SetLinearAndAngularVelocity(JPH::BodyID(bodyId), linear, angular);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
}

// ============================================================================
// Body dynamics (used by Rigidbody component)
// ============================================================================

void PhysicsWorld::SetBodyMotionType(uint32_t bodyId, int motionType)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::EMotionType mt;
    JPH::ObjectLayer layer;
    switch (motionType) {
    case 0:
        mt = JPH::EMotionType::Static;
        layer = PhysicsObjectLayers::Encode(0, false);
        break;
    case 1:
        mt = JPH::EMotionType::Kinematic;
        layer = PhysicsObjectLayers::Encode(0, true);
        break;
    case 2:
    default:
        mt = JPH::EMotionType::Dynamic;
        layer = PhysicsObjectLayers::Encode(0, true);
        break;
    }

    if (auto it = m_bodies.find(bodyId); it != m_bodies.end() && it->second.gameObject) {
        layer = BodyLayerForMembers(it->second.members, *it->second.gameObject, motionType != 0);
    }

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    if (bi.GetMotionType(JPH::BodyID(bodyId)) == mt)
        return;
    bi.SetMotionType(JPH::BodyID(bodyId), mt, JPH::EActivation::Activate);
    bi.SetObjectLayer(JPH::BodyID(bodyId), layer);
    if (mt != JPH::EMotionType::Dynamic) {
        m_staticContinuousBodyIds.erase(bodyId);
        m_continuousBodyIds.erase(bodyId);
    }
    // An explicit motion-type change supersedes any pending drag tracking —
    // never restore this body to Static behind the caller's back.
    m_kinematicMoveStates.erase(bodyId);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::RefreshBodyLayers(uint32_t bodyId)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    const JPH::EMotionType motionType = bi.GetMotionType(JPH::BodyID(bodyId));
    const bool moving = motionType != JPH::EMotionType::Static;
    const auto &record = m_bodies.at(bodyId);
    bi.SetObjectLayer(JPH::BodyID(bodyId), BodyLayerForMembers(record.members, *record.gameObject, moving));
    bi.InvalidateContactCache(JPH::BodyID(bodyId));
    PublishBodyQueryIdentitiesUnlocked(bodyId);
    WakeBodiesTouchingStatic(bodyId);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::SetBodyDamping(uint32_t bodyId, float linearDamping, float angularDamping)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (!body.IsStatic()) {
            JPH::MotionProperties *mp = body.GetMotionProperties();
            mp->SetLinearDamping(linearDamping);
            mp->SetAngularDamping(angularDamping);
        }
    }
}

void PhysicsWorld::SetBodyGravityFactor(uint32_t bodyId, float factor)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    bool activate = false;
    {
        JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
        if (lock.Succeeded()) {
            JPH::Body &body = lock.GetBody();
            if (!body.IsStatic()) {
                auto *motion = body.GetMotionProperties();
                activate = body.IsDynamic() && motion->GetGravityFactor() != factor;
                motion->SetGravityFactor(factor);
            }
        }
    }
    auto &bodies = m_physicsSystem->GetBodyInterface();
    if (activate && bodies.IsAdded(JPH::BodyID(bodyId)))
        bodies.ActivateBody(JPH::BodyID(bodyId));
}

void PhysicsWorld::SetBodyFriction(uint32_t bodyId, float friction)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetFriction(JPH::BodyID(bodyId), friction);
}

void PhysicsWorld::SetBodyRestitution(uint32_t bodyId, float restitution)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetRestitution(JPH::BodyID(bodyId), restitution);
}

glm::vec3 PhysicsWorld::GetBodyLinearVelocity(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::vec3(0.0f);

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::Vec3 v = bi.GetLinearVelocity(JPH::BodyID(bodyId));
    return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
}

void PhysicsWorld::SetBodyLinearVelocity(uint32_t bodyId, const glm::vec3 &vel)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetLinearVelocity(JPH::BodyID(bodyId), JPH::Vec3(vel.x, vel.y, vel.z));

    // An explicit velocity supersedes a pending MoveKinematic arrival stop
    // (e.g. constant-velocity kinematic platforms). Drag-converted statics
    // stay tracked so they are still restored to Static.
    if (auto it = m_kinematicMoveStates.find(bodyId); it != m_kinematicMoveStates.end() && !it->second.restoreStatic)
        m_kinematicMoveStates.erase(it);
}

glm::vec3 PhysicsWorld::GetBodyAngularVelocity(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::vec3(0.0f);

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::Vec3 v = bi.GetAngularVelocity(JPH::BodyID(bodyId));
    return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
}

void PhysicsWorld::SetBodyAngularVelocity(uint32_t bodyId, const glm::vec3 &vel)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetAngularVelocity(JPH::BodyID(bodyId), JPH::Vec3(vel.x, vel.y, vel.z));

    if (auto it = m_kinematicMoveStates.find(bodyId); it != m_kinematicMoveStates.end() && !it->second.restoreStatic)
        m_kinematicMoveStates.erase(it);
}

void PhysicsWorld::AddBodyForce(uint32_t bodyId, const glm::vec3 &force)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.AddForce(JPH::BodyID(bodyId), JPH::Vec3(force.x, force.y, force.z));
}

namespace
{
template <typename Apply> void ApplyBodyImpulse(JPH::PhysicsSystem &system, JPH::BodyID id, Apply apply)
{
    JPH::BodyLockWrite lock(system.GetBodyLockInterface(), id);
    if (!lock.Succeeded() || !lock.GetBody().IsDynamic())
        return;

    apply(lock.GetBody());
    // An already active body can be one step away from sleeping. Refresh
    // its sleep timer along with the impulse, under the same body lock.
    system.GetBodyInterfaceNoLock().ActivateBody(id);
}
} // namespace

void PhysicsWorld::AddBodyImpulse(uint32_t bodyId, const glm::vec3 &impulse)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    ApplyBodyImpulse(*m_physicsSystem, JPH::BodyID(bodyId),
                     [&](JPH::Body &body) { body.AddImpulse(JPH::Vec3(impulse.x, impulse.y, impulse.z)); });
}

void PhysicsWorld::AddBodyTorque(uint32_t bodyId, const glm::vec3 &torque)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.AddTorque(JPH::BodyID(bodyId), JPH::Vec3(torque.x, torque.y, torque.z));
}

void PhysicsWorld::AddBodyAngularImpulse(uint32_t bodyId, const glm::vec3 &impulse)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    ApplyBodyImpulse(*m_physicsSystem, JPH::BodyID(bodyId),
                     [&](JPH::Body &body) { body.AddAngularImpulse(JPH::Vec3(impulse.x, impulse.y, impulse.z)); });
}

// ---- Forces at position ----

void PhysicsWorld::AddBodyForceAtPosition(uint32_t bodyId, const glm::vec3 &force, const glm::vec3 &point)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.AddForce(JPH::BodyID(bodyId), JPH::Vec3(force.x, force.y, force.z), JPH::RVec3(point.x, point.y, point.z));
}

void PhysicsWorld::AddBodyImpulseAtPosition(uint32_t bodyId, const glm::vec3 &impulse, const glm::vec3 &point)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    ApplyBodyImpulse(*m_physicsSystem, JPH::BodyID(bodyId), [&](JPH::Body &body) {
        body.AddImpulse(JPH::Vec3(impulse.x, impulse.y, impulse.z), JPH::RVec3(point.x, point.y, point.z));
    });
}

// ---- Constraints / Motion quality ----

void PhysicsWorld::ValidateBodyMassProperties(const Rigidbody &body, float mass,
                                              const RigidbodyMassProperties *properties) const
{
    auto *owner = body.GetGameObject();
    const auto scale = owner ? owner->GetTransform()->GetWorldScale() : glm::vec3(1.0f);
    if (properties) {
        (void)BodyMassConfiguration(nullptr, mass, properties, scale);
        return;
    }
    if (!owner || !m_initialized || body.IsKinematic())
        return; // Geometry facts do not exist until a collider creates its body.
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    const auto id = body.GetBodyId();
    if (id == 0xFFFFFFFF)
        return;
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(id));
    if (lock.Succeeded() && lock.GetBody().IsDynamic()) {
        const JPH::Shape *shape = lock.GetBody().GetShape();
        if (shape->GetSubType() == JPH::EShapeSubType::OffsetCenterOfMass)
            shape = static_cast<const JPH::OffsetCenterOfMassShape *>(shape)->GetInnerShape();
        (void)BodyMassConfiguration(shape, mass, nullptr, scale);
    }
}

void PhysicsWorld::SetBodyAllowedDOFs(uint32_t bodyId, int allowedDOFs, float mass)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (body.IsDynamic()) {
            auto found = m_bodies.find(bodyId);
            auto *rigidbody = found == m_bodies.end() ? nullptr : found->second.rigidbody;
            const auto *properties = rigidbody && rigidbody->GetAuthoredMassProperties()
                                         ? &*rigidbody->GetAuthoredMassProperties()
                                         : nullptr;
            const auto scale =
                rigidbody ? rigidbody->GetGameObject()->GetTransform()->GetWorldScale() : glm::vec3(1.0f);
            const auto prepared = BodyMassConfiguration(body.GetShape(), mass, properties, scale);
            ApplyMassConfiguration(*body.GetMotionProperties(), static_cast<JPH::EAllowedDOFs>(allowedDOFs), prepared);
        }
    }
}

void PhysicsWorld::SetBodyMotionQuality(uint32_t bodyId, int quality)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    const int mappedQuality = MapMotionQualityMode(quality);
    // Jolt LinearCast also catches dynamic targets. Keep it only for the
    // ContinuousDynamic contract; plain Continuous is swept against statics by
    // FindEarliestStaticCCDFraction and otherwise remains discrete.
    JPH::EMotionQuality mq =
        (mappedQuality == 1 && quality == 2) ? JPH::EMotionQuality::LinearCast : JPH::EMotionQuality::Discrete;
    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetMotionQuality(JPH::BodyID(bodyId), mq);
    if (quality == 1)
        m_staticContinuousBodyIds.insert(bodyId);
    else
        m_staticContinuousBodyIds.erase(bodyId);
    if (quality == 2)
        m_continuousBodyIds.insert(bodyId);
    else
        m_continuousBodyIds.erase(bodyId);
}

void PhysicsWorld::SetBodyMaxAngularVelocity(uint32_t bodyId, float maxVel)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (!body.IsStatic()) {
            body.GetMotionProperties()->SetMaxAngularVelocity(maxVel);
        }
    }
}

void PhysicsWorld::SetBodyMaxLinearVelocity(uint32_t bodyId, float maxVel)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (!body.IsStatic()) {
            body.GetMotionProperties()->SetMaxLinearVelocity(maxVel);
        }
    }
}

// ---- Kinematic move ----

void PhysicsWorld::MoveBodyKinematic(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot,
                                     float deltaTime, float maxSpeed)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    MoveBodyKinematicUnlocked(bodyId, targetPos, targetRot, deltaTime, maxSpeed);
}

void PhysicsWorld::MoveBodyKinematicPosition(uint32_t bodyId, const glm::vec3 &targetPos, float deltaTime)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    const auto pending = m_kinematicMoveStates.find(bodyId);
    const glm::quat rotation = pending != m_kinematicMoveStates.end() && pending->second.movedThisStep
                                   ? pending->second.targetRotation
                                   : GetBodyRotation(bodyId);
    MoveBodyKinematicUnlocked(bodyId, targetPos, rotation, deltaTime, 0.0f);
}

void PhysicsWorld::MoveBodyKinematicRotation(uint32_t bodyId, const glm::quat &targetRot, float deltaTime)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    const auto pending = m_kinematicMoveStates.find(bodyId);
    const glm::vec3 position = pending != m_kinematicMoveStates.end() && pending->second.movedThisStep
                                   ? pending->second.targetPosition
                                   : GetBodyPosition(bodyId);
    MoveBodyKinematicUnlocked(bodyId, position, targetRot, deltaTime, 0.0f);
}

void PhysicsWorld::MoveBodyKinematicUnlocked(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot,
                                             float deltaTime, float maxSpeed)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    const JPH::BodyID id(bodyId);
    const JPH::RVec3 target(targetPos.x, targetPos.y, targetPos.z);
    const JPH::Quat targetQuat = ToJoltQuat(targetRot);

    bool posePublished = false;
    if (maxSpeed > 0.0f && deltaTime > 0.0f) {
        // Cap the contact velocity the move can impart (matches PhysX's
        // default maxDepenetrationVelocity). Excess displacement — a scripted
        // long-range teleport, or the first sync after un-pausing — is applied
        // as a teleport so the body never lags behind its Transform, and
        // trigger/contact detection still sees the body at its new location
        // in the very next step.
        const JPH::Vec3 delta(target - bi.GetPosition(id));
        const float maxStepDistance = maxSpeed * deltaTime;
        const float distance = delta.Length();
        if (distance > maxStepDistance) {
            const JPH::RVec3 nearTarget = target - JPH::RVec3(delta * (maxStepDistance / distance));
            bi.SetPositionAndRotation(id, nearTarget, targetQuat, JPH::EActivation::Activate);
            posePublished = true;
        }
    }

    bi.MoveKinematic(id, target, targetQuat, deltaTime);

    // Track the move so SettleKinematicMoves() can zero the velocity once the
    // body has arrived — MoveKinematic velocity persists in Jolt otherwise.
    auto &state = m_kinematicMoveStates[bodyId];
    state.targetPosition = targetPos;
    state.targetRotation = targetRot;
    state.movedThisStep = true;
    state.idleSteps = 0;
    if (posePublished)
        m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::MoveStaticBodyWithVelocity(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot,
                                              float deltaTime)
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (!m_initialized || bodyId == 0xFFFFFFFF || deltaTime <= 0.0f)
        return;

    const JPH::BodyID id(bodyId);
    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();

    const JPH::EMotionType motionType = bi.GetMotionType(id);
    if (motionType == JPH::EMotionType::Dynamic) {
        // A Rigidbody took ownership of this body mid-drag — plain teleport.
        SetBodyPositionUnlocked(bodyId, targetPos, targetRot);
        return;
    }

    if (motionType == JPH::EMotionType::Static) {
        // The object layer is intentionally left unchanged (still non-moving):
        // statics never collide with each other, and dynamic bodies pair with
        // the non-moving layer, which is all a drag push needs.
        bi.SetMotionType(id, JPH::EMotionType::Kinematic, JPH::EActivation::Activate);
    }

    MoveBodyKinematicUnlocked(bodyId, targetPos, targetRot, deltaTime, kMaxTransformDriveSpeed);
    m_kinematicMoveStates[bodyId].restoreStatic = true;
}

bool PhysicsWorld::IsBodySleeping(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return true;

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    return !bi.IsActive(JPH::BodyID(bodyId));
}

bool PhysicsWorld::IsBodySensor(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return false;

    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    return lock.Succeeded() ? lock.GetBody().IsSensor() : false;
}

void PhysicsWorld::ActivateBody(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    // Inactive objects and retained scenes keep their allocated bodies. Only
    // resident bodies belong to Jolt's active simulation set.
    if (bi.IsAdded(JPH::BodyID(bodyId)))
        bi.ActivateBody(JPH::BodyID(bodyId));
}

void PhysicsWorld::ActivateBodiesInAABB(const glm::vec3 &min, const glm::vec3 &max)
{
    if (!m_initialized)
        return;

    JPH::AABox box(JPH::Vec3(min.x, min.y, min.z), JPH::Vec3(max.x, max.y, max.z));
    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.ActivateBodiesInAABox(box, JPH::BroadPhaseLayerFilter(), JPH::ObjectLayerFilter());
}

void PhysicsWorld::WakeBodiesTouchingStatic(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    // Read the body's world-space AABB, then RELEASE the lock before
    // calling ActivateBodiesInAABox (which takes its own internal locks).
    // Holding BodyLockRead while calling the locking BodyInterface causes
    // a deadlock on Jolt's striped mutex.
    JPH::AABox bounds;
    {
        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
        if (!lock.Succeeded())
            return;

        const JPH::Body &body = lock.GetBody();
        bounds = body.GetWorldSpaceBounds();
    } // lock released

    // Expand by a small margin so bodies resting exactly on the surface are caught
    bounds.ExpandBy(JPH::Vec3::sReplicate(0.1f));

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.ActivateBodiesInAABox(bounds, JPH::BroadPhaseLayerFilter(), JPH::ObjectLayerFilter());
}

void PhysicsWorld::DeactivateBody(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.DeactivateBody(JPH::BodyID(bodyId));
    for (auto &[id, record] : m_constraints)
        if (record.constraint->GetSubType() == JPH::EConstraintSubType::Hinge &&
            (record.bodyIdA == bodyId || record.bodyIdB == bodyId))
            static_cast<JPH::HingeConstraint *>(record.constraint)->ResetRotorState();
}

glm::vec3 PhysicsWorld::GetBodyPosition(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::vec3(0.0f);

    // NoLock: safe because this is called from main thread AFTER Step().
    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::RVec3 p = bi.GetPosition(JPH::BodyID(bodyId));
    return glm::vec3(static_cast<float>(p.GetX()), static_cast<float>(p.GetY()), static_cast<float>(p.GetZ()));
}

glm::quat PhysicsWorld::GetBodyRotation(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::quat(1, 0, 0, 0);

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::Quat q = bi.GetRotation(JPH::BodyID(bodyId));
    return glm::normalize(glm::quat(q.GetW(), q.GetX(), q.GetY(), q.GetZ()));
}

glm::vec3 PhysicsWorld::GetBodyCenterOfMassPosition(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::vec3(0.0f);

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::RVec3 p = bi.GetCenterOfMassPosition(JPH::BodyID(bodyId));
    return glm::vec3(static_cast<float>(p.GetX()), static_cast<float>(p.GetY()), static_cast<float>(p.GetZ()));
}

PhysicsBodyMotionState PhysicsWorld::GetBodyMotionState(uint32_t bodyId, PhysicsStateFields fields) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        throw std::logic_error("Motion state requires an active physics body");
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (!lock.Succeeded() || !lock.GetBody().IsInBroadPhase())
        throw std::logic_error("Motion state requires an active physics body");
    const auto &body = lock.GetBody();
    const auto vector = [](const auto &v) {
        return glm::vec3(static_cast<float>(v.GetX()), static_cast<float>(v.GetY()), static_cast<float>(v.GetZ()));
    };
    PhysicsBodyMotionState state;
    state.bodyId = body.GetID().GetIndexAndSequenceNumber();
    if (HasPhysicsStateField(fields, PhysicsStateFields::Position))
        state.position = vector(body.GetPosition());
    if (HasPhysicsStateField(fields, PhysicsStateFields::Rotation)) {
        const auto rotation = body.GetRotation();
        state.rotation = glm::quat(rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ());
    }
    if (HasPhysicsStateField(fields, PhysicsStateFields::CenterOfMass))
        state.centerOfMass = vector(body.GetCenterOfMassPosition());
    if (HasPhysicsStateField(fields, PhysicsStateFields::LinearVelocity))
        state.linearVelocity = vector(body.GetLinearVelocity());
    if (HasPhysicsStateField(fields, PhysicsStateFields::AngularVelocity))
        state.angularVelocity = vector(body.GetAngularVelocity());
    if (body.IsDynamic()) {
        if (HasPhysicsStateField(fields, PhysicsStateFields::InverseMass)) {
            const auto *motion = body.GetMotionProperties();
            state.inverseMass = vector(motion->LockTranslation(JPH::Vec3::sReplicate(motion->GetInverseMass())));
        }
        if (HasPhysicsStateField(fields, PhysicsStateFields::InverseInertia)) {
            const auto inertia = body.GetInverseInertia();
            for (int column = 0; column < 3; ++column)
                state.inverseInertia[column] = vector(inertia.GetColumn3(column));
        }
    }
    return state;
}

void PhysicsWorld::SetColliderPairIgnored(Collider *colliderA, Collider *colliderB, bool ignored)
{
    if (!m_contactListener || !m_physicsSystem || !colliderA || !colliderB)
        throw std::logic_error("collision ignore requires an initialized physics world and two Colliders");
    m_contactListener->SetColliderPairIgnored(colliderA->GetComponentID(), colliderB->GetComponentID(), ignored);

    const uint32_t bodyIdA = colliderA->GetBodyId();
    const uint32_t bodyIdB = colliderB->GetBodyId();
    if (bodyIdA == 0xFFFFFFFF || bodyIdB == 0xFFFFFFFF || bodyIdA == bodyIdB)
        return;
    m_contactListener->InvalidatePairsForBody(bodyIdA);
    m_contactListener->InvalidatePairsForBody(bodyIdB);
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.InvalidateContactCache(JPH::BodyID(bodyIdA));
    bodyInterface.InvalidateContactCache(JPH::BodyID(bodyIdB));
    bodyInterface.ActivateBody(JPH::BodyID(bodyIdA));
    bodyInterface.ActivateBody(JPH::BodyID(bodyIdB));
}

bool PhysicsWorld::GetColliderPairIgnored(const Collider *colliderA, const Collider *colliderB) const
{
    if (!m_contactListener || !colliderA || !colliderB)
        return false;
    return m_contactListener->GetColliderPairIgnored(colliderA->GetComponentID(), colliderB->GetComponentID());
}

void PhysicsWorld::RemoveIgnoredPairsForCollider(const Collider *collider)
{
    if (m_contactListener && collider)
        m_contactListener->RemoveIgnoredPairsForCollider(collider->GetComponentID());
}

std::optional<PhysicsPenetrationResult> PhysicsWorld::ComputePenetration(const Collider &a, const glm::vec3 &positionA,
                                                                         const glm::quat &rotationA, const Collider &b,
                                                                         const glm::vec3 &positionB,
                                                                         const glm::quat &rotationB) const
{
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    if (!m_initialized)
        throw std::logic_error("compute_penetration requires initialized physics");
    struct QueryShape
    {
        JPH::RefConst<JPH::Shape> shape;
        JPH::Vec3 scale = JPH::Vec3::sReplicate(1.0f);
        JPH::Vec3 offset = JPH::Vec3::sZero();

        JPH::Vec3 CenterOfMass() const
        {
            return offset + scale * shape->GetCenterOfMass();
        }
    };
    const auto shapeForQuery = [this](const Collider &collider) -> QueryShape {
        if (const auto *mesh = dynamic_cast<const MeshCollider *>(&collider)) {
            const auto &store = PhysicsECSStore::Instance();
            const auto &state = store.GetCollider(mesh->GetECSHandle());
            const uint32_t bodyId =
                store.IsValid(state.actorHandle) ? store.GetActor(state.actorHandle).bodyId : UINT32_MAX;
            const PhysicsShapeGeometry *geometry = nullptr;
            const auto record = m_bodies.find(bodyId);
            if (record != m_bodies.end()) {
                JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
                if (lock.Succeeded()) {
                    for (const auto &leaf : record->second.shapeLeaves) {
                        const auto *identity = ResolvePublishedQueryIdentity(lock.GetBody(), bodyId, leaf.subShapeId);
                        if (identity && identity->collider == mesh) {
                            geometry = leaf.geometry.get();
                            break;
                        }
                    }
                }
            }
            if (!(geometry ? geometry->convex : mesh->IsConvex()))
                throw std::invalid_argument(
                    "compute_penetration requires published MeshCollider.convex = true geometry");
            if (!geometry)
                throw std::logic_error("compute_penetration MeshCollider geometry is not published");
            const auto &scale = geometry->scale;
            const auto &center = state.publishedRigidCenter;
            return {geometry->shape.get(), {scale.x, scale.y, scale.z}, {center.x, center.y, center.z}};
        }
        if (!dynamic_cast<const BoxCollider *>(&collider) && !dynamic_cast<const SphereCollider *>(&collider) &&
            !dynamic_cast<const CapsuleCollider *>(&collider) && !dynamic_cast<const CylinderCollider *>(&collider))
            throw std::invalid_argument(
                "compute_penetration supports Box, Sphere, Capsule, Cylinder and convex Mesh colliders");
        return {collider.CreateJoltShape()};
    };
    const auto shapeA = shapeForQuery(a), shapeB = shapeForQuery(b);
    // Work relative to A's origin, then restore world coordinates in the result.
    const auto delta = positionB - positionA;
    const auto transformA =
        JPH::Mat44::sRotation(ToJoltQuat(rotationA)) * JPH::Mat44::sTranslation(shapeA.CenterOfMass());
    const auto transformB =
        JPH::Mat44::sRotationTranslation(ToJoltQuat(rotationB), JPH::Vec3(delta.x, delta.y, delta.z)) *
        JPH::Mat44::sTranslation(shapeB.CenterOfMass());
    JPH::ClosestHitCollisionCollector<JPH::CollideShapeCollector> collector;
    JPH::CollisionDispatch::sCollideShapeVsShape(shapeA.shape, shapeB.shape, shapeA.scale, shapeB.scale, transformA,
                                                 transformB, JPH::SubShapeIDCreator(), JPH::SubShapeIDCreator(),
                                                 JPH::CollideShapeSettings(), collector);
    if (!collector.HadHit() || collector.mHit.mPenetrationDepth <= 0)
        return std::nullopt;
    const auto &hit = collector.mHit;
    const auto normal = -hit.mPenetrationAxis.Normalized();
    PhysicsPenetrationResult result;
    result.direction = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
    result.distance = hit.mPenetrationDepth;
    result.pointA =
        positionA + glm::vec3(hit.mContactPointOn1.GetX(), hit.mContactPointOn1.GetY(), hit.mContactPointOn1.GetZ());
    result.pointB =
        positionA + glm::vec3(hit.mContactPointOn2.GetX(), hit.mContactPointOn2.GetY(), hit.mContactPointOn2.GetZ());
    return result;
}

glm::mat3 PhysicsWorld::GetBodyWorldSpaceInertiaTensor(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::mat3(0.0f);

    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (!lock.Succeeded())
        return glm::mat3(0.0f);

    const JPH::Body &body = lock.GetBody();
    if (!body.IsDynamic())
        return glm::mat3(0.0f);

    const JPH::Mat44 invInertia = body.GetInverseInertia();
    const JPH::Vec3 c0 = invInertia.GetColumn3(0);
    const JPH::Vec3 c1 = invInertia.GetColumn3(1);
    const JPH::Vec3 c2 = invInertia.GetColumn3(2);

    const glm::mat3 invTensor(glm::vec3(c0.GetX(), c0.GetY(), c0.GetZ()), glm::vec3(c1.GetX(), c1.GetY(), c1.GetZ()),
                              glm::vec3(c2.GetX(), c2.GetY(), c2.GetZ()));

    const auto allowedDofs = static_cast<uint8_t>(body.GetMotionProperties()->GetAllowedDOFs());
    return InvertAllowedAngularSubspace(invTensor, allowedDofs);
}

// ============================================================================
// Raycast
// ============================================================================

bool PhysicsWorld::Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance, RaycastHit &outHit,
                           uint32_t layerMask, bool queryTriggers) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    return RaycastCurrent(origin, direction, maxDistance, outHit, layerMask, queryTriggers);
}

void PhysicsWorld::RaycastBatch(const float *originsXYZ, const float *directionsXYZ, size_t count, float maxDistance,
                                RaycastHit *outHits, uint8_t *outHitMask, uint32_t layerMask, bool queryTriggers,
                                uint64_t *outQueryGeneration, RaycastBatchProfile *outProfile) const
{
    const auto profileNowNs = []() noexcept {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    };
    const uint64_t syncStartNs = outProfile ? profileNowNs() : 0;
    const bool synchronized = PrepareRaycastBatchQuery();
    const uint64_t syncNs = outProfile && synchronized ? profileNowNs() - syncStartNs : 0;
    RaycastBatchPublished(originsXYZ, directionsXYZ, count, maxDistance, outHits, outHitMask, layerMask, queryTriggers,
                          outQueryGeneration, outProfile);
    if (outProfile)
        outProfile->snapshotSyncNs = syncNs;
}

bool PhysicsWorld::PrepareRaycastBatchQuery() const
{
    {
        std::shared_lock snapshotRead(m_querySnapshotMutex);
        if (std::this_thread::get_id() != m_ownerThreadId)
            return false;
    }
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    return true;
}

void PhysicsWorld::RaycastBatchPublished(const float *originsXYZ, const float *directionsXYZ, size_t count,
                                         float maxDistance, RaycastHit *outHits, uint8_t *outHitMask,
                                         uint32_t layerMask, bool queryTriggers, uint64_t *outQueryGeneration,
                                         RaycastBatchProfile *outProfile, bool directionsNormalized) const
{
    if ((count != 0 && (!originsXYZ || !directionsXYZ || !outHits || !outHitMask)))
        throw std::invalid_argument("raycast batch requires non-null storage");

    const auto profileNowNs = []() noexcept {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    };
    if (outProfile)
        *outProfile = {};
    const uint64_t lockStartNs = outProfile ? profileNowNs() : 0;
    std::shared_lock snapshotRead(m_querySnapshotMutex, std::defer_lock);
    snapshotRead.lock();
    if (outProfile) {
        const uint64_t lockedNs = profileNowNs();
        outProfile->snapshotLockWaitNs = lockedNs - lockStartNs;
    }
    const uint64_t queryGeneration = GetQueryGeneration();
    if (outQueryGeneration)
        *outQueryGeneration = queryGeneration;
    // The synchronized live-world query epoch is stable for this call. Keep
    // one native query boundary, then fan out large batches through the
    // engine JobSystem instead of making the caller serialize thousands of
    // independent narrow-phase casts on the owner thread.  Small batches stay
    // inline: scheduling overhead is larger than the query itself there.
    const auto castOne = [&](uint32_t index, RaycastBatchProfile *profile) {
        const size_t offset = index * 3;
        const glm::vec3 origin(originsXYZ[offset], originsXYZ[offset + 1], originsXYZ[offset + 2]);
        const glm::vec3 direction(directionsXYZ[offset], directionsXYZ[offset + 1], directionsXYZ[offset + 2]);
        outHits[index] = RaycastHit{};
        outHitMask[index] = RaycastCurrent(origin, direction, maxDistance, outHits[index], layerMask, queryTriggers,
                                           true, profile, directionsNormalized, queryGeneration)
                                ? uint8_t{1}
                                : uint8_t{0};
    };

    const uint64_t dispatchStartNs = outProfile ? profileNowNs() : 0;
    constexpr size_t kParallelRaycastThreshold = 256;
    if (count >= kParallelRaycastThreshold && count <= std::numeric_limits<uint32_t>::max() &&
        JobSystem::IsAvailable() && !JobSystem::Get().IsInline() && JobSystem::Get().GetWorkerCount() > 1) {
        // One task per ray makes large batches spend most of their time in
        // queueing, allocation and completion bookkeeping. Keep a few chunks
        // per worker so Jolt's narrow-phase work remains parallel without
        // turning the batch into thousands of tiny scheduler operations.
        const uint32_t workerCount = JobSystem::Get().GetWorkerCount();
        // Dense ray batches have near-uniform work per ray. One contiguous
        // chunk per worker avoids allocating and queueing 4x as many tiny
        // std::function jobs on every gameplay query while the waiting owner
        // thread can still help execute a queued chunk.
        const uint32_t targetChunks = workerCount == std::numeric_limits<uint32_t>::max()
                                          ? workerCount
                                          : std::max<uint32_t>(workerCount + 1u, 1u);
        const uint32_t chunkSize = std::max<uint32_t>(64, 1u + (static_cast<uint32_t>(count) - 1u) / targetChunks);
        std::vector<RaycastBatchProfile> chunkProfiles;
        if (outProfile)
            chunkProfiles.resize(1u + ((static_cast<uint32_t>(count) - 1u) / chunkSize));
        JobSystem::Get().ParallelForChunks(
            static_cast<uint32_t>(count), chunkSize,
            [&](uint32_t begin, uint32_t end) {
                RaycastBatchProfile *profile = outProfile ? &chunkProfiles[begin / chunkSize] : nullptr;
                for (uint32_t index = begin; index < end; ++index) {
                    castOne(index, profile);
                }
            },
            JobDomain::Physics, JobPriority::Critical);
        if (outProfile) {
            outProfile->dispatchWallNs = profileNowNs() - dispatchStartNs;
            for (const auto &profile : chunkProfiles) {
                outProfile->joltQueryCpuNs += profile.joltQueryCpuNs;
                outProfile->narrowPhaseCpuNs += profile.narrowPhaseCpuNs;
                outProfile->sortFilterCpuNs += profile.sortFilterCpuNs;
                outProfile->hitPublishCpuNs += profile.hitPublishCpuNs;
                outProfile->broadphaseCandidates += profile.broadphaseCandidates;
                outProfile->narrowphaseHits += profile.narrowphaseHits;
                outProfile->publishedHits += profile.publishedHits;
            }
        }
        if (GetQueryGeneration() != queryGeneration)
            throw std::logic_error("physics query world changed during raycast batch");
        return;
    }

    for (size_t index = 0; index < count; ++index) {
        castOne(static_cast<uint32_t>(index), outProfile);
    }
    if (outProfile)
        outProfile->dispatchWallNs = profileNowNs() - dispatchStartNs;
    if (GetQueryGeneration() != queryGeneration)
        throw std::logic_error("physics query world changed during raycast batch");
}

bool PhysicsWorld::RaycastCurrent(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance,
                                  RaycastHit &outHit, uint32_t layerMask, bool queryTriggers, bool queryEpochHeld,
                                  RaycastBatchProfile *profile, bool directionNormalized,
                                  uint64_t queryGeneration) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(origin))
        return false;

    glm::vec3 normalizedDirection = direction;
    if (!directionNormalized && !NormalizeQueryDirection(direction, maxDistance, normalizedDirection))
        return false;

    const JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                            JPH::Vec3(normalizedDirection.x * maxDistance, normalizedDirection.y * maxDistance,
                                      normalizedDirection.z * maxDistance));
    // The batch owns the engine query epoch across every worker. All Jolt
    // pose/shape/body/broadphase publication takes the write side of that
    // epoch, so taking Jolt's striped body lock again per candidate is
    // redundant. Single casts retain the ordinary locking interfaces.
    const JPH::NarrowPhaseQuery &query =
        queryEpochHeld ? m_physicsSystem->GetNarrowPhaseQueryNoLock() : m_physicsSystem->GetNarrowPhaseQuery();
    const JPH::BodyLockInterface &bodyLocks =
        queryEpochHeld ? static_cast<const JPH::BodyLockInterface &>(m_physicsSystem->GetBodyLockInterfaceNoLock())
                       : static_cast<const JPH::BodyLockInterface &>(m_physicsSystem->GetBodyLockInterface());
    LayerMaskObjectFilter objectFilter(layerMask);
    const auto profileNowNs = []() noexcept {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    };

    const auto publishBody = [&](const JPH::RayCastResult &result, const JPH::Body &body, bool filterTriggers) {
        outHit.distance = result.mFraction * maxDistance;
        outHit.point = origin + normalizedDirection * outHit.distance;
        outHit.bodyId = result.mBodyID.GetIndexAndSequenceNumber();
        outHit.collider = nullptr;
        outHit.gameObject = nullptr;
        outHit.colliderId = 0;
        outHit.gameObjectId = 0;
        outHit.target = PhysicsTargetReference{};

        // Resolve application identity from the immutable data published
        // with this query epoch. Collider properties are owner-authored
        // mutable state and must never be read by parallel query workers.
        const QueryTargetIdentity *identity = ResolvePublishedQueryIdentity(
            body, result.mBodyID.GetIndexAndSequenceNumber(), result.mSubShapeID2.GetValue());
        if (!identity || !(identity->layerBit & layerMask))
            return false;
        if (filterTriggers && (body.IsSensor() || (identity && identity->isTrigger)))
            return false;
        outHit.collider = identity ? identity->collider : nullptr;
        outHit.gameObject = identity ? identity->gameObject : nullptr;
        outHit.colliderId = identity ? identity->colliderId : 0;
        outHit.gameObjectId = identity ? identity->gameObjectId : 0;
        outHit.target = identity ? identity->target : PhysicsTargetReference{};
        const JPH::Vec3 normal = body.GetWorldSpaceSurfaceNormal(
            result.mSubShapeID2, JPH::RVec3(outHit.point.x, outHit.point.y, outHit.point.z));
        outHit.normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
        PublishRaycastSubShape(body, result.mSubShapeID2, outHit);
        return true;
    };

    const auto publish = [&](const JPH::RayCastResult &result, bool filterTriggers) {
        JPH::BodyLockRead lock(bodyLocks, result.mBodyID);
        return lock.Succeeded() && publishBody(result, lock.GetBody(), filterTriggers);
    };

    if (queryTriggers) {
        static thread_local RaycastPublishedBodyCache publishedBodyCache;
        DeterministicClosestRayBodyCollector collector(
            ray, bodyLocks, profile, queryEpochHeld ? &publishedBodyCache : nullptr, queryGeneration, layerMask);
        const uint64_t queryStartNs = profile ? profileNowNs() : 0;
        if (queryEpochHeld) {
            m_physicsSystem->GetBroadPhaseNoLock().CastRayNoLock(static_cast<JPH::RayCast>(ray), collector,
                                                                 JPH::BroadPhaseLayerFilter(), objectFilter);
        } else {
            m_physicsSystem->GetBroadPhaseQuery().CastRay(static_cast<JPH::RayCast>(ray), collector,
                                                          JPH::BroadPhaseLayerFilter(), objectFilter);
        }
        if (profile)
            profile->joltQueryCpuNs += profileNowNs() - queryStartNs;
        if (!collector.HadHit())
            return false;
        const uint64_t publishStartNs = profile ? profileNowNs() : 0;
        const bool accepted = publishBody(collector.Hit(), collector.HitBody(), false);
        if (profile) {
            profile->hitPublishCpuNs += profileNowNs() - publishStartNs;
            profile->publishedHits += accepted ? 1u : 0u;
        }
        return accepted;
    }

    const auto consumeAllHits = [&](auto &collector) {
        const uint64_t queryStartNs = profile ? profileNowNs() : 0;
        query.CastRay(ray, JPH::RayCastSettings(), collector, JPH::BroadPhaseLayerFilter(), objectFilter);
        if (profile)
            profile->joltQueryCpuNs += profileNowNs() - queryStartNs;
        if (!collector.HadHit())
            return false;
        const uint64_t sortStartNs = profile ? profileNowNs() : 0;
        std::sort(collector.mHits.begin(), collector.mHits.end(), [](const auto &left, const auto &right) {
            return std::tuple(left.mFraction, left.mBodyID.GetIndexAndSequenceNumber(), left.mSubShapeID2.GetValue()) <
                   std::tuple(right.mFraction, right.mBodyID.GetIndexAndSequenceNumber(),
                              right.mSubShapeID2.GetValue());
        });
        if (profile)
            profile->sortFilterCpuNs += profileNowNs() - sortStartNs;
        const uint64_t publishStartNs = profile ? profileNowNs() : 0;
        for (const JPH::RayCastResult &result : collector.mHits) {
            if (publish(result, true)) {
                if (profile) {
                    profile->hitPublishCpuNs += profileNowNs() - publishStartNs;
                    ++profile->publishedHits;
                }
                return true;
            }
        }
        if (profile)
            profile->hitPublishCpuNs += profileNowNs() - publishStartNs;
        return false;
    };
    if (profile) {
        ProfiledAllHitRayCollector collector(profile);
        return consumeAllHits(collector);
    }
    JPH::AllHitCollisionCollector<JPH::CastRayCollector> collector;
    return consumeAllHits(collector);
}

bool PhysicsWorld::RaycastCollider(const Collider &collider, const glm::vec3 &origin, const glm::vec3 &direction,
                                   float maxDistance, RaycastHit &outHit) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    if (!m_initialized || !IsFinite(origin))
        return false;

    glm::vec3 normalizedDirection(0.0f);
    if (!NormalizeQueryDirection(direction, maxDistance, normalizedDirection))
        return false;

    const uint32_t bodyId = collider.GetBodyId();
    if (bodyId == 0xFFFFFFFF)
        return false;

    const JPH::BodyID joltBodyId(bodyId);
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), joltBodyId);
    if (!lock.Succeeded())
        return false;

    const JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                            JPH::Vec3(normalizedDirection.x * maxDistance, normalizedDirection.y * maxDistance,
                                      normalizedDirection.z * maxDistance));
    const JPH::TransformedShape transformedShape = lock.GetBody().GetTransformedShape();
    JPH::AllHitCollisionCollector<JPH::CastRayCollector> collector;
    transformedShape.CastRay(ray, JPH::RayCastSettings(), collector);
    if (!collector.HadHit())
        return false;

    std::sort(collector.mHits.begin(), collector.mHits.end(), [](const auto &left, const auto &right) {
        return std::tuple(left.mFraction, left.mBodyID.GetIndexAndSequenceNumber(), left.mSubShapeID2.GetValue()) <
               std::tuple(right.mFraction, right.mBodyID.GetIndexAndSequenceNumber(), right.mSubShapeID2.GetValue());
    });
    for (const JPH::RayCastResult &result : collector.mHits) {
        if (ResolveColliderForSubShape(lock.GetBody(), bodyId, result.mSubShapeID2.GetValue()) != &collider)
            continue;

        outHit.distance = result.mFraction * maxDistance;
        outHit.point = origin + normalizedDirection * outHit.distance;
        outHit.bodyId = bodyId;
        outHit.collider = const_cast<Collider *>(&collider);
        outHit.gameObject = collider.GetGameObject();
        outHit.colliderId = collider.GetComponentID();
        outHit.gameObjectId = outHit.gameObject ? outHit.gameObject->GetID() : 0;
        outHit.target = PhysicsTargetReference(&collider);
        const JPH::Vec3 normal = transformedShape.GetWorldSpaceSurfaceNormal(
            result.mSubShapeID2, JPH::RVec3(outHit.point.x, outHit.point.y, outHit.point.z));
        outHit.normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
        PublishRaycastSubShape(lock.GetBody(), result.mSubShapeID2, outHit);
        return true;
    }
    return false;
}

glm::vec3 PhysicsWorld::ClosestPointOnCollider(const Collider &collider, const glm::vec3 &point) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    if (!collider.IsEnabled() || !collider.GetGameObject()->IsActiveInHierarchy())
        return point;
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    if (!m_initialized)
        throw std::logic_error("closest_point requires initialized physics");
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(collider.GetBodyId()));
    if (!lock.SucceededAndIsInBroadPhase())
        throw std::logic_error("closest_point requires a registered collider");
    const auto &body = lock.GetBody();
    const auto transformed = body.GetTransformedShape();
    JPH::AllHitCollisionCollector<JPH::CollidePointCollector> inside;
    transformed.CollidePoint(JPH::RVec3(point.x, point.y, point.z), inside);
    for (const auto &hit : inside.mHits)
        if (ResolveColliderForSubShape(body, collider.GetBodyId(), hit.mSubShapeID2.GetValue()) == &collider)
            return point;

    JPH::AllHitCollisionCollector<JPH::TransformedShapeCollector> leaves;
    transformed.CollectTransformedShapes(body.GetWorldSpaceBounds(), leaves);
    const JPH::SphereShape pointShape(1.0e-5f);
    float minimumDistance = std::numeric_limits<float>::infinity();
    glm::vec3 closestPoint{};
    for (const auto &leaf : leaves.mHits) {
        if (ResolveColliderForSubShape(body, collider.GetBodyId(), leaf.mSubShapeIDCreator.GetID().GetValue()) !=
            &collider)
            continue;
        const auto &center = leaf.mShapePositionCOM;
        const glm::vec3 base(center.GetX(), center.GetY(), center.GetZ());
        const auto relative = point - base;
        const JPH::AABox bounds = leaf.mShape->GetLocalBounds();
        JPH::CollideShapeSettings settings;
        settings.mMaxSeparationDistance =
            glm::length(relative) +
            ((bounds.GetCenter().Abs() + bounds.GetExtent()) * leaf.GetShapeScale().Abs()).Length() + 1.0f;
        JPH::ClosestHitCollisionCollector<JPH::CollideShapeCollector> collector;
        JPH::CollisionDispatch::sCollideShapeVsShape(
            &pointShape, leaf.mShape, JPH::Vec3::sReplicate(1), leaf.GetShapeScale(),
            JPH::Mat44::sTranslation(JPH::Vec3(relative.x, relative.y, relative.z)),
            JPH::Mat44::sRotation(leaf.mShapeRotation), JPH::SubShapeIDCreator(), JPH::SubShapeIDCreator(), settings,
            collector);
        if (!collector.HadHit())
            continue;
        const auto &contact = collector.mHit.mContactPointOn2;
        const auto candidate = base + glm::vec3(contact.GetX(), contact.GetY(), contact.GetZ());
        const float distance = glm::dot(candidate - point, candidate - point);
        if (distance < minimumDistance) {
            minimumDistance = distance;
            closestPoint = candidate;
        }
    }
    if (!std::isfinite(minimumDistance))
        throw std::runtime_error("closest_point collider geometry is not published");
    return closestPoint;
}

std::vector<RaycastHit> PhysicsWorld::RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance,
                                                 uint32_t layerMask, bool queryTriggers) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    std::vector<RaycastHit> hits;
    if (!m_initialized || layerMask == 0 || !IsFinite(origin))
        return hits;

    glm::vec3 dir(0.0f);
    if (!NormalizeQueryDirection(direction, maxDistance, dir)) {
        return hits;
    }

    JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                      JPH::Vec3(dir.x * maxDistance, dir.y * maxDistance, dir.z * maxDistance));

    const JPH::NarrowPhaseQuery &npQuery = m_physicsSystem->GetNarrowPhaseQuery();
    JPH::AllHitCollisionCollector<JPH::CastRayCollector> collector;
    LayerMaskObjectFilter objectFilter(layerMask);
    npQuery.CastRay(ray, JPH::RayCastSettings(), collector, JPH::BroadPhaseLayerFilter(), objectFilter);

    if (!collector.HadHit()) {
        return hits;
    }

    std::sort(collector.mHits.begin(), collector.mHits.end(), [](const auto &left, const auto &right) {
        return std::tuple(left.mFraction, left.mBodyID.GetIndexAndSequenceNumber(), left.mSubShapeID2.GetValue()) <
               std::tuple(right.mFraction, right.mBodyID.GetIndexAndSequenceNumber(), right.mSubShapeID2.GetValue());
    });
    hits.reserve(static_cast<size_t>(collector.mHits.size()));

    for (const JPH::RayCastResult &result : collector.mHits) {
        const uint32_t bodyId = result.mBodyID.GetIndexAndSequenceNumber();

        RaycastHit hit;
        hit.distance = result.mFraction * maxDistance;
        hit.point = origin + dir * hit.distance;
        hit.bodyId = bodyId;

        hit.collider = ResolveColliderForSubShape(hit.bodyId, result.mSubShapeID2.GetValue());
        if (!hit.collider || !(layerMask & (uint32_t{1} << hit.collider->GetGameObject()->GetLayer())))
            continue;
        if (!queryTriggers && (IsBodySensor(bodyId) || (hit.collider && hit.collider->IsTrigger())))
            continue;
        if (hit.collider && hit.collider->GetGameObject()) {
            hit.gameObject = hit.collider->GetGameObject();
        }
        hit.colliderId = hit.collider ? hit.collider->GetComponentID() : 0;
        hit.gameObjectId = hit.gameObject ? hit.gameObject->GetID() : 0;
        hit.target = PhysicsTargetReference(hit.collider);

        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), result.mBodyID);
        if (lock.Succeeded()) {
            const JPH::Body &body = lock.GetBody();
            JPH::Vec3 normal =
                body.GetWorldSpaceSurfaceNormal(result.mSubShapeID2, JPH::RVec3(hit.point.x, hit.point.y, hit.point.z));
            hit.normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
            PublishRaycastSubShape(body, result.mSubShapeID2, hit);
        }

        hits.push_back(hit);
    }

    // Jolt's broadphase traversal order is deliberately unspecified.  Keep
    // the public result stable for equal-distance surfaces so Scene picking,
    // MCP observation and gameplay do not change merely because a tree was
    // rebuilt.  Geometry identity is the final deterministic tie-breaker.
    std::sort(hits.begin(), hits.end(), [](const RaycastHit &left, const RaycastHit &right) {
        return std::tie(left.distance, left.bodyId, left.subShapeId, left.triangleIndex) <
               std::tie(right.distance, right.bodyId, right.subShapeId, right.triangleIndex);
    });

    return hits;
}

// ============================================================================
// Shared overlap / shape-cast implementations
// ============================================================================

std::vector<Collider *> PhysicsWorld::OverlapShapeImpl(const JPH::Shape &shape, const glm::vec3 &center,
                                                       const glm::quat &orientation, uint32_t layerMask,
                                                       bool queryTriggers) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    std::vector<Collider *> results;
    JPH::CollideShapeSettings settings;
    JPH::RMat44 transform =
        JPH::RMat44::sRotationTranslation(ToJoltQuat(orientation), JPH::RVec3(center.x, center.y, center.z));

    const JPH::NarrowPhaseQuery &npQuery = m_physicsSystem->GetNarrowPhaseQuery();
    JPH::AllHitCollisionCollector<JPH::CollideShapeCollector> collector;
    LayerMaskObjectFilter objectFilter(layerMask);
    npQuery.CollideShape(&shape, JPH::Vec3::sReplicate(1.0f), transform, settings,
                         JPH::RVec3(center.x, center.y, center.z), collector, JPH::BroadPhaseLayerFilter(),
                         objectFilter);

    if (!collector.HadHit())
        return results;

    std::unordered_set<Collider *> seen;
    for (const auto &hit : collector.mHits) {
        uint32_t bodyId = hit.mBodyID2.GetIndexAndSequenceNumber();
        Collider *col = ResolveColliderForSubShape(bodyId, hit.mSubShapeID2.GetValue());
        if (!col || !(layerMask & (uint32_t{1} << col->GetGameObject()->GetLayer())))
            continue;
        if (!queryTriggers && (IsBodySensor(bodyId) || (col && col->IsTrigger())))
            continue;
        if (col && seen.insert(col).second) {
            results.push_back(col);
        }
    }
    std::sort(results.begin(), results.end(), [](const Collider *left, const Collider *right) {
        const uint64_t leftId = left ? left->GetComponentID() : 0;
        const uint64_t rightId = right ? right->GetComponentID() : 0;
        return leftId < rightId;
    });
    return results;
}

bool PhysicsWorld::ShapeCastImpl(const JPH::Shape &shape, const glm::vec3 &origin, const glm::quat &orientation,
                                 const glm::vec3 &direction, float maxDistance, RaycastHit &outHit, uint32_t layerMask,
                                 bool queryTriggers, const Collider *movingCollider) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    glm::vec3 dir(0.0f);
    if (!NormalizeQueryDirection(direction, maxDistance, dir))
        return false;

    JPH::RShapeCast shapeCast = JPH::RShapeCast::sFromWorldTransform(
        &shape, JPH::Vec3::sReplicate(1.0f),
        JPH::RMat44::sRotationTranslation(ToJoltQuat(orientation), JPH::RVec3(origin.x, origin.y, origin.z)),
        JPH::Vec3(dir.x * maxDistance, dir.y * maxDistance, dir.z * maxDistance));

    JPH::ShapeCastSettings castSettings;
    // Initial overlap also promises a target surface point. Jolt otherwise
    // permits an interior GJK point; EPA runs only for penetrating starts.
    castSettings.mReturnDeepestPoint = true;
    castSettings.mUseShrunkenShapeAndConvexRadius = movingCollider != nullptr;
    JPH::AllHitCollisionCollector<JPH::CastShapeCollector> collector;
    LayerMaskObjectFilter objectFilter(layerMask);

    const JPH::NarrowPhaseQuery &npQuery = m_physicsSystem->GetNarrowPhaseQuery();
    npQuery.CastShape(shapeCast, castSettings, JPH::RVec3(origin.x, origin.y, origin.z), collector,
                      JPH::BroadPhaseLayerFilter(), objectFilter);

    if (!collector.HadHit())
        return false;

    std::sort(collector.mHits.begin(), collector.mHits.end(), [](const auto &left, const auto &right) {
        return std::tuple(left.mFraction, left.mBodyID2.GetIndexAndSequenceNumber(), left.mSubShapeID2.GetValue()) <
               std::tuple(right.mFraction, right.mBodyID2.GetIndexAndSequenceNumber(), right.mSubShapeID2.GetValue());
    });
    const JPH::ShapeCastResult *selected = nullptr;
    Collider *selectedCollider = nullptr;
    const JPH::Vec3 castDirection(dir.x, dir.y, dir.z);
    const uint32_t movingBody = movingCollider ? movingCollider->GetBodyId() : UINT32_MAX;
    for (const auto &candidate : collector.mHits) {
        uint32_t candidateBodyId = candidate.mBodyID2.GetIndexAndSequenceNumber();
        Collider *candidateCollider = ResolveColliderForSubShape(candidateBodyId, candidate.mSubShapeID2.GetValue());
        if (!candidateCollider || !(layerMask & (uint32_t{1} << candidateCollider->GetGameObject()->GetLayer())))
            continue;
        if (!queryTriggers && (IsBodySensor(candidateBodyId) || (candidateCollider && candidateCollider->IsTrigger())))
            continue;
        if (movingCollider &&
            (candidateBodyId == movingBody || candidateCollider->GetGameObject() == movingCollider->GetGameObject() ||
             GetColliderPairIgnored(movingCollider, candidateCollider) ||
             (candidate.mFraction <= 0.0f &&
              candidate.mPenetrationAxis.Dot(castDirection) <= 1e-3f * candidate.mPenetrationAxis.Length())))
            continue;
        selected = &candidate;
        selectedCollider = candidateCollider;
        break;
    }
    if (!selected)
        return false;

    const auto &result = *selected;
    uint32_t bodyId = result.mBodyID2.GetIndexAndSequenceNumber();

    outHit.distance = result.mFraction * maxDistance;
    // CastShape reports contacts relative to its base offset (origin above).
    // Distance is shape travel; point belongs to the selected target surface.
    const auto &contact = result.mContactPointOn2;
    outHit.point = origin + glm::vec3(contact.GetX(), contact.GetY(), contact.GetZ());
    outHit.bodyId = bodyId;
    outHit.normal =
        glm::vec3(-result.mPenetrationAxis.GetX(), -result.mPenetrationAxis.GetY(), -result.mPenetrationAxis.GetZ());
    float nLen = glm::length(outHit.normal);
    if (nLen > kEpsilon)
        outHit.normal /= nLen;

    outHit.collider = selectedCollider;
    if (outHit.collider && outHit.collider->GetGameObject())
        outHit.gameObject = outHit.collider->GetGameObject();
    outHit.colliderId = outHit.collider ? outHit.collider->GetComponentID() : 0;
    outHit.gameObjectId = outHit.gameObject ? outHit.gameObject->GetID() : 0;
    outHit.target = PhysicsTargetReference(outHit.collider);

    return true;
}

// ============================================================================
// Overlap queries (Unity: Physics.OverlapSphere / OverlapBox)
// ============================================================================

std::vector<Collider *> PhysicsWorld::OverlapSphere(const glm::vec3 &center, float radius, uint32_t layerMask,
                                                    bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(center) || !std::isfinite(radius) || radius <= 0.0f)
        return {};
    JPH::SphereShape sphere(radius);
    return OverlapShapeImpl(sphere, center, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), layerMask, queryTriggers);
}

std::vector<Collider *> PhysicsWorld::OverlapBox(const glm::vec3 &center, const glm::vec3 &halfExtents,
                                                 const glm::quat &orientation, uint32_t layerMask,
                                                 bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(center) || !HasPositiveFiniteExtents(halfExtents) ||
        !IsFinite(orientation))
        return {};
    JPH::BoxShape box(JPH::Vec3(halfExtents.x, halfExtents.y, halfExtents.z));
    return OverlapShapeImpl(box, center, orientation, layerMask, queryTriggers);
}

std::vector<Rigidbody *> PhysicsWorld::QueryRigidbodiesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum,
                                                                uint32_t layerMask, bool queryTriggers) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    std::vector<Rigidbody *> results;
    if (!m_initialized || layerMask == 0 || !IsFinite(minimum) || !IsFinite(maximum) || minimum.x > maximum.x ||
        minimum.y > maximum.y || minimum.z > maximum.z)
        return results;

    JPH::AllHitCollisionCollector<JPH::CollideShapeBodyCollector> collector;
    LayerMaskObjectFilter objectFilter(layerMask);
    const JPH::AABox bounds(JPH::Vec3(minimum.x, minimum.y, minimum.z), JPH::Vec3(maximum.x, maximum.y, maximum.z));
    m_physicsSystem->GetBroadPhaseQuery().CollideAABox(bounds, collector, JPH::BroadPhaseLayerFilter(), objectFilter);

    results.reserve(static_cast<size_t>(collector.mHits.size()));
    std::unordered_set<Rigidbody *> seen;
    for (const JPH::BodyID &bodyId : collector.mHits) {
        const uint32_t id = bodyId.GetIndexAndSequenceNumber();
        const auto &targets = m_bodies.at(id).targets;
        if (!std::any_of(targets.begin(), targets.end(), [&](const auto &target) {
                return (target.layerBit & layerMask) && (queryTriggers || !target.isTrigger);
            }))
            continue;
        Rigidbody *rigidbody = FindRigidbodyByBodyId(id);
        if (rigidbody && rigidbody->IsEnabled() && seen.insert(rigidbody).second)
            results.push_back(rigidbody);
    }
    std::sort(results.begin(), results.end(), [](const Rigidbody *left, const Rigidbody *right) {
        const uint64_t leftId = left ? left->GetComponentID() : 0;
        const uint64_t rightId = right ? right->GetComponentID() : 0;
        return leftId < rightId;
    });
    return results;
}

std::vector<Collider *> PhysicsWorld::OverlapCapsule(const glm::vec3 &point0, const glm::vec3 &point1, float radius,
                                                     uint32_t layerMask, bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(point0) || !IsFinite(point1) || !std::isfinite(radius) ||
        radius <= 0.0f)
        return {};
    const float segmentLength = glm::length(point1 - point0);
    const glm::vec3 center = (point0 + point1) * 0.5f;
    if (segmentLength <= 1e-6f) {
        JPH::SphereShape sphere(radius);
        return OverlapShapeImpl(sphere, center, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), layerMask, queryTriggers);
    }
    JPH::CapsuleShape capsule(segmentLength * 0.5f, radius);
    return OverlapShapeImpl(capsule, center, CapsuleOrientation(point0, point1), layerMask, queryTriggers);
}

// ============================================================================
// Shape cast queries (Unity: Physics.SphereCast / BoxCast)
// ============================================================================

bool PhysicsWorld::SphereCast(const glm::vec3 &origin, float radius, const glm::vec3 &direction, float maxDistance,
                              RaycastHit &outHit, uint32_t layerMask, bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(origin) || !std::isfinite(radius) || radius <= 0.0f)
        return false;
    JPH::SphereShape sphere(radius);
    return ShapeCastImpl(sphere, origin, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), direction, maxDistance, outHit, layerMask,
                         queryTriggers);
}

bool PhysicsWorld::BoxCast(const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction,
                           const glm::quat &orientation, float maxDistance, RaycastHit &outHit, uint32_t layerMask,
                           bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(center) || !HasPositiveFiniteExtents(halfExtents) ||
        !IsFinite(orientation))
        return false;

    JPH::BoxShape box(JPH::Vec3(halfExtents.x, halfExtents.y, halfExtents.z));
    return ShapeCastImpl(box, center, orientation, direction, maxDistance, outHit, layerMask, queryTriggers);
}

bool PhysicsWorld::SweepBox(const BoxCollider &box, const glm::vec3 &position, const glm::quat &rotation,
                            const glm::vec3 &displacement, RaycastHit &outHit) const
{
    const auto *object = box.GetGameObject();
    if (!m_initialized || !object)
        return false;
    const auto layerMask = TagLayerManager::Instance().GetLayerCollisionMask(object->GetLayer());
    const auto shape = box.CreateJoltShape();
    return ShapeCastImpl(*shape, position, rotation, displacement, glm::length(displacement), outHit, layerMask, false,
                         &box);
}

bool PhysicsWorld::CapsuleCast(const glm::vec3 &point0, const glm::vec3 &point1, float radius,
                               const glm::vec3 &direction, float maxDistance, RaycastHit &outHit, uint32_t layerMask,
                               bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(point0) || !IsFinite(point1) || !std::isfinite(radius) ||
        radius <= 0.0f)
        return false;
    const float segmentLength = glm::length(point1 - point0);
    const glm::vec3 center = (point0 + point1) * 0.5f;
    if (segmentLength <= 1e-6f) {
        JPH::SphereShape sphere(radius);
        return ShapeCastImpl(sphere, center, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), direction, maxDistance, outHit,
                             layerMask, queryTriggers);
    }
    JPH::CapsuleShape capsule(segmentLength * 0.5f, radius);
    return ShapeCastImpl(capsule, center, CapsuleOrientation(point0, point1), direction, maxDistance, outHit, layerMask,
                         queryTriggers);
}

// ============================================================================
// Lookup
// ============================================================================

void PhysicsWorld::SetBodyActorOwner(uint32_t bodyId, const PhysicsActorData &actor)
{
    auto &record = m_bodies[bodyId];
    record.gameObject = actor.owner;
    record.rigidbody = actor.rigidbody;
}

Rigidbody *PhysicsWorld::FindRigidbodyByBodyId(uint32_t bodyId) const
{
    const auto found = m_bodies.find(bodyId);
    return found == m_bodies.end() ? nullptr : found->second.rigidbody;
}

std::vector<glm::vec3> PhysicsShapeGeometry::GetTriangles() const
{
    JPH::Shape::GetTrianglesContext context;
    const JPH::Vec3 leafScale(scale.x, scale.y, scale.z);
    shape->GetTrianglesStart(context, shape->GetLocalBounds().Scaled(leafScale), JPH::Vec3::sZero(),
                             JPH::Quat::sIdentity(), leafScale);
    // A convex face is emitted as one block (up to 254 triangles).
    constexpr int batchSize = 256;
    JPH::Float3 vertices[3 * batchSize];
    std::vector<glm::vec3> result;
    for (int count; (count = shape->GetTrianglesNext(context, batchSize, vertices)) != 0;)
        for (int index = 0; index < count * 3; ++index)
            result.emplace_back(vertices[index].x, vertices[index].y, vertices[index].z);
    return result;
}

void PhysicsWorld::PublishBodyShapeGeometryUnlocked(uint32_t bodyId)
{
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    const auto &body = lock.GetBody();
    auto &record = m_bodies.at(bodyId);
    auto previous = std::move(record.shapeLeaves);
    JPH::AllHitCollisionCollector<JPH::TransformedShapeCollector> collector;
    body.GetTransformedShape().CollectTransformedShapes(body.GetWorldSpaceBounds(), collector);
    record.shapeLeaves.reserve(collector.mHits.size());
    const auto inverseBodyRotation = body.GetRotation().Conjugated();
    for (const auto &leaf : collector.mHits) {
        if (leaf.mShape->GetSubType() == JPH::EShapeSubType::Empty)
            continue;
        const auto nativeScale = leaf.GetShapeScale();
        const glm::vec3 scale(nativeScale.GetX(), nativeScale.GetY(), nativeScale.GetZ());
        std::shared_ptr<const PhysicsShapeGeometry> geometry;
        for (const auto &old : previous)
            if (old.geometry->shape.get() == leaf.mShape.GetPtr() && old.geometry->scale == scale) {
                geometry = old.geometry;
                break;
            }
        if (!geometry) {
            auto built = std::make_shared<PhysicsShapeGeometry>();
            built->identity = ++m_nextShapeGeometryIdentity;
            leaf.mShape->AddRef();
            built->shape = {leaf.mShape.GetPtr(), [](const JPH::Shape *shape) { shape->Release(); }};
            built->scale = scale;
            built->convex = leaf.mShape->GetType() == JPH::EShapeType::Convex;
            const auto bounds = leaf.mShape->GetLocalBounds().Scaled(nativeScale);
            built->boundsMin = {bounds.mMin.GetX(), bounds.mMin.GetY(), bounds.mMin.GetZ()};
            built->boundsMax = {bounds.mMax.GetX(), bounds.mMax.GetY(), bounds.mMax.GetZ()};
            switch (leaf.mShape->GetSubType()) {
            case JPH::EShapeSubType::Box: {
                const auto half =
                    static_cast<const JPH::BoxShape *>(leaf.mShape.GetPtr())->GetHalfExtent() * nativeScale.Abs();
                built->type = PhysicsShapeType::Box;
                built->dimensions = {2 * half.GetX(), 2 * half.GetY(), 2 * half.GetZ(), 0};
                break;
            }
            case JPH::EShapeSubType::Sphere:
                built->type = PhysicsShapeType::Sphere;
                built->dimensions.x =
                    static_cast<const JPH::SphereShape *>(leaf.mShape.GetPtr())->GetRadius() * nativeScale.Abs().GetX();
                break;
            case JPH::EShapeSubType::Capsule: {
                const auto *capsule = static_cast<const JPH::CapsuleShape *>(leaf.mShape.GetPtr());
                const float radius = capsule->GetRadius() * nativeScale.Abs().GetX();
                built->type = PhysicsShapeType::Capsule;
                built->dimensions = {
                    radius, 2 * (capsule->GetHalfHeightOfCylinder() * nativeScale.Abs().GetY() + radius), 1, 0};
                break;
            }
            default:
                built->type = PhysicsShapeType::Mesh;
                break;
            }
            geometry = std::move(built);
        }
        const auto position = inverseBodyRotation * JPH::Vec3(leaf.mShapePositionCOM - body.GetCenterOfMassPosition());
        const auto rotation = inverseBodyRotation * leaf.mShapeRotation;
        record.shapeLeaves.push_back({leaf.mSubShapeIDCreator.GetID().GetValue(),
                                      {position.GetX(), position.GetY(), position.GetZ()},
                                      {rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ()},
                                      std::move(geometry)});
    }
    for (auto *member : record.members)
        if (auto *mesh = dynamic_cast<MeshCollider *>(member))
            mesh->PublishCookedGeometry();
}

std::vector<PhysicsShapeState> PhysicsWorld::GetBodyShapeStates(uint32_t bodyId, bool queryTriggers) const
{
    std::shared_lock snapshotRead(m_querySnapshotMutex);
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (!lock.SucceededAndIsInBroadPhase())
        throw std::runtime_error("shape state requires an active physics body");
    const auto &body = lock.GetBody();
    const auto nativeCenter = body.GetCenterOfMassPosition();
    const glm::vec3 center(nativeCenter.GetX(), nativeCenter.GetY(), nativeCenter.GetZ());
    const auto nativeRotation = body.GetRotation();
    const glm::quat rotation(nativeRotation.GetW(), nativeRotation.GetX(), nativeRotation.GetY(),
                             nativeRotation.GetZ());
    const auto velocity = body.GetLinearVelocity();
    const auto angular = body.GetAngularVelocity();
    std::vector<PhysicsShapeState> states;
    const auto &leaves = m_bodies.at(bodyId).shapeLeaves;
    states.reserve(leaves.size());
    for (const auto &leaf : leaves) {
        const auto *identity = ResolvePublishedQueryIdentity(body, bodyId, leaf.subShapeId);
        if (!identity || (!queryTriggers && identity->isTrigger))
            continue;
        states.push_back({identity->collider,
                          identity->memberIndex,
                          identity->layerBit,
                          identity->isTrigger,
                          !body.IsStatic(),
                          leaf.geometry,
                          center + rotation * leaf.center,
                          rotation * leaf.rotation,
                          center,
                          {velocity.GetX(), velocity.GetY(), velocity.GetZ()},
                          {angular.GetX(), angular.GetY(), angular.GetZ()}});
    }
    return states;
}

std::vector<PhysicsBoxShapeState> PhysicsWorld::GetBodyBoxShapeStates(uint32_t bodyId, bool queryTriggers) const
{
    std::vector<PhysicsBoxShapeState> states;
    for (const auto &leaf : GetBodyShapeStates(bodyId, queryTriggers)) {
        if (leaf.geometry->type == PhysicsShapeType::Box && dynamic_cast<BoxCollider *>(leaf.collider))
            states.push_back({leaf.memberIndex, leaf.center, leaf.rotation, glm::vec3(leaf.geometry->dimensions) * .5f,
                              leaf.collider->GetFriction(), leaf.collider->GetBounciness()});
    }
    std::sort(states.begin(), states.end(),
              [](const auto &left, const auto &right) { return left.memberIndex < right.memberIndex; });
    return states;
}

Collider *PhysicsWorld::FindColliderByBodyId(uint32_t bodyId) const
{
    const auto found = m_bodies.find(bodyId);
    return found != m_bodies.end() && !found->second.targets.empty() ? found->second.targets.front().collider : nullptr;
}

void PhysicsWorld::PublishBodyQueryIdentitiesUnlocked(uint32_t bodyId)
{
    auto &record = m_bodies.at(bodyId);
    auto &published = record.targets;
    published.clear();
    published.reserve(record.members.size());

    // Retired members keep their original subshape indices until the next shape commit.
    for (size_t index = 0; index < record.members.size(); ++index) {
        auto *collider = record.members[index];
        if (!collider)
            continue;
        GameObject *gameObject = collider->GetGameObject();
        QueryTargetIdentity identity;
        identity.collider = collider;
        identity.gameObject = gameObject;
        identity.colliderId = collider->GetComponentID();
        identity.gameObjectId = gameObject ? gameObject->GetID() : 0;
        identity.memberIndex = static_cast<uint32_t>(index);
        identity.target = PhysicsTargetReference(collider);
        identity.isTrigger = collider->IsTrigger();
        identity.layerBit = uint32_t{1} << gameObject->GetLayer();
        published.push_back(identity);
    }
}

const PhysicsWorld::QueryTargetIdentity *
PhysicsWorld::ResolvePublishedQueryIdentity(const JPH::Body &body, uint32_t bodyId, uint32_t subShapeIdValue) const
{
    const auto published = m_bodies.find(bodyId);
    if (published == m_bodies.end() || published->second.targets.empty())
        return nullptr;

    const QueryTargetIdentity *fallback = &published->second.targets.front();

    const auto memberIndex = CompoundMemberIndex(body.GetShape(), subShapeIdValue);
    if (!memberIndex)
        return fallback;

    const auto match = std::find_if(
        published->second.targets.begin(), published->second.targets.end(),
        [memberIndex](const QueryTargetIdentity &identity) { return identity.memberIndex == *memberIndex; });
    return match != published->second.targets.end() ? &*match : nullptr;
}

bool PhysicsWorld::AcceptsSubShapeQuery(const JPH::Body &body, uint32_t subShapeId, uint32_t layerMask) const
{
    const auto *identity = ResolvePublishedQueryIdentity(body, body.GetID().GetIndexAndSequenceNumber(), subShapeId);
    return identity && (identity->layerBit & layerMask) != 0;
}

bool PhysicsWorld::AcceptsCCDCandidate(const JPH::Body &source, const JPH::Body &target, uint32_t sourceSubShape,
                                       uint32_t targetSubShape) const
{
    const auto sourceId = source.GetID().GetIndexAndSequenceNumber();
    const auto targetId = target.GetID().GetIndexAndSequenceNumber();
    const auto *a = ResolvePublishedQueryIdentity(source, sourceId, sourceSubShape);
    const auto *b = ResolvePublishedQueryIdentity(target, targetId, targetSubShape);
    return a && b && !a->isTrigger && !b->isTrigger &&
           TagLayerManager::Instance().GetLayersCollide(a->gameObject->GetLayer(), b->gameObject->GetLayer()) &&
           (!m_contactListener || (!m_contactListener->GetBodyPairIgnored(sourceId, targetId) &&
                                   !m_contactListener->GetColliderPairIgnored(a->colliderId, b->colliderId)));
}

Collider *PhysicsWorld::ResolveColliderForSubShape(uint32_t bodyId, uint32_t subShapeIdValue) const
{
    Collider *fallback = FindColliderByBodyId(bodyId);
    if (!fallback) {
        return nullptr;
    }

    JPH::BodyID joltBodyId(bodyId);
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), joltBodyId);
    if (!lock.Succeeded()) {
        return fallback;
    }

    return ResolveColliderForSubShape(lock.GetBody(), bodyId, subShapeIdValue);
}

Collider *PhysicsWorld::ResolveColliderForSubShape(const JPH::Body &body, uint32_t bodyId,
                                                   uint32_t subShapeIdValue) const
{
    Collider *fallback = FindColliderByBodyId(bodyId);
    if (!fallback) {
        return nullptr;
    }

    const auto memberIndex = CompoundMemberIndex(body.GetShape(), subShapeIdValue);
    if (!memberIndex) {
        return fallback;
    }

    const auto cached = m_bodies.find(bodyId);
    if (cached == m_bodies.end() || *memberIndex >= cached->second.members.size()) {
        return fallback;
    }
    return cached->second.members[*memberIndex];
}

void PhysicsWorld::EnsureSceneBodiesRegistered(Scene *scene)
{
    if (!m_initialized || !scene)
        return;
    auto &manager = SceneManager::Instance();
    manager.SyncTransforms();
}

void PhysicsWorld::OptimizeBroadPhase()
{
    std::unique_lock snapshotWrite(m_querySnapshotMutex);
    if (m_initialized && m_physicsSystem) {
        m_physicsSystem->OptimizeBroadPhase();
    }
}

} // namespace infernux
