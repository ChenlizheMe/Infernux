#include "JoltCpuQueries.h"

#include <Jolt/Jolt.h>
#include <Jolt/Geometry/ConvexSupport.h>
#include <Jolt/Geometry/GJKClosestPoint.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/BodyLockMulti.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <Jolt/Physics/Collision/TransformedShape.h>
#include "JoltCpuScene.h"
#include "PhysicsLayers.h"
#include "PhysicsWorld.h"
#include <algorithm>
#include <cmath>
#include <function/scene/BoxCollider.h>
#include <function/scene/GameObject.h>
#include <function/scene/MeshCollider.h>
#include <function/scene/Rigidbody.h>
#include <function/scene/TransformECSStore.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace infernux
{
struct JoltCpuShapeLeaves
{
    struct Triangle
    {
        JPH::Vec3 centroid;
        PhysicsShapeGeometry geometry;
    };

    struct Leaf
    {
        JPH::TransformedShape native;
        uint32_t memberIndex;
        mutable std::shared_ptr<const PhysicsShapeGeometry> geometry;
        mutable std::shared_ptr<const std::vector<Triangle>> mesh;
    };

    std::vector<Leaf> entries;
};

namespace
{
bool Finite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

JPH::RVec3 Position(const glm::vec3 &value)
{
    if (!Finite(value))
        throw std::invalid_argument("Physics vector must be finite");
    return JPH::RVec3(value.x, value.y, value.z);
}

JPH::Vec3 CastDisplacement(const glm::vec3 &direction, float maximum)
{
    if (!Finite(direction) || !std::isfinite(maximum) || maximum <= 0.0f)
        throw std::invalid_argument("Physics cast requires a finite direction and positive finite distance");
    const double length = std::hypot(double(direction.x), double(direction.y), double(direction.z));
    if (length == 0.0)
        throw std::invalid_argument("Physics cast direction must be nonzero");
    const double scale = maximum / length;
    return JPH::Vec3(float(direction.x * scale), float(direction.y * scale), float(direction.z * scale));
}

JPH::Quat Rotation(const glm::quat &value)
{
    const float norm = glm::dot(value, value);
    if (!std::isfinite(norm) || norm <= 1e-12f)
        throw std::invalid_argument("Physics rotation must be finite and nonzero");
    const auto unit = glm::normalize(value);
    return JPH::Quat(unit.x, unit.y, unit.z, unit.w);
}

void RequireRadius(float radius)
{
    if (!std::isfinite(radius) || radius <= 0.0f)
        throw std::invalid_argument("Physics query radius must be positive and finite");
}

template <typename Operation> auto WithShape(const JoltCpuQueries::Sphere &sphere, Operation &run)
{
    const auto center = Position(sphere.center);
    RequireRadius(sphere.radius);
    return run(JPH::SphereShape(sphere.radius), JPH::RMat44::sTranslation(center));
}

template <typename Operation> auto WithShape(const JoltCpuQueries::Box &box, Operation &run)
{
    const auto center = Position(box.center);
    const auto &half = box.halfExtents;
    if (!Finite(half) || half.x <= 0.0f || half.y <= 0.0f || half.z <= 0.0f)
        throw std::invalid_argument("Physics query box requires positive finite half extents");
    const float convexRadius = std::min({0.05f, half.x, half.y, half.z});
    const JPH::BoxShape shape(JPH::Vec3(half.x, half.y, half.z), convexRadius);
    return run(shape, JPH::RMat44::sRotationTranslation(Rotation(box.rotation), center));
}

template <typename Operation> auto WithShape(const JoltCpuQueries::Capsule &capsule, Operation &run)
{
    const auto a = Position(capsule.point0);
    const auto b = Position(capsule.point1);
    RequireRadius(capsule.radius);
    const auto center = a * 0.5f + b * 0.5f;
    const glm::dvec3 delta = glm::dvec3(capsule.point1) - glm::dvec3(capsule.point0);
    const double radial = std::hypot(delta.x, delta.z);
    const double halfLength = std::hypot(radial, delta.y) * 0.5;
    if (halfLength > std::numeric_limits<float>::max())
        throw std::invalid_argument("Physics query capsule is too long");
    if (static_cast<float>(halfLength) == 0.0f)
        return run(JPH::SphereShape(capsule.radius), JPH::RMat44::sTranslation(center));

    auto rotation = JPH::Quat::sIdentity();
    if (radial > 0.0) {
        const double halfAngle = std::atan2(radial, delta.y) * 0.5;
        const double sine = std::sin(halfAngle);
        rotation = JPH::Quat(float(delta.z / radial * sine), 0.0f, float(-delta.x / radial * sine),
                             float(std::cos(halfAngle)));
    }
    // A capsule's geometry is symmetric about its center, including when its
    // endpoint direction is exactly -Y.
    const JPH::CapsuleShape shape(static_cast<float>(halfLength), capsule.radius);
    return run(shape, JPH::RMat44::sRotationTranslation(rotation, center));
}

template <typename Operation> auto WithShape(const JoltCpuQueries::Shape &shape, Operation run)
{
    return std::visit([&](const auto &primitive) { return WithShape(primitive, run); }, shape);
}

template <typename Vec> glm::vec3 Vector(const Vec &value)
{
    return {float(value.GetX()), float(value.GetY()), float(value.GetZ())};
}

glm::quat Quaternion(JPH::QuatArg value)
{
    return {value.GetW(), value.GetX(), value.GetY(), value.GetZ()};
}

JPH::TransformedShape UnwrapDecorated(JPH::TransformedShape shape)
{
    while (shape.mShape->GetType() == JPH::EShapeType::Decorated) {
        JPH::SubShapeID remainder;
        shape = shape.GetSubShapeTransformedShape({}, remainder);
    }
    return shape;
}

std::optional<glm::vec3> ClosestPointOnShape(const JPH::TransformedShape &shape, JPH::RVec3Arg point)
{
    const JPH::Vec3 local(shape.GetInverseCenterOfMassTransform() * point);
    const auto scale = shape.GetShapeScale();
    JPH::Vec3 nearest;
    if (shape.mShape->GetType() == JPH::EShapeType::Convex) {
        const auto &convex = static_cast<const JPH::ConvexShape &>(*shape.mShape);
        JPH::ConvexShape::SupportBuffer buffer;
        const auto *support =
            convex.GetSupportFunction(JPH::ConvexShape::ESupportMode::IncludeConvexRadius, buffer, scale);
        JPH::GJKClosestPoint query;
        JPH::Vec3 axis = JPH::Vec3::sAxisX(), queryWitness;
        const float distanceSq =
            query.GetClosestPoints(*support, JPH::PointConvexSupport{local}, JPH::cDefaultCollisionTolerance, FLT_MAX,
                                   axis, nearest, queryWitness);
        if (distanceSq == 0.0f)
            return Vector(point);
        if (distanceSq == FLT_MAX)
            throw std::runtime_error("Closest-point query exceeds native numerical range");
    } else if (shape.mShape->GetSubType() == JPH::EShapeSubType::Mesh) {
        const auto &mesh = static_cast<const JPH::MeshShape &>(*shape.mShape);
        if (!mesh.GetClosestSurfacePoint(local, scale, nearest))
            return std::nullopt;
    } else {
        return std::nullopt;
    }
    return Vector(shape.GetCenterOfMassTransform() * nearest);
}

bool IncludeLayer(uint32_t mask, uint32_t layer)
{
    return layer < PhysicsObjectLayers::LAYER_COUNT && (mask & (uint32_t{1} << layer)) != 0;
}

class QueryLayerFilter final : public JPH::ObjectLayerFilter
{
  public:
    explicit QueryLayerFilter(uint32_t layers) : m_layers(layers)
    {
    }

    bool ShouldCollide(JPH::ObjectLayer layer) const override
    {
        return PhysicsObjectLayers::IsMixed(layer) ||
               IncludeLayer(m_layers, PhysicsObjectLayers::DecodeGameLayer(layer));
    }

  private:
    uint32_t m_layers;
};

class QueryBodyFilter final : public JPH::BodyFilter
{
  public:
    explicit QueryBodyFilter(const JoltCpuQueries::Filter &filter, JPH::BodyID exclude = {})
        : m_filter(filter), m_exclude(exclude)
    {
    }

    bool ShouldCollide(const JPH::BodyID &body) const override
    {
        return body != m_exclude;
    }

    bool ShouldCollideLocked(const JPH::Body &body) const override
    {
        return !m_filter.excludeBody || !m_filter.excludeBody(static_cast<uint32_t>(body.GetUserData()));
    }

  private:
    const JoltCpuQueries::Filter &m_filter;
    JPH::BodyID m_exclude;
};

template <typename Collector> class SurfaceCollector : public Collector
{
  public:
    SurfaceCollector(const JoltCpuScene &scene, const JoltCpuQueries::Filter &filter) : m_scene(scene), m_filter(filter)
    {
    }

    void OnBody(const JPH::Body &body) override
    {
        m_bodyId = static_cast<uint32_t>(body.GetUserData());
    }

  protected:
    const JoltCpuScene::Surface *AcceptedSurface(JPH::SubShapeID subshape) const
    {
        const auto &surface = m_scene.GetSurface(m_bodyId, subshape);
        if (!IncludeLayer(m_filter.layers, surface.layer) || (!m_filter.triggers && surface.sensor))
            return nullptr;
        if (m_filter.target &&
            (surface.colliderId != m_filter.target->id || surface.generation != m_filter.target->generation))
            return nullptr;
        return &surface;
    }

    uint32_t m_bodyId = UINT32_MAX;

  private:
    const JoltCpuScene &m_scene;
    const JoltCpuQueries::Filter &m_filter;
};

class OverlapCollector final : public SurfaceCollector<JPH::CollideShapeCollector>
{
  public:
    using SurfaceCollector::SurfaceCollector;

    void AddHit(const JPH::CollideShapeResult &result) override
    {
        if (const auto *surface = AcceptedSurface(result.mSubShapeID2))
            m_targets.try_emplace({surface->colliderId, surface->generation}, surface->target);
    }

    std::vector<Collider *> Colliders() const
    {
        std::vector<Collider *> colliders;
        colliders.reserve(m_targets.size());
        for (const auto &entry : m_targets)
            if (auto *collider = entry.second.GetCollider())
                colliders.push_back(collider);
        return colliders;
    }

  private:
    std::map<std::pair<uint64_t, uint64_t>, PhysicsTargetReference> m_targets;
};

void SetHitGeometry(RaycastHit &hit, const JPH::RayCastResult &result, const JPH::TransformedShape &target,
                    const JPH::RRayCast &ray)
{
    const auto point = ray.GetPointOnRay(result.mFraction);
    hit.point = Vector(point);
    hit.normal = Vector(target.GetWorldSpaceSurfaceNormal(result.mSubShapeID2, point));
}

void SetHitGeometry(RaycastHit &hit, const JPH::ShapeCastResult &result, const JPH::TransformedShape &target,
                    const JPH::RRayCast &ray)
{
    const auto point = ray.mOrigin + JPH::RVec3(result.mContactPointOn2);
    hit.point = Vector(point);
    hit.normal = result.mPenetrationAxis.LengthSq() > 0.0f
                     ? Vector(-result.mPenetrationAxis.Normalized())
                     : Vector(target.GetWorldSpaceSurfaceNormal(result.mSubShapeID2, point));
}

void ResolveTarget(RaycastHit &hit)
{
    hit.collider = hit.target.GetCollider();
    hit.gameObject = hit.target.GetGameObject();
    if (hit.gameObject)
        hit.gameObjectId = hit.gameObject->GetID();
}

struct SweepPolicy
{
    JPH::BodyID excludeBody;
    uint64_t object;
    JoltCpuQueries::ColliderIdentity collider;
    const PhysicsColliderPairs *ignored;

    bool Accepts(const JoltCpuScene::Surface &surface) const
    {
        if (const auto *owner = surface.target.GetGameObject(); owner && owner->GetID() == object)
            return false;
        if (!ignored)
            return true;
        auto a = std::pair(collider.id, collider.generation);
        auto b = std::pair(surface.colliderId, surface.generation);
        if (b < a)
            std::swap(a, b);
        const PhysicsColliderPair pair{a.first, a.second, b.first, b.second};
        return !std::binary_search(ignored->begin(), ignored->end(), pair);
    }
};

template <typename Collector> class CastCollector final : public SurfaceCollector<Collector>
{
  public:
    CastCollector(const JoltCpuScene &scene, const JoltCpuQueries::Filter &filter, const JPH::RRayCast &ray,
                  float maximum, bool all, float collisionTolerance = 0.0f, const SweepPolicy *sweep = nullptr)
        : SurfaceCollector<Collector>(scene, filter), m_ray(ray), m_maximum(maximum), m_all(all), m_sweep(sweep),
          m_earlyOutSlack(std::numeric_limits<float>::epsilon() +
                          float(std::min(1.0, double(collisionTolerance) / maximum)))
    {
    }

    void AddHit(const typename Collector::ResultType &result) override
    {
        const uint32_t subshape = result.mSubShapeID2.GetValue();
        const auto order = std::tuple(result.mFraction, this->m_bodyId, subshape);
        if (!(result.mFraction >= 0.0f && result.mFraction <= 1.0f) || (!m_all && order >= m_closestOrder))
            return;
        const auto *surface = this->AcceptedSurface(result.mSubShapeID2);
        if (!surface || (m_sweep && !m_sweep->Accepts(*surface)))
            return;

        RaycastHit hit;
        hit.bodyId = this->m_bodyId;
        hit.subShapeId = subshape;
        hit.distance = result.mFraction * m_maximum;
        hit.colliderId = surface->colliderId;
        hit.target = surface->target;
        const auto &target = *this->GetContext();
        SetHitGeometry(hit, result, target, m_ray);
        if (m_sweep && glm::dot(hit.normal, Vector(m_ray.mDirection) / m_maximum) >= -1.0e-3f)
            return;
        JPH::SubShapeID remainder;
        const auto *leaf = target.mShape->GetLeafShape(result.mSubShapeID2, remainder);
        if (leaf && leaf->GetSubType() == JPH::EShapeSubType::Mesh)
            hit.triangleIndex = static_cast<const JPH::MeshShape *>(leaf)->GetTriangleUserData(remainder);

        if (m_all) {
            m_hits.push_back(std::move(hit));
        } else {
            m_closest = std::move(hit);
            m_closestOrder = order;
            // Dispatched casts can converge slightly before the broadphase
            // bound. Keep coincident hits eligible for the identity tiebreak.
            const float bound = std::min(this->GetEarlyOutFraction(), result.mFraction + m_earlyOutSlack);
            this->UpdateEarlyOutFraction(bound);
        }
    }

    bool ClosestHit(RaycastHit &hit) const
    {
        hit = m_closest;
        if (hit.bodyId == UINT32_MAX)
            return false;
        ResolveTarget(hit);
        return true;
    }

    std::vector<RaycastHit> TakeHits()
    {
        std::sort(m_hits.begin(), m_hits.end(), [](const RaycastHit &a, const RaycastHit &b) {
            return std::tie(a.distance, a.bodyId, a.subShapeId) < std::tie(b.distance, b.bodyId, b.subShapeId);
        });
        for (auto &hit : m_hits)
            ResolveTarget(hit);
        return std::move(m_hits);
    }

  private:
    JPH::RRayCast m_ray;
    float m_maximum;
    bool m_all;
    const SweepPolicy *m_sweep;
    float m_earlyOutSlack;
    RaycastHit m_closest;
    std::tuple<float, uint32_t, uint32_t> m_closestOrder{std::numeric_limits<float>::infinity(), UINT32_MAX,
                                                         UINT32_MAX};
    std::vector<RaycastHit> m_hits;
};

CastCollector<JPH::CastRayCollector> CastRay(const JoltCpuScene &scene, const JPH::RRayCast &ray, float maximum,
                                             const JoltCpuQueries::Filter &filter, bool all)
{
    CastCollector<JPH::CastRayCollector> collector(scene, filter, ray, maximum, all);
    if (filter.layers != 0)
        scene.System().GetNarrowPhaseQuery().CastRay(ray, {}, collector, {}, QueryLayerFilter(filter.layers),
                                                     QueryBodyFilter(filter));
    return collector;
}

CastCollector<JPH::CastShapeCollector> CastShape(const JoltCpuScene &scene, const JPH::Shape &native,
                                                 JPH::RMat44Arg pose, JPH::Vec3Arg displacement, float maximum,
                                                 const JoltCpuQueries::Filter &filter, bool all,
                                                 const SweepPolicy *sweep = nullptr)
{
    const JPH::RRayCast ray(pose.GetTranslation(), displacement);
    JPH::ShapeCastSettings settings;
    settings.mReturnDeepestPoint = true;
    settings.mUseShrunkenShapeAndConvexRadius = true;
    settings.mBackFaceModeConvex = JPH::EBackFaceMode::CollideWithBackFaces;
    CastCollector<JPH::CastShapeCollector> collector(scene, filter, ray, maximum, all, settings.mCollisionTolerance,
                                                     sweep);
    if (filter.layers != 0) {
        const auto cast = JPH::RShapeCast::sFromWorldTransform(&native, JPH::Vec3::sOne(), pose, displacement);
        const QueryBodyFilter bodyFilter(filter, sweep ? sweep->excludeBody : JPH::BodyID());
        scene.System().GetNarrowPhaseQuery().CastShape(cast, settings, ray.mOrigin, collector, {},
                                                       QueryLayerFilter(filter.layers), bodyFilter);
    }
    return collector;
}

CastCollector<JPH::CastShapeCollector> CastShape(const JoltCpuScene &scene, const JoltCpuQueries::Shape &shape,
                                                 const glm::vec3 &direction, float maximum,
                                                 const JoltCpuQueries::Filter &filter, bool all)
{
    const auto displacement = CastDisplacement(direction, maximum);
    return WithShape(shape, [&](const JPH::Shape &native, JPH::RMat44Arg pose) {
        return CastShape(scene, native, pose, displacement, maximum, filter, all);
    });
}
} // namespace

bool JoltCpuQueries::Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maximum, RaycastHit &hit,
                             const Filter &filter) const
{
    const JPH::RRayCast ray(Position(origin), CastDisplacement(direction, maximum));
    return CastRay(m_scene, ray, maximum, filter, false).ClosestHit(hit);
}

void JoltCpuQueries::RaycastBatch(const float *origins, const float *directions, size_t count, float maximum,
                                  RaycastHit *hits, uint8_t *mask, const Filter &filter) const
{
    if (count == 0)
        return;
    if (!origins || !directions || !hits || !mask || count > UINT32_MAX)
        throw std::invalid_argument("Raycast batch requires valid arrays within the query index range");
    std::vector<JPH::RRayCast> rays;
    rays.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto *p = origins + i * 3, *d = directions + i * 3;
        rays.emplace_back(Position({p[0], p[1], p[2]}), CastDisplacement({d[0], d[1], d[2]}, maximum));
    }
    for (size_t i = 0; i < count; ++i)
        mask[i] = CastRay(m_scene, rays[i], maximum, filter, false).ClosestHit(hits[i]) ? 1 : 0;
}

bool JoltCpuQueries::SweepBox(const BoxCollider &box, const glm::vec3 &position, const glm::quat &rotation,
                              const glm::vec3 &displacement, RaycastHit &hit, const BodyExclusion &exclude,
                              const JPH::Shape *capturedShape) const
{
    const auto pose = JPH::RMat44::sRotationTranslation(Rotation(rotation), Position(position));
    if (!Finite(displacement))
        throw std::invalid_argument("BoxCollider sweep displacement must be finite");
    const double distance = std::hypot(double(displacement.x), double(displacement.y), double(displacement.z));
    hit = {};
    const auto *object = box.GetGameObject();
    if (distance == 0.0 || !object)
        return false;
    if (distance > std::numeric_limits<float>::max())
        throw std::invalid_argument("BoxCollider sweep displacement exceeds native coordinates");
    const TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto shape = capturedShape ? JPH::RefConst<JPH::Shape>(capturedShape) : box.CreateJoltShape();
    if (!shape)
        throw std::runtime_error("Collider geometry is not ready for an explicit geometry query");
    if (shape->GetSubType() == JPH::EShapeSubType::Empty)
        return false;

    const Filter filter{TagLayerManager::Instance().GetLayerCollisionMask(object->GetLayer()), false, {}, exclude};
    SweepPolicy sweep{JPH::BodyID(),
                      object->GetID(),
                      {box.GetComponentID(), box.GetLifetimeGeneration()},
                      m_scene.m_ignoredPairs.get()};
    const auto accepted = m_scene.m_bodies.find(box.GetBodyId());
    if (accepted != m_scene.m_bodies.end() &&
        std::any_of(accepted->second.surfaces.begin(), accepted->second.surfaces.end(), [&](const auto &surface) {
            return surface.colliderId == sweep.collider.id && surface.generation == sweep.collider.generation;
        }))
        sweep.excludeBody = accepted->second.native;
    return CastShape(m_scene, *shape, pose, JPH::Vec3(displacement.x, displacement.y, displacement.z),
                     static_cast<float>(distance), filter, false, &sweep)
        .ClosestHit(hit);
}

std::vector<RaycastHit> JoltCpuQueries::RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maximum,
                                                   const Filter &filter) const
{
    const JPH::RRayCast ray(Position(origin), CastDisplacement(direction, maximum));
    return CastRay(m_scene, ray, maximum, filter, true).TakeHits();
}

std::vector<Collider *> JoltCpuQueries::Overlap(const Shape &shape, const Filter &filter) const
{
    return WithShape(shape, [&](const JPH::Shape &native, JPH::RMat44Arg pose) {
        OverlapCollector collector(m_scene, filter);
        if (filter.layers != 0) {
            const auto com = pose.PreTranslated(native.GetCenterOfMass());
            m_scene.System().GetNarrowPhaseQuery().CollideShape(
                &native, JPH::Vec3::sOne(), com, {}, com.GetTranslation(), collector, {},
                QueryLayerFilter(filter.layers), QueryBodyFilter(filter));
        }
        return collector.Colliders();
    });
}

bool JoltCpuQueries::Cast(const Shape &shape, const glm::vec3 &direction, float maximum, RaycastHit &hit,
                          const Filter &filter) const
{
    return CastShape(m_scene, shape, direction, maximum, filter, false).ClosestHit(hit);
}

std::vector<RaycastHit> JoltCpuQueries::CastAll(const Shape &shape, const glm::vec3 &direction, float maximum,
                                                const Filter &filter) const
{
    return CastShape(m_scene, shape, direction, maximum, filter, true).TakeHits();
}

std::optional<glm::vec3> JoltCpuQueries::ClosestPoint(uint32_t bodyId, ColliderIdentity target,
                                                      const glm::vec3 &point) const
{
    const auto position = Position(point);
    const auto actor = m_scene.m_bodies.find(bodyId);
    if (actor == m_scene.m_bodies.end())
        return std::nullopt;
    const auto &record = actor->second;
    const auto surface = std::find_if(record.surfaces.begin(), record.surfaces.end(), [&](const auto &entry) {
        return entry.colliderId == target.id && entry.generation == target.generation;
    });
    if (surface == record.surfaces.end())
        return std::nullopt;

    JPH::TransformedShape shape;
    {
        const JPH::BodyLockRead lock(m_scene.System().GetBodyLockInterface(), record.native);
        if (!lock.SucceededAndIsInBroadPhase())
            return std::nullopt;
        shape = lock.GetBody().GetTransformedShape();
    }
    shape = UnwrapDecorated(std::move(shape));
    if (const auto *compound = record.memberCompound) {
        const auto member = static_cast<uint32_t>(surface - record.surfaces.begin());
        for (JPH::uint i = 0; i < compound->GetNumSubShapes(); ++i)
            if (compound->GetCompoundUserData(i) == member) {
                const auto id = compound->GetSubShapeIDFromIndex(static_cast<int>(i), {}).GetID();
                JPH::SubShapeID remainder;
                return ClosestPointOnShape(UnwrapDecorated(shape.GetSubShapeTransformedShape(id, remainder)), position);
            }
        return std::nullopt;
    }
    return ClosestPointOnShape(shape, position);
}

std::optional<PhysicsPenetrationResult>
JoltCpuQueries::ComputePenetration(const Collider &a, const glm::vec3 &positionA, const glm::quat &rotationA,
                                   const Collider &b, const glm::vec3 &positionB, const glm::quat &rotationB)
{
    const auto originA = Position(positionA), originB = Position(positionB);
    const auto orientationA = Rotation(rotationA), orientationB = Rotation(rotationB);
    const TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto geometry = [](const Collider &collider) {
        if (const auto *mesh = dynamic_cast<const MeshCollider *>(&collider))
            return mesh->GetPublishedConvexQueryShape();
        const auto shape = collider.CreateJoltShape();
        if (!shape)
            throw std::runtime_error("Collider geometry is not ready for an explicit geometry query");
        return shape;
    };
    const auto first = geometry(a), second = geometry(b);
    if (first->GetSubType() == JPH::EShapeSubType::Empty || second->GetSubType() == JPH::EShapeSubType::Empty)
        return std::nullopt;
    const auto comA = JPH::RMat44::sRotationTranslation(orientationA, originA).PreTranslated(first->GetCenterOfMass());
    const JPH::TransformedShape target(originB + orientationB * second->GetCenterOfMass(), orientationB,
                                       second.GetPtr(), JPH::BodyID());
    JPH::ClosestHitCollisionCollector<JPH::CollideShapeCollector> hits;
    target.CollideShape(first.GetPtr(), JPH::Vec3::sOne(), comA, {}, originA, hits);
    if (!hits.HadHit() || hits.mHit.mPenetrationDepth <= 0.0f)
        return std::nullopt;
    const auto &hit = hits.mHit;
    return PhysicsPenetrationResult{Vector(-hit.mPenetrationAxis.Normalized()), hit.mPenetrationDepth,
                                    Vector(originA + hit.mContactPointOn1), Vector(originA + hit.mContactPointOn2)};
}

const JoltCpuShapeLeaves &JoltCpuScene::BodyRecord::QueryLeaves() const
{
    if (!queryLeaves) {
        auto catalog = std::make_shared<JoltCpuShapeLeaves>();
        if (!surfaces.empty()) {
            JPH::AllHitCollisionCollector<JPH::TransformedShapeCollector> collector;
            shape->CollectTransformedShapes(JPH::AABox::sBiggest(), JPH::Vec3::sZero(), JPH::Quat::sIdentity(),
                                            JPH::Vec3::sOne(), JPH::SubShapeIDCreator(), collector, JPH::ShapeFilter());
            catalog->entries.reserve(collector.mHits.size());
            for (const auto &leaf : collector.mHits) {
                if (leaf.mShape->GetSubType() == JPH::EShapeSubType::Empty)
                    continue;
                if (leaf.GetShapeScale() != JPH::Vec3::sOne())
                    throw std::logic_error("Accepted collider geometry must have its scale baked into the leaf");
                const auto member = MemberIndex(leaf.mSubShapeIDCreator.GetID());
                catalog->entries.push_back({leaf, member, {}, {}});
            }
        }
        queryLeaves = std::move(catalog);
    }
    return *queryLeaves;
}

void JoltCpuScene::BodyRecord::ReuseQueryGeometry(const BodyRecord &previous)
{
    if (!previous.queryLeaves)
        return;
    if (shape.GetPtr() == previous.shape.GetPtr()) {
        queryLeaves = previous.queryLeaves;
        return;
    }

    // Rebuilding a compound/decorator can leave the accepted leaf shapes intact.
    std::unordered_map<const JPH::Shape *, const JoltCpuShapeLeaves::Leaf *> cached;
    for (const auto &leaf : previous.queryLeaves->entries)
        if (leaf.geometry || leaf.mesh)
            cached.emplace(leaf.native.mShape.GetPtr(), &leaf);
    if (cached.empty())
        return;
    for (const auto &leaf : QueryLeaves().entries) {
        if (const auto found = cached.find(leaf.native.mShape.GetPtr()); found != cached.end()) {
            leaf.geometry = found->second->geometry;
            leaf.mesh = found->second->mesh;
        }
    }
}

std::shared_ptr<const PhysicsShapeGeometry> JoltCpuQueries::ResolveGeometry(const JPH::Shape &shape)
{
    auto geometry = std::make_shared<PhysicsShapeGeometry>();
    auto owner = std::make_shared<JPH::RefConst<JPH::Shape>>(&shape);
    geometry->shape = std::shared_ptr<const JPH::Shape>(std::move(owner), &shape);
    geometry->identity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&shape));
    geometry->convex = shape.GetType() == JPH::EShapeType::Convex;
    const auto bounds = shape.GetLocalBounds();
    geometry->boundsMin = Vector(bounds.mMin);
    geometry->boundsMax = Vector(bounds.mMax);
    switch (shape.GetSubType()) {
    case JPH::EShapeSubType::Box: {
        const auto &box = static_cast<const JPH::BoxShape &>(shape);
        geometry->type = PhysicsShapeType::Box;
        geometry->dimensions = glm::vec4(2.0f * Vector(box.GetHalfExtent()), box.GetConvexRadius());
        break;
    }
    case JPH::EShapeSubType::Sphere:
        geometry->type = PhysicsShapeType::Sphere;
        geometry->dimensions.x = static_cast<const JPH::SphereShape &>(shape).GetRadius();
        break;
    case JPH::EShapeSubType::Capsule: {
        const auto &capsule = static_cast<const JPH::CapsuleShape &>(shape);
        const float radius = capsule.GetRadius();
        geometry->type = PhysicsShapeType::Capsule;
        geometry->dimensions = {radius, 2.0f * (capsule.GetHalfHeightOfCylinder() + radius), 1.0f, 0.0f};
        break;
    }
    default:
        // Cylinders and convex hulls expose their
        // immutable surface through GetTriangles.
        geometry->type = PhysicsShapeType::Mesh;
        break;
    }
    return geometry;
}

void JoltCpuQueries::CacheMeshGeometry(const JoltCpuShapeLeaves &leaves, size_t index)
{
    const auto &leaf = leaves.entries[index];
    if (leaf.mesh)
        return;
    const auto &mesh = static_cast<const JPH::MeshShape &>(*leaf.native.mShape);
    JPH::Array<JPH::MeshShape::TriangleData> triangles;
    mesh.GetTriangleData(triangles);
    auto view = std::make_shared<std::vector<JoltCpuShapeLeaves::Triangle>>(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto &source = triangles[i];
        auto &triangle = (*view)[i];
        triangle.centroid = (source.mV0 + source.mV1 + source.mV2) / 3.0f;
        const auto a = source.mV0 - triangle.centroid;
        const auto b = source.mV1 - triangle.centroid;
        const auto c = source.mV2 - triangle.centroid;
        auto &geometry = triangle.geometry;
        geometry.identity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&geometry));
        geometry.type = PhysicsShapeType::Mesh;
        geometry.convex = true;
        geometry.boundsMin = Vector(JPH::Vec3::sMin(a, JPH::Vec3::sMin(b, c)));
        geometry.boundsMax = Vector(JPH::Vec3::sMax(a, JPH::Vec3::sMax(b, c)));
        geometry.triangle = std::array{Vector(a), Vector(b), Vector(c)};
    }
    leaf.mesh = std::move(view);
}

namespace
{
template <typename Append>
void VisitBodyBoxes(const JPH::Body &body, const JoltCpuShapeLeaves &leaves,
                    const std::vector<PreparedCpuActor::Surface> &surfaces, bool triggers, Append &&append)
{
    const auto position = body.GetCenterOfMassPosition();
    const auto rotation = body.GetRotation();
    for (const auto &leaf : leaves.entries) {
        if (leaf.native.mShape->GetSubType() != JPH::EShapeSubType::Box)
            continue;
        const auto &surface = surfaces.at(leaf.memberIndex);
        if (surface.sensor && !triggers)
            continue;
        const auto &box = static_cast<const JPH::BoxShape &>(*leaf.native.mShape);
        const auto center = position + rotation * JPH::Vec3(leaf.native.mShapePositionCOM);
        const auto orientation = (rotation * leaf.native.mShapeRotation).Normalized();
        append(PhysicsBoxShapeState{leaf.memberIndex, Vector(center), Quaternion(orientation),
                                    Vector(box.GetHalfExtent()), surface.friction, surface.restitution});
    }
}
} // namespace

std::vector<Rigidbody *> JoltCpuQueries::QueryRigidbodiesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum,
                                                                  const Filter &filter) const
{
    const auto low = Position(minimum), high = Position(maximum);
    if (glm::any(glm::greaterThan(minimum, maximum)))
        throw std::invalid_argument("Physics query bounds require minimum <= maximum");
    if (!filter.layers)
        return {};
    const JPH::AABox bounds{JPH::Vec3(low), JPH::Vec3(high)};
    JPH::AllHitCollisionCollector<JPH::CollideShapeBodyCollector> hits;
    m_scene.System().GetBroadPhaseQuery().CollideAABox(bounds, hits, {}, QueryLayerFilter(filter.layers));
    std::vector<Rigidbody *> bodies;
    bodies.reserve(hits.mHits.size());
    for (const auto nativeId : hits.mHits) {
        const JPH::BodyLockRead lock(m_scene.System().GetBodyLockInterface(), nativeId);
        if (!lock.SucceededAndIsInBroadPhase())
            continue;
        const auto &body = lock.GetBody();
        const auto id = static_cast<uint32_t>(body.GetUserData());
        const auto &record = m_scene.m_bodies.at(id);
        auto *rigidbody = static_cast<Rigidbody *>(record.bodyTarget.GetComponent());
        if (!rigidbody || !bounds.Overlaps(body.GetWorldSpaceBounds()) ||
            (filter.excludeBody && filter.excludeBody(id)))
            continue;
        if (std::any_of(record.surfaces.begin(), record.surfaces.end(), [&](const auto &surface) {
                return IncludeLayer(filter.layers, surface.layer) && (filter.triggers || !surface.sensor);
            }))
            bodies.push_back(rigidbody);
    }
    return bodies;
}

std::vector<PhysicsBoxShapeState> JoltCpuQueries::GatherBodyBoxes(uint32_t bodyId, bool triggers) const
{
    const auto found = m_scene.m_bodies.find(bodyId);
    if (found == m_scene.m_bodies.end())
        throw std::invalid_argument("CPU box states require an accepted body identity");
    const JPH::BodyLockRead lock(m_scene.System().GetBodyLockInterface(), found->second.native);
    std::vector<PhysicsBoxShapeState> result;
    const auto &record = found->second;
    if (lock.SucceededAndIsInBroadPhase())
        VisitBodyBoxes(lock.GetBody(), record.QueryLeaves(), record.surfaces, triggers,
                       [&](PhysicsBoxShapeState shape) { result.push_back(std::move(shape)); });
    return result;
}

PhysicsRigidbodyStateBatch JoltCpuQueries::GatherRigidbodyStates(const std::vector<BodyRequest> &requests,
                                                                 PhysicsStateFields fields, bool includeBoxes,
                                                                 bool triggers) const
{
    PhysicsRigidbodyStateBatch batch;
    if (requests.empty())
        return batch;
    if (requests.size() > size_t(std::numeric_limits<int>::max()))
        throw std::length_error("CPU rigidbody state batch exceeds native lock capacity");

    batch.bodies.reserve(requests.size());
    batch.states.reserve(requests.size());
    std::vector<const JoltCpuScene::BodyRecord *> records;
    std::vector<JPH::BodyID> nativeIds;
    records.reserve(requests.size());
    nativeIds.reserve(requests.size());
    for (const auto &request : requests) {
        const auto found = m_scene.m_bodies.find(request.bodyId);
        if (!request.rigidbody || found == m_scene.m_bodies.end() ||
            found->second.bodyTarget.GetComponent() != request.rigidbody)
            throw std::invalid_argument("CPU state batch requires the accepted Rigidbody identity");
        batch.bodies.push_back(request.rigidbody);
        records.push_back(&found->second);
        nativeIds.push_back(found->second.native);
    }

    const auto &interface = m_scene.System().GetBodyLockInterface();
    const JPH::BodyLockMultiRead lock(interface, nativeIds.data(), static_cast<int>(nativeIds.size()));
    for (uint32_t row = 0; row < requests.size(); ++row) {
        const auto &record = *records[row];
        const auto *body = lock.GetBody(static_cast<int>(row));
        if (!body)
            throw std::logic_error("CPU state batch lost an accepted body");
        batch.states.push_back(JoltCpuScene::SampleMotionState(requests[row].bodyId, record, *body, fields));
        if (includeBoxes && body->IsInBroadPhase())
            VisitBodyBoxes(*body, record.QueryLeaves(), record.surfaces, triggers,
                           [&](PhysicsBoxShapeState shape) { batch.boxes.push_back({row, std::move(shape)}); });
    }
    batch.boxCount = batch.boxes.size();
    return batch;
}

std::vector<PhysicsShapeState> JoltCpuQueries::GatherBodyShapes(uint32_t bodyId, bool triggers) const
{
    const auto found = m_scene.m_bodies.find(bodyId);
    if (found == m_scene.m_bodies.end())
        throw std::invalid_argument("CPU shape states require an accepted body identity");
    const auto &record = found->second;
    const JPH::BodyLockRead lock(m_scene.System().GetBodyLockInterface(), record.native);
    if (!lock.SucceededAndIsInBroadPhase())
        return {};

    const auto &body = lock.GetBody();
    const auto position = body.GetCenterOfMassPosition();
    const auto rotation = body.GetRotation();
    const auto linear = Vector(body.GetLinearVelocity()), angular = Vector(body.GetAngularVelocity());
    const auto &leaves = record.QueryLeaves();
    size_t rowCount = 0;
    for (size_t index = 0; index < leaves.entries.size(); ++index) {
        const auto &leaf = leaves.entries[index];
        if (record.surfaces.at(leaf.memberIndex).sensor && !triggers)
            continue;
        if (leaf.native.mShape->GetSubType() == JPH::EShapeSubType::Mesh) {
            CacheMeshGeometry(leaves, index);
            rowCount += leaf.mesh->size();
        } else {
            ++rowCount;
        }
    }

    std::vector<PhysicsShapeState> result;
    result.reserve(rowCount);
    for (const auto &leaf : leaves.entries) {
        const auto &surface = record.surfaces.at(leaf.memberIndex);
        if (surface.sensor && !triggers)
            continue;
        PhysicsShapeState state;
        state.collider = surface.target.GetCollider();
        state.memberIndex = leaf.memberIndex;
        state.layerBit = 1u << surface.layer;
        state.trigger = surface.sensor;
        state.moving = !body.IsStatic();
        const auto center = position + rotation * JPH::Vec3(leaf.native.mShapePositionCOM);
        const auto orientation = (rotation * leaf.native.mShapeRotation).Normalized();
        state.rotation = Quaternion(orientation);
        state.bodyCenterOfMass = Vector(position);
        state.linearVelocity = linear;
        state.angularVelocity = angular;
        state.friction = surface.friction;
        state.bounciness = surface.restitution;
        state.frictionCombine = static_cast<uint32_t>(surface.frictionCombine);
        state.bounceCombine = static_cast<uint32_t>(surface.bounceCombine);
        if (leaf.mesh) {
            for (const auto &triangle : *leaf.mesh) {
                state.center = Vector(center + orientation * triangle.centroid);
                state.geometry = std::shared_ptr<const PhysicsShapeGeometry>(leaf.mesh, &triangle.geometry);
                result.push_back(state);
            }
        } else {
            if (!leaf.geometry)
                leaf.geometry = ResolveGeometry(*leaf.native.mShape);
            state.geometry = leaf.geometry;
            state.center = Vector(center);
            result.push_back(std::move(state));
        }
    }
    return result;
}
} // namespace infernux
