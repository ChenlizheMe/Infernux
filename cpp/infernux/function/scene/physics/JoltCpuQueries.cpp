#include "JoltCpuQueries.h"

#include <Jolt/Jolt.h>
#include <Jolt/Geometry/ConvexSupport.h>
#include <Jolt/Geometry/GJKClosestPoint.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Body/BodyLock.h>
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
#include <Jolt/Physics/Collision/TransformedShape.h>
#include "JoltCpuScene.h"
#include "PhysicsLayers.h"
#include "PhysicsWorld.h"
#include <algorithm>
#include <cmath>
#include <function/scene/BoxCollider.h>
#include <function/scene/GameObject.h>
#include <function/scene/MeshCollider.h>
#include <function/scene/TransformECSStore.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace infernux
{
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
        scene.System().GetNarrowPhaseQuery().CastRay(ray, {}, collector, {}, QueryLayerFilter(filter.layers));
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
        const JPH::IgnoreSingleBodyFilter bodyFilter(sweep ? sweep->excludeBody : JPH::BodyID());
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
                              const glm::vec3 &displacement, RaycastHit &hit) const
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
    const auto shape = box.CreateJoltShape();
    if (!shape)
        throw std::runtime_error("Collider geometry is not ready for an explicit geometry query");
    if (shape->GetSubType() == JPH::EShapeSubType::Empty)
        return false;

    const Filter filter{TagLayerManager::Instance().GetLayerCollisionMask(object->GetLayer()), false};
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
            m_scene.System().GetNarrowPhaseQuery().CollideShape(&native, JPH::Vec3::sOne(), com, {},
                                                                com.GetTranslation(), collector, {},
                                                                QueryLayerFilter(filter.layers));
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
} // namespace infernux
