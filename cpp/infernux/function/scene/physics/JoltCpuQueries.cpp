#include "JoltCpuQueries.h"

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/TransformedShape.h>
#include "JoltCpuScene.h"
#include "PhysicsLayers.h"
#include "PhysicsWorld.h"
#include <algorithm>
#include <cmath>
#include <function/scene/GameObject.h>
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

template <typename Collector> class CastCollector final : public SurfaceCollector<Collector>
{
  public:
    CastCollector(const JoltCpuScene &scene, const JoltCpuQueries::Filter &filter, const JPH::RRayCast &ray,
                  float maximum, bool all, float collisionTolerance = 0.0f)
        : SurfaceCollector<Collector>(scene, filter), m_ray(ray), m_maximum(maximum), m_all(all),
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
        if (!surface)
            return;

        RaycastHit hit;
        hit.bodyId = this->m_bodyId;
        hit.subShapeId = subshape;
        hit.distance = result.mFraction * m_maximum;
        hit.colliderId = surface->colliderId;
        hit.target = surface->target;
        const auto &target = *this->GetContext();
        SetHitGeometry(hit, result, target, m_ray);
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
    float m_earlyOutSlack;
    RaycastHit m_closest;
    std::tuple<float, uint32_t, uint32_t> m_closestOrder{std::numeric_limits<float>::infinity(), UINT32_MAX,
                                                         UINT32_MAX};
    std::vector<RaycastHit> m_hits;
};

CastCollector<JPH::CastRayCollector> CastRay(const JoltCpuScene &scene, const glm::vec3 &origin,
                                             const glm::vec3 &direction, float maximum,
                                             const JoltCpuQueries::Filter &filter, bool all)
{
    const JPH::RRayCast ray(Position(origin), CastDisplacement(direction, maximum));
    CastCollector<JPH::CastRayCollector> collector(scene, filter, ray, maximum, all);
    if (filter.layers != 0)
        scene.System().GetNarrowPhaseQuery().CastRay(ray, {}, collector, {}, QueryLayerFilter(filter.layers));
    return collector;
}

CastCollector<JPH::CastShapeCollector> CastShape(const JoltCpuScene &scene, const JoltCpuQueries::Shape &shape,
                                                 const glm::vec3 &direction, float maximum,
                                                 const JoltCpuQueries::Filter &filter, bool all)
{
    const auto displacement = CastDisplacement(direction, maximum);
    return WithShape(shape, [&](const JPH::Shape &native, JPH::RMat44Arg pose) {
        const JPH::RRayCast ray(pose.GetTranslation(), displacement);
        JPH::ShapeCastSettings settings;
        settings.mReturnDeepestPoint = true;
        settings.mUseShrunkenShapeAndConvexRadius = true;
        settings.mBackFaceModeConvex = JPH::EBackFaceMode::CollideWithBackFaces;
        CastCollector<JPH::CastShapeCollector> collector(scene, filter, ray, maximum, all,
                                                         settings.mCollisionTolerance);
        if (filter.layers != 0) {
            const auto cast = JPH::RShapeCast::sFromWorldTransform(&native, JPH::Vec3::sOne(), pose, displacement);
            scene.System().GetNarrowPhaseQuery().CastShape(cast, settings, ray.mOrigin, collector, {},
                                                           QueryLayerFilter(filter.layers));
        }
        return collector;
    });
}
} // namespace

bool JoltCpuQueries::Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maximum, RaycastHit &hit,
                             const Filter &filter) const
{
    return CastRay(m_scene, origin, direction, maximum, filter, false).ClosestHit(hit);
}

std::vector<RaycastHit> JoltCpuQueries::RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maximum,
                                                   const Filter &filter) const
{
    return CastRay(m_scene, origin, direction, maximum, filter, true).TakeHits();
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
} // namespace infernux
