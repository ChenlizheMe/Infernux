#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace infernux
{

class BoxCollider;
class Collider;
class JoltCpuScene;
struct RaycastHit;
struct PhysicsPenetrationResult;

/// Queries against one published native world. The caller keeps its bodies and
/// surface metadata stable throughout the query.
class JoltCpuQueries final
{
  public:
    struct ColliderIdentity
    {
        uint64_t id;
        uint64_t generation;
    };

    struct Filter
    {
        uint32_t layers;
        bool triggers;
        std::optional<ColliderIdentity> target = std::nullopt;
    };

    struct Sphere
    {
        glm::vec3 center;
        float radius;
    };

    struct Box
    {
        glm::vec3 center;
        glm::vec3 halfExtents;
        glm::quat rotation;
    };

    struct Capsule
    {
        glm::vec3 point0;
        glm::vec3 point1;
        float radius;
    };

    using Shape = std::variant<Sphere, Box, Capsule>;

    explicit JoltCpuQueries(const JoltCpuScene &scene) : m_scene(scene)
    {
    }

    [[nodiscard]] bool Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maximum, RaycastHit &hit,
                               const Filter &filter) const;
    /// Read a batch against the same accepted CPU scene, in request order.
    void RaycastBatch(const float *originsXYZ, const float *directionsXYZ, size_t count, float maximum,
                      RaycastHit *hits, uint8_t *mask, const Filter &filter) const;
    [[nodiscard]] bool SweepBox(const BoxCollider &box, const glm::vec3 &position, const glm::quat &rotation,
                                const glm::vec3 &displacement, RaycastHit &hit) const;

    [[nodiscard]] std::vector<RaycastHit> RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maximum,
                                                     const Filter &filter) const;
    [[nodiscard]] std::vector<Collider *> Overlap(const Shape &shape, const Filter &filter) const;
    [[nodiscard]] bool Cast(const Shape &shape, const glm::vec3 &direction, float maximum, RaycastHit &hit,
                            const Filter &filter) const;
    [[nodiscard]] std::vector<RaycastHit> CastAll(const Shape &shape, const glm::vec3 &direction, float maximum,
                                                  const Filter &filter) const;

    /// Restrict nearest point to one accepted Collider lifetime on this body.
    [[nodiscard]] std::optional<glm::vec3> ClosestPoint(uint32_t bodyId, ColliderIdentity target,
                                                        const glm::vec3 &point) const;
    /// Query exactly two supplied Collider poses, independently of residency or collision policy.
    [[nodiscard]] static std::optional<PhysicsPenetrationResult>
    ComputePenetration(const Collider &a, const glm::vec3 &positionA, const glm::quat &rotationA, const Collider &b,
                       const glm::vec3 &positionB, const glm::quat &rotationB);

  private:
    const JoltCpuScene &m_scene;
};

} // namespace infernux
