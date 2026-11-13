#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace infernux
{
class Collider;
class Component;
class GameObject;
class Transform;

/// The actual component of a retained physics value, independent of body
/// rebuilds and Scene membership. Capture before publishing a query
/// epoch or delivering a callback; workers only copy this immutable identity.
class PhysicsTargetReference
{
  public:
    PhysicsTargetReference() = default;
    explicit PhysicsTargetReference(const Component *component);

    [[nodiscard]] Component *GetComponent() const;
    [[nodiscard]] Collider *GetCollider() const;
    [[nodiscard]] GameObject *GetGameObject() const;

  private:
    // Native embedded hosts and the extension can own different static runtime
    // registries. Resolve against the registry that published this value.
    using Resolver = Component *(*)(uint64_t, uint64_t);
    Resolver m_resolve = nullptr;
    uint64_t m_componentId = 0;
    uint64_t m_componentGeneration = 0;
    // A GameObject owns its intrinsic Transform for its entire lifetime. Its
    // registry identity lets the owner remain resolvable after collider removal.
    uint64_t m_ownerTransformId = 0;
    uint64_t m_ownerTransformGeneration = 0;
};

enum class PhysicsTargetChannels : uint8_t
{
    Position = 1,
    Rotation = 2
};

/// Immutable author identity for one ordered command. This contains no live
/// numerical physics state; both component references retain their generations.
struct PhysicsTargetAuthor
{
    PhysicsTargetReference rigidbody;
    PhysicsTargetReference ownerTransform;
    uint32_t bodyId = UINT32_MAX;
    uint64_t positionRevision = 0;
    uint64_t rotationRevision = 0;
    uint8_t channels = 0;
};

/// A control prefix appends records, then retains this same owner through its
/// candidate and any channel guards. Existing records are never modified.
struct PhysicsTargetBatch
{
    std::vector<PhysicsTargetAuthor> records;
};

struct PhysicsTargetSource
{
    std::shared_ptr<const PhysicsTargetBatch> batch;
    uint32_t index = UINT32_MAX;

    [[nodiscard]] const PhysicsTargetAuthor *Get() const noexcept
    {
        return batch && index < batch->records.size() ? &batch->records[index] : nullptr;
    }
    explicit operator bool() const noexcept
    {
        return Get() != nullptr;
    }
    friend bool operator==(const PhysicsTargetSource &left, const PhysicsTargetSource &right) noexcept
    {
        return left.batch == right.batch && left.index == right.index;
    }
    friend bool operator!=(const PhysicsTargetSource &left, const PhysicsTargetSource &right) noexcept
    {
        return !(left == right);
    }
};
} // namespace infernux
