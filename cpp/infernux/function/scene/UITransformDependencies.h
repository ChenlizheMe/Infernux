#pragma once

#include "GameObject.h"
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infernux
{

/// Geometry dependencies of a recorded UI command list. Retain ECS lifetime
/// handles, never Python wrappers or raw pointers to destroyed scene objects.
/// Python rebuilds membership on its existing scene/UI revision boundary.
class UITransformDependencies
{
  public:
    UITransformDependencies(const std::vector<GameObject *> &screen, const std::vector<GameObject *> &world)
    {
        m_entries.reserve(screen.size() + world.size());
        m_changedEntries.reserve(screen.size() + world.size());
        for (auto *object : screen)
            Add(object, false);
        for (auto *object : world)
            Add(object, true);
    }

    uint64_t Poll()
    {
        auto &store = TransformECSStore::Instance();
        const auto serial = store.GetGlobalTransformSerial();
        const bool checkPose = !m_initialized || serial != m_transformSerial;
        bool changed = !m_initialized;
        m_changedEntries.clear();
        for (uint32_t index = 0; index < m_entries.size(); ++index) {
            auto &entry = m_entries[index];
            bool entryChanged = !m_initialized;
            // Even an unchanged transform serial must not make an expired
            // generational handle safe to dereference.
            if (!store.IsValid(entry.handle))
                throw std::runtime_error("UI dependency membership outlived its Transform; rebuild the snapshot");
            auto *transform = store.GetOwner(entry.handle);
            if (entry.world) {
                const int layer = transform->GetGameObject()->GetLayer();
                entryChanged |= layer != entry.layer;
                entry.layer = layer;
            }
            // Local UI poses do not resolve world matrices. Several local
            // edits can therefore share the same already-dirty serial; read
            // their three SoA values directly instead of treating that serial
            // as an all-mutations counter.
            // A fixed/update callback can query before deferred frame-cache
            // writes are committed to the SoA serial. Read that live pose too.
            if (!entry.world || checkPose || store.GetWorldMatrixDirty(entry.handle) || store.IsFrameCacheActive()) {
                glm::vec3 position;
                glm::quat rotation;
                if (entry.world) {
                    position = transform->GetPosition();
                    rotation = transform->GetRotation();
                } else {
                    // Screen geometry ignores Z, X/Y tilt and Scale by contract.
                    const auto local = transform->GetLocalPosition();
                    position = {local.x, local.y, transform->GetLocalEulerAngles().z};
                    rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                }
                entryChanged |= position != entry.position || rotation != entry.rotation;
                entry.position = position;
                entry.rotation = rotation;
            }
            if (entryChanged) {
                changed = true;
                m_changedEntries.push_back(index);
            }
        }
        m_transformSerial = serial;
        m_initialized = true;
        if (changed)
            ++m_revision;
        return m_revision;
    }

    /// Changed indices from the last Poll, in screen-then-world input order.
    /// This is a publication delta, not a history retained across polls.
    const std::vector<uint32_t> &GetChangedEntries() const noexcept
    {
        return m_changedEntries;
    }

    // Shared by runtime input and editor picking. Return unbounded local
    // coordinates: a captured drag must continue beyond the element's quad.
    // UI layout owns pixel sizes; this batch only owns native scene geometry.
    std::vector<std::array<double, 3>> ProjectWorldRay(const glm::vec3 &origin, const glm::vec3 &direction,
                                                       uint32_t layerMask)
    {
        Poll();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double infinity = std::numeric_limits<double>::infinity();
        std::vector<std::array<double, 3>> result;
        result.reserve(m_entries.size());
        for (const auto &entry : m_entries) {
            if (!entry.world)
                continue;
            std::array<double, 3> hit{nan, nan, infinity};
            if ((layerMask & (uint32_t(1) << entry.layer)) != 0) {
                const auto inverse = glm::conjugate(glm::dquat(entry.rotation));
                const auto offset = inverse * (glm::dvec3(origin) - glm::dvec3(entry.position));
                const auto ray = inverse * glm::dvec3(direction);
                if (std::abs(ray.z) > 1e-7) {
                    const double distance = -offset.z / ray.z;
                    if (distance > 0.0)
                        hit = {offset.x + ray.x * distance, offset.y + ray.y * distance, distance};
                }
            }
            result.push_back(hit);
        }
        return result;
    }

  private:
    struct Entry
    {
        TransformECSStore::Handle handle;
        bool world;
        int layer = -1;
        glm::vec3 position{0.0f};
        glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    };

    void Add(GameObject *object, bool world)
    {
        if (!object)
            throw std::invalid_argument("UI geometry dependencies require an attached GameObject");
        m_entries.push_back({object->GetTransform()->GetECSHandle(), world});
    }

    std::vector<Entry> m_entries;
    std::vector<uint32_t> m_changedEntries;
    uint64_t m_transformSerial = 0;
    uint64_t m_revision = 0;
    bool m_initialized = false;
};

} // namespace infernux
