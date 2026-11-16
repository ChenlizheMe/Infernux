#include "PhysicsBody.h"
#include "RuntimeWorld.h"

#include <stdexcept>

namespace infernux
{
void PhysicsBody::ValidateGpuSelection(bool gpu) const
{
    if (gpu == m_gpu)
        return;
    if (const auto *world = GetBoundPhysicsWorld(); world && world->simulation.isPlaying)
        throw std::logic_error("Live CPU/GPU body migration is not implemented yet; change GPU while stopped");
}

void PhysicsBody::SetGpuEnabled(bool gpu)
{
    if (gpu == m_gpu)
        return;
    ValidateGpuSelection(gpu);
    const bool previous = m_gpu;
    m_gpu = gpu;
    try {
        OnGpuSelectionChanged();
    } catch (...) {
        m_gpu = previous;
        throw;
    }
}

void PhysicsBody::OnDisable()
{
    InvalidatePhysicsResults();
}

nlohmann::json PhysicsBody::SerializeDocument() const
{
    auto document = Component::SerializeDocument();
    document["gpu"] = m_gpu;
    return document;
}

bool PhysicsBody::DeserializeDocument(const nlohmann::json &document)
{
    const bool gpu = document.value("gpu", false);
    ValidateGpuSelection(gpu);
    if (!Component::DeserializeDocument(document))
        return false;
    m_gpu = gpu;
    return true;
}

void PhysicsBody::CopyPhysicsBodyAuthoringTo(PhysicsBody &clone) const noexcept
{
    clone.m_enabled = m_enabled;
    clone.m_executionOrder = m_executionOrder;
    clone.m_gpu = m_gpu;
}
} // namespace infernux
