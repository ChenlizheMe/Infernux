#include "PhysicsBody.h"

namespace infernux
{
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
    document["cross_backend_interaction"] = m_crossBackendInteraction;
    return document;
}

bool PhysicsBody::DeserializeDocument(const nlohmann::json &document)
{
    const bool gpu = document.value("gpu", false);
    const bool crossBackendInteraction = document.value("cross_backend_interaction", false);
    if (!Component::DeserializeDocument(document))
        return false;
    m_gpu = gpu;
    m_crossBackendInteraction = crossBackendInteraction;
    return true;
}

void PhysicsBody::CopyPhysicsBodyAuthoringTo(PhysicsBody &clone) const noexcept
{
    clone.m_enabled = m_enabled;
    clone.m_executionOrder = m_executionOrder;
    clone.m_gpu = m_gpu;
    clone.m_crossBackendInteraction = m_crossBackendInteraction;
}
} // namespace infernux
