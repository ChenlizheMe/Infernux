#include "TrailRenderer.h"
#include "ComponentFactory.h"
#include "Transform.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace infernux
{
namespace
{
float NonNegative(float value, const char *name)
{
    if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    return value;
}
void FinitePosition(const glm::vec3 &value)
{
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
        throw std::invalid_argument("Trail positions must be finite");
}
} // namespace

INFERNUX_REGISTER_VALIDATED_COMPONENT("TrailRenderer", TrailRenderer)

TrailRenderer::TrailRenderer()
{
    LineRenderer::SetPositions({});
    SetWidthCurve({{0.0f, 1.0f}, {1.0f, 1.0f}});
}

void TrailRenderer::SetTime(float value)
{
    m_time = NonNegative(value, "Trail time");
    Trim();
    Publish();
}

void TrailRenderer::SetMinVertexDistance(float value)
{
    m_minVertexDistance = NonNegative(value, "Trail minimum vertex distance");
}

void TrailRenderer::SetEmitting(bool value)
{
    if (m_emitting && !value && m_hasLiveHead && !m_history.empty() && m_liveHead != m_history.back().position)
        m_history.push_back({m_liveHead, m_clock});
    m_emitting = value;
    m_hasLiveHead = false;
    Publish();
}

glm::vec3 TrailRenderer::GetTrailPosition(size_t index) const
{
    if (index >= m_history.size())
        throw std::out_of_range("Trail position index out of range");
    return m_history[index].position;
}

std::vector<glm::vec3> TrailRenderer::GetTrailPositions() const
{
    std::vector<glm::vec3> positions;
    positions.reserve(m_history.size());
    for (const auto &point : m_history)
        positions.push_back(point.position);
    return positions;
}

void TrailRenderer::SetTrailPosition(size_t index, const glm::vec3 &position)
{
    FinitePosition(position);
    if (index >= m_history.size())
        throw std::out_of_range("Trail position index out of range");
    m_history[index].position = position;
    Publish();
}

void TrailRenderer::SetTrailPositions(const std::vector<glm::vec3> &positions)
{
    if (positions.size() != m_history.size())
        throw std::invalid_argument("SetPositions replaces existing trail points; use AddPositions to append");
    for (const auto &position : positions)
        FinitePosition(position);
    for (size_t index = 0; index < positions.size(); ++index)
        m_history[index].position = positions[index];
    Publish();
}

void TrailRenderer::AddPosition(const glm::vec3 &position)
{
    AddPositions({position});
}

void TrailRenderer::AddPositions(const std::vector<glm::vec3> &positions)
{
    for (const auto &position : positions)
        FinitePosition(position);
    for (const auto &position : positions)
        m_history.push_back({position, m_clock});
    m_hasLiveHead = false;
    Trim();
    Publish();
}

void TrailRenderer::Clear()
{
    m_history.clear();
    m_hasLiveHead = false;
    LineRenderer::SetPositions({});
}

void TrailRenderer::Trim()
{
    const double cutoff = m_clock - m_time;
    // Retain a boundary segment until both of its endpoints expire; Publish
    // clips its oldest endpoint continuously instead of popping a whole edge.
    while (m_history.size() > 1 && m_history[1].created <= cutoff)
        m_history.pop_front();
    if (m_history.size() == 1 && m_history.front().created <= cutoff)
        m_history.clear();
}

void TrailRenderer::Publish()
{
    std::vector<glm::vec3> positions;
    positions.reserve(m_history.size() + 1);
    if (!m_history.empty()) {
        if (m_hasLiveHead && m_liveHead != m_history.back().position)
            positions.push_back(m_liveHead);
        // The API returns creation order. Rendering is head to tail so curve
        // time zero is the spawning end, matching TrailRenderer's width API.
        for (auto it = m_history.rbegin(); it != m_history.rend(); ++it)
            positions.push_back(it->position);
        const double cutoff = m_clock - m_time;
        if (m_history.size() > 1 && m_history.front().created < cutoff) {
            const auto &a = m_history[0];
            const auto &b = m_history[1];
            const double duration = b.created - a.created;
            if (duration > 0)
                positions.back() =
                    glm::mix(a.position, b.position, static_cast<float>((cutoff - a.created) / duration));
        }
    }
    LineRenderer::SetPositions(std::move(positions));
}

void TrailRenderer::LateUpdate(float deltaTime)
{
    if (!std::isfinite(deltaTime) || deltaTime < 0)
        return;
    m_clock += deltaTime;
    Trim();
    if (m_emitting && GetTransform() && m_time > 0) {
        const auto position = GetTransform()->GetPosition();
        FinitePosition(position);
        const float distance = m_history.empty() ? 0 : glm::length(position - m_history.back().position);
        if (m_history.empty() || (distance > 0 && distance >= m_minVertexDistance))
            m_history.push_back({position, m_clock});
        m_liveHead = position;
        m_hasLiveHead = true;
    } else {
        m_hasLiveHead = false;
    }
    Publish();
}

nlohmann::json TrailRenderer::SerializeDocument() const
{
    auto document = LineRenderer::SerializeDocument();
    document["type"] = "TrailRenderer";
    // History and GPU ribbon vertices are transient, never scene assets.
    document["positions"] = nlohmann::json::array();
    document["time"] = m_time;
    document["minVertexDistance"] = m_minVertexDistance;
    document["emitting"] = m_emitting;
    return document;
}

void TrailRenderer::ValidateSerializedDocument(const nlohmann::json &document)
{
    ValidateSerializedDocumentForType(document, "TrailRenderer");
}

bool TrailRenderer::DeserializeDocument(const nlohmann::json &document)
{
    if (!LineRenderer::DeserializeDocument(document))
        return false;
    m_time = document["time"].get<float>();
    m_minVertexDistance = document["minVertexDistance"].get<float>();
    m_emitting = document["emitting"].get<bool>();
    m_clock = 0;
    Clear();
    return true;
}

std::unique_ptr<Component> TrailRenderer::Clone() const
{
    auto clone = std::make_unique<TrailRenderer>();
    auto document = SerializeDocument();
    document.erase("component_id");
    if (!clone->DeserializeDocument(document))
        throw std::runtime_error("TrailRenderer clone could not restore its authored state");
    CopyMaterialBindingsTo(*clone);
    return clone;
}
} // namespace infernux
