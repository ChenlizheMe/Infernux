#pragma once

#include "LineRenderer.h"
#include <deque>

namespace infernux
{
/// Timed world-space history. Reuses the native ribbon renderer, materials,
/// per-camera expansion, bounds and batching rather than creating a second
/// drawing implementation or running geometry generation in Python.
class TrailRenderer final : public LineRenderer
{
  public:
    TrailRenderer();
    [[nodiscard]] const char *GetTypeName() const override
    {
        return "TrailRenderer";
    }
    [[nodiscard]] bool WantsRuntimeLateUpdate() const override
    {
        return true;
    }
    void LateUpdate(float deltaTime) override;

    [[nodiscard]] float GetTime() const
    {
        return m_time;
    }
    void SetTime(float value);
    [[nodiscard]] float GetMinVertexDistance() const
    {
        return m_minVertexDistance;
    }
    void SetMinVertexDistance(float value);
    [[nodiscard]] bool GetEmitting() const
    {
        return m_emitting;
    }
    void SetEmitting(bool value);
    [[nodiscard]] size_t GetTrailPositionCount() const
    {
        return m_history.size();
    }
    [[nodiscard]] glm::vec3 GetTrailPosition(size_t index) const;
    [[nodiscard]] std::vector<glm::vec3> GetTrailPositions() const;
    void SetTrailPosition(size_t index, const glm::vec3 &position);
    void SetTrailPositions(const std::vector<glm::vec3> &positions);
    void AddPosition(const glm::vec3 &position);
    void AddPositions(const std::vector<glm::vec3> &positions);
    void Clear();

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

  private:
    struct Point
    {
        glm::vec3 position;
        double created;
    };
    void Trim();
    void Publish();
    std::deque<Point> m_history;
    double m_clock = 0.0;
    float m_time = 5.0f;
    float m_minVertexDistance = 0.1f;
    bool m_emitting = true;
    bool m_hasLiveHead = false;
    glm::vec3 m_liveHead{0.0f};
};
} // namespace infernux
