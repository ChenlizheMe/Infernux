#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace infernux
{

enum class PhysicsMaterialCombine : uint8_t
{
    Average = 0,
    Minimum = 1,
    Multiply = 2,
    Maximum = 3,
};

class PhysicMaterial
{
  public:
    /// Surface mutations invalidate collision mirrors without coupling assets to a scene.
    [[nodiscard]] static uint64_t GetSurfaceRevision() noexcept;
    [[nodiscard]] float GetFriction() const
    {
        return m_friction;
    }
    void SetFriction(float value);

    [[nodiscard]] float GetBounciness() const
    {
        return m_bounciness;
    }
    void SetBounciness(float value);

    [[nodiscard]] PhysicsMaterialCombine GetFrictionCombine() const
    {
        return m_frictionCombine;
    }
    void SetFrictionCombine(PhysicsMaterialCombine value);

    [[nodiscard]] PhysicsMaterialCombine GetBounceCombine() const
    {
        return m_bounceCombine;
    }
    void SetBounceCombine(PhysicsMaterialCombine value);

    [[nodiscard]] const std::string &GetName() const
    {
        return m_name;
    }
    void SetName(std::string value)
    {
        m_name = std::move(value);
    }

    [[nodiscard]] const std::string &GetGuid() const
    {
        return m_guid;
    }
    void SetGuid(std::string value)
    {
        m_guid = std::move(value);
    }

    [[nodiscard]] const std::string &GetFilePath() const
    {
        return m_filePath;
    }
    void SetFilePath(std::string value)
    {
        m_filePath = std::move(value);
    }

    [[nodiscard]] nlohmann::json SerializeDocument() const;
    void DeserializeDocument(const nlohmann::json &document);
    void CopyValuesFrom(const PhysicMaterial &other);
    void SaveToFile() const;
    void SaveToFile(const std::string &path);

    [[nodiscard]] size_t GetRuntimeMemoryBytes() const noexcept
    {
        return sizeof(*this) + m_name.capacity() + m_guid.capacity() + m_filePath.capacity();
    }

  private:
    friend class Collider;
    void AddColliderBinding() noexcept
    {
        ++m_colliderBindings;
    }
    void RemoveColliderBinding() noexcept
    {
        --m_colliderBindings;
    }
    [[nodiscard]] size_t GetColliderBindingCount() const noexcept
    {
        return m_colliderBindings;
    }
    void NotifySurfaceChanged() const noexcept;
    static void ValidateCombine(PhysicsMaterialCombine value, const char *fieldName);

    float m_friction = 0.6f;
    float m_bounciness = 0.0f;
    PhysicsMaterialCombine m_frictionCombine = PhysicsMaterialCombine::Average;
    PhysicsMaterialCombine m_bounceCombine = PhysicsMaterialCombine::Average;
    std::string m_name;
    std::string m_guid;
    std::string m_filePath;
    size_t m_colliderBindings = 0;
};

} // namespace infernux
