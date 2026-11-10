/**
 * @file CapsuleCollider.h
 * @brief Capsule collider component (Unity: CapsuleCollider).
 */

#pragma once

#include "Collider.h"

namespace infernux
{

class CapsuleCollider : public Collider
{
  public:
    CapsuleCollider() = default;
    ~CapsuleCollider() override = default;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "CapsuleCollider";
    }

    // ====================================================================
    // Properties (Unity-style)
    // ====================================================================

    [[nodiscard]] float GetRadius() const
    {
        return m_radius;
    }
    void SetRadius(float radius);

    /// @brief Total height of the capsule (including hemispherical caps).
    [[nodiscard]] float GetHeight() const
    {
        return m_height;
    }
    void SetHeight(float height);

    /// @brief Direction axis: 0 = X, 1 = Y (default), 2 = Z
    [[nodiscard]] int GetDirection() const
    {
        return m_direction;
    }
    void SetDirection(int dir);

    // ====================================================================
    // Jolt shape
    // ====================================================================

    [[nodiscard]] JPH::RefConst<JPH::Shape> CreateJoltShape() const override;

    // ====================================================================
    // Serialization
    // ====================================================================

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

    void AutoFitToMesh() override;

  private:
    float m_radius = 0.5f;
    float m_height = 2.0f; // Total height (Unity convention)
    int m_direction = 1;   // Y-axis by default
};

} // namespace infernux
