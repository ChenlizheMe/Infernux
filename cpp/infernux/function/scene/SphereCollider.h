/**
 * @file SphereCollider.h
 * @brief Sphere collider component (Unity: SphereCollider).
 */

#pragma once

#include "Collider.h"

namespace infernux
{

class SphereCollider : public Collider
{
  public:
    SphereCollider() = default;
    ~SphereCollider() override = default;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "SphereCollider";
    }

    // ====================================================================
    // Properties
    // ====================================================================

    [[nodiscard]] float GetRadius() const
    {
        return m_radius;
    }
    void SetRadius(float radius);

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
};

} // namespace infernux
