/**
 * @file BoxCollider.h
 * @brief Axis-aligned box collider (Unity: BoxCollider).
 */

#pragma once

#include "Collider.h"
#include <glm/glm.hpp>

namespace infernux
{

class BoxCollider : public Collider
{
  public:
    BoxCollider() = default;
    ~BoxCollider() override = default;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "BoxCollider";
    }

    // ====================================================================
    // Properties
    // ====================================================================

    /// @brief Half-extents of the box (Unity: BoxCollider.size stores full size,
    ///        but Jolt uses half-extents. We expose full size like Unity.)
    [[nodiscard]] glm::vec3 GetSize() const
    {
        return m_size;
    }
    void SetSize(const glm::vec3 &size);

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
    glm::vec3 m_size{1.0f, 1.0f, 1.0f}; // Full size (Unity convention)
};

} // namespace infernux
