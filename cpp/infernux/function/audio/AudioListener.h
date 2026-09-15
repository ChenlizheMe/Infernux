#pragma once

#include <function/scene/Component.h>
#include <string>

namespace infernux
{

/**
 * @brief AudioListener component — represents the "ears" in the scene.
 *
 * Only one AudioListener is active in the shared World at a time (typically
 * on the main camera). The listener's
 * Transform position is used for 3D
 * spatialization of all AudioSources.
 *
 * Unity API alignment:
 * - AudioListener is a component attached to a GameObject
 * - Only one active listener across all resident scenes in the World
 * - Position/orientation from the Transform
 * drives 3D audio
 *
 * Wwise extensibility:
 * - gameObjectId() → owning GameObject ID for Wwise listener registration
 */
class AudioListener : public Component
{
  public:
    AudioListener() = default;
    ~AudioListener() override = default;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "AudioListener";
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    void Awake() override;
    void OnEnable() override;
    void OnDisable() override;
    void OnDestroy() override;

    // ========================================================================
    // Serialization
    // ========================================================================

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;

    // ========================================================================
    // Wwise extensibility hook
    // ========================================================================

    /// @brief Get the owning GameObject ID (for Wwise listener registration)
    [[nodiscard]] uint64_t GetGameObjectId() const;
};

} // namespace infernux
