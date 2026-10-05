/**
 * @file TagLayerManager.h
 * @brief Unity-style Tag and Layer management system.
 *
 * Provides a singleton manager for project-wide tag/layer definitions.
 * Tags are string identifiers (one per GameObject), layers are integer
 * indices (0-31) usable for rendering culling masks and physics collision.
 *
 * Built-in tags: Untagged, Respawn, Finish, EditorOnly, MainCamera, Player, GameController
 * Built-in layers: 0=Default, 1=TransparentFX, 2=IgnoreRaycast, 4=Water, 5=UI
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{

class TagLayerManager
{
  public:
    static TagLayerManager &Instance();

    static constexpr int kMaxLayers = 32;
    static constexpr int kBuiltinTagCount = 7;

    // ========================================================================
    // Tags
    // ========================================================================

    /// @brief Get tag string by index
    [[nodiscard]] const std::string &GetTag(int index) const;

    /// @brief Get tag index by name (-1 if not found)
    [[nodiscard]] int GetTagIndex(const std::string &tag) const;

    /// @brief Add a custom tag. Returns its index, or existing index if duplicate.
    int AddTag(const std::string &tag);

    /// @brief Remove a custom tag. Built-in tags cannot be removed.
    bool RemoveTag(const std::string &tag);

    /// @brief Get all tags (built-in + custom)
    [[nodiscard]] const std::vector<std::string> &GetAllTags() const;

    /// @brief Check if a tag is built-in (cannot be removed)
    [[nodiscard]] bool IsBuiltinTag(const std::string &tag) const;

    // ========================================================================
    // Layers
    // ========================================================================

    /// @brief Get layer name by index (0-31)
    [[nodiscard]] const std::string &GetLayerName(int layer) const;

    /// @brief Get layer index by name (-1 if not found)
    [[nodiscard]] int GetLayerByName(const std::string &name) const;

    /// @brief Set a layer name (built-in layers cannot be renamed)
    bool SetLayerName(int layer, const std::string &name);

    /// @brief Get all 32 layer names
    [[nodiscard]] const std::vector<std::string> &GetAllLayers() const;

    /// @brief Check if a layer is built-in (cannot be renamed)
    [[nodiscard]] bool IsBuiltinLayer(int layer) const;

    // ========================================================================
    // Physics collision matrix
    // ========================================================================

    /// @brief Get the 32-bit collision mask for a layer.
    ///        Bit N indicates whether this layer collides with layer N.
    [[nodiscard]] uint32_t GetLayerCollisionMask(int layer) const;

    /// @brief Set the full 32-bit collision mask for a layer.
    bool SetLayerCollisionMask(int layer, uint32_t mask);

    /// @brief Check whether two layers should collide in physics.
    [[nodiscard]] bool GetLayersCollide(int layerA, int layerB) const;

    /// @brief Enable/disable collision between two layers (symmetric).
    bool SetLayersCollide(int layerA, int layerB, bool shouldCollide);

    // ========================================================================
    // Layer mask helpers
    // ========================================================================

    /// @brief Create a layer mask from layer indices
    [[nodiscard]] static uint32_t LayerToMask(int layer);

    /// @brief Create a mask from multiple layer names
    [[nodiscard]] uint32_t GetMask(const std::vector<std::string> &layerNames) const;

    // ========================================================================
    // Serialization (project settings)
    // ========================================================================

    /// @brief Serialize to JSON string
    [[nodiscard]] std::string Serialize() const;

    /// @brief Serialize project defaults independently of the active singleton.
    [[nodiscard]] static std::string SerializeDefaults();

    /// @brief Deserialize from JSON string
    bool Deserialize(const std::string &json);

    /// @brief Save to file
    bool SaveToFile(const std::string &path) const;

    /// @brief Load from file
    bool LoadFromFile(const std::string &path);

  private:
    TagLayerManager();
    ~TagLayerManager() = default;
    TagLayerManager(const TagLayerManager &) = delete;
    TagLayerManager &operator=(const TagLayerManager &) = delete;

    void InitDefaults();

    std::vector<std::string> m_tags;             // Dynamic list, starts with built-ins
    std::vector<std::string> m_layers;           // Always 32 slots
    std::vector<uint32_t> m_layerCollisionMasks; // Always 32 slots, symmetric 32x32 matrix as bitmasks

    static const std::string s_emptyString;
};

} // namespace infernux
