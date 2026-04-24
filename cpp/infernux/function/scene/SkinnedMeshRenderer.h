#pragma once

#include "MeshRenderer.h"

#include <string>
#include <vector>

namespace infernux
{

/**
 * @brief SkinnedMeshRenderer — structural placeholder for animated model instances.
 *
 * This currently reuses MeshRenderer's static draw path so animated FBX models
 * can already be distinguished in scene data, inspector UI, and future runtime
 * animation plumbing. A dedicated skinned render path will be added later.
 */
class SkinnedMeshRenderer : public MeshRenderer
{
  public:
    SkinnedMeshRenderer() = default;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "SkinnedMeshRenderer";
    }

    void SetSourceModelGuid(const std::string &guid)
    {
        m_sourceModelGuid = guid;
    }
    [[nodiscard]] const std::string &GetSourceModelGuid() const
    {
        return m_sourceModelGuid;
    }

    void SetSourceModelPath(const std::string &path)
    {
        m_sourceModelPath = path;
    }
    [[nodiscard]] const std::string &GetSourceModelPath() const
    {
        return m_sourceModelPath;
    }

    void SetAnimationTakeNames(std::vector<std::string> names)
    {
        m_animationTakeNames = std::move(names);
    }
    [[nodiscard]] const std::vector<std::string> &GetAnimationTakeNames() const
    {
        return m_animationTakeNames;
    }

    void SetActiveTakeName(const std::string &name)
    {
        m_activeTakeName = name;
    }
    [[nodiscard]] const std::string &GetActiveTakeName() const
    {
        return m_activeTakeName;
    }

    [[nodiscard]] bool HasAnimationTakes() const
    {
        return !m_animationTakeNames.empty();
    }

    [[nodiscard]] std::string Serialize() const override;
    bool Deserialize(const std::string &jsonStr) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

  private:
    std::string m_sourceModelGuid;
    std::string m_sourceModelPath;
    std::vector<std::string> m_animationTakeNames;
    std::string m_activeTakeName;
};

} // namespace infernux
