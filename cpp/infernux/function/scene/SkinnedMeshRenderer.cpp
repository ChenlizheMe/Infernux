#include "SkinnedMeshRenderer.h"
#include "ComponentFactory.h"
#include "SceneManager.h"

#include <core/config/MathConstants.h>
#include <function/resources/InxSkinnedMesh/SkinnedModelCache.h>

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infernux
{

INFERNUX_REGISTER_COMPONENT("SkinnedMeshRenderer", SkinnedMeshRenderer)

void SkinnedMeshRenderer::SetSourceModelGuid(const std::string &guid)
{
    if (m_sourceModelGuid == guid)
        return;
    SkinnedModelCache::Instance().Invalidate(m_sourceModelGuid, m_sourceModelPath);
    m_runtimeModel.reset();
    m_sourceModelGuid = guid;
    ClearRuntimeSkinnedMesh();
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetSourceModelPath(const std::string &path)
{
    if (m_sourceModelPath == path)
        return;
    SkinnedModelCache::Instance().Invalidate(m_sourceModelGuid, m_sourceModelPath);
    m_runtimeModel.reset();
    m_sourceModelPath = path;
    ClearRuntimeSkinnedMesh();
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetActiveTakeName(const std::string &name)
{
    if (m_activeTakeName == name)
        return;
    m_activeTakeName = name;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetRuntimeAnimationTime(float t)
{
    m_runtimeAnimationTime = t;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SubmitAnimationPose(const std::string &takeName, float timeSeconds, float normalizedTime,
                                              const std::string &blendTakeName, float blendTimeSeconds,
                                              float blendWeight)
{
    m_activeTakeName = takeName;
    m_runtimeAnimationTime = timeSeconds;
    m_runtimeAnimationNormalized = normalizedTime;
    m_blendTakeName = blendTakeName;
    m_blendAnimationTime = blendTimeSeconds;
    m_blendWeight = std::clamp(blendWeight, 0.0f, 1.0f);
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetBlendTakeName(const std::string &name)
{
    if (m_blendTakeName == name)
        return;
    m_blendTakeName = name;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetBlendAnimationTime(float t)
{
    m_blendAnimationTime = t;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetBlendWeight(float w)
{
    const float clamped = std::clamp(w, 0.0f, 1.0f);
    if (std::abs(m_blendWeight - clamped) <= kEpsilon)
        return;
    m_blendWeight = clamped;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::ClearAnimationBlend()
{
    if (m_blendTakeName.empty() && m_blendAnimationTime == 0.0f && m_blendWeight == 0.0f)
        return;
    m_blendTakeName.clear();
    m_blendAnimationTime = 0.0f;
    m_blendWeight = 0.0f;
    RefreshRuntimeSkinnedMesh();
}

float SkinnedMeshRenderer::GetAnimationDurationSeconds(const std::string &takeName) const
{
    auto model = GetOrLoadRuntimeModel();
    return model ? model->GetAnimationDurationSeconds(takeName) : 0.0f;
}

void SkinnedMeshRenderer::ClearRuntimeSkinnedMesh()
{
    m_runtimeSkinnedVertices.clear();
    m_runtimeSkinnedIndices.clear();
    m_runtimeSkinnedSubMeshes.clear();
    m_runtimeSkinBoneMatrices.clear();
    MarkMeshBufferDirty();
}

std::shared_ptr<InxSkinnedMesh> SkinnedMeshRenderer::GetOrLoadRuntimeModel() const
{
    if (m_runtimeModel && m_runtimeModel->IsValid())
        return m_runtimeModel;
    if (m_sourceModelGuid.empty() && m_sourceModelPath.empty())
        return nullptr;
    m_runtimeModel = SkinnedModelCache::Instance().Load(m_sourceModelGuid, m_sourceModelPath);
    return m_runtimeModel;
}

void SkinnedMeshRenderer::RefreshRuntimeSkinnedMesh()
{
    if (m_sourceModelGuid.empty() && m_sourceModelPath.empty())
        return;
    if (m_activeTakeName.empty())
        return;

    auto model = GetOrLoadRuntimeModel();
    if (!model || !model->IsValid()) {
        ClearRuntimeSkinnedMesh();
        return;
    }

    SkinnedSampleRequest request;
    request.takeName = m_activeTakeName;
    request.timeSeconds = m_runtimeAnimationTime;
    request.blendTakeName = m_blendTakeName;
    request.blendTimeSeconds = m_blendAnimationTime;
    request.blendWeight = m_blendWeight;

    const bool wasEmpty = m_runtimeSkinnedVertices.empty();
    if (wasEmpty || m_runtimeSkinnedIndices.empty()) {
        m_runtimeSkinnedVertices = model->baseVertices;
        m_runtimeSkinnedIndices = model->indices;
        m_runtimeSkinnedSubMeshes = model->subMeshes;
        MarkMeshBufferDirty();
    }

    m_runtimeSkinBoneMatrices = model->BuildGpuBonePalette(request);
    if (wasEmpty)
        SceneManager::Instance().NotifyMeshRendererChanged(this);
}

std::string SkinnedMeshRenderer::Serialize() const
{
    json j = json::parse(MeshRenderer::Serialize());
    if (!m_sourceModelGuid.empty())
        j["sourceModelGuid"] = m_sourceModelGuid;
    if (!m_sourceModelPath.empty())
        j["sourceModelPath"] = m_sourceModelPath;
    if (!m_animationTakeNames.empty())
        j["animationTakeNames"] = m_animationTakeNames;
    if (!m_activeTakeName.empty())
        j["activeTakeName"] = m_activeTakeName;
    return j.dump(2);
}

bool SkinnedMeshRenderer::Deserialize(const std::string &jsonStr)
{
    if (!MeshRenderer::Deserialize(jsonStr))
        return false;

    try {
        json j = json::parse(jsonStr);
        m_sourceModelGuid = j.value("sourceModelGuid", std::string());
        m_sourceModelPath = j.value("sourceModelPath", std::string());
        m_activeTakeName = j.value("activeTakeName", std::string());
        m_animationTakeNames.clear();
        if (j.contains("animationTakeNames") && j["animationTakeNames"].is_array()) {
            for (const auto &v : j["animationTakeNames"]) {
                if (v.is_string())
                    m_animationTakeNames.push_back(v.get<std::string>());
            }
        }
        RefreshRuntimeSkinnedMesh();
        return true;
    } catch (...) {
        return false;
    }
}

std::unique_ptr<Component> SkinnedMeshRenderer::Clone() const
{
    auto clone = std::make_unique<SkinnedMeshRenderer>();
    const uint64_t newId = clone->GetComponentID();
    clone->Deserialize(Serialize());
    clone->SetComponentID(newId);
    return clone;
}

} // namespace infernux
