#include "RenderTextureLoader.h"
#include "RenderTextureArtifact.h"
#include <function/resources/AssetDatabase/AssetDatabase.h>

namespace infernux
{
RuntimeAssetPayload RenderTextureLoader::Load(const std::string &, const std::string &guid, AssetDatabase *adb)
{
    if (!adb)
        throw std::invalid_argument("RenderTextureLoader requires an AssetDatabase");
    const auto path = adb->GetRuntimeArtifactPath(guid, ResourceType::RenderTexture);
    std::vector<char> bytes;
    if (path.empty() || !adb->ReadFile(path, bytes))
        throw std::runtime_error("RenderTexture requires an imported .inxrtex artifact: " + guid);
    return std::make_shared<rhi::RenderTextureDesc>(
        RenderTextureArtifact::Decode(std::string_view(bytes.data(), bytes.size())));
}

bool RenderTextureLoader::Reload(const RuntimeAssetPayload &existing, const std::string &path, const std::string &guid,
                                 AssetDatabase *adb)
{
    const auto candidate = Load(path, guid, adb).Get<rhi::RenderTextureDesc>();
    if (m_beforePublish)
        m_beforePublish(guid, *candidate);
    *existing.Get<rhi::RenderTextureDesc>() = *candidate;
    return true;
}

size_t RenderTextureLoader::EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const
{
    if (!payload.Get<rhi::RenderTextureDesc>())
        throw std::invalid_argument("RenderTextureLoader cannot estimate an empty payload");
    return sizeof(rhi::RenderTextureDesc);
}

void RenderTextureLoader::CreateMeta(const char *content, size_t size, const std::string &path,
                                     InxResourceMeta &meta) const
{
    meta.Init(content, size, path, ResourceType::RenderTexture);
}
} // namespace infernux
