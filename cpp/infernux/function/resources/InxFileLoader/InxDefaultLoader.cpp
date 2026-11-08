#include "InxDefaultLoader.hpp"
#include <function/resources/AssetRegistry/AssetBytes.h>
#include <platform/filesystem/InxPath.h>

namespace infernux
{
InxDefaultLoader::InxDefaultLoader(ResourceType resourceType) : m_resourceType(resourceType)
{
    if (resourceType != ResourceType::DefaultText && resourceType != ResourceType::DefaultBinary &&
        resourceType != ResourceType::RenderEffect && resourceType != ResourceType::ParticleGraph &&
        resourceType != ResourceType::DataAsset)
        throw std::invalid_argument("InxDefaultLoader requires a raw source or metadata-only resource type");
}

void InxDefaultLoader::CreateMeta(AssetSource &source, InxResourceMeta &metadata) const
{
    metadata.InitFromSource(source, m_resourceType);
}

bool InxDefaultLoader::SupportsWorkerLoad() const noexcept
{
    return m_resourceType == ResourceType::DefaultText || m_resourceType == ResourceType::DefaultBinary;
}

RuntimeAssetPayload InxDefaultLoader::Load(const std::string &path, const std::string &, AssetDatabase *)
{
    if (!SupportsWorkerLoad())
        return nullptr;
    return std::make_shared<AssetBytes>(AssetSource(path).TakeBytes());
}

WorkerAssetLoadPlan InxDefaultLoader::PlanWorkerLoad(const std::string &path, const std::string &, AssetDatabase *)
{
    const auto size = AssetSource(path).Size();
    if (size > std::string{}.max_size())
        throw std::length_error("Asset source is too large to materialize: " + path);
    // Size discovery closes its file. Open the retained revision only after
    // the asset's destination has been admitted by the owner.
    auto advance = [path, size, source = std::shared_ptr<AssetSource>{}](
                       uint64_t bytes, WorkerAssetLoadPlan::Read &next) mutable -> RuntimeAssetPayload {
        if (!source) {
            const auto mode = bytes < size ? AssetSource::ReadMode::Retained : AssetSource::ReadMode::Once;
            source = std::make_shared<AssetSource>(path, mode);
            if (source->Size() != size)
                throw std::runtime_error("Asset source changed size before its planned read: " + path);
        }
        const bool complete = source->ReadNext(static_cast<size_t>(bytes));
        next.bytes -= bytes;
        if (!complete)
            return {};
        return std::make_shared<AssetBytes>(std::move(*source).TakeBytes());
    };
    return {size + sizeof(AssetBytes) + sizeof(AssetSource) + sizeof(FileInput) + path.size() + 64,
            std::move(advance),
            {size}};
}

size_t InxDefaultLoader::EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const
{
    return payload ? payload.Get<AssetBytes>()->OwnedBytes() : 0;
}
} // namespace infernux
