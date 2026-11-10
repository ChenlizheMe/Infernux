#include "AudioClipLoader.h"

#include <core/log/InxLog.h>
#include <function/audio/AudioClip.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>

#include <platform/filesystem/InxPath.h>

#include <filesystem>

namespace infernux
{
void AudioClipLoader::OnRelocated(const RuntimeAssetPayload &payload, const std::string &filePath)
{
    payload.Get<AudioClip>()->SetFilePath(filePath);
}

// =============================================================================
// Load — decode audio file and create a new AudioClip
// =============================================================================

RuntimeAssetPayload AudioClipLoader::Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb)
{
    if (filePath.empty() || guid.empty()) {
        INXLOG_WARN("AudioClipLoader::Load: empty filePath or guid");
        return nullptr;
    }

    auto fsPath = ToFsPath(filePath);
    if (!std::filesystem::exists(fsPath)) {
        INXLOG_ERROR("AudioClipLoader::Load: file not found: ", filePath);
        return nullptr;
    }

    auto clip = std::make_shared<AudioClip>();
    const auto metadata = adb ? adb->GetMetaByGuid(guid) : nullptr;
    if (!clip->LoadFromFile(filePath, metadata.get())) {
        INXLOG_ERROR("AudioClipLoader::Load: failed to decode: ", filePath);
        return nullptr;
    }

    clip->SetGuid(guid);

    return clip;
}

// =============================================================================
// Reload — re-decode audio and replace PCM data in-place
// =============================================================================

bool AudioClipLoader::Reload(const RuntimeAssetPayload &existing, const std::string &filePath, const std::string &guid,
                             AssetDatabase *adb)
{
    auto clip = existing.Get<AudioClip>();
    if (!clip) {
        INXLOG_WARN("AudioClipLoader::Reload: null existing instance");
        return false;
    }

    const auto metadata = adb ? adb->GetMetaByGuid(guid) : nullptr;
    AudioClip candidate;
    if (!candidate.LoadFromFile(filePath, metadata.get())) {
        INXLOG_ERROR("AudioClipLoader::Reload: failed to decode: ", filePath);
        return false;
    }

    candidate.SetGuid(guid);
    *clip = std::move(candidate);

    return true;
}

// =============================================================================
// ScanDependencies — audio clips have no outgoing asset dependencies
// =============================================================================

size_t AudioClipLoader::EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const
{
    const auto clip = payload.Get<AudioClip>();
    if (!clip)
        throw std::invalid_argument("AudioClipLoader cannot estimate an empty runtime payload");
    return clip->GetRuntimeMemoryBytes();
}

std::set<std::string> AudioClipLoader::ScanDependencies(const std::string & /*filePath*/, AssetDatabase * /*adb*/)
{
    return {};
}

// =============================================================================
// CreateMeta — audio-specific .meta creation
// =============================================================================

void AudioClipLoader::CreateMeta(AssetSource &source, InxResourceMeta &metaData) const
{
    metaData.InitFromSource(source, ResourceType::Audio);

    const auto path = ToFsPath(source.Path());
    std::string resourceName = FromFsPath(path.stem());

    metaData.AddMetadata("resource_name", resourceName);
    metaData.AddMetadata("file_type", std::string("audio"));
    metaData.AddMetadata("extension", FromFsPath(path.extension()));
}

} // namespace infernux
