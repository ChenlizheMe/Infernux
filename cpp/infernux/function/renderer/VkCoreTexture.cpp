#include "InxError.h"
#include "InxVkCoreModular.h"
#include "TextureUploadBuilder.h"
#include "vk/RhiVulkanTypes.h"
#include <algorithm>
#include <core/threading/FrameWorkOwner.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/InxTexture/TextureReadTask.h>

namespace infernux
{
size_t InxVkCoreModular::GetPendingTextureCpuLoadCount() const noexcept
{
    return m_pendingTextureAssetLoads.size() +
           std::count_if(m_pendingTextureUploads.begin(), m_pendingTextureUploads.end(), [](const auto &entry) {
               return !entry.second.read->IsRetired() || !entry.second.read->GetError().empty();
           });
}

size_t InxVkCoreModular::GetPendingTextureUploadCount() const noexcept
{
    return std::count_if(m_pendingTextureUploads.begin(), m_pendingTextureUploads.end(),
                         [](const auto &entry) { return bool(entry.second.ticket); });
}

bool InxVkCoreModular::AdvanceTextureUpload(PendingTextureUpload &pending, bool gpu, FrameWorkOwner *owner)
{
    if (!gpu && !pending.ticket && pending.read->GetState() != TextureReadState::Opening) {
        TextureUploadBatch batch(pending.read->GetDescription(), pending.sampler);
        pending.ticket = m_resourceManager.CreateTextureUpload(batch.GetRequest());
        return true;
    }
    if (!gpu) {
        if (owner)
            return pending.read->TryAdvance(*owner);
        pending.read->Advance();
        return true;
    } else {
        const auto &output = pending.read->GetOutput();
        const auto bytes = output.size() - pending.outputOffset;
        const auto submitted = pending.ticket->GetSubmittedBytes();
        if (!m_resourceManager.TryAppendTextureUpload(pending.ticket, output.data() + pending.outputOffset, bytes,
                                                      owner))
            return false;
        pending.outputOffset += pending.ticket->GetSubmittedBytes() - submitted;
        if (pending.outputOffset == output.size()) {
            pending.read->ConsumeOutput();
            pending.outputOffset = 0;
        }
        if (!submitted) {
            ++m_submittedTextureUploadCount;
            if (pending.ticket->IsAsync())
                ++m_asyncTextureUploadCount;
        }
        return true;
    }
}

bool InxVkCoreModular::PublishTextureUpload(const std::string &cacheKey, const PendingTextureUpload &pending)
{
    // An uploaded final block is still private until file validation succeeds.
    if (pending.read->IsPending() || pending.read->GetState() != TextureReadState::Complete || !pending.ticket ||
        !m_resourceManager.TryPublishTextureUpload(pending.ticket))
        return false;
    (void)m_textureCache.Insert(cacheKey, pending.ticket->GetTexture(), m_ensureFrameCounter, false, pending.guid,
                                pending.runtimeVersion, pending.contentGeneration);
    ++m_completedTextureUploadCount;
    return true;
}

FrameWorkSource InxVkCoreModular::GetTextureWorkSource(FrameWorkOwner &owner)
{
    m_frameWorkOwner = &owner;
    m_resourceManager.RegisterImageReadbackWork(owner);
    FrameWorkSource source;
    source.interactive = true;
    source.observe = [this] {
        m_resourceManager.PollGpuUploads();
        AssetRegistry::Instance().PollTextureReads();
        PumpPendingTextureLoads();
    };
    source.refresh = source.observe;
    source.describe = [this, &owner] {
        FrameWorkFacts facts;
        const auto &registry = AssetRegistry::Instance();
        const auto *database = registry.GetAssetDatabase();
        facts.workerBusy = registry.HasPendingTextureReads();
        facts.gpuBusy = m_resourceManager.GetPendingAsyncGraphicsSubmissionCount() != 0 ||
                        m_resourceManager.GetPendingGpuTransferCount() != 0;
        facts.pending =
            !m_pendingTextureAssetLoads.empty() || !m_pendingTextureUploads.empty() || registry.HasTextureReads();
        facts.observe = facts.pending;
        for (const auto &[key, pending] : m_pendingTextureUploads) {
            const bool gpuPending = pending.ticket && pending.ticket->IsPending();
            facts.gpuBusy |= gpuPending;
            if (database->IsStagingRuntimeArtifact(pending.guid, ResourceType::Texture))
                continue;
            if (pending.read->IsPending() || !pending.read->GetError().empty())
                continue;
            const auto state = pending.read->GetState();
            const bool create = !pending.ticket && state != TextureReadState::Opening;
            const bool read = state == TextureReadState::Opening || state == TextureReadState::Reading ||
                              state == TextureReadState::Closing;
            const bool readReady = !create && read && pending.read->CanAdvance(owner);
            const bool host = create || readReady;
            const bool gpu = pending.ticket && !gpuPending && state == TextureReadState::Output &&
                             m_resourceManager.CanAppendTextureUpload(
                                 *pending.ticket, pending.read->GetOutput().size() - pending.outputOffset, owner);
            const auto token = pending.token;
            const auto select = [token](FrameWorkCandidate &candidate, bool ready, uint64_t next) {
                if (ready && (!candidate.ready ||
                              std::pair(token < next, token) < std::pair(candidate.token < next, candidate.token))) {
                    candidate = {token, 0, true, false};
                    return true;
                }
                return false;
            };
            if (select(facts.host, host, m_nextTextureHost))
                facts.ioReady = readReady;
            select(facts.gpu, gpu, m_nextTextureGpu);
        }
        return facts;
    };
    source.advance = [this, &owner](bool gpu, uint64_t token) {
        for (auto pending = m_pendingTextureUploads.begin(); pending != m_pendingTextureUploads.end(); ++pending) {
            if (pending->second.token != token)
                continue;
            const auto guid = pending->second.guid;
            bool progressed;
            try {
                progressed = AdvanceTextureUpload(pending->second, gpu, &owner);
            } catch (const std::exception &error) {
                m_failedTextureUploads[pending->first] = {guid, pending->second.runtimeVersion,
                                                          pending->second.contentGeneration};
                INXLOG_ERROR("Texture step failed for GUID '", guid, "': ", error.what());
                m_pendingTextureUploads.erase(pending);
                m_materialPipelineManager.RefreshMaterialsUsingTexture(guid);
                progressed = true;
            }
            (gpu ? m_nextTextureGpu : m_nextTextureHost) = token + 1;
            return progressed;
        }
        return false;
    };
    return source;
}

void InxVkCoreModular::PumpPendingTextureLoads()
{
    auto &registry = AssetRegistry::Instance();
    const auto *database = registry.GetAssetDatabase();
    std::vector<std::string> changedTextures;
    for (auto pending = m_pendingTextureAssetLoads.begin(); pending != m_pendingTextureAssetLoads.end();) {
        if (!registry.IsAssetLoadContentCurrent(pending->second)) {
            changedTextures.push_back(pending->first);
            pending = m_pendingTextureAssetLoads.erase(pending);
            continue;
        }
        try {
            if (!registry.TryCommitAssetLoad(pending->second)) {
                ++pending;
                continue;
            }
            changedTextures.push_back(pending->first);
        } catch (const AssetLoadSuperseded &) {
            // Refresh the consumer so it requests the current preparation.
            changedTextures.push_back(pending->first);
        } catch (const std::exception &exception) {
            m_failedTextureUploads[pending->first] = {pending->first, registry.GetAssetVersion(pending->first),
                                                      registry.GetAssetContentGeneration(pending->first)};
            INXLOG_ERROR("Texture CPU load failed for GUID '", pending->first, "': ", exception.what());
        }
        pending = m_pendingTextureAssetLoads.erase(pending);
    }

    for (auto pending = m_pendingTextureUploads.begin(); pending != m_pendingTextureUploads.end();) {
        const auto &request = pending->second;
        if (database->IsStagingRuntimeArtifact(request.guid, ResourceType::Texture)) {
            ++pending;
            continue;
        }
        if (!registry.IsLoaded(request.guid) || registry.GetAssetVersion(request.guid) != request.runtimeVersion ||
            registry.GetAssetContentGeneration(request.guid) != request.contentGeneration) {
            changedTextures.push_back(request.guid);
            pending = m_pendingTextureUploads.erase(pending);
            continue;
        }
        try {
            if (request.read->IsPending()) {
                ++pending;
                continue;
            }
            if (!request.read->GetError().empty())
                throw std::runtime_error(request.read->GetError());
            if (!PublishTextureUpload(pending->first, request)) {
                ++pending;
                continue;
            }
            changedTextures.push_back(request.guid);
        } catch (const std::exception &error) {
            m_failedTextureUploads[pending->first] = {request.guid, request.runtimeVersion, request.contentGeneration};
            changedTextures.push_back(request.guid);
            INXLOG_ERROR("Texture preparation failed for GUID '", request.guid, "': ", error.what());
        }
        pending = m_pendingTextureUploads.erase(pending);
    }
    std::sort(changedTextures.begin(), changedTextures.end());
    changedTextures.erase(std::unique(changedTextures.begin(), changedTextures.end()), changedTextures.end());
    for (const auto &guid : changedTextures)
        m_materialPipelineManager.RefreshMaterialsUsingTexture(guid);
}

TextureResolveResult InxVkCoreModular::ResolveTextureForMaterial(const std::string &textureRef,
                                                                 const std::string &bindingName,
                                                                 const MaterialTextureSampler *sampler,
                                                                 bool waitForPreparation)
{
    bool progressed = false;
    auto resolved = ResolveTextureAsset(textureRef, bindingName, TextureDimension::Texture2D, nullptr, nullptr, sampler,
                                        progressed, waitForPreparation);
    if (progressed && m_frameWorkOwner)
        m_frameWorkOwner->ReportProgress();
    return resolved;
}

TextureResolveResult InxVkCoreModular::ResolveTextureForVectorField(const std::string &textureGuid,
                                                                    bool linearFiltering, bool repeat,
                                                                    bool waitForPreparation)
{
    bool progressed = false;
    auto resolved = ResolveTextureAsset(textureGuid, "ParticleVectorField", TextureDimension::Texture3D,
                                        linearFiltering ? "bilinear" : "point", repeat ? "repeat" : "clamp", nullptr,
                                        progressed, waitForPreparation);
    if (progressed && m_frameWorkOwner)
        m_frameWorkOwner->ReportProgress();
    return resolved;
}

TextureResolveResult InxVkCoreModular::ResolveTextureForGraph(const std::string &textureGuid, bool volume,
                                                              bool waitForPreparation)
{
    bool progressed = false;
    auto resolved = ResolveTextureAsset(textureGuid, "RenderGraph",
                                        volume ? TextureDimension::Texture3D : TextureDimension::Texture2D, nullptr,
                                        nullptr, nullptr, progressed, waitForPreparation);
    if (progressed && m_frameWorkOwner)
        m_frameWorkOwner->ReportProgress();
    return resolved;
}

TextureResolveResult InxVkCoreModular::ResolveTextureAsset(const std::string &textureGuid,
                                                           const std::string &bindingName,
                                                           TextureDimension expectedDimension,
                                                           const char *filterOverride, const char *wrapOverride,
                                                           const MaterialTextureSampler *samplerOverride,
                                                           bool &progressed, bool waitForPreparation)
{
    // Material texture properties store asset GUIDs. Path normalization belongs
    // at the public material/asset boundary, never in the renderer.
    std::string texturePath;
    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    if (adb && waitForPreparation && (!m_frameWorkOwner || !m_frameWorkOwner->IsActive()) &&
        adb->IsStagingRuntimeArtifact(textureGuid, ResourceType::Texture))
        adb->CompleteRuntimeArtifactPublication();
    const bool staging = adb && adb->IsStagingRuntimeArtifact(textureGuid, ResourceType::Texture);
    if (adb)
        texturePath = adb->GetPathFromGuid(textureGuid);

    if (texturePath.empty()) {
        if (staging)
            return {TextureResolveStatus::Pending, {}};
        INXLOG_WARN("TextureResolver: texture reference '", textureGuid, "' is not a resolvable asset GUID (binding='",
                    bindingName, "'). Texture properties must hold GUIDs.");
        return {TextureResolveStatus::Failed, {}};
    }

    // A published GPU view remains authoritative after its decoded CPU asset
    // has been evicted. Resolve it first so CPU residency policy cannot turn a
    // stable material into a per-frame decode/upload loop.
    const std::string filterIdentity = filterOverride ? filterOverride : "asset";
    const std::string wrapIdentity = wrapOverride ? wrapOverride : "asset";
    const std::string samplerIdentity =
        samplerOverride ? std::to_string(static_cast<uint32_t>(samplerOverride->minFilter)) + "," +
                              std::to_string(static_cast<uint32_t>(samplerOverride->magFilter)) + "," +
                              std::to_string(static_cast<uint32_t>(samplerOverride->mipFilter)) + "," +
                              std::to_string(static_cast<uint32_t>(samplerOverride->addressU)) + "," +
                              std::to_string(static_cast<uint32_t>(samplerOverride->addressV)) + "," +
                              std::to_string(static_cast<uint32_t>(samplerOverride->addressW))
                        : "asset";
    const std::string cacheKey = textureGuid + "::dim" + std::to_string(static_cast<uint32_t>(expectedDimension)) +
                                 "::filter=" + filterIdentity + "::wrap=" + wrapIdentity +
                                 "::bindingSampler=" + samplerIdentity;
    const uint64_t publishedRuntimeVersion = registry.GetAssetVersion(textureGuid);
    const uint64_t contentGeneration = registry.GetAssetContentGeneration(textureGuid);
    if (publishedRuntimeVersion != 0) {
        auto cachedSlot = m_textureCache.FindAsset(cacheKey, textureGuid, publishedRuntimeVersion, contentGeneration,
                                                   m_ensureFrameCounter);
        auto cached = cachedSlot ? cachedSlot->Acquire() : nullptr;
        if (cached && cached->IsValid()) {
            auto &rhiDevice = m_backend.Device().GetRhiDevice();
            return {TextureResolveStatus::Ready,
                    {rhiDevice.Resolve(cached->GetView()), rhiDevice.Resolve(cached->GetSampler()),
                     std::move(cachedSlot), std::move(cached)},
                    contentGeneration};
        }
    }

    if (staging)
        return {TextureResolveStatus::Pending, {}};

    const auto failed = m_failedTextureUploads.find(cacheKey);
    if (failed != m_failedTextureUploads.end() && failed->second.runtimeVersion == publishedRuntimeVersion &&
        failed->second.contentGeneration == contentGeneration)
        return {TextureResolveStatus::Failed, {}};

    auto infTex = registry.GetAsset<InxTexture>(textureGuid);
    if (!infTex) {
        const auto failedLoad = m_failedTextureUploads.find(textureGuid);
        if (failedLoad != m_failedTextureUploads.end() &&
            failedLoad->second.runtimeVersion == publishedRuntimeVersion &&
            failedLoad->second.contentGeneration == contentGeneration)
            return {TextureResolveStatus::Failed, {}};
        auto pending = m_pendingTextureAssetLoads.find(textureGuid);
        if (pending == m_pendingTextureAssetLoads.end()) {
            try {
                pending = m_pendingTextureAssetLoads
                              .emplace(textureGuid, registry.BeginLoadAsset(textureGuid, ResourceType::Texture))
                              .first;
                progressed = true;
            } catch (const std::exception &exception) {
                m_failedTextureUploads[textureGuid] = {textureGuid, publishedRuntimeVersion, contentGeneration};
                progressed = true;
                INXLOG_ERROR("TextureResolver: failed to schedule texture metadata load for '", textureGuid,
                             "': ", exception.what());
                return {TextureResolveStatus::Failed, {}};
            }
        }
        try {
            if (!registry.IsAssetLoadContentCurrent(pending->second)) {
                m_pendingTextureAssetLoads.erase(pending);
                progressed = true;
                return {TextureResolveStatus::Pending, {}};
            }
            if (waitForPreparation)
                pending->second->Wait();
            if (!registry.TryCommitAssetLoad(pending->second))
                return {TextureResolveStatus::Pending, {}};
            m_pendingTextureAssetLoads.erase(pending);
            progressed = true;
            infTex = registry.GetAsset<InxTexture>(textureGuid);
            if (!infTex)
                return {TextureResolveStatus::Pending, {}};
        } catch (const AssetLoadSuperseded &) {
            m_pendingTextureAssetLoads.erase(textureGuid);
            progressed = true;
            return {TextureResolveStatus::Pending, {}};
        } catch (const std::exception &exception) {
            m_failedTextureUploads[textureGuid] = {textureGuid, publishedRuntimeVersion, contentGeneration};
            INXLOG_ERROR("TextureResolver: metadata load failed for '", textureGuid, "': ", exception.what());
            m_pendingTextureAssetLoads.erase(textureGuid);
            progressed = true;
            return {TextureResolveStatus::Failed, {}};
        }
    }
    if (!infTex || infTex->GetMipCount() == 0) {
        INXLOG_ERROR("TextureResolver: texture asset has no current imported artifact description: ", textureGuid);
        return {TextureResolveStatus::Failed, {}};
    }
    if (infTex->GetDimension() != expectedDimension) {
        INXLOG_ERROR("TextureResolver: binding '", bindingName, "' expected texture dimension ",
                     static_cast<uint32_t>(expectedDimension), " but asset uses ",
                     static_cast<uint32_t>(infTex->GetDimension()), ": ", textureGuid);
        return {TextureResolveStatus::Failed, {}};
    }

    const uint64_t runtimeVersion = registry.GetAssetVersion(textureGuid);
    if (runtimeVersion == 0)
        throw std::logic_error("TextureResolver resolved a payload without a published runtime version");
    const std::string filterMode = filterOverride ? filterOverride : infTex->GetFilterMode();
    const std::string wrapMode = wrapOverride ? wrapOverride : infTex->GetWrapMode();
    // Simulation volumes use point or linear interpolation at a single position.
    const int anisoLevel = filterOverride ? 1 : infTex->GetAnisoLevel();
    rhi::SamplerDesc sampler;
    sampler.maxLod = static_cast<float>(infTex->GetMipCount() - 1);
    const auto &capabilities = m_backend.Device().GetCapabilities();
    const float deviceMaxAnisotropy =
        capabilities.features.samplerAnisotropy ? (std::max)(1.0f, capabilities.limits.maxSamplerAnisotropy) : 1.0f;
    const float requestedAnisotropy = anisoLevel < 0 ? deviceMaxAnisotropy : static_cast<float>(anisoLevel);
    sampler.maxAnisotropy = (std::min)((std::max)(1.0f, requestedAnisotropy), deviceMaxAnisotropy);
    if (filterMode == "point") {
        sampler.minFilter = rhi::FilterMode::Nearest;
        sampler.magFilter = rhi::FilterMode::Nearest;
        sampler.mipFilter = rhi::FilterMode::Nearest;
        sampler.maxAnisotropy = 1.0f;
    } else if (filterMode == "trilinear") {
        sampler.mipFilter = rhi::FilterMode::Linear;
    } else if (filterMode == "linear" || filterMode == "bilinear") {
        sampler.mipFilter = rhi::FilterMode::Nearest;
    } else {
        INXLOG_ERROR("TextureResolver: unsupported filter mode '", filterMode, "' for texture ", textureGuid);
        return {TextureResolveStatus::Failed, {}};
    }

    if (wrapMode == "clamp")
        sampler.addressU = sampler.addressV = sampler.addressW = rhi::AddressMode::ClampToEdge;
    else if (wrapMode == "mirror")
        sampler.addressU = sampler.addressV = sampler.addressW = rhi::AddressMode::MirroredRepeat;

    if (samplerOverride) {
        const auto filter = [](MaterialSamplerFilter value, rhi::FilterMode inherited) {
            if (value == MaterialSamplerFilter::Nearest)
                return rhi::FilterMode::Nearest;
            if (value == MaterialSamplerFilter::Linear)
                return rhi::FilterMode::Linear;
            return inherited;
        };
        const auto address = [](MaterialSamplerAddress value, rhi::AddressMode inherited) {
            if (value == MaterialSamplerAddress::Repeat)
                return rhi::AddressMode::Repeat;
            if (value == MaterialSamplerAddress::Clamp)
                return rhi::AddressMode::ClampToEdge;
            if (value == MaterialSamplerAddress::Mirror)
                return rhi::AddressMode::MirroredRepeat;
            return inherited;
        };
        sampler.minFilter = filter(samplerOverride->minFilter, sampler.minFilter);
        sampler.magFilter = filter(samplerOverride->magFilter, sampler.magFilter);
        sampler.mipFilter = filter(samplerOverride->mipFilter, sampler.mipFilter);
        sampler.addressU = address(samplerOverride->addressU, sampler.addressU);
        sampler.addressV = address(samplerOverride->addressV, sampler.addressV);
        sampler.addressW = address(samplerOverride->addressW, sampler.addressW);
        if (sampler.minFilter == rhi::FilterMode::Nearest && sampler.magFilter == rhi::FilterMode::Nearest)
            sampler.maxAnisotropy = 1.0f;
    }

    // Cache key uses GUID so that a renamed file still shares its cache entry
    // The cache key identifies a stable consumer slot, not the mutable import
    // result. Format, color space and asset-controlled sampler settings may
    // change and must publish a new revision into the same slot.
    // Check texture cache again after CPU resolution: another consumer may
    // have completed publication while this request was pending.
    {
        auto cachedSlot =
            m_textureCache.FindAsset(cacheKey, textureGuid, runtimeVersion, contentGeneration, m_ensureFrameCounter);
        auto cached = cachedSlot ? cachedSlot->Acquire() : nullptr;
        if (cached && cached->IsValid()) {
            auto &rhiDevice = m_backend.Device().GetRhiDevice();
            return {TextureResolveStatus::Ready,
                    {rhiDevice.Resolve(cached->GetView()), rhiDevice.Resolve(cached->GetSampler()),
                     std::move(cachedSlot), std::move(cached)},
                    contentGeneration};
        }
    }

    auto pending = m_pendingTextureUploads.find(cacheKey);
    if (pending != m_pendingTextureUploads.end() &&
        (pending->second.runtimeVersion != runtimeVersion || pending->second.contentGeneration != contentGeneration)) {
        m_pendingTextureUploads.erase(pending);
        pending = m_pendingTextureUploads.end();
        progressed = true;
    }
    try {
        if (pending == m_pendingTextureUploads.end()) {
            PendingTextureUpload request;
            request.guid = textureGuid;
            request.runtimeVersion = runtimeVersion;
            request.contentGeneration = contentGeneration;
            request.token = m_nextTextureWorkToken++;
            request.sampler = sampler;
            request.read = registry.BeginTextureRead(textureGuid);
            pending = m_pendingTextureUploads.emplace(cacheKey, std::move(request)).first;
            progressed = true;
        }
        auto &request = pending->second;
        // Explicit cold preparation drives the exact same steps to completion.
        // Ordinary frame consumers only enqueue and return Pending.
        if (waitForPreparation) {
            while (true) {
                if (request.read->IsPending())
                    request.read->Wait();
                if (m_frameWorkOwner)
                    request.read->Poll(*m_frameWorkOwner);
                if (!request.read->GetError().empty())
                    throw std::runtime_error(request.read->GetError());
                const auto state = request.read->GetState();
                if (state == TextureReadState::Complete)
                    break;
                if (request.ticket && request.ticket->IsPending())
                    m_resourceManager.DrainBufferUploads();
                const bool gpu = request.ticket && state == TextureReadState::Output;
                (void)AdvanceTextureUpload(request, gpu);
            }
            if (request.ticket && request.ticket->IsPending())
                m_resourceManager.DrainBufferUploads();
        }
        if (!PublishTextureUpload(cacheKey, request))
            return {TextureResolveStatus::Pending, {}};
        m_pendingTextureUploads.erase(pending);
        progressed = true;
        auto slot =
            m_textureCache.FindAsset(cacheKey, textureGuid, runtimeVersion, contentGeneration, m_ensureFrameCounter);
        auto resident = slot->Acquire();
        auto &device = m_backend.Device().GetRhiDevice();
        return {TextureResolveStatus::Ready,
                {device.Resolve(resident->GetView()), device.Resolve(resident->GetSampler()), std::move(slot),
                 std::move(resident)},
                contentGeneration};
    } catch (const std::exception &error) {
        m_failedTextureUploads[cacheKey] = {textureGuid, runtimeVersion, contentGeneration};
        m_pendingTextureUploads.erase(cacheKey);
        progressed = true;
        INXLOG_ERROR("Texture preparation failed for GUID '", textureGuid, "': ", error.what());
        return {TextureResolveStatus::Failed, {}};
    }
}

std::shared_ptr<const rhi::TextureGpuView>
InxVkCoreModular::ResolveTextureForEditorPreview(const std::string &textureGuid, bool &progressed, bool &pending)
{
    TextureResolveResult resolved = ResolveTextureAsset(
        textureGuid, "EditorTexturePreview", TextureDimension::Texture2D, nullptr, nullptr, nullptr, progressed);
    pending = resolved.status == TextureResolveStatus::Pending;
    return resolved.status == TextureResolveStatus::Ready ? std::move(resolved.binding.gpuView) : nullptr;
}

} // namespace infernux
