#pragma once

#include "../MaterialDescriptor.h"
#include "../vk/VulkanRhiDevice.h"

#include <shared_mutex>

namespace infernux
{

// Builtin UI textures have no asset GUID to resolve through the ordinary
// material texture path. Keep an overridden sampler alive with the image view
// retained by the UI descriptor, including while an old descriptor retires.
inline TextureResolveResult ResolveBuiltinUIMaterialTexture(const std::shared_ptr<rhi::TextureGpuViewSlot> &slot,
                                                            vk::VulkanRhiDevice &device,
                                                            const MaterialTextureSampler *samplerOverride)
{
    const auto source = slot ? slot->Acquire() : nullptr;
    if (!source || !source->IsValid())
        return {TextureResolveStatus::Pending, {}};

    auto publication = source;
    if (samplerOverride && samplerOverride->HasOverrides()) {
        rhi::SamplerDesc desc; // Builtin solid-color textures use the RHI defaults.
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
        desc.minFilter = filter(samplerOverride->minFilter, desc.minFilter);
        desc.magFilter = filter(samplerOverride->magFilter, desc.magFilter);
        desc.mipFilter = filter(samplerOverride->mipFilter, desc.mipFilter);
        desc.addressU = address(samplerOverride->addressU, desc.addressU);
        desc.addressV = address(samplerOverride->addressV, desc.addressV);
        desc.addressW = address(samplerOverride->addressW, desc.addressW);

        struct SamplerOwner
        {
            rhi::Device &device;
            std::shared_ptr<rhi::DeviceLifetime> lifetime;
            std::shared_ptr<const rhi::TextureGpuView> source;
            rhi::SamplerHandle sampler;

            SamplerOwner(rhi::Device &ownerDevice, std::shared_ptr<rhi::DeviceLifetime> ownerLifetime,
                         std::shared_ptr<const rhi::TextureGpuView> ownerSource)
                : device(ownerDevice), lifetime(std::move(ownerLifetime)), source(std::move(ownerSource))
            {
            }

            ~SamplerOwner()
            {
                if (!sampler.IsValid())
                    return;
                if (lifetime) {
                    std::shared_lock lock(lifetime->gate);
                    if (lifetime->alive.load(std::memory_order_acquire))
                        device.Release(sampler);
                } else {
                    device.Release(sampler);
                }
            }
        };
        auto owner = std::make_shared<SamplerOwner>(device, device.GetLifetime(), source);
        owner->sampler = device.CreateSampler(desc);
        if (!owner->sampler.IsValid())
            return {TextureResolveStatus::Failed, {}};
        const auto sampler = owner->sampler;
        publication = std::make_shared<const rhi::TextureGpuView>(
            source->GetSourceId(), source->GetRevision(), source->GetTexture(), source->GetView(), sampler,
            source->GetResidentBytes(), std::move(owner), source->GetFormat(), source->GetViewDesc());
    }

    return {TextureResolveStatus::Ready,
            {device.Resolve(publication->GetView()), device.Resolve(publication->GetSampler()), slot,
             std::move(publication)}};
}

} // namespace infernux
