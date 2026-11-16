#pragma once

#include "../MaterialDescriptor.h"
#include "../MaterialSamplerState.h"
#include "../vk/VulkanRhiDevice.h"

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
        ApplyMaterialSamplerOverrides(desc, *samplerOverride);

        publication = rhi::WithTextureSampler(device, source, desc);
        if (!publication)
            return {TextureResolveStatus::Failed, {}};
    }

    return {TextureResolveStatus::Ready,
            {device.Resolve(publication->GetView()), device.Resolve(publication->GetSampler()), slot,
             std::move(publication)}};
}

} // namespace infernux
