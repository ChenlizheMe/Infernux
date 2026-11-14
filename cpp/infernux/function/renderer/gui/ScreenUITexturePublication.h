#pragma once

#include "../rhi/RhiRenderTexture.h"
#include "ImGuiTextureAllocationLease.h"
#include <cstdint>
#include <memory>

namespace infernux
{
// Created by the texture producer. No runtime Component, registry lookup or
// mutable allocation slot is needed after the UI command captures this value.
struct ScreenUITexturePublication final
{
    uint64_t textureId = 0;
    uint64_t generation = 0;
    bool requiresDisplayEncoding = false;
    std::shared_ptr<ImGuiTextureAllocationLease> allocation;
    std::shared_ptr<rhi::RenderTexture> renderTexture;
    std::shared_ptr<const rhi::RenderTextureGeneration> renderTextureGeneration;
    std::shared_ptr<const void> pixelLease;
};
} // namespace infernux
