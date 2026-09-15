#pragma once

#include "../rhi/RhiDescriptors.h"
#include "../rhi/RhiUpload.h"

#include <utility>
#include <vulkan/vulkan.h>

namespace infernux::rhi
{

/// Graph masks retain an empty stage set; semaphore waits explicitly request
/// AllCommands for an unspecified wait stage (Vulkan requires a nonzero mask).
[[nodiscard]] inline VkPipelineStageFlags ToVkPipelineStages(PipelineStage stages,
                                                             VkPipelineStageFlags emptyStages = 0) noexcept
{
    VkPipelineStageFlags result = 0;
    const auto has = [&](PipelineStage bit) { return (stages & bit) != PipelineStage::None; };
    if (has(PipelineStage::Top))
        result |= VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    if (has(PipelineStage::DrawIndirect))
        result |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    if (has(PipelineStage::VertexInput))
        result |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    if (has(PipelineStage::VertexShader))
        result |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    if (has(PipelineStage::FragmentShader))
        result |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (has(PipelineStage::EarlyDepth))
        result |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    if (has(PipelineStage::LateDepth))
        result |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    if (has(PipelineStage::ColorOutput))
        result |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (has(PipelineStage::ComputeShader))
        result |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (has(PipelineStage::Transfer))
        result |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (has(PipelineStage::Bottom))
        result |= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    if (has(PipelineStage::Host))
        result |= VK_PIPELINE_STAGE_HOST_BIT;
    if (has(PipelineStage::AllGraphics))
        result |= VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    if (has(PipelineStage::AllCommands))
        result |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    return result != 0 ? result : emptyStages;
}

[[nodiscard]] inline VkAccessFlags ToVkAccessFlags(Access access) noexcept
{
    constexpr std::pair<Access, VkAccessFlags> mapping[] = {
        {Access::IndirectRead, VK_ACCESS_INDIRECT_COMMAND_READ_BIT},
        {Access::IndexRead, VK_ACCESS_INDEX_READ_BIT},
        {Access::VertexRead, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT},
        {Access::UniformRead, VK_ACCESS_UNIFORM_READ_BIT},
        {Access::ShaderRead, VK_ACCESS_SHADER_READ_BIT},
        {Access::ShaderWrite, VK_ACCESS_SHADER_WRITE_BIT},
        {Access::ColorRead, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT},
        {Access::ColorWrite, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT},
        {Access::DepthRead, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT},
        {Access::DepthWrite, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
        {Access::TransferRead, VK_ACCESS_TRANSFER_READ_BIT},
        {Access::TransferWrite, VK_ACCESS_TRANSFER_WRITE_BIT},
        {Access::HostRead, VK_ACCESS_HOST_READ_BIT},
        {Access::HostWrite, VK_ACCESS_HOST_WRITE_BIT},
        {Access::MemoryRead, VK_ACCESS_MEMORY_READ_BIT},
        {Access::MemoryWrite, VK_ACCESS_MEMORY_WRITE_BIT}};
    VkAccessFlags result = 0;
    for (const auto &entry : mapping)
        if (HasAny(access, entry.first))
            result |= entry.second;
    return result;
}

struct DynamicRenderingCommands final
{
    PFN_vkCmdBeginRendering begin = nullptr;
    PFN_vkCmdEndRendering end = nullptr;

    [[nodiscard]] constexpr bool IsValid() const noexcept
    {
        return begin != nullptr && end != nullptr;
    }
};

struct Synchronization2Commands final
{
    PFN_vkCmdPipelineBarrier2 barrier = nullptr;

    [[nodiscard]] constexpr bool IsValid() const noexcept
    {
        return barrier != nullptr;
    }
};

[[nodiscard]] constexpr bool SelectDynamicRenderingPath(bool capabilityEnabled, bool corePairAvailable,
                                                        bool extensionPairAvailable) noexcept
{
    return capabilityEnabled && (corePairAvailable || extensionPairAvailable);
}

[[nodiscard]] inline DynamicRenderingCommands ResolveDynamicRenderingCommands(VkDevice device) noexcept
{
    if (device == VK_NULL_HANDLE)
        return {};

    DynamicRenderingCommands commands;
    commands.begin = reinterpret_cast<PFN_vkCmdBeginRendering>(vkGetDeviceProcAddr(device, "vkCmdBeginRendering"));
    commands.end = reinterpret_cast<PFN_vkCmdEndRendering>(vkGetDeviceProcAddr(device, "vkCmdEndRendering"));
    if (commands.IsValid())
        return commands;

    commands.begin = reinterpret_cast<PFN_vkCmdBeginRendering>(vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR"));
    commands.end = reinterpret_cast<PFN_vkCmdEndRendering>(vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR"));
    return commands.IsValid() ? commands : DynamicRenderingCommands{};
}

[[nodiscard]] inline Synchronization2Commands ResolveSynchronization2Commands(VkDevice device) noexcept
{
    if (device == VK_NULL_HANDLE)
        return {};

    Synchronization2Commands commands;
    commands.barrier =
        reinterpret_cast<PFN_vkCmdPipelineBarrier2>(vkGetDeviceProcAddr(device, "vkCmdPipelineBarrier2"));
    if (!commands.IsValid()) {
        commands.barrier =
            reinterpret_cast<PFN_vkCmdPipelineBarrier2>(vkGetDeviceProcAddr(device, "vkCmdPipelineBarrier2KHR"));
    }
    return commands.IsValid() ? commands : Synchronization2Commands{};
}

[[nodiscard]] inline VkShaderStageFlags ToVkShaderStages(ShaderStage stages) noexcept
{
    VkShaderStageFlags result = 0;
    if (HasShaderStage(stages, ShaderStage::Vertex))
        result |= VK_SHADER_STAGE_VERTEX_BIT;
    if (HasShaderStage(stages, ShaderStage::Fragment))
        result |= VK_SHADER_STAGE_FRAGMENT_BIT;
    if (HasShaderStage(stages, ShaderStage::Compute))
        result |= VK_SHADER_STAGE_COMPUTE_BIT;
    return result;
}

[[nodiscard]] constexpr VkFormat ToVkFormat(PixelFormat format) noexcept
{
    switch (format) {
    case PixelFormat::R8UNorm:
        return VK_FORMAT_R8_UNORM;
    case PixelFormat::RG8UNorm:
        return VK_FORMAT_R8G8_UNORM;
    case PixelFormat::RGBA8UNorm:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case PixelFormat::RGBA8Srgb:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case PixelFormat::BGRA8UNorm:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case PixelFormat::BGRA8Srgb:
        return VK_FORMAT_B8G8R8A8_SRGB;
    case PixelFormat::R16SFloat:
        return VK_FORMAT_R16_SFLOAT;
    case PixelFormat::RG16SFloat:
        return VK_FORMAT_R16G16_SFLOAT;
    case PixelFormat::RGBA16SFloat:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case PixelFormat::RGBA16UNorm:
        return VK_FORMAT_R16G16B16A16_UNORM;
    case PixelFormat::R32SFloat:
        return VK_FORMAT_R32_SFLOAT;
    case PixelFormat::RG32UInt:
        return VK_FORMAT_R32G32_UINT;
    case PixelFormat::RGBA32SFloat:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case PixelFormat::RGB10A2UNorm:
        return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    case PixelFormat::RGBA4UNormPack16:
        return VK_FORMAT_R4G4B4A4_UNORM_PACK16;
    case PixelFormat::BC1RgbaUNorm:
        return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case PixelFormat::BC1RgbaSrgb:
        return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case PixelFormat::BC3UNorm:
        return VK_FORMAT_BC3_UNORM_BLOCK;
    case PixelFormat::BC3Srgb:
        return VK_FORMAT_BC3_SRGB_BLOCK;
    case PixelFormat::BC4UNorm:
        return VK_FORMAT_BC4_UNORM_BLOCK;
    case PixelFormat::BC5UNorm:
        return VK_FORMAT_BC5_UNORM_BLOCK;
    case PixelFormat::BC6HUFloat:
        return VK_FORMAT_BC6H_UFLOAT_BLOCK;
    case PixelFormat::BC7UNorm:
        return VK_FORMAT_BC7_UNORM_BLOCK;
    case PixelFormat::BC7Srgb:
        return VK_FORMAT_BC7_SRGB_BLOCK;
    case PixelFormat::D32SFloat:
        return VK_FORMAT_D32_SFLOAT;
    case PixelFormat::D24UNormS8UInt:
        return VK_FORMAT_D24_UNORM_S8_UINT;
    case PixelFormat::Count:
    case PixelFormat::Undefined:
        return VK_FORMAT_UNDEFINED;
    }
    return VK_FORMAT_UNDEFINED;
}

[[nodiscard]] constexpr PixelFormat FromVkFormat(VkFormat format) noexcept
{
    switch (format) {
    case VK_FORMAT_R8_UNORM:
        return PixelFormat::R8UNorm;
    case VK_FORMAT_R8G8_UNORM:
        return PixelFormat::RG8UNorm;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return PixelFormat::RGBA8UNorm;
    case VK_FORMAT_R8G8B8A8_SRGB:
        return PixelFormat::RGBA8Srgb;
    case VK_FORMAT_B8G8R8A8_UNORM:
        return PixelFormat::BGRA8UNorm;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return PixelFormat::BGRA8Srgb;
    case VK_FORMAT_R16_SFLOAT:
        return PixelFormat::R16SFloat;
    case VK_FORMAT_R16G16_SFLOAT:
        return PixelFormat::RG16SFloat;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return PixelFormat::RGBA16SFloat;
    case VK_FORMAT_R16G16B16A16_UNORM:
        return PixelFormat::RGBA16UNorm;
    case VK_FORMAT_R32_SFLOAT:
        return PixelFormat::R32SFloat;
    case VK_FORMAT_R32G32_UINT:
        return PixelFormat::RG32UInt;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return PixelFormat::RGBA32SFloat;
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        return PixelFormat::RGB10A2UNorm;
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
        return PixelFormat::RGBA4UNormPack16;
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        return PixelFormat::BC1RgbaUNorm;
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        return PixelFormat::BC1RgbaSrgb;
    case VK_FORMAT_BC3_UNORM_BLOCK:
        return PixelFormat::BC3UNorm;
    case VK_FORMAT_BC3_SRGB_BLOCK:
        return PixelFormat::BC3Srgb;
    case VK_FORMAT_BC4_UNORM_BLOCK:
        return PixelFormat::BC4UNorm;
    case VK_FORMAT_BC5_UNORM_BLOCK:
        return PixelFormat::BC5UNorm;
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        return PixelFormat::BC6HUFloat;
    case VK_FORMAT_BC7_UNORM_BLOCK:
        return PixelFormat::BC7UNorm;
    case VK_FORMAT_BC7_SRGB_BLOCK:
        return PixelFormat::BC7Srgb;
    case VK_FORMAT_D32_SFLOAT:
        return PixelFormat::D32SFloat;
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return PixelFormat::D24UNormS8UInt;
    default:
        return PixelFormat::Undefined;
    }
}

[[nodiscard]] constexpr VkImageAspectFlags ToVkImageAspectMask(VkFormat format) noexcept
{
    switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_S8_UINT:
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

[[nodiscard]] constexpr SampleCount FromVkSampleCount(VkSampleCountFlagBits samples) noexcept
{
    switch (samples) {
    case VK_SAMPLE_COUNT_2_BIT:
        return SampleCount::Two;
    case VK_SAMPLE_COUNT_4_BIT:
        return SampleCount::Four;
    case VK_SAMPLE_COUNT_8_BIT:
        return SampleCount::Eight;
    case VK_SAMPLE_COUNT_1_BIT:
    default:
        return SampleCount::One;
    }
}

[[nodiscard]] constexpr VkSampleCountFlagBits ToVkSampleCount(SampleCount samples) noexcept
{
    switch (samples) {
    case SampleCount::One:
        return VK_SAMPLE_COUNT_1_BIT;
    case SampleCount::Two:
        return VK_SAMPLE_COUNT_2_BIT;
    case SampleCount::Four:
        return VK_SAMPLE_COUNT_4_BIT;
    case SampleCount::Eight:
        return VK_SAMPLE_COUNT_8_BIT;
    }
    return VK_SAMPLE_COUNT_1_BIT;
}

[[nodiscard]] inline bool
BuildVkPipelineRenderingInfo(const GraphicsRenderingSignature &signature,
                             std::array<VkFormat, GraphicsRenderingSignature::MaxColorTargets> &colorFormats,
                             VkPipelineRenderingCreateInfo &info) noexcept
{
    if (!signature.IsValid())
        return false;

    colorFormats.fill(VK_FORMAT_UNDEFINED);
    for (uint32_t index = 0; index < signature.colorFormatCount; ++index) {
        colorFormats[index] = ToVkFormat(signature.colorFormats[index]);
        if (colorFormats[index] == VK_FORMAT_UNDEFINED)
            return false;
    }

    info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    info.viewMask = signature.viewMask;
    info.colorAttachmentCount = signature.colorFormatCount;
    info.pColorAttachmentFormats = signature.colorFormatCount > 0 ? colorFormats.data() : nullptr;
    info.depthAttachmentFormat = ToVkFormat(signature.depthFormat);
    info.stencilAttachmentFormat = ToVkFormat(signature.stencilFormat);
    return true;
}

[[nodiscard]] constexpr VkBufferUsageFlags ToVkBufferUsage(BufferUsage usage) noexcept
{
    switch (usage) {
    case BufferUsage::Vertex:
        return VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    case BufferUsage::Index:
        return VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    case BufferUsage::Storage:
        return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    return 0;
}

} // namespace infernux::rhi
