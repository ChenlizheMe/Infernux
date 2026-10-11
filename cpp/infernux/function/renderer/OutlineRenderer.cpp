/**
 * @file OutlineRenderer.cpp
 * @brief Post-process selection outline renderer implementation
 *
 * Extracted from InxVkCoreModular.cpp during editor/renderer separation.
 */

#include "OutlineRenderer.h"
#include "InxVkCoreModular.h"
#include "MaterialDescriptor.h"
#include "MaterialPipelineManager.h"
#include "SceneRenderTarget.h"
#include "VertexInputFilter.h"
#include "shader/ShaderProgram.h"
#include "shader/ShaderReflection.h"
#include "vk/DescriptorBindTrace.h"
#include "vk/RhiVulkanTypes.h"
#include "vk/VkPipelineHelpers.h"
#include "vk/VkRenderUtils.h"
#include <core/types/ColorSpace.h>
#include <function/resources/InxMaterial/InxMaterial.h>

#include <core/error/InxError.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vk_mem_alloc.h>

#include <array>
#include <cstring>
#include <vector>

namespace infernux
{

namespace
{

using infernux::vkrender::MakeMultisampleState;
using infernux::vkrender::MakeShaderStageInfo;
using infernux::vkrender::MakeTriangleListInputAssembly;
using DynamicViewportState = infernux::vkrender::DynamicViewportScissorState;

struct MeshVertexInputState
{
    VkVertexInputBindingDescription bindingDesc{};
    std::vector<VkVertexInputAttributeDescription> attrDescs;
    VkPipelineVertexInputStateCreateInfo createInfo{};

    MeshVertexInputState()
        : bindingDesc(vk::GetVertexBindingDescription()),
          attrDescs(FilterVertexAttributesForReflection(ShaderReflection{}))
    {
        initCreateInfo();
    }

    explicit MeshVertexInputState(const ShaderReflection &vertexReflection)
        : bindingDesc(vk::GetVertexBindingDescription()),
          attrDescs(FilterVertexAttributesForReflection(vertexReflection))
    {
        initCreateInfo();
    }

    void initCreateInfo()
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        createInfo.vertexBindingDescriptionCount = attrDescs.empty() ? 0u : 1u;
        createInfo.pVertexBindingDescriptions = attrDescs.empty() ? nullptr : &bindingDesc;
        createInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
        createInfo.pVertexAttributeDescriptions = attrDescs.empty() ? nullptr : attrDescs.data();
    }
};

VkPipelineRasterizationStateCreateInfo MakeRasterizationState(VkCullModeFlags cullMode)
{
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1.0f;
    raster.cullMode = cullMode;
    raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
    return raster;
}

VkPipelineDepthStencilStateCreateInfo MakeDepthStencilState(VkBool32 depthTestEnable, VkBool32 depthWriteEnable)
{
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = depthTestEnable;
    depthStencil.depthWriteEnable = depthWriteEnable;
    return depthStencil;
}

VkPipelineColorBlendAttachmentState MakeOpaqueColorBlendAttachment()
{
    VkPipelineColorBlendAttachmentState colorBlendAttach{};
    colorBlendAttach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttach.blendEnable = VK_FALSE;
    return colorBlendAttach;
}

VkPipelineColorBlendAttachmentState MakeAlphaBlendAttachment()
{
    VkPipelineColorBlendAttachmentState colorBlendAttach = MakeOpaqueColorBlendAttachment();
    colorBlendAttach.blendEnable = VK_TRUE;
    colorBlendAttach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttach.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttach.alphaBlendOp = VK_BLEND_OP_ADD;
    return colorBlendAttach;
}

VkPipelineColorBlendStateCreateInfo MakeColorBlendState(const VkPipelineColorBlendAttachmentState &attachment)
{
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &attachment;
    return colorBlend;
}

} // namespace

// ============================================================================
// Destructor
// ============================================================================

OutlineRenderer::~OutlineRenderer()
{
    Cleanup();
}

// ============================================================================
// Lifecycle
// ============================================================================

bool OutlineRenderer::Initialize(InxVkCoreModular *core, SceneRenderTarget *sceneTarget)
{
    if (m_resourcesReady)
        return true;

    if (!core || !sceneTarget || !sceneTarget->IsReady()) {
        INXLOG_WARN("OutlineRenderer::Initialize: core or SceneRenderTarget not ready");
        return false;
    }

    m_core = core;
    m_sceneRenderTarget = sceneTarget;

    // Editor support shaders are lazy assets. Resolve them through the same
    // catalog path as materials/fullscreen effects instead of polling the raw
    // module cache forever when startup has deliberately skipped eager scans.
    if (!m_core->EnsureShaderAvailable("Outline Mask", "vertex") ||
        !m_core->EnsureShaderAvailable("Outline Mask", "fragment") ||
        !m_core->EnsureShaderAvailable("Outline Composite", "vertex") ||
        !m_core->EnsureShaderAvailable("Outline Composite", "fragment")) {
        if (!m_missingShadersReported) {
            INXLOG_WARN("OutlineRenderer::Initialize: outline shaders are unavailable");
            m_missingShadersReported = true;
        }
        return false;
    }
    m_missingShadersReported = false;

    CreateOutlineDescriptorResources();
    CreateOutlinePipelineLayouts();
    CreateOutlineMaterialResources();

    m_resourcesReady = true;
    return true;
}

void OutlineRenderer::Cleanup(bool waitForIdle)
{
    const bool hasOwnedResources =
        m_resourcesReady || m_outlineMaskPipeline != VK_NULL_HANDLE || m_outlineMaskPipelineLayout != VK_NULL_HANDLE ||
        m_outlineMaskDescSetLayout != VK_NULL_HANDLE || m_outlineCompositePipeline != VK_NULL_HANDLE ||
        m_outlineCompositePipelineLayout != VK_NULL_HANDLE || m_outlineCompositeDescSetLayout != VK_NULL_HANDLE ||
        m_outlineCompositeDescSet != VK_NULL_HANDLE || m_outlineCompositeDescLease.IsValid() ||
        !m_outlineInstanceBufs.empty() || !m_outlineSkinInstanceBufs.empty() || !m_outlineSkinPaletteBufs.empty() ||
        !m_outlineInstanceAuxBufs.empty() || !m_outlineGlobalsDescSets.empty() || !m_outlineGlobalsDescLeases.empty() ||
        !m_perMtlOutlinePipelines.empty();
    if (!hasOwnedResources)
        return;

    VkDevice device = m_core ? m_core->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE)
        return;

    if (waitForIdle && !m_core->IsShuttingDown()) {
        m_core->GetDeviceContext().WaitIdle();
    }

    auto &descriptorManager = m_core->GetDeviceContext().GetRhiDevice().GetDescriptorManager();
    descriptorManager.Retire(m_outlineCompositeDescLease);
    for (const auto &lease : m_outlineGlobalsDescLeases)
        descriptorManager.Retire(lease);
    m_outlineCompositeDescLease = {};
    m_outlineGlobalsDescLeases.clear();

    DestroyOutlinePipelines();
    vkrender::SafeDestroy(device, m_outlineMaskPipelineLayout);
    vkrender::SafeDestroy(device, m_outlineCompositePipelineLayout);
    vkrender::SafeDestroy(device, m_outlineMaskDescSetLayout);
    vkrender::SafeDestroy(device, m_outlineCompositeDescSetLayout);

    m_outlineGlobalsDescSets.clear();

    VmaAllocator allocator = m_core->GetDeviceContext().GetVmaAllocator();
    for (auto &instBuf : m_outlineInstanceBufs) {
        if (instBuf.buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator, instBuf.buffer, instBuf.allocation);
    }
    m_outlineInstanceBufs.clear();
    for (auto &skinBuf : m_outlineSkinInstanceBufs) {
        if (skinBuf.buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator, skinBuf.buffer, skinBuf.allocation);
    }
    m_outlineSkinInstanceBufs.clear();
    for (auto &skinBuf : m_outlineSkinPaletteBufs) {
        if (skinBuf.buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator, skinBuf.buffer, skinBuf.allocation);
    }
    m_outlineSkinPaletteBufs.clear();
    for (auto &auxBuf : m_outlineInstanceAuxBufs) {
        if (auxBuf.buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator, auxBuf.buffer, auxBuf.allocation);
    }
    m_outlineInstanceAuxBufs.clear();

    m_outlineCompositeDescSet = VK_NULL_HANDLE;
    m_outlineMaskRenderingSignature = {};
    m_outlineCompositeRenderingSignature = {};
    m_resourcesReady = false;
}

// ============================================================================
// Rendering
// ============================================================================

bool OutlineRenderer::EnsureGraphPipelines(const rhi::GraphicsRenderingSignature &maskSignature,
                                           const rhi::GraphicsRenderingSignature &compositeSignature)
{
    if (!m_resourcesReady || !maskSignature.IsValid() || !compositeSignature.IsValid())
        return false;
    if (m_outlineMaskRenderingSignature == maskSignature &&
        m_outlineCompositeRenderingSignature == compositeSignature && m_outlineMaskPipeline != VK_NULL_HANDLE &&
        m_outlineCompositePipeline != VK_NULL_HANDLE) {
        return true;
    }

    DestroyOutlinePipelines();
    m_outlineMaskRenderingSignature = maskSignature;
    m_outlineCompositeRenderingSignature = compositeSignature;
    return CreateOutlinePipelines();
}

// ============================================================================
// Internal: Vulkan Resource Creation
// ============================================================================

void OutlineRenderer::CreateOutlineDescriptorResources()
{
    VkDevice device = m_core->GetDevice();
    auto &descriptorManager = m_core->GetDeviceContext().GetRhiDevice().GetDescriptorManager();

    // --- Empty material set 0. Camera state lives in canonical set 1. ---
    {
        vkrender::CreateDescriptorSetLayout(device, nullptr, 0, m_outlineMaskDescSetLayout);
    }

    // --- Composite descriptor set layout: binding 0 = sampler (mask texture) ---
    {
        const VkDescriptorSetLayoutBinding samplerBinding = vkrender::MakeDescriptorSetLayoutBinding(
            0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);

        vkrender::CreateDescriptorSetLayout(device, &samplerBinding, 1, m_outlineCompositeDescSetLayout);
        m_outlineCompositeDescLease =
            descriptorManager.Allocate(m_outlineCompositeDescSetLayout, vk::DescriptorArena::ViewPersistent);
        m_outlineCompositeDescSet = m_outlineCompositeDescLease.set;
        if (!m_outlineCompositeDescLease.IsValid()) {
            INXLOG_ERROR("OutlineRenderer: Failed to allocate outline composite descriptor set");
            return;
        }

        // Write mask texture to binding 0
        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = m_sceneRenderTarget->GetOutlineMaskSampler();
        imageInfo.imageView = m_sceneRenderTarget->GetOutlineMaskImageView();
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkrender::UpdateDescriptorSetWithImage(device, m_outlineCompositeDescSet, 0, imageInfo);
    }
}

void OutlineRenderer::CreateOutlinePipelineLayouts()
{
    VkDevice device = m_core->GetDevice();

    {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushRange.offset = 0;
        pushRange.size = 128; // 2 x mat4

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VkDescriptorSetLayout setLayouts[3] = {m_outlineMaskDescSetLayout, m_core->GetPerViewDescSetLayout(),
                                               m_core->GetGlobalsDescSetLayout()};
        layoutInfo.setLayoutCount = 3;
        layoutInfo.pSetLayouts = setLayouts;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_outlineMaskPipelineLayout);
    }

    {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = 32; // vec4 color + vec2 texelSize + float width + float padding

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_outlineCompositeDescSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_outlineCompositePipelineLayout);
    }
}

bool OutlineRenderer::CreateOutlinePipelines()
{
    if (!m_core || !m_outlineMaskRenderingSignature.IsValid() || !m_outlineCompositeRenderingSignature.IsValid() ||
        m_outlineMaskPipelineLayout == VK_NULL_HANDLE || m_outlineCompositePipelineLayout == VK_NULL_HANDLE) {
        return false;
    }

    VkDevice device = m_core->GetDevice();
    {
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
            MakeShaderStageInfo(VK_SHADER_STAGE_VERTEX_BIT, m_core->GetShaderModule("Outline Mask", "vertex")),
            MakeShaderStageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, m_core->GetShaderModule("Outline Mask", "fragment")),
        };

        ShaderReflection outlineMaskVertRefl;
        const auto *outlineMaskVertSpv = m_core->GetShaderCache().FindVertCode("Outline Mask");
        if (!outlineMaskVertSpv || !outlineMaskVertRefl.Reflect(*outlineMaskVertSpv, VK_SHADER_STAGE_VERTEX_BIT))
            outlineMaskVertRefl.Clear();

        m_outlineMaskPipeline = CreateMaskPipeline(stages.data(), m_outlineMaskPipelineLayout, outlineMaskVertRefl);
        if (m_outlineMaskPipeline == VK_NULL_HANDLE)
            INXLOG_ERROR("OutlineRenderer: Failed to create graph-compatible outline mask pipeline");
    }

    {
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
            MakeShaderStageInfo(VK_SHADER_STAGE_VERTEX_BIT, m_core->GetShaderModule("Outline Composite", "vertex")),
            MakeShaderStageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, m_core->GetShaderModule("Outline Composite", "fragment")),
        };

        // No vertex input (fullscreen triangle is procedural)
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = MakeTriangleListInputAssembly();
        DynamicViewportState viewportState;
        VkPipelineRasterizationStateCreateInfo raster = MakeRasterizationState(VK_CULL_MODE_NONE);
        VkPipelineDepthStencilStateCreateInfo depthStencil = MakeDepthStencilState(VK_FALSE, VK_FALSE);
        VkPipelineMultisampleStateCreateInfo multisampling =
            MakeMultisampleState(rhi::ToVkSampleCount(m_outlineCompositeRenderingSignature.samples));
        VkPipelineColorBlendAttachmentState colorBlendAttach = MakeAlphaBlendAttachment();
        VkPipelineColorBlendStateCreateInfo colorBlend = MakeColorBlendState(colorBlendAttach);

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState.viewportState;
        pipelineInfo.pRasterizationState = &raster;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlend;
        pipelineInfo.pDynamicState = &viewportState.dynamicState;
        pipelineInfo.layout = m_outlineCompositePipelineLayout;
        std::array<VkFormat, rhi::GraphicsRenderingSignature::MaxColorTargets> colorFormats{};
        VkPipelineRenderingCreateInfo renderingInfo{};
        if (!rhi::BuildVkPipelineRenderingInfo(m_outlineCompositeRenderingSignature, colorFormats, renderingInfo))
            return false;
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.renderPass = VK_NULL_HANDLE;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_outlineCompositePipeline) !=
            VK_SUCCESS) {
            INXLOG_ERROR("OutlineRenderer: Failed to create graph-compatible outline composite pipeline");
        }
    }

    return m_outlineMaskPipeline != VK_NULL_HANDLE && m_outlineCompositePipeline != VK_NULL_HANDLE;
}

void OutlineRenderer::DestroyOutlinePipelines()
{
    if (!m_core)
        return;
    VkDevice device = m_core->GetDevice();
    std::vector<VkPipeline> pipelines;
    if (m_outlineMaskPipeline != VK_NULL_HANDLE)
        pipelines.push_back(m_outlineMaskPipeline);
    if (m_outlineCompositePipeline != VK_NULL_HANDLE)
        pipelines.push_back(m_outlineCompositePipeline);
    for (const auto &[key, pipeline] : m_perMtlOutlinePipelines) {
        (void)key;
        if (pipeline != VK_NULL_HANDLE)
            pipelines.push_back(pipeline);
    }
    m_outlineMaskPipeline = VK_NULL_HANDLE;
    m_outlineCompositePipeline = VK_NULL_HANDLE;
    m_perMtlOutlinePipelines.clear();
    m_perMtlOutlineProgramIds.clear();

    if (pipelines.empty())
        return;
    auto destroy = [device, pipelines = std::move(pipelines)] {
        for (VkPipeline pipeline : pipelines)
            vkDestroyPipeline(device, pipeline, nullptr);
    };
    if (!m_core->IsShuttingDown())
        m_core->RetireGpuResource(std::move(destroy));
    else
        destroy();
}

// ============================================================================
// Per-material outline mask pipeline resources
// ============================================================================

void OutlineRenderer::CreateOutlineMaterialResources()
{
    VkDevice device = m_core->GetDevice();
    const uint32_t framesInFlight = m_core->GetMaxFramesInFlight();
    m_outlineInstanceBufs.resize(framesInFlight);
    m_outlineSkinInstanceBufs.resize(framesInFlight);
    m_outlineSkinPaletteBufs.resize(framesInFlight);
    m_outlineInstanceAuxBufs.resize(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; ++i)
        EnsureOutlineBufferCapacity(i, 1, 1);

    // --- Per-frame outline globals descriptor sets ---
    {
        VkDescriptorSetLayout globalsLayout = m_core->GetGlobalsDescSetLayout();
        auto &descriptorManager = m_core->GetDeviceContext().GetRhiDevice().GetDescriptorManager();
        m_outlineGlobalsDescSets.resize(framesInFlight);
        m_outlineGlobalsDescLeases.reserve(framesInFlight);

        for (uint32_t i = 0; i < framesInFlight; ++i) {
            auto lease = descriptorManager.Allocate(globalsLayout, vk::DescriptorArena::ViewPersistent);
            if (!lease.IsValid()) {
                INXLOG_ERROR("OutlineRenderer: Failed to allocate outline globals descriptor set for frame ", i);
                continue;
            }
            m_outlineGlobalsDescSets[i] = lease.set;
            m_outlineGlobalsDescLeases.push_back(lease);

            // Binding 0: globals UBO (same as engine frame i)
            VkDescriptorBufferInfo uboBufInfo{};
            uboBufInfo.buffer = m_core->GetGlobalsBuffer(i);
            uboBufInfo.offset = 0;
            uboBufInfo.range = VK_WHOLE_SIZE;
            vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[i], 0,
                                                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uboBufInfo);

            // Binding 1: transforms for selected draws
            VkDescriptorBufferInfo ssboBufInfo{};
            ssboBufInfo.buffer = m_outlineInstanceBufs[i].buffer;
            ssboBufInfo.offset = 0;
            ssboBufInfo.range = VK_WHOLE_SIZE;
            vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[i], 1,
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ssboBufInfo);

            if (i < m_outlineSkinInstanceBufs.size() && i < m_outlineSkinPaletteBufs.size()) {
                if (m_outlineSkinInstanceBufs[i].buffer != VK_NULL_HANDLE) {
                    VkDescriptorBufferInfo skinInstInfo{};
                    skinInstInfo.buffer = m_outlineSkinInstanceBufs[i].buffer;
                    skinInstInfo.offset = 0;
                    skinInstInfo.range = VK_WHOLE_SIZE;
                    vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[i], 2,
                                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, skinInstInfo);
                }

                if (m_outlineSkinPaletteBufs[i].buffer != VK_NULL_HANDLE) {
                    VkDescriptorBufferInfo skinPaletteInfo{};
                    skinPaletteInfo.buffer = m_outlineSkinPaletteBufs[i].buffer;
                    skinPaletteInfo.offset = 0;
                    skinPaletteInfo.range = VK_WHOLE_SIZE;
                    vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[i], 3,
                                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, skinPaletteInfo);
                }
            }

            if (i < m_outlineInstanceAuxBufs.size() && m_outlineInstanceAuxBufs[i].buffer != VK_NULL_HANDLE) {
                VkDescriptorBufferInfo instanceAuxInfo{};
                instanceAuxInfo.buffer = m_outlineInstanceAuxBufs[i].buffer;
                instanceAuxInfo.offset = 0;
                instanceAuxInfo.range = sizeof(GPUInstanceAuxData);
                vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[i], 4,
                                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, instanceAuxInfo);
            }
        }
    }
}

void OutlineRenderer::EnsureOutlineBufferCapacity(uint32_t frameIndex, size_t instanceCount, size_t boneMatrixCount)
{
    if (!m_core || frameIndex >= m_outlineSkinInstanceBufs.size() || frameIndex >= m_outlineSkinPaletteBufs.size())
        return;

    VkDevice device = m_core->GetDevice();
    VmaAllocator allocator = m_core->GetDeviceContext().GetVmaAllocator();
    if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE)
        return;

    auto createStorageBuffer = [&](OutlineStorageBuffer &buf, size_t elementCount, size_t elementSize) {
        const VkDeviceSize byteSize = static_cast<VkDeviceSize>(std::max<size_t>(1, elementCount) * elementSize);
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = byteSize;
        bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCreateInfo{};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        VmaAllocationInfo allocInfo{};
        if (vmaCreateBuffer(allocator, &bufInfo, &allocCreateInfo, &buf.buffer, &buf.allocation, &allocInfo) ==
            VK_SUCCESS) {
            buf.mapped = allocInfo.pMappedData;
            buf.capacity = elementCount;
        }
    };

    auto ensureCapacity = [&](OutlineStorageBuffer &buffer, size_t count, size_t stride) {
        const size_t required = std::max<size_t>(1, count);
        if (buffer.buffer != VK_NULL_HANDLE && buffer.capacity >= required)
            return;
        if (buffer.buffer != VK_NULL_HANDLE) {
            const VkBuffer retiredBuffer = buffer.buffer;
            const VmaAllocation retiredAllocation = buffer.allocation;
            m_core->RetireGpuResource([allocator, retiredBuffer, retiredAllocation] {
                vmaDestroyBuffer(allocator, retiredBuffer, retiredAllocation);
            });
        }
        buffer = OutlineStorageBuffer{};
        createStorageBuffer(buffer, std::max<size_t>(required, 64), stride);
    };
    ensureCapacity(m_outlineInstanceBufs[frameIndex], instanceCount, sizeof(glm::mat4));
    ensureCapacity(m_outlineInstanceAuxBufs[frameIndex], instanceCount, sizeof(GPUInstanceAuxData));
    ensureCapacity(m_outlineSkinInstanceBufs[frameIndex], instanceCount, sizeof(GPUSkinInstanceData));
    ensureCapacity(m_outlineSkinPaletteBufs[frameIndex], boneMatrixCount, sizeof(glm::mat4));
    auto &skinInstance = m_outlineSkinInstanceBufs[frameIndex];
    auto &skinPalette = m_outlineSkinPaletteBufs[frameIndex];

    if (frameIndex < m_outlineGlobalsDescSets.size() && m_outlineGlobalsDescSets[frameIndex] != VK_NULL_HANDLE) {
        for (const auto &[binding, buffer] : std::array<std::pair<uint32_t, VkBuffer>, 2>{
                 {{1, m_outlineInstanceBufs[frameIndex].buffer}, {4, m_outlineInstanceAuxBufs[frameIndex].buffer}}}) {
            VkDescriptorBufferInfo info{buffer, 0, VK_WHOLE_SIZE};
            vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[frameIndex], binding,
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, info);
        }
        if (skinInstance.buffer != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo skinInstInfo{};
            skinInstInfo.buffer = skinInstance.buffer;
            skinInstInfo.offset = 0;
            skinInstInfo.range = VK_WHOLE_SIZE;
            vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[frameIndex], 2,
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, skinInstInfo);
        }

        if (skinPalette.buffer != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo skinPaletteInfo{};
            skinPaletteInfo.buffer = skinPalette.buffer;
            skinPaletteInfo.offset = 0;
            skinPaletteInfo.range = VK_WHOLE_SIZE;
            vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[frameIndex], 3,
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, skinPaletteInfo);
        }
    }
}

VkPipeline OutlineRenderer::CreateMaskPipeline(const VkPipelineShaderStageCreateInfo stages[2], VkPipelineLayout layout,
                                               const ShaderReflection &vertexReflection)
{
    MeshVertexInputState vertexInput(vertexReflection);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = MakeTriangleListInputAssembly();
    DynamicViewportState viewportState;
    VkPipelineRasterizationStateCreateInfo raster = MakeRasterizationState(VK_CULL_MODE_NONE);
    VkPipelineDepthStencilStateCreateInfo depthStencil = MakeDepthStencilState(VK_FALSE, VK_FALSE);
    VkPipelineMultisampleStateCreateInfo multisampling = MakeMultisampleState(VK_SAMPLE_COUNT_1_BIT);
    VkPipelineColorBlendAttachmentState colorBlendAttach = MakeOpaqueColorBlendAttachment();
    VkPipelineColorBlendStateCreateInfo colorBlend = MakeColorBlendState(colorBlendAttach);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput.createInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState.viewportState;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &viewportState.dynamicState;
    pipelineInfo.layout = layout;
    std::array<VkFormat, rhi::GraphicsRenderingSignature::MaxColorTargets> colorFormats{};
    VkPipelineRenderingCreateInfo renderingInfo{};
    if (!rhi::BuildVkPipelineRenderingInfo(m_outlineMaskRenderingSignature, colorFormats, renderingInfo))
        return VK_NULL_HANDLE;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    pipelineInfo.subpass = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(m_core->GetDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    return pipeline;
}

VkPipeline OutlineRenderer::GetOrCreateMtlOutlinePipeline(InxMaterial *material)
{
    const ShaderProgram *program = material->GetPassShaderProgram(ShaderCompileTarget::Forward);
    if (!program)
        return VK_NULL_HANDLE;
    const std::string programId = program->GetShaderId();
    std::string key = material->GetMaterialKey();
    if (key.empty())
        key = material->GetName();

    auto it = m_perMtlOutlinePipelines.find(key);
    if (it != m_perMtlOutlinePipelines.end()) {
        if (m_perMtlOutlineProgramIds.at(key) == programId)
            return it->second;
        const VkDevice device = m_core->GetDevice();
        const VkPipeline retired = it->second;
        m_core->RetireGpuResource([device, retired] { vkDestroyPipeline(device, retired, nullptr); });
        m_perMtlOutlinePipelines.erase(it);
        m_perMtlOutlineProgramIds.erase(key);
    }

    VkShaderModule vertModule = program->GetVertexModule();
    VkShaderModule fragModule = m_core->GetShaderModule("Outline Mask", "fragment");
    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;

    VkDevice device = m_core->GetDevice();

    std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
        MakeShaderStageInfo(VK_SHADER_STAGE_VERTEX_BIT, vertModule),
        MakeShaderStageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragModule),
    };

    VkPipeline pipeline =
        CreateMaskPipeline(stages.data(), program->GetPipelineLayout(), program->GetVertexReflection());
    if (pipeline == VK_NULL_HANDLE) {
        INXLOG_WARN("OutlineRenderer: Failed to create per-material outline pipeline for '", material->GetName(), "'");
        return VK_NULL_HANDLE;
    }

    m_perMtlOutlinePipelines[key] = pipeline;
    m_perMtlOutlineProgramIds[key] = programId;
    return pipeline;
}

// ============================================================================
// Internal: Mask Pass
// ============================================================================

void OutlineRenderer::RecordMaskDraws(VkCommandBuffer cmdBuf, const std::vector<DrawCall> &drawCalls,
                                      rhi::BindGroupHandle perViewGroup)
{
    if (!m_resourcesReady || !HasActiveOutline() || cmdBuf == VK_NULL_HANDLE ||
        m_outlineMaskPipeline == VK_NULL_HANDLE || m_outlineInstanceBufs.empty()) {
        return;
    }
    const VkDescriptorSet perViewDescSet = m_core->GetDeviceContext().GetRhiDevice().Resolve(perViewGroup);
    if (perViewDescSet == VK_NULL_HANDLE) {
        INXLOG_ERROR("OutlineRenderer::RecordMaskDraws received an invalid per-view bind group");
        return;
    }

    uint32_t frameIdx = m_core->GetCurrentFrameSlot() % static_cast<uint32_t>(m_outlineInstanceBufs.size());
    size_t selectedDrawCount = 0, selectedBoneCount = 0;
    for (const auto &dc : drawCalls) {
        if (!IsOutlinedObject(dc.objectId))
            continue;
        ++selectedDrawCount;
        if (dc.skinBoneMatrices)
            selectedBoneCount += dc.skinBoneMatrices->size();
    }
    EnsureOutlineBufferCapacity(frameIdx, selectedDrawCount, selectedBoneCount);
    uint32_t instanceIndex = 0, boneOffset = 0;

    // Render the selected object
    for (const auto &dc : drawCalls) {
        if (!IsOutlinedObject(dc.objectId))
            continue;

        // Get per-object buffer via core accessor
        VkBuffer vertBuf = m_core->GetObjectVertexBuffer(dc.objectId);
        VkBuffer idxBuf = m_core->GetObjectIndexBuffer(dc.objectId);

        if (vertBuf == VK_NULL_HANDLE || idxBuf == VK_NULL_HANDLE)
            continue;

        VkBuffer vertBuffers[] = {vertBuf};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cmdBuf, 0, 1, vertBuffers, offsets);
        const MeshIndexFormat indexFormat = m_core->GetObjectIndexFormat(dc.objectId);
        vkCmdBindIndexBuffer(cmdBuf, idxBuf, 0,
                             indexFormat == MeshIndexFormat::UInt16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);

        // Push per-object model matrix + normal matrix
        struct PushConstants
        {
            glm::mat4 model;
            glm::mat4 normalMat;
        };

        PushConstants pushData;
        pushData.model = dc.worldMatrix;
        glm::mat3 normalMat3 = glm::transpose(glm::inverse(glm::mat3(dc.worldMatrix)));
        pushData.normalMat = glm::mat4(normalMat3);

        // Every published material vertex stage defines its silhouette. Deformation
        // may use time, textures or constants without a vertex material UBO.
        bool usePerMaterialPipeline = false;
        if (dc.material) {
            const ShaderProgram *fwdProgram = dc.material->GetPassShaderProgram(ShaderCompileTarget::Forward);
            if (fwdProgram) {
                VkPipeline mtlPipeline = GetOrCreateMtlOutlinePipeline(dc.material.get());
                MaterialRenderData *renderData =
                    m_core->GetMaterialPipelineManager().GetRenderData(dc.material->GetMaterialKey());
                VkDescriptorSet mtlDescSet =
                    renderData && renderData->isValid ? renderData->descriptorSet : VK_NULL_HANDLE;
                if (dc.parameterBlock) {
                    const auto *descriptor =
                        m_core->GetMaterialPipelineManager().GetDescriptorManager().GetOrCreateRendererDescriptorSet(
                            *dc.material, *fwdProgram, dc.parameterBlock);
                    mtlDescSet = descriptor && descriptor->isValid ? descriptor->descriptorSet : VK_NULL_HANDLE;
                }
                const VkPipelineLayout layout = fwdProgram->GetPipelineLayout();

                if (mtlPipeline != VK_NULL_HANDLE && mtlDescSet != VK_NULL_HANDLE) {
                    // Write the selected object's transform and skin palette in distinct storage for each recorded
                    // draw.
                    std::memcpy(static_cast<glm::mat4 *>(m_outlineInstanceBufs[frameIdx].mapped) + instanceIndex,
                                &dc.worldMatrix, sizeof(glm::mat4));
                    if (frameIdx < m_outlineInstanceAuxBufs.size() && m_outlineInstanceAuxBufs[frameIdx].mapped) {
                        GPUInstanceAuxData aux{};
                        aux.previousModel = dc.worldMatrix;
                        aux.objectId = PackGPUObjectId(dc.pickingObjectId != 0 ? dc.pickingObjectId : dc.objectId);
                        aux.layerMask = dc.layerMask;
                        std::memcpy(static_cast<GPUInstanceAuxData *>(m_outlineInstanceAuxBufs[frameIdx].mapped) +
                                        instanceIndex,
                                    &aux, sizeof(aux));
                    }
                    GPUSkinInstanceData skinData{};
                    if (dc.skinBoneMatrices && !dc.skinBoneMatrices->empty() &&
                        frameIdx < m_outlineSkinInstanceBufs.size() && frameIdx < m_outlineSkinPaletteBufs.size()) {
                        auto &skinInstance = m_outlineSkinInstanceBufs[frameIdx];
                        auto &skinPalette = m_outlineSkinPaletteBufs[frameIdx];
                        if (skinInstance.mapped && skinPalette.mapped &&
                            skinPalette.capacity >= boneOffset + dc.skinBoneMatrices->size()) {
                            skinData.boneOffset = boneOffset;
                            skinData.boneCount = static_cast<uint32_t>(dc.skinBoneMatrices->size());
                            skinData.flags = kGPUSkinFlagEnabled;
                            std::memcpy(static_cast<glm::mat4 *>(skinPalette.mapped) + boneOffset,
                                        dc.skinBoneMatrices->data(), dc.skinBoneMatrices->size() * sizeof(glm::mat4));
                        }
                    }
                    if (frameIdx < m_outlineSkinInstanceBufs.size() && m_outlineSkinInstanceBufs[frameIdx].mapped)
                        std::memcpy(static_cast<GPUSkinInstanceData *>(m_outlineSkinInstanceBufs[frameIdx].mapped) +
                                        instanceIndex,
                                    &skinData, sizeof(GPUSkinInstanceData));

                    // Bind per-material pipeline
                    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, mtlPipeline);

                    // Set 0: the published material resources, including vertex properties and textures
                    vkdebug::CmdBindDescriptorSetsTracked("OutlineRenderer.RenderOutlineMask.Set0", cmdBuf,
                                                          VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &mtlDescSet, 0,
                                                          nullptr);

                    vkdebug::CmdBindDescriptorSetsTracked("OutlineRenderer.RenderOutlineMask.Set1", cmdBuf,
                                                          VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1,
                                                          &perViewDescSet, 0, nullptr);

                    // Set 2: outline globals (globals UBO + outline instance buffer)
                    VkDescriptorSet globalsDescSet = m_outlineGlobalsDescSets[frameIdx];
                    vkdebug::CmdBindDescriptorSetsTracked("OutlineRenderer.RenderOutlineMask.Set2", cmdBuf,
                                                          VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 2, 1,
                                                          &globalsDescSet, 0, nullptr);

                    if (fwdProgram->UsesBindlessTextureABI()) {
                        auto &device = m_core->GetDeviceContext().GetRhiDevice();
                        const VkDescriptorSet textures = device.Resolve(device.GetBindlessTextureTableBinding().group);
                        vkdebug::CmdBindDescriptorSetsTracked("OutlineRenderer.RenderOutlineMask.Set3", cmdBuf,
                                                              VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 3, 1, &textures,
                                                              0, nullptr);
                        const auto *indices =
                            m_core->GetMaterialPipelineManager().GetDescriptorManager().GetBindlessTextureIndices(
                                mtlDescSet);
                        device.MarkBindlessTexturesUsed(indices && !indices->empty() ? indices->data() : nullptr,
                                                        indices ? indices->size() : 0);
                    }

                    vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pushData);

                    vkCmdDrawIndexed(cmdBuf, dc.indexCount, 1, dc.indexStart, dc.vertexStart, instanceIndex++);
                    boneOffset += skinData.boneCount;
                    usePerMaterialPipeline = true;
                }
            }
        }

        // Geometry without a published material uses the built-in vertex stage.
        if (!usePerMaterialPipeline && !dc.material) {
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_outlineMaskPipeline);
            vkdebug::CmdBindDescriptorSetsTracked("OutlineRenderer.RenderOutlineMask.FallbackSet1", cmdBuf,
                                                  VK_PIPELINE_BIND_POINT_GRAPHICS, m_outlineMaskPipelineLayout, 1, 1,
                                                  &perViewDescSet, 0, nullptr);
            vkCmdPushConstants(cmdBuf, m_outlineMaskPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(PushConstants), &pushData);
            vkCmdDrawIndexed(cmdBuf, dc.indexCount, 1, dc.indexStart, dc.vertexStart, 0);
        }
    }
}

// ============================================================================
// Internal: Composite Pass
// ============================================================================

void OutlineRenderer::RecordCompositeDraw(VkCommandBuffer cmdBuf)
{
    if (!m_resourcesReady || !HasActiveOutline() || cmdBuf == VK_NULL_HANDLE ||
        m_outlineCompositePipeline == VK_NULL_HANDLE || m_outlineCompositeDescSet == VK_NULL_HANDLE) {
        return;
    }

    uint32_t w = m_sceneRenderTarget->GetWidth();
    uint32_t h = m_sceneRenderTarget->GetHeight();

    // Bind composite pipeline
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_outlineCompositePipeline);

    // Bind composite descriptor set (mask texture sampler)
    vkdebug::CmdBindDescriptorSetsTracked("OutlineRenderer.RenderOutlineComposite.Set0", cmdBuf,
                                          VK_PIPELINE_BIND_POINT_GRAPHICS, m_outlineCompositePipelineLayout, 0, 1,
                                          &m_outlineCompositeDescSet, 0, nullptr);

    // Push constants: outline color, texel size, outline width
    struct CompositePushConstants
    {
        glm::vec4 outlineColor;
        glm::vec2 texelSize;
        float outlineWidth;
        float _padding;
    };

    CompositePushConstants pushData;
    // Authored sRGB -> linear: the composite writes into the linear scene
    // buffer, which is sRGB-encoded later by the display encode pass.
    pushData.outlineColor = inx::color::SrgbToLinear(m_outlineColor);
    pushData.texelSize = glm::vec2(1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h));
    pushData.outlineWidth = m_outlinePixelWidth;
    pushData._padding = 0.0f;

    vkCmdPushConstants(cmdBuf, m_outlineCompositePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(CompositePushConstants), &pushData);

    // Draw fullscreen triangle (3 vertices, no vertex buffer)
    vkCmdDraw(cmdBuf, 3, 1, 0, 0);
}

} // namespace infernux
