/**
 * @file FullscreenRenderer.cpp
 * @brief Implementation of the FullscreenRenderer utility
 */

#include "FullscreenRenderer.h"
#include "InxVkCoreModular.h"
#include "shader/ShaderProgram.h"
#include "vk/DescriptorBindTrace.h"
#include "vk/VkDeviceContext.h"
#include "vk/VkPipelineHelpers.h"
#include "vk/VkPipelineManager.h"
#include "vk/VkSwapchainManager.h"
#include <algorithm>
#include <core/config/EngineConfig.h>
#include <core/error/InxError.h>

namespace infernux
{

namespace
{

void DestroyPipelineEntry(VkDevice device, FullscreenPipelineEntry &entry)
{
    if (entry.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, entry.pipeline, nullptr);
        entry.pipeline = VK_NULL_HANDLE;
    }
    if (entry.layoutOwned && entry.layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, entry.layout, nullptr);
        entry.layout = VK_NULL_HANDLE;
        entry.layoutOwned = false;
    }
    if (entry.descSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, entry.descSetLayout, nullptr);
        entry.descSetLayout = VK_NULL_HANDLE;
    }
    if (entry.emptyGapLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, entry.emptyGapLayout, nullptr);
        entry.emptyGapLayout = VK_NULL_HANDLE;
    }
}

VkDescriptorSetLayout CreateDescriptorSetLayout(VkDevice device,
                                                const std::vector<VkDescriptorSetLayoutBinding> &bindings)
{
    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutCI.pBindings = bindings.empty() ? nullptr : bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &layout) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return layout;
}

bool CreatePipelineLayout(VkDevice device, VkDescriptorSetLayout textureLayout, VkDescriptorSetLayout globalsLayout,
                          FullscreenPipelineEntry &entry)
{
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(FullscreenPushConstants);

    std::vector<VkDescriptorSetLayout> setLayouts = {textureLayout};
    if (globalsLayout != VK_NULL_HANDLE) {
        entry.emptyGapLayout = CreateDescriptorSetLayout(device, {});
        if (entry.emptyGapLayout == VK_NULL_HANDLE) {
            return false;
        }
        setLayouts.push_back(entry.emptyGapLayout);
        setLayouts.push_back(globalsLayout);
    }

    VkPipelineLayoutCreateInfo pipeLayoutCI{};
    pipeLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutCI.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipeLayoutCI.pSetLayouts = setLayouts.data();
    pipeLayoutCI.pushConstantRangeCount = 1;
    pipeLayoutCI.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &pipeLayoutCI, nullptr, &entry.layout) != VK_SUCCESS) {
        return false;
    }
    entry.layoutOwned = true;
    return true;
}

void AppendImageWrite(std::vector<VkDescriptorImageInfo> &imageInfos, std::vector<VkWriteDescriptorSet> &writes,
                      VkDescriptorSet descSet, uint32_t binding, VkImageView imageView, VkSampler sampler,
                      VkImageLayout imageLayout)
{
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = sampler;
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = imageLayout;
    imageInfos.push_back(imageInfo);

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descSet;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfos.back();
    writes.push_back(write);
}

VkSamplerCreateInfo MakeSamplerCreateInfo(VkFilter filter, VkSamplerMipmapMode mipmapMode)
{
    VkSamplerCreateInfo samplerCI{};
    samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCI.magFilter = filter;
    samplerCI.minFilter = filter;
    samplerCI.mipmapMode = mipmapMode;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.mipLodBias = 0.0f;
    samplerCI.maxAnisotropy = 1.0f;
    samplerCI.minLod = 0.0f;
    samplerCI.maxLod = 0.0f;
    return samplerCI;
}

bool CreateSampler(VkDevice device, VkFilter filter, VkSamplerMipmapMode mipmapMode, VkSampler &sampler,
                   const char *label)
{
    VkSamplerCreateInfo samplerCI = MakeSamplerCreateInfo(filter, mipmapMode);
    if (vkCreateSampler(device, &samplerCI, nullptr, &sampler) != VK_SUCCESS) {
        INXLOG_ERROR("FullscreenRenderer: Failed to create ", label, " sampler");
        sampler = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

} // namespace

// ============================================================================
// Lifecycle
// ============================================================================

FullscreenRenderer::~FullscreenRenderer()
{
    Destroy();
}

void FullscreenRenderer::Initialize(InxVkCoreModular *vkCore)
{
    if (!vkCore) {
        INXLOG_ERROR("FullscreenRenderer::Initialize: null vkCore");
        return;
    }
    m_vkCore = vkCore;
    m_device = vkCore->GetDevice();

    CreateLinearSampler();
    CreateNearestSampler();
    CreateDescriptorPools();
}

void FullscreenRenderer::Destroy()
{
    if (m_device == VK_NULL_HANDLE)
        return;

    vkDeviceWaitIdle(m_device);

    for (auto &[key, entry] : m_pipelineCache) {
        DestroyPipelineEntry(m_device, entry);
    }
    m_pipelineCache.clear();

    for (VkDescriptorPool pool : m_descriptorPools) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, pool, nullptr);
        }
    }
    m_descriptorPools.clear();

    if (m_linearSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_linearSampler, nullptr);
        m_linearSampler = VK_NULL_HANDLE;
    }

    if (m_nearestSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_nearestSampler, nullptr);
        m_nearestSampler = VK_NULL_HANDLE;
    }

    m_device = VK_NULL_HANDLE;
    m_vkCore = nullptr;
}

// ============================================================================
// Pipeline management
// ============================================================================

const FullscreenPipelineEntry &FullscreenRenderer::EnsurePipeline(const FullscreenPipelineKey &key)
{
    auto it = m_pipelineCache.find(key);
    if (it != m_pipelineCache.end()) {
        return it->second;
    }

    auto entry = CreatePipeline(key);
    auto [insertIt, inserted] = m_pipelineCache.emplace(key, entry);
    return insertIt->second;
}

FullscreenPipelineEntry FullscreenRenderer::CreatePipeline(const FullscreenPipelineKey &key)
{
    FullscreenPipelineEntry entry{};

    if (!m_vkCore) {
        INXLOG_ERROR("FullscreenRenderer::CreatePipeline: not initialized");
        return entry;
    }

    // ------------------------------------------------------------------
    // 1. Descriptor set layout: N combined image samplers (fragment)
    // ------------------------------------------------------------------
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(key.inputTextureCount);
    for (uint32_t i = 0; i < key.inputTextureCount; ++i) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = i;
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b.pImmutableSamplers = nullptr;
        bindings.push_back(b);
    }

    entry.descSetLayout = CreateDescriptorSetLayout(m_device, bindings);
    if (entry.descSetLayout == VK_NULL_HANDLE) {
        INXLOG_ERROR("FullscreenRenderer: Failed to create descriptor set layout for '", key.shaderName, "'");
        return entry;
    }

    // ------------------------------------------------------------------
    // 2. Pipeline layout: desc sets + push constants (128 bytes, fragment)
    //    set 0 = textures, set 1 = empty (per-view gap), set 2 = globals
    // ------------------------------------------------------------------
    VkDescriptorSetLayout globalsLayout = ShaderProgram::GetGlobalsDescSetLayout();

    if (!CreatePipelineLayout(m_device, entry.descSetLayout, globalsLayout, entry)) {
        INXLOG_ERROR("FullscreenRenderer: Failed to create pipeline layout for '", key.shaderName, "'");
        DestroyPipelineEntry(m_device, entry);
        return entry;
    }

    // ------------------------------------------------------------------
    // 3. Shader modules
    // ------------------------------------------------------------------
    // Try effect-specific vertex shader first; fall back to the shared
    // fullscreen_triangle vertex shader if not found.
    VkShaderModule vertModule = m_vkCore->GetShaderModule(key.shaderName, "vertex");
    if (vertModule == VK_NULL_HANDLE) {
        vertModule = m_vkCore->GetShaderModule("fullscreen_triangle", "vertex");
    }
    VkShaderModule fragModule = m_vkCore->GetShaderModule(key.shaderName, "fragment");

    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        INXLOG_ERROR("FullscreenRenderer: Missing shader modules for '", key.shaderName,
                     "' (vert=", (vertModule != VK_NULL_HANDLE ? "OK" : "MISSING"),
                     ", frag=", (fragModule != VK_NULL_HANDLE ? "OK" : "MISSING"), ")");
        DestroyPipelineEntry(m_device, entry);
        return entry;
    }

    // ------------------------------------------------------------------
    // 4. Graphics pipeline (no vertex input, no depth, no cull)
    // ------------------------------------------------------------------
    // Shader stages
    auto shaderStages = vkrender::MakeVertFragStages(vertModule, fragModule);

    // Empty vertex input (procedural fullscreen triangle)
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto inputAssembly = vkrender::MakeTriangleListInputAssembly();

    // Dynamic viewport + scissor
    vkrender::DynamicViewportScissorState dynState;

    // Rasterization: no cull, fill
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    // Multisample
    auto multisampling = vkrender::MakeMultisampleState(key.samples);

    // No depth
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;

    // Color blend: no blend (overwrite), single attachment
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    // Create pipeline
    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.stageCount = 2;
    pipelineCI.pStages = shaderStages.data();
    pipelineCI.pVertexInputState = &vertexInput;
    pipelineCI.pInputAssemblyState = &inputAssembly;
    pipelineCI.pViewportState = &dynState.viewportState;
    pipelineCI.pRasterizationState = &rasterizer;
    pipelineCI.pMultisampleState = &multisampling;
    pipelineCI.pDepthStencilState = &depthStencil;
    pipelineCI.pColorBlendState = &colorBlending;
    pipelineCI.pDynamicState = &dynState.dynamicState;
    pipelineCI.layout = entry.layout;
    pipelineCI.renderPass = key.renderPass;
    pipelineCI.subpass = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline) != VK_SUCCESS) {
        INXLOG_ERROR("FullscreenRenderer: Failed to create pipeline for '", key.shaderName, "'");
        DestroyPipelineEntry(m_device, entry);
        return entry;
    }
    entry.pipeline = pipeline;

    return entry;
}

// ============================================================================
// Per-frame pool reset
// ============================================================================

void FullscreenRenderer::ResetPool()
{
    VkDescriptorPool pool = GetCurrentDescriptorPool();
    if (pool != VK_NULL_HANDLE) {
        vkResetDescriptorPool(m_device, pool, 0);
    }
}

// ============================================================================
// Descriptor set allocation
// ============================================================================

VkDescriptorSet FullscreenRenderer::AllocateDescriptorSet(VkDescriptorSetLayout layout, const VkImageView *inputViews,
                                                          uint32_t inputViewCount, const bool *depthInputs,
                                                          VkSampler colorSampler)
{
    VkDescriptorPool pool = GetCurrentDescriptorPool();
    if (pool == VK_NULL_HANDLE || layout == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet descSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &descSet) != VK_SUCCESS) {
        INXLOG_ERROR("FullscreenRenderer: Failed to allocate descriptor set");
        return VK_NULL_HANDLE;
    }

    // Write image descriptors
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkWriteDescriptorSet> writes;
    imageInfos.reserve(inputViewCount);
    writes.reserve(inputViewCount);

    for (uint32_t i = 0; i < inputViewCount; ++i) {
        const bool isDepthInput = (depthInputs != nullptr) ? depthInputs[i] : false;
        AppendImageWrite(imageInfos, writes, descSet, i, inputViews[i], isDepthInput ? m_nearestSampler : colorSampler,
                         isDepthInput ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                      : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    return descSet;
}

// ============================================================================
// Draw
// ============================================================================

void FullscreenRenderer::Draw(VkCommandBuffer cmdBuf, const FullscreenPipelineEntry &entry, VkDescriptorSet descSet,
                              const FullscreenPushConstants &pushConstants, uint32_t pushConstantSize)
{
    if (entry.pipeline == VK_NULL_HANDLE)
        return;

    // Bind pipeline
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, entry.pipeline);

    // Bind descriptor set 0 (input textures)
    if (descSet != VK_NULL_HANDLE) {
        vkdebug::CmdBindDescriptorSetsTracked("FullscreenRenderer.Draw.Set0", cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                              entry.layout, 0, 1, &descSet, 0, nullptr);
    }

    // Bind descriptor set 2 (engine globals UBO) — provides _Globals to fullscreen shaders
    if (m_vkCore && entry.emptyGapLayout != VK_NULL_HANDLE) {
        VkDescriptorSet globalsDescSet = m_vkCore->GetCurrentGlobalsDescSet();
        if (globalsDescSet != VK_NULL_HANDLE) {
            vkdebug::CmdBindDescriptorSetsTracked("FullscreenRenderer.Draw.Set2", cmdBuf,
                                                  VK_PIPELINE_BIND_POINT_GRAPHICS, entry.layout, 2, 1, &globalsDescSet,
                                                  0, nullptr);
        }
    }

    // Push constants
    if (pushConstantSize > 0) {
        vkCmdPushConstants(cmdBuf, entry.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, pushConstantSize,
                           pushConstants.values);
    }

    // Draw fullscreen triangle (3 vertices, no vertex buffer)
    vkCmdDraw(cmdBuf, 3, 1, 0, 0);
}

// ============================================================================
// Private helpers
// ============================================================================

uint32_t FullscreenRenderer::GetCurrentFrameIndex() const
{
    if (!m_vkCore) {
        return 0;
    }
    return m_vkCore->GetSwapchain().GetCurrentFrame() %
           static_cast<uint32_t>(m_descriptorPools.empty() ? 1 : m_descriptorPools.size());
}

VkDescriptorPool FullscreenRenderer::GetCurrentDescriptorPool() const
{
    if (m_descriptorPools.empty()) {
        return VK_NULL_HANDLE;
    }
    return m_descriptorPools[GetCurrentFrameIndex()];
}

void FullscreenRenderer::CreateDescriptorPools()
{
    const EngineConfig &config = EngineConfig::Get();
    const uint32_t framesInFlight = std::max(1u, m_vkCore ? m_vkCore->GetMaxFramesInFlight() : 1u);
    const uint32_t samplerDescriptorsPerFrame = std::max(1u, config.fullscreenSamplerDescriptorsPerFrame);
    const uint32_t maxSetsPerFrame = std::max(1u, config.fullscreenDescriptorSetsPerFrame);

    m_descriptorPools.assign(framesInFlight, VK_NULL_HANDLE);

    for (size_t frameIndex = 0; frameIndex < m_descriptorPools.size(); ++frameIndex) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = samplerDescriptorsPerFrame;

        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolCI.maxSets = maxSetsPerFrame;
        poolCI.poolSizeCount = 1;
        poolCI.pPoolSizes = &poolSize;

        if (vkCreateDescriptorPool(m_device, &poolCI, nullptr, &m_descriptorPools[frameIndex]) != VK_SUCCESS) {
            INXLOG_ERROR("FullscreenRenderer: Failed to create descriptor pool for frame ", frameIndex);
            for (VkDescriptorPool pool : m_descriptorPools) {
                if (pool != VK_NULL_HANDLE) {
                    vkDestroyDescriptorPool(m_device, pool, nullptr);
                }
            }
            m_descriptorPools.clear();
            return;
        }
    }
}

void FullscreenRenderer::CreateLinearSampler()
{
    CreateSampler(m_device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, m_linearSampler, "linear");
}

void FullscreenRenderer::CreateNearestSampler()
{
    CreateSampler(m_device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, m_nearestSampler, "nearest");
}

} // namespace infernux
