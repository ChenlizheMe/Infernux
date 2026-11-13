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
#include "vk/VulkanRhiDevice.h"
#include <core/types/ColorSpace.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/ShaderAsset/SlangNativeModule.h>

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

VkPipelineColorBlendStateCreateInfo MakeColorBlendState(const VkPipelineColorBlendAttachmentState &attachment)
{
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &attachment;
    return colorBlend;
}

const ShaderTargetCode &OutlineMaskCode()
{
    static const auto code = CompileSlangNativeModule({"inx_outline_mask", "<infernux/editor/outline_mask>", R"slang(
import inx_camera;
struct ObjectTransform { column_major float4x4 model; };
INX_ROOT_CONSTANTS ConstantBuffer<ObjectTransform> object_transform;
[shader("vertex")]
float4 vertex_main(float3 position : POSITION0) : SV_Position
{
    float4 projected = mul(inx_view.projection, mul(inx_view.view, mul(object_transform.model, float4(position, 1.0))));
    projected.y = -projected.y;
    return projected;
}
[shader("fragment")]
float4 fragment_main() : SV_Target0
{
    return float4(1.0);
}
)slang"},
                                                      ShaderCodeTarget::SpirV);
    return code;
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

    m_compositeRenderer.Initialize(m_core);
    const auto program = m_compositeRenderer.AcquireProgram("infernux/editor/outline_composite");
    if (!program) {
        if (!m_missingShadersReported)
            INXLOG_WARN("OutlineRenderer: outline composite program is unavailable");
        m_missingShadersReported = true;
        return false;
    }
    m_missingShadersReported = false;
    m_compositeProgram = program;
    m_compositeParameters = FullscreenRenderer::PrepareParameters(
        *program,
        {{"outline_color", std::array<float, 4>{}}, {"texel_size", std::array<float, 2>{}}, {"outline_width", 3.0f}});
    auto &device = m_core->GetDeviceContext().GetRhiDevice();
    m_outlineMaskInput.view = device.RegisterTextureView(m_sceneRenderTarget->GetOutlineMaskImageView());
    m_outlineMaskInput.format = rhi::PixelFormat::RGBA8UNorm;
    m_outlineMaskInput.sampler = device.RegisterSampler(m_sceneRenderTarget->GetOutlineMaskSampler());
    CreateOutlineMaskProgram();
    CreateOutlinePipelineLayout();
    CreateOutlineMaterialResources();

    m_resourcesReady = true;
    return true;
}

void OutlineRenderer::Cleanup(bool waitForIdle)
{
    if (!m_core)
        return;

    VkDevice device = m_core ? m_core->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE)
        return;

    if (waitForIdle && !m_core->IsShuttingDown()) {
        m_core->GetDeviceContext().WaitIdle();
    }

    auto &descriptorManager = m_core->GetDeviceContext().GetRhiDevice().GetDescriptorManager();
    for (const auto &lease : m_outlineGlobalsDescLeases)
        descriptorManager.Retire(lease);
    m_outlineGlobalsDescLeases.clear();

    DestroyOutlinePipelines();
    vkrender::SafeDestroy(device, m_outlineMaskPipelineLayout);
    vkrender::SafeDestroy(device, m_outlineMaskDescSetLayout);

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

    m_compositeRenderer.Destroy();
    auto &rhiDevice = m_core->GetDeviceContext().GetRhiDevice();
    rhiDevice.Release(m_outlineMaskVertex);
    rhiDevice.Release(m_outlineMaskFragment);
    rhiDevice.Release(m_outlineMaskInput.view);
    rhiDevice.Release(m_outlineMaskInput.sampler);
    m_outlineMaskVertex = {};
    m_outlineMaskFragment = {};
    m_outlineMaskInput = {};
    m_compositePipeline = {};
    m_compositeProgram.reset();
    m_outlineMaskRenderingSignature = {};
    m_resourcesReady = false;
    m_core = nullptr;
    m_sceneRenderTarget = nullptr;
}

// ============================================================================
// Rendering
// ============================================================================

bool OutlineRenderer::EnsureGraphPipelines(const rhi::GraphicsRenderingSignature &maskSignature,
                                           const rhi::GraphicsRenderingSignature &compositeSignature)
{
    if (!m_resourcesReady || !maskSignature.IsValid() || !compositeSignature.IsValid())
        return false;
    const auto program = m_compositeRenderer.AcquireProgram("infernux/editor/outline_composite");
    if (program->code != m_compositeProgram->code) {
        m_compositeParameters =
            FullscreenRenderer::PrepareParameters(*program, {{"outline_color", std::array<float, 4>{}},
                                                             {"texel_size", std::array<float, 2>{}},
                                                             {"outline_width", 3.0f}});
        m_compositeProgram = program;
    }
    FullscreenPipelineKey key;
    key.shaderName = program->shaderId;
    key.program = program;
    key.colorFormat = compositeSignature.colorFormats[0];
    key.depthFormat = compositeSignature.depthFormat;
    key.samples = compositeSignature.samples;
    key.alphaBlend = true;
    key.colorWriteMask = 0x07; // Selection color does not change scene coverage.
    key.inputResourceCount = 1;
    key.useDynamicRendering = true;
    m_compositePipeline = m_compositeRenderer.EnsurePipeline(key);
    if (m_outlineMaskRenderingSignature == maskSignature && m_outlineMaskPipeline != VK_NULL_HANDLE)
        return true;
    DestroyOutlinePipelines();
    m_outlineMaskRenderingSignature = maskSignature;
    return CreateOutlinePipelines();
}

// ============================================================================
// Internal: Vulkan Resource Creation
// ============================================================================

void OutlineRenderer::CreateOutlineMaskProgram()
{
    auto &device = m_core->GetDeviceContext().GetRhiDevice();
    vkrender::CreateDescriptorSetLayout(m_core->GetDevice(), nullptr, 0, m_outlineMaskDescSetLayout);
    for (const auto &entry : OutlineMaskCode().entries) {
        const auto module =
            device.CreateShaderModule({rhi::ShaderSourceLanguage::SpirV, entry.code.data(), entry.code.size()});
        if (entry.stage == rhi::ShaderStage::Vertex)
            m_outlineMaskVertex = module;
        else
            m_outlineMaskFragment = module;
    }
}

void OutlineRenderer::CreateOutlinePipelineLayout()
{
    const VkPushConstantRange pushRange{VK_SHADER_STAGE_VERTEX_BIT, 0, 128};
    const VkDescriptorSetLayout sets[] = {m_outlineMaskDescSetLayout, m_core->GetPerViewDescSetLayout()};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 2;
    layout.pSetLayouts = sets;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(m_core->GetDevice(), &layout, nullptr, &m_outlineMaskPipelineLayout);
}

bool OutlineRenderer::CreateOutlinePipelines()
{
    auto &device = m_core->GetDeviceContext().GetRhiDevice();
    const std::array stages = {
        MakeShaderStageInfo(VK_SHADER_STAGE_VERTEX_BIT, device.Resolve(m_outlineMaskVertex), "vertex_main"),
        MakeShaderStageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, device.Resolve(m_outlineMaskFragment), "fragment_main"),
    };
    ShaderReflection vertexReflection;
    vertexReflection.LoadTargetLayout(OutlineMaskCode(), "vertex_main");
    m_outlineMaskPipeline = CreateMaskPipeline(stages.data(), m_outlineMaskPipelineLayout, vertexReflection);
    return m_outlineMaskPipeline != VK_NULL_HANDLE;
}

void OutlineRenderer::DestroyOutlinePipelines()
{
    if (!m_core)
        return;
    VkDevice device = m_core->GetDevice();
    std::vector<VkPipeline> pipelines;
    if (m_outlineMaskPipeline != VK_NULL_HANDLE)
        pipelines.push_back(m_outlineMaskPipeline);
    for (const auto &[key, entry] : m_perMtlOutlinePipelines) {
        (void)key;
        pipelines.push_back(entry.pipeline);
    }
    m_outlineMaskPipeline = VK_NULL_HANDLE;
    m_perMtlOutlinePipelines.clear();

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

            // Binding 1: transforms for selected draws
            VkDescriptorBufferInfo ssboBufInfo{};
            ssboBufInfo.buffer = m_outlineInstanceBufs[i].buffer;
            ssboBufInfo.offset = 0;
            ssboBufInfo.range = VK_WHOLE_SIZE;
            vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[i], 1,
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ssboBufInfo);
            for (uint32_t binding : {5u, 6u})
                vkrender::UpdateDescriptorSetWithBuffer(device, m_outlineGlobalsDescSets[i], binding,
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
        for (const auto &[binding, buffer] :
             std::array<std::pair<uint32_t, VkBuffer>, 4>{{{1, m_outlineInstanceBufs[frameIndex].buffer},
                                                           {4, m_outlineInstanceAuxBufs[frameIndex].buffer},
                                                           {5, m_outlineInstanceBufs[frameIndex].buffer},
                                                           {6, m_outlineInstanceBufs[frameIndex].buffer}}}) {
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
        if (it->second.programId == programId)
            return it->second.pipeline;
        const VkDevice device = m_core->GetDevice();
        const VkPipeline retired = it->second.pipeline;
        m_core->RetireGpuResource([device, retired] { vkDestroyPipeline(device, retired, nullptr); });
        m_perMtlOutlinePipelines.erase(it);
    }

    VkShaderModule vertModule = program->GetVertexModule();
    VkShaderModule fragModule = m_core->GetDeviceContext().GetRhiDevice().Resolve(m_outlineMaskFragment);
    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;

    VkDevice device = m_core->GetDevice();

    std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
        MakeShaderStageInfo(VK_SHADER_STAGE_VERTEX_BIT, vertModule, program->GetVertexEntryPoint().c_str()),
        MakeShaderStageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "fragment_main"),
    };

    VkPipeline pipeline =
        CreateMaskPipeline(stages.data(), program->GetPipelineLayout(), program->GetVertexReflection());
    if (pipeline == VK_NULL_HANDLE) {
        INXLOG_WARN("OutlineRenderer: Failed to create per-material outline pipeline for '", material->GetName(), "'");
        return VK_NULL_HANDLE;
    }

    m_perMtlOutlinePipelines[key] = {programId, pipeline};
    return pipeline;
}

// ============================================================================
// Internal: Mask Pass
// ============================================================================

void OutlineRenderer::RecordMaskDraws(VkCommandBuffer cmdBuf, const std::vector<DrawCall> &drawCalls,
                                      rhi::BindGroupHandle perViewGroup, VkBuffer viewGlobalsBuffer)
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
    const VkDescriptorBufferInfo globals{viewGlobalsBuffer, 0, sizeof(EngineGlobalsUBO)};
    vkrender::UpdateDescriptorSetWithBuffer(m_core->GetDevice(), m_outlineGlobalsDescSets[frameIdx], 0,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, globals);
    uint32_t instanceIndex = 0, boneOffset = 0;

    // Render the selected object
    for (const auto &dc : drawCalls) {
        if (!IsOutlinedObject(dc.objectId))
            continue;

        if (!m_core->BindObjectGeometry(cmdBuf, dc.objectId))
            continue;

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
            MaterialRenderData *renderData = m_core->ResolveMeshMaterial(dc.material);
            const ShaderProgram *fwdProgram = dc.material->GetPassShaderProgram(ShaderCompileTarget::Forward);
            if (dc.devicePose.value && fwdProgram && !fwdProgram->UsesDevicePoseABI())
                continue;
            if (fwdProgram) {
                VkPipeline mtlPipeline = GetOrCreateMtlOutlinePipeline(dc.material.get());
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
                        ApplyDevicePose(aux, dc.devicePose);
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
                    VkDescriptorSet globalsDescSet =
                        m_core->ResolvePoseGlobalsBinding(m_outlineGlobalsDescSets[frameIdx], dc.devicePose);
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

                    if (!fwdProgram->GetVertexReflection().GetPushConstants().empty())
                        vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants),
                                           &pushData);

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
    if (!m_resourcesReady || !HasActiveOutline() || cmdBuf == VK_NULL_HANDLE)
        return;
    const auto color = inx::color::SrgbToLinear(m_outlineColor);
    const ShaderParameterValue values[] = {
        std::array<float, 4>{color.r, color.g, color.b, color.a},
        std::array<float, 2>{1.0f / m_sceneRenderTarget->GetWidth(), 1.0f / m_sceneRenderTarget->GetHeight()},
        m_outlinePixelWidth,
    };
    m_compositeRenderer.ResetPool();
    auto &device = m_core->GetDeviceContext().GetRhiDevice();
    vk::VulkanGraphicsCommandContext context;
    auto encoder = device.MakeGraphicsCommandEncoder(context, cmdBuf);
    m_compositeRenderer.DrawProgram(encoder, m_compositePipeline, &m_outlineMaskInput, m_compositeParameters, values);
}

} // namespace infernux
