/**
 * @file InxScreenUIRenderer.cpp
 * @brief Implementation of GPU-based 2D screen-space UI renderer
 *
 * Uses ImGui's ImDrawList for command accumulation (provides text rendering
 * via font atlas for free) and renders with a standalone Vulkan pipeline
 * inside the scene render graph's MSAA render passes.
 *
 * The pipeline is a direct replica of ImGui's internal 2D pipeline:
 *   - Vertex format: ImDrawVert (pos, uv, col)
 *   - Push constants: vec2 scale + vec2 translate (orthographic projection)
 *   - Descriptor: single combined_image_sampler (font atlas)
 *   - Alpha blending, no depth test, no cull
 *
 * Screen Overlay encodes linear sampled colors; Camera and World UI stay
 * linear until the graph's display-encoding
 * pass.
 */

#include "InxScreenUIRenderer.h"
#include "InxTextLayout.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <core/log/InxLog.h>
#include <cstring>
#include <function/renderer/vk/DescriptorBindTrace.h>
#include <function/renderer/vk/RhiVulkanTypes.h>
#include <function/renderer/vk/VkPipelineHelpers.h>
#include <function/renderer/vk/VkRenderUtils.h>
#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <function/scene/GameObject.h>
#include <function/scene/WorldUIProjection.h>
#include <glm/gtc/type_ptr.hpp>
#include <imgui_internal.h> // for ImGui::GetDrawListSharedData()
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace infernux
{

namespace
{
constexpr float kTransformRotationEpsilon = 0.001f;
constexpr float kPi = 3.14159265358979f;
constexpr const char *kShaderEntryPoint = "main";
constexpr uint32_t kFontTextureBinding = 0;
constexpr float kWorldUILogicalPixelsPerUnit = 100.0f;

constexpr const char *kScreenVertexShader = R"glsl(
#version 450
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
layout(push_constant) uniform ScreenUIConstants {
    vec2 scale;
    vec2 translate;
    float encodeSample;
    float _materialPadding0;
    float _materialPadding1;
    float _materialPadding2;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 _tailPadding;
} pc;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;
void main() {
    outColor = aColor;
    outUV = aUV;
    gl_Position = vec4(aPosition * pc.scale + pc.translate, 0, 1);
}
)glsl";

constexpr const char *kScreenFragmentShader = R"glsl(
#version 450
layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inUV;
layout(set = 0, binding = 0) uniform sampler2D uiTexture;
layout(push_constant) uniform ScreenUIConstants {
    vec2 scale;
    vec2 translate;
    float encodeSample;
    float _materialPadding0;
    float _materialPadding1;
    float _materialPadding2;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 _tailPadding;
} pc;
layout(location = 0) out vec4 outColor;
void main() {
    vec4 sampleColor = texture(uiTexture, inUV);
    if (pc.encodeSample > 0.5) {
        vec3 rgb = max(sampleColor.rgb, vec3(0));
        sampleColor.rgb = mix(1.055 * pow(rgb, vec3(1.0 / 2.4)) - 0.055,
                             12.92 * rgb, lessThanEqual(rgb, vec3(0.0031308)));
    }
    outColor = inColor * sampleColor * pc.materialColor;
    if (pc.alphaClipEnabled > 0.5 && outColor.a < pc.alphaClipThreshold)
        discard;
}
)glsl";

void WorldElementBoundaryCallback(const ImDrawList *, const ImDrawCmd *)
{
}

constexpr const char *kWorldVertexShader = R"glsl(
#version 450
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
layout(location = 3) in vec2 aLocalPosition;
layout(location = 4) in vec3 aAnchor;
layout(location = 5) in vec2 aLocalOffset;
layout(location = 6) in float aPolicy;
layout(push_constant) uniform WorldUIConstants {
    mat4 viewProjection;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 screenScale;
    vec4 cameraRight;
    vec4 cameraUp;
} pc;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec2 outLocalPosition;
void main() {
    outColor = aColor;
    outUV = aUV;
    outLocalPosition = aLocalPosition;
    vec3 position = aPosition;
    int policy = int(aPolicy + 0.5);
    if (policy != 0) {
        vec3 offset = aPosition - aAnchor;
        if ((policy & 1) != 0)
            offset = pc.cameraRight.xyz * aLocalOffset.x + pc.cameraUp.xyz * aLocalOffset.y;
        if ((policy & 2) != 0) {
            float clipW = (pc.viewProjection * vec4(aAnchor, 1.0)).w;
            offset *= max(clipW, 0.0) * pc.screenScale.x * 100.0;
        }
        position = aAnchor + offset;
    }
    gl_Position = pc.viewProjection * vec4(position, 1.0);
}
)glsl";

constexpr const char *kWorldFragmentShader = R"glsl(
#version 450
layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inUV;
layout(set = 0, binding = 0) uniform sampler2D uiTexture;
layout(push_constant) uniform WorldUIConstants {
    mat4 viewProjection;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 screenScale;
    vec4 cameraRight;
    vec4 cameraUp;
} pc;
layout(location = 0) out vec4 outColor;
void main() {
    outColor = inColor * texture(uiTexture, inUV) * pc.materialColor;
    // World UI participates in the scene depth buffer. Fully transparent
    // glyph/image texels therefore must not publish depth for their quad.
    // Keep partially covered antialiased pixels; only empty coverage is cut.
    if ((pc.alphaClipEnabled > 0.5 && outColor.a < pc.alphaClipThreshold) || outColor.a <= 0.0)
        discard;
}
)glsl";

struct alignas(16) ScreenUIPushConstants
{
    std::array<float, 2> scale{};
    std::array<float, 2> translate{};
    float encodeSample = 0.0f;
    std::array<float, 3> materialPadding{};
    std::array<float, 4> materialColor{1.0f, 1.0f, 1.0f, 1.0f};
    float alphaClipThreshold = 0.0f;
    float alphaClipEnabled = 0.0f;
    std::array<float, 2> tailPadding{};
};

static_assert(offsetof(ScreenUIPushConstants, materialColor) == 32);
static_assert(sizeof(ScreenUIPushConstants) == 64);

struct alignas(16) WorldUIPushConstants
{
    glm::mat4 viewProjection{1.0f};
    glm::vec4 materialColor{1.0f};
    float alphaClipThreshold = 0.0f;
    float alphaClipEnabled = 0.0f;
    std::array<float, 2> screenScale{};
    glm::vec4 cameraRight{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 cameraUp{0.0f, 1.0f, 0.0f, 0.0f};
};

static_assert(sizeof(WorldUIPushConstants) == 128);

struct VertexTransform
{
    ImVec2 pivot{};
    float cosAngle = 1.0f;
    float sinAngle = 0.0f;
    bool mirrorH = false;
    bool mirrorV = false;
    bool enabled = false;
};

float ResolveFontSize(float fontSize)
{
    return textlayout::ResolveFontSize(fontSize);
}

float ExtractHDRScale(float &r, float &g, float &b)
{
    const float maxRGB = std::max(r, std::max(g, b));
    if (maxRGB <= 1.0f) {
        return 1.0f;
    }

    r /= maxRGB;
    g /= maxRGB;
    b /= maxRGB;
    return maxRGB;
}

ImTextureID ToImTextureID(uint64_t textureId)
{
    if constexpr (std::is_pointer_v<ImTextureID>) {
        return (ImTextureID)(static_cast<uintptr_t>(textureId));
    }
    return static_cast<ImTextureID>(textureId);
}

void ResetDrawListForFrame(ImDrawList &drawList, uint32_t width, uint32_t height, bool screenBounded = true)
{
    drawList._ResetForNewFrame();
    drawList.PushTextureID(ImGui::GetIO().Fonts->TexRef.GetTexID());
    if (screenBounded) {
        drawList.PushClipRect(ImVec2(0.0f, 0.0f), ImVec2(static_cast<float>(width), static_cast<float>(height)));
    } else {
        // World UI elements are ordinary scene geometry and do not inherit a
        // world Canvas, viewport, or parent range.
        constexpr float unbounded = 1.0e20f;
        drawList.PushClipRect(ImVec2(-unbounded, -unbounded), ImVec2(unbounded, unbounded));
    }
}

float NormalizeRotationDegrees(float rotation)
{
    rotation = std::fmod(rotation, 360.0f);
    if (rotation < 0.0f)
        rotation += 360.0f;
    return rotation;
}

VertexTransform MakeVertexTransform(float minX, float minY, float maxX, float maxY, float rotation, bool mirrorH,
                                    bool mirrorV)
{
    rotation = NormalizeRotationDegrees(rotation);

    VertexTransform transform{};
    transform.enabled = mirrorH || mirrorV || std::fabs(rotation) >= kTransformRotationEpsilon;
    if (!transform.enabled) {
        return transform;
    }

    const float radians = rotation * kPi / 180.0f;
    transform.pivot = ImVec2((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);
    transform.cosAngle = std::cos(radians);
    transform.sinAngle = std::sin(radians);
    transform.mirrorH = mirrorH;
    transform.mirrorV = mirrorV;
    return transform;
}

void ApplyVertexTransform(ImDrawList &drawList, int vertexStart, const VertexTransform &transform)
{
    if (!transform.enabled) {
        return;
    }

    for (int i = vertexStart; i < drawList.VtxBuffer.Size; ++i) {
        ImVec2 local(drawList.VtxBuffer[i].pos.x - transform.pivot.x, drawList.VtxBuffer[i].pos.y - transform.pivot.y);
        if (transform.mirrorH)
            local.x = -local.x;
        if (transform.mirrorV)
            local.y = -local.y;

        const float rx = local.x * transform.cosAngle - local.y * transform.sinAngle;
        const float ry = local.x * transform.sinAngle + local.y * transform.cosAngle;
        drawList.VtxBuffer[i].pos = ImVec2(transform.pivot.x + rx, transform.pivot.y + ry);
    }
}

bool RefreshFontDescriptorSet(VkDescriptorSet &descriptorSet)
{
    const ImTextureID texId = ImGui::GetIO().Fonts->TexRef.GetTexID();
    if (texId == 0) {
        return false;
    }

    descriptorSet = reinterpret_cast<VkDescriptorSet>(static_cast<uintptr_t>(texId));
    return true;
}

VkAttachmentDescription MakeColorAttachmentDescription(VkFormat format, VkSampleCountFlagBits samples,
                                                       VkImageLayout finalLayout)
{
    VkAttachmentDescription attachment{};
    attachment.format = format;
    attachment.samples = samples;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = finalLayout;
    return attachment;
}

VkAttachmentReference MakeColorAttachmentReference(uint32_t attachmentIndex = 0)
{
    VkAttachmentReference colorRef{};
    colorRef.attachment = attachmentIndex;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    return colorRef;
}

VkSubpassDescription MakeSingleColorSubpass(const VkAttachmentReference &colorRef)
{
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    return subpass;
}

VkViewport MakeViewport(uint32_t width, uint32_t height)
{
    VkViewport viewport{};
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    return viewport;
}

ScreenUIPushConstants MakeOrthoPushConstants(uint32_t width, uint32_t height)
{
    ScreenUIPushConstants constants{};
    constants.scale = {2.0f / static_cast<float>(width), 2.0f / static_cast<float>(height)};
    constants.translate = {-1.0f, -1.0f};
    return constants;
}

bool MakeClampedScissor(const ImDrawCmd &cmd, float frameWidth, float frameHeight, VkRect2D &outScissor)
{
    const float clipMinX = std::clamp(cmd.ClipRect.x, 0.0f, frameWidth);
    const float clipMinY = std::clamp(cmd.ClipRect.y, 0.0f, frameHeight);
    const float clipMaxX = std::clamp(cmd.ClipRect.z, 0.0f, frameWidth);
    const float clipMaxY = std::clamp(cmd.ClipRect.w, 0.0f, frameHeight);
    if (clipMaxX <= clipMinX || clipMaxY <= clipMinY) {
        return false;
    }

    outScissor.offset.x = static_cast<int32_t>(clipMinX);
    outScissor.offset.y = static_cast<int32_t>(clipMinY);
    outScissor.extent.width = static_cast<uint32_t>(clipMaxX - clipMinX);
    outScissor.extent.height = static_cast<uint32_t>(clipMaxY - clipMinY);
    return true;
}

bool CreateShaderModule(VkDevice device, const uint32_t *code, size_t codeSize, VkShaderModule &outModule)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = code;

    outModule = VK_NULL_HANDLE;
    return vkCreateShaderModule(device, &createInfo, nullptr, &outModule) == VK_SUCCESS;
}

bool CreatePipelineLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
                          const VkPushConstantRange &pushConstantRange, VkPipelineLayout &outLayout)
{
    VkPipelineLayoutCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    createInfo.setLayoutCount = 1;
    createInfo.pSetLayouts = &descriptorSetLayout;
    createInfo.pushConstantRangeCount = 1;
    createInfo.pPushConstantRanges = &pushConstantRange;

    outLayout = VK_NULL_HANDLE;
    return vkCreatePipelineLayout(device, &createInfo, nullptr, &outLayout) == VK_SUCCESS;
}

VkPushConstantRange MakeVertexPushConstantRange(uint32_t size)
{
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.size = size;
    return pushConstantRange;
}

using infernux::vkrender::MakeMultisampleState;
using infernux::vkrender::MakeShaderStageInfo;
using infernux::vkrender::MakeTriangleListInputAssembly;

VkPipelineVertexInputStateCreateInfo MakeVertexInputState(const VkVertexInputBindingDescription &bindingDesc,
                                                          const VkVertexInputAttributeDescription *attrDesc,
                                                          uint32_t attrCount)
{
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = attrCount;
    vertexInput.pVertexAttributeDescriptions = attrDesc;
    return vertexInput;
}

VkPipelineViewportStateCreateInfo MakeDynamicViewportState()
{
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    return viewportState;
}

VkPipelineRasterizationStateCreateInfo MakeRasterizationState()
{
    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterization.lineWidth = 1.0f;
    return rasterization;
}

VkPipelineColorBlendAttachmentState MakeAlphaBlendAttachment()
{
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.blendEnable = VK_TRUE;
    attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.colorBlendOp = VK_BLEND_OP_ADD;
    attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    return attachment;
}

VkPipelineColorBlendStateCreateInfo MakeColorBlendState(const VkPipelineColorBlendAttachmentState &attachment)
{
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &attachment;
    return colorBlend;
}

VkPipelineDepthStencilStateCreateInfo MakeDisabledDepthStencilState()
{
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    return depthStencil;
}

VkPipelineDynamicStateCreateInfo MakeDynamicStateInfo(const VkDynamicState *dynamicStates, uint32_t dynamicStateCount)
{
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = dynamicStateCount;
    dynamicState.pDynamicStates = dynamicStates;
    return dynamicState;
}

bool UploadAllocation(VmaAllocator allocator, VmaAllocation allocation, const void *data, size_t size)
{
    if (size == 0) {
        return true;
    }

    void *mappedData = nullptr;
    if (vmaMapMemory(allocator, allocation, &mappedData) != VK_SUCCESS || mappedData == nullptr) {
        return false;
    }

    std::memcpy(mappedData, data, size);
    const VkResult result = vmaFlushAllocation(allocator, allocation, 0, size);
    vmaUnmapMemory(allocator, allocation);
    return result == VK_SUCCESS;
}

VkDeviceSize GrowBufferSize(VkDeviceSize requiredSize)
{
    return requiredSize + (requiredSize >> 1);
}

void DestroyBufferWithQueue(VmaAllocator allocator, GpuRetirementQueue *deletionQueue, VkBuffer &buffer,
                            VmaAllocation &allocation, VkDeviceSize &bufferSize)
{
    if (buffer == VK_NULL_HANDLE) {
        allocation = VK_NULL_HANDLE;
        bufferSize = 0;
        return;
    }

    if (deletionQueue != nullptr) {
        const VkBuffer oldBuffer = buffer;
        const VmaAllocation oldAllocation = allocation;
        deletionQueue->Retire(
            [allocator, oldBuffer, oldAllocation]() { vmaDestroyBuffer(allocator, oldBuffer, oldAllocation); });
    } else {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }

    buffer = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
    bufferSize = 0;
}

bool EnsureHostVisibleBuffer(VmaAllocator allocator, GpuRetirementQueue *deletionQueue, VkBufferUsageFlags usage,
                             VkBuffer &buffer, VmaAllocation &allocation, VkDeviceSize &bufferSize,
                             VkDeviceSize requiredSize)
{
    if (requiredSize == 0) {
        return true;
    }
    if (buffer != VK_NULL_HANDLE && bufferSize >= requiredSize) {
        return true;
    }

    DestroyBufferWithQueue(allocator, deletionQueue, buffer, allocation, bufferSize);

    VkBufferCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size = GrowBufferSize(requiredSize);
    createInfo.usage = usage;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    if (vmaCreateBuffer(allocator, &createInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
        bufferSize = 0;
        return false;
    }

    bufferSize = createInfo.size;
    return true;
}
} // namespace

struct InxScreenUIRenderer::CommandPacket::Data
{
    struct List
    {
        bool used = false;
        bool hasVertexOffsets = false;
        std::vector<ImDrawVert> vertices;
        std::vector<ImDrawIdx> indices;
        std::vector<ImDrawCmd> commands;
        std::vector<UIShaderMaterialBinding> bindings;
        std::vector<InxScreenUIRenderer::CommandBindingEvent> bindingEvents;
        std::vector<HDRColorRange> hdr;
    };
    std::array<List, 3> lists;
    std::vector<WorldElementSpan> worlds;
    std::vector<ScreenElementSpan> screens;
};

InxScreenUIRenderer::CommandPacket::CommandPacket() : m_data(std::make_unique<Data>())
{
}
InxScreenUIRenderer::CommandPacket::~CommandPacket() = default;

std::array<uint64_t, 3> InxScreenUIRenderer::GetCommandPacketEpoch() const
{
    return {textlayout::FontCacheGeneration(), static_cast<uint64_t>(ImGui::GetIO().Fonts->TexRef.GetTexID()),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ImGui::GetFont()))};
}

void InxScreenUIRenderer::SetMaterialBinding(ScreenUIList list, const std::string &materialGuid, uint64_t generation,
                                             const std::string &pipelineKey)
{
    SetMaterialBinding(list, materialGuid, generation, pipelineKey, std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}, false,
                       0.0f);
}

void InxScreenUIRenderer::SetMaterialBinding(ScreenUIList list, const std::string &materialGuid, uint64_t generation,
                                             const std::string &pipelineKey, const std::array<float, 4> &baseColor,
                                             bool alphaClipEnabled, float alphaClipThreshold)
{
    if (!m_recordingPacket)
        throw std::logic_error("UI material binding requires an active command packet");
    const bool clearBinding = materialGuid.empty() && generation == 0 && pipelineKey.empty();
    if (!clearBinding && (materialGuid.empty() || generation == 0 || pipelineKey.empty()))
        throw std::invalid_argument("UI material binding requires GUID, generation, and pipeline key");
    const int index = ListIndex(list);
    auto *drawList = GetDrawList(list);
    if (!drawList)
        throw std::logic_error("UI material binding list is not initialized");
    if (drawList->CmdBuffer.empty())
        drawList->AddDrawCmd();
    if (drawList->CmdBuffer.back().ElemCount != 0)
        drawList->AddDrawCmd();
    const int commandIndex = drawList->CmdBuffer.Size - 1;
    auto &events = m_recordingPacket->m_data->lists[index].bindingEvents;
    UIShaderMaterialBinding binding{};
    binding.materialGuid = materialGuid;
    binding.generation = generation;
    binding.pipelineKey = pipelineKey;
    binding.baseColor = baseColor;
    binding.alphaClipEnabled = alphaClipEnabled;
    binding.alphaClipThreshold = std::clamp(alphaClipThreshold, 0.0f, 1.0f);
    if (!events.empty() && events.back().commandIndex == commandIndex && events.back().binding == binding)
        return;
    events.push_back({commandIndex, binding});
}

const std::vector<UIShaderMaterialBinding> &InxScreenUIRenderer::GetCommandBindings(ScreenUIList list) const
{
    return m_commandBindings[ListIndex(list)];
}

void InxScreenUIRenderer::BeginCommandPacket()
{
    if (!m_initialized || m_recordingPacket || m_worldElementStart >= 0)
        throw std::logic_error("UI packet capture requires an initialized renderer outside another element");
    m_recordingPacket = std::make_shared<CommandPacket>();
}

void InxScreenUIRenderer::AbortCommandPacket()
{
    m_commandCacheValid = false;
    m_recordingPacket.reset();
    m_worldElementStart = -1;
    m_pendingWorldElement = {};
}

std::shared_ptr<InxScreenUIRenderer::CommandPacket> InxScreenUIRenderer::EndCommandPacket()
{
    if (!m_recordingPacket || m_worldElementStart >= 0)
        throw std::logic_error("UI packet capture must close all world elements before publication");
    for (size_t index = 0; index < 3; ++index) {
        auto &output = m_recordingPacket->m_data->lists[index];
        if (!output.used)
            continue;
        const auto &source = *m_packetDrawLists[index];
        output.vertices.assign(source.VtxBuffer.begin(), source.VtxBuffer.end());
        output.indices.assign(source.IdxBuffer.begin(), source.IdxBuffer.end());
        output.commands.assign(source.CmdBuffer.begin(), source.CmdBuffer.end());
        while (!output.commands.empty() && !output.commands.back().ElemCount && !output.commands.back().UserCallback)
            output.commands.pop_back();
        // Retained commands must not retain pointers into ImGui's atlas data.
        for (auto &command : output.commands) {
            command.TexRef = ImTextureRef(command.GetTexID());
            output.hasVertexOffsets |= command.VtxOffset != 0;
        }
        output.bindings.assign(output.commands.size(), {});
        for (const auto &event : output.bindingEvents) {
            if (event.commandIndex >= 0 && static_cast<size_t>(event.commandIndex) < output.bindings.size())
                output.bindings[static_cast<size_t>(event.commandIndex)] = event.binding;
        }
        ++m_geometryStats[index].packetCaptures;
    }
    return std::exchange(m_recordingPacket, {});
}

void InxScreenUIRenderer::AppendCommandPackets(const std::vector<std::shared_ptr<CommandPacket>> &packets)
{
    if (!m_initialized || m_recordingPacket || m_worldElementStart >= 0)
        throw std::logic_error("UI packets require an initialized renderer outside element capture");
    for (const auto &packet : packets) {
        if (!packet)
            throw std::invalid_argument("UI command packets must not be null");
        for (int index = 0; index < 3; ++index) {
            const auto &source = packet->m_data->lists[index];
            if (source.vertices.empty())
                continue;
            const auto list =
                index == 0 ? ScreenUIList::Camera : (index == 1 ? ScreenUIList::Overlay : ScreenUIList::World);
            auto &destination = *GetDrawList(list);
            const int vertexStart = destination.VtxBuffer.Size;
            const int indexStart = destination.IdxBuffer.Size;
            // Keep small adjacent packets in the same 16-bit vertex bucket,
            // so retaining elements does not turn one texture batch into N draws.
            const uint64_t indexLimit = uint64_t{1} << (sizeof(ImDrawIdx) * 8);
            const bool sameBucket =
                !source.hasVertexOffsets && source.vertices.size() + destination._VtxCurrentIdx < indexLimit;
            const unsigned int bias = sameBucket ? destination._VtxCurrentIdx : 0;
            const unsigned int base = sameBucket ? destination._CmdHeader.VtxOffset : vertexStart;
            destination.VtxBuffer.resize(vertexStart + static_cast<int>(source.vertices.size()));
            std::memcpy(destination.VtxBuffer.Data + vertexStart, source.vertices.data(),
                        source.vertices.size() * sizeof(ImDrawVert));
            destination.IdxBuffer.resize(indexStart + static_cast<int>(source.indices.size()));
            for (size_t i = 0; i < source.indices.size(); ++i)
                destination.IdxBuffer.Data[indexStart + i] = static_cast<ImDrawIdx>(source.indices[i] + bias);
            if (!destination.CmdBuffer.empty() && !destination.CmdBuffer.back().ElemCount &&
                !destination.CmdBuffer.back().UserCallback) {
                destination.CmdBuffer.pop_back();
                if (m_commandBindings[index].size() > static_cast<size_t>(destination.CmdBuffer.Size))
                    m_commandBindings[index].pop_back();
            }
            const int commandStart = destination.CmdBuffer.Size;
            unsigned int lastBase = base;
            for (size_t sourceIndex = 0; sourceIndex < source.commands.size(); ++sourceIndex) {
                auto command = source.commands[sourceIndex];
                const auto sourceBinding =
                    sourceIndex < source.bindings.size() ? source.bindings[sourceIndex] : UIShaderMaterialBinding{};
                command.IdxOffset += indexStart;
                command.VtxOffset += base;
                lastBase = command.VtxOffset;
                if (index != 2 && !destination.CmdBuffer.empty()) {
                    auto &previous = destination.CmdBuffer.back();
                    const auto previousBinding =
                        m_commandBindings[index].size() > static_cast<size_t>(destination.CmdBuffer.Size - 1)
                            ? m_commandBindings[index][static_cast<size_t>(destination.CmdBuffer.Size - 1)]
                            : UIShaderMaterialBinding{};
                    if (!previous.UserCallback && !command.UserCallback &&
                        previous.IdxOffset + previous.ElemCount == command.IdxOffset &&
                        previous.VtxOffset == command.VtxOffset && previous.GetTexID() == command.GetTexID() &&
                        std::memcmp(&previous.ClipRect, &command.ClipRect, sizeof(ImVec4)) == 0 &&
                        previousBinding == sourceBinding) {
                        previous.ElemCount += command.ElemCount;
                        continue;
                    }
                }
                destination.CmdBuffer.push_back(command);
                m_commandBindings[index].push_back(sourceBinding);
            }
            for (auto range : source.hdr) {
                range.vertexStart += vertexStart;
                range.vertexEnd += vertexStart;
                GetHDRRanges(list).push_back(range);
            }
            if (index == 2) {
                for (auto span : packet->m_data->worlds) {
                    ResolveWorldPose(span);
                    span.vertexStart += vertexStart;
                    span.vertexEnd += vertexStart;
                    span.commandStart += commandStart;
                    span.commandEnd += commandStart;
                    m_hasSelectiveWorldOcclusion |= span.ignoredOccluderId != 0 && !span.alwaysOnTop;
                    m_worldElementSpans.push_back(span);
                }
            } else {
                for (auto span : packet->m_data->screens) {
                    if (span.list != list)
                        continue;
                    span.vertexStart += vertexStart;
                    span.vertexEnd += vertexStart;
                    m_screenElementSpans.push_back(span);
                }
            }
            destination._CmdHeader.VtxOffset = lastBase;
            destination._VtxCurrentIdx = destination.VtxBuffer.Size - lastBase;
            destination._VtxWritePtr = destination.VtxBuffer.end();
            destination._IdxWritePtr = destination.IdxBuffer.end();
            destination.AddDrawCmd();
            m_commandBindings[index].push_back({});
            ++m_geometryRevision[index];
            ++m_geometryStats[index].packetAppends;
        }
    }
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

InxScreenUIRenderer::InxScreenUIRenderer() = default;

InxScreenUIRenderer::~InxScreenUIRenderer()
{
    Destroy();
}

// ============================================================================
// Initialization
// ============================================================================

bool InxScreenUIRenderer::Initialize(VkDevice device, VmaAllocator allocator, VkFormat colorFormat,
                                     VkFormat depthFormat, VkSampleCountFlagBits msaaSamples, uint32_t frameCount)
{
    if (m_initialized)
        return true;

    m_device = device;
    m_allocator = allocator;
    m_colorFormat = colorFormat;
    m_depthFormat = depthFormat;
    m_msaaSamples = msaaSamples;
    m_frameBuffers.resize(frameCount);
    if (!CreatePipeline() || !CreateWorldPipeline()) {
        INXLOG_ERROR("InxScreenUIRenderer: Failed to create screen/world pipeline");
        return false;
    }

    // Create standalone ImDrawList instances (not attached to any ImGui window)
    ImDrawListSharedData *sharedData = ImGui::GetDrawListSharedData();
    m_cameraDrawList = IM_NEW(ImDrawList)(sharedData);
    m_overlayDrawList = IM_NEW(ImDrawList)(sharedData);
    m_worldDrawList = IM_NEW(ImDrawList)(sharedData);

    m_initialized = true;
    // INXLOG_INFO("InxScreenUIRenderer initialized (format=", static_cast<int>(colorFormat),
    //             ", MSAA=", static_cast<int>(msaaSamples), ")");
    return true;
}

void InxScreenUIRenderer::Destroy()
{
    m_commandCacheValid = false;
    AbortCommandPacket();
    m_worldElementSpans.clear();
    m_hasSelectiveWorldOcclusion = false;
    for (auto &drawList : m_packetDrawLists) {
        if (drawList)
            IM_DELETE(drawList);
        drawList = nullptr;
    }
    if (m_cameraDrawList) {
        IM_DELETE(m_cameraDrawList);
        m_cameraDrawList = nullptr;
    }
    if (m_overlayDrawList) {
        IM_DELETE(m_overlayDrawList);
        m_overlayDrawList = nullptr;
    }
    if (m_worldDrawList) {
        IM_DELETE(m_worldDrawList);
        m_worldDrawList = nullptr;
    }

    if (m_device != VK_NULL_HANDLE) {
        for (auto &frame : m_frameBuffers) {
            for (auto &buf : frame) {
                if (buf.vertexBuffer)
                    vmaDestroyBuffer(m_allocator, buf.vertexBuffer, buf.vertexAlloc);
                if (buf.indexBuffer)
                    vmaDestroyBuffer(m_allocator, buf.indexBuffer, buf.indexAlloc);
            }
        }
        if (m_pipeline)
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
        for (const auto &variant : m_worldPipelineVariants)
            vkDestroyPipeline(m_device, variant.pipeline, nullptr);
        m_worldPipelineVariants.clear();
        if (m_worldPipeline)
            vkDestroyPipeline(m_device, m_worldPipeline, nullptr);
        if (m_worldTopPipeline)
            vkDestroyPipeline(m_device, m_worldTopPipeline, nullptr);
        if (m_pipelineLayout)
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        if (m_worldPipelineLayout)
            vkDestroyPipelineLayout(m_device, m_worldPipelineLayout, nullptr);
        if (m_descriptorSetLayout)
            vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
        if (m_vertShader)
            vkDestroyShaderModule(m_device, m_vertShader, nullptr);
        if (m_worldVertShader)
            vkDestroyShaderModule(m_device, m_worldVertShader, nullptr);
        if (m_worldFragShader)
            vkDestroyShaderModule(m_device, m_worldFragShader, nullptr);
        if (m_fragShader)
            vkDestroyShaderModule(m_device, m_fragShader, nullptr);
    }

    m_frameBuffers.clear();
    m_preparedRevision = {};
    m_geometryStats = {};
    m_screenVertices = {};
    m_worldVertices.clear();
    m_commandBindings = {};
    m_pipeline = VK_NULL_HANDLE;
    m_worldPipeline = VK_NULL_HANDLE;
    m_worldTopPipeline = VK_NULL_HANDLE;
    m_pipelineLayout = VK_NULL_HANDLE;
    m_worldPipelineLayout = VK_NULL_HANDLE;
    m_descriptorSetLayout = VK_NULL_HANDLE;
    m_fontDescriptorSet = VK_NULL_HANDLE;
    m_vertShader = VK_NULL_HANDLE;
    m_worldVertShader = VK_NULL_HANDLE;
    m_worldFragShader = VK_NULL_HANDLE;
    m_fragShader = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_allocator = VK_NULL_HANDLE;
    m_initialized = false;
}

// ============================================================================
// Per-Frame Command Accumulation
// ============================================================================

void InxScreenUIRenderer::BeginFrame(uint32_t width, uint32_t height)
{
    if (!m_initialized)
        return;
    if (m_recordingPacket)
        throw std::logic_error("UI packet capture must finish before beginning a frame");
    m_cachedWidth = width;
    m_cachedHeight = height;

    m_cameraHDRRanges.clear();
    m_overlayHDRRanges.clear();
    m_worldHDRRanges.clear();
    m_worldElementSpans.clear();
    m_hasSelectiveWorldOcclusion = false;
    m_screenElementSpans.clear();
    m_worldElementStart = -1;
    m_commandCacheValid = false;

    for (auto &bindings : m_commandBindings)
        bindings.clear();

    for (auto &revision : m_geometryRevision)
        ++revision;
    ResetDrawListForFrame(*m_cameraDrawList, width, height);
    ResetDrawListForFrame(*m_overlayDrawList, width, height);
    ResetDrawListForFrame(*m_worldDrawList, width, height, false);
    for (auto &bindings : m_commandBindings)
        bindings.push_back({}); // ImDrawList starts with one empty command.
}

bool InxScreenUIRenderer::BeginFrameCached(uint32_t width, uint32_t height, uint64_t contentRevision)
{
    if (!m_initialized)
        return false;

    const auto packetEpoch = GetCommandPacketEpoch();
    if (m_commandCacheValid && m_cachedWidth == width && m_cachedHeight == height &&
        m_cachedContentRevision == contentRevision && m_cachedPacketEpoch == packetEpoch) {
        return true;
    }

    BeginFrame(width, height);
    m_cachedWidth = width;
    m_cachedHeight = height;
    m_cachedContentRevision = contentRevision;
    m_cachedPacketEpoch = packetEpoch;
    m_commandCacheValid = true;
    return false;
}

bool UploadAllocationRange(VmaAllocator allocator, VmaAllocation allocation, size_t offset, const void *data,
                           size_t size)
{
    if (size == 0)
        return true;
    void *mappedData = nullptr;
    if (vmaMapMemory(allocator, allocation, &mappedData) != VK_SUCCESS || mappedData == nullptr)
        return false;
    std::memcpy(static_cast<std::byte *>(mappedData) + offset, data, size);
    const VkResult result = vmaFlushAllocation(allocator, allocation, offset, size);
    vmaUnmapMemory(allocator, allocation);
    return result == VK_SUCCESS;
}

VkPipelineDepthStencilStateCreateInfo MakeWorldDepthStencilState()
{
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    return depthStencil;
}

void InxScreenUIRenderer::PushClipRect(ScreenUIList list, float minX, float minY, float maxX, float maxY)
{
    if (ImDrawList *drawList = GetDrawList(list))
        drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);
}

void InxScreenUIRenderer::PopClipRect(ScreenUIList list)
{
    if (ImDrawList *drawList = GetDrawList(list))
        drawList->PopClipRect();
}

void InxScreenUIRenderer::BeginWorldElement(const std::array<float, 16> &localToWorld, float pivotX, float pivotY,
                                            uint32_t layerMask, bool alwaysOnTop, bool billboard, bool constantScreenSize)
{
    auto *drawList = GetDrawList(ScreenUIList::World);
    if (m_worldElementStart >= 0)
        throw std::logic_error("World UI elements cannot be nested");
    if (!drawList)
        throw std::logic_error("World UI renderer is not initialized");

    // Each element is a transparency-sorting unit. Keep a hard command
    // boundary so one ImDrawCmd can never contain geometry from two world
    // objects, even when both use the same texture and clip rectangle.
    if (drawList->CmdBuffer.empty())
        throw std::logic_error("World UI draw list has no active command");
    if (drawList->CmdBuffer.back().ElemCount != 0)
        drawList->AddDrawCmd();

    m_worldElementStart = drawList->VtxBuffer.Size;
    m_pendingWorldElement.commandStart = drawList->CmdBuffer.Size - 1;
    std::memcpy(glm::value_ptr(m_pendingWorldElement.localToWorld), localToWorld.data(), sizeof(float) * 16);
    m_pendingWorldElement.pivotX = pivotX;
    m_pendingWorldElement.pivotY = pivotY;
    m_pendingWorldElement.layerMask = layerMask;
    m_pendingWorldElement.alwaysOnTop = alwaysOnTop;
    m_pendingWorldElement.billboard = billboard;
    m_pendingWorldElement.constantScreenSize = constantScreenSize;
    m_pendingWorldElement.ignoredOccluderId = 0;
}

void InxScreenUIRenderer::ResolveWorldPose(WorldElementSpan &span)
{
    if (!span.transform.IsValid())
        return; // Explicit-matrix commands do not bind a scene object.
    auto &store = TransformECSStore::Instance();
    if (!store.IsValid(span.transform))
        throw std::runtime_error("World UI packet outlived its Transform; rebuild membership");
    auto *transform = store.GetOwner(span.transform);
    span.localToWorld = glm::mat4_cast(transform->GetRotation());
    span.localToWorld[3] = glm::vec4(transform->GetPosition(), 1.0f);
    span.layerMask = uint32_t(1) << transform->GetGameObject()->GetLayer();
}

void InxScreenUIRenderer::BeginWorldObject(GameObject *object, float pivotX, float pivotY, bool alwaysOnTop,
                                           bool billboard, bool constantScreenSize, uint64_t ignoredOccluderId)
{
    if (!object)
        throw std::invalid_argument("World UI geometry requires a scene object");
    BeginWorldElement({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, pivotX, pivotY, 0xffffffffu,
                      alwaysOnTop, billboard, constantScreenSize);
    m_pendingWorldElement.transform = object->GetTransform()->GetECSHandle();
    m_pendingWorldElement.ignoredOccluderId = ignoredOccluderId;
    ResolveWorldPose(m_pendingWorldElement);
}

bool InxScreenUIRenderer::ResolveScreenPose(ScreenElementSpan &span)
{
    if (!span.transform.IsValid())
        return false;
    auto &store = TransformECSStore::Instance();
    if (!store.IsValid(span.transform))
        throw std::runtime_error("Screen UI packet outlived its Transform; rebuild membership");
    auto *transform = store.GetOwner(span.transform);
    // Screen-space UI is laid out in the Canvas' viewport coordinates.  A
    // Canvas (or any other scene parent) must not inject its world pose into
    // that layout; only the element's own local pose is meaningful here.
    // This also keeps the native pose path consistent with
    // UITransformDependencies and the Python screen-layout contract.
    const auto position = transform->GetLocalPosition();
    const auto angles = transform->GetLocalEulerAngles();
    const auto transformScale = transform->GetLocalScale();
    const glm::vec3 nextDelta{(position.x - span.position.x) * span.scaleX,
                              -(position.y - span.position.y) * span.scaleY, 0.0f};
    const float nextDeltaRotation = glm::radians(angles.z - span.rotation);
    const float nextScaleX = transformScale.x / span.transformScaleX;
    const float nextScaleY = transformScale.y / span.transformScaleY;
    const bool changed =
        glm::length(nextDelta - span.delta) > 1.0e-6f || std::fabs(nextDeltaRotation - span.deltaRotation) > 1.0e-6f ||
        std::fabs(nextScaleX - span.currentScaleX) > 1.0e-6f || std::fabs(nextScaleY - span.currentScaleY) > 1.0e-6f;
    span.delta = nextDelta;
    span.deltaRotation = nextDeltaRotation;
    span.currentScaleX = nextScaleX;
    span.currentScaleY = nextScaleY;
    return changed;
}

void InxScreenUIRenderer::ApplyScreenPose(const ScreenElementSpan &span, const ImVec2 &source, ImVec2 &target)
{
    const float cs = std::cos(span.deltaRotation);
    const float sn = std::sin(span.deltaRotation);
    const float x = (source.x - span.pivotX) * span.currentScaleX;
    const float y = (source.y - span.pivotY) * span.currentScaleY;
    target = ImVec2(span.pivotX + x * cs - y * sn + span.delta.x, span.pivotY + x * sn + y * cs + span.delta.y);
}

void InxScreenUIRenderer::BeginScreenObject(GameObject *object, ScreenUIList list, float pivotX, float pivotY,
                                            float scaleX, float scaleY)
{
    if (!object || !m_recordingPacket)
        throw std::logic_error("Screen UI object binding requires an active packet and scene object");
    auto *drawList = GetDrawList(list);
    if (!drawList)
        throw std::logic_error("Screen UI renderer is not initialized");
    ScreenElementSpan span;
    span.vertexStart = drawList->VtxBuffer.Size;
    span.list = list;
    span.transform = object->GetTransform()->GetECSHandle();
    // Capture the local screen pose.  Using world values here makes moving or
    // scaling a Canvas move all of its children in screen space.
    const auto position = object->GetTransform()->GetLocalPosition();
    span.position = position;
    span.rotation = object->GetTransform()->GetLocalEulerAngles().z;
    const auto transformScale = object->GetTransform()->GetLocalScale();
    span.transformScaleX = transformScale.x;
    span.transformScaleY = transformScale.y;
    span.pivotX = pivotX;
    span.pivotY = pivotY;
    span.scaleX = scaleX;
    span.scaleY = scaleY;
    m_recordingPacket->m_data->screens.push_back(span);
}

void InxScreenUIRenderer::EndScreenObject()
{
    if (!m_recordingPacket || m_recordingPacket->m_data->screens.empty())
        throw std::logic_error("Screen UI object binding requires BeginScreenObject");
    auto &span = m_recordingPacket->m_data->screens.back();
    auto *drawList = GetDrawList(span.list);
    span.vertexEnd = drawList->VtxBuffer.Size;
    if (span.vertexEnd == span.vertexStart)
        m_recordingPacket->m_data->screens.pop_back();
}

void InxScreenUIRenderer::EndWorldElement()
{
    auto *drawList = GetDrawList(ScreenUIList::World);
    if (m_worldElementStart < 0)
        throw std::logic_error("EndWorldElement requires a matching BeginWorldElement");
    m_pendingWorldElement.vertexStart = m_worldElementStart;
    m_pendingWorldElement.vertexEnd = drawList->VtxBuffer.Size;
    m_pendingWorldElement.commandEnd = drawList->CmdBuffer.Size;
    while (m_pendingWorldElement.commandEnd > m_pendingWorldElement.commandStart &&
           drawList->CmdBuffer[m_pendingWorldElement.commandEnd - 1].ElemCount == 0)
        --m_pendingWorldElement.commandEnd;
    if (m_pendingWorldElement.vertexEnd > m_pendingWorldElement.vertexStart) {
        if (m_pendingWorldElement.commandEnd <= m_pendingWorldElement.commandStart)
            throw std::logic_error("World UI element produced vertices without a draw command");
        auto &spans = m_recordingPacket ? m_recordingPacket->m_data->worlds : m_worldElementSpans;
        spans.push_back(m_pendingWorldElement);
        if (!m_recordingPacket)
            m_hasSelectiveWorldOcclusion |=
                m_pendingWorldElement.ignoredOccluderId != 0 && !m_pendingWorldElement.alwaysOnTop;
        // Texture/clip changes may merge an empty ImDrawCmd with the previous
        // one. A callback command is the explicit, non-mergeable separator
        // between independently sorted world elements. RenderWorld consumes
        // only the recorded element command spans, so the marker is never
        // submitted to Vulkan.
        drawList->AddCallback(WorldElementBoundaryCallback, nullptr);
    }
    m_worldElementStart = -1;
    m_pendingWorldElement = {};
}

void InxScreenUIRenderer::AddFilledRect(ScreenUIList list, float minX, float minY, float maxX, float maxY, float r,
                                        float g, float b, float a, float rounding, float rotation, bool mirrorH,
                                        bool mirrorV)
{
    ImDrawList *dl = GetDrawList(list);
    if (!dl)
        return;
    const int vtxStart = dl->VtxBuffer.Size;
    const float hdrScale = ExtractHDRScale(r, g, b);
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
    const VertexTransform transform = MakeVertexTransform(minX, minY, maxX, maxY, rotation, mirrorH, mirrorV);
    dl->AddRectFilled(ImVec2(minX, minY), ImVec2(maxX, maxY), col, rounding);
    ApplyVertexTransform(*dl, vtxStart, transform);
    TrackHDRColorRange(list, vtxStart, dl->VtxBuffer.Size, hdrScale);
}

void InxScreenUIRenderer::AddImage(ScreenUIList list, uint64_t textureId, float minX, float minY, float maxX,
                                   float maxY, float uv0X, float uv0Y, float uv1X, float uv1Y, float r, float g,
                                   float b, float a, float rotation, bool mirrorH, bool mirrorV, float rounding)
{
    ImDrawList *dl = GetDrawList(list);
    if (!dl || textureId == 0)
        return;

    const int vtxStart = dl->VtxBuffer.Size;
    const float hdrScale = ExtractHDRScale(r, g, b);
    ImU32 tint = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
    const VertexTransform transform = MakeVertexTransform(minX, minY, maxX, maxY, rotation, mirrorH, mirrorV);
    if (rounding > 0.5f)
        dl->AddImageRounded(ToImTextureID(textureId), ImVec2(minX, minY), ImVec2(maxX, maxY), ImVec2(uv0X, uv0Y),
                            ImVec2(uv1X, uv1Y), tint, rounding);
    else
        dl->AddImage(ToImTextureID(textureId), ImVec2(minX, minY), ImVec2(maxX, maxY), ImVec2(uv0X, uv0Y),
                     ImVec2(uv1X, uv1Y), tint);

    ApplyVertexTransform(*dl, vtxStart, transform);
    TrackHDRColorRange(list, vtxStart, dl->VtxBuffer.Size, hdrScale);
}

void InxScreenUIRenderer::AddText(ScreenUIList list, float minX, float minY, float maxX, float maxY,
                                  const std::string &text, float r, float g, float b, float a, float alignX,
                                  float alignY, float fontSize, float wrapWidth, float rotation, bool mirrorH,
                                  bool mirrorV, const std::string &fontPath, float lineHeight, float letterSpacing,
                                  bool clip, const std::vector<std::string> &fallbackFontPaths)
{
    ImDrawList *dl = GetDrawList(list);
    if (!dl || text.empty())
        return;

    const textlayout::TextLayoutResult layout = textlayout::LayoutText(
        {text, fontPath, ResolveFontSize(fontSize), wrapWidth, lineHeight, letterSpacing, fallbackFontPaths});

    const float hdrScale = ExtractHDRScale(r, g, b);
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
    const int vtxStart = dl->VtxBuffer.Size;
    const VertexTransform transform = MakeVertexTransform(minX, minY, maxX, maxY, rotation, mirrorH, mirrorV);
    if (clip)
        dl->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);
    dl->PushTextureID(ImGui::GetIO().Fonts->TexRef);
    textlayout::RenderTextBox(dl, minX, minY, maxX, maxY, layout, col, alignX, alignY, letterSpacing);
    dl->PopTextureID();
    if (clip)
        dl->PopClipRect();

    ApplyVertexTransform(*dl, vtxStart, transform);
    TrackHDRColorRange(list, vtxStart, dl->VtxBuffer.Size, hdrScale);
}

std::pair<float, float> InxScreenUIRenderer::MeasureText(const std::string &text, float fontSize, float wrapWidth,
                                                         const std::string &fontPath, float lineHeight,
                                                         float letterSpacing,
                                                         const std::vector<std::string> &fallbackFontPaths) const
{
    const textlayout::TextLayoutResult layout = textlayout::LayoutText(
        {text, fontPath, ResolveFontSize(fontSize), wrapWidth, lineHeight, letterSpacing, fallbackFontPaths});
    return {layout.totalWidth, layout.totalHeight};
}

bool InxScreenUIRenderer::HasCommands(ScreenUIList list) const
{
    const ImDrawList *dl = GetDrawList(list);
    return dl && dl->CmdBuffer.Size > 0 && dl->VtxBuffer.Size > 0;
}

void InxScreenUIRenderer::TrackHDRColorRange(ScreenUIList list, int vertexStart, int vertexEnd, float rgbScale)
{
    if (vertexEnd > vertexStart)
        ++m_geometryRevision[ListIndex(list)];
    if (rgbScale <= 1.0f || vertexEnd <= vertexStart) {
        return;
    }

    auto &ranges = GetHDRRanges(list);
    ranges.push_back({vertexStart, vertexEnd, rgbScale});
}

std::vector<InxScreenUIRenderer::HDRColorRange> &InxScreenUIRenderer::GetHDRRanges(ScreenUIList list)
{
    if (m_recordingPacket)
        return m_recordingPacket->m_data->lists[ListIndex(list)].hdr;
    if (list == ScreenUIList::Camera)
        return m_cameraHDRRanges;
    if (list == ScreenUIList::Overlay)
        return m_overlayHDRRanges;
    return m_worldHDRRanges;
}

const std::vector<InxScreenUIRenderer::HDRColorRange> &InxScreenUIRenderer::GetHDRRanges(ScreenUIList list) const
{
    if (list == ScreenUIList::Camera)
        return m_cameraHDRRanges;
    if (list == ScreenUIList::Overlay)
        return m_overlayHDRRanges;
    return m_worldHDRRanges;
}

// ============================================================================
// Rendering
// ============================================================================

bool InxScreenUIRenderer::UploadGeometry(ListBuffers &buf, ScreenUIList list, const void *vertices, size_t vertexBytes)
{
    const int index = ListIndex(list);
    if (buf.uploadedRevision == m_geometryRevision[index])
        return true;
    const auto &indices = GetDrawList(list)->IdxBuffer;
    const size_t indexBytes = static_cast<size_t>(indices.Size) * sizeof(ImDrawIdx);
    if (!EnsureBuffers(buf, vertexBytes, indexBytes) ||
        !UploadAllocation(m_allocator, buf.vertexAlloc, vertices, vertexBytes) ||
        !UploadAllocation(m_allocator, buf.indexAlloc, indices.Data, indexBytes)) {
        INXLOG_ERROR("InxScreenUIRenderer: Failed to upload UI geometry");
        return false;
    }
    buf.uploadedRevision = m_geometryRevision[index];
    ++m_geometryStats[index].uploads;
    m_geometryStats[index].uploadedBytes += vertexBytes + indexBytes;
    return true;
}

void InxScreenUIRenderer::Render(VkCommandBuffer cmdBuf, ScreenUIList list, uint32_t width, uint32_t height,
                                 uint32_t frameSlot)
{
    if (list == ScreenUIList::World)
        throw std::invalid_argument("World UI requires RenderWorld with the active camera matrix");
    const int listIndex = ListIndex(list);
    m_lastSubmittedDrawCounts[listIndex] = 0;
    m_lastSubmittedIndexCounts[listIndex] = 0;
    if (!m_initialized || !m_pipeline || width == 0 || height == 0 || !m_enabled)
        return;

    ImDrawList *dl = GetDrawList(list);
    if (!dl || dl->VtxBuffer.Size == 0 || dl->IdxBuffer.Size == 0)
        return;

    // With ImGui 1.92+ dynamic font atlas, the texture may be recreated
    // at any time. Always pull the current descriptor set before drawing.
    if (!RefreshFontDescriptorSet(m_fontDescriptorSet)) {
        return;
    }

    auto &gpuVertices = m_screenVertices[listIndex];
    if (m_preparedRevision[listIndex] != m_geometryRevision[listIndex]) {
        gpuVertices.resize(static_cast<size_t>(dl->VtxBuffer.Size));
        auto &localPositions = m_screenLocalPositions[listIndex];
        localPositions.resize(static_cast<size_t>(dl->VtxBuffer.Size));

        const auto &hdrRanges = GetHDRRanges(list);
        size_t rangeIndex = 0;
        for (int i = 0; i < dl->VtxBuffer.Size; ++i) {
            while (rangeIndex < hdrRanges.size() && i >= hdrRanges[rangeIndex].vertexEnd) {
                ++rangeIndex;
            }

            float rgbScale = 1.0f;
            if (rangeIndex < hdrRanges.size()) {
                const HDRColorRange &range = hdrRanges[rangeIndex];
                if (i >= range.vertexStart && i < range.vertexEnd) {
                    rgbScale = range.rgbScale;
                }
            }

            const ImDrawVert &src = dl->VtxBuffer[i];
            localPositions[static_cast<size_t>(i)] = src.pos;
            GPUVertex &dst = gpuVertices[static_cast<size_t>(i)];
            dst.pos = src.pos;
            dst.uv = src.uv;

            const ImVec4 unpacked = ImGui::ColorConvertU32ToFloat4(src.col);
            dst.color[0] = unpacked.x * rgbScale;
            dst.color[1] = unpacked.y * rgbScale;
            dst.color[2] = unpacked.z * rgbScale;
            dst.color[3] = unpacked.w;
        }
        m_preparedRevision[listIndex] = m_geometryRevision[listIndex];
        ++m_geometryStats[listIndex].preparations;
    }

    // Pose-only updates stay on the native side. Re-read the immutable local
    // vertex positions and apply each bound Transform delta without invoking
    // Python extraction or rebuilding text/image geometry.
    bool poseDirty = false;
    int poseMinVertex = static_cast<int>(gpuVertices.size());
    int poseMaxVertex = 0;
    for (auto &span : m_screenElementSpans) {
        if (span.list != list)
            continue;
        const bool spanDirty = ResolveScreenPose(span);
        poseDirty = spanDirty || poseDirty;
        if (spanDirty) {
            poseMinVertex = std::min(poseMinVertex, std::max(0, span.vertexStart));
            poseMaxVertex = std::max(poseMaxVertex, std::min(span.vertexEnd, static_cast<int>(gpuVertices.size())));
        }
        const auto &localPositions = m_screenLocalPositions[listIndex];
        const int end = std::min(span.vertexEnd, static_cast<int>(gpuVertices.size()));
        for (int i = std::max(0, span.vertexStart); i < end; ++i) {
            if (i < static_cast<int>(localPositions.size()))
                ApplyScreenPose(span, localPositions[static_cast<size_t>(i)], gpuVertices[static_cast<size_t>(i)].pos);
        }
    }

    const VkDeviceSize vtxSize = gpuVertices.size() * sizeof(GPUVertex);
    ListBuffers &buf = m_frameBuffers[frameSlot][listIndex];
    const bool geometryDirty = buf.uploadedRevision != m_geometryRevision[listIndex];
    if (poseDirty && !geometryDirty && poseMaxVertex > poseMinVertex) {
        const size_t firstByte = static_cast<size_t>(poseMinVertex) * sizeof(GPUVertex);
        const size_t rangeBytes = static_cast<size_t>(poseMaxVertex - poseMinVertex) * sizeof(GPUVertex);
        if (!UploadAllocationRange(m_allocator, buf.vertexAlloc, firstByte, gpuVertices.data() + poseMinVertex,
                                   rangeBytes)) {
            INXLOG_ERROR("InxScreenUIRenderer: Failed to upload screen UI pose range");
            return;
        }
        ++m_geometryStats[listIndex].uploads;
        m_geometryStats[listIndex].uploadedBytes += rangeBytes;
    }
    if (!UploadGeometry(buf, list, gpuVertices.data(), static_cast<size_t>(vtxSize)))
        return;

    // ---- Bind pipeline ----
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    // ---- Bind vertex/index buffers ----
    const VkBuffer vertexBuffer = buf.vertexBuffer;
    const VkDeviceSize vertexBufferOffset = 0;
    vkCmdBindVertexBuffers(cmdBuf, 0, 1, &vertexBuffer, &vertexBufferOffset);
    vkCmdBindIndexBuffer(cmdBuf, buf.indexBuffer, 0,
                         sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);

    const VkViewport viewport = MakeViewport(width, height);
    vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

    // The projection is shared by every draw, while authored material values
    // are command-local.  The complete block is pushed per draw so the
    // fragment shader consumes the same GUID-backed material contract that
    // produced the retained command.
    const ScreenUIPushConstants projection = MakeOrthoPushConstants(width, height);

    // ---- Bind font atlas descriptor set ----
    {
        const uint64_t fontDescRaw = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(m_fontDescriptorSet));
        const uint32_t lo = static_cast<uint32_t>(fontDescRaw & 0xffffffffull);
        const uint32_t hi = static_cast<uint32_t>((fontDescRaw >> 32) & 0xffffffffull);
        if (hi == lo && lo != 0u && lo <= 0x000fffffu) {
            static int badUiDescWarnCount = 0;
            if (badUiDescWarnCount++ < 16) {
                INXLOG_WARN("[InxScreenUIRenderer] suspicious font set0 desc=0x", fontDescRaw,
                            " -- skip UI batch to avoid invalid bind");
            }
            return;
        }
    }
    vkdebug::CmdBindDescriptorSetsTracked("InxScreenUIRenderer.Render.Set0Font", cmdBuf,
                                          VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_fontDescriptorSet,
                                          0, nullptr);
    VkDescriptorSet lastBoundDescSet = m_fontDescriptorSet;

    // ---- Issue draw commands ----
    const float frameWidth = static_cast<float>(width);
    const float frameHeight = static_cast<float>(height);

    uint32_t submittedDraws = 0;
    uint64_t submittedIndices = 0;
    for (int cmdI = 0; cmdI < dl->CmdBuffer.Size; cmdI++) {
        const ImDrawCmd &cmd = dl->CmdBuffer[cmdI];

        if (cmd.UserCallback != nullptr || cmd.ElemCount == 0) {
            // User callbacks are not supported in scene render passes
            continue;
        }

        // Per-command texture (usually font atlas)
        VkDescriptorSet texDescSet = reinterpret_cast<VkDescriptorSet>(static_cast<uintptr_t>(cmd.GetTexID()));
        if (texDescSet == VK_NULL_HANDLE) {
            m_commandCacheValid = false;
            continue;
        }
        if (texDescSet != m_fontDescriptorSet && m_textureUsageValidator &&
            !m_textureUsageValidator(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(texDescSet)))) {
            // Cached ImDrawLists retain raw descriptor handles. A texture replacement or
            // residency eviction invalidates that handle, so reject this command and force
            // Python to rebuild the list with the currently published texture next UI tick.
            m_commandCacheValid = false;
            continue;
        }
        {
            const uint64_t texDescRaw = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(texDescSet));
            const uint32_t lo = static_cast<uint32_t>(texDescRaw & 0xffffffffull);
            const uint32_t hi = static_cast<uint32_t>((texDescRaw >> 32) & 0xffffffffull);
            if (hi == lo && lo != 0u && lo <= 0x000fffffu) {
                static int badUiTexDescWarnCount = 0;
                if (badUiTexDescWarnCount++ < 24) {
                    INXLOG_WARN("[InxScreenUIRenderer] suspicious cmd texture set0 desc=0x", texDescRaw,
                                " -- skip draw command");
                }
                continue;
            }
        }
        if (texDescSet != lastBoundDescSet) {
            vkdebug::CmdBindDescriptorSetsTracked("InxScreenUIRenderer.Render.Set0Tex", cmdBuf,
                                                  VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &texDescSet,
                                                  0, nullptr);
            lastBoundDescSet = texDescSet;
        }

        // Scissor rect from ImDrawCmd clip rect — clamped to render area
        // to prevent Vulkan validation errors and potential DEVICE_LOST.
        VkRect2D scissor{};
        if (!MakeClampedScissor(cmd, frameWidth, frameHeight, scissor))
            continue; // Degenerate scissor — skip draw

        vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

        const auto binding = (static_cast<size_t>(cmdI) < m_commandBindings[listIndex].size())
                                 ? m_commandBindings[listIndex][static_cast<size_t>(cmdI)]
                                 : UIShaderMaterialBinding{};
        ScreenUIPushConstants pushConstants = projection;
        pushConstants.materialColor = binding.baseColor;
        pushConstants.alphaClipEnabled = binding.alphaClipEnabled ? 1.0f : 0.0f;
        pushConstants.alphaClipThreshold = binding.alphaClipThreshold;
        // Camera UI is drawn in linear space; Overlay is after display encoding.
        // Font alpha and display-space uploads must not be gamma transformed.
        pushConstants.encodeSample =
            list == ScreenUIList::Overlay && m_textureColorSpaceQuery &&
                    m_textureColorSpaceQuery(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(texDescSet)))
                ? 1.0f
                : 0.0f;
        vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(pushConstants), &pushConstants);

        vkCmdDrawIndexed(cmdBuf, cmd.ElemCount, 1, cmd.IdxOffset, static_cast<int32_t>(cmd.VtxOffset), 0);
        ++submittedDraws;
        submittedIndices += cmd.ElemCount;
    }

    if (!m_reportedFirstDraw && submittedDraws > 0) {
        m_reportedFirstDraw = true;
        INXLOG_INFO("INFERNUX_SCREEN_UI_READY list=", list == ScreenUIList::Camera ? "camera" : "overlay",
                    " width=", width, " height=", height, " vertices=", dl->VtxBuffer.Size,
                    " indices=", submittedIndices, " draws=", submittedDraws);
    }
    m_lastSubmittedDrawCounts[listIndex] = submittedDraws;
    m_lastSubmittedIndexCounts[listIndex] = submittedIndices;
}

// ============================================================================
// Pipeline Creation
// ============================================================================

bool InxScreenUIRenderer::CreatePipeline()
{
    // ---- Shader modules ----
    InxShaderLoader compiler(false, true, false, true, false, true, false, false, false, false);
    const auto vertex = compiler.CompileVertexGlsl(kScreenVertexShader, "Infernux/ScreenUI.vert");
    const auto fragment = compiler.CompileFragmentGlsl(kScreenFragmentShader, "Infernux/ScreenUI.frag");
    if (!CreateShaderModule(m_device, reinterpret_cast<const uint32_t *>(vertex.data()), vertex.size(), m_vertShader))
        return false;
    if (!CreateShaderModule(m_device, reinterpret_cast<const uint32_t *>(fragment.data()), fragment.size(),
                            m_fragShader))
        return false;

    // ---- Descriptor set layout (identical to ImGui's) ----
    const VkDescriptorSetLayoutBinding binding = vkrender::MakeDescriptorSetLayoutBinding(
        kFontTextureBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
    if (!vkrender::CreateDescriptorSetLayout(m_device, &binding, 1, m_descriptorSetLayout))
        return false;

    // Ortho projection plus the sampled texture's display-encoding flag.
    auto pushConstRange = MakeVertexPushConstantRange(sizeof(ScreenUIPushConstants));
    pushConstRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    if (!CreatePipelineLayout(m_device, m_descriptorSetLayout, pushConstRange, m_pipelineLayout))
        return false;

    // ---- Graphics pipeline (replicates ImGui's pipeline for scene render target) ----
    const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
        MakeShaderStageInfo(VK_SHADER_STAGE_VERTEX_BIT, m_vertShader),
        MakeShaderStageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, m_fragShader),
    };

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.stride = sizeof(GPUVertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDesc[3]{};
    attrDesc[0].location = 0;
    attrDesc[0].format = VK_FORMAT_R32G32_SFLOAT;
    attrDesc[0].offset = offsetof(GPUVertex, pos);
    attrDesc[1].location = 1;
    attrDesc[1].format = VK_FORMAT_R32G32_SFLOAT;
    attrDesc[1].offset = offsetof(GPUVertex, uv);
    attrDesc[2].location = 2;
    attrDesc[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrDesc[2].offset = offsetof(GPUVertex, color);

    const VkPipelineVertexInputStateCreateInfo vertInput = MakeVertexInputState(bindingDesc, attrDesc, 3);
    const VkPipelineInputAssemblyStateCreateInfo iaInfo = MakeTriangleListInputAssembly();
    const VkPipelineViewportStateCreateInfo vpInfo = MakeDynamicViewportState();
    const VkPipelineRasterizationStateCreateInfo rsInfo = MakeRasterizationState();
    const VkPipelineMultisampleStateCreateInfo msInfo = MakeMultisampleState(m_msaaSamples);
    const VkPipelineColorBlendAttachmentState blendAttach = MakeAlphaBlendAttachment();
    const VkPipelineColorBlendStateCreateInfo cbInfo = MakeColorBlendState(blendAttach);
    const VkPipelineDepthStencilStateCreateInfo dsInfo = MakeDisabledDepthStencilState();
    const std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynInfo = MakeDynamicStateInfo(dynStates.data(), dynStates.size());

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipeInfo.pStages = stages.data();
    pipeInfo.pVertexInputState = &vertInput;
    pipeInfo.pInputAssemblyState = &iaInfo;
    pipeInfo.pViewportState = &vpInfo;
    pipeInfo.pRasterizationState = &rsInfo;
    pipeInfo.pMultisampleState = &msInfo;
    pipeInfo.pDepthStencilState = &dsInfo;
    pipeInfo.pColorBlendState = &cbInfo;
    pipeInfo.pDynamicState = &dynInfo;
    pipeInfo.layout = m_pipelineLayout;
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &m_colorFormat;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.renderPass = VK_NULL_HANDLE;
    pipeInfo.subpass = 0;

    return vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_pipeline) == VK_SUCCESS;
}

bool InxScreenUIRenderer::CreateWorldPipeline()
{
    if (m_depthFormat == VK_FORMAT_UNDEFINED)
        return false;

    InxShaderLoader compiler(false, true, false, true, false, true, false, false, false, false);
    const std::vector<char> shaderBytes = compiler.CompileVertexGlsl(kWorldVertexShader, "Infernux/WorldUI.vert");
    const std::vector<char> fragmentBytes = compiler.CompileFragmentGlsl(kWorldFragmentShader, "Infernux/WorldUI.frag");
    if (shaderBytes.size() < sizeof(uint32_t) * 5 || shaderBytes.size() % sizeof(uint32_t) != 0)
        return false;
    if (fragmentBytes.size() < sizeof(uint32_t) * 5 || fragmentBytes.size() % sizeof(uint32_t) != 0)
        return false;
    if (!CreateShaderModule(m_device, reinterpret_cast<const uint32_t *>(shaderBytes.data()), shaderBytes.size(),
                            m_worldVertShader))
        return false;
    if (!CreateShaderModule(m_device, reinterpret_cast<const uint32_t *>(fragmentBytes.data()), fragmentBytes.size(),
                            m_worldFragShader))
        return false;

    VkPushConstantRange pushConstants = MakeVertexPushConstantRange(sizeof(WorldUIPushConstants));
    pushConstants.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    if (!CreatePipelineLayout(m_device, m_descriptorSetLayout, pushConstants, m_worldPipelineLayout))
        return false;

    rhi::GraphicsRenderingSignature target;
    target.colorFormatCount = 1;
    target.colorFormats[0] = rhi::FromVkFormat(m_colorFormat);
    target.depthFormat = rhi::FromVkFormat(m_depthFormat);
    target.samples = rhi::FromVkSampleCount(m_msaaSamples);
    return CreateWorldPipeline(target, m_worldPipeline) && CreateWorldPipeline(target, m_worldTopPipeline, true);
}

bool InxScreenUIRenderer::CreateWorldPipeline(const rhi::GraphicsRenderingSignature &target, VkPipeline &result,
                                              bool alwaysOnTop)
{

    const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
        MakeShaderStageInfo(VK_SHADER_STAGE_VERTEX_BIT, m_worldVertShader),
        MakeShaderStageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, m_worldFragShader),
    };
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.stride = sizeof(WorldGPUVertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attributes[7]{};
    attributes[0].location = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = offsetof(WorldGPUVertex, pos);
    attributes[1].location = 1;
    attributes[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[1].offset = offsetof(WorldGPUVertex, uv);
    attributes[2].location = 2;
    attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[2].offset = offsetof(WorldGPUVertex, color);
    attributes[3].location = 3;
    attributes[3].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[3].offset = offsetof(WorldGPUVertex, localPos);
    attributes[4].location = 4;
    attributes[4].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[4].offset = offsetof(WorldGPUVertex, anchor);
    attributes[5].location = 5;
    attributes[5].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[5].offset = offsetof(WorldGPUVertex, localOffset);
    attributes[6].location = 6;
    attributes[6].format = VK_FORMAT_R32_SFLOAT;
    attributes[6].offset = offsetof(WorldGPUVertex, policy);

    const VkPipelineVertexInputStateCreateInfo vertexInput = MakeVertexInputState(bindingDescription, attributes, 7);
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly = MakeTriangleListInputAssembly();
    const VkPipelineViewportStateCreateInfo viewport = MakeDynamicViewportState();
    const VkPipelineRasterizationStateCreateInfo rasterization = MakeRasterizationState();
    const VkPipelineMultisampleStateCreateInfo multisample = MakeMultisampleState(rhi::ToVkSampleCount(target.samples));
    const VkPipelineColorBlendAttachmentState blendAttachment = MakeAlphaBlendAttachment();
    const VkPipelineColorBlendStateCreateInfo blend = MakeColorBlendState(blendAttachment);
    VkPipelineDepthStencilStateCreateInfo depth = MakeWorldDepthStencilState();
    // Keep the same attachment and no-write contract; only the explicit
    // policy changes comparison so scene depth cannot reject this element.
    if (alwaysOnTop)
        depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    const std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic = MakeDynamicStateInfo(dynamicStates.data(), dynamicStates.size());

    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1;
    const VkFormat color = rhi::ToVkFormat(target.colorFormats[0]);
    rendering.pColorAttachmentFormats = &color;
    rendering.depthAttachmentFormat = rhi::ToVkFormat(target.depthFormat);
    rendering.stencilAttachmentFormat = rhi::ToVkFormat(target.stencilFormat);

    VkGraphicsPipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = &rendering;
    pipeline.stageCount = static_cast<uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &vertexInput;
    pipeline.pInputAssemblyState = &inputAssembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &rasterization;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = m_worldPipelineLayout;
    return vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &result) == VK_SUCCESS;
}

VkPipeline InxScreenUIRenderer::GetWorldPipeline(const rhi::GraphicsRenderingSignature &target, bool alwaysOnTop)
{
    if (target.colorFormatCount != 1 || target.depthFormat == rhi::PixelFormat::Undefined)
        throw std::invalid_argument("World UI needs one color and one depth attachment");
    if (target.colorFormats[0] == rhi::FromVkFormat(m_colorFormat) &&
        target.depthFormat == rhi::FromVkFormat(m_depthFormat) && target.stencilFormat == rhi::PixelFormat::Undefined &&
        target.samples == rhi::FromVkSampleCount(m_msaaSamples))
        return alwaysOnTop ? m_worldTopPipeline : m_worldPipeline;
    for (const auto &variant : m_worldPipelineVariants) {
        if (variant.target == target && variant.alwaysOnTop == alwaysOnTop)
            return variant.pipeline;
    }
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (!CreateWorldPipeline(target, pipeline, alwaysOnTop))
        throw std::runtime_error("Failed to create World UI pipeline for the actual target attachments");
    m_worldPipelineVariants.push_back({target, alwaysOnTop, pipeline});
    return pipeline;
}

// ============================================================================
// Buffer Management
// ============================================================================

bool InxScreenUIRenderer::EnsureBuffers(ListBuffers &buf, VkDeviceSize vertexSize, VkDeviceSize indexSize)
{
    return EnsureHostVisibleBuffer(m_allocator, m_deletionQueue, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, buf.vertexBuffer,
                                   buf.vertexAlloc, buf.vertexBufferSize, vertexSize) &&
           EnsureHostVisibleBuffer(m_allocator, m_deletionQueue, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, buf.indexBuffer,
                                   buf.indexAlloc, buf.indexBufferSize, indexSize);
}

// ============================================================================
// Helpers
// ============================================================================

ImDrawList *InxScreenUIRenderer::GetDrawList(ScreenUIList list)
{
    if (m_recordingPacket) {
        const int index = ListIndex(list);
        auto &output = m_recordingPacket->m_data->lists[index];
        auto *&drawList = m_packetDrawLists[index];
        if (!output.used) {
            if (!drawList)
                drawList = IM_NEW(ImDrawList)(ImGui::GetDrawListSharedData());
            ResetDrawListForFrame(*drawList, m_cachedWidth, m_cachedHeight, list != ScreenUIList::World);
            output.used = true;
        }
        return drawList;
    }
    if (list == ScreenUIList::Camera)
        return m_cameraDrawList;
    if (list == ScreenUIList::Overlay)
        return m_overlayDrawList;
    return m_worldDrawList;
}

bool InxScreenUIRenderer::HasSelectiveWorldOcclusion(uint32_t cullingMask) const
{
    if (!m_initialized || !m_hasSelectiveWorldOcclusion)
        return false;
    return std::any_of(m_worldElementSpans.begin(), m_worldElementSpans.end(), [cullingMask](const auto &element) {
        return element.ignoredOccluderId != 0 && !element.alwaysOnTop &&
               (element.layerMask & cullingMask) != 0;
    });
}

std::vector<InxScreenUIRenderer::WorldDepthRun>
InxScreenUIRenderer::GetWorldDepthRuns(const glm::mat4 &viewProjection, uint32_t cullingMask) const
{
    struct OrderedElement
    {
        size_t index;
        float depth;
    };
    std::vector<OrderedElement> order;
    order.reserve(m_worldElementSpans.size());
    for (size_t index = 0; index < m_worldElementSpans.size(); ++index) {
        const auto &element = m_worldElementSpans[index];
        if ((element.layerMask & cullingMask) == 0)
            continue;
        const glm::vec4 clipCenter = viewProjection * element.localToWorld[3];
        order.push_back({index, clipCenter.w > 0.0f ? clipCenter.z / clipCenter.w : -1.0f});
    }
    std::stable_sort(order.begin(), order.end(), [&](const OrderedElement &lhs, const OrderedElement &rhs) {
        const bool lhsTop = m_worldElementSpans[lhs.index].alwaysOnTop;
        const bool rhsTop = m_worldElementSpans[rhs.index].alwaysOnTop;
        return lhsTop != rhsTop ? !lhsTop : lhs.depth > rhs.depth;
    });
    std::vector<WorldUIOcclusionPolicy> policies;
    policies.reserve(order.size());
    for (const auto &entry : order) {
        const auto &element = m_worldElementSpans[entry.index];
        policies.push_back({element.ignoredOccluderId, element.alwaysOnTop});
    }
    return BuildWorldUIOcclusionPlan(policies).runs;
}

void InxScreenUIRenderer::RenderWorld(VkCommandBuffer cmdBuf, uint32_t width, uint32_t height,
                                      const glm::mat4 &viewProjection, const rhi::GraphicsRenderingSignature &target,
                                      uint32_t frameSlot, uint32_t cullingMask, const glm::mat4 &view,
                                      const glm::mat4 &projection, uint32_t firstOrdinal, uint32_t endOrdinal)
{
    constexpr ScreenUIList list = ScreenUIList::World;
    constexpr int listIndex = 2;
    if (firstOrdinal == 0) {
        m_lastSubmittedDrawCounts[listIndex] = 0;
        m_lastSubmittedIndexCounts[listIndex] = 0;
    }
    if (!m_initialized || !m_worldPipeline || width == 0 || height == 0 || !m_enabled)
        return;
    if (m_worldElementStart >= 0)
        throw std::logic_error("World UI element submission was not closed before rendering");

    ImDrawList *drawList = m_worldDrawList;
    if (!drawList || drawList->VtxBuffer.Size == 0 || drawList->IdxBuffer.Size == 0)
        return;
    struct ElementDepth
    {
        size_t index;
        float depth;
    };
    std::vector<ElementDepth> elementOrder;
    elementOrder.reserve(m_worldElementSpans.size());
    for (size_t index = 0; index < m_worldElementSpans.size(); ++index) {
        const auto &element = m_worldElementSpans[index];
        if ((element.layerMask & cullingMask) != 0) {
            const glm::vec4 clipCenter = viewProjection * element.localToWorld[3];
            elementOrder.push_back({index, clipCenter.w > 0.0f ? clipCenter.z / clipCenter.w : -1.0f});
        }
    }
    if (elementOrder.empty())
        return;
    if (!RefreshFontDescriptorSet(m_fontDescriptorSet))
        return;
    auto &vertices = m_worldVertices;
    if (m_preparedRevision[listIndex] != m_geometryRevision[listIndex]) {
        if (m_worldElementSpans.empty() || m_worldElementSpans.front().vertexStart != 0 ||
            m_worldElementSpans.back().vertexEnd != drawList->VtxBuffer.Size) {
            throw std::logic_error("World UI vertices must belong to one explicit element");
        }
        for (size_t index = 1; index < m_worldElementSpans.size(); ++index) {
            if (m_worldElementSpans[index - 1].vertexEnd != m_worldElementSpans[index].vertexStart)
                throw std::logic_error("World UI element spans must be contiguous");
        }
        vertices.resize(static_cast<size_t>(drawList->VtxBuffer.Size));
        const auto &hdrRanges = m_worldHDRRanges;
        size_t hdrIndex = 0;
        size_t elementIndex = 0;
        for (int vertexIndex = 0; vertexIndex < drawList->VtxBuffer.Size; ++vertexIndex) {
            while (hdrIndex < hdrRanges.size() && vertexIndex >= hdrRanges[hdrIndex].vertexEnd)
                ++hdrIndex;
            while (elementIndex + 1 < m_worldElementSpans.size() &&
                   vertexIndex >= m_worldElementSpans[elementIndex].vertexEnd)
                ++elementIndex;

            float rgbScale = 1.0f;
            if (hdrIndex < hdrRanges.size() && vertexIndex >= hdrRanges[hdrIndex].vertexStart)
                rgbScale = hdrRanges[hdrIndex].rgbScale;

            const ImDrawVert &source = drawList->VtxBuffer[vertexIndex];
            const WorldElementSpan &element = m_worldElementSpans[elementIndex];
            // UI +X follows Transform +X; UI +Y points down while the world +Y
            // axis points up. This keeps world UI manipulation identical to
            // ordinary scene objects without introducing a second pose.
            const glm::vec4 local((source.pos.x - element.pivotX) / kWorldUILogicalPixelsPerUnit,
                                  -(source.pos.y - element.pivotY) / kWorldUILogicalPixelsPerUnit, 0.0f, 1.0f);
            const glm::vec4 world = element.localToWorld * local;
            WorldGPUVertex &target = vertices[static_cast<size_t>(vertexIndex)];
            target.pos[0] = world.x;
            target.pos[1] = world.y;
            target.pos[2] = world.z;
            const glm::vec3 anchor = glm::vec3(element.localToWorld[3]);
            target.anchor[0] = anchor.x;
            target.anchor[1] = anchor.y;
            target.anchor[2] = anchor.z;
            target.localOffset = ImVec2(local.x, local.y);
            target.policy = float((element.billboard ? WorldUIBillboard : 0u) |
                                  (element.constantScreenSize ? WorldUIConstantScreenSize : 0u));
            target.uv = source.uv;
            target.localPos = source.pos;
            const ImVec4 color = ImGui::ColorConvertU32ToFloat4(source.col);
            target.color[0] = color.x * rgbScale;
            target.color[1] = color.y * rgbScale;
            target.color[2] = color.z * rgbScale;
            target.color[3] = color.w;
        }
        m_preparedRevision[listIndex] = m_geometryRevision[listIndex];
        ++m_geometryStats[listIndex].preparations;
    }

    const VkDeviceSize vertexBytes = vertices.size() * sizeof(WorldGPUVertex);
    ListBuffers &buffers = m_frameBuffers[frameSlot][listIndex];
    if (!UploadGeometry(buffers, list, vertices.data(), static_cast<size_t>(vertexBytes)))
        return;

    // Resolve both immutable pipeline variants before recording draw commands.
    const VkPipeline depthPipeline = GetWorldPipeline(target);
    const bool hasTopElement = std::any_of(elementOrder.begin(), elementOrder.end(), [&](const ElementDepth &entry) {
        return m_worldElementSpans[entry.index].alwaysOnTop;
    });
    const VkPipeline topPipeline = hasTopElement ? GetWorldPipeline(target, true) : VK_NULL_HANDLE;
    VkPipeline lastPipeline = VK_NULL_HANDLE;
    const VkDeviceSize vertexOffset = 0;
    vkCmdBindVertexBuffers(cmdBuf, 0, 1, &buffers.vertexBuffer, &vertexOffset);
    vkCmdBindIndexBuffer(cmdBuf, buffers.indexBuffer, 0,
                         sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    const VkViewport viewport = MakeViewport(width, height);
    vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
    const VkRect2D scissor{{0, 0}, {width, height}};
    vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
    VkDescriptorSet lastDescriptor = VK_NULL_HANDLE;
    uint32_t submittedDraws = 0;
    uint64_t submittedIndices = 0;

    // Every camera owns its own spatial transparency order. The command list
    // itself is camera-independent and may be replayed by several cameras, so
    // sorting at Python submission time would be incorrect. Sort elements
    // back-to-front in this camera's clip space, while preserving the authored
    // draw order inside each element.
    std::stable_sort(elementOrder.begin(), elementOrder.end(), [&](const ElementDepth &lhs, const ElementDepth &rhs) {
        const bool lhsTop = m_worldElementSpans[lhs.index].alwaysOnTop;
        const bool rhsTop = m_worldElementSpans[rhs.index].alwaysOnTop;
        return lhsTop != rhsTop ? !lhsTop : lhs.depth > rhs.depth;
    });

    // The camera matrix is shared by every draw, while authored material
    // values are command-local.  Push the complete block for each draw so
    // World UI consumes the same material contract as Screen UI.
    const bool hasBillboard = std::any_of(elementOrder.begin(), elementOrder.end(), [&](const ElementDepth &entry) {
        return m_worldElementSpans[entry.index].billboard;
    });
    const size_t first = std::min<size_t>(firstOrdinal, elementOrder.size());
    const size_t end = std::min<size_t>(endOrdinal, elementOrder.size());
    if (first >= end)
        return;
    elementOrder.erase(elementOrder.begin() + end, elementOrder.end());
    elementOrder.erase(elementOrder.begin(), elementOrder.begin() + first);
    const bool hasConstantSize = std::any_of(elementOrder.begin(), elementOrder.end(), [&](const ElementDepth &entry) {
        return m_worldElementSpans[entry.index].constantScreenSize;
    });
    const glm::mat4 cameraToWorld = hasBillboard ? glm::inverse(view) : glm::mat4(1.0f);
    const glm::vec4 cameraRight(glm::vec3(cameraToWorld[0]), 0.0f);
    const glm::vec4 cameraUp(glm::vec3(cameraToWorld[1]), 0.0f);
    const float screenPixelScale = hasConstantSize
                                       ? 2.0f / (std::max(std::abs(projection[1][1]), 1e-6f) * float(height))
                                       : 0.0f;
    const auto drawCommand = [&](const ImDrawCmd &command, int commandIndex, bool alwaysOnTop) {
        if (command.ElemCount == 0)
            return;
        VkDescriptorSet descriptor = reinterpret_cast<VkDescriptorSet>(static_cast<uintptr_t>(command.GetTexID()));
        if (descriptor == VK_NULL_HANDLE)
            return;
        const VkPipeline pipeline = alwaysOnTop ? topPipeline : depthPipeline;
        if (pipeline != lastPipeline) {
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            lastPipeline = pipeline;
        }
        if (descriptor != m_fontDescriptorSet && m_textureUsageValidator &&
            !m_textureUsageValidator(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(descriptor))))
            return;
        if (descriptor != lastDescriptor) {
            vkdebug::CmdBindDescriptorSetsTracked("InxScreenUIRenderer.RenderWorld.Set0", cmdBuf,
                                                  VK_PIPELINE_BIND_POINT_GRAPHICS, m_worldPipelineLayout, 0, 1,
                                                  &descriptor, 0, nullptr);
            lastDescriptor = descriptor;
        }
        const auto binding = (commandIndex >= 0 && static_cast<size_t>(commandIndex) <
                                                       m_commandBindings[ListIndex(ScreenUIList::World)].size())
                                 ? m_commandBindings[ListIndex(ScreenUIList::World)][static_cast<size_t>(commandIndex)]
                                 : UIShaderMaterialBinding{};
        WorldUIPushConstants constants{};
        constants.viewProjection = viewProjection;
        constants.materialColor =
            glm::vec4(binding.baseColor[0], binding.baseColor[1], binding.baseColor[2], binding.baseColor[3]);
        constants.alphaClipEnabled = binding.alphaClipEnabled ? 1.0f : 0.0f;
        constants.alphaClipThreshold = binding.alphaClipThreshold;
        constants.cameraRight = cameraRight;
        constants.cameraUp = cameraUp;
        constants.screenScale[0] = screenPixelScale;
        vkCmdPushConstants(cmdBuf, m_worldPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(constants), &constants);
        vkCmdDrawIndexed(cmdBuf, command.ElemCount, 1, command.IdxOffset, static_cast<int32_t>(command.VtxOffset), 0);
        ++submittedDraws;
        submittedIndices += command.ElemCount;
    };
    // Merge only consecutive index ranges after spatial sorting. Texture and
    // base vertex remain identical; primitive/blend order is preserved. World
    // UI ignores canvas clips, so ClipRect cannot be a batch boundary here.
    ImDrawCmd pending{};
    int pendingCommandIndex = -1;
    UIShaderMaterialBinding pendingBinding{};
    bool pendingAlwaysOnTop = false;
    for (const auto &entry : elementOrder) {
        const WorldElementSpan &element = m_worldElementSpans[entry.index];
        if (element.commandStart < 0 || element.commandEnd > drawList->CmdBuffer.Size)
            throw std::logic_error("World UI element command span is invalid");
        for (int commandIndex = element.commandStart; commandIndex < element.commandEnd; ++commandIndex) {
            const auto &command = drawList->CmdBuffer[commandIndex];
            if (command.UserCallback || !command.ElemCount)
                continue;
            const auto binding =
                static_cast<size_t>(commandIndex) < m_commandBindings[ListIndex(ScreenUIList::World)].size()
                    ? m_commandBindings[ListIndex(ScreenUIList::World)][static_cast<size_t>(commandIndex)]
                    : UIShaderMaterialBinding{};
            if (pending.ElemCount && pending.IdxOffset + pending.ElemCount == command.IdxOffset &&
                pending.VtxOffset == command.VtxOffset && pending.GetTexID() == command.GetTexID() &&
                pendingBinding == binding && pendingAlwaysOnTop == element.alwaysOnTop) {
                pending.ElemCount += command.ElemCount;
            } else {
                drawCommand(pending, pendingCommandIndex, pendingAlwaysOnTop);
                pending = command;
                pendingCommandIndex = commandIndex;
                pendingBinding = binding;
                pendingAlwaysOnTop = element.alwaysOnTop;
            }
        }
    }
    drawCommand(pending, pendingCommandIndex, pendingAlwaysOnTop);
    m_lastSubmittedDrawCounts[listIndex] += submittedDraws;
    m_lastSubmittedIndexCounts[listIndex] += submittedIndices;
}

std::vector<std::shared_ptr<rhi::RenderTexture>> InxScreenUIRenderer::GetRenderTextureReads(ScreenUIList list,
                                                                                            uint32_t cullingMask) const
{
    std::vector<std::shared_ptr<rhi::RenderTexture>> reads;
    const auto *drawList = GetDrawList(list);
    if (!m_enabled || !drawList || !m_renderTextureResolver)
        return reads;
    const auto add = [&](int index) {
        const auto &command = drawList->CmdBuffer[index];
        if (!command.ElemCount || command.UserCallback)
            return;
        auto texture = m_renderTextureResolver(static_cast<uint64_t>(command.GetTexID()));
        if (texture && std::find(reads.begin(), reads.end(), texture) == reads.end())
            reads.push_back(std::move(texture));
    };
    if (list == ScreenUIList::World) {
        for (const auto &element : m_worldElementSpans) {
            if ((element.layerMask & cullingMask) != 0) {
                for (int index = element.commandStart; index < element.commandEnd; ++index)
                    add(index);
            }
        }
    } else {
        for (int index = 0; index < drawList->CmdBuffer.Size; ++index)
            add(index);
    }
    return reads;
}

const ImDrawList *InxScreenUIRenderer::GetDrawList(ScreenUIList list) const
{
    if (list == ScreenUIList::Camera)
        return m_cameraDrawList;
    if (list == ScreenUIList::Overlay)
        return m_overlayDrawList;
    return m_worldDrawList;
}

} // namespace infernux
