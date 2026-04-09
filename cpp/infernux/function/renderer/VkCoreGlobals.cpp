/**
 * @file VkCoreGlobals.cpp
 * @brief InxVkCoreModular — Engine globals UBO (set 2, binding 0)
 *
 * Split from InxVkCoreModular.cpp for maintainability.
 * Contains: StageGlobals, CreateGlobalsBuffers,
 *           CreateGlobalsDescriptorResources, DestroyGlobalsDescriptorResources,
 *           CmdUpdateGlobals.
 */

#include "InxError.h"
#include "InxVkCoreModular.h"
#include "vk/VkRenderUtils.h"

#include <function/renderer/shader/ShaderProgram.h>

#include <cstring>

namespace infernux
{

// ============================================================================
// Public API
// ============================================================================

void InxVkCoreModular::StageGlobals(const EngineGlobalsUBO &globals)
{
    m_stagedGlobals = globals;
    m_globalsDirty = true;
}

// ============================================================================
// Buffer creation
// ============================================================================

void InxVkCoreModular::CreateGlobalsBuffers()
{
    constexpr VkDeviceSize bufferSize = sizeof(EngineGlobalsUBO);

    m_globalsBuffers.resize(m_maxFramesInFlight);
    for (size_t i = 0; i < m_maxFramesInFlight; ++i) {
        m_globalsBuffers[i] = m_resourceManager.CreateUniformBuffer(bufferSize);
    }

    INXLOG_INFO("Created engine globals UBO buffers: ", bufferSize, " bytes x ", m_maxFramesInFlight, " frames");
}

// ============================================================================
// Descriptor resources (layout + pool + sets)
// ============================================================================

bool InxVkCoreModular::CreateGlobalsDescriptorResources()
{
    VkDevice device = GetDevice();
    if (device == VK_NULL_HANDLE)
        return false;

    // Layout: set 2, binding 0 = uniform buffer (globals UBO), vertex + fragment
    //         set 2, binding 1 = storage buffer (instance models), vertex only
    VkDescriptorSetLayoutBinding bindings[2]{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[0].pImmutableSamplers = nullptr;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_globalsDescSetLayout) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create globals descriptor set layout");
        return false;
    }

    // Publish to ShaderProgram so all pipelines pick up the shared layout at set 2
    ShaderProgram::SetGlobalsDescSetLayout(m_globalsDescSetLayout);

    // Pool: one UBO + one SSBO per frame-in-flight
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(m_maxFramesInFlight);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(m_maxFramesInFlight);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.maxSets = static_cast<uint32_t>(m_maxFramesInFlight);
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_globalsDescPool) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create globals descriptor pool");
        vkDestroyDescriptorSetLayout(device, m_globalsDescSetLayout, nullptr);
        m_globalsDescSetLayout = VK_NULL_HANDLE;
        ShaderProgram::SetGlobalsDescSetLayout(VK_NULL_HANDLE);
        return false;
    }

    // Allocate descriptor sets
    std::vector<VkDescriptorSetLayout> layouts(m_maxFramesInFlight, m_globalsDescSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_globalsDescPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(m_maxFramesInFlight);
    allocInfo.pSetLayouts = layouts.data();

    m_globalsDescSets.resize(m_maxFramesInFlight);
    if (vkAllocateDescriptorSets(device, &allocInfo, m_globalsDescSets.data()) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to allocate globals descriptor sets");
        vkDestroyDescriptorPool(device, m_globalsDescPool, nullptr);
        m_globalsDescPool = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(device, m_globalsDescSetLayout, nullptr);
        m_globalsDescSetLayout = VK_NULL_HANDLE;
        ShaderProgram::SetGlobalsDescSetLayout(VK_NULL_HANDLE);
        return false;
    }

    // Write each descriptor set to point at the corresponding globals buffer
    // and a placeholder instance SSBO
    m_instanceBuffers.resize(m_maxFramesInFlight);
    for (size_t i = 0; i < m_maxFramesInFlight; ++i) {
        // Create initial instance buffer for this frame
        const VkDeviceSize initialBytes = INSTANCE_BUFFER_INITIAL_CAPACITY * sizeof(glm::mat4);
        m_instanceBuffers[i].buffer = m_resourceManager.CreateStorageBuffer(initialBytes, /*deviceLocal=*/false);
        m_instanceBuffers[i].capacity = INSTANCE_BUFFER_INITIAL_CAPACITY;

        if (!m_instanceBuffers[i].buffer) {
            INXLOG_ERROR("Failed to create instance buffer for frame ", i);
            continue;
        }

        m_instanceBuffers[i].mapped = m_instanceBuffers[i].buffer->Map();

        VkWriteDescriptorSet writes[2]{};

        // Binding 0: Globals UBO
        VkDescriptorBufferInfo uboBufInfo{};
        if (m_globalsBuffers[i]) {
            uboBufInfo.buffer = m_globalsBuffers[i]->GetBuffer();
            uboBufInfo.offset = 0;
            uboBufInfo.range = sizeof(EngineGlobalsUBO);
        }

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_globalsDescSets[i];
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &uboBufInfo;

        // Binding 1: Instance SSBO
        VkDescriptorBufferInfo ssboBufInfo{};
        ssboBufInfo.buffer = m_instanceBuffers[i].buffer->GetBuffer();
        ssboBufInfo.offset = 0;
        ssboBufInfo.range = VK_WHOLE_SIZE;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_globalsDescSets[i];
        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &ssboBufInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }

    INXLOG_INFO("Created globals descriptor set layout, pool, and ", m_maxFramesInFlight,
                " sets (set 2) with instance SSBO");
    return true;
}

void InxVkCoreModular::DestroyGlobalsDescriptorResources()
{
    VkDevice device = GetDevice();
    if (device == VK_NULL_HANDLE)
        return;

    m_instanceBuffers.clear();
    m_globalsDescSets.clear();

    if (m_globalsDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_globalsDescPool, nullptr);
        m_globalsDescPool = VK_NULL_HANDLE;
    }
    if (m_globalsDescSetLayout != VK_NULL_HANDLE) {
        ShaderProgram::SetGlobalsDescSetLayout(VK_NULL_HANDLE);
        vkDestroyDescriptorSetLayout(device, m_globalsDescSetLayout, nullptr);
        m_globalsDescSetLayout = VK_NULL_HANDLE;
    }
}

// ============================================================================
// Instance buffer management
// ============================================================================

void InxVkCoreModular::EnsureInstanceBufferCapacity(uint32_t frameIndex, size_t instanceCount)
{
    if (frameIndex >= m_instanceBuffers.size())
        return;

    auto &frame = m_instanceBuffers[frameIndex];
    if (frame.capacity >= instanceCount && frame.buffer) {
        if (!frame.mapped) {
            frame.mapped = frame.buffer->Map();
        }
        return;
    }

    std::unique_ptr<vk::VkBufferHandle> oldBuffer = std::move(frame.buffer);
    void *oldMapped = frame.mapped;
    frame.mapped = nullptr;

    // Grow to next power-of-two that fits
    size_t newCapacity = frame.capacity > 0 ? frame.capacity : INSTANCE_BUFFER_INITIAL_CAPACITY;
    while (newCapacity < instanceCount)
        newCapacity *= 2;

    const VkDeviceSize newBytes = newCapacity * sizeof(glm::mat4);
    auto newBuffer = m_resourceManager.CreateStorageBuffer(newBytes, /*deviceLocal=*/false);
    if (!newBuffer) {
        INXLOG_ERROR("Failed to grow instance buffer to ", newCapacity, " instances (", newBytes, " bytes)");
        frame.buffer = std::move(oldBuffer);
        frame.mapped = oldMapped;
        return;
    }

    void *newMapped = newBuffer->Map();

    // Preserve existing data written earlier in this frame
    if (oldMapped && newMapped && m_instanceWriteOffset > 0) {
            std::memcpy(newMapped, oldMapped, m_instanceWriteOffset * sizeof(glm::mat4));
    }

    frame.buffer = std::move(newBuffer);
    frame.capacity = newCapacity;
    frame.mapped = newMapped;

    // NOTE: Do NOT call UpdateInstanceBufferDescriptor() here.
    // This function can be called mid-recording, and updating a descriptor set
    // that is already bound in the command buffer invalidates it.
    // The descriptor is updated once before any draws in PreallocateInstances().

    // Old instance buffers may still be referenced by commands already
    // recorded earlier in this same command buffer. Defer final destruction
    // until the frame deletion queue says all in-flight use is complete.
    if (oldBuffer) {
        auto retiredBuffer = std::shared_ptr<vk::VkBufferHandle>(oldBuffer.release());
        m_deletionQueue.Push([retiredBuffer]() mutable { retiredBuffer.reset(); });
    }
}

void InxVkCoreModular::PreallocateInstances(size_t totalDrawCalls)
{
    if (totalDrawCalls == 0)
        return;

    const uint32_t frameIndex = m_currentFrame % m_maxFramesInFlight;

    // Reset per-frame write offset on new frame (mirrors logic in RecordGroupedFilteredDrawCalls).
    if (m_lastInstanceFrame != m_currentFrame) {
        m_instanceWriteOffset = 0;
        m_lastInstanceFrame = m_currentFrame;
    }

    // Upper bound: every draw call can appear once in an opaque pass and
    // once per shadow cascade.  Pre-allocating here guarantees the buffer
    // never grows mid-recording, avoiding descriptor-set-update-while-bound.
    const size_t maxInstances = totalDrawCalls * (1 + NUM_SHADOW_CASCADES);
    EnsureInstanceBufferCapacity(frameIndex, maxInstances);

    // Safe to update the descriptor now — no draws have been recorded yet.
    UpdateInstanceBufferDescriptor(frameIndex);
}

bool InxVkCoreModular::WriteInstanceMatrix(uint32_t frameIndex, uint32_t instanceIndex, const glm::mat4 &matrix)
{
    if (frameIndex >= m_instanceBuffers.size())
        return false;

    const size_t requiredCount = static_cast<size_t>(instanceIndex) + 1;
    const VkBuffer previousBuffer =
        m_instanceBuffers[frameIndex].buffer ? m_instanceBuffers[frameIndex].buffer->GetBuffer() : VK_NULL_HANDLE;

    EnsureInstanceBufferCapacity(frameIndex, requiredCount);

    auto &frame = m_instanceBuffers[frameIndex];
    if (!frame.buffer)
        return false;

    if (previousBuffer != frame.buffer->GetBuffer())
        UpdateInstanceBufferDescriptor(frameIndex);

    void *mapped = frame.mapped;
    if (!mapped) {
        mapped = frame.buffer->Map();
        frame.mapped = mapped;
    }
    if (!mapped)
        return false;

    auto *matrices = static_cast<glm::mat4 *>(mapped);
    matrices[instanceIndex] = matrix;
    return true;
}

void InxVkCoreModular::UpdateInstanceBufferDescriptor(uint32_t frameIndex)
{
    VkDevice device = GetDevice();
    if (device == VK_NULL_HANDLE || frameIndex >= m_globalsDescSets.size() || frameIndex >= m_instanceBuffers.size())
        return;

    const auto &frame = m_instanceBuffers[frameIndex];
    if (!frame.buffer)
        return;

    VkDescriptorBufferInfo ssboBufInfo{};
    ssboBufInfo.buffer = frame.buffer->GetBuffer();
    ssboBufInfo.offset = 0;
    ssboBufInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_globalsDescSets[frameIndex];
    write.dstBinding = 1;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &ssboBufInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

// ============================================================================
// Command-buffer inline update
// ============================================================================

void InxVkCoreModular::CmdUpdateGlobals(VkCommandBuffer cmdBuf)
{
    if (!m_globalsDirty)
        return;

    // Update ALL per-frame globals buffers so every frame-in-flight sees the
    // latest values.  Each m_globalsDescSets[i] points at m_globalsBuffers[i],
    // so we must write to every buffer — not just buffer 0.
    for (size_t i = 0; i < m_globalsBuffers.size(); ++i) {
        if (!m_globalsBuffers[i])
            continue;

        VkBuffer buffer = m_globalsBuffers[i]->GetBuffer();

        // Barrier: ensure previous shader reads from the globals UBO are complete
        vkrender::CmdBarrierUniformReadToTransferWrite(cmdBuf);

        // Update the globals UBO inline in the command buffer
        // vkCmdUpdateBuffer has a 65536-byte limit; EngineGlobalsUBO is 128 bytes.
        vkCmdUpdateBuffer(cmdBuf, buffer, 0, sizeof(EngineGlobalsUBO), &m_stagedGlobals);

        // Barrier: ensure write is visible before subsequent shader reads
        vkrender::CmdBarrierTransferWriteToUniformRead(cmdBuf);
    }

    m_globalsDirty = false;
}

} // namespace infernux
