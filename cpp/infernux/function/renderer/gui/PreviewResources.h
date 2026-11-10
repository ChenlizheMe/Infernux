#pragma once

#include <array>
#include <core/threading/FrameWorkOwner.h>
#include <function/renderer/InxVkCoreModular.h>
#include <function/renderer/vk/VkResourceManager.h>
#include <stdexcept>
#include <utility>

namespace infernux
{

inline FrameWorkOwner *ActivePreviewOwner(InxVkCoreModular &core)
{
    auto *owner = core.GetFrameWorkOwner();
    return owner && owner->IsActive() ? owner : nullptr;
}

inline VkDeviceSize PreviewUploadBytes(FrameWorkOwner *owner, VkDeviceSize remaining, VkDeviceSize unit)
{
    auto bytes = (std::min)(VkDeviceSize{64 * 1024}, remaining);
    if (!owner)
        return bytes;
    bytes = (std::min)(bytes, owner->ResourceRemaining(FrameWorkResource::Transfer));
    bytes -= bytes % unit;
    if (bytes)
        return bytes;
    return owner->CanAdmitResource(FrameWorkResource::Transfer, unit, true) ? unit : 0;
}

// Static engine geometry uses the same retained upload as imported meshes.
inline bool PreparePreviewBuffer(InxVkCoreModular &core, std::shared_ptr<vk::BufferUploadTicket> &ticket,
                                 const void *data, VkDeviceSize bytes, rhi::BufferUsageFlags usage, bool &progressed)
{
    auto &resources = core.GetResourceManager();
    if (!ticket) {
        ticket = resources.CreateBufferUpload(bytes, usage);
        progressed = true;
    }
    const bool published = ticket->IsPublished();
    if (resources.TryPublishBufferUpload(ticket)) {
        progressed |= !published;
        return true;
    }
    if (ticket->IsPending())
        return false;
    auto *owner = ActivePreviewOwner(core);
    const auto count = PreviewUploadBytes(owner, bytes - ticket->GetSubmittedBytes(), sizeof(float));
    if (count)
        progressed |=
            resources.TryAppendBufferUpload(ticket, static_cast<const std::byte *>(data) + ticket->GetSubmittedBytes(),
                                            count, owner, count == sizeof(float), sizeof(float));
    return false;
}

class PreviewCommands
{
  public:
    PreviewCommands(InxVkCoreModular &core, VkDeviceSize transferBytes, uint64_t gpuNanoseconds)
        : m_manager(core.GetResourceManager())
    {
        m_work.lastExecutionNanoseconds = gpuNanoseconds;
        if (auto *owner = ActivePreviewOwner(core)) {
            m_commands = m_manager.TryBeginManagedGraphics(m_work, *owner, transferBytes);
            return;
        }
        m_commands = m_manager.BeginSingleTimeCommands();
        if (m_commands == VK_NULL_HANDLE)
            throw std::runtime_error("Failed to begin preview commands");
    }
    ~PreviewCommands()
    {
        if (m_commands != VK_NULL_HANDLE)
            m_manager.DiscardSingleTimeCommands(m_commands);
    }
    PreviewCommands(const PreviewCommands &) = delete;
    PreviewCommands &operator=(const PreviewCommands &) = delete;
    [[nodiscard]] VkCommandBuffer Get() const noexcept
    {
        return m_commands;
    }
    void ReadGeometry(const vk::VkBufferHandle &vertices, const vk::VkBufferHandle &indices)
    {
        ReadUpload(vertices.GetUploadSubmission(), VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
        ReadUpload(indices.GetUploadSubmission(), VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    }
    void ReadMaterial(const MaterialDescriptorSet &descriptor)
    {
        for (const auto &[binding, texture] : descriptor.textureBindings) {
            if (texture.gpuView)
                ReadUpload(texture.gpuView->GetUploadSubmission(),
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }
    }
    void AddDependency(vk::SubmissionWait wait)
    {
        if (wait.timeline != VK_NULL_HANDLE)
            m_dependencies.push_back(wait);
    }
    std::shared_ptr<vk::GraphicsSubmissionTicket> Submit(std::function<void()> releaseResources)
    {
        return m_manager.EndSingleTimeCommandsAsync(std::exchange(m_commands, VK_NULL_HANDLE),
                                                    std::move(releaseResources), std::move(m_work), m_dependencies);
    }

  private:
    void ReadUpload(rhi::SubmissionTicket upload, VkPipelineStageFlags stages)
    {
        AddDependency(m_manager.UploadReadDependency(upload, rhi::QueueRole::Graphics, stages));
    }

    vk::VkResourceManager &m_manager;
    VkCommandBuffer m_commands = VK_NULL_HANDLE;
    vk::GpuWorkReceipt m_work;
    std::vector<vk::SubmissionWait> m_dependencies;
};

inline void RetirePreviewAttachments(InxVkCoreModular &core, vk::VkImageHandle &color, vk::VkImageHandle &resolve,
                                     vk::VkImageHandle &depth)
{
    if (!color.IsValid() && !resolve.IsValid() && !depth.IsValid())
        return;
    auto retired = std::make_shared<std::array<vk::VkImageHandle, 3>>();
    (*retired)[0] = std::move(color);
    (*retired)[1] = std::move(resolve);
    (*retired)[2] = std::move(depth);
    core.RetireGpuResource([retired = std::move(retired)] {});
}

} // namespace infernux
