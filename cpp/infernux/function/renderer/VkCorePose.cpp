#include "InxVkCoreModular.h"

namespace infernux
{

void InxVkCoreModular::CapturePoseRead(const std::shared_ptr<const RenderPoseValue> &value)
{
    if (!m_framePoseReads.insert(value.get()).second)
        return;
    if (value->models->GetHost().device.GetDeviceId() != m_backend.Device().GetDeviceId())
        throw std::invalid_argument("Renderer pose source belongs to another device");
    for (const auto &buffer : {value->models, value->bounds}) {
        RegisterFrameBufferReadDependency(buffer->GetLastWriteSubmission(), rhi::PipelineStage::VertexShader);
        m_frameSubmission.Retain(buffer);
    }
}

void InxVkCoreModular::RetirePoseBindings()
{
    auto &descriptors = m_backend.Device().GetRhiDevice().GetDescriptorManager();
    for (const auto &binding : m_poseGlobalsBindings)
        descriptors.Retire(binding.lease);
    m_poseGlobalsBindings.clear();
    m_framePoseReads.clear();
}

VkDescriptorSet InxVkCoreModular::ResolvePoseGlobalsBinding(VkDescriptorSet base, const RenderPoseBinding &current,
                                                            const RenderPoseBinding &previous)
{
    if (!current.value)
        return base;
    const auto &history = previous.value ? previous : current;
    for (const auto &entry : m_poseGlobalsBindings)
        if (entry.base == base && entry.current == current.value.get() && entry.previous == history.value.get())
            return entry.lease.set;

    CapturePoseRead(current.value);
    CapturePoseRead(history.value);
    auto &device = m_backend.Device().GetRhiDevice();
    auto &descriptors = device.GetDescriptorManager();
    const auto lease = descriptors.Allocate(m_globalsDescSetLayout, vk::DescriptorArena::FrameTransient);
    if (!lease.IsValid())
        throw std::runtime_error("Failed to allocate resident geometry binding");

    // Keep camera time, skinning and per-draw identity from the pass's existing
    // stream. Models/bounds use whole allocations plus shader-side indices, so
    // a record offset need not satisfy a device's descriptor offset alignment.
    std::array<VkCopyDescriptorSet, 4> copies{};
    constexpr std::array<uint32_t, 4> retainedBindings{0, 2, 3, 4};
    for (size_t index = 0; index < copies.size(); ++index) {
        auto &copy = copies[index];
        copy.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
        copy.srcSet = base;
        copy.srcBinding = retainedBindings[index];
        copy.dstSet = lease.set;
        copy.dstBinding = retainedBindings[index];
        copy.descriptorCount = 1;
    }
    const std::array<std::shared_ptr<rhi::ComputeBuffer>, 3> buffers{current.value->models, history.value->models,
                                                                     current.value->bounds};
    constexpr std::array<uint32_t, 3> bindings{1, 5, 6};
    std::array<VkDescriptorBufferInfo, 3> infos{};
    std::array<VkWriteDescriptorSet, 3> writes{};
    vk::SubmissionReads reads;
    for (size_t index = 0; index < buffers.size(); ++index) {
        infos[index] = {device.Resolve(buffers[index]->GetBuffer()), 0, VK_WHOLE_SIZE};
        auto &write = writes[index];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = lease.set;
        write.dstBinding = bindings[index];
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &infos[index];
        reads.Add(buffers[index]->GetLastWriteSubmission(), VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
    }
    vkUpdateDescriptorSets(GetDevice(), static_cast<uint32_t>(writes.size()), writes.data(),
                           static_cast<uint32_t>(copies.size()), copies.data());
    descriptors.SetUploadSources(lease, reads);
    m_poseGlobalsBindings.push_back({base, current.value.get(), history.value.get(), lease});
    return lease.set;
}

} // namespace infernux
