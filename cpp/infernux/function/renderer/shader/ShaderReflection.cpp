#include "ShaderReflection.h"

#include <algorithm>

namespace infernux
{

std::vector<VkDescriptorSetLayoutBinding> ShaderReflection::GetDescriptorSetLayoutBindings(uint32_t set) const
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    // Add uniform buffer bindings
    for (const auto &ubo : m_uniformBuffers) {
        if (ubo.set == set) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = ubo.binding;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = ubo.stageFlags;
            binding.pImmutableSamplers = nullptr;
            bindings.push_back(binding);
        }
    }

    // Add sampled image bindings
    for (const auto &image : m_sampledImages) {
        if (image.set == set) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = image.binding;
            binding.descriptorType = image.descriptorType;
            binding.descriptorCount = image.arraySize;
            binding.stageFlags = image.stageFlags;
            binding.pImmutableSamplers = nullptr;
            bindings.push_back(binding);
        }
    }

    for (const auto &buffer : m_storageBuffers) {
        if (buffer.set == set) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = buffer.binding;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = buffer.arraySize;
            binding.stageFlags = buffer.stageFlags;
            bindings.push_back(binding);
        }
    }

    for (const auto &image : m_storageImages) {
        if (image.set == set) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = image.binding;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            binding.descriptorCount = image.arraySize;
            binding.stageFlags = image.stageFlags;
            bindings.push_back(binding);
        }
    }

    std::sort(bindings.begin(), bindings.end(),
              [](const auto &left, const auto &right) { return left.binding < right.binding; });

    return bindings;
}

std::vector<uint32_t> ShaderReflection::GetUsedDescriptorSets() const
{
    std::vector<uint32_t> sets;

    for (const auto &ubo : m_uniformBuffers) {
        if (std::find(sets.begin(), sets.end(), ubo.set) == sets.end()) {
            sets.push_back(ubo.set);
        }
    }

    for (const auto &image : m_sampledImages) {
        if (std::find(sets.begin(), sets.end(), image.set) == sets.end()) {
            sets.push_back(image.set);
        }
    }

    for (const auto &buffer : m_storageBuffers) {
        if (std::find(sets.begin(), sets.end(), buffer.set) == sets.end())
            sets.push_back(buffer.set);
    }

    for (const auto &image : m_storageImages) {
        if (std::find(sets.begin(), sets.end(), image.set) == sets.end())
            sets.push_back(image.set);
    }

    std::sort(sets.begin(), sets.end());
    return sets;
}

} // namespace infernux
