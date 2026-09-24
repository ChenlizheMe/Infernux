#include "MaterialDescriptor.h"
#include <algorithm>
#include <core/log/InxLog.h>
#include <core/types/ColorSpace.h>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infernux
{

namespace
{

void AppendBufferWrite(std::vector<VkWriteDescriptorSet> &writes, std::vector<VkDescriptorBufferInfo> &bufferInfos,
                       VkDescriptorSet dstSet, uint32_t binding, VkDescriptorType descriptorType,
                       const VkDescriptorBufferInfo &bufferInfo, uint32_t descriptorCount = 1)
{
    bufferInfos.push_back(bufferInfo);

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = dstSet;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = descriptorCount;
    write.descriptorType = descriptorType;
    write.pBufferInfo = &bufferInfos.back();
    writes.push_back(write);
}

void AppendImageWrite(std::vector<VkWriteDescriptorSet> &writes, std::vector<VkDescriptorImageInfo> &imageInfos,
                      VkDescriptorSet dstSet, uint32_t binding, VkImageView imageView, VkSampler sampler,
                      VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
{
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = imageLayout;
    imageInfo.imageView = imageView;
    imageInfo.sampler = sampler;
    imageInfos.push_back(imageInfo);

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = dstSet;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfos.back();
    writes.push_back(write);
}

bool HasSameGpuBinding(const MaterialDescriptorSet::TextureBinding &left,
                       const MaterialDescriptorSet::TextureBinding &right)
{
    return left.imageView == right.imageView && left.sampler == right.sampler &&
           left.resourceIndex == right.resourceIndex && left.resolvedExplicitTexture == right.resolvedExplicitTexture;
}

} // namespace

// ============================================================================
// MaterialUBO Implementation
// ============================================================================

MaterialUBO::~MaterialUBO()
{
    Destroy();
}

bool MaterialUBO::Create(VmaAllocator allocator, VkDevice device, const MaterialUBOLayout &layout)
{
    m_allocator = allocator;
    m_device = device;
    m_layout = layout;
    m_size = layout.size;

    if (m_size == 0) {
        INXLOG_WARN("Creating MaterialUBO with size 0");
        return true;
    }

    // Create buffer via VMA
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = m_size;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VmaAllocationInfo allocInfo{};
    VkResult result = vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &m_buffer, &m_allocation, &allocInfo);
    if (result != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create material UBO buffer via VMA");
        return false;
    }

    m_mappedData = allocInfo.pMappedData;

    // Zero-initialize
    std::memset(m_mappedData, 0, m_size);

    return true;
}

void MaterialUBO::Destroy()
{
    if (m_buffer != VK_NULL_HANDLE && m_allocator != VK_NULL_HANDLE) {
        m_mappedData = nullptr;
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        m_buffer = VK_NULL_HANDLE;
        m_allocation = VK_NULL_HANDLE;
    }

    m_allocator = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}

void MaterialUBO::Update(const InxMaterial &material)
{
    if (!m_mappedData || m_size == 0) {
        return;
    }

    const auto &properties = material.GetAllProperties();

    for (const auto &[name, prop] : properties) {
        uint32_t offset, size;
        if (!m_layout.GetMemberInfo(name, offset, size)) {
            continue; // Property not in UBO layout
        }

        switch (prop.type) {
        case MaterialPropertyType::Float: {
            float value = std::get<float>(prop.value);
            WriteData(offset, &value, sizeof(float));
            break;
        }
        case MaterialPropertyType::Float2: {
            glm::vec2 value = std::get<glm::vec2>(prop.value);
            WriteData(offset, &value, sizeof(glm::vec2));
            break;
        }
        case MaterialPropertyType::Float3: {
            glm::vec3 value = std::get<glm::vec3>(prop.value);
            WriteData(offset, &value, sizeof(glm::vec3));
            break;
        }
        case MaterialPropertyType::Float4: {
            glm::vec4 value = std::get<glm::vec4>(prop.value);
            WriteData(offset, &value, sizeof(glm::vec4));
            break;
        }
        case MaterialPropertyType::Color: {
            // Authored colors are sRGB; shading runs in linear space.
            glm::vec4 value = inx::color::SrgbToLinear(std::get<glm::vec4>(prop.value));
            WriteData(offset, &value, sizeof(glm::vec4));
            break;
        }
        case MaterialPropertyType::Int: {
            int value = std::get<int>(prop.value);
            WriteData(offset, &value, sizeof(int));
            break;
        }
        case MaterialPropertyType::Mat4: {
            glm::mat4 value = std::get<glm::mat4>(prop.value);
            WriteData(offset, &value, sizeof(glm::mat4));
            break;
        }
        case MaterialPropertyType::Texture2D:
            // Textures are bound separately, not in UBO
            break;
        case MaterialPropertyType::FloatArray:
            SetFloatArray(name, std::get<std::vector<float>>(prop.value));
            break;
        case MaterialPropertyType::Float4Array:
            SetVec4Array(name, std::get<std::vector<glm::vec4>>(prop.value));
            break;
        }
    }
}

void MaterialUBO::Apply(const RendererParameterBlock &parameters)
{
    for (const auto &[name, property] : parameters.properties) {
        switch (property.type) {
        case MaterialPropertyType::Float:
            SetFloat(name, std::get<float>(property.value));
            break;
        case MaterialPropertyType::Float2:
            SetVec2(name, std::get<glm::vec2>(property.value));
            break;
        case MaterialPropertyType::Float3:
            SetVec3(name, std::get<glm::vec3>(property.value));
            break;
        case MaterialPropertyType::Float4:
            SetVec4(name, std::get<glm::vec4>(property.value));
            break;
        case MaterialPropertyType::Color:
            SetVec4(name, inx::color::SrgbToLinear(std::get<glm::vec4>(property.value)));
            break;
        case MaterialPropertyType::Int:
            SetInt(name, std::get<int>(property.value));
            break;
        case MaterialPropertyType::Mat4:
            SetMat4(name, std::get<glm::mat4>(property.value));
            break;
        case MaterialPropertyType::Texture2D:
            break;
        case MaterialPropertyType::FloatArray:
            SetFloatArray(name, std::get<std::vector<float>>(property.value));
            break;
        case MaterialPropertyType::Float4Array:
            SetVec4Array(name, std::get<std::vector<glm::vec4>>(property.value));
            break;
        }
    }
}

void MaterialUBO::UpdateTextureIndices(const std::array<uint32_t, ShaderProgram::MaterialTextureIndexCapacity> &indices)
{
    if (!m_mappedData || m_size == 0)
        return;
    const size_t memberCount = (std::min)(m_layout.members.size(), indices.size());
    for (size_t slot = 0; slot < memberCount; ++slot) {
        const UniformMember &member = m_layout.members[slot];
        WriteData(member.offset, &indices[slot], (std::min)(member.size, static_cast<uint32_t>(sizeof(indices[slot]))));
    }
}

void MaterialUBO::UpdateTextureIndex(std::string_view name, uint32_t index)
{
    uint32_t offset = 0;
    uint32_t size = 0;
    if (!m_layout.GetMemberInfo(std::string(name), offset, size))
        return;
    WriteData(offset, &index, (std::min)(size, static_cast<uint32_t>(sizeof(index))));
}

void MaterialUBO::WriteData(uint32_t offset, const void *data, uint32_t size)
{
    if (!m_mappedData || offset + size > m_size) {
        INXLOG_WARN("MaterialUBO write out of bounds: offset=", offset, " size=", size, " bufferSize=", m_size);
        return;
    }

    std::memcpy(static_cast<char *>(m_mappedData) + offset, data, size);
}

void MaterialUBO::SetFloat(const std::string &name, float value)
{
    uint32_t offset, size;
    if (m_layout.GetMemberInfo(name, offset, size)) {
        WriteData(offset, &value, sizeof(float));
    }
}

void MaterialUBO::SetVec2(const std::string &name, const glm::vec2 &value)
{
    uint32_t offset, size;
    if (m_layout.GetMemberInfo(name, offset, size)) {
        WriteData(offset, &value, sizeof(glm::vec2));
    }
}

void MaterialUBO::SetVec3(const std::string &name, const glm::vec3 &value)
{
    uint32_t offset, size;
    if (m_layout.GetMemberInfo(name, offset, size)) {
        WriteData(offset, &value, sizeof(glm::vec3));
    }
}

void MaterialUBO::SetVec4(const std::string &name, const glm::vec4 &value)
{
    uint32_t offset, size;
    if (m_layout.GetMemberInfo(name, offset, size)) {
        WriteData(offset, &value, sizeof(glm::vec4));
    }
}

void MaterialUBO::SetInt(const std::string &name, int value)
{
    uint32_t offset, size;
    if (m_layout.GetMemberInfo(name, offset, size)) {
        WriteData(offset, &value, sizeof(int));
    }
}

void MaterialUBO::SetMat4(const std::string &name, const glm::mat4 &value)
{
    uint32_t offset, size;
    if (m_layout.GetMemberInfo(name, offset, size)) {
        WriteData(offset, &value, sizeof(glm::mat4));
    }
}

void MaterialUBO::SetFloatArray(const std::string &name, const std::vector<float> &values)
{
    const auto member = std::find_if(m_layout.members.begin(), m_layout.members.end(),
                                     [&](const UniformMember &candidate) { return candidate.name == name; });
    if (member == m_layout.members.end() || member->arraySize != values.size() || member->size < values.size() * 16u)
        return;
    for (size_t index = 0; index < values.size(); ++index)
        WriteData(member->offset + static_cast<uint32_t>(index * 16u), &values[index], sizeof(float));
}

void MaterialUBO::SetVec4Array(const std::string &name, const std::vector<glm::vec4> &values)
{
    const auto member = std::find_if(m_layout.members.begin(), m_layout.members.end(),
                                     [&](const UniformMember &candidate) { return candidate.name == name; });
    if (member == m_layout.members.end() || member->arraySize != values.size() || member->size < values.size() * 16u)
        return;
    if (!values.empty())
        WriteData(member->offset, values.data(), static_cast<uint32_t>(values.size() * sizeof(glm::vec4)));
}

// ============================================================================
// MaterialDescriptorManager Implementation
// ============================================================================

MaterialDescriptorManager::~MaterialDescriptorManager()
{
    Shutdown();
}

bool MaterialDescriptorManager::IsDescriptorSetComplete(VkDescriptorSet descriptorSet) const
{
    if (descriptorSet == VK_NULL_HANDLE)
        return false;
    const auto found = m_liveDescriptorHandles.find(reinterpret_cast<uint64_t>(descriptorSet));
    const MaterialDescriptorSet *descriptor = found == m_liveDescriptorHandles.end() ? nullptr : found->second;
    return descriptor && descriptor->isValid && !descriptor->hasUnboundRequiredBuffers;
}

void MaterialDescriptorManager::Initialize(VmaAllocator allocator, VkDevice device, VkPhysicalDevice physicalDevice,
                                           vk::VkDescriptorManager *descriptorManager)
{
    m_vmaAllocator = allocator;
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_descriptorManager = descriptorManager;
    if (!m_descriptorManager)
        INXLOG_ERROR("MaterialDescriptorManager requires the Vulkan descriptor manager");
}

void MaterialDescriptorManager::Shutdown()
{
    Clear();

    // Default bindings participate in the same revisioned texture publication
    // ownership as material bindings. Release them explicitly while the RHI
    // device is still alive; relying on member destruction would keep the
    // TextureResource alive until after InxVkCoreModular destroys its device.
    m_defaultGpuView.reset();
    m_defaultNormalGpuView.reset();
    m_defaultImageView = VK_NULL_HANDLE;
    m_defaultSampler = VK_NULL_HANDLE;
    m_defaultNormalImageView = VK_NULL_HANDLE;
    m_defaultNormalSampler = VK_NULL_HANDLE;

    if (m_descriptorManager)
        (void)m_descriptorManager->Collect((std::numeric_limits<rhi::SubmissionSerial>::max)());

    m_device = VK_NULL_HANDLE;
    m_physicalDevice = VK_NULL_HANDLE;
    m_descriptorManager = nullptr;
    m_liveDescriptorHandles.clear();
    m_bufferResolver = {};
}

bool MaterialDescriptorManager::IsPlaceholderTexturePath(std::string_view texturePath) const
{
    return texturePath == "white" || texturePath == "black" || texturePath == "normal";
}

bool MaterialDescriptorManager::IsNormalBindingName(std::string_view bindingName) const
{
    return bindingName.find("normal") != std::string_view::npos || bindingName.find("Normal") != std::string_view::npos;
}

bool MaterialDescriptorManager::TryGetDefaultTextureBinding(std::string_view bindingName,
                                                            MaterialDescriptorSet::TextureBinding &outBinding) const
{
    if (IsNormalBindingName(bindingName) && m_defaultNormalImageView != VK_NULL_HANDLE &&
        m_defaultNormalSampler != VK_NULL_HANDLE) {
        outBinding = {m_defaultNormalImageView, m_defaultNormalSampler, {}, m_defaultNormalGpuView};
        return ResolveBindlessIndex(outBinding);
    }

    if (m_defaultImageView != VK_NULL_HANDLE && m_defaultSampler != VK_NULL_HANDLE) {
        outBinding = {m_defaultImageView, m_defaultSampler, {}, m_defaultGpuView};
        return ResolveBindlessIndex(outBinding);
    }

    outBinding = {};
    return false;
}

bool MaterialDescriptorManager::ResolveBindlessIndex(MaterialDescriptorSet::TextureBinding &binding) const
{
    if (!m_bindlessMaterialMode)
        return true;
    if (!binding.gpuView || !binding.gpuView->IsValid() || !m_bindlessTextureResolver)
        return false;
    binding.resourceIndex = m_bindlessTextureResolver(binding.gpuView);
    return binding.resourceIndex.IsValid();
}

TextureResolveStatus
MaterialDescriptorManager::ResolveRenderTextureBinding(const std::shared_ptr<rhi::RenderTexture> &texture,
                                                       MaterialDescriptorSet::TextureBinding &binding) const
{
    binding = m_renderTextureResolver(texture);
    if (!ResolveBindlessIndex(binding))
        throw std::runtime_error("RenderTexture could not be published to the material texture table");
    binding.resolvedExplicitTexture = true;
    return TextureResolveStatus::Ready;
}

TextureResolveStatus
MaterialDescriptorManager::ResolveExplicitTextureBinding(const std::string &texturePath, const std::string &bindingName,
                                                         MaterialDescriptorSet::TextureBinding &outBinding,
                                                         const MaterialTextureSampler *sampler) const
{
    if (!m_textureResolver || texturePath.empty()) {
        return TextureResolveStatus::Failed;
    }

    TextureResolveResult result = m_textureResolver(texturePath, bindingName, sampler);
    if (result.status != TextureResolveStatus::Ready) {
        outBinding = {};
        return result.status;
    }

    outBinding = std::move(result.binding);
    if (outBinding.imageView == VK_NULL_HANDLE || outBinding.sampler == VK_NULL_HANDLE || !outBinding.gpuView ||
        !outBinding.gpuView->IsValid()) {
        outBinding = {};
        INXLOG_ERROR("Texture resolver returned Ready without a complete GPU binding for texture '", texturePath,
                     "' (binding='", bindingName, "')");
        return TextureResolveStatus::Failed;
    }
    if (!ResolveBindlessIndex(outBinding)) {
        outBinding = {};
        return TextureResolveStatus::Failed;
    }
    outBinding.resolvedExplicitTexture = true;
    return TextureResolveStatus::Ready;
}

MaterialDescriptorSet *MaterialDescriptorManager::GetOrCreateDescriptorSet(const InxMaterial &material,
                                                                           const ShaderProgram &program)
{
    const std::string materialName = material.GetMaterialKey();

    VkDescriptorSetLayout requiredLayout = program.GetDescriptorSetLayout(0);
    if (requiredLayout == VK_NULL_HANDLE) {
        INXLOG_ERROR("Shader program has no descriptor set layout");
        return nullptr;
    }

    // SetBuffer is intentionally authorable before the first Forward program
    // exists. Once the real pass is available, however, every authored buffer
    // must resolve to exactly one reflected set-0 storage binding. Do this
    // before accepting a cached descriptor so first-use ordering never turns
    // an invalid binding into a silently ignored value.
    for (const auto &[name, buffer] : material.GetBuffers()) {
        const MergedDescriptorBinding *binding = nullptr;
        for (const auto &candidate : program.GetDescriptorBindings()) {
            if (candidate.set == 0 && candidate.name == name && candidate.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                binding = &candidate;
                break;
            }
        }
        if (!buffer || !binding || binding->descriptorCount != 1) {
            INXLOG_ERROR("Material buffer '", name, "' does not match one reflected set-0 storage binding");
            return nullptr;
        }
    }

    // Check if already exists AND uses the same layout
    auto it = m_descriptorSets.find(materialName);
    if (it != m_descriptorSets.end() && it->second->isValid) {
        const MaterialUBOLayout *requiredMaterialLayout = program.GetMaterialUBOLayout();
        const MaterialUBOLayout *requiredVertexMaterialLayout = program.GetVertexMaterialUBOLayout();
        bool needsMaterialUBO = requiredMaterialLayout != nullptr && requiredMaterialLayout->size > 0;
        bool needsVertexMaterialUBO = requiredVertexMaterialLayout != nullptr && requiredVertexMaterialLayout->size > 0;
        bool hasMaterialUBO = it->second->materialUBO && it->second->materialUBO->IsValid();
        bool hasVertexMaterialUBO = it->second->vertexMaterialUBO && it->second->vertexMaterialUBO->IsValid();
        const bool needsBindlessTextureUBO = m_bindlessMaterialMode && program.UsesBindlessTextureABI();
        const bool hasBindlessTextureUBO = it->second->textureIndexUBO && it->second->textureIndexUBO->IsValid();
        const bool hasSameTextureABI = it->second->usesBindlessTextureABI == needsBindlessTextureUBO;
        bool hasSameStorageBuffers = true;
        for (const auto &binding : program.GetDescriptorBindings()) {
            if (binding.set != 0 || binding.type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                continue;
            const auto current = material.GetBuffer(binding.name);
            const auto published = it->second->storageBufferBindings.find(binding.binding);
            if ((published == it->second->storageBufferBindings.end() ? nullptr : published->second) != current) {
                hasSameStorageBuffers = false;
                break;
            }
        }

        // CRITICAL: Must verify layout matches - shader may have changed
        if (it->second->layout == requiredLayout && needsMaterialUBO == hasMaterialUBO &&
            needsVertexMaterialUBO == hasVertexMaterialUBO && needsBindlessTextureUBO == hasBindlessTextureUBO &&
            hasSameTextureABI && hasSameStorageBuffers) {
            return it->second.get();
        } else {
            INXLOG_INFO("Material '", materialName, "' descriptor requirements changed, recreating descriptor set");
            auto staleEntry = std::shared_ptr<MaterialDescriptorSet>(std::move(it->second));
            if (staleEntry && staleEntry->descriptorSet != VK_NULL_HANDLE) {
                m_liveDescriptorHandles.erase(reinterpret_cast<uint64_t>(staleEntry->descriptorSet));
            }
            m_descriptorSets.erase(it);
            RetireDescriptorSet(std::move(staleEntry));
        }
    }

    // Create new descriptor set
    auto matDescSet = std::make_unique<MaterialDescriptorSet>();
    matDescSet->layout = requiredLayout; // Track which layout we're using
    matDescSet->bindings = program.GetDescriptorBindings();
    matDescSet->usesBindlessTextureABI = m_bindlessMaterialMode && program.UsesBindlessTextureABI();

    if (!m_descriptorManager) {
        INXLOG_ERROR("Material descriptor manager is unavailable for material: ", materialName);
        return nullptr;
    }
    const auto arena =
        m_updateAfterBindEnabled ? vk::DescriptorArena::UpdateAfterBind : vk::DescriptorArena::Persistent;
    matDescSet->descriptorLease = m_descriptorManager->Allocate(requiredLayout, arena);
    if (!matDescSet->descriptorLease.IsValid()) {
        INXLOG_ERROR("Failed to allocate descriptor set for material: ", materialName);
        return nullptr;
    }
    matDescSet->descriptorSet = matDescSet->descriptorLease.set;

    // Track this handle so callers can verify it's still live before binding.
    m_liveDescriptorHandles.emplace(reinterpret_cast<uint64_t>(matDescSet->descriptorSet), matDescSet.get());

    // Create material UBO if shader has one
    const MaterialUBOLayout *uboLayout = program.GetMaterialUBOLayout();
    if (uboLayout != nullptr && uboLayout->size > 0) {
        matDescSet->materialUBO = std::make_unique<MaterialUBO>();
        if (!matDescSet->materialUBO->Create(m_vmaAllocator, m_device, *uboLayout)) {
            INXLOG_ERROR("Failed to create material UBO for: ", materialName);
        } else {
            // Update UBO with current material values
            matDescSet->materialUBO->Update(material);
        }
    }

    // Create the vertex-stage material UBO when ShaderInfo declares Properties (binding 14).
    const MaterialUBOLayout *vertUboLayout = program.GetVertexMaterialUBOLayout();
    if (vertUboLayout != nullptr && vertUboLayout->size > 0) {
        matDescSet->vertexMaterialUBO = std::make_unique<MaterialUBO>();
        if (!matDescSet->vertexMaterialUBO->Create(m_vmaAllocator, m_device, *vertUboLayout)) {
            INXLOG_ERROR("Failed to create vertex material UBO for: ", materialName);
        } else {
            matDescSet->vertexMaterialUBO->Update(material);
        }
    }

    if (matDescSet->usesBindlessTextureABI) {
        MaterialUBOLayout textureIndexLayout{};
        if (const auto *reflectedLayout = program.GetBindlessTextureIndexLayout())
            textureIndexLayout = *reflectedLayout;
        else {
            textureIndexLayout.binding = ShaderProgram::MaterialTextureIndexBinding;
            textureIndexLayout.size = ShaderProgram::MaterialTextureIndexCapacity * sizeof(uint32_t);
        }
        matDescSet->textureIndexUBO = std::make_unique<MaterialUBO>();
        if (!matDescSet->textureIndexUBO->Create(m_vmaAllocator, m_device, textureIndexLayout)) {
            INXLOG_ERROR("Failed to create bindless texture-index UBO for: ", materialName);
            matDescSet->textureIndexUBO.reset();
        }
    }

    // Update descriptor bindings
    for (const auto &binding : matDescSet->bindings) {
        if (binding.set != 0 || binding.type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            continue;
        if (binding.descriptorCount != 1) {
            INXLOG_ERROR("Material storage-buffer arrays are not supported for binding '", binding.name, "'");
            continue;
        }
        if (auto buffer = material.GetBuffer(binding.name))
            matDescSet->storageBufferBindings[binding.binding] = std::move(buffer);
    }
    if (!UpdateDescriptorBindings(*matDescSet, program)) {
        m_liveDescriptorHandles.erase(reinterpret_cast<uint64_t>(matDescSet->descriptorSet));
        RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(matDescSet)));
        return nullptr;
    }

    // Resolve material Texture2D properties → actual GPU textures
    if (m_textureResolver) {
        const auto &properties = material.GetAllProperties();
        const auto &bindings = program.GetDescriptorBindings();

        // Collect all descriptor writes and flush as a single batch
        std::vector<VkWriteDescriptorSet> texWrites;
        std::vector<VkDescriptorImageInfo> texImageInfos;
        texWrites.reserve(properties.size());
        texImageInfos.reserve(properties.size());

        if (matDescSet->usesBindlessTextureABI) {
            const auto *indexLayout = program.GetBindlessTextureIndexLayout();
            if (!indexLayout) {
                INXLOG_ERROR("Bindless material program has no reflected texture-index layout: ", materialName);
                m_liveDescriptorHandles.erase(reinterpret_cast<uint64_t>(matDescSet->descriptorSet));
                RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(matDescSet)));
                return nullptr;
            }

            std::array<uint32_t, ShaderProgram::MaterialTextureIndexCapacity> indices{};
            matDescSet->bindlessTextureIndices.reserve(indexLayout->members.size());
            for (size_t slot = 0; slot < indexLayout->members.size(); ++slot) {
                if (slot >= indices.size()) {
                    INXLOG_ERROR("Bindless material texture count exceeds the fixed shader ABI for: ", materialName);
                    m_liveDescriptorHandles.erase(reinterpret_cast<uint64_t>(matDescSet->descriptorSet));
                    RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(matDescSet)));
                    return nullptr;
                }

                const std::string &propName = indexLayout->members[slot].name;
                const auto property = properties.find(propName);
                const auto renderTexture = material.GetRenderTexture(propName);
                const std::string *texturePath = nullptr;
                if (property != properties.end() && property->second.type == MaterialPropertyType::Texture2D)
                    texturePath = std::get_if<std::string>(&property->second.value);

                MaterialDescriptorSet::TextureBinding resolvedBinding{};
                TextureResolveStatus resolveStatus = TextureResolveStatus::Failed;
                const bool hasExplicitTexture =
                    renderTexture || (texturePath && !texturePath->empty() && !IsPlaceholderTexturePath(*texturePath));
                if (hasExplicitTexture)
                    resolveStatus = renderTexture
                                        ? ResolveRenderTextureBinding(renderTexture, resolvedBinding)
                                        : ResolveExplicitTextureBinding(*texturePath, propName, resolvedBinding,
                                                                        material.GetTextureSampler(propName));
                if (hasExplicitTexture && resolveStatus != TextureResolveStatus::Ready) {
                    matDescSet->hasUnresolvedExplicitTextures = true;
                    if (resolveStatus == TextureResolveStatus::Pending)
                        matDescSet->hasPendingTextures = true;
                }

                const bool resolvedExplicit = resolveStatus == TextureResolveStatus::Ready;
                if (!resolvedExplicit && !TryGetDefaultTextureBinding(propName, resolvedBinding)) {
                    INXLOG_ERROR("Failed to initialize required bindless texture property '", propName,
                                 "' for material '", materialName, "'");
                    m_liveDescriptorHandles.erase(reinterpret_cast<uint64_t>(matDescSet->descriptorSet));
                    RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(matDescSet)));
                    return nullptr;
                }

                const uint32_t shaderIndex = resolvedBinding.resourceIndex.IsValid()
                                                 ? resolvedBinding.resourceIndex.index
                                                 : rhi::ResourceIndex::FallbackIndex;
                matDescSet->textureBindings[static_cast<uint32_t>(slot)] = resolvedBinding;
                matDescSet->bindlessTextureIndices.push_back(resolvedBinding.resourceIndex);
                indices[slot] = shaderIndex;
            }
            if (matDescSet->textureIndexUBO)
                matDescSet->textureIndexUBO->UpdateTextureIndices(indices);
        } else {
            for (const auto &[propName, prop] : properties) {
                if (prop.type != MaterialPropertyType::Texture2D)
                    continue;

                const std::string *texturePath = std::get_if<std::string>(&prop.value);
                const auto renderTexture = material.GetRenderTexture(propName);
                if (!renderTexture && (!texturePath || texturePath->empty()))
                    continue;
                const bool isPlaceholderTexture = !renderTexture && IsPlaceholderTexturePath(*texturePath);

                // Find the matching sampler binding by name (set 0 only)
                for (const auto &binding : bindings) {
                    if (binding.set != 0 || binding.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        continue;

                    if (binding.name != propName)
                        continue;
                    MaterialDescriptorSet::TextureBinding resolvedBinding{};
                    const TextureResolveStatus resolveStatus =
                        renderTexture ? ResolveRenderTextureBinding(renderTexture, resolvedBinding)
                        : isPlaceholderTexture
                            ? TextureResolveStatus::Pending
                            : ResolveExplicitTextureBinding(*texturePath, binding.name, resolvedBinding,
                                                            material.GetTextureSampler(binding.name));
                    const bool resolvedExplicit = resolveStatus == TextureResolveStatus::Ready;

                    if (!isPlaceholderTexture && resolveStatus != TextureResolveStatus::Ready) {
                        matDescSet->hasUnresolvedExplicitTextures = true;
                        if (resolveStatus == TextureResolveStatus::Pending)
                            matDescSet->hasPendingTextures = true;
                    }

                    if (!resolvedExplicit && !TryGetDefaultTextureBinding(binding.name, resolvedBinding)) {
                        matDescSet->textureBindings.erase(binding.binding);
                        break;
                    }

                    matDescSet->textureBindings[binding.binding] = resolvedBinding;

                    if (!resolvedExplicit && resolveStatus == TextureResolveStatus::Failed) {
                        INXLOG_WARN("Failed to resolve texture '", *texturePath, "' for material '", materialName,
                                    "' property '", propName, "' — binding default texture");
                    }

                    AppendImageWrite(texWrites, texImageInfos, matDescSet->descriptorSet, binding.binding,
                                     resolvedBinding.imageView, resolvedBinding.sampler);
                    break;
                }
            }
        }

        if (!texWrites.empty()) {
            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(texWrites.size()), texWrites.data(), 0, nullptr);
        }
    }

    matDescSet->isValid = true;

    MaterialDescriptorSet *result = matDescSet.get();
    m_descriptorSets[materialName] = std::move(matDescSet);

    return result;
}

MaterialDescriptorSet *MaterialDescriptorManager::GetOrCreateRendererDescriptorSet(
    const InxMaterial &material, const ShaderProgram &program,
    const std::shared_ptr<const RendererParameterBlock> &parameters)
{
    // Every compatible material pass borrows the Forward set-0 ABI. Building
    // an override from the active pass layout would replace the material's
    // authoritative descriptor whenever a non-primary camera/pass consumes a
    // short-lived payload. Keep one owner/layout and bind that publication in
    // all reflection-compatible pass pipelines.
    const ShaderProgram *baseProgram = material.GetPassShaderProgram(ShaderCompileTarget::Forward);
    if (!baseProgram)
        baseProgram = &program;
    if (!parameters || (parameters->properties.empty() && parameters->buffers.empty()))
        return GetOrCreateDescriptorSet(material, *baseProgram);

    MaterialDescriptorSet *base = GetOrCreateDescriptorSet(material, *baseProgram);
    if (!base || !base->isValid)
        return nullptr;
    const VkDescriptorSetLayout layout = base->layout;
    const std::string key = material.GetMaterialKey() + "|" + std::to_string(reinterpret_cast<uintptr_t>(layout)) +
                            "|" + std::to_string(reinterpret_cast<uintptr_t>(parameters.get())) + "|" +
                            std::to_string(material.GetVersion()) + "|" +
                            std::to_string(reinterpret_cast<uintptr_t>(base->descriptorSet));
    const auto cached = m_rendererDescriptorSets.find(key);
    if (cached != m_rendererDescriptorSets.end() && cached->second.descriptor && cached->second.descriptor->isValid) {
        // The address is part of the key for speed, but allocators may reuse an
        // expired block's address. Confirm ownership before accepting a hit so
        // a later draw can never inherit the retired payload.
        const auto owner = cached->second.parameters.lock();
        if (owner && owner.get() == parameters.get())
            return cached->second.descriptor.get();
    }

    // A block is immutable. A new block identity is the publication boundary;
    // a changed base material or descriptor replaces only this block/layout
    // variant and retires its previous GPU generation.
    for (auto it = m_rendererDescriptorSets.begin(); it != m_rendererDescriptorSets.end();) {
        const auto owner = it->second.parameters.lock();
        const bool stale = !owner || (owner.get() == parameters.get() && it->second.layout == layout);
        if (!stale) {
            ++it;
            continue;
        }
        auto retired = std::shared_ptr<MaterialDescriptorSet>(std::move(it->second.descriptor));
        if (retired && retired->descriptorSet != VK_NULL_HANDLE)
            m_liveDescriptorHandles.erase(reinterpret_cast<uint64_t>(retired->descriptorSet));
        it = m_rendererDescriptorSets.erase(it);
        RetireDescriptorSet(std::move(retired));
    }

    auto descriptor = std::make_unique<MaterialDescriptorSet>();
    descriptor->layout = layout;
    descriptor->bindings = baseProgram->GetDescriptorBindings();
    descriptor->usesBindlessTextureABI = base->usesBindlessTextureABI;
    descriptor->textureBindings = base->textureBindings;
    descriptor->storageBufferBindings = base->storageBufferBindings;
    descriptor->hasPendingTextures = base->hasPendingTextures;
    descriptor->hasUnresolvedExplicitTextures = base->hasUnresolvedExplicitTextures;

    const auto arena =
        m_updateAfterBindEnabled ? vk::DescriptorArena::UpdateAfterBind : vk::DescriptorArena::Persistent;
    descriptor->descriptorLease = m_descriptorManager->Allocate(layout, arena);
    if (!descriptor->descriptorLease.IsValid())
        return nullptr;
    descriptor->descriptorSet = descriptor->descriptorLease.set;

    if (const auto *materialLayout = baseProgram->GetMaterialUBOLayout(); materialLayout && materialLayout->size > 0) {
        descriptor->materialUBO = std::make_unique<MaterialUBO>();
        if (!descriptor->materialUBO->Create(m_vmaAllocator, m_device, *materialLayout)) {
            RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(descriptor)));
            return nullptr;
        }
        descriptor->materialUBO->Update(material);
        descriptor->materialUBO->Apply(*parameters);
    }
    if (const auto *vertexLayout = baseProgram->GetVertexMaterialUBOLayout(); vertexLayout && vertexLayout->size > 0) {
        descriptor->vertexMaterialUBO = std::make_unique<MaterialUBO>();
        if (!descriptor->vertexMaterialUBO->Create(m_vmaAllocator, m_device, *vertexLayout)) {
            RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(descriptor)));
            return nullptr;
        }
        descriptor->vertexMaterialUBO->Update(material);
        descriptor->vertexMaterialUBO->Apply(*parameters);
    }

    const auto resolveTexture = [&](const std::string &name, const std::string &guid,
                                    MaterialDescriptorSet::TextureBinding &binding) {
        if (!guid.empty() && !IsPlaceholderTexturePath(guid)) {
            const TextureResolveStatus status =
                ResolveExplicitTextureBinding(guid, name, binding, material.GetTextureSampler(name));
            if (status == TextureResolveStatus::Ready)
                return true;
            if (status == TextureResolveStatus::Pending) {
                descriptor->hasPendingTextures = true;
                return false;
            }
        }
        return TryGetDefaultTextureBinding(name, binding);
    };

    for (const auto &[name, property] : parameters->properties) {
        if (property.type != MaterialPropertyType::Texture2D)
            continue;
        const auto *guid = std::get_if<std::string>(&property.value);
        if (!guid)
            continue;
        if (descriptor->usesBindlessTextureABI) {
            const auto *indexLayout = baseProgram->GetBindlessTextureIndexLayout();
            if (!indexLayout)
                continue;
            for (size_t slot = 0; slot < indexLayout->members.size(); ++slot) {
                if (indexLayout->members[slot].name != name)
                    continue;
                MaterialDescriptorSet::TextureBinding binding{};
                if (resolveTexture(name, *guid, binding))
                    descriptor->textureBindings[static_cast<uint32_t>(slot)] = std::move(binding);
                break;
            }
        } else {
            for (const auto &declaredBinding : descriptor->bindings) {
                if (declaredBinding.set != 0 || declaredBinding.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
                    declaredBinding.name != name)
                    continue;
                MaterialDescriptorSet::TextureBinding binding{};
                if (resolveTexture(name, *guid, binding))
                    descriptor->textureBindings[declaredBinding.binding] = std::move(binding);
                break;
            }
        }
    }

    for (const auto &[name, buffer] : parameters->buffers) {
        const MergedDescriptorBinding *declaredBinding = nullptr;
        for (const auto &binding : descriptor->bindings) {
            if (binding.set == 0 && binding.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && binding.name == name) {
                declaredBinding = &binding;
                break;
            }
        }
        if (!declaredBinding || declaredBinding->descriptorCount != 1) {
            INXLOG_ERROR("Renderer parameter buffer '", name, "' does not match one reflected storage binding");
            RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(descriptor)));
            return nullptr;
        }
        descriptor->storageBufferBindings[declaredBinding->binding] = buffer;
    }

    if (descriptor->usesBindlessTextureABI) {
        MaterialUBOLayout indexLayout{};
        if (const auto *reflected = baseProgram->GetBindlessTextureIndexLayout())
            indexLayout = *reflected;
        else {
            indexLayout.binding = ShaderProgram::MaterialTextureIndexBinding;
            indexLayout.size = ShaderProgram::MaterialTextureIndexCapacity * sizeof(uint32_t);
        }
        descriptor->textureIndexUBO = std::make_unique<MaterialUBO>();
        if (!descriptor->textureIndexUBO->Create(m_vmaAllocator, m_device, indexLayout)) {
            RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(descriptor)));
            return nullptr;
        }
        std::array<uint32_t, ShaderProgram::MaterialTextureIndexCapacity> indices{};
        descriptor->bindlessTextureIndices.reserve(descriptor->textureBindings.size());
        for (const auto &[slot, binding] : descriptor->textureBindings) {
            if (slot >= indices.size())
                continue;
            indices[slot] =
                binding.resourceIndex.IsValid() ? binding.resourceIndex.index : rhi::ResourceIndex::FallbackIndex;
            descriptor->bindlessTextureIndices.push_back(binding.resourceIndex);
        }
        descriptor->textureIndexUBO->UpdateTextureIndices(indices);
    }

    if (!UpdateDescriptorBindings(*descriptor, *baseProgram)) {
        RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(descriptor)));
        return nullptr;
    }
    if (descriptor->hasUnboundRequiredBuffers) {
        INXLOG_ERROR("Renderer parameter block does not bind every reflected material storage buffer");
        RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet>(std::move(descriptor)));
        return nullptr;
    }

    descriptor->isValid = true;
    m_liveDescriptorHandles.emplace(reinterpret_cast<uint64_t>(descriptor->descriptorSet), descriptor.get());
    MaterialDescriptorSet *result = descriptor.get();
    RendererDescriptorEntry entry;
    entry.parameters = parameters;
    entry.layout = layout;
    entry.descriptor = std::move(descriptor);
    m_rendererDescriptorSets.emplace(key, std::move(entry));
    return result;
}

bool MaterialDescriptorManager::UpdateDescriptorBindings(MaterialDescriptorSet &matDescSet,
                                                         const ShaderProgram &program)
{
    matDescSet.bufferBindings.clear();
    matDescSet.hasUnboundRequiredBuffers = false;
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkDescriptorImageInfo> imageInfos;

    // Reserve space to avoid reallocation invalidating pointers
    const auto &bindings = program.GetDescriptorBindings();
    bufferInfos.reserve(bindings.size());
    imageInfos.reserve(bindings.size());

    for (const auto &binding : bindings) {
        // Only write set 0 bindings into the material descriptor set.
        // Set 1 (per-view shadow map) is handled separately per render graph.
        if (binding.set != 0) {
            continue;
        }

        if (binding.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            VkDescriptorBufferInfo bufferInfo{};

            // Set 0 is exclusively material-owned. Camera and lighting data
            // belong to the active RenderView descriptor set (set 1).
            const MaterialUBOLayout *matLayout = program.GetMaterialUBOLayout();
            bool isMaterialUBOBinding = matLayout && matLayout->size > 0 && binding.binding == matLayout->binding;

            const MaterialUBOLayout *vertMatLayout = program.GetVertexMaterialUBOLayout();
            bool isVertexMaterialUBOBinding =
                vertMatLayout && vertMatLayout->size > 0 && binding.binding == vertMatLayout->binding;

            if (matDescSet.usesBindlessTextureABI && binding.binding == ShaderProgram::MaterialTextureIndexBinding &&
                matDescSet.textureIndexUBO && matDescSet.textureIndexUBO->IsValid()) {
                bufferInfo.buffer = matDescSet.textureIndexUBO->GetBuffer();
                bufferInfo.offset = 0;
                bufferInfo.range = matDescSet.textureIndexUBO->GetSize();
            } else if (isVertexMaterialUBOBinding && matDescSet.vertexMaterialUBO &&
                       matDescSet.vertexMaterialUBO->IsValid()) {
                // Vertex-stage material UBO at binding 14
                bufferInfo.buffer = matDescSet.vertexMaterialUBO->GetBuffer();
                bufferInfo.offset = 0;
                bufferInfo.range = matDescSet.vertexMaterialUBO->GetSize();
            } else if (isMaterialUBOBinding && matDescSet.materialUBO && matDescSet.materialUBO->IsValid()) {
                // Material UBO — identified by shader reflection binding number
                bufferInfo.buffer = matDescSet.materialUBO->GetBuffer();
                bufferInfo.offset = 0;
                bufferInfo.range = matDescSet.materialUBO->GetSize();
            } else {
                INXLOG_ERROR("Material shader ABI violation: set 0 uniform buffer '", binding.name, "' at binding ",
                             binding.binding,
                             " is not a reflected material Properties block. Camera and lighting uniforms must use "
                             "the engine-owned RenderView set 1 contract.");
                return false;
            }

            AppendBufferWrite(writes, bufferInfos, matDescSet.descriptorSet, binding.binding, binding.type, bufferInfo,
                              binding.descriptorCount);
            matDescSet.bufferBindings[binding.binding] = bufferInfo;
        } else if (binding.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            const auto storage = matDescSet.storageBufferBindings.find(binding.binding);
            if (binding.descriptorCount != 1 || storage == matDescSet.storageBufferBindings.end() || !storage->second ||
                !m_bufferResolver) {
                matDescSet.hasUnboundRequiredBuffers = true;
                continue;
            }
            const VkDescriptorBufferInfo bufferInfo = m_bufferResolver(storage->second);
            if (bufferInfo.buffer == VK_NULL_HANDLE || bufferInfo.range == 0) {
                INXLOG_ERROR("Material storage buffer '", binding.name, "' is not resident on this render device");
                return false;
            }
            AppendBufferWrite(writes, bufferInfos, matDescSet.descriptorSet, binding.binding, binding.type, bufferInfo);
            matDescSet.bufferBindings[binding.binding] = bufferInfo;
        } else if (binding.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            if (matDescSet.usesBindlessTextureABI)
                continue;
            // Check if we have a texture bound for this slot
            auto texIt = matDescSet.textureBindings.find(binding.binding);
            MaterialDescriptorSet::TextureBinding textureBinding{};
            if (texIt != matDescSet.textureBindings.end()) {
                textureBinding = texIt->second;
            } else if (!TryGetDefaultTextureBinding(binding.name, textureBinding)) {
                continue; // Skip if no valid image
            }

            AppendImageWrite(writes, imageInfos, matDescSet.descriptorSet, binding.binding, textureBinding.imageView,
                             textureBinding.sampler);
        }
    }

    if (matDescSet.usesBindlessTextureABI && matDescSet.textureIndexUBO &&
        matDescSet.bufferBindings.find(ShaderProgram::MaterialTextureIndexBinding) == matDescSet.bufferBindings.end()) {
        VkDescriptorBufferInfo indexBuffer{};
        indexBuffer.buffer = matDescSet.textureIndexUBO->GetBuffer();
        indexBuffer.offset = 0;
        indexBuffer.range = matDescSet.textureIndexUBO->GetSize();
        AppendBufferWrite(writes, bufferInfos, matDescSet.descriptorSet, ShaderProgram::MaterialTextureIndexBinding,
                          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, indexBuffer);
        matDescSet.bufferBindings[ShaderProgram::MaterialTextureIndexBinding] = indexBuffer;
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
    return true;
}

bool MaterialDescriptorManager::PublishDescriptorReplacement(
    MaterialDescriptorSet &descriptorSet,
    const std::unordered_map<uint32_t, MaterialDescriptorSet::TextureBinding> &textureBindings)
{
    if (!m_descriptorManager || descriptorSet.layout == VK_NULL_HANDLE)
        return false;

    const auto arena =
        m_updateAfterBindEnabled ? vk::DescriptorArena::UpdateAfterBind : vk::DescriptorArena::Persistent;
    const vk::DescriptorLease replacement = m_descriptorManager->Allocate(descriptorSet.layout, arena);
    if (!replacement.IsValid()) {
        INXLOG_ERROR("Failed to allocate copy-on-write material descriptor set");
        return false;
    }

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkDescriptorImageInfo> imageInfos;
    writes.reserve(descriptorSet.bindings.size());
    bufferInfos.reserve(descriptorSet.bindings.size());
    imageInfos.reserve(descriptorSet.bindings.size());

    std::unique_ptr<MaterialUBO> replacementTextureIndexUBO;
    if (descriptorSet.usesBindlessTextureABI && descriptorSet.textureIndexUBO) {
        replacementTextureIndexUBO = std::make_unique<MaterialUBO>();
        if (!replacementTextureIndexUBO->Create(m_vmaAllocator, m_device, descriptorSet.textureIndexUBO->GetLayout())) {
            m_descriptorManager->Retire(replacement);
            return false;
        }
        std::array<uint32_t, ShaderProgram::MaterialTextureIndexCapacity> indices{};
        for (const auto &[binding, textureBinding] : textureBindings) {
            if (binding < indices.size())
                indices[binding] = textureBinding.resourceIndex.IsValid() ? textureBinding.resourceIndex.index
                                                                          : rhi::ResourceIndex::FallbackIndex;
        }
        replacementTextureIndexUBO->UpdateTextureIndices(indices);
    }

    for (const auto &binding : descriptorSet.bindings) {
        if (binding.set != 0)
            continue;
        if (binding.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            VkDescriptorBufferInfo bufferInfo{};
            if (descriptorSet.usesBindlessTextureABI && binding.binding == ShaderProgram::MaterialTextureIndexBinding &&
                replacementTextureIndexUBO) {
                bufferInfo.buffer = replacementTextureIndexUBO->GetBuffer();
                bufferInfo.offset = 0;
                bufferInfo.range = replacementTextureIndexUBO->GetSize();
            } else {
                const auto buffer = descriptorSet.bufferBindings.find(binding.binding);
                if (buffer == descriptorSet.bufferBindings.end()) {
                    INXLOG_ERROR("Cannot publish material descriptor replacement: uniform binding ", binding.binding,
                                 " has no buffer snapshot");
                    m_descriptorManager->Retire(replacement);
                    return false;
                }
                bufferInfo = buffer->second;
            }
            if (bufferInfo.buffer == VK_NULL_HANDLE) {
                INXLOG_ERROR("Cannot publish material descriptor replacement: uniform binding ", binding.binding,
                             " has no buffer snapshot");
                m_descriptorManager->Retire(replacement);
                return false;
            }
            AppendBufferWrite(writes, bufferInfos, replacement.set, binding.binding, binding.type, bufferInfo,
                              binding.descriptorCount);
            continue;
        }
        if (binding.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            const auto buffer = descriptorSet.bufferBindings.find(binding.binding);
            if (buffer == descriptorSet.bufferBindings.end() || buffer->second.buffer == VK_NULL_HANDLE) {
                // An incomplete base material descriptor remains incomplete;
                // renderer parameter publications may provide this binding.
                continue;
            }
            AppendBufferWrite(writes, bufferInfos, replacement.set, binding.binding, binding.type, buffer->second,
                              binding.descriptorCount);
            continue;
        }
        if (binding.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            INXLOG_ERROR("Cannot publish material descriptor replacement: unsupported set-0 descriptor type ",
                         static_cast<int>(binding.type), " at binding ", binding.binding);
            m_descriptorManager->Retire(replacement);
            return false;
        }

        if (descriptorSet.usesBindlessTextureABI)
            continue;

        MaterialDescriptorSet::TextureBinding textureBinding{};
        const auto texture = textureBindings.find(binding.binding);
        if (texture != textureBindings.end()) {
            textureBinding = texture->second;
        } else if (!TryGetDefaultTextureBinding(binding.name, textureBinding)) {
            INXLOG_ERROR("Cannot publish material descriptor replacement: image binding ", binding.binding, " ('",
                         binding.name, "') has neither an explicit texture nor a default");
            m_descriptorManager->Retire(replacement);
            return false;
        }
        if (textureBinding.imageView == VK_NULL_HANDLE || textureBinding.sampler == VK_NULL_HANDLE) {
            INXLOG_ERROR("Cannot publish material descriptor replacement: image binding ", binding.binding, " ('",
                         binding.name, "') resolves to a null image or sampler");
            m_descriptorManager->Retire(replacement);
            return false;
        }
        AppendImageWrite(writes, imageInfos, replacement.set, binding.binding, textureBinding.imageView,
                         textureBinding.sampler);
    }

    if (descriptorSet.usesBindlessTextureABI && replacementTextureIndexUBO &&
        descriptorSet.bufferBindings.find(ShaderProgram::MaterialTextureIndexBinding) ==
            descriptorSet.bufferBindings.end()) {
        VkDescriptorBufferInfo indexBuffer{};
        indexBuffer.buffer = replacementTextureIndexUBO->GetBuffer();
        indexBuffer.offset = 0;
        indexBuffer.range = replacementTextureIndexUBO->GetSize();
        AppendBufferWrite(writes, bufferInfos, replacement.set, ShaderProgram::MaterialTextureIndexBinding,
                          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, indexBuffer);
    }

    if (!writes.empty())
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    if (!m_deletionQueue && (!descriptorSet.textureBindings.empty() || descriptorSet.textureIndexUBO != nullptr)) {
        INXLOG_ERROR("Cannot publish material descriptor replacement without a GPU retirement queue");
        m_descriptorManager->Retire(replacement);
        return false;
    }

    const vk::DescriptorLease retiredLease = descriptorSet.descriptorLease;
    const VkDescriptorSet retiredSet = descriptorSet.descriptorSet;
    auto retiredTextureBindings = std::move(descriptorSet.textureBindings);
    auto retiredTextureIndexUBO = std::move(descriptorSet.textureIndexUBO);
    descriptorSet.descriptorLease = replacement;
    descriptorSet.descriptorSet = replacement.set;
    descriptorSet.textureBindings = textureBindings;
    descriptorSet.textureIndexUBO = std::move(replacementTextureIndexUBO);
    descriptorSet.bindlessTextureIndices.clear();
    if (descriptorSet.usesBindlessTextureABI) {
        descriptorSet.bindlessTextureIndices.reserve(textureBindings.size());
        std::array<uint32_t, ShaderProgram::MaterialTextureIndexCapacity> indices{};
        for (const auto &[binding, textureBinding] : textureBindings) {
            if (binding >= indices.size())
                continue;
            const uint32_t shaderIndex = textureBinding.resourceIndex.IsValid() ? textureBinding.resourceIndex.index
                                                                                : rhi::ResourceIndex::FallbackIndex;
            indices[binding] = shaderIndex;
            descriptorSet.bindlessTextureIndices.push_back(textureBinding.resourceIndex);
        }
        if (descriptorSet.textureIndexUBO) {
            descriptorSet.textureIndexUBO->UpdateTextureIndices(indices);
            VkDescriptorBufferInfo indexBuffer{};
            indexBuffer.buffer = descriptorSet.textureIndexUBO->GetBuffer();
            indexBuffer.offset = 0;
            indexBuffer.range = descriptorSet.textureIndexUBO->GetSize();
            descriptorSet.bufferBindings[ShaderProgram::MaterialTextureIndexBinding] = indexBuffer;
        }
    }
    m_liveDescriptorHandles.erase(reinterpret_cast<uint64_t>(retiredSet));
    m_liveDescriptorHandles.emplace(reinterpret_cast<uint64_t>(replacement.set), &descriptorSet);
    m_descriptorManager->Retire(retiredLease);
    if (m_deletionQueue && !retiredTextureBindings.empty()) {
        m_deletionQueue->Retire([bindings = std::move(retiredTextureBindings)]() mutable { bindings.clear(); });
    }
    if (m_deletionQueue && retiredTextureIndexUBO) {
        auto retiredUbo = std::shared_ptr<MaterialUBO>(std::move(retiredTextureIndexUBO));
        m_deletionQueue->Retire([ubo = std::move(retiredUbo)]() mutable { ubo.reset(); });
    }
    return true;
}

void MaterialDescriptorManager::UpdateMaterialUBO(const std::string &materialName, const InxMaterial &material)
{
    auto it = m_descriptorSets.find(materialName);
    if (it != m_descriptorSets.end()) {
        if (it->second->materialUBO) {
            it->second->materialUBO->Update(material);
        }
        if (it->second->vertexMaterialUBO) {
            it->second->vertexMaterialUBO->Update(material);
        }
    }
}

void MaterialDescriptorManager::ResolveTextureProperties(const std::string &materialName, const InxMaterial &material,
                                                         const ShaderProgram &program)
{
    if (!m_textureResolver) {
        return;
    }

    auto it = m_descriptorSets.find(materialName);
    if (it == m_descriptorSets.end() || !it->second->isValid) {
        return;
    }

    auto &matDescSet = *it->second;
    auto candidateBindings = matDescSet.textureBindings;
    bool candidateHasPendingTextures = false;
    bool candidateHasUnresolvedExplicitTextures = false;
    bool bindingsChanged = false;
    const auto &properties = material.GetAllProperties();
    const auto &bindings = program.GetDescriptorBindings();

    if (matDescSet.usesBindlessTextureABI) {
        // The bindless shader exposes texture property names as members of
        // binding 15 instead of creating one sampler descriptor per property.
        // Resolve those members directly and publish the complete index set as
        // one copy-on-write material update.
        const auto *indexLayout = program.GetBindlessTextureIndexLayout();
        if (!indexLayout)
            return;
        for (size_t slot = 0; slot < indexLayout->members.size(); ++slot) {
            const std::string &propName = indexLayout->members[slot].name;
            const auto property = properties.find(propName);
            if (property == properties.end() || property->second.type != MaterialPropertyType::Texture2D)
                continue;
            const auto &prop = property->second;
            if (prop.type != MaterialPropertyType::Texture2D)
                continue;
            const std::string *texturePath = std::get_if<std::string>(&prop.value);
            const auto renderTexture = material.GetRenderTexture(propName);
            if (!renderTexture && (!texturePath || texturePath->empty())) {
                MaterialDescriptorSet::TextureBinding defaultBinding{};
                if (TryGetDefaultTextureBinding(propName, defaultBinding)) {
                    const auto previous = candidateBindings.find(static_cast<uint32_t>(slot));
                    bindingsChanged = bindingsChanged || previous == candidateBindings.end() ||
                                      !HasSameGpuBinding(previous->second, defaultBinding);
                    candidateBindings[static_cast<uint32_t>(slot)] = defaultBinding;
                }
                continue;
            }

            MaterialDescriptorSet::TextureBinding resolvedBinding{};
            const bool placeholder = !renderTexture && IsPlaceholderTexturePath(*texturePath);
            const TextureResolveStatus resolveStatus =
                renderTexture ? ResolveRenderTextureBinding(renderTexture, resolvedBinding)
                : placeholder ? TextureResolveStatus::Pending
                              : ResolveExplicitTextureBinding(*texturePath, propName, resolvedBinding,
                                                              material.GetTextureSampler(propName));
            const bool resolvedExplicit = resolveStatus == TextureResolveStatus::Ready;
            if (!placeholder && !resolvedExplicit) {
                if (resolveStatus == TextureResolveStatus::Pending)
                    candidateHasPendingTextures = true;

                // Texture uploads publish a complete immutable GPU view. Keep
                // the previous revision visible for both pending work and a
                // failed replacement. A failure must never overwrite a good
                // icon with the default white descriptor.
                const auto previous = candidateBindings.find(static_cast<uint32_t>(slot));
                if (previous != candidateBindings.end() && previous->second.resolvedExplicitTexture &&
                    previous->second.gpuView && previous->second.gpuView->IsValid())
                    continue;
                candidateHasUnresolvedExplicitTextures = true;
            }

            if (!resolvedExplicit && !TryGetDefaultTextureBinding(propName, resolvedBinding)) {
                continue;
            }
            const auto previous = candidateBindings.find(static_cast<uint32_t>(slot));
            bindingsChanged = bindingsChanged || previous == candidateBindings.end() ||
                              !HasSameGpuBinding(previous->second, resolvedBinding);
            candidateBindings[static_cast<uint32_t>(slot)] = resolvedBinding;
        }

        const bool previousHasPendingTextures = matDescSet.hasPendingTextures;
        const bool previousHasUnresolvedExplicitTextures = matDescSet.hasUnresolvedExplicitTextures;
        matDescSet.hasPendingTextures = candidateHasPendingTextures;
        matDescSet.hasUnresolvedExplicitTextures = candidateHasUnresolvedExplicitTextures;
        if (!bindingsChanged)
            return;
        if (!PublishDescriptorReplacement(matDescSet, candidateBindings)) {
            matDescSet.hasPendingTextures = previousHasPendingTextures;
            matDescSet.hasUnresolvedExplicitTextures = previousHasUnresolvedExplicitTextures;
            INXLOG_ERROR("Bindless material texture publication failed for '", materialName,
                         "'; the previous complete descriptor set remains active");
        }
        return;
    }

    for (const auto &[propName, prop] : properties) {
        if (prop.type != MaterialPropertyType::Texture2D) {
            continue;
        }

        const std::string *texturePath = std::get_if<std::string>(&prop.value);
        const auto renderTexture = material.GetRenderTexture(propName);

        for (const auto &binding : bindings) {
            if (binding.set != 0 || binding.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                continue;
            }
            if (binding.name == propName) {
                MaterialDescriptorSet::TextureBinding resolvedBinding{};

                if (!renderTexture && (!texturePath || texturePath->empty())) {
                    if (TryGetDefaultTextureBinding(binding.name, resolvedBinding)) {
                        const auto previous = candidateBindings.find(binding.binding);
                        bindingsChanged = bindingsChanged || previous == candidateBindings.end() ||
                                          !HasSameGpuBinding(previous->second, resolvedBinding);
                        candidateBindings[binding.binding] = resolvedBinding;
                    }
                    break;
                }

                const bool isPlaceholder = !renderTexture && IsPlaceholderTexturePath(*texturePath);
                const TextureResolveStatus resolveStatus =
                    renderTexture   ? ResolveRenderTextureBinding(renderTexture, resolvedBinding)
                    : isPlaceholder ? TextureResolveStatus::Pending
                                    : ResolveExplicitTextureBinding(*texturePath, binding.name, resolvedBinding,
                                                                    material.GetTextureSampler(binding.name));
                const bool resolvedExplicit = resolveStatus == TextureResolveStatus::Ready;

                if (!isPlaceholder && !resolvedExplicit) {
                    if (resolveStatus == TextureResolveStatus::Pending)
                        candidateHasPendingTextures = true;
                    const auto previous = candidateBindings.find(binding.binding);
                    if (previous != candidateBindings.end() && previous->second.resolvedExplicitTexture &&
                        previous->second.gpuView && previous->second.gpuView->IsValid()) {
                        break;
                    }
                    candidateHasUnresolvedExplicitTextures = true;
                }

                const bool hasBinding = resolvedExplicit || TryGetDefaultTextureBinding(binding.name, resolvedBinding);

                if (hasBinding) {
                    const auto previous = candidateBindings.find(binding.binding);
                    bindingsChanged = bindingsChanged || previous == candidateBindings.end() ||
                                      !HasSameGpuBinding(previous->second, resolvedBinding);
                    candidateBindings[binding.binding] = resolvedBinding;
                } else {
                    if (resolveStatus == TextureResolveStatus::Failed) {
                        INXLOG_WARN("Failed to resolve texture '", *texturePath, "' for material '", materialName,
                                    "' property '", propName, "' — binding default texture");
                    }
                }
                break;
            }
        }
    }

    const bool previousHasPendingTextures = matDescSet.hasPendingTextures;
    const bool previousHasUnresolvedExplicitTextures = matDescSet.hasUnresolvedExplicitTextures;
    matDescSet.hasPendingTextures = candidateHasPendingTextures;
    matDescSet.hasUnresolvedExplicitTextures = candidateHasUnresolvedExplicitTextures;
    if (!bindingsChanged)
        return;

    if (!PublishDescriptorReplacement(matDescSet, candidateBindings)) {
        matDescSet.hasPendingTextures = previousHasPendingTextures;
        matDescSet.hasUnresolvedExplicitTextures = previousHasUnresolvedExplicitTextures;
        INXLOG_ERROR("Material texture publication failed for '", materialName,
                     "'; the previous complete descriptor set remains active");
    }
}

bool MaterialDescriptorManager::HasPendingTextureProperties(const std::string &materialName) const
{
    const auto it = m_descriptorSets.find(materialName);
    if (it == m_descriptorSets.end() || !it->second || !it->second->isValid)
        return false;
    return it->second->hasPendingTextures ||
           std::any_of(it->second->textureBindings.begin(), it->second->textureBindings.end(), [](const auto &entry) {
               const auto &binding = entry.second;
               return binding.gpuSlot && binding.gpuSlot->Acquire() != binding.gpuView;
           });
}

bool MaterialDescriptorManager::HasUnresolvedExplicitTextureProperties(const std::string &materialName) const
{
    const auto it = m_descriptorSets.find(materialName);
    return it != m_descriptorSets.end() && it->second && it->second->isValid &&
           it->second->hasUnresolvedExplicitTextures;
}

const std::vector<rhi::ResourceIndex> *
MaterialDescriptorManager::GetBindlessTextureIndices(const std::string &materialName) const noexcept
{
    const auto it = m_descriptorSets.find(materialName);
    if (it == m_descriptorSets.end() || !it->second || !it->second->isValid || !it->second->usesBindlessTextureABI)
        return nullptr;
    return &it->second->bindlessTextureIndices;
}

const std::vector<rhi::ResourceIndex> *
MaterialDescriptorManager::GetBindlessTextureIndices(VkDescriptorSet descriptorSet) const noexcept
{
    if (descriptorSet == VK_NULL_HANDLE)
        return nullptr;
    for (const auto &[key, value] : m_descriptorSets) {
        (void)key;
        if (value && value->descriptorSet == descriptorSet && value->usesBindlessTextureABI)
            return &value->bindlessTextureIndices;
    }
    for (const auto &[key, entry] : m_rendererDescriptorSets) {
        (void)key;
        if (entry.descriptor && entry.descriptor->descriptorSet == descriptorSet &&
            entry.descriptor->usesBindlessTextureABI)
            return &entry.descriptor->bindlessTextureIndices;
    }
    return nullptr;
}

void MaterialDescriptorManager::BindTexture(const std::string &materialName, uint32_t binding, VkImageView imageView,
                                            VkSampler sampler)
{
    auto it = m_descriptorSets.find(materialName);
    if (it == m_descriptorSets.end())
        return;

    if (imageView == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
        INXLOG_ERROR("Cannot bind a null texture descriptor to material '", materialName, "' at binding ", binding);
        return;
    }

    auto &descriptorSet = *it->second;
    if (descriptorSet.usesBindlessTextureABI) {
        INXLOG_ERROR("BindTexture cannot mutate a bindless material directly; resolve the TextureGpuView through "
                     "the material texture resolver instead (material='",
                     materialName, "', binding=", binding, ")");
        return;
    }
    const auto previousBinding = descriptorSet.textureBindings.find(binding);
    if (previousBinding != descriptorSet.textureBindings.end() &&
        HasSameGpuBinding(previousBinding->second, {imageView, sampler})) {
        return;
    }

    auto candidateBindings = descriptorSet.textureBindings;
    candidateBindings[binding] = {imageView, sampler};
    if (!PublishDescriptorReplacement(descriptorSet, candidateBindings)) {
        INXLOG_ERROR("Texture binding publication failed for material '", materialName,
                     "'; the previous complete descriptor set remains active");
    }
}

void MaterialDescriptorManager::RemoveDescriptorSet(const std::string &materialName)
{
    auto it = m_descriptorSets.find(materialName);
    if (it != m_descriptorSets.end()) {
        auto retiredEntry = std::shared_ptr<MaterialDescriptorSet>(std::move(it->second));
        if (retiredEntry && retiredEntry->descriptorSet != VK_NULL_HANDLE) {
            const uint64_t handle = reinterpret_cast<uint64_t>(retiredEntry->descriptorSet);
            m_liveDescriptorHandles.erase(handle);
        }
        m_descriptorSets.erase(it);
        RetireDescriptorSet(std::move(retiredEntry));
    }
    const std::string prefix = materialName + "|";
    for (auto rendererIt = m_rendererDescriptorSets.begin(); rendererIt != m_rendererDescriptorSets.end();) {
        if (rendererIt->first.compare(0, prefix.size(), prefix) != 0) {
            ++rendererIt;
            continue;
        }
        auto retiredEntry = std::shared_ptr<MaterialDescriptorSet>(std::move(rendererIt->second.descriptor));
        if (retiredEntry && retiredEntry->descriptorSet != VK_NULL_HANDLE)
            m_liveDescriptorHandles.erase(reinterpret_cast<uint64_t>(retiredEntry->descriptorSet));
        rendererIt = m_rendererDescriptorSets.erase(rendererIt);
        RetireDescriptorSet(std::move(retiredEntry));
    }
}

size_t MaterialDescriptorManager::CollectExpiredRendererDescriptorSets()
{
    size_t retiredCount = 0;
    for (auto it = m_rendererDescriptorSets.begin(); it != m_rendererDescriptorSets.end();) {
        if (!it->second.parameters.expired()) {
            ++it;
            continue;
        }
        auto retired = std::shared_ptr<MaterialDescriptorSet>(std::move(it->second.descriptor));
        if (retired && retired->descriptorSet != VK_NULL_HANDLE)
            m_liveDescriptorHandles.erase(reinterpret_cast<uint64_t>(retired->descriptorSet));
        it = m_rendererDescriptorSets.erase(it);
        if (retired)
            RetireDescriptorSet(std::move(retired));
        ++retiredCount;
    }
    return retiredCount;
}

void MaterialDescriptorManager::RetireDescriptorSet(std::shared_ptr<MaterialDescriptorSet> descriptorSet)
{
    if (!descriptorSet)
        throw std::invalid_argument("Cannot retire an empty material descriptor set");

    const vk::DescriptorLease lease = descriptorSet->descriptorLease;
    auto pending = m_pendingDescriptorSetReleases;
    if (m_descriptorManager)
        m_descriptorManager->Retire(lease);

    if (m_deletionQueue) {
        pending->fetch_add(1, std::memory_order_relaxed);
        m_deletionQueue->Retire([descriptorSet = std::move(descriptorSet), pending = std::move(pending)]() mutable {
            descriptorSet->descriptorSet = VK_NULL_HANDLE;
            descriptorSet->descriptorLease = {};
            descriptorSet.reset();
            pending->fetch_sub(1, std::memory_order_relaxed);
        });
        return;
    }

    vkDeviceWaitIdle(m_device);
    if (m_descriptorManager)
        (void)m_descriptorManager->Collect((std::numeric_limits<rhi::SubmissionSerial>::max)());
    descriptorSet->descriptorSet = VK_NULL_HANDLE;
    descriptorSet->descriptorLease = {};
}

void MaterialDescriptorManager::Clear()
{
    for (auto &[name, descriptorSet] : m_descriptorSets) {
        (void)name;
        if (descriptorSet && m_descriptorManager)
            m_descriptorManager->Retire(descriptorSet->descriptorLease);
    }
    for (auto &[key, entry] : m_rendererDescriptorSets) {
        (void)key;
        if (entry.descriptor && m_descriptorManager)
            m_descriptorManager->Retire(entry.descriptor->descriptorLease);
    }
    m_descriptorSets.clear();
    m_rendererDescriptorSets.clear();
    // All handles are now invalid — clear the live-handle tracking set.
    m_liveDescriptorHandles.clear();
}

void MaterialDescriptorManager::SetDefaultTexture(VkImageView imageView, VkSampler sampler,
                                                  std::shared_ptr<const rhi::TextureGpuView> gpuView)
{
    m_defaultImageView = imageView;
    m_defaultSampler = sampler;
    m_defaultGpuView = std::move(gpuView);
}

void MaterialDescriptorManager::SetDefaultNormalTexture(VkImageView imageView, VkSampler sampler,
                                                        std::shared_ptr<const rhi::TextureGpuView> gpuView)
{
    m_defaultNormalImageView = imageView;
    m_defaultNormalSampler = sampler;
    m_defaultNormalGpuView = std::move(gpuView);
}

} // namespace infernux
