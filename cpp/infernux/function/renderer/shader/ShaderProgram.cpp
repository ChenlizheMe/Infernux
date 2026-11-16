#include "ShaderProgram.h"
#include "DeviceShaderProgram.h"
#include <algorithm>
#include <core/log/InxLog.h>
#include <function/renderer/vk/RhiVulkanTypes.h>
#include <platform/filesystem/InxPath.h>
#include <stdexcept>

namespace infernux
{

VkDescriptorSetLayout ShaderProgram::s_globalsDescSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout ShaderProgram::s_perViewDescSetLayout = VK_NULL_HANDLE;
bool ShaderProgram::s_updateAfterBindEnabled = false;
VkDescriptorSetLayout ShaderProgram::s_bindlessTextureDescSetLayout = VK_NULL_HANDLE;
bool ShaderProgram::s_bindlessTextureEnabled = false;

void ShaderProgram::SetGlobalsDescSetLayout(VkDescriptorSetLayout layout)
{
    s_globalsDescSetLayout = layout;
}

VkDescriptorSetLayout ShaderProgram::GetGlobalsDescSetLayout()
{
    return s_globalsDescSetLayout;
}

void ShaderProgram::SetPerViewDescSetLayout(VkDescriptorSetLayout layout)
{
    s_perViewDescSetLayout = layout;
}

VkDescriptorSetLayout ShaderProgram::GetPerViewDescSetLayout()
{
    return s_perViewDescSetLayout;
}

void ShaderProgram::SetUpdateAfterBindEnabled(bool enabled)
{
    s_updateAfterBindEnabled = enabled;
}

bool ShaderProgram::IsUpdateAfterBindEnabled()
{
    return s_updateAfterBindEnabled;
}

void ShaderProgram::SetBindlessTextureDescSetLayout(VkDescriptorSetLayout layout)
{
    s_bindlessTextureDescSetLayout = layout;
}

VkDescriptorSetLayout ShaderProgram::GetBindlessTextureDescSetLayout()
{
    return s_bindlessTextureDescSetLayout;
}

void ShaderProgram::SetBindlessTextureEnabled(bool enabled)
{
    s_bindlessTextureEnabled = enabled;
}

bool ShaderProgram::IsBindlessTextureEnabled()
{
    return s_bindlessTextureEnabled && s_bindlessTextureDescSetLayout != VK_NULL_HANDLE;
}

// ============================================================================
// ShaderProgram Implementation
// ============================================================================

ShaderProgram::~ShaderProgram()
{
    Destroy();
}

ShaderProgram::ShaderProgram(ShaderProgram &&other) noexcept
    : m_device(other.m_device), m_shaderId(std::move(other.m_shaderId)), m_variantKey(std::move(other.m_variantKey)),
      m_vertModule(other.m_vertModule), m_fragModule(other.m_fragModule),
      m_vertexEntryPoint(std::move(other.m_vertexEntryPoint)),
      m_fragmentEntryPoint(std::move(other.m_fragmentEntryPoint)), m_domain(other.m_domain),
      m_vertReflection(std::move(other.m_vertReflection)), m_fragReflection(std::move(other.m_fragReflection)),
      m_descriptorBindings(std::move(other.m_descriptorBindings)),
      m_descriptorSetLayouts(std::move(other.m_descriptorSetLayouts)),
      m_materialLayoutOwner(std::move(other.m_materialLayoutOwner)),
      m_borrowedMaterialLayout(other.m_borrowedMaterialLayout), m_pipelineLayout(other.m_pipelineLayout),
      m_materialUBOLayout(std::move(other.m_materialUBOLayout)), m_hasMaterialUBO(other.m_hasMaterialUBO),
      m_vertexMaterialUBOLayout(std::move(other.m_vertexMaterialUBOLayout)),
      m_hasVertexMaterialUBO(other.m_hasVertexMaterialUBO),
      m_bindlessTextureIndexLayout(std::move(other.m_bindlessTextureIndexLayout)),
      m_hasBindlessTextureIndexLayout(other.m_hasBindlessTextureIndexLayout),
      m_usesBindlessTextureABI(other.m_usesBindlessTextureABI)
{
    other.m_device = VK_NULL_HANDLE;
    other.m_vertModule = VK_NULL_HANDLE;
    other.m_fragModule = VK_NULL_HANDLE;
    other.m_pipelineLayout = VK_NULL_HANDLE;
    other.m_borrowedMaterialLayout = VK_NULL_HANDLE;
    other.m_hasMaterialUBO = false;
    other.m_hasVertexMaterialUBO = false;
    other.m_hasBindlessTextureIndexLayout = false;
    other.m_usesBindlessTextureABI = false;
    other.m_descriptorSetLayouts.clear();
}

ShaderProgram &ShaderProgram::operator=(ShaderProgram &&other) noexcept
{
    if (this != &other) {
        Destroy();

        m_device = other.m_device;
        m_shaderId = std::move(other.m_shaderId);
        m_variantKey = std::move(other.m_variantKey);
        m_vertModule = other.m_vertModule;
        m_fragModule = other.m_fragModule;
        m_vertexEntryPoint = std::move(other.m_vertexEntryPoint);
        m_fragmentEntryPoint = std::move(other.m_fragmentEntryPoint);
        m_domain = other.m_domain;
        m_vertReflection = std::move(other.m_vertReflection);
        m_fragReflection = std::move(other.m_fragReflection);
        m_descriptorBindings = std::move(other.m_descriptorBindings);
        m_descriptorSetLayouts = std::move(other.m_descriptorSetLayouts);
        m_materialLayoutOwner = std::move(other.m_materialLayoutOwner);
        m_borrowedMaterialLayout = other.m_borrowedMaterialLayout;
        m_pipelineLayout = other.m_pipelineLayout;
        m_materialUBOLayout = std::move(other.m_materialUBOLayout);
        m_hasMaterialUBO = other.m_hasMaterialUBO;
        m_vertexMaterialUBOLayout = std::move(other.m_vertexMaterialUBOLayout);
        m_hasVertexMaterialUBO = other.m_hasVertexMaterialUBO;
        m_bindlessTextureIndexLayout = std::move(other.m_bindlessTextureIndexLayout);
        m_hasBindlessTextureIndexLayout = other.m_hasBindlessTextureIndexLayout;
        m_usesBindlessTextureABI = other.m_usesBindlessTextureABI;

        other.m_device = VK_NULL_HANDLE;
        other.m_vertModule = VK_NULL_HANDLE;
        other.m_fragModule = VK_NULL_HANDLE;
        other.m_pipelineLayout = VK_NULL_HANDLE;
        other.m_borrowedMaterialLayout = VK_NULL_HANDLE;
        other.m_hasMaterialUBO = false;
        other.m_hasVertexMaterialUBO = false;
        other.m_hasBindlessTextureIndexLayout = false;
        other.m_usesBindlessTextureABI = false;
        other.m_descriptorSetLayouts.clear();
    }
    return *this;
}

bool ShaderProgram::Create(VkDevice device, const std::vector<char> &vertSpirv, const std::vector<char> &fragSpirv,
                           const ShaderProgramVariantKey &variantKey, const ShaderProgramMetadata &metadata)
{
    m_device = device;
    m_variantKey = variantKey;
    m_shaderId = variantKey.ToString();
    m_vertexEntryPoint = metadata.vertexEntryPoint;
    m_fragmentEntryPoint = metadata.fragmentEntryPoint;
    m_domain = metadata.domain;
    m_materialLayoutOwner = metadata.materialLayoutOwner;

    const auto select = [](const ShaderTargetCode &code, std::string_view name) -> ShaderStageSelection {
        const auto entry = std::find_if(code.entries.begin(), code.entries.end(),
                                        [&](const auto &value) { return value.name == name; });
        if (entry == code.entries.end())
            throw std::invalid_argument("Graphics program entry is absent from its compiler layout: " +
                                        std::string(name));
        return {code, static_cast<uint32_t>(entry - code.entries.begin())};
    };
    ShaderStageLayout linked;
    try {
        linked = LinkShaderStages({select(metadata.vertexTarget, metadata.vertexEntryPoint),
                                   select(metadata.fragmentTarget, metadata.fragmentEntryPoint)});
    } catch (const std::invalid_argument &error) {
        INXLOG_ERROR("Shader program '", variantKey.program.stages.ToString(), "': ", error.what());
        return false; // The publication cache records this failed revision once.
    }
    MergeBindings(linked.resources);

    // Create shader modules
    m_vertModule = CreateShaderModule(vertSpirv);
    if (m_vertModule == VK_NULL_HANDLE) {
        INXLOG_ERROR("Failed to create vertex shader module for program: ", variantKey.program.stages.ToString());
        return false;
    }

    m_fragModule = CreateShaderModule(fragSpirv);
    if (m_fragModule == VK_NULL_HANDLE) {
        INXLOG_ERROR("Failed to create fragment shader module for program: ", variantKey.program.stages.ToString());
        vkDestroyShaderModule(m_device, m_vertModule, nullptr);
        m_vertModule = VK_NULL_HANDLE;
        return false;
    }

    m_vertReflection.LoadTargetLayout(metadata.vertexTarget, metadata.vertexEntryPoint);
    m_fragReflection.LoadTargetLayout(metadata.fragmentTarget, metadata.fragmentEntryPoint);

    const bool hasReflectedBindlessTextureABI =
        std::any_of(m_descriptorBindings.begin(), m_descriptorBindings.end(), [](const auto &binding) {
            return binding.set == BindlessTextureSet && binding.binding == BindlessTextureBinding &&
                   binding.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && binding.descriptorCount != 1;
        });
    if (hasReflectedBindlessTextureABI &&
        (!IsBindlessTextureEnabled() || GetBindlessTextureDescSetLayout() == VK_NULL_HANDLE)) {
        INXLOG_ERROR("Shader program '", variantKey.program.stages.ToString(),
                     "' contains the bindless material texture ABI, but the active device does not provide the "
                     "required descriptor-indexing table; use the bounded sampler compilation");
        Destroy();
        return false;
    }
    m_usesBindlessTextureABI = hasReflectedBindlessTextureABI;

    // Extract material UBO layout
    ExtractMaterialUBOLayout();

    // Create descriptor set layouts
    if (!CreateDescriptorSetLayouts()) {
        INXLOG_ERROR("Failed to create descriptor set layouts for program: ", variantKey.program.stages.ToString());
        Destroy();
        return false;
    }

    // Create pipeline layout
    if (!CreatePipelineLayout()) {
        INXLOG_ERROR("Failed to create pipeline layout for program: ", variantKey.program.stages.ToString());
        Destroy();
        return false;
    }

    // INXLOG_INFO("Created shader program: ", shaderId, " with ", m_descriptorBindings.size(), " bindings");
    return true;
}

void ShaderProgram::Destroy()
{
    if (m_device == VK_NULL_HANDLE) {
        return;
    }

    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }

    for (auto &[set, layout] : m_descriptorSetLayouts) {
        // Skip the shared globals layout — it is owned by VkCore and
        // destroyed in DestroyGlobalsDescriptorResources().
        if (layout != VK_NULL_HANDLE && layout != s_globalsDescSetLayout && layout != s_perViewDescSetLayout &&
            layout != s_bindlessTextureDescSetLayout && layout != m_borrowedMaterialLayout) {
            vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
        }
    }
    m_descriptorSetLayouts.clear();
    m_materialLayoutOwner.reset();
    m_borrowedMaterialLayout = VK_NULL_HANDLE;

    if (m_fragModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device, m_fragModule, nullptr);
        m_fragModule = VK_NULL_HANDLE;
    }

    if (m_vertModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device, m_vertModule, nullptr);
        m_vertModule = VK_NULL_HANDLE;
    }

    m_descriptorBindings.clear();
    m_hasMaterialUBO = false;
    m_hasVertexMaterialUBO = false;
    m_bindlessTextureIndexLayout = {};
    m_hasBindlessTextureIndexLayout = false;
    m_usesBindlessTextureABI = false;
    m_device = VK_NULL_HANDLE;
}

VkShaderModule ShaderProgram::CreateShaderModule(const std::vector<char> &code)
{
    if (code.empty()) {
        INXLOG_ERROR("Cannot create shader module from empty code");
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule module;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        INXLOG_ERROR("vkCreateShaderModule failed");
        return VK_NULL_HANDLE;
    }

    return module;
}

void ShaderProgram::MergeBindings(const std::vector<ShaderResourceLayout> &resources)
{
    m_descriptorBindings.clear();
    m_descriptorBindings.reserve(resources.size());
    constexpr std::array types{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                               VK_DESCRIPTOR_TYPE_SAMPLER,        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    for (const auto &resource : resources) {
        const auto &binding = resource.binding;
        const bool readOnly = binding.type == rhi::BindingType::StorageBuffer
                                  ? binding.readOnlyStorage
                                  : binding.type == rhi::BindingType::StorageTexture &&
                                        binding.storageAccess == rhi::StorageTextureAccess::ReadOnly;
        m_descriptorBindings.push_back({binding.binding, resource.set, types[static_cast<size_t>(binding.type)],
                                        binding.count, rhi::ToVkShaderStages(binding.visibility), resource.name,
                                        readOnly});
    }
}

void ShaderProgram::ExtractMaterialUBOLayout()
{
    m_hasMaterialUBO = false;
    m_hasVertexMaterialUBO = false;
    m_hasBindlessTextureIndexLayout = false;

    const uint32_t materialSet = m_domain == ShaderProgramDomain::ParticleSprite ? 2u : MaterialDescriptorSet;

    const auto extract = [](const UniformBufferInfo &ubo) {
        return MaterialUBOLayout{ubo.binding, ubo.size, ubo.members};
    };

    // Material ownership comes from the program domain and binding set, not
    // the compiler's emitted block name. Slang may mangle ConstantBuffer names.
    for (const auto &ubo : m_fragReflection.GetUniformBuffers()) {
        if (ubo.name == "InxMaterialTextureIndices") {
            m_bindlessTextureIndexLayout = extract(ubo);
            m_hasBindlessTextureIndexLayout = true;
            continue;
        }
        if (ubo.set == materialSet) {
            m_materialUBOLayout = extract(ubo);
            m_hasMaterialUBO = true;

            break;
        }
    }

    // The geometry ABI places vertex material parameters at binding 14.
    for (const auto &ubo : m_vertReflection.GetUniformBuffers()) {
        if (ubo.set == materialSet && ubo.binding == 14) {
            m_vertexMaterialUBOLayout = extract(ubo);
            m_hasVertexMaterialUBO = true;

            break;
        }
    }

    // A linked program exposes one canonical MaterialProperties block to both
    // stages. When reflection reports the same layout twice, bind one buffer
    // with merged stage visibility instead of allocating a duplicate vertex UBO.
    if (m_hasMaterialUBO && m_hasVertexMaterialUBO &&
        m_materialUBOLayout.binding == m_vertexMaterialUBOLayout.binding &&
        m_materialUBOLayout.size == m_vertexMaterialUBOLayout.size &&
        m_materialUBOLayout.members == m_vertexMaterialUBOLayout.members) {
        m_vertexMaterialUBOLayout = {};
        m_hasVertexMaterialUBO = false;
    }
}

bool ShaderProgram::CreateDescriptorSetLayouts()
{
    // Group bindings by descriptor set
    std::unordered_map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> setBindings;
    if (m_materialLayoutOwner) {
        m_borrowedMaterialLayout = m_materialLayoutOwner->GetDescriptorSetLayout(MaterialDescriptorSet);
        m_descriptorSetLayouts[MaterialDescriptorSet] = m_borrowedMaterialLayout;
    }

    for (const auto &merged : m_descriptorBindings) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = merged.binding;
        binding.descriptorType = merged.type;
        binding.descriptorCount = merged.descriptorCount;
        binding.stageFlags = merged.stageFlags;
        binding.pImmutableSamplers = nullptr;

        setBindings[merged.set].push_back(binding);
    }

    if (m_usesBindlessTextureABI) {
        uint32_t textureIndexSet = 0;
        uint32_t textureIndexBinding = MaterialTextureIndexBinding;
        const auto reflectedIndexBuffer =
            std::find_if(m_fragReflection.GetUniformBuffers().begin(), m_fragReflection.GetUniformBuffers().end(),
                         [](const auto &buffer) { return buffer.name == "InxMaterialTextureIndices"; });
        if (reflectedIndexBuffer != m_fragReflection.GetUniformBuffers().end()) {
            textureIndexSet = reflectedIndexBuffer->set;
            textureIndexBinding = reflectedIndexBuffer->binding;
        }
        auto &materialBindings = setBindings[textureIndexSet];
        const auto existing =
            std::find_if(materialBindings.begin(), materialBindings.end(),
                         [textureIndexBinding](const auto &binding) { return binding.binding == textureIndexBinding; });
        if (existing == materialBindings.end()) {
            VkDescriptorSetLayoutBinding indexBinding{};
            indexBinding.binding = textureIndexBinding;
            indexBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            indexBinding.descriptorCount = 1;
            indexBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            materialBindings.push_back(indexBinding);
        } else if (existing->descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || existing->descriptorCount != 1) {
            INXLOG_ERROR("Bindless material ABI reserves set ", textureIndexSet, " binding ", textureIndexBinding,
                         " for a uniform buffer, but shader ", m_variantKey.program.stages.ToString(),
                         " declares another descriptor type");
            return false;
        }
    }

    // Create a layout for each set.
    //
    // For the material set (set 0) we retain UPDATE_AFTER_BIND compatibility when
    // supported. MaterialDescriptorManager now publishes immutable replacement sets,
    // so live-update correctness no longer depends on mutating an in-flight set.
    //
    // Sets 1, 2 (per-view, globals) keep the legacy layout — they are written once
    // per frame on the CPU before any submission and have no live-update concern.
    const bool enableUpdateAfterBind = IsUpdateAfterBindEnabled();
    for (auto &[setIndex, bindings] : setBindings) {
        if (setIndex == MaterialDescriptorSet && m_materialLayoutOwner)
            continue;
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        std::vector<VkDescriptorBindingFlags> bindingFlags;
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
        if (enableUpdateAfterBind && setIndex == MaterialDescriptorSet) {
            bindingFlags.assign(bindings.size(), VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                                     VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
            bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
            bindingFlagsInfo.pBindingFlags = bindingFlags.data();
            layoutInfo.pNext = &bindingFlagsInfo;
            layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        }

        VkDescriptorSetLayout layout;
        if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create descriptor set layout for set ", setIndex);
            return false;
        }

        m_descriptorSetLayouts[setIndex] = layout;
    }

    // If no bindings, create an empty layout for set 0
    if (m_descriptorSetLayouts.empty()) {
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 0;
        layoutInfo.pBindings = nullptr;

        VkDescriptorSetLayout layout;
        if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create empty descriptor set layout");
            return false;
        }
        m_descriptorSetLayouts[MaterialDescriptorSet] = layout;
    }

    return true;
}

bool ShaderProgram::CreatePipelineLayout()
{
    if (s_perViewDescSetLayout != VK_NULL_HANDLE) {
        auto it = m_descriptorSetLayouts.find(ViewDescriptorSet);
        if (it != m_descriptorSetLayouts.end() && it->second != s_perViewDescSetLayout)
            vkDestroyDescriptorSetLayout(m_device, it->second, nullptr);
        m_descriptorSetLayouts[ViewDescriptorSet] = s_perViewDescSetLayout;
    }

    // If a globals descriptor set layout was registered, ensure set 2 exists
    // in the layout map. Replace any reflection-created set 2 with the
    // canonical engine layout so descriptor set compatibility is guaranteed.
    if (s_globalsDescSetLayout != VK_NULL_HANDLE) {
        // If reflection already created a set 2, destroy it — we use the shared one
        auto it = m_descriptorSetLayouts.find(EngineDescriptorSet);
        if (it != m_descriptorSetLayouts.end() && it->second != s_globalsDescSetLayout) {
            vkDestroyDescriptorSetLayout(m_device, it->second, nullptr);
        }
        m_descriptorSetLayouts[EngineDescriptorSet] = s_globalsDescSetLayout;
    }

    if (m_usesBindlessTextureABI) {
        auto it = m_descriptorSetLayouts.find(BindlessTextureSet);
        if (it != m_descriptorSetLayouts.end() && it->second != s_bindlessTextureDescSetLayout)
            vkDestroyDescriptorSetLayout(m_device, it->second, nullptr);
        m_descriptorSetLayouts[BindlessTextureSet] = s_bindlessTextureDescSetLayout;
    }

    // Get layouts in order (set 0, 1, 2, ...)
    std::vector<VkDescriptorSetLayout> layouts;
    uint32_t maxSet = 0;

    for (const auto &[setIndex, layout] : m_descriptorSetLayouts) {
        maxSet = std::max(maxSet, setIndex);
    }

    for (uint32_t i = 0; i <= maxSet; ++i) {
        auto it = m_descriptorSetLayouts.find(i);
        if (it != m_descriptorSetLayouts.end()) {
            layouts.push_back(it->second);
        } else {
            // Create empty layout for gaps
            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 0;
            layoutInfo.pBindings = nullptr;

            VkDescriptorSetLayout emptyLayout;
            if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &emptyLayout) != VK_SUCCESS) {
                INXLOG_ERROR("Failed to create empty descriptor set layout for gap at set ", i);
                return false;
            }
            m_descriptorSetLayouts[i] = emptyLayout;
            layouts.push_back(emptyLayout);
        }
    }

    // Collect push constant ranges
    std::vector<VkPushConstantRange> pushConstantRanges;
    for (const auto &pc : m_vertReflection.GetPushConstants()) {
        VkPushConstantRange range{};
        range.stageFlags = pc.stageFlags;
        range.offset = pc.offset;
        range.size = pc.size;
        pushConstantRanges.push_back(range);
    }
    for (const auto &pc : m_fragReflection.GetPushConstants()) {
        // Check if we need to merge with existing range
        bool merged = false;
        for (auto &existing : pushConstantRanges) {
            if (existing.offset == pc.offset && existing.size == pc.size) {
                existing.stageFlags |= pc.stageFlags;
                merged = true;
                break;
            }
        }
        if (!merged) {
            VkPushConstantRange range{};
            range.stageFlags = pc.stageFlags;
            range.offset = pc.offset;
            range.size = pc.size;
            pushConstantRanges.push_back(range);
        }
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
    pipelineLayoutInfo.pSetLayouts = layouts.empty() ? nullptr : layouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
    pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges.empty() ? nullptr : pushConstantRanges.data();

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create pipeline layout");
        return false;
    }

    return true;
}

bool ShaderProgram::HasDeclaredDescriptorSet(uint32_t set) const
{
    return std::any_of(m_descriptorBindings.begin(), m_descriptorBindings.end(),
                       [set](const MergedDescriptorBinding &binding) { return binding.set == set; });
}

uint32_t ShaderProgram::GetTextureBindingCount() const
{
    uint32_t count = 0;
    for (const auto &binding : m_descriptorBindings) {
        if (binding.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            binding.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
            count += binding.descriptorCount;
        }
    }
    return count;
}

// ============================================================================
// ShaderProgramCache Implementation
// ============================================================================

void ShaderProgramCache::Initialize(VkDevice device)
{
    m_device = device;
}

void ShaderProgramCache::Shutdown()
{
    // Shared publications may still be held by scene materials after the
    // cache and render-data maps release their own references. Destroy the
    // Vulkan objects while the device is valid; the later shared_ptr
    // destructors then perform an idempotent no-op.
    for (auto &weakProgram : m_devicePrograms) {
        if (auto program = weakProgram.lock())
            program->Destroy();
    }
    m_devicePrograms.clear();
    Clear();
    m_device = VK_NULL_HANDLE;
}

ShaderProgramPublication ShaderProgramCache::GetOrCreateProgram(const ShaderProgramVariantKey &variantKey,
                                                                const std::vector<char> &vertSpirv,
                                                                const std::vector<char> &fragSpirv,
                                                                const ShaderProgramMetadata &metadata)
{
    const ShaderProgramVariantKey canonicalKey = CanonicalKey(variantKey);
    const ProgramCacheKey cacheKey{canonicalKey, metadata.materialLayoutOwner};
    // Check cache first
    auto it = m_programs.find(cacheKey);
    if (it != m_programs.end()) {
        return it->second;
    }

    // Check if this program previously failed creation (don't retry every frame)
    if (m_failedPrograms.count(cacheKey)) {
        return nullptr;
    }

    // Create new program
    auto program = std::make_shared<ShaderProgram>();
    if (!program->Create(m_device, vertSpirv, fragSpirv, canonicalKey, metadata)) {
        INXLOG_ERROR("Failed to create shader program: ", canonicalKey.program.stages.ToString(), " (",
                     ShaderCompileTargetName(canonicalKey.target), ")");
        m_failedPrograms.insert(cacheKey);
        return nullptr;
    }

    if (m_devicePrograms.size() == m_devicePrograms.capacity()) {
        m_devicePrograms.erase(std::remove_if(m_devicePrograms.begin(), m_devicePrograms.end(),
                                              [](const auto &weakProgram) { return weakProgram.expired(); }),
                               m_devicePrograms.end());
    }
    m_devicePrograms.emplace_back(program);
    ShaderProgramPublication publication = std::move(program);
    m_programs[cacheKey] = publication;
    return publication;
}

ShaderProgramPublication ShaderProgramCache::GetProgram(const ShaderProgramKey &programKey) const
{
    return GetProgram({programKey, ShaderCompileTarget::Forward});
}

ShaderProgramPublication ShaderProgramCache::GetProgram(const ShaderProgramVariantKey &variantKey,
                                                        const ShaderProgramPublication &materialLayoutOwner) const
{
    const ProgramCacheKey key{CanonicalKey(variantKey), materialLayoutOwner};
    auto it = m_programs.find(key);
    return it != m_programs.end() ? it->second : nullptr;
}

bool ShaderProgramCache::HasProgram(const ShaderProgramKey &programKey) const
{
    return HasProgram({programKey, ShaderCompileTarget::Forward});
}

bool ShaderProgramCache::HasProgram(const ShaderProgramVariantKey &variantKey,
                                    const ShaderProgramPublication &materialLayoutOwner) const
{
    return m_programs.find(ProgramCacheKey{CanonicalKey(variantKey), materialLayoutOwner}) != m_programs.end();
}

ShaderProgramPublication ShaderProgramCache::TakeProgram(const ShaderProgramKey &programKey)
{
    return TakeProgram({programKey, ShaderCompileTarget::Forward});
}

ShaderProgramPublication ShaderProgramCache::TakeProgram(const ShaderProgramVariantKey &variantKey,
                                                         const ShaderProgramPublication &materialLayoutOwner)
{
    const ProgramCacheKey key{CanonicalKey(variantKey), materialLayoutOwner};
    auto found = m_programs.find(key);
    if (found == m_programs.end()) {
        m_failedPrograms.erase(key);
        return nullptr;
    }
    auto program = std::move(found->second);
    m_programs.erase(found);
    m_failedPrograms.erase(key);
    return program;
}

std::vector<ShaderProgramPublication> ShaderProgramCache::TakePrograms(const ShaderProgramKey &programKey)
{
    // Remove every actual layout owner and device-contract incarnation of
    // this artifact directly; re-canonicalizing an old key could miss one.
    std::vector<ShaderProgramPublication> programs;
    for (auto it = m_programs.begin(); it != m_programs.end();) {
        if (it->first.variant.program == programKey) {
            programs.push_back(std::move(it->second));
            it = m_programs.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = m_failedPrograms.begin(); it != m_failedPrograms.end();) {
        if (it->variant.program == programKey)
            it = m_failedPrograms.erase(it);
        else
            ++it;
    }
    return programs;
}

std::vector<ShaderProgramPublication> ShaderProgramCache::TakeProgramsContainingShader(const std::string &shaderName)
{
    if (shaderName.empty())
        throw std::invalid_argument("Shader program invalidation requires a non-empty shader identifier");
    const auto matchesShader = [&](const ShaderStagePair &stages) {
        const auto matchesStage = [&](const std::string &stageId) {
            return stageId == shaderName || stageId.rfind(shaderName + "/", 0) == 0;
        };
        return matchesStage(stages.vertexKey) || matchesStage(stages.fragmentKey);
    };
    std::vector<ShaderProgramPublication> retired;
    for (auto it = m_programs.begin(); it != m_programs.end();) {
        if (matchesShader(it->first.variant.program.stages)) {
            retired.push_back(std::move(it->second));
            it = m_programs.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = m_failedPrograms.begin(); it != m_failedPrograms.end();) {
        if (matchesShader(it->variant.program.stages))
            it = m_failedPrograms.erase(it);
        else
            ++it;
    }
    return retired;
}
void ShaderProgramCache::Clear()
{
    m_programs.clear();
    m_failedPrograms.clear();
}

} // namespace infernux
