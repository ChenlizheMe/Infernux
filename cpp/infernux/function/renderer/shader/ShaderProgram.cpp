#include "ShaderProgram.h"
#include <algorithm>
#include <core/log/InxLog.h>

namespace infernux
{

// ============================================================================
// ShaderProgram Implementation
// ============================================================================

ShaderProgram::~ShaderProgram()
{
    Destroy();
}

ShaderProgram::ShaderProgram(ShaderProgram &&other) noexcept
    : m_device(other.m_device), m_shaderId(std::move(other.m_shaderId)), m_vertModule(other.m_vertModule),
      m_fragModule(other.m_fragModule), m_vertReflection(std::move(other.m_vertReflection)),
      m_fragReflection(std::move(other.m_fragReflection)), m_descriptorBindings(std::move(other.m_descriptorBindings)),
      m_descriptorSetLayouts(std::move(other.m_descriptorSetLayouts)), m_pipelineLayout(other.m_pipelineLayout),
      m_materialUBOLayout(std::move(other.m_materialUBOLayout)), m_hasMaterialUBO(other.m_hasMaterialUBO),
      m_vertexMaterialUBOLayout(std::move(other.m_vertexMaterialUBOLayout)),
      m_hasVertexMaterialUBO(other.m_hasVertexMaterialUBO)
{
    other.m_device = VK_NULL_HANDLE;
    other.m_vertModule = VK_NULL_HANDLE;
    other.m_fragModule = VK_NULL_HANDLE;
    other.m_pipelineLayout = VK_NULL_HANDLE;
    other.m_hasMaterialUBO = false;
    other.m_hasVertexMaterialUBO = false;
    other.m_descriptorSetLayouts.clear();
}

ShaderProgram &ShaderProgram::operator=(ShaderProgram &&other) noexcept
{
    if (this != &other) {
        Destroy();

        m_device = other.m_device;
        m_shaderId = std::move(other.m_shaderId);
        m_vertModule = other.m_vertModule;
        m_fragModule = other.m_fragModule;
        m_vertReflection = std::move(other.m_vertReflection);
        m_fragReflection = std::move(other.m_fragReflection);
        m_descriptorBindings = std::move(other.m_descriptorBindings);
        m_descriptorSetLayouts = std::move(other.m_descriptorSetLayouts);
        m_pipelineLayout = other.m_pipelineLayout;
        m_materialUBOLayout = std::move(other.m_materialUBOLayout);
        m_hasMaterialUBO = other.m_hasMaterialUBO;
        m_vertexMaterialUBOLayout = std::move(other.m_vertexMaterialUBOLayout);
        m_hasVertexMaterialUBO = other.m_hasVertexMaterialUBO;

        other.m_device = VK_NULL_HANDLE;
        other.m_vertModule = VK_NULL_HANDLE;
        other.m_fragModule = VK_NULL_HANDLE;
        other.m_pipelineLayout = VK_NULL_HANDLE;
        other.m_hasMaterialUBO = false;
        other.m_hasVertexMaterialUBO = false;
        other.m_descriptorSetLayouts.clear();
    }
    return *this;
}

bool ShaderProgram::Create(VkDevice device, const std::vector<char> &vertSpirv, const std::vector<char> &fragSpirv,
                           const std::string &shaderId)
{
    m_device = device;
    m_shaderId = shaderId;

    // Create shader modules
    m_vertModule = CreateShaderModule(vertSpirv);
    if (m_vertModule == VK_NULL_HANDLE) {
        INXLOG_ERROR("Failed to create vertex shader module for program: ", shaderId);
        return false;
    }

    m_fragModule = CreateShaderModule(fragSpirv);
    if (m_fragModule == VK_NULL_HANDLE) {
        INXLOG_ERROR("Failed to create fragment shader module for program: ", shaderId);
        vkDestroyShaderModule(m_device, m_vertModule, nullptr);
        m_vertModule = VK_NULL_HANDLE;
        return false;
    }

    // Reflect shader resources
    if (!m_vertReflection.Reflect(vertSpirv, VK_SHADER_STAGE_VERTEX_BIT)) {
        INXLOG_ERROR("Failed to reflect vertex shader: ", shaderId);
    }

    if (!m_fragReflection.Reflect(fragSpirv, VK_SHADER_STAGE_FRAGMENT_BIT)) {
        INXLOG_ERROR("Failed to reflect fragment shader: ", shaderId);
    }

    // Merge reflection data
    MergeReflectionData();

    // Validate vertex→fragment stage interface
    if (!ValidateStageInterface()) {
        INXLOG_ERROR("Shader interface validation failed for program: ", shaderId,
                     ". Vertex outputs and fragment inputs are incompatible.");
        Destroy();
        return false;
    }

    // Extract material UBO layout
    ExtractMaterialUBOLayout();

    // Create descriptor set layouts
    if (!CreateDescriptorSetLayouts()) {
        INXLOG_ERROR("Failed to create descriptor set layouts for program: ", shaderId);
        Destroy();
        return false;
    }

    // Create pipeline layout
    if (!CreatePipelineLayout()) {
        INXLOG_ERROR("Failed to create pipeline layout for program: ", shaderId);
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
        if (layout != VK_NULL_HANDLE && layout != s_globalsDescSetLayout) {
            vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
        }
    }
    m_descriptorSetLayouts.clear();

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

void ShaderProgram::MergeReflectionData()
{
    m_descriptorBindings.clear();

    // Helper to add or merge a binding
    auto addBinding = [this](uint32_t binding, uint32_t set, VkDescriptorType type, uint32_t count,
                             VkShaderStageFlags stage, const std::string &name) {
        // Check if binding already exists
        for (auto &existing : m_descriptorBindings) {
            if (existing.binding == binding && existing.set == set) {
                // Merge stage flags
                existing.stageFlags |= stage;
                return;
            }
        }

        // Add new binding
        MergedDescriptorBinding merged;
        merged.binding = binding;
        merged.set = set;
        merged.type = type;
        merged.descriptorCount = count;
        merged.stageFlags = stage;
        merged.name = name;
        m_descriptorBindings.push_back(merged);
    };

    // Process vertex shader UBOs
    for (const auto &ubo : m_vertReflection.GetUniformBuffers()) {
        addBinding(ubo.binding, ubo.set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, ubo.name);
    }

    // Process vertex shader samplers
    for (const auto &sampler : m_vertReflection.GetSampledImages()) {
        addBinding(sampler.binding, sampler.set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampler.arraySize,
                   VK_SHADER_STAGE_VERTEX_BIT, sampler.name);
    }

    // Process fragment shader UBOs
    for (const auto &ubo : m_fragReflection.GetUniformBuffers()) {
        addBinding(ubo.binding, ubo.set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, ubo.name);
    }

    // Process fragment shader samplers
    for (const auto &sampler : m_fragReflection.GetSampledImages()) {
        addBinding(sampler.binding, sampler.set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampler.arraySize,
                   VK_SHADER_STAGE_FRAGMENT_BIT, sampler.name);
    }

    // Sort by set, then by binding
    std::sort(m_descriptorBindings.begin(), m_descriptorBindings.end(),
              [](const MergedDescriptorBinding &a, const MergedDescriptorBinding &b) {
                  if (a.set != b.set)
                      return a.set < b.set;
                  return a.binding < b.binding;
              });

    INXLOG_DEBUG("Merged ", m_descriptorBindings.size(), " descriptor bindings for shader: ", m_shaderId);
    for (const auto &b : m_descriptorBindings) {
        INXLOG_DEBUG("  binding=", b.binding, " set=", b.set, " type=", static_cast<int>(b.type), " name=", b.name);
    }
}

// Helper: human-readable name for VkFormat
static const char *VkFormatName(VkFormat fmt)
{
    switch (fmt) {
    case VK_FORMAT_R32_SFLOAT:
        return "float";
    case VK_FORMAT_R32G32_SFLOAT:
        return "vec2";
    case VK_FORMAT_R32G32B32_SFLOAT:
        return "vec3";
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return "vec4";
    case VK_FORMAT_R32_SINT:
        return "int";
    case VK_FORMAT_R32G32_SINT:
        return "ivec2";
    case VK_FORMAT_R32G32B32_SINT:
        return "ivec3";
    case VK_FORMAT_R32G32B32A32_SINT:
        return "ivec4";
    default:
        return "unknown";
    }
}

bool ShaderProgram::ValidateStageInterface() const
{
    const auto &vertOutputs = m_vertReflection.GetOutputs();
    const auto &fragInputs = m_fragReflection.GetInputs();
    bool valid = true;

    for (const auto &fragIn : fragInputs) {
        // Find the matching vertex output at the same location
        const ShaderIOVariable *matchedOutput = nullptr;
        for (const auto &vertOut : vertOutputs) {
            if (vertOut.location == fragIn.location) {
                matchedOutput = &vertOut;
                break;
            }
        }

        if (!matchedOutput) {
            INXLOG_ERROR("Shader interface mismatch in '", m_shaderId, "': fragment input '", fragIn.name,
                         "' (location ", fragIn.location, ", type ", VkFormatName(fragIn.format),
                         ") has no matching vertex output. "
                         "The fragment shader will receive undefined values.");
            valid = false;
            continue;
        }

        if (matchedOutput->format != fragIn.format) {
            INXLOG_ERROR("Shader interface mismatch in '", m_shaderId, "': vertex output '", matchedOutput->name,
                         "' is ", VkFormatName(matchedOutput->format), " but fragment input '", fragIn.name,
                         "' expects ", VkFormatName(fragIn.format), " at location ", fragIn.location,
                         ". This will cause a Vulkan validation error and potential GPU crash.");
            valid = false;
        }
    }

    return valid;
}

void ShaderProgram::ExtractMaterialUBOLayout()
{
    m_hasMaterialUBO = false;
    m_hasVertexMaterialUBO = false;

    // Look for a UBO named "MaterialProperties" or "Material" in fragment shader
    // Convention: Material UBO can be at binding 2 or 3 depending on shader type (unlit vs lit)
    for (const auto &ubo : m_fragReflection.GetUniformBuffers()) {
        bool isMaterialUBO =
            (ubo.name == "MaterialProperties" || ubo.name == "Material" || ubo.name == "MaterialUBO" ||
             ((ubo.binding == 2 || ubo.binding == 3) && ubo.set == 0)); // Fallback: binding 2 or 3, set 0

        if (isMaterialUBO) {
            m_materialUBOLayout.binding = ubo.binding;
            m_materialUBOLayout.size = ubo.size;
            m_materialUBOLayout.members = ubo.members;
            m_hasMaterialUBO = true;

            INXLOG_DEBUG("Found material UBO '", ubo.name, "' at binding ", ubo.binding, " with ", ubo.members.size(),
                         " members, size=", ubo.size);
            break;
        }
    }

    // Also look for a vertex-stage MaterialProperties UBO at binding 14
    // (used when the vertex shader declares @property fields)
    for (const auto &ubo : m_vertReflection.GetUniformBuffers()) {
        if ((ubo.name == "MaterialProperties" || ubo.name == "Material" || ubo.name == "MaterialUBO") &&
            ubo.binding == 14 && ubo.set == 0) {
            m_vertexMaterialUBOLayout.binding = ubo.binding;
            m_vertexMaterialUBOLayout.size = ubo.size;
            m_vertexMaterialUBOLayout.members = ubo.members;
            m_hasVertexMaterialUBO = true;

            INXLOG_DEBUG("Found vertex material UBO '", ubo.name, "' at binding 14 with ", ubo.members.size(),
                         " members, size=", ubo.size);
            break;
        }
    }
}

bool ShaderProgram::CreateDescriptorSetLayouts()
{
    // Group bindings by descriptor set
    std::unordered_map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> setBindings;

    for (const auto &merged : m_descriptorBindings) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = merged.binding;
        binding.descriptorType = merged.type;
        binding.descriptorCount = merged.descriptorCount;
        binding.stageFlags = merged.stageFlags;
        binding.pImmutableSamplers = nullptr;

        setBindings[merged.set].push_back(binding);
    }

    // Create a layout for each set
    for (auto &[setIndex, bindings] : setBindings) {
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VkDescriptorSetLayout layout;
        if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create descriptor set layout for set ", setIndex);
            return false;
        }

        m_descriptorSetLayouts[setIndex] = layout;
        INXLOG_DEBUG("Created descriptor set layout for set ", setIndex, " with ", bindings.size(), " bindings");
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
        m_descriptorSetLayouts[0] = layout;
    }

    return true;
}

bool ShaderProgram::CreatePipelineLayout()
{
    // If a globals descriptor set layout was registered, ensure set 2 exists
    // in the layout map. Replace any reflection-created set 2 with the
    // canonical engine layout so descriptor set compatibility is guaranteed.
    if (s_globalsDescSetLayout != VK_NULL_HANDLE) {
        // If reflection already created a set 2, destroy it — we use the shared one
        auto it = m_descriptorSetLayouts.find(2);
        if (it != m_descriptorSetLayouts.end() && it->second != s_globalsDescSetLayout) {
            vkDestroyDescriptorSetLayout(m_device, it->second, nullptr);
        }
        m_descriptorSetLayouts[2] = s_globalsDescSetLayout;
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
        if (binding.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
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
    Clear();
    m_device = VK_NULL_HANDLE;
}

ShaderProgram *ShaderProgramCache::GetOrCreateProgram(const std::string &shaderId, const std::vector<char> &vertSpirv,
                                                      const std::vector<char> &fragSpirv)
{
    // Check cache first
    auto it = m_programs.find(shaderId);
    if (it != m_programs.end()) {
        return it->second.get();
    }

    // Check if this program previously failed creation (don't retry every frame)
    if (m_failedPrograms.count(shaderId)) {
        return nullptr;
    }

    // Create new program
    auto program = std::make_unique<ShaderProgram>();
    if (!program->Create(m_device, vertSpirv, fragSpirv, shaderId)) {
        INXLOG_ERROR("Failed to create shader program: ", shaderId);
        m_failedPrograms.insert(shaderId);
        return nullptr;
    }

    ShaderProgram *result = program.get();
    m_programs[shaderId] = std::move(program);
    return result;
}

ShaderProgram *ShaderProgramCache::GetProgram(const std::string &shaderId)
{
    auto it = m_programs.find(shaderId);
    return it != m_programs.end() ? it->second.get() : nullptr;
}

bool ShaderProgramCache::HasProgram(const std::string &shaderId) const
{
    return m_programs.find(shaderId) != m_programs.end();
}

void ShaderProgramCache::RemoveProgram(const std::string &shaderId)
{
    m_programs.erase(shaderId);
    m_failedPrograms.erase(shaderId);
}

void ShaderProgramCache::RemoveProgramsContainingShader(const std::string &shaderName)
{
    // Helper to extract shader name from path
    auto extractShaderName = [](const std::string &path) -> std::string {
        if (path.empty())
            return "";
        size_t lastSlash = path.find_last_of("/\\");
        std::string fileName = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
        size_t dotPos = fileName.find_last_of('.');
        if (dotPos != std::string::npos) {
            return fileName.substr(0, dotPos);
        }
        return fileName;
    };

    std::vector<std::string> toRemove;
    for (const auto &[key, program] : m_programs) {
        // Key format is "vert_path|frag_path"
        // Check if either path's shader name matches
        size_t pipePos = key.find('|');
        if (pipePos != std::string::npos) {
            std::string vertPath = key.substr(0, pipePos);
            std::string fragPath = key.substr(pipePos + 1);

            if (extractShaderName(vertPath) == shaderName || extractShaderName(fragPath) == shaderName) {
                toRemove.push_back(key);
                INXLOG_DEBUG("Removing cached shader program: ", key);
            }
        } else if (key == shaderName || extractShaderName(key) == shaderName) {
            // Fallback for simple key format
            toRemove.push_back(key);
            INXLOG_DEBUG("Removing cached shader program (simple key): ", key);
        }
    }

    for (const auto &key : toRemove) {
        m_programs.erase(key);
        m_failedPrograms.erase(key);
    }

    INXLOG_INFO("Removed ", toRemove.size(), " shader programs containing shader '", shaderName, "'");
}

void ShaderProgramCache::Clear()
{
    m_programs.clear();
    m_failedPrograms.clear();
}

} // namespace infernux
