/**
 * @file VkCoreMaterial.cpp
 * @brief InfVkCoreModular — Material system, lighting, and buffer accessors
 *
 * Split from InfVkCoreModular.cpp for maintainability.
 * Contains: UpdateMaterialUBO, EnsureMaterialUBO, CreateBuffer,
 *           InitializeMaterialSystem, RefreshMaterialPipeline,
 *           SetAmbientColor, UpdateLightingUBO,
 *           GetObjectBuffer, GetUniformBuffer, GetShaderModule.
 */

#include "InfError.h"
#include "InfVkCoreModular.h"

#include <function/renderer/shader/ShaderProgram.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InfMaterial/InfMaterial.h>
#include <function/resources/InfResource/InfResourceMeta.h>
#include <function/resources/InfTexture/InfTexture.h>

#include <algorithm>
#include <cctype>
#include <glm/glm.hpp>

#include <cstring>

namespace infengine
{

static std::string ToLowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

static bool IsLinearMaterialTextureBinding(const std::string &bindingName)
{
    const std::string lower = ToLowerCopy(bindingName);
    return lower.find("normal") != std::string::npos || lower.find("metal") != std::string::npos ||
           lower.find("rough") != std::string::npos || lower.find("smooth") != std::string::npos ||
           lower.find("ao") != std::string::npos || lower.find("occlusion") != std::string::npos ||
           lower.find("mask") != std::string::npos || lower.find("height") != std::string::npos;
}

// ============================================================================
// Shared texture resolution for material Texture2D properties
// ============================================================================

std::pair<VkImageView, VkSampler> InfVkCoreModular::ResolveTextureForMaterial(const std::string &textureRef,
                                                                              const std::string &bindingName)
{
    // Resolve texture GUID → file path via AssetDatabase.
    std::string textureGuid = textureRef;
    std::string texturePath;
    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    if (adb) {
        std::string resolved = adb->GetPathFromGuid(textureRef);
        if (!resolved.empty()) {
            texturePath = resolved;
        }
    }

    if (texturePath.empty()) {
        // Could not resolve to a file path — GUID mapping is stale or asset deleted
        INFLOG_WARN("TextureResolver: cannot resolve texture reference '", textureRef, "' to a file path (binding='",
                    bindingName, "')");
        return {VK_NULL_HANDLE, VK_NULL_HANDLE};
    }

    // ── Load InfTexture via AssetRegistry (caches import settings) ──────────
    // This replaces the ad-hoc .meta reading that was here before.
    bool isLinearTexture = IsLinearMaterialTextureBinding(bindingName);
    bool generateMipmaps = true;
    bool normalMapMode = false;
    int maxSize = 0; // 0 = no clamping

    auto infTex = registry.LoadAsset<InfTexture>(textureGuid, ResourceType::Texture);
    if (infTex) {
        // InfTexture import settings take full precedence over binding-name heuristic.
        // Two-way: if texture says sRGB, honour it even when the binding name
        // would otherwise default to linear (e.g. user overrides a normal-map slot).
        isLinearTexture = infTex->IsLinear();
        generateMipmaps = infTex->GenerateMipmaps();
        normalMapMode = infTex->IsNormalMapMode();
        maxSize = infTex->GetMaxSize();
    } else {
        // Fallback: read .meta directly (texture not in AssetDatabase, e.g. engine-internal)
        // Use explicit metadata values as the single source of truth.
        std::string metaPath = InfResourceMeta::GetMetaFilePath(texturePath);
        InfResourceMeta meta;
        if (meta.LoadFromFile(metaPath)) {
            if (meta.HasKey("texture_type")) {
                normalMapMode = meta.GetDataAs<std::string>("texture_type") == "normal_map";
            }
            if (meta.HasKey("srgb")) {
                isLinearTexture = !meta.GetDataAs<bool>("srgb");
            }
            if (meta.HasKey("generate_mipmaps")) {
                generateMipmaps = meta.GetDataAs<bool>("generate_mipmaps");
            }
            if (meta.HasKey("max_size")) {
                maxSize = meta.GetDataAs<int>("max_size");
            }
        }
    }

    VkFormat format = isLinearTexture ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_SRGB;

    // Cache key uses GUID so that a renamed file still shares its cache entry
    std::string cacheKey =
        textureGuid + (isLinearTexture ? "::unorm" : "::srgb") + (normalMapMode ? "::normalmap" : "::raw");

    // Check texture cache (thread-safe)
    {
        auto *cached = m_textureCache.Find(cacheKey);
        if (cached) {
            return {cached->GetView(), cached->GetSampler()};
        }
    }

    // Load texture from disk → GPU with correct format, mipmaps, and size limit
    auto texture = m_resourceManager.LoadTexture(texturePath, generateMipmaps, format, maxSize, normalMapMode);
    if (!texture) {
        INFLOG_WARN("TextureResolver: failed to load '", texturePath, "'");
        return {VK_NULL_HANDLE, VK_NULL_HANDLE};
    }

    VkImageView view = texture->GetView();
    VkSampler sampler = texture->GetSampler();
    m_textureCache.Insert(cacheKey, std::move(texture));
    INFLOG_INFO("TextureResolver: loaded texture '", texturePath, "' (", isLinearTexture ? "UNORM" : "SRGB",
                ", binding='", bindingName, "', key='", cacheKey, "')");
    return {view, sampler};
}

// ============================================================================
// Material UBO Management
// ============================================================================

void InfVkCoreModular::UpdateMaterialUBO(InfMaterial &material)
{
    if (!material.IsPropertiesDirty()) {
        return;
    }

    if (m_materialPipelineManagerInitialized && material.GetPassShaderProgram(ShaderCompileTarget::Forward) &&
        material.GetPassDescriptorSet(ShaderCompileTarget::Forward) != VK_NULL_HANDLE) {
        m_materialPipelineManager.UpdateMaterialProperties(material.GetMaterialKey(), material);
        material.ClearPropertiesDirty();
        return;
    }

    ShaderProgram *shaderProgram = material.GetPassShaderProgram(ShaderCompileTarget::Forward);
    const MaterialUBOLayout *uboLayout = shaderProgram ? shaderProgram->GetMaterialUBOLayout() : nullptr;

    // Use reflection size when available; legacy fallback uses 256 bytes
    size_t uboSize = (uboLayout && uboLayout->size > 0) ? uboLayout->size : 256;

    const auto &properties = material.GetAllProperties();

    std::vector<uint8_t> uboData(uboSize, 0);

    if (uboLayout && !uboLayout->members.empty()) {
        for (const auto &[name, prop] : properties) {
            uint32_t memberOffset = 0;
            uint32_t memberSize = 0;

            if (!uboLayout->GetMemberInfo(name, memberOffset, memberSize)) {
                continue;
            }

            switch (prop.type) {
            case MaterialPropertyType::Float4:
            case MaterialPropertyType::Color: {
                if (memberOffset + sizeof(glm::vec4) <= uboSize) {
                    glm::vec4 value = std::get<glm::vec4>(prop.value);
                    std::memcpy(uboData.data() + memberOffset, &value, sizeof(glm::vec4));
                }
                break;
            }
            case MaterialPropertyType::Float3: {
                if (memberOffset + sizeof(glm::vec3) <= uboSize) {
                    glm::vec3 value = std::get<glm::vec3>(prop.value);
                    std::memcpy(uboData.data() + memberOffset, &value, sizeof(glm::vec3));
                }
                break;
            }
            case MaterialPropertyType::Float2: {
                if (memberOffset + sizeof(glm::vec2) <= uboSize) {
                    glm::vec2 value = std::get<glm::vec2>(prop.value);
                    std::memcpy(uboData.data() + memberOffset, &value, sizeof(glm::vec2));
                }
                break;
            }
            case MaterialPropertyType::Float: {
                if (memberOffset + sizeof(float) <= uboSize) {
                    float value = std::get<float>(prop.value);
                    std::memcpy(uboData.data() + memberOffset, &value, sizeof(float));
                }
                break;
            }
            case MaterialPropertyType::Int: {
                if (memberOffset + sizeof(int) <= uboSize) {
                    int value = std::get<int>(prop.value);
                    std::memcpy(uboData.data() + memberOffset, &value, sizeof(int));
                }
                break;
            }
            default:
                break;
            }
        }
    } else {
        size_t offset = 0;

        for (const auto &[name, prop] : properties) {
            if (prop.type == MaterialPropertyType::Float4) {
                offset = (offset + 15) & ~15;
                if (offset + sizeof(glm::vec4) <= uboSize) {
                    glm::vec4 value = std::get<glm::vec4>(prop.value);
                    std::memcpy(uboData.data() + offset, &value, sizeof(glm::vec4));
                    offset += sizeof(glm::vec4);
                }
            }
        }

        for (const auto &[name, prop] : properties) {
            if (prop.type == MaterialPropertyType::Float3) {
                offset = (offset + 15) & ~15;
                if (offset + sizeof(glm::vec3) <= uboSize) {
                    glm::vec3 value = std::get<glm::vec3>(prop.value);
                    std::memcpy(uboData.data() + offset, &value, sizeof(glm::vec3));
                    offset += 16;
                }
            }
        }

        for (const auto &[name, prop] : properties) {
            if (prop.type == MaterialPropertyType::Float2) {
                offset = (offset + 7) & ~7;
                if (offset + sizeof(glm::vec2) <= uboSize) {
                    glm::vec2 value = std::get<glm::vec2>(prop.value);
                    std::memcpy(uboData.data() + offset, &value, sizeof(glm::vec2));
                    offset += sizeof(glm::vec2);
                }
            }
        }

        for (const auto &[name, prop] : properties) {
            if (prop.type == MaterialPropertyType::Float) {
                offset = (offset + 3) & ~3;
                if (offset + sizeof(float) <= uboSize) {
                    float value = std::get<float>(prop.value);
                    std::memcpy(uboData.data() + offset, &value, sizeof(float));
                    offset += sizeof(float);
                }
            }
        }

        for (const auto &[name, prop] : properties) {
            if (prop.type == MaterialPropertyType::Int) {
                offset = (offset + 3) & ~3;
                if (offset + sizeof(int) <= uboSize) {
                    int value = std::get<int>(prop.value);
                    std::memcpy(uboData.data() + offset, &value, sizeof(int));
                    offset += sizeof(int);
                }
            }
        }
    }

    if (material.HasUBO()) {
        void *matMappedData = material.GetUBOMappedData();
        if (matMappedData) {
            std::memcpy(matMappedData, uboData.data(), uboSize);
        }
    } else {
        for (size_t i = 0; i < m_materialUboMapped.size(); ++i) {
            if (m_materialUboMapped[i]) {
                std::memcpy(m_materialUboMapped[i], uboData.data(), uboSize);
            }
        }
    }

    material.ClearPropertiesDirty();
}

void InfVkCoreModular::EnsureMaterialUBO(std::shared_ptr<InfMaterial> material)
{
    if (!material) {
        return;
    }

    if (material->HasUBO()) {
        return;
    }

    VkBuffer uboBuffer = VK_NULL_HANDLE;
    VmaAllocation uboAllocation = VK_NULL_HANDLE;
    void *uboMappedData = nullptr;

    // Use reflection size when available; legacy fallback uses 256 bytes
    ShaderProgram *shaderProgram = material->GetPassShaderProgram(ShaderCompileTarget::Forward);
    const MaterialUBOLayout *uboLayout = shaderProgram ? shaderProgram->GetMaterialUBOLayout() : nullptr;
    size_t uboSize = (uboLayout && uboLayout->size > 0) ? uboLayout->size : 256;
    CreateBuffer(uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboBuffer, uboAllocation);

    VmaAllocator allocator = m_deviceContext.GetVmaAllocator();
    vmaMapMemory(allocator, uboAllocation, &uboMappedData);
    if (uboMappedData) {
        std::memset(uboMappedData, 0, uboSize);
    }

    material->SetUBOBuffer(allocator, uboBuffer, uboAllocation, uboMappedData);
}

// ============================================================================
// Material / Pipeline System
// ============================================================================

void InfVkCoreModular::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                    VkBuffer &buffer, VmaAllocation &allocation)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocator allocator = m_deviceContext.GetVmaAllocator();
    VmaAllocationCreateInfo allocCreateInfo{};

    if (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            allocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        } else {
            allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        }
    } else if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocCreateInfo.requiredFlags = properties;
        if (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        } else {
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        }
    } else {
        allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    }

    VkResult result = vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &buffer, &allocation, nullptr);
    if (result != VK_SUCCESS) {
        INFLOG_ERROR("CreateBuffer: failed to create buffer via VMA");
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
    }
}

void InfVkCoreModular::InitializeMaterialSystem()
{
    if (m_materialSystemInitialized) {
        return;
    }

    AssetRegistry::Instance().InitializeBuiltinMaterials();

    if (!m_materialPipelineManagerInitialized) {
        // Use SceneRenderTarget-compatible formats: HDR R16G16B16A16_SFLOAT color + device depth format
        VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkFormat depthFormat = m_deviceContext.FindDepthFormat();
        m_materialPipelineManager.Initialize(m_deviceContext.GetVmaAllocator(), GetDevice(), GetPhysicalDevice(),
                                             colorFormat, depthFormat, m_msaaSampleCount,
                                             m_shaderCache.GetProgramCache(), &m_deletionQueue);
        m_materialPipelineManagerInitialized = true;

        auto *whiteTex = m_textureCache.Find("white");
        if (whiteTex) {
            m_materialPipelineManager.SetDefaultTexture(whiteTex->GetView(), whiteTex->GetSampler());
        }

        auto *normalTex = m_textureCache.Find("_default_normal");
        if (normalTex) {
            m_materialPipelineManager.SetDefaultNormalTexture(normalTex->GetView(), normalTex->GetSampler());
        }

        // Set up texture resolver for material Texture2D properties
        // Delegates to ResolveTextureForMaterial which uses GUID-based cache keys.
        m_materialPipelineManager.SetTextureResolver(
            [this](const std::string &textureRef, const std::string &bindingName) -> std::pair<VkImageView, VkSampler> {
                return ResolveTextureForMaterial(textureRef, bindingName);
            });
    }

    auto defaultMaterial = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
    if (defaultMaterial) {
        const std::string &vertId = defaultMaterial->GetVertShaderName();
        const std::string &fragId = defaultMaterial->GetFragShaderName();

        const auto *vertCode = m_shaderCache.FindVertCode(vertId);
        const auto *fragCode = m_shaderCache.FindFragCode(fragId);

        if (vertCode && fragCode) {
            VkBuffer lightingBuffer =
                m_lightingUboBuffers.empty() ? VK_NULL_HANDLE : m_lightingUboBuffers[0]->GetBuffer();
            m_materialPipelineManager.GetOrCreateRenderDataWithReflection(
                defaultMaterial, *vertCode, *fragCode, defaultMaterial->GetShaderId(),
                m_uniformBuffers.empty() ? VK_NULL_HANDLE : m_uniformBuffers[0]->GetBuffer(),
                sizeof(UniformBufferObject), lightingBuffer, sizeof(ShaderLightingUBO));
        } else {
            INFLOG_ERROR("InitializeMaterialSystem: SPIR-V shader codes not found for default material "
                         "(vert='",
                         vertId, "', frag='", fragId, "'). Reflection path requires shader code cache.");
        }
    }

    // Pre-build error material pipeline (unlit magenta-black checkerboard).
    auto errorMaterial = AssetRegistry::Instance().GetBuiltinMaterial("ErrorMaterial");
    if (errorMaterial) {
        const std::string &errVertId = errorMaterial->GetVertShaderName();
        const std::string &errFragId = errorMaterial->GetFragShaderName();

        const auto *errVertCode = m_shaderCache.FindVertCode(errVertId);
        const auto *errFragCode = m_shaderCache.FindFragCode(errFragId);

        if (errVertCode && errFragCode) {
            VkBuffer lightingBuffer =
                m_lightingUboBuffers.empty() ? VK_NULL_HANDLE : m_lightingUboBuffers[0]->GetBuffer();
            auto *renderData = m_materialPipelineManager.GetOrCreateRenderDataWithReflection(
                errorMaterial, *errVertCode, *errFragCode, errorMaterial->GetShaderId(),
                m_uniformBuffers.empty() ? VK_NULL_HANDLE : m_uniformBuffers[0]->GetBuffer(),
                sizeof(UniformBufferObject), lightingBuffer, sizeof(ShaderLightingUBO));
            if (renderData && renderData->isValid) {
                INFLOG_INFO("Error material pipeline created successfully (shaders: ", errVertId, "/", errFragId, ")");
            } else {
                INFLOG_WARN("InitializeMaterialSystem: error material pipeline deferred to lazy build");
            }
        } else {
            INFLOG_WARN("InitializeMaterialSystem: error shader SPIR-V not yet in cache "
                        "(vert='",
                        errVertId, "', frag='", errFragId, "'), will be built lazily on first use");
        }
    }

    m_materialSystemInitialized = true;
}

void InfVkCoreModular::ReinitializeMaterialPipelines(VkSampleCountFlagBits newSampleCount)
{
    if (!m_materialPipelineManagerInitialized) {
        return;
    }

    INFLOG_INFO("ReinitializeMaterialPipelines: changing MSAA sample count to ", static_cast<int>(newSampleCount));

    // Shutdown existing pipelines (caller must have called WaitIdle already)
    m_materialPipelineManager.Shutdown(/* skipWaitIdle */ true);
    m_materialPipelineManagerInitialized = false;

    // Re-initialize with new sample count
    VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat depthFormat = m_deviceContext.FindDepthFormat();
    m_materialPipelineManager.Initialize(m_deviceContext.GetVmaAllocator(), GetDevice(), GetPhysicalDevice(),
                                         colorFormat, depthFormat, newSampleCount, m_shaderCache.GetProgramCache(),
                                         &m_deletionQueue);
    m_materialPipelineManagerInitialized = true;

    // Restore default textures
    auto *whiteTex = m_textureCache.Find("white");
    if (whiteTex) {
        m_materialPipelineManager.SetDefaultTexture(whiteTex->GetView(), whiteTex->GetSampler());
    }
    auto *normalTex = m_textureCache.Find("_default_normal");
    if (normalTex) {
        m_materialPipelineManager.SetDefaultNormalTexture(normalTex->GetView(), normalTex->GetSampler());
    }

    // Restore texture resolver
    m_materialPipelineManager.SetTextureResolver(
        [this](const std::string &textureRef, const std::string &bindingName) -> std::pair<VkImageView, VkSampler> {
            return ResolveTextureForMaterial(textureRef, bindingName);
        });

    INFLOG_INFO("ReinitializeMaterialPipelines: complete — pipelines will be lazily rebuilt on next draw");
}

bool InfVkCoreModular::RefreshMaterialPipeline(std::shared_ptr<InfMaterial> material, const std::string &vertShaderName,
                                               const std::string &fragShaderName)
{
    if (!material) {
        return false;
    }

    // Apply shader render-state annotations to the material before pipeline creation.
    // Fragment shader annotations take priority (they define the surface behaviour).
    const auto *renderMeta = m_shaderCache.GetRenderMeta(fragShaderName);
    if (renderMeta) {
        material->ApplyShaderRenderMeta(renderMeta->cullMode, renderMeta->depthWrite, renderMeta->depthTest,
                                        renderMeta->blend, renderMeta->queue, renderMeta->passTag, renderMeta->stencil,
                                        renderMeta->alphaClip);
    }

    const auto *vertCode = m_shaderCache.FindVertCode(vertShaderName);
    const auto *fragCode = m_shaderCache.FindFragCode(fragShaderName);

    if (vertCode && fragCode && m_materialPipelineManagerInitialized) {
        VkBuffer sceneUbo = m_uniformBuffers.empty() ? VK_NULL_HANDLE : m_uniformBuffers[0]->GetBuffer();
        VkDeviceSize sceneUboSize = sizeof(UniformBufferObject);
        VkBuffer lightingUbo = m_lightingUboBuffers.empty() ? VK_NULL_HANDLE : m_lightingUboBuffers[0]->GetBuffer();
        VkDeviceSize lightingUboSize = sizeof(ShaderLightingUBO);
        auto *renderData = m_materialPipelineManager.GetOrCreateRenderDataWithReflection(
            material, *vertCode, *fragCode, material->GetShaderId(), sceneUbo, sceneUboSize, lightingUbo,
            lightingUboSize);

        bool forwardOk = renderData && renderData->isValid;

        if (forwardOk && m_shadowPipelineReady) {
            std::string shadowFragName = fragShaderName + "/shadow";
            bool hasShadowFrag = (GetShaderModule(shadowFragName, "fragment") != VK_NULL_HANDLE);
            if (hasShadowFrag) {
                VkPipeline oldShadow = material->GetPassPipeline(ShaderCompileTarget::Shadow);
                if (oldShadow != VK_NULL_HANDLE) {
                    VkDevice dev = GetDevice();
                    m_deletionQueue.Push([dev, oldShadow]() { vkDestroyPipeline(dev, oldShadow, nullptr); });
                    material->SetPassPipeline(ShaderCompileTarget::Shadow, VK_NULL_HANDLE);
                }

                CreateMaterialShadowPipeline(material, vertShaderName, fragShaderName);
            }
        }

        return forwardOk;
    }

    INFLOG_WARN("RefreshMaterialPipeline: shader codes not found or MPM not initialized for '", material->GetName(),
                "' (vert='", vertShaderName, "', frag='", fragShaderName, "')");

    // Dump available shader keys for debugging
    static int dumpCount = 0;
    if (dumpCount++ < 2) {
        std::string vertKeys, fragKeys;
        m_shaderCache.DumpAvailableKeys(vertKeys, fragKeys);
        INFLOG_WARN("  Available vert shaders:", vertKeys);
        INFLOG_WARN("  Available frag shaders:", fragKeys);
        INFLOG_WARN("  MPM initialized: ", m_materialPipelineManagerInitialized ? "true" : "false",
                    ", vertCode found: ", (vertCode ? "yes" : "no"), ", fragCode found: ", (fragCode ? "yes" : "no"));
    }
    return false;
}

// ============================================================================
// Lighting System
// ============================================================================

void InfVkCoreModular::SetAmbientColor(const glm::vec3 &color, float intensity)
{
    m_lightCollector.SetAmbientColor(color, intensity);
    INFLOG_DEBUG("SetAmbientColor: (", color.r, ", ", color.g, ", ", color.b, ") intensity=", intensity);
}

void InfVkCoreModular::UpdateLightingUBO(const glm::vec3 &cameraPosition)
{
    // Delegate to StageLightingUBO — the actual GPU write now happens
    // inline in the command buffer via CmdUpdateLightingUBO().
    StageLightingUBO(cameraPosition);
}

void InfVkCoreModular::StageLightingUBO(const glm::vec3 &cameraPosition)
{
    // Phase 2.1: Sync ambient color from skybox material properties
    auto skyMat = AssetRegistry::Instance().GetBuiltinMaterial("SkyboxProcedural");
    if (skyMat) {
        const auto *skyTopProp = skyMat->GetProperty("skyTopColor");
        const auto *horizonProp = skyMat->GetProperty("skyHorizonColor");
        const auto *groundProp = skyMat->GetProperty("groundColor");
        const auto *exposureProp = skyMat->GetProperty("exposure");
        if (skyTopProp && groundProp) {
            glm::vec3 skyTop = glm::vec3(std::get<glm::vec4>(skyTopProp->value));
            glm::vec3 ground = glm::vec3(std::get<glm::vec4>(groundProp->value));
            glm::vec3 equator;
            if (horizonProp) {
                equator = glm::vec3(std::get<glm::vec4>(horizonProp->value));
            } else {
                equator = glm::mix(ground, skyTop, 0.5f);
            }
            float exposure = 0.8f;
            if (exposureProp) {
                exposure = std::get<float>(exposureProp->value);
            }
            m_lightCollector.SetAmbientGradient(skyTop * exposure, equator * exposure, ground * exposure);
        }
    }

    // Build the shader-compatible UBO from collected lights
    m_lightCollector.BuildShaderLightingUBO();
    m_stagedLightingUBO = m_lightCollector.GetShaderLightingUBO();
    m_stagedLightingUBO.cameraPos = glm::vec4(cameraPosition, 1.0f);
    m_lightingUBODirty = true;
}

void InfVkCoreModular::CmdUpdateLightingCameraPos(VkCommandBuffer cmdBuf, const glm::vec3 &cameraPos)
{
    if (m_lightingUboBuffers.empty() || !m_lightingUboBuffers[0])
        return;

    VkBuffer buffer = m_lightingUboBuffers[0]->GetBuffer();

    // cameraPos sits at offset 32 in ShaderLightingUBO (after lightCounts + ambientColor).
    constexpr VkDeviceSize cameraPosOffset = offsetof(ShaderLightingUBO, cameraPos);
    glm::vec4 cameraPosVec4(cameraPos, 1.0f);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    vkCmdUpdateBuffer(cmdBuf, buffer, cameraPosOffset, sizeof(glm::vec4), &cameraPosVec4);

    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
}

void InfVkCoreModular::CmdUpdateLightingUBO(VkCommandBuffer cmdBuf)
{
    if (!m_lightingUBODirty)
        return;
    if (m_lightingUboBuffers.empty() || !m_lightingUboBuffers[0])
        return;

    VkBuffer buffer = m_lightingUboBuffers[0]->GetBuffer();

    // Barrier: ensure previous shader reads from the lighting UBO are complete
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Update the lighting UBO inline in the command buffer
    // vkCmdUpdateBuffer has a 65536-byte limit; ShaderLightingUBO is well within that.
    vkCmdUpdateBuffer(cmdBuf, buffer, 0, sizeof(ShaderLightingUBO), &m_stagedLightingUBO);

    // Barrier: ensure write is visible before subsequent shader reads
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);

    m_lightingUBODirty = false;
}

void InfVkCoreModular::CmdUpdateShadowDataForCamera(VkCommandBuffer cmdBuf, const glm::mat4 *lightVPs,
                                                    uint32_t cascadeCount, const float *cascadeSplits,
                                                    float mapResolution)
{
    if (m_lightingUboBuffers.empty() || !m_lightingUboBuffers[0])
        return;

    VkBuffer buffer = m_lightingUboBuffers[0]->GetBuffer();

    // Build the shadow portion we need to patch
    glm::mat4 vpData[NUM_SHADOW_CASCADES];
    for (uint32_t i = 0; i < NUM_SHADOW_CASCADES; ++i)
        vpData[i] = (i < cascadeCount) ? lightVPs[i] : glm::mat4(1.0f);

    glm::vec4 splitVec(cascadeCount > 0 ? cascadeSplits[0] : 0.0f, cascadeCount > 1 ? cascadeSplits[1] : 0.0f,
                       cascadeCount > 2 ? cascadeSplits[2] : 0.0f, cascadeCount > 3 ? cascadeSplits[3] : 0.0f);

    float cascadeRes = mapResolution * 0.5f;
    glm::vec4 params(mapResolution, cascadeCount > 0 ? 1.0f : 0.0f, static_cast<float>(cascadeCount), cascadeRes);

    // Offsets into ShaderLightingUBO
    constexpr VkDeviceSize vpOffset = offsetof(ShaderLightingUBO, lightVP);
    constexpr VkDeviceSize splitOffset = offsetof(ShaderLightingUBO, shadowCascadeSplits);
    constexpr VkDeviceSize paramsOffset = offsetof(ShaderLightingUBO, shadowMapParams);

    // Barrier before writes
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    vkCmdUpdateBuffer(cmdBuf, buffer, vpOffset, sizeof(vpData), vpData);
    vkCmdUpdateBuffer(cmdBuf, buffer, splitOffset, sizeof(glm::vec4), &splitVec);
    vkCmdUpdateBuffer(cmdBuf, buffer, paramsOffset, sizeof(glm::vec4), &params);

    // Also update per-cascade shadow UBOs (used by shadow caster rendering)
    const uint32_t frameIndex = m_currentFrame % m_maxFramesInFlight;
    struct ShadowUBO
    {
        glm::mat4 model, view, proj;
    };
    for (uint32_t ci = 0; ci < cascadeCount && ci < NUM_SHADOW_CASCADES; ++ci) {
        uint32_t bufIdx = frameIndex * NUM_SHADOW_CASCADES + ci;
        if (bufIdx >= m_shadowUboBuffers.size())
            break;
        VkBuffer shadowBuffer = m_shadowUboBuffers[bufIdx];
        if (shadowBuffer == VK_NULL_HANDLE)
            continue;
        ShadowUBO ubo{glm::mat4(1.0f), glm::mat4(1.0f), lightVPs[ci]};
        vkCmdUpdateBuffer(cmdBuf, shadowBuffer, 0, sizeof(ShadowUBO), &ubo);
    }

    // Barrier after writes
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
}

// ============================================================================
// Buffer / Shader Accessors (for OutlineRenderer)
// ============================================================================

VkBuffer InfVkCoreModular::GetObjectVertexBuffer(uint64_t objectId) const
{
    auto it = m_perObjectBuffers.find(objectId);
    if (it != m_perObjectBuffers.end() && it->second.vertexBuffer)
        return it->second.vertexBuffer->GetBuffer();
    return VK_NULL_HANDLE;
}

VkBuffer InfVkCoreModular::GetObjectIndexBuffer(uint64_t objectId) const
{
    auto it = m_perObjectBuffers.find(objectId);
    if (it != m_perObjectBuffers.end() && it->second.indexBuffer)
        return it->second.indexBuffer->GetBuffer();
    return VK_NULL_HANDLE;
}

VkBuffer InfVkCoreModular::GetUniformBuffer(size_t index) const
{
    if (index < m_uniformBuffers.size() && m_uniformBuffers[index])
        return m_uniformBuffers[index]->GetBuffer();
    return VK_NULL_HANDLE;
}

VkShaderModule InfVkCoreModular::GetShaderModule(const std::string &name, const std::string &type) const
{
    return m_shaderCache.GetModule(name, type);
}

// ============================================================================
// Per-material shadow pipeline creation
// ============================================================================

void InfVkCoreModular::CreateMaterialShadowPipeline(std::shared_ptr<InfMaterial> material,
                                                    const std::string &vertShaderName,
                                                    const std::string &fragShaderName)
{
    // Shared shadow resources must be ready
    if (m_shadowCompatRenderPass == VK_NULL_HANDLE || m_shadowPipelineLayout == VK_NULL_HANDLE)
        return;

    VkDevice device = GetDevice();

    // Vertex shader: prefer shadow vertex variant, fall back to forward pass vertex shader
    std::string shadowVertName = vertShaderName + "/shadow";
    std::string shadowFragName = fragShaderName + "/shadow";

    VkShaderModule vertModule = GetShaderModule(shadowVertName, "vertex");
    if (vertModule == VK_NULL_HANDLE)
        vertModule = GetShaderModule(vertShaderName, "vertex");

    // Fragment shader: only create a per-material shadow pipeline when a
    // generated shadow variant exists for this material.
    VkShaderModule fragModule = GetShaderModule(shadowFragName, "fragment");
    if (fragModule == VK_NULL_HANDLE) {
        static int s_missingShadowFragWarnCount = 0;
        if (s_missingShadowFragWarnCount++ < 16) {
            INFLOG_WARN("CreateMaterialShadowPipeline: missing shadow fragment module '", shadowFragName,
                        "' for material '", material->GetName(), "'");
        }
        return;
    }

    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        static int s_missingShadowModuleWarnCount = 0;
        if (s_missingShadowModuleWarnCount++ < 16) {
            INFLOG_WARN("CreateMaterialShadowPipeline: shader modules unavailable for material '", material->GetName(),
                        "' (vert='", shadowVertName, "' fallback='", vertShaderName, "', frag='", shadowFragName, "')");
        }
        return;
    }

    // Shader stages
    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {vertStage, fragStage};

    // Vertex input — same layout as the main scene rendering
    auto bindingDesc = Vertex::getBindingDescription();
    auto attrDescs = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInput.pVertexAttributeDescriptions = attrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Rasterization: front-face culling + depth bias (matches EnsureShadowPipeline)
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 1.5f;
    rasterizer.depthBiasSlopeFactor = 1.0f;
    rasterizer.depthBiasClamp = 0.01f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 0;

    std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_shadowPipelineLayout;
    pipelineInfo.renderPass = m_shadowCompatRenderPass;
    pipelineInfo.subpass = 0;

    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPipeline) != VK_SUCCESS) {
        INFLOG_WARN("Failed to create per-material shadow pipeline for '", material->GetName(), "' (vert='",
                    shadowVertName, "', frag='", shadowFragName, "')");
        return;
    }

    material->SetPassPipeline(ShaderCompileTarget::Shadow, shadowPipeline);
    INFLOG_DEBUG("Created per-material shadow pipeline for '", material->GetName(), "'");
}

} // namespace infengine
