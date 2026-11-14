#pragma once

#include "ParticleOutputSemantics.h"

#include <core/types/ShaderProgramArtifact.h>
#include <function/renderer/rhi/RhiTexture.h>
#include <function/renderer/shader/ShaderProgramCode.h>
#include <function/renderer/shader/ShaderValueCopy.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace infernux
{
class InxMaterial;
class GpuRetirementQueue;
} // namespace infernux

namespace infernux::particle
{

struct GpuBillboardMaterialState
{
    int32_t renderQueue = 3000;
    bool blendEnabled = true;
    bool depthTestEnabled = true;
    bool depthWriteEnabled = false;
    bool premultipliedAlpha = false;
};

enum class GpuBillboardTextureStatus : uint8_t
{
    Ready,
    Pending,
    Failed,
};

struct GpuBillboardTextureLease
{
    GpuBillboardTextureStatus status = GpuBillboardTextureStatus::Failed;
    rhi::TextureViewHandle texture;
    rhi::SamplerHandle sampler;
    std::shared_ptr<rhi::TextureGpuViewSlot> gpuSlot;
    std::shared_ptr<const rhi::TextureGpuView> gpuView;
};

enum class GpuParticleTextureRequest : uint8_t
{
    Poll,
    Prepare,
};

using GpuBillboardTextureResolver = std::function<GpuBillboardTextureLease(
    const std::string &textureGuid, const std::string &bindingName, GpuParticleTextureRequest request)>;

/// Owns an authored particle material's reflected set-zero resources.
/// Geometry and view resources are composed from independent programs.
/// Logical values use the same prepared copies as ordinary materials.
class ParticleGpuSurfaceBinding
{
  public:
    ParticleGpuSurfaceBinding() = default;
    ~ParticleGpuSurfaceBinding();

    ParticleGpuSurfaceBinding(const ParticleGpuSurfaceBinding &) = delete;
    ParticleGpuSurfaceBinding &operator=(const ParticleGpuSurfaceBinding &) = delete;
    ParticleGpuSurfaceBinding(ParticleGpuSurfaceBinding &&) = delete;
    ParticleGpuSurfaceBinding &operator=(ParticleGpuSurfaceBinding &&) = delete;

    [[nodiscard]] bool Create(rhi::Device &device, std::shared_ptr<const ShaderProgramArtifact> shaderProgram,
                              std::shared_ptr<InxMaterial> material, GpuBillboardMaterialState fallbackMaterial,
                              ParticleOutputSemantics semantics, GpuBillboardTextureResolver textureResolver,
                              GpuRetirementQueue *deletionQueue);
    void Destroy() noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] rhi::BindingLayoutHandle Layout() const noexcept
    {
        return m_layout;
    }
    [[nodiscard]] std::shared_ptr<const ShaderProgramArtifact> ShaderProgram() const noexcept
    {
        return m_shaderProgram;
    }

    [[nodiscard]] GpuBillboardMaterialState ResolveMaterialState() const noexcept;
    [[nodiscard]] float ResolveMaterialFloat(const char *name, float fallback) const noexcept;

    [[nodiscard]] bool RefreshMaterialBuffer(bool force);
    [[nodiscard]] bool RefreshTextureBindings(bool force);
    [[nodiscard]] rhi::BindGroupHandle ResolveBindGroup() const noexcept
    {
        return m_group;
    }

  private:
    struct TextureBindingState
    {
        uint32_t binding = 0;
        uint32_t samplerBinding = 0;
        std::string name;
        std::string defaultGuid;
        std::string requestedGuid;
        uint64_t requestedTextureAssetRevision = 0;
        rhi::TextureViewHandle texture;
        rhi::SamplerHandle sampler;
        std::shared_ptr<rhi::TextureGpuViewSlot> gpuSlot;
        std::shared_ptr<const rhi::TextureGpuView> gpuView;
        bool pending = false;
        bool fallback = false;
    };

    [[nodiscard]] rhi::BindGroupHandle CreateBindGroup(const std::vector<TextureBindingState> &textures,
                                                       rhi::BufferHandle materialBuffer) const;
    [[nodiscard]] std::string ResolveMaterialTextureGuid(const TextureBindingState &binding) const;
    [[nodiscard]] bool RebuildBindGroup();
    void RetireBindGroup(rhi::BindGroupHandle group);
    void RetireBuffer(rhi::BufferHandle buffer);
    void RetireTexture(std::shared_ptr<const rhi::TextureGpuView> gpuView);

    rhi::Device *m_device = nullptr;
    std::shared_ptr<const ShaderProgramArtifact> m_shaderProgram;
    std::shared_ptr<InxMaterial> m_material;
    GpuBillboardMaterialState m_fallbackMaterial{};
    ParticleOutputSemantics m_semantics{};
    GpuBillboardTextureResolver m_textureResolver;
    GpuRetirementQueue *m_deletionQueue = nullptr;
    rhi::BindingLayoutHandle m_layout;
    rhi::BindGroupHandle m_group;
    rhi::BufferHandle m_materialBuffer;
    std::vector<uint8_t> m_materialBytes;
    const ShaderTargetCode *m_target = nullptr;
    uint32_t m_materialBufferSize = 0;
    std::vector<std::pair<std::string, ShaderValueCopy>> m_memberCopies;
    std::vector<TextureBindingState> m_textures;
    uint64_t m_materialVersion = 0;
    bool m_materialVersionInitialized = false;
};

} // namespace infernux::particle
