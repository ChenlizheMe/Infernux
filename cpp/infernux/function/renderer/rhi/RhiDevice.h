#pragma once

#include "RhiCapabilities.h"
#include "RhiDescriptors.h"
#include "RhiResourceIndex.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <shared_mutex>
#include <string_view>

namespace infernux::rhi
{

/// RHI ABI version. The major component changes whenever a plugin must be
/// rebuilt; the minor component is for additive contracts understood by the
/// existing device vtable.
inline constexpr uint32_t kRhiApiVersionMajor = 2;
inline constexpr uint32_t kRhiApiVersionMinor = 0;
inline constexpr uint32_t kRhiApiVersion = (kRhiApiVersionMajor << 16u) | kRhiApiVersionMinor;

[[nodiscard]] constexpr bool IsRhiApiVersionCompatible(uint32_t version) noexcept
{
    return (version >> 16u) == kRhiApiVersionMajor && (version & 0xffffu) <= kRhiApiVersionMinor;
}

enum class DeviceContractDiagnosticCode : uint8_t
{
    None = 0,
    IncompatibleApiVersion,
    MissingBackendId,
};

struct DeviceContractCheck final
{
    DeviceContractDiagnosticCode code = DeviceContractDiagnosticCode::None;
    uint32_t apiVersion = kRhiApiVersion;

    [[nodiscard]] constexpr bool IsValid() const noexcept
    {
        return code == DeviceContractDiagnosticCode::None;
    }

    [[nodiscard]] constexpr std::string_view Message() const noexcept
    {
        switch (code) {
        case DeviceContractDiagnosticCode::None:
            return {};
        case DeviceContractDiagnosticCode::IncompatibleApiVersion:
            return "RHI device API version is incompatible with this engine";
        case DeviceContractDiagnosticCode::MissingBackendId:
            return "RHI device did not publish a backend identity";
        }
        return "unknown RHI device contract diagnostic";
    }
};

class TextureGpuView;

struct BindlessTextureTableBinding final
{
    BindingLayoutHandle layout;
    BindGroupHandle group;

    [[nodiscard]] constexpr bool IsValid() const noexcept
    {
        return layout.IsValid() && group.IsValid();
    }
};

/// Small deterministic hash used to identify an opaque backend in a shader
/// contract key. This is an identity tag, not a security checksum.
[[nodiscard]] constexpr uint32_t ShaderBackendTag(std::string_view backendId) noexcept
{
    uint32_t hash = 2166136261u;
    for (const char character : backendId) {
        hash ^= static_cast<uint8_t>(character);
        hash *= 16777619u;
    }
    return hash;
}

/// Stable shader-ABI key for enabled descriptor and arithmetic capabilities.
/// Host execution features (dynamic rendering, synchronization2, submit2)
/// deliberately do not participate: they do not change shader code. Backend
/// identity and shader IR version do, so Vulkan and plugin caches cannot alias.
[[nodiscard]] constexpr uint64_t ComputeDeviceShaderContractKey(const PortableCaps &caps, std::string_view backendId,
                                                                uint32_t shaderIrVersion = 1) noexcept
{
    uint64_t key = 0x494e000000000000ull; // INXSH, contract key version 4
    key |= static_cast<uint64_t>(shaderIrVersion & 0xffu) << 40u;
    key |= static_cast<uint64_t>(ShaderBackendTag(backendId) & 0xffffffu) << 16u;
    const bool bindless = caps.bindlessSampledTextures;
    key |= static_cast<uint64_t>(bindless) << 0u;
    key |= static_cast<uint64_t>(caps.shaderInt16) << 1u;
    key |= static_cast<uint64_t>(caps.shaderInt64) << 2u;
    key |= static_cast<uint64_t>(caps.shaderFloat64) << 3u;
    key |= static_cast<uint64_t>(caps.shaderFloat16) << 4u;
    return key;
}

/// Shared lifetime state for resources that can outlive their owning device
/// wrapper. Resources must not call a backend after this flag becomes false.
struct DeviceLifetime final
{
    // Resource releases take a shared lock while device Reset/destruction
    // takes the exclusive lock. The atomic remains useful for cheap validity
    // probes, but is not used as a release-vs-teardown synchronization point.
    mutable std::shared_mutex gate;
    std::atomic<bool> alive{true};
};

/// Backend-neutral resource creation surface. Runtime systems retain only
/// RHI handles; Vulkan/WebGPU ownership stays in the concrete adapter.
class Device
{
  public:
    virtual ~Device() = default;

    [[nodiscard]] virtual uint32_t GetApiVersion() const noexcept
    {
        return kRhiApiVersion;
    }

    [[nodiscard]] virtual DeviceId GetDeviceId() const noexcept
    {
        return InvalidDeviceId;
    }
    [[nodiscard]] virtual const DeviceCaps &GetCapabilities() const noexcept
    {
        static const DeviceCaps empty{};
        return empty;
    }

    [[nodiscard]] virtual std::shared_ptr<DeviceLifetime> GetLifetime() const noexcept
    {
        return {};
    }

    /// Optional device-global sampled-texture table. Backends publish this
    /// capability once the table and its fallback descriptor are complete.
    /// Renderers consume only RHI handles and stable ResourceIndex values.
    [[nodiscard]] virtual BindlessTextureTableBinding GetBindlessTextureTableBinding() const noexcept
    {
        return {};
    }
    [[nodiscard]] virtual ResourceIndex
    PublishBindlessTexture(const std::shared_ptr<const TextureGpuView> &texture) noexcept
    {
        (void)texture;
        return {};
    }
    virtual void MarkBindlessTexturesUsed(const ResourceIndex *resources, size_t count) noexcept
    {
        (void)resources;
        (void)count;
    }

    [[nodiscard]] virtual BufferHandle CreateBuffer(const BufferDesc &desc) = 0;
    [[nodiscard]] virtual TextureHandle CreateTexture(const TextureDesc &desc) = 0;
    [[nodiscard]] virtual TextureViewHandle CreateTextureView(const TextureViewDesc &desc) = 0;
    [[nodiscard]] virtual SamplerHandle CreateSampler(const SamplerDesc &desc) = 0;
    [[nodiscard]] virtual ShaderModuleHandle CreateShaderModule(const ShaderModuleDesc &desc) = 0;
    [[nodiscard]] virtual BindingLayoutHandle CreateBindingLayout(const BindingLayoutDesc &desc) = 0;
    [[nodiscard]] virtual BindGroupHandle CreateBindGroup(const BindGroupDesc &desc) = 0;
    [[nodiscard]] virtual GraphicsPipelineHandle CreateGraphicsPipeline(const GraphicsPipelineDesc &desc) = 0;
    [[nodiscard]] virtual ComputePipelineHandle CreateComputePipeline(const ComputePipelineDesc &desc) = 0;

    virtual bool WriteBuffer(BufferHandle handle, uint64_t offset, const void *data, uint64_t byteSize) = 0;
    /// Borrow a host-visible range; this neither copies nor waits for GPU work.
    /// Caller must retain the buffer, complete conflicting submissions before
    /// mapping, and call UnmapBuffer with the same range/access before submitting
    /// new work or releasing the buffer. Read/ReadWrite require Readback memory;
    /// Write supports Upload and Readback memory. DeviceLocal cannot be mapped.
    /// A null result means unsupported mapping or invalid handle/range/access.
    [[nodiscard]] virtual void *MapBuffer(BufferHandle, uint64_t, uint64_t, BufferMapAccess)
    {
        return nullptr;
    }
    /// Publish CPU writes and end the borrow. Does not release the resource.
    [[nodiscard]] virtual bool UnmapBuffer(BufferHandle, uint64_t, uint64_t, BufferMapAccess)
    {
        return false;
    }
    /// Copy bytes from a host-visible readback buffer after the submission
    /// that populated it has completed. Device-local and upload buffers are
    /// intentionally rejected by concrete backends.
    [[nodiscard]] virtual bool ReadBuffer(BufferHandle handle, uint64_t offset, void *data, uint64_t byteSize)
    {
        (void)handle;
        (void)offset;
        (void)data;
        (void)byteSize;
        return false;
    }

    virtual void Release(BufferHandle handle) noexcept = 0;
    virtual void Release(TextureHandle handle) noexcept = 0;
    virtual void Release(TextureViewHandle handle) noexcept = 0;
    virtual void Release(SamplerHandle handle) noexcept = 0;
    virtual void Release(ShaderModuleHandle handle) noexcept = 0;
    virtual void Release(BindingLayoutHandle handle) noexcept = 0;
    virtual void Release(BindGroupHandle handle) noexcept = 0;
    virtual void Release(GraphicsPipelineHandle handle) noexcept = 0;
    virtual void Release(ComputePipelineHandle handle) noexcept = 0;
};

/// Validate the seam before renderer-specific resources are created. The
/// backend identity is opaque so plugins can add implementations without
/// changing this header; the ABI version remains the compatibility gate.
[[nodiscard]] inline DeviceContractCheck CheckDeviceContract(uint32_t apiVersion,
                                                             const DeviceCaps &capabilities) noexcept
{
    if (!IsRhiApiVersionCompatible(apiVersion))
        return {DeviceContractDiagnosticCode::IncompatibleApiVersion, apiVersion};
    if (capabilities.BackendName().empty())
        return {DeviceContractDiagnosticCode::MissingBackendId, apiVersion};
    return {};
}

} // namespace infernux::rhi
