#pragma once

#include "RhiQuery.h"
#include "RhiTypes.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace infernux::rhi
{

enum class BackendType : uint8_t
{
    Unknown = 0,
    Vulkan,
    // Source-compatibility bridge for out-of-tree adapters. New code uses
    // DeviceCaps::backendId so plugins do not require a central enum entry.
    WebGPU,
};

/// Stable backend identity carried across the RHI seam. The core treats this
/// as an opaque identifier and never dispatches on a plugin-specific enum.
struct BackendId final
{
    static constexpr size_t Capacity = 32;

    std::array<char, Capacity> value{};

    constexpr BackendId() noexcept = default;

    explicit BackendId(std::string_view id) noexcept
    {
        Set(id);
    }

    void Set(std::string_view id) noexcept
    {
        value.fill('\0');
        const size_t count = id.size() < Capacity - 1 ? id.size() : Capacity - 1;
        for (size_t index = 0; index < count; ++index)
            value[index] = id[index];
    }

    [[nodiscard]] constexpr std::string_view View() const noexcept
    {
        size_t length = 0;
        while (length < value.size() && value[length] != '\0')
            ++length;
        return {value.data(), length};
    }

    [[nodiscard]] constexpr bool Empty() const noexcept
    {
        return value[0] == '\0';
    }

    friend constexpr bool operator==(const BackendId &lhs, const BackendId &rhs) noexcept
    {
        return lhs.View() == rhs.View();
    }

    friend constexpr bool operator!=(const BackendId &lhs, const BackendId &rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

inline constexpr std::string_view kVulkanBackendId = "vulkan";
inline constexpr std::string_view kWebGpuBackendId = "webgpu";

enum class AdapterType : uint8_t
{
    Unknown = 0,
    Integrated,
    Discrete,
    Virtual,
    Cpu,
};

enum class FormatFeature : uint16_t
{
    None = 0,
    Sampled = 1u << 0,
    FilterLinear = 1u << 1,
    Storage = 1u << 2,
    ColorAttachment = 1u << 3,
    DepthStencilAttachment = 1u << 4,
    TransferSource = 1u << 5,
    TransferDestination = 1u << 6,
    BlitSource = 1u << 7,
    BlitDestination = 1u << 8,
};

[[nodiscard]] constexpr FormatFeature operator|(FormatFeature lhs, FormatFeature rhs) noexcept
{
    return static_cast<FormatFeature>(static_cast<uint16_t>(lhs) | static_cast<uint16_t>(rhs));
}

constexpr FormatFeature &operator|=(FormatFeature &lhs, FormatFeature rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

[[nodiscard]] constexpr bool HasAllFormatFeatures(FormatFeature available, FormatFeature required) noexcept
{
    return (static_cast<uint16_t>(available) & static_cast<uint16_t>(required)) == static_cast<uint16_t>(required);
}

using SampleCountMask = uint8_t;

[[nodiscard]] constexpr SampleCountMask SampleCountBit(SampleCount samples) noexcept
{
    switch (samples) {
    case SampleCount::One:
        return 1u << 0;
    case SampleCount::Two:
        return 1u << 1;
    case SampleCount::Four:
        return 1u << 2;
    case SampleCount::Eight:
        return 1u << 3;
    }
    return 0;
}

enum class CapabilityDiagnosticCode : uint8_t
{
    None = 0,
    InvalidFormat,
    UnsupportedFormatFeatures,
    UnsupportedSampleCount,
};

struct CapabilityCheck
{
    CapabilityDiagnosticCode code = CapabilityDiagnosticCode::None;
    uint64_t required = 0;
    uint64_t available = 0;

    [[nodiscard]] constexpr bool IsSupported() const noexcept
    {
        return code == CapabilityDiagnosticCode::None;
    }
};

struct FormatCapabilities
{
    PixelFormat format = PixelFormat::Undefined;
    FormatFeature optimalTiling = FormatFeature::None;
    SampleCountMask sampleCounts = 0;
};

struct DeviceLimits
{
    uint32_t maxTextureDimension1D = 0;
    uint32_t maxTextureDimension2D = 0;
    uint32_t maxTextureDimension3D = 0;
    uint32_t maxTextureArrayLayers = 0;
    uint32_t maxColorAttachments = 0;
    uint32_t maxPushConstantBytes = 0;
    uint32_t maxBindingLayouts = 0;
    uint32_t maxSampledTexturesPerStage = 0;
    uint32_t maxUpdateAfterBindDescriptors = 0;
    uint32_t maxUpdateAfterBindResourcesPerStage = 0;
    uint32_t maxUpdateAfterBindSamplersPerStage = 0;
    uint32_t maxUpdateAfterBindSampledTexturesPerStage = 0;
    uint32_t maxUpdateAfterBindSamplersPerSet = 0;
    uint32_t maxUpdateAfterBindSampledTexturesPerSet = 0;
    uint32_t maxStorageBuffersPerStage = 0;
    float maxSamplerAnisotropy = 1.0f;
    uint32_t maxComputeWorkgroupCount[3] = {};
    uint32_t maxComputeWorkgroupSize[3] = {};
    uint32_t maxComputeWorkgroupInvocations = 0;
};

struct DeviceFeatures
{
    bool samplerAnisotropy = false;
    bool fillModeNonSolid = false;
    bool wideLines = false;
    bool descriptorIndexing = false;
    bool timelineSemaphore = false;
    bool independentComputeQueue = false;
    bool dedicatedTransferQueue = false;
};

/// Backend-neutral limits and feature declarations. Shared compute, particle,
/// and future NN code consumes this subset; backend adapters may retain their
/// private feature-chain state separately.
struct PortableCaps final
{
    bool bindlessSampledTextures = false;
    uint32_t maxBindlessSampledTextures = 0;
    bool asyncCompute = false;
    bool timelineCompletion = false;
    bool storageTextures = false;
    bool shaderFloat16 = false;
    bool shaderInt64 = false;
    uint32_t maxWorkgroupSize[3] = {};
    uint32_t maxWorkgroupInvocations = 0;
    uint32_t maxStorageBufferBinding = 0;
    uint32_t pushConstantBytes = 0;
    bool mappableReadback = false;
    bool externalMemory = false;
};

struct DeviceCaps
{
    static constexpr size_t AdapterNameCapacity = 128;

    // Kept for source compatibility with existing adapters. Shared code uses
    // backendId/BackendName instead of branching on a central enum.
    BackendType backend = BackendType::Unknown;
    BackendId backendId;
    AdapterType adapterType = AdapterType::Unknown;
    std::array<char, AdapterNameCapacity> adapterName{};
    uint32_t vendorId = 0;
    uint32_t deviceId = 0;
    uint32_t driverVersion = 0;
    uint32_t apiVersionMajor = 0;
    uint32_t apiVersionMinor = 0;
    uint32_t apiVersionPatch = 0;
    DeviceLimits limits;
    DeviceFeatures features;
    PortableCaps portable;
    TimestampQueryCapabilities timestampQueries;
    std::array<FormatCapabilities, kPixelFormatCount> formats{};

    void SetBackendId(std::string_view value) noexcept
    {
        backendId.Set(value);
    }

    [[nodiscard]] std::string_view BackendName() const noexcept
    {
        if (!backendId.Empty())
            return backendId.View();
        switch (backend) {
        case BackendType::Vulkan:
            return kVulkanBackendId;
        case BackendType::WebGPU:
            return kWebGpuBackendId;
        case BackendType::Unknown:
            break;
        }
        return {};
    }

    void SetAdapterName(std::string_view value) noexcept
    {
        adapterName.fill('\0');
        const size_t count = std::min(value.size(), adapterName.size() - 1);
        std::copy_n(value.data(), count, adapterName.data());
    }

    [[nodiscard]] std::string_view AdapterName() const noexcept
    {
        const auto end = std::find(adapterName.begin(), adapterName.end(), '\0');
        return {adapterName.data(), static_cast<size_t>(end - adapterName.begin())};
    }

    [[nodiscard]] const FormatCapabilities *FindFormat(PixelFormat format) const noexcept
    {
        if (!IsValidPixelFormat(format)) {
            return nullptr;
        }
        const size_t index = static_cast<size_t>(format);
        return index < formats.size() ? &formats[index] : nullptr;
    }

    [[nodiscard]] CapabilityCheck CheckFormat(PixelFormat format, FormatFeature required) const noexcept
    {
        const auto *capability = FindFormat(format);
        if (!capability) {
            return {CapabilityDiagnosticCode::InvalidFormat, static_cast<uint64_t>(required), 0};
        }
        if (!HasAllFormatFeatures(capability->optimalTiling, required)) {
            return {CapabilityDiagnosticCode::UnsupportedFormatFeatures, static_cast<uint64_t>(required),
                    static_cast<uint64_t>(capability->optimalTiling)};
        }
        return {};
    }

    [[nodiscard]] CapabilityCheck CheckSampleCount(PixelFormat format, SampleCount samples) const noexcept
    {
        const auto *capability = FindFormat(format);
        const SampleCountMask required = SampleCountBit(samples);
        if (!capability) {
            return {CapabilityDiagnosticCode::InvalidFormat, required, 0};
        }
        if ((capability->sampleCounts & required) == 0) {
            return {CapabilityDiagnosticCode::UnsupportedSampleCount, required, capability->sampleCounts};
        }
        return {};
    }
};

} // namespace infernux::rhi
