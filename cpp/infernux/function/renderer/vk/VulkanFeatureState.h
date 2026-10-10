#pragma once

#include <function/renderer/rhi/RhiCapabilities.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>

namespace infernux::vk
{

/// Vulkan-only feature negotiation; this is not part of the plugin RHI ABI.
enum class DeviceCapability : uint8_t
{
    DescriptorIndexing = 0,
    DynamicRendering,
    TimelineSemaphore,
    Synchronization2,
    Submit2,
    ShaderInt16,
    ShaderInt64,
    ShaderFloat64,
};

enum class DeviceCapabilityDiagnosticCode : uint8_t
{
    None = 0,
    Unsupported,
    NotEnabled,
    IncompleteDescriptorIndexing,
};

struct DeviceCapabilityStatus final
{
    bool supported = false;
    bool enabled = false;

    [[nodiscard]] constexpr bool IsSupported() const noexcept
    {
        return supported;
    }

    [[nodiscard]] constexpr bool IsEnabled() const noexcept
    {
        return enabled;
    }
};

/// Descriptor-indexing is deliberately represented by the individual Vulkan
/// subfeatures. A single descriptorIndexing boolean is not sufficient to
/// prove that a bindless layout can be created.
struct BindlessCapabilityStatus final
{
    DeviceCapabilityStatus descriptorIndexing;
    DeviceCapabilityStatus runtimeDescriptorArray;
    DeviceCapabilityStatus shaderSampledImageArrayNonUniformIndexing;
    DeviceCapabilityStatus descriptorBindingPartiallyBound;
    DeviceCapabilityStatus descriptorBindingVariableDescriptorCount;
    DeviceCapabilityStatus descriptorBindingSampledImageUpdateAfterBind;
    DeviceCapabilityStatus descriptorBindingUniformBufferUpdateAfterBind;
    DeviceCapabilityStatus descriptorBindingStorageBufferUpdateAfterBind;
    DeviceCapabilityStatus descriptorBindingUpdateUnusedWhilePending;

    [[nodiscard]] constexpr bool IsSupported() const noexcept
    {
        return descriptorIndexing.supported && runtimeDescriptorArray.supported &&
               shaderSampledImageArrayNonUniformIndexing.supported && descriptorBindingPartiallyBound.supported &&
               descriptorBindingVariableDescriptorCount.supported &&
               descriptorBindingSampledImageUpdateAfterBind.supported;
    }

    [[nodiscard]] constexpr bool IsEnabled() const noexcept
    {
        return descriptorIndexing.enabled && runtimeDescriptorArray.enabled &&
               shaderSampledImageArrayNonUniformIndexing.enabled && descriptorBindingPartiallyBound.enabled &&
               descriptorBindingVariableDescriptorCount.enabled && descriptorBindingSampledImageUpdateAfterBind.enabled;
    }
};

struct VulkanDescriptorLimits final
{
    uint32_t maxUpdateAfterBindDescriptors = 0;
    uint32_t maxUpdateAfterBindResourcesPerStage = 0;
    uint32_t maxUpdateAfterBindSamplersPerStage = 0;
    uint32_t maxUpdateAfterBindSampledTexturesPerStage = 0;
    uint32_t maxUpdateAfterBindSamplersPerSet = 0;
    uint32_t maxUpdateAfterBindSampledTexturesPerSet = 0;
};

struct VulkanFeatureState final
{
    VulkanDescriptorLimits descriptorLimits;
    /// Publish only features enabled on the logical device. Physical support
    /// alone must never select a shader variant that cannot execute.
    constexpr void PublishShaderCapabilities(rhi::PortableCaps &caps) const noexcept
    {
        caps.bindlessSampledTextures = bindless.IsEnabled();
        caps.shaderInt16 = shaderInt16.IsEnabled();
        caps.shaderInt64 = shaderInt64.IsEnabled();
        caps.shaderFloat64 = shaderFloat64.IsEnabled();
    }

    BindlessCapabilityStatus bindless;
    DeviceCapabilityStatus dynamicRendering;
    DeviceCapabilityStatus timelineSemaphore;
    DeviceCapabilityStatus synchronization2;
    DeviceCapabilityStatus submit2;
    // Arithmetic capabilities only: neither 16-bit storage nor 64-bit atomics
    // are implied by these shader scalar types.
    DeviceCapabilityStatus shaderInt16;
    DeviceCapabilityStatus shaderInt64;
    DeviceCapabilityStatus shaderFloat64;

    [[nodiscard]] constexpr DeviceCapabilityStatus Get(DeviceCapability capability) const noexcept
    {
        switch (capability) {
        case DeviceCapability::DescriptorIndexing:
            return {bindless.IsSupported(), bindless.IsEnabled()};
        case DeviceCapability::DynamicRendering:
            return dynamicRendering;
        case DeviceCapability::TimelineSemaphore:
            return timelineSemaphore;
        case DeviceCapability::Synchronization2:
            return synchronization2;
        case DeviceCapability::Submit2:
            return submit2;
        case DeviceCapability::ShaderInt16:
            return shaderInt16;
        case DeviceCapability::ShaderInt64:
            return shaderInt64;
        case DeviceCapability::ShaderFloat64:
            return shaderFloat64;
        }
        return {};
    }
};

struct DeviceCapabilityRequest final
{
    bool descriptorIndexing = false;
    bool dynamicRendering = false;
    bool timelineSemaphore = false;
    bool synchronization2 = false;
    bool submit2 = false;
    bool shaderInt16 = false;
    bool shaderInt64 = false;
    bool shaderFloat64 = false;
};

struct DeviceCapabilityCheck final
{
    DeviceCapabilityDiagnosticCode code = DeviceCapabilityDiagnosticCode::None;
    DeviceCapability capability = DeviceCapability::DescriptorIndexing;

    [[nodiscard]] constexpr bool IsSupported() const noexcept
    {
        return code == DeviceCapabilityDiagnosticCode::None;
    }

    [[nodiscard]] constexpr std::string_view Message() const noexcept
    {
        switch (code) {
        case DeviceCapabilityDiagnosticCode::None:
            return {};
        case DeviceCapabilityDiagnosticCode::Unsupported:
            return "requested Vulkan capability is not supported by the physical device";
        case DeviceCapabilityDiagnosticCode::NotEnabled:
            return "requested Vulkan capability is supported but was not enabled at device creation";
        case DeviceCapabilityDiagnosticCode::IncompleteDescriptorIndexing:
            return "descriptor indexing is incomplete for the requested bindless contract";
        }
        return "unknown Vulkan capability diagnostic";
    }
};

[[nodiscard]] constexpr DeviceCapabilityCheck CheckDeviceCapability(const VulkanFeatureState &state,
                                                                    DeviceCapability capability) noexcept
{
    const auto status = state.Get(capability);
    if (!status.supported)
        return {DeviceCapabilityDiagnosticCode::Unsupported, capability};
    if (!status.enabled)
        return {DeviceCapabilityDiagnosticCode::NotEnabled, capability};
    return {};
}

[[nodiscard]] constexpr DeviceCapabilityCheck CheckDeviceCapabilities(const VulkanFeatureState &state,
                                                                      const DeviceCapabilityRequest &request) noexcept
{
    if (request.descriptorIndexing) {
        if (!state.bindless.IsSupported())
            return {DeviceCapabilityDiagnosticCode::IncompleteDescriptorIndexing, DeviceCapability::DescriptorIndexing};
        if (!state.bindless.IsEnabled())
            return {DeviceCapabilityDiagnosticCode::NotEnabled, DeviceCapability::DescriptorIndexing};
    }
    const DeviceCapability requested[] = {DeviceCapability::DynamicRendering, DeviceCapability::TimelineSemaphore,
                                          DeviceCapability::Synchronization2, DeviceCapability::Submit2,
                                          DeviceCapability::ShaderInt16,      DeviceCapability::ShaderInt64,
                                          DeviceCapability::ShaderFloat64};
    const bool enabled[] = {request.dynamicRendering, request.timelineSemaphore, request.synchronization2,
                            request.submit2,          request.shaderInt16,       request.shaderInt64,
                            request.shaderFloat64};
    for (size_t i = 0; i < std::size(requested); ++i) {
        if (enabled[i]) {
            const auto result = CheckDeviceCapability(state, requested[i]);
            if (!result.IsSupported())
                return result;
        }
    }
    return {};
}

} // namespace infernux::vk
