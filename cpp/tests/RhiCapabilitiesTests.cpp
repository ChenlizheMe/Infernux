#include <function/renderer/rhi/RhiDevice.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <set>
#include <string>

using namespace infernux::rhi;

int main()
{
    DeviceCaps capabilities;
    capabilities.backend = BackendType::Vulkan;
    capabilities.adapterType = AdapterType::Discrete;
    capabilities.SetAdapterName("Test Adapter");
    assert(capabilities.AdapterName() == "Test Adapter");

    auto &rgba = capabilities.formats[static_cast<size_t>(PixelFormat::RGBA8UNorm)];
    rgba.format = PixelFormat::RGBA8UNorm;
    rgba.optimalTiling = FormatFeature::Sampled | FormatFeature::FilterLinear | FormatFeature::ColorAttachment;
    rgba.sampleCounts = SampleCountBit(SampleCount::One) | SampleCountBit(SampleCount::Four);

    assert(capabilities.CheckFormat(PixelFormat::RGBA8UNorm, FormatFeature::Sampled).IsSupported());
    const auto storageCheck = capabilities.CheckFormat(PixelFormat::RGBA8UNorm, FormatFeature::Storage);
    assert(!storageCheck.IsSupported());
    assert(storageCheck.code == CapabilityDiagnosticCode::UnsupportedFormatFeatures);

    assert(capabilities.CheckSampleCount(PixelFormat::RGBA8UNorm, SampleCount::Four).IsSupported());
    const auto sampleCheck = capabilities.CheckSampleCount(PixelFormat::RGBA8UNorm, SampleCount::Eight);
    assert(!sampleCheck.IsSupported());
    assert(sampleCheck.code == CapabilityDiagnosticCode::UnsupportedSampleCount);

    const auto invalidCheck = capabilities.CheckFormat(PixelFormat::Undefined, FormatFeature::Sampled);
    assert(invalidCheck.code == CapabilityDiagnosticCode::InvalidFormat);

    const std::string longName(DeviceCaps::AdapterNameCapacity * 2, 'x');
    capabilities.SetAdapterName(longName);
    assert(capabilities.AdapterName().size() == DeviceCaps::AdapterNameCapacity - 1);

    DeviceCapabilityState state;
    DeviceCapabilityRequest emptyRequest;
    assert(CheckDeviceCapabilities(state, emptyRequest).IsSupported());

    state.dynamicRendering.supported = true;
    const auto notEnabled = CheckDeviceCapability(state, DeviceCapability::DynamicRendering);
    assert(!notEnabled.IsSupported());
    assert(notEnabled.code == DeviceCapabilityDiagnosticCode::NotEnabled);

    state.dynamicRendering.enabled = true;
    assert(CheckDeviceCapability(state, DeviceCapability::DynamicRendering).IsSupported());

    DeviceCapabilityRequest bindlessRequest;
    bindlessRequest.descriptorIndexing = true;
    const auto incompleteBindless = CheckDeviceCapabilities(state, bindlessRequest);
    assert(!incompleteBindless.IsSupported());
    assert(incompleteBindless.code == DeviceCapabilityDiagnosticCode::IncompleteDescriptorIndexing);

    state.bindless.descriptorIndexing.supported = true;
    state.bindless.runtimeDescriptorArray.supported = true;
    state.bindless.shaderSampledImageArrayNonUniformIndexing.supported = true;
    state.bindless.descriptorBindingPartiallyBound.supported = true;
    state.bindless.descriptorBindingVariableDescriptorCount.supported = true;
    state.bindless.descriptorBindingSampledImageUpdateAfterBind.supported = true;
    assert(CheckDeviceCapabilities(state, bindlessRequest).code == DeviceCapabilityDiagnosticCode::NotEnabled);

    state.bindless.descriptorIndexing.enabled = true;
    state.bindless.runtimeDescriptorArray.enabled = true;
    state.bindless.shaderSampledImageArrayNonUniformIndexing.enabled = true;
    state.bindless.descriptorBindingPartiallyBound.enabled = true;
    state.bindless.descriptorBindingVariableDescriptorCount.enabled = true;
    state.bindless.descriptorBindingSampledImageUpdateAfterBind.enabled = true;
    assert(CheckDeviceCapabilities(state, bindlessRequest).IsSupported());

    DeviceCapabilityRequest syncRequest;
    syncRequest.synchronization2 = true;
    state.synchronization2 = {true, false};
    assert(CheckDeviceCapabilities(state, syncRequest).code == DeviceCapabilityDiagnosticCode::NotEnabled);
    state.synchronization2.enabled = true;
    assert(CheckDeviceCapabilities(state, syncRequest).IsSupported());

    DeviceCapabilityState computeState;
    DeviceCapabilityRequest computeRequest;
    computeRequest.shaderFloat64 = true;
    auto computeCheck = CheckDeviceCapabilities(computeState, computeRequest);
    assert(computeCheck.code == DeviceCapabilityDiagnosticCode::Unsupported);
    assert(computeCheck.capability == DeviceCapability::ShaderFloat64);
    computeState.shaderFloat64.supported = true;
    assert(CheckDeviceCapabilities(computeState, computeRequest).code == DeviceCapabilityDiagnosticCode::NotEnabled);
    computeState.shaderFloat64.enabled = true;
    assert(CheckDeviceCapabilities(computeState, computeRequest).IsSupported());
    computeRequest.shaderInt16 = true;
    computeRequest.shaderInt64 = true;
    computeState.shaderInt16 = {true, true};
    computeState.shaderInt64 = {true, false};
    computeCheck = CheckDeviceCapabilities(computeState, computeRequest);
    assert(computeCheck.capability == DeviceCapability::ShaderInt64);
    assert(computeCheck.code == DeviceCapabilityDiagnosticCode::NotEnabled);
    computeState.shaderInt64.enabled = true;
    assert(CheckDeviceCapabilities(computeState, computeRequest).IsSupported());

    // Every enabled shader contract must have a distinct cache key. Physical
    // support without logical-device enablement must not select another key.
    std::set<uint64_t> shaderContracts;
    for (uint32_t bits = 0; bits < 128; ++bits) {
        DeviceCapabilityState contract;
        const DeviceCapabilityStatus bindless{true, (bits & 1u) != 0};
        contract.bindless.descriptorIndexing = bindless;
        contract.bindless.runtimeDescriptorArray = bindless;
        contract.bindless.shaderSampledImageArrayNonUniformIndexing = bindless;
        contract.bindless.descriptorBindingPartiallyBound = bindless;
        contract.bindless.descriptorBindingVariableDescriptorCount = bindless;
        contract.bindless.descriptorBindingSampledImageUpdateAfterBind = bindless;
        contract.dynamicRendering = {true, (bits & 2u) != 0};
        contract.synchronization2 = {true, (bits & 4u) != 0};
        contract.submit2 = {true, (bits & 8u) != 0};
        contract.shaderInt16 = {true, (bits & 16u) != 0};
        contract.shaderInt64 = {true, (bits & 32u) != 0};
        contract.shaderFloat64 = {true, (bits & 64u) != 0};
        const auto key = ComputeDeviceShaderContractKey(contract);
        if (bits == 0)
            assert(key == ComputeDeviceShaderContractKey({}));
        assert(shaderContracts.insert(key).second);
    }

    return 0;
}
