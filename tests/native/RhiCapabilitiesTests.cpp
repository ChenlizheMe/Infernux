#include <function/renderer/rhi/RhiCommand.h>
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
    capabilities.SetBackendId("vulkan");
    assert(capabilities.BackendName() == "vulkan");
    assert(CheckDeviceContract(kRhiApiVersion, capabilities).IsValid());
    assert(CheckDeviceContract((kRhiApiVersionMajor + 1u) << 16u, capabilities).code ==
           DeviceContractDiagnosticCode::IncompatibleApiVersion);
    DeviceCaps missingBackend;
    assert(CheckDeviceContract(kRhiApiVersion, missingBackend).code == DeviceContractDiagnosticCode::MissingBackendId);
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

    // Plugins add identities without registering a central backend enum.
    DeviceCaps plugin;
    plugin.SetBackendId("example.compute");
    assert(plugin.BackendName() == "example.compute");
    assert(CheckDeviceContract(kRhiApiVersion, plugin).IsValid());
    assert(!CheckDeviceContract(1u << 16u, plugin).IsValid());

    // Every enabled shader feature has a distinct cache key. Host execution
    // limits and scheduling capabilities must not create shader variants.
    std::set<uint64_t> shaderContracts;
    for (uint32_t bits = 0; bits < 32; ++bits) {
        PortableCaps contract;
        contract.bindlessSampledTextures = (bits & 1u) != 0;
        contract.shaderInt16 = (bits & 2u) != 0;
        contract.shaderInt64 = (bits & 4u) != 0;
        contract.shaderFloat64 = (bits & 8u) != 0;
        contract.shaderFloat16 = (bits & 16u) != 0;
        const auto key = ComputeDeviceShaderContractKey(contract, "example.compute");
        assert(shaderContracts.insert(key).second);
        contract.timelineCompletion = true;
        contract.asyncCompute = true;
        contract.maxStorageBufferBinding = 1ull << 33u;
        assert(key == ComputeDeviceShaderContractKey(contract, "example.compute"));
    }
    assert(ComputeDeviceShaderContractKey({}, "vulkan") != ComputeDeviceShaderContractKey({}, "example.compute"));
    assert(ComputeDeviceShaderContractKey({}, "vulkan", 1) != ComputeDeviceShaderContractKey({}, "vulkan", 2));

    GraphicsCommandEncoder::Dispatch graphicsDispatch;
    assert(!graphicsDispatch.IsComplete());
    ComputeCommandEncoder::DispatchTable computeDispatch;
    assert(!computeDispatch.IsComplete());
    TransferCommandEncoder::DispatchTable transferDispatch;
    assert(!transferDispatch.IsComplete());

    return 0;
}
