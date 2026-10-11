// Standalone browser probe linked against the production WebGPU RHI backend.
#include "WebGpuRhiDevice.h"
#include <emscripten.h>
#include <emscripten/eventloop.h>
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>

using namespace infernux;
namespace
{
wgpu::Instance instance;
wgpu::Adapter adapter;
wgpu::Device native;
std::unique_ptr<web::WebGpuRhiDevice> device;
rhi::BufferHandle first, current;
unsigned released = 0;
bool failed = false;

void Fail(const char *message)
{
    failed = true;
    std::fprintf(stderr, "RHI_IDENTITY_FAIL %s\n", message);
    EM_ASM({ window.identityProof.failed = true; window.identityProof.done = true; });
}

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void Recycle(void *)
{
    if (failed)
        return;
    try {
        rhi::BufferDesc desc;
        desc.byteSize = 64;
        desc.memory = rhi::BufferMemory::Readback;
        desc.usage = rhi::BufferUsageFlags::TransferDestination;
        for (unsigned batch = 0; batch < 512 && released < 65535; ++batch, ++released) {
            device->Release(current);
            current = device->CreateBuffer(desc);
            Require(current.IsValid(), "WebGPU buffer allocation failed");
            Require(!device->GetNativeBuffer(first), "Old WebGPU handle resolved a new buffer");
            device->Release(first);
            Require(bool(device->GetNativeBuffer(current)), "Old WebGPU release destroyed a new buffer");
        }
        if (released < 65535) {
            emscripten_set_timeout(Recycle, 0, nullptr);
            return;
        }
        Require(current.index != first.index, "WebGPU reused an exhausted slot");
        const std::array<uint32_t, 16> values{0x241abcde, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        const auto buffer = device->GetNativeBuffer(current);
        native.GetQueue().WriteBuffer(buffer, 0, values.data(), sizeof(values));
        buffer.MapAsync(wgpu::MapMode::Read, 0, sizeof(values), wgpu::CallbackMode::AllowSpontaneous,
            [buffer, values](wgpu::MapAsyncStatus status, wgpu::StringView) {
                if (status != wgpu::MapAsyncStatus::Success) {
                    Fail("WebGPU readback mapping failed");
                    return;
                }
                const auto *actual = static_cast<const uint32_t *>(buffer.GetConstMappedRange(0, sizeof(values)));
                for (size_t i = 0; i < values.size(); ++i) {
                    if (actual[i] != values[i]) {
                        Fail("WebGPU readback mismatch after slot retirement");
                        return;
                    }
                }
                buffer.Unmap();
                device->Release(current);
                std::puts("RHI_IDENTITY_PASS real_allocations=65536 readback_words=16");
                EM_ASM({ window.identityProof.passed = true; window.identityProof.done = true; });
            });
    } catch (const std::exception &error) {
        Fail(error.what());
    }
}
} // namespace

int main()
{
    instance = wgpu::CreateInstance();
    wgpu::RequestAdapterOptions options;
    instance.RequestAdapter(&options, wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter result, wgpu::StringView) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                Fail("WebGPU adapter unavailable");
                return;
            }
            adapter = std::move(result);
            wgpu::DeviceDescriptor descriptor;
            descriptor.SetUncapturedErrorCallback([](const wgpu::Device &, wgpu::ErrorType, wgpu::StringView message) {
                std::fprintf(stderr, "%.*s\n", int(message.length), message.data);
                Fail("Uncaptured WebGPU validation error");
            });
            adapter.RequestDevice(&descriptor, wgpu::CallbackMode::AllowSpontaneous,
                [](wgpu::RequestDeviceStatus status, wgpu::Device result, wgpu::StringView) {
                    if (status != wgpu::RequestDeviceStatus::Success) {
                        Fail("WebGPU device unavailable");
                        return;
                    }
                    native = std::move(result);
                    device = std::make_unique<web::WebGpuRhiDevice>(native, native.GetQueue(), 8);
                    rhi::BufferDesc desc;
                    desc.byteSize = 64;
                    desc.memory = rhi::BufferMemory::Readback;
                    desc.usage = rhi::BufferUsageFlags::TransferDestination;
                    first = current = device->CreateBuffer(desc);
                    if (!first.IsValid()) {
                        Fail("Initial WebGPU buffer allocation failed");
                        return;
                    }
                    emscripten_set_timeout(Recycle, 0, nullptr);
                });
        });
    emscripten_exit_with_live_runtime();
}
