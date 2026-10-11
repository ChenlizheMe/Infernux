// Real Vulkan allocations; retirement serials are controlled contract inputs.
#include <function/renderer/vk/VkDescriptorManager.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <SDL3/SDL.h>

#include <cassert>
#include <iostream>
#include <optional>
#include <string>
#ifdef _WIN32
#include <Windows.h>
#endif

using namespace infernux;

static VkDescriptorSetLayout MakeLayout(VkDevice device)
{
    VkDescriptorSetLayoutBinding binding{};
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_ALL;
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = 1;
    info.pBindings = &binding;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    assert(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout) == VK_SUCCESS);
    return layout;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    assert(argc == 2);
    const std::string scenario = argv[1];
    assert(SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("Descriptor lease ownership", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    assert(window != nullptr);
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    vk::VkDeviceContext firstDevice, secondDevice;
    assert(firstDevice.Initialize(window, config));
    const bool separateDevice = scenario.rfind("device_", 0) == 0 || scenario.rfind("identity_", 0) == 0;
    if (separateDevice) {
        assert(secondDevice.Initialize(window, config));
        assert(firstDevice.GetDeviceId() != secondDevice.GetDeviceId());
    }
    const auto firstLayout = MakeLayout(firstDevice.GetDevice());
    const auto secondLayout = separateDevice ? MakeLayout(secondDevice.GetDevice()) : firstLayout;
    {
        std::optional<vk::VkDescriptorManager> first(std::in_place, firstDevice.GetDevice(), firstDevice.GetDeviceId());
        vk::VkDescriptorManager second(separateDevice ? secondDevice.GetDevice() : firstDevice.GetDevice(),
                                        separateDevice ? secondDevice.GetDeviceId() : firstDevice.GetDeviceId());
        const auto original = first->Allocate(firstLayout);
        assert(original.IsValid() && first->Owns(original));
        if (scenario == "own") {
            first->Retire(original);
            assert(!first->Owns(original) && first->GetStats().liveSets == 0);
        } else if (scenario == "deferred" || scenario == "native_mark") {
            if (scenario == "deferred")
                first->MarkUsed(original, 9);
            else
                first->MarkUsed(original.set, 9);
            first->Retire(original, 5);
            assert(first->Collect(5) == 0);
            assert(first->Collect(9) == 1);
            assert(!first->Owns(original));
        } else {
            auto *victim = &second;
            vk::DescriptorLease live;
            auto rejected = original;
            if (scenario.rfind("identity_", 0) == 0) {
                // Even a recognized allocation ID cannot override the supplied device identity.
                victim = &*first;
                live = original;
                rejected.device = secondDevice.GetDeviceId();
            } else if (scenario.rfind("reset_", 0) == 0) {
                first->Reset(firstDevice.GetDevice(), firstDevice.GetDeviceId());
                victim = &*first;
                live = first->Allocate(firstLayout);
            } else if (scenario.rfind("recreate_", 0) == 0) {
                // Reuse exactly the same C++ storage: an owner pointer is not an identity.
                const auto *address = &*first;
                first.reset();
                first.emplace(firstDevice.GetDevice(), firstDevice.GetDeviceId());
                assert(&*first == address);
                victim = &*first;
                live = first->Allocate(firstLayout);
            } else {
                assert(separateDevice || scenario.rfind("owner_", 0) == 0);
                live = second.Allocate(secondLayout);
            }
            assert(live.IsValid() && victim->Owns(live));
            const auto operation = scenario.substr(scenario.find('_') + 1);
            if (operation == "owns") {
                assert(!victim->Owns(rejected));
            } else if (operation == "retire") {
                victim->Retire(rejected);
                assert(victim->Owns(live) && victim->GetStats().liveSets == 1);
            } else {
                assert(operation == "mark");
                victim->MarkUsed(rejected, 9);
                victim->Retire(live, 5);
                assert(victim->Collect(5) == 1);
            }
        }
    }
    if (scenario == "deferred") {
        // Particle view retirement drops the RHI handle immediately. Verify the
        // real Vulkan lease survives until its last recorded use, even when the
        // current retirement source points to an earlier submission.
        auto &device = firstDevice.GetRhiDevice();
        device.UseSubmissionSerials([] { return rhi::SubmissionSerial{5}; });
        const auto baseline = device.GetDescriptorStats();
        const auto buffer = device.CreateBuffer({256, rhi::BufferUsageFlags::Storage});
        rhi::BindingLayoutDesc layoutDesc;
        layoutDesc.entries[layoutDesc.entryCount++] = {0, rhi::BindingType::StorageBuffer,
                                                        rhi::ShaderStage::Vertex, 1};
        const auto layout = device.CreateBindingLayout(layoutDesc);
        assert(buffer.IsValid() && layout.IsValid());
        for (unsigned cycle = 0; cycle < 32; ++cycle) {
            rhi::BindGroupDesc desc;
            desc.layout = layout;
            desc.buffers[desc.bufferCount++] = {0, rhi::BindingType::StorageBuffer, buffer, 0, 256};
            const auto group = device.CreateBindGroup(desc);
            assert(group.IsValid());
            device.GetDescriptorManager().MarkUsed(device.Resolve(group), 9);
            device.Release(group);
            assert(device.Resolve(group) == VK_NULL_HANDLE);
            assert(device.CollectDescriptorRetirements(5) == 0);
            assert(device.GetDescriptorStats().retiredSets == baseline.retiredSets + 1);
            assert(device.CollectDescriptorRetirements(9) == 1);
            assert(device.GetDescriptorStats().retiredSets == baseline.retiredSets);
            assert(device.GetDescriptorStats().liveSets == baseline.liveSets);
        }
        device.Release(layout);
        device.Release(buffer);
        device.CollectResourceRetirements(9);
    }
    if (separateDevice)
        vkDestroyDescriptorSetLayout(secondDevice.GetDevice(), secondLayout, nullptr);
    vkDestroyDescriptorSetLayout(firstDevice.GetDevice(), firstLayout, nullptr);
    secondDevice.Destroy();
    firstDevice.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << "cleanup_complete " << scenario << '\n';
}
