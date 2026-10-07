// Each CTest case runs in its own process: a teardown crash must fail the case.
#include <function/renderer/InxVkCoreModular.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#ifdef _WIN32
#include <Windows.h>
#endif

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    assert(argc == 2);
    const std::string scenario = argv[1];
    auto core = std::make_unique<infernux::InxVkCoreModular>(2);
    const infernux::InxAppMetadata metadata("VulkanStartupLifecycle", 1, 0, 0, "tests.local");
    SDL_Window *window = nullptr;
    if (scenario == "rejected_init") {
        assert(!core->Init(metadata, metadata, 0, nullptr));
    } else if (scenario == "instance_failure") {
        const char *extensions[] = {"VK_INFERNUX_nonexistent_test_extension"};
        assert(!core->Init(metadata, metadata, 1, extensions));
    } else if (scenario != "default") {
        assert(SDL_Init(SDL_INIT_VIDEO));
        window = SDL_CreateWindow("Vulkan startup lifecycle", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
        assert(window != nullptr);
        uint32_t count = 0;
        const auto *extensions = SDL_Vulkan_GetInstanceExtensions(&count);
        assert(extensions != nullptr && count != 0);
        std::vector<const char *> instanceExtensions(extensions, extensions + count);
        assert(core->Init(metadata, metadata, count, instanceExtensions.data()));
        auto &device = core->GetDeviceContext();
        assert(device.GetInstance() != VK_NULL_HANDLE && !device.IsValid());

        if (scenario == "missing_surface") {
            assert(!core->PrepareSurface());
        } else if (scenario == "destroy_twice") {
            device.Destroy();
            device.Destroy();
            assert(device.GetInstance() == VK_NULL_HANDLE && device.GetSurface() == VK_NULL_HANDLE);
            assert(!device.IsValid());
        } else if (scenario != "instance_only") {
            assert(SDL_Vulkan_CreateSurface(window, device.GetInstance(), nullptr, &core->m_surface));
            // The device owns the surface immediately, before logical-device startup.
            device.SetExternalSurface(core->m_surface);
            if (scenario == "surface_suspend") {
                core->SuspendPresentationSurface();
                assert(core->m_surface == VK_NULL_HANDLE && device.GetSurface() == VK_NULL_HANDLE);
                core->SuspendPresentationSurface();
            } else if (scenario == "device_ready") {
                assert(device.InitializeDevice(core->m_surface));
                assert(device.IsValid() && device.GetDeviceId() != infernux::rhi::InvalidDeviceId);
            } else {
                assert(scenario == "surface_only");
            }
        }
    }
    std::cout << "before_destroy " << scenario << std::endl;
    core.reset();
    if (window) {
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
    std::cout << "cleanup_complete " << scenario << std::endl;
}
