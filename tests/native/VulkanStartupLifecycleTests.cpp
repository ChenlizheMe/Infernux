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

#include "SwapchainRecreationTests.h"
#include "RhiIdentityGpuTests.h"

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
        window = SDL_CreateWindow("Vulkan startup lifecycle", 64, 64,
                                  SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN | SDL_WINDOW_RESIZABLE);
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
            } else if (scenario == "presentation_resize") {
                core->SetWindowSize(64, 64);
                assert(core->PrepareSurface());
                const auto initialRevision = core->GetPresentationViewContext().revision;
                assert(core->RefreshPresentationSize());
                assert(core->GetPresentationViewContext().revision == initialRevision);

                // No acquire or present is performed in this scenario. The
                // driver therefore cannot rescue missing resize notification
                // by returning OUT_OF_DATE. Exercise the same frame-start
                // entry point used by the Editor and the native Player.
                for (const auto size : {VkExtent2D{1600, 900}, VkExtent2D{2560, 1494},
                                        VkExtent2D{1024, 700}, VkExtent2D{800, 600}, VkExtent2D{1600, 900}}) {
                    assert(SDL_SetWindowSize(window, int(size.width), int(size.height)));
                    assert(SDL_SyncWindow(window));
                    SDL_PumpEvents();
                    int width = 0, height = 0;
                    assert(SDL_GetWindowSizeInPixels(window, &width, &height));
                    const auto revision = core->GetPresentationViewContext().revision;
                    core->SetWindowSize(uint32_t(width), uint32_t(height));
                    assert(core->RefreshPresentationSize());
                    assert(core->GetSwapchainExtent().width == uint32_t(width));
                    assert(core->GetSwapchainExtent().height == uint32_t(height));
                    assert(core->GetPresentationViewContext().width == uint32_t(width));
                    assert(core->GetPresentationViewContext().height == uint32_t(height));
                    assert(core->GetPresentationViewContext().revision == revision + 1);
                    for (int frame = 0; frame < 8; ++frame) {
                        core->SetWindowSize(uint32_t(width), uint32_t(height));
                        assert(core->RefreshPresentationSize());
                        assert(core->GetPresentationViewContext().revision == revision + 1);
                    }

                    // Minimize must not create zero-sized images or consume a
                    // pending resize. Restoring the same dimensions is valid.
                    core->SetWindowSize(0, 0);
                    assert(!core->RefreshPresentationSize());
                    assert(!core->RefreshPresentationSize());
                    assert(core->GetPresentationViewContext().revision == revision + 1);
                    core->SetWindowSize(uint32_t(width), uint32_t(height));
                    assert(core->RefreshPresentationSize());
                    assert(core->GetPresentationViewContext().revision == revision + 2);
                }
                std::cout << "presentation_resize_without_acquire_present_passed" << std::endl;
            } else if (scenario == "device_ready") {
                assert(device.InitializeDevice(core->m_surface));
                assert(device.IsValid() && device.GetDeviceId() != infernux::rhi::InvalidDeviceId);
                rhi_identity_test::RealBufferRecycling(device);
                swapchain_recreation_test::Run(device, window);
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
