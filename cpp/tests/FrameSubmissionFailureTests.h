bool VerifyFrameSubmissionFailureIsTerminal()
{
    using namespace infernux;
    for (int mode = 0; mode < 4; ++mode) {
        uint32_t recordings = 0;
        std::vector<bool> notifications;
        struct Owner
        {
            SDL_Window *window = nullptr;
            std::unique_ptr<InxVkCoreModular> core;
            ~Owner()
            {
                core.reset();
                if (window)
                    SDL_DestroyWindow(window);
            }
        } owner;
        owner.window =
            SDL_CreateWindow("Frame submission failure regression", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
        if (!Require(owner.window != nullptr, SDL_GetError()))
            return false;
        owner.core = std::make_unique<InxVkCoreModular>(2);
        auto &core = *owner.core;
        Uint32 extensionCount = 0;
        const auto nativeExtensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
        if (!Require(nativeExtensions && extensionCount != 0, "Frame fixture has no Vulkan window extensions"))
            return false;
        std::vector<const char *> extensions(nativeExtensions, nativeExtensions + extensionCount);
        core.SetWindowSize(64, 64);
        if (!Require(core.Init({}, {}, extensionCount, extensions.data()) &&
                         SDL_Vulkan_CreateSurface(owner.window, core.m_instance, nullptr, &core.m_surface) &&
                         core.PrepareSurface(),
                     "Frame failure fixture initialization failed"))
            return false;
        core.SetFrameComputeSubmissionCallback([&](bool submitted) { notifications.push_back(submitted); });
        core.SetFrameSubmissionBuilder([&](vk::VulkanFrameSubmission &submission, uint32_t setup, uint32_t) {
            if (mode == 2)
                return false;
            const auto work = submission.AddWork(
                core.GetDeviceContext().GetDeviceId(), rhi::QueueRole::Graphics, rhi::SubmissionDomain::Frame,
                rhi::InvalidRenderViewId, rhi::PipelineStage::AllCommands, {setup},
                [&](VkCommandBuffer) {
                    ++recordings;
                    if (mode == 1)
                        throw std::runtime_error("frame recorder sentinel");
                    return mode != 0;
                },
                "Tests/FrameFailure");
            return work != 0;
        });
        const float position[] = {0, 0, 3};
        const float target[] = {0, 0, 0};
        const float up[] = {0, 1, 0};
        std::string failure;
        core.WaitForCurrentFrame();
        try {
            core.DrawFrame(position, target, up);
        } catch (const std::exception &error) {
            failure = error.what();
        }
        if (mode == 3) {
            if (!Require(failure.empty() && recordings == 1 && notifications == std::vector<bool>{true},
                         "New renderer session failed to submit a valid frame"))
                return false;
            core.WaitForCurrentFrame();
            core.DrawFrame(position, target, up);
            if (!Require(recordings == 2 && notifications == std::vector<bool>({true, true}),
                         "Successful renderer session did not continue normally"))
                return false;
            continue;
        }
        if (!Require(!failure.empty(), "Failed frame returned normally instead of terminating the renderer session"))
            return false;
        if (!Require(mode != 1 || failure == "frame recorder sentinel", "Recorder exception identity was lost"))
            return false;
        const auto expectedNotifications = mode == 2 ? std::vector<bool>{} : std::vector<bool>{false};
        if (!Require(notifications == expectedNotifications, "Failed submission acknowledgement was incorrect"))
            return false;
        const uint32_t recordedBeforeRetry = recordings;
        // Removing the faulty callback cannot make an unconsumed acquired
        // image or partially recorded simulation safe for another frame.
        core.SetFrameSubmissionBuilder([](vk::VulkanFrameSubmission &, uint32_t, uint32_t) { return true; });
        std::string waitFailure;
        try {
            core.WaitForCurrentFrame();
        } catch (const std::exception &error) {
            waitFailure = error.what();
        }
        if (!Require(waitFailure == failure, "Failed session entered the next frame fence wait"))
            return false;
        std::string repeatedFailure;
        try {
            core.DrawFrame(position, target, up);
        } catch (const std::exception &error) {
            repeatedFailure = error.what();
        }
        if (!Require(repeatedFailure == failure && recordings == recordedBeforeRetry &&
                         notifications == expectedNotifications,
                     "Failed renderer session continued recording, acquisition or acknowledgements"))
            return false;
    }
    std::cout << "Frame failures: rejected recorder, exception and rejected composition are terminal; "
                 "a new renderer submits normally\n";
    return true;
}
