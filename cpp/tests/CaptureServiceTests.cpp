#include <function/renderer/CaptureService.h>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace
{
infernux::rhi::RenderViewContext CaptureView()
{
    infernux::rhi::RenderViewContext view;
    view.id = 2;
    view.source = 1;
    view.device = 1;
    view.kind = infernux::rhi::RenderViewKind::Capture;
    view.output = infernux::rhi::RenderOutputKind::Readback;
    view.width = 64;
    view.height = 64;
    return view;
}
} // namespace

int main()
{
    using namespace std::chrono_literals;
    infernux::CaptureService service(0ms);
    const auto output = (std::filesystem::temp_directory_path() / "infernux-capture-timeout.png").string();
    {
        std::ofstream existing(output, std::ios::binary | std::ios::trunc);
        existing << "existing";
    }
    const uint64_t id = service.Request(infernux::CaptureSource::Editor, CaptureView(), 1, 42, output);
    assert(service.HasPending());
    assert(service.Query(id).status == infernux::CaptureStatus::PendingGpu);

    service.Poll();
    const auto terminal = service.Query(id);
    assert(terminal.status == infernux::CaptureStatus::Failed);
    assert(terminal.error == "Capture source frame was not submitted before timeout");
    assert(!service.HasPending());
    assert(!service.Cancel(id));
    {
        std::ifstream existing(output, std::ios::binary);
        std::string contents;
        existing >> contents;
        assert(contents == "existing");
    }

    // Terminal requests without live encoder work do not consume one of the
    // four actual in-flight slots.
    for (int index = 0; index < 8; ++index) {
        const uint64_t completed = service.Request(infernux::CaptureSource::Editor, CaptureView(), 1,
                                                   43 + static_cast<uint64_t>(index), output + std::to_string(index));
        service.Poll();
        assert(service.Query(completed).status == infernux::CaptureStatus::Failed);
    }

    bool rejectedNegativeTimeout = false;
    try {
        infernux::CaptureService invalid(-1ms);
    } catch (const std::invalid_argument &) {
        rejectedNegativeTimeout = true;
    }
    assert(rejectedNegativeTimeout);
    std::filesystem::remove(output);
    return 0;
}
