#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

#include <apps/player/PlayerHost.h>

int main()
{
    using namespace infernux::playerhost;
#ifdef _WIN32
    const auto layout = ResolveLayout(std::filesystem::path(L"C:/Games/测试/Star.exe"));
    assert(layout.installRoot == std::filesystem::path(L"C:/Games/测试"));
    assert(layout.dataRoot == std::filesystem::path(L"C:/Games/测试/Star_Data"));
    assert(layout.runtimeRoot == layout.dataRoot / L"Runtime");
    const auto nearLimit = ResolveLayout(std::filesystem::path(L"C:\\" + std::wstring(246, L'x') + L"\\Player.exe"));
    assert(nearLimit.hostExecutable.native().size() == 260);
    assert(!IsSupportedExecutableLocation(nearLimit.hostExecutable));
    const auto supported = ResolveLayout(std::filesystem::path(L"C:\\" + std::wstring(245, L'x') + L"\\Player.exe"));
    assert(supported.hostExecutable.native().size() == 259);
    assert(IsSupportedExecutableLocation(supported.hostExecutable));
    assert(supported.runtimeRoot.native().size() > 260);
    // A supplementary Unicode character counts as two Windows path units.
    assert(!IsSupportedExecutableLocation(
        std::filesystem::path(L"C:\\" + std::wstring(244, L'x') + L"\U0001F680\\Player.exe")));
#else
    const auto layout = ResolveLayout(std::filesystem::path("/opt/games/Star"));
    assert(layout.installRoot == std::filesystem::path("/opt/games"));
    assert(layout.dataRoot == std::filesystem::path("/opt/games/Star_Data"));
    assert(layout.runtimeRoot == layout.dataRoot / "Runtime");
    const auto unicodeArgs = BuildPythonArguments(std::filesystem::u8path(u8"/home/player/桌面/Star"), {});
    assert(unicodeArgs == std::vector<std::wstring>{L"/home/player/桌面/Star"});
    assert(IsSupportedExecutableLocation(std::filesystem::path("/opt/" + std::string(280, 'x') + "/Player")));
#endif
    const auto args = BuildPythonArguments(layout.hostExecutable, {L"--scene", L"场景"});
    assert(args.size() == 3);
    assert(args[1] == L"--scene");
    assert(args[2] == L"场景");
    return 0;
}
