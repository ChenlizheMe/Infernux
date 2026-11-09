#include <platform/filesystem/AtomicFile.h>
#include <platform/filesystem/InxPath.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace
{

bool Expect(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

std::string ReadText(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::string MakeCrashDocument(char fill)
{
    std::string document = "INFERNUX_ATOMIC_CRASH_GATE\n";
    document.append(4 * 1024 * 1024, fill);
    document += "\nEND\n";
    return document;
}

bool RunInterruptedWrite(const std::filesystem::path &executable, const std::filesystem::path &target, char fill,
                         unsigned delayMs)
{
#ifdef _WIN32
    std::wstring command = L"\"" + executable.wstring() + L"\" --atomic-crash-child \"" + target.wstring() + L"\" " +
                           static_cast<wchar_t>(fill);
    std::vector<wchar_t> commandBuffer(command.begin(), command.end());
    commandBuffer.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!::CreateProcessW(nullptr, commandBuffer.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr,
                          &startup, &process))
        return false;

    ::Sleep(delayMs);
    DWORD exitCode = 0;
    const bool running = ::GetExitCodeProcess(process.hProcess, &exitCode) && exitCode == STILL_ACTIVE;
    if (running)
        ::TerminateProcess(process.hProcess, 91);
    ::WaitForSingleObject(process.hProcess, INFINITE);
    ::CloseHandle(process.hThread);
    ::CloseHandle(process.hProcess);
    return running;
#else
    const pid_t child = ::fork();
    if (child < 0)
        return false;
    if (child == 0) {
        const std::string fillArgument(1, fill);
        ::execl(executable.c_str(), executable.c_str(), "--atomic-crash-child", target.c_str(), fillArgument.c_str(),
                static_cast<char *>(nullptr));
        ::_exit(127);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    const bool interrupted = ::kill(child, SIGKILL) == 0;
    int status = 0;
    ::waitpid(child, &status, 0);
    return interrupted;
#endif
}

bool RunCrashInjectionGate(const std::filesystem::path &executable, const std::filesystem::path &directory)
{
    const std::filesystem::path target = directory / "crash-gate.scene";
    const std::string first = MakeCrashDocument('A');
    const std::string second = MakeCrashDocument('B');
    std::string error;
    if (!infernux::WriteTextFileAtomically(target.u8string(), first, error)) {
        std::cerr << error << '\n';
        return false;
    }

    size_t interruptions = 0;
    size_t attempts = 0;
    while (interruptions < 100 && attempts < 300) {
        ++attempts;
        const std::string before = ReadText(target);
        if (before != first && before != second) {
            std::cerr << "crash gate found an invalid document before interruption " << interruptions << '\n';
            return false;
        }
        const char next = before == first ? 'B' : 'A';
        if (!RunInterruptedWrite(executable, target, next, static_cast<unsigned>(attempts % 4)))
            continue;
        ++interruptions;

        const std::string after = ReadText(target);
        if (after != first && after != second) {
            std::cerr << "crash gate found a partial document after interruption " << interruptions << '\n';
            return false;
        }
    }
    if (interruptions != 100) {
        std::cerr << "crash gate could only force " << interruptions << " interruptions\n";
        return false;
    }
    return true;
}

bool RunTransientReplaceLockGate(const std::filesystem::path &target)
{
#ifdef _WIN32
    HANDLE lock = ::CreateFileW(target.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    if (lock == INVALID_HANDLE_VALUE)
        return false;

    bool written = false;
    std::string writeError;
    std::thread writer(
        [&] { written = infernux::WriteTextFileAtomically(target.u8string(), "after-lock", writeError).has_value(); });
    ::Sleep(40);
    ::CloseHandle(lock);
    writer.join();
    if (!written)
        std::cerr << "transient replace lock was not retried: " << writeError << '\n';
    return written && ReadText(target) == "after-lock";
#else
    (void)target;
    return true;
#endif
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 4 && std::string(argv[1]) == "--atomic-crash-child") {
        std::string error;
        return infernux::WriteTextFileAtomically(argv[2], MakeCrashDocument(argv[3][0]), error) ? 0 : 2;
    }

    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / ("infernux_atomic_file_" + std::to_string(suffix));
    std::filesystem::create_directory(directory);

    const std::filesystem::path target = directory / infernux::ToFsPath(u8"可靠写入.txt");
    std::string error;
    bool passed = true;

    passed &= Expect(infernux::WriteTextFileAtomically(target.u8string(), "first", error).has_value(), error.c_str());
    passed &= Expect(ReadText(target) == "first", "initial atomic write content mismatch");
    const auto snapshot = infernux::ReadTextFileSnapshot(target.u8string());
    passed &= Expect(snapshot.content == "first", "snapshot bytes differ from loaded content");
    passed &= Expect(snapshot.state == infernux::CaptureAtomicFileState(target.u8string()),
                     "snapshot identity differs from an unchanged file");
    passed &= Expect(infernux::WriteTextFileAtomically(target.u8string(), "second", error).has_value(), error.c_str());
    passed &= Expect(ReadText(target) == "second", "replacement atomic write content mismatch");
    passed &= Expect(snapshot.content == "first", "retained snapshot changed after external replace");
    passed &= Expect(!infernux::WriteTextFileAtomically(target.u8string(), "stale", error,
                                                        infernux::AtomicWriteOptions{infernux::FileWriteMode::Replace, snapshot.state}),
                     "snapshot from old bytes allowed overwriting a newer document");
    passed &= Expect(ReadText(target) == "second", "rejected snapshot write changed disk content");
    // Pinned inputs expose the previous complete generation across replacement.
    for (const auto &generations : {std::pair{"second", "third"}, std::pair{"third", "fourth"}}) {
        infernux::FileInput previous(target);
        const auto result = infernux::WriteTextFileAtomically(target.u8string(), generations.second, error);
        passed &= Expect(result.has_value(), error.c_str());
        passed &= Expect(ReadText(target) == generations.second, "atomic replacement published incorrect content");
        std::string previousBytes(static_cast<size_t>(previous.Size()), '\0');
        previous.Read(previousBytes.data(), previousBytes.size());
        passed &= Expect(previousBytes == generations.first, "previous generation changed across replacement");
    }
    passed &= Expect(RunTransientReplaceLockGate(target), "transient destination lock was not tolerated");
    passed &= Expect(infernux::RemoveFileDurably(target.u8string(), error), error.c_str());
    passed &= Expect(!std::filesystem::exists(target), "durable removal left the target behind");
    passed &= Expect(infernux::RemoveFileDurably(target.u8string(), error), "durable removal rejected a missing file");

    size_t temporaryCount = 0;
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
        if (infernux::IsAtomicTemporaryPath(entry.path()))
            ++temporaryCount;
    }
    passed &= Expect(temporaryCount == 0, "successful writes left temporary files behind");

    const std::filesystem::path missingTarget = directory / "missing" / "failure.txt";
    passed &= Expect(!infernux::WriteTextFileAtomically(missingTarget.u8string(), "failure", error),
                     "write to missing parent unexpectedly succeeded");

    passed &= Expect(RunCrashInjectionGate(std::filesystem::absolute(argv[0]), directory),
                     "100-interruption atomic write crash gate failed");

    std::filesystem::remove_all(directory);
    return passed ? 0 : 1;
}
