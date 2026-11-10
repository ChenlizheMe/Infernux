#include "BindingRegistration.h"
#include "InxPackPyBridge.h"
#include <platform/filesystem/InxPack.h>
#include <platform/filesystem/InxPath.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <signal.h>
#include <sys/types.h>
#endif

namespace py = pybind11;

PYBIND11_MODULE(_InfernuxBootstrap, m)
{
    infernux::binding::BindPackFunctions(m);
    m.def(
        "_inxplayer_show_error",
        [](const py::str &title, const py::str &message) {
#if defined(_WIN32)
            const std::wstring wideTitle = title.cast<std::wstring>();
            const std::wstring wideMessage = message.cast<std::wstring>();
            MessageBoxW(nullptr, wideMessage.c_str(), wideTitle.c_str(), MB_OK | MB_ICONERROR | MB_TASKMODAL);
#else
            const std::string titleText = title.cast<std::string>();
            const std::string messageText = message.cast<std::string>();
            std::fprintf(stderr, "%s: %s\n", titleText.c_str(), messageText.c_str());
            std::fflush(stderr);
#endif
        },
        py::arg("title"), py::arg("message"), "Show a startup error without importing the Python ctypes stack.");
    m.def(
        "_inxplayer_process_is_alive",
        [](const int64_t pid) {
            if (pid <= 0)
                return false;
#if defined(_WIN32)
            if (static_cast<uint64_t>(pid) > std::numeric_limits<DWORD>::max())
                return false;
            const HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
            if (process == nullptr)
                return GetLastError() == ERROR_ACCESS_DENIED;

            DWORD exitCode = 0;
            const BOOL queried = GetExitCodeProcess(process, &exitCode);
            CloseHandle(process);
            return queried == FALSE || exitCode == STILL_ACTIVE;
#else
            if (::kill(static_cast<pid_t>(pid), 0) == 0)
                return true;
            return errno == EPERM;
#endif
        },
        py::arg("pid"), "Return whether a Player cache publisher process is alive.");
}
