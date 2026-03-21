#pragma once

#include <algorithm>
#include <core/log/InfLog.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <core/config/InfPlatform.h>

#ifdef INF_PLATFORM_WINDOWS
namespace infengine
{
inline const char *GetExecutableDir()
{
    // Thread-safe via C++11 magic statics (initialized exactly once).
    static const std::string path = []() -> std::string {
        char buffer[MAX_PATH];
        DWORD len = GetModuleFileNameA(NULL, buffer, MAX_PATH);
        if (len == 0) {
            INFLOG_ERROR("Failed to get executable path, using current directory as fallback.");
            return ".";
        }
        return std::filesystem::path(buffer).parent_path().string();
    }();
    return path.c_str();
}

} // namespace infengine
#else
#include <limits.h>
#include <string.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace infengine
{
inline const char *GetExecutableDir()
{
    // Thread-safe via C++11 magic statics (initialized exactly once).
    static const std::string path = []() -> std::string {
        char result[PATH_MAX];
        ssize_t len = 0;

#if defined(__linux__)
        len = readlink("/proc/self/exe", result, PATH_MAX);
#elif defined(__APPLE__)
        uint32_t size = sizeof(result);
        if (_NSGetExecutablePath(result, &size) != 0) {
            INFLOG_ERROR("Buffer too small for executable path, using current directory as fallback.");
            return ".";
        }
        len = strlen(result);
#endif

        if (len <= 0) {
            INFLOG_ERROR("Failed to get executable path, using current directory as fallback.");
            return ".";
        }
        return std::filesystem::path(result, result + len).parent_path().string();
    }();
    return path.c_str();
}
} // namespace infengine
#endif
namespace infengine
{

/**
 * @brief Normalize a file path: replace backslashes with forward slashes.
 */
inline std::string NormalizePath(const std::string &path)
{
    std::string result = path;
    std::replace(result.begin(), result.end(), '\\', '/');
    return result;
}

/**
 * @brief Convert a UTF-8 path string to std::filesystem::path.
 *
 * On Windows the native encoding for narrow strings is the active code-page,
 * NOT UTF-8.  This helper ensures paths that were produced by Python
 * (which always outputs UTF-8) are correctly converted to wide-char
 * internally so that std::ifstream / std::filesystem operations work with
 * non-ASCII characters (e.g. Chinese filenames).
 */
inline std::filesystem::path ToFsPath(const std::string &utf8Path)
{
#ifdef INF_PLATFORM_WINDOWS
    // MultiByteToWideChar: convert UTF-8 → wchar_t
    if (utf8Path.empty())
        return {};
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8Path.data(), static_cast<int>(utf8Path.size()), nullptr, 0);
    if (wlen <= 0)
        return std::filesystem::path(utf8Path);
    std::wstring wstr(static_cast<size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8Path.data(), static_cast<int>(utf8Path.size()), wstr.data(), wlen);
    return std::filesystem::path(std::move(wstr));
#else
    // On Linux / macOS the native encoding is UTF-8 already.
    return std::filesystem::path(utf8Path);
#endif
}

/**
 * @brief Convert a std::filesystem::path to a UTF-8 std::string.
 *
 * On Windows, std::filesystem::path::string() / generic_string() encode
 * using the active code-page (e.g. GBK), NOT UTF-8.  This helper always
 * produces a UTF-8 string so that the result is compatible with ToFsPath()
 * and with Python (which expects UTF-8).
 */
inline std::string FromFsPath(const std::filesystem::path &p)
{
#ifdef INF_PLATFORM_WINDOWS
    std::wstring ws = p.generic_wstring();
    if (ws.empty())
        return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0)
        return p.generic_string();
    std::string result(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()), result.data(), len, nullptr, nullptr);
    return result;
#else
    return p.generic_string();
#endif
}

/**
 * @brief Read a file into a byte vector, supporting Unicode paths on Windows.
 */
inline bool ReadFileBytes(const std::string &filePath, std::vector<unsigned char> &out)
{
    std::ifstream file(ToFsPath(filePath), std::ios::binary | std::ios::ate);
    if (!file.is_open())
        return false;
    auto size = file.tellg();
    if (size <= 0) {
        out.clear();
        return size == 0;
    }
    out.resize(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(out.data()), size);
    return !file.fail();
}

inline std::string JoinPath(std::initializer_list<const char *> parts)
{
    std::filesystem::path path;
    for (const auto &part : parts) {
        path /= ToFsPath(part);
    }
    return FromFsPath(path);
}

inline std::string JoinPath(std::initializer_list<std::string> parts)
{
    std::filesystem::path path;
    for (const auto &part : parts) {
        path /= ToFsPath(part);
    }
    return FromFsPath(path);
}
} // namespace infengine