#pragma once

#include <filesystem>

namespace infernux
{

/// Native file/loader spelling only. Catalog identities, document keys and
/// portable metadata keep their normal paths.
[[nodiscard]] inline std::filesystem::path NativeFilePath(const std::filesystem::path &path)
{
#ifdef _WIN32
    const auto absolute = std::filesystem::absolute(path).lexically_normal().make_preferred().native();
    if (absolute.rfind(L"\\\\?\\", 0) == 0)
        return absolute;
    if (absolute.rfind(L"\\\\", 0) == 0)
        return L"\\\\?\\UNC\\" + absolute.substr(2);
    return L"\\\\?\\" + absolute;
#else
    return path;
#endif
}

} // namespace infernux
