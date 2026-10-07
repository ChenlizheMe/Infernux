#pragma once

#include <platform/filesystem/InxPath.h>

#ifdef INX_PLATFORM_WINDOWS
#pragma push_macro("CreateDirectory")
#pragma push_macro("DeleteFile")
#undef CreateDirectory
#undef DeleteFile
#endif

#include <assimp/DefaultIOStream.h>
#include <assimp/IOSystem.hpp>
#include <assimp/Importer.hpp>

#include <cstdio>
#include <memory>
#include <string_view>

namespace infernux
{
// Keep Assimp's regular-file read/seek semantics, but open Windows files with
// their native Unicode spelling rather than the active narrow code page.
class ModelSourceStream final : public Assimp::DefaultIOStream
{
  public:
    ModelSourceStream(FILE *file, const std::string &path) : DefaultIOStream(file, path) {}
};

// One importer owns one source directory. Sidecars never resolve against a
// mutable process cwd, and importing cannot change that cwd or write files.
class ModelSourceIO final : public Assimp::IOSystem
{
  public:
    explicit ModelSourceIO(const std::string &sourcePath)
        : m_directory(ToFsPath(NormalizeFilesystemPathLexically(sourcePath)).parent_path())
    {
    }

    bool Exists(const char *path) const override
    {
        if (!path || !*path)
            return false;
        std::error_code error;
        return std::filesystem::is_regular_file(Resolve(path), error);
    }

    char getOsSeparator() const override { return '/'; }

    Assimp::IOStream *Open(const char *path, const char *mode = "rb") override
    {
        if (!path || !*path || !mode)
            return nullptr;
        const std::string_view access(mode);
        if (access != "r" && access != "rb" && access != "rt")
            return nullptr;
        const auto resolved = Resolve(path);
        std::error_code error;
        if (!std::filesystem::is_regular_file(resolved, error))
            return nullptr;
#ifdef INX_PLATFORM_WINDOWS
        FILE *file = _wfopen(resolved.c_str(), L"rb");
#else
        FILE *file = std::fopen(resolved.c_str(), "rb");
#endif
        if (!file)
            return nullptr;
        std::unique_ptr<FILE, decltype(&std::fclose)> owner(file, &std::fclose);
        auto *stream = new ModelSourceStream(file, FromFsPath(resolved));
        // Ownership moves only after the stream's construction succeeds.
        owner.release();
        return stream;
    }

    void Close(Assimp::IOStream *stream) override { delete stream; }

    bool ComparePaths(const char *left, const char *right) const override
    {
        return left && right && FilesystemPathsEquivalent(FromFsPath(Resolve(left)), FromFsPath(Resolve(right)));
    }

    bool CreateDirectory(const std::string &) override { return false; }
    bool ChangeDirectory(const std::string &) override { return false; }
    bool DeleteFile(const std::string &) override { return false; }

  private:
    std::filesystem::path Resolve(const std::string &path) const
    {
        auto candidate = ToFsPath(NormalizePortablePath(path));
        if (candidate.is_relative())
            candidate = m_directory / candidate;
        return ToFsPath(NormalizeFilesystemPathLexically(FromFsPath(candidate)));
    }

    std::filesystem::path m_directory;
};

inline const aiScene *ReadModelSource(Assimp::Importer &importer, const std::string &path, unsigned int flags)
{
    const auto source = NormalizeFilesystemPathLexically(path);
    importer.SetIOHandler(new ModelSourceIO(source));
    return importer.ReadFile(source, flags);
}
} // namespace infernux

#ifdef INX_PLATFORM_WINDOWS
#pragma pop_macro("DeleteFile")
#pragma pop_macro("CreateDirectory")
#endif
