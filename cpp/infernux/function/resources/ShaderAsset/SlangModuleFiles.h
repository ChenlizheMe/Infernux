#pragma once

#include "SlangCompiler.h"
#include "SlangFrameModule.h"
#include "SlangGeometryModule.h"
#include "SlangLightingModule.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <slang-com-helper.h>
#include <unordered_map>

namespace infernux
{

class SlangModuleBlob final : public ISlangBlob
{
  public:
    explicit SlangModuleBlob(std::string bytes) : m_bytes(std::move(bytes))
    {
    }
    SLANG_IUNKNOWN_ALL
    ISlangUnknown *getInterface(const SlangUUID &uuid)
    {
        return uuid == ISlangBlob::getTypeGuid() || uuid == ISlangUnknown::getTypeGuid() ? this : nullptr;
    }
    SLANG_NO_THROW const void *SLANG_MCALL getBufferPointer() override
    {
        return m_bytes.data();
    }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() override
    {
        return m_bytes.size();
    }

  private:
    std::atomic<uint32_t> m_refCount{0};
    std::string m_bytes;
};

/// One immutable compilation snapshot. Slang resolves imports on demand through
/// this mount; unrelated modules never enter its session or diagnostics.
class SlangModuleFiles final : public ISlangFileSystemExt
{
  public:
    static constexpr const char *Mount = "__infernux_modules__";

    SlangModuleFiles(const ShaderSourceModule &source, const std::vector<ShaderSourceModule> &libraries)
    {
        Add(GetSlangGeometryModule());
        Add(GetSlangLightingModule());
        Add(GetSlangFrameModule());
        for (const auto &library : libraries)
            Add(library);
        Add(source);
    }
    SLANG_IUNKNOWN_ALL
    ISlangUnknown *getInterface(const SlangUUID &uuid)
    {
        return uuid == ISlangFileSystemExt::getTypeGuid() || uuid == ISlangFileSystem::getTypeGuid() ||
                       uuid == ISlangCastable::getTypeGuid() || uuid == ISlangUnknown::getTypeGuid()
                   ? this
                   : nullptr;
    }
    SLANG_NO_THROW void *SLANG_MCALL castAs(const SlangUUID &uuid) override
    {
        return getInterface(uuid);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(const char *path, ISlangBlob **outBlob) override
    try {
        *outBlob = nullptr;
        const auto file = Path(path);
        const auto found = m_sources.find(file.generic_u8string());
        std::string bytes;
        if (found != m_sources.end()) {
            const auto &module = *found->second;
            // Preserve the author's original source location despite the
            // virtual module-name path used for import resolution.
            bytes = "#line 1 \"";
            for (const char character : module.path) {
                if (character == '\\' || character == '"')
                    bytes += '\\';
                bytes += character;
            }
            bytes += "\"\n";
            bytes += module.source;
        } else if (file.extension() == ".slang-module") {
            // Slang resolves packaged standard modules beside its library, or
            // through the host's configured standard-module search path.
            std::ifstream input(file, std::ios::binary);
            if (!input)
                return SLANG_E_NOT_FOUND;
            bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        } else {
            return SLANG_E_NOT_FOUND;
        }
        return Blob(std::move(bytes), outBlob);
    } catch (const std::bad_alloc &) {
        return SLANG_E_OUT_OF_MEMORY;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getFileUniqueIdentity(const char *path, ISlangBlob **out) override
    {
        SlangPathType type;
        if (SLANG_FAILED(getPathType(path, &type)))
            return SLANG_E_NOT_FOUND;
        return PathBlob(OriginalPath(path), out);
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(SlangPathType type, const char *from, const char *relative,
                                                            ISlangBlob **out) override
    {
        auto base = OriginalPath(from);
        if (type == SLANG_PATH_TYPE_FILE)
            base = base.parent_path();
        return PathBlob((base / Path(relative)).lexically_normal(), out);
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL getPathType(const char *path, SlangPathType *out) override
    {
        const auto file = Path(path);
        if (m_sources.find(file.generic_u8string()) != m_sources.end()) {
            *out = SLANG_PATH_TYPE_FILE;
            return SLANG_OK;
        }
        std::error_code error;
        if (file.extension() == ".slang-module" && std::filesystem::is_regular_file(file, error)) {
            *out = SLANG_PATH_TYPE_FILE;
            return SLANG_OK;
        }
        return SLANG_E_NOT_FOUND;
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL getPath(PathKind kind, const char *path, ISlangBlob **out) override
    {
        if (kind == PathKind::OperatingSystem)
            return SLANG_E_NOT_IMPLEMENTED;
        return PathBlob(OriginalPath(path), out);
    }
    SLANG_NO_THROW void SLANG_MCALL clearCache() override
    {
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(const char *, FileSystemContentsCallBack,
                                                                 void *) override
    {
        return SLANG_E_NOT_IMPLEMENTED;
    }
    SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() override
    {
        return OSPathKind::None;
    }

  private:
    static SlangResult Blob(std::string bytes, ISlangBlob **out)
    {
        *out = new SlangModuleBlob(std::move(bytes));
        (*out)->addRef();
        return SLANG_OK;
    }
    static SlangResult PathBlob(const std::filesystem::path &path, ISlangBlob **out)
    {
        const auto text = path.generic_u8string();
        return Blob(std::string(text.begin(), text.end()), out);
    }
    std::filesystem::path OriginalPath(const char *path) const
    {
        auto file = Path(path);
        const auto found = m_sources.find(file.generic_u8string());
        return found == m_sources.end() ? file : Path(found->second->path);
    }
    static std::filesystem::path Path(std::string_view value)
    {
        return std::filesystem::u8path(value).lexically_normal();
    }
    void Add(const ShaderSourceModule &module)
    {
        const auto name = (Path(Mount) / Path(module.name + ".slang")).generic_u8string();
        if (!m_sources.emplace(name, &module).second)
            throw ShaderCompilationError({{"compiler.module_duplicate",
                                           "error",
                                           module.path,
                                           1,
                                           1,
                                           1,
                                           1,
                                           "Duplicate shader module name: " + module.name,
                                           "module",
                                           {}}});
        m_sources.emplace(Path(module.path).generic_u8string(), &module);
    }

    std::atomic<uint32_t> m_refCount{0};
    using PathKey = decltype(std::filesystem::path{}.generic_u8string());
    std::unordered_map<PathKey, const ShaderSourceModule *> m_sources;
};

} // namespace infernux
