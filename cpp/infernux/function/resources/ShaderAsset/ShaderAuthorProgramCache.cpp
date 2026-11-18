#include "ShaderAuthorProgramCache.h"
#include "SlangCompiler.h"
#include <function/renderer/shader/ShaderTargetArchive.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <platform/filesystem/InxPath.h>
#include <algorithm>
#include <unordered_map>

namespace infernux
{
namespace
{
using namespace shader_target_archive;

void AppendDiagnostic(std::string &out, const ShaderDiagnostic &diagnostic)
{
    AppendString(out, diagnostic.code);
    AppendString(out, diagnostic.severity);
    AppendString(out, diagnostic.path);
    AppendU32(out, diagnostic.line);
    AppendU32(out, diagnostic.column);
    AppendU32(out, diagnostic.endLine);
    AppendU32(out, diagnostic.endColumn);
    AppendString(out, diagnostic.message);
    AppendString(out, diagnostic.phase);
    AppendString(out, diagnostic.target);
}

ShaderDiagnostic ReadDiagnostic(Reader &in)
{
    ShaderDiagnostic diagnostic;
    diagnostic.code = in.ReadString();
    diagnostic.severity = in.ReadString();
    diagnostic.path = in.ReadString();
    diagnostic.line = in.ReadU32();
    diagnostic.column = in.ReadU32();
    diagnostic.endLine = in.ReadU32();
    diagnostic.endColumn = in.ReadU32();
    diagnostic.message = in.ReadString();
    diagnostic.phase = in.ReadString();
    diagnostic.target = in.ReadString();
    return diagnostic;
}

std::optional<ShaderAuthorProgramRecord> DecodeProgram(const std::string &bytes)
{
    try {
        Reader in(bytes);
        ShaderAuthorProgramRecord record;
        record.schema = in.ReadString(false);
        record.dependencies = ReadVector<std::string>(in, [](Reader &source) { return source.ReadString(false); });
        auto compiled = std::make_shared<ShaderCompilation>();
        compiled->diagnostics = ReadVector<ShaderDiagnostic>(in, ReadDiagnostic);
        compiled->targets = ReadVector<ShaderTargetCode>(
            in, [](Reader &source) { return DecodeShaderTarget(source.ReadString(false)); });
        if (!in.AtEnd() || compiled->targets.empty())
            return std::nullopt;
        record.compilation = std::move(compiled);
        return record;
    } catch (const std::invalid_argument &) {
        return std::nullopt;
    }
}
} // namespace

ShaderCodeTarget RuntimeShaderAuthorTarget() noexcept
{
#ifdef __EMSCRIPTEN__
    return ShaderCodeTarget::Wgsl;
#else
    return ShaderCodeTarget::SpirV;
#endif
}

std::string ShaderAuthorProgramSelection(std::string_view authorGuid, std::string_view vertexGuid,
                                         ShaderCodeTarget target)
{
    std::string selection(SlangCompiler::GetAuthorContract());
    shader_target_archive::AppendString(selection, authorGuid);
    shader_target_archive::AppendString(selection, vertexGuid);
    shader_target_archive::AppendEnum(selection, target);
    return selection;
}

std::string ShaderAuthorInputId(const AssetDatabase &database, const std::string &path)
{
    const auto guid = database.GetGuidFromPath(path);
    std::string relative;
    if (TryMakeRelativeFilesystemPath(path, database.GetProjectRoot(), relative)) {
        if (guid.empty())
            return "path:" + relative;
        // GameBuilder retains authored namespaces below the cooked blob directory.
        constexpr std::string_view cookedPrefix = "Library/Artifacts/Blob/";
        if (relative.compare(0, cookedPrefix.size(), cookedPrefix) == 0)
            relative.erase(0, cookedPrefix.size());
        // Moving an author with the same GUID can change its relative imports.
        // Project and package paths survive the cook. Engine built-ins instead
        // have a different installation directory on each platform.
        if (relative.compare(0, 7, "Assets/") == 0 || relative.compare(0, 9, "Packages/") == 0)
            return guid + "@" + relative;
    }
    return guid;
}

std::optional<ShaderAuthorProgramRecord> FindShaderAuthorProgram(const AssetDatabase &database,
                                                                 std::string_view selection,
                                                                 const std::vector<ShaderCacheInput> &loadedInputs,
                                                                 std::vector<ShaderCacheInput> *matchedInputs,
                                                                 const std::string &buildCache)
{
    const auto &project = database.GetProjectRoot();
    if (project.empty())
        return std::nullopt;
    std::unordered_map<std::string, std::optional<uint64_t>> revisions;
    for (const auto &input : loadedInputs)
        revisions.emplace(input.id, input.revision);
    std::vector<ShaderCacheInput> verified;
    const auto revision = [&](const std::string &id) -> std::optional<uint64_t> {
        auto [it, inserted] = revisions.try_emplace(id);
        if (inserted) {
            const auto location = id.find('@');
            const bool byPath = id.compare(0, 5, "path:") == 0;
            std::string relative;
            if (byPath && !TryNormalizePortableRelativePath(id.substr(5), relative))
                return std::nullopt;
            const auto path = byPath ? FromFsPath(ToFsPath(project) / ToFsPath(relative))
                                     : database.GetPathFromGuid(id.substr(0, location));
            if (!byPath && location != std::string::npos && (path.empty() || ShaderAuthorInputId(database, path) != id))
                return std::nullopt;
            std::vector<char> bytes;
            if (!path.empty() && database.ReadFile(path, bytes))
                it->second = binary_codec::Fnv1a64({bytes.data(), bytes.size()});
        }
        if (matchedInputs && it->second)
            verified.push_back({id, *it->second});
        return it->second;
    };
    for (const auto &root : {buildCache.empty() ? ProjectCacheDirectory(project) : buildCache,
                             FromFsPath(ToFsPath(project) / "Library")}) {
        verified.clear();
        auto bytes = ReadShaderProgramCache(root, selection, revision);
        if (bytes) {
            if (auto record = DecodeProgram(*bytes)) {
                if (matchedInputs)
                    *matchedInputs = std::move(verified);
                return record;
            }
        }
    }
    return std::nullopt;
}

bool StoreShaderAuthorProgram(const std::string &cacheDirectory, std::string_view selection,
                              const std::vector<ShaderCacheInput> &inputs, const ShaderAuthorProgramRecord &record)
{
    using namespace shader_target_archive;
    if (inputs.empty() || std::any_of(inputs.begin(), inputs.end(), [](const auto &input) { return input.id.empty(); }))
        return false;
    std::string out;
    AppendString(out, record.schema);
    AppendVector(out, record.dependencies, [](auto &bytes, const auto &id) { AppendString(bytes, id); });
    AppendVector(out, record.compilation->diagnostics, AppendDiagnostic);
    AppendVector(out, record.compilation->targets,
                 [](auto &bytes, const auto &target) { AppendString(bytes, EncodeShaderTarget(target)); });
    return WriteShaderProgramCache(cacheDirectory, selection, inputs, out);
}
} // namespace infernux
