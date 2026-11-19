#include "ShaderAuthorProgramCache.h"
#include "SlangCompiler.h"
#include <algorithm>
#include <function/renderer/shader/ShaderTargetArchive.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetFormatRegistry.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <platform/filesystem/InxPath.h>
#include <unordered_map>

namespace infernux
{
namespace
{
using namespace shader_target_archive;
constexpr std::string_view AuthorProgramMagic = "INXSAUTH2";

const ShaderCacheInput *PathInput(const std::string &path, const std::vector<ShaderCacheInput> &inputs)
{
    if (path.empty())
        return nullptr;
    const auto key = LexicalFilesystemPathKey(NormalizePortablePath(path));
    const auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto &input) {
        return !input.path.empty() && LexicalFilesystemPathKey(NormalizePortablePath(input.path)) == key;
    });
    return found == inputs.end() ? nullptr : &*found;
}

using InputPaths = std::unordered_map<std::string, std::string>;

std::string RestoredPath(const std::string &id, const InputPaths &paths)
{
    const auto found = paths.find(id);
    if (found == paths.end() || found->second.empty())
        throw std::invalid_argument("shader author cache source identity is unavailable");
    return found->second;
}

std::string StoredSchema(const std::string &schema, const std::vector<ShaderCacheInput> &inputs)
{
    auto value = nlohmann::json::parse(schema);
    const auto vertex = value.find("vertex_source");
    if (vertex != value.end() && vertex->is_string())
        if (const auto *input = PathInput(vertex->get<std::string>(), inputs))
            *vertex = nlohmann::json{{"shader_cache_input", input->id}};
    return value.dump();
}

std::string RestoredSchema(const std::string &schema, const InputPaths &paths)
{
    auto value = nlohmann::json::parse(schema);
    const auto vertex = value.find("vertex_source");
    if (vertex != value.end() && vertex->is_object() && vertex->size() == 1 && vertex->contains("shader_cache_input"))
        *vertex = RestoredPath(vertex->at("shader_cache_input").get<std::string>(), paths);
    return value.dump();
}

void AppendDiagnostic(std::string &out, const ShaderDiagnostic &diagnostic, const std::vector<ShaderCacheInput> &inputs)
{
    AppendString(out, diagnostic.code);
    AppendString(out, diagnostic.severity);
    const auto *input = PathInput(diagnostic.path, inputs);
    binary_codec::AppendUnsigned(out, static_cast<uint8_t>(input != nullptr));
    AppendString(out, input ? input->id : diagnostic.path);
    AppendU32(out, diagnostic.line);
    AppendU32(out, diagnostic.column);
    AppendU32(out, diagnostic.endLine);
    AppendU32(out, diagnostic.endColumn);
    AppendString(out, diagnostic.message);
    AppendString(out, diagnostic.phase);
    AppendString(out, diagnostic.target);
}

ShaderDiagnostic ReadDiagnostic(Reader &in, const InputPaths &paths)
{
    ShaderDiagnostic diagnostic;
    diagnostic.code = in.ReadString();
    diagnostic.severity = in.ReadString();
    const auto sourceReference = in.ReadUnsigned<uint8_t>();
    if (sourceReference > 1)
        throw std::invalid_argument("shader author cache has an invalid source reference");
    diagnostic.path = in.ReadString();
    if (sourceReference)
        diagnostic.path = RestoredPath(diagnostic.path, paths);
    diagnostic.line = in.ReadU32();
    diagnostic.column = in.ReadU32();
    diagnostic.endLine = in.ReadU32();
    diagnostic.endColumn = in.ReadU32();
    diagnostic.message = in.ReadString();
    diagnostic.phase = in.ReadString();
    diagnostic.target = in.ReadString();
    return diagnostic;
}

std::optional<ShaderAuthorProgramRecord> DecodeProgram(const std::string &bytes, const InputPaths &paths)
{
    try {
        Reader in(bytes);
        // Older author payloads embedded cooker-machine paths. They must be
        // rebuilt; the shared cache envelope and target archive stay unchanged.
        if (in.ReadString(false) != AuthorProgramMagic)
            return std::nullopt;
        ShaderAuthorProgramRecord record;
        record.schema = RestoredSchema(in.ReadString(false), paths);
        record.dependencies = ReadVector<std::string>(in, [](Reader &source) { return source.ReadString(false); });
        auto compiled = std::make_shared<ShaderCompilation>();
        compiled->diagnostics =
            ReadVector<ShaderDiagnostic>(in, [&paths](Reader &source) { return ReadDiagnostic(source, paths); });
        compiled->targets = ReadVector<ShaderTargetCode>(
            in, [](Reader &source) { return DecodeShaderTarget(source.ReadString(false)); });
        if (!in.AtEnd() || compiled->targets.empty())
            return std::nullopt;
        record.compilation = std::move(compiled);
        return record;
    } catch (const std::invalid_argument &) {
        return std::nullopt;
    } catch (const nlohmann::json::exception &) {
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

ShaderAuthorSurfaceReader ShaderProjectSurfaceReader(const AssetDatabase &database)
{
    return [&database,
            sources =
                std::shared_ptr<const ShaderAuthorSurfaceSources>{}]() mutable -> const ShaderAuthorSurfaceSources & {
        if (sources)
            return *sources;
        // The database already owns a stable, immutable asset index. Rebuild this
        // shader-only view once per index, shared by all actual ray consumers.
        struct Entry
        {
            std::weak_ptr<const AssetCatalogSnapshot> snapshot;
            std::shared_ptr<const ShaderAuthorSurfaceSources> sources;
        };
        static std::mutex mutex;
        static std::vector<Entry> cache;
        const auto snapshot = database.GetCatalogSnapshot();
        const std::lock_guard lock(mutex);
        cache.erase(
            std::remove_if(cache.begin(), cache.end(), [](const auto &entry) { return entry.snapshot.expired(); }),
            cache.end());
        for (const auto &entry : cache)
            if (entry.snapshot.lock() == snapshot) {
                sources = entry.sources;
                return *sources;
            }
        auto result = std::make_shared<ShaderAuthorSurfaceSources>();
        for (const auto &[guid, path] : snapshot->GetShaderSources()) {
            const auto *kind = asset_formats::FindShaderSourceKind(path);
            if (!kind || std::string_view(kind->stage) != "fragment")
                continue;
            std::string relative;
            if (!TryMakeRelativeFilesystemPath(path, database.GetProjectRoot(), relative))
                continue;
            constexpr std::string_view cookedPrefix = "Library/Artifacts/Blob/";
            if (relative.compare(0, cookedPrefix.size(), cookedPrefix) == 0)
                relative.erase(0, cookedPrefix.size());
            if (relative.compare(0, 7, "Assets/") == 0 || relative.compare(0, 9, "Packages/") == 0)
                result->emplace_back(guid, path);
        }
        std::sort(result->begin(), result->end());
        sources = std::move(result);
        cache.push_back({snapshot, sources});
        return *sources;
    };
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

std::vector<std::string> ShaderAuthorDependencies(const AssetDatabase &database,
                                                  const std::vector<ShaderCacheInput> &inputs)
{
    // Several authored files can contribute to one generated Slang module.
    // The first recorded input owns the program and cannot be its own dependency.
    std::vector<std::string> dependencies;
    for (size_t i = 1; i < inputs.size(); ++i) {
        const auto &input = inputs[i];
        if (input.id != inputs.front().id && asset_formats::FindShaderSourceKind(input.path)) {
            auto guid = database.GetGuidFromPath(input.path);
            if (!guid.empty())
                dependencies.push_back(std::move(guid));
        }
    }
    std::sort(dependencies.begin(), dependencies.end());
    dependencies.erase(std::unique(dependencies.begin(), dependencies.end()), dependencies.end());
    return dependencies;
}

std::optional<ShaderAuthorProgramRecord>
FindShaderAuthorProgram(const AssetDatabase &database, std::string_view selection,
                        const ShaderAuthorSurfaceReader &surfaces, const std::vector<ShaderCacheInput> &loadedInputs,
                        std::vector<ShaderCacheInput> *matchedInputs, const std::string &buildCache)
{
    const auto &project = database.GetProjectRoot();
    if (project.empty())
        return std::nullopt;
    std::unordered_map<std::string, std::optional<uint64_t>> revisions;
    InputPaths paths;
    for (const auto &input : loadedInputs) {
        revisions.emplace(input.id, input.revision);
        if (!input.path.empty())
            paths.emplace(input.id, input.path);
    }
    std::vector<ShaderCacheInput> verified;
    const auto revision = [&](const std::string &id) -> std::optional<uint64_t> {
        auto [it, inserted] = revisions.try_emplace(id);
        if (inserted || paths.find(id) == paths.end()) {
            const auto location = id.find('@');
            const bool byPath = id.compare(0, 5, "path:") == 0;
            std::string relative;
            if (byPath && !TryNormalizePortableRelativePath(id.substr(5), relative))
                return std::nullopt;
            const auto path = byPath ? FromFsPath(ToFsPath(project) / ToFsPath(relative))
                                     : database.GetPathFromGuid(id.substr(0, location));
            if (!byPath && location != std::string::npos && (path.empty() || ShaderAuthorInputId(database, path) != id))
                return std::nullopt;
            if (!path.empty()) {
                paths.emplace(id, path);
                if (inserted) {
                    std::vector<char> bytes;
                    if (database.ReadFile(path, bytes))
                        it->second = binary_codec::Fnv1a64({bytes.data(), bytes.size()});
                }
            }
        }
        if (it->second)
            verified.push_back({id, *it->second, paths[id]});
        return it->second;
    };
    for (const auto &root : {buildCache.empty() ? ProjectCacheDirectory(project) : buildCache,
                             FromFsPath(ToFsPath(project) / "Library")}) {
        verified.clear();
        auto bytes = ReadShaderProgramCache(root, selection, revision);
        if (bytes) {
            // Only dependencies proven by this exact envelope can restore a
            // source reference, including inputs needed only by diagnostics.
            InputPaths verifiedPaths;
            for (const auto &input : verified)
                verifiedPaths.emplace(input.id, input.path);
            if (auto record = DecodeProgram(*bytes, verifiedPaths)) {
                const auto schema = nlohmann::json::parse(record->schema);
                const auto catalog = schema.find("ray_surface_catalog");
                if (catalog != schema.end() && catalog->is_array()) {
                    const auto &current = surfaces();
                    bool same = catalog->size() == current.size();
                    for (size_t i = 0; same && i < current.size(); ++i)
                        same = catalog->at(i).get<std::string>() == current[i].first;
                    if (!same)
                        continue;
                }

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
    AppendString(out, AuthorProgramMagic);
    AppendString(out, StoredSchema(record.schema, inputs));
    AppendVector(out, record.dependencies, [](auto &bytes, const auto &id) { AppendString(bytes, id); });
    AppendVector(out, record.compilation->diagnostics,
                 [&inputs](auto &bytes, const auto &diagnostic) { AppendDiagnostic(bytes, diagnostic, inputs); });
    AppendVector(out, record.compilation->targets,
                 [](auto &bytes, const auto &target) { AppendString(bytes, EncodeShaderTarget(target)); });
    return WriteShaderProgramCache(cacheDirectory, selection, inputs, out);
}
} // namespace infernux
