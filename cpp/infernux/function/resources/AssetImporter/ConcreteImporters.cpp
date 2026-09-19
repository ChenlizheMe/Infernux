#include "ConcreteImporters.h"
#include <function/resources/InxMesh/InxMesh.h>

#include <core/log/InxLog.h>
#include <function/resources/InxMaterial/MaterialDocumentValidation.h>
#include <function/resources/InxMesh/MeshArtifact.h>
#include <function/resources/InxMesh/MeshImportSettings.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxSkinnedMesh/SkinnedMeshArtifact.h>
#include <function/resources/InxTexture/TextureArtifact.h>
#include <function/resources/InxTexture/TextureDecoder.h>
#include <function/resources/RenderTexture/RenderTextureArtifact.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <unordered_set>
#include <vector>
#if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
#include <SDL3/SDL.h>
#endif

namespace infernux
{

namespace
{

// A .blend file is an editor source, never a runtime format. The conversion
// lives inside this import request and is discarded once binary artifacts exist.
class BlenderSource
{
  public:
    explicit BlenderSource(const ImportRequest &request)
    {
        if (request.blenderExecutable.empty() || request.blenderExportScript.empty())
            throw std::runtime_error(
                "Blender 5.2 model import is not configured; install Model Authoring in Infernux Hub or select Blender in editor preferences");
        if (request.projectRoot.empty())
            throw std::logic_error("Blender import requires a project Library directory");
        const auto parent = ToFsPath(request.projectRoot) / "Library" / "ModelImport";
        std::filesystem::create_directories(parent);
        static std::atomic<uint64_t> sequence{0};
        directory = parent / (std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                              std::to_string(sequence.fetch_add(1)));
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("Cannot reserve Blender import directory");
    }

    ~BlenderSource()
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
        if (error)
            INXLOG_WARN("Cannot remove Blender import staging directory: ", error.message());
    }

    std::string Convert(const ImportRequest &request) const
    {
#if defined(__EMSCRIPTEN__) || defined(__ANDROID__)
        throw std::runtime_error("Blender source import is available only in desktop editors");
#else
        const std::string output = FromFsPath(directory / "model.glb");
        const char *args[] = {request.blenderExecutable.c_str(),
                              "--background",
                              "--factory-startup",
                              "--disable-autoexec",
                              "--python-exit-code",
                              "1",
                              "--python",
                              request.blenderExportScript.c_str(),
                              "--",
                              request.sourcePath.c_str(),
                              output.c_str(),
                              nullptr};
        const SDL_PropertiesID props = SDL_CreateProperties();
        SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, const_cast<char **>(args));
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_APP);
        SDL_SetBooleanProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_TO_STDOUT_BOOLEAN, true);
        std::unique_ptr<SDL_Process, decltype(&SDL_DestroyProcess)> process(SDL_CreateProcessWithProperties(props),
                                                                            SDL_DestroyProcess);
        SDL_DestroyProperties(props);
        if (!process)
            throw std::runtime_error("Cannot launch Blender: " + std::string(SDL_GetError()));
        // Drain a non-blocking pipe and retain a bounded diagnostic tail. No
        // inherited log-file handles can keep staging files locked on Windows.
        std::string detail;
        SDL_IOStream *stream = SDL_GetProcessOutput(process.get());
        const auto drain = [&]() {
            char buffer[4096];
            for (int block = 0; block < 16; ++block) {
                const size_t count = SDL_ReadIO(stream, buffer, sizeof(buffer));
                if (count == 0)
                    break;
                detail.append(buffer, count);
                if (detail.size() > 8000)
                    detail.erase(0, detail.size() - 8000);
            }
        };
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        int exitCode = 0;
        bool timedOut = false;
        while (!SDL_WaitProcess(process.get(), false, &exitCode)) {
            drain();
            if (std::chrono::steady_clock::now() >= deadline) {
                SDL_KillProcess(process.get(), true);
                SDL_WaitProcess(process.get(), true, &exitCode);
                timedOut = true;
                break;
            }
            SDL_Delay(10);
        }
        drain();
        if (timedOut || exitCode != 0 || !std::filesystem::is_regular_file(ToFsPath(output))) {
            throw std::runtime_error(std::string(timedOut ? "Blender import timed out" : "Blender import failed") +
                                     " for '" + request.sourcePath + "':\n" + detail);
        }
        return output;
#endif
    }

  private:
    std::filesystem::path directory;
};

bool IsBuiltinTextureToken(const std::string &value)
{
    return value == "white" || value == "black" || value == "normal";
}

void RejectPathOnlyReference(const std::string &guid, const std::string &pathHint, const std::string &location)
{
    if (guid.empty() && !pathHint.empty()) {
        throw std::runtime_error(location +
                                 " must provide a GUID; path_hint is non-authoritative and cannot resolve an asset");
    }
}

// A source node's path is the public lookup key, but it is not an identity:
// artists are allowed to rename a node or reorder siblings in the DCC file.
// Persist compact geometry signatures beside the path so a unique renamed
// node can retain its subresource GUID without introducing a second identity
// system. These are deliberately fast FNV-1a signatures, not content
// integrity checks; the source content hash still owns import invalidation.
std::string ModelNodeIdentityKey(const InxMesh &mesh, size_t nodeIndex, const std::vector<std::string> &path,
                                 bool includeParentPath)
{
    const auto &nodes = mesh.GetModelNodes();
    if (nodeIndex >= nodes.size() || nodes[nodeIndex].nodeGroup < 0)
        return {};
    const auto geometry = mesh.GetModelSourceGeometry();
    if (!geometry)
        return {};

    uint64_t hash = 14695981039346656037ull;
    const auto mix = [&hash](const void *data, size_t size) {
        const auto *bytes = static_cast<const unsigned char *>(data);
        for (size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ull;
        }
    };
    const auto mixString = [&mix](std::string_view value) {
        mix(value.data(), value.size());
        const unsigned char separator = 0xff;
        mix(&separator, sizeof(separator));
    };
    const auto mixFloat = [&mix](float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        mix(&bits, sizeof(bits));
    };
    const auto mixVec2 = [&mixFloat](const glm::vec2 &value) {
        mixFloat(value.x);
        mixFloat(value.y);
    };
    const auto mixVec3 = [&mixFloat](const glm::vec3 &value) {
        mixFloat(value.x);
        mixFloat(value.y);
        mixFloat(value.z);
    };
    const auto mixVec4 = [&mixFloat](const glm::vec4 &value) {
        mixFloat(value.x);
        mixFloat(value.y);
        mixFloat(value.z);
        mixFloat(value.w);
    };
    const auto mixUvec4 = [&mix](const glm::uvec4 &value) { mix(&value.x, sizeof(uint32_t) * 4); };
    // The strict key includes the parent source path so identical meshes under
    // different pivots remain distinct while a leaf rename stays matchable.
    // A second geometry-only key is persisted for a parent rename; it is only
    // accepted when it is unique in the previous import, never guessed.
    if (includeParentPath) {
        for (size_t index = 0; index + 1 < path.size(); ++index)
            mixString(path[index]);
    }
    for (const auto &submesh : geometry->subMeshes) {
        if (static_cast<int32_t>(submesh.nodeGroup) != nodes[nodeIndex].nodeGroup)
            continue;
        mix(&submesh.materialSlot, sizeof(submesh.materialSlot));
        mix(&submesh.vertexCount, sizeof(submesh.vertexCount));
        mix(&submesh.indexCount, sizeof(submesh.indexCount));
        mixVec3(submesh.boundsMin);
        mixVec3(submesh.boundsMax);
        if (submesh.vertexStart + submesh.vertexCount <= geometry->vertices.size()) {
            for (uint32_t vertexIndex = 0; vertexIndex < submesh.vertexCount; ++vertexIndex) {
                const auto &vertex = geometry->vertices[submesh.vertexStart + vertexIndex];
                mixVec3(vertex.pos);
                mixVec3(vertex.normal);
                mixVec4(vertex.tangent);
                mixVec3(vertex.color);
                mixVec2(vertex.texCoord);
                mixUvec4(vertex.boneIndices);
                mixVec4(vertex.boneWeights);
            }
        }
        for (uint32_t index = 0; index < submesh.indexCount; ++index) {
            // Source buffers concatenate node geometry; allocation offsets
            // must not distinguish identical meshes or change on a reorder.
            const uint32_t local = geometry->indices.at(submesh.indexStart + index) - submesh.vertexStart;
            mix(&local, sizeof(local));
        }
    }
    std::ostringstream result;
    result << (includeParentPath ? "v2/" : "g2/") << std::hex << std::setw(16) << std::setfill('0') << hash;
    return result.str();
}

} // namespace

ImportArtifact PrefabImporter::Import(const ImportRequest &request) const
{
    std::ifstream stream(ToFsPath(request.sourcePath));
    if (!stream)
        throw std::runtime_error("Cannot open Prefab source: " + request.sourcePath);
    const auto document = nlohmann::json::parse(stream);
    if (!document.is_object() || !document.contains("root_object") || !document["root_object"].is_object())
        throw std::runtime_error("Prefab source requires a root_object: " + request.sourcePath);
    ImportArtifact artifact(request.metadata);
    artifact.dependenciesAuthoritative = true;
    const auto variant = document.find("variant");
    if (variant != document.end()) {
        if (!variant->is_object() || !variant->contains("guid") || !(*variant)["guid"].is_string() ||
            (*variant)["guid"].get<std::string>().empty())
            throw std::runtime_error("Prefab Variant requires a base GUID: " + request.sourcePath);
        artifact.dependencies.push_back((*variant)["guid"].get<std::string>());
    }
    return artifact;
}

ImportArtifact RenderTextureImporter::Import(const ImportRequest &request) const
{
    std::ifstream stream(ToFsPath(request.sourcePath));
    if (!stream)
        throw std::runtime_error("Cannot open RenderTexture source: " + request.sourcePath);
    const auto description = RenderTextureArtifact::ParseDocument(nlohmann::json::parse(stream));
    ImportArtifact result(request.metadata);
    result.dependenciesAuthoritative = true;
    result.runtimeCpuArtifacts.push_back(
        {ImportArtifact::RuntimeArtifactKind::Primary, ResourceType::RenderTexture,
         RenderTextureArtifact::Encode(description, request.metadata.GetDataAs<std::string>("content_hash"))});
    return result;
}

ImportArtifact TextureImporter::Import(const ImportRequest &request) const
{
    ImportArtifact artifact(request.metadata);
    EnsureDefaultSettings(artifact.metadata);
    std::string extension = FromFsPath(ToFsPath(request.sourcePath).extension());
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == ".inxvfield") {
        artifact.metadata.AddMetadata("texture_type", std::string("vector_field"));
        artifact.metadata.AddMetadata("srgb", false);
        artifact.metadata.AddMetadata("texture_compression", std::string("none"));
        const std::string format = artifact.metadata.GetDataAs<std::string>("texture_format");
        if (format == "auto")
            artifact.metadata.AddMetadata("texture_format", std::string("rgba16_float"));
        else if (format != "rgba16_float" && format != "rgba32_float")
            throw std::invalid_argument("VectorField textures require rgba16_float or rgba32_float storage");
    } else if (extension == ".inxsdf") {
        artifact.metadata.AddMetadata("texture_type", std::string("sdf"));
        artifact.metadata.AddMetadata("srgb", false);
        artifact.metadata.AddMetadata("generate_mipmaps", false);
        artifact.metadata.AddMetadata("wrap_mode", std::string("clamp"));
        artifact.metadata.AddMetadata("texture_compression", std::string("none"));
        const std::string format = artifact.metadata.GetDataAs<std::string>("texture_format");
        if (format == "auto")
            artifact.metadata.AddMetadata("texture_format", std::string("rgba16_float"));
        else if (format != "rgba16_float" && format != "rgba32_float")
            throw std::invalid_argument("SignedDistanceField textures require rgba16_float or rgba32_float storage");
    } else if (artifact.metadata.GetDataAs<std::string>("texture_type") == "vector_field") {
        throw std::invalid_argument("VectorField textures must use the .inxvfield source format");
    } else if (artifact.metadata.GetDataAs<std::string>("texture_type") == "sdf") {
        throw std::invalid_argument("SignedDistanceField textures must use the .inxsdf source format");
    }
    if (!artifact.metadata.HasKey("content_hash"))
        throw std::logic_error("TextureImporter metadata has no source content hash");
    const auto cpuData = TextureDecoder::Decode(request.sourcePath, artifact.metadata);
    if (!cpuData || !cpuData->IsValid())
        throw std::runtime_error("TextureImporter failed to build the runtime texture artifact");
    artifact.metadata.AddMetadata("artifact_width", static_cast<int>(cpuData->mipLevels.front().width));
    artifact.metadata.AddMetadata("artifact_height", static_cast<int>(cpuData->mipLevels.front().height));
    artifact.metadata.AddMetadata("artifact_depth", static_cast<int>(cpuData->mipLevels.front().depth));
    artifact.metadata.AddMetadata("artifact_mip_count", static_cast<int>(cpuData->mipLevels.size()));
    artifact.metadata.AddMetadata("artifact_dimension",
                                  std::string(cpuData->dimension == TextureDimension::Texture3D ? "3d" : "2d"));
    artifact.metadata.AddMetadata("artifact_srgb", TextureFormatIsSrgb(cpuData->format));
    artifact.metadata.AddMetadata("artifact_format", std::string(TextureFormatName(cpuData->format)));
    artifact.runtimeCpuArtifacts.push_back(ImportArtifact::RuntimeCpuArtifact{
        ImportArtifact::RuntimeArtifactKind::Primary, ResourceType::Texture,
        TextureArtifact::Serialize(*cpuData, artifact.metadata.GetDataAs<std::string>("content_hash"))});
    return artifact;
}

ImportArtifact MaterialImporter::Import(const ImportRequest &request) const
{
    ImportArtifact artifact(request.metadata);
    artifact.dependencies = ScanDependencies(request);
    artifact.dependenciesAuthoritative = true;
    return artifact;
}

std::vector<std::string> MaterialImporter::ScanDependencies(const ImportRequest &request) const
{
    std::unordered_set<std::string> deps;

    nlohmann::json root;
    try {
        std::ifstream file(ToFsPath(request.sourcePath));
        if (!file.is_open())
            throw std::runtime_error("failed to open material document");
        file >> root;
    } catch (const std::exception &e) {
        throw std::runtime_error("MaterialImporter failed to parse '" + request.sourcePath + "': " + e.what());
    } catch (...) {
        throw std::runtime_error("MaterialImporter failed to parse '" + request.sourcePath + "'");
    }
    try {
        material_document_validation::ValidateMaterialDocument(root, request.sourcePath);
    } catch (const std::exception &e) {
        throw std::runtime_error("MaterialImporter rejected '" + request.sourcePath + "': " + e.what());
    }

    // Canonical shader references carry GUID plus stable compiler/path hints.
    auto shadersIt = root.find("shaders");
    if (shadersIt != root.end() && shadersIt->is_object()) {
        for (const auto &key : {"vertex", "fragment"}) {
            auto it = shadersIt->find(key);
            if (it == shadersIt->end())
                continue;
            std::string depGuid;
            const auto guidIt = it->find("guid");
            if (guidIt != it->end() && guidIt->is_string())
                depGuid = guidIt->get<std::string>();
            const auto pathIt = it->find("path_hint");
            const std::string pathHint =
                pathIt != it->end() && pathIt->is_string() ? pathIt->get<std::string>() : std::string{};
            RejectPathOnlyReference(depGuid, pathHint, "material shaders." + std::string(key));
            if (!depGuid.empty())
                deps.insert(depGuid);
        }
    }

    // Texture dependencies (properties with type == 6 == Texture2D)
    auto propsIt = root.find("properties");
    if (propsIt != root.end() && propsIt->is_object()) {
        for (auto &[propName, propVal] : propsIt->items()) {
            if (!propVal.is_object())
                continue;
            auto typeIt = propVal.find("type");
            if (typeIt == propVal.end() || !typeIt->is_number_integer())
                continue;
            int ptype = typeIt->get<int>();
            if (ptype != 6) // 6 == Texture2D
                continue;
            auto guidIt = propVal.find("guid");
            if (guidIt != propVal.end() && guidIt->is_string()) {
                std::string texGuid = guidIt->get<std::string>();
                if (!texGuid.empty() && !IsBuiltinTextureToken(texGuid))
                    deps.insert(texGuid);
            }
        }
    }

    std::vector<std::string> ordered(deps.begin(), deps.end());
    std::sort(ordered.begin(), ordered.end());
    return ordered;
}

ImportArtifact RenderEffectImporter::Import(const ImportRequest &request) const
{
    ImportArtifact artifact(request.metadata);
    artifact.dependencies = ScanDependencies(request);
    artifact.dependenciesAuthoritative = true;
    return artifact;
}

std::vector<std::string> RenderEffectImporter::ScanDependencies(const ImportRequest &request) const
{
    nlohmann::json root;
    try {
        std::ifstream file(ToFsPath(request.sourcePath));
        if (!file.is_open())
            throw std::runtime_error("failed to open render effect document");
        file >> root;
    } catch (const std::exception &e) {
        throw std::runtime_error("RenderEffectImporter failed to parse '" + request.sourcePath + "': " + e.what());
    }

    const auto requireExactKeys = [](const nlohmann::json &value, std::initializer_list<const char *> expected,
                                     const std::string &location) {
        if (!value.is_object())
            throw std::runtime_error(location + " must be an object");
        std::unordered_set<std::string> keys;
        for (const char *key : expected)
            keys.emplace(key);
        std::vector<std::string> missing;
        std::vector<std::string> unknown;
        for (const auto &key : keys) {
            if (!value.contains(key))
                missing.push_back(key);
        }
        for (const auto &[key, ignored] : value.items()) {
            (void)ignored;
            if (keys.find(key) == keys.end())
                unknown.push_back(key);
        }
        if (!missing.empty() || !unknown.empty()) {
            std::sort(missing.begin(), missing.end());
            std::sort(unknown.begin(), unknown.end());
            const auto join = [](const std::vector<std::string> &items) {
                std::string result;
                for (const auto &item : items) {
                    if (!result.empty())
                        result += ", ";
                    result += "'" + item + "'";
                }
                return result;
            };
            throw std::runtime_error(location + " schema mismatch; missing=[" + join(missing) + "], unknown=[" +
                                     join(unknown) + "]");
        }
    };

    if (!root.is_object())
        throw std::runtime_error("render effect document root must be an object");
    if (!root.contains("$schema") || !root["$schema"].is_string())
        throw std::runtime_error("render effect $schema must be a string");

    std::unordered_set<std::string> dependencies;
    const auto readReference = [&](const nlohmann::json &reference, const std::string &location) {
        requireExactKeys(reference, {"guid", "path_hint"}, location);
        if (!reference["guid"].is_string() || !reference["path_hint"].is_string())
            throw std::runtime_error(location + " guid and path_hint must be strings");
        const std::string guid = reference["guid"].get<std::string>();
        const std::string pathHint = reference["path_hint"].get<std::string>();
        RejectPathOnlyReference(guid, pathHint, location);
        if (guid.empty())
            throw std::runtime_error(location + " must provide a non-empty asset GUID");
        dependencies.insert(guid);
    };

    const std::string schema = root["$schema"].get<std::string>();
    if (schema == "infernux.render_effect") {
        requireExactKeys(root, {"$schema", "feature_type", "parameters", "dependencies"}, "render effect");
        static const std::regex featureTypePattern("^[a-z][a-z0-9_]*(\\.[a-z][a-z0-9_]*)+$");
        if (!root["feature_type"].is_string() ||
            !std::regex_match(root["feature_type"].get_ref<const std::string &>(), featureTypePattern))
            throw std::runtime_error("render effect feature_type must be a lowercase namespaced identifier");
        if (!root["parameters"].is_object())
            throw std::runtime_error("render effect parameters must be an object");
        if (!root["dependencies"].is_array())
            throw std::runtime_error("render effect dependencies must be an array");
        for (size_t index = 0; index < root["dependencies"].size(); ++index)
            readReference(root["dependencies"][index], "dependencies[" + std::to_string(index) + "]");
    } else if (schema == "infernux.render_effect_group") {
        requireExactKeys(root, {"$schema", "entries"}, "render effect group");
        if (!root["entries"].is_array())
            throw std::runtime_error("render effect group entries must be an array");
        std::unordered_set<std::string> entryIds;
        for (size_t index = 0; index < root["entries"].size(); ++index) {
            const auto &entry = root["entries"][index];
            const std::string location = "entries[" + std::to_string(index) + "]";
            requireExactKeys(entry, {"entry_id", "asset", "enabled", "overrides"}, location);
            if (!entry["entry_id"].is_string() || entry["entry_id"].get_ref<const std::string &>().empty())
                throw std::runtime_error(location + ".entry_id must be a non-empty string");
            if (!entryIds.insert(entry["entry_id"].get<std::string>()).second)
                throw std::runtime_error("render effect group entry_id values must be unique");
            if (!entry["enabled"].is_boolean())
                throw std::runtime_error(location + ".enabled must be a boolean");
            if (!entry["overrides"].is_object())
                throw std::runtime_error(location + ".overrides must be an object");
            readReference(entry["asset"], location + ".asset");
        }
    } else {
        throw std::runtime_error("unsupported render effect schema '" + schema + "'");
    }

    std::vector<std::string> ordered(dependencies.begin(), dependencies.end());
    std::sort(ordered.begin(), ordered.end());
    return ordered;
}

ImportArtifact ParticleGraphImporter::Import(const ImportRequest &request) const
{
    ImportArtifact artifact(request.metadata);
    artifact.dependencies = ScanDependencies(request);
    artifact.dependenciesAuthoritative = true;
    return artifact;
}

std::vector<std::string> ParticleGraphImporter::ScanDependencies(const ImportRequest &request) const
{
    nlohmann::json root;
    try {
        std::ifstream file(ToFsPath(request.sourcePath));
        if (!file.is_open())
            throw std::runtime_error("failed to open particle graph document");
        file >> root;
    } catch (const std::exception &e) {
        throw std::runtime_error("ParticleGraphImporter failed to parse '" + request.sourcePath + "': " + e.what());
    }

    const auto requireExactKeys = [](const nlohmann::json &value, std::initializer_list<const char *> expected,
                                     const std::string &location) {
        if (!value.is_object())
            throw std::runtime_error(location + " must be an object");
        std::unordered_set<std::string> keys;
        for (const char *key : expected)
            keys.emplace(key);
        if (value.size() != keys.size())
            throw std::runtime_error(location + " contains missing or unknown fields");
        for (const auto &[key, ignored] : value.items()) {
            (void)ignored;
            if (keys.find(key) == keys.end())
                throw std::runtime_error(location + " contains unknown field '" + key + "'");
        }
    };

    requireExactKeys(root, {"$schema", "stable_id", "name", "emitters", "parameters", "event_types"}, "particle graph");
    if (!root["$schema"].is_string() || root["$schema"].get<std::string>() != "infernux.particle_graph")
        throw std::runtime_error("particle graph has an unsupported $schema");
    if (!root["stable_id"].is_string() || root["stable_id"].get_ref<const std::string &>().empty() ||
        !root["name"].is_string() || root["name"].get_ref<const std::string &>().empty())
        throw std::runtime_error("particle graph stable_id and name must be non-empty strings");
    if (!root["emitters"].is_array() || root["emitters"].empty() || !root["parameters"].is_array() ||
        !root["event_types"].is_array())
        throw std::runtime_error("particle graph requires emitters, parameters, and event_types arrays");

    std::unordered_set<std::string> dependencies;
    const auto readReference = [&](const nlohmann::json &reference, const std::string &location) {
        requireExactKeys(reference, {"guid", "path_hint"}, location);
        if (!reference["guid"].is_string() || !reference["path_hint"].is_string())
            throw std::runtime_error(location + " guid and path_hint must be strings");
        const std::string guid = reference["guid"].get<std::string>();
        const std::string pathHint = reference["path_hint"].get<std::string>();
        RejectPathOnlyReference(guid, pathHint, location);
        // Built-in texture tokens are renderer symbols, not AssetDatabase
        // identities, so they deliberately do not create dependency edges.
        if (!guid.empty() && !IsBuiltinTextureToken(guid))
            dependencies.insert(guid);
    };
    std::function<void(const nlohmann::json &, const std::string &)> scanReferences;
    scanReferences = [&](const nlohmann::json &value, const std::string &location) {
        if (value.is_object()) {
            if (value.size() == 2 && value.contains("guid") && value.contains("path_hint")) {
                readReference(value, location);
                return;
            }
            for (const auto &[key, child] : value.items())
                scanReferences(child, location + "." + key);
            return;
        }
        if (value.is_array()) {
            for (size_t index = 0; index < value.size(); ++index)
                scanReferences(value[index], location + "[" + std::to_string(index) + "]");
        }
    };
    scanReferences(root, "particle graph");
    for (size_t emitterIndex = 0; emitterIndex < root["emitters"].size(); ++emitterIndex) {
        const auto &emitter = root["emitters"][emitterIndex];
        const std::string emitterLocation = "emitters[" + std::to_string(emitterIndex) + "]";
        requireExactKeys(emitter,
                         {"stable_id", "name", "settings", "attribute_defaults", "data_interfaces", "stages", "events"},
                         emitterLocation);
        if (!emitter["attribute_defaults"].is_object() || !emitter["data_interfaces"].is_array() ||
            !emitter["events"].is_array())
            throw std::runtime_error(emitterLocation +
                                     ".attribute_defaults must be an object and data_interfaces/events must be arrays");
        for (size_t interfaceIndex = 0; interfaceIndex < emitter["data_interfaces"].size(); ++interfaceIndex) {
            const auto &dataInterface = emitter["data_interfaces"][interfaceIndex];
            const std::string interfaceLocation =
                emitterLocation + ".data_interfaces[" + std::to_string(interfaceIndex) + "]";
            if (!dataInterface.is_object() || !dataInterface.contains("kind") || !dataInterface["kind"].is_string())
                throw std::runtime_error(interfaceLocation + " must be a typed object");
            const std::string kind = dataInterface["kind"].get<std::string>();
            if (kind == "vector_field") {
                requireExactKeys(dataInterface,
                                 {"kind", "stable_id", "name", "texture", "space", "field_to_space", "vector_scale",
                                  "boundary", "filtering"},
                                 interfaceLocation);
                readReference(dataInterface["texture"], interfaceLocation + ".texture");
            } else if (kind == "sdf_volume") {
                requireExactKeys(
                    dataInterface,
                    {"kind", "stable_id", "name", "texture", "space", "field_to_space", "distance_scale", "filtering"},
                    interfaceLocation);
                readReference(dataInterface["texture"], interfaceLocation + ".texture");
            } else {
                throw std::runtime_error(interfaceLocation + " has unsupported kind '" + kind + "'");
            }
        }
        if (!emitter["stages"].is_object())
            throw std::runtime_error(emitterLocation + ".stages must be an object");
        requireExactKeys(emitter["stages"],
                         {"init", "update", "collision_enter", "collision_stay", "collision_exit", "rendering"},
                         emitterLocation + ".stages");

        for (const char *stageName :
             {"init", "update", "collision_enter", "collision_stay", "collision_exit", "rendering"}) {
            const auto &stage = emitter["stages"][stageName];
            const std::string stageLocation = emitterLocation + ".stages." + stageName;
            const bool optionalCollisionStage = std::string(stageName).rfind("collision_", 0) == 0;
            if (optionalCollisionStage && stage.is_null())
                continue;
            if (stage.is_null())
                throw std::runtime_error(stageLocation + " cannot be null");
            requireExactKeys(stage, {"$schema", "domain", "nodes", "links", "metadata"}, stageLocation);
            if (!stage["nodes"].is_array() || !stage["links"].is_array())
                throw std::runtime_error(stageLocation + " nodes and links must be arrays");
            for (const auto &node : stage["nodes"]) {
                if (!node.is_object() || !node.contains("properties") || !node["properties"].is_object())
                    throw std::runtime_error(stageLocation + " contains an invalid node");
                scanReferences(node["properties"], stageLocation + ".properties");
            }
        }
    }

    std::vector<std::string> ordered(dependencies.begin(), dependencies.end());
    std::sort(ordered.begin(), ordered.end());
    return ordered;
}

ImportArtifact DataAssetImporter::Import(const ImportRequest &request) const
{
    ImportArtifact artifact(request.metadata);
    artifact.dependencies = ScanDependencies(request);
    artifact.dependenciesAuthoritative = true;
    return artifact;
}

std::vector<std::string> DataAssetImporter::ScanDependencies(const ImportRequest &request) const
{
    nlohmann::json root;
    try {
        std::ifstream file(ToFsPath(request.sourcePath));
        if (!file.is_open())
            throw std::runtime_error("failed to open data asset document");
        file >> root;
    } catch (const std::exception &e) {
        throw std::runtime_error("DataAssetImporter failed to parse '" + request.sourcePath + "': " + e.what());
    }

    const bool legacyDocument = root.is_object() && root.size() == 3;
    const bool versionedDocument = root.is_object() && root.size() == 4 && root.contains("schema_version") &&
                                   root["schema_version"].is_number_unsigned() &&
                                   root["schema_version"].get<uint64_t>() > 0;
    if ((!legacyDocument && !versionedDocument) || root.value("$type", std::string{}) != "data_asset" ||
        !root.contains("type_id") || !root["type_id"].is_string() ||
        root["type_id"].get_ref<const std::string &>().empty() || !root.contains("fields") ||
        !root["fields"].is_object()) {
        throw std::runtime_error("DataAsset document must contain $type='data_asset', type_id, fields, and an optional "
                                 "positive schema_version");
    }

    std::unordered_set<std::string> dependencies;
    std::function<void(const nlohmann::json &, const std::string &)> scan;
    scan = [&](const nlohmann::json &value, const std::string &location) {
        if (value.is_array()) {
            for (size_t index = 0; index < value.size(); ++index)
                scan(value[index], location + "[" + std::to_string(index) + "]");
            return;
        }
        if (!value.is_object())
            return;
        if (value.value("$type", std::string{}) == "asset_ref") {
            if (value.size() != 4 || !value.contains("asset_type") || !value["asset_type"].is_string() ||
                !value.contains("guid") || !value["guid"].is_string() || !value.contains("path_hint") ||
                !value["path_hint"].is_string()) {
                throw std::runtime_error(location + " contains a malformed asset reference");
            }
            const std::string guid = value["guid"].get<std::string>();
            const std::string pathHint = value["path_hint"].get<std::string>();
            RejectPathOnlyReference(guid, pathHint, location);
            if (!guid.empty())
                dependencies.insert(guid);
            return;
        }
        for (const auto &[key, child] : value.items())
            scan(child, location + "." + key);
    };
    scan(root["fields"], "data asset fields");

    std::vector<std::string> ordered(dependencies.begin(), dependencies.end());
    std::sort(ordered.begin(), ordered.end());
    return ordered;
}

// ============================================================================
// ModelImporter — scan model file with Assimp and extract metadata into .meta
// ============================================================================

ImportArtifact ModelImporter::Import(const ImportRequest &request) const
{
    ImportArtifact artifact(request.metadata);
    EnsureDefaultSettings(artifact.metadata);
    std::string sourcePath = request.sourcePath;
    std::string extension = FromFsPath(ToFsPath(sourcePath).extension());
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::unique_ptr<BlenderSource> blender;
    if (extension == ".blend") {
        blender = std::make_unique<BlenderSource>(request);
        sourcePath = blender->Convert(request);
    }
    auto imported = MeshLoader::ImportSourceDetailed(sourcePath, request.guid, artifact.metadata);
    if (!imported.mesh)
        throw std::logic_error("ModelImporter detailed source import returned no runtime mesh");
    imported.mesh->SetFilePath(request.sourcePath);
    if (blender)
        imported.mesh->SetName(FromFsPath(ToFsPath(request.sourcePath).stem()));
    if (imported.skinnedMesh)
        imported.skinnedMesh->sourcePath = request.sourcePath;

    // Embedded images are owned sub-assets: the source sidecar owns identity,
    // while their runtime payloads use the same Texture artifacts as file assets.
    nlohmann::json previousTextures = nlohmann::json::array();
    if (artifact.metadata.HasKey("model_textures"))
        previousTextures = nlohmann::json::parse(artifact.metadata.GetDataAs<std::string>("model_textures"));
    nlohmann::json modelTextures = nlohmann::json::array();
    std::unordered_map<std::string, std::string> embeddedGuids;
    const auto publishEmbedded = [&](size_t index, const std::string &semantic) -> std::string {
        const auto &image = imported.embeddedImages.at(index);
        const std::string key = image.key + "/" + semantic;
        if (const auto found = embeddedGuids.find(key); found != embeddedGuids.end())
            return found->second;
        InxResourceMeta metadata;
        metadata.Init("", 0, request.sourcePath, ResourceType::Texture);
        metadata.AddMetadata("srgb", semantic == "color");
        metadata.AddMetadata("texture_type", semantic == "normal" ? std::string("normal_map") :
            semantic == "color" ? std::string("default") : semantic);
        for (const auto &previous : previousTextures)
            if (previous.at("key") == key) {
                metadata.DeserializeDocument(previous.at("metadata"));
                break;
            }
        const std::string guid = metadata.GetGuid();
        metadata.UpdateFilePath(request.sourcePath + "::subtex:" + guid);
        TextureImporter{}.EnsureDefaultSettings(metadata);
        metadata.AddMetadata("import_owner_guid", request.guid);
        metadata.AddMetadata("resource_name", image.name);
        const auto sourceHash = request.metadata.GetDataAs<std::string>("content_hash");
        metadata.AddMetadata("content_hash", sourceHash);
        const auto pixels = image.height
            ? TextureDecoder::DecodeRgba8(image.bytes, image.width, image.height, metadata)
            : TextureDecoder::DecodeMemory(image.bytes, metadata, image.name);
        metadata.AddMetadata("artifact_width", static_cast<int>(pixels->mipLevels.front().width));
        metadata.AddMetadata("artifact_height", static_cast<int>(pixels->mipLevels.front().height));
        metadata.AddMetadata("artifact_depth", 1);
        modelTextures.push_back({{"key", key}, {"guid", guid}, {"name", image.name},
                                 {"semantic", semantic}, {"metadata", metadata.SerializeDocument()}});
        artifact.runtimeCpuArtifacts.push_back({ImportArtifact::RuntimeArtifactKind::Primary,
            ResourceType::Texture, TextureArtifact::Serialize(*pixels, sourceHash), guid});
        embeddedGuids.emplace(key, guid);
        return guid;
    };
    auto materials = imported.mesh->GetMaterialSlotData();
    for (const auto &texture : imported.textureSources) {
        auto &guid = materials.at(texture.materialSlot).textureGuids.at(texture.channel);
        const bool linear = texture.channel != static_cast<uint32_t>(ModelTexture::BaseColor) &&
                            texture.channel != static_cast<uint32_t>(ModelTexture::Emission);
        if (texture.embeddedIndex >= 0) {
            guid = publishEmbedded(static_cast<size_t>(texture.embeddedIndex),
                texture.channel == static_cast<uint32_t>(ModelTexture::Normal) ? "normal" : linear ? "data" : "color");
            continue;
        }
        std::string path = texture.path;
        if (path.rfind("//", 0) == 0)
            path.erase(0, 2);
        std::replace(path.begin(), path.end(), '\\', '/');
        auto texturePath = ToFsPath(path);
        if (texturePath.is_relative())
            texturePath = ToFsPath(sourcePath).parent_path() / texturePath;
        if (!request.resolveTextureGuid)
            throw std::logic_error("model texture import requires an immutable asset catalog");
        const auto normalizedPath = NormalizeFilesystemPathLexically(FromFsPath(texturePath));
        guid = request.resolveTextureGuid(normalizedPath, linear);
        artifact.resolvedTextureSources.emplace_back(normalizedPath, guid);
    }
    // Keep previously published views while the image exists: other materials
    // can reference them independently of this model's current material remaps.
    for (size_t index = 0; index < imported.embeddedImages.size(); ++index)
        for (const auto &previous : previousTextures) {
            const auto semantic = previous.at("semantic").get<std::string>();
            if (previous.at("key") == imported.embeddedImages[index].key + "/" + semantic)
                (void)publishEmbedded(index, semantic);
        }
    // Images unused by the current material remaps remain available to authors.
    for (size_t index = 0; index < imported.embeddedImages.size(); ++index) {
        const auto &image = imported.embeddedImages[index];
        if (!embeddedGuids.count(image.key + "/color") && !embeddedGuids.count(image.key + "/data") &&
            !embeddedGuids.count(image.key + "/normal"))
            (void)publishEmbedded(index, "color");
    }
    artifact.metadata.AddMetadata("model_textures", modelTextures.dump());
    imported.mesh->SetMaterialSlotData(std::move(materials));

    const auto checkedMetadataInt = [](uint64_t value, std::string_view field) {
        if (value > static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("ModelImporter metadata count exceeds int range: " + std::string(field));
        return static_cast<int>(value);
    };
    const auto joinCsv = [](const std::vector<std::string> &values) {
        std::string joined;
        for (size_t index = 0; index < values.size(); ++index) {
            if (index > 0)
                joined += ',';
            joined += values[index];
        }
        return joined;
    };

    // ── Write metadata to .meta ─────────────────────────────────────────

    artifact.metadata.AddMetadata("mesh_count", checkedMetadataInt(imported.meshCount, "mesh_count"));
    artifact.metadata.AddMetadata("vertex_count", checkedMetadataInt(imported.vertexCount, "vertex_count"));
    artifact.metadata.AddMetadata("index_count", checkedMetadataInt(imported.indexCount, "index_count"));
    artifact.metadata.AddMetadata("material_slot_count",
                                  checkedMetadataInt(imported.materialSlots.size(), "material_slot_count"));

    // Store material slot names as a comma-separated string for .meta
    // (InxResourceMeta uses std::any; a string is the simplest portable choice)
    artifact.metadata.AddMetadata("material_slots", joinCsv(imported.materialSlots));

    // Model mesh children are addressable sub-resources.  Their identity is
    // persisted by canonical source path rather than by the node array index,
    // so reordering sibling nodes does not invalidate scene references.  Keep
    // the existing id on reimport; newly discovered paths receive an ordinary
    // resource GUID once and are then persisted in this metadata document.
    nlohmann::json previousModelMeshes = nlohmann::json::array();
    if (artifact.metadata.HasKey("model_meshes"))
        previousModelMeshes = nlohmann::json::parse(artifact.metadata.GetDataAs<std::string>("model_meshes"));
    nlohmann::json modelMeshes = nlohmann::json::array();
    const auto &modelNodes = imported.mesh->GetModelNodes();
    for (size_t i = 0; i < modelNodes.size(); ++i) {
        if (modelNodes[i].nodeGroup < 0)
            continue;
        const auto path = imported.mesh->GetModelNodePath(i);
        const std::string identityKey = ModelNodeIdentityKey(*imported.mesh, i, path, true);
        const std::string geometryKey = ModelNodeIdentityKey(*imported.mesh, i, path, false);
        modelMeshes.push_back({{"name", modelNodes[i].name}, {"path", path},
                               {"subresource_id", ""}, {"identity_key", identityKey},
                               {"geometry_key", geometryKey}});
    }
    std::unordered_set<std::string> consumedSubresourceIds;
    // Reserve every exact path before considering renames. Otherwise a new
    // duplicate visited first can steal an ID still owned by an existing node.
    // Each subsequent pass matches only one-to-one unmatched candidates on
    // BOTH sides, rather than letting iteration order choose the winner.
    const auto matchUnique = [&](const char *key) {
        std::unordered_map<std::string, std::vector<size_t>> oldByKey, newByKey;
        for (size_t i = 0; i < previousModelMeshes.size(); ++i) {
            const auto &entry = previousModelMeshes[i];
            const auto id = entry.value("subresource_id", std::string{});
            if (!id.empty() && consumedSubresourceIds.find(id) == consumedSubresourceIds.end() && entry.contains(key))
                oldByKey[entry[key].dump()].push_back(i);
        }
        for (size_t i = 0; i < modelMeshes.size(); ++i) {
            const auto &entry = modelMeshes[i];
            if (entry["subresource_id"] == "")
                newByKey[entry[key].dump()].push_back(i);
        }
        for (const auto &[value, newIndices] : newByKey) {
            const auto found = oldByKey.find(value);
            if (newIndices.size() != 1 || found == oldByKey.end() || found->second.size() != 1)
                continue;
            const auto id = previousModelMeshes[found->second.front()]["subresource_id"].get<std::string>();
            if (consumedSubresourceIds.insert(id).second)
                modelMeshes[newIndices.front()]["subresource_id"] = id;
        }
    };
    matchUnique("path");
    matchUnique("identity_key");
    matchUnique("geometry_key");
    for (auto &entry : modelMeshes) {
        if (entry["subresource_id"] == "") {
            InxResourceMeta subresource;
            subresource.Init(nullptr, 0, request.sourcePath + "::submesh:" + entry["path"].dump(),
                             ResourceType::Mesh);
            entry["subresource_id"] = subresource.GetGuid();
        }
    }
    artifact.metadata.AddMetadata("model_meshes", modelMeshes.dump());

    artifact.metadata.AddMetadata("bone_count", checkedMetadataInt(imported.boneNames.size(), "bone_count"));
    artifact.metadata.AddMetadata("bone_names_csv", joinCsv(imported.boneNames));

    artifact.metadata.AddMetadata("animation_count",
                                  checkedMetadataInt(imported.animationNames.size(), "animation_count"));
    artifact.metadata.AddMetadata("animation_names_csv", joinCsv(imported.animationNames));
    artifact.metadata.AddMetadata("source_animations", imported.sourceAnimations.dump());
    // Local clip IDs describe source takes / authored slices. Asset GUIDs belong
    // to this model and survive temporarily disabling an output clip.
    auto animationIdentities = artifact.metadata.HasKey("model_animation_identities")
        ? nlohmann::json::parse(artifact.metadata.GetDataAs<std::string>("model_animation_identities"))
        : nlohmann::json::object();
    auto modelAnimations = nlohmann::json::array();
    if (imported.skinnedMesh) {
        for (const auto &animation : imported.skinnedMesh->animations) {
            InxResourceMeta metadata;
            metadata.Init("", 0, request.sourcePath, ResourceType::DefaultText);
            if (animationIdentities.contains(animation.id))
                metadata.AddMetadata("guid", animationIdentities.at(animation.id).get<std::string>());
            const auto guid = metadata.GetGuid();
            animationIdentities[animation.id] = guid;
            const double duration = animation.durationTicks / animation.ticksPerSecond;
            const nlohmann::json document = {
                {"name", animation.name}, {"source_model_guid", request.guid}, {"source_model_path", ""},
                {"take_name", animation.id}, {"bind_pose_bone_names", imported.boneNames},
                {"duration_hint", duration}, {"events", nlohmann::json::array()}};
            metadata.UpdateFilePath(request.sourcePath + "::subanim:" + animation.id);
            metadata.AddMetadata("import_owner_guid", request.guid);
            metadata.AddMetadata("resource_name", animation.name);
            metadata.AddMetadata("file_extension", std::string(".animclip3d"));
            metadata.AddMetadata("content_hash", request.metadata.GetDataAs<std::string>("content_hash"));
            metadata.AddMetadata("import_document", document.dump());
            modelAnimations.push_back({{"id", animation.id}, {"guid", guid}, {"name", animation.name},
                                       {"duration", duration}, {"metadata", metadata.SerializeDocument()}});
        }
    }
    artifact.metadata.AddMetadata("model_animation_identities", animationIdentities.dump());
    artifact.metadata.AddMetadata("model_animations", modelAnimations.dump());

    if (MeshImportSettings::Read(artifact.metadata).materialImportMode != "none") {
        const auto externalTextures = MeshLoader::ScanExternalTexturePaths(sourcePath);
        artifact.dependencyPathHints.assign(externalTextures.begin(), externalTextures.end());
    }
    artifact.dependenciesAuthoritative = true;
    std::set<std::string> materialDependencies;
    for (const auto &material : imported.mesh->GetMaterialSlotData()) {
        if (!material.materialGuid.empty())
            materialDependencies.insert(material.materialGuid);
        for (const auto &textureGuid : material.textureGuids)
            if (!textureGuid.empty())
                materialDependencies.insert(textureGuid);
    }
    artifact.dependencies.assign(materialDependencies.begin(), materialDependencies.end());

    if (!artifact.metadata.HasKey("content_hash"))
        throw std::logic_error("ModelImporter metadata has no source content hash");
    artifact.runtimeCpuArtifacts.push_back(ImportArtifact::RuntimeCpuArtifact{
        ImportArtifact::RuntimeArtifactKind::Primary, ResourceType::Mesh,
        MeshArtifact::Serialize(*imported.mesh, artifact.metadata.GetDataAs<std::string>("content_hash"))});
    const std::string sourceHash = artifact.metadata.GetDataAs<std::string>("content_hash");
    artifact.runtimeCpuArtifacts.push_back(ImportArtifact::RuntimeCpuArtifact{
        ImportArtifact::RuntimeArtifactKind::SkinnedMesh, ResourceType::Mesh,
        imported.skinnedMesh ? SkinnedMeshArtifact::Serialize(*imported.skinnedMesh, sourceHash)
                             : SkinnedMeshArtifact::SerializeEmpty(sourceHash)});

    // INXLOG_INFO("ModelImporter: imported '", FromFsPath(ToFsPath(request.sourcePath).filename()), "' — ",
    //             imported.meshCount, " mesh(es), ", imported.vertexCount, " verts, ", imported.indexCount, " indices,
    //             ", imported.materialSlots.size(), " material slot(s), ", imported.boneNames.size(), " bone(s), ",
    //             imported.animationNames.size(), " anim(s)");

    return artifact;
}

} // namespace infernux
