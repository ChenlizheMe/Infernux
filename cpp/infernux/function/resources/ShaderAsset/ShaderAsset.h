#pragma once

#include <core/types/ShaderProgramArtifact.h>
#include <core/types/ShaderTypes.h>
#include <function/renderer/shader/ShaderProgramCode.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernux
{
inline constexpr char kStandardVertexShaderId[] = "infernux/standard";

struct ShaderAsset;

struct ShaderMaterialPreparation
{
    std::shared_ptr<const ShaderCompilation> vertex;
    std::shared_ptr<const ShaderProgramArtifact> program;
    std::string error;
    std::vector<ShaderDiagnostic> diagnostics;
};

struct ShaderMaterialPrograms
{
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<const ShaderMaterialPreparation>> entries;
};

/// Frozen typed source publication, shared by graphics and compute consumers.
struct ShaderAsset
{
    /// Shader identity (e.g. "pbr", "unlit", "surface_water")
    std::string shaderId;

    /// Author source kind.
    std::string shaderType;

    /// Source file path (for hot-reload cache key)
    std::string filePath;

    /// Render-state annotations (fragment shaders only)
    ShaderRenderMeta renderMeta;

    /// Statically authored programs retain both targets and their logical
    /// fields. Device publication chooses the target without recompiling.
    std::shared_ptr<const ShaderCompilation> program;
    std::string authoringSchema;
    std::string authorSourceText;
    std::optional<ShaderProgramDomain> programDomain;
    std::shared_ptr<const ShaderProgramArtifact> materialProgram;
    std::vector<std::string> sourceDependencies;
    // Author selection belongs to a material. Each fragment snapshot caches
    // its requested vertex combinations, including one rejected source pair.
    std::shared_ptr<ShaderMaterialPrograms> materialPrograms = std::make_shared<ShaderMaterialPrograms>();

    [[nodiscard]] size_t GetRuntimeMemoryBytes() const noexcept
    {
        size_t bytes = sizeof(*this) + shaderId.capacity() + shaderType.capacity() + filePath.capacity() +
                       renderMeta.cullMode.capacity() + renderMeta.depthWrite.capacity() +
                       renderMeta.depthTest.capacity() + renderMeta.blend.capacity() + renderMeta.passTag.capacity() +
                       renderMeta.stencil.capacity() + renderMeta.alphaClip.capacity();
        bytes += authoringSchema.capacity() + authorSourceText.capacity();
        bytes += sourceDependencies.capacity() * sizeof(std::string);
        for (const auto &dependency : sourceDependencies)
            bytes += dependency.capacity();
        if (program) {
            bytes += program->diagnostics.capacity() * sizeof(ShaderDiagnostic) +
                     program->targets.capacity() * sizeof(ShaderTargetCode);
            for (const auto &diagnostic : program->diagnostics)
                bytes += diagnostic.GetRuntimeMemoryBytes() - sizeof(diagnostic);
            for (const auto &target : program->targets) {
                bytes += target.reflection.capacity() + target.entries.capacity() * sizeof(ShaderEntryPointCode);
                bytes += target.resources.capacity() * sizeof(ShaderResourceLayout);
                for (const auto &resource : target.resources)
                    bytes += resource.GetRuntimeMemoryBytes() - sizeof(resource);
                for (const auto &entry : target.entries) {
                    bytes += entry.name.capacity() + entry.code.capacity() +
                             entry.bindings.capacity() * sizeof(entry.bindings[0]);
                    for (const auto *variables : {&entry.inputs, &entry.outputs}) {
                        bytes += variables->capacity() * sizeof(ShaderStageVariable);
                        for (const auto &variable : *variables)
                            bytes += variable.name.capacity();
                    }
                }
            }
        }
        return bytes;
    }
};

} // namespace infernux
