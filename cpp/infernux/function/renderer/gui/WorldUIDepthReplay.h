#pragma once

#include "../RenderGraphDescription.h"
#include <algorithm>
#include <optional>
#include <string_view>

namespace infernux
{

// A compiled replay owns its command. A newer Python graph may replace its
// source description while the old graph is still waiting to be retired.
struct WorldUIDepthReplay
{
    std::string passName;
    GraphCommandDesc draw;
};

// Replay the one cleared Forward depth producer preceding this WorldUI pass. Graph and
// pass names are diagnostic labels, not capabilities of the depth producer.
[[nodiscard]] inline std::optional<WorldUIDepthReplay> BuildWorldUIDepthReplay(const RenderGraphDescription &graph,
                                                                               bool preservesCameraDepth,
                                                                               std::string_view worldUIPassName)
{
    if (preservesCameraDepth)
        return std::nullopt;
    const auto depth = std::find_if(graph.textures.begin(), graph.textures.end(),
                                    [](const auto &texture) { return texture.IsViewDepth(); });
    if (depth == graph.textures.end())
        return std::nullopt;

    std::optional<WorldUIDepthReplay> source;
    for (const auto &pass : graph.passes) {
        const auto *command = pass.commands.empty() ? nullptr : &pass.commands.front();
        const bool writesViewDepth = pass.WritesTexture(depth->name);
        if (command && command->type == GraphCommandType::DrawWorldUI) {
            if (pass.name == worldUIPassName)
                return pass.commands.size() == 1 && writesViewDepth && !pass.clearDepth ? source : std::nullopt;
            // Ordinary UI only reads scene depth. Clearing it changes the
            // depth contract seen by a later stage and cannot be replayed.
            if (writesViewDepth && pass.clearDepth)
                return std::nullopt;
            continue;
        }
        if (!writesViewDepth)
            continue;
        if (source || !command || pass.type != GraphPassType::Raster || pass.commands.size() != 1 ||
            command->type != GraphCommandType::DrawRenderers || command->shaderTarget != ShaderCompileTarget::Forward ||
            !pass.clearDepth || pass.clearDepthValue != 0.0f || command->rendererSelection ||
            !command->overrideMaterial.empty() || command->materialFilter != GraphMaterialFilter::All)
            return std::nullopt;
        source = WorldUIDepthReplay{pass.name, *command};
    }
    return std::nullopt;
}

} // namespace infernux
