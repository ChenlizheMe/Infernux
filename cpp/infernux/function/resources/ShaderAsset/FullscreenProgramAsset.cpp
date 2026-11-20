#include "FullscreenProgramAsset.h"

#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>

#include <algorithm>

namespace infernux
{
std::shared_ptr<const FullscreenProgram> LoadFullscreenProgram(const std::string &name, ShaderCodeTarget targetKind)
{
    auto &registry = AssetRegistry::Instance();
    auto *database = registry.GetAssetDatabase();
    auto path = database->GetPathFromGuid(name);
    std::string guid = path.empty() ? std::string{} : name;
    if (path.empty()) {
        if (asset_formats::FindShaderSourceKind(name)) {
            path = name;
            guid = database->GetGuidFromPath(path);
        } else {
            guid = database->FindShaderGuidById(name, "fragment");
            path = database->GetPathFromGuid(guid);
        }
    }
    if (!asset_formats::FindShaderSourceKind(path))
        return {};
    const auto asset = registry.LoadAsset<ShaderAsset>(guid, ResourceType::Shader);
    if (!asset || !asset->program)
        throw ShaderProgramSelectionError("Fullscreen author program is unavailable: " + path);
    const auto target = std::find_if(asset->program->targets.begin(), asset->program->targets.end(),
                                     [targetKind](const auto &code) { return code.target == targetKind; });
    if (target == asset->program->targets.end())
        throw ShaderProgramSelectionError("Fullscreen shader asset has no requested target: " + path);
    auto result = std::make_shared<FullscreenProgram>();
    result->assetGuid = std::move(guid);
    result->shaderId = asset->shaderId;
    result->code = std::shared_ptr<const ShaderTargetCode>(asset->program, &*target);
    result->schema = asset->authoringSchema;
    return result;
}
} // namespace infernux
