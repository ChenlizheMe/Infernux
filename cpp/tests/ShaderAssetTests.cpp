#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>
#include <platform/filesystem/InxPath.h>

#include <cassert>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{

std::vector<char> Spirv(size_t words, char value)
{
    return std::vector<char>(words * sizeof(uint32_t), value);
}

void VerifyPreparedStagePublication()
{
    using namespace infernux;
    const auto root = std::filesystem::temp_directory_path() /
        ("infernux-prepared-shader-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto path = root / "Assets" / "Publication.vert";
    std::filesystem::create_directories(path.parent_path());
    {
        std::ofstream output(path);
        output << "#version 450\nShaderInfo { Name \"Prepared Publication\" Capabilities [Fullscreen] }\n"
                  "void main(){ gl_Position = vec4(0.0); }\n";
    }
    JobSystem::Initialize(2);
    auto database = std::make_unique<AssetDatabase>();
    database->Initialize(FromFsPath(root));
    auto &registry = AssetRegistry::Instance();
    registry.Initialize(std::move(database));
    registry.RegisterLoader(ResourceType::Shader, std::make_unique<ShaderLoader>());
    registry.PopulateAssetDatabaseLoaders();
    const auto imported = registry.GetAssetDatabase()->ImportAsset(FromFsPath(path));
    assert(imported.succeeded);
    const auto prepare = [&] {
        return registry.GetLoader(ResourceType::Shader)
            ->Load(FromFsPath(path), imported.guid, registry.GetAssetDatabase()).Get<ShaderAsset>();
    };
    auto candidate = prepare();
    assert(candidate && !registry.IsLoaded(imported.guid));
    assert(registry.GetAssetVersion(imported.guid) == 0);
    registry.PublishShader(imported.guid, candidate);
    auto resident = registry.GetAsset<ShaderAsset>(imported.guid);
    assert(resident == candidate);
    const auto version = registry.GetAssetVersion(imported.guid);
    auto replacement = prepare();
    assert(replacement != resident);
    registry.PublishShader(imported.guid, replacement);
    assert(registry.GetAsset<ShaderAsset>(imported.guid) == resident);
    assert(registry.GetAssetVersion(imported.guid) == version + 1);
    auto invalid = prepare();
    invalid->variants.clear();
    bool rejected = false;
    try {
        registry.PublishShader(imported.guid, invalid);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected && registry.GetAssetVersion(imported.guid) == version + 1);
    assert(registry.GetAsset<ShaderAsset>(imported.guid) == resident);
    assert(resident->HasVariant(ShaderCompileTarget::Forward));
    registry.Shutdown();
    JobSystem::Shutdown();
    std::filesystem::remove_all(root);
}

} // namespace

int main()
{
    infernux::ShaderAsset asset;
    asset.shaderId = "Tests/VariantAsset";
    asset.shaderType = "fragment";

    assert(!asset.HasVariant(infernux::ShaderCompileTarget::Forward));
    assert(asset.SetVariant(infernux::ShaderCompileTarget::Forward, Spirv(1, 1)));
    assert(asset.HasVariant(infernux::ShaderCompileTarget::Forward));
    assert(asset.variants.size() == 1);

    assert(asset.SetVariant(infernux::ShaderCompileTarget::Forward, Spirv(2, 2)));
    assert(asset.variants.size() == 1);
    assert(asset.FindVariant(infernux::ShaderCompileTarget::Forward)->spirv.size() == 2 * sizeof(uint32_t));

    assert(asset.SetVariant(infernux::ShaderCompileTarget::Shadow, Spirv(1, 3)));
    assert(asset.SetVariant(infernux::ShaderCompileTarget::GBuffer, Spirv(1, 4)));
    assert(asset.variants.size() == 3);
    assert(asset.HasVariant(infernux::ShaderCompileTarget::Shadow));
    assert(asset.HasVariant(infernux::ShaderCompileTarget::GBuffer));

    assert(!asset.SetVariant(infernux::ShaderCompileTarget::Depth, {}));
    assert(!asset.SetVariant(infernux::ShaderCompileTarget::Count, Spirv(1, 5)));
    assert(asset.variants.size() == 3);
    assert(asset.GetRuntimeMemoryBytes() >= sizeof(asset) + 4 * sizeof(uint32_t));

    infernux::InxShaderLoader compiler(true, false, false, false, false, true, false, false, false, false);
    const std::string source = "#version 450\nvoid main() { gl_Position = vec4(0.0); }\n";
    infernux::InxResourceMeta metadata;
    compiler.CreateMeta(source.data(), source.size(), "Tests/CacheReset.vert", metadata);
    const std::string cachePath = metadata.GetDataAs<std::string>("file_path");
    const auto compiled = compiler.CompileVertexGlsl(source, cachePath);
    assert(!compiled.empty());
    assert(infernux::InxShaderLoader::TakeCompiledVariants(cachePath).empty());
    assert(infernux::InxShaderLoader::TakeCompiledVariants(cachePath).empty());

    VerifyPreparedStagePublication();
    std::cout << "Shader asset tests passed\n";
    return 0;
}
