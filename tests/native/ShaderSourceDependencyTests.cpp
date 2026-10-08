#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace
{
void Write(const std::filesystem::path &path, const std::string &source)
{
    std::ofstream stream(path);
    stream << source;
    assert(stream.good());
}

bool HasRoot(const std::filesystem::path &dependency, const std::filesystem::path &root,
             const std::string &declaration = "")
{
    const auto roots = infernux::InxShaderLoader::GetDependentStageSources(infernux::FromFsPath(dependency), declaration);
    return std::find(roots.begin(), roots.end(), infernux::ResolveFilesystemPath(infernux::FromFsPath(root))) != roots.end();
}
}

int main()
{
    using infernux::FromFsPath;
    using infernux::InxShaderLoader;
    const auto directory = std::filesystem::temp_directory_path() /
        ("infernux-source-dependencies-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    const auto leaf = directory / "leaf.glsl";
    const auto outer = directory / "outer.glsl";
    const auto missing = directory / "arriving.glsl";
    const auto root = directory / "root.frag";
    const auto unrelated = directory / "other.glsl";
    const std::string rootSource = R"(#version 450
ShaderInfo { Name "Dependency Root" Capabilities [Fullscreen] Imports ["Outer"] }
void main() {}
)";
    Write(root, rootSource);
    Write(leaf, "ShaderInfo { Name \"Leaf\" }\nfloat leaf() { return 0.25; }\n");
    Write(outer, "ShaderInfo { Name \"Outer\" Imports [\"Leaf\"] }\nfloat outer() { return leaf(); }\n");
    InxShaderLoader::SetProjectShaderSearchPaths({FromFsPath(directory)});
    InxShaderLoader compiler(true, false, false, false, false, true, false, false, false, false);
    assert(!InxShaderLoader::AreSourceDiagnosticsCaptured());
    {
        const InxShaderLoader::SourceDiagnosticScope outerOwner;
        assert(InxShaderLoader::AreSourceDiagnosticsCaptured());
        {
            const InxShaderLoader::SourceDiagnosticScope innerOwner;
            assert(InxShaderLoader::AreSourceDiagnosticsCaptured());
        }
        assert(InxShaderLoader::AreSourceDiagnosticsCaptured());
    }
    assert(!InxShaderLoader::AreSourceDiagnosticsCaptured());
    auto compile = [&] {
        infernux::InxResourceMeta metadata;
        compiler.CreateMeta(rootSource.data(), rootSource.size(), FromFsPath(root), metadata);
        return compiler.Compile(rootSource.data(), rootSource.size(), metadata);
    };
    assert(compile());
    assert(HasRoot(leaf, root));
    assert(HasRoot(outer, root));
    assert(!HasRoot(unrelated, root));

    // Failed candidates subscribe to missing imports without dropping the
    // accepted program's dependencies. Creating the declaration can recover.
    Write(outer, "ShaderInfo { Name \"Outer\" Imports [\"Arriving\"] }\nfloat outer() { return arriving(); }\n");
    InxShaderLoader::InvalidateDirectoryCache();
    assert(!compile());
    assert(HasRoot(leaf, root));
    assert(HasRoot(missing, root, "Arriving"));
    Write(missing, "ShaderInfo { Name \"Arriving\" }\nfloat arriving() { return 0.5; }\n");
    InxShaderLoader::InvalidateDirectoryCache();
    assert(compile());
    assert(HasRoot(missing, root));
    assert(!HasRoot(leaf, root));

    // A deleted source remains discoverable by path, and a moved/restored
    // declaration at a new path remains discoverable by its exact Name.
    std::filesystem::remove(missing);
    assert(HasRoot(missing, root));
    InxShaderLoader::InvalidateDirectoryCache();
    assert(!compile());
    assert(HasRoot(directory / "moved.glsl", root, "Arriving"));
    Write(directory / "moved.glsl", "ShaderInfo { Name \"Arriving\" }\nfloat arriving() { return 0.75; }\n");
    InxShaderLoader::InvalidateDirectoryCache();
    assert(compile());
    assert(!HasRoot(missing, root));
    assert(HasRoot(directory / "moved.glsl", root));

    // One successful sibling compile cannot drop the running program's
    // dependency closure while another candidate in the batch is rejected.
    Write(outer, "ShaderInfo { Name \"Outer\" Imports [\"Leaf\"] }\nfloat outer() { return leaf(); }\n");
    InxShaderLoader::InvalidateDirectoryCache();
    {
        InxShaderLoader::SourceDependencyPublication rejectedBatch;
        assert(compile());
        assert(HasRoot(directory / "moved.glsl", root));
        assert(HasRoot(leaf, root));
    }
    assert(HasRoot(directory / "moved.glsl", root));
    assert(HasRoot(leaf, root));
    Write(outer, "ShaderInfo { Name \"Outer\" Imports [\"Arriving\"] }\nfloat outer() { return arriving(); }\n");
    InxShaderLoader::InvalidateDirectoryCache();
    {
        InxShaderLoader::SourceDependencyPublication acceptedBatch;
        assert(compile());
        acceptedBatch.Commit();
    }
    assert(!HasRoot(leaf, root));
    assert(HasRoot(directory / "moved.glsl", root));

    // Worker compilation may prepare a different closure, but only the owner
    // accepting that program can change the live dependency subscriptions.
    using Publication = InxShaderLoader::SourceDependencyPublication;
    std::shared_ptr<const Publication::Prepared> prepared;
    Write(outer, "ShaderInfo { Name \"Outer\" Imports [\"Leaf\"] }\nfloat outer() { return leaf(); }\n");
    InxShaderLoader::InvalidateDirectoryCache();
    std::thread worker([&] {
        Publication batch;
        assert(compile());
        prepared = batch.Detach();
    });
    worker.join();
    assert(!HasRoot(leaf, root));
    assert(HasRoot(directory / "moved.glsl", root));
    // Dropping a superseded result does not install candidate subscriptions.
    prepared.reset();
    assert(!HasRoot(leaf, root));
    {
        Publication batch;
        assert(compile());
        prepared = batch.Detach();
    }
    assert(!HasRoot(leaf, root));
    Publication::Publish(prepared);
    assert(HasRoot(leaf, root));
    assert(!HasRoot(directory / "moved.glsl", root));
    // Rejected worker candidates are isolated too, until explicitly accepted.
    Write(outer, "ShaderInfo { Name \"Outer\" Imports [\"NotHere\"] }\n");
    InxShaderLoader::InvalidateDirectoryCache();
    {
        Publication batch;
        assert(!compile());
        prepared = batch.Detach();
    }
    assert(!HasRoot(directory / "not-here.glsl", root, "NotHere"));
    Publication::Publish(prepared);
    assert(HasRoot(directory / "not-here.glsl", root, "NotHere"));
    assert(HasRoot(leaf, root));
    Write(outer, "ShaderInfo { Name \"Outer\" Imports [\"Arriving\"] }\nfloat outer() { return arriving(); }\n");
    InxShaderLoader::InvalidateDirectoryCache();
    assert(compile());
    assert(!HasRoot(directory / "not-here.glsl", root, "NotHere"));

    // Invalid declaration syntax must still leave its dependency subscription.
    Write(directory / "moved.glsl", "ShaderInfo { Name \"Arriving\" Imports [123] }\nfloat arriving() { return 1.0; }\n");
    InxShaderLoader::InvalidateDirectoryCache();
    assert(!compile());
    assert(HasRoot(directory / "moved.glsl", root));

    // Selected models include their library closure. Deferred additionally
    // subscribes to model catalog membership, including future/unsupported IDs.
    InxShaderLoader::AddShaderSearchPath(INFERNUX_TEST_SHADER_ROOT);
    const auto model = directory / "dependent.shadingmodel";
    const auto modelRoot = directory / "model.frag";
    const auto deferredRoot = directory / "deferred.frag";
    Write(model, R"(ShadingModelInfo { Name "Dependency Model" Imports ["Leaf"] }
void shading(in SurfaceData s, out vec4 color) { color = vec4(leaf()); }
)" );
    const std::string modelSource = R"(#version 450
ShaderInfo { Name "Model Root" ShadingModel "Dependency Model" }
void surface(out SurfaceData s) { s.albedo = vec3(0.5); }
)";
    const std::string deferredSource = R"(#version 450
ShaderInfo { Name "Deferred Root" Capabilities [Fullscreen, DeferredLighting] }
void main() {}
)";
    Write(modelRoot, modelSource);
    Write(deferredRoot, deferredSource);
    InxShaderLoader::InvalidateDirectoryCache();
    const auto modelPrepared = compiler.PrepareAuthoredStageGlsl(modelSource, FromFsPath(modelRoot),
                                                                infernux::ShaderCompileTarget::Forward);
    assert(!modelPrepared.empty());
    const auto deferred = compiler.PrepareAuthoredStageGlsl(deferredSource, FromFsPath(deferredRoot),
                                                            infernux::ShaderCompileTarget::Forward);
    assert(deferred.find("Dependency Model") != std::string::npos);
    assert(HasRoot(model, modelRoot));
    assert(HasRoot(leaf, modelRoot));
    assert(HasRoot(model, deferredRoot));
    assert(HasRoot(leaf, deferredRoot));
    assert(HasRoot(directory / "future.shadingmodel", deferredRoot, "shadingmodel/Future"));
    assert(!HasRoot(directory / "future.shadingmodel", modelRoot, "shadingmodel/Future"));

    // Engine/project changes retire subscriptions, while a repeated refresh
    // of the same project must keep them.
    InxShaderLoader::SetProjectShaderSearchPaths({FromFsPath(directory)});
    assert(HasRoot(directory / "moved.glsl", root));
    InxShaderLoader::SetProjectShaderSearchPaths({FromFsPath(directory / "next")});
    assert(!HasRoot(directory / "moved.glsl", root));
    InxShaderLoader::SetProjectShaderSearchPaths({});
    std::filesystem::remove_all(directory);
}
