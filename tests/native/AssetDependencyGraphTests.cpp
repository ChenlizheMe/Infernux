#include <function/resources/AssetDependencyGraph.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main()
{
    try {
        using namespace infernux;
        auto &graph = AssetDependencyGraph::Instance();
        graph.Clear();
        // Content revisions fan out once on publication. Stable queries do
        // not walk dependencies; unrelated assets and runtime objects stay out.
        graph.SetAssetDependencies("mesh", {"material"});
        graph.SetAssetDependencies("prefab", {"mesh"});
        graph.SetAssetDependencies("material", {"texture"});
        graph.AddRuntimeDependency("temporary-component", "texture");
        const auto baseline = graph.GetContentRevision("mesh");
        graph.PublishContentChange("texture");
        const auto changed = graph.GetContentRevision("mesh");
        Require(changed > baseline, "transitive texture edit did not reach mesh");
        Require(graph.GetContentRevision("prefab") == changed, "nested prefab missed texture edit");
        Require(graph.GetContentRevision("unrelated") == 1, "unrelated content was invalidated");
        Require(graph.GetContentRevision("temporary-component") == 1, "content revisions retained a runtime object");
        for (int i = 0; i < 1000; ++i)
            Require(graph.GetContentRevision("mesh") == changed, "reading revision caused a new publication");
        graph.AddAssetDependency("texture", "prefab");
        graph.PublishContentChange("texture");
        Require(graph.GetContentRevision("mesh") > changed, "cyclic dependencies stopped content publication");
        const auto cyclic = graph.GetContentRevision("mesh");
        graph.SetAssetDependencies("mesh", {"replacement-material"});
        graph.PublishContentChange("texture");
        Require(graph.GetContentRevision("mesh") == cyclic, "old dependency survived material remapping");
        graph.PublishContentChange("replacement-material");
        Require(graph.GetContentRevision("mesh") > cyclic, "new material dependency was not observed");
        graph.Clear();
        Require(graph.GetContentRevision("mesh") == 1, "new engine retained prior content revisions");

        constexpr size_t userCount = 10'000;
        std::unordered_map<std::string, std::vector<std::string>> dependencies;
        dependencies.reserve(userCount);
        for (size_t index = 0; index < userCount; ++index) {
            const std::string suffix = std::to_string(index);
            dependencies.emplace("asset-" + suffix,
                                 std::vector<std::string>{"dependency-a-" + suffix, "dependency-b-" + suffix});
        }

        const auto initial = graph.GetAssetSnapshot();
        const auto started = std::chrono::steady_clock::now();
        const auto built = AssetDependencyGraph::BuildAssetSnapshot(dependencies, 1);
        const double buildMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        Require(buildMilliseconds < 2'000.0, "10k dependency snapshot build exceeded 2 seconds");
        Require(built->GetEdgeCount() == userCount * 2, "dependency snapshot edge count mismatch");
        Require(built->GetNodeCount() == userCount * 3, "dependency snapshot node count mismatch");

        graph.AddRuntimeDependency("runtime-object", "dependency-a-0");
        graph.InstallAssetSnapshot(built);
        std::vector<std::string> notified;
        graph.RegisterCallback(ResourceType::Mesh,
                               [&](const std::string &dependent, const std::string &source, AssetEvent event) {
                                   Require(source == "dependency-a-0", "notification changed source identity");
                                   notified.push_back(dependent);
                               });
        graph.NotifyEvent("dependency-a-0", ResourceType::Mesh, AssetEvent::RuntimeModified);
        Require(notified == std::vector<std::string>{"runtime-object"},
                "runtime publication notified source assets or missed its runtime user");
        notified.clear();
        graph.NotifyEvent("dependency-a-0", ResourceType::Mesh, AssetEvent::Modified);
        Require(notified.size() == 2, "file modification stopped notifying the asset/runtime union");
        Require(graph.GetAssetGeneration() == 1, "asset dependency generation did not publish");
        Require(graph.HasDependency("asset-0", "dependency-a-0"), "asset edge was not published");
        Require(graph.HasDependency("runtime-object", "dependency-a-0"), "runtime overlay was replaced");
        const auto dependents = graph.GetDependents("dependency-a-0");
        Require(dependents.count("asset-0") == 1 && dependents.count("runtime-object") == 1,
                "asset/runtime dependent union is incomplete");
        Require(initial->GetGeneration() == 0 && initial->GetEdgeCount() == 0,
                "retained dependency snapshot changed after publication");

        graph.RemoveAsset("dependency-a-0");
        Require(graph.HasDependency("asset-0", "dependency-a-0"),
                "asset deletion discarded a serialized missing reference");
        Require(graph.HasDependency("runtime-object", "dependency-a-0"),
                "asset deletion discarded a runtime missing reference");

        bool rejectedStale = false;
        try {
            graph.InstallAssetSnapshot(built);
        } catch (const std::logic_error &) {
            rejectedStale = true;
        }
        Require(rejectedStale, "stale dependency generation was accepted");

        auto replacement = dependencies;
        replacement.erase("asset-0");
        graph.InstallAssetSnapshot(AssetDependencyGraph::BuildAssetSnapshot(replacement, 3));
        Require(!graph.HasDependency("asset-0", "dependency-a-0"), "replacement retained a removed asset edge");
        Require(graph.HasDependency("runtime-object", "dependency-a-0"), "replacement removed runtime usage");
        Require(built->GetEdgeCount() == userCount * 2, "retained generation changed after replacement");
        const auto sourceSnapshot = graph.GetAssetSnapshot();
        graph.AddRuntimeDependency("runtime-object", "second-runtime-asset");
        graph.RekeyRuntimeDependencies("runtime-object", "published-object");
        Require(!graph.HasDependency("runtime-object", "dependency-a-0"), "staged owner retained an edge");
        Require(graph.HasDependency("published-object", "dependency-a-0") &&
                    graph.HasDependency("published-object", "second-runtime-asset"),
                "publication lost an asset edge");
        notified.clear();
        graph.NotifyEvent("dependency-a-0", ResourceType::Mesh, AssetEvent::RuntimeModified);
        Require(notified == std::vector<std::string>{"published-object"}, "notification used the old owner ID");
        graph.RekeyRuntimeDependencies("published-object", "published-object");
        graph.RekeyRuntimeDependencies("no-runtime-assets", "unused-owner");
        Require(graph.GetAssetSnapshot() == sourceSnapshot, "runtime identity change rebuilt the source graph");
        graph.ClearRuntimeDependenciesOf("published-object");
        Require(graph.GetDependents("second-runtime-asset").empty(), "published owner cleanup retained an edge");

        graph.ClearRuntimeDependenciesOf("runtime-object");
        graph.Clear();
        std::cout << "AssetDependencyGraph tests passed; build_ms=" << buildMilliseconds << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "AssetDependencyGraph tests failed: " << error.what() << '\n';
        return 1;
    }
}
