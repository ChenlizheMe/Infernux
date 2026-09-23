#include <function/renderer/CommandBuffer.h>
#include <function/renderer/RendererSelection.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace infernux;

int main()
{
    auto mesh = std::make_shared<InxMesh>("draw-capture");
    std::vector<Vertex> vertices(3);
    vertices[0].pos = {0.0F, 0.0F, 0.0F};
    vertices[1].pos = {1.0F, 0.0F, 0.0F};
    vertices[2].pos = {0.0F, 1.0F, 0.0F};
    mesh->SetData(std::move(vertices), {0, 1, 2}, {});
    mesh->SetIndexFormat(MeshIndexFormat::UInt32);

    auto material = std::make_shared<InxMaterial>("draw-capture-material");
    material->SetColor("_BaseColor", {1.0F, 1.0F, 1.0F, 1.0F});
    material->SetFloat("_Roughness", 0.5F);
    material->SetVector2("_Wind", {0.0F, 0.0F});
    material->SetVector3("_Axis", {0.0F, 1.0F, 0.0F});
    material->SetVector4("_Weights", {1.0F, 0.0F, 0.0F, 0.0F});
    material->SetInt("_Variant", 0);
    material->SetMatrix("_Local", glm::mat4(1.0F));
    material->SetTextureGuid("_Albedo", "white");
    material->SetFloatArray("_Curve", {0.0F, 0.5F, 1.0F});
    material->SetVector4Array("_Palette", {{1.0F, 0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F, 1.0F}});

    DrawParameterBlock parameters;
    parameters.SetColor("_BaseColor", {1.0F, 0.0F, 0.0F, 1.0F});
    parameters.SetFloat("_Roughness", 0.25F);
    parameters.SetVector2("_Wind", {1.0F, -0.5F});
    parameters.SetVector3("_Axis", {1.0F, 0.0F, 0.0F});
    parameters.SetVector4("_Weights", {0.5F, 0.25F, 0.125F, 0.125F});
    parameters.SetInt("_Variant", 3);
    parameters.SetMatrix("_Local", glm::mat4(2.0F));
    parameters.SetTexture("_Albedo", "white");
    parameters.SetFloatArray("_Curve", {0.25F, 0.5F, 0.75F});
    parameters.SetVector4Array("_Palette", {{0.5F, 0.0F, 0.0F, 1.0F}, {0.0F, 0.5F, 0.0F, 1.0F}});
    CommandBuffer commands("draw-capture");
    commands.DrawMesh(mesh, glm::mat4(1.0F), material, 0, 0, &parameters);

    parameters.SetColor("_BaseColor", {0.0F, 1.0F, 0.0F, 1.0F});
    commands.DrawMesh(mesh, glm::mat4(1.0F), material, 0, 0, &parameters);
    commands.DrawMesh(mesh, glm::mat4(1.0F), material, 0, 0, &parameters);
    const auto &recorded = std::get<DrawMeshParams>(commands.GetCommands().at(0).data);
    assert(recorded.parameterBlock);
    const auto captured = recorded.parameterBlock->properties.at("_BaseColor");
    assert(std::get<glm::vec4>(captured.value) == glm::vec4(1.0F, 0.0F, 0.0F, 1.0F));
    assert(std::get<float>(recorded.parameterBlock->properties.at("_Roughness").value) == 0.25F);
    assert(std::get<glm::vec2>(recorded.parameterBlock->properties.at("_Wind").value) == glm::vec2(1.0F, -0.5F));
    assert(std::get<glm::vec3>(recorded.parameterBlock->properties.at("_Axis").value) == glm::vec3(1.0F, 0.0F, 0.0F));
    assert(std::get<glm::vec4>(recorded.parameterBlock->properties.at("_Weights").value) ==
           glm::vec4(0.5F, 0.25F, 0.125F, 0.125F));
    assert(std::get<int>(recorded.parameterBlock->properties.at("_Variant").value) == 3);
    assert(std::get<glm::mat4>(recorded.parameterBlock->properties.at("_Local").value) == glm::mat4(2.0F));
    assert(std::get<std::string>(recorded.parameterBlock->properties.at("_Albedo").value) == "white");
    assert(std::get<std::vector<float>>(recorded.parameterBlock->properties.at("_Curve").value) ==
           std::vector<float>({0.25F, 0.5F, 0.75F}));
    assert(std::get<std::vector<glm::vec4>>(recorded.parameterBlock->properties.at("_Palette").value).size() == 2);
    assert(recorded.geometry);
    assert(recorded.vertices == &recorded.geometry->vertices);
    assert(recorded.indices == &recorded.geometry->indices);
    assert(recorded.meshIndexFormat == MeshIndexFormat::UInt32);

    // The public CPU topology remains uint32 while Auto/UInt16 select the
    // actual GPU encoding. Explicit UInt16 is range-checked, never truncated.
    assert(ResolveMeshIndexFormat(MeshIndexFormat::Auto, 3, recorded.geometry->indices) ==
           MeshIndexFormat::UInt16);
    assert(ResolveMeshIndexFormat(MeshIndexFormat::UInt32, 3, recorded.geometry->indices) ==
           MeshIndexFormat::UInt32);
    bool rejectedNarrowIndex = false;
    try {
        (void)ResolveMeshIndexFormat(MeshIndexFormat::UInt16, 65537, std::vector<uint32_t>{0, 65536, 0});
    } catch (const std::invalid_argument &) {
        rejectedNarrowIndex = true;
    }
    assert(rejectedNarrowIndex);

    const auto &second = std::get<DrawMeshParams>(commands.GetCommands().at(1).data);
    const auto &third = std::get<DrawMeshParams>(commands.GetCommands().at(2).data);
    assert(second.parameterBlock && third.parameterBlock);
    assert(second.parameterBlock == third.parameterBlock); // stable identity preserves instancing
    assert(second.parameterBlock != recorded.parameterBlock);
    assert(std::get<glm::vec4>(second.parameterBlock->properties.at("_BaseColor").value) ==
           glm::vec4(0.0F, 1.0F, 0.0F, 1.0F));

    // VkCoreDraw uses immutable publication identity as part of its instance
    // batch key. Quantify the authored stream here: the changed payload opens
    // one new batch, while two consecutive captures of the unchanged block do
    // not. This is deliberately a publication count, not a value comparison.
    const auto publicationBatchCount = [&] {
        const RendererParameterBlock *previous = nullptr;
        size_t count = 0;
        for (const auto &command : commands.GetCommands()) {
            const auto *draw = std::get_if<DrawMeshParams>(&command.data);
            if (!draw)
                continue;
            const auto *publication = draw->parameterBlock.get();
            if (publication != previous) {
                ++count;
                previous = publication;
            }
        }
        return count;
    };
    assert(publicationBatchCount() == 2);

    // Measure the author-facing update-frequency boundary without inventing a
    // synthetic fast path: one unchanged publication keeps a 4k instance
    // stream in one batch; changing a value for every draw necessarily creates
    // 4k immutable publications/batches. Timings are diagnostic evidence, not
    // a machine-specific pass threshold.
    constexpr size_t measuredDrawCount = 4096;
    const auto measureBatches = [](const CommandBuffer &buffer) {
        const RendererParameterBlock *previous = nullptr;
        size_t batches = 0;
        for (const auto &command : buffer.GetCommands()) {
            const auto *draw = std::get_if<DrawMeshParams>(&command.data);
            if (!draw)
                continue;
            if (draw->parameterBlock.get() != previous) {
                previous = draw->parameterBlock.get();
                ++batches;
            }
        }
        return batches;
    };
    DrawParameterBlock stableParameters;
    stableParameters.SetFloat("_Roughness", 0.5F);
    CommandBuffer stableBatch("stable-parameter-frequency");
    const auto stableStart = std::chrono::steady_clock::now();
    for (size_t index = 0; index < measuredDrawCount; ++index)
        stableBatch.DrawMesh(mesh, glm::mat4(1.0F), material, 0, 0, &stableParameters);
    const auto stableElapsed = std::chrono::steady_clock::now() - stableStart;
    assert(measureBatches(stableBatch) == 1);

    DrawParameterBlock perDrawParameters;
    CommandBuffer perDrawBatch("per-draw-parameter-frequency");
    const auto dynamicStart = std::chrono::steady_clock::now();
    for (size_t index = 0; index < measuredDrawCount; ++index) {
        perDrawParameters.SetFloat("_Roughness", static_cast<float>(index) / measuredDrawCount);
        perDrawBatch.DrawMesh(mesh, glm::mat4(1.0F), material, 0, 0, &perDrawParameters);
    }
    const auto dynamicElapsed = std::chrono::steady_clock::now() - dynamicStart;
    assert(measureBatches(perDrawBatch) == measuredDrawCount);
    std::cout << "Renderer parameter frequency: stable="
              << std::chrono::duration<double, std::milli>(stableElapsed).count() << "ms/1 batch, per-draw="
              << std::chrono::duration<double, std::milli>(dynamicElapsed).count() << "ms/" << measuredDrawCount
              << " batches\n";

    // Failed writes do not poison the last complete publication.
    bool rejectedNonFinite = false;
    try {
        parameters.SetFloat("_BaseColor", std::numeric_limits<float>::quiet_NaN());
    } catch (const std::invalid_argument &) {
        rejectedNonFinite = true;
    }
    assert(rejectedNonFinite);
    commands.DrawMesh(mesh, glm::mat4(1.0F), material, 0, 0, &parameters);
    const auto &afterRejected = std::get<DrawMeshParams>(commands.GetCommands().at(3).data);
    assert(afterRejected.parameterBlock == second.parameterBlock);
    assert(publicationBatchCount() == 2);

    DrawParameterBlock invalid;
    invalid.SetFloat("_BaseColor", 1.0F);
    bool rejected = false;
    try {
        CommandBuffer bad;
        bad.DrawMesh(mesh, glm::mat4(1.0F), material, 0, 0, &invalid);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    DrawParameterBlock wrongArrayLength;
    wrongArrayLength.SetFloatArray("_Curve", {1.0F, 2.0F});
    rejected = false;
    try {
        CommandBuffer bad;
        bad.DrawMesh(mesh, glm::mat4(1.0F), material, 0, 0, &wrongArrayLength);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    RendererSelection selection(material);
    const auto renderer = RenderProxyHandle::FromScene({10, 1, 7}, {11, 1, 7});
    const auto other = RenderProxyHandle::FromScene({20, 1, 7}, {21, 1, 7});
    selection.Set(renderer, -1, &parameters);
    const auto *all = selection.Find(renderer.MakeDrawIdentity(4));
    assert(all && *all);
    const auto retained = *all;
    parameters.SetColor("_BaseColor", {0.0F, 0.0F, 1.0F, 1.0F});
    assert(std::get<glm::vec4>(retained->properties.at("_BaseColor").value) == glm::vec4(0, 1, 0, 1));
    selection.Set(renderer, 4, &parameters);
    assert(selection.Size() == 2);
    assert(selection.Find(renderer.MakeDrawIdentity(4))->get() != retained.get());
    assert(selection.Find(renderer.MakeDrawIdentity(3))->get() == retained.get());
    assert(!selection.Find(other.MakeDrawIdentity(4)));
    // The same authored IDs in another world or a replacement component are
    // different lifetimes. Neither may inherit the retired selection.
    assert(!selection.Find(RenderProxyHandle::FromScene({10, 1, 8}, {11, 1, 8}).MakeDrawIdentity(4)));
    assert(!selection.Find(RenderProxyHandle::FromScene({10, 1, 7}, {11, 2, 7}).MakeDrawIdentity(4)));
    const auto revision = selection.Revision();
    rejected = false;
    try {
        selection.Set(renderer, 4, &invalid);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected && selection.Revision() == revision && selection.Size() == 2);
    assert(selection.Remove(renderer, 4));
    assert(selection.Find(renderer.MakeDrawIdentity(4))->get() == retained.get());
    assert(!selection.Remove(renderer, 4));
    selection.Set(renderer, 4); // A selected submesh with material defaults.
    assert(selection.Find(renderer.MakeDrawIdentity(4)) && !*selection.Find(renderer.MakeDrawIdentity(4)));
    selection.Clear();
    assert(selection.Size() == 0 && !selection.Find(renderer.MakeDrawIdentity()));
    for (int submesh : {-2, -19}) {
        rejected = false;
        try {
            selection.Set(renderer, submesh);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected && selection.Size() == 0);
    }
    return 0;
}
