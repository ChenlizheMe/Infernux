#include <function/renderer/CommandBuffer.h>
#include <function/renderer/RendererSelection.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
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

    auto material = std::make_shared<InxMaterial>("draw-capture-material");
    material->SetColor("_BaseColor", {1.0F, 1.0F, 1.0F, 1.0F});

    DrawParameterBlock parameters;
    parameters.SetColor("_BaseColor", {1.0F, 0.0F, 0.0F, 1.0F});
    CommandBuffer commands("draw-capture");
    commands.DrawMesh(mesh, glm::mat4(1.0F), material, 0, 0, &parameters);

    parameters.SetColor("_BaseColor", {0.0F, 1.0F, 0.0F, 1.0F});
    const auto &recorded = std::get<DrawMeshParams>(commands.GetCommands().at(0).data);
    assert(recorded.parameterBlock);
    const auto captured = recorded.parameterBlock->properties.at("_BaseColor");
    assert(std::get<glm::vec4>(captured.value) == glm::vec4(1.0F, 0.0F, 0.0F, 1.0F));
    assert(recorded.geometry);
    assert(recorded.vertices == &recorded.geometry->vertices);
    assert(recorded.indices == &recorded.geometry->indices);

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
