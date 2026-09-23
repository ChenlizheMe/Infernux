#include <function/renderer/GizmosDrawCallBuffer.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
infernux::GizmosDrawCallBuffer::DrawDescriptor Descriptor(float x)
{
    infernux::GizmosDrawCallBuffer::DrawDescriptor descriptor;
    descriptor.indexCount = 2;
    const glm::mat4 world = glm::translate(glm::mat4(1.0f), glm::vec3(x, 0.0f, 0.0f));
    std::memcpy(descriptor.worldMatrix, &world, sizeof(descriptor.worldMatrix));
    return descriptor;
}

std::vector<infernux::Vertex> Line(float end)
{
    return {infernux::Vertex::Create({0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}),
            infernux::Vertex::Create({end, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f})};
}
} // namespace

int main()
{
    infernux::GizmosDrawCallBuffer buffer;
    buffer.SetData(Line(1.0f), {0, 1}, {Descriptor(0.0f)});
    auto first = buffer.GetDrawCalls(nullptr);
    assert(first.drawCalls.size() == 1);
    const uint64_t initialRevision = first.drawCalls[0].meshRuntimeVersion;
    assert(initialRevision != 0 && !first.drawCalls[0].forceBufferUpdate);

    // Transform-only changes belong to the draw parameters and must retain
    // the resident mesh generation.
    buffer.SetData(Line(1.0f), {0, 1}, {Descriptor(3.0f)});
    auto moved = buffer.GetDrawCalls(nullptr);
    assert(moved.drawCalls.size() == 1);
    assert(moved.drawCalls[0].meshRuntimeVersion == initialRevision);
    assert(moved.drawCalls[0].worldMatrix[3].x == 3.0f);

    buffer.SetData(Line(2.0f), {0, 1}, {Descriptor(3.0f)});
    auto changed = buffer.GetDrawCalls(nullptr);
    assert(changed.drawCalls.size() == 1);
    assert(changed.drawCalls[0].meshRuntimeVersion != initialRevision);

    infernux::GizmosDrawCallBuffer::IconEntry icon;
    icon.objectId = 42;
    icon.position = {0.0f, 0.0f, 0.0f};
    buffer.SetIconData({icon});
    infernux::GizmosDrawCallBuffer::IconMaterials materials;
    // A null material intentionally suppresses the draw, so use an aliasing
    // non-owning token; GetIconDrawCalls only tests pointer truth here.
    auto token = std::shared_ptr<infernux::InxMaterial>(reinterpret_cast<infernux::InxMaterial *>(1),
                                                       [](infernux::InxMaterial *) {});
    materials.fallback = token;
    auto iconFirst = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, 10.0f}, {1.0f, 0.0f, 0.0f},
                                             {0.0f, 1.0f, 0.0f});
    assert(iconFirst.drawCalls.size() == 1);
    assert(iconFirst.drawCalls[0].objectId != icon.objectId);
    assert(iconFirst.drawCalls[0].pickingObjectId == icon.objectId);
    const uint64_t iconRevision = iconFirst.drawCalls[0].meshRuntimeVersion;
    auto iconStable = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, 10.0f}, {1.0f, 0.0f, 0.0f},
                                              {0.0f, 1.0f, 0.0f});
    assert(iconStable.drawCalls[0].meshRuntimeVersion == iconRevision);
    auto iconCameraMoved = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, 20.0f}, {1.0f, 0.0f, 0.0f},
                                                   {0.0f, 1.0f, 0.0f});
    assert(iconCameraMoved.drawCalls[0].meshRuntimeVersion != iconRevision);

    // Built-in kinds are strict routes.  Falling back to the generic white
    // material while a dedicated material is being published produces the
    // intermittent white camera/light squares seen in Scene view.
    auto cameraToken = std::shared_ptr<infernux::InxMaterial>(reinterpret_cast<infernux::InxMaterial *>(2),
                                                             [](infernux::InxMaterial *) {});
    auto lightToken = std::shared_ptr<infernux::InxMaterial>(reinterpret_cast<infernux::InxMaterial *>(3),
                                                            [](infernux::InxMaterial *) {});
    materials.camera = cameraToken;
    materials.light = lightToken;

    icon.iconKind = infernux::GizmosDrawCallBuffer::ICON_KIND_CAMERA;
    buffer.SetIconData({icon});
    auto cameraIcon = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, 10.0f}, {1.0f, 0.0f, 0.0f},
                                              {0.0f, 1.0f, 0.0f});
    assert(cameraIcon.drawCalls.size() == 1);
    assert(cameraIcon.drawCalls[0].material == cameraToken);

    icon.iconKind = infernux::GizmosDrawCallBuffer::ICON_KIND_LIGHT;
    buffer.SetIconData({icon});
    auto lightIcon = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, 10.0f}, {1.0f, 0.0f, 0.0f},
                                             {0.0f, 1.0f, 0.0f});
    assert(lightIcon.drawCalls.size() == 1);
    assert(lightIcon.drawCalls[0].material == lightToken);

    materials.camera.reset();
    icon.iconKind = infernux::GizmosDrawCallBuffer::ICON_KIND_CAMERA;
    buffer.SetIconData({icon});
    auto unpublishedCamera = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, 10.0f}, {1.0f, 0.0f, 0.0f},
                                                     {0.0f, 1.0f, 0.0f});
    assert(unpublishedCamera.drawCalls.empty());

    std::cout << "Gizmo draw buffer generation tests passed\n";
    return 0;
}
