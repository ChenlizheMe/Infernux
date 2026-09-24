#include <function/renderer/GizmosDrawCallBuffer.h>
#include <function/resources/AssetDatabase/BuiltinSceneIconMetadata.h>
#include <function/resources/AssetImporter/ConcreteImporters.h>
#include <function/resources/InxTexture/TextureArtifact.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
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
    const glm::mat4 iconProjection = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    // Infernux Transform.forward is +Z; the editor camera is behind an icon
    // at the origin when it sits at -Z. This was previously tested backwards,
    // allowing billboards to collapse to a subpixel quad in the real editor.
    auto iconFirst = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, -10.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                                             iconProjection, 800, 1.0f);
    assert(iconFirst.drawCalls.size() == 1);
    assert(iconFirst.drawCalls[0].objectId != icon.objectId);
    assert(iconFirst.drawCalls[0].pickingObjectId == icon.objectId);
    const auto &iconVertices = *iconFirst.drawCalls[0].meshVertices;
    const float halfWidth = iconVertices[1].pos.x;
    assert(std::abs(halfWidth * 800.0f * iconProjection[1][1] / (2.0f * 10.0f) -
                    infernux::GizmosDrawCallBuffer::ICON_HALF_SIZE_PIXELS) < 1.0e-4f);
    const uint64_t iconRevision = iconFirst.drawCalls[0].meshRuntimeVersion;
    auto iconStable = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, -10.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                                              iconProjection, 800, 1.0f);
    assert(iconStable.drawCalls[0].meshRuntimeVersion == iconRevision);
    auto iconHiDpi = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, -10.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                                             iconProjection, 800, 2.0f);
    assert(std::abs(iconHiDpi.drawCalls[0].meshVertices->at(1).pos.x - 2.0f * halfWidth) < 1.0e-5f);
    const glm::mat4 orthoProjection = glm::ortho(-5.0f, 5.0f, -5.0f, 5.0f, 0.1f, 100.0f);
    auto iconOrtho = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, -10.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                                             orthoProjection, 800, 1.0f);
    assert(std::abs(iconOrtho.drawCalls[0].meshVertices->at(1).pos.x * 800.0f * orthoProjection[1][1] / 2.0f -
                    infernux::GizmosDrawCallBuffer::ICON_HALF_SIZE_PIXELS) < 1.0e-4f);
    auto iconCameraMoved = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, -20.0f}, {1.0f, 0.0f, 0.0f},
                                                   {0.0f, 1.0f, 0.0f}, iconProjection, 800, 1.0f);
    assert(iconCameraMoved.drawCalls[0].meshRuntimeVersion != iconRevision);
    assert(std::abs(iconCameraMoved.drawCalls[0].meshVertices->at(1).pos.x - 2.0f * halfWidth) < 1.0e-5f);

    // An explicit view-matrix override can flip the camera basis handedness.
    // Its axial distance, and therefore projected icon size, must agree.
    const float oppositeBasisSize = infernux::GizmosDrawCallBuffer::ComputeIconHalfWorldSize(
        icon.position, {0.0f, 0.0f, -10.0f}, {0.0f, 0.0f, -1.0f}, iconProjection, 800, 1.0f);
    assert(std::abs(oppositeBasisSize - halfWidth) < 1.0e-5f);
    auto iconMirrored = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, 10.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                                                iconProjection, 800, 1.0f);
    assert(std::abs(std::abs(iconMirrored.drawCalls[0].meshVertices->at(1).pos.x) - halfWidth) < 1.0e-5f);

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
    auto cameraIcon = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, -10.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                                              iconProjection, 800, 1.0f);
    assert(cameraIcon.drawCalls.size() == 1);
    assert(cameraIcon.drawCalls[0].material == cameraToken);

    icon.iconKind = infernux::GizmosDrawCallBuffer::ICON_KIND_LIGHT;
    buffer.SetIconData({icon});
    auto lightIcon = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, -10.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                                             iconProjection, 800, 1.0f);
    assert(lightIcon.drawCalls.size() == 1);
    assert(lightIcon.drawCalls[0].material == lightToken);

    materials.camera.reset();
    icon.iconKind = infernux::GizmosDrawCallBuffer::ICON_KIND_CAMERA;
    buffer.SetIconData({icon});
    auto unpublishedCamera = buffer.GetIconDrawCalls(materials, {0.0f, 0.0f, -10.0f}, {1.0f, 0.0f, 0.0f},
                                                     {0.0f, 1.0f, 0.0f}, iconProjection, 800, 1.0f);
    assert(unpublishedCamera.drawCalls.empty());

    const auto sourcePath = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
                            "python/Infernux/resources/icons/gizmo_light.png";
    const std::string iconPath = infernux::FromFsPath(sourcePath);
    std::ifstream source(sourcePath, std::ios::binary);
    assert(source.good());
    const std::string bytes((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    infernux::ImportRequest iconRequest;
    iconRequest.sourcePath = iconPath;
    iconRequest.resourceType = infernux::ResourceType::Texture;
    iconRequest.metadata.Init(bytes.data(), bytes.size(), iconPath, infernux::ResourceType::Texture);
    assert(!infernux::HasCurrentBuiltinSceneIconMetadata(iconRequest.metadata, iconPath, true));
    infernux::ApplyBuiltinSceneIconMetadata(iconRequest.metadata, iconPath, true);
    assert(infernux::HasCurrentBuiltinSceneIconMetadata(iconRequest.metadata, iconPath, true));
    const std::string libraryMirrorPath = "D:/Projects/GizmoLab/Library/Resources/icons/gizmo_light.png";
    infernux::InxResourceMeta mirrorMetadata;
    assert(!infernux::HasCurrentBuiltinSceneIconMetadata(mirrorMetadata, libraryMirrorPath, true));
    infernux::ApplyBuiltinSceneIconMetadata(mirrorMetadata, libraryMirrorPath, true);
    assert(infernux::HasCurrentBuiltinSceneIconMetadata(mirrorMetadata, libraryMirrorPath, true));
    assert(iconRequest.metadata.GetDataAs<std::string>("texture_type") == "ui");
    assert(iconRequest.metadata.GetDataAs<std::string>("texture_compression") == "none");
    assert(iconRequest.metadata.GetDataAs<std::string>("wrap_mode") == "clamp");
    const auto importedIcon = infernux::TextureImporter{}.Import(iconRequest);
    const auto iconTexture = infernux::TextureArtifact::Deserialize(
        importedIcon.runtimeCpuArtifacts.front().bytes, importedIcon.metadata.GetDataAs<std::string>("content_hash"));
    assert(iconTexture->semantic == infernux::TextureSemantic::UserInterface);
    assert(iconTexture->format == infernux::TextureFormat::Rgba8Srgb);
    assert(iconTexture->mipLevels.size() > 1);
    const auto &baseMip = iconTexture->mipLevels.front();
    bool hasTransparentTexel = false;
    bool hasVisibleTexel = false;
    for (uint64_t offset = baseMip.byteOffset + 3; offset < baseMip.byteOffset + baseMip.byteSize; offset += 4) {
        const uint8_t alpha = iconTexture->bytes.at(static_cast<size_t>(offset));
        hasTransparentTexel = hasTransparentTexel || alpha == 0;
        hasVisibleTexel = hasVisibleTexel || alpha != 0;
    }
    assert(hasTransparentTexel && hasVisibleTexel);
    iconRequest.metadata.AddMetadata("texture_compression", std::string("auto"));
    assert(!infernux::HasCurrentBuiltinSceneIconMetadata(iconRequest.metadata, iconPath, true));
    assert(infernux::HasCurrentBuiltinSceneIconMetadata(iconRequest.metadata, iconPath, false));

    std::cout << "Gizmo draw buffer generation tests passed\n";
    return 0;
}
