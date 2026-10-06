#pragma once

#include "ParticleGpuSystemManager.h"
#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/SkinnedMeshRenderer.h>
#include <function/scene/Transform.h>

namespace infernux::particle
{

// Scene ownership and lifetime checks stay at the scene-to-particle boundary;
// the GPU manager only consumes immutable mesh and pose snapshots.
inline std::optional<GpuParticleSkinnedMeshSnapshot> ResolveSceneSkinnedMeshSource(const ObjectHandle &handle)
{
    Scene *scene = SceneManager::Instance().GetSceneByWorldId(handle.worldId);
    auto *renderer = scene ? dynamic_cast<SkinnedMeshRenderer *>(scene->ResolveComponent(handle)) : nullptr;
    if (!renderer || !renderer->GetGameObject() || !renderer->GetGameObject()->GetTransform())
        return std::nullopt;
    const auto pose = renderer->GetRuntimeSkinPoseSnapshot();
    auto mesh = renderer->GetMeshAssetRef().Get();
    auto model = renderer->GetRuntimeModelSnapshot();
    if (!pose || !pose->IsValid() || !mesh || !model)
        return std::nullopt;
    GpuParticleSkinnedMeshSnapshot snapshot;
    snapshot.mesh = std::move(mesh);
    snapshot.model = std::move(model);
    snapshot.currentPalette = pose->current;
    snapshot.previousPalette = pose->previous;
    snapshot.revision = pose->revision;
    const glm::mat4 &world = renderer->GetGameObject()->GetTransform()->GetWorldMatrix();
    for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t column = 0; column < 4; ++column)
            snapshot.sourceToWorld[row * 4 + column] = world[column][row];
    }
    return snapshot;
}

} // namespace infernux::particle
