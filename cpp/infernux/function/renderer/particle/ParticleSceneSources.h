#pragma once

#include "ParticleGpuSystemManager.h"
#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/SkinnedMeshRenderer.h>
#include <function/scene/Transform.h>

namespace infernux::particle
{

// Registered only while a graph borrows this component. Scene transfers keep
// its native lifetime; destruction retires the observer before storage is
// freed. No scene search, Python callback, or per-frame allocation is needed.
class SceneSkinnedMeshBinding final
{
    struct Observer final : NativeLifetimeObserver
    {
        explicit Observer(SkinnedMeshRenderer &source) : renderer(&source), generation(source.GetLifetimeGeneration())
        {
        }
        void RetireNativeObject() noexcept override
        {
            renderer = nullptr;
        }
        SkinnedMeshRenderer *renderer;
        const uint64_t generation;
    };

  public:
    explicit SceneSkinnedMeshBinding(SkinnedMeshRenderer &renderer) : m_observer(std::make_shared<Observer>(renderer))
    {
        renderer.ObserveNativeLifetime(m_observer);
    }
    ~SceneSkinnedMeshBinding()
    {
        if (m_observer->renderer)
            m_observer->renderer->ForgetNativeLifetimeObserver(m_observer.get());
    }
    SceneSkinnedMeshBinding(const SceneSkinnedMeshBinding &) = delete;
    SceneSkinnedMeshBinding &operator=(const SceneSkinnedMeshBinding &) = delete;

    [[nodiscard]] ObjectHandle GetHandle() const
    {
        const auto *renderer = m_observer->renderer;
        return renderer && renderer->GetLifetimeGeneration() == m_observer->generation ? renderer->GetHandle()
                                                                                       : ObjectHandle{};
    }

  private:
    std::shared_ptr<Observer> m_observer;
};

inline std::function<ObjectHandle()> BindSceneSkinnedMeshSource(SkinnedMeshRenderer &renderer)
{
    return [binding = std::make_shared<SceneSkinnedMeshBinding>(renderer)] { return binding->GetHandle(); };
}

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
