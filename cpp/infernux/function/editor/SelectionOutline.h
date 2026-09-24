#pragma once

#include <function/scene/GameObject.h>
#include <function/scene/SceneManager.h>
#include <unordered_set>
#include <vector>

namespace infernux
{

// Selection belongs to the loaded World, not the active authoring Scene.
// Keep the primary selection separate; only the mask expands descendants.
inline std::vector<uint64_t> ExpandSelectionOutlineIds(const SceneManager &manager,
                                                       const std::vector<uint64_t> &objectIds)
{
    std::vector<uint64_t> expanded;
    std::unordered_set<uint64_t> seen;
    const auto collect = [&](const auto &self, GameObject *object) -> void {
        if (!object || !object->IsActiveInHierarchy())
            return;
        const uint64_t id = object->GetID();
        if (id == 0 || !seen.insert(id).second)
            return;
        expanded.push_back(id);
        for (size_t i = 0; i < object->GetChildCount(); ++i)
            self(self, object->GetChild(i));
    };
    for (const uint64_t id : objectIds)
        collect(collect, manager.FindRuntimeObjectByID(id));
    return expanded;
}

} // namespace infernux
