#include "HierarchyPanel.h"

#include <function/renderer/gui/InxGUISemantics.h>
#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/Transform.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

// ImGui key constants (must match imgui.h ImGuiKey enum)
static constexpr int kKeyLeftCtrl = ImGuiKey_LeftCtrl;
static constexpr int kKeyRightCtrl = ImGuiKey_RightCtrl;
static constexpr int kKeyLeftShift = ImGuiKey_LeftShift;
static constexpr int kKeyRightShift = ImGuiKey_RightShift;

namespace infernux
{

// ════════════════════════════════════════════════════════════════════
// Helpers — casefold for search
// ════════════════════════════════════════════════════════════════════

static std::string ExternalDropCommandArgument(const std::string &reference, uint64_t parentId, bool isGuid)
{
    return reference + "\t" + std::to_string(parentId) + "\t" + (isGuid ? "1" : "0");
}

static std::string RenameCommandArgument(uint64_t objectId, const std::string &name)
{
    return std::to_string(objectId) + "\t" + name;
}

static std::string TreeExpandedCommandArgument(uint64_t objectId, bool expanded)
{
    return std::to_string(objectId) + "\t" + (expanded ? "1" : "0");
}

static std::string MoveCommandArgument(const std::vector<uint64_t> &objectIds, const std::string &mode,
                                       uint64_t targetId, bool after, uint64_t destinationWorldId = 0)
{
    std::string ids;
    for (const uint64_t objectId : objectIds) {
        if (!ids.empty())
            ids.push_back(',');
        ids += std::to_string(objectId);
    }
    return ids + "\t" + mode + "\t" + std::to_string(targetId) + "\t" + (after ? "1" : "0") + "\t" +
           std::to_string(destinationWorldId);
}

// ════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════

HierarchyPanel::HierarchyPanel() : EditorPanel("Hierarchy", "hierarchy")
{
}

std::unordered_map<std::string, double> HierarchyPanel::ConsumeSubTimings()
{
    std::unordered_map<std::string, double> out;
    out["pre_hidden"] = m_subPreHidden;
    out["pre_select"] = m_subPreSelection;
    out["pre_shortcuts"] = m_subPreShortcuts;
    out["pre_pending"] = m_subPrePendingSelect;
    out["header"] = m_subHeader;
    out["search"] = m_subSearch;
    out["refresh"] = m_subRefreshRoots;
    out["filterRoots"] = m_subFilterRoots;
    out["flatBuild"] = m_subFlatBuild;
    out["rows"] = m_subRows;
    out["popup"] = m_subPopup;
    out["tailDrop"] = m_subTailDrop;

    m_subPreHidden = 0.0;
    m_subPreSelection = 0.0;
    m_subPreShortcuts = 0.0;
    m_subPrePendingSelect = 0.0;
    m_subHeader = 0.0;
    m_subSearch = 0.0;
    m_subRefreshRoots = 0.0;
    m_subFilterRoots = 0.0;
    m_subFlatBuild = 0.0;
    m_subRows = 0.0;
    m_subPopup = 0.0;
    m_subTailDrop = 0.0;
    return out;
}

// ════════════════════════════════════════════════════════════════════
// Translation helper
// ════════════════════════════════════════════════════════════════════

const std::string &HierarchyPanel::Tr(const std::string &key)
{
    auto it = m_trCache.find(key);
    if (it != m_trCache.end())
        return it->second;
    if (translate)
        m_trCache[key] = translate(key);
    else
        m_trCache[key] = key;
    return m_trCache[key];
}

// ════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::InvalidateSceneStructureCache()
{
    m_cachedSceneKey.clear();
    m_cachedStructureVer = UINT64_MAX;
    m_lastRootRefreshTime = 0.0f;
    m_orderedIdsDirty = true;
    m_searchVisCache.clear();
    m_itemHeightMeasured = false;
    m_flatItems.clear();
    m_cachedScenes.clear();
    m_flatListDirty = true;
}

void HierarchyPanel::ClearSearch()
{
    SetSearchQuery("");
}

void HierarchyPanel::ClearSelectionAndNotify()
{
    if (clearSelection)
        clearSelection();
    SyncSelectionCache();
}

void HierarchyPanel::SetSelectedObjectById(uint64_t id, bool clearSearchFirst)
{
    if (id == 0)
        id = 0;
    if (clearSearchFirst)
        ClearSearch();

    uint64_t curPrimary = getPrimary ? getPrimary() : 0;
    int curCount = selectionCount ? selectionCount() : 0;
    bool changed = (curPrimary != id || curCount != 1);
    if (changed && selectId)
        selectId(id);

    // This API represents an explicit single selection. Keep the native
    // snapshot coherent immediately; push-mode listeners may run before or
    // after the Python selection callback depending on where the action
    // originated in the current ImGui frame.
    m_selIds.clear();
    m_selOrderedIds.clear();
    if (id) {
        m_selIds.insert(id);
        m_selOrderedIds.push_back(id);
        m_selCount = 1;
    } else {
        m_selCount = 0;
    }
    m_selPrimary = id;
    if (id) {
        m_selectedSceneWorldId = 0;
        m_pendingSceneSelectWorldId = 0;
    }

    // Always expand the parent chain
    if (id) {
        ExpandToObject(id);
        m_scrollToObjectId = id;
        const bool missingFromCache = std::none_of(m_flatItems.begin(), m_flatItems.end(), [id](const FlatItem &item) {
            return item.obj && item.obj->GetID() == id;
        });
        m_forceRootRefresh = missingFromCache;
        if (missingFromCache) {
            Scene *scene = SceneManager::Instance().GetActiveScene();
            GameObject *selected = SceneManager::Instance().FindRuntimeObjectByID(id);
            if (selected) {
                // Runtime-only IDs can be recycled after leaving Play Mode.
                // An explicitly revealed live scene object must not inherit a
                // stale hidden marker from the previous runtime world.
                m_hiddenIds.erase(id);
                RefreshRootObjects(scene, false, true);
                m_forceRootRefresh = false;
            }
        }
    }
}

void HierarchyPanel::ExpandToObject(uint64_t objId)
{
    if (objId == 0)
        return;
    GameObject *go = SceneManager::Instance().FindRuntimeObjectByID(objId);
    if (!go)
        return;
    GameObject *parent = go->GetParent();
    while (parent) {
        uint64_t pid = parent->GetID();
        if (m_treeProjection.SetExpanded(pid, true))
            m_flatListDirty = true;
        m_forceExpandIds.insert(pid);
        parent = parent->GetParent();
    }
}

// ════════════════════════════════════════════════════════════════════
// Selection cache — sync once per frame
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::SyncSelectionCache()
{
    if (m_selectionPushMode)
        return;
    m_selIds.clear();
    m_selOrderedIds.clear();
    if (getSelectedIds) {
        m_selOrderedIds = getSelectedIds();
        for (auto id : m_selOrderedIds)
            m_selIds.insert(id);
    }
    m_selPrimary = getPrimary ? getPrimary() : 0;
    m_selCount = selectionCount ? selectionCount() : 0;
    if (m_selCount > 0) {
        m_selectedSceneWorldId = 0;
        m_pendingSceneSelectWorldId = 0;
    }
}

void HierarchyPanel::SetSelectionSnapshot(const std::vector<uint64_t> &ids, uint64_t primary)
{
    m_selectionPushMode = true;
    m_selOrderedIds = ids;
    m_selIds.clear();
    m_selIds.insert(ids.begin(), ids.end());
    m_selPrimary = primary;
    m_selCount = static_cast<int>(ids.size());
    if (!ids.empty()) {
        m_selectedSceneWorldId = 0;
        m_pendingSceneSelectWorldId = 0;
    }
}

std::vector<uint64_t> HierarchyPanel::GetExpandedObjectIds() const
{
    std::vector<uint64_t> ids(m_treeProjection.ExpandedIds().begin(), m_treeProjection.ExpandedIds().end());
    std::sort(ids.begin(), ids.end());
    return ids;
}

void HierarchyPanel::SetExpandedObjectIds(const std::vector<uint64_t> &ids)
{
    const std::unordered_set<uint64_t> expanded(ids.begin(), ids.end());
    if (m_treeProjection.ReplaceExpanded(expanded)) {
        m_flatListDirty = true;
        m_orderedIdsDirty = true;
    }
}

void HierarchyPanel::SetRuntimeHiddenIds(const std::unordered_set<uint64_t> &ids)
{
    m_runtimeHiddenPushMode = true;
    if (m_hiddenIds == ids)
        return;
    m_hiddenIds = ids;
    m_forceRootRefresh = true;
    m_searchVisCache.clear();
    m_flatListDirty = true;
}

void HierarchyPanel::SetSceneHeaderSnapshot(const std::string &sceneDisplayName, bool prefabMode,
                                            const std::string &prefabDisplayName)
{
    m_sceneHeaderPushMode = true;
    m_sceneDisplayName = sceneDisplayName;
    m_cachedPrefabMode = prefabMode;
    m_prefabDisplayName = prefabDisplayName;
}

bool HierarchyPanel::IsPrefabModeActive() const
{
    return m_sceneHeaderPushMode ? m_cachedPrefabMode : (isPrefabMode && isPrefabMode());
}

std::string HierarchyPanel::SceneDisplayName() const
{
    return m_sceneHeaderPushMode ? m_sceneDisplayName : (getSceneDisplayName ? getSceneDisplayName() : "");
}

std::string HierarchyPanel::PrefabDisplayName() const
{
    return m_sceneHeaderPushMode ? m_prefabDisplayName : (getPrefabDisplayName ? getPrefabDisplayName() : "Prefab");
}

// ════════════════════════════════════════════════════════════════════
// Hidden-object filtering
// ════════════════════════════════════════════════════════════════════

bool HierarchyPanel::IsHidden(uint64_t id) const
{
    return m_hiddenIds.count(id) > 0;
}

std::vector<GameObject *> HierarchyPanel::FilterHidden(const std::vector<std::unique_ptr<GameObject>> &objects) const
{
    std::vector<GameObject *> out;
    out.reserve(objects.size());
    for (auto &obj : objects) {
        if (!IsHidden(obj->GetID()))
            out.push_back(obj.get());
    }
    return out;
}

void HierarchyPanel::RefreshRootObjects(Scene *scene, bool allowStale, bool forceRefresh)
{
    (void)allowStale;
    if (!scene) {
        m_cachedRoots.clear();
        m_cachedScenes.clear();
        m_cachedRawRootCount = 0;
        m_selectedSceneWorldId = 0;
        m_pendingSceneSelectWorldId = 0;
        return;
    }

    std::vector<Scene *> scenes;
    if (IsPrefabModeActive()) {
        scenes.push_back(scene);
    } else {
        const auto &loaded = SceneManager::Instance().GetAllScenes();
        scenes.reserve(loaded.size());
        for (const auto &loadedScene : loaded) {
            if (loadedScene)
                scenes.push_back(loadedScene.get());
        }
    }
    if (m_selectedSceneWorldId != 0 && std::none_of(scenes.begin(), scenes.end(), [this](Scene *loadedScene) {
            return loadedScene && loadedScene->GetWorldId() == m_selectedSceneWorldId;
        }))
        m_selectedSceneWorldId = 0;

    std::string sceneKey;
    size_t rawRootCount = 0;
    for (Scene *loadedScene : scenes) {
        sceneKey += std::to_string(loadedScene->GetWorldId());
        sceneKey += ':';
        sceneKey += std::to_string(loadedScene->GetStructureVersion());
        sceneKey += ':';
        sceneKey += std::to_string(loadedScene->GetRootObjects().size());
        sceneKey += ';';
        rawRootCount += loadedScene->GetRootObjects().size();
    }

    if (forceRefresh || rawRootCount != m_cachedRawRootCount || sceneKey != m_cachedSceneKey) {
        m_cachedScenes = std::move(scenes);
        m_cachedRoots.clear();
        m_cachedRoots.reserve(rawRootCount);
        for (Scene *loadedScene : m_cachedScenes) {
            auto roots = FilterHidden(loadedScene->GetRootObjects());
            m_cachedRoots.insert(m_cachedRoots.end(), roots.begin(), roots.end());
        }
        m_orderedIdsDirty = true;
        m_searchVisCache.clear();
        m_itemHeightMeasured = false;
        m_flatListDirty = true;
        m_cachedSceneKey = sceneKey;
        m_cachedStructureVer = scene->GetStructureVersion();
        m_cachedRawRootCount = rawRootCount;
        m_lastRootRefreshTime = ImGui::GetTime();
    }
}

// ════════════════════════════════════════════════════════════════════
// Search
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::SetSearchQuery(const char *text)
{
    if (!m_search.SetQuery(text ? text : ""))
        return;
    m_searchVisCache.clear();
    m_flatListDirty = true;
}

bool HierarchyPanel::MatchesSearch(GameObject *obj) const
{
    return m_search.Matches(obj->GetName());
}

bool HierarchyPanel::IsVisibleInSearch(GameObject *obj)
{
    if (!m_search.IsActive())
        return true;
    uint64_t id = obj->GetID();
    auto it = m_searchVisCache.find(id);
    if (it != m_searchVisCache.end())
        return it->second;

    bool visible = MatchesSearch(obj);
    if (!visible) {
        for (auto &child : obj->GetChildren()) {
            if (!IsHidden(child->GetID()) && IsVisibleInSearch(child.get())) {
                visible = true;
                break;
            }
        }
    }
    m_searchVisCache[id] = visible;
    return visible;
}

std::vector<GameObject *> HierarchyPanel::FilterForSearch(const std::vector<GameObject *> &objects)
{
    if (!HasActiveSearch())
        return objects;
    std::vector<GameObject *> out;
    for (auto *obj : objects) {
        if (IsVisibleInSearch(obj))
            out.push_back(obj);
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// Flat virtual scrolling — build a flat list of visible items
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::BuildFlatVisibleList(const std::vector<GameObject *> &roots)
{
    m_flatItems.clear();
    m_flatItems.reserve(roots.size() * 2 + m_cachedScenes.size()); // heuristic
    if (IsPrefabModeActive()) {
        for (auto *root : roots)
            BuildFlatListRecurse(root, 0);
    } else {
        for (Scene *scene : m_cachedScenes) {
            bool hasVisibleRoot = false;
            for (GameObject *root : roots) {
                if (root && root->GetScene() == scene) {
                    hasVisibleRoot = true;
                    break;
                }
            }
            if (HasActiveSearch() && !hasVisibleRoot)
                continue;
            m_flatItems.push_back({nullptr, 0, hasVisibleRoot, scene, true});
            if (m_collapsedSceneWorldIds.count(scene->GetWorldId()) != 0 && !HasActiveSearch())
                continue;
            for (GameObject *root : roots) {
                if (root && root->GetScene() == scene)
                    BuildFlatListRecurse(root, 1);
            }
        }
    }
    m_flatListDirty = false;
}

void HierarchyPanel::RebuildFlatListIfNeeded(const std::vector<GameObject *> &roots)
{
    if (m_flatListDirty) {
        BuildFlatVisibleList(roots);
    }
}

void HierarchyPanel::BuildFlatListRecurse(GameObject *obj, int depth)
{
    BuildFlatListRecurse(obj, depth, m_flatItems);
}

void HierarchyPanel::BuildFlatListRecurse(GameObject *obj, int depth, std::vector<FlatItem> &items)
{
    if (!obj)
        return;
    if (HasActiveSearch() && !IsVisibleInSearch(obj))
        return;

    uint64_t objId = obj->GetID();
    const auto &children = obj->GetChildren();

    // Check for visible children without allocating a vector
    bool hasVisibleChildren = false;
    for (const auto &child : children) {
        if (IsHidden(child->GetID()))
            continue;
        if (HasActiveSearch() && !IsVisibleInSearch(child.get()))
            continue;
        hasVisibleChildren = true;
        break;
    }

    items.push_back({obj, depth, hasVisibleChildren, obj->GetScene(), false});

    // Determine expanded state
    bool isExpanded = m_treeProjection.IsExpanded(objId);

    // Search expansion is transient.  Do not write it into the persistent
    // user-authored expansion set, otherwise clearing a search permanently
    // changes the tree layout.
    if (HasActiveSearch() && hasVisibleChildren) {
        isExpanded = true;
    }

    if (hasVisibleChildren && isExpanded) {
        for (const auto &child : children) {
            if (IsHidden(child->GetID()))
                continue;
            BuildFlatListRecurse(child.get(), depth + 1, items);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Keyboard helpers
// ════════════════════════════════════════════════════════════════════

bool HierarchyPanel::IsCtrl(InxGUIContext *ctx) const
{
    return ctx->IsKeyDown(kKeyLeftCtrl) || ctx->IsKeyDown(kKeyRightCtrl);
}

bool HierarchyPanel::IsShift(InxGUIContext *ctx) const
{
    return ctx->IsKeyDown(kKeyLeftShift) || ctx->IsKeyDown(kKeyRightShift);
}

// ════════════════════════════════════════════════════════════════════
// Ordered IDs (for shift-range select)
// ════════════════════════════════════════════════════════════════════

std::vector<uint64_t> HierarchyPanel::CollectOrderedIds(const std::vector<GameObject *> &roots) const
{
    std::vector<uint64_t> result;
    // Iterative DFS
    std::vector<GameObject *> stack;
    for (auto it = roots.rbegin(); it != roots.rend(); ++it)
        stack.push_back(*it);

    while (!stack.empty()) {
        auto *obj = stack.back();
        stack.pop_back();
        if (!obj || IsHidden(obj->GetID()))
            continue;
        result.push_back(obj->GetID());
        auto &children = obj->GetChildren();
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
            if (!IsHidden(it->get()->GetID()))
                stack.push_back(it->get());
        }
    }
    return result;
}

// ════════════════════════════════════════════════════════════════════
// Drag-drop helpers
// ════════════════════════════════════════════════════════════════════

std::vector<uint64_t> HierarchyPanel::GetDragIds(uint64_t primaryId)
{
    if (m_selIds.count(primaryId) && m_selCount > 1)
        return m_selOrderedIds;
    return {primaryId};
}

std::vector<uint64_t> HierarchyPanel::TopoSortIds(Scene *scene, const std::vector<uint64_t> &ids)
{
    std::unordered_set<uint64_t> idSet(ids.begin(), ids.end());
    std::vector<uint64_t> ordered;
    ordered.reserve(ids.size());

    std::function<void(GameObject *)> walk = [&](GameObject *go) {
        uint64_t gid = go->GetID();
        if (idSet.count(gid)) {
            ordered.push_back(gid);
            idSet.erase(gid);
        }
        for (auto &child : go->GetChildren())
            walk(child.get());
    };

    for (auto &root : scene->GetRootObjects()) {
        walk(root.get());
        if (idSet.empty())
            break;
    }
    Scene *persistentScene = SceneManager::Instance().GetRuntimePersistentScene();
    if (persistentScene && persistentScene != scene) {
        for (const auto &root : persistentScene->GetRootObjects()) {
            walk(root.get());
            if (idSet.empty())
                break;
        }
    }
    // Append any remaining IDs not found in tree
    for (auto id : ids) {
        if (std::find(ordered.begin(), ordered.end(), id) == ordered.end())
            ordered.push_back(id);
    }
    return ordered;
}

bool HierarchyPanel::IsDescendantOf(GameObject *potentialChild, GameObject *potentialParent)
{
    GameObject *cur = potentialChild;
    while (cur) {
        if (cur->GetID() == potentialParent->GetID())
            return true;
        cur = cur->GetParent();
    }
    return false;
}

bool HierarchyPanel::HasUiScreenComponentInSubtree(GameObject *obj) const
{
    if (!obj)
        return false;
    if (goHasUiScreenComponent && goHasUiScreenComponent(obj->GetID()))
        return true;
    for (const auto &child : obj->GetChildren()) {
        if (HasUiScreenComponentInSubtree(child.get()))
            return true;
    }
    return false;
}

bool HierarchyPanel::ValidateReparent(GameObject *obj, uint64_t newParentId, GameObject *newParent)
{
    if (goHasCanvas && goHasCanvas(obj->GetID())) {
        if (showWarning)
            showWarning("Canvas can only be a root object.");
        return false;
    }
    if (HasUiScreenComponentInSubtree(obj)) {
        if (newParent == nullptr || (parentHasCanvasAncestor && !parentHasCanvasAncestor(newParentId))) {
            if (showWarning)
                showWarning("UI components must be placed under a Canvas.");
            return false;
        }
    }
    return true;
}

bool HierarchyPanel::ValidateMoveAdjacent(GameObject *obj, uint64_t newParentId, GameObject *newParent)
{
    const bool isCanvas = goHasCanvas && goHasCanvas(obj->GetID());
    if (isCanvas && newParentId != 0) {
        if (showWarning)
            showWarning("Canvas can only be a root object.");
        return false;
    }
    if (!isCanvas && HasUiScreenComponentInSubtree(obj)) {
        if (newParentId == 0 || (newParent && parentHasCanvasAncestor && !parentHasCanvasAncestor(newParentId))) {
            if (showWarning)
                showWarning("UI components must be placed under a Canvas.");
            return false;
        }
    }
    return true;
}

void HierarchyPanel::ReparentObject(uint64_t draggedId, uint64_t newParentId)
{
    GameObject *newParent = SceneManager::Instance().FindRuntimeObjectByID(newParentId);
    if (!newParent)
        return;
    GameObject *dragged = SceneManager::Instance().FindRuntimeObjectByID(draggedId);
    if (!dragged || !dragged->GetScene())
        return;
    Scene *scene = dragged->GetScene();

    auto dragIds = GetDragIds(draggedId);
    auto sorted = TopoSortIds(scene, dragIds);

    std::vector<uint64_t> validIds;
    for (uint64_t did : sorted) {
        if (did == newParentId)
            continue;
        auto *obj = SceneManager::Instance().FindRuntimeObjectByID(did);
        if (!obj)
            continue;
        if (obj->GetScene() != scene)
            continue;
        if (IsDescendantOf(newParent, obj))
            continue;
        if (!ValidateReparent(obj, newParentId, newParent))
            continue;

        validIds.push_back(did);
    }
    if (!validIds.empty() && ExecuteEditorCommand("scene.move_hierarchy",
                                                  MoveCommandArgument(validIds, "parent", newParentId, false,
                                                                      newParent->GetScene()->GetWorldId()),
                                                  "drag_drop"))
        m_pendingExpandId = newParentId;
}

void HierarchyPanel::MoveObjectAdjacent(uint64_t draggedId, uint64_t targetId, bool after)
{
    auto *targetObj = SceneManager::Instance().FindRuntimeObjectByID(targetId);
    if (!targetObj)
        return;
    GameObject *dragged = SceneManager::Instance().FindRuntimeObjectByID(draggedId);
    if (!dragged || !dragged->GetScene())
        return;
    Scene *scene = dragged->GetScene();

    auto *newParent = targetObj->GetParent();
    uint64_t newParentId = newParent ? newParent->GetID() : 0;

    auto dragIds = GetDragIds(draggedId);
    auto sorted = TopoSortIds(scene, dragIds);

    std::vector<uint64_t> validIds;
    for (uint64_t did : sorted) {
        if (did == targetId)
            continue;
        auto *obj = SceneManager::Instance().FindRuntimeObjectByID(did);
        if (!obj)
            continue;
        if (obj->GetScene() != scene)
            continue;
        if (IsDescendantOf(targetObj, obj))
            continue;
        if (!ValidateMoveAdjacent(obj, newParentId, newParent))
            continue;
        validIds.push_back(did);
    }
    if (validIds.empty())
        return;

    if (ExecuteEditorCommand(
            "scene.move_hierarchy",
            MoveCommandArgument(validIds, "adjacent", targetId, after, targetObj->GetScene()->GetWorldId()),
            "drag_drop") &&
        newParentId != 0)
        m_pendingExpandId = newParentId;
}

void HierarchyPanel::ReparentToRoot(uint64_t draggedId, uint64_t destinationWorldId)
{
    GameObject *dragged = SceneManager::Instance().FindRuntimeObjectByID(draggedId);
    if (!dragged)
        return;
    Scene *scene = dragged->GetScene();

    auto dragIds = GetDragIds(draggedId);
    auto sorted = TopoSortIds(scene, dragIds);

    std::vector<uint64_t> validIds;
    for (uint64_t did : sorted) {
        auto *obj = SceneManager::Instance().FindRuntimeObjectByID(did);
        if (!obj || obj->GetScene() != scene)
            continue;
        const bool isCanvas = goHasCanvas && goHasCanvas(obj->GetID());
        if (!isCanvas && HasUiScreenComponentInSubtree(obj)) {
            if (showWarning)
                showWarning("UI components must be placed under a Canvas.");
            continue;
        }

        validIds.push_back(did);
    }
    if (!validIds.empty()) {
        if (destinationWorldId == 0)
            destinationWorldId = scene->GetWorldId();
        ExecuteEditorCommand("scene.move_hierarchy",
                             MoveCommandArgument(validIds, "root", 0, false, destinationWorldId), "drag_drop");
    }
}

void HierarchyPanel::HandleExternalDrop(const std::string &dropType, uint64_t payload, uint64_t parentId)
{
    // In Prefab Mode, force under prefab root
    if (IsPrefabModeActive() && parentId == 0) {
        Scene *scene = SceneManager::Instance().GetActiveScene();
        if (scene && !scene->GetRootObjects().empty())
            parentId = scene->GetRootObjects()[0]->GetID();
    }

    if (dropType == DRAG_DROP_TYPE) {
        if (parentId == 0)
            ReparentToRoot(payload);
        else
            ReparentObject(payload, parentId);
    }
}

void HierarchyPanel::HandleExternalDropStr(const std::string &dropType, const std::string &payload, uint64_t parentId)
{
    // In Prefab Mode, force under prefab root
    if (IsPrefabModeActive() && parentId == 0) {
        Scene *scene = SceneManager::Instance().GetActiveScene();
        if (scene && !scene->GetRootObjects().empty())
            parentId = scene->GetRootObjects()[0]->GetID();
    }

    if (dropType == "PREFAB_GUID" || dropType == "PREFAB_FILE") {
        const bool isGuid = (dropType == "PREFAB_GUID");
        ExecuteEditorCommand("scene.instantiate_prefab", ExternalDropCommandArgument(payload, parentId, isGuid),
                             "drag_drop");
    } else if (dropType == "MODEL_GUID" || dropType == "MODEL_FILE") {
        const bool isGuid = (dropType == "MODEL_GUID");
        ExecuteEditorCommand("scene.create_model", ExternalDropCommandArgument(payload, parentId, isGuid), "drag_drop");
    } else if (dropType == "SCENE_FILE") {
        ExecuteEditorCommand("scene.open_additive", payload, "drag_drop");
    }
}

// ════════════════════════════════════════════════════════════════════
// Rename
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::BeginRename(uint64_t objId)
{
    auto *obj = SceneManager::Instance().FindRuntimeObjectByID(objId);
    if (!obj)
        return;
    m_renameId = objId;
    std::strncpy(m_renameBuf, obj->GetName().c_str(), sizeof(m_renameBuf) - 1);
    m_renameBuf[sizeof(m_renameBuf) - 1] = '\0';
    m_renameFocus = true;
    m_renameSkipDeactivateFrames = 2;
    BeginTransientInteraction("rename", "inline_rename", 100, [this]() {
        CancelRename();
        return true;
    });
}

void HierarchyPanel::BeginRenameObject(uint64_t objId)
{
    BeginRename(objId);
}

void HierarchyPanel::CommitRename()
{
    if (!m_renameId)
        return;
    std::string newName(m_renameBuf);
    // Trim
    while (!newName.empty() && newName.front() == ' ')
        newName.erase(newName.begin());
    while (!newName.empty() && newName.back() == ' ')
        newName.pop_back();
    if (!newName.empty()) {
        Scene *scene = SceneManager::Instance().GetActiveScene();
        if (scene) {
            auto *obj = SceneManager::Instance().FindRuntimeObjectByID(m_renameId);
            if (obj && obj->GetName() != newName) {
                ExecuteEditorCommand("scene.rename_object", RenameCommandArgument(m_renameId, newName), "inline_edit");
            }
        }
    }
    m_renameId = 0;
    m_renameBuf[0] = '\0';
    m_renameFocus = false;
    m_renameSkipDeactivateFrames = 0;
    EndTransientInteraction("rename");
}

void HierarchyPanel::CancelRename()
{
    m_renameId = 0;
    m_renameBuf[0] = '\0';
    m_renameFocus = false;
    m_renameSkipDeactivateFrames = 0;
    EndTransientInteraction("rename");
}

// ════════════════════════════════════════════════════════════════════
// Reorder separator helper
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::RenderReorderSep(InxGUIContext *ctx, const char *sepId, std::function<void(uint64_t)> onDrop,
                                      float indentPx, bool consumeSpace)
{
    const ImGuiPayload *activePayload = ImGui::GetDragDropPayload();
    if (activePayload == nullptr || !activePayload->IsDataType(DRAG_DROP_TYPE))
        return;

    float savedY = ctx->GetCursorPosY();
    float savedX = ctx->GetCursorPosX();
    float availW = ctx->GetContentRegionAvailWidth();
    if (indentPx > 0.0f) {
        ctx->SetCursorPosX(savedX + indentPx);
        availW = (std::max)(1.0f, availW - indentPx);
    }
    ctx->SetNextItemAllowOverlap();
    const float dpi = ctx->GetDpiScale();
    ctx->SetCursorPosY(savedY - EditorTheme::DND_REORDER_HIT_ABOVE * dpi);
    ctx->InvisibleButton(sepId, availW, EditorTheme::DND_REORDER_SEPARATOR_H * dpi);
    ctx->PushStyleColor(ImGuiCol_DragDropTarget, 0.0f, 0.0f, 0.0f, 0.0f);
    if (ctx->BeginDragDropTarget()) {
        // A filled rectangle produces one exact solid-white line. AddLine is
        // anti-aliased and creates the gray fringe visible around thin lines.
        const float lineY = ctx->GetItemRectMinY() + EditorTheme::DND_REORDER_HIT_ABOVE * dpi;
        float x1 = ctx->GetItemRectMinX();
        float x2 = x1 + availW;
        const float lineH = EditorTheme::DND_REORDER_LINE_THICKNESS * dpi;
        ctx->DrawFilledRect(x1, std::floor(lineY), x2, std::floor(lineY) + lineH, EditorTheme::DND_REORDER_LINE.x,
                            EditorTheme::DND_REORDER_LINE.y, EditorTheme::DND_REORDER_LINE.z,
                            EditorTheme::DND_REORDER_LINE.w, 0.0f);
        uint64_t payload = 0;
        if (ctx->AcceptDragDropPayload(DRAG_DROP_TYPE, &payload)) {
            if (onDrop)
                onDrop(payload);
        }
        ctx->EndDragDropTarget();
    }
    ctx->PopStyleColor(1);
    ctx->SetCursorPosX(savedX);
    if (!consumeSpace)
        ctx->SetCursorPosY(savedY);
}

// ════════════════════════════════════════════════════════════════════
// Multi-drop target helper
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::RenderMultiDropTarget(InxGUIContext *ctx, uint64_t parentId)
{
    if (ImGui::GetDragDropPayload() == nullptr)
        return;

    ctx->PushStyleColor(ImGuiCol_DragDropTarget, 0.0f, 0.0f, 0.0f, 0.0f);
    if (ctx->BeginDragDropTarget()) {
        // Accept HIERARCHY_GAMEOBJECT (uint64_t payload)
        uint64_t payload = 0;
        if (ctx->AcceptDragDropPayload(DRAG_DROP_TYPE, &payload)) {
            HandleExternalDrop(DRAG_DROP_TYPE, payload, parentId);
        }
        // Accept string payloads
        for (const char *dt : {"MODEL_GUID", "MODEL_FILE", "PREFAB_GUID", "PREFAB_FILE", "SCENE_FILE"}) {
            std::string strPayload;
            if (ctx->AcceptDragDropPayload(dt, &strPayload)) {
                HandleExternalDropStr(dt, strPayload, parentId);
                break;
            }
        }
        ctx->EndDragDropTarget();
    }
    ctx->PopStyleColor(1);
}

// ════════════════════════════════════════════════════════════════════
// Context menus
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::RenderItemContextMenu(InxGUIContext *ctx, GameObject *obj)
{
    if (!obj || !renderContextMenu)
        return;
    renderContextMenu(ctx, obj->GetID(), obj->IsPrefabInstance(), obj->GetID());
}

bool HierarchyPanel::ExecuteEditorCommand(const std::string &commandId, const std::string &argument,
                                          const std::string &source) const
{
    return executeCommand && executeCommand(commandId, source, argument);
}

// ════════════════════════════════════════════════════════════════════
// Inline rename rendering
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::RenderRenameInput(InxGUIContext *ctx, GameObject *obj)
{
    if (m_renameFocus) {
        ctx->SetKeyboardFocusHere();
        m_renameFocus = false;
    }

    float availW = ctx->GetContentRegionAvailWidth();
    ctx->SetNextItemWidth(availW);
    const bool submitted =
        ctx->InputTextWithHint("##rename", "", m_renameBuf, sizeof(m_renameBuf), ImGuiInputTextFlags_EnterReturnsTrue);
    ctx->RecordSemanticItem("hierarchy_rename", obj->GetName(), true,
                            "hierarchy.object." + std::to_string(obj->GetID()) + ".rename", std::nullopt, std::nullopt,
                            std::string(m_renameBuf));

    if (m_renameSkipDeactivateFrames > 0)
        --m_renameSkipDeactivateFrames;

    if (submitted) {
        CommitRename();
        return;
    }
    if (m_renameSkipDeactivateFrames == 0 && ctx->IsItemDeactivated())
        CommitRename();
}

// ════════════════════════════════════════════════════════════════════
// Flat item rendering (replaces recursive RenderGameObjectTree for
// the main scrollable body; the old recursive function is kept for
// reference but no longer called from OnRenderContent).
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::MoveSceneAdjacent(uint64_t draggedWorldId, uint64_t targetWorldId, bool after)
{
    if (SceneManager::Instance().MoveSceneAdjacent(draggedWorldId, targetWorldId, after)) {
        m_forceRootRefresh = true;
        m_flatListDirty = true;
    }
}

void HierarchyPanel::RenderSceneHeader(InxGUIContext *ctx, Scene *scene)
{
    if (!scene)
        return;
    const uint64_t worldId = scene->GetWorldId();
    const bool active = SceneManager::Instance().GetActiveScene() == scene;
    ctx->PushID("HierarchyScene_" + std::to_string(worldId));
    // Scene groups are first-class rows.  Keep their appearance separate from
    // GameObject rows: this is the same compact, framed treatment used by
    // Inspector component headers, while the hierarchy tree below remains
    // unchanged.
    const float dpi = ctx->GetDpiScale();
    const bool dragged = m_draggedSceneWorldId == worldId;
    const bool selectedOrDragged = m_selectedSceneWorldId == worldId || dragged;
    const ImVec2 sceneRowMin = ImGui::GetCursorScreenPos();
    const float sceneRowHeight = ImGui::GetFontSize() * EditorTheme::INSPECTOR_HEADER_PRIMARY_FONT_SCALE +
                                 2.0f * EditorTheme::INSPECTOR_HEADER_PRIMARY_FRAME_PAD.y * dpi;
    const ImVec2 sceneRowMax{sceneRowMin.x + ImGui::GetContentRegionAvail().x, sceneRowMin.y + sceneRowHeight};
    const bool dragTargetHovered =
        ImGui::GetDragDropPayload() != nullptr && ImGui::IsMouseHoveringRect(sceneRowMin, sceneRowMax, false);
    const ImVec4 &header = selectedOrDragged   ? EditorTheme::HIERARCHY_ROW_SELECTED
                           : dragTargetHovered ? EditorTheme::HIERARCHY_ROW_HOVER
                                               : EditorTheme::INSPECTOR_HEADER_PRIMARY;
    const ImVec4 &hovered = selectedOrDragged ? EditorTheme::HIERARCHY_ROW_SELECTED : EditorTheme::HIERARCHY_ROW_HOVER;
    const ImVec4 &pressed = selectedOrDragged ? EditorTheme::HIERARCHY_ROW_SELECTED : EditorTheme::HIERARCHY_ROW_HOVER;
    ctx->PushStyleColor(ImGuiCol_Header, header.x, header.y, header.z, header.w);
    ctx->PushStyleColor(ImGuiCol_HeaderHovered, hovered.x, hovered.y, hovered.z, hovered.w);
    ctx->PushStyleColor(ImGuiCol_HeaderActive, pressed.x, pressed.y, pressed.z, pressed.w);
    ctx->PushStyleVarVec2(ImGuiStyleVar_FramePadding, EditorTheme::INSPECTOR_HEADER_PRIMARY_FRAME_PAD.x * dpi,
                          EditorTheme::INSPECTOR_HEADER_PRIMARY_FRAME_PAD.y * dpi);
    ctx->PushStyleVarVec2(ImGuiStyleVar_ItemSpacing, EditorTheme::INSPECTOR_HEADER_ITEM_SPC.x * dpi,
                          EditorTheme::INSPECTOR_HEADER_ITEM_SPC.y * dpi);
    ctx->PushStyleVarFloat(ImGuiStyleVar_FrameBorderSize, EditorTheme::INSPECTOR_HEADER_BORDER_SIZE * dpi);
    ImGui::SetWindowFontScale(EditorTheme::INSPECTOR_HEADER_PRIMARY_FONT_SCALE);

    const bool wasOpen = m_collapsedSceneWorldIds.count(worldId) == 0;
    ctx->SetNextItemOpen(wasOpen, ImGuiCond_Always);
    std::string label = scene->GetName();
    if (isSceneDirty && isSceneDirty(worldId))
        label += " *";
    label += "###HierarchySceneHeader";
    // Unity semantics: the header body selects the scene; only the disclosure
    // arrow changes the expanded state.
    const bool isOpen =
        ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow);
    const bool leftClicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
    if (leftClicked) {
        m_pendingSceneSelectWorldId = worldId;
        ExecuteEditorCommand("scene.set_active", std::to_string(worldId), "pointer");
    }
    if (isOpen != wasOpen) {
        if (isOpen)
            m_collapsedSceneWorldIds.erase(worldId);
        else
            m_collapsedSceneWorldIds.insert(worldId);
        m_flatListDirty = true;
    }

    if (ctx->BeginDragDropSource(0)) {
        m_pendingSceneSelectWorldId = 0;
        m_draggedSceneWorldId = worldId;
        ctx->SetDragDropPayload(SCENE_DRAG_DROP_TYPE, worldId);
        ctx->Label(scene->GetName());
        ctx->EndDragDropSource();
    }

    ctx->PushStyleColor(ImGuiCol_DragDropTarget, 0.0f, 0.0f, 0.0f, 0.0f);
    if (ctx->BeginDragDropTarget()) {
        const ImGuiPayload *payloadType = ImGui::GetDragDropPayload();
        if (payloadType && payloadType->IsDataType(SCENE_DRAG_DROP_TYPE)) {
            const bool after = ImGui::GetMousePos().y >= (ctx->GetItemRectMinY() + ctx->GetItemRectMaxY()) * 0.5f;
            const float lineY = after ? ctx->GetItemRectMaxY() : ctx->GetItemRectMinY();
            ctx->DrawFilledRect(ctx->GetItemRectMinX(), std::floor(lineY), ctx->GetItemRectMaxX(),
                                std::floor(lineY) + EditorTheme::DND_REORDER_LINE_THICKNESS * dpi,
                                EditorTheme::DND_REORDER_LINE.x, EditorTheme::DND_REORDER_LINE.y,
                                EditorTheme::DND_REORDER_LINE.z, EditorTheme::DND_REORDER_LINE.w, 0.0f);
            uint64_t draggedWorldId = 0;
            if (ctx->AcceptDragDropPayload(SCENE_DRAG_DROP_TYPE, &draggedWorldId))
                MoveSceneAdjacent(draggedWorldId, worldId, after);
        } else {
            uint64_t payload = 0;
            if (ctx->AcceptDragDropPayload(DRAG_DROP_TYPE, &payload))
                ReparentToRoot(payload, worldId);
        }
        ctx->EndDragDropTarget();
    }
    ctx->PopStyleColor(1);
    if (ctx->BeginPopupContextItem("##HierarchySceneContext", 1)) {
        if (ctx->MenuItem(Tr("hierarchy.scene.set_active").c_str(), "", active, true))
            ExecuteEditorCommand("scene.set_active", std::to_string(worldId), "context_menu");
        if (ctx->MenuItem(Tr("hierarchy.scene.save").c_str(), "Ctrl+S", false, true))
            ExecuteEditorCommand("scene.save", std::to_string(worldId), "context_menu");
        ctx->Separator();
        if (ctx->MenuItem(Tr("hierarchy.scene.unload").c_str(), "", false, true))
            ExecuteEditorCommand("scene.unload", std::to_string(worldId), "context_menu");
        ctx->EndPopup();
    }
    if (InxGUISemantics::IsCaptureEnabled())
        ctx->RecordSemanticItem("hierarchy_scene", scene->GetName(), true, "hierarchy.scene." + std::to_string(worldId),
                                active);
    ImGui::SetWindowFontScale(1.0f);
    ctx->PopID();
    ctx->PopStyleVar(3);
    ctx->PopStyleColor(3);
}

void HierarchyPanel::RenderFlatItem(InxGUIContext *ctx, const FlatItem &item, float baseIndentX, float indentStep)
{
    GameObject *obj = item.obj;
    if (!obj)
        return;

    uint64_t objId = obj->GetID();
    ctx->PushID(static_cast<int>(objId & 0x7FFFFFFF));

    // ── Inline rename mode ──────────────────────────────────────
    if (m_renameId == objId) {
        float indentPx = static_cast<float>(item.depth) * indentStep;
        if (indentPx > 0)
            ImGui::Indent(indentPx);
        RenderRenameInput(ctx, obj);
        if (indentPx > 0)
            ImGui::Unindent(indentPx);
        ctx->PopID();
        return;
    }

    // Tree node flags — always use NoTreePushOnOpen so no TreePop needed
    int nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth |
                    ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_NoTreePushOnOpen;

    const bool selectedOrDragged = m_selIds.count(objId) != 0 || m_draggedObjectId == objId;
    if (selectedOrDragged)
        nodeFlags |= ImGuiTreeNodeFlags_Selected;

    bool isLeaf = !item.hasVisibleChildren;
    if (isLeaf)
        nodeFlags |= ImGuiTreeNodeFlags_Leaf;

    // Keep our stable object-ID set authoritative. ImGui normally owns tree
    // state by widget ID; rows disappearing during reparenting or changing
    // their visible labels during rename must not reset that state.
    const bool requestedOpen = HasActiveSearch() || m_treeProjection.IsExpanded(objId);
    ctx->SetNextItemOpen(requestedOpen, ImGuiCond_Always);
    m_forceExpandIds.erase(objId);

    // Display name with prefab decoration
    bool isPrefab = obj->IsPrefabInstance();
    const std::string &objectName = obj->GetName();
    const std::string *displayName = &objectName;
    std::string prefabDisplayName;
    if (isPrefab) {
        prefabDisplayName.reserve(objectName.size() + sizeof(EditorTheme::PREFAB_ICON) + 1);
        prefabDisplayName = EditorTheme::PREFAB_ICON;
        prefabDisplayName += " ";
        prefabDisplayName += objectName;
        displayName = &prefabDisplayName;
    }

    const bool inactiveDimmed = !obj->IsActiveInHierarchy();
    int textColorPushed = 0;
    if (inactiveDimmed) {
        ctx->PushStyleColor(ImGuiCol_Text, EditorTheme::TEXT_DISABLED.x, EditorTheme::TEXT_DISABLED.y,
                            EditorTheme::TEXT_DISABLED.z, EditorTheme::TEXT_DISABLED.w);
        textColorPushed = 1;
    } else if (isPrefab) {
        ctx->PushStyleColor(ImGuiCol_Text, EditorTheme::PREFAB_TEXT.x, EditorTheme::PREFAB_TEXT.y,
                            EditorTheme::PREFAB_TEXT.z, EditorTheme::PREFAB_TEXT.w);
        textColorPushed = 1;
    }

    // Manual indentation for flat rendering
    float indentPx = static_cast<float>(item.depth) * indentStep;
    if (indentPx > 0)
        ImGui::Indent(indentPx);

    std::string stableLabel;
    stableLabel.reserve(displayName->size() + 40);
    stableLabel = *displayName;
    stableLabel += "###HierarchyObject_";
    stableLabel += std::to_string(objId);
    // A drag blocks ImGui's ordinary hover detection. Resolve the prospective
    // parent under the pointer before submitting the row and feed it through
    // the exact same Header color as an ordinary mouse hover. Drawing a drop
    // overlay afterward would tint or cover the row text.
    const ImVec2 rowMin = ImGui::GetCursorScreenPos();
    const ImVec2 rowMax{rowMin.x + ImGui::GetContentRegionAvail().x, rowMin.y + ImGui::GetFrameHeight()};
    const bool dragBodyHovered =
        ImGui::GetDragDropPayload() != nullptr && ImGui::IsMouseHoveringRect(rowMin, rowMax, false);
    if (dragBodyHovered && !selectedOrDragged)
        nodeFlags |= ImGuiTreeNodeFlags_Selected;
    const ImVec4 &rowBase = selectedOrDragged ? EditorTheme::HIERARCHY_ROW_SELECTED
                            : dragBodyHovered ? EditorTheme::HIERARCHY_ROW_HOVER
                                              : EditorTheme::ROW_NONE;
    const ImVec4 &rowHover = selectedOrDragged ? EditorTheme::HIERARCHY_ROW_SELECTED : EditorTheme::HIERARCHY_ROW_HOVER;
    ctx->PushStyleColor(ImGuiCol_Header, rowBase.x, rowBase.y, rowBase.z, rowBase.w);
    ctx->PushStyleColor(ImGuiCol_HeaderHovered, rowHover.x, rowHover.y, rowHover.z, rowHover.w);
    ctx->PushStyleColor(ImGuiCol_HeaderActive, rowHover.x, rowHover.y, rowHover.z, rowHover.w);
    bool isOpen = ctx->TreeNodeEx(stableLabel, nodeFlags);
    ctx->PopStyleColor(3);
    if (InxGUISemantics::IsCaptureEnabled())
        ctx->RecordSemanticItem("hierarchy_object", objectName, true, "hierarchy.object." + std::to_string(objId),
                                m_selIds.count(objId) > 0);

    if (indentPx > 0)
        ImGui::Unindent(indentPx);

    if (textColorPushed)
        ctx->PopStyleColor(1);

    // Search forces matching branches open only for presentation. Preserve the
    // user's pre-search expansion choices until normal tree interaction resumes.
    const bool toggledOpen = !isLeaf && ImGui::IsItemToggledOpen();
    if (!HasActiveSearch() && toggledOpen)
        ExecuteEditorCommand("hierarchy.set_expanded", TreeExpandedCommandArgument(objId, isOpen), "pointer");

    // ── Selection ───────────────────────────────────────────────
    if (ctx->IsItemClicked(0)) {
        m_selectedSceneWorldId = 0;
        m_pendingSceneSelectWorldId = 0;
        if (m_renameId && m_renameId != objId)
            CancelRename();
        m_pendingSelectId = objId;
        m_pendingCtrl = IsCtrl(ctx);
        m_pendingShift = IsShift(ctx);
    }
    if (ctx->IsItemClicked(1)) {
        m_selectedSceneWorldId = 0;
        m_pendingSceneSelectWorldId = 0;
        if (!m_selIds.count(objId)) {
            if (selectId)
                selectId(objId);
            SyncSelectionCache();
        }
        m_rightClickedObjId = objId;
        // The shared popup is rendered after the flat rows, outside this
        // object's ID scope. Open it in that same scope so ImGui hashes the
        // popup ID consistently.
        ctx->PopID();
        ctx->OpenPopup("##HierarchyItemContext");
        ctx->PushID(std::to_string(objId));
    }

    // Double-click frame selected through the shared editor command path.
    // A rapid expand/collapse gesture on the disclosure arrow is not a row
    // activation. Keep it from leaking into the double-click frame command.
    if (!toggledOpen && ctx->IsMouseDoubleClicked(0) && ctx->IsItemHovered()) {
        ExecuteEditorCommand("scene.frame_selected", std::to_string(objId), "pointer");
    }

    // ── Drag source ─────────────────────────────────────────────
    if (ctx->BeginDragDropSource(0)) {
        // Unity selects an unselected row when its drag actually begins. This
        // clears a stale selection on the prospective parent, so that row can
        // display the ordinary neutral hover while the source owns the muted
        // accent selection.
        if (m_selIds.count(objId) == 0) {
            m_selectedSceneWorldId = 0;
            m_pendingSceneSelectWorldId = 0;
            if (selectId)
                selectId(objId);
            SyncSelectionCache();
        }
        m_draggedObjectId = objId;
        ctx->SetDragDropPayload(DRAG_DROP_TYPE, objId);
        int n = m_selIds.count(objId) ? m_selCount : 1;
        if (n > 1)
            ctx->Label(obj->GetName() + " (+" + std::to_string(n - 1) + ")");
        else
            ctx->Label(obj->GetName());
        ctx->EndDragDropSource();
    }

    // ── Drop target on body → reparent as child ─────────────────
    RenderMultiDropTarget(ctx, objId);

    ctx->PopID();
}

// ════════════════════════════════════════════════════════════════════
// VisiblePreRender — keyboard shortcuts + deferred selection
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::VisiblePreRender(InxGUIContext *ctx)
{
    using Clock = std::chrono::high_resolution_clock;
    auto msSince = [](const Clock::time_point &start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };

    // Refresh hidden IDs
    auto preHiddenStart = Clock::now();
    if (!m_runtimeHiddenPushMode && getRuntimeHiddenIds)
        m_hiddenIds = getRuntimeHiddenIds();
    else if (!m_runtimeHiddenPushMode)
        m_hiddenIds.clear();
    m_subPreHidden += msSince(preHiddenStart);

    // Sync selection once per frame
    auto preSelectionStart = Clock::now();
    SyncSelectionCache();
    if (m_selPrimary != m_lastObservedPrimaryId) {
        m_lastObservedPrimaryId = m_selPrimary;
        m_scrollToObjectId = m_selPrimary;
        if (m_selPrimary) {
            ExpandToObject(m_selPrimary);
            const bool selectionMissingFromCache =
                std::none_of(m_flatItems.begin(), m_flatItems.end(),
                             [this](const FlatItem &item) { return item.obj && item.obj->GetID() == m_selPrimary; });
            m_forceRootRefresh = selectionMissingFromCache;
        }
    }
    m_subPreSelection += msSince(preSelectionStart);

    // Deferred left-click selection
    auto pendingStart = Clock::now();
    if (m_pendingSelectId != 0) {
        if (!ctx->IsMouseButtonDown(0)) {
            if (!ctx->IsMouseDragging(0)) {
                uint64_t pid = m_pendingSelectId;

                if (m_pendingCtrl) {
                    if (toggleId)
                        toggleId(pid);
                } else if (m_pendingShift) {
                    Scene *scene = SceneManager::Instance().GetActiveScene();
                    if (scene) {
                        if (m_orderedIdsDirty) {
                            m_cachedOrderedIds = CollectOrderedIds(m_cachedRoots);
                            m_orderedIdsDirty = false;
                        }
                        auto searchFiltered =
                            HasActiveSearch() ? CollectOrderedIds(FilterForSearch(m_cachedRoots)) : m_cachedOrderedIds;
                        if (setOrderedIds)
                            setOrderedIds(searchFiltered);
                    }
                    if (rangeSelectId)
                        rangeSelectId(pid);
                } else {
                    if (selectId)
                        selectId(pid);
                }
                SyncSelectionCache();
            }
            m_pendingSelectId = 0;
            m_pendingCtrl = false;
            m_pendingShift = false;
        } else if (ctx->IsMouseDragging(0)) {
            m_pendingSelectId = 0;
            m_pendingCtrl = false;
            m_pendingShift = false;
        }
    }

    // Scene Header selection is also release-driven so beginning a drag does
    // not leave the header selected after the reorder gesture completes.
    if (m_pendingSceneSelectWorldId != 0) {
        if (!ctx->IsMouseButtonDown(0)) {
            if (!ctx->IsMouseDragging(0)) {
                const uint64_t worldId = m_pendingSceneSelectWorldId;
                ClearSelectionAndNotify();
                m_selectedSceneWorldId = worldId;
            }
            m_pendingSceneSelectWorldId = 0;
        } else if (ctx->IsMouseDragging(0)) {
            m_pendingSceneSelectWorldId = 0;
        }
    }
    m_subPrePendingSelect += msSince(pendingStart);
}

// ════════════════════════════════════════════════════════════════════
// OnRenderContent — the main hierarchy body
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::OnRenderContent(InxGUIContext *ctx)
{
    using Clock = std::chrono::high_resolution_clock;
    auto msSince = [](const Clock::time_point &start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };
    const float dpi = ctx->GetDpiScale();
    const ImGuiPayload *framePayload = ImGui::GetDragDropPayload();
    if (framePayload == nullptr) {
        m_draggedObjectId = 0;
        m_draggedSceneWorldId = 0;
    }
    if (std::abs(dpi - m_lastDpiScale) >= 0.01f) {
        m_lastDpiScale = dpi;
        m_cachedItemHeight = 18.0f * dpi;
        m_itemHeightMeasured = false;
    }

    // ── Prefab mode header (the scene is represented by its own group row)
    auto headerStart = Clock::now();
    if (IsPrefabModeActive()) {
        std::string prefabName = PrefabDisplayName();
        ctx->PushStyleColor(ImGuiCol_Text, EditorTheme::PREFAB_TEXT.x, EditorTheme::PREFAB_TEXT.y,
                            EditorTheme::PREFAB_TEXT.z, EditorTheme::PREFAB_TEXT.w);
        ctx->Label(prefabName);
        ctx->PopStyleColor(1);
    }
    m_subHeader += msSince(headerStart);

    // ── Search bar ──────────────────────────────────────────────
    auto searchStart = Clock::now();
    ctx->SetNextItemWidth(ctx->GetContentRegionAvailWidth());
    if (m_focusSearchNextFrame) {
        ctx->SetKeyboardFocusHere();
        m_focusSearchNextFrame = false;
    }
    std::strncpy(m_searchBuf, m_search.Query().c_str(), sizeof(m_searchBuf) - 1);
    m_searchBuf[sizeof(m_searchBuf) - 1] = '\0';
    ctx->InputTextWithHint("##HierarchySearch", Tr("hierarchy.search_placeholder").c_str(), m_searchBuf,
                           sizeof(m_searchBuf), 0);
    if (InxGUISemantics::IsCaptureEnabled())
        ctx->RecordSemanticItem("hierarchy_search", Tr("hierarchy.search_placeholder"), true, "hierarchy.search");
    SetSearchQuery(m_searchBuf);

    ctx->Separator();
    m_subSearch += msSince(searchStart);

    // ── Scene tree ──────────────────────────────────────────────
    Scene *scene = SceneManager::Instance().GetActiveScene();
    if (scene) {
        ctx->PushStyleVarVec2(ImGuiStyleVar_ItemSpacing, EditorTheme::TREE_ITEM_SPC.x * dpi,
                              EditorTheme::TREE_ITEM_SPC.y * dpi);
        ctx->PushStyleVarVec2(ImGuiStyleVar_FramePadding, EditorTheme::TREE_FRAME_PAD.x * dpi,
                              EditorTheme::TREE_FRAME_PAD.y * dpi);
        ctx->PushStyleVarFloat(ImGuiStyleVar_IndentSpacing, EditorTheme::TREE_INDENT * dpi);

        bool allowStale =
            !m_forceRootRefresh && !ctx->IsWindowFocused(0) && !ctx->IsWindowHovered() && !m_cachedRoots.empty();
        {
            auto t0 = Clock::now();
            RefreshRootObjects(scene, allowStale, m_forceRootRefresh);
            m_forceRootRefresh = false;
            m_subRefreshRoots += msSince(t0);
        }

        // Apply the pending expansion requested by the latest hierarchy action.
        if (m_pendingExpandId) {
            m_treeProjection.SetExpanded(m_pendingExpandId, true);
            m_forceExpandIds.insert(m_pendingExpandId);
            m_pendingExpandId = 0;
            m_flatListDirty = true;
        }

        // Use cachedRoots directly when no search is active to avoid O(n) copy
        const std::vector<GameObject *> *pVisibleRoots = &m_cachedRoots;
        std::vector<GameObject *> filteredRoots;
        if (HasActiveSearch()) {
            auto t0 = Clock::now();
            filteredRoots = FilterForSearch(m_cachedRoots);
            m_subFilterRoots += msSince(t0);
            pVisibleRoots = &filteredRoots;
        }
        const auto &visibleRoots = *pVisibleRoots;
        int nRoots = static_cast<int>(visibleRoots.size());

        // Build flat list of all visible items (roots + expanded children)
        // Only rebuild when structure, search, or expand state changes
        {
            auto t0 = Clock::now();
            RebuildFlatListIfNeeded(visibleRoots);
            m_subFlatBuild += msSince(t0);
        }

        // A live, explicitly selected root must never disappear from the
        // Hierarchy because a stale structure or runtime-hidden snapshot still
        // carries the same recycled object ID. This path is intentionally
        // exceptional and O(n); the normal cached list remains untouched.
        if (!HasActiveSearch() && m_selPrimary != 0) {
            const bool selectedVisible =
                std::any_of(m_flatItems.begin(), m_flatItems.end(),
                            [this](const FlatItem &item) { return item.obj && item.obj->GetID() == m_selPrimary; });
            if (!selectedVisible) {
                GameObject *selected = SceneManager::Instance().FindRuntimeObjectByID(m_selPrimary);
                if (selected && selected->GetParent() == nullptr) {
                    const bool hasVisibleChildren =
                        std::any_of(selected->GetChildren().begin(), selected->GetChildren().end(),
                                    [this](const auto &child) { return child && !IsHidden(child->GetID()); });
                    m_flatItems.push_back(
                        {selected, IsPrefabModeActive() ? 0 : 1, hasVisibleChildren, selected->GetScene(), false});
                }
            }
        }
        int nItems = static_cast<int>(m_flatItems.size());

        const ImGuiPayload *activePayload = ImGui::GetDragDropPayload();
        const bool hasObjectDrag = activePayload && activePayload->IsDataType(DRAG_DROP_TYPE);
        const bool hasAnyDrag = activePayload != nullptr;
        // Prefab mode has no Scene Header, so its first-root insertion target
        // remains at the top of the flat tree.
        if (hasObjectDrag && IsPrefabModeActive()) {
            if (nRoots > 0) {
                uint64_t firstRootId = visibleRoots[0]->GetID();
                RenderReorderSep(ctx, "##sep_before_first_root", [this, firstRootId](uint64_t payload) {
                    MoveObjectAdjacent(payload, firstRootId, false);
                });
            } else {
                RenderReorderSep(ctx, "##sep_empty_root", [this](uint64_t payload) { ReparentToRoot(payload); });
            }
        }

        // ── Flat virtual scrolling ──────────────────────────────
        if (nItems > 0) {
            auto rowsStart = Clock::now();
            float availW = ctx->GetContentRegionAvailWidth();
            float scrollY = ctx->GetScrollY();
            float viewportH = ctx->GetContentRegionAvailHeight();
            if (viewportH <= 0)
                viewportH = 400.0f * dpi;
            float startY = ctx->GetCursorPosY();
            float itemH = m_cachedItemHeight;
            float indentStep = EditorTheme::TREE_INDENT * dpi;

            // Selection can be changed by creation services, undo/redo, scene
            // picking, or another panel. Ensure a newly selected row is not
            // discarded by virtual scrolling when it lies outside the current
            // viewport. Keep this one-shot so normal hierarchy scrolling is
            // never pulled back to an old selection.
            if (m_scrollToObjectId != 0) {
                auto selectedIt = std::find_if(m_flatItems.begin(), m_flatItems.end(), [this](const FlatItem &item) {
                    return item.obj && item.obj->GetID() == m_scrollToObjectId;
                });
                if (selectedIt != m_flatItems.end()) {
                    const int selectedIndex = static_cast<int>(std::distance(m_flatItems.begin(), selectedIt));
                    const float rowTop = startY + static_cast<float>(selectedIndex) * itemH;
                    const float rowBottom = rowTop + itemH;
                    if (rowTop < scrollY || rowBottom > scrollY + viewportH) {
                        const float targetY = (std::max)(0.0f, rowTop - (viewportH - itemH) * 0.5f);
                        ImGui::SetScrollY(targetY);
                        scrollY = targetY;
                    }
                    m_scrollToObjectId = 0;
                } else if (!HasActiveSearch()) {
                    // The object no longer exists or is hidden by a collapsed
                    // branch that could not be expanded.
                    m_scrollToObjectId = 0;
                }
            }

            // ImGui may preserve the previous scroll offset for one frame after
            // a hierarchy shrinks. Clamp both ends to the current flat list so
            // that the final object is still rendered instead of producing an
            // empty clipped range at the bottom.
            int firstVis = (std::max)(0, static_cast<int>((scrollY - startY) / itemH) - 2);
            firstVis = (std::min)(nItems - 1, firstVis);
            int lastVis = (std::min)(nItems - 1, static_cast<int>((scrollY + viewportH - startY) / itemH) + 5);
            lastVis = (std::max)(firstVis, lastVis);

            if (firstVis > 0)
                ctx->Dummy(availW, static_cast<float>(firstVis) * itemH);

            float baseIndentX = ctx->GetCursorPosX();
            const float baseIndentScreenX = ImGui::GetCursorScreenPos().x;
            for (int i = firstVis; i <= lastVis; i++) {
                float beforeY = ctx->GetCursorPosY();
                const FlatItem &currentItem = m_flatItems[i];

                if (currentItem.sceneHeader) {
                    RenderSceneHeader(ctx, currentItem.scene);
                    if (hasObjectDrag && i + 1 < nItems) {
                        const bool firstRootVisible = i + 1 < nItems && !m_flatItems[i + 1].sceneHeader &&
                                                      m_flatItems[i + 1].scene == currentItem.scene;
                        const uint64_t destinationWorldId = currentItem.scene->GetWorldId();
                        if (firstRootVisible) {
                            const uint64_t firstRootId = m_flatItems[i + 1].obj->GetID();
                            const float rootIndent = static_cast<float>(m_flatItems[i + 1].depth) * indentStep;
                            const std::string separatorId = "##scene_first_root_" + std::to_string(destinationWorldId);
                            RenderReorderSep(
                                ctx, separatorId.c_str(),
                                [this, firstRootId](uint64_t payload) {
                                    MoveObjectAdjacent(payload, firstRootId, false);
                                },
                                rootIndent);
                        } else {
                            const std::string separatorId = "##scene_empty_root_" + std::to_string(destinationWorldId);
                            RenderReorderSep(
                                ctx, separatorId.c_str(),
                                [this, destinationWorldId](uint64_t payload) {
                                    ReparentToRoot(payload, destinationWorldId);
                                },
                                indentStep);
                        }
                    }
                    continue;
                }

                RenderFlatItem(ctx, currentItem, baseIndentX, indentStep);

                // One visual boundary owns one insertion target. If the next
                // row is a child, this object's "after" location belongs below
                // that complete subtree and is intentionally not drawn here.
                // At a closing subtree boundary, horizontal pointer position
                // selects the desired ancestor depth, yielding exactly one
                // line aligned to the intended parent level.
                if (hasObjectDrag && i + 1 < nItems) {
                    const int rootDepth = IsPrefabModeActive() ? 0 : 1;
                    int minimumDepth = rootDepth;
                    if (i + 1 < nItems && !m_flatItems[i + 1].sceneHeader &&
                        m_flatItems[i + 1].scene == currentItem.scene)
                        minimumDepth = m_flatItems[i + 1].depth;

                    if (minimumDepth <= currentItem.depth) {
                        const float relativeMouseX = ImGui::GetMousePos().x - baseIndentScreenX;
                        const int pointerDepth = static_cast<int>(std::floor(relativeMouseX / indentStep + 0.5f));
                        const int desiredDepth = (std::max)(minimumDepth, (std::min)(currentItem.depth, pointerDepth));
                        GameObject *afterObject = currentItem.obj;
                        int afterDepth = currentItem.depth;
                        while (afterObject && afterDepth > desiredDepth) {
                            afterObject = afterObject->GetParent();
                            --afterDepth;
                        }
                        if (afterObject) {
                            const uint64_t afterObjectId = afterObject->GetID();
                            const std::string separatorId =
                                "##boundary_after_" + std::to_string(currentItem.obj->GetID());
                            RenderReorderSep(
                                ctx, separatorId.c_str(),
                                [this, afterObjectId](uint64_t payload) {
                                    MoveObjectAdjacent(payload, afterObjectId, true);
                                },
                                static_cast<float>(desiredDepth) * indentStep);
                        }
                    }
                }

                float afterY = ctx->GetCursorPosY();
                float actualH = afterY - beforeY;
                if (!hasAnyDrag && actualH > 1.0f && !m_itemHeightMeasured) {
                    m_cachedItemHeight = actualH;
                    itemH = actualH;
                    m_itemHeightMeasured = true;
                }
            }

            int remaining = nItems - lastVis - 1;
            if (remaining > 0)
                ctx->Dummy(availW, static_cast<float>(remaining) * itemH);
            m_subRows += msSince(rowsStart);
        }

        // Unity-style runtime residency is a real, separate Scene. Present it
        // after all authored Scene groups.
        Scene *persistentScene = SceneManager::Instance().GetRuntimePersistentScene();
        if (persistentScene && !persistentScene->GetRootObjects().empty()) {
            std::vector<GameObject *> persistentRoots = FilterHidden(persistentScene->GetRootObjects());
            if (HasActiveSearch())
                persistentRoots = FilterForSearch(persistentRoots);
            if (!persistentRoots.empty()) {
                ctx->Separator();
                ctx->PushStyleColor(ImGuiCol_Text, EditorTheme::TEXT_DISABLED.x, EditorTheme::TEXT_DISABLED.y,
                                    EditorTheme::TEXT_DISABLED.z, EditorTheme::TEXT_DISABLED.w);
                ctx->Label("DontDestroyOnLoad");
                ctx->PopStyleColor(1);

                std::vector<FlatItem> persistentItems;
                persistentItems.reserve(persistentRoots.size() * 2);
                for (GameObject *root : persistentRoots)
                    BuildFlatListRecurse(root, 0, persistentItems);
                const float baseIndentX = ctx->GetCursorPosX();
                for (const FlatItem &item : persistentItems)
                    RenderFlatItem(ctx, item, baseIndentX, EditorTheme::TREE_INDENT * dpi);
            }
        }

        auto popupStart = Clock::now();
        if (ctx->BeginPopup("##HierarchyItemContext")) {
            GameObject *popupObj = SceneManager::Instance().FindRuntimeObjectByID(m_rightClickedObjId);
            if (popupObj)
                RenderItemContextMenu(ctx, popupObj);
            else
                m_rightClickedObjId = 0;
            ctx->EndPopup();
        } else if (!ImGui::IsPopupOpen("##HierarchyItemContext")) {
            m_rightClickedObjId = 0;
        }
        m_subPopup += msSince(popupStart);

        // ── Tail drop zone ──────────────────────────────────────
        auto tailDropStart = Clock::now();
        bool tailContextMenuRequested = false;
        float remainingH = ctx->GetContentRegionAvailHeight();
        if (remainingH > 4.0f * dpi) {
            float tailW = ctx->GetContentRegionAvailWidth();
            ctx->InvisibleButton("##drop_to_root_tail", tailW, remainingH);
            if (InxGUISemantics::IsCaptureEnabled())
                ctx->RecordSemanticItem("hierarchy_background", "Hierarchy Background", true, "hierarchy.background");

            if (ctx->IsItemClicked(0)) {
                CancelRename();
                m_selectedSceneWorldId = 0;
                m_pendingSceneSelectWorldId = 0;
                ClearSelectionAndNotify();
            }
            if (ctx->IsItemClicked(1))
                tailContextMenuRequested = true;

            // Drop target with top-edge line
            ctx->PushStyleColor(ImGuiCol_DragDropTarget, 0.0f, 0.0f, 0.0f, 0.0f);
            if (ctx->BeginDragDropTarget()) {
                float lineY = ctx->GetItemRectMinY();
                const float tailMinX = ctx->GetItemRectMinX();
                float lineX1 = tailMinX;
                float lineX2 = tailMinX + tailW;
                GameObject *tailAfterObject = nullptr;
                const ImGuiPayload *tailPayload = ImGui::GetDragDropPayload();
                if (tailPayload && tailPayload->IsDataType(DRAG_DROP_TYPE) && nItems > 0 &&
                    !m_flatItems.back().sceneHeader && m_flatItems.back().obj) {
                    const FlatItem &lastItem = m_flatItems.back();
                    const int rootDepth = IsPrefabModeActive() ? 0 : 1;
                    const int pointerDepth = static_cast<int>(
                        std::floor((ImGui::GetMousePos().x - tailMinX) / (EditorTheme::TREE_INDENT * dpi) + 0.5f));
                    const int desiredDepth = (std::max)(rootDepth, (std::min)(lastItem.depth, pointerDepth));
                    tailAfterObject = lastItem.obj;
                    int afterDepth = lastItem.depth;
                    while (tailAfterObject && afterDepth > desiredDepth) {
                        tailAfterObject = tailAfterObject->GetParent();
                        --afterDepth;
                    }
                    lineX1 += static_cast<float>(desiredDepth) * EditorTheme::TREE_INDENT * dpi;
                }
                ctx->DrawFilledRect(lineX1, std::floor(lineY), lineX2,
                                    std::floor(lineY) + EditorTheme::DND_REORDER_LINE_THICKNESS * dpi,
                                    EditorTheme::DND_REORDER_LINE.x, EditorTheme::DND_REORDER_LINE.y,
                                    EditorTheme::DND_REORDER_LINE.z, EditorTheme::DND_REORDER_LINE.w, 0.0f);
                // Accept uint64_t payload
                uint64_t payload = 0;
                bool accepted = false;
                if (ctx->AcceptDragDropPayload(DRAG_DROP_TYPE, &payload)) {
                    if (tailAfterObject)
                        MoveObjectAdjacent(payload, tailAfterObject->GetID(), true);
                    else {
                        const uint64_t lastWorldId = m_cachedScenes.empty() ? 0 : m_cachedScenes.back()->GetWorldId();
                        ReparentToRoot(payload, lastWorldId);
                    }
                    accepted = true;
                }
                uint64_t draggedWorldId = 0;
                if (!accepted && ctx->AcceptDragDropPayload(SCENE_DRAG_DROP_TYPE, &draggedWorldId)) {
                    if (!m_cachedScenes.empty())
                        MoveSceneAdjacent(draggedWorldId, m_cachedScenes.back()->GetWorldId(), true);
                    accepted = true;
                }
                if (!accepted) {
                    for (const char *dt : {"MODEL_GUID", "MODEL_FILE", "PREFAB_GUID", "PREFAB_FILE", "SCENE_FILE"}) {
                        std::string strPayload;
                        if (ctx->AcceptDragDropPayload(dt, &strPayload)) {
                            HandleExternalDropStr(dt, strPayload, 0);
                            break;
                        }
                    }
                }
                ctx->EndDragDropTarget();
            }
            ctx->PopStyleColor(1);
        }

        // The tail is an InvisibleButton so it can accept root-level drops.
        // BeginPopupContextWindow(..., noOpenOverItems=true) intentionally
        // ignores it; forward its right click to a dedicated background popup.
        if (tailContextMenuRequested && m_rightClickedObjId == 0)
            ctx->OpenPopup("##HierarchyBackgroundContext");

        // Fallback: deselect when clicking the scrollable background
        // (the tail InvisibleButton only works when remainingH > 4)
        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsAnyItemHovered()) {
            CancelRename();
            m_selectedSceneWorldId = 0;
            m_pendingSceneSelectWorldId = 0;
            ClearSelectionAndNotify();
        }
        m_subTailDrop += msSince(tailDropStart);

        ctx->PopStyleVar(3); // IndentSpacing + FramePadding + ItemSpacing

        if (HasActiveSearch() && nItems == 0)
            ctx->Label(Tr("hierarchy.no_search_results"));
    }

    // ── Parent for background-menu creations ─────────────────────
    // A blank-area context menu creates a root object. Creating a child is
    // deliberately reserved for an object's own context menu, otherwise a
    // selected object silently turns the next root creation into a collapsed
    // child and makes it appear to disappear from the Hierarchy.
    uint64_t parentIdForNew = 0;
    if (IsPrefabModeActive()) {
        Scene *pscene = SceneManager::Instance().GetActiveScene();
        if (pscene && !pscene->GetRootObjects().empty())
            parentIdForNew = pscene->GetRootObjects()[0]->GetID();
    }

    // ── Background context menu ─────────────────────────────────
    // An object row opens the shared item popup earlier in this frame. Do not
    // also let the window-level popup consume that same right click.
    bool backgroundContextOpen = false;
    if (m_rightClickedObjId == 0) {
        backgroundContextOpen = ctx->BeginPopup("##HierarchyBackgroundContext");
        if (!backgroundContextOpen)
            backgroundContextOpen = ctx->BeginPopupContextWindow("", 1, true);
    }
    if (backgroundContextOpen) {
        ctx->RecordSemanticWindow("context_menu", "Hierarchy Create", "hierarchy.context.root");
        if (renderContextMenu)
            renderContextMenu(ctx, 0, false, parentIdForNew);
        ctx->EndPopup();
    }
}

} // namespace infernux
