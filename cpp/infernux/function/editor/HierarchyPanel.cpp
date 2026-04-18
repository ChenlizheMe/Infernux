#include "HierarchyPanel.h"

#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/Transform.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>

// ImGui key constants (must match imgui.h ImGuiKey enum)
static constexpr int kKeyLeftCtrl = ImGuiKey_LeftCtrl;
static constexpr int kKeyRightCtrl = ImGuiKey_RightCtrl;
static constexpr int kKeyLeftShift = ImGuiKey_LeftShift;
static constexpr int kKeyRightShift = ImGuiKey_RightShift;
static constexpr int kKeyF2 = ImGuiKey_F2;
static constexpr int kKeyDelete = ImGuiKey_Delete;
static constexpr int kKeyEnter = ImGuiKey_Enter;
static constexpr int kKeyEscape = ImGuiKey_Escape;
static constexpr int kKeyC = ImGuiKey_C;
static constexpr int kKeyV = ImGuiKey_V;
static constexpr int kKeyX = ImGuiKey_X;

namespace infernux
{

// ════════════════════════════════════════════════════════════════════
// Helpers — casefold for search
// ════════════════════════════════════════════════════════════════════

static std::string CaseFold(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s)
        out.push_back(static_cast<char>(std::tolower(c)));
    return out;
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
    out["canvasRoots"] = m_subCanvasRoots;
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
    m_subCanvasRoots = 0.0;
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

void HierarchyPanel::SetUiMode(bool enabled)
{
    m_uiMode = enabled;
    // Invalidate caches
    m_cachedSceneKey.clear();
    m_cachedStructureVer = UINT64_MAX;
    m_lastRootRefreshTime = 0.0f;
    m_orderedIdsDirty = true;
    m_canvasRootsDirty = true;
    m_searchVisCache.clear();
    m_itemHeightMeasured = false;
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
    NotifySelectionChanged();
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

    // Always expand the parent chain
    if (id)
        ExpandToObject(id);

    if (changed)
        NotifySelectionChanged();
}

void HierarchyPanel::ExpandToObject(uint64_t objId)
{
    if (objId == 0)
        return;
    Scene *scene = SceneManager::Instance().GetActiveScene();
    if (!scene)
        return;
    GameObject *go = scene->FindByID(objId);
    if (!go)
        return;
    GameObject *parent = go->GetParent();
    while (parent) {
        uint64_t pid = parent->GetID();
        if (m_expandedNodes.insert(pid).second)
            m_flatListDirty = true;
        m_forceExpandIds.insert(pid);
        parent = parent->GetParent();
    }
}

void HierarchyPanel::SetPendingExpandId(uint64_t id)
{
    EnsureNodeExpanded(id);
}

void HierarchyPanel::EnsureNodeExpanded(uint64_t id)
{
    if (!id)
        return;
    if (m_expandedNodes.insert(id).second)
        m_flatListDirty = true;
    m_forceExpandIds.insert(id);
}

// ════════════════════════════════════════════════════════════════════
// Selection cache — sync once per frame
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::SyncSelectionCache()
{
    m_selIds.clear();
    if (getSelectedIds) {
        auto ids = getSelectedIds();
        for (auto id : ids)
            m_selIds.insert(id);
    }
    m_selPrimary = getPrimary ? getPrimary() : 0;
    m_selCount = selectionCount ? selectionCount() : 0;
}

// ════════════════════════════════════════════════════════════════════
// Notification
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::NotifySelectionChanged()
{
    uint64_t primary = getPrimary ? getPrimary() : 0;

    // In UI mode, skip inspector for non-canvas objects
    if (m_uiMode && primary != 0) {
        Scene *scene = SceneManager::Instance().GetActiveScene();
        GameObject *go = scene ? scene->FindByID(primary) : nullptr;
        if (go && !IsInCanvasTree(go)) {
            if (onSelectionChangedUiEditor)
                onSelectionChangedUiEditor(primary);
            return;
        }
    }

    if (onSelectionChanged)
        onSelectionChanged(primary);
    if (onSelectionChangedUiEditor)
        onSelectionChangedUiEditor(primary);
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

void HierarchyPanel::RefreshRootObjects(Scene *scene, bool allowStale)
{
    if (!scene) {
        m_cachedRoots.clear();
        return;
    }
    std::string sceneKey = scene->GetName();
    uint64_t ver = scene->GetStructureVersion();

    float now = ImGui::GetTime();
    bool canReuseStale = (allowStale && m_cachedSceneKey == sceneKey && !m_cachedRoots.empty() &&
                          static_cast<int>(m_cachedRoots.size()) >= STALE_ROOT_THRESHOLD &&
                          (now - m_lastRootRefreshTime) < STALE_ROOT_INTERVAL);

    if (sceneKey != m_cachedSceneKey || (ver != m_cachedStructureVer && !canReuseStale)) {
        m_cachedRoots = FilterHidden(scene->GetRootObjects());
        m_orderedIdsDirty = true;
        m_canvasRootsDirty = true;
        m_searchVisCache.clear();
        m_itemHeightMeasured = false;
        m_flatListDirty = true;
        m_cachedSceneKey = sceneKey;
        m_cachedStructureVer = ver;
        m_lastRootRefreshTime = now;
    }
}

// ════════════════════════════════════════════════════════════════════
// Search
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::SetSearchQuery(const char *text)
{
    std::string s(text ? text : "");
    std::string norm = CaseFold(s);
    // Trim
    while (!norm.empty() && norm.front() == ' ')
        norm.erase(norm.begin());
    while (!norm.empty() && norm.back() == ' ')
        norm.pop_back();

    if (s == m_searchQuery && norm == m_searchQueryNorm)
        return;
    m_searchQuery = std::move(s);
    m_searchQueryNorm = std::move(norm);
    m_searchVisCache.clear();
    m_flatListDirty = true;
}

bool HierarchyPanel::MatchesSearch(GameObject *obj) const
{
    if (m_searchQueryNorm.empty())
        return true;
    std::string name = CaseFold(obj->GetName());
    return name.find(m_searchQueryNorm) != std::string::npos;
}

bool HierarchyPanel::IsVisibleInSearch(GameObject *obj)
{
    if (m_searchQueryNorm.empty())
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
    m_flatItems.reserve(roots.size() * 2); // heuristic
    for (auto *root : roots)
        BuildFlatListRecurse(root, 0);
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

    m_flatItems.push_back({obj, depth, hasVisibleChildren});

    // Determine expanded state
    bool isExpanded = m_expandedNodes.count(objId) > 0;

    // Auto-expand when search is active
    if (HasActiveSearch() && hasVisibleChildren) {
        isExpanded = true;
        m_expandedNodes.insert(objId);
        m_forceExpandIds.insert(objId);
    }

    if (hasVisibleChildren && isExpanded) {
        for (const auto &child : children) {
            if (IsHidden(child->GetID()))
                continue;
            BuildFlatListRecurse(child.get(), depth + 1);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Canvas helpers
// ════════════════════════════════════════════════════════════════════

bool HierarchyPanel::IsInCanvasTree(GameObject *obj) const
{
    // Walk to root and check if that root is in canvas root set
    GameObject *cur = obj;
    while (true) {
        GameObject *p = cur->GetParent();
        if (!p)
            break;
        cur = p;
    }
    return m_canvasRootIds.count(cur->GetID()) > 0;
}

void HierarchyPanel::RefreshCanvasRootIds(const std::vector<GameObject *> &roots)
{
    if (!m_canvasRootsDirty)
        return;
    m_canvasRootIds.clear();
    if (getCanvasRootIds) {
        auto rootIds = getCanvasRootIds();
        m_canvasRootIds.insert(rootIds.begin(), rootIds.end());
    } else if (hasCanvasDescendant) {
        for (auto *go : roots) {
            if (hasCanvasDescendant(go->GetID()))
                m_canvasRootIds.insert(go->GetID());
        }
    }
    m_canvasRootsDirty = false;
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
    if (m_selIds.count(primaryId) && m_selCount > 1 && getSelectedIds)
        return getSelectedIds();
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

bool HierarchyPanel::ValidateReparent(GameObject *obj, uint64_t newParentId, GameObject *newParent)
{
    if (m_uiMode) {
        if (!IsInCanvasTree(obj))
            return false;
        if (goHasCanvas && goHasCanvas(obj->GetID())) {
            if (showWarning)
                showWarning("Canvas can only be a root object.");
            return false;
        }
    } else {
        if (IsInCanvasTree(obj))
            return false;
    }
    if (goHasUiScreenComponent && goHasUiScreenComponent(obj->GetID())) {
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
    if (m_uiMode) {
        if (!IsInCanvasTree(obj))
            return false;
        if (goHasCanvas && goHasCanvas(obj->GetID()) && newParentId != 0) {
            if (showWarning)
                showWarning("Canvas can only be a root object.");
            return false;
        }
        if (goHasCanvas && !goHasCanvas(obj->GetID()) && newParentId == 0) {
            if (showWarning)
                showWarning("UI elements must be placed under a Canvas.");
            return false;
        }
    } else {
        if (IsInCanvasTree(obj))
            return false;
    }
    if (goHasUiScreenComponent && goHasUiScreenComponent(obj->GetID())) {
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
    Scene *scene = SceneManager::Instance().GetActiveScene();
    if (!scene)
        return;
    GameObject *newParent = scene->FindByID(newParentId);
    if (!newParent)
        return;

    auto dragIds = GetDragIds(draggedId);
    auto sorted = TopoSortIds(scene, dragIds);

    for (uint64_t did : sorted) {
        if (did == newParentId)
            continue;
        auto *obj = scene->FindByID(did);
        if (!obj)
            continue;
        if (IsDescendantOf(newParent, obj))
            continue;
        if (!ValidateReparent(obj, newParentId, newParent))
            continue;

        auto *oldP = obj->GetParent();
        uint64_t oldPid = oldP ? oldP->GetID() : 0;
        int oldIdx = obj->GetTransform() ? obj->GetTransform()->GetSiblingIndex() : 0;
        int newIdx = static_cast<int>(newParent->GetChildren().size());
        if (oldPid == newParentId && oldIdx < newIdx)
            newIdx--;

        if (undoRecordMove)
            undoRecordMove(did, oldPid, newParentId, oldIdx, newIdx);
    }
    EnsureNodeExpanded(newParentId);
}

void HierarchyPanel::MoveObjectAdjacent(uint64_t draggedId, uint64_t targetId, bool after)
{
    Scene *scene = SceneManager::Instance().GetActiveScene();
    if (!scene)
        return;
    auto *targetObj = scene->FindByID(targetId);
    if (!targetObj)
        return;

    auto *newParent = targetObj->GetParent();
    uint64_t newParentId = newParent ? newParent->GetID() : 0;

    auto dragIds = GetDragIds(draggedId);
    auto sorted = TopoSortIds(scene, dragIds);

    std::vector<uint64_t> validIds;
    for (uint64_t did : sorted) {
        if (did == targetId)
            continue;
        auto *obj = scene->FindByID(did);
        if (!obj)
            continue;
        if (IsDescendantOf(targetObj, obj))
            continue;
        if (!ValidateMoveAdjacent(obj, newParentId, newParent))
            continue;
        validIds.push_back(did);
    }
    if (validIds.empty())
        return;

    int anchorIdx = targetObj->GetTransform() ? targetObj->GetTransform()->GetSiblingIndex() : 0;
    int insertIdx = anchorIdx + (after ? 1 : 0);

    for (uint64_t did : validIds) {
        auto *obj = scene->FindByID(did);
        if (!obj)
            continue;
        auto *oldP = obj->GetParent();
        uint64_t oldPid = oldP ? oldP->GetID() : 0;
        int oldIdx = obj->GetTransform() ? obj->GetTransform()->GetSiblingIndex() : 0;

        int effIdx = insertIdx;
        if (oldPid == newParentId && oldIdx < effIdx)
            effIdx--;
        if (oldPid == newParentId && oldIdx == effIdx) {
            insertIdx++;
            continue;
        }

        if (undoRecordMove)
            undoRecordMove(did, oldPid, newParentId, oldIdx, effIdx);
        insertIdx = effIdx + 1;
    }

    if (newParentId != 0)
        EnsureNodeExpanded(newParentId);
}

void HierarchyPanel::ReparentToRoot(uint64_t draggedId)
{
    Scene *scene = SceneManager::Instance().GetActiveScene();
    if (!scene)
        return;

    auto dragIds = GetDragIds(draggedId);
    auto sorted = TopoSortIds(scene, dragIds);

    for (uint64_t did : sorted) {
        auto *obj = scene->FindByID(did);
        if (!obj)
            continue;
        if (m_uiMode) {
            if (!IsInCanvasTree(obj))
                continue;
            if (goHasCanvas && !goHasCanvas(obj->GetID())) {
                if (showWarning)
                    showWarning("UI elements must be placed under a Canvas.");
                continue;
            }
        } else {
            if (IsInCanvasTree(obj))
                continue;
        }
        if (goHasUiScreenComponent && goHasUiScreenComponent(obj->GetID())) {
            if (showWarning)
                showWarning("UI components must be placed under a Canvas.");
            continue;
        }

        auto *oldParent = obj->GetParent();
        uint64_t oldPid = oldParent ? oldParent->GetID() : 0;
        int oldIdx = obj->GetTransform() ? obj->GetTransform()->GetSiblingIndex() : 0;
        int rootCount = static_cast<int>(scene->GetRootObjects().size());
        int newIdx = (std::max)(0, rootCount - (oldPid == 0 ? 1 : 0));
        if (oldPid != 0 || oldIdx != newIdx) {
            if (undoRecordMove)
                undoRecordMove(did, oldPid, 0, oldIdx, newIdx);
        }
    }
}

void HierarchyPanel::HandleExternalDrop(const std::string &dropType, uint64_t payload, uint64_t parentId)
{
    // In Prefab Mode, force under prefab root
    if (isPrefabMode && isPrefabMode() && parentId == 0) {
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
    if (isPrefabMode && isPrefabMode() && parentId == 0) {
        Scene *scene = SceneManager::Instance().GetActiveScene();
        if (scene && !scene->GetRootObjects().empty())
            parentId = scene->GetRootObjects()[0]->GetID();
    }

    if (dropType == "PREFAB_GUID" || dropType == "PREFAB_FILE") {
        bool isGuid = (dropType == "PREFAB_GUID");
        if (instantiatePrefab)
            instantiatePrefab(payload, parentId, isGuid);
    } else if (dropType == "MODEL_GUID" || dropType == "MODEL_FILE") {
        bool isGuid = (dropType == "MODEL_GUID");
        if (createModelObject)
            createModelObject(payload, parentId, isGuid);
    }
}

// ════════════════════════════════════════════════════════════════════
// Rename
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::BeginRename(uint64_t objId)
{
    Scene *scene = SceneManager::Instance().GetActiveScene();
    if (!scene)
        return;
    auto *obj = scene->FindByID(objId);
    if (!obj)
        return;
    m_renameId = objId;
    std::strncpy(m_renameBuf, obj->GetName().c_str(), sizeof(m_renameBuf) - 1);
    m_renameBuf[sizeof(m_renameBuf) - 1] = '\0';
    m_renameFocus = true;
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
            auto *obj = scene->FindByID(m_renameId);
            if (obj)
                obj->SetName(newName);
        }
    }
    m_renameId = 0;
    m_renameBuf[0] = '\0';
}

void HierarchyPanel::CancelRename()
{
    m_renameId = 0;
    m_renameBuf[0] = '\0';
}

// ════════════════════════════════════════════════════════════════════
// Clipboard shortcuts
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::HandleClipboardShortcuts(InxGUIContext *ctx)
{
    if (!ctx->IsWindowFocused(0) || ctx->WantTextInput())
        return;
    if (!IsCtrl(ctx))
        return;

    if (ctx->IsKeyPressed(kKeyC)) {
        if (copySelected)
            copySelected(false);
        return;
    }
    if (ctx->IsKeyPressed(kKeyX)) {
        if (copySelected)
            copySelected(true);
        return;
    }
    if (ctx->IsKeyPressed(kKeyV)) {
        if (pasteClipboard)
            pasteClipboard();
    }
}

// ════════════════════════════════════════════════════════════════════
// Reorder separator helper
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::RenderReorderSep(InxGUIContext *ctx, const char *sepId, std::function<void(uint64_t)> onDrop)
{
    if (ImGui::GetDragDropPayload() == nullptr)
        return;

    float savedY = ctx->GetCursorPosY();
    float availW = ctx->GetContentRegionAvailWidth();
    ctx->SetNextItemAllowOverlap();
    ctx->InvisibleButton(sepId, availW, EditorTheme::DND_REORDER_SEPARATOR_H);
    ctx->PushStyleColor(ImGuiCol_DragDropTarget, 0.0f, 0.0f, 0.0f, 0.0f);
    if (ctx->BeginDragDropTarget()) {
        // Draw separator line at midpoint
        float minY = ctx->GetItemRectMinY();
        float maxY = ctx->GetItemRectMaxY();
        float midY = (minY + maxY) * 0.5f;
        float x1 = ctx->GetItemRectMinX();
        float x2 = x1 + availW;
        ctx->DrawLine(x1, midY, x2, midY, EditorTheme::DND_REORDER_LINE.x, EditorTheme::DND_REORDER_LINE.y,
                      EditorTheme::DND_REORDER_LINE.z, EditorTheme::DND_REORDER_LINE.w,
                      EditorTheme::DND_REORDER_LINE_THICKNESS);
        uint64_t payload = 0;
        if (ctx->AcceptDragDropPayload(DRAG_DROP_TYPE, &payload)) {
            if (onDrop)
                onDrop(payload);
        }
        ctx->EndDragDropTarget();
    }
    ctx->PopStyleColor(1);
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
        for (const char *dt : {"MODEL_GUID", "MODEL_FILE", "PREFAB_GUID", "PREFAB_FILE"}) {
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
    if (!obj)
        return;

    const uint64_t objId = obj->GetID();
    const bool isPrefab = obj->IsPrefabInstance();

    if (ctx->BeginMenu(Tr("hierarchy.create_child"))) {
        if (ctx->BeginMenu(Tr("hierarchy.create_3d_object"))) {
            ShowCreatePrimitiveMenu(ctx, objId);
            ctx->EndMenu();
        }
        if (ctx->BeginMenu(Tr("hierarchy.light_menu"))) {
            ShowCreateLightMenu(ctx, objId);
            ctx->EndMenu();
        }
        if (ctx->BeginMenu(Tr("hierarchy.rendering_menu"))) {
            ShowCreateRenderingMenu(ctx, objId);
            ctx->EndMenu();
        }
        if (ctx->Selectable(Tr("hierarchy.empty_object"), false, 0, 0, 0)) {
            if (createEmpty)
                createEmpty(objId);
        }
        ctx->EndMenu();
    }
    ctx->Separator();
    if (ctx->Selectable(Tr("project.copy"), false, 0, 0, 0)) {
        if (copySelected)
            copySelected(false);
    }
    if (ctx->Selectable(Tr("project.cut"), false, 0, 0, 0)) {
        if (copySelected)
            copySelected(true);
    }
    if (ctx->Selectable(Tr("project.paste"), false, 0, 0, 0)) {
        if (pasteClipboard)
            pasteClipboard();
    }
    ctx->Separator();
    if (ctx->Selectable(Tr("hierarchy.rename"), false, 0, 0, 0))
        BeginRename(objId);
    ctx->Separator();
    if (ctx->Selectable(Tr("hierarchy.save_as_prefab"), false, 0, 0, 0)) {
        if (saveAsPrefab)
            saveAsPrefab(objId);
    }

    if (isPrefab) {
        ctx->Separator();
        ctx->PushStyleColor(ImGuiCol_Text, EditorTheme::PREFAB_TEXT.x, EditorTheme::PREFAB_TEXT.y,
                            EditorTheme::PREFAB_TEXT.z, EditorTheme::PREFAB_TEXT.w);
        ctx->Label(Tr("hierarchy.prefab_label"));
        ctx->PopStyleColor(1);
        if (ctx->Selectable(Tr("hierarchy.select_prefab_asset"), false, 0, 0, 0)) {
            if (prefabSelectAsset)
                prefabSelectAsset(objId);
        }
        if (ctx->Selectable(Tr("hierarchy.open_prefab"), false, 0, 0, 0)) {
            if (prefabOpenAsset)
                prefabOpenAsset(objId);
        }
        if (ctx->Selectable(Tr("hierarchy.apply_all_overrides"), false, 0, 0, 0)) {
            if (prefabApplyOverrides)
                prefabApplyOverrides(objId);
        }
        if (ctx->Selectable(Tr("hierarchy.revert_all_overrides"), false, 0, 0, 0)) {
            if (prefabRevertOverrides)
                prefabRevertOverrides(objId);
        }
        ctx->Separator();
        if (ctx->Selectable(Tr("hierarchy.unpack_prefab"), false, 0, 0, 0)) {
            if (prefabUnpack)
                prefabUnpack(objId);
        }
    }

    ctx->Separator();
    if (ctx->Selectable(Tr("hierarchy.delete"), false, 0, 0, 0)) {
        if (undoRecordDelete)
            undoRecordDelete(objId, "Delete GameObject");
        if (m_selIds.count(objId)) {
            if (clearSelection)
                clearSelection();
            SyncSelectionCache();
            NotifySelectionChanged();
        }
    }
}

void HierarchyPanel::ShowCreatePrimitiveMenu(InxGUIContext *ctx, uint64_t parentId)
{
    struct PrimEntry
    {
        const char *key;
        int typeIdx;
    };
    static const PrimEntry entries[] = {
        {"hierarchy.primitive_cube", 0},     {"hierarchy.primitive_sphere", 1}, {"hierarchy.primitive_capsule", 2},
        {"hierarchy.primitive_cylinder", 3}, {"hierarchy.primitive_plane", 4},  {"hierarchy.primitive_quad", 5},
    };
    for (auto &e : entries) {
        if (ctx->Selectable(Tr(e.key), false, 0, 0, 0)) {
            if (createPrimitive)
                createPrimitive(e.typeIdx, parentId);
        }
    }
}

void HierarchyPanel::ShowCreateLightMenu(InxGUIContext *ctx, uint64_t parentId)
{
    struct LightEntry
    {
        const char *key;
        int typeIdx;
    };
    static const LightEntry entries[] = {
        {"hierarchy.light_directional", 0},
        {"hierarchy.light_point", 1},
        {"hierarchy.light_spot", 2},
    };
    for (auto &e : entries) {
        if (ctx->Selectable(Tr(e.key), false, 0, 0, 0)) {
            if (createLight)
                createLight(e.typeIdx, parentId);
        }
    }
}

void HierarchyPanel::ShowCreateRenderingMenu(InxGUIContext *ctx, uint64_t parentId)
{
    for (auto &entry : createEntries) {
        if (entry.category == "Rendering") {
            if (ctx->Selectable(Tr(entry.localeKey), false, 0, 0, 0)) {
                if (entry.callback)
                    entry.callback(parentId);
            }
        }
    }
}

void HierarchyPanel::ShowUiMenu(InxGUIContext *ctx, uint64_t parentId)
{
    for (auto &entry : createEntries) {
        if (entry.category == "UI") {
            if (ctx->Selectable(Tr(entry.localeKey), false, 0, 0, 0)) {
                if (entry.callback)
                    entry.callback(parentId);
            }
        }
    }
}

void HierarchyPanel::ShowUiModeContextMenu(InxGUIContext *ctx, uint64_t parentId)
{
    for (auto &entry : createEntries) {
        if (entry.category == "UI") {
            if (ctx->Selectable(Tr(entry.localeKey), false, 0, 0, 0)) {
                if (entry.callback)
                    entry.callback(parentId);
            }
        }
    }
}

void HierarchyPanel::AddCreateEntry(const std::string &category, const std::string &localeKey,
                                    std::function<void(uint64_t)> callback)
{
    createEntries.push_back({category, localeKey, std::move(callback)});
}

void HierarchyPanel::ClearCreateEntries()
{
    createEntries.clear();
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
    ctx->InputTextWithHint("##rename", "", m_renameBuf, sizeof(m_renameBuf), 0);

    if (ctx->IsKeyPressed(kKeyEnter)) {
        CommitRename();
        return;
    }
    if (ctx->IsKeyPressed(kKeyEscape)) {
        CancelRename();
        return;
    }
    if (ctx->IsItemDeactivated())
        CommitRename();
}

// ════════════════════════════════════════════════════════════════════
// Flat item rendering — virtualised hierarchy rows (single scroll body).
// ════════════════════════════════════════════════════════════════════

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

    if (m_selIds.count(objId))
        nodeFlags |= ImGuiTreeNodeFlags_Selected;

    bool isLeaf = !item.hasVisibleChildren;
    if (isLeaf)
        nodeFlags |= ImGuiTreeNodeFlags_Leaf;

    // Force-expand (one-shot for auto-expand, selection expand, etc.)
    if (m_forceExpandIds.count(objId)) {
        ctx->SetNextItemOpen(true);
        m_forceExpandIds.erase(objId);
    }

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

    // Dim objects that don't belong to the current mode's domain
    bool uiDimmed = false;
    if (m_uiMode) {
        uiDimmed = !IsInCanvasTree(obj);
    } else if (!m_canvasRootIds.empty()) {
        uiDimmed = IsInCanvasTree(obj);
    }
    int textColorPushed = 0;
    if (uiDimmed) {
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

    bool isOpen = ctx->TreeNodeEx(*displayName, nodeFlags);

    if (indentPx > 0)
        ImGui::Unindent(indentPx);

    if (textColorPushed)
        ctx->PopStyleColor(1);

    // Sync expand state from TreeNodeEx return value
    if (isOpen && !isLeaf) {
        if (m_expandedNodes.insert(objId).second)
            m_flatListDirty = true; // newly expanded
    } else {
        if (m_expandedNodes.erase(objId) > 0)
            m_flatListDirty = true; // newly collapsed
    }

    // ── Selection ───────────────────────────────────────────────
    if (ctx->IsItemClicked(0)) {
        if (m_renameId && m_renameId != objId)
            CancelRename();
        m_pendingSelectId = objId;
        m_pendingCtrl = IsCtrl(ctx);
        m_pendingShift = IsShift(ctx);
    }
    if (ctx->IsItemClicked(1)) {
        if (!m_selIds.count(objId)) {
            if (selectId)
                selectId(objId);
            SyncSelectionCache();
            NotifySelectionChanged();
        }
        m_rightClickedObjId = objId;
        ctx->OpenPopup("##HierarchyItemContext");
    }

    // Double-click focus
    if (ctx->IsMouseDoubleClicked(0) && ctx->IsItemHovered()) {
        if (onDoubleClickFocus)
            onDoubleClickFocus(objId);
    }

    // ── Drag source ─────────────────────────────────────────────
    if (ctx->BeginDragDropSource(0)) {
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
// PreRender — keyboard shortcuts + deferred selection
// ════════════════════════════════════════════════════════════════════

void HierarchyPanel::PreRender(InxGUIContext *ctx)
{
    using Clock = std::chrono::high_resolution_clock;
    auto msSince = [](const Clock::time_point &start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };

    // Refresh hidden IDs
    auto preHiddenStart = Clock::now();
    if (getRuntimeHiddenIds)
        m_hiddenIds = getRuntimeHiddenIds();
    else
        m_hiddenIds.clear();
    m_subPreHidden += msSince(preHiddenStart);

    // Sync selection once per frame
    auto preSelectionStart = Clock::now();
    SyncSelectionCache();
    m_subPreSelection += msSince(preSelectionStart);

    // Keyboard shortcuts (F2 rename, Delete)
    auto shortcutStart = Clock::now();
    if (!ctx->WantTextInput() && m_selCount > 0) {
        if (ctx->IsKeyPressed(kKeyF2) && m_renameId == 0) {
            if (m_selPrimary)
                BeginRename(m_selPrimary);
        }
        if (ctx->IsKeyPressed(kKeyDelete)) {
            if (deleteSelectedObjects)
                deleteSelectedObjects();
            SyncSelectionCache();
        }
    }
    m_subPreShortcuts += msSince(shortcutStart);

    // Deferred left-click selection
    auto pendingStart = Clock::now();
    if (m_pendingSelectId != 0) {
        if (!ctx->IsMouseButtonDown(0)) {
            if (!ctx->IsMouseDragging(0)) {
                uint64_t pid = m_pendingSelectId;

                // In UI mode, block selection of non-canvas objects
                if (m_uiMode) {
                    Scene *scene = SceneManager::Instance().GetActiveScene();
                    auto *go = scene ? scene->FindByID(pid) : nullptr;
                    if (go && !IsInCanvasTree(go)) {
                        m_pendingSelectId = 0;
                        m_pendingCtrl = false;
                        m_pendingShift = false;
                        return;
                    }
                }

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
                NotifySelectionChanged();
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

    HandleClipboardShortcuts(ctx);

    // ── Header: scene name / prefab mode / ui mode ──────────────
    auto headerStart = Clock::now();
    if (m_uiMode) {
        ctx->Label(Tr("hierarchy.ui_mode"));
    } else if (isPrefabMode && isPrefabMode()) {
        std::string prefabName = getPrefabDisplayName ? getPrefabDisplayName() : "Prefab";
        ctx->PushStyleColor(ImGuiCol_Text, EditorTheme::PREFAB_TEXT.x, EditorTheme::PREFAB_TEXT.y,
                            EditorTheme::PREFAB_TEXT.z, EditorTheme::PREFAB_TEXT.w);
        ctx->Label(prefabName);
        ctx->PopStyleColor(1);
    } else {
        std::string displayName = getSceneDisplayName ? getSceneDisplayName() : "";
        if (!displayName.empty())
            ctx->Label(displayName);
        else {
            Scene *scene = SceneManager::Instance().GetActiveScene();
            ctx->Label(scene ? scene->GetName() : Tr("hierarchy.no_scene"));
        }
    }
    m_subHeader += msSince(headerStart);

    // ── Search bar ──────────────────────────────────────────────
    auto searchStart = Clock::now();
    ctx->SetNextItemWidth(ctx->GetContentRegionAvailWidth());
    std::strncpy(m_searchBuf, m_searchQuery.c_str(), sizeof(m_searchBuf) - 1);
    m_searchBuf[sizeof(m_searchBuf) - 1] = '\0';
    ctx->InputTextWithHint("##HierarchySearch", Tr("hierarchy.search_placeholder").c_str(), m_searchBuf,
                           sizeof(m_searchBuf), 0);
    SetSearchQuery(m_searchBuf);

    ctx->Separator();
    m_subSearch += msSince(searchStart);

    // ── Scene tree ──────────────────────────────────────────────
    Scene *scene = SceneManager::Instance().GetActiveScene();
    if (scene) {
        ctx->PushStyleVarVec2(ImGuiStyleVar_ItemSpacing, EditorTheme::TREE_ITEM_SPC.x, EditorTheme::TREE_ITEM_SPC.y);
        ctx->PushStyleVarVec2(ImGuiStyleVar_FramePadding, EditorTheme::TREE_FRAME_PAD.x, EditorTheme::TREE_FRAME_PAD.y);
        ctx->PushStyleVarFloat(ImGuiStyleVar_IndentSpacing, EditorTheme::TREE_INDENT);

        bool allowStale = !ctx->IsWindowFocused(0) && !ctx->IsWindowHovered() && !m_cachedRoots.empty();
        {
            auto t0 = Clock::now();
            RefreshRootObjects(scene, allowStale);
            m_subRefreshRoots += msSince(t0);
        }

        // Refresh canvas roots
        {
            auto t0 = Clock::now();
            RefreshCanvasRootIds(m_cachedRoots);
            m_subCanvasRoots += msSince(t0);
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
        int nItems = static_cast<int>(m_flatItems.size());

        // Root-level insertion line before first root (only when dragging)
        bool hasDrag = (ImGui::GetDragDropPayload() != nullptr);
        if (hasDrag) {
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
                viewportH = 400.0f;
            float startY = ctx->GetCursorPosY();
            float itemH = m_cachedItemHeight;
            float indentStep = EditorTheme::TREE_INDENT;

            int firstVis = (std::max)(0, static_cast<int>((scrollY - startY) / itemH) - 2);
            int lastVis = (std::min)(nItems - 1, static_cast<int>((scrollY + viewportH - startY) / itemH) + 3);

            if (firstVis > 0)
                ctx->Dummy(availW, static_cast<float>(firstVis) * itemH);

            float baseIndentX = ctx->GetCursorPosX();
            for (int i = firstVis; i <= lastVis; i++) {
                float beforeY = ctx->GetCursorPosY();

                // Reorder separator before first child (only when dragging)
                if (hasDrag && i > 0 && m_flatItems[i].depth > m_flatItems[i - 1].depth) {
                    uint64_t childId = m_flatItems[i].obj->GetID();
                    std::string sepId = "##sep_fc_" + std::to_string(m_flatItems[i - 1].obj->GetID());
                    RenderReorderSep(ctx, sepId.c_str(), [this, childId](uint64_t payload) {
                        MoveObjectAdjacent(payload, childId, false);
                    });
                }

                RenderFlatItem(ctx, m_flatItems[i], baseIndentX, indentStep);

                // Reorder separator after each item (only when dragging)
                if (hasDrag) {
                    uint64_t afterObjId = m_flatItems[i].obj->GetID();
                    std::string sepAfterId = "##sep_a_" + std::to_string(afterObjId);
                    RenderReorderSep(ctx, sepAfterId.c_str(), [this, afterObjId](uint64_t payload) {
                        MoveObjectAdjacent(payload, afterObjId, true);
                    });
                }

                float afterY = ctx->GetCursorPosY();
                float actualH = afterY - beforeY;
                if (actualH > 1.0f && !m_itemHeightMeasured) {
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

        auto popupStart = Clock::now();
        if (ctx->BeginPopup("##HierarchyItemContext")) {
            GameObject *popupObj = scene ? scene->FindByID(m_rightClickedObjId) : nullptr;
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
        float remainingH = ctx->GetContentRegionAvailHeight();
        if (remainingH > 4.0f) {
            float tailW = ctx->GetContentRegionAvailWidth();
            ctx->InvisibleButton("##drop_to_root_tail", tailW, remainingH);

            if (ctx->IsItemClicked(0)) {
                CancelRename();
                ClearSelectionAndNotify();
            }

            // Drop target with top-edge line
            ctx->PushStyleColor(ImGuiCol_DragDropTarget, 0.0f, 0.0f, 0.0f, 0.0f);
            if (ctx->BeginDragDropTarget()) {
                float lineY = ctx->GetItemRectMinY();
                float lineX1 = ctx->GetItemRectMinX();
                float lineX2 = lineX1 + tailW;
                ctx->DrawLine(lineX1, lineY, lineX2, lineY, EditorTheme::DND_REORDER_LINE.x,
                              EditorTheme::DND_REORDER_LINE.y, EditorTheme::DND_REORDER_LINE.z,
                              EditorTheme::DND_REORDER_LINE.w, EditorTheme::DND_REORDER_LINE_THICKNESS);
                // Accept uint64_t payload
                uint64_t payload = 0;
                bool accepted = false;
                if (ctx->AcceptDragDropPayload(DRAG_DROP_TYPE, &payload)) {
                    HandleExternalDrop(DRAG_DROP_TYPE, payload, 0);
                    accepted = true;
                }
                if (!accepted) {
                    for (const char *dt : {"MODEL_GUID", "MODEL_FILE", "PREFAB_GUID", "PREFAB_FILE"}) {
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

        // Fallback: deselect when clicking the scrollable background
        // (the tail InvisibleButton only works when remainingH > 4)
        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsAnyItemHovered()) {
            CancelRename();
            ClearSelectionAndNotify();
        }
        m_subTailDrop += msSince(tailDropStart);

        ctx->PopStyleVar(3); // IndentSpacing + FramePadding + ItemSpacing

        if (HasActiveSearch() && nItems == 0)
            ctx->Label(Tr("hierarchy.no_search_results"));
    }

    // ── Parent for new objects ───────────────────────────────────
    uint64_t parentIdForNew = 0;
    if (isPrefabMode && isPrefabMode()) {
        Scene *pscene = SceneManager::Instance().GetActiveScene();
        if (pscene && !pscene->GetRootObjects().empty())
            parentIdForNew = pscene->GetRootObjects()[0]->GetID();
    } else if (m_selCount > 0) {
        parentIdForNew = m_selPrimary;
    }

    // ── Background context menu ─────────────────────────────────
    if (ctx->BeginPopupContextWindow("", 1)) {
        if (m_uiMode) {
            ShowUiModeContextMenu(ctx, parentIdForNew);
        } else {
            if (ctx->BeginMenu(Tr("hierarchy.create_3d_object"))) {
                ShowCreatePrimitiveMenu(ctx, parentIdForNew);
                ctx->EndMenu();
            }
            if (ctx->BeginMenu(Tr("hierarchy.light_menu"))) {
                ShowCreateLightMenu(ctx, parentIdForNew);
                ctx->EndMenu();
            }
            if (ctx->BeginMenu(Tr("hierarchy.rendering_menu"))) {
                ShowCreateRenderingMenu(ctx, parentIdForNew);
                ctx->EndMenu();
            }
            if (ctx->BeginMenu(Tr("hierarchy.ui_menu"))) {
                ShowUiMenu(ctx, parentIdForNew);
                ctx->EndMenu();
            }
            if (ctx->Selectable(Tr("hierarchy.create_empty"), false, 0, 0, 0)) {
                if (createEmpty)
                    createEmpty(parentIdForNew);
            }
        }

        bool hasClip = hasClipboardData && hasClipboardData();
        if (m_selCount > 0 || hasClip) {
            ctx->Separator();
            if (m_selCount > 0) {
                if (ctx->Selectable(Tr("project.copy"), false, 0, 0, 0)) {
                    if (copySelected)
                        copySelected(false);
                }
                if (ctx->Selectable(Tr("project.cut"), false, 0, 0, 0)) {
                    if (copySelected)
                        copySelected(true);
                }
            }
            if (hasClip) {
                if (ctx->Selectable(Tr("project.paste"), false, 0, 0, 0)) {
                    if (pasteClipboard)
                        pasteClipboard();
                }
            }
        }

        if (m_selCount > 0) {
            ctx->Separator();
            if (ctx->Selectable(Tr("hierarchy.delete_selected"), false, 0, 0, 0)) {
                if (deleteSelectedObjects)
                    deleteSelectedObjects();
                SyncSelectionCache();
            }
        }

        ctx->EndPopup();
    }
}

} // namespace infernux
