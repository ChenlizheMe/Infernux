#include <function/editor/InspectorPanel.h>

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace infernux
{

// ── Drag speeds (match Python inspector_utils constants) ─────────
static constexpr float DRAG_SPEED_DEFAULT = 0.1f;
static constexpr float DRAG_SPEED_FINE = 0.01f;

// ============================================================================
// Construction
// ============================================================================

InspectorPanel::InspectorPanel() : EditorPanel("Inspector", "inspector")
{
}

// ============================================================================
// Sub-timing snapshot (consumed + reset by profile system)
// ============================================================================

std::unordered_map<std::string, double> InspectorPanel::ConsumeSubTimings()
{
    std::unordered_map<std::string, double> out;
    out["info"] = m_subGetInfo;
    out["transform"] = m_subTransform;
    out["compList"] = m_subGetComponents;
    out["compBody"] = m_subComponentBodies;
    out["material"] = m_subMaterials;
    if (consumeComponentBodyProfile) {
        auto bodyMetrics = consumeComponentBodyProfile();
        for (const auto &kv : bodyMetrics)
            out[kv.first] += kv.second;
    }
    m_subGetInfo = 0.0;
    m_subTransform = 0.0;
    m_subGetComponents = 0.0;
    m_subComponentBodies = 0.0;
    m_subMaterials = 0.0;
    return out;
}

// ============================================================================
// Translation cache
// ============================================================================

const std::string &InspectorPanel::Tr(const std::string &key)
{
    auto it = m_trCache.find(key);
    if (it != m_trCache.end())
        return it->second;
    if (translate) {
        auto &ref = m_trCache[key];
        ref = translate(key);
        return ref;
    }
    return m_trCache[key] = key;
}

// ============================================================================
// Public API
// ============================================================================

void InspectorPanel::SetSelectedObjectId(uint64_t id)
{
    if (m_selectedObjectId != id) {
        m_cachedObjInfoId = 0; // invalidate cache
        m_cachedComponentListObjId = 0;
        m_cachedComponents.clear();
        m_cachedValueGeneration = 0;
        m_cachedValueRefreshTime = 0.0f;
    }
    m_selectedObjectId = id;
    if (id != 0) {
        m_selectedFile.clear();
        m_assetCategory.clear();
        m_mode = InspectorMode::Object;
    }
}

void InspectorPanel::ClearSelectedObject()
{
    m_selectedObjectId = 0;
    m_cachedObjInfoId = 0;
    m_cachedComponentListObjId = 0;
    m_cachedComponents.clear();
    m_cachedValueGeneration = 0;
    m_cachedValueRefreshTime = 0.0f;
    m_mode = InspectorMode::Object;
}

void InspectorPanel::SetSelectedFile(const std::string &filePath, const std::string &category)
{
    if (filePath != m_selectedFile) {
        m_selectedFile = filePath;
        m_cachedObjInfoId = 0;
        m_cachedComponentListObjId = 0;
        m_cachedComponents.clear();
        m_cachedValueGeneration = 0;
        m_cachedValueRefreshTime = 0.0f;
    }
    if (!filePath.empty()) {
        m_assetCategory = category;
        m_mode = category.empty() ? InspectorMode::Preview : InspectorMode::Asset;
        // Clear object selection when file is selected
        m_selectedObjectId = 0;
    } else {
        m_assetCategory.clear();
        m_mode = InspectorMode::Object;
    }
}

void InspectorPanel::ClearSelectedFile()
{
    m_selectedFile.clear();
    m_assetCategory.clear();
    m_cachedValueGeneration = 0;
    m_cachedValueRefreshTime = 0.0f;
    m_mode = InspectorMode::Object;
}

void InspectorPanel::SetDetailFile(const std::string &filePath, const std::string &category)
{
    if (filePath != m_selectedFile) {
        m_selectedFile = filePath;
    }
    if (!filePath.empty()) {
        m_assetCategory = category;
        m_mode = category.empty() ? InspectorMode::Preview : InspectorMode::Asset;
        // NOTE: unlike SetSelectedFile, does NOT clear object selection
    } else {
        m_assetCategory.clear();
        m_mode = InspectorMode::Object;
    }
}

// ============================================================================
// PreRender
// ============================================================================

void InspectorPanel::PreRender(InxGUIContext * /*ctx*/)
{
    auto now = std::chrono::steady_clock::now();
    m_frameTimeNow = std::chrono::duration<float>(now.time_since_epoch()).count();
}

// ============================================================================
// OnRenderContent — main entry
// ============================================================================

void InspectorPanel::OnRenderContent(InxGUIContext *ctx)
{
    float totalHeight = ImGui::GetContentRegionAvail().y;

    bool hasDetailContent = !m_selectedFile.empty();
    bool fileOnly = hasDetailContent && m_selectedObjectId == 0;

    if (fileOnly) {
        // Full-height file view (asset inspector or generic preview)
        RenderRawDataModule(ctx, 0.0f);
    } else if (hasDetailContent &&
               totalHeight > (EditorTheme::INSPECTOR_MIN_PROPS_H + EditorTheme::INSPECTOR_MIN_RAWDATA_H +
                              EditorTheme::INSPECTOR_SPLITTER_H)) {
        float usableHeight = totalHeight - EditorTheme::INSPECTOR_SPLITTER_H;
        float propsHeight = usableHeight * m_propertiesRatio;
        float rawDataHeight = usableHeight - propsHeight;

        if (propsHeight < EditorTheme::INSPECTOR_MIN_PROPS_H) {
            propsHeight = EditorTheme::INSPECTOR_MIN_PROPS_H;
            rawDataHeight = usableHeight - propsHeight;
        }
        if (rawDataHeight < EditorTheme::INSPECTOR_MIN_RAWDATA_H) {
            rawDataHeight = EditorTheme::INSPECTOR_MIN_RAWDATA_H;
            propsHeight = usableHeight - rawDataHeight;
        }

        RenderPropertiesModule(ctx, propsHeight);
        RenderSplitter(ctx, totalHeight);
        RenderRawDataModule(ctx, rawDataHeight);
    } else {
        RenderPropertiesModule(ctx, 0.0f);
    }
}

// ============================================================================
// Properties module (top half)
// ============================================================================

void InspectorPanel::RenderPropertiesModule(InxGUIContext *ctx, float height)
{
    if (undoBeginFrame)
        undoBeginFrame();

    bool childHovered = false;
    bool childVisible = ImGui::BeginChild("PropertiesModule", ImVec2(0, height), ImGuiChildFlags_Borders);
    if (childVisible) {
        childHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_None);

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, EditorTheme::INSPECTOR_FRAME_PAD);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, EditorTheme::INSPECTOR_ITEM_SPC);

        bool multi = isMultiSelection && isMultiSelection();
        if (multi) {
            auto ids = getSelectedIds ? getSelectedIds() : std::vector<uint64_t>{};
            if (ids.size() > 1) {
                RenderMultiEdit(ctx, ids);
            } else {
                ImGui::TextUnformatted(Tr("inspector.no_objects_selected").c_str());
            }
        } else {
            if (m_selectedObjectId != 0) {
                RenderSingleObject(ctx, m_selectedObjectId);
            } else {
                ImGui::TextUnformatted(Tr("inspector.no_object_selected").c_str());
            }
        }

        ImGui::PopStyleVar(2);
    }
    ImGui::EndChild();

    bool anyActive = ImGui::IsAnyItemActive();
    if (undoEndFrame)
        undoEndFrame(anyActive);

    // Idle-skip counter
    if (childHovered || anyActive)
        m_idleFrames = 0;
    else
        ++m_idleFrames;

    // Drag-drop target for scripts on the whole PropertiesModule
    if (m_selectedObjectId != 0 && ImGui::BeginDragDropTarget()) {
        const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("SCRIPT_FILE");
        if (payload && handleScriptDrop) {
            std::string path(static_cast<const char *>(payload->Data), payload->DataSize);
            // Remove trailing null if present
            if (!path.empty() && path.back() == '\0')
                path.pop_back();
            handleScriptDrop(path);
        }
        ImGui::EndDragDropTarget();
    }
}

// ============================================================================
// Raw data module (bottom half)
// ============================================================================

void InspectorPanel::RenderRawDataModule(InxGUIContext *ctx, float height)
{
    bool childVisible = ImGui::BeginChild("RawDataModule", ImVec2(0, height), ImGuiChildFlags_Borders);
    if (childVisible) {
        if (!m_selectedFile.empty() && m_mode == InspectorMode::Asset) {
            if (renderAssetInspector)
                renderAssetInspector(ctx, m_selectedFile, m_assetCategory);
            else
                ImGui::TextUnformatted("(Asset inspector not available)");
        } else if (!m_selectedFile.empty()) {
            if (renderFilePreview)
                renderFilePreview(ctx, m_selectedFile);
            else
                ImGui::TextUnformatted(m_selectedFile.c_str());
        } else {
            ImGui::TextUnformatted(Tr("inspector.no_selection").c_str());
        }
    }
    ImGui::EndChild();
}

// ============================================================================
// Splitter
// ============================================================================

float InspectorPanel::RenderSplitter(InxGUIContext * /*ctx*/, float totalHeight)
{
    ImGui::Separator();

    float availWidth = ImGui::GetContentRegionAvail().x;
    ImGui::InvisibleButton("##InspectorSplitter", ImVec2(availWidth, EditorTheme::INSPECTOR_SPLITTER_H));

    bool hovered = ImGui::IsItemHovered();
    bool active = ImGui::IsItemActive();

    if (hovered || active)
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);

    if (active) {
        float deltaY = ImGui::GetMouseDragDelta(0).y;
        if (std::abs(deltaY) > 1.0f) {
            float usableHeight = totalHeight - EditorTheme::INSPECTOR_SPLITTER_H;
            if (usableHeight > 0.0f) {
                float newPropsHeight = m_propertiesRatio * usableHeight + deltaY;
                float newRatio = newPropsHeight / usableHeight;
                float minRatio = EditorTheme::INSPECTOR_MIN_PROPS_H / usableHeight;
                float maxRatio = 1.0f - (EditorTheme::INSPECTOR_MIN_RAWDATA_H / usableHeight);
                m_propertiesRatio = std::clamp(newRatio, minRatio, maxRatio);
            }
            ImGui::ResetMouseDragDelta(0);
        }
    }

    ImGui::Separator();
    return m_propertiesRatio;
}

// ============================================================================
// Single object inspector
// ============================================================================

void InspectorPanel::RenderSingleObject(InxGUIContext *ctx, uint64_t objId)
{
    using clock = std::chrono::high_resolution_clock;

    uint64_t valueGeneration = getValueGeneration ? getValueGeneration() : 0;
    bool refreshSnapshots = (m_cachedObjInfoId != objId) || (m_cachedComponentListObjId != objId) ||
                            (m_idleFrames <= 0) || (valueGeneration != m_cachedValueGeneration) ||
                            ((m_frameTimeNow - m_cachedValueRefreshTime) >= VALUE_CACHE_TTL);

    // ── Sub-timing: getObjectInfo + getPrefabInfo ────────────────────
    auto t0 = clock::now();

    if (refreshSnapshots) {
        if (getObjectInfo)
            m_cachedObjInfo = getObjectInfo(objId);
        else
            m_cachedObjInfo = {};
        m_cachedObjInfoId = objId;
        if (!m_cachedObjInfo.prefabGuid.empty() && getPrefabInfo)
            m_cachedPrefabInfo = getPrefabInfo(objId);
        else
            m_cachedPrefabInfo = {};
    }

    auto t1 = clock::now();
    m_subGetInfo += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const auto &info = m_cachedObjInfo;
    const auto &pinfo = m_cachedPrefabInfo;
    bool isPrefabReadonly = pinfo.isReadonly;
    bool isPrefabTransformReadonly = pinfo.isTransformReadonly;

    ImGui::PushID(static_cast<int>(objId));

    // Prefab header bar (if applicable)
    if (!info.prefabGuid.empty()) {
        RenderPrefabHeader(ctx, objId, pinfo);
    }

    // Active, Name, Tag, Layer are scene-instance properties — always editable,
    // even on prefab instances (they don't come from the prefab asset).

    // Object header: active checkbox + name input
    RenderObjectHeader(ctx, objId, info);

    // Tag & Layer
    RenderTagLayerRow(ctx, objId, info);

    ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_TITLE_GAP));
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP));

    // ── Sub-timing: Transform ────────────────────────────────────────
    auto t2 = clock::now();

    // Transform (skip for screen-space UI elements)
    if (!info.hideTransform) {
        if (m_cachedTransformIconId == 0 && getComponentIconId)
            m_cachedTransformIconId = getComponentIconId("Transform", false);
        auto [headerOpen, _unused] = RenderComponentHeader(ctx, "Transform", "transform", m_cachedTransformIconId,
                                                           /*showEnabled=*/false, /*isEnabled=*/true, /*suffix=*/"",
                                                           /*defaultOpen=*/true);

        if (headerOpen) {
            if (isPrefabTransformReadonly)
                ImGui::BeginDisabled();
            RenderTransform(ctx, objId);
            if (isPrefabTransformReadonly)
                ImGui::EndDisabled();
        }
    }

    auto t3 = clock::now();
    m_subTransform += std::chrono::duration<double, std::milli>(t3 - t2).count();

    // --- Prefab instance: disable everything below Transform ---
    if (isPrefabReadonly)
        ImGui::BeginDisabled();

    // ── Sub-timing: getComponentList ─────────────────────────────────
    auto t4 = clock::now();

    if (refreshSnapshots) {
        if (getComponentList)
            m_cachedComponents = getComponentList(objId);
        else
            m_cachedComponents.clear();
        m_cachedComponentListObjId = objId;
        m_cachedValueGeneration = valueGeneration;
        m_cachedValueRefreshTime = m_frameTimeNow;
    }

    const auto &components = m_cachedComponents;

    auto t5 = clock::now();
    m_subGetComponents += std::chrono::duration<double, std::milli>(t5 - t4).count();

    // ── Sub-timing: Component bodies ─────────────────────────────────
    auto t6 = clock::now();

    // Render each component
    for (const auto &comp : components) {
        ImGui::PushID(static_cast<int>(comp.componentId));

        auto [headerOpen, newEnabled] =
            RenderComponentHeader(ctx, comp.typeName, "comp_" + std::to_string(comp.componentId), comp.iconId,
                                  /*showEnabled=*/true, comp.enabled, comp.isScript ? " (Script)" : "",
                                  /*defaultOpen=*/true);

        // Right-click context menu — only call Python when popup is open
        bool componentRemoved = false;
        {
            const char *ctxPopupId = comp.isScript ? "py_comp_ctx" : "comp_ctx";
            if (ImGui::BeginPopupContextItem(ctxPopupId)) {
                if (renderComponentContextMenu) {
                    componentRemoved =
                        renderComponentContextMenu(ctx, objId, comp.typeName, comp.componentId, comp.isNative);
                }
                ImGui::EndPopup();
            }
        }

        if (!componentRemoved) {
            // Enabled toggle
            if (newEnabled != comp.enabled && setComponentEnabled) {
                setComponentEnabled(objId, comp.componentId, newEnabled, comp.isNative);
            }

            // Component body
            if (headerOpen) {
                if (comp.isBroken && !comp.brokenError.empty()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::ERROR_TEXT);
                    ImGui::TextWrapped("%s", comp.brokenError.c_str());
                    ImGui::PopStyleColor();
                } else if (renderComponentBody) {
                    renderComponentBody(ctx, objId, comp.typeName, comp.componentId, comp.isNative);
                }
            }
        }

        ImGui::PopID();
    }

    // Add Component button + popup
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP));
    RenderAddComponentButton(ctx);
    ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP));
    RenderAddComponentPopup(ctx);

    auto t7 = clock::now();
    m_subComponentBodies += std::chrono::duration<double, std::milli>(t7 - t6).count();

    // ── Sub-timing: Material sections ────────────────────────────────
    auto t8 = clock::now();

    // Material override sections
    if (renderMaterialSections)
        renderMaterialSections(ctx, objId);

    auto t9 = clock::now();
    m_subMaterials += std::chrono::duration<double, std::milli>(t9 - t8).count();

    if (isPrefabReadonly)
        ImGui::EndDisabled();

    ImGui::PopID();
}

// ============================================================================
// Multi-edit inspector
// ============================================================================

void InspectorPanel::RenderMultiEdit(InxGUIContext *ctx, const std::vector<uint64_t> &ids)
{
    int n = static_cast<int>(ids.size());
    ImGui::PushID("multi_edit");

    ImGui::Text("%d objects selected", n);

    // For multi-edit, we render Transform for the first object as primary
    // and component body rendering is delegated to Python callbacks
    if (!ids.empty()) {
        if (getObjectInfo)
            m_cachedObjInfo = getObjectInfo(ids[0]);

        ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_TITLE_GAP));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP));

        // Transform header
        uint64_t transformIcon = getComponentIconId ? getComponentIconId("Transform", false) : 0;
        auto [headerOpen, _unused] = RenderComponentHeader(ctx, "Transform", "multi_transform", transformIcon,
                                                           /*showEnabled=*/false, /*isEnabled=*/true, /*suffix=*/"",
                                                           /*defaultOpen=*/true);

        if (headerOpen) {
            // Render transform for primary object
            RenderTransform(ctx, ids[0]);
        }

        // Common components — delegated to Python callback which handles
        // intersection logic and multi-object display
        std::vector<ComponentInfo> components;
        if (getComponentList)
            components = getComponentList(ids[0]);

        for (const auto &comp : components) {
            ImGui::PushID(static_cast<int>(comp.componentId));

            uint64_t iconId = getComponentIconId ? getComponentIconId(comp.typeName, comp.isScript) : 0;

            auto [compOpen, newEnabled] =
                RenderComponentHeader(ctx, comp.typeName, "multi_comp_" + std::to_string(comp.componentId), iconId,
                                      true, comp.enabled, comp.isScript ? " (Script)" : "", true);

            if (newEnabled != comp.enabled && setComponentEnabled)
                setComponentEnabled(ids[0], comp.componentId, newEnabled, comp.isNative);

            if (compOpen && renderComponentBody)
                renderComponentBody(ctx, ids[0], comp.typeName, comp.componentId, comp.isNative);

            ImGui::PopID();
        }

        // Add Component
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP));
        RenderAddComponentButton(ctx);
        ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP));
        RenderAddComponentPopup(ctx);
    }

    ImGui::PopID();
}

// ============================================================================
// Object header (active + name)
// ============================================================================

void InspectorPanel::RenderObjectHeader(InxGUIContext *ctx, uint64_t objId, const ObjectInfo &info)
{
    // Active checkbox
    bool active = info.active;
    bool newActive = RenderInspectorCheckbox(ctx, "##obj_active", active);
    if (newActive != active && setObjectProperty) {
        m_cachedObjInfo.active = newActive;
        setObjectProperty(objId, "active", newActive ? "true" : "false");
    }

    ImGui::SameLine(0, 6);

    // Editable name
    ImGui::SetNextItemWidth(-1);
    char nameBuf[256];
    std::strncpy(nameBuf, info.name.c_str(), sizeof(nameBuf) - 1);
    nameBuf[sizeof(nameBuf) - 1] = '\0';
    if (ImGui::InputText("##obj_name", nameBuf, sizeof(nameBuf))) {
        std::string newName(nameBuf);
        if (newName != info.name && setObjectProperty) {
            m_cachedObjInfo.name = newName;
            setObjectProperty(objId, "name", newName);
        }
    }
}

// ============================================================================
// Tag & Layer row
// ============================================================================

void InspectorPanel::RefreshTagLayerCache()
{
    if (m_frameTimeNow - m_tagLayerCacheTime < TAG_LAYER_CACHE_TTL && !m_cachedTagItems.empty() &&
        !m_cachedLayerItems.empty())
        return;

    m_tagLayerCacheTime = m_frameTimeNow;

    if (getAllTags) {
        m_cachedTags = getAllTags();
        m_cachedTagItems = m_cachedTags;
        m_cachedTagItems.push_back("Add Tag...");
    }

    if (getAllLayers) {
        m_cachedLayers = getAllLayers();
        m_cachedLayerItems.clear();
        for (size_t i = 0; i < m_cachedLayers.size(); ++i) {
            std::string label = std::to_string(i) + ": ";
            label += m_cachedLayers[i].empty() ? "---" : m_cachedLayers[i];
            m_cachedLayerItems.push_back(std::move(label));
        }
        m_cachedLayerItems.push_back("Add Layer...");
    }
}

void InspectorPanel::RenderTagLayerRow(InxGUIContext *ctx, uint64_t objId, const ObjectInfo &info)
{
    RefreshTagLayerCache();

    float availW = ImGui::GetContentRegionAvail().x;
    float halfW = availW * 0.5f - 4.0f;

    // --- Tag (left column) ---
    ImGui::TextUnformatted(Tr("inspector.tag").c_str());
    ImGui::SameLine(0, 4.0f);

    int tagIdx = 0;
    for (size_t i = 0; i < m_cachedTags.size(); ++i) {
        if (m_cachedTags[i] == info.tag) {
            tagIdx = static_cast<int>(i);
            break;
        }
    }

    int newTagIdx = SearchableCombo(ctx, "Tag", tagIdx, m_cachedTagItems, halfW - 30);
    if (newTagIdx != tagIdx) {
        if (newTagIdx == static_cast<int>(m_cachedTags.size())) {
            // "Add Tag..." selected
            if (openWindow)
                openWindow("tag_layer_settings");
        } else if (newTagIdx >= 0 && newTagIdx < static_cast<int>(m_cachedTags.size())) {
            m_cachedObjInfo.tag = m_cachedTags[newTagIdx];
            if (setObjectProperty)
                setObjectProperty(objId, "tag", m_cachedTags[newTagIdx]);
        }
    }

    // --- Layer (right column) ---
    ImGui::SameLine(halfW + 8);
    ImGui::TextUnformatted(Tr("inspector.layer").c_str());
    ImGui::SameLine(0, 4.0f);
    float layerComboW = ImGui::GetContentRegionAvail().x;

    int newLayer = SearchableCombo(ctx, "Layer", info.layer, m_cachedLayerItems, layerComboW);
    if (newLayer != info.layer) {
        if (newLayer == static_cast<int>(m_cachedLayerItems.size()) - 1) {
            // "Add Layer..." selected
            if (openWindow)
                openWindow("tag_layer_settings");
        } else if (setObjectProperty) {
            m_cachedObjInfo.layer = newLayer;
            setObjectProperty(objId, "layer", std::to_string(newLayer));
        }
    }
}

// ============================================================================
// Transform rendering
// ============================================================================

void InspectorPanel::RenderTransform(InxGUIContext *ctx, uint64_t objId)
{
    if (!getTransformData)
        return;

    TransformData td = getTransformData(objId);
    float labelW = EditorTheme::INSPECTOR_MIN_LABEL_WIDTH;

    // Vector3Control modifies the array in-place
    float pos[3] = {td.px, td.py, td.pz};
    float rot[3] = {td.rx, td.ry, td.rz};
    float scl[3] = {td.sx, td.sy, td.sz};

    ctx->Vector3Control(Tr("Position"), pos, DRAG_SPEED_DEFAULT, labelW);
    ctx->Vector3Control(Tr("Rotation"), rot, DRAG_SPEED_DEFAULT, labelW);
    ctx->Vector3Control(Tr("Scale"), scl, DRAG_SPEED_FINE, labelW);

    if (setTransformData) {
        bool changed = false;
        changed |= std::abs(pos[0] - td.px) > 1e-6f;
        changed |= std::abs(pos[1] - td.py) > 1e-6f;
        changed |= std::abs(pos[2] - td.pz) > 1e-6f;
        changed |= std::abs(rot[0] - td.rx) > 1e-6f;
        changed |= std::abs(rot[1] - td.ry) > 1e-6f;
        changed |= std::abs(rot[2] - td.rz) > 1e-6f;
        changed |= std::abs(scl[0] - td.sx) > 1e-6f;
        changed |= std::abs(scl[1] - td.sy) > 1e-6f;
        changed |= std::abs(scl[2] - td.sz) > 1e-6f;

        if (changed) {
            TransformData newTd;
            newTd.px = pos[0];
            newTd.py = pos[1];
            newTd.pz = pos[2];
            newTd.rx = rot[0];
            newTd.ry = rot[1];
            newTd.rz = rot[2];
            newTd.sx = scl[0];
            newTd.sy = scl[1];
            newTd.sz = scl[2];
            setTransformData(objId, newTd);
        }
    }
}

// ============================================================================
// Prefab header
// ============================================================================

void InspectorPanel::RenderPrefabHeader(InxGUIContext * /*ctx*/, uint64_t objId, const PrefabInfo &pinfo)
{
    ImGui::Dummy(ImVec2(0, 4));

    ImGui::PushStyleColor(ImGuiCol_ChildBg, EditorTheme::PREFAB_HEADER_BG);
    ImGui::BeginChild("##prefab_header_bar", ImVec2(0, EditorTheme::PREFAB_HEADER_H), ImGuiChildFlags_Borders);

    ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::PREFAB_TEXT);
    ImGui::TextUnformatted(Tr("inspector.prefab_label").c_str());
    ImGui::PopStyleColor();

    float gap = EditorTheme::PREFAB_HEADER_BTN_GAP;

    ImGui::SameLine(0, gap * 2);
    EditorTheme::PushFlatButtonStyle(EditorTheme::INSPECTOR_INLINE_BTN_IDLE);
    if (ImGui::Button(Tr("inspector.prefab_select").c_str())) {
        if (prefabAction)
            prefabAction(objId, "select");
    }
    ImGui::PopStyleColor(3);

    ImGui::SameLine(0, gap);
    EditorTheme::PushFlatButtonStyle(EditorTheme::INSPECTOR_INLINE_BTN_IDLE);
    if (ImGui::Button(Tr("inspector.prefab_open").c_str())) {
        if (prefabAction)
            prefabAction(objId, "open");
    }
    ImGui::PopStyleColor(3);

    ImGui::SameLine(0, gap * 3);

    if (pinfo.overrideCount > 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::WARNING_TEXT);
        ImGui::Text("%d %s", pinfo.overrideCount, Tr("inspector.overrides").c_str());
        ImGui::PopStyleColor();

        ImGui::SameLine(0, gap * 2);
        EditorTheme::PushFlatButtonStyle(EditorTheme::INSPECTOR_INLINE_BTN_IDLE);
        if (ImGui::Button(Tr("inspector.prefab_apply").c_str())) {
            if (prefabAction)
                prefabAction(objId, "apply");
        }
        ImGui::PopStyleColor(3);

        ImGui::SameLine(0, gap);
        EditorTheme::PushFlatButtonStyle(EditorTheme::INSPECTOR_INLINE_BTN_IDLE);
        if (ImGui::Button(Tr("inspector.prefab_revert").c_str())) {
            if (prefabAction)
                prefabAction(objId, "revert");
        }
        ImGui::PopStyleColor(3);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::TEXT_DIM2);
        ImGui::TextUnformatted(Tr("inspector.no_overrides").c_str());
        ImGui::PopStyleColor();
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

// ============================================================================
// Component header (icon + checkbox + collapsing header)
// ============================================================================

std::pair<bool, bool> InspectorPanel::RenderComponentHeader(InxGUIContext * /*ctx*/, const std::string &typeName,
                                                            const std::string &headerId, uint64_t iconId,
                                                            bool showEnabled, bool isEnabled, const std::string &suffix,
                                                            bool defaultOpen)
{
    bool newEnabled = isEnabled;

    // Build display name: insert spaces before uppercase chars
    std::string displayName;
    for (size_t i = 0; i < typeName.size(); ++i) {
        char c = typeName[i];
        if (i > 0 && std::isupper(static_cast<unsigned char>(c))) {
            char prev = typeName[i - 1];
            if (!std::isupper(static_cast<unsigned char>(prev)) && prev != ' ')
                displayName += ' ';
        }
        displayName += c;
    }
    displayName += suffix;

    // Styling
    ImGui::PushStyleColor(ImGuiCol_Header, EditorTheme::INSPECTOR_HEADER_PRIMARY);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, EditorTheme::INSPECTOR_HEADER_PRIMARY_HOVERED);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, EditorTheme::INSPECTOR_HEADER_PRIMARY_ACTIVE);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, EditorTheme::INSPECTOR_HEADER_PRIMARY_FRAME_PAD);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, EditorTheme::INSPECTOR_HEADER_ITEM_SPC);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, EditorTheme::INSPECTOR_HEADER_BORDER_SIZE);
    ImGui::SetWindowFontScale(EditorTheme::INSPECTOR_HEADER_PRIMARY_FONT_SCALE);

    // Full-width collapsing header
    if (defaultOpen)
        ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextItemAllowOverlap();

    std::string headerKey = "##comp_" + headerId;
    {
        float clipMaxX = ImGui::GetWindowPos().x + ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x -
                         EditorTheme::INSPECTOR_HEADER_RIGHT_MARGIN;
        ImGui::GetWindowDrawList()->PushClipRect(ImVec2(0.0f, 0.0f), ImVec2(clipMaxX, 1e7f), true);
    }
    bool headerOpen = ImGui::CollapsingHeader(headerKey.c_str());
    ImGui::GetWindowDrawList()->PopClipRect();

    float headerMinY = ImGui::GetItemRectMin().y;
    float headerMaxY = ImGui::GetItemRectMax().y;
    float headerHeight = (std::max)(0.0f, headerMaxY - headerMinY);

    // Overlay: icon + checkbox + label on the same row
    float indent = EditorTheme::INSPECTOR_HEADER_CONTENT_INDENT;
    ImGui::SameLine(indent, 0);

    if (iconId != 0) {
        float iconSize = EditorTheme::COMPONENT_ICON_SIZE;
        ImGui::Dummy(ImVec2(iconSize, (std::max)(headerHeight, iconSize)));
        ImVec2 slotMin = ImGui::GetItemRectMin();
        ImVec2 slotMax = ImGui::GetItemRectMax();
        float drawSize = (std::min)({iconSize, slotMax.x - slotMin.x, slotMax.y - slotMin.y});
        float drawX = slotMin.x + (std::max)(0.0f, (slotMax.x - slotMin.x - drawSize) * 0.5f);
        float drawY = slotMin.y + (std::max)(0.0f, (slotMax.y - slotMin.y - drawSize) * 0.5f);

        ImDrawList *drawList = ImGui::GetWindowDrawList();
        ImTextureRef texRef(static_cast<ImTextureID>(iconId));
        drawList->AddImage(texRef, ImVec2(drawX, drawY), ImVec2(drawX + drawSize, drawY + drawSize));

        ImGui::SameLine(0, EditorTheme::INSPECTOR_HEADER_ITEM_SPC.x);
    }

    if (showEnabled) {
        newEnabled = RenderInspectorCheckbox(nullptr, "##hdr_en", isEnabled);
        ImGui::SameLine(0, EditorTheme::INSPECTOR_HEADER_ITEM_SPC.x);
    }

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(displayName.c_str());

    // Cleanup
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(3);

    return {headerOpen, newEnabled};
}

// ============================================================================
// Inspector checkbox
// ============================================================================

bool InspectorPanel::RenderInspectorCheckbox(InxGUIContext * /*ctx*/, const char *label, bool value)
{
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, EditorTheme::INSPECTOR_CHECKBOX_FRAME_PAD);
    ImGui::SetWindowFontScale(EditorTheme::INSPECTOR_CHECKBOX_FONT_SCALE);
    bool newValue = value;
    ImGui::Checkbox(label, &newValue);
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleVar();
    return newValue;
}

// ============================================================================
// Add Component button + popup
// ============================================================================

void InspectorPanel::RenderAddComponentButton(InxGUIContext * /*ctx*/)
{
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, EditorTheme::ADD_COMP_FRAME_PAD);
    ImGui::SetCursorPosX(EditorTheme::INSPECTOR_ACTION_ALIGN_X);
    if (ImGui::Button(Tr("inspector.add_component").c_str(), ImVec2(-1, 0))) {
        m_addCompSearch[0] = '\0';
        if (getAddComponentEntries)
            m_addCompEntries = getAddComponentEntries();
        ImGui::OpenPopup("##add_component_popup");
    }
    ImGui::PopStyleVar();
}

void InspectorPanel::RenderAddComponentPopup(InxGUIContext * /*ctx*/)
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, EditorTheme::POPUP_ADD_COMP_PAD);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, EditorTheme::POPUP_ADD_COMP_SPC);

    if (ImGui::BeginPopup("##add_component_popup")) {
        // Search field
        ImGui::SetNextItemWidth(EditorTheme::ADD_COMP_SEARCH_W);
        ImGui::InputTextWithHint("##comp_search", Tr("inspector.search_components").c_str(), m_addCompSearch,
                                 sizeof(m_addCompSearch));

        ImGui::Separator();

        // Scrollable region
        if (ImGui::BeginChild("##comp_list", ImVec2(0, 350), ImGuiChildFlags_None)) {
            std::string searchLower(m_addCompSearch);
            std::transform(searchLower.begin(), searchLower.end(), searchLower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            bool foundAny = false;
            int uid = 0;

            // Group entries by category
            std::unordered_map<std::string, std::vector<const AddComponentEntry *>> categories;
            std::vector<std::string> categoryOrder;

            for (const auto &entry : m_addCompEntries) {
                // Filter by search
                if (!searchLower.empty()) {
                    std::string nameLower = entry.displayName;
                    std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    if (nameLower.find(searchLower) == std::string::npos)
                        continue;
                }

                std::string cat = entry.category.empty() ? "Miscellaneous" : entry.category;
                if (categories.find(cat) == categories.end())
                    categoryOrder.push_back(cat);
                categories[cat].push_back(&entry);
            }

            std::sort(categoryOrder.begin(), categoryOrder.end());

            for (const auto &cat : categoryOrder) {
                ImGui::TextUnformatted(cat.c_str());
                ImGui::Separator();

                for (const auto *entry : categories[cat]) {
                    foundAny = true;
                    ++uid;
                    std::string selectLabel = "  " + entry->displayName + "##" + std::to_string(uid);
                    if (ImGui::Selectable(selectLabel.c_str())) {
                        if (addComponent)
                            addComponent(entry->displayName, entry->isNative, entry->scriptPath);
                        ImGui::CloseCurrentPopup();
                    }
                }
                ImGui::Dummy(ImVec2(0, 4));
            }

            if (!foundAny) {
                ImGui::TextUnformatted(Tr("inspector.no_components_found").c_str());
            }
        }
        ImGui::EndChild();
        ImGui::EndPopup();
    }

    ImGui::PopStyleVar(2);
}

// ============================================================================
// Searchable combo helper
// ============================================================================

int InspectorPanel::SearchableCombo(InxGUIContext * /*ctx*/, const char *label, int currentIdx,
                                    const std::vector<std::string> &items, float width)
{
    if (items.empty())
        return currentIdx;

    int safeIdx = (currentIdx >= 0 && currentIdx < static_cast<int>(items.size())) ? currentIdx : 0;
    const std::string &currentText = items[safeIdx];

    std::string comboId = std::string("##combo_") + label;
    std::string popupId = std::string("##combopop_") + label;

    if (width > 0.0f)
        ImGui::SetNextItemWidth(width);

    if (ImGui::Button((currentText + "##" + label).c_str(), ImVec2(width > 0.0f ? width : 0.0f, 0))) {
        auto &state = m_comboStates[label];
        state.filter[0] = '\0';
        state.needsFocus = true;
        ImGui::OpenPopup(popupId.c_str());
    }

    int result = currentIdx;

    if (ImGui::BeginPopup(popupId.c_str())) {
        auto &state = m_comboStates[label];

        if (state.needsFocus) {
            ImGui::SetKeyboardFocusHere();
            state.needsFocus = false;
        }

        ImGui::SetNextItemWidth(EditorTheme::ADD_COMP_SEARCH_W);
        ImGui::InputTextWithHint("##filter", "Filter...", state.filter, sizeof(state.filter));

        std::string filterLower(state.filter);
        std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        for (int i = 0; i < static_cast<int>(items.size()); ++i) {
            if (!filterLower.empty()) {
                std::string itemLower = items[i];
                std::transform(itemLower.begin(), itemLower.end(), itemLower.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (itemLower.find(filterLower) == std::string::npos)
                    continue;
            }

            bool selected = (i == currentIdx);
            if (ImGui::Selectable(items[i].c_str(), selected)) {
                result = i;
                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::EndPopup();
    }

    return result;
}

} // namespace infernux
