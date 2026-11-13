#include <core/log/InxLog.h>
#include <function/editor/EditorObjectIcon.h>
#include <function/editor/EditorThemeRegistry.h>
#include <function/editor/InspectorPanel.h>
#include <function/renderer/ProfileConfig.h>
#include <function/renderer/gui/InxGUISemantics.h>
#include <platform/filesystem/InxPath.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <unordered_map>
#include <utility>

namespace
{
constexpr uint32_t DRIVEN_TRANSFORM_POSITION = 1u << 0u;
constexpr uint32_t DRIVEN_TRANSFORM_ROTATION = 1u << 1u;
constexpr uint32_t DRIVEN_TRANSFORM_SCALE = 1u << 2u;

template <typename Callback> class InspectorScopeExit final
{
  public:
    explicit InspectorScopeExit(Callback callback) : m_callback(std::move(callback))
    {
    }
    InspectorScopeExit(const InspectorScopeExit &) = delete;
    InspectorScopeExit &operator=(const InspectorScopeExit &) = delete;
    ~InspectorScopeExit()
    {
        m_callback();
    }

  private:
    Callback m_callback;
};

template <typename Callback> InspectorScopeExit<Callback> MakeInspectorScopeExit(Callback callback)
{
    return InspectorScopeExit<Callback>(std::move(callback));
}

const char *InspectorComponentDomain(const std::string &identity)
{
    static const std::unordered_map<std::string, const char *> domains{
        {"Transform", "DOMAIN_CORE"},
        {"Camera", "DOMAIN_GEOMETRY"},
        {"Light", "DOMAIN_LIGHT"},
        {"MeshRenderer", "DOMAIN_GEOMETRY"},
        {"SkinnedMeshRenderer", "DOMAIN_GEOMETRY"},
        {"SpriteRenderer", "DOMAIN_GEOMETRY"},
        {"LineRenderer", "DOMAIN_GEOMETRY"},
        {"TrailRenderer", "DOMAIN_GEOMETRY"},
        {"ParticleSystem", "DOMAIN_VFX"},
        {"RenderStack", "DOMAIN_POST_FX"},
        {"NeuralPolicy", "DOMAIN_NEURAL"},
        {"Rigidbody", "DOMAIN_PHYSICS"},
        {"Collider", "DOMAIN_PHYSICS"},
        {"BoxCollider", "DOMAIN_PHYSICS"},
        {"SphereCollider", "DOMAIN_PHYSICS"},
        {"CapsuleCollider", "DOMAIN_PHYSICS"},
        {"CylinderCollider", "DOMAIN_PHYSICS"},
        {"MeshCollider", "DOMAIN_PHYSICS"},
        {"HingeJoint", "DOMAIN_PHYSICS"},
        {"SliderJoint", "DOMAIN_PHYSICS"},
        {"FixedJoint", "DOMAIN_PHYSICS"},
        {"DistanceJoint", "DOMAIN_PHYSICS"},
        {"SwingTwistJoint", "DOMAIN_PHYSICS"},
        {"SixDofJoint", "DOMAIN_PHYSICS"},
        {"Ragdoll", "DOMAIN_PHYSICS"},
        {"SkeletalAnimator", "DOMAIN_ANIMATION"},
        {"SkeletonInstance", "DOMAIN_ANIMATION"},
        {"SpriteAnimator", "DOMAIN_ANIMATION"},
        {"TimelineAction", "DOMAIN_ANIMATION"},
        {"AudioSource", "DOMAIN_AUDIO"},
        {"AudioListener", "DOMAIN_AUDIO"},
    };
    const auto domain = domains.find(identity);
    return domain != domains.end() ? domain->second : identity.rfind("UI", 0) == 0 ? "DOMAIN_UI" : "DOMAIN_PYTHON";
}

class InspectorComponentCard
{
  public:
    InspectorComponentCard(infernux::InxGUIContext *context, const std::string &identity) : m_context(context)
    {
        m_color = infernux::EditorThemeRegistry::Color(InspectorComponentDomain(identity));
        const auto base = infernux::EditorThemeRegistry::Color("ROLE_BG_SURFACE");
        const ImVec4 background(base.x + (m_color.x - base.x) * 0.06f, base.y + (m_color.y - base.y) * 0.06f,
                                base.z + (m_color.z - base.z) * 0.06f, 1.0f);
        m_context->BeginFramedGroup(0, 0, 0, {background.x, background.y, background.z, 1}, {0, 0, 0, 0});
    }
    ~InspectorComponentCard()
    {
        m_context->EndFramedGroup();
        const auto minimum = ImGui::GetItemRectMin(), maximum = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddRectFilled(minimum, ImVec2(minimum.x + 3 * m_context->GetDpiScale(), maximum.y),
                                                  ImGui::GetColorU32(m_color));
    }

  private:
    infernux::InxGUIContext *m_context;
    ImVec4 m_color;
};

class InspectorComponentBody
{
  public:
    explicit InspectorComponentBody(float inset)
        : m_window(ImGui::GetCurrentWindow()), m_inset(inset), m_workMax(m_window->WorkRect.Max.x),
          m_contentMax(m_window->ContentRegionRect.Max.x)
    {
        ImGui::Indent(m_inset);
        m_window->WorkRect.Max.x -= m_inset;
        m_window->ContentRegionRect.Max.x -= m_inset;
        ImGui::PushItemWidth(-FLT_MIN);
    }
    ~InspectorComponentBody()
    {
        ImGui::PopItemWidth();
        m_window->WorkRect.Max.x = m_workMax;
        m_window->ContentRegionRect.Max.x = m_contentMax;
        ImGui::Unindent(m_inset);
    }

  private:
    ImGuiWindow *m_window;
    float m_inset, m_workMax, m_contentMax;
};

static std::string MakeComponentAddCommandArgument(const std::string &typeName, bool isNative,
                                                   const std::string &scriptPath, uint64_t targetComponentId = 0,
                                                   bool insertAfter = false, bool insertAtStart = false)
{
    return nlohmann::json{{"type_name", typeName},       {"is_native", isNative},
                          {"script_path", scriptPath},   {"target_component_id", targetComponentId},
                          {"insert_after", insertAfter}, {"insert_at_start", insertAtStart}}
        .dump();
}

static std::string MakeComponentEnabledCommandArgument(const std::vector<uint64_t> &objectIds,
                                                       const std::vector<uint64_t> &componentIds, bool enabled,
                                                       bool isNative)
{
    nlohmann::json targets = nlohmann::json::array();
    const size_t count = (std::min)(objectIds.size(), componentIds.size());
    for (size_t index = 0; index < count; ++index)
        targets.push_back({{"object_id", objectIds[index]}, {"component_id", componentIds[index]}});
    return nlohmann::json{{"targets", std::move(targets)}, {"enabled", enabled}, {"is_native", isNative}}.dump();
}

static std::string MakeComponentEnabledCommandArgument(uint64_t objectId, uint64_t componentId, bool enabled,
                                                       bool isNative)
{
    return MakeComponentEnabledCommandArgument(std::vector<uint64_t>{objectId}, std::vector<uint64_t>{componentId},
                                               enabled, isNative);
}

static std::string MakeObjectPropertyCommandArgument(uint64_t objectId, const char *property,
                                                     const nlohmann::json &value)
{
    return nlohmann::json{{"object_id", objectId}, {"property", property}, {"value", value}}.dump();
}

static nlohmann::json MakeTransformValue(float px, float py, float pz, float rx, float ry, float rz, float sx, float sy,
                                         float sz)
{
    return nlohmann::json{{"position", {px, py, pz}}, {"rotation", {rx, ry, rz}}, {"scale", {sx, sy, sz}}};
}

static uint32_t CaptureInspectorEditLifecycle(bool changed)
{
    uint32_t flags = changed ? infernux::InxGUIContext::EditChanged : 0u;
    if (ImGui::IsItemActive())
        flags |= infernux::InxGUIContext::EditActive;
    if (ImGui::IsItemActivated())
        flags |= infernux::InxGUIContext::EditActivated;
    if (ImGui::IsItemDeactivatedAfterEdit())
        flags |= infernux::InxGUIContext::EditDeactivatedAfterEdit;
    if (ImGui::IsItemDeactivated())
        flags |= infernux::InxGUIContext::EditDeactivated;
    return flags;
}

static std::string MakeComponentReorderCommandArgument(const std::vector<uint64_t> &objectIds,
                                                       const std::vector<uint64_t> &draggedComponentIds,
                                                       const std::vector<uint64_t> &targetComponentIds,
                                                       bool insertAfter)
{
    return nlohmann::json{{"object_ids", objectIds},
                          {"dragged_component_ids", draggedComponentIds},
                          {"target_component_ids", targetComponentIds},
                          {"insert_after", insertAfter}}
        .dump();
}
} // namespace

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

bool InspectorPanel::ExecuteEditorCommand(const std::string &commandId, const std::string &argument,
                                          const std::string &source) const
{
    return executeCommand && executeCommand(commandId, source, argument);
}

bool InspectorPanel::CanExecuteEditorCommand(const std::string &commandId, const std::string &argument) const
{
    return canExecuteCommand && canExecuteCommand(commandId, argument);
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

void InspectorPanel::InvalidateObjectCaches()
{
    // Field/selection callbacks run synchronously inside rendering. The active
    // frame still borrows these component and transform snapshots; invalidate
    // their keys now and replace their payloads on the next draw.
    m_cachedObjInfoId = 0;
    m_cachedComponentListObjId = 0;
    m_cachedObjectTargetRevision = 0;
    m_cachedObjectValueRevision = 0;
    m_cachedComponentTargetRevision = 0;
    m_cachedComponentSchemaRevision = 0;
    m_cachedTransformObjId = 0;
    m_cachedTransformValueRevision = 0;
    m_transformGestureIds = {};

    m_cachedMultiComponentsValid = false;
    m_cachedMultiComponentTargetRevision = 0;
    m_cachedMultiComponentSchemaRevision = 0;

    m_cachedMultiTransformValid = false;
    m_cachedMultiTransformTargetRevision = 0;
    m_cachedMultiTransformValueRevision = 0;
}

// ============================================================================
// Public API
// ============================================================================

void InspectorPanel::SetSelectedObjectId(uint64_t id)
{
    if (m_selectedObjectId != id)
        InvalidateObjectCaches();
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
    InvalidateObjectCaches();
    m_mode = InspectorMode::Object;
}

void InspectorPanel::SetSelectedFile(const std::string &filePath, const std::string &category)
{
    if (AssetPathKey(filePath) != AssetPathKey(m_selectedFile)) {
        m_selectedFile = filePath;
        InvalidateObjectCaches();
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
    InvalidateObjectCaches();
    m_mode = InspectorMode::Object;
}

void InspectorPanel::ShowDiagnostic()
{
    m_selectedObjectId = 0;
    m_selectedFile.clear();
    m_assetCategory.clear();
    InvalidateObjectCaches();
    m_mode = InspectorMode::Diagnostic;
}

void InspectorPanel::SetDetailFile(const std::string &filePath, const std::string &category)
{
    if (AssetPathKey(filePath) != AssetPathKey(m_selectedFile)) {
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

void InspectorPanel::SetSelectedComponentIds(const std::vector<uint64_t> &componentIds)
{
    m_selectedComponentIds.clear();
    for (const uint64_t componentId : componentIds) {
        if (componentId != 0)
            m_selectedComponentIds.insert(componentId);
    }
}

void InspectorPanel::ClearSelectedComponents()
{
    m_selectedComponentIds.clear();
}

bool InspectorPanel::IsComponentSelected(uint64_t componentId) const
{
    return componentId != 0 && m_selectedComponentIds.find(componentId) != m_selectedComponentIds.end();
}

bool InspectorPanel::AreComponentsSelected(const std::vector<uint64_t> &componentIds) const
{
    return !componentIds.empty() && std::all_of(componentIds.begin(), componentIds.end(), [this](uint64_t componentId) {
        return IsComponentSelected(componentId);
    });
}

// ============================================================================
// VisiblePreRender
// ============================================================================

void InspectorPanel::VisiblePreRender(InxGUIContext * /*ctx*/)
{
    auto now = std::chrono::steady_clock::now();
    m_frameTimeNow = std::chrono::duration<float>(now.time_since_epoch()).count();
}

// ============================================================================
// OnRenderContent — main entry
// ============================================================================

void InspectorPanel::OnRenderContent(InxGUIContext *ctx)
{
    if (m_mode == InspectorMode::Diagnostic) {
        if (renderDiagnosticInspector)
            renderDiagnosticInspector(ctx);
        return;
    }
    const float dpi = ctx->GetDpiScale();
    if (std::abs(dpi - m_lastDpiScale) >= 0.01f) {
        m_lastDpiScale = dpi;
        m_cachedTransformBodyHeight = 0.0f;
        m_cachedComponentBodyHeights.clear();
    }
    if (!getRevisionSnapshot)
        throw std::runtime_error("Inspector revision snapshot callback is required");
    if (m_selectedObjectId != 0 || !m_selectedFile.empty()) {
        try {
            m_frameRevisions = getRevisionSnapshot();
            m_revisionSnapshotErrorReported = false;
        } catch (const std::exception &error) {
            // Keep the preceding immutable packet. A transient Python retirement
            // race must not abort DrawFrame or read stale component lifetimes.
            if (!m_revisionSnapshotErrorReported) {
                INXLOG_ERROR("Inspector revision snapshot failed; retaining the previous packet: ", error.what());
                m_revisionSnapshotErrorReported = true;
            }
        }
    }

    float totalHeight = ImGui::GetContentRegionAvail().y;

    bool hasDetailContent = !m_selectedFile.empty();
    bool fileOnly = hasDetailContent && m_selectedObjectId == 0;

    if (fileOnly) {
        // Full-height file view (asset inspector or generic preview)
        RenderRawDataModule(ctx, 0.0f);
    } else if (hasDetailContent &&
               totalHeight > (EditorTheme::INSPECTOR_MIN_PROPS_H + EditorTheme::INSPECTOR_MIN_RAWDATA_H +
                              EditorTheme::INSPECTOR_SPLITTER_H) *
                                 dpi) {
        float usableHeight = totalHeight - EditorTheme::INSPECTOR_SPLITTER_H * dpi;
        float propsHeight = usableHeight * m_propertiesRatio;
        float rawDataHeight = usableHeight - propsHeight;

        if (propsHeight < EditorTheme::INSPECTOR_MIN_PROPS_H * dpi) {
            propsHeight = EditorTheme::INSPECTOR_MIN_PROPS_H * dpi;
            rawDataHeight = usableHeight - propsHeight;
        }
        if (rawDataHeight < EditorTheme::INSPECTOR_MIN_RAWDATA_H * dpi) {
            rawDataHeight = EditorTheme::INSPECTOR_MIN_RAWDATA_H * dpi;
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
    const float dpi = ctx->GetDpiScale();
    {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, EditorThemeRegistry::Color("ROLE_BG_BASE"));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        bool childVisible =
            ImGui::BeginChild("PropertiesModule", ImVec2(0, height), ImGuiChildFlags_AlwaysUseWindowPadding,
                              ImGuiWindowFlags_AlwaysVerticalScrollbar);
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
        auto childGuard = MakeInspectorScopeExit([] { ImGui::EndChild(); });
        if (childVisible) {
            const ImVec2 fieldPadding = EditorThemeRegistry::Vec2("INSPECTOR_FRAME_PAD");
            const ImVec2 fieldSpacing = EditorThemeRegistry::Vec2("INSPECTOR_ITEM_SPC");
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(fieldPadding.x * dpi, fieldPadding.y * dpi));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(fieldSpacing.x * dpi, fieldSpacing.y * dpi));
            auto styleGuard = MakeInspectorScopeExit([] { ImGui::PopStyleVar(2); });

            bool multi = isMultiSelection && isMultiSelection();
            if (multi) {
                auto ids = getSelectedIds ? getSelectedIds() : std::vector<uint64_t>{};
                if (ids.size() > 1) {
                    RenderMultiEdit(ctx, ids);
                } else {
                    const ImVec2 avail = ImGui::GetContentRegionAvail();
                    const char *hint = Tr("inspector.no_objects_selected").c_str();
                    const ImVec2 textSize = ImGui::CalcTextSize(hint);
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (std::max)(0.0f, (avail.x - textSize.x) * 0.5f));
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (std::max)(0.0f, (avail.y - textSize.y) * 0.35f));
                    ImGui::PushStyleColor(ImGuiCol_Text, EditorThemeRegistry::Color("TEXT_DISABLED"));
                    ImGui::TextUnformatted(hint);
                    ImGui::PopStyleColor();
                }
            } else {
                if (m_selectedObjectId != 0) {
                    RenderSingleObject(ctx, m_selectedObjectId);
                } else {
                    ctx->EmptyState("##inspector_empty", "glyphs/picker", Tr("inspector.no_object_selected"),
                                    Tr("inspector.empty_hint"));
                }
            }
        }
    }

    // Drag-drop target for scripts on the whole PropertiesModule
    if (m_selectedObjectId != 0 && ImGui::BeginDragDropTarget()) {
        std::string path;
        if (ctx->AcceptDragDropPayload("SCRIPT_FILE", &path)) {
            ExecuteEditorCommand("component.add", MakeComponentAddCommandArgument("", false, path), "drag_drop");
        }
        ImGui::EndDragDropTarget();
    }
}

// ============================================================================
// Raw data module (bottom half)
// ============================================================================

void InspectorPanel::RenderRawDataModule(InxGUIContext *ctx, float height)
{
    {
        bool childVisible = ImGui::BeginChild("RawDataModule", ImVec2(0, height), ImGuiChildFlags_Borders,
                                              ImGuiWindowFlags_AlwaysVerticalScrollbar);
        auto childGuard = MakeInspectorScopeExit([] { ImGui::EndChild(); });
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
    }
}

// ============================================================================
// Splitter
// ============================================================================

float InspectorPanel::RenderSplitter(InxGUIContext *ctx, float totalHeight)
{
    const float dpi = ctx->GetDpiScale();
    ImGui::Separator();

    float availWidth = ImGui::GetContentRegionAvail().x;
    ImGui::InvisibleButton("##InspectorSplitter", ImVec2(availWidth, EditorTheme::INSPECTOR_SPLITTER_H * dpi));

    bool hovered = ImGui::IsItemHovered();
    bool active = ImGui::IsItemActive();

    if (hovered || active) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        const ImVec2 minimum = ImGui::GetItemRectMin(), maximum = ImGui::GetItemRectMax();
        const float y = (minimum.y + maximum.y) * 0.5f;
        ImGui::GetWindowDrawList()->AddLine(ImVec2(minimum.x, y), ImVec2(maximum.x, y),
                                            ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_ACCENT")), dpi);
    }

    if (active) {
        float deltaY = ImGui::GetMouseDragDelta(0).y;
        if (std::abs(deltaY) > 1.0f) {
            float usableHeight = totalHeight - EditorTheme::INSPECTOR_SPLITTER_H * dpi;
            if (usableHeight > 0.0f) {
                float newPropsHeight = m_propertiesRatio * usableHeight + deltaY;
                float newRatio = newPropsHeight / usableHeight;
                float minRatio = EditorTheme::INSPECTOR_MIN_PROPS_H * dpi / usableHeight;
                float maxRatio = 1.0f - (EditorTheme::INSPECTOR_MIN_RAWDATA_H * dpi / usableHeight);
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
#if INFERNUX_FRAME_PROFILE
    using clock = std::chrono::high_resolution_clock;
#endif

    const bool objectChanged = (m_cachedObjInfoId != objId) || (m_cachedComponentListObjId != objId);
    const bool refreshObject = objectChanged || m_cachedObjectTargetRevision != m_frameRevisions.target ||
                               m_cachedObjectValueRevision != m_frameRevisions.value;
    const bool refreshComponents = objectChanged || m_cachedComponentTargetRevision != m_frameRevisions.target ||
                                   m_cachedComponentSchemaRevision != m_frameRevisions.schema;

    if (objectChanged) {
        m_cachedComponentBodyHeights.clear();
        m_cachedTransformBodyHeight = 0.0f;
    }

    // ── Sub-timing: getObjectInfo + getPrefabInfo ────────────────────
#if INFERNUX_FRAME_PROFILE
    auto t0 = clock::now();
#endif

    if (refreshObject) {
        if (getObjectInfo)
            m_cachedObjInfo = getObjectInfo(objId);
        else
            m_cachedObjInfo = {};
        m_cachedObjInfoId = objId;
        if (!m_cachedObjInfo.prefabGuid.empty() && getPrefabInfo)
            m_cachedPrefabInfo = getPrefabInfo(objId);
        else
            m_cachedPrefabInfo = {};
        m_cachedObjectTargetRevision = m_frameRevisions.target;
        m_cachedObjectValueRevision = m_frameRevisions.value;
    }

#if INFERNUX_FRAME_PROFILE
    auto t1 = clock::now();
    m_subGetInfo += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    // ── Sub-timing: getComponentList ─────────────────────────────────
#if INFERNUX_FRAME_PROFILE
    auto t4 = clock::now();
#endif

    if (refreshComponents) {
        if (getComponentList)
            m_cachedComponents = getComponentList(objId);
        else
            m_cachedComponents.clear();
        m_cachedComponentListObjId = objId;
        m_cachedComponentTargetRevision = m_frameRevisions.target;
        m_cachedComponentSchemaRevision = m_frameRevisions.schema;
    }

    const auto &components = m_cachedComponents;

#if INFERNUX_FRAME_PROFILE
    auto t5 = clock::now();
    m_subGetComponents += std::chrono::duration<double, std::milli>(t5 - t4).count();
#endif

    const auto &info = m_cachedObjInfo;
    const auto &pinfo = m_cachedPrefabInfo;
    bool isPrefabReadonly = pinfo.isReadonly;
    bool isPrefabTransformReadonly = pinfo.isTransformReadonly;

    ImGui::PushID(static_cast<int>(objId));
    auto objectIdGuard = MakeInspectorScopeExit([] { ImGui::PopID(); });

    // Prefab header bar (if applicable)
    if (!info.prefabGuid.empty()) {
        RenderPrefabHeader(ctx, objId, pinfo);
    }

    // Active, Name, Tag, Layer are scene-instance properties — always editable,
    // even on prefab instances (they don't come from the prefab asset).

    const float dpi = ctx->GetDpiScale();
    const ImVec4 cardColor = EditorThemeRegistry::Color("ROLE_BG_SURFACE");
    ctx->BeginFramedGroup(12.0f, 10.0f, 0.0f, {cardColor.x, cardColor.y, cardColor.z, cardColor.w}, {0, 0, 0, 0});
    RenderObjectHeader(ctx, objId, info);

    // Tag & Layer
    ImGui::Dummy(ImVec2(0.0f, 3.0f * dpi));
    RenderTagLayerRow(ctx, objId, info);

    // ── Sub-timing: Transform ────────────────────────────────────────
#if INFERNUX_FRAME_PROFILE
    auto t2 = clock::now();
#endif

    // Transform (skip for screen-space UI elements)
    if (!info.hideTransform) {
        ImGui::Separator();
        {
            if (isPrefabTransformReadonly)
                ImGui::BeginDisabled();
            auto transformDisabledGuard = MakeInspectorScopeExit([isPrefabTransformReadonly] {
                if (isPrefabTransformReadonly)
                    ImGui::EndDisabled();
            });
            if (m_cachedTransformBodyHeight > 0.0f && ctx &&
                !ctx->IsVirtualizedRegionVisible(m_cachedTransformBodyHeight)) {
                ImGui::Dummy(ImVec2(0.0f, m_cachedTransformBodyHeight));
            } else {
                const float bodyStartY = ImGui::GetCursorPosY();
                RenderTransform(ctx, objId, info.hideTransformScale, info.drivenTransformProperties);
                const float bodyHeight = ImGui::GetCursorPosY() - bodyStartY;
                if (bodyHeight > 0.0f)
                    m_cachedTransformBodyHeight = bodyHeight;
            }
        }
    }

#if INFERNUX_FRAME_PROFILE
    auto t3 = clock::now();
    m_subTransform += std::chrono::duration<double, std::milli>(t3 - t2).count();
#endif

    ctx->EndFramedGroup();
    const ImVec2 cardHi = ImGui::GetItemRectMax();
    const float cut = 6.0f * dpi;
    ImGui::GetWindowDrawList()->AddTriangleFilled(ImVec2(cardHi.x - cut, ImGui::GetItemRectMin().y),
                                                  ImVec2(cardHi.x, ImGui::GetItemRectMin().y),
                                                  ImVec2(cardHi.x, ImGui::GetItemRectMin().y + cut),
                                                  ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_BG_BASE")));
    ctx->RecordSemanticItem("object_header", info.name, true, "inspector.object." + std::to_string(objId) + ".header");
    ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP * dpi));

    // --- Prefab instance: disable everything below Transform ---
    if (isPrefabReadonly)
        ImGui::BeginDisabled();
    auto prefabDisabledGuard = MakeInspectorScopeExit([isPrefabReadonly] {
        if (isPrefabReadonly)
            ImGui::EndDisabled();
    });

    // ── Sub-timing: Component bodies ─────────────────────────────────
#if INFERNUX_FRAME_PROFILE
    auto t6 = clock::now();
#endif

    // Render each component
    for (const auto &comp : components) {
        ImGui::PushID(static_cast<int>(comp.componentId));
        auto componentIdGuard = MakeInspectorScopeExit([] { ImGui::PopID(); });

        InspectorComponentCard card(ctx, comp.typeName);

        // No "(Script)" suffix — type is shown via icon + mono type name in the header.
        // Broken scripts keep a short localized mark without English parentheses.
        const char *scriptSuffix = comp.isBroken ? u8" · 缺失" : "";
        const std::string &componentLabel = comp.displayName.empty() ? comp.typeName : comp.displayName;
        const uint64_t iconId = getComponentIconId ? getComponentIconId(comp.typeName, comp.isScript) : comp.iconId;
        const auto header = RenderComponentHeader(
            ctx, componentLabel, "comp_" + std::to_string(comp.componentId), iconId,
            /*showEnabled=*/true, comp.enabled, comp.isScript ? scriptSuffix : "",
            /*defaultOpen=*/true,
            "inspector.object." + std::to_string(objId) + ".component." + std::to_string(comp.componentId),
            comp.isScript ? "py_comp_ctx" : "comp_ctx", IsComponentSelected(comp.componentId), {objId},
            {comp.componentId}, true, comp.typeName);

        if (header.selectionRequested && onComponentSelectionChanged)
            onComponentSelectionChanged({objId}, {comp.componentId}, comp.isNative);

        // Right-click context menu — only call Python when popup is open
        bool componentRemoved = false;
        {
            const char *ctxPopupId = comp.isScript ? "py_comp_ctx" : "comp_ctx";
            if (ctx->BeginPopup(ctxPopupId)) {
                if (renderComponentContextMenu) {
                    componentRemoved =
                        renderComponentContextMenu(ctx, objId, comp.typeName, comp.componentId, comp.isNative);
                }
                ctx->EndPopup();
            }
        }

        if (!componentRemoved) {
            // Enabled toggle
            if (header.enabled != comp.enabled)
                ExecuteEditorCommand(
                    "component.set_enabled",
                    MakeComponentEnabledCommandArgument(objId, comp.componentId, header.enabled, comp.isNative));

            // Component body
            if (header.open) {
                const float inset = EditorThemeRegistry::Float("INSPECTOR_COMPONENT_INSET") * dpi;
                InspectorComponentBody bodyScope(inset);
                ImGui::Dummy(ImVec2(0.0f, 4.0f * dpi));
                if (comp.isBroken && !comp.brokenError.empty()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, EditorThemeRegistry::Colors().at("ERROR_TEXT"));
                    ImGui::TextWrapped("%s", comp.brokenError.c_str());
                    ImGui::PopStyleColor();
                } else if (renderComponentBody) {
                    const auto heightIt = m_cachedComponentBodyHeights.find(comp.componentId);
                    const float cachedHeight = heightIt != m_cachedComponentBodyHeights.end() ? heightIt->second : 0.0f;
                    if (cachedHeight > 0.0f && ctx && !ctx->IsVirtualizedRegionVisible(cachedHeight)) {
                        ImGui::Dummy(ImVec2(0.0f, cachedHeight));
                    } else {
                        const float bodyStartY = ImGui::GetCursorPosY();
                        renderComponentBody(ctx, objId, comp.typeName, comp.componentId, comp.isNative);
                        const float bodyHeight = ImGui::GetCursorPosY() - bodyStartY;
                        if (bodyHeight > 0.0f)
                            m_cachedComponentBodyHeights[comp.componentId] = bodyHeight;
                    }
                }
            }
        }
        ImGui::Dummy(ImVec2(0.0f, EditorThemeRegistry::Float("INSPECTOR_SECTION_GAP") * dpi));
    }

    // Add Component button + popup
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP * dpi));
    RenderAddComponentButton(ctx);
    ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP * dpi));
    RenderAddComponentPopup(ctx);

#if INFERNUX_FRAME_PROFILE
    auto t7 = clock::now();
    m_subComponentBodies += std::chrono::duration<double, std::milli>(t7 - t6).count();
#endif

    // ── Sub-timing: Material sections ────────────────────────────────
#if INFERNUX_FRAME_PROFILE
    auto t8 = clock::now();
#endif

    // Material override sections
    if (renderMaterialSections)
        renderMaterialSections(ctx, objId);

#if INFERNUX_FRAME_PROFILE
    auto t9 = clock::now();
    m_subMaterials += std::chrono::duration<double, std::milli>(t9 - t8).count();
#endif
}

// ============================================================================
// Multi-edit inspector
// ============================================================================

const std::vector<InspectorPanel::CommonComponent> &
InspectorPanel::GetCommonComponentsForMultiSelection(const std::vector<uint64_t> &ids)
{
    const bool refreshSnapshots = !m_cachedMultiComponentsValid || (m_cachedMultiComponentIds != ids) ||
                                  m_cachedMultiComponentTargetRevision != m_frameRevisions.target ||
                                  m_cachedMultiComponentSchemaRevision != m_frameRevisions.schema;
    if (!refreshSnapshots)
        return m_cachedMultiCommonComponents;

    m_cachedMultiComponentsValid = true;
    m_cachedMultiComponentIds = ids;
    m_cachedMultiComponentTargetRevision = m_frameRevisions.target;
    m_cachedMultiComponentSchemaRevision = m_frameRevisions.schema;
    m_cachedMultiCommonComponents.clear();

    if (!getComponentList || ids.empty())
        return m_cachedMultiCommonComponents;

    std::vector<std::vector<ComponentInfo>> perObjectComponents;
    perObjectComponents.reserve(ids.size());
    for (uint64_t objectId : ids)
        perObjectComponents.push_back(getComponentList(objectId));

    if (perObjectComponents.empty())
        return m_cachedMultiCommonComponents;

    auto makeKindKey = [](const ComponentInfo &component) {
        return component.typeName + (component.isNative ? "#n" : "#p");
    };
    auto makeKey = [&](const ComponentInfo &component, int occurrence) {
        return makeKindKey(component) + "#" + std::to_string(occurrence);
    };

    std::vector<std::pair<std::string, ComponentInfo>> firstKeys;
    std::unordered_map<std::string, int> firstCounts;
    for (const auto &component : perObjectComponents[0]) {
        const int occurrence = firstCounts[makeKindKey(component)]++;
        firstKeys.push_back({makeKey(component, occurrence), component});
    }

    std::vector<std::unordered_map<std::string, ComponentInfo>> componentMaps;
    componentMaps.reserve(perObjectComponents.size());
    for (const auto &componentList : perObjectComponents) {
        std::unordered_map<std::string, ComponentInfo> componentMap;
        std::unordered_map<std::string, int> counts;
        for (const auto &component : componentList) {
            const int occurrence = counts[makeKindKey(component)]++;
            componentMap[makeKey(component, occurrence)] = component;
        }
        componentMaps.push_back(std::move(componentMap));
    }

    for (const auto &[key, firstComponent] : firstKeys) {
        CommonComponent entry;
        entry.display = firstComponent;
        bool presentOnAll = true;
        for (const auto &componentMap : componentMaps) {
            auto iter = componentMap.find(key);
            if (iter == componentMap.end()) {
                presentOnAll = false;
                break;
            }
            entry.componentIds.push_back(iter->second.componentId);
        }
        if (presentOnAll)
            m_cachedMultiCommonComponents.push_back(std::move(entry));
    }

    return m_cachedMultiCommonComponents;
}

const InspectorPanel::MultiTransformSnapshot &
InspectorPanel::GetMultiTransformSnapshot(const std::vector<uint64_t> &ids)
{
    const bool refreshSnapshot = !m_cachedMultiTransformValid || (m_cachedMultiTransformIds != ids) ||
                                 m_cachedMultiTransformTargetRevision != m_frameRevisions.target ||
                                 m_cachedMultiTransformValueRevision != m_frameRevisions.value;
    if (!refreshSnapshot)
        return m_cachedMultiTransformSnapshot;

    m_cachedMultiTransformValid = true;
    m_cachedMultiTransformIds = ids;
    m_cachedMultiTransformTargetRevision = m_frameRevisions.target;
    m_cachedMultiTransformValueRevision = m_frameRevisions.value;
    m_cachedMultiTransformSnapshot = {};

    if (!getTransformData || ids.empty())
        return m_cachedMultiTransformSnapshot;

    if (getObjectInfo) {
        m_cachedMultiTransformSnapshot.componentIds.reserve(ids.size());
        for (const uint64_t objectId : ids) {
            const uint64_t componentId = getObjectInfo(objectId).transformComponentId;
            if (componentId != 0)
                m_cachedMultiTransformSnapshot.componentIds.push_back(componentId);
        }
    }

    TransformData first = getTransformData(ids[0]);
    m_cachedMultiTransformSnapshot.first = first;

    for (size_t objectIndex = 1; objectIndex < ids.size(); ++objectIndex) {
        TransformData transformData = getTransformData(ids[objectIndex]);
        auto &mixed = m_cachedMultiTransformSnapshot.mixed;
        mixed[0] = mixed[0] || std::abs(transformData.px - first.px) > 1e-6f;
        mixed[1] = mixed[1] || std::abs(transformData.py - first.py) > 1e-6f;
        mixed[2] = mixed[2] || std::abs(transformData.pz - first.pz) > 1e-6f;
        mixed[3] = mixed[3] || std::abs(transformData.rx - first.rx) > 1e-6f;
        mixed[4] = mixed[4] || std::abs(transformData.ry - first.ry) > 1e-6f;
        mixed[5] = mixed[5] || std::abs(transformData.rz - first.rz) > 1e-6f;
        mixed[6] = mixed[6] || std::abs(transformData.sx - first.sx) > 1e-6f;
        mixed[7] = mixed[7] || std::abs(transformData.sy - first.sy) > 1e-6f;
        mixed[8] = mixed[8] || std::abs(transformData.sz - first.sz) > 1e-6f;
    }

    return m_cachedMultiTransformSnapshot;
}

void InspectorPanel::RenderMultiEdit(InxGUIContext *ctx, const std::vector<uint64_t> &ids)
{
#if INFERNUX_FRAME_PROFILE
    using clock = std::chrono::high_resolution_clock;
#endif

    const float dpi = ctx->GetDpiScale();
    int n = static_cast<int>(ids.size());
    ImGui::PushID("multi_edit");

    if (!ids.empty()) {
        const auto &transformSnapshot = GetMultiTransformSnapshot(ids);
        const auto &transformComponentIds = transformSnapshot.componentIds;
        const auto color = EditorThemeRegistry::Color("ROLE_BG_SURFACE");
        ctx->BeginFramedGroup(12, 10, 0, {color.x, color.y, color.z, color.w}, {0, 0, 0, 0});
        const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
        ctx->PushUIFont(EditorThemeRegistry::Float("EDITOR_TITLE_FONT_SIZE"));
        ImGui::AlignTextToFramePadding();
        ImGui::Text(Tr("inspector.multiple_selection").c_str(), n);
        ctx->PopFont();
        ImGui::SameLine();
        ImGui::SetCursorScreenPos(ImVec2(right - 28 * dpi, ImGui::GetCursorScreenPos().y));
        ImGui::BeginDisabled(transformComponentIds.size() != ids.size());
        if (ctx->IconButton("object_transform_options", "glyphs/more", Tr("inspector.transform_options"), 28 * dpi,
                            28 * dpi))
            ImGui::OpenPopup("multi_transform_ctx");
        ImGui::EndDisabled();

        if (ctx->BeginPopup("multi_transform_ctx")) {
            if (renderComponentContextMenu && transformComponentIds.size() == ids.size())
                renderComponentContextMenu(ctx, ids.front(), "Transform", transformComponentIds.front(), true);
            ctx->EndPopup();
        }

#if INFERNUX_FRAME_PROFILE
        auto transformStart = clock::now();
#endif
        bool hideTransformScale = !ids.empty() && static_cast<bool>(getObjectInfo);
        uint32_t drivenTransformProperties = 0;
        if (hideTransformScale) {
            for (uint64_t id : ids) {
                const auto objectInfo = getObjectInfo(id);
                hideTransformScale = hideTransformScale && objectInfo.hideTransformScale;
                drivenTransformProperties |= objectInfo.drivenTransformProperties;
            }
        } else if (getObjectInfo) {
            for (uint64_t id : ids)
                drivenTransformProperties |= getObjectInfo(id).drivenTransformProperties;
        }
        ImGui::Separator();
        RenderMultiTransform(ctx, ids, hideTransformScale, drivenTransformProperties);
        ctx->EndFramedGroup();
        ctx->RecordSemanticItem("object_header", Tr("inspector.multiple_selection"), true, "inspector.multi.header");
        ImGui::Dummy(ImVec2(0.0f, EditorThemeRegistry::Float("INSPECTOR_SECTION_GAP") * dpi));
#if INFERNUX_FRAME_PROFILE
        auto transformEnd = clock::now();
        m_subTransform += std::chrono::duration<double, std::milli>(transformEnd - transformStart).count();
#endif

#if INFERNUX_FRAME_PROFILE
        auto componentsStart = clock::now();
#endif
        const auto &commonComponents = GetCommonComponentsForMultiSelection(ids);
#if INFERNUX_FRAME_PROFILE
        auto componentsEnd = clock::now();
        m_subGetComponents += std::chrono::duration<double, std::milli>(componentsEnd - componentsStart).count();
#endif

#if INFERNUX_FRAME_PROFILE
        auto componentBodiesStart = clock::now();
#endif
        for (const auto &entry : commonComponents) {
            const auto &comp = entry.display;
            InspectorComponentCard card(ctx, comp.typeName);
            ImGui::PushID(static_cast<int>(comp.componentId));

            const uint64_t iconId = getComponentIconId ? getComponentIconId(comp.typeName, comp.isScript) : comp.iconId;

            const std::string &componentLabel = comp.displayName.empty() ? comp.typeName : comp.displayName;
            const char *multiScriptSuffix = comp.isBroken ? u8" · 缺失" : "";
            const auto componentHeader = RenderComponentHeader(
                ctx, componentLabel, "multi_comp_" + std::to_string(comp.componentId), iconId, true, comp.enabled,
                comp.isScript ? multiScriptSuffix : "", true, "",
                comp.isScript ? "multi_py_comp_ctx" : "multi_comp_ctx", AreComponentsSelected(entry.componentIds), ids,
                entry.componentIds, true, comp.typeName);

            if (componentHeader.selectionRequested && onComponentSelectionChanged)
                onComponentSelectionChanged(ids, entry.componentIds, comp.isNative);

            bool componentRemoved = false;
            const char *contextPopupId = comp.isScript ? "multi_py_comp_ctx" : "multi_comp_ctx";
            if (ctx->BeginPopup(contextPopupId)) {
                if (renderComponentContextMenu)
                    componentRemoved = renderComponentContextMenu(ctx, ids.front(), comp.typeName,
                                                                  entry.componentIds.front(), comp.isNative);
                ctx->EndPopup();
            }

            if (!componentRemoved && componentHeader.enabled != comp.enabled) {
                ExecuteEditorCommand("component.set_enabled",
                                     MakeComponentEnabledCommandArgument(ids, entry.componentIds,
                                                                         componentHeader.enabled, comp.isNative));
                m_cachedMultiComponentsValid = false;
            }

            if (!componentRemoved && componentHeader.open) {
                const float inset = EditorThemeRegistry::Float("INSPECTOR_COMPONENT_INSET") * dpi;
                InspectorComponentBody bodyScope(inset);
                ImGui::Dummy(ImVec2(0.0f, 4.0f * dpi));
                if (comp.isBroken && !comp.brokenError.empty()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, EditorThemeRegistry::Colors().at("ERROR_TEXT"));
                    ImGui::TextWrapped("%s", comp.brokenError.c_str());
                    ImGui::PopStyleColor();
                } else if (renderMultiComponentBody)
                    renderMultiComponentBody(ctx, ids, comp.typeName, entry.componentIds, comp.isNative);
                else if (renderComponentBody)
                    renderComponentBody(ctx, ids[0], comp.typeName, comp.componentId, comp.isNative);
            }
            ImGui::Dummy(ImVec2(0.0f, EditorThemeRegistry::Float("INSPECTOR_SECTION_GAP") * dpi));

            ImGui::PopID();
        }
#if INFERNUX_FRAME_PROFILE
        auto componentBodiesEnd = clock::now();
        m_subComponentBodies +=
            std::chrono::duration<double, std::milli>(componentBodiesEnd - componentBodiesStart).count();
#endif

        // Add Component
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP * dpi));
        RenderAddComponentButton(ctx);
        ImGui::Dummy(ImVec2(0, EditorTheme::INSPECTOR_SECTION_GAP * dpi));
        RenderAddComponentPopup(ctx);
    }

    ImGui::PopID();
}

// ============================================================================
// Object header (active + name)
// ============================================================================

void InspectorPanel::RenderObjectHeader(InxGUIContext *ctx, uint64_t objId, const ObjectInfo &info)
{
    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    ImGui::BeginGroup();
    const float dpi = ctx->GetDpiScale();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + (26.0f * dpi - ImGui::GetTextLineHeight()) * 0.5f));
    // Active checkbox
    bool active = info.active;
    bool newActive = RenderInspectorCheckbox(ctx, "##obj_active", active);
    if (captureSemantics)
        ctx->RecordSemanticItem("inspector_active", "Active", true,
                                "inspector.object." + std::to_string(objId) + ".active");
    if (newActive != active &&
        ExecuteEditorCommand("scene.set_object_property", MakeObjectPropertyCommandArgument(objId, "active", newActive),
                             "pointer")) {
        m_cachedObjInfo.active = newActive;
    }

    ImGui::SameLine(0, 6.0f * ctx->GetDpiScale());
    ImGui::SetCursorScreenPos(ImVec2(ImGui::GetCursorScreenPos().x, origin.y));
    ImGui::Dummy(ImVec2(26.0f * dpi, 26.0f * dpi));
    EditorObjectIcon objectIcon;
    for (const auto &component : m_cachedComponents)
        objectIcon.Consider(component.isScript ? "Script" : component.typeName);
    const std::string iconResource = objectIcon.Resource(!info.prefabGuid.empty());
    const auto icon = ctx->EditorIcon(iconResource);
    if (icon)
        ImGui::GetWindowDrawList()->AddImage(ImTextureRef(static_cast<ImTextureID>(icon)), ImGui::GetItemRectMin(),
                                             ImGui::GetItemRectMax());
    if (captureSemantics)
        ctx->RecordSemanticItem("object_icon", info.name, true, "inspector.object." + std::to_string(objId) + ".icon",
                                std::nullopt, std::nullopt, iconResource);
    ImGui::SameLine(0, 8.0f * dpi);
    ImGui::SetCursorScreenPos(
        ImVec2(ImGui::GetCursorScreenPos().x, origin.y + (26.0f * dpi - ImGui::GetFrameHeight()) * 0.5f));

    // Editable name
    const float optionsWidth = info.hideTransform ? 0 : 28 * dpi;
    ImGui::SetNextItemWidth(
        (std::max)(32.0f * dpi, ImGui::GetContentRegionAvail().x - optionsWidth - (info.hideTransform ? 0 : 6 * dpi)));
    std::string newName = info.name;
    const bool nameChanged = ImGui::InputText("##obj_name", &newName);
    if (captureSemantics)
        ctx->RecordSemanticItem("inspector_name", "Name", true, "inspector.object." + std::to_string(objId) + ".name");
    if (nameChanged) {
        if (newName != info.name &&
            ExecuteEditorCommand("scene.set_object_property", MakeObjectPropertyCommandArgument(objId, "name", newName),
                                 "inline_edit")) {
            m_cachedObjInfo.name = newName;
        }
    }
    if (!info.hideTransform) {
        ImGui::SameLine(0, 6 * dpi);
        if (ctx->IconButton("object_transform_options", "glyphs/more", Tr("inspector.transform_options"), optionsWidth,
                            28 * dpi))
            ImGui::OpenPopup("transform_ctx");
        if (captureSemantics)
            ctx->RecordSemanticItem("button", Tr("inspector.transform_options"), true,
                                    "inspector.object." + std::to_string(objId) + ".transform_options");
        if (ctx->BeginPopup("transform_ctx")) {
            if (renderComponentContextMenu)
                renderComponentContextMenu(ctx, objId, "Transform", info.transformComponentId, true);
            ctx->EndPopup();
        }
    }
    ImGui::EndGroup();
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
        m_cachedTagItems.push_back(Tr("inspector.add_tag"));
    }

    if (getAllLayers) {
        m_cachedLayers = getAllLayers();
        m_cachedLayerItems.clear();
        for (size_t i = 0; i < m_cachedLayers.size(); ++i) {
            std::string label = std::to_string(i) + ": ";
            label += m_cachedLayers[i].empty() ? Tr("igui.none") : m_cachedLayers[i];
            m_cachedLayerItems.push_back(std::move(label));
        }
        m_cachedLayerItems.push_back(Tr("inspector.add_layer"));
    }
}

void InspectorPanel::RenderTagLayerRow(InxGUIContext *ctx, uint64_t objId, const ObjectInfo &info)
{
    const float dpi = ctx->GetDpiScale();
    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    RefreshTagLayerCache();

    const float rowStartX = ImGui::GetCursorPosX();
    const float availW = ImGui::GetContentRegionAvail().x;
    const float outerInset = 3.0f * dpi;
    const float columnGap = 10.0f * dpi;
    const float columnW = (std::max)(40.0f * dpi, (availW - outerInset * 2.0f - columnGap) * 0.5f);

    // --- Tag (left column) ---
    ImGui::SetCursorPosX(rowStartX + outerInset);
    ImGui::BeginGroup();
    ctx->PushMonoFont();
    ImGui::PushStyleColor(ImGuiCol_Text, EditorThemeRegistry::Color("TEXT_DISABLED"));
    ImGui::TextUnformatted(Tr("inspector.tag_kicker").c_str());
    ImGui::PopStyleColor();
    ctx->PopFont();

    int tagIdx = 0;
    for (size_t i = 0; i < m_cachedTags.size(); ++i) {
        if (m_cachedTags[i] == info.tag) {
            tagIdx = static_cast<int>(i);
            break;
        }
    }

    const float tagComboW = columnW;
    int newTagIdx = ctx->SearchableCombo("Inspector.Tag", tagIdx, m_cachedTagItems, tagComboW, 8,
                                         Tr("igui.search_hint").c_str(), Tr("igui.no_results").c_str());
    if (captureSemantics)
        ctx->RecordSemanticItem("inspector_tag", Tr("inspector.tag"), true,
                                "inspector.object." + std::to_string(objId) + ".tag");
    if (newTagIdx != tagIdx) {
        if (newTagIdx == static_cast<int>(m_cachedTags.size())) {
            // "Add Tag..." selected
            ExecuteEditorCommand("window.open", "tag_layer_settings", "pointer");
        } else if (newTagIdx >= 0 && newTagIdx < static_cast<int>(m_cachedTags.size())) {
            const std::string &newTag = m_cachedTags[newTagIdx];
            if (ExecuteEditorCommand("scene.set_object_property",
                                     MakeObjectPropertyCommandArgument(objId, "tag", newTag), "pointer"))
                m_cachedObjInfo.tag = newTag;
        }
    }

    ImGui::EndGroup();
    // --- Layer (right column) ---
    ImGui::SameLine(0.0f, columnGap);
    ImGui::BeginGroup();
    ctx->PushMonoFont();
    ImGui::PushStyleColor(ImGuiCol_Text, EditorThemeRegistry::Color("TEXT_DISABLED"));
    ImGui::TextUnformatted(Tr("inspector.layer_kicker").c_str());
    ImGui::PopStyleColor();
    ctx->PopFont();
    const float layerComboW = columnW;

    int newLayer = ctx->SearchableCombo("Inspector.Layer", info.layer, m_cachedLayerItems, layerComboW, 8,
                                        Tr("igui.search_hint").c_str(), Tr("igui.no_results").c_str());
    if (captureSemantics)
        ctx->RecordSemanticItem("inspector_layer", Tr("inspector.layer"), true,
                                "inspector.object." + std::to_string(objId) + ".layer");
    if (newLayer != info.layer) {
        if (newLayer == static_cast<int>(m_cachedLayerItems.size()) - 1) {
            // "Add Layer..." selected
            ExecuteEditorCommand("window.open", "tag_layer_settings", "pointer");
        } else if (ExecuteEditorCommand("scene.set_object_property",
                                        MakeObjectPropertyCommandArgument(objId, "layer", newLayer), "pointer")) {
            m_cachedObjInfo.layer = newLayer;
        }
    }
    ImGui::EndGroup();
}

// ============================================================================
// Transform rendering
// ============================================================================

std::string InspectorPanel::UpdateTransformGesture(size_t rowIndex, uint32_t lifecycleFlags)
{
    if (rowIndex >= m_transformGestureIds.size())
        return {};
    auto &gestureId = m_transformGestureIds[rowIndex];
    const bool activated = (lifecycleFlags & InxGUIContext::EditActivated) != 0u;
    const bool changed = (lifecycleFlags & InxGUIContext::EditChanged) != 0u;
    if (activated || (changed && gestureId.empty())) {
        gestureId = "inspector.transform:" + std::to_string(++m_transformGestureSerial);
    }
    return gestureId;
}

void InspectorPanel::FinishTransformGesture(size_t rowIndex, uint32_t lifecycleFlags)
{
    if (rowIndex >= m_transformGestureIds.size())
        return;
    if ((lifecycleFlags & InxGUIContext::EditDeactivated) != 0u)
        m_transformGestureIds[rowIndex].clear();
}

void InspectorPanel::RenderTransform(InxGUIContext *ctx, uint64_t objId, bool hideScale, uint32_t drivenProperties)
{
    if (!getTransformData)
        return;

    if (m_cachedTransformObjId != objId || m_cachedTransformValueRevision != m_frameRevisions.value) {
        m_cachedTransformData = getTransformData(objId);
        m_cachedTransformObjId = objId;
        m_cachedTransformValueRevision = m_frameRevisions.value;
    }
    TransformData td = m_cachedTransformData;
    float labelW = ctx->InspectorLabelWidth();

    // Vector3Control modifies the array in-place
    float pos[3] = {td.px, td.py, td.pz};
    float rot[3] = {td.rx, td.ry, td.rz};
    float scl[3] = {td.sx, td.sy, td.sz};

    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    const std::string transformBase =
        captureSemantics ? "inspector.object." + std::to_string(objId) + ".transform." : std::string{};
    const std::string positionSemanticId = captureSemantics ? transformBase + "position" : std::string{};
    const std::string rotationSemanticId = captureSemantics ? transformBase + "rotation" : std::string{};
    const std::string scaleSemanticId = captureSemantics ? transformBase + "scale" : std::string{};

    const bool positionDriven = (drivenProperties & DRIVEN_TRANSFORM_POSITION) != 0u;
    const bool rotationDriven = (drivenProperties & DRIVEN_TRANSFORM_ROTATION) != 0u;
    const bool scaleDriven = (drivenProperties & DRIVEN_TRANSFORM_SCALE) != 0u;

    if (positionDriven)
        ImGui::BeginDisabled();
    ctx->Vector3Control(Tr("inspector.transform.position"), pos, DRAG_SPEED_DEFAULT, labelW, positionSemanticId);
    if (positionDriven)
        ImGui::EndDisabled();
    const uint32_t positionLifecycle = ctx->GetLastEditLifecycleFlags();
    const std::string positionGesture = UpdateTransformGesture(0, positionLifecycle);
    if (captureSemantics)
        ctx->RecordSemanticItem("inspector_transform", Tr("inspector.transform.position"), !positionDriven,
                                positionSemanticId);
    if (rotationDriven)
        ImGui::BeginDisabled();
    ctx->Vector3Control(Tr("inspector.transform.rotation"), rot, DRAG_SPEED_DEFAULT, labelW, rotationSemanticId);
    if (rotationDriven)
        ImGui::EndDisabled();
    const uint32_t rotationLifecycle = ctx->GetLastEditLifecycleFlags();
    const std::string rotationGesture = UpdateTransformGesture(1, rotationLifecycle);
    if (captureSemantics)
        ctx->RecordSemanticItem("inspector_transform", Tr("inspector.transform.rotation"), !rotationDriven,
                                rotationSemanticId);
    uint32_t scaleLifecycle = 0;
    std::string scaleGesture;
    if (!hideScale) {
        if (scaleDriven)
            ImGui::BeginDisabled();
        ctx->Vector3Control(Tr("inspector.transform.scale"), scl, DRAG_SPEED_FINE, labelW, scaleSemanticId);
        if (scaleDriven)
            ImGui::EndDisabled();
        scaleLifecycle = ctx->GetLastEditLifecycleFlags();
        scaleGesture = UpdateTransformGesture(2, scaleLifecycle);
        if (captureSemantics)
            ctx->RecordSemanticItem("inspector_transform", Tr("inspector.transform.scale"), !scaleDriven,
                                    scaleSemanticId);
    }

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
        const bool positionChanged =
            std::abs(pos[0] - td.px) > 1e-6f || std::abs(pos[1] - td.py) > 1e-6f || std::abs(pos[2] - td.pz) > 1e-6f;
        const bool rotationChanged =
            std::abs(rot[0] - td.rx) > 1e-6f || std::abs(rot[1] - td.ry) > 1e-6f || std::abs(rot[2] - td.rz) > 1e-6f;
        const std::string &gestureId = positionChanged   ? positionGesture
                                       : rotationChanged ? rotationGesture
                                                         : scaleGesture;
        const nlohmann::json payload{
            {"object_ids", {objId}},
            {"transforms",
             {MakeTransformValue(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], scl[0], scl[1], scl[2])}},
            {"gesture_id", gestureId}};
        ExecuteEditorCommand("scene.set_transforms", payload.dump(), "pointer");
    }
    FinishTransformGesture(0, positionLifecycle);
    FinishTransformGesture(1, rotationLifecycle);
    FinishTransformGesture(2, scaleLifecycle);
}

void InspectorPanel::RenderMultiTransform(InxGUIContext *ctx, const std::vector<uint64_t> &ids, bool hideScale,
                                          uint32_t drivenProperties)
{
    if (!getTransformData || ids.empty())
        return;

    const auto &snapshot = GetMultiTransformSnapshot(ids);
    TransformData first = snapshot.first;
    bool mixed[9] = {};
    for (size_t axisIndex = 0; axisIndex < snapshot.mixed.size(); ++axisIndex)
        mixed[axisIndex] = snapshot.mixed[axisIndex];

    const float dpi = ctx->GetDpiScale();
    float labelW = ctx->InspectorLabelWidth();
    float pos[3] = {first.px, first.py, first.pz};
    float rot[3] = {first.rx, first.ry, first.rz};
    float scl[3] = {first.sx, first.sy, first.sz};
    bool axisChanged[9] = {};
    uint32_t rowLifecycle[3] = {};

    auto renderRow = [&](const std::string &label, const char *rowId, float *values, const float *originalValues,
                         const bool *rowMixed, int baseIndex, float speed, bool driven) {
        if (driven)
            ImGui::BeginDisabled();
        ImGui::TextUnformatted(label.c_str());
        ImGui::SameLine(labelW);

        const ImGuiStyle &style = ImGui::GetStyle();
        const float avail = std::max(1.0f * dpi, ImGui::GetContentRegionAvail().x);
        const float cellWidth = std::max(24.0f * dpi, (avail - style.ItemSpacing.x * 2.0f) / 3.0f);

        ImGui::PushID(rowId);
        for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
            if (axisIndex > 0)
                ImGui::SameLine(0.0f, style.ItemSpacing.x);

            ImGui::PushID(axisIndex);
            ImGui::SetNextItemWidth(cellWidth);
            if (rowMixed[axisIndex]) {
                static std::unordered_map<std::string, std::array<char, 64>> mixedBuffers;
                const std::string bufferKey = std::string(rowId) + ":" + std::to_string(axisIndex);
                auto &buffer = mixedBuffers[bufferKey];
                const bool submitted = ImGui::InputTextWithHint("##mixed", "--", buffer.data(), buffer.size(),
                                                                ImGuiInputTextFlags_EnterReturnsTrue);
                rowLifecycle[baseIndex / 3] |= CaptureInspectorEditLifecycle(submitted);
                const bool committed = submitted || ImGui::IsItemDeactivatedAfterEdit();
                if (committed) {
                    char *end = nullptr;
                    const float parsed = std::strtof(buffer.data(), &end);
                    while (end && *end != '\0' && std::isspace(static_cast<unsigned char>(*end)))
                        ++end;
                    if (end != buffer.data() && end && *end == '\0') {
                        values[axisIndex] = parsed;
                        axisChanged[baseIndex + axisIndex] = true;
                    }
                    buffer.fill('\0');
                }
            } else {
                const bool edited =
                    ImGui::DragFloat("##value", &values[axisIndex], speed, -1000000.0f, 1000000.0f, "%.3f");
                rowLifecycle[baseIndex / 3] |= CaptureInspectorEditLifecycle(edited);
                axisChanged[baseIndex + axisIndex] = std::abs(values[axisIndex] - originalValues[axisIndex]) > 1e-6f;
            }
            ImGui::PopID();
        }
        ImGui::PopID();
        if (driven)
            ImGui::EndDisabled();
    };

    const float originalPos[3] = {first.px, first.py, first.pz};
    const float originalRot[3] = {first.rx, first.ry, first.rz};
    const float originalScale[3] = {first.sx, first.sy, first.sz};

    renderRow(Tr("inspector.transform.position"), "position", pos, originalPos, mixed, 0, DRAG_SPEED_DEFAULT,
              (drivenProperties & DRIVEN_TRANSFORM_POSITION) != 0u);
    renderRow(Tr("inspector.transform.rotation"), "rotation", rot, originalRot, mixed + 3, 3, DRAG_SPEED_DEFAULT,
              (drivenProperties & DRIVEN_TRANSFORM_ROTATION) != 0u);
    if (!hideScale)
        renderRow(Tr("inspector.transform.scale"), "scale", scl, originalScale, mixed + 6, 6, DRAG_SPEED_FINE,
                  (drivenProperties & DRIVEN_TRANSFORM_SCALE) != 0u);

    const std::array<std::string, 3> gestureIds = {
        UpdateTransformGesture(0, rowLifecycle[0]),
        UpdateTransformGesture(1, rowLifecycle[1]),
        UpdateTransformGesture(2, rowLifecycle[2]),
    };

    bool anyAxisChanged = false;
    for (bool changed : axisChanged)
        anyAxisChanged |= changed;
    if (!anyAxisChanged) {
        FinishTransformGesture(0, rowLifecycle[0]);
        FinishTransformGesture(1, rowLifecycle[1]);
        FinishTransformGesture(2, rowLifecycle[2]);
        return;
    }

    nlohmann::json transforms = nlohmann::json::array();
    for (uint64_t id : ids) {
        TransformData td = getTransformData(id);
        if (axisChanged[0])
            td.px = pos[0];
        if (axisChanged[1])
            td.py = pos[1];
        if (axisChanged[2])
            td.pz = pos[2];
        if (axisChanged[3])
            td.rx = rot[0];
        if (axisChanged[4])
            td.ry = rot[1];
        if (axisChanged[5])
            td.rz = rot[2];
        if (axisChanged[6])
            td.sx = scl[0];
        if (axisChanged[7])
            td.sy = scl[1];
        if (axisChanged[8])
            td.sz = scl[2];
        transforms.push_back(MakeTransformValue(td.px, td.py, td.pz, td.rx, td.ry, td.rz, td.sx, td.sy, td.sz));
    }
    size_t changedRow = 0;
    while (changedRow < 2 && !axisChanged[changedRow * 3] && !axisChanged[changedRow * 3 + 1] &&
           !axisChanged[changedRow * 3 + 2])
        ++changedRow;
    const nlohmann::json payload{
        {"object_ids", ids}, {"transforms", std::move(transforms)}, {"gesture_id", gestureIds[changedRow]}};
    if (ExecuteEditorCommand("scene.set_transforms", payload.dump(), "pointer"))
        m_cachedMultiTransformValid = false;
    FinishTransformGesture(0, rowLifecycle[0]);
    FinishTransformGesture(1, rowLifecycle[1]);
    FinishTransformGesture(2, rowLifecycle[2]);
}

// ============================================================================
// Prefab header
// ============================================================================

void InspectorPanel::RenderPrefabHeader(InxGUIContext *ctx, uint64_t objId, const PrefabInfo &pinfo)
{
    const float dpi = ctx->GetDpiScale();
    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    const std::string semanticBase =
        captureSemantics ? "inspector.object." + std::to_string(objId) + ".prefab" : std::string{};
    ImGui::Dummy(ImVec2(0, 4.0f * dpi));

    ImGui::PushStyleColor(ImGuiCol_ChildBg, EditorThemeRegistry::Colors().at("PREFAB_HEADER_BG"));
    ImGui::BeginChild("##prefab_header_bar", ImVec2(0, EditorTheme::PREFAB_HEADER_H * dpi), ImGuiChildFlags_Borders);

    ImGui::PushStyleColor(ImGuiCol_Text, EditorThemeRegistry::Colors().at("PREFAB_TEXT"));
    ImGui::TextUnformatted(Tr("inspector.prefab_label").c_str());
    if (captureSemantics)
        ctx->RecordSemanticItem("prefab_header", Tr("inspector.prefab_label"), true, semanticBase, std::nullopt,
                                static_cast<double>(pinfo.overrideCount));
    ImGui::PopStyleColor();

    float gap = EditorTheme::PREFAB_HEADER_BTN_GAP * dpi;

    ImGui::SameLine(0, gap * 2);
    EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Colors().at("INSPECTOR_INLINE_BTN_IDLE"));
    const std::string selectLabel = Tr("inspector.prefab_select");
    if (ImGui::Button(selectLabel.c_str())) {
        ExecuteEditorCommand("prefab.select_asset", std::to_string(objId));
    }
    if (captureSemantics)
        ctx->RecordSemanticItem("prefab_action", selectLabel, true, semanticBase + ".select");
    ImGui::PopStyleColor(3);

    ImGui::SameLine(0, gap);
    EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Colors().at("INSPECTOR_INLINE_BTN_IDLE"));
    const std::string openLabel = Tr("inspector.prefab_open");
    if (ImGui::Button(openLabel.c_str())) {
        ExecuteEditorCommand("prefab.open", std::to_string(objId));
    }
    if (captureSemantics)
        ctx->RecordSemanticItem("prefab_action", openLabel, true, semanticBase + ".open");
    ImGui::PopStyleColor(3);

    ImGui::SameLine(0, gap * 3);

    if (pinfo.overrideCount > 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, EditorThemeRegistry::Colors().at("WARNING_TEXT"));
        ImGui::Text("%d %s", pinfo.overrideCount, Tr("inspector.overrides").c_str());
        if (captureSemantics)
            ctx->RecordSemanticItem("prefab_override_count", Tr("inspector.overrides"), false,
                                    semanticBase + ".override_count", std::nullopt,
                                    static_cast<double>(pinfo.overrideCount));
        ImGui::PopStyleColor();

        ImGui::SameLine(0, gap * 2);
        EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Colors().at("INSPECTOR_INLINE_BTN_IDLE"));
        const std::string applyLabel = Tr("inspector.prefab_apply");
        if (ImGui::Button(applyLabel.c_str())) {
            ExecuteEditorCommand("prefab.apply", std::to_string(objId));
        }
        if (captureSemantics)
            ctx->RecordSemanticItem("prefab_action", applyLabel, true, semanticBase + ".apply");
        ImGui::PopStyleColor(3);

        ImGui::SameLine(0, gap);
        EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Colors().at("INSPECTOR_INLINE_BTN_IDLE"));
        const std::string revertLabel = Tr("inspector.prefab_revert");
        if (ImGui::Button(revertLabel.c_str())) {
            ExecuteEditorCommand("prefab.revert", std::to_string(objId));
        }
        if (captureSemantics)
            ctx->RecordSemanticItem("prefab_action", revertLabel, true, semanticBase + ".revert");
        ImGui::PopStyleColor(3);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::TEXT_DIM2);
        ImGui::TextUnformatted(Tr("inspector.no_overrides").c_str());
        if (captureSemantics)
            ctx->RecordSemanticItem("prefab_override_count", Tr("inspector.no_overrides"), false,
                                    semanticBase + ".override_count", std::nullopt, 0.0);
        ImGui::PopStyleColor();
    }

    if (!pinfo.structuralRows.empty()) {
        ImGui::Separator();
        ImGui::TextUnformatted(Tr("inspector.prefab_structure").c_str());
        if (captureSemantics)
            ctx->RecordSemanticItem("prefab_structure_header", Tr("inspector.prefab_structure"), false,
                                    semanticBase + ".structure", std::nullopt,
                                    static_cast<double>(pinfo.structuralRows.size()));
        for (size_t index = 0; index < pinfo.structuralRows.size(); ++index) {
            const auto &row = pinfo.structuralRows[index];
            const char *kindKey = "inspector.prefab_structure_change";
            if (row.kind == "child_added")
                kindKey = "inspector.prefab_structure_child_added";
            else if (row.kind == "child_removed")
                kindKey = "inspector.prefab_structure_child_removed";
            else if (row.kind == "component_added")
                kindKey = "inspector.prefab_structure_component_added";
            else if (row.kind == "component_removed")
                kindKey = "inspector.prefab_structure_component_removed";
            else if (row.kind == "child_reordered")
                kindKey = "inspector.prefab_structure_child_reordered";
            else if (row.kind == "component_reordered")
                kindKey = "inspector.prefab_structure_component_reordered";
            std::string label = Tr(kindKey) + " · " + row.nodePath;
            if (!row.key.empty())
                label += " · " + row.key;
            ImGui::BulletText("%s", label.c_str());
            if (captureSemantics)
                ctx->RecordSemanticItem("prefab_structure_row", label, false,
                                        semanticBase + ".structure." + std::to_string(index));
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

// ============================================================================
// Component header (icon + checkbox + collapsing header)
// ============================================================================

InspectorPanel::ComponentHeaderResult InspectorPanel::RenderComponentHeader(
    InxGUIContext *ctx, const std::string &typeName, const std::string &headerId, uint64_t iconId, bool showEnabled,
    bool isEnabled, const std::string &suffix, bool defaultOpen, const std::string &semanticId,
    const std::string &contextPopupId, bool selected, const std::vector<uint64_t> &dragObjectIds,
    const std::vector<uint64_t> &dragComponentIds, bool allowComponentReorder, const std::string &typeIdentity)
{
    const float dpi = ctx->GetDpiScale();
    bool newEnabled = isEnabled;
    const std::string &identity = typeIdentity.empty() ? typeName : typeIdentity;
    std::string displayName = typeName;
    if (displayName == identity) {
        std::string key = "component.";
        for (size_t i = 0; i < identity.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(identity[i]);
            if (i > 0 && std::isupper(c) &&
                (!std::isupper(static_cast<unsigned char>(identity[i - 1])) ||
                 (i + 1 < identity.size() && std::islower(static_cast<unsigned char>(identity[i + 1])))))
                key += '_';
            key += static_cast<char>(std::tolower(c));
        }
        const auto localized = Tr(key);
        if (localized != key)
            displayName = localized;
    }
    displayName += suffix;
    const char *domainToken = InspectorComponentDomain(identity);

    // Styling
    ImGui::PushStyleColor(ImGuiCol_Header, EditorThemeRegistry::Color(selected ? "INSPECTOR_HEADER_SELECTED"
                                                                               : "INSPECTOR_HEADER_PRIMARY"));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                          EditorThemeRegistry::Color(selected ? "INSPECTOR_HEADER_SELECTED_HOVERED"
                                                              : "INSPECTOR_HEADER_PRIMARY_HOVERED"));
    ImGui::PushStyleColor(
        ImGuiCol_HeaderActive,
        EditorThemeRegistry::Color(selected ? "INSPECTOR_HEADER_SELECTED_ACTIVE" : "INSPECTOR_HEADER_PRIMARY_ACTIVE"));
    const float rowHeight = EditorThemeRegistry::Float("INSPECTOR_COMPONENT_HEADER_H") * dpi;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(8.0f * dpi, (std::max)(0.0f, (rowHeight - ImGui::GetTextLineHeight()) * 0.5f)));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(EditorTheme::INSPECTOR_HEADER_ITEM_SPC.x * dpi,
                                                          EditorTheme::INSPECTOR_HEADER_ITEM_SPC.y * dpi));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, EditorTheme::INSPECTOR_HEADER_BORDER_SIZE * dpi);
    ImGui::SetWindowFontScale(EditorTheme::INSPECTOR_HEADER_PRIMARY_FONT_SCALE);

    // Full-width collapsing header
    if (defaultOpen)
        ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextItemAllowOverlap();

    std::string headerKey = "##comp_" + headerId;
    {
        float clipMaxX = ImGui::GetWindowPos().x + ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x -
                         EditorTheme::INSPECTOR_HEADER_RIGHT_MARGIN * dpi;
        ImGui::GetWindowDrawList()->PushClipRect(ImVec2(0.0f, 0.0f), ImVec2(clipMaxX, 1e7f), true);
    }
    bool headerOpen = ImGui::CollapsingHeader(headerKey.c_str());
    const bool headerLeftClicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
    const bool headerRightClicked = ImGui::IsItemClicked(ImGuiMouseButton_Right);
    ImGui::GetWindowDrawList()->PopClipRect();
    // CollapsingHeader owns layout for the row.  The controls below are an
    // overlay and must restore this cursor before component contents render.
    const ImVec2 contentCursor = ImGui::GetCursorScreenPos();
    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    const std::string semanticBase =
        captureSemantics ? (semanticId.empty() ? "inspector.component." + headerId : semanticId) : std::string{};
    if (ctx && captureSemantics)
        ctx->RecordSemanticItem("component_header", displayName, true, semanticBase, selected);

    const ImVec2 headerMin = ImGui::GetItemRectMin();
    const ImVec2 headerMax((std::min)(ImGui::GetItemRectMax().x, ImGui::GetCurrentWindow()->WorkRect.Max.x),
                           ImGui::GetItemRectMax().y);
    ImGui::GetWindowDrawList()->AddRectFilled(headerMin, ImVec2(headerMin.x + 3.0f * dpi, headerMax.y),
                                              ImGui::GetColorU32(EditorThemeRegistry::Color(domainToken)));
    ImGui::GetWindowDrawList()->AddLine(headerMin, ImVec2(headerMax.x, headerMin.y),
                                        ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_BORDER")), dpi);

    constexpr const char *componentDragType = "INFERNUX_COMPONENT_ORDER";
    const bool hasInsertionTarget =
        !dragObjectIds.empty() && dragObjectIds.size() == dragComponentIds.size() &&
        std::none_of(dragObjectIds.begin(), dragObjectIds.end(), [](uint64_t value) { return value == 0; }) &&
        std::none_of(dragComponentIds.begin(), dragComponentIds.end(), [](uint64_t value) { return value == 0; });
    const bool canDrag = allowComponentReorder && hasInsertionTarget;
    uint64_t targetComponentId = 0;
    if (hasInsertionTarget) {
        const auto selectedIt = std::find(dragObjectIds.begin(), dragObjectIds.end(), m_selectedObjectId);
        if (selectedIt != dragObjectIds.end())
            targetComponentId = dragComponentIds[static_cast<size_t>(selectedIt - dragObjectIds.begin())];
        else if (dragComponentIds.size() == 1)
            targetComponentId = dragComponentIds.front();
    }
    if (canDrag) {
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceNoDisableHover)) {
            std::vector<uint64_t> payload;
            payload.reserve(1 + dragObjectIds.size() * 2);
            payload.push_back(static_cast<uint64_t>(dragObjectIds.size()));
            payload.insert(payload.end(), dragObjectIds.begin(), dragObjectIds.end());
            payload.insert(payload.end(), dragComponentIds.begin(), dragComponentIds.end());
            ImGui::SetDragDropPayload(componentDragType, payload.data(), payload.size() * sizeof(uint64_t));
            ImGui::TextUnformatted(displayName.c_str());
            ImGui::EndDragDropSource();
        }
    }
    if (ImGui::BeginDragDropTarget()) {
        if (canDrag) {
            if (const ImGuiPayload *payload =
                    ImGui::AcceptDragDropPayload(componentDragType, ImGuiDragDropFlags_AcceptBeforeDelivery)) {
                if (payload->DataSize >= static_cast<int>(sizeof(uint64_t))) {
                    const auto *values = static_cast<const uint64_t *>(payload->Data);
                    const size_t count = static_cast<size_t>(values[0]);
                    const size_t expectedBytes = (1 + count * 2) * sizeof(uint64_t);
                    if (count == dragObjectIds.size() && payload->DataSize == static_cast<int>(expectedBytes)) {
                        const std::vector<uint64_t> draggedObjectIds(values + 1, values + 1 + count);
                        const std::vector<uint64_t> draggedComponentIds(values + 1 + count, values + 1 + count * 2);
                        const bool sameObjects = draggedObjectIds == dragObjectIds;
                        const bool selfDrop = draggedComponentIds == dragComponentIds;
                        if (sameObjects && !selfDrop) {
                            const bool insertAfter = ImGui::GetMousePos().y >= (headerMin.y + headerMax.y) * 0.5f;
                            const float lineY = insertAfter ? headerMax.y : headerMin.y;
                            ImGui::GetWindowDrawList()->AddLine(ImVec2(headerMin.x, lineY), ImVec2(headerMax.x, lineY),
                                                                ImGui::GetColorU32(ImGuiCol_DragDropTarget),
                                                                2.0f * dpi);
                            if (payload->IsDelivery())
                                ExecuteEditorCommand("component.reorder",
                                                     MakeComponentReorderCommandArgument(dragObjectIds,
                                                                                         draggedComponentIds,
                                                                                         dragComponentIds, insertAfter),
                                                     "drag_drop");
                        }
                    }
                }
            }
        }
        if (targetComponentId != 0) {
            std::string path;
            if (ctx->AcceptDragDropPayload("SCRIPT_FILE", &path, ImGuiDragDropFlags_AcceptBeforeDelivery)) {
                const bool insertAfter =
                    !allowComponentReorder || ImGui::GetMousePos().y >= (headerMin.y + headerMax.y) * 0.5f;
                const float lineY = insertAfter ? headerMax.y : headerMin.y;
                ImGui::GetWindowDrawList()->AddLine(ImVec2(headerMin.x, lineY), ImVec2(headerMax.x, lineY),
                                                    ImGui::GetColorU32(ImGuiCol_DragDropTarget), 2.0f * dpi);
                if (ImGui::GetDragDropPayload()->IsDelivery()) {
                    const uint64_t commandTargetId = allowComponentReorder ? targetComponentId : 0;
                    const bool insertAtStart = !allowComponentReorder;
                    if (!ExecuteEditorCommand("component.add",
                                              MakeComponentAddCommandArgument("", false, path, commandTargetId,
                                                                              insertAfter, insertAtStart),
                                              "drag_drop"))
                        INXLOG_ERROR("Inspector rejected script component drop for '", path, "' at component ",
                                     targetComponentId);
                }
            }
        }
        ImGui::EndDragDropTarget();
    }
    float headerMinY = headerMin.y;
    float headerMaxY = headerMax.y;
    float headerHeight = (std::max)(0.0f, headerMaxY - headerMinY);

    // Overlay: icon + checkbox + label on the same row. The header rectangle
    // already contains ImGui's scrolling transform, so keep every overlay item
    // in that same screen-space coordinate system.
    const float overlayX = headerMin.x + 36.0f * dpi;
    ImGui::SetCursorScreenPos(ImVec2(overlayX, headerMin.y));

    const float iconSize = EditorTheme::COMPONENT_ICON_SIZE * dpi;
    // Reserve this slot even while the texture upload is pending. Header
    // geometry must not depend on asynchronous descriptor publication.
    ImGui::Dummy(ImVec2(iconSize, (std::max)(headerHeight, iconSize)));
    const ImVec2 slotMin = ImGui::GetItemRectMin();
    const ImVec2 slotMax = ImGui::GetItemRectMax();
    if (ctx && captureSemantics)
        ctx->RecordSemanticItem("component_icon", displayName, true, semanticBase + ".icon", iconId != 0);
    if (iconId != 0) {
        float drawSize = (std::min)({iconSize, slotMax.x - slotMin.x, slotMax.y - slotMin.y});
        float drawX = slotMin.x + (std::max)(0.0f, (slotMax.x - slotMin.x - drawSize) * 0.5f);
        float drawY = slotMin.y + (std::max)(0.0f, (slotMax.y - slotMin.y - drawSize) * 0.5f);
        // Snap to integer pixels to avoid bilinear mush on subpixel edges (common vs Unity crisp UI).
        drawSize = std::floor(drawSize);
        drawX = std::floor(drawX);
        drawY = std::floor(drawY);

        ImDrawList *drawList = ImGui::GetWindowDrawList();
        ImTextureRef texRef(static_cast<ImTextureID>(iconId));
        drawList->AddImage(texRef, ImVec2(drawX, drawY), ImVec2(drawX + drawSize, drawY + drawSize));
    }

    const float controlsX = overlayX + iconSize + 8.0f * dpi;
    const float optionsWidth = contextPopupId.empty() ? 0.0f : 24.0f * dpi;
    const float checkboxWidth = showEnabled ? 22.0f * dpi : 0.0f;
    const float controlsRight = headerMax.x - 8.0f * dpi;
    const float labelRight = controlsRight - optionsWidth - checkboxWidth - 8.0f * dpi;
    ctx->PushUIFont(EditorThemeRegistry::Float("EDITOR_TITLE_FONT_SIZE"));
    ImGui::SetCursorScreenPos(ImVec2(controlsX, headerMin.y + (headerHeight - ImGui::GetTextLineHeight()) * 0.5f));
    ImGui::PushStyleColor(ImGuiCol_Text, EditorThemeRegistry::Color("TEXT"));
    ImGui::RenderTextClipped(ImGui::GetCursorScreenPos(), ImVec2(labelRight, headerMax.y), displayName.c_str(), nullptr,
                             nullptr, ImVec2(0.0f, 0.0f));
    const float titleWidth = ImGui::CalcTextSize(displayName.c_str()).x;
    ImGui::PopStyleColor();
    ctx->PopFont();
    if (displayName != identity && controlsX + titleWidth + 40.0f * dpi < labelRight) {
        ctx->PushMonoFont();
        auto *draw = ImGui::GetWindowDrawList();
        draw->PushClipRect(ImVec2(controlsX, headerMin.y), ImVec2(labelRight, headerMax.y), true);
        const float fontSize = EditorThemeRegistry::Float("EDITOR_SMALL_FONT_SIZE") * dpi;
        draw->AddText(ImGui::GetFont(), fontSize,
                      ImVec2(controlsX + titleWidth + 10.0f * dpi, headerMin.y + (headerHeight - fontSize) * 0.5f),
                      ImGui::GetColorU32(ImGuiCol_TextDisabled), identity.c_str());
        draw->PopClipRect();
        ctx->PopFont();
    }
    if (ctx && captureSemantics)
        ctx->RecordSemanticRect("component_label", displayName, controlsX, headerMin.y,
                                (std::max)(0.0f, labelRight - controlsX), headerHeight, true, semanticBase + ".label");
    bool enabledClicked = false;
    if (showEnabled) {
        const float checkboxRowHeight =
            (std::max)(EditorThemeRegistry::Float("INSPECTOR_CHECKBOX_BOX_PX") * dpi, ImGui::GetTextLineHeight());
        ImGui::SetCursorScreenPos(ImVec2(controlsRight - optionsWidth - checkboxWidth,
                                         headerMin.y + (headerHeight - checkboxRowHeight) * 0.5f));
        ctx->CheckboxInspector("##hdr_en", &newEnabled);
        enabledClicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
        if (ctx && captureSemantics)
            ctx->RecordSemanticItem("component_enabled", displayName, true, semanticBase + ".enabled");
    }

    bool optionsClicked = false;
    if (!contextPopupId.empty()) {
        ImGui::SetCursorScreenPos(
            ImVec2(controlsRight - optionsWidth, headerMin.y + (headerHeight - 22.0f * dpi) * 0.5f));
        if (ctx->IconButton("component_options", "glyphs/more", Tr("inspector.component_options"), optionsWidth,
                            22.0f * dpi)) {
            optionsClicked = true;
            ImGui::OpenPopup(contextPopupId.c_str());
        }
        const std::string optionsLabel = Tr("inspector.component_options");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
            ImGui::SetTooltip("%s", optionsLabel.c_str());
        if (ctx && captureSemantics)
            ctx->RecordSemanticItem("component_options", optionsLabel, true, semanticBase + ".options");
        if (ImGui::IsMouseHoveringRect(headerMin, headerMax) && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
            ImGui::OpenPopup(contextPopupId.c_str());
    }

    // The icon, label, enabled checkbox and options button are intentionally
    // drawn over the collapsing header.  Treat the complete visible row as
    // the component selection target; otherwise clicks over non-interactive
    // overlay text miss the header and selection appears to require a second
    // click.  Child controls still keep their own action semantics below.
    const bool headerRowClicked =
        ImGui::IsMouseHoveringRect(headerMin, headerMax, false) &&
        (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Right));

    ImGui::SetCursorScreenPos(contentCursor);

    // Cleanup
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(3);

    return {headerOpen, newEnabled,
            (headerLeftClicked || headerRightClicked || headerRowClicked || optionsClicked) && !enabledClicked};
}

// ============================================================================
// Inspector checkbox
// ============================================================================

bool InspectorPanel::RenderInspectorCheckbox(InxGUIContext *ctx, const char *label, bool value)
{
    bool newValue = value;
    if (ctx)
        ctx->CheckboxInspector(label ? label : "##cb", &newValue);
    return newValue;
}

// ============================================================================
// Add Component button + popup
// ============================================================================

void InspectorPanel::RenderAddComponentButton(InxGUIContext *ctx)
{
    const float dpi = ctx->GetDpiScale();
    const std::string label = Tr("inspector.add_component");
    const bool clicked =
        ImGui::InvisibleButton("##add_component", ImVec2(ImGui::GetContentRegionAvail().x, 30.0f * dpi));
    const ImVec2 lo = ImGui::GetItemRectMin();
    const ImVec2 hi = ImGui::GetItemRectMax();
    auto *draw = ImGui::GetWindowDrawList();
    if (ImGui::IsItemHovered())
        draw->AddRectFilled(lo, hi, ImGui::GetColorU32(EditorThemeRegistry::Color("BTN_GHOST_HOVERED")));
    const ImU32 edge = ImGui::GetColorU32(ImGuiCol_TextDisabled);
    for (float x = lo.x; x < hi.x; x += 7.0f * dpi) {
        const float end = (std::min)(x + 4.0f * dpi, hi.x);
        draw->AddLine(ImVec2(x, lo.y), ImVec2(end, lo.y), edge, dpi);
        draw->AddLine(ImVec2(x, hi.y - dpi), ImVec2(end, hi.y - dpi), edge, dpi);
    }
    for (float y = lo.y; y < hi.y; y += 7.0f * dpi) {
        const float end = (std::min)(y + 4.0f * dpi, hi.y);
        draw->AddLine(ImVec2(lo.x, y), ImVec2(lo.x, end), edge, dpi);
        draw->AddLine(ImVec2(hi.x - dpi, y), ImVec2(hi.x - dpi, end), edge, dpi);
    }
    const float textW = ImGui::CalcTextSize(label.c_str()).x;
    const float left = lo.x + (hi.x - lo.x - textW - 22.0f * dpi) * 0.5f;
    const auto icon = ctx->EditorIcon("glyphs/plus");
    if (icon)
        draw->AddImage(ImTextureRef(static_cast<ImTextureID>(icon)), ImVec2(left, lo.y + 8.0f * dpi),
                       ImVec2(left + 14.0f * dpi, lo.y + 22.0f * dpi));
    draw->AddText(ImVec2(left + 22.0f * dpi, lo.y + (hi.y - lo.y - ImGui::GetTextLineHeight()) * 0.5f),
                  ImGui::GetColorU32(ImGuiCol_Text), label.c_str());
    if (ctx && InxGUISemantics::IsCaptureEnabled())
        ctx->RecordSemanticItem("add_component", label, true, "inspector.add_component");
    if (clicked) {
        m_addCompSearch[0] = '\0';
        if (getAddComponentEntries)
            m_addCompEntries = getAddComponentEntries();
        m_addCompNeedsFocus = true;
        m_addCompCloseRequested = false;
        ImGui::OpenPopup("##add_component_popup");
    }
}

void InspectorPanel::RenderAddComponentPopup(InxGUIContext *ctx)
{
    // Unity-style component browser: generous bleed, full-width search and
    // tall rows with a clear hover band (shared popup chrome below).
    const float dpi = ctx->GetDpiScale();
    const float em = ImGui::GetFontSize();
    ImGui::SetNextWindowSizeConstraints(ImVec2(em * 18.0f, 0.0f), ImVec2(FLT_MAX, FLT_MAX));

    const bool popupVisible = ctx->BeginPopup("##add_component_popup");
    if (popupVisible) {
        if (!m_addCompPopupOpen) {
            m_addCompPopupOpen = true;
            BeginTransientInteraction("add_component_popup", "popup", 200, [this]() {
                m_addCompCloseRequested = true;
                return true;
            });
        }
        const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
        if (ctx && captureSemantics)
            ctx->RecordSemanticWindow("add_component_popup", "Add Component", "inspector.add_component.popup");
        // Search field
        if (m_addCompNeedsFocus)
            ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(std::max(EditorTheme::ADD_COMP_SEARCH_W * dpi, ImGui::GetContentRegionAvail().x));
        const bool submitFirst =
            ImGui::InputTextWithHint("##comp_search", Tr("inspector.search_components").c_str(), m_addCompSearch,
                                     sizeof(m_addCompSearch), ImGuiInputTextFlags_EnterReturnsTrue);
        if (ctx && captureSemantics)
            ctx->RecordSemanticItem("component_search", Tr("inspector.search_components"), true,
                                    "inspector.add_component.search");
        const bool searchFocused = ImGui::IsItemFocused();
        if (searchFocused)
            m_addCompNeedsFocus = false;
        const bool focusFirst = searchFocused && ImGui::IsKeyPressed(ImGuiKey_DownArrow);
        if (m_addCompCloseRequested) {
            ImGui::CloseCurrentPopup();
            m_addCompCloseRequested = false;
            m_addCompPopupOpen = false;
        }

        ImGui::Separator();

        // Scrollable region (transparent bg inside popup via BeginChild)
        if (ctx->BeginChild("##comp_list", 0, 350.0f * dpi, false, 0)) {
            std::string searchLower(m_addCompSearch);
            std::transform(searchLower.begin(), searchLower.end(), searchLower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            bool foundAny = false;
            bool assignedKeyboardFocus = false;
            bool handledKeyboardSelection = false;
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

            const float rowHeight = ImGui::GetTextLineHeight() + ImGui::GetFontSize() * 0.55f;
            for (const auto &cat : categoryOrder) {
                // Muted category header, Unity-style section grouping.
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.55f, 0.55f, 1.0f));
                ImGui::TextUnformatted(cat.c_str());
                ImGui::PopStyleColor();
                ImGui::Separator();

                for (const auto *entry : categories[cat]) {
                    foundAny = true;
                    ++uid;
                    std::string selectLabel = "  " + entry->displayName + "##" + std::to_string(uid);
                    if (focusFirst && !assignedKeyboardFocus) {
                        ImGui::SetKeyboardFocusHere();
                        assignedKeyboardFocus = true;
                    }
                    const bool activateFromKeyboard = submitFirst && !handledKeyboardSelection;
                    // Taller rows with vertically centered labels give the
                    // hover highlight a solid, easy-to-track band.
                    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.0f, 0.5f));
                    const bool selected = ImGui::Selectable(selectLabel.c_str(), false, 0, ImVec2(0.0f, rowHeight));
                    ImGui::PopStyleVar();
                    if (ctx && captureSemantics)
                        ctx->RecordSemanticItem("component_option", entry->displayName, true,
                                                "inspector.add_component.option." + std::to_string(uid));
                    if (selected || activateFromKeyboard) {
                        if (ExecuteEditorCommand("component.add",
                                                 MakeComponentAddCommandArgument(entry->displayName, entry->isNative,
                                                                                 entry->scriptPath))) {
                            // The selection stays unchanged, so the normal selection-driven
                            // invalidation path does not run after adding a component.
                            InvalidateObjectCaches();
                        }
                        ImGui::CloseCurrentPopup();
                        m_addCompPopupOpen = false;
                        EndTransientInteraction("add_component_popup");
                        handledKeyboardSelection = true;
                    }
                }
                ImGui::Dummy(ImVec2(0, 4.0f * dpi));
            }

            if (!foundAny) {
                ImGui::TextUnformatted(Tr("inspector.no_components_found").c_str());
            }
        }
        ctx->EndChild();
        ctx->EndPopup();
    } else if (m_addCompPopupOpen) {
        m_addCompPopupOpen = false;
        m_addCompCloseRequested = false;
        EndTransientInteraction("add_component_popup");
    }
}

} // namespace infernux
