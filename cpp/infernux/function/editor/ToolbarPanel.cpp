#include "ToolbarPanel.h"
#include <function/editor/EditorThemeRegistry.h>
#include <function/renderer/gui/InxGUISemantics.h>

#include <algorithm>
#include <cstdio>
#include <iterator>

namespace infernux
{

// ════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════

ToolbarPanel::ToolbarPanel() : EditorPanel("Toolbar", "toolbar")
{
}

// ════════════════════════════════════════════════════════════════════
// Camera settings
// ════════════════════════════════════════════════════════════════════

ToolbarPanel::CameraSettings ToolbarPanel::GetCameraSettings() const
{
    return m_cameraSettings;
}

void ToolbarPanel::SetCameraSettings(const CameraSettings &settings)
{
    m_cameraSettings = settings;
    if (applyCameraToEngine)
        applyCameraToEngine(m_cameraSettings);
}

// ════════════════════════════════════════════════════════════════════
// Translation helper
// ════════════════════════════════════════════════════════════════════

std::string ToolbarPanel::T(const std::string &key) const
{
    if (translate)
        return translate(key);
    // Fallback: return the key suffix after the last dot
    auto dot = key.rfind('.');
    return (dot != std::string::npos) ? key.substr(dot + 1) : key;
}

// ════════════════════════════════════════════════════════════════════
// Window flags & pre-render
// ════════════════════════════════════════════════════════════════════

ImGuiWindowFlags ToolbarPanel::GetWindowFlags() const
{
    return ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
           ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
           ImGuiWindowFlags_NoScrollWithMouse;
}

void ToolbarPanel::PreRender(InxGUIContext *ctx)
{
    // These authored overrides replace the already-scaled global style.
    const float dpi = ctx->GetDpiScale();
    const auto scaled = [dpi](ImVec2 size) { return ImVec2(size.x * dpi, size.y * dpi); };
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, scaled(EditorTheme::TOOLBAR_WIN_PAD));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, scaled(EditorTheme::TOOLBAR_FRAME_PAD));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, scaled(EditorTheme::TOOLBAR_ITEM_SPC));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, EditorTheme::TOOLBAR_FRAME_RND * dpi);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, EditorTheme::TOOLBAR_FRAME_BRD * dpi);
}

// ════════════════════════════════════════════════════════════════════
// Main content
// ════════════════════════════════════════════════════════════════════

void ToolbarPanel::OnRenderContent(InxGUIContext *ctx)
{
    if (!ImGui::BeginMenuBar())
        return;
    const float dpi = ctx->GetDpiScale();
    ImGui::SetCursorPosY((EditorThemeRegistry::Float("TOP_BAR_HEIGHT") - 26.0f) * 0.5f * dpi);
    RenderPlayControls(ctx, ctx->GetWindowWidth());
    const PlayState state = getPlayState ? getPlayState() : PlayState::Edit;
    if (state != PlayState::Edit) {
        const float dpi = ctx->GetDpiScale();
        const ImVec2 origin = ImGui::GetWindowPos();
        const ImVec2 end(origin.x + ImGui::GetWindowWidth(), origin.y + ImGui::GetWindowHeight());
        const ImU32 color = ImGui::ColorConvertFloat4ToU32(
            EditorThemeRegistry::Color(state == PlayState::Paused ? "LOG_WARNING" : "ROLE_ACCENT"));
        auto *draw = ImGui::GetWindowDrawList();
        draw->PushClipRect(ImVec2(origin.x, end.y - 3.0f * dpi), end, true);
        for (float x = origin.x; x < end.x + 6.0f * dpi; x += 12.0f * dpi)
            draw->AddLine(ImVec2(x, end.y), ImVec2(x + 6.0f * dpi, end.y - 6.0f * dpi), color, 3.0f * dpi);
        draw->PopClipRect();
    }
    ImGui::EndMenuBar();
}

void ToolbarPanel::PostRender(InxGUIContext * /*ctx*/)
{
    // Pop the 5 style vars pushed in PreRender (always, even when collapsed)
    ImGui::PopStyleVar(5);
}

// ════════════════════════════════════════════════════════════════════
// Play controls (centered)
// ════════════════════════════════════════════════════════════════════

void ToolbarPanel::RenderPlayControls(InxGUIContext *ctx, float winW)
{
    const float dpi = ctx->GetDpiScale();
    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    PlayState state = getPlayState ? getPlayState() : PlayState::Edit;
    bool isPlaying = (state == PlayState::Playing || state == PlayState::Paused);
    bool isPaused = (state == PlayState::Paused);
    const bool canTogglePlay = canExecuteCommand && canExecuteCommand("play.toggle", "");
    const bool canTogglePause = canExecuteCommand && canExecuteCommand("play.pause", "");
    const bool canStep = canExecuteCommand && canExecuteCommand("play.step", "");

    float btnW = 106.0f * dpi;
    float cx = (winW - btnW) * 0.5f;
    if (cx < 6.0f * dpi)
        cx = 6.0f * dpi;
    ImGui::SetCursorPosX(cx);

    // ── Play / Stop ──────────────────────────────────────────────
    if (isPlaying && !isPaused)
        EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Color("PLAY_ACTIVE"));
    else
        EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Color("BTN_IDLE"));

    std::string playLabel = isPlaying ? T("toolbar.stop") : T("toolbar.play");
    ImGui::BeginDisabled(!canTogglePlay);
    const bool playClicked =
        ctx->IconButton("toolbar_play", isPlaying ? "glyphs/stop" : "glyphs/play", playLabel, 34.0f * dpi, 26.0f * dpi);
    ImGui::EndDisabled();
    if (captureSemantics)
        ctx->RecordSemanticItem("toolbar_play_stop", playLabel, canTogglePlay, "toolbar.play_stop");
    if (playClicked && canTogglePlay && executeCommand)
        executeCommand("play.toggle", "toolbar", "");
    ImGui::PopStyleColor(3);

    ImGui::SameLine(0.0f, 2.0f * dpi);

    // ── Pause / Resume ───────────────────────────────────────────
    if (!isPlaying)
        EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Color("BTN_DISABLED"));
    else if (isPaused)
        EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Color("PAUSE_ACTIVE"));
    else
        EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Color("BTN_IDLE"));

    std::string pauseLabel = isPaused ? T("toolbar.resume") : T("toolbar.pause");
    ImGui::BeginDisabled(!canTogglePause);
    const bool pauseClicked = ctx->IconButton("toolbar_pause", "glyphs/pause", pauseLabel, 34.0f * dpi, 26.0f * dpi);
    ImGui::EndDisabled();
    if (captureSemantics)
        ctx->RecordSemanticItem("toolbar_pause_resume", pauseLabel, canTogglePause, "toolbar.pause_resume");
    if (pauseClicked && canTogglePause && executeCommand)
        executeCommand("play.pause", "toolbar", "");
    ImGui::PopStyleColor(3);

    ImGui::SameLine(0.0f, 2.0f * dpi);

    // ── Step ─────────────────────────────────────────────────────
    if (isPaused)
        EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Color("BTN_IDLE"));
    else
        EditorTheme::PushFlatButtonStyle(EditorThemeRegistry::Color("BTN_DISABLED"));

    std::string stepLabel = T("toolbar.step");
    ImGui::BeginDisabled(!canStep);
    const bool stepClicked = ctx->IconButton("toolbar_step", "glyphs/step_f", stepLabel, 34.0f * dpi, 26.0f * dpi);
    ImGui::EndDisabled();
    if (captureSemantics)
        ctx->RecordSemanticItem("toolbar_step", stepLabel, canStep, "toolbar.step");
    if (stepClicked && canStep && executeCommand)
        executeCommand("play.step", "toolbar", "");
    ImGui::PopStyleColor(3);

    // ── Time label while playing ─────────────────────────────────
    {
        std::string timeStr = getPlayTimeStr ? getPlayTimeStr() : "00:00.000";
        const std::string status = "MET  " + timeStr;
        const float rightControlsX = std::max(winW - 200.0f * dpi, 300.0f * dpi);
        const float statusStartX = ImGui::GetItemRectMax().x + 8.0f * dpi;
        const float rightControlsScreenX = ImGui::GetWindowPos().x + rightControlsX;
        const float statusWidth = ImGui::CalcTextSize(status.c_str()).x;
        if (statusStartX + statusWidth + 8.0f * dpi <= rightControlsScreenX) {
            ImGui::SameLine(0.0f, 8.0f * dpi);
            ctx->PushMonoFont();
            ImGui::TextColored(
                EditorThemeRegistry::Color(isPaused ? "LOG_WARNING" : (isPlaying ? "ROLE_ACCENT" : "TEXT_DISABLED")),
                "%s", status.c_str());
            ctx->PopFont();
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Right-aligned dropdowns
// ════════════════════════════════════════════════════════════════════

void ToolbarPanel::RenderViewportControls(InxGUIContext *ctx)
{
    const float dpi = ctx->GetDpiScale();
    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    ImGui::BeginGroup();
    // Toolbar dropdowns sit on the bar as ghost frames; the chevron marks them.
    // The ghost colours wrap only the frame so popup content keeps field styling.
    auto beginDropdown = [ctx](const char *id, const std::string &label) {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, EditorThemeRegistry::Colors().at("BTN_GHOST"));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, EditorThemeRegistry::Colors().at("BTN_GHOST_HOVERED"));
        const bool open = ctx->BeginCombo(id, label, ImGuiComboFlags_WidthFitPreview | ImGuiComboFlags_HeightLargest);
        ImGui::PopStyleColor(2);
        return open;
    };

    // Gizmos dropdown
    const std::string gizLabel = T("toolbar.gizmos");
    if (beginDropdown("##giz", gizLabel)) {
        if (captureSemantics)
            ctx->RecordSemanticWindow("toolbar_popup", gizLabel, "toolbar.gizmos.popup");
        PopupGizmos(ctx);
        ctx->EndCombo();
    }
    if (captureSemantics)
        ctx->RecordSemanticItem("toolbar_gizmos", gizLabel, true, "toolbar.gizmos");

    ImGui::SameLine(0.0f, 4.0f * dpi);

    // Camera dropdown
    const std::string camLabel = T("toolbar.camera");
    if (beginDropdown("##cam", camLabel)) {
        if (captureSemantics)
            ctx->RecordSemanticWindow("toolbar_popup", camLabel, "toolbar.camera.popup");
        PopupCamera(ctx);
        ctx->EndCombo();
    }
    if (captureSemantics)
        ctx->RecordSemanticItem("toolbar_camera", camLabel, true, "toolbar.camera");
    ImGui::EndGroup();
}

// ════════════════════════════════════════════════════════════════════
// Popup: Gizmos
// ════════════════════════════════════════════════════════════════════

void ToolbarPanel::PopupGizmos(InxGUIContext *ctx)
{
    if (!isShowGrid || !isShowGizmos) {
        ImGui::TextUnformatted(T("toolbar.engine_not_available").c_str());
        return;
    }

    ImGui::Dummy(ImVec2(200.0f * ctx->GetDpiScale(), 0.0f)); // minimum popup width
    ImGui::TextUnformatted(T("toolbar.gizmos_header").c_str());
    ImGui::Separator();

    bool gizmos = isShowGizmos();
    const std::string gizmosLabel = T("toolbar.show_gizmos");
    const bool gizmosChanged = ctx->Checkbox(gizmosLabel, &gizmos);
    if (InxGUISemantics::IsCaptureEnabled())
        ctx->RecordSemanticItem("toolbar_show_gizmos", gizmosLabel, true, "toolbar.gizmos.show_gizmos");
    if (gizmosChanged && executeCommand)
        executeCommand("scene.toggle_gizmos", "toolbar", "");

    bool grid = isShowGrid();
    const std::string gridLabel = T("toolbar.show_grid");
    const bool gridChanged = ctx->Checkbox(gridLabel, &grid);
    if (InxGUISemantics::IsCaptureEnabled())
        ctx->RecordSemanticItem("toolbar_show_grid", gridLabel, true, "toolbar.gizmos.show_grid");
    if (gridChanged && executeCommand)
        executeCommand("scene.toggle_grid", "toolbar", "");
}

// ════════════════════════════════════════════════════════════════════
// Popup: Camera
// ════════════════════════════════════════════════════════════════════

void ToolbarPanel::PopupCamera(InxGUIContext *ctx)
{
    const float dpi = ctx->GetDpiScale();
    // Sync from engine
    if (syncCameraFromEngine)
        m_cameraSettings = syncCameraFromEngine();

    ImGui::Dummy(ImVec2(360.0f * dpi, 0.0f)); // minimum popup width
    ImGui::TextUnformatted(T("toolbar.scene_camera").c_str());
    ImGui::Separator();

    ImGui::TextUnformatted(T("toolbar.projection_mode").c_str());
    ImGui::SameLine(145.0f * dpi);
    ImGui::SetNextItemWidth(200.0f * dpi);
    int projection = m_cameraSettings.orthographic ? 1 : 0;
    const CameraSettings projectionBefore = m_cameraSettings;
    const bool projectionChanged =
        ctx->Combo("##camera_projection", &projection, {T("toolbar.perspective"), T("toolbar.orthographic")});
    m_cameraSettings.orthographic = projection == 1;
    if (ImGui::IsItemActivated() && beginCameraEdit)
        beginCameraEdit("projection", projectionBefore);
    if (projectionChanged && applyCameraToEngine)
        applyCameraToEngine(m_cameraSettings);
    if (ImGui::IsItemDeactivatedAfterEdit() && endCameraEdit)
        endCameraEdit("projection", m_cameraSettings);

    struct CamParam
    {
        const char *key;
        float *value;
        float mn, mx, step, stepFast;
        const char *headerKey; // null if no header
        // Percent remap: when nonzero, the slider works in display-percent space
        // (value*scale), clamped to [mn, mx], and writes value/scale back. This
        // lets raw values with awkward magnitudes (e.g. 0.005 deg/px, 5.0 m/s)
        // be presented as an intuitive percentage instead of leading zeros.
        // The baseline differs per parameter: "100%" always means the default.
        float percentScale = 0.0f;
    };

    CamParam perspectiveParams[] = {
        {"toolbar.field_of_view", &m_cameraSettings.fov, 10.0f, 120.0f, 1.0f, 10.0f, nullptr},
    };
    CamParam orthographicParams[] = {
        {"toolbar.orthographic_size", &m_cameraSettings.orthographicSize, 0.01f, 1000.0f, 0.1f, 1.0f, nullptr},
    };
    // 100% == default for every navigation parameter. Allow up to 4x the
    // default for navigating large scenes without changing existing settings.
    constexpr float kRotationPercentScale = 100.0f / CAMERA_DEFAULTS_ROTATION;
    constexpr float kPanPercentScale = 100.0f / CAMERA_DEFAULTS_PAN;
    constexpr float kZoomPercentScale = 100.0f / CAMERA_DEFAULTS_ZOOM;
    constexpr float kMovePercentScale = 100.0f / CAMERA_DEFAULTS_MOVE;
    // Speed boost baseline: 100% == default (3.0x).
    constexpr float kBoostPercentScale = 100.0f / CAMERA_DEFAULTS_BOOST;
    CamParam navigationParams[] = {
        {"toolbar.rotation_sensitivity", &m_cameraSettings.rotationSpeed, 25.0f, 400.0f, 5.0f, 25.0f,
         "toolbar.navigation_header", kRotationPercentScale},
        {"toolbar.pan_speed", &m_cameraSettings.panSpeed, 25.0f, 400.0f, 5.0f, 25.0f, nullptr, kPanPercentScale},
        {"toolbar.zoom_speed", &m_cameraSettings.zoomSpeed, 25.0f, 400.0f, 5.0f, 25.0f, nullptr, kZoomPercentScale},
        {"toolbar.move_speed", &m_cameraSettings.moveSpeed, 25.0f, 400.0f, 5.0f, 25.0f, nullptr, kMovePercentScale},
        {"toolbar.speed_boost", &m_cameraSettings.moveSpeedBoost, 25.0f, 400.0f, 5.0f, 25.0f, nullptr,
         kBoostPercentScale},
    };

    auto renderParams = [&](CamParam *params, size_t count) {
        for (size_t index = 0; index < count; ++index) {
            auto &p = params[index];
            if (p.headerKey) {
                ImGui::TextUnformatted(T(p.headerKey).c_str());
                ImGui::Separator();
            }

            ImGui::TextUnformatted(T(p.key).c_str());
            ImGui::SameLine(145.0f * dpi);

            // Unity-style ranged slider (thin track + grab + numeric input),
            // matching the Game View scale slider. No FrameBg box.
            char sliderId[64];
            snprintf(sliderId, sizeof(sliderId), "##%s_value", p.key);
            ImGui::SetNextItemWidth(210.0f * dpi);
            const CameraSettings valueBefore = m_cameraSettings;
            const float valueBeforeFloat = *p.value;

            const bool isPercent = p.percentScale > 0.0f;
            float displayValue = isPercent ? (*p.value * p.percentScale) : *p.value;
            ctx->FloatSlider(sliderId, &displayValue, p.mn, p.mx, isPercent ? "%.0f%%" : nullptr);
            // Clamp in display space, then map back to the raw value.
            displayValue = (std::min)((std::max)(displayValue, p.mn), p.mx);
            *p.value = isPercent ? (displayValue / p.percentScale) : displayValue;

            const bool valueChanged = *p.value != valueBeforeFloat;
            if (ImGui::IsItemActivated() && beginCameraEdit)
                beginCameraEdit(p.key, valueBefore);
            if (valueChanged && applyCameraToEngine)
                applyCameraToEngine(m_cameraSettings);
            if (ImGui::IsItemDeactivatedAfterEdit() && endCameraEdit)
                endCameraEdit(p.key, m_cameraSettings);
        }
    };

    if (m_cameraSettings.orthographic)
        renderParams(orthographicParams, std::size(orthographicParams));
    else
        renderParams(perspectiveParams, std::size(perspectiveParams));
    renderParams(navigationParams, std::size(navigationParams));

    // Reset button
    ImGui::Dummy(ImVec2(0.0f, 2.0f * dpi));
    if (ImGui::Button(T("toolbar.reset_camera_settings").c_str(), ImVec2(-1.0f, 0.0f))) {
        const CameraSettings resetBefore = m_cameraSettings;
        if (beginCameraEdit)
            beginCameraEdit("reset", resetBefore);
        m_cameraSettings.fov = CAMERA_DEFAULTS_FOV;
        m_cameraSettings.orthographic = false;
        m_cameraSettings.orthographicSize = CAMERA_DEFAULTS_ORTHOGRAPHIC_SIZE;
        m_cameraSettings.rotationSpeed = CAMERA_DEFAULTS_ROTATION;
        m_cameraSettings.panSpeed = CAMERA_DEFAULTS_PAN;
        m_cameraSettings.zoomSpeed = CAMERA_DEFAULTS_ZOOM;
        m_cameraSettings.moveSpeed = CAMERA_DEFAULTS_MOVE;
        m_cameraSettings.moveSpeedBoost = CAMERA_DEFAULTS_BOOST;
        if (applyCameraToEngine)
            applyCameraToEngine(m_cameraSettings);
        if (endCameraEdit)
            endCameraEdit("reset", m_cameraSettings);
    }
    ImGui::Dummy(ImVec2(0.0f, 4.0f * dpi));
}

} // namespace infernux
