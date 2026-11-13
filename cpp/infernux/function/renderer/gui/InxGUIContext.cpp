#include "InxGUIContext.h"
#include "EditorMenuItem.h"
#include "EditorWindowBounds.h"
#include "ImageFillDraw.h"
#include "InxGUISemantics.h"
#include "InxTextGeometry.h"
#include "InxTextLayout.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <function/editor/EditorTheme.h>
#include <function/editor/EditorThemeRegistry.h>
#include <imgui_internal.h>
#include <limits>
#include <misc/cpp/imgui_stdlib.h>
#include <stdexcept>
#include <type_traits>

namespace infernux
{

float InxGUIContext::s_dpiScale = 1.0f;
ImFont *InxGUIContext::s_monoFont = nullptr;

float InxGUIContext::GetDpiScale() const
{
    return s_dpiScale;
}

float InxGUIContext::InspectorLabelWidth() const
{
    return ImGui::GetCursorPosX() +
           (std::max)(EditorThemeRegistry::Float("INSPECTOR_MIN_LABEL_WIDTH") * GetDpiScale(),
                      ImGui::GetContentRegionAvail().x * EditorThemeRegistry::Float("INSPECTOR_LABEL_FRACTION"));
}

void InxGUIContext::InspectorFieldLabel(const std::string &text, float valueColumn)
{
    if (valueColumn <= 0.0f)
        valueColumn = InspectorLabelWidth();
    FieldLabel(text, (std::max)(1.0f, valueColumn - ImGui::GetCursorPosX()));
    // SameLine's explicit offset includes GroupOffset. Inspector columns are
    // window-local coordinates, including inside nested serialized groups.
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::SetCursorPosX(valueColumn);
    ImGui::SetNextItemWidth(-1.0f);
}

float InxGUIContext::TextLineHeightWithSpacing() const
{
    return ImGui::GetTextLineHeightWithSpacing();
}

void InxGUIContext::InspectorFieldInfo(const std::string &text, float valueColumn)
{
    const float dpi = GetDpiScale();
    ImGui::SetCursorPosX(valueColumn > 0.0f ? valueColumn : InspectorLabelWidth());
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(9.0f * dpi, ImGui::GetTextLineHeight()));
    const auto icon = EditorIcon("glyphs/info");
    if (icon)
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)icon, ImVec2(origin.x, origin.y + 2.0f * dpi),
                                             ImVec2(origin.x + 9.0f * dpi, origin.y + 11.0f * dpi), ImVec2(0, 0),
                                             ImVec2(1, 1), ImGui::GetColorU32(ImGuiCol_TextDisabled));
    ImGui::SameLine(0.0f, 4.0f * dpi);
    ImGui::PushStyleColor(ImGuiCol_Text, EditorThemeRegistry::Color("TEXT_DISABLED"));
    ImGui::TextWrapped("%s", text.c_str());
    ImGui::PopStyleColor();
}

void InxGUIContext::ReadonlyField(const std::string &id, const std::string &text)
{
    const float dpi = GetDpiScale();
    const float height = ImGui::GetFrameHeight();
    const float width = (std::max)(1.0f, ImGui::GetContentRegionAvail().x - 20.0f * dpi);
    ImGui::InvisibleButton(id.c_str(), ImVec2(width, height));
    const ImVec2 lo = ImGui::GetItemRectMin();
    const ImVec2 hi = ImGui::GetItemRectMax();
    const ImU32 color = ImGui::GetColorU32(ImGuiCol_TextDisabled);
    auto *draw = ImGui::GetWindowDrawList();
    for (float x = lo.x; x < hi.x; x += 5.0f * dpi) {
        const float end = (std::min)(x + 3.0f * dpi, hi.x);
        draw->AddLine(ImVec2(x, lo.y), ImVec2(end, lo.y), color, dpi);
        draw->AddLine(ImVec2(x, hi.y - dpi), ImVec2(end, hi.y - dpi), color, dpi);
    }
    for (float y = lo.y; y < hi.y; y += 5.0f * dpi) {
        const float end = (std::min)(y + 3.0f * dpi, hi.y);
        draw->AddLine(ImVec2(lo.x, y), ImVec2(lo.x, end), color, dpi);
        draw->AddLine(ImVec2(hi.x - dpi, y), ImVec2(hi.x - dpi, end), color, dpi);
    }
    PushMonoFont();
    draw->PushClipRect(lo, hi, true);
    draw->AddText(ImVec2(lo.x + 6.0f * dpi, lo.y + (height - ImGui::GetTextLineHeight()) * 0.5f), color, text.c_str());
    draw->PopClipRect();
    PopFont();
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        ImGui::SetClipboardText(text.c_str());
    const auto lock = EditorIcon("glyphs/lock");
    if (lock)
        draw->AddImage((ImTextureID)(intptr_t)lock, ImVec2(hi.x + 5.0f * dpi, lo.y + 5.0f * dpi),
                       ImVec2(hi.x + 15.0f * dpi, lo.y + 15.0f * dpi), ImVec2(0, 0), ImVec2(1, 1), color);
}

void InxGUIContext::EmptyState(const std::string &id, const std::string &icon, const std::string &title,
                               const std::string &hint, float height, bool interactive)
{
    const float dpi = GetDpiScale();
    const ImVec2 size(ImGui::GetContentRegionAvail().x,
                      height > 0.0f ? height : (std::max)(100.0f * dpi, ImGui::GetContentRegionAvail().y));
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    if (interactive)
        ImGui::InvisibleButton(id.c_str(), size);
    else
        ImGui::Dummy(size);
    const float centerX = origin.x + size.x * 0.5f;
    const float centerY = origin.y + size.y * 0.5f;
    auto *draw = ImGui::GetWindowDrawList();
    const ImU32 dim = ImGui::GetColorU32(ImGuiCol_TextDisabled);
    const auto texture = EditorIcon(icon);
    if (texture)
        draw->AddImage((ImTextureID)(intptr_t)texture, ImVec2(centerX - 13.0f * dpi, centerY - 48.0f * dpi),
                       ImVec2(centerX + 13.0f * dpi, centerY - 22.0f * dpi), ImVec2(0, 0), ImVec2(1, 1), dim);
    const ImVec2 titleSize = ImGui::CalcTextSize(title.c_str());
    draw->AddText(ImVec2(centerX - titleSize.x * 0.5f, centerY - 10.0f * dpi), ImGui::GetColorU32(ImGuiCol_Text),
                  title.c_str());
    const ImVec2 hintSize = ImGui::CalcTextSize(hint.c_str());
    draw->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), true);
    draw->AddText(ImVec2(centerX - hintSize.x * 0.5f, centerY + 14.0f * dpi), dim, hint.c_str());
    draw->PopClipRect();
}

void InxGUIContext::DrawGradientRect(float x0, float y0, float x1, float y1, const std::array<float, 4> &left,
                                     const std::array<float, 4> &right)
{
    const auto l = ImGui::ColorConvertFloat4ToU32(ImVec4(left[0], left[1], left[2], left[3]));
    const auto r = ImGui::ColorConvertFloat4ToU32(ImVec4(right[0], right[1], right[2], right[3]));
    ImGui::GetWindowDrawList()->AddRectFilledMultiColor(ImVec2(x0, y0), ImVec2(x1, y1), l, r, r, l);
}

void InxGUIContext::DrawTriangleFilled(const std::array<float, 2> &a, const std::array<float, 2> &b,
                                       const std::array<float, 2> &c, const std::array<float, 4> &color)
{
    ImGui::GetWindowDrawList()->AddTriangleFilled(
        ImVec2(a[0], a[1]), ImVec2(b[0], b[1]), ImVec2(c[0], c[1]),
        ImGui::ColorConvertFloat4ToU32(ImVec4(color[0], color[1], color[2], color[3])));
}

void InxGUIContext::DrawPolyline(const std::vector<std::array<float, 2>> &points, const std::array<float, 4> &color,
                                 float thickness, bool closed)
{
    std::vector<ImVec2> vertices;
    vertices.reserve(points.size());
    for (const auto &point : points)
        vertices.emplace_back(point[0], point[1]);
    ImGui::GetWindowDrawList()->AddPolyline(
        vertices.data(), static_cast<int>(vertices.size()),
        ImGui::ColorConvertFloat4ToU32(ImVec4(color[0], color[1], color[2], color[3])),
        closed ? ImDrawFlags_Closed : ImDrawFlags_None, thickness);
}

void InxGUIContext::PushUIFont(float fontSize)
{
    ImGui::PushFont(ImGui::GetIO().FontDefault, fontSize * GetDpiScale());
}

void InxGUIContext::PushMonoFont()
{
    if (s_monoFont == nullptr)
        throw std::logic_error(
            "InxGUIContext::PushMonoFont(): mono font is not loaded; set_gui_fonts() must succeed first");
    // Size 0 keeps the current size, so window font scales still apply.
    ImGui::PushFont(s_monoFont, 0.0f);
}

void InxGUIContext::PopFont()
{
    ImGui::PopFont();
}

uint64_t InxGUIContext::EditorIcon(const std::string &name) const
{
    return resolveEditorIcon(name);
}

bool InxGUIContext::IconButton(const std::string &id, const std::string &icon, const std::string &tooltip, float width,
                               float height)
{
    ImGui::PushID(id.c_str());
    const bool clicked = ImGui::Button("##icon", ImVec2(width, height));
    const ImVec2 lo = ImGui::GetItemRectMin();
    const ImVec2 hi = ImGui::GetItemRectMax();
    const float size = (std::min)(16.0f * GetDpiScale(), (std::min)(hi.x - lo.x, hi.y - lo.y) - 6.0f * GetDpiScale());
    const auto texture = EditorIcon(icon);
    if (texture && size > 0.0f) {
        const ImVec2 a((lo.x + hi.x - size) * 0.5f, (lo.y + hi.y - size) * 0.5f);
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)texture, a, ImVec2(a.x + size, a.y + size),
                                             ImVec2(0, 0), ImVec2(1, 1), ImGui::GetColorU32(ImGuiCol_Text));
    }
    if (!tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", tooltip.c_str());
    ImGui::PopID();
    return clicked;
}

bool InxGUIContext::CanRenderWidgets() const
{
    ImGuiContext *context = ImGui::GetCurrentContext();
    if (context == nullptr || !context->WithinFrameScope || context->CurrentWindowStack.Size <= 0)
        return false;
    ImGuiWindow *stackWindow = context->CurrentWindowStack.back().Window;
    return stackWindow != nullptr && context->CurrentWindow == stackWindow;
}

void InxGUIContext::BeginFrameInteractionState()
{
    m_popupOwnedPointerAtFrameStart = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
}

bool InxGUIContext::IsPointerActivationBlockedByPopup() const
{
    return m_popupOwnedPointerAtFrameStart || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
}

namespace
{
ImTextureID ToImTextureID(uint64_t textureId)
{
    if constexpr (std::is_pointer_v<ImTextureID>) {
        return (ImTextureID)(static_cast<uintptr_t>(textureId));
    }
    return static_cast<ImTextureID>(textureId);
}

ImGuiPopupFlags ContextPopupFlagsForMouseButton(int mouseButton)
{
    switch (mouseButton) {
    case 0:
        return ImGuiPopupFlags_MouseButtonLeft;
    case 1:
        return ImGuiPopupFlags_MouseButtonRight;
    case 2:
        return ImGuiPopupFlags_MouseButtonMiddle;
    default:
        return ImGuiPopupFlags_MouseButtonRight;
    }
}

// Numeric values render in the `mono` face. A widget whose label is visible
// draws that label itself, so only hidden-label widgets switch faces; labels
// drawn beside the value (field label column) stay in `ui`.
class MonoValueFontScope
{
  public:
    explicit MonoValueFontScope(const char *label = "") : m_active(ImGui::FindRenderedTextEnd(label) == label)
    {
        if (!m_active)
            return;
        if (InxGUIContext::s_monoFont == nullptr)
            throw std::logic_error("Numeric field rendered before the mono GUI font was installed");
        ImGui::PushFont(InxGUIContext::s_monoFont, 0.0f);
    }
    ~MonoValueFontScope()
    {
        if (m_active)
            ImGui::PopFont();
    }
    MonoValueFontScope(const MonoValueFontScope &) = delete;
    MonoValueFontScope &operator=(const MonoValueFontScope &) = delete;

  private:
    bool m_active;
};

std::string WindowSemanticId(const std::string &name)
{
    const size_t separator = name.rfind("###");
    if (separator == std::string::npos || separator + 3 >= name.size())
        return name;
    return name.substr(separator + 3);
}

bool DrawInspectorSliderScalar(const char *id, ImGuiDataType dataType, void *data, const void *minimum,
                               const void *maximum, float dpi)
{
    ImGuiWindow *window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return false;

    ImGuiContext &g = *GImGui;
    const ImGuiID widgetId = window->GetID(id);
    const float width = ImGui::CalcItemWidth();
    const float height = ImGui::GetFrameHeight();
    const ImVec2 cursor = window->DC.CursorPos;
    const ImRect frame(cursor, ImVec2(cursor.x + width, cursor.y + height));

    ImGui::ItemSize(frame, 0.0f);
    if (!ImGui::ItemAdd(frame, widgetId, &frame, ImGuiItemFlags_Inputable))
        return false;

    const char *format = ImGui::DataTypeGetInfo(dataType)->PrintFmt;
    const bool hovered = ImGui::ItemHoverable(frame, widgetId, g.LastItemData.ItemFlags);
    const bool clicked = hovered && ImGui::IsMouseClicked(0, ImGuiInputFlags_None, widgetId);
    const bool navActivated = g.NavActivateId == widgetId;

    if (clicked) {
        ImGui::SetKeyOwner(ImGuiKey_MouseLeft, widgetId);
        std::memcpy(&g.ActiveIdValueOnActivation, data, ImGui::DataTypeGetInfo(dataType)->Size);
    }
    if (clicked || navActivated) {
        ImGui::SetActiveID(widgetId, window);
        ImGui::SetFocusID(widgetId, window);
        ImGui::FocusWindow(window);
        g.ActiveIdUsingNavDirMask |= (1 << ImGuiDir_Left) | (1 << ImGuiDir_Right);
    }

    ImRect grab;
    const bool changed =
        ImGui::SliderBehavior(frame, widgetId, dataType, data, minimum, maximum, format, ImGuiSliderFlags_None, &grab);
    if (changed)
        ImGui::MarkItemEdited(widgetId);

    // 3 px hairline track, accent fill up to the grab, warm-white 8x14 grab.
    ImDrawList *drawList = window->DrawList;
    const bool active = g.ActiveId == widgetId;
    const float centerY = std::floor((frame.Min.y + frame.Max.y) * 0.5f);
    const float halfTrack = EditorTheme::INSPECTOR_SLIDER_TRACK_HEIGHT * dpi * 0.5f;
    const float trackLeft = frame.Min.x + EditorTheme::INSPECTOR_SLIDER_LINE_PAD * dpi;
    const float trackRight = frame.Max.x - EditorTheme::INSPECTOR_SLIDER_LINE_PAD * dpi;
    const ImVec4 border = ImGui::GetStyleColorVec4(ImGuiCol_Border);
    const ImVec4 text = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    const float trackLift = hovered || active ? 0.20f : 0.10f;
    const ImVec4 trackColor(border.x + (text.x - border.x) * trackLift, border.y + (text.y - border.y) * trackLift,
                            border.z + (text.z - border.z) * trackLift, 1.0f);
    drawList->AddRectFilled(ImVec2(trackLeft, centerY - halfTrack), ImVec2(trackRight, centerY + halfTrack),
                            ImGui::GetColorU32(trackColor));

    if (grab.Max.x > grab.Min.x) {
        const float grabX = std::clamp(std::floor((grab.Min.x + grab.Max.x) * 0.5f), trackLeft, trackRight);
        drawList->AddRectFilled(ImVec2(trackLeft, centerY - halfTrack), ImVec2(grabX, centerY + halfTrack),
                                ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_ACCENT")));
        const float halfGrabW = EditorTheme::INSPECTOR_SLIDER_GRAB_W * dpi * 0.5f;
        const float halfGrabH = std::min(EditorTheme::INSPECTOR_SLIDER_GRAB_H * dpi, frame.GetHeight()) * 0.5f;
        drawList->AddRectFilled(ImVec2(grabX - halfGrabW, centerY - halfGrabH),
                                ImVec2(grabX + halfGrabW, centerY + halfGrabH),
                                ImGui::GetColorU32(active ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab));
    }

    if (g.NavId == widgetId)
        ImGui::RenderNavCursor(frame, widgetId);

    return changed;
}

bool DrawUnityRangedFloat(const char *baseId, float *value, float min, float max, const char *format, float dpi)
{
    const ImGuiStyle &style = ImGui::GetStyle();
    const float spacing = style.ItemInnerSpacing.x;
    const float totalW = ImGui::CalcItemWidth();
    const float inputW = EditorTheme::INSPECTOR_RANGED_INPUT_WIDTH * dpi;
    const float sliderW = std::max(24.0f * dpi, totalW - inputW - spacing);

    ImGui::BeginGroup();
    ImGui::PushID(baseId);
    ImGui::SetNextItemWidth(sliderW);
    bool changed = DrawInspectorSliderScalar("##slider", ImGuiDataType_Float, value, &min, &max, dpi);
    ImGui::SameLine(0.0f, spacing);
    ImGui::SetNextItemWidth(inputW);
    if (MonoValueFontScope mono;
        ImGui::InputFloat("##input", value, 0.0f, 0.0f, format ? format : "%.3f", ImGuiInputTextFlags_CharsDecimal))
        changed = true;
    *value = std::clamp(*value, min, max);
    ImGui::PopID();
    ImGui::EndGroup();
    return changed;
}

template <typename Integer>
bool DrawUnityRangedInt(const char *baseId, Integer *value, Integer min, Integer max, float dpi)
{
    const ImGuiStyle &style = ImGui::GetStyle();
    const float spacing = style.ItemInnerSpacing.x;
    const float totalW = ImGui::CalcItemWidth();
    const float inputW = EditorTheme::INSPECTOR_RANGED_INPUT_WIDTH * dpi;
    const float sliderW = std::max(24.0f * dpi, totalW - inputW - spacing);

    ImGui::BeginGroup();
    ImGui::PushID(baseId);
    ImGui::SetNextItemWidth(sliderW);
    constexpr auto dataType = sizeof(Integer) == 8 ? ImGuiDataType_S64 : ImGuiDataType_S32;
    bool changed = DrawInspectorSliderScalar("##slider", dataType, value, &min, &max, dpi);
    ImGui::SameLine(0.0f, spacing);
    ImGui::SetNextItemWidth(inputW);
    if (MonoValueFontScope mono; ImGui::InputScalar("##input", dataType, value))
        changed = true;
    *value = std::clamp(*value, min, max);
    ImGui::PopID();
    ImGui::EndGroup();
    return changed;
}

} // namespace

/* basic text & labels */
void InxGUIContext::Label(const std::string &text)
{
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(text.c_str());
}

void InxGUIContext::TextWrapped(const std::string &text)
{
    ImGui::TextWrapped("%s", text.c_str());
}

/* buttons / clickables */
bool InxGUIContext::Button(const std::string &label, std::function<void()> onClick, float width, float height)
{
    bool clicked = ImGui::Button(label.c_str(), ImVec2(width, height));
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("button", label);
    if (clicked && onClick)
        onClick();
    return clicked;
}

bool InxGUIContext::RadioButton(const std::string &label, bool active)
{
    const bool clicked = ImGui::RadioButton(label.c_str(), active);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("radio_button", label);
    return clicked;
}

static bool DrawEditorSelectable(const std::string &label, bool selected, int flags, float width, float height,
                                 float dpi)
{
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float rowWidth = width > 0 ? width : ImGui::GetContentRegionAvail().x;
    const float rowHeight = height > 0
                                ? height
                                : std::max(ImGui::GetFontSize(), EditorThemeRegistry::Float("EDITOR_ROW_HEIGHT") * dpi -
                                                                     ImGui::GetStyle().ItemSpacing.y);
    ImGui::PushStyleColor(ImGuiCol_Header, EditorThemeRegistry::Color("ROW_SELECTED"));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, EditorThemeRegistry::Color("ROLE_BG_HOVER"));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, EditorThemeRegistry::Color("ROW_SELECTED"));
    // Preserve the caller's native id while separating its hit area from the
    // padded caption. This also keeps hidden-label custom rows unchanged.
    ImGui::PushOverrideID(ImGui::GetID(label.c_str()));
    const bool clicked = ImGui::Selectable("", selected, flags, ImVec2(rowWidth, rowHeight));
    ImGui::PopID();
    const ImVec2 minimum = ImGui::GetItemRectMin(), maximum = ImGui::GetItemRectMax();
    if (selected)
        ImGui::GetWindowDrawList()->AddRectFilled(minimum, ImVec2(minimum.x + 2 * dpi, maximum.y),
                                                  ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_ACCENT")));
    const ImVec2 captionMin(origin.x + 8 * dpi, origin.y);
    const ImVec2 captionMax(std::max(captionMin.x, origin.x + rowWidth - 8 * dpi), origin.y + rowHeight);
    const ImRect clip(captionMin, captionMax);
    ImGui::RenderTextClipped(captionMin, captionMax, label.c_str(), ImGui::FindRenderedTextEnd(label.c_str()), nullptr,
                             ImVec2(0, 0.5f), &clip);
    ImGui::PopStyleColor(3);
    return clicked;
}

bool InxGUIContext::Selectable(const std::string &label, bool selected, int flags, float width, float height)
{
    const bool clicked = DrawEditorSelectable(label, selected, flags, width, height, GetDpiScale());
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("selectable", label);
    return clicked;
}

bool InxGUIContext::SelectableRow(const std::string &label, bool selected, const std::string &status,
                                  const std::array<float, 4> &statusColor, float height, float paddingX)
{
    if (ImGui::GetCurrentWindow()->SkipItems)
        return false;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
    const float rowHeight = std::max(height, ImGui::GetTextLineHeight());
    const float inset = std::clamp(paddingX, 0.0f, width * 0.5f);
    // Selectable owns the full-row input/background. Draw both captions inside
    // that row using the same font, inset and vertical alignment.
    const size_t idStart = label.find("###");
    const std::string widgetId = idStart == std::string::npos ? "##" + label : label.substr(idStart);
    const bool clicked = DrawEditorSelectable(widgetId, selected, 0, width, rowHeight, GetDpiScale());
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("selectable", label);

    const ImVec2 statusSize = ImGui::CalcTextSize(status.c_str());
    const float right = origin.x + width - inset;
    const float statusLeft = std::max(origin.x + inset, right - statusSize.x);
    const ImRect nameClip(ImVec2(origin.x + inset, origin.y),
                          ImVec2(std::max(origin.x + inset, statusLeft - inset), origin.y + rowHeight));
    ImGui::RenderTextClipped(nameClip.Min, nameClip.Max, label.c_str(), ImGui::FindRenderedTextEnd(label.c_str()),
                             nullptr, ImVec2(0.0f, 0.5f), &nameClip);
    const ImRect statusClip(ImVec2(statusLeft, origin.y), ImVec2(right, origin.y + rowHeight));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(statusColor[0], statusColor[1], statusColor[2], statusColor[3]));
    ImGui::RenderTextClipped(statusClip.Min, statusClip.Max, status.c_str(), nullptr, &statusSize, ImVec2(1.0f, 0.5f),
                             &statusClip);
    ImGui::PopStyleColor();
    return clicked;
}

int InxGUIContext::SelectableListClipped(size_t itemCount, const std::function<std::string(size_t)> &labelAt)
{
    if (!labelAt || itemCount == 0)
        return -1;
    if (itemCount > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("SelectableListClipped item count exceeds ImGuiListClipper range");

    int selectedIndex = -1;
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(itemCount));
    while (clipper.Step()) {
        for (int index = clipper.DisplayStart; index < clipper.DisplayEnd; ++index) {
            ImGui::PushID(index);
            const std::string label = labelAt(static_cast<size_t>(index));
            const bool clicked = ImGui::Selectable(label.c_str(), false);
            if (InxGUISemantics::IsCaptureEnabled())
                RecordSemanticItem("selectable", label);
            ImGui::PopID();
            if (clicked && selectedIndex < 0)
                selectedIndex = index;
        }
    }
    return selectedIndex;
}

/* value editors */
bool InxGUIContext::Checkbox(const std::string &label, bool *value)
{
    // Unified compact checkbox (75% square, ambient-size label) everywhere,
    // matching CheckboxInspector. Semantics are recorded here.
    const bool changed = CheckboxInspector(label, value);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("checkbox", label, true, "", *value);
    return changed;
}

namespace
{

bool DrawInspectorCheckboxSquare(const std::string &label, bool *value, std::string *outVisible, float dpi,
                                 bool framedRow = false)
{
    // Split "Visible##id" so only the square is drawn — label text stays at
    // the ambient font size.
    std::string visible;
    std::string id;
    const size_t hashPos = label.find("##");
    if (hashPos != std::string::npos) {
        visible = label.substr(0, hashPos);
        id = label.substr(hashPos);
    } else {
        visible = label;
        id = "##inx_cb_" + label;
    }
    if (id.empty() || id == "##")
        id = "##inx_cb";

    ImGuiWindow *window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return false;

    // One fixed square size everywhere. The square and its optional label are
    // submitted as one item so ImGui advances the row exactly once.
    const float boxSize = EditorThemeRegistry::Float("INSPECTOR_CHECKBOX_BOX_PX") * dpi;
    const float textHeight = ImGui::GetTextLineHeight();
    const float rowHeight = (std::max)(boxSize, framedRow ? ImGui::GetFrameHeight() : textHeight);
    const float labelWidth = visible.empty() ? 0.0f : ImGui::CalcTextSize(visible.c_str()).x;
    const float labelSpacing = visible.empty() ? 0.0f : ImGui::GetStyle().ItemInnerSpacing.x;
    const float itemWidth = boxSize + labelSpacing + labelWidth;

    const bool clicked = ImGui::InvisibleButton(id.c_str(), ImVec2(itemWidth, rowHeight));
    const bool hovered = ImGui::IsItemHovered();

    const ImVec2 itemMin = ImGui::GetItemRectMin();
    const float boxOffsetY = (rowHeight - boxSize) * 0.5f;
    const ImVec2 boxMin(itemMin.x, itemMin.y + boxOffsetY);
    const ImVec2 boxMax(itemMin.x + boxSize, itemMin.y + boxOffsetY + boxSize);
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    const bool mixed = (GImGui->LastItemData.ItemFlags & ImGuiItemFlags_MixedValue) != 0;
    const ImU32 accent = ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_ACCENT"));
    const ImU32 bgColor =
        *value && !mixed ? accent : ImGui::GetColorU32(hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg);
    const ImU32 borderColor = hovered || mixed || *value ? accent : ImGui::GetColorU32(ImGuiCol_TextDisabled);
    drawList->AddRectFilled(boxMin, boxMax, bgColor, 0.0f);
    drawList->AddRect(boxMin, boxMax, borderColor, 0.0f, 0, 1.0f * dpi);

    if (mixed) {
        const float y = (boxMin.y + boxMax.y) * 0.5f;
        drawList->AddLine(ImVec2(boxMin.x + 3.0f * dpi, y), ImVec2(boxMax.x - 3.0f * dpi, y), accent, 2.0f * dpi);
    } else if (*value) {
        const ImU32 checkColor = ImGui::GetColorU32(ImGuiCol_Text);
        const float s = boxSize;
        const ImVec2 check[3] = {
            {boxMin.x + s * 0.18f, boxMin.y + s * 0.52f},
            {boxMin.x + s * 0.42f, boxMin.y + s * 0.76f},
            {boxMin.x + s * 0.84f, boxMin.y + s * 0.26f},
        };
        drawList->AddPolyline(check, 3, checkColor, ImDrawFlags_None, 2.0f * dpi);
    }

    if (clicked)
        *value = !*value;

    if (!visible.empty()) {
        const ImVec2 textPos(boxMax.x + labelSpacing, itemMin.y + (rowHeight - textHeight) * 0.5f);
        drawList->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), visible.c_str());
    }

    if (outVisible)
        *outVisible = std::move(visible);
    return clicked;
}

} // namespace

bool InxGUIContext::CheckboxInspector(const std::string &label, bool *value)
{
    // Semantics are recorded by callers (they own the semantic id).
    return DrawInspectorCheckboxSquare(label, value, nullptr, GetDpiScale());
}

bool InxGUIContext::CheckboxField(const std::string &label, bool *value)
{
    // Inspector values share the input frame's height. Menus and component
    // header overlays keep their compact checkbox footprint.
    const bool changed = DrawInspectorCheckboxSquare(label, value, nullptr, GetDpiScale(), true);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("checkbox", label, true, "", *value);
    return changed;
}

void InxGUIContext::IntSlider(const std::string &label, int *value, int min, int max)
{
    DrawUnityRangedInt(label.c_str(), value, min, max, GetDpiScale());
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("int_slider", label, true, "", std::nullopt, static_cast<double>(*value));
}

void InxGUIContext::FloatSlider(const std::string &label, float *value, float min, float max, const char *format)
{
    DrawUnityRangedFloat(label.c_str(), value, min, max, format, GetDpiScale());
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("float_slider", label, true, "", std::nullopt, static_cast<double>(*value));
}

bool InxGUIContext::DragFloat(const std::string &label, float *value, float speed, float min, float max,
                              const char *fmt, float power, const std::string &semanticId)
{
    CompensateWarp();
    (void)power;
    const ImGuiSliderFlags flags = min < max ? ImGuiSliderFlags_AlwaysClamp : ImGuiSliderFlags_None;
    bool changed = false;
    {
        MonoValueFontScope mono(label.c_str());
        changed = ImGui::DragFloat(label.c_str(), value, speed, min, max, fmt, flags);
    }
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("drag_float", label, true, semanticId, std::nullopt, static_cast<double>(*value));
    HandleDragCapture();
    return changed;
}

bool InxGUIContext::DragInt(const std::string &label, int *value, float speed, int min, int max, const char *fmt)
{
    CompensateWarp();
    const ImGuiSliderFlags flags = min < max ? ImGuiSliderFlags_AlwaysClamp : ImGuiSliderFlags_None;
    bool changed = false;
    {
        MonoValueFontScope mono(label.c_str());
        changed = ImGui::DragInt(label.c_str(), value, speed, min, max, fmt, flags);
    }
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("drag_int", label, true, "", std::nullopt, static_cast<double>(*value));
    HandleDragCapture();
    return changed;
}

void InxGUIContext::TextInput(const std::string &label, char *buffer, size_t bufferSize)
{
    ImGui::InputText(label.c_str(), buffer, bufferSize);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("text_input", label, true, "", std::nullopt, std::nullopt, std::string(buffer));
}

void InxGUIContext::TextArea(const std::string &label, char *buffer, size_t bufferSize)
{
    ImGui::InputTextMultiline(label.c_str(), buffer, bufferSize, ImVec2(-FLT_MIN, 100.0f * GetDpiScale()));
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("text_area", label, true, "", std::nullopt, std::nullopt, std::string(buffer));
}

bool InxGUIContext::TextInput(const std::string &label, std::string &value)
{
    const bool changed = ImGui::InputText(label.c_str(), &value);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("text_input", label, true, "", std::nullopt, std::nullopt, value);
    return changed;
}

bool InxGUIContext::TextArea(const std::string &label, std::string &value)
{
    const bool changed = ImGui::InputTextMultiline(label.c_str(), &value, ImVec2(-FLT_MIN, 100.0f * GetDpiScale()));
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("text_area", label, true, "", std::nullopt, std::nullopt, value);
    return changed;
}

bool InxGUIContext::InputTextWithHint(const std::string &label, const std::string &hint, std::string &value, int flags)
{
    const bool changed = ImGui::InputTextWithHint(label.c_str(), hint.c_str(), &value, flags);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("text_input", label.empty() || label.rfind("##", 0) == 0 ? hint : label, true, label,
                           std::nullopt, std::nullopt, value);
    return changed;
}

bool InxGUIContext::InputTextWithHint(const std::string &label, const std::string &hint, char *buffer,
                                      size_t bufferSize, int flags)
{
    const bool changed = ImGui::InputTextWithHint(label.c_str(), hint.c_str(), buffer, bufferSize, flags);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("text_input", label.empty() || label.rfind("##", 0) == 0 ? hint : label, true, label,
                           std::nullopt, std::nullopt, std::string(buffer));
    return changed;
}

bool InxGUIContext::InputInt(const std::string &label, int *value, int step, int stepFast, int flags,
                             const std::string &semanticId)
{
    bool changed = false;
    {
        MonoValueFontScope mono(label.c_str());
        changed = ImGui::InputInt(label.c_str(), value, step, stepFast, flags);
    }
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("int_input", label, true, semanticId, std::nullopt, static_cast<double>(*value));
    return changed;
}

bool InxGUIContext::InputUInt(const std::string &label, uint32_t *value, uint32_t step, uint32_t stepFast, int flags,
                              const std::string &semanticId)
{
    bool changed = false;
    {
        MonoValueFontScope mono(label.c_str());
        changed = ImGui::InputScalar(label.c_str(), ImGuiDataType_U32, value, &step, &stepFast, "%u", flags);
    }
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("uint_input", label, true, semanticId, std::nullopt, static_cast<double>(*value));
    return changed;
}

bool InxGUIContext::InputFloat(const std::string &label, float *value, float step, float stepFast, int flags,
                               const std::string &semanticId)
{
    bool changed = false;
    {
        MonoValueFontScope mono(label.c_str());
        changed =
            ImGui::InputFloat(label.c_str(), value, step, stepFast, "%.3f", static_cast<ImGuiInputTextFlags>(flags));
    }
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("float_input", label, true, semanticId, std::nullopt, static_cast<double>(*value));
    return changed;
}
void InxGUIContext::ColorEdit(const std::string &label, float color[4], bool hdr)
{
    const ImGuiColorEditFlags flags = hdr ? ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float |
                                                ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB
                                          : ImGuiColorEditFlags_None;
    ImGui::ColorEdit4(label.c_str(), color, flags);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("color_edit", label);
}

bool InxGUIContext::ColorSwatch(const std::string &id, const std::array<float, 4> &color, float width, float height,
                                bool hdr)
{
    const float dpi = GetDpiScale();
    const float w = width > 0 ? width : ImGui::GetContentRegionAvail().x;
    const float h = height > 0 ? height : ImGui::GetFrameHeight();
    const bool clicked = ImGui::InvisibleButton(id.c_str(), ImVec2(w, h));
    const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    auto *draw = ImGui::GetWindowDrawList();
    const float peak = std::max(1.0f, std::max(color[0], std::max(color[1], color[2])));
    const ImVec4 base(color[0] / peak, color[1] / peak, color[2] / peak, std::clamp(color[3], 0.0f, 1.0f));
    const float hexWidth = w >= 190 * dpi ? 96 * dpi : 0;
    const float swatchRight = hi.x - hexWidth;
    const float split = lo.x + (swatchRight - lo.x) * 0.62f;
    draw->AddRectFilled(lo, ImVec2(split, hi.y), ImGui::GetColorU32(ImVec4(base.x, base.y, base.z, 1)));
    ImGui::RenderColorRectWithAlphaCheckerboard(draw, ImVec2(split, lo.y), ImVec2(swatchRight, hi.y),
                                                ImGui::GetColorU32(base), 5 * dpi, ImVec2(0, 0), 0);
    PushMonoFont();
    const float size = EditorThemeRegistry::Float("EDITOR_SMALL_FONT_SIZE") * dpi;
    const float textY = (lo.y + hi.y - size) * 0.5f;
    if (hexWidth > 0) {
        draw->AddRectFilled(ImVec2(swatchRight, lo.y), hi, ImGui::GetColorU32(ImGuiCol_FrameBg));
        char text[12];
        snprintf(text, sizeof(text), "#%02X%02X%02X%02X",
                 static_cast<unsigned>(std::round(std::clamp(base.x, 0.0f, 1.0f) * 255)),
                 static_cast<unsigned>(std::round(std::clamp(base.y, 0.0f, 1.0f) * 255)),
                 static_cast<unsigned>(std::round(std::clamp(base.z, 0.0f, 1.0f) * 255)),
                 static_cast<unsigned>(std::round(base.w * 255)));
        const float textWidth = ImGui::GetFont()->CalcTextSizeA(size, FLT_MAX, 0, text).x;
        draw->AddText(ImGui::GetFont(), size, ImVec2(swatchRight + (hexWidth - textWidth) * 0.5f, textY),
                      ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_TEXT_DIM")), text);
    }
    if (hdr && peak > 1.0f && split - lo.x > 72 * dpi) {
        char label[32];
        snprintf(label, sizeof(label), "HDR ×%.2g", static_cast<double>(peak));
        const float luminance = base.x * 0.2126f + base.y * 0.7152f + base.z * 0.0722f;
        draw->AddText(ImGui::GetFont(), size, ImVec2(lo.x + 6 * dpi, textY),
                      ImGui::GetColorU32(EditorThemeRegistry::Color(luminance > 0.45f ? "ROLE_BG_BASE" : "ROLE_TEXT")),
                      label);
    }
    PopFont();
    const bool focused = ImGui::IsItemActive() || ImGui::IsItemFocused();
    const ImVec4 accent = EditorThemeRegistry::Color("ROLE_ACCENT");
    draw->AddRect(lo, hi, ImGui::GetColorU32(focused ? accent : EditorThemeRegistry::Color("ROLE_BORDER")));
    if (focused)
        draw->AddRect(ImVec2(lo.x - dpi, lo.y - dpi), ImVec2(hi.x + dpi, hi.y + dpi),
                      ImGui::GetColorU32(ImVec4(accent.x, accent.y, accent.z, 0.35f)));
    return clicked;
}

bool InxGUIContext::ColorPicker(const std::string &label, float color[4], int flags)
{
    const bool changed = ImGui::ColorPicker4(label.c_str(), color, static_cast<ImGuiColorEditFlags>(flags));
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("color_picker", label);
    return changed;
}

// Unity-style helper: label on the left, DragFloatN on the right.
static uint32_t CaptureEditLifecycle(bool changed)
{
    uint32_t flags = changed ? InxGUIContext::EditChanged : 0u;
    if (ImGui::IsItemActive())
        flags |= InxGUIContext::EditActive;
    if (ImGui::IsItemActivated())
        flags |= InxGUIContext::EditActivated;
    if (ImGui::IsItemDeactivatedAfterEdit())
        flags |= InxGUIContext::EditDeactivatedAfterEdit;
    if (ImGui::IsItemDeactivated())
        flags |= InxGUIContext::EditDeactivated;
    return flags;
}

static uint32_t LabeledDragFloatN(InxGUIContext &ctx, const char *label, float *value, int components, float speed,
                                  float labelWidth = 0.0f, const std::string &axisSemanticBase = "")
{
    const bool visibleLabel = ImGui::FindRenderedTextEnd(label) != label;
    if (visibleLabel) {
        ctx.InspectorFieldLabel(label, labelWidth);
    }
    float avail = ImGui::GetContentRegionAvail().x;
    ImGui::SetNextItemWidth(avail);
    std::string hiddenLabel = visibleLabel ? std::string("##") + label : std::string(label);
    MonoValueFontScope mono;

    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    uint32_t lifecycle = 0;
    if (captureSemantics && components >= 2 && components <= 4 && !axisSemanticBase.empty()) {
        // Match ImGui::DragFloatN's layout so each actual field has a stable,
        // focusable semantic target instead of exposing only the aggregate row.
        static constexpr const char *kAxisNames[] = {"x", "y", "z", "w"};
        static constexpr const char *kAxisLabels[] = {"X", "Y", "Z", "W"};
        ImGui::BeginGroup();
        ImGui::PushID(hiddenLabel.c_str());
        ImGui::PushMultiItemsWidths(components, ImGui::CalcItemWidth());
        for (int axis = 0; axis < components; ++axis) {
            ImGui::PushID(axis);
            if (axis > 0)
                ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            // Keep the editor ID below the per-axis PushID scope. An empty
            // label aliases the current ID-stack seed and can collide when
            // DragFloat temporarily swaps to its text-input representation.
            const bool changed = ImGui::DragFloat("##axis_value", &value[axis], speed);
            lifecycle |= CaptureEditLifecycle(changed);
            ctx.RecordSemanticItem("vector_axis", kAxisLabels[axis], true, axisSemanticBase + "." + kAxisNames[axis],
                                   std::nullopt, static_cast<double>(value[axis]));
            ImGui::PopID();
            ImGui::PopItemWidth();
        }
        ImGui::PopID();
        ImGui::EndGroup();
    } else {
        bool changed = false;
        switch (components) {
        case 2:
            changed = ImGui::DragFloat2(hiddenLabel.c_str(), value, speed);
            break;
        case 3:
            changed = ImGui::DragFloat3(hiddenLabel.c_str(), value, speed);
            break;
        case 4:
            changed = ImGui::DragFloat4(hiddenLabel.c_str(), value, speed);
            break;
        default:
            changed = ImGui::DragFloat(hiddenLabel.c_str(), value, speed);
            break;
        }
        lifecycle = CaptureEditLifecycle(changed);
    }
    if (captureSemantics)
        ctx.RecordSemanticItem("vector", label);
    return lifecycle;
}

void InxGUIContext::Vector2Control(const std::string &label, float value[2], float speed, float labelWidth,
                                   const std::string &axisSemanticBase)
{
    CompensateWarp();
    m_lastEditLifecycleFlags = LabeledDragFloatN(*this, label.c_str(), value, 2, speed, labelWidth, axisSemanticBase);
    HandleDragCapture();
}

void InxGUIContext::Vector3Control(const std::string &label, float value[3], float speed, float labelWidth,
                                   const std::string &axisSemanticBase)
{
    CompensateWarp();
    m_lastEditLifecycleFlags = LabeledDragFloatN(*this, label.c_str(), value, 3, speed, labelWidth, axisSemanticBase);
    HandleDragCapture();
}

void InxGUIContext::Vector4Control(const std::string &label, float value[4], float speed, float labelWidth)
{
    CompensateWarp();
    m_lastEditLifecycleFlags = LabeledDragFloatN(*this, label.c_str(), value, 4, speed, labelWidth);
    HandleDragCapture();
}

/* combo & lists */
bool InxGUIContext::Combo(const std::string &label, int *currentItem, const std::vector<std::string> &items,
                          int popupMaxHeightInItems)
{
    if (currentItem == nullptr)
        return false;
    if (ImGui::GetCurrentWindow()->SkipItems)
        return false;

    ImGuiContext &context = *ImGui::GetCurrentContext();
    if (popupMaxHeightInItems >= 0 &&
        (context.NextWindowData.HasFlags & ImGuiNextWindowDataFlags_HasSizeConstraint) == 0) {
        // Rows inside the popup use the popup content spacing, not the panel's.
        const float rowSpacing = EditorThemeRegistry::Vec2("POPUP_ITEM_SPC").y * s_dpiScale;
        const float windowPadding = EditorThemeRegistry::Vec2("POPUP_WIN_PAD").y * s_dpiScale;
        const float popupMaxHeight = (context.FontSize + rowSpacing) * static_cast<float>(popupMaxHeightInItems) -
                                     rowSpacing + windowPadding * 2.0f;
        ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(FLT_MAX, std::max(0.0f, popupMaxHeight)));
    }

    const bool hasSelection = *currentItem >= 0 && *currentItem < static_cast<int>(items.size());
    const std::string preview = hasSelection ? items[*currentItem] : std::string{};
    const bool open = BeginCombo(label, preview);

    const size_t hiddenMarker = label.find("##");
    const std::string semanticBase =
        hiddenMarker != std::string::npos && hiddenMarker + 2 < label.size() ? label.substr(hiddenMarker + 2) : label;
    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    if (!open) {
        if (captureSemantics)
            RecordSemanticItem("combo", preview.empty() ? label : preview, true, semanticBase, std::nullopt,
                               std::nullopt, preview);
        return false;
    }

    if (captureSemantics)
        RecordSemanticWindow("combo_popup", preview.empty() ? label : preview, semanticBase);
    bool changed = false;
    for (int index = 0; index < static_cast<int>(items.size()); ++index) {
        ImGui::PushID(index);
        const bool selected = index == *currentItem;
        if (ImGui::Selectable(items[index].c_str(), selected)) {
            *currentItem = index;
            changed = true;
        }
        if (captureSemantics)
            RecordSemanticItem("combo_option", items[index], true, semanticBase + ":option:" + std::to_string(index));
        if (selected)
            ImGui::SetItemDefaultFocus();
        ImGui::PopID();
    }
    EndCombo();

    // EndCombo restores its parent's LastItemData. Capturing it after
    // BeginCombo would capture the popup window header when the menu is open.
    if (changed)
        ImGui::MarkItemEdited(context.LastItemData.ID);
    if (captureSemantics)
        RecordSemanticItem("combo", preview.empty() ? label : preview, true, semanticBase, std::nullopt, std::nullopt,
                           preview);
    return changed;
}

uint64_t InxGUIContext::FlagsCombo(const std::string &label, uint64_t value, const std::vector<std::string> &items,
                                   const std::vector<uint64_t> &masks, const std::string &preview)
{
    if (items.size() != masks.size())
        throw std::invalid_argument("FlagsCombo requires one mask for each label");
    if (ImGui::GetCurrentWindow()->SkipItems)
        return value;
    auto &context = *ImGui::GetCurrentContext();
    const uint64_t previous = value;
    const bool open = BeginCombo(label, preview);
    const auto marker = label.find("##");
    const std::string semanticBase = marker == std::string::npos ? label : label.substr(marker + 2);
    const bool capture = InxGUISemantics::IsCaptureEnabled();
    if (!open) {
        if (capture)
            RecordSemanticItem("combo", preview, true, semanticBase, std::nullopt, std::nullopt, std::to_string(value));
        return value;
    }
    if (capture)
        RecordSemanticWindow("combo_popup", preview, semanticBase);
    for (size_t index = 0; index < items.size(); ++index) {
        ImGui::PushID(static_cast<int>(index));
        const uint64_t mask = masks[index];
        const bool selected = mask == 0 ? value == 0 : (value & mask) == mask;
        if (ImGui::Selectable(items[index].c_str(), selected, ImGuiSelectableFlags_NoAutoClosePopups))
            value = mask == 0 ? 0 : selected ? value & ~mask : value | mask;
        if (capture)
            RecordSemanticItem("combo_option", items[index], true, semanticBase + ":option:" + std::to_string(index),
                               mask == 0 ? value == 0 : (value & mask) == mask);
        ImGui::PopID();
    }
    EndCombo();
    if (value != previous)
        ImGui::MarkItemEdited(context.LastItemData.ID);
    if (capture)
        RecordSemanticItem("combo", preview, true, semanticBase, std::nullopt, std::nullopt, std::to_string(value));
    return value;
}

bool InxGUIContext::ListBox(const std::string &label, int *currentItem, const std::vector<std::string> &items,
                            int heightInItems)
{
    std::vector<const char *> cstrs;
    cstrs.reserve(items.size());
    for (const auto &s : items)
        cstrs.push_back(s.c_str());
    const bool changed =
        ImGui::ListBox(label.c_str(), currentItem, cstrs.data(), static_cast<int>(cstrs.size()), heightInItems);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("list_box", label);
    return changed;
}

/* progress & indicators */
void InxGUIContext::ProgressBar(float fraction, float width, float height, const std::string &overlay)
{
    ImGui::ProgressBar(fraction, ImVec2(width, height), overlay.c_str());
}

/* layout helpers */
void InxGUIContext::BeginGroup(const std::string &name)
{
    ImGui::BeginGroup();
    if (!name.empty())
        ImGui::TextUnformatted(name.c_str());
}

void InxGUIContext::EndGroup()
{
    ImGui::EndGroup();
}

void InxGUIContext::FieldLabel(const std::string &text, float columnWidth)
{
    const float width = std::min(columnWidth, std::max(0.0f, ImGui::GetContentRegionAvail().x - 48.0f * GetDpiScale()));
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::AlignTextToFramePadding();
    ImGui::Dummy(ImVec2(width, ImGui::GetFrameHeight()));
    const ImRect clip(origin, ImVec2(origin.x + width, origin.y + ImGui::GetFrameHeight()));
    ImGui::RenderTextClipped(clip.Min, clip.Max, text.c_str(), nullptr, nullptr, ImVec2(0, 0.5f), &clip);
    if (ImGui::IsItemHovered() && ImGui::CalcTextSize(text.c_str()).x > width)
        ImGui::SetTooltip("%s", text.c_str());
    ImGui::SameLine(0, 0);
    ImGui::SetNextItemWidth(-FLT_MIN);
}

void InxGUIContext::BeginFramedGroup(float paddingX, float paddingY, float rounding,
                                     const std::array<float, 4> &background, const std::array<float, 4> &border)
{
    if (m_framedGroupDepth == m_framedGroups.size())
        m_framedGroups.emplace_back();
    auto &group = m_framedGroups[m_framedGroupDepth++];
    group.drawList = ImGui::GetWindowDrawList();
    group.paddingX = paddingX * GetDpiScale();
    group.paddingY = paddingY * GetDpiScale();
    group.origin = ImGui::GetCursorScreenPos();
    group.width = ImGui::GetContentRegionAvail().x;
    auto *window = ImGui::GetCurrentWindow();
    group.workMaxX = window->WorkRect.Max.x;
    group.contentMaxX = window->ContentRegionRect.Max.x;
    group.rounding = rounding;
    group.background = ImGui::GetColorU32(ImVec4(background[0], background[1], background[2], background[3]));
    group.border = ImGui::GetColorU32(ImVec4(border[0], border[1], border[2], border[3]));
    group.splitter.Split(group.drawList, 2);
    group.splitter.SetCurrentChannel(group.drawList, 1);
    ImGui::BeginGroup();
    if (group.paddingX != 0.0f)
        ImGui::Indent(group.paddingX);
    window->WorkRect.Max.x -= group.paddingX;
    window->ContentRegionRect.Max.x -= group.paddingX;
    ImGui::PushItemWidth(-FLT_MIN);
    ImGui::Dummy(ImVec2(0, group.paddingY));
}

void InxGUIContext::EndFramedGroup()
{
    auto &group = m_framedGroups[--m_framedGroupDepth];
    ImGui::Dummy(ImVec2(0, group.paddingY));
    ImGui::PopItemWidth();
    auto *window = ImGui::GetCurrentWindow();
    window->WorkRect.Max.x = group.workMaxX;
    window->ContentRegionRect.Max.x = group.contentMaxX;
    if (group.paddingX != 0.0f)
        ImGui::Unindent(group.paddingX);
    ImGui::EndGroup();
    const float bottom = ImGui::GetItemRectMax().y;
    const ImVec2 maximum(group.origin.x + group.width, bottom);
    group.splitter.SetCurrentChannel(group.drawList, 0);
    group.drawList->AddRectFilled(group.origin, maximum, group.background, group.rounding);
    group.drawList->AddRect(group.origin, maximum, group.border, group.rounding);
    group.splitter.Merge(group.drawList);
}

void InxGUIContext::SameLine(float offsetFromStartX, float spacing)
{
    ImGui::SameLine(offsetFromStartX, spacing);
}

void InxGUIContext::AlignTextToFramePadding()
{
    ImGui::AlignTextToFramePadding();
}

void InxGUIContext::SetScrollHereY(float centerYRatio)
{
    ImGui::SetScrollHereY(centerYRatio);
}

float InxGUIContext::GetScrollY()
{
    return ImGui::GetScrollY();
}

float InxGUIContext::GetScrollMaxY()
{
    return ImGui::GetScrollMaxY();
}

void InxGUIContext::SetScrollX(float scrollX)
{
    ImGui::SetScrollX(scrollX);
}

void InxGUIContext::SetScrollY(float scrollY)
{
    ImGui::SetScrollY(scrollY);
}

void InxGUIContext::CloseCurrentPopup()
{
    ImGui::CloseCurrentPopup();
}

void InxGUIContext::Separator()
{
    ImGui::Separator();
}

void InxGUIContext::Spacing()
{
    ImGui::Spacing();
}

void InxGUIContext::Dummy(float width, float height)
{
    ImGui::Dummy(ImVec2(width, height));
}

void InxGUIContext::NewLine()
{
    ImGui::NewLine();
}

/* tree & collapsing */
bool InxGUIContext::TreeNode(const std::string &label)
{
    const bool open = ImGui::TreeNode(label.c_str());
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("tree_node", label);
    return open;
}

bool InxGUIContext::TreeNodeEx(const std::string &label, int flags)
{
    const bool open = ImGui::TreeNodeEx(label.c_str(), static_cast<ImGuiTreeNodeFlags>(flags));
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("tree_node", label);
    return open;
}

void InxGUIContext::TreePop()
{
    ImGui::TreePop();
}

void InxGUIContext::SetNextItemOpen(bool is_open, int cond)
{
    ImGui::SetNextItemOpen(is_open, cond);
}

void InxGUIContext::SetNextItemAllowOverlap()
{
    ImGui::SetNextItemAllowOverlap();
}

bool InxGUIContext::CollapsingHeader(const std::string &label)
{
    const bool open = ImGui::CollapsingHeader(label.c_str());
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("collapsing_header", label);
    return open;
}

bool InxGUIContext::RenderCompactSectionHeader(
    const std::string &label, uint64_t iconId, bool defaultOpen, int openCondition, bool allowOverlap, float framePadX,
    float framePadY, float itemSpacingX, float itemSpacingY, float borderSize, bool zeroIndent, float fontScale,
    float rightMargin, float iconSize, const std::array<float, 4> &headerColor, const std::array<float, 4> &hoverColor,
    const std::array<float, 4> &activeColor, bool useTextColor, const std::array<float, 4> &textColor)
{
    if (defaultOpen)
        ImGui::SetNextItemOpen(true, openCondition);
    if (allowOverlap)
        ImGui::SetNextItemAllowOverlap();

    auto asColor = [](const std::array<float, 4> &color) { return ImVec4(color[0], color[1], color[2], color[3]); };
    ImGui::PushStyleColor(ImGuiCol_Header, asColor(headerColor));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, asColor(hoverColor));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, asColor(activeColor));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(framePadX, framePadY));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(itemSpacingX, itemSpacingY));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, borderSize);
    if (zeroIndent)
        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 0.0f);
    ImGui::SetWindowFontScale(fontScale);
    if (useTextColor)
        ImGui::PushStyleColor(ImGuiCol_Text, asColor(textColor));

    const float maxX =
        ImGui::GetWindowPos().x + ImGui::GetCursorPos().x + ImGui::GetContentRegionAvail().x - rightMargin;
    ImGui::GetWindowDrawList()->PushClipRect(ImVec2(0.0f, 0.0f), ImVec2(maxX, 1e7f), true);
    const bool open = iconId != 0 ? ImGui::TreeNodeEx(label.c_str(), ImGuiTreeNodeFlags_CollapsingHeader, "%s", "")
                                  : ImGui::CollapsingHeader(label.c_str());
    if (iconId != 0) {
        const ImVec2 lo = ImGui::GetItemRectMin();
        const ImVec2 hi = ImGui::GetItemRectMax();
        const float x = lo.x + ImGui::GetTreeNodeToLabelSpacing();
        const float y = lo.y + (hi.y - lo.y - iconSize) * 0.5f;
        ImGui::GetWindowDrawList()->AddImage(ImTextureRef(static_cast<ImTextureID>(iconId)), ImVec2(x, y),
                                             ImVec2(x + iconSize, y + iconSize));
        const char *end = ImGui::FindRenderedTextEnd(label.c_str());
        ImGui::RenderTextClipped(ImVec2(x + iconSize + 6.0f * GetDpiScale(), lo.y), hi, label.c_str(), end, nullptr,
                                 ImVec2(0.0f, 0.5f));
    }
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("collapsing_header", label);
    ImGui::GetWindowDrawList()->PopClipRect();

    if (useTextColor)
        ImGui::PopStyleColor();
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(zeroIndent ? 4 : 3);
    return open;
}

bool InxGUIContext::IsItemClicked(int mouseButton)
{
    return ImGui::IsItemClicked(static_cast<ImGuiMouseButton>(mouseButton));
}

/* tab bars */
bool InxGUIContext::BeginTabBar(const std::string &id)
{
    return ImGui::BeginTabBar(id.c_str());
}

void InxGUIContext::EndTabBar()
{
    ImGui::EndTabBar();
}

bool InxGUIContext::BeginTabItem(const std::string &label, bool *open, bool selected)
{
    const bool visible = ImGui::BeginTabItem(label.c_str(), open, selected ? ImGuiTabItemFlags_SetSelected : 0);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("tab", label, open == nullptr || *open);
    return visible;
}

void InxGUIContext::EndTabItem()
{
    ImGui::EndTabItem();
}

namespace
{
// ─── Unity-style popup / menu chrome ─────────────────────────────────────
// Every popup uses DPI-scaled padding and row spacing. One style stack entry
// per popup/menu window keeps nested Begin/End pairs balanced.
struct PopupContentStyle
{
    int styleVars = 0;
    int styleColors = 0;
};
std::vector<PopupContentStyle> s_popupContentStyleStack;

// Floating layers get a 1 px edge lifted above the panel hairline so a popup
// reads as raised over the surface it covers. Begin latches it for the frame.
void PushPopupEdgeColor()
{
    const ImVec4 border = ImGui::GetStyleColorVec4(ImGuiCol_Border);
    const ImVec4 text = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    constexpr float lift = 0.12f;
    ImGui::PushStyleColor(ImGuiCol_Border,
                          ImVec4(border.x + (text.x - border.x) * lift, border.y + (text.y - border.y) * lift,
                                 border.z + (text.z - border.z) * lift, 1.0f));
}

// Pushed *before* Begin so the popup window latches padding and edge colour.
// Pair with PopPopupWindowChrome right after Begin.
void PushPopupWindowChrome()
{
    const float dpi = InxGUIContext::s_dpiScale;
    const ImVec2 winPad = EditorThemeRegistry::Vec2("POPUP_WIN_PAD");
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(winPad.x * dpi, winPad.y * dpi));
    PushPopupEdgeColor();
}

void PopPopupWindowChrome()
{
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// Pushed when the popup is open; popped in the matching End call.
void PushPopupContentStyle()
{
    PopupContentStyle entry;
    const float dpi = InxGUIContext::s_dpiScale;
    const ImVec2 itemSpc = EditorThemeRegistry::Vec2("POPUP_ITEM_SPC");
    const ImVec2 framePad = EditorThemeRegistry::Vec2("POPUP_FRAME_PAD");
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(itemSpc.x * dpi, itemSpc.y * dpi));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(framePad.x * dpi, framePad.y * dpi));
    entry.styleVars = 2;

    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, EditorThemeRegistry::Color("ROLE_BG_HOVER"));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, EditorThemeRegistry::Color("SELECTION_BG"));
    entry.styleColors = 2;

    s_popupContentStyleStack.push_back(entry);
}

void PopPopupContentStyle()
{
    if (s_popupContentStyleStack.empty())
        return;
    const PopupContentStyle entry = s_popupContentStyleStack.back();
    s_popupContentStyleStack.pop_back();
    ImGui::PopStyleColor(entry.styleColors);
    ImGui::PopStyleVar(entry.styleVars);
}

// Context menus get a Unity-like minimum width so entries and shortcuts
// are not clipped into a narrow strip.
void ApplyContextMenuMinWidth()
{
    const float em = ImGui::GetFontSize();
    ImGui::SetNextWindowSizeConstraints(ImVec2(em * 14.0f, 0.0f), ImVec2(FLT_MAX, FLT_MAX));
}

// One dropdown look for every combo: a flat field frame, the preview clipped
// with an ellipsis, and a vector chevron (chev_d) in a square slot on the
// right. An open dropdown carries the focus ring. Drawn into the current
// (owning) window before any popup begins.
void DrawComboFrame(const ImRect &frame, const char *preview, bool hovered, bool open, float dpi)
{
    const ImGuiStyle &style = ImGui::GetStyle();
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(frame.Min, frame.Max,
                            ImGui::GetColorU32(hovered || open ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg));
    if (open) {
        const ImVec4 accent = EditorThemeRegistry::Color("ROLE_ACCENT");
        const float ring = 1.0f * dpi;
        drawList->AddRect(ImVec2(frame.Min.x - ring, frame.Min.y - ring),
                          ImVec2(frame.Max.x + ring, frame.Max.y + ring),
                          ImGui::GetColorU32(ImVec4(accent.x, accent.y, accent.z, 0.35f)), 0.0f, 0, ring);
        drawList->AddRect(frame.Min, frame.Max, ImGui::GetColorU32(accent), 0.0f, 0, ring);
    } else {
        ImGui::RenderFrameBorder(frame.Min, frame.Max, 0.0f);
    }

    const float slot = frame.GetHeight();
    const ImVec2 textMin(frame.Min.x + style.FramePadding.x, frame.Min.y + style.FramePadding.y);
    const ImVec2 textMax(frame.Max.x - slot, frame.Max.y);
    const char *previewEnd = ImGui::FindRenderedTextEnd(preview);
    if (previewEnd != preview && textMax.x > textMin.x)
        ImGui::RenderTextEllipsis(drawList, textMin, textMax, textMax.x, preview, previewEnd, nullptr);

    const float half = std::round(ImGui::GetFontSize() * 0.3f);
    const ImVec2 center(std::floor(frame.Max.x - slot * 0.5f) + 0.5f,
                        std::floor((frame.Min.y + frame.Max.y) * 0.5f) + 0.5f);
    const ImVec2 chevron[3] = {ImVec2(center.x - half, center.y - half * 0.5f),
                               ImVec2(center.x, center.y + half * 0.5f),
                               ImVec2(center.x + half, center.y - half * 0.5f)};
    const ImU32 chevronColor = hovered || open ? ImGui::GetColorU32(ImGuiCol_Text)
                                               : ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_TEXT_DIM"));
    drawList->AddPolyline(chevron, 3, chevronColor, ImDrawFlags_None, std::max(1.0f, 1.5f * dpi));
}

} // namespace

/* combo */
bool InxGUIContext::BeginCombo(const std::string &label, const std::string &preview, int flags)
{
    ImGuiContext &g = *GImGui;
    ImGuiWindow *window = ImGui::GetCurrentWindow();
    // Like ImGui::BeginCombo: consume the caller's next-window data now and
    // hand it to the popup only if it opens.
    const ImGuiNextWindowDataFlags nextWindowFlags = g.NextWindowData.HasFlags;
    g.NextWindowData.ClearFlags();
    if (window->SkipItems)
        return false;

    const ImGuiStyle &style = g.Style;
    const ImGuiID id = window->GetID(label.c_str());
    const float frameHeight = ImGui::GetFrameHeight();
    ImGuiComboFlags comboFlags = static_cast<ImGuiComboFlags>(flags);
    const float width = (comboFlags & ImGuiComboFlags_WidthFitPreview)
                            ? ImGui::CalcTextSize(preview.c_str(), nullptr, true).x + style.FramePadding.x + frameHeight
                            : ImGui::CalcItemWidth();
    comboFlags &= ~ImGuiComboFlags_WidthFitPreview;

    const ImVec2 labelSize = ImGui::CalcTextSize(label.c_str(), nullptr, true);
    const ImVec2 cursor = window->DC.CursorPos;
    const ImRect frame(cursor, ImVec2(cursor.x + width, cursor.y + frameHeight));
    const float labelExtra = labelSize.x > 0.0f ? style.ItemInnerSpacing.x + labelSize.x : 0.0f;
    const ImRect total(frame.Min, ImVec2(frame.Max.x + labelExtra, frame.Max.y));
    ImGui::ItemSize(total, style.FramePadding.y);
    if (!ImGui::ItemAdd(total, id, &frame))
        return false;

    bool hovered = false;
    bool held = false;
    const bool pressed = ImGui::ButtonBehavior(frame, id, &hovered, &held);
    const ImGuiID popupId = ImHashStr("##ComboPopup", 0, id);
    bool popupOpen = ImGui::IsPopupOpen(popupId, ImGuiPopupFlags_None);
    if (pressed && !popupOpen) {
        ImGui::OpenPopupEx(popupId, ImGuiPopupFlags_None);
        popupOpen = true;
    }

    ImGui::RenderNavCursor(frame, id);
    DrawComboFrame(frame, preview.c_str(), hovered, popupOpen, GetDpiScale());
    if (labelSize.x > 0.0f)
        ImGui::RenderText(ImVec2(frame.Max.x + style.ItemInnerSpacing.x, frame.Min.y + style.FramePadding.y),
                          label.c_str());

    if (!popupOpen)
        return false;

    g.NextWindowData.HasFlags = nextWindowFlags;
    PushPopupWindowChrome();
    const bool open = ImGui::BeginComboPopup(popupId, frame, comboFlags);
    PopPopupWindowChrome();
    if (open)
        PushPopupContentStyle();
    return open;
}

void InxGUIContext::EndCombo()
{
    PopPopupContentStyle();
    ImGui::EndCombo();
}

/* main menu / menus */
bool InxGUIContext::BeginMainMenuBar()
{
    return ImGui::BeginMainMenuBar();
}

void InxGUIContext::EndMainMenuBar()
{
    ImGui::EndMainMenuBar();
}

bool InxGUIContext::BeginMenu(const std::string &label, bool enabled, const std::string &semanticId)
{
    // Nested BeginMenu windows must NOT inherit the root context-menu
    // WindowPadding push. ImGui places submenus with a 1px overlap against
    // the parent, so an inflated WindowPadding makes 1st/2nd/3rd-level menu
    // edges visually collide. Content style (row spacing / hover colours)
    // is still applied once the submenu is open.
    PushPopupEdgeColor();
    const bool open = ImGui::BeginMenu(label.c_str(), enabled);
    ImGui::PopStyleColor();
    if (open)
        PushPopupContentStyle();
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("menu", label, enabled, semanticId, open);
    return open;
}

void InxGUIContext::EndMenu()
{
    PopPopupContentStyle();
    ImGui::EndMenu();
}

bool InxGUIContext::MenuItem(const std::string &label, const std::string &shortcut, bool selected, bool enabled)
{
    const bool clicked =
        EditorMenuItem(label.c_str(), shortcut.empty() ? nullptr : shortcut.c_str(), selected, enabled);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("menu_item", label, enabled);
    return clicked;
}

/* child & windows */
bool InxGUIContext::BeginChild(const std::string &id, float width, float height, bool border, int flags, bool padding)
{
    // Inside a popup, child scroll regions must not draw their own background
    // (the popup window already provides one). Keep the stack balanced via the
    // matching EndChild.
    const bool inPopup = !s_popupContentStyleStack.empty();
    if (inPopup) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ++m_childBgTransparentCount;
    }
    const ImGuiChildFlags childFlags =
        (border ? ImGuiChildFlags_Borders : 0) | (padding ? ImGuiChildFlags_AlwaysUseWindowPadding : 0);
    return ImGui::BeginChild(id.c_str(), ImVec2(width, height), childFlags, static_cast<ImGuiWindowFlags>(flags));
}

void InxGUIContext::EndChild()
{
    ImGui::EndChild();
    if (m_childBgTransparentCount > 0) {
        --m_childBgTransparentCount;
        ImGui::PopStyleColor();
    }
}

/* popups & tooltips */

void InxGUIContext::OpenPopup(const std::string &id)
{
    ImGui::OpenPopup(id.c_str());
}

bool InxGUIContext::BeginPopup(const std::string &id, bool resizable)
{
    // Panels that build context menus manually (OpenPopup + BeginPopup, e.g.
    // "##HierarchyItemContext") name them accordingly; give those the same
    // minimum width as the BeginPopupContext* variants.
    if (id.find("Context") != std::string::npos || id.find("context") != std::string::npos)
        ApplyContextMenuMinWidth();
    PushPopupWindowChrome();
    const bool open = resizable ? ImGui::BeginPopupEx(ImGui::GetID(id.c_str()),
                                                      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings)
                                : ImGui::BeginPopup(id.c_str());
    PopPopupWindowChrome();
    if (open) {
        PushPopupContentStyle();
        RecordSemanticWindow("popup", id, id);
    }
    return open;
}

bool InxGUIContext::BeginPopupModal(const std::string &title, int flags)
{
    // Detached editor panels are separate native child windows of the main
    // viewport. A modal merged into the main viewport therefore cannot cover
    // them, regardless of ImGui's internal display order. Give every modal its
    // own short-lived platform viewport parented to the main editor window.
    ImGuiWindowClass modalClass{};
    modalClass.ClassId = ImHashStr("Infernux.GlobalModal");
    modalClass.ParentViewportId = ImGui::GetMainViewport()->ID;
    modalClass.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge | ImGuiViewportFlags_NoTaskBarIcon |
                                          ImGuiViewportFlags_NoDecoration | ImGuiViewportFlags_TopMost;
    ImGui::SetNextWindowClass(&modalClass);

    PushPopupEdgeColor();
    const bool open = ImGui::BeginPopupModal(title.c_str(), nullptr, static_cast<ImGuiWindowFlags>(flags));
    ImGui::PopStyleColor();
    if (open) {
        // Modals are closed through the shared EndPopup wrapper, which pops
        // one content-style entry — keep the stack balanced.
        PushPopupContentStyle();
        ImGuiWindow *window = ImGui::GetCurrentWindow();
        // Docking preserves the display order of undocked windows even when
        // they are submitted before this popup. A modal must remain above
        // every floating editor window for its entire lifetime.
        ImGui::BringWindowToDisplayFront(window);
        ImGui::BringWindowToFocusFront(window->RootWindow);

        // Keep the modal as the sole ImGui focus owner for its whole lifetime.
        // A detached panel may otherwise regain focus from the title-bar close
        // interaction on the following frame and cover the confirmation.
        ImGui::FocusWindow(window);

        // A new platform viewport is created after the first ImGui frame. The
        // old Appearing-only check therefore ran too early and never raised the
        // native modal. Remember the request until the viewport actually exists.
        static const ImGuiID nativeRaiseState = ImHashStr("Infernux.GlobalModal.NativeRaiseComplete");
        if (window->Appearing)
            window->StateStorage.SetBool(nativeRaiseState, false);
        ImGuiViewport *viewport = window->Viewport;
        if (viewport != nullptr && viewport->PlatformWindowCreated &&
            !window->StateStorage.GetBool(nativeRaiseState, false)) {
            ImGuiPlatformIO &platformIO = ImGui::GetPlatformIO();
            if (platformIO.Platform_SetWindowFocus != nullptr)
                platformIO.Platform_SetWindowFocus(viewport);
            window->StateStorage.SetBool(nativeRaiseState, true);
        }
        RecordSemanticWindow("modal", title, title);
    }
    return open;
}

bool InxGUIContext::BeginPopupContextItem(const std::string &id, int mouseButton)
{
    ApplyContextMenuMinWidth();
    PushPopupWindowChrome();
    const bool open =
        ImGui::BeginPopupContextItem(id.empty() ? nullptr : id.c_str(), ContextPopupFlagsForMouseButton(mouseButton));
    PopPopupWindowChrome();
    if (open) {
        PushPopupContentStyle();
        RecordSemanticWindow("context_menu", id, id);
    }
    return open;
}

bool InxGUIContext::BeginPopupContextWindow(const std::string &id, int mouseButton, bool noOpenOverItems)
{
    ImGuiPopupFlags flags = ContextPopupFlagsForMouseButton(mouseButton);
    if (noOpenOverItems)
        flags |= ImGuiPopupFlags_NoOpenOverItems;
    ApplyContextMenuMinWidth();
    PushPopupWindowChrome();
    const bool open = ImGui::BeginPopupContextWindow(id.empty() ? nullptr : id.c_str(), flags);
    PopPopupWindowChrome();
    if (open) {
        PushPopupContentStyle();
        RecordSemanticWindow("context_menu", id, id);
    }
    return open;
}

void InxGUIContext::EndPopup()
{
    PopPopupContentStyle();
    ImGui::EndPopup();
}

void InxGUIContext::BeginTooltip()
{
    ImGui::BeginTooltip();
}

void InxGUIContext::EndTooltip()
{
    ImGui::EndTooltip();
}

void InxGUIContext::SetTooltip(const std::string &text)
{
    ImGui::SetTooltip("%s", text.c_str());
}

/* images */
void InxGUIContext::Image(void *textureId, float width, float height, float uv0_x, float uv0_y, float uv1_x,
                          float uv1_y)
{
    if (!textureId)
        return;
    ImGui::Image(reinterpret_cast<ImTextureID>(textureId), ImVec2(width, height), ImVec2(uv0_x, uv0_y),
                 ImVec2(uv1_x, uv1_y));
}

bool InxGUIContext::ImageButton(const std::string &id, void *textureId, float width, float height, float uv0_x,
                                float uv0_y, float uv1_x, float uv1_y)
{
    if (!textureId)
        return false;
    const bool clicked = ImGui::ImageButton(id.c_str(), reinterpret_cast<ImTextureID>(textureId), ImVec2(width, height),
                                            ImVec2(uv0_x, uv0_y), ImVec2(uv1_x, uv1_y));
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("image_button", id, true, id);
    return clicked;
}

/* tables */
bool InxGUIContext::BeginTable(const std::string &id, int columns, int flags, float innerWidth)
{
    return ImGui::BeginTable(id.c_str(), columns, flags, ImVec2(innerWidth, 0));
}

void InxGUIContext::EndTable()
{
    ImGui::EndTable();
}

void InxGUIContext::TableSetupColumn(const std::string &label, int flags, float initWidthOrWeight, int userID)
{
    ImGui::TableSetupColumn(label.c_str(), flags, initWidthOrWeight, userID);
}

void InxGUIContext::TableHeadersRow()
{
    ImGui::TableHeadersRow();
}

void InxGUIContext::TableNextRow()
{
    ImGui::TableNextRow();
}

void InxGUIContext::TableSetColumnIndex(int columnIndex)
{
    ImGui::TableSetColumnIndex(columnIndex);
}

bool InxGUIContext::TableNextColumn()
{
    return ImGui::TableNextColumn();
}

/* misc helpers */
bool InxGUIContext::CheckboxFlags(const std::string &label, unsigned int *flags, unsigned int flagValue)
{
    const bool changed = ImGui::CheckboxFlags(label.c_str(), flags, flagValue);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("checkbox", label, true, "", (*flags & flagValue) == flagValue);
    return changed;
}

void InxGUIContext::SetNextItemWidth(float width)
{
    ImGui::SetNextItemWidth(width);
}

void InxGUIContext::SetNextWindowSize(float width, float height, int cond)
{
    ImGui::SetNextWindowSize(ImVec2(width, height), static_cast<ImGuiCond>(cond));
}

void InxGUIContext::SetNextWindowPos(float x, float y, int cond, float pivot_x, float pivot_y)
{
    ImGui::SetNextWindowPos(ImVec2(x, y), static_cast<ImGuiCond>(cond), ImVec2(pivot_x, pivot_y));
}

void InxGUIContext::SetNextWindowFocus()
{
    ImGui::SetNextWindowFocus();
}

void InxGUIContext::SetWindowFocus()
{
    ImGui::SetWindowFocus();
}

bool InxGUIContext::BeginWindow(const std::string &name, bool *open, int flags)
{
    ConstrainNextFloatingWindowToMainViewport(name.c_str(), flags);
    const bool visible = ImGui::Begin(name.c_str(), open, flags);
    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    if (captureSemantics)
        RecordSemanticWindow("window", name, WindowSemanticId(name));
    if (captureSemantics && open != nullptr)
        InxGUISemantics::RecordCurrentWindowCloseButton(WindowSemanticId(name) + ".close");
    return visible;
}

bool InxGUIContext::IsCurrentWindowContentPresented()
{
    ImGuiContext *context = ImGui::GetCurrentContext();
    ImGuiWindow *window = context ? ImGui::GetCurrentWindowRead() : nullptr;
    if (window == nullptr)
        return false;

    ImGuiWindow *root = window->RootWindow != nullptr ? window->RootWindow : window;
    if (root->Hidden || root->Collapsed)
        return false;

#ifdef IMGUI_HAS_DOCK
    if (const ImGuiDockNode *dockNode = root->DockNode; dockNode != nullptr) {
        if (!root->DockNodeIsVisible || !root->DockTabIsVisible)
            return false;
        if (dockNode->VisibleWindow != nullptr)
            return dockNode->VisibleWindow == root;
    }
#endif

    return true;
}

void InxGUIContext::EndWindow()
{
    ImGui::End();
}

/* layout query */
float InxGUIContext::GetContentRegionAvailWidth()
{
    return ImGui::GetContentRegionAvail().x;
}

float InxGUIContext::GetContentRegionAvailHeight()
{
    return ImGui::GetContentRegionAvail().y;
}

float InxGUIContext::GetCursorPosX()
{
    return ImGui::GetCursorPosX();
}

float InxGUIContext::GetCursorPosY()
{
    return ImGui::GetCursorPosY();
}

bool InxGUIContext::IsVirtualizedRegionVisible(float height)
{
    if (height <= 0.0f)
        return true;

    // Visibility must be local to this region. Expanding every virtualized
    // Inspector body while any item is held or any popup is open changes the
    // child window's content height between mouse-down and mouse-up. At the
    // bottom of the Inspector that moves both the scroll position and the
    // item under the cursor. The active body and its popup are already inside
    // the visible clip rect when interaction begins, so they remain rendered
    // without a global interaction bypass.
    const float width = (std::max)(ImGui::GetContentRegionAvail().x, 1.0f);
    return ImGui::IsRectVisible(ImVec2(width, height));
}

void InxGUIContext::SetCursorPosX(float x)
{
    ImGui::SetCursorPosX(x);
}

void InxGUIContext::SetCursorPosY(float y)
{
    ImGui::SetCursorPosY(y);
}

void InxGUIContext::SetCursorScreenPos(float x, float y)
{
    ImGui::SetCursorScreenPos(ImVec2(x, y));
}

float InxGUIContext::GetWindowPosX()
{
    return ImGui::GetWindowPos().x;
}

float InxGUIContext::GetWindowPosY()
{
    return ImGui::GetWindowPos().y;
}

float InxGUIContext::CalcTextWidth(const std::string &text)
{
    return ImGui::CalcTextSize(text.c_str()).x;
}

float InxGUIContext::GetWindowWidth()
{
    return ImGui::GetWindowWidth();
}

float InxGUIContext::GetItemRectMinX()
{
    return ImGui::GetItemRectMin().x;
}

float InxGUIContext::GetItemRectMinY()
{
    return ImGui::GetItemRectMin().y;
}

float InxGUIContext::GetItemRectMaxX()
{
    return ImGui::GetItemRectMax().x;
}

float InxGUIContext::GetItemRectMaxY()
{
    return ImGui::GetItemRectMax().y;
}

std::array<float, 4> InxGUIContext::GetItemVisibleRect()
{
    ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    rect.ClipWithFull(ImGui::GetCurrentWindow()->ClipRect);
    return {rect.Min.x, rect.Min.y, rect.GetWidth(), rect.GetHeight()};
}

void InxGUIContext::RecordSemanticItem(const std::string &kind, const std::string &label, bool enabled,
                                       const std::string &semanticId, std::optional<bool> boolValue,
                                       std::optional<double> numericValue, std::optional<std::string> stringValue)
{
    InxGUISemantics::RecordLastItem(kind, label, enabled, semanticId, boolValue, numericValue, stringValue);
}

void InxGUIContext::RecordSemanticRect(const std::string &kind, const std::string &label, float x, float y, float width,
                                       float height, bool enabled, const std::string &semanticId)
{
    InxGUISemantics::RecordRect(kind, label, x, y, width, height, enabled, semanticId);
}

void InxGUIContext::RecordSemanticWindow(const std::string &kind, const std::string &label,
                                         const std::string &semanticId)
{
    InxGUISemantics::RecordCurrentWindow(kind, label, semanticId);
}

/* invisible button (for splitter) */
bool InxGUIContext::InvisibleButton(const std::string &id, float width, float height)
{
    const bool clicked = ImGui::InvisibleButton(id.c_str(), ImVec2(width, height));
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("invisible_button", id, true, id);
    return clicked;
}

bool InxGUIContext::IsItemActive()
{
    return ImGui::IsItemActive();
}

bool InxGUIContext::IsItemEdited()
{
    return ImGui::IsItemEdited();
}

bool InxGUIContext::IsAnyItemActive()
{
    return ImGui::IsAnyItemActive();
}

bool InxGUIContext::IsItemHovered()
{
    return ImGui::IsItemHovered();
}

int InxGUIContext::SearchableCombo(const std::string &id, int currentItem, const std::vector<std::string> &items,
                                   float width, int maxVisibleItems, const std::string &searchHint,
                                   const std::string &emptyText)
{
    if (id.empty())
        throw std::invalid_argument("SearchableCombo id cannot be empty");
    if (maxVisibleItems <= 0)
        throw std::invalid_argument("SearchableCombo maxVisibleItems must be positive");

    auto &state = m_searchableComboStates[id];
    const int safeCurrent = currentItem >= 0 && currentItem < static_cast<int>(items.size()) ? currentItem : -1;
    const std::string display = safeCurrent >= 0 ? items[safeCurrent] : std::string{};
    int result = currentItem;

    ImGui::PushID(id.c_str());
    const float triggerWidth = width > 0.0f ? width : ImGui::CalcItemWidth();
    const bool triggerClicked = ImGui::InvisibleButton("##trigger", ImVec2(triggerWidth, ImGui::GetFrameHeight()));
    const ImRect trigger(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    DrawComboFrame(trigger, display.c_str(), ImGui::IsItemHovered(), ImGui::IsPopupOpen("##popup"), s_dpiScale);
    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    if (captureSemantics)
        RecordSemanticItem("searchable_combo", display, true, id);
    if (triggerClicked) {
        state.filter.fill('\0');
        state.highlightedItem = safeCurrent;
        state.needsSearchFocus = true;
        state.scrollToHighlight = true;
        state.closeRequested = false;
        ImGui::OpenPopup("##popup");
    }
    if (state.restoreTriggerFocus) {
        ImGui::SetItemDefaultFocus();
        state.restoreTriggerFocus = false;
    }

    const float popupWidth = std::max(width, 220.0f * s_dpiScale);
    const float rowHeight = ImGui::GetTextLineHeightWithSpacing();
    const float listHeight = rowHeight * static_cast<float>(maxVisibleItems) + ImGui::GetStyle().WindowPadding.y * 2.0f;
    ImGui::SetNextWindowSizeConstraints(ImVec2(popupWidth, 0.0f), ImVec2(popupWidth, FLT_MAX));
    ImGui::SetNextWindowPos(ImVec2(trigger.Min.x, trigger.Max.y), ImGuiCond_Appearing);
    PushPopupWindowChrome();
    const bool comboOpen = ImGui::BeginPopup("##popup");
    PopPopupWindowChrome();
    if (comboOpen) {
        PushPopupContentStyle();
        if (captureSemantics)
            RecordSemanticWindow("combo_popup", id, id);
        const std::string transientToken = "searchable_combo:" + id;
        if (!state.wasOpen) {
            state.wasOpen = true;
            if (m_transientBegin) {
                m_transientBegin(transientToken, "popup", 150, [this, id]() {
                    auto it = m_searchableComboStates.find(id);
                    if (it == m_searchableComboStates.end())
                        return false;
                    it->second.closeRequested = true;
                    return true;
                });
            }
        }
        if (state.needsSearchFocus) {
            ImGui::SetKeyboardFocusHere();
            state.needsSearchFocus = false;
        }

        ImGui::SetNextItemWidth(-FLT_MIN);
        const std::string previousFilter(state.filter.data());
        const bool submitted =
            ImGui::InputTextWithHint("##search", searchHint.c_str(), state.filter.data(), state.filter.size(),
                                     ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue);
        if (captureSemantics)
            RecordSemanticItem("text_input", searchHint, true, id + ":search");
        const bool searchFocused = ImGui::IsItemFocused();
        const bool filterChanged = previousFilter != state.filter.data();

        std::string filterLower(state.filter.data());
        std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::vector<int> filtered;
        filtered.reserve(items.size());
        for (int index = 0; index < static_cast<int>(items.size()); ++index) {
            std::string itemLower = items[index];
            std::transform(itemLower.begin(), itemLower.end(), itemLower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (filterLower.empty() || itemLower.find(filterLower) != std::string::npos)
                filtered.push_back(index);
        }

        auto highlighted = std::find(filtered.begin(), filtered.end(), state.highlightedItem);
        if (filterChanged || highlighted == filtered.end()) {
            auto current = std::find(filtered.begin(), filtered.end(), safeCurrent);
            state.highlightedItem = current != filtered.end() ? *current : (filtered.empty() ? -1 : filtered.front());
            state.scrollToHighlight = true;
            highlighted = std::find(filtered.begin(), filtered.end(), state.highlightedItem);
        }

        if (searchFocused && !filtered.empty()) {
            int position = highlighted == filtered.end() ? 0 : static_cast<int>(highlighted - filtered.begin());
            if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
                position = (position + 1) % static_cast<int>(filtered.size());
                state.highlightedItem = filtered[position];
                state.scrollToHighlight = true;
            } else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
                position = (position + static_cast<int>(filtered.size()) - 1) % static_cast<int>(filtered.size());
                state.highlightedItem = filtered[position];
                state.scrollToHighlight = true;
            }
        }

        bool closePopup = false;
        if (state.closeRequested) {
            closePopup = true;
            state.closeRequested = false;
        } else if (submitted && state.highlightedItem >= 0) {
            result = state.highlightedItem;
            closePopup = true;
        }

        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        if (ImGui::BeginChild("##results", ImVec2(0.0f, listHeight), ImGuiChildFlags_None)) {
            if (filtered.empty()) {
                ImGui::TextDisabled("%s", emptyText.c_str());
            } else {
                for (int index : filtered) {
                    const bool highlightedItem = index == state.highlightedItem;
                    ImGui::PushID(index);
                    const bool selected = ImGui::Selectable((items[index] + "##item").c_str(), highlightedItem);
                    if (captureSemantics)
                        RecordSemanticItem("combo_option", items[index], true, id + ":option:" + std::to_string(index));
                    if (selected) {
                        result = index;
                        closePopup = true;
                    }
                    if (ImGui::IsItemHovered())
                        state.highlightedItem = index;
                    if (highlightedItem && state.scrollToHighlight) {
                        ImGui::SetScrollHereY(0.5f);
                        state.scrollToHighlight = false;
                    }
                    ImGui::PopID();
                }
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(1);

        if (closePopup) {
            ImGui::CloseCurrentPopup();
            state.wasOpen = false;
            state.restoreTriggerFocus = true;
            if (m_transientEnd)
                m_transientEnd(transientToken);
        }
        PopPopupContentStyle();
        ImGui::EndPopup();
    } else if (state.wasOpen) {
        state.wasOpen = false;
        state.closeRequested = false;
        state.restoreTriggerFocus = true;
        if (m_transientEnd)
            m_transientEnd("searchable_combo:" + id);
    }
    ImGui::PopID();
    return result;
}

bool InxGUIContext::IsItemFocused()
{
    return ImGui::IsItemFocused();
}

/* focus & activation */
void InxGUIContext::SetKeyboardFocusHere(int offset)
{
    ImGui::SetKeyboardFocusHere(offset);
}

bool InxGUIContext::IsItemDeactivated()
{
    return ImGui::IsItemDeactivated();
}

bool InxGUIContext::IsItemDeactivatedAfterEdit()
{
    return ImGui::IsItemDeactivatedAfterEdit();
}

float InxGUIContext::GetMouseDragDeltaY(int button)
{
    // Use 0.0f threshold for immediate response (no lock threshold)
    return ImGui::GetMouseDragDelta(button, 0.0f).y;
}

void InxGUIContext::ResetMouseDragDelta(int button)
{
    ImGui::ResetMouseDragDelta(button);
}

/* ID stack */
void InxGUIContext::PushID(int id)
{
    ImGui::PushID(id);
}

void InxGUIContext::PushID(const std::string &id)
{
    ImGui::PushID(id.c_str());
}

void InxGUIContext::PopID()
{
    ImGui::PopID();
}

void InxGUIContext::PushStyleColor(int idx, float r, float g, float b, float a)
{
    ImGui::PushStyleColor(static_cast<ImGuiCol_>(idx), ImVec4(r, g, b, a));
}

void InxGUIContext::PopStyleColor(int count)
{
    ImGui::PopStyleColor(count);
}

void InxGUIContext::PushStyleVarFloat(int idx, float val)
{
    ImGui::PushStyleVar(static_cast<ImGuiStyleVar_>(idx), val);
}

void InxGUIContext::PushStyleVarVec2(int idx, float x, float y)
{
    ImGui::PushStyleVar(static_cast<ImGuiStyleVar_>(idx), ImVec2(x, y));
}

void InxGUIContext::PopStyleVar(int count)
{
    ImGui::PopStyleVar(count);
}

void InxGUIContext::BeginDisabled(bool disabled)
{
    ImGui::BeginDisabled(disabled);
}

void InxGUIContext::EndDisabled()
{
    ImGui::EndDisabled();
}

/* Drag and Drop */
namespace
{
// ImGui copies opaque bytes. A small explicit tag preserves the value kind
// across native/Python panels without guessing from its length or source.
enum class DragValueKind : unsigned char
{
    Unknown = 0,
    UInt64 = 1,
    String = 2
};
constexpr size_t DragValueHeaderSize = 4;

DragValueKind GetDragValueKind(const ImGuiPayload *payload)
{
    if (!payload || !payload->Data || payload->DataSize < static_cast<int>(DragValueHeaderSize))
        return DragValueKind::Unknown;
    const auto *bytes = static_cast<const unsigned char *>(payload->Data);
    if (bytes[0] != 'I' || bytes[1] != 'N' || bytes[2] != 'X')
        return DragValueKind::Unknown;
    const auto kind = static_cast<DragValueKind>(bytes[3]);
    if (kind == DragValueKind::String)
        return kind;
    if (kind == DragValueKind::UInt64 && payload->DataSize == DragValueHeaderSize + sizeof(uint64_t))
        return kind;
    return DragValueKind::Unknown;
}

const char *DragValueData(const ImGuiPayload *payload)
{
    return static_cast<const char *>(payload->Data) + DragValueHeaderSize;
}

void ReadDragString(const ImGuiPayload *payload, std::string *value)
{
    value->assign(DragValueData(payload), payload->DataSize - DragValueHeaderSize);
}
} // namespace

bool InxGUIContext::BeginDragDropSource(int flags)
{
    return ImGui::BeginDragDropSource(static_cast<ImGuiDragDropFlags>(flags));
}

bool InxGUIContext::SetDragDropPayload(const std::string &type, uint64_t data)
{
    std::array<unsigned char, DragValueHeaderSize + sizeof(data)> bytes = {
        'I', 'N', 'X', static_cast<unsigned char>(DragValueKind::UInt64)};
    std::memcpy(bytes.data() + DragValueHeaderSize, &data, sizeof(data));
    return ImGui::SetDragDropPayload(type.c_str(), bytes.data(), bytes.size());
}

bool InxGUIContext::SetDragDropPayload(const std::string &type, const std::string &data)
{
    std::string bytes{'I', 'N', 'X', static_cast<char>(DragValueKind::String)};
    bytes.append(data);
    return ImGui::SetDragDropPayload(type.c_str(), bytes.data(), bytes.size());
}

void InxGUIContext::EndDragDropSource()
{
    ImGui::EndDragDropSource();
}

bool InxGUIContext::BeginDragDropTarget()
{
    return ImGui::BeginDragDropTarget();
}

bool InxGUIContext::BeginDragDropTargetRect(float minX, float minY, float maxX, float maxY, const std::string &targetId)
{
    ImGuiContext *context = ImGui::GetCurrentContext();
    if (context == nullptr || !context->DragDropActive)
        return false;
    const ImRect bounds(ImVec2(minX, minY), ImVec2(maxX, maxY));
    return ImGui::BeginDragDropTargetCustom(bounds, ImGui::GetID(targetId.c_str()));
}

bool InxGUIContext::AcceptDragDropPayload(const std::string &type, uint64_t *outData)
{
    const ImGuiPayload *preview = ImGui::GetDragDropPayload();
    if (GetDragValueKind(preview) != DragValueKind::UInt64 || !preview->IsDataType(type.c_str()))
        return false;
    const ImGuiPayload *payload = ImGui::AcceptDragDropPayload(type.c_str());
    if (payload) {
        std::memcpy(outData, DragValueData(payload), sizeof(*outData));
        return true;
    }
    return false;
}

bool InxGUIContext::AcceptDragDropPayload(const std::string &type, std::string *outData, int flags)
{
    const ImGuiPayload *preview = ImGui::GetDragDropPayload();
    if (GetDragValueKind(preview) != DragValueKind::String || !preview->IsDataType(type.c_str()))
        return false;
    const ImGuiPayload *payload = ImGui::AcceptDragDropPayload(type.c_str(), flags);
    if (payload) {
        ReadDragString(payload, outData);
        return true;
    }
    return false;
}

bool InxGUIContext::AcceptAnyDragDropPayload(std::string *outType, uint64_t *outU64, std::string *outStr,
                                             bool *outIsU64)
{
    const ImGuiPayload *preview = ImGui::GetDragDropPayload();
    const DragValueKind kind = GetDragValueKind(preview);
    if (kind == DragValueKind::Unknown)
        return false;
    const ImGuiPayload *acc = ImGui::AcceptDragDropPayload(preview->DataType);
    if (!acc)
        return false;
    outType->assign(preview->DataType);
    if (kind == DragValueKind::UInt64) {
        *outIsU64 = true;
        std::memcpy(outU64, DragValueData(acc), sizeof(*outU64));
        return true;
    }
    *outIsU64 = false;
    ReadDragString(acc, outStr);
    return true;
}

void InxGUIContext::EndDragDropTarget()
{
    ImGui::EndDragDropTarget();
}

void InxGUIContext::SetMouseCursor(int cursorType)
{
    ImGui::SetMouseCursor(static_cast<ImGuiMouseCursor>(cursorType));
}

// ========================================================================
// Scene View Input API implementation
// ========================================================================

bool InxGUIContext::IsMouseButtonDown(int button)
{
    return ImGui::IsMouseDown(button);
}

bool InxGUIContext::IsMouseButtonClicked(int button)
{
    return ImGui::IsMouseClicked(button);
}

bool InxGUIContext::IsMouseDoubleClicked(int button)
{
    return ImGui::IsMouseDoubleClicked(button);
}

bool InxGUIContext::IsMouseDragging(int button, float lockThreshold)
{
    return ImGui::IsMouseDragging(button, lockThreshold);
}

float InxGUIContext::GetMouseDragDeltaX(int button)
{
    // Use 0.0f threshold for immediate response (no lock threshold)
    return ImGui::GetMouseDragDelta(button, 0.0f).x;
}

float InxGUIContext::GetMousePosX()
{
    return ImGui::GetMousePos().x;
}

float InxGUIContext::GetMousePosY()
{
    return ImGui::GetMousePos().y;
}

float InxGUIContext::GetMouseWheelDelta()
{
    return ImGui::GetIO().MouseWheel;
}

bool InxGUIContext::IsKeyDown(int keyCode)
{
    return ImGui::IsKeyDown(static_cast<ImGuiKey>(keyCode));
}

bool InxGUIContext::IsKeyPressed(int keyCode)
{
    return ImGui::IsKeyPressed(static_cast<ImGuiKey>(keyCode));
}

bool InxGUIContext::IsKeyReleased(int keyCode)
{
    return ImGui::IsKeyReleased(static_cast<ImGuiKey>(keyCode));
}

bool InxGUIContext::IsWindowFocused(int flags)
{
    return ImGui::IsWindowFocused(flags);
}

bool InxGUIContext::IsWindowHovered(int flags)
{
    return ImGui::IsWindowHovered(flags);
}

bool InxGUIContext::WantTextInput()
{
    return ImGui::GetIO().WantTextInput;
}

void InxGUIContext::CaptureMouseFromApp(bool capture)
{
    ImGui::GetIO().WantCaptureMouse = capture;
}

void InxGUIContext::CaptureKeyboardFromApp(bool capture)
{
    ImGui::GetIO().WantCaptureKeyboard = capture;
}

void InxGUIContext::WarpMouseGlobal(float x, float y)
{
    SDL_WarpMouseGlobal(x, y);
}

float InxGUIContext::GetGlobalMousePosX()
{
    float gx = 0, gy = 0;
    SDL_GetGlobalMouseState(&gx, &gy);
    return gx;
}

float InxGUIContext::GetGlobalMousePosY()
{
    float gx = 0, gy = 0;
    SDL_GetGlobalMouseState(&gx, &gy);
    return gy;
}

void InxGUIContext::GetMainViewportBounds(float *x, float *y, float *w, float *h)
{
    ImGuiViewport *vp = ImGui::GetMainViewport();
    *x = vp->Pos.x;
    *y = vp->Pos.y;
    *w = vp->Size.x;
    *h = vp->Size.y;
}

void InxGUIContext::SetClipboardText(const std::string &text)
{
    ImGui::SetClipboardText(text.c_str());
}

std::string InxGUIContext::GetClipboardText()
{
    const char *t = ImGui::GetClipboardText();
    return t ? std::string(t) : std::string();
}

std::string InxGUIContext::InputTextMultiline(const std::string &label, const std::string &text, size_t bufferSize,
                                              float width, float height, int flags)
{
    if (!CanRenderWidgets())
        return text;

    const size_t capacity = std::max<size_t>(bufferSize, 1);
    std::vector<char> buffer(capacity, 0);
    const size_t copyLength = std::min(text.size(), capacity - 1);
    std::copy_n(text.begin(), copyLength, buffer.begin());
    ImGui::InputTextMultiline(label.c_str(), buffer.data(), buffer.size(), ImVec2(width, height), flags);
    if (InxGUISemantics::IsCaptureEnabled())
        RecordSemanticItem("text_area", label, true, label, std::nullopt, std::nullopt, std::string(buffer.data()));
    return std::string(buffer.data());
}

void InxGUIContext::GetDisplayBounds(float *x, float *y, float *w, float *h)
{
    int count = 0;
    SDL_DisplayID *displays = SDL_GetDisplays(&count);
    if (displays && count > 0) {
        SDL_Rect bounds{};
        if (SDL_GetDisplayBounds(displays[0], &bounds)) {
            *x = static_cast<float>(bounds.x);
            *y = static_cast<float>(bounds.y);
            *w = static_cast<float>(bounds.w);
            *h = static_cast<float>(bounds.h);
            SDL_free(displays);
            return;
        }
        SDL_free(displays);
    }
    // Fallback
    *x = 0;
    *y = 0;
    *w = 1920;
    *h = 1080;
}

// ==========================================================================
// Infinite drag — warp cursor to opposite screen edge when it hits a
// boundary, giving a Unity-style infinite-drag feel for DragFloat etc.
// Does NOT use SDL relative mouse mode (that conflicts with ImGui).
// ==========================================================================

void InxGUIContext::HandleDragCapture()
{
    const bool active = ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left);

    if (active && !m_dragCaptured) {
        // Just started dragging
        m_dragCaptured = true;
    }

    if (active) {
        // Warp cursor to opposite edge when hitting screen boundary
        float mx, my;
        SDL_GetGlobalMouseState(&mx, &my);

        SDL_DisplayID did = SDL_GetPrimaryDisplay();
        SDL_Rect bounds;
        if (SDL_GetDisplayBounds(did, &bounds)) {
            const float margin = 2.0f;
            const float left = static_cast<float>(bounds.x) + margin;
            const float right = static_cast<float>(bounds.x + bounds.w) - margin;
            const float top = static_cast<float>(bounds.y) + margin;
            const float bottom = static_cast<float>(bounds.y + bounds.h) - margin;

            float newMx = mx, newMy = my;
            bool warped = false;
            if (mx <= left) {
                newMx = right - margin;
                warped = true;
            } else if (mx >= right) {
                newMx = left + margin;
                warped = true;
            }
            if (my <= top) {
                newMy = bottom - margin;
                warped = true;
            } else if (my >= bottom) {
                newMy = top + margin;
                warped = true;
            }

            if (warped) {
                SDL_WarpMouseGlobal(newMx, newMy);
                // Ignore two frames of mouse delta after warp to prevent
                // artificial value jumps in DragFloat / DragInt.
                // (Some backends report the warp jump on the next frame,
                // others one frame later.)
                m_ignoreMouseDeltaFrames = 2;
            }
        }
    } else if (m_dragCaptured) {
        // Drag ended — do NOT snap cursor back to start position.
        // Snap-back causes visible stutter and introduces artificial deltas.
        m_dragCaptured = false;
    }
}

void InxGUIContext::CompensateWarp()
{
    ImGuiIO &io = ImGui::GetIO();

    if (m_ignoreMouseDeltaFrames > 0) {
        // Drop delta right after warp (teleport artifact).
        io.MouseDelta = ImVec2(0.0f, 0.0f);
        --m_ignoreMouseDeltaFrames;
        return;
    }

    // Safety net (same idea as Scene panel): ignore implausibly large deltas
    // during drag, which are almost always edge-wrap teleport artifacts.
    constexpr float kWarpJumpThreshold = 400.0f;
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) &&
        (std::fabs(io.MouseDelta.x) > kWarpJumpThreshold || std::fabs(io.MouseDelta.y) > kWarpJumpThreshold)) {
        io.MouseDelta = ImVec2(0.0f, 0.0f);
    }
}

void InxGUIContext::SetWindowFontScale(float scale)
{
    ImGui::SetWindowFontScale(scale);
}

void InxGUIContext::DrawRect(float minX, float minY, float maxX, float maxY, float r, float g, float b, float a,
                             float thickness, float rounding)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (drawList) {
        ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
        drawList->AddRect(ImVec2(minX, minY), ImVec2(maxX, maxY), col, rounding, 0, thickness);
    }
}

void InxGUIContext::DrawFilledRect(float minX, float minY, float maxX, float maxY, float r, float g, float b, float a,
                                   float rounding)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (drawList) {
        ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
        drawList->AddRectFilled(ImVec2(minX, minY), ImVec2(maxX, maxY), col, rounding);
    }
}

void InxGUIContext::DrawFilledRectRotated(float minX, float minY, float maxX, float maxY, float r, float g, float b,
                                          float a, float rotation, bool mirrorH, bool mirrorV, float rounding)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (!drawList)
        return;
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
    const int vtxStart = drawList->VtxBuffer.Size;
    drawList->AddRectFilled(ImVec2(minX, minY), ImVec2(maxX, maxY), col, rounding);

    rotation = std::fmod(rotation, 360.0f);
    if (rotation < 0.0f)
        rotation += 360.0f;
    if ((std::fabs(rotation) < 0.001f) && !mirrorH && !mirrorV)
        return;

    const float radians = rotation * 3.14159265358979f / 180.0f;
    const float cosA = std::cos(radians);
    const float sinA = std::sin(radians);
    const ImVec2 pivot((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);
    for (int i = vtxStart; i < drawList->VtxBuffer.Size; ++i) {
        ImVec2 local(drawList->VtxBuffer[i].pos.x - pivot.x, drawList->VtxBuffer[i].pos.y - pivot.y);
        if (mirrorH)
            local.x = -local.x;
        if (mirrorV)
            local.y = -local.y;
        const float rx = local.x * cosA - local.y * sinA;
        const float ry = local.x * sinA + local.y * cosA;
        drawList->VtxBuffer[i].pos = ImVec2(pivot.x + rx, pivot.y + ry);
    }
}

void InxGUIContext::DrawLine(float x1, float y1, float x2, float y2, float r, float g, float b, float a,
                             float thickness)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (drawList) {
        ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
        drawList->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), col, thickness);
    }
}

void InxGUIContext::DrawCircle(float centerX, float centerY, float radius, float r, float g, float b, float a,
                               float thickness, int segments)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (drawList) {
        ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
        drawList->AddCircle(ImVec2(centerX, centerY), radius, col, segments, thickness);
    }
}

void InxGUIContext::DrawFilledCircle(float centerX, float centerY, float radius, float r, float g, float b, float a,
                                     int segments)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (drawList) {
        ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
        drawList->AddCircleFilled(ImVec2(centerX, centerY), radius, col, segments);
    }
}

void InxGUIContext::DrawImageRect(uint64_t textureId, float minX, float minY, float maxX, float maxY, float uv0_x,
                                  float uv0_y, float uv1_x, float uv1_y, float tintR, float tintG, float tintB,
                                  float tintA, float rotation, bool mirrorH, bool mirrorV, float rounding)
{
    if (textureId == 0)
        return;
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (!drawList)
        return;
    ImU32 tint = ImGui::ColorConvertFloat4ToU32(ImVec4(tintR, tintG, tintB, tintA));
    const int vtxStart = drawList->VtxBuffer.Size;
    if (rounding > 0.5f)
        drawList->AddImageRounded(ToImTextureID(textureId), ImVec2(minX, minY), ImVec2(maxX, maxY),
                                  ImVec2(uv0_x, uv0_y), ImVec2(uv1_x, uv1_y), tint, rounding);
    else
        drawList->AddImage(ToImTextureID(textureId), ImVec2(minX, minY), ImVec2(maxX, maxY), ImVec2(uv0_x, uv0_y),
                           ImVec2(uv1_x, uv1_y), tint);

    rotation = std::fmod(rotation, 360.0f);
    if (rotation < 0.0f)
        rotation += 360.0f;
    if ((std::fabs(rotation) < 0.001f) && !mirrorH && !mirrorV)
        return;

    const float radians = rotation * 3.14159265358979f / 180.0f;
    const float cosA = std::cos(radians);
    const float sinA = std::sin(radians);
    const ImVec2 pivot((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);
    for (int i = vtxStart; i < drawList->VtxBuffer.Size; ++i) {
        ImVec2 local(drawList->VtxBuffer[i].pos.x - pivot.x, drawList->VtxBuffer[i].pos.y - pivot.y);
        if (mirrorH)
            local.x = -local.x;
        if (mirrorV)
            local.y = -local.y;
        const float rx = local.x * cosA - local.y * sinA;
        const float ry = local.x * sinA + local.y * cosA;
        drawList->VtxBuffer[i].pos = ImVec2(pivot.x + rx, pivot.y + ry);
    }
}

void InxGUIContext::DrawFilledImage(uint64_t textureId, float minX, float minY, float maxX, float maxY, float u0,
                                    float v0, float u1, float v1, float r, float g, float b, float a, float amount,
                                    int method, int origin, bool clockwise, float rotation, bool mirrorH, bool mirrorV)
{
    if (auto *draw = ImGui::GetWindowDrawList())
        AppendFilledImage(*draw, ToImTextureID(textureId), minX, minY, maxX, maxY, u0, v0, u1, v1,
                          ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a)), amount, method, origin, clockwise,
                          rotation, mirrorH, mirrorV);
}

void InxGUIContext::DrawText(float x, float y, const std::string &text, float r, float g, float b, float a,
                             float fontSize)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (!drawList)
        return;
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
    textlayout::TextLayoutParams params{text, "", fontSize, 0.0f, 1.0f, 0.0f};
    params.fontOverride = ImGui::GetFont();
    const textlayout::TextLayoutResult layout = textlayout::LayoutText(params);
    if (!layout.lines.empty()) {
        drawList->PushTextureID(ImGui::GetIO().Fonts->TexRef);
        textlayout::RenderLine(drawList, layout, layout.lines.front(), x, y, col);
        drawList->PopTextureID();
    }
}

void InxGUIContext::DrawTextAligned(float minX, float minY, float maxX, float maxY, const std::string &text, float r,
                                    float g, float b, float a, float alignX, float alignY, float fontSize, bool clip)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (!drawList)
        return;

    textlayout::TextLayoutParams params{text, "", fontSize, 0.0f, 1.0f, 0.0f};
    params.fontOverride = ImGui::GetFont();
    const textlayout::TextLayoutResult layout = textlayout::LayoutText(params);

    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));

    if (clip)
        drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);

    drawList->PushTextureID(ImGui::GetIO().Fonts->TexRef);
    textlayout::RenderTextBox(drawList, minX, minY, maxX, maxY, layout, col, alignX, alignY);
    drawList->PopTextureID();

    if (clip)
        drawList->PopClipRect();
}

void InxGUIContext::DrawTextRotated90Aligned(float minX, float minY, float maxX, float maxY, const std::string &text,
                                             float r, float g, float b, float a, float alignX, float alignY,
                                             float fontSize, bool clockwise, bool clip)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (!drawList || text.empty())
        return;

    textlayout::TextLayoutParams params{text, "", fontSize, 0.0f, 1.0f, 0.0f};
    params.fontOverride = ImGui::GetFont();
    const textlayout::TextLayoutResult layout = textlayout::LayoutText(params);
    const ImVec2 textSize(layout.totalWidth, layout.totalHeight);

    float rotatedW = textSize.y;
    float rotatedH = textSize.x;
    float targetX = minX + (maxX - minX - rotatedW) * alignX;
    float targetY = minY + (maxY - minY - rotatedH) * alignY;

    const int vtxStart = drawList->VtxBuffer.Size;
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));

    if (clip)
        drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);

    drawList->PushTextureID(ImGui::GetIO().Fonts->TexRef);
    textlayout::RenderTextBox(drawList, minX, minY, minX + layout.totalWidth, minY + layout.totalHeight, layout, col,
                              0.0f, 0.0f);
    drawList->PopTextureID();

    if (clip)
        drawList->PopClipRect();

    if (drawList->VtxBuffer.Size <= vtxStart)
        return;

    ImVec2 boundsMin(FLT_MAX, FLT_MAX);
    ImVec2 boundsMax(-FLT_MAX, -FLT_MAX);
    for (int i = vtxStart; i < drawList->VtxBuffer.Size; ++i) {
        const ImVec2 &p = drawList->VtxBuffer[i].pos;
        boundsMin.x = std::min(boundsMin.x, p.x);
        boundsMin.y = std::min(boundsMin.y, p.y);
        boundsMax.x = std::max(boundsMax.x, p.x);
        boundsMax.y = std::max(boundsMax.y, p.y);
    }

    ImVec2 rotatedMin(FLT_MAX, FLT_MAX);
    ImVec2 rotatedMax(-FLT_MAX, -FLT_MAX);
    for (int i = vtxStart; i < drawList->VtxBuffer.Size; ++i) {
        ImVec2 local(drawList->VtxBuffer[i].pos.x - boundsMin.x, drawList->VtxBuffer[i].pos.y - boundsMin.y);
        ImVec2 rotated = clockwise ? ImVec2(local.y, textSize.x - local.x) : ImVec2(textSize.y - local.y, local.x);
        drawList->VtxBuffer[i].pos = rotated;
        rotatedMin.x = std::min(rotatedMin.x, rotated.x);
        rotatedMin.y = std::min(rotatedMin.y, rotated.y);
        rotatedMax.x = std::max(rotatedMax.x, rotated.x);
        rotatedMax.y = std::max(rotatedMax.y, rotated.y);
    }

    ImVec2 delta(targetX - rotatedMin.x, targetY - rotatedMin.y);
    for (int i = vtxStart; i < drawList->VtxBuffer.Size; ++i) {
        drawList->VtxBuffer[i].pos.x += delta.x;
        drawList->VtxBuffer[i].pos.y += delta.y;
    }
}

void InxGUIContext::DrawTextExAligned(float minX, float minY, float maxX, float maxY, const std::string &text, float r,
                                      float g, float b, float a, float alignX, float alignY, float fontSize,
                                      float wrapWidth, float rotation, bool mirrorH, bool mirrorV, bool clip,
                                      const std::string &fontPath, float lineHeight, float letterSpacing,
                                      const std::vector<std::string> &fallbackFontPaths)
{
    // Normalise rotation to [0, 360)
    rotation = std::fmod(rotation, 360.0f);
    if (rotation < 0.0f)
        rotation += 360.0f;

    // General path: layout the text inside the element box first, then transform
    // the generated vertices around the box center. This keeps editor and game
    // rendering aligned with the component's bounding box semantics.
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (!drawList || text.empty())
        return;

    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, fontPath, fontSize, wrapWidth, lineHeight, letterSpacing, fallbackFontPaths});
    const ImVec2 textSize(layout.totalWidth, layout.totalHeight);

    if (std::fabs(rotation) < 0.001f && !mirrorH && !mirrorV) {
        if (clip)
            drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);
        const ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
        drawList->PushTextureID(ImGui::GetIO().Fonts->TexRef);
        textlayout::RenderTextBox(drawList, minX, minY, maxX, maxY, layout, col, alignX, alignY);
        drawList->PopTextureID();
        if (clip)
            drawList->PopClipRect();
        return;
    }

    const ImVec2 pivot((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);

    if (clip)
        drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);

    const int vtxStart = drawList->VtxBuffer.Size;
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
    drawList->PushTextureID(ImGui::GetIO().Fonts->TexRef);
    textlayout::RenderTextBox(drawList, minX, minY, maxX, maxY, layout, col, alignX, alignY);
    drawList->PopTextureID();

    if (clip)
        drawList->PopClipRect();

    if (drawList->VtxBuffer.Size <= vtxStart)
        return;

    // Compute sin/cos for arbitrary angle
    float radians = rotation * 3.14159265358979f / 180.0f;
    float cosA = std::cos(radians);
    float sinA = std::sin(radians);

    for (int i = vtxStart; i < drawList->VtxBuffer.Size; ++i) {
        ImVec2 local(drawList->VtxBuffer[i].pos.x - pivot.x, drawList->VtxBuffer[i].pos.y - pivot.y);
        if (mirrorH)
            local.x = -local.x;
        if (mirrorV)
            local.y = -local.y;
        float rx = local.x * cosA - local.y * sinA;
        float ry = local.x * sinA + local.y * cosA;
        drawList->VtxBuffer[i].pos = ImVec2(pivot.x + rx, pivot.y + ry);
    }
}

void InxGUIContext::DrawTextGeometry(float minX, float minY, float maxX, float maxY, textlayout::TextGeometry &geometry,
                                     float r, float g, float b, float a, float alignX, float alignY, float scale,
                                     float rotation, bool mirrorH, bool mirrorV, bool clip, float opacity,
                                     const std::array<float, 4> &materialColor)
{
    geometry.Prepare(scale);
    ImDrawList *draw = ImGui::GetWindowDrawList();
    const int firstVertex = draw->VtxBuffer.Size;
    draw->PushTextureID(ImGui::GetIO().Fonts->TexRef);
    geometry.Render(*draw, minX, minY, maxX, maxY, {r, g, b, a}, alignX, alignY, scale, rotation, mirrorH, mirrorV,
                    false, clip, nullptr, 0, opacity);
    draw->PopTextureID();
    if (materialColor != std::array<float, 4>{1.f, 1.f, 1.f, 1.f}) {
        for (int index = firstVertex; index < draw->VtxBuffer.Size; ++index) {
            auto &vertex = draw->VtxBuffer[index];
            auto color = ImGui::ColorConvertU32ToFloat4(vertex.col);
            color.x *= materialColor[0];
            color.y *= materialColor[1];
            color.z *= materialColor[2];
            color.w *= materialColor[3];
            vertex.col = ImGui::ColorConvertFloat4ToU32(color);
        }
    }
}

std::pair<float, float> InxGUIContext::CalcTextSizeA(const std::string &text, float fontSize,
                                                     const std::string &fontPath, float lineHeight, float letterSpacing,
                                                     const std::vector<std::string> &fallbackFontPaths)
{
    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, fontPath, fontSize, 0.0f, lineHeight, letterSpacing, fallbackFontPaths});
    return {layout.totalWidth, layout.totalHeight};
}

std::pair<float, float> InxGUIContext::CalcTextSizeWrappedA(const std::string &text, float fontSize, float wrapWidth,
                                                            const std::string &fontPath, float lineHeight,
                                                            float letterSpacing,
                                                            const std::vector<std::string> &fallbackFontPaths)
{
    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, fontPath, fontSize, wrapWidth, lineHeight, letterSpacing, fallbackFontPaths});
    return {layout.totalWidth, layout.totalHeight};
}

void InxGUIContext::PushDrawListClipRect(float minX, float minY, float maxX, float maxY, bool intersectWithCurrent)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (drawList)
        drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), intersectWithCurrent);
}

void InxGUIContext::PopDrawListClipRect()
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (drawList)
        drawList->PopClipRect();
}

// ═════════════════════════════════════════════════════════════════════════
//  Batch property renderer
// ═════════════════════════════════════════════════════════════════════════

std::vector<PropertyChange> InxGUIContext::RenderPropertyBatch(const std::vector<PropertyDesc> &descriptors,
                                                               float labelWidth, int *activeIndex,
                                                               int *deactivatedAfterEditIndex)
{
    std::vector<PropertyChange> changes;
    if (activeIndex)
        *activeIndex = -1;
    if (deactivatedAfterEditIndex)
        *deactivatedAfterEditIndex = -1;

    auto doLabel = [&](const std::string &label) {
        if (!label.empty()) {
            InspectorFieldLabel(label, labelWidth);
        } else {
            ImGui::SetNextItemWidth(-1);
        }
    };

    auto drawMixedOverlay = [](bool mixed) {
        if (!mixed || ImGui::IsItemActive())
            return;
        ImDrawList *drawList = ImGui::GetWindowDrawList();
        if (!drawList)
            return;
        const ImVec2 min = ImGui::GetItemRectMin();
        const ImVec2 max = ImGui::GetItemRectMax();
        const ImGuiStyle &style = ImGui::GetStyle();
        const ImVec2 textPos(min.x + style.FramePadding.x, min.y + style.FramePadding.y);
        drawList->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_FrameBg), style.FrameRounding);
        drawList->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), "—");
    };

    const bool captureSemantics = InxGUISemantics::IsCaptureEnabled();
    const int count = static_cast<int>(descriptors.size());
    for (int i = 0; i < count; ++i) {
        const auto &d = descriptors[i];
        const std::string &semanticId = d.semanticId.empty() ? d.widgetId : d.semanticId;

        // All field paths share space, section, value, then help text order.
        if (d.space > 0)
            ImGui::Dummy(ImVec2(0, d.space * GetDpiScale()));
        if (!d.header.empty()) {
            ImGui::Spacing();
            PushMonoFont();
            ImGui::TextColored(EditorThemeRegistry::Color("TEXT_DISABLED"), "%s", d.header.c_str());
            PopFont();
            ImGui::Separator();
        }

        if (d.readonly) {
            doLabel(d.label);
            ReadonlyField(d.widgetId, d.readonlyText);
            if (captureSemantics)
                RecordSemanticItem("inspector_field", d.label, false, semanticId);
        } else
            switch (d.type) {
            case PropertyDesc::Float: {
                doLabel(d.label);
                float val = d.fVal[0];
                float orig = val;
                CompensateWarp();
                if (d.slider && d.hasRange)
                    DrawUnityRangedFloat(d.widgetId.c_str(), &val, d.rangeMin, d.rangeMax, "%.3f", GetDpiScale());
                else {
                    const ImGuiSliderFlags flags = d.hasRange ? ImGuiSliderFlags_AlwaysClamp : ImGuiSliderFlags_None;
                    MonoValueFontScope mono(d.widgetId.c_str());
                    ImGui::DragFloat(d.widgetId.c_str(), &val, d.speed, d.hasRange ? d.rangeMin : 0.0f,
                                     d.hasRange ? d.rangeMax : 0.0f, "%.3f", flags);
                }
                if (captureSemantics)
                    RecordSemanticItem(d.slider ? "float_slider" : "drag_float", d.label, true, semanticId,
                                       std::nullopt, static_cast<double>(val));
                HandleDragCapture();
                drawMixedOverlay(d.mixed);
                if (val != orig) {
                    PropertyChange c;
                    c.index = i;
                    c.type = PropertyDesc::Float;
                    c.fVal[0] = val;
                    changes.push_back(c);
                }
                break;
            }
            case PropertyDesc::Int: {
                doLabel(d.label);
                int64_t val = d.iVal;
                const int64_t orig = val;
                CompensateWarp();
                if (d.slider && d.hasRange) {
                    DrawUnityRangedInt(d.widgetId.c_str(), &val, d.intRangeMin, d.intRangeMax, GetDpiScale());
                } else {
                    const ImGuiSliderFlags flags = d.hasRange ? ImGuiSliderFlags_AlwaysClamp : ImGuiSliderFlags_None;
                    MonoValueFontScope mono(d.widgetId.c_str());
                    ImGui::DragScalar(d.widgetId.c_str(), ImGuiDataType_S64, &val, d.speed,
                                      d.hasRange ? &d.intRangeMin : nullptr, d.hasRange ? &d.intRangeMax : nullptr,
                                      nullptr, flags);
                }
                if (captureSemantics)
                    RecordSemanticItem(d.slider ? "int_slider" : "drag_int", d.label, true, semanticId, std::nullopt,
                                       static_cast<double>(val));
                HandleDragCapture();
                drawMixedOverlay(d.mixed);
                if (val != orig) {
                    PropertyChange c;
                    c.index = i;
                    c.type = PropertyDesc::Int;
                    c.iVal = val;
                    changes.push_back(c);
                }
                break;
            }
            case PropertyDesc::Bool: {
                bool val = d.bVal;
                bool orig = val;
                if (d.fieldLabel)
                    doLabel(d.label);
                // When the label is drawn separately, only pass an id so the
                // square can scale without shrinking any text.
                std::string cbLabel;
                if (d.fieldLabel) {
                    cbLabel = d.widgetId.find("##") != std::string::npos ? d.widgetId : ("##" + d.widgetId);
                } else if (!d.label.empty()) {
                    cbLabel = d.label + "##" + d.widgetId;
                } else {
                    cbLabel = d.widgetId;
                }
                if (d.mixed)
                    ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, true);
                DrawInspectorCheckboxSquare(cbLabel, &val, nullptr, GetDpiScale(), d.fieldLabel);
                if (captureSemantics)
                    RecordSemanticItem("checkbox", d.label.empty() ? cbLabel : d.label, true, semanticId, val);
                if (d.mixed)
                    ImGui::PopItemFlag();
                if (val != orig) {
                    PropertyChange c;
                    c.index = i;
                    c.type = PropertyDesc::Bool;
                    c.bVal = val;
                    changes.push_back(c);
                }
                break;
            }
            case PropertyDesc::String: {
                doLabel(d.label);
                std::string value = d.mixed ? std::string("—") : d.sVal;
                const bool edited =
                    d.multiline ? ImGui::InputTextMultiline(d.widgetId.c_str(), &value,
                                                            ImVec2(-1, 4.0f * ImGui::GetTextLineHeightWithSpacing()))
                                : ImGui::InputText(d.widgetId.c_str(), &value);
                if (captureSemantics)
                    RecordSemanticItem(d.multiline ? "text_area" : "text_input", d.label, true, semanticId);
                if (edited && (d.mixed || value != d.sVal)) {
                    PropertyChange c;
                    c.index = i;
                    c.type = PropertyDesc::String;
                    c.sVal = std::move(value);
                    changes.push_back(c);
                }
                break;
            }
            case PropertyDesc::Vec2: {
                float v[2] = {d.fVal[0], d.fVal[1]};
                float lw = d.label.empty() ? 1.0f : labelWidth;
                ImGui::PushID(d.widgetId.c_str());
                Vector2Control(d.label.empty() ? " " : d.label, v, d.speed, lw);
                ImGui::PopID();
                if (captureSemantics)
                    RecordSemanticItem("vector", d.label, true, semanticId);
                drawMixedOverlay(d.mixed);
                if (v[0] != d.fVal[0] || v[1] != d.fVal[1]) {
                    PropertyChange c;
                    c.index = i;
                    c.type = PropertyDesc::Vec2;
                    c.fVal[0] = v[0];
                    c.fVal[1] = v[1];
                    changes.push_back(c);
                }
                break;
            }
            case PropertyDesc::Vec3: {
                float v[3] = {d.fVal[0], d.fVal[1], d.fVal[2]};
                float lw = d.label.empty() ? 1.0f : labelWidth;
                ImGui::PushID(d.widgetId.c_str());
                Vector3Control(d.label.empty() ? " " : d.label, v, d.speed, lw);
                ImGui::PopID();
                if (captureSemantics)
                    RecordSemanticItem("vector", d.label, true, semanticId);
                drawMixedOverlay(d.mixed);
                if (v[0] != d.fVal[0] || v[1] != d.fVal[1] || v[2] != d.fVal[2]) {
                    PropertyChange c;
                    c.index = i;
                    c.type = PropertyDesc::Vec3;
                    c.fVal[0] = v[0];
                    c.fVal[1] = v[1];
                    c.fVal[2] = v[2];
                    changes.push_back(c);
                }
                break;
            }
            case PropertyDesc::Vec4: {
                float v[4] = {d.fVal[0], d.fVal[1], d.fVal[2], d.fVal[3]};
                float lw = d.label.empty() ? 1.0f : labelWidth;
                ImGui::PushID(d.widgetId.c_str());
                Vector4Control(d.label.empty() ? " " : d.label, v, d.speed, lw);
                ImGui::PopID();
                if (captureSemantics)
                    RecordSemanticItem("vector", d.label, true, semanticId);
                drawMixedOverlay(d.mixed);
                if (v[0] != d.fVal[0] || v[1] != d.fVal[1] || v[2] != d.fVal[2] || v[3] != d.fVal[3]) {
                    PropertyChange c;
                    c.index = i;
                    c.type = PropertyDesc::Vec4;
                    c.fVal[0] = v[0];
                    c.fVal[1] = v[1];
                    c.fVal[2] = v[2];
                    c.fVal[3] = v[3];
                    changes.push_back(c);
                }
                break;
            }
            case PropertyDesc::Enum: {
                doLabel(d.label);
                int idx = static_cast<int>(d.iVal);
                int orig = idx;
                Combo(d.widgetId, &idx, d.enumNames);
                if (captureSemantics)
                    RecordSemanticItem("combo", d.label, true, semanticId);
                drawMixedOverlay(d.mixed);
                if (idx != orig) {
                    PropertyChange c;
                    c.index = i;
                    c.type = PropertyDesc::Enum;
                    c.iVal = idx;
                    changes.push_back(c);
                }
                break;
            }
            case PropertyDesc::Color: {
                doLabel(d.label);
                float col[4] = {d.fVal[0], d.fVal[1], d.fVal[2], d.fVal[3]};
                if (ImGui::ColorEdit4(d.widgetId.c_str(), col)) {
                    PropertyChange c;
                    c.index = i;
                    c.type = PropertyDesc::Color;
                    c.fVal[0] = col[0];
                    c.fVal[1] = col[1];
                    c.fVal[2] = col[2];
                    c.fVal[3] = col[3];
                    changes.push_back(c);
                }
                if (captureSemantics)
                    RecordSemanticItem("color_edit", d.label, true, semanticId);
                drawMixedOverlay(d.mixed);
                break;
            }
            } // switch

        // Tooltip for the last rendered widget (label hover is not tracked).
        if (!d.tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", d.tooltip.c_str());
        if (activeIndex && ImGui::IsItemActive())
            *activeIndex = i;
        if (deactivatedAfterEditIndex && ImGui::IsItemDeactivatedAfterEdit())
            *deactivatedAfterEditIndex = i;
        if (!d.infoText.empty()) {
            InspectorFieldInfo(d.infoText, labelWidth);
        }
    }
    return changes;
}

uint32_t InxGUIContext::RenderObjectFieldChrome(const std::string &fieldId, const std::string &displayText,
                                                const std::string &typeHint, bool selected, bool clickable,
                                                bool hasPicker, uint64_t pickerTextureId, const std::string &semanticId,
                                                float fixedWidth)
{
    const float dpi = GetDpiScale();
    // Unity ObjectField: one shared frame height, left text inset, picker flush
    // to the right. Hover/active fill covers the whole control (body + picker).
    const ImVec4 bodyIdleColor = EditorThemeRegistry::Color("OBJECT_FIELD_BG");
    const ImVec4 bodyHoverColor = EditorThemeRegistry::Color("FRAME_BG_HOVERED");
    const ImVec4 bodyActiveColor = EditorThemeRegistry::Color("FRAME_BG_ACTIVE");
    const ImVec4 borderIdleColor = EditorThemeRegistry::Color("BORDER");
    const ImVec4 borderHoverColor = EditorThemeRegistry::Color("ROLE_BORDER");
    const ImVec4 borderActiveColor = EditorThemeRegistry::Color("ROLE_ACCENT");
    const ImVec4 buttonIdleColor = EditorThemeRegistry::Color("INSPECTOR_INLINE_BTN_IDLE");
    const ImVec4 buttonHoverColor = EditorThemeRegistry::Color("INSPECTOR_INLINE_BTN_HOVER");
    const ImVec4 buttonActiveColor = EditorThemeRegistry::Color("INSPECTOR_INLINE_BTN_ACTIVE");
    const float textInsetX = EditorThemeRegistry::Float("OBJECT_FIELD_TEXT_INSET_X") * dpi;

    if (fieldId.empty())
        throw std::invalid_argument("ObjectField fieldId must not be empty.");
    const std::string resolvedSemanticId = semanticId.empty() ? "object_field." + fieldId : semanticId;

    ImGui::PushID(fieldId.c_str());

    const std::string &fullText = displayText;

    const float availableWidth = ImGui::GetContentRegionAvail().x;
    const float fieldHeight = ImGui::GetFrameHeight();
    const float buttonWidth = hasPicker ? fieldHeight : 0.0f;
    if (hasPicker)
        pickerTextureId = EditorIcon("glyphs/picker");
    // fixedWidth > 0 caps the field (node-graph pin slots); otherwise fill the
    // available region like a normal Inspector row.
    const float totalWidth = fixedWidth > 0.0f ? (std::max)(fixedWidth, buttonWidth + 10.0f * dpi)
                                               : (std::max)(availableWidth, buttonWidth + 10.0f * dpi);
    const float bodyWidth = (std::max)(totalWidth - buttonWidth, 10.0f * dpi);

    const ImVec2 start = ImGui::GetCursorScreenPos();
    ImGui::BeginGroup();
    ImGui::InvisibleButton("##object_field_body", ImVec2(bodyWidth, fieldHeight));
    const bool bodyHovered = ImGui::IsItemHovered();
    const bool bodyPressed = ImGui::IsItemActive();
    uint32_t result = 0;
    // The body always reports navigation intent. `clickable` only describes
    // whether the current reference can resolve to a useful target; it must
    // not create a second hit-test contract for read-only fields.
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
        result |= 1u;
    // Bit 4: body double-click — domains may use this to open the reference.
    if (bodyHovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        result |= 4u;
    // Bit 8: keyboard activation mirrors a body double-click.  The Python
    // interaction model decides whether that opens the resource or locates it.
    if (ImGui::IsItemFocused() &&
        (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)))
        result |= 8u;
    // Bit 16: focused ObjectField clear. Text editors retain Delete/Backspace
    // because their active input owns WantTextInput instead of this field.
    if (ImGui::IsItemFocused() && !ImGui::GetIO().WantTextInput &&
        (ImGui::IsKeyPressed(ImGuiKey_Delete, false) || ImGui::IsKeyPressed(ImGuiKey_Backspace, false)))
        result |= 16u;
    // Bit 32: shared ObjectField context menu request. Python opens the
    // stable field-id popup after native batch rendering has completed.
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right))
        result |= 32u;

    bool pickerHovered = false;
    bool pickerPressed = false;
    ImVec2 pickerMin{};
    ImVec2 pickerMax{};
    if (hasPicker) {
        ImGui::SameLine(0.0f, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
        if (ImGui::Button("##picker", ImVec2(buttonWidth, fieldHeight)))
            result |= 2u;
        pickerHovered = ImGui::IsItemHovered();
        pickerPressed = ImGui::IsItemActive();
        if (ImGui::IsItemClicked(ImGuiMouseButton_Right))
            result |= 32u;
        pickerMin = ImGui::GetItemRectMin();
        pickerMax = ImGui::GetItemRectMax();
        if (InxGUISemantics::IsCaptureEnabled())
            RecordSemanticItem("button", "Select " + typeHint, true, resolvedSemanticId + ".picker");
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);
    }

    // Only the picker button opens the object selector. Python applies the
    // shared ObjectField contract to the body flags: single-click locates the
    // current reference, while double-click/Enter may open it when a domain
    // supplies an opener.
    if ((result & 2u) != 0 && hasPicker)
        ImGui::OpenPopup("##obj_picker");

    ImGui::EndGroup();
    // Keep the whole ObjectField (body + picker) as the last item so drag-drop
    // targets and semantic outlines cover the Unity-aligned control bounds.
    RecordSemanticItem("object_field", displayText, clickable || hasPicker, resolvedSemanticId);

    const ImVec2 end{start.x + totalWidth, start.y + fieldHeight};
    const bool groupHovered = bodyHovered || pickerHovered || ImGui::IsItemHovered();
    const bool groupPressed =
        bodyPressed || pickerPressed || (groupHovered && ImGui::IsMouseDown(ImGuiMouseButton_Left));
    const ImVec4 &fill =
        selected ? bodyActiveColor : (groupPressed ? bodyActiveColor : (groupHovered ? bodyHoverColor : bodyIdleColor));
    const ImVec4 &drawBorder = groupPressed ? borderActiveColor : (groupHovered ? borderHoverColor : borderIdleColor);

    ImDrawList *drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(start, end, ImGui::ColorConvertFloat4ToU32(fill), 0.0f);
    drawList->AddRect(start, end, ImGui::ColorConvertFloat4ToU32(drawBorder), 0.0f, 0, 1.0f * dpi);
    static const std::unordered_map<std::string, const char *> referenceIcons{
        {"GameObject", "components/component_transform"},
        {"Component", "components/component_script"},
        {"Material", "material"},
        {"Mesh", "model_3d"},
        {"Texture", "texture"},
        {"Texture2D", "texture"},
        {"Shader", "shader_glsl"},
        {"Vert", "shader_vert"},
        {"Frag", "shader_frag"},
        {"Script", "script_py"},
        {"RenderEffect", "render_effect"},
        {"RenderEffectGroup", "render_effect_group"},
        {"AudioClip", "audio"},
        {"Prefab", "prefab"},
        {"RenderTexture", "rendertexture"},
    };
    const auto refIcon = referenceIcons.find(typeHint);
    const auto icon = EditorIcon(refIcon != referenceIcons.end() ? refIcon->second : "file");
    if (icon) {
        const float size = 12.0f * dpi;
        const ImVec2 lo(start.x + 6.0f * dpi, start.y + (fieldHeight - size) * 0.5f);
        drawList->AddImage(ImTextureRef(static_cast<ImTextureID>(icon)), lo, ImVec2(lo.x + size, lo.y + size));
    }

    const ImVec2 textSize = ImGui::CalcTextSize(fullText.c_str());
    PushMonoFont();
    const ImVec2 typeSize = ImGui::CalcTextSize(typeHint.c_str());
    PopFont();
    const bool showType = bodyWidth > 150.0f * dpi && typeSize.x < bodyWidth * 0.4f;
    const float typeWidth = showType ? typeSize.x + 12.0f * dpi : 0.0f;
    const float textMaxWidth = (std::max)(0.0f, bodyWidth - textInsetX - typeWidth - 6.0f * dpi);
    std::string clipped = fullText;
    if (textSize.x > textMaxWidth && textMaxWidth > 0.0f) {
        while (!clipped.empty() && ImGui::CalcTextSize((clipped + "…").c_str()).x > textMaxWidth) {
            size_t utf8End = clipped.size() - 1;
            while (utf8End > 0 && (static_cast<unsigned char>(clipped[utf8End]) & 0xc0) == 0x80)
                --utf8End;
            clipped.resize(utf8End);
        }
        clipped += "…";
    }
    drawList->AddText(ImVec2(start.x + textInsetX, start.y + (fieldHeight - textSize.y) * 0.5f),
                      ImGui::GetColorU32(ImGuiCol_Text), clipped.c_str());
    if (showType) {
        PushMonoFont();
        drawList->AddText(
            ImVec2(start.x + bodyWidth - typeSize.x - 6.0f * dpi, start.y + (fieldHeight - typeSize.y) * 0.5f),
            ImGui::GetColorU32(ImGuiCol_TextDisabled), typeHint.c_str());
        PopFont();
    }

    if (hasPicker) {
        const ImVec4 &pickerFill =
            pickerPressed ? buttonActiveColor : (pickerHovered ? buttonHoverColor : buttonIdleColor);
        drawList->AddRectFilled(pickerMin, pickerMax, ImGui::ColorConvertFloat4ToU32(pickerFill), 0.0f);
        const float drawSize =
            (std::max)(0.0f, (std::min)(10.0f * dpi, (std::min)(pickerMax.x - pickerMin.x - 6.0f * dpi,
                                                                pickerMax.y - pickerMin.y - 4.0f * dpi)));
        const ImVec2 drawMin{pickerMin.x + ((pickerMax.x - pickerMin.x) - drawSize) * 0.5f,
                             pickerMin.y + ((pickerMax.y - pickerMin.y) - drawSize) * 0.5f};
        if (pickerTextureId != 0 && drawSize > 0.0f) {
            drawList->AddImage(ToImTextureID(pickerTextureId), drawMin,
                               ImVec2(drawMin.x + drawSize, drawMin.y + drawSize));
        }
    }

    ImGui::PopID();
    return result;
}

std::vector<ObjectFieldInteraction> InxGUIContext::RenderMeshRendererInspectorFields(
    const std::string &meshFieldId, const std::string &meshLabel, const std::string &meshDisplay,
    const std::vector<std::string> &slotLabels, const std::vector<std::string> &slotDisplays, uint64_t pickerTextureId,
    float labelWidth, const std::string &materialsLabel)
{
    if (slotLabels.size() != slotDisplays.size())
        throw std::invalid_argument("Material slot label/display counts do not match.");

    const auto &colors = EditorThemeRegistry::Colors();
    const auto &floats = EditorThemeRegistry::Floats();
    const auto outlineIt = colors.find("DND_DROP_OUTLINE");
    const ImVec4 dropOutline = outlineIt != colors.end() ? outlineIt->second : ImVec4(1, 1, 1, 0.85f);
    const auto thicknessIt = floats.find("DND_DROP_OUTLINE_THICKNESS");
    const float dropOutlineThickness = thicknessIt != floats.end() ? thicknessIt->second : 1.5f;
    std::vector<ObjectFieldInteraction> interactions;

    auto label = [&](const std::string &text) { InspectorFieldLabel(text, labelWidth); };

    auto renderField = [&](int index, const std::string &fieldId, const std::string &fieldLabel,
                           const std::string &display, const std::string &typeHint,
                           const std::vector<std::string> &acceptTypes) {
        label(fieldLabel);
        ObjectFieldInteraction interaction;
        interaction.index = index;
        interaction.flags = RenderObjectFieldChrome(fieldId, display, typeHint, false, true, true, pickerTextureId);

        ImGui::PushStyleColor(ImGuiCol_DragDropTarget, ImVec4(0, 0, 0, 0));
        if (ImGui::BeginDragDropTarget()) {
            const ImVec2 min = ImGui::GetItemRectMin();
            const ImVec2 max = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddRect(min, max, ImGui::ColorConvertFloat4ToU32(dropOutline), 0.0f, 0,
                                                dropOutlineThickness);
            for (const auto &acceptType : acceptTypes) {
                if (AcceptDragDropPayload(acceptType, &interaction.payload)) {
                    interaction.payloadType = acceptType;
                    break;
                }
            }
            ImGui::EndDragDropTarget();
        }
        ImGui::PopStyleColor();

        ImGui::PushID(fieldId.c_str());
        interaction.popupOpen = ImGui::IsPopupOpen("##obj_picker");
        ImGui::PopID();
        if (interaction.flags != 0 || interaction.popupOpen || !interaction.payloadType.empty())
            interactions.push_back(std::move(interaction));
    };

    renderField(-1, meshFieldId, meshLabel, meshDisplay, "Mesh", {"MODEL_GUID", "MODEL_FILE"});
    ImGui::Spacing();
    PushMonoFont();
    ImGui::TextColored(EditorThemeRegistry::Color("TEXT_DISABLED"), "%s", materialsLabel.c_str());
    ImGui::SameLine();
    ImGui::TextColored(EditorThemeRegistry::Color("TEXT_DISABLED"), "%zu", slotLabels.size());
    PopFont();
    ImGui::Separator();
    for (size_t index = 0; index < slotLabels.size(); ++index) {
        renderField(static_cast<int>(index), "mat_" + std::to_string(index), slotLabels[index], slotDisplays[index],
                    "Material", {"MATERIAL_FILE"});
    }
    return interactions;
}

} // namespace infernux
