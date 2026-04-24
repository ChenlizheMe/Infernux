#include "InxGUIContext.h"
#include "InxTextLayout.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace infernux
{

float InxGUIContext::s_dpiScale = 1.0f;

namespace
{
ImTextureID ToImTextureID(uint64_t textureId)
{
    if constexpr (std::is_pointer_v<ImTextureID>) {
        return (ImTextureID)(static_cast<uintptr_t>(textureId));
    }
    return static_cast<ImTextureID>(textureId);
}

float ResolveFontSize(float fontSize)
{
    return textlayout::ResolveFontSize(fontSize);
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
    if (clicked && onClick)
        onClick();
    return clicked;
}

bool InxGUIContext::RadioButton(const std::string &label, bool active)
{
    return ImGui::RadioButton(label.c_str(), active);
}

bool InxGUIContext::Selectable(const std::string &label, bool selected, int flags, float width, float height)
{
    return ImGui::Selectable(label.c_str(), selected, flags, ImVec2(width, height));
}

/* value editors */
void InxGUIContext::Checkbox(const std::string &label, bool *value)
{
    ImGui::Checkbox(label.c_str(), value);
}

void InxGUIContext::IntSlider(const std::string &label, int *value, int min, int max)
{
    ImGui::SliderInt(label.c_str(), value, min, max);
}

void InxGUIContext::FloatSlider(const std::string &label, float *value, float min, float max)
{
    ImGui::SliderFloat(label.c_str(), value, min, max);
}

bool InxGUIContext::DragFloat(const std::string &label, float *value, float speed, float min, float max,
                              const char *fmt, float power)
{
    CompensateWarp();
    bool changed = ImGui::DragFloat(label.c_str(), value, speed, min, max, fmt, power);
    HandleDragCapture();
    return changed;
}

bool InxGUIContext::DragInt(const std::string &label, int *value, float speed, int min, int max, const char *fmt)
{
    CompensateWarp();
    bool changed = ImGui::DragInt(label.c_str(), value, speed, min, max, fmt);
    HandleDragCapture();
    return changed;
}

void InxGUIContext::TextInput(const std::string &label, char *buffer, size_t bufferSize)
{
    ImGui::InputText(label.c_str(), buffer, bufferSize);
}

void InxGUIContext::TextArea(const std::string &label, char *buffer, size_t bufferSize)
{
    ImGui::InputTextMultiline(label.c_str(), buffer, bufferSize, ImVec2(-FLT_MIN, 100));
}

bool InxGUIContext::InputTextWithHint(const std::string &label, const std::string &hint, char *buffer,
                                      size_t bufferSize, int flags)
{
    return ImGui::InputTextWithHint(label.c_str(), hint.c_str(), buffer, bufferSize, flags);
}

bool InxGUIContext::InputInt(const std::string &label, int *value, int step, int stepFast, int flags)
{
    return ImGui::InputInt(label.c_str(), value, step, stepFast, flags);
}

bool InxGUIContext::InputFloat(const std::string &label, float *value, float step, float stepFast, int flags)
{
    return ImGui::InputFloat(label.c_str(), value, step, stepFast, "%.3f", static_cast<ImGuiInputTextFlags>(flags));
}
void InxGUIContext::ColorEdit(const std::string &label, float color[4])
{
    ImGui::ColorEdit4(label.c_str(), color);
}

bool InxGUIContext::ColorPicker(const std::string &label, float color[4], int flags)
{
    return ImGui::ColorPicker4(label.c_str(), color, static_cast<ImGuiColorEditFlags>(flags));
}

// Unity-style helper: label on the left, DragFloatN on the right
static void LabeledDragFloatN(const char *label, float *value, int components, float speed, float labelWidth = 0.0f)
{
    if (labelWidth <= 0.0f)
        labelWidth = ImGui::CalcTextSize(label).x + 20.0f;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(labelWidth);
    float avail = ImGui::GetContentRegionAvail().x;
    ImGui::SetNextItemWidth(avail);
    std::string hiddenLabel = std::string("##") + label;
    switch (components) {
    case 2:
        ImGui::DragFloat2(hiddenLabel.c_str(), value, speed);
        break;
    case 3:
        ImGui::DragFloat3(hiddenLabel.c_str(), value, speed);
        break;
    case 4:
        ImGui::DragFloat4(hiddenLabel.c_str(), value, speed);
        break;
    default:
        ImGui::DragFloat(hiddenLabel.c_str(), value, speed);
        break;
    }
}

void InxGUIContext::Vector2Control(const std::string &label, float value[2], float speed, float labelWidth)
{
    CompensateWarp();
    LabeledDragFloatN(label.c_str(), value, 2, speed, labelWidth);
    HandleDragCapture();
}

void InxGUIContext::Vector3Control(const std::string &label, float value[3], float speed, float labelWidth)
{
    CompensateWarp();
    LabeledDragFloatN(label.c_str(), value, 3, speed, labelWidth);
    HandleDragCapture();
}

void InxGUIContext::Vector4Control(const std::string &label, float value[4], float speed, float labelWidth)
{
    CompensateWarp();
    LabeledDragFloatN(label.c_str(), value, 4, speed, labelWidth);
    HandleDragCapture();
}

/* combo & lists */
bool InxGUIContext::Combo(const std::string &label, int *currentItem, const std::vector<std::string> &items,
                          int popupMaxHeightInItems)
{
    std::vector<const char *> cstrs;
    cstrs.reserve(items.size());
    for (const auto &s : items)
        cstrs.push_back(s.c_str());
    return ImGui::Combo(label.c_str(), currentItem, cstrs.data(), static_cast<int>(cstrs.size()),
                        popupMaxHeightInItems);
}

bool InxGUIContext::ListBox(const std::string &label, int *currentItem, const std::vector<std::string> &items,
                            int heightInItems)
{
    std::vector<const char *> cstrs;
    cstrs.reserve(items.size());
    for (const auto &s : items)
        cstrs.push_back(s.c_str());
    return ImGui::ListBox(label.c_str(), currentItem, cstrs.data(), static_cast<int>(cstrs.size()), heightInItems);
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
    return ImGui::TreeNode(label.c_str());
}

bool InxGUIContext::TreeNodeEx(const std::string &label, int flags)
{
    return ImGui::TreeNodeEx(label.c_str(), static_cast<ImGuiTreeNodeFlags>(flags));
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
    return ImGui::CollapsingHeader(label.c_str());
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

bool InxGUIContext::BeginTabItem(const std::string &label, bool *open)
{
    return ImGui::BeginTabItem(label.c_str(), open);
}

void InxGUIContext::EndTabItem()
{
    ImGui::EndTabItem();
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

bool InxGUIContext::BeginMenu(const std::string &label, bool enabled)
{
    return ImGui::BeginMenu(label.c_str(), enabled);
}

void InxGUIContext::EndMenu()
{
    ImGui::EndMenu();
}

bool InxGUIContext::MenuItem(const std::string &label, const std::string &shortcut, bool selected, bool enabled)
{
    return ImGui::MenuItem(label.c_str(), shortcut.empty() ? nullptr : shortcut.c_str(), selected, enabled);
}

/* child & windows */
bool InxGUIContext::BeginChild(const std::string &id, float width, float height, bool border)
{
    return ImGui::BeginChild(id.c_str(), ImVec2(width, height), border);
}

void InxGUIContext::EndChild()
{
    ImGui::EndChild();
}

/* popups & tooltips */
void InxGUIContext::OpenPopup(const std::string &id)
{
    ImGui::OpenPopup(id.c_str());
}

bool InxGUIContext::BeginPopup(const std::string &id)
{
    return ImGui::BeginPopup(id.c_str());
}

bool InxGUIContext::BeginPopupModal(const std::string &title, int flags)
{
    return ImGui::BeginPopupModal(title.c_str(), nullptr, static_cast<ImGuiWindowFlags>(flags));
}

bool InxGUIContext::BeginPopupContextItem(const std::string &id, int mouseButton)
{
    return ImGui::BeginPopupContextItem(id.empty() ? nullptr : id.c_str(), static_cast<ImGuiPopupFlags>(mouseButton));
}

bool InxGUIContext::BeginPopupContextWindow(const std::string &id, int mouseButton)
{
    return ImGui::BeginPopupContextWindow(id.empty() ? nullptr : id.c_str(), static_cast<ImGuiPopupFlags>(mouseButton));
}

void InxGUIContext::EndPopup()
{
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
    return ImGui::ImageButton(id.c_str(), reinterpret_cast<ImTextureID>(textureId), ImVec2(width, height),
                              ImVec2(uv0_x, uv0_y), ImVec2(uv1_x, uv1_y));
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
    return ImGui::CheckboxFlags(label.c_str(), flags, flagValue);
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
    return ImGui::Begin(name.c_str(), open, flags);
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

void InxGUIContext::SetCursorPosX(float x)
{
    ImGui::SetCursorPosX(x);
}

void InxGUIContext::SetCursorPosY(float y)
{
    ImGui::SetCursorPosY(y);
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

/* invisible button (for splitter) */
bool InxGUIContext::InvisibleButton(const std::string &id, float width, float height)
{
    return ImGui::InvisibleButton(id.c_str(), ImVec2(width, height));
}

bool InxGUIContext::IsItemActive()
{
    return ImGui::IsItemActive();
}

bool InxGUIContext::IsAnyItemActive()
{
    return ImGui::IsAnyItemActive();
}

bool InxGUIContext::IsItemHovered()
{
    return ImGui::IsItemHovered();
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
bool InxGUIContext::BeginDragDropSource(int flags)
{
    return ImGui::BeginDragDropSource(static_cast<ImGuiDragDropFlags>(flags));
}

bool InxGUIContext::SetDragDropPayload(const std::string &type, uint64_t data)
{
    return ImGui::SetDragDropPayload(type.c_str(), &data, sizeof(data));
}

bool InxGUIContext::SetDragDropPayload(const std::string &type, const std::string &data)
{
    return ImGui::SetDragDropPayload(type.c_str(), data.c_str(), data.size() + 1);
}

void InxGUIContext::EndDragDropSource()
{
    ImGui::EndDragDropSource();
}

bool InxGUIContext::BeginDragDropTarget()
{
    return ImGui::BeginDragDropTarget();
}

bool InxGUIContext::AcceptDragDropPayload(const std::string &type, uint64_t *outData)
{
    const ImGuiPayload *payload = ImGui::AcceptDragDropPayload(type.c_str());
    if (payload && payload->DataSize == sizeof(uint64_t)) {
        *outData = *static_cast<const uint64_t *>(payload->Data);
        return true;
    }
    return false;
}

bool InxGUIContext::AcceptDragDropPayload(const std::string &type, std::string *outData)
{
    const ImGuiPayload *payload = ImGui::AcceptDragDropPayload(type.c_str());
    if (payload && payload->DataSize > 0) {
        *outData = std::string(static_cast<const char *>(payload->Data), payload->DataSize - 1);
        return true;
    }
    return false;
}

bool InxGUIContext::AcceptAnyDragDropPayload(std::string *outType, uint64_t *outU64, std::string *outStr, bool *outIsU64)
{
    const ImGuiPayload *preview = ImGui::GetDragDropPayload();
    if (!preview || preview->DataType[0] == '\0')
        return false;
    const ImGuiPayload *acc = ImGui::AcceptDragDropPayload(preview->DataType);
    if (!acc)
        return false;
    outType->assign(preview->DataType);
    if (acc->DataSize == sizeof(uint64_t)) {
        *outIsU64 = true;
        *outU64 = *reinterpret_cast<const uint64_t *>(acc->Data);
        return true;
    }
    *outIsU64 = false;
    if (acc->DataSize > 0) {
        *outStr = std::string(static_cast<const char *>(acc->Data), acc->DataSize - 1);
        return true;
    }
    return false;
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

void InxGUIContext::InputTextMultiline(const std::string &label, const std::string &text, float width, float height,
                                       int flags)
{
    // We use a temporary buffer so ImGui can render selection / scrolling,
    // but the content is effectively read-only (ImGuiInputTextFlags_ReadOnly = 1 << 14).
    std::vector<char> buf(text.size() + 1, 0);
    std::copy(text.begin(), text.end(), buf.begin());
    ImGui::InputTextMultiline(label.c_str(), buf.data(), buf.size(), ImVec2(width, height), flags);
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

void InxGUIContext::DrawText(float x, float y, const std::string &text, float r, float g, float b, float a,
                             float fontSize)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (!drawList)
        return;
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, "", ResolveFontSize(fontSize), 0.0f, 1.0f, 0.0f});
    if (!layout.lines.empty()) {
        drawList->PushTextureID(ImGui::GetIO().Fonts->TexRef);
        textlayout::RenderLine(drawList, layout, layout.lines.front(), x, y, col, 0.0f);
        drawList->PopTextureID();
    }
}

void InxGUIContext::DrawTextAligned(float minX, float minY, float maxX, float maxY, const std::string &text, float r,
                                    float g, float b, float a, float alignX, float alignY, float fontSize, bool clip)
{
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (!drawList)
        return;

    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, "", ResolveFontSize(fontSize), 0.0f, 1.0f, 0.0f});

    float boxW = maxX - minX;
    float boxH = maxY - minY;

    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));

    if (clip)
        drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);

    drawList->PushTextureID(ImGui::GetIO().Fonts->TexRef);
    textlayout::RenderTextBox(drawList, minX, minY, maxX, maxY, layout, col, alignX, alignY, 0.0f);
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

    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, "", ResolveFontSize(fontSize), 0.0f, 1.0f, 0.0f});
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
                              0.0f, 0.0f, 0.0f);
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
                                      const std::string &fontPath, float lineHeight, float letterSpacing)
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
        textlayout::LayoutText({text, fontPath, ResolveFontSize(fontSize), wrapWidth, lineHeight, letterSpacing});
    const ImVec2 textSize(layout.totalWidth, layout.totalHeight);

    if (std::fabs(rotation) < 0.001f && !mirrorH && !mirrorV) {
        if (clip)
            drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);
        const ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
        drawList->PushTextureID(ImGui::GetIO().Fonts->TexRef);
        textlayout::RenderTextBox(drawList, minX, minY, maxX, maxY, layout, col, alignX, alignY, letterSpacing);
        drawList->PopTextureID();
        if (clip)
            drawList->PopClipRect();
        return;
    }

    float boxW = maxX - minX;
    float boxH = maxY - minY;
    const ImVec2 pivot((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);

    if (clip)
        drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);

    const int vtxStart = drawList->VtxBuffer.Size;
    ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a));
    drawList->PushTextureID(ImGui::GetIO().Fonts->TexRef);
    textlayout::RenderTextBox(drawList, minX, minY, maxX, maxY, layout, col, alignX, alignY, letterSpacing);
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

std::pair<float, float> InxGUIContext::CalcTextSizeA(const std::string &text, float fontSize,
                                                     const std::string &fontPath, float lineHeight, float letterSpacing)
{
    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, fontPath, ResolveFontSize(fontSize), 0.0f, lineHeight, letterSpacing});
    return {layout.totalWidth, layout.totalHeight};
}

std::pair<float, float> InxGUIContext::CalcTextSizeWrappedA(const std::string &text, float fontSize, float wrapWidth,
                                                            const std::string &fontPath, float lineHeight,
                                                            float letterSpacing)
{
    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, fontPath, ResolveFontSize(fontSize), wrapWidth, lineHeight, letterSpacing});
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
                                                               float labelWidth)
{
    std::vector<PropertyChange> changes;

    // Checkbox styling constants (match EditorTheme)
    constexpr ImVec2 kCheckboxPad{4.0f, 2.0f};
    constexpr float kCheckboxFontScale = 1.0f;
    constexpr float kMinLabelWidth = 156.0f;

    auto doLabel = [&](const std::string &label) {
        if (!label.empty()) {
            float w = labelWidth;
            if (w <= 0.0f)
                w = std::max(ImGui::CalcTextSize(label.c_str()).x + 18.0f, kMinLabelWidth);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(label.c_str());
            ImGui::SameLine(w);
            ImGui::SetNextItemWidth(-1);
        } else {
            ImGui::SetNextItemWidth(-1);
        }
    };

    const int count = static_cast<int>(descriptors.size());
    for (int i = 0; i < count; ++i) {
        const auto &d = descriptors[i];

        // Layout: header / space
        if (!d.header.empty()) {
            ImGui::Spacing();
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(d.header.c_str());
        }
        if (d.space > 0)
            ImGui::Dummy(ImVec2(0, d.space));

        switch (d.type) {
        case PropertyDesc::Float: {
            doLabel(d.label);
            float val = d.fVal[0];
            float orig = val;
            CompensateWarp();
            if (d.slider && d.rangeMin > -1e5f)
                ImGui::SliderFloat(d.widgetId.c_str(), &val, d.rangeMin, d.rangeMax);
            else
                ImGui::DragFloat(d.widgetId.c_str(), &val, d.speed, d.rangeMin, d.rangeMax);
            HandleDragCapture();
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
            int val = d.iVal;
            int orig = val;
            CompensateWarp();
            if (d.slider && static_cast<int>(d.rangeMin) > -1000000)
                ImGui::SliderInt(d.widgetId.c_str(), &val, static_cast<int>(d.rangeMin), static_cast<int>(d.rangeMax));
            else
                ImGui::DragInt(d.widgetId.c_str(), &val, d.speed, static_cast<int>(d.rangeMin),
                               static_cast<int>(d.rangeMax));
            HandleDragCapture();
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
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, kCheckboxPad);
            ImGui::SetWindowFontScale(kCheckboxFontScale);
            bool val = d.bVal;
            bool orig = val;
            std::string cbLabel = !d.label.empty() ? d.label : d.widgetId;
            ImGui::Checkbox(cbLabel.c_str(), &val);
            ImGui::SetWindowFontScale(1.0f);
            ImGui::PopStyleVar(1);
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
            char buf[4096];
            size_t len = std::min(d.sVal.size(), sizeof(buf) - 1);
            std::memcpy(buf, d.sVal.c_str(), len);
            buf[len] = '\0';
            if (d.multiline)
                ImGui::InputTextMultiline(d.widgetId.c_str(), buf, sizeof(buf), ImVec2(-1, 80));
            else
                ImGui::InputText(d.widgetId.c_str(), buf, 256);
            std::string newStr(buf);
            if (newStr != d.sVal) {
                PropertyChange c;
                c.index = i;
                c.type = PropertyDesc::String;
                c.sVal = std::move(newStr);
                changes.push_back(c);
            }
            break;
        }
        case PropertyDesc::Vec2: {
            float v[2] = {d.fVal[0], d.fVal[1]};
            float lw = d.label.empty() ? 1.0f : labelWidth;
            Vector2Control(d.label.empty() ? " " : d.label, v, d.speed, lw);
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
            Vector3Control(d.label.empty() ? " " : d.label, v, d.speed, lw);
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
            Vector4Control(d.label.empty() ? " " : d.label, v, d.speed, lw);
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
            int idx = d.iVal;
            int orig = idx;
            std::vector<const char *> cstrs;
            cstrs.reserve(d.enumNames.size());
            for (const auto &s : d.enumNames)
                cstrs.push_back(s.c_str());
            ImGui::Combo(d.widgetId.c_str(), &idx, cstrs.data(), static_cast<int>(cstrs.size()));
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
            break;
        }
        } // switch

        // Tooltip for the last rendered widget (label hover is not tracked).
        if (!d.tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", d.tooltip.c_str());
    }
    return changes;
}

} // namespace infernux
