#include "gui/InfGUIContext.h"
#include "gui/InfGUIRenderable.h"
#include "gui/InfResourcePreviewer.h"
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infengine
{
class PyGUIRenderable : public InfGUIRenderable
{
  public:
    using InfGUIRenderable::InfGUIRenderable;

    void OnRender(InfGUIContext *ctx) override
    {
        PYBIND11_OVERRIDE_NAME(void, InfGUIRenderable, "on_render", OnRender, ctx);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
//  Dedicated UI helper — render Position/Rotation/Scale in one bridge call
// ═══════════════════════════════════════════════════════════════════════════

namespace
{

// Render 3 Vector3 controls (Position, Rotation, Scale) and return all 9
// floats in a single flat tuple.  This avoids the per-call pybind11
// dispatch overhead that a generic batch would incur.
py::tuple RenderTransformFields(InfGUIContext &ctx, float px, float py, float pz, float rx, float ry, float rz,
                                float sx, float sy, float sz, float speedPos, float speedRot, float speedScl,
                                float labelWidth)
{
    float pos[3] = {px, py, pz};
    float rot[3] = {rx, ry, rz};
    float scl[3] = {sx, sy, sz};
    ctx.Vector3Control("Position", pos, speedPos, labelWidth);
    ctx.Vector3Control("Rotation", rot, speedRot, labelWidth);
    ctx.Vector3Control("Scale", scl, speedScl, labelWidth);
    return py::make_tuple(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], scl[0], scl[1], scl[2]);
}

} // anonymous namespace

void RegisterGUIBindings(py::module_ &m)
{
    py::class_<InfGUIContext>(m, "InfGUIContext")
        .def("label", &InfGUIContext::Label)
        .def("button", &InfGUIContext::Button, py::arg("label"), py::arg("on_click") = py::none(),
             py::arg("width") = 0.0f, py::arg("height") = 0.0f,
             "Button widget. width=-1 fills available width, height=0 uses default.")
        .def("radio_button", &InfGUIContext::RadioButton)
        .def("selectable", &InfGUIContext::Selectable, py::arg("label"), py::arg("selected") = false,
             py::arg("flags") = 0, py::arg("width") = 0.0f, py::arg("height") = 0.0f)
        .def("checkbox",
             [](InfGUIContext &ctx, const std::string &label, bool value) {
                 ctx.Checkbox(label, &value);
                 return value;
             })
        .def("int_slider",
             [](InfGUIContext &ctx, const std::string &label, int value, int min, int max) {
                 ctx.IntSlider(label, &value, min, max);
                 return value;
             })
        .def("float_slider",
             [](InfGUIContext &ctx, const std::string &label, float value, float min, float max) {
                 ctx.FloatSlider(label, &value, min, max);
                 return value;
             })
        .def("drag_int",
             [](InfGUIContext &ctx, const std::string &label, int value, float speed, int min, int max) {
                 ctx.DragInt(label, &value, speed, min, max);
                 return value;
             })
        .def("drag_float",
             [](InfGUIContext &ctx, const std::string &label, float value, float speed, float min, float max) {
                 ctx.DragFloat(label, &value, speed, min, max);
                 return value;
             })
        .def("text_input",
             [](InfGUIContext &ctx, const std::string &label, const std::string &value, size_t buffer_size) {
                 std::vector<char> buffer(buffer_size, 0);
                 if (value.size() < buffer_size) {
                     std::copy(value.begin(), value.end(), buffer.begin());
                 } else {
                     std::copy(value.begin(), value.begin() + buffer_size - 1, buffer.begin());
                 }
                 ctx.TextInput(label, buffer.data(), buffer_size);
                 return std::string(buffer.data());
             })
        .def("text_area", &InfGUIContext::TextArea)
        .def(
            "input_text_with_hint",
            [](InfGUIContext &ctx, const std::string &label, const std::string &hint, const std::string &value,
               size_t buffer_size, int flags) {
                std::vector<char> buffer(buffer_size, 0);
                if (value.size() < buffer_size) {
                    std::copy(value.begin(), value.end(), buffer.begin());
                } else {
                    std::copy(value.begin(), value.begin() + buffer_size - 1, buffer.begin());
                }
                ctx.InputTextWithHint(label, hint, buffer.data(), buffer_size, flags);
                return std::string(buffer.data());
            },
            py::arg("label"), py::arg("hint"), py::arg("value"), py::arg("buffer_size") = 256, py::arg("flags") = 0)
        .def(
            "input_int",
            [](InfGUIContext &ctx, const std::string &label, int value, int step, int step_fast, int flags) {
                ctx.InputInt(label, &value, step, step_fast, flags);
                return value;
            },
            py::arg("label"), py::arg("value"), py::arg("step") = 1, py::arg("step_fast") = 100, py::arg("flags") = 0)
        .def(
            "input_float",
            [](InfGUIContext &ctx, const std::string &label, float value, float step, float step_fast, int flags) {
                ctx.InputFloat(label, &value, step, step_fast, flags);
                return value;
            },
            py::arg("label"), py::arg("value"), py::arg("step") = 0.0f, py::arg("step_fast") = 0.0f,
            py::arg("flags") = 0)
        .def(
            "color_edit",
            [](InfGUIContext &ctx, const std::string &label, float r, float g, float b, float a) -> py::tuple {
                float color[4] = {r, g, b, a};
                ctx.ColorEdit(label, color);
                return py::make_tuple(py::float_(color[0]), py::float_(color[1]), py::float_(color[2]),
                                      py::float_(color[3]));
            },
            py::arg("label"), py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a") = 1.0f)
        .def(
            "color_picker",
            [](InfGUIContext &ctx, const std::string &label, float r, float g, float b, float a,
               int flags) -> py::tuple {
                float color[4] = {r, g, b, a};
                bool changed = ctx.ColorPicker(label, color, flags);
                return py::make_tuple(py::bool_(changed), py::float_(color[0]), py::float_(color[1]),
                                      py::float_(color[2]), py::float_(color[3]));
            },
            py::arg("label"), py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a") = 1.0f, py::arg("flags") = 0)
        .def(
            "vector2",
            [](InfGUIContext &ctx, const std::string &label, float x, float y, float speed,
               float labelWidth) -> py::tuple {
                float value[2] = {x, y};
                ctx.Vector2Control(label, value, speed, labelWidth);
                return py::make_tuple(py::float_(value[0]), py::float_(value[1]));
            },
            py::arg("label"), py::arg("x"), py::arg("y"), py::arg("speed") = 0.1f, py::arg("label_width") = 0.0f)
        .def(
            "vector3",
            [](InfGUIContext &ctx, const std::string &label, float x, float y, float z, float speed,
               float labelWidth) -> py::tuple {
                float value[3] = {x, y, z};
                ctx.Vector3Control(label, value, speed, labelWidth);
                return py::make_tuple(py::float_(value[0]), py::float_(value[1]), py::float_(value[2]));
            },
            py::arg("label"), py::arg("x"), py::arg("y"), py::arg("z"), py::arg("speed") = 0.1f,
            py::arg("label_width") = 0.0f)
        .def(
            "vector4",
            [](InfGUIContext &ctx, const std::string &label, float x, float y, float z, float w, float speed,
               float labelWidth) -> py::tuple {
                float value[4] = {x, y, z, w};
                ctx.Vector4Control(label, value, speed, labelWidth);
                return py::make_tuple(py::float_(value[0]), py::float_(value[1]), py::float_(value[2]),
                                      py::float_(value[3]));
            },
            py::arg("label"), py::arg("x"), py::arg("y"), py::arg("z"), py::arg("w"), py::arg("speed") = 0.1f,
            py::arg("label_width") = 0.0f)
        .def(
            "combo",
            [](InfGUIContext &ctx, const std::string &label, int currentItem, const std::vector<std::string> &items,
               int popupMaxHeightInItems) {
                ctx.Combo(label, &currentItem, items, popupMaxHeightInItems);
                return currentItem;
            },
            py::arg("label"), py::arg("current_item"), py::arg("items"), py::arg("popup_max_height_in_items") = -1)
        .def(
            "list_box",
            [](InfGUIContext &ctx, const std::string &label, int currentItem, const std::vector<std::string> &items,
               int heightInItems) {
                ctx.ListBox(label, &currentItem, items, heightInItems);
                return currentItem;
            },
            py::arg("label"), py::arg("current_item"), py::arg("items"), py::arg("height_in_items") = -1)
        .def("progress_bar", &InfGUIContext::ProgressBar)
        .def("begin_group", &InfGUIContext::BeginGroup, py::arg("name") = "")
        .def("end_group", &InfGUIContext::EndGroup)
        .def("same_line", &InfGUIContext::SameLine, py::arg("offset_from_start_x") = 0.0f, py::arg("spacing") = -1.0f)
        .def("align_text_to_frame_padding", &InfGUIContext::AlignTextToFramePadding,
             "Vertically align upcoming text baseline to FramePadding.y so it aligns with framed widgets")
        .def("set_scroll_here_y", &InfGUIContext::SetScrollHereY, py::arg("center_y_ratio") = 0.5f,
             "Adjust scrolling amount to make current cursor position visible. 0.0=top, 0.5=center, 1.0=bottom")
        .def("get_scroll_y", &InfGUIContext::GetScrollY, "Get current scroll Y position")
        .def("get_scroll_max_y", &InfGUIContext::GetScrollMaxY, "Get maximum scroll Y value")
        .def("separator", &InfGUIContext::Separator)
        .def("spacing", &InfGUIContext::Spacing)
        .def("dummy", &InfGUIContext::Dummy)
        .def("new_line", &InfGUIContext::NewLine)
        .def("tree_node", &InfGUIContext::TreeNode)
        .def("tree_node_ex", &InfGUIContext::TreeNodeEx, py::arg("label"), py::arg("flags"),
             "Create tree node with flags (ImGuiTreeNodeFlags)")
        .def("tree_pop", &InfGUIContext::TreePop)
        .def("set_next_item_open", &InfGUIContext::SetNextItemOpen, py::arg("is_open"), py::arg("cond") = 0)
        .def("set_next_item_allow_overlap", &InfGUIContext::SetNextItemAllowOverlap,
             "Allow the next item to be overlapped by a subsequent item (e.g. checkbox over CollapsingHeader).")
        .def("collapsing_header", &InfGUIContext::CollapsingHeader)
        .def("is_item_clicked", &InfGUIContext::IsItemClicked, py::arg("mouse_button") = 0)
        .def("begin_tab_bar", &InfGUIContext::BeginTabBar)
        .def("end_tab_bar", &InfGUIContext::EndTabBar)
        .def("begin_tab_item", &InfGUIContext::BeginTabItem)
        .def("end_tab_item", &InfGUIContext::EndTabItem)
        .def("begin_main_menu_bar", &InfGUIContext::BeginMainMenuBar)
        .def("end_main_menu_bar", &InfGUIContext::EndMainMenuBar)
        .def("begin_menu", &InfGUIContext::BeginMenu, py::arg("label"), py::arg("enabled") = true)
        .def("end_menu", &InfGUIContext::EndMenu)
        .def("menu_item", &InfGUIContext::MenuItem)
        .def("begin_child", &InfGUIContext::BeginChild)
        .def("end_child", &InfGUIContext::EndChild)
        .def("open_popup", &InfGUIContext::OpenPopup)
        .def("begin_popup", &InfGUIContext::BeginPopup)
        .def("begin_popup_modal", &InfGUIContext::BeginPopupModal, py::arg("title"), py::arg("flags") = 0,
             "Open a modal popup. Returns true while open. Must call end_popup() when true.")
        .def("begin_popup_context_item", &InfGUIContext::BeginPopupContextItem, py::arg("id") = "",
             py::arg("mouse_button") = 1, "Open context popup on right-click of last item")
        .def("begin_popup_context_window", &InfGUIContext::BeginPopupContextWindow, py::arg("id") = "",
             py::arg("mouse_button") = 1, "Open context popup on right-click anywhere in window")
        .def("end_popup", &InfGUIContext::EndPopup)
        .def("close_current_popup", &InfGUIContext::CloseCurrentPopup, "Close current popup")
        .def("begin_tooltip", &InfGUIContext::BeginTooltip)
        .def("end_tooltip", &InfGUIContext::EndTooltip)
        .def("set_tooltip", &InfGUIContext::SetTooltip)
        .def(
            "image",
            [](InfGUIContext &ctx, uint64_t textureId, float width, float height, float uv0_x, float uv0_y, float uv1_x,
               float uv1_y) {
                ctx.Image(reinterpret_cast<void *>(textureId), width, height, uv0_x, uv0_y, uv1_x, uv1_y);
            },
            py::arg("texture_id"), py::arg("width"), py::arg("height"), py::arg("uv0_x") = 0.0f,
            py::arg("uv0_y") = 0.0f, py::arg("uv1_x") = 1.0f, py::arg("uv1_y") = 1.0f)
        .def(
            "image_button",
            [](InfGUIContext &ctx, const std::string &id, uint64_t textureId, float width, float height, float uv0_x,
               float uv0_y, float uv1_x, float uv1_y) {
                return ctx.ImageButton(id, reinterpret_cast<void *>(textureId), width, height, uv0_x, uv0_y, uv1_x,
                                       uv1_y);
            },
            py::arg("id"), py::arg("texture_id"), py::arg("width"), py::arg("height"), py::arg("uv0_x") = 0.0f,
            py::arg("uv0_y") = 0.0f, py::arg("uv1_x") = 1.0f, py::arg("uv1_y") = 1.0f)
        .def("begin_table", &InfGUIContext::BeginTable)
        .def("end_table", &InfGUIContext::EndTable)
        .def("table_setup_column", &InfGUIContext::TableSetupColumn)
        .def("table_headers_row", &InfGUIContext::TableHeadersRow)
        .def("table_next_row", &InfGUIContext::TableNextRow)
        .def("table_set_column_index", &InfGUIContext::TableSetColumnIndex)
        .def("table_next_column", &InfGUIContext::TableNextColumn)
        .def("checkbox_flags", &InfGUIContext::CheckboxFlags)
        .def("set_next_item_width", &InfGUIContext::SetNextItemWidth)
        .def("set_next_window_size", &InfGUIContext::SetNextWindowSize)
        .def("set_next_window_pos", &InfGUIContext::SetNextWindowPos)
        .def("set_next_window_focus", &InfGUIContext::SetNextWindowFocus)
        .def("begin_window", &InfGUIContext::BeginWindow)
        // begin_window_closable returns tuple (is_visible, is_open) for closable windows
        .def(
            "begin_window_closable",
            [](InfGUIContext &ctx, const std::string &name, bool is_open, int flags) -> std::tuple<bool, bool> {
                bool open = is_open;
                bool visible = ctx.BeginWindow(name, &open, flags);
                return std::make_tuple(visible, open);
            },
            py::arg("name"), py::arg("is_open") = true, py::arg("flags") = 0,
            "Begin a closable window. Returns (is_visible, is_open). "
            "When user clicks close button, is_open becomes False.")
        .def("end_window", &InfGUIContext::EndWindow)
        // Layout query methods
        .def("calc_text_width", &InfGUIContext::CalcTextWidth, py::arg("text"),
             "Calculate the pixel width of the given text string")
        .def("get_content_region_avail_width", &InfGUIContext::GetContentRegionAvailWidth)
        .def("get_content_region_avail_height", &InfGUIContext::GetContentRegionAvailHeight)
        .def("get_cursor_pos_x", &InfGUIContext::GetCursorPosX)
        .def("get_cursor_pos_y", &InfGUIContext::GetCursorPosY)
        .def("set_cursor_pos_x", &InfGUIContext::SetCursorPosX)
        .def("set_cursor_pos_y", &InfGUIContext::SetCursorPosY)
        .def("get_window_pos_x", &InfGUIContext::GetWindowPosX)
        .def("get_window_pos_y", &InfGUIContext::GetWindowPosY)
        .def("get_window_width", &InfGUIContext::GetWindowWidth)
        .def("get_item_rect_min_x", &InfGUIContext::GetItemRectMinX)
        .def("get_item_rect_min_y", &InfGUIContext::GetItemRectMinY)
        .def("get_item_rect_max_x", &InfGUIContext::GetItemRectMaxX)
        .def("get_item_rect_max_y", &InfGUIContext::GetItemRectMaxY)
        // Splitter helper methods
        .def("invisible_button", &InfGUIContext::InvisibleButton)
        .def("is_item_active", &InfGUIContext::IsItemActive)
        .def("is_any_item_active", &InfGUIContext::IsAnyItemActive)
        .def("is_item_hovered", &InfGUIContext::IsItemHovered)
        .def("set_keyboard_focus_here", &InfGUIContext::SetKeyboardFocusHere, py::arg("offset") = 0,
             "Set keyboard focus to the next item (or previous with negative offset)")
        .def("is_item_deactivated", &InfGUIContext::IsItemDeactivated,
             "Check if the last item was deactivated (focused -> unfocused)")
        .def("is_item_deactivated_after_edit", &InfGUIContext::IsItemDeactivatedAfterEdit,
             "Check if the last item was deactivated and value was modified")
        .def("get_mouse_drag_delta_y", &InfGUIContext::GetMouseDragDeltaY, py::arg("button") = 0)
        .def("reset_mouse_drag_delta", &InfGUIContext::ResetMouseDragDelta, py::arg("button") = 0)
        // ID stack for unique widget IDs
        .def("push_id", py::overload_cast<int>(&InfGUIContext::PushID), py::arg("id"),
             "Push integer ID onto the ID stack")
        .def("push_id_str", py::overload_cast<const std::string &>(&InfGUIContext::PushID), py::arg("id"),
             "Push string ID onto the ID stack")
        .def("pop_id", &InfGUIContext::PopID, "Pop from the ID stack") // Style
        .def("push_style_color", &InfGUIContext::PushStyleColor, py::arg("idx"), py::arg("r"), py::arg("g"),
             py::arg("b"), py::arg("a"), "Push style color (ImGuiCol enum value)")
        .def("pop_style_color", &InfGUIContext::PopStyleColor, py::arg("count") = 1, "Pop style color")
        .def("push_style_var_float", &InfGUIContext::PushStyleVarFloat, py::arg("idx"), py::arg("val"),
             "Push style var (float) by ImGuiStyleVar enum value")
        .def("push_style_var_vec2", &InfGUIContext::PushStyleVarVec2, py::arg("idx"), py::arg("x"), py::arg("y"),
             "Push style var (ImVec2) by ImGuiStyleVar enum value")
        .def("pop_style_var", &InfGUIContext::PopStyleVar, py::arg("count") = 1, "Pop style var")
        .def("begin_disabled", &InfGUIContext::BeginDisabled, py::arg("disabled") = true,
             "Begin a disabled section (grayed out, no interaction)")
        .def("end_disabled", &InfGUIContext::EndDisabled, "End disabled section") // Drag and Drop
        .def("begin_drag_drop_source", &InfGUIContext::BeginDragDropSource, py::arg("flags") = 0,
             "Begin a drag source on the last item")
        .def("set_drag_drop_payload",
             py::overload_cast<const std::string &, uint64_t>(&InfGUIContext::SetDragDropPayload), py::arg("type"),
             py::arg("data"), "Set drag-drop payload (uint64 data)")
        .def("set_drag_drop_payload_str",
             py::overload_cast<const std::string &, const std::string &>(&InfGUIContext::SetDragDropPayload),
             py::arg("type"), py::arg("data"), "Set drag-drop payload (string data)")
        .def("end_drag_drop_source", &InfGUIContext::EndDragDropSource, "End drag source")
        .def("begin_drag_drop_target", &InfGUIContext::BeginDragDropTarget, "Begin a drag-drop target on last item")
        .def(
            "accept_drag_drop_payload",
            [](InfGUIContext &ctx, const std::string &type) -> py::
                                                                object {
                                                                    // Try uint64_t first
                                                                    uint64_t data_int = 0;
                                                                    if (ctx.AcceptDragDropPayload(type, &data_int)) {
                                                                        return py::cast(data_int);
                                                                    }
                                                                    // Try string
                                                                    std::string data_str;
                                                                    if (ctx.AcceptDragDropPayload(type, &data_str)) {
                                                                        return py::cast(data_str);
                                                                    }
                                                                    return py::none();
                                                                },
            py::arg("type"), "Accept drag-drop payload, returns data (int or str) or None")
        .def("end_drag_drop_target", &InfGUIContext::EndDragDropTarget, "End drag-drop target")
        // Mouse cursor
        .def("set_mouse_cursor", &InfGUIContext::SetMouseCursor, py::arg("cursor_type"),
             "Set mouse cursor: 0=Arrow, 1=TextInput, 2=ResizeAll, 3=ResizeNS, 4=ResizeEW, 5=ResizeNESW, 6=ResizeNWSE, "
             "7=Hand, 8=NotAllowed")
        // ========================================================================
        // Scene View Input API - Unity-style camera controls
        // ========================================================================
        // Mouse state
        .def("is_mouse_button_down", &InfGUIContext::IsMouseButtonDown, py::arg("button"),
             "Check if mouse button is held down (0=left, 1=right, 2=middle)")
        .def("is_mouse_button_clicked", &InfGUIContext::IsMouseButtonClicked, py::arg("button"),
             "Check if mouse button was clicked this frame")
        .def("is_mouse_double_clicked", &InfGUIContext::IsMouseDoubleClicked, py::arg("button") = 0,
             "Check if mouse button was double-clicked this frame")
        .def("is_mouse_dragging", &InfGUIContext::IsMouseDragging, py::arg("button"), py::arg("lock_threshold") = -1.0f,
             "Check if mouse is being dragged")
        .def("get_mouse_drag_delta_x", &InfGUIContext::GetMouseDragDeltaX, py::arg("button") = 0,
             "Get horizontal mouse drag delta")
        .def("get_mouse_pos_x", &InfGUIContext::GetMousePosX, "Get current mouse X position")
        .def("get_mouse_pos_y", &InfGUIContext::GetMousePosY, "Get current mouse Y position")
        .def("get_mouse_wheel_delta", &InfGUIContext::GetMouseWheelDelta, "Get mouse wheel scroll delta")
        // Keyboard state
        .def("is_key_down", &InfGUIContext::IsKeyDown, py::arg("key_code"),
             "Check if key is held down (ImGuiKey enum values)")
        .def("is_key_pressed", &InfGUIContext::IsKeyPressed, py::arg("key_code"), "Check if key was pressed this frame")
        .def("is_key_released", &InfGUIContext::IsKeyReleased, py::arg("key_code"),
             "Check if key was released this frame")
        // Window focus state
        .def("is_window_focused", &InfGUIContext::IsWindowFocused, py::arg("flags") = 0,
             "Check if current window is focused")
        .def("is_window_hovered", &InfGUIContext::IsWindowHovered, py::arg("flags") = 0,
             "Check if mouse is over current window")
        .def("want_text_input", &InfGUIContext::WantTextInput,
             "Returns true when ImGui wants keyboard input (e.g. text field is active)")
        // Input capture
        .def("capture_mouse_from_app", &InfGUIContext::CaptureMouseFromApp, py::arg("capture"),
             "Capture mouse input from application")
        .def("capture_keyboard_from_app", &InfGUIContext::CaptureKeyboardFromApp, py::arg("capture"),
             "Capture keyboard input from application")
        // Mouse warp / global pos for Unity-style screen-edge wrapping
        .def("warp_mouse_global", &InfGUIContext::WarpMouseGlobal, py::arg("x"), py::arg("y"),
             "Warp mouse cursor to global screen coordinates")
        .def("get_global_mouse_pos_x", &InfGUIContext::GetGlobalMousePosX, "Get global (screen) mouse X coordinate")
        .def("get_global_mouse_pos_y", &InfGUIContext::GetGlobalMousePosY, "Get global (screen) mouse Y coordinate")
        .def(
            "get_main_viewport_bounds",
            [](InfGUIContext &ctx) -> py::
                                       tuple {
                                           float x = 0, y = 0, w = 0, h = 0;
                                           ctx.GetMainViewportBounds(&x, &y, &w, &h);
                                           return py::make_tuple(py::float_(x), py::float_(y), py::float_(w),
                                                                 py::float_(h));
                                       },
            "Returns (x, y, width, height) of the main ImGui viewport (app window client area)")
        .def("set_clipboard_text", &InfGUIContext::SetClipboardText, py::arg("text"), "Set the system clipboard text")
        .def("get_clipboard_text", &InfGUIContext::GetClipboardText, "Get the system clipboard text")
        .def(
            "input_text_multiline",
            [](InfGUIContext &ctx, const std::string &label, const std::string &value, size_t buffer_size, float width,
               float height, int flags) {
                std::vector<char> buffer(buffer_size, 0);
                size_t copyLen = std::min(value.size(), buffer_size - 1);
                std::copy(value.begin(), value.begin() + copyLen, buffer.begin());
                ImGui::InputTextMultiline(label.c_str(), buffer.data(), buffer.size(), ImVec2(width, height), flags);
                return std::string(buffer.data());
            },
            py::arg("label"), py::arg("text"), py::arg("buffer_size") = 4096, py::arg("width") = -1.0f,
            py::arg("height") = -1.0f, py::arg("flags") = 0,
            "Editable multiline text input. Returns the (possibly modified) text.")
        .def("draw_rect", &InfGUIContext::DrawRect, py::arg("min_x"), py::arg("min_y"), py::arg("max_x"),
             py::arg("max_y"), py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"), py::arg("thickness") = 1.0f,
             py::arg("rounding") = 0.0f, "Draw a rectangle outline on the current window's draw list (screen coords)")
        .def("draw_filled_rect", &InfGUIContext::DrawFilledRect, py::arg("min_x"), py::arg("min_y"), py::arg("max_x"),
             py::arg("max_y"), py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"), py::arg("rounding") = 0.0f,
             "Draw a filled rectangle on the current window's draw list (screen coords)")
        .def("draw_filled_rect_rotated", &InfGUIContext::DrawFilledRectRotated, py::arg("min_x"), py::arg("min_y"),
             py::arg("max_x"), py::arg("max_y"), py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"),
             py::arg("rotation") = 0.0f, py::arg("mirror_h") = false, py::arg("mirror_v") = false,
             py::arg("rounding") = 0.0f,
             "Draw a filled rectangle with rotation/mirror on the current window's draw list")
        .def("draw_line", &InfGUIContext::DrawLine, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"), py::arg("thickness") = 1.0f,
             "Draw a line on the current window's draw list (screen coords)")
        .def("draw_circle", &InfGUIContext::DrawCircle, py::arg("center_x"), py::arg("center_y"), py::arg("radius"),
             py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"), py::arg("thickness") = 1.0f,
             py::arg("segments") = 0, "Draw a circle outline on the current window's draw list (screen coords)")
        .def("draw_filled_circle", &InfGUIContext::DrawFilledCircle, py::arg("center_x"), py::arg("center_y"),
             py::arg("radius"), py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"), py::arg("segments") = 0,
             "Draw a filled circle on the current window's draw list (screen coords)")
        .def("draw_image_rect", &InfGUIContext::DrawImageRect, py::arg("texture_id"), py::arg("min_x"),
             py::arg("min_y"), py::arg("max_x"), py::arg("max_y"), py::arg("uv0_x") = 0.0f, py::arg("uv0_y") = 0.0f,
             py::arg("uv1_x") = 1.0f, py::arg("uv1_y") = 1.0f, py::arg("tint_r") = 1.0f, py::arg("tint_g") = 1.0f,
             py::arg("tint_b") = 1.0f, py::arg("tint_a") = 1.0f, py::arg("rotation") = 0.0f,
             py::arg("mirror_h") = false, py::arg("mirror_v") = false, py::arg("rounding") = 0.0f,
             "Draw an image quad in absolute screen coordinates with optional rotation, mirroring and rounding")
        .def("set_window_font_scale", &InfGUIContext::SetWindowFontScale, py::arg("scale"),
             "Set font scale for the current window (1.0 = default)")
        .def("get_dpi_scale", &InfGUIContext::GetDpiScale,
             "Get the OS display scale factor (e.g. 2.0 for 200% scaling)")
        .def("draw_text", &InfGUIContext::DrawText, py::arg("x"), py::arg("y"), py::arg("text"), py::arg("r"),
             py::arg("g"), py::arg("b"), py::arg("a"), py::arg("font_size") = 0.0f,
             "Draw text at absolute screen coordinates with colour and optional font size")
        .def("draw_text_aligned", &InfGUIContext::DrawTextAligned, py::arg("min_x"), py::arg("min_y"), py::arg("max_x"),
             py::arg("max_y"), py::arg("text"), py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"),
             py::arg("align_x") = 0.0f, py::arg("align_y") = 0.0f, py::arg("font_size") = 0.0f, py::arg("clip") = false,
             "Draw aligned text within a bounding box (align 0=left/top, 0.5=center, 1=right/bottom)")
        .def("draw_text_rotated_90_aligned", &InfGUIContext::DrawTextRotated90Aligned, py::arg("min_x"),
             py::arg("min_y"), py::arg("max_x"), py::arg("max_y"), py::arg("text"), py::arg("r"), py::arg("g"),
             py::arg("b"), py::arg("a"), py::arg("align_x") = 0.0f, py::arg("align_y") = 0.0f,
             py::arg("font_size") = 0.0f, py::arg("clockwise") = false, py::arg("clip") = false,
             "Draw text rotated by 90 degrees inside a bounding box")
        .def("draw_text_ex_aligned", &InfGUIContext::DrawTextExAligned, py::arg("min_x"), py::arg("min_y"),
             py::arg("max_x"), py::arg("max_y"), py::arg("text"), py::arg("r"), py::arg("g"), py::arg("b"),
             py::arg("a"), py::arg("align_x") = 0.0f, py::arg("align_y") = 0.0f, py::arg("font_size") = 0.0f,
             py::arg("wrap_width") = 0.0f, py::arg("rotation") = 0.0f, py::arg("mirror_h") = false,
             py::arg("mirror_v") = false, py::arg("clip") = false, py::arg("font_path") = std::string(),
             py::arg("line_height") = 1.0f, py::arg("letter_spacing") = 0.0f,
             "Draw aligned text with arbitrary rotation (degrees) and optional horizontal/vertical mirror")
        .def(
            "calc_text_size",
            [](InfGUIContext &ctx, const std::string &text, float fontSize, const std::string &fontPath,
               float lineHeight, float letterSpacing) -> py::tuple {
                auto [w, h] = ctx.CalcTextSizeA(text, fontSize, fontPath, lineHeight, letterSpacing);
                return py::make_tuple(py::float_(w), py::float_(h));
            },
            py::arg("text"), py::arg("font_size") = 0.0f, py::arg("font_path") = std::string(),
            py::arg("line_height") = 1.0f, py::arg("letter_spacing") = 0.0f,
            "Calculate pixel size of text at given font size. Returns (width, height).")
        .def(
            "calc_text_size_wrapped",
            [](InfGUIContext &ctx, const std::string &text, float fontSize, float wrapWidth,
               const std::string &fontPath, float lineHeight, float letterSpacing) -> py::tuple {
                auto [w, h] = ctx.CalcTextSizeWrappedA(text, fontSize, wrapWidth, fontPath, lineHeight, letterSpacing);
                return py::make_tuple(py::float_(w), py::float_(h));
            },
            py::arg("text"), py::arg("font_size") = 0.0f, py::arg("wrap_width") = 0.0f,
            py::arg("font_path") = std::string(), py::arg("line_height") = 1.0f, py::arg("letter_spacing") = 0.0f,
            "Calculate wrapped pixel size of text at given font size. Returns (width, height).")
        .def("push_draw_list_clip_rect", &InfGUIContext::PushDrawListClipRect, py::arg("min_x"), py::arg("min_y"),
             py::arg("max_x"), py::arg("max_y"), py::arg("intersect_with_current") = true,
             "Push a clip rect onto the draw list for subsequent draw calls")
        .def("pop_draw_list_clip_rect", &InfGUIContext::PopDrawListClipRect,
             "Pop the last clip rect from the draw list")
        .def(
            "get_display_bounds",
            [](InfGUIContext &ctx) -> py::
                                       tuple {
                                           float x, y, w, h;
                                           ctx.GetDisplayBounds(&x, &y, &w, &h);
                                           return py::make_tuple(py::float_(x), py::float_(y), py::float_(w),
                                                                 py::float_(h));
                                       },
            "Get primary display bounds as (x, y, width, height)")
        // ── Dedicated Transform renderer (1 call for 3 vector3) ────
        .def(
            "render_transform_fields",
            [](InfGUIContext &ctx, float px, float py, float pz, float rx, float ry, float rz, float sx, float sy,
               float sz, float speedPos, float speedRot, float speedScl, float labelWidth) {
                return RenderTransformFields(ctx, px, py, pz, rx, ry, rz, sx, sy, sz, speedPos, speedRot, speedScl,
                                             labelWidth);
            },
            py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("rx"), py::arg("ry"), py::arg("rz"), py::arg("sx"),
            py::arg("sy"), py::arg("sz"), py::arg("speed_pos"), py::arg("speed_rot"), py::arg("speed_scl"),
            py::arg("label_width"), "Render Position/Rotation/Scale vector3 controls in one call.");

    py::class_<InfGUIRenderable, PyGUIRenderable, std::shared_ptr<InfGUIRenderable>>(m, "InfGUIRenderable",
                                                                                     py::dynamic_attr())
        .def(py::init<>());

    // ResourcePreviewManager - manages resource previewers for Inspector
    py::class_<ResourcePreviewManager>(m, "ResourcePreviewManager")
        .def("has_previewer", &ResourcePreviewManager::HasPreviewer, py::arg("extension"),
             "Check if there's a previewer for the given file extension")
        .def("get_previewer_type_name", &ResourcePreviewManager::GetPreviewerTypeName, py::arg("extension"),
             "Get the previewer type name for a file extension")
        .def("get_all_supported_extensions", &ResourcePreviewManager::GetAllSupportedExtensions,
             "Get all supported extensions")
        .def("load_preview", &ResourcePreviewManager::LoadPreview, py::arg("file_path"), "Load a file for preview")
        .def("render_preview", &ResourcePreviewManager::RenderPreview, py::arg("ctx"), py::arg("avail_width"),
             py::arg("avail_height"), "Render the current preview")
        .def("render_metadata", &ResourcePreviewManager::RenderMetadata, py::arg("ctx"),
             "Render metadata for the current preview")
        .def("unload_preview", &ResourcePreviewManager::UnloadPreview, "Unload the current preview")
        .def("is_preview_loaded", &ResourcePreviewManager::IsPreviewLoaded, "Check if a preview is currently loaded")
        .def("get_loaded_path", &ResourcePreviewManager::GetLoadedPath, "Get the currently loaded file path")
        .def("get_current_type_name", &ResourcePreviewManager::GetCurrentTypeName,
             "Get the current previewer type name")
        .def("set_preview_settings", &ResourcePreviewManager::SetPreviewSettings, py::arg("display_mode"),
             py::arg("max_size"), py::arg("srgb"), "Set preview settings (display mode, max size, sRGB)");
}

} // namespace infengine