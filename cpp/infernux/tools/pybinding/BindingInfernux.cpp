#include "Infernux.h"
// Explicit includes for types now only forward-declared in InxRenderer.h
#include <SDL3/SDL.h>
#include <cmath>
#include <core/config/EngineConfig.h>
#include <core/log/InxLog.h>
#include <function/renderer/EditorTools.h>
#include <function/renderer/GizmosDrawCallBuffer.h>
#include <function/renderer/SceneRenderGraph.h>
#include <function/renderer/ScriptableRenderContext.h>
#include <function/renderer/gui/InxGUIContext.h>
#include <function/renderer/gui/InxGUIRenderable.h>
#include <function/renderer/gui/InxResourcePreviewer.h>
#include <function/renderer/gui/InxScreenUIRenderer.h>
#include <function/scene/EditorCameraController.h>
#include <glm/glm.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace infernux;
namespace py = pybind11;

namespace infernux
{
void RegisterGUIBindings(py::module_ &m);
void RegisterVector2Bindings(py::module_ &m);
void RegisterVector3Bindings(py::module_ &m);
void RegisterVec4fBindings(py::module_ &m);
void RegisterResourceBindings(py::module_ &m);
void RegisterSceneBindings(py::module_ &m);
void RegisterAssetDatabaseBindings(py::module_ &m);
void RegisterAssetRegistryBindings(py::module_ &m);
void RegisterRenderGraphBindings(py::module_ &m);
void RegisterRenderPipelineBindings(py::module_ &m);
void RegisterCommandBufferBindings(py::module_ &m);
void RegisterTagLayerBindings(py::module_ &m);
void RegisterInputBindings(py::module_ &m);
void RegisterPhysicsBindings(py::module_ &m);
void RegisterAudioBindings(py::module_ &m);
void RegisterBatchBindings(py::module_ &m);
} // namespace infernux

PYBIND11_MODULE(_Infernux, m)
{
    m.doc() = "Python bindings for Infernux";

    // ---- Editor gizmo handle IDs (exposed so Python can identify gizmo picks) ----
    m.attr("GIZMO_X_AXIS_ID") = EditorTools::X_AXIS_ID;
    m.attr("GIZMO_Y_AXIS_ID") = EditorTools::Y_AXIS_ID;
    m.attr("GIZMO_Z_AXIS_ID") = EditorTools::Z_AXIS_ID;
    m.attr("GIZMO_XY_PLANE_ID") = EditorTools::XY_PLANE_ID;
    m.attr("GIZMO_XZ_PLANE_ID") = EditorTools::XZ_PLANE_ID;
    m.attr("GIZMO_YZ_PLANE_ID") = EditorTools::YZ_PLANE_ID;

    py::enum_<LogLevel>(m, "LogLevel")
        .value("Debug", LogLevel::LOG_DEBUG)
        .value("Info", LogLevel::LOG_INFO)
        .value("Warn", LogLevel::LOG_WARN)
        .value("Error", LogLevel::LOG_ERROR)
        .value("Fatal", LogLevel::LOG_FATAL)
        .export_values();

    // ---- EngineConfig (centralised runtime configuration) ----
    py::class_<EngineConfig>(m, "EngineConfig",
                             "Centralised engine configuration singleton.\n"
                             "Modify values BEFORE the corresponding subsystem initializes.\n"
                             "Access via EngineConfig.get().")
        .def_static("get", &EngineConfig::Get, py::return_value_policy::reference,
                    "Get the singleton EngineConfig instance.")
        // Rendering — Descriptor Pools
        .def_readwrite("max_materials_per_pool", &EngineConfig::maxMaterialsPerPool)
        .def_readwrite("ubo_descriptors_per_material", &EngineConfig::uboDescriptorsPerMaterial)
        .def_readwrite("sampler_descriptors_per_material", &EngineConfig::samplerDescriptorsPerMaterial)
        .def_readwrite("fullscreen_descriptor_sets_per_frame", &EngineConfig::fullscreenDescriptorSetsPerFrame)
        .def_readwrite("fullscreen_sampler_descriptors_per_frame", &EngineConfig::fullscreenSamplerDescriptorsPerFrame)
        // Rendering — Textures
        .def_readwrite("enable_mipmap", &EngineConfig::enableMipmap)
        .def_readwrite("anisotropy_scale", &EngineConfig::anisotropyScale)
        // Rendering — Swapchain
        .def_readwrite("preferred_swapchain_image_count", &EngineConfig::preferredSwapchainImageCount)
        .def_readwrite("max_frames_in_flight", &EngineConfig::maxFramesInFlight)
        // Physics — Jolt Configuration
        .def_readwrite("physics_temp_allocator_size", &EngineConfig::physicsTempAllocatorSize)
        .def_readwrite("physics_max_jobs", &EngineConfig::physicsMaxJobs)
        .def_readwrite("physics_max_barriers", &EngineConfig::physicsMaxBarriers)
        .def_readwrite("physics_max_bodies", &EngineConfig::physicsMaxBodies)
        .def_readwrite("physics_max_body_pairs", &EngineConfig::physicsMaxBodyPairs)
        .def_readwrite("physics_max_contact_constraints", &EngineConfig::physicsMaxContactConstraints)
        .def_readwrite("physics_collision_steps", &EngineConfig::physicsCollisionSteps)
        .def_property(
            "physics_gravity", [](EngineConfig &self) { return self.physicsGravity; },
            [](EngineConfig &self, const glm::vec3 &v) { self.physicsGravity = v; },
            "Default gravity vector (applied on physics init)")
        .def_readwrite("physics_max_worker_threads", &EngineConfig::physicsMaxWorkerThreads)
        // Physics — Default Collider Properties
        .def_readwrite("default_collider_friction", &EngineConfig::defaultColliderFriction)
        .def_readwrite("default_collider_bounciness", &EngineConfig::defaultColliderBounciness)
        // Physics — Default Rigidbody Properties
        .def_readwrite("default_rigidbody_mass", &EngineConfig::defaultRigidbodyMass)
        .def_readwrite("default_rigidbody_drag", &EngineConfig::defaultRigidbodyDrag)
        .def_readwrite("default_rigidbody_angular_drag", &EngineConfig::defaultRigidbodyAngularDrag)
        .def_readwrite("default_max_angular_velocity", &EngineConfig::defaultMaxAngularVelocity)
        .def_readwrite("default_max_linear_velocity", &EngineConfig::defaultMaxLinearVelocity)
        .def_readwrite("default_max_depenetration_velocity", &EngineConfig::defaultMaxDepenetrationVelocity)
        // Physics — Layers
        .def_readwrite("physics_layer_count", &EngineConfig::physicsLayerCount)
        .def_readwrite("default_query_layer_mask", &EngineConfig::defaultQueryLayerMask)
        // Render Queue Ranges (read-only from Python; change via code if needed)
        .def_readwrite("opaque_queue_min", &EngineConfig::opaqueQueueMin)
        .def_readwrite("opaque_queue_max", &EngineConfig::opaqueQueueMax)
        .def_readwrite("transparent_queue_min", &EngineConfig::transparentQueueMin)
        .def_readwrite("transparent_queue_max", &EngineConfig::transparentQueueMax)
        .def_readwrite("shadow_caster_queue_min", &EngineConfig::shadowCasterQueueMin)
        .def_readwrite("shadow_caster_queue_max", &EngineConfig::shadowCasterQueueMax)
        .def_readwrite("component_gizmo_queue_min", &EngineConfig::componentGizmoQueueMin)
        .def_readwrite("component_gizmo_queue_max", &EngineConfig::componentGizmoQueueMax)
        .def_readwrite("editor_gizmo_queue_min", &EngineConfig::editorGizmoQueueMin)
        .def_readwrite("editor_gizmo_queue_max", &EngineConfig::editorGizmoQueueMax)
        .def_readwrite("editor_tools_queue_min", &EngineConfig::editorToolsQueueMin)
        .def_readwrite("editor_tools_queue_max", &EngineConfig::editorToolsQueueMax)
        .def_readwrite("skybox_queue", &EngineConfig::skyboxQueue);

    // ---- EditorCamera (property-based camera access) ----
    py::class_<EditorCameraController>(m, "EditorCamera",
                                       "Editor camera controller with property-based access.\n"
                                       "Access via engine.editor_camera.")
        .def_property(
            "fov",
            [](EditorCameraController &self) -> float {
                auto *cam = self.GetCamera();
                return cam ? cam->GetFieldOfView() : 60.0f;
            },
            [](EditorCameraController &self, float v) {
                auto *cam = self.GetCamera();
                if (cam)
                    cam->SetFieldOfView(v);
            },
            "Vertical field of view in degrees")
        .def_property(
            "near_clip",
            [](EditorCameraController &self) -> float {
                auto *cam = self.GetCamera();
                return cam ? cam->GetNearClip() : 0.01f;
            },
            [](EditorCameraController &self, float v) {
                auto *cam = self.GetCamera();
                if (cam)
                    cam->SetNearClip(v);
            },
            "Near clipping distance")
        .def_property(
            "far_clip",
            [](EditorCameraController &self) -> float {
                auto *cam = self.GetCamera();
                return cam ? cam->GetFarClip() : 1000.0f;
            },
            [](EditorCameraController &self, float v) {
                auto *cam = self.GetCamera();
                if (cam)
                    cam->SetFarClip(v);
            },
            "Far clipping distance")
        .def_property_readonly(
            "position",
            [](EditorCameraController &self) -> glm::vec3 {
                auto *cam = self.GetCamera();
                if (cam && cam->GetGameObject()) {
                    return cam->GetGameObject()->GetTransform()->GetPosition();
                }
                return glm::vec3(0.0f);
            },
            "Camera position as Vector3")
        .def_property_readonly(
            "rotation",
            [](EditorCameraController &self) -> py::tuple { return py::make_tuple(self.GetYaw(), self.GetPitch()); },
            "Camera rotation as (yaw, pitch) tuple")
        .def("reset", &EditorCameraController::Reset, "Reset camera to default position and orientation")
        .def(
            "focus_on",
            [](EditorCameraController &self, float x, float y, float z, float distance) {
                self.FocusOn(glm::vec3(x, y, z), distance);
            },
            py::arg("x"), py::arg("y"), py::arg("z"), py::arg("distance") = 10.0f,
            "Focus camera on a world-space point")
        .def_readwrite("rotation_speed", &EditorCameraController::rotationSpeed, "Mouse rotation sensitivity")
        .def_readwrite("pan_speed", &EditorCameraController::panSpeed, "Middle-mouse pan sensitivity")
        .def_readwrite("zoom_speed", &EditorCameraController::zoomSpeed, "Scroll wheel zoom sensitivity")
        .def_readwrite("move_speed", &EditorCameraController::moveSpeed, "WASD movement speed")
        .def_readwrite("move_speed_boost", &EditorCameraController::moveSpeedBoost, "Shift speed multiplier")
        .def_property_readonly(
            "focus_point", [](EditorCameraController &self) -> glm::vec3 { return self.GetFocusPoint(); },
            "Camera focus/orbit point as Vector3")
        .def_property_readonly(
            "focus_distance", [](EditorCameraController &self) -> float { return self.GetFocusDistance(); },
            "Distance from the camera to the focus/orbit point")
        .def(
            "restore_state",
            [](EditorCameraController &self, float pos_x, float pos_y, float pos_z, float focus_x, float focus_y,
               float focus_z, float focus_dist, float yaw, float pitch) {
                self.RestoreState(glm::vec3(pos_x, pos_y, pos_z), glm::vec3(focus_x, focus_y, focus_z), focus_dist, yaw,
                                  pitch);
            },
            py::arg("pos_x"), py::arg("pos_y"), py::arg("pos_z"), py::arg("focus_x"), py::arg("focus_y"),
            py::arg("focus_z"), py::arg("focus_dist"), py::arg("yaw"), py::arg("pitch"),
            "Restore full camera state (position, focus, orientation)")
        .def(
            "world_to_screen_point",
            [](EditorCameraController &self, float x, float y, float z) -> glm::vec2 {
                auto *camera = self.GetCamera();
                if (!camera)
                    return glm::vec2(0.0f);
                return camera->WorldToScreenPoint(glm::vec3(x, y, z));
            },
            py::arg("x"), py::arg("y"), py::arg("z"),
            "Project world position into current Scene View render target coordinates");

    // ========================================================================
    // ScreenUIList enum and InxScreenUIRenderer bindings
    // ========================================================================
    py::enum_<ScreenUIList>(m, "ScreenUIList")
        .value("Camera", ScreenUIList::Camera)
        .value("Overlay", ScreenUIList::Overlay);

    py::class_<InxScreenUIRenderer>(m, "InxScreenUIRenderer")
        .def("begin_frame", &InxScreenUIRenderer::BeginFrame, py::arg("width"), py::arg("height"),
             "Reset draw lists for a new frame")
        .def("add_filled_rect", &InxScreenUIRenderer::AddFilledRect, py::arg("list"), py::arg("min_x"),
             py::arg("min_y"), py::arg("max_x"), py::arg("max_y"), py::arg("r") = 1.0f, py::arg("g") = 1.0f,
             py::arg("b") = 1.0f, py::arg("a") = 1.0f, py::arg("rounding") = 0.0f,
             "Add a filled rectangle to the specified draw list")
        .def("add_image", &InxScreenUIRenderer::AddImage, py::arg("list"), py::arg("texture_id"), py::arg("min_x"),
             py::arg("min_y"), py::arg("max_x"), py::arg("max_y"), py::arg("uv0_x") = 0.0f, py::arg("uv0_y") = 0.0f,
             py::arg("uv1_x") = 1.0f, py::arg("uv1_y") = 1.0f, py::arg("r") = 1.0f, py::arg("g") = 1.0f,
             py::arg("b") = 1.0f, py::arg("a") = 1.0f, py::arg("rotation") = 0.0f, py::arg("mirror_h") = false,
             py::arg("mirror_v") = false, py::arg("rounding") = 0.0f,
             "Add a textured image quad to the specified draw list with optional rotation, mirroring and rounding")
        .def("add_text", &InxScreenUIRenderer::AddText, py::arg("list"), py::arg("min_x"), py::arg("min_y"),
             py::arg("max_x"), py::arg("max_y"), py::arg("text"), py::arg("r") = 1.0f, py::arg("g") = 1.0f,
             py::arg("b") = 1.0f, py::arg("a") = 1.0f, py::arg("align_x") = 0.5f, py::arg("align_y") = 0.5f,
             py::arg("font_size") = 0.0f, py::arg("wrap_width") = 0.0f, py::arg("rotation") = 0.0f,
             py::arg("mirror_h") = false, py::arg("mirror_v") = false, py::arg("font_path") = std::string(),
             py::arg("line_height") = 1.0f, py::arg("letter_spacing") = 0.0f,
             "Add text within a bounding box to the specified draw list with optional rotation and mirroring")
        .def(
            "measure_text",
            [](const InxScreenUIRenderer &renderer, const std::string &text, float font_size, float wrap_width,
               const std::string &font_path, float line_height, float letter_spacing) -> py::tuple {
                auto [w, h] = renderer.MeasureText(text, font_size, wrap_width, font_path, line_height, letter_spacing);
                return py::make_tuple(py::float_(w), py::float_(h));
            },
            py::arg("text"), py::arg("font_size") = 0.0f, py::arg("wrap_width") = 0.0f,
            py::arg("font_path") = std::string(), py::arg("line_height") = 1.0f, py::arg("letter_spacing") = 0.0f,
            "Measure text size using the active UI font. Returns (width, height).")
        .def("has_commands", &InxScreenUIRenderer::HasCommands, py::arg("list"),
             "Check if the specified draw list has any draw commands")
        .def("set_enabled", &InxScreenUIRenderer::SetEnabled, py::arg("enabled"),
             "Enable or disable rendering (commands still accumulate)")
        .def("is_enabled", &InxScreenUIRenderer::IsEnabled, "Check if rendering is enabled");

    py::class_<Infernux>(m, "Infernux")
        .def(py::init<std::string>(), py::arg("dll_path"))
        .def("init_renderer", &Infernux::InitRenderer, py::arg("width"), py::arg("height"), py::arg("project_path"),
             py::arg("builtin_resource_path") = std::string())
        .def(
            "set_gui_font",
            [](Infernux &self, const std::string &fontPath, float fontSize) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetGUIFont(fontPath.c_str(), fontSize);
            },
            py::arg("font_path"), py::arg("font_size"))
        .def(
            "get_display_scale",
            [](Infernux &self) -> float {
                auto *r = self.GetRenderer();
                return r ? r->GetDisplayScale() : 1.0f;
            },
            "Get the OS display scale factor (e.g. 2.0 for 200%% scaling)")
        .def("run", &Infernux::Run)
        .def(
            "set_pre_gui_callback",
            [](Infernux &self, py::object callback) {
                auto *r = self.GetRenderer();
                if (!r)
                    return;
                if (callback.is_none()) {
                    r->SetPreGuiCallback(nullptr);
                } else {
                    // GIL is already held during DrawFrame (Run() keeps it),
                    // so no acquire needed in the callback.
                    py::function fn = py::cast<py::function>(callback);
                    r->SetPreGuiCallback([fn]() {
                        try {
                            fn();
                        } catch (py::error_already_set &e) {
                            e.restore();
                        }
                    });
                }
            },
            py::arg("callback"),
            "Set a Python callback invoked each frame before GUI rendering.\n"
            "Used for DeferredTaskRunner to ensure scene mutations finish before panels render.")
        .def(
            "set_post_draw_callback",
            [](Infernux &self, py::object callback) {
                auto *r = self.GetRenderer();
                if (!r)
                    return;
                if (callback.is_none()) {
                    r->SetPostDrawCallback(nullptr);
                } else {
                    py::function fn = py::cast<py::function>(callback);
                    r->SetPostDrawCallback([fn]() {
                        try {
                            fn();
                        } catch (py::error_already_set &e) {
                            e.restore();
                        }
                    });
                }
            },
            py::arg("callback"),
            "Set a Python callback invoked each frame after GPU submit + present.\n"
            "Heavy scene loads run here, sandwiched by SDL_PumpEvents to prevent\n"
            "Windows from flagging the application as Not Responding.")
        .def(
            "pump_events", [](Infernux & /*self*/) { SDL_PumpEvents(); },
            "Pump the OS message queue to prevent Windows Not Responding during long operations")
        .def("set_log_level", &Infernux::SetLogLevel)
        .def(
            "register_gui_renderable",
            [](Infernux &self, const std::string &name, std::shared_ptr<InxGUIRenderable> renderable) {
                auto *r = self.GetRenderer();
                if (r)
                    r->RegisterGUIRenderable(name.c_str(), renderable);
            },
            py::arg("name"), py::arg("renderable"))
        .def(
            "unregister_gui_renderable",
            [](Infernux &self, const std::string &name) {
                auto *r = self.GetRenderer();
                if (r)
                    r->UnregisterGUIRenderable(name.c_str());
            },
            py::arg("name"))
        .def("select_docked_window", &Infernux::SelectDockedWindow,
             "Select and focus a docked ImGui window by its stable window_id", py::arg("window_id"))
        .def("reset_imgui_layout", &Infernux::ResetImGuiLayout, "Clear ImGui docking layout and delete saved ini")
        .def("exit", &Infernux::Exit, "Exit the Infernux application")
        .def("cleanup", &Infernux::Cleanup, "Destroy renderer and release all GPU resources")
        .def(
            "is_close_requested",
            [](Infernux &self) -> bool {
                auto *r = self.GetRenderer();
                return r && r->IsCloseRequested();
            },
            "True when the user clicked the window close button but Python has not yet confirmed")
        .def(
            "confirm_close",
            [](Infernux &self) {
                auto *r = self.GetRenderer();
                if (r)
                    r->ConfirmClose();
            },
            "Actually close the engine (call after save dialogs are handled)")
        .def(
            "cancel_close",
            [](Infernux &self) {
                auto *r = self.GetRenderer();
                if (r)
                    r->CancelClose();
            },
            "Cancel a pending close request (user chose Cancel in save dialog)")
        .def(
            "show",
            [](Infernux &self) {
                auto *r = self.GetRenderer();
                if (r)
                    r->ShowWindow();
            },
            "Show the Infernux window")
        .def(
            "hide",
            [](Infernux &self) {
                auto *r = self.GetRenderer();
                if (r)
                    r->HideWindow();
            },
            "Hide the Infernux window")
        .def(
            "set_window_icon",
            [](Infernux &self, const std::string &iconPath) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetWindowIcon(iconPath);
            },
            py::arg("icon_path"), "Set the window icon from a PNG file")
        .def(
            "set_fullscreen",
            [](Infernux &self, bool fullscreen) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetWindowFullscreen(fullscreen);
            },
            py::arg("fullscreen"), "Set the window to fullscreen or windowed mode")
        .def(
            "set_window_title",
            [](Infernux &self, const std::string &title) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetWindowTitle(title);
            },
            py::arg("title"), "Set the window title bar text")
        .def(
            "set_maximized",
            [](Infernux &self, bool maximized) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetWindowMaximized(maximized);
            },
            py::arg("maximized"), "Maximize or restore the window")
        .def(
            "set_resizable",
            [](Infernux &self, bool resizable) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetWindowResizable(resizable);
            },
            py::arg("resizable"), "Set whether the window is resizable")
        .def("modify_resources", &Infernux::ModifyResources, py::arg("file_path"))
        .def("delete_resources", &Infernux::DeleteResources, py::arg("file_path"))
        .def("move_resources", &Infernux::MoveResources, py::arg("old_file_path"), py::arg("new_file_path"))
        .def("reload_shader", &Infernux::ReloadShader, py::arg("shader_path"),
             "Reload a shader file and refresh materials using it. Returns empty string on success, error message on "
             "failure.")
        .def("reload_texture", &Infernux::ReloadTexture, py::arg("texture_path"),
             "Invalidate cached texture and force materials to reload it")
        .def("reload_mesh", &Infernux::ReloadMesh, py::arg("mesh_path"),
             "Reload a mesh asset and notify dependent MeshRenderers")
        .def("reload_audio", &Infernux::ReloadAudio, py::arg("audio_path"),
             "Reload an audio clip asset and notify dependents")
        .def("get_asset_database", &Infernux::GetAssetDatabase, py::return_value_policy::reference,
             "Get the asset database instance")
        .def(
            "upload_texture_for_imgui",
            [](Infernux &self, const std::string &name, const std::vector<unsigned char> &pixels, int width, int height,
               bool nearest) -> uint64_t {
                auto *r = self.GetRenderer();
                VkFilter f = nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
                return r ? r->UploadTextureForImGui(name, pixels.data(), width, height, f) : 0;
            },
            py::arg("name"), py::arg("pixels"), py::arg("width"), py::arg("height"), py::arg("nearest") = false,
            "Upload texture data for ImGui display, returns texture ID")
        .def(
            "remove_imgui_texture",
            [](Infernux &self, const std::string &name) {
                auto *r = self.GetRenderer();
                if (r)
                    r->RemoveImGuiTexture(name);
            },
            py::arg("name"), "Remove a previously uploaded ImGui texture")
        .def(
            "has_imgui_texture",
            [](Infernux &self, const std::string &name) -> bool {
                auto *r = self.GetRenderer();
                return r && r->HasImGuiTexture(name);
            },
            py::arg("name"), "Check if an ImGui texture with the given name exists")
        .def(
            "get_imgui_texture_id",
            [](Infernux &self, const std::string &name) -> uint64_t {
                auto *r = self.GetRenderer();
                return r ? r->GetImGuiTextureId(name) : 0;
            },
            py::arg("name"), "Get texture ID for an already uploaded texture")
        .def(
            "get_resource_preview_manager",
            [](Infernux &self) -> ResourcePreviewManager * {
                auto *r = self.GetRenderer();
                return r ? r->GetResourcePreviewManager() : nullptr;
            },
            py::return_value_policy::reference, "Get the resource preview manager for file previews")
        .def(
            "render_material_preview_pixels",
            [](Infernux &self, const std::string &matFilePath, int size) -> py::object {
                std::vector<unsigned char> pixels;
                AssetDatabase *adb = self.GetAssetDatabase();
                InxRenderer *renderer = self.GetRenderer();
                if (!MaterialPreviewer::RenderToPixels(matFilePath, size, pixels, adb, renderer))
                    return py::none();
                return py::cast(std::move(pixels));
            },
            py::arg("mat_file_path"), py::arg("size") = 128,
            "Render a PBR sphere preview for a .mat file (GPU with CPU fallback). Returns list[int] of RGBA pixels, or "
            "None on failure.")
        // ========================================================================
        // Editor Camera (property-based object access — preferred API)
        // ========================================================================
        .def_property_readonly("editor_camera", &Infernux::GetEditorCamera, py::return_value_policy::reference,
                               "Get the editor camera controller (EditorCamera object with property access)")
        // ========================================================================
        // Scene Camera Control API - for Scene View with Unity-style controls
        // ========================================================================
        .def("process_scene_view_input", &Infernux::ProcessSceneViewInput, py::arg("delta_time"),
             py::arg("right_mouse_down"), py::arg("middle_mouse_down"), py::arg("mouse_delta_x"),
             py::arg("mouse_delta_y"), py::arg("scroll_delta"), py::arg("key_w"), py::arg("key_a"), py::arg("key_s"),
             py::arg("key_d"), py::arg("key_q"), py::arg("key_e"), py::arg("key_shift"),
             "Process scene view input for editor camera control")
        // ========================================================================
        // Scene Render Target API - for offscreen scene rendering to ImGui
        // ========================================================================
        .def(
            "get_scene_texture_id",
            [](Infernux &self) -> uint64_t {
                auto *r = self.GetRenderer();
                return r ? r->GetSceneTextureId() : 0;
            },
            "Get scene render target texture ID for ImGui display")
        .def(
            "wait_for_gpu_idle",
            [](Infernux &self) {
                auto *r = self.GetRenderer();
                if (r)
                    r->WaitForGpuIdle();
            },
            "Drain pending GPU work before destructive scene replacement")
        .def(
            "resize_scene_render_target",
            [](Infernux &self, uint32_t width, uint32_t height) {
                auto *r = self.GetRenderer();
                if (r)
                    r->ResizeSceneRenderTarget(width, height);
            },
            py::arg("width"), py::arg("height"), "Resize the scene render target to match viewport size")
        // ========================================================================
        // Game Camera Render Target API - for Game View panel
        // ========================================================================
        .def(
            "get_game_texture_id",
            [](Infernux &self) -> uint64_t {
                auto *r = self.GetRenderer();
                return r ? r->GetGameTextureId() : 0;
            },
            "Get game render target texture ID for ImGui display")
        .def(
            "resize_game_render_target",
            [](Infernux &self, uint32_t width, uint32_t height) {
                auto *r = self.GetRenderer();
                if (r)
                    r->ResizeGameRenderTarget(width, height);
            },
            py::arg("width"), py::arg("height"), "Resize the game render target (lazy-initializes on first call)")
        .def(
            "set_game_camera_enabled",
            [](Infernux &self, bool enabled) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetGameCameraEnabled(enabled);
            },
            py::arg("enabled"), "Enable/disable game camera rendering")
        .def(
            "set_scene_view_visible",
            [](Infernux &self, bool visible) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetSceneViewVisible(visible);
            },
            py::arg("visible"), "Enable/disable scene view rendering")
        .def(
            "set_gui_player_mode",
            [](Infernux &self, bool enabled) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetGUIPlayerMode(enabled);
            },
            py::arg("enabled"), "Skip DockSpace/layout overhead in standalone player mode")
        .def(
            "is_game_camera_enabled",
            [](Infernux &self) -> bool {
                auto *r = self.GetRenderer();
                return r && r->IsGameCameraEnabled();
            },
            "Check if game camera rendering is enabled")
        .def(
            "get_last_game_render_ms",
            [](Infernux &self) -> double {
                auto *r = self.GetRenderer();
                return r ? r->GetLastGameRenderMs() : 0.0;
            },
            "Get last frame's game view render time (CPU command recording) in ms, excluding editor panels")
        .def(
            "get_game_only_frame_ms",
            [](Infernux &self) -> double {
                auto *r = self.GetRenderer();
                return r ? r->GetGameOnlyFrameMs() : 0.0;
            },
            "Get game-only frame cost in ms (SceneUpdate + PrepareFrame + GameRender), excluding editor panels")
        .def(
            "get_scene_update_ms",
            [](Infernux &self) -> double {
                auto *r = self.GetRenderer();
                return r ? r->GetSceneUpdateMs() : 0.0;
            },
            "Get SceneManager::Update + LateUpdate time in ms")
        .def(
            "get_gui_build_ms",
            [](Infernux &self) -> double {
                auto *r = self.GetRenderer();
                return r ? r->GetGuiBuildMs() : 0.0;
            },
            "Get GUI::BuildFrame (all ImGui panels) time in ms")
        .def(
            "get_prepare_frame_ms",
            [](Infernux &self) -> double {
                auto *r = self.GetRenderer();
                return r ? r->GetPrepareFrameMs() : 0.0;
            },
            "Get PrepareFrame (collect/cull renderables) time in ms")
        .def(
            "get_screen_ui_renderer",
            [](Infernux &self) -> InxScreenUIRenderer * {
                auto *r = self.GetRenderer();
                return r ? r->GetScreenUIRenderer() : nullptr;
            },
            py::return_value_policy::reference,
            "Get the screen UI renderer for GPU-based 2D screen-space UI (returns None before game RT init)")
        // ========================================================================
        // MSAA Configuration
        // ========================================================================
        .def(
            "set_msaa_samples",
            [](Infernux &self, int samples) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetMsaaSamples(samples);
            },
            py::arg("samples"), "Set MSAA sample count (1=off, 2, 4, 8) for both scene and game render targets")
        .def(
            "get_msaa_samples",
            [](Infernux &self) -> int {
                auto *r = self.GetRenderer();
                return r ? r->GetMsaaSamples() : 4;
            },
            "Get current MSAA sample count (1=off)")
        // ========================================================================
        // Present Mode
        // ========================================================================
        .def(
            "set_present_mode",
            [](Infernux &self, int mode) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetPresentMode(mode);
            },
            py::arg("mode"), "Set present mode: 0=IMMEDIATE, 1=MAILBOX, 2=FIFO, 3=FIFO_RELAXED")
        .def(
            "get_present_mode",
            [](Infernux &self) -> int {
                auto *r = self.GetRenderer();
                return r ? r->GetPresentMode() : 1;
            },
            "Get current present mode (0=IMMEDIATE, 1=MAILBOX, 2=FIFO, 3=FIFO_RELAXED)")
        // ========================================================================
        // Editor Power-Save / Idle Mode
        // ========================================================================
        .def(
            "set_editor_idle_enabled",
            [](Infernux &self, bool enabled) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetEditorIdleEnabled(enabled);
            },
            py::arg("enabled"), "Enable/disable editor idle mode (reduced FPS when no input)")
        .def(
            "is_editor_idle_enabled",
            [](Infernux &self) -> bool {
                auto *r = self.GetRenderer();
                return r && r->IsEditorIdleEnabled();
            },
            "Check if editor idle mode is enabled")
        .def(
            "set_editor_idle_fps",
            [](Infernux &self, float fps) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetEditorIdleFps(fps);
            },
            py::arg("fps"), "Set idle-mode target FPS (e.g. 10). 0 disables idling.")
        .def(
            "get_editor_idle_fps",
            [](Infernux &self) -> float {
                auto *r = self.GetRenderer();
                return r ? r->GetEditorIdleFps() : 0.0f;
            },
            "Get idle-mode target FPS")
        .def(
            "is_editor_idling",
            [](Infernux &self) -> bool {
                auto *r = self.GetRenderer();
                return r && r->IsEditorIdling();
            },
            "Check if editor is currently in idle (reduced FPS) state")
        .def(
            "request_full_speed_frame",
            [](Infernux &self) {
                auto *r = self.GetRenderer();
                if (r)
                    r->RequestFullSpeedFrame();
            },
            "Force full-speed rendering for the next few frames")
        .def(
            "set_editor_fps_cap",
            [](Infernux &self, float fps) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetEditorFpsCap(fps);
            },
            py::arg("fps"), "Set editor-mode FPS cap (e.g. 60). 0 = uncapped. Only applies outside play mode.")
        .def(
            "get_editor_fps_cap",
            [](Infernux &self) -> float {
                auto *r = self.GetRenderer();
                return r ? r->GetEditorFpsCap() : 0.0f;
            },
            "Get editor-mode FPS cap")
        .def(
            "set_play_mode_rendering",
            [](Infernux &self, bool play) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetPlayModeRendering(play);
            },
            py::arg("play"), "Enable/disable play-mode rendering (uncapped FPS, no idle)")
        .def(
            "is_play_mode_rendering",
            [](Infernux &self) -> bool {
                auto *r = self.GetRenderer();
                return r && r->IsPlayModeRendering();
            },
            "Check if renderer is in play-mode (uncapped FPS)")
        // ========================================================================
        // Scene Picking API - for editor selection
        // ========================================================================
        .def("pick_scene_object_id", &Infernux::PickSceneObjectId, py::arg("screen_x"), py::arg("screen_y"),
             py::arg("viewport_width"), py::arg("viewport_height"),
             "Pick a scene object or gizmo arrow by screen-space coordinates and return its ID (0 if none)")
        .def("pick_scene_object_ids", &Infernux::PickSceneObjectIds, py::arg("screen_x"), py::arg("screen_y"),
             py::arg("viewport_width"), py::arg("viewport_height"),
             "Pick ordered scene object candidate IDs from screen coordinates")
        .def("pick_gizmo_axis", &Infernux::PickGizmoAxis, py::arg("screen_x"), py::arg("screen_y"),
             py::arg("viewport_width"), py::arg("viewport_height"),
             "Lightweight gizmo axis proximity test for hover highlighting (no scene raycast)")
        .def("set_editor_tool_highlight", &Infernux::SetEditorToolHighlight, py::arg("axis"),
             "Set the highlighted gizmo axis. 0=None, 1=X, 2=Y, 3=Z.")
        .def("set_editor_tool_mode", &Infernux::SetEditorToolMode, py::arg("mode"),
             "Set the active tool mode. 0=None, 1=Translate, 2=Rotate, 3=Scale.")
        .def("get_editor_tool_mode", &Infernux::GetEditorToolMode,
             "Get the active tool mode. 0=None, 1=Translate, 2=Rotate, 3=Scale.")
        .def("set_editor_tool_local_mode", &Infernux::SetEditorToolLocalMode, py::arg("local"),
             "Enable/disable local coordinate mode for editor tools (gizmo aligns to object rotation)")
        .def("screen_to_world_ray", &Infernux::ScreenToWorldRay, py::arg("screen_x"), py::arg("screen_y"),
             py::arg("viewport_width"), py::arg("viewport_height"),
             "Build a world-space ray from screen coords. Returns (ox,oy,oz, dx,dy,dz).")
        // ========================================================================
        // Editor Gizmos API - for toggling visual aids in scene view
        // ========================================================================
        .def(
            "set_show_grid",
            [](Infernux &self, bool show) {
                auto *r = self.GetRenderer();
                if (r)
                    r->SetShowGrid(show);
            },
            py::arg("show"), "Set visibility of ground grid")
        .def(
            "is_show_grid",
            [](Infernux &self) -> bool {
                auto *r = self.GetRenderer();
                return r && r->IsShowGrid();
            },
            "Get visibility of ground grid")
        .def("set_selection_outline", &Infernux::SetSelectionOutline, py::arg("object_id"),
             "Set selection outline for a game object (Unity-style orange wireframe). Pass 0 to clear.")
        .def("set_selection_outlines", &Infernux::SetSelectionOutlines, py::arg("object_ids"),
             "Set combined selection outline for multiple game objects.")
        .def("get_selected_object_id", &Infernux::GetSelectedObjectId,
             "Get the currently selected object ID (0 if none).")
        .def("clear_selection_outline", &Infernux::ClearSelectionOutline, "Clear selection outline")
        // ========================================================================
        // Component Gizmos API — upload per-component gizmo geometry from Python
        // ========================================================================
        .def(
            "upload_component_gizmos",
            [](Infernux &self, py::buffer vertices, int64_t vertexCount, py::buffer indices, py::buffer descriptors,
               int64_t descriptorCount) {
                auto *renderer = self.GetRenderer();
                if (!renderer)
                    return;
                GizmosDrawCallBuffer *buf = renderer->GetGizmosDrawCallBuffer();
                if (!buf)
                    return;

                // vertices: flat float buffer, stride 6 (pos3 + color3) per vertex
                constexpr int64_t kVertStride = 6;
                py::buffer_info vInfo = vertices.request();
                const float *vPtr = static_cast<const float *>(vInfo.ptr);

                std::vector<Vertex> verts;
                verts.reserve(static_cast<size_t>(vertexCount));
                for (int64_t i = 0; i < vertexCount; ++i) {
                    const float *b = vPtr + i * kVertStride;
                    Vertex v;
                    v.pos = glm::vec3(b[0], b[1], b[2]);
                    v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
                    v.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
                    v.color = glm::vec3(b[3], b[4], b[5]);
                    v.texCoord = glm::vec2(0.0f);
                    verts.push_back(v);
                }

                // indices: flat uint32 buffer
                py::buffer_info iInfo = indices.request();
                const uint32_t *iPtr = static_cast<const uint32_t *>(iInfo.ptr);
                std::vector<uint32_t> idx(iPtr, iPtr + iInfo.size);

                // descriptors: flat float buffer, stride 18 (indexStart + indexCount + mat4x4)
                constexpr int64_t kDescStride = 18;
                py::buffer_info dInfo = descriptors.request();
                const float *dPtr = static_cast<const float *>(dInfo.ptr);

                std::vector<GizmosDrawCallBuffer::DrawDescriptor> descs;
                descs.reserve(static_cast<size_t>(descriptorCount));
                for (int64_t i = 0; i < descriptorCount; ++i) {
                    const float *b = dPtr + i * kDescStride;
                    GizmosDrawCallBuffer::DrawDescriptor d;
                    d.indexStart = static_cast<uint32_t>(b[0]);
                    d.indexCount = static_cast<uint32_t>(b[1]);
                    for (int j = 0; j < 16; ++j) {
                        d.worldMatrix[j] = b[2 + j];
                    }
                    descs.push_back(d);
                }

                buf->SetData(std::move(verts), std::move(idx), std::move(descs));
            },
            py::arg("vertices"), py::arg("vertex_count"), py::arg("indices"), py::arg("descriptors"),
            py::arg("descriptor_count"),
            "Upload per-component gizmo geometry via buffer protocol (no numpy). "
            "vertices: flat float32 (N*6), indices: flat uint32, descriptors: flat float32 (D*18)")
        .def(
            "clear_component_gizmos",
            [](Infernux &self) {
                auto *renderer = self.GetRenderer();
                if (!renderer)
                    return;
                GizmosDrawCallBuffer *buf = renderer->GetGizmosDrawCallBuffer();
                if (buf)
                    buf->Clear();
            },
            "Clear all component gizmo geometry")
        .def(
            "upload_component_gizmo_icons",
            [](Infernux &self, py::buffer positions, py::buffer objectIds, py::buffer iconKinds, int64_t iconCount) {
                auto *renderer = self.GetRenderer();
                if (!renderer)
                    return;
                GizmosDrawCallBuffer *buf = renderer->GetGizmosDrawCallBuffer();
                if (!buf || iconCount <= 0)
                    return;

                // positions: flat float buffer, stride 6 (pos3 + color3) per icon
                constexpr int64_t kPosStride = 6;
                py::buffer_info posInfo = positions.request();
                const float *posPtr = static_cast<const float *>(posInfo.ptr);

                // objectIds: flat uint32 buffer, stride 2 (lo + hi) per icon
                py::buffer_info idInfo = objectIds.request();
                const uint32_t *idPtr = static_cast<const uint32_t *>(idInfo.ptr);

                py::buffer_info kindInfo = iconKinds.request();
                const uint32_t *kindPtr = static_cast<const uint32_t *>(kindInfo.ptr);

                std::vector<GizmosDrawCallBuffer::IconEntry> entries;
                entries.reserve(static_cast<size_t>(iconCount));
                for (int64_t i = 0; i < iconCount; ++i) {
                    const float *p = posPtr + i * kPosStride;
                    const uint32_t *id = idPtr + i * 2;

                    GizmosDrawCallBuffer::IconEntry entry;
                    entry.position = glm::vec3(p[0], p[1], p[2]);
                    entry.color = glm::vec3(p[3], p[4], p[5]);
                    entry.objectId = (static_cast<uint64_t>(id[1]) << 32) | static_cast<uint64_t>(id[0]);
                    entry.iconKind = kindPtr[i];
                    entries.push_back(entry);
                }

                static int64_t s_lastIconUploadCount = -1;
                if (s_lastIconUploadCount != iconCount) {
                    uint32_t firstKind = entries.empty() ? 0u : entries.front().iconKind;
                    s_lastIconUploadCount = iconCount;
                }

                buf->SetIconData(std::move(entries));
            },
            py::arg("positions"), py::arg("object_ids"), py::arg("icon_kinds"), py::arg("icon_count"),
            "Upload component gizmo icon entries via buffer protocol (no numpy). "
            "positions: flat float32 (N*6: x,y,z,r,g,b), object_ids: flat uint32 (N*2: lo,hi), "
            "icon_kinds: flat uint32 (N)")
        .def(
            "clear_component_gizmo_icons",
            [](Infernux &self) {
                auto *renderer = self.GetRenderer();
                if (!renderer)
                    return;
                GizmosDrawCallBuffer *buf = renderer->GetGizmosDrawCallBuffer();
                if (buf)
                    buf->ClearIcons();
            },
            "Clear all component gizmo icon data")
        // ========================================================================
        // Material Pipeline API - for refreshing material shaders at runtime
        // ========================================================================
        .def("refresh_material_pipeline", &Infernux::RefreshMaterialPipeline, py::arg("material"),
             "Refresh a material's rendering pipeline by reloading its shaders")
        .def(
            "remove_material_pipeline",
            [](Infernux &self, const std::string &materialName) {
                auto *r = self.GetRenderer();
                if (r)
                    r->RemoveMaterialPipeline(materialName);
            },
            py::arg("material_name"), "Remove pipeline render data for a deleted material (releases GPU resources)")
        // ========================================================================
        // Render Pipeline API - for custom Python render pipelines (SRP)
        // ========================================================================
        .def(
            "set_render_pipeline",
            [](Infernux &self, py::object pipeline) {
                auto *r = self.GetRenderer();
                if (!r)
                    return;
                if (pipeline.is_none()) {
                    r->SetRenderPipeline(nullptr);
                } else {
                    r->SetRenderPipeline(pipeline.cast<std::shared_ptr<RenderPipelineCallback>>());
                }
            },
            py::arg("pipeline"),
            "Set a custom RenderPipelineCallback to control rendering from Python. Pass None to revert to default.")
        // ========================================================================
        // Render Graph API
        // ========================================================================
        .def(
            "get_scene_render_graph",
            [](Infernux &self) -> SceneRenderGraph * {
                auto *r = self.GetRenderer();
                return r ? r->GetSceneRenderGraph() : nullptr;
            },
            py::return_value_policy::reference, "Get the scene render graph for pass configuration");

    // ── Logging bridge: let Python write to the C++ InxLog (engine.log) ──
    m.def(
        "inflog_warn", [](const std::string &msg) { INXLOG_WARN(msg); }, py::arg("msg"),
        "Write a WARN-level message to the engine log.");

    m.def(
        "inflog_internal", [](const std::string &msg) { INXLOG_INFO_INTERNAL(msg); }, py::arg("msg"),
        "Write an internal INFO-level message to the engine log without surfacing it in the editor console.");

    // Register all binding modules
    RegisterGUIBindings(m);
    RegisterVector2Bindings(m);
    RegisterVector3Bindings(m);
    RegisterVec4fBindings(m);
    RegisterResourceBindings(m);
    RegisterAssetDatabaseBindings(m);
    RegisterAssetRegistryBindings(m);
    RegisterSceneBindings(m);
    RegisterTagLayerBindings(m);
    RegisterRenderGraphBindings(m);
    RegisterCommandBufferBindings(m); // Must come before RenderPipeline (provides VkFormat, RenderTargetHandle, etc.)
    RegisterRenderPipelineBindings(m);
    RegisterInputBindings(m);
    RegisterPhysicsBindings(m);
    RegisterAudioBindings(m);
    RegisterBatchBindings(m);

    // ====================================================================
    // Gizmo geometry generation helpers (pure math, no engine state needed)
    // Returns pre-packed flat float arrays ready for Python Gizmos system.
    // ====================================================================

    m.def(
        "generate_wire_sphere",
        [](float cx, float cy, float cz, float radius, int segments, float cr, float cg, float cb) -> py::tuple {
            // Generate 3 axis-aligned circles: YZ, XZ, XY
            const int totalVerts = segments * 3;
            const int totalIndices = segments * 3 * 2; // 2 indices per line segment

            // Pre-compute trig table
            std::vector<float> cosTab(segments), sinTab(segments);
            const float twoPi = 2.0f * 3.14159265358979323846f;
            for (int i = 0; i < segments; ++i) {
                float angle = twoPi * static_cast<float>(i) / static_cast<float>(segments);
                cosTab[i] = std::cos(angle);
                sinTab[i] = std::sin(angle);
            }

            // Flat vertex buffer: x,y,z,r,g,b per vertex
            std::vector<float> verts(totalVerts * 6);
            std::vector<int32_t> indices(totalIndices);

            int vi = 0; // vertex float index
            int ii = 0; // index index
            for (int axis = 0; axis < 3; ++axis) {
                int base = axis * segments;
                for (int i = 0; i < segments; ++i) {
                    float ca = cosTab[i] * radius;
                    float sa = sinTab[i] * radius;
                    float px, py, pz;
                    if (axis == 0) {
                        px = cx;
                        py = cy + ca;
                        pz = cz + sa;
                    } else if (axis == 1) {
                        px = cx + ca;
                        py = cy;
                        pz = cz + sa;
                    } else {
                        px = cx + ca;
                        py = cy + sa;
                        pz = cz;
                    }
                    verts[vi++] = px;
                    verts[vi++] = py;
                    verts[vi++] = pz;
                    verts[vi++] = cr;
                    verts[vi++] = cg;
                    verts[vi++] = cb;

                    indices[ii++] = base + i;
                    indices[ii++] = base + (i + 1) % segments;
                }
            }

            auto vertArr = py::array_t<float>(verts.size(), verts.data());
            auto idxArr = py::array_t<int32_t>(indices.size(), indices.data());
            return py::make_tuple(vertArr, totalVerts, idxArr);
        },
        py::arg("cx"), py::arg("cy"), py::arg("cz"), py::arg("radius"), py::arg("segments"), py::arg("cr"),
        py::arg("cg"), py::arg("cb"),
        "Generate wire sphere vertices and indices. Returns (vert_flat, vert_count, idx_flat).");

    m.def(
        "generate_wire_arc",
        [](float cx, float cy, float cz, float nx, float ny, float nz, float radius, float startDeg, float arcDeg,
           int segments, float cr, float cg, float cb) -> py::tuple {
            // Normalize normal
            float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len < 1e-8f)
                return py::make_tuple(py::array_t<float>(0), 0, py::array_t<int32_t>(0));
            nx /= len;
            ny /= len;
            nz /= len;

            // Build basis from normal
            float ax, ay, az;
            if (std::fabs(ny) < 0.99f) {
                ax = 0;
                ay = 1;
                az = 0;
            } else {
                ax = 1;
                ay = 0;
                az = 0;
            }

            // u = normalize(cross(normal, arbitrary))
            float ux = ny * az - nz * ay;
            float uy = nz * ax - nx * az;
            float uz = nx * ay - ny * ax;
            float ul = std::sqrt(ux * ux + uy * uy + uz * uz);
            ux /= ul;
            uy /= ul;
            uz /= ul;

            // v = cross(normal, u)
            float vx = ny * uz - nz * uy;
            float vy = nz * ux - nx * uz;
            float vz = nx * uy - ny * ux;

            int numPts = segments + 1;
            std::vector<float> verts(numPts * 6);
            std::vector<int32_t> indices(segments * 2);

            float startRad = startDeg * 3.14159265358979323846f / 180.0f;
            float arcRad = arcDeg * 3.14159265358979323846f / 180.0f;

            int vi = 0, ii = 0;
            for (int i = 0; i <= segments; ++i) {
                float angle = startRad + arcRad * static_cast<float>(i) / static_cast<float>(segments);
                float ca = std::cos(angle), sa = std::sin(angle);
                verts[vi++] = cx + radius * (ca * ux + sa * vx);
                verts[vi++] = cy + radius * (ca * uy + sa * vy);
                verts[vi++] = cz + radius * (ca * uz + sa * vz);
                verts[vi++] = cr;
                verts[vi++] = cg;
                verts[vi++] = cb;
                if (i > 0) {
                    indices[ii++] = i - 1;
                    indices[ii++] = i;
                }
            }

            auto vertArr = py::array_t<float>(verts.size(), verts.data());
            auto idxArr = py::array_t<int32_t>(indices.size(), indices.data());
            return py::make_tuple(vertArr, numPts, idxArr);
        },
        py::arg("cx"), py::arg("cy"), py::arg("cz"), py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("radius"),
        py::arg("start_deg"), py::arg("arc_deg"), py::arg("segments"), py::arg("cr"), py::arg("cg"), py::arg("cb"),
        "Generate wire arc vertices and indices. Returns (vert_flat, vert_count, idx_flat).");
}
