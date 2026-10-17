#include "BindingSceneInternal.h"
#include "MatrixPyBridge.h"
#include "function/renderer/rhi/RhiRenderTexture.h"
#include "function/scene/Camera.h"
#include "function/scene/Light.h"
#include "function/scene/LineRenderer.h"
#include "function/scene/SceneManager.h"
#include "function/scene/SkinnedMeshRenderer.h"
#include "function/scene/SpriteRenderer.h"
#include <array>

namespace py = pybind11;

namespace infernux::scene_binding
{

void BindLineRenderer(py::module_ &m)
{
    py::enum_<LineAlignment>(m, "LineAlignment")
        .value("View", LineAlignment::View)
        .value("TransformZ", LineAlignment::TransformZ)
        .export_values();

    py::enum_<LineTextureMode>(m, "LineTextureMode")
        .value("Stretch", LineTextureMode::Stretch)
        .value("Tile", LineTextureMode::Tile)
        .value("DistributePerSegment", LineTextureMode::DistributePerSegment)
        .value("RepeatPerSegment", LineTextureMode::RepeatPerSegment)
        .value("Static", LineTextureMode::Static)
        .export_values();

    py::enum_<LineCurveWrapMode>(m, "LineCurveWrapMode")
        .value("Clamp", LineCurveWrapMode::Clamp)
        .value("Repeat", LineCurveWrapMode::Repeat)
        .value("PingPong", LineCurveWrapMode::PingPong)
        .export_values();

    py::enum_<LineGradientMode>(m, "LineGradientMode")
        .value("Linear", LineGradientMode::Linear)
        .value("Fixed", LineGradientMode::Fixed)
        .value("PerceptualBlend", LineGradientMode::PerceptualBlend)
        .export_values();

    py::class_<LineWidthKey>(m, "LineWidthKey")
        .def(py::init([](float time, float value, float inTangent, float outTangent) {
                 return LineWidthKey{time, value, inTangent, outTangent};
             }),
             py::arg("time"), py::arg("value"), py::arg("in_tangent") = 0.0f, py::arg("out_tangent") = 0.0f)
        .def_readwrite("time", &LineWidthKey::time)
        .def_readwrite("value", &LineWidthKey::value)
        .def_readwrite("in_tangent", &LineWidthKey::inTangent)
        .def_readwrite("out_tangent", &LineWidthKey::outTangent);

    py::class_<LineColorKey>(m, "LineColorKey")
        .def(py::init([](float time, const glm::vec4 &color) { return LineColorKey{time, color}; }), py::arg("time"),
             py::arg("color"))
        .def_readwrite("time", &LineColorKey::time)
        .def_readwrite("color", &LineColorKey::color);

    NativeClass<LineRenderer, MeshRenderer>(m, "LineRenderer")
        .def(py::init<>())
        .def_property(
            "position_count", &LineRenderer::GetPositionCount,
            [](LineRenderer &renderer, py::ssize_t count) {
                if (count < 0)
                    throw py::value_error("position_count must be non-negative");
                renderer.SetPositionCount(static_cast<size_t>(count));
            },
            "Number of authored line positions")
        .def("get_position", &LineRenderer::GetPosition, py::arg("index"), "Get one authored position")
        .def("set_position", &LineRenderer::SetPosition, py::arg("index"), py::arg("position"),
             "Set one authored position")
        .def("get_positions", &LineRenderer::GetPositions, py::return_value_policy::copy, "Get all authored positions")
        .def("set_positions", &LineRenderer::SetPositions, py::arg("positions"), "Replace all authored positions")
        .def_property("start_width", &LineRenderer::GetStartWidth, &LineRenderer::SetStartWidth)
        .def_property("end_width", &LineRenderer::GetEndWidth, &LineRenderer::SetEndWidth)
        .def_property("width_multiplier", &LineRenderer::GetWidthMultiplier, &LineRenderer::SetWidthMultiplier)
        .def_property("width_curve", &LineRenderer::GetWidthCurve, &LineRenderer::SetWidthCurve)
        .def_property("width_curve_pre_wrap", &LineRenderer::GetWidthCurvePreWrap, &LineRenderer::SetWidthCurvePreWrap)
        .def_property("width_curve_post_wrap", &LineRenderer::GetWidthCurvePostWrap,
                      &LineRenderer::SetWidthCurvePostWrap)
        .def_property("start_color", &LineRenderer::GetStartColor, &LineRenderer::SetStartColor)
        .def_property("end_color", &LineRenderer::GetEndColor, &LineRenderer::SetEndColor)
        .def_property("color_gradient", &LineRenderer::GetColorGradient, &LineRenderer::SetColorGradient)
        .def_property("color_gradient_mode", &LineRenderer::GetColorGradientMode, &LineRenderer::SetColorGradientMode)
        .def_property("loop", &LineRenderer::GetLoop, &LineRenderer::SetLoop)
        .def_property("use_world_space", &LineRenderer::GetUseWorldSpace, &LineRenderer::SetUseWorldSpace)
        .def_property("alignment", &LineRenderer::GetAlignment, &LineRenderer::SetAlignment)
        .def_property("texture_mode", &LineRenderer::GetTextureMode, &LineRenderer::SetTextureMode)
        .def_property("texture_scale", &LineRenderer::GetTextureScale, &LineRenderer::SetTextureScale)
        .def_property("num_corner_vertices", &LineRenderer::GetNumCornerVertices, &LineRenderer::SetNumCornerVertices)
        .def_property("num_cap_vertices", &LineRenderer::GetNumCapVertices, &LineRenderer::SetNumCapVertices)
        .def_property("shadow_bias", &LineRenderer::GetShadowBias, &LineRenderer::SetShadowBias)
        .def_property("generate_lighting_data", &LineRenderer::GetGenerateLightingData,
                      &LineRenderer::SetGenerateLightingData)
        .def("bake_mesh", &LineRenderer::BakeMesh, py::arg("target"), py::arg("camera") = nullptr,
             py::arg("use_transform") = false, "Bake the expanded line snapshot into a MeshRenderer inline mesh")
        .def("simplify", &LineRenderer::Simplify, py::arg("tolerance"),
             "Reduce the position list using Ramer-Douglas-Peucker simplification");
}

void BindSkinnedMeshRenderer(py::module_ &m)
{
    NativeClass<SkinnedMeshRenderer, MeshRenderer>(m, "SkinnedMeshRenderer")
        .def(py::init<>())
        .def_property_readonly("source_model_guid", &SkinnedMeshRenderer::GetSourceModelGuid,
                               "GUID of the animated source model asset")
        .def_property("animation_source_guid", &SkinnedMeshRenderer::GetAnimationSourceGuid,
                      &SkinnedMeshRenderer::SetAnimationSourceGuid,
                      "GUID of the independently imported model that owns the active animation take")
        .def(
            "set_source_model_guid",
            [](SkinnedMeshRenderer &sr, const std::string &guid) { sr.SetSourceModelGuid(guid); }, py::arg("guid"),
            "Assign the skinned source model by GUID")
        .def_property("active_take_name", &SkinnedMeshRenderer::GetActiveTakeName,
                      &SkinnedMeshRenderer::SetActiveTakeName, "Currently selected animation take name")
        .def_property_readonly("has_animation_takes", &SkinnedMeshRenderer::HasAnimationTakes,
                               "Whether this renderer has any imported animation takes")
        .def_property_readonly("animation_take_count", &SkinnedMeshRenderer::GetAnimationTakeCount,
                               "Number of imported animation takes on the source model")
        .def(
            "get_animation_take_names", [](const SkinnedMeshRenderer &sr) { return sr.GetAnimationTakeNames(); },
            "Get imported animation take names from the source model")
        .def("get_skeleton_node_names", &SkinnedMeshRenderer::GetSkeletonNodeNames,
             "Get unique skeleton node names, including nodes without skin weights")
        .def(
            "get_skeleton_node_world_matrices",
            [](const SkinnedMeshRenderer &renderer, const std::vector<std::string> &nodeNames) {
                const auto matrices = renderer.GetSkeletonNodeWorldMatrices(nodeNames);
                py::array_t<float> result(py::array::ShapeContainer{static_cast<py::ssize_t>(matrices.size()),
                                                                    py::ssize_t{4}, py::ssize_t{4}});
                auto values = result.mutable_unchecked<3>();
                for (size_t index = 0; index < matrices.size(); ++index)
                    for (int row = 0; row < 4; ++row)
                        for (int column = 0; column < 4; ++column)
                            values(index, row, column) = matrices[index][column][row];
                return result;
            },
            py::arg("node_names"),
            "Read one latest submitted pose as float32[N,4,4], indexed [node,row,column]. "
            "Matrices map engine-unit bone-local coordinates into rendered world space; unknown names raise.")
        .def("get_animation_duration_seconds", &SkinnedMeshRenderer::GetAnimationDurationSeconds, py::arg("take_name"),
             py::arg("animation_source_guid") = "", "Get imported animation take duration in seconds")
        .def(
            "get_root_motion_delta",
            [](const SkinnedMeshRenderer &renderer, const std::string &takeName, float fromSeconds, float toSeconds,
               bool loop, const std::string &animationSourceGuid) {
                const RootMotionDelta delta =
                    renderer.GetRootMotionDelta(takeName, fromSeconds, toSeconds, loop, animationSourceGuid);
                return py::make_tuple(delta.translation, delta.rotation);
            },
            py::arg("take_name"), py::arg("from_seconds"), py::arg("to_seconds"), py::arg("loop") = true,
            py::arg("animation_source_guid") = "",
            "Return the imported root-motion translation and rotation delta for one playback interval")
        .def("submit_animation_pose", &SkinnedMeshRenderer::SubmitAnimationPose, py::arg("take_name"),
             py::arg("time_seconds"), py::arg("normalized_time"), py::arg("blend_take_name") = "",
             py::arg("blend_time_seconds") = 0.0f, py::arg("blend_weight") = 0.0f, py::arg("loop") = true,
             py::arg("animation_source_guid") = "", py::arg("blend_animation_source_guid") = "",
             "Submit active and blend animation state in one native call. "
             "Empty take_name renders the bind pose; loop=False holds the end pose.")
        .def_property(
            "runtime_animation_time", &SkinnedMeshRenderer::GetRuntimeAnimationTime,
            [](SkinnedMeshRenderer &renderer, float time) {
                const float previous = renderer.GetRuntimeAnimationTime();
                renderer.SetRuntimeAnimationTime(time);
                if (previous != renderer.GetRuntimeAnimationTime())
                    SceneManager::Instance().MarkActiveSceneTemporalDiscontinuity();
            },
            "Explicitly seek the current clip time in seconds")
        .def_property("runtime_animation_normalized_time", &SkinnedMeshRenderer::GetRuntimeAnimationNormalizedTime,
                      &SkinnedMeshRenderer::SetRuntimeAnimationNormalizedTime,
                      "Normalized clip time 0..1 (runtime; driven when duration is known)")
        .def_property("blend_take_name", &SkinnedMeshRenderer::GetBlendTakeName, &SkinnedMeshRenderer::SetBlendTakeName,
                      "Secondary animation take name used for runtime pose blending")
        .def_property("blend_animation_time", &SkinnedMeshRenderer::GetBlendAnimationTime,
                      &SkinnedMeshRenderer::SetBlendAnimationTime,
                      "Secondary animation time in seconds used for runtime pose blending")
        .def_property("blend_weight", &SkinnedMeshRenderer::GetBlendWeight, &SkinnedMeshRenderer::SetBlendWeight,
                      "Runtime pose blend weight from active take to blend take")
        .def("clear_animation_blend", &SkinnedMeshRenderer::ClearAnimationBlend, "Clear runtime pose blending state")
        .def(
            "submit_pose_stack",
            [](SkinnedMeshRenderer &sr, const py::list &layers) {
                std::vector<PoseStackLayer> stack;
                stack.reserve(py::len(layers));
                for (const auto &item : layers) {
                    py::dict d = item.cast<py::dict>();
                    PoseStackLayer layer;
                    if (d.contains("take_name"))
                        layer.takeName = d["take_name"].cast<std::string>();
                    if (d.contains("source_model_guid"))
                        layer.sourceModelGuid = d["source_model_guid"].cast<std::string>();
                    if (d.contains("time"))
                        layer.timeSeconds = d["time"].cast<float>();
                    if (d.contains("weight"))
                        layer.weight = d["weight"].cast<float>();
                    if (d.contains("additive"))
                        layer.additive = d["additive"].cast<bool>();
                    if (d.contains("loop"))
                        layer.loop = d["loop"].cast<bool>();
                    if (d.contains("bone_mask"))
                        layer.boneMask = d["bone_mask"].cast<std::vector<std::string>>();
                    if (d.contains("blend_group"))
                        layer.blendGroup = d["blend_group"].cast<std::string>();
                    if (d.contains("group_weight"))
                        layer.groupWeight = d["group_weight"].cast<float>();
                    stack.push_back(std::move(layer));
                }
                sr.SubmitPoseStack(stack);
            },
            py::arg("layers"),
            "Submit a multi-layer pose stack (AnimationTree output). Each layer is a dict: "
            "{take_name:str, source_model_guid:str, time:float, weight:float, additive:bool, loop:bool, "
            "bone_mask:list[str], blend_group:str, group_weight:float}. "
            "Enables N-way weighted + additive + bone-masked blending beyond the 2-clip crossfade.")
        .def("clear_pose_stack", &SkinnedMeshRenderer::ClearPoseStack,
             "Clear the pose stack and revert to the single-clip / crossfade path")
        .def("has_pose_stack", &SkinnedMeshRenderer::HasPoseStack, "Whether a pose stack is currently active");
}

void BindSpriteRenderer(py::module_ &m)
{
    NativeClass<SpriteRenderer, MeshRenderer>(m, "SpriteRenderer")
        .def(py::init<>())
        .def_property("sprite_guid", &SpriteRenderer::GetSpriteGuid, &SpriteRenderer::SetSpriteGuid,
                      "Asset GUID of the sprite texture")
        .def_property("frame_id", &SpriteRenderer::GetFrameId, &SpriteRenderer::SetFrameId,
                      "Stable ID of the sprite frame to display")
        .def_property(
            "sprite_color",
            [](const SpriteRenderer &sr) -> py::tuple {
                const auto &c = sr.GetColor();
                return py::make_tuple(c.r, c.g, c.b, c.a);
            },
            [](SpriteRenderer &sr, const py::tuple &t) {
                sr.SetColor(glm::vec4(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(), t[3].cast<float>()));
            },
            "Sprite tint color (r, g, b, a)")
        .def_property("flip_x", &SpriteRenderer::GetFlipX, &SpriteRenderer::SetFlipX, "Flip sprite horizontally")
        .def_property("flip_y", &SpriteRenderer::GetFlipY, &SpriteRenderer::SetFlipY, "Flip sprite vertically")
        .def("_publish_sprite_texture", &SpriteRenderer::PublishSpriteTexture)
        .def("_publish_sprite_color", &SpriteRenderer::PublishSpriteColor)
        .def("_publish_sprite_uv",
             [](py::object renderer, const std::array<float, 4> &uvRect, const std::array<float, 4> &displayScale) {
                 renderer.cast<SpriteRenderer &>().PublishSpriteUV(
                     glm::vec4(uvRect[0], uvRect[1], uvRect[2], uvRect[3]),
                     glm::vec4(displayScale[0], displayScale[1], displayScale[2], displayScale[3]));
             });
}

void BindLight(py::module_ &m)
{
    py::enum_<LightType>(m, "LightType")
        .value("Directional", LightType::Directional)
        .value("Point", LightType::Point)
        .value("Spot", LightType::Spot)
        .value("Area", LightType::Area)
        .export_values();

    py::enum_<LightShadows>(m, "LightShadows")
        .value("NoShadows", LightShadows::None)
        .value("Hard", LightShadows::Hard)
        .value("Soft", LightShadows::Soft)
        .export_values();

    py::enum_<LightRenderMode>(m, "LightRenderMode")
        .value("Auto", LightRenderMode::Auto)
        .value("ForcePixel", LightRenderMode::ForcePixel)
        .value("ForceVertex", LightRenderMode::ForceVertex);

    py::enum_<LightColorMode>(m, "LightColorMode")
        .value("Color", LightColorMode::Color)
        .value("FilterAndTemperature", LightColorMode::FilterAndTemperature);

    // Light component binding (Unity-like API)
    NativeClass<Light, Component>(m, "Light")
        // Light type
        .def_property("light_type", &Light::GetLightType, &Light::SetLightType,
                      "Type of light (Directional, Point, Spot, Area)")

        // Color & intensity (Unity-style)
        .def_property(
            "color", [](Light *l) { return glm::vec3(l->GetColor()); },
            [](Light *l, const glm::vec3 &v) { l->SetColor(v.x, v.y, v.z); },
            "Authored sRGB color or temperature filter")
        .def_property("use_color_temperature", &Light::GetUseColorTemperature, &Light::SetUseColorTemperature,
                      "Use color as a filter over Kelvin blackbody color")
        .def_property("color_mode", &Light::GetColorMode, &Light::SetColorMode,
                      "Color or Filter and Temperature appearance mode")
        .def_property("color_temperature", &Light::GetColorTemperature, &Light::SetColorTemperature,
                      "Blackbody color temperature in Kelvin (1000-20000)")
        .def_property_readonly("effective_color", &Light::GetEffectiveColor,
                               "Final emitted color in sRGB before intensity")
        .def_property_readonly("effective_linear_color", &Light::GetLinearColor,
                               "Final emitted color in linear RGB before intensity")
        .def_property("intensity", &Light::GetIntensity, &Light::SetIntensity, "Light intensity multiplier")

        // Range (Point/Spot)
        .def_property("range", &Light::GetRange, &Light::SetRange, "Light range (Point/Spot lights)")

        // Spot angle (Spot only)
        .def_property("spot_angle", &Light::GetSpotAngle, &Light::SetSpotAngle, "Inner spot angle in degrees")
        .def_property("outer_spot_angle", &Light::GetOuterSpotAngle, &Light::SetOuterSpotAngle,
                      "Outer spot angle in degrees")
        .def_property("area_size", &Light::GetAreaSize, &Light::SetAreaSize, "Rectangle area-light width and height")
        .def_property("area_two_sided", &Light::GetAreaTwoSided, &Light::SetAreaTwoSided,
                      "Whether the rectangle emits from both sides")

        // Shadows
        .def_property("shadows", &Light::GetShadows, &Light::SetShadows, "Shadow type (None, Hard, Soft)")
        .def_property("shadow_strength", &Light::GetShadowStrength, &Light::SetShadowStrength, "Shadow strength (0-1)")
        .def_property_readonly("shadow_bias", &Light::GetShadowBias, "Engine-managed shadow depth bias")
        .def_property_readonly("shadow_normal_bias", &Light::GetShadowNormalBias, "Engine-managed shadow normal bias")
        .def_property("shadow_softness", &Light::GetShadowSoftness, &Light::SetShadowSoftness,
                      "Soft-shadow filter radius in shadow-map texels")

        // Influence domains are orthogonal to the GameObject layer mask.
        .def_property("affect_geometry", &Light::GetAffectGeometry, &Light::SetAffectGeometry,
                      "Whether this light affects geometry renderers")
        .def_property("affect_particles", &Light::GetAffectParticles, &Light::SetAffectParticles,
                      "Whether this light affects particle renderers")
        .def_property("culling_mask", &Light::GetCullingMask, &Light::SetCullingMask,
                      "Layer bitmask selecting which GameObjects this light affects")

        // Serialization
        .def("serialize", &Light::Serialize, "Serialize Light to JSON string");
}

void BindCamera(py::module_ &m)
{
    py::enum_<CameraProjection>(m, "CameraProjection")
        .value("Perspective", CameraProjection::Perspective)
        .value("Orthographic", CameraProjection::Orthographic)
        .export_values();

    py::enum_<PhysicalGateFit>(m, "PhysicalGateFit")
        .value("None", PhysicalGateFit::None)
        // ``None`` is retained for serialized/API parity; Python callers can
        // spell the reserved member as ``PhysicalGateFit.None_``.
        .value("None_", PhysicalGateFit::None)
        .value("Vertical", PhysicalGateFit::Vertical)
        .value("Horizontal", PhysicalGateFit::Horizontal)
        .value("Fill", PhysicalGateFit::Fill)
        .value("Overscan", PhysicalGateFit::Overscan)
        .export_values();

    py::enum_<CameraSensorType>(m, "CameraSensorType")
        .value("Film8mm", CameraSensorType::Film8mm)
        .value("Super8mm", CameraSensorType::Super8mm)
        .value("Film16mm", CameraSensorType::Film16mm)
        .value("Super16mm", CameraSensorType::Super16mm)
        .value("Film35mm2Perf", CameraSensorType::Film35mm2Perf)
        .value("Film35mmAcademy", CameraSensorType::Film35mmAcademy)
        .value("Super35", CameraSensorType::Super35)
        .value("Film35mmTVProjection", CameraSensorType::Film35mmTVProjection)
        .value("Film35mmFullAperture", CameraSensorType::Film35mmFullAperture)
        .value("Film35mm185Projection", CameraSensorType::Film35mm185Projection)
        .value("Film35mmAnamorphic", CameraSensorType::Film35mmAnamorphic)
        .value("Film65mmAlexa", CameraSensorType::Film65mmAlexa)
        .value("Film70mm", CameraSensorType::Film70mm)
        .value("Film70mmImax", CameraSensorType::Film70mmImax)
        .value("Custom", CameraSensorType::Custom)
        .export_values();

    // CameraClearFlags enum
    py::enum_<CameraClearFlags>(m, "CameraClearFlags")
        .value("Skybox", CameraClearFlags::Skybox)
        .value("SolidColor", CameraClearFlags::SolidColor)
        .value("DepthOnly", CameraClearFlags::DepthOnly)
        .value("DontClear", CameraClearFlags::DontClear)
        .export_values();

    // Camera component binding (Unity-like API)
    NativeClass<Camera, Component>(m, "Camera")
        .def_property("target_texture", &Camera::GetTargetTexture, &Camera::SetTargetTexture,
                      "Shared RenderTexture output owner")
        .def_property("target_texture_guid", &Camera::GetTargetTextureGuid, &Camera::SetTargetTextureGuid,
                      "Authored output identity; resolved by the renderer, never by scene decoding")
        // Projection mode
        .def_property("projection_mode", &Camera::GetProjectionMode, &Camera::SetProjectionMode,
                      "Camera projection mode (Perspective or Orthographic)")
        // Perspective settings
        .def_property("field_of_view", &Camera::GetFieldOfView, &Camera::SetFieldOfView,
                      "Field of view in degrees (Perspective mode)")
        .def_property("use_physical_properties", &Camera::GetUsePhysicalProperties, &Camera::SetUsePhysicalProperties,
                      "Use physical lens and sensor properties for perspective projection")
        .def_property("iso", &Camera::GetIso, &Camera::SetIso, "Physical camera sensor sensitivity")
        .def_property("shutter_speed", &Camera::GetShutterSpeed, &Camera::SetShutterSpeed,
                      "Physical camera exposure time in seconds")
        .def_property("aperture", &Camera::GetAperture, &Camera::SetAperture, "Physical camera aperture in f-stops")
        .def_property("focus_distance", &Camera::GetFocusDistance, &Camera::SetFocusDistance,
                      "Physical camera focus-plane distance")
        .def_property("blade_count", &Camera::GetBladeCount, &Camera::SetBladeCount,
                      "Physical camera diaphragm blade count")
        .def_property("curvature", &Camera::GetCurvature, &Camera::SetCurvature,
                      "Aperture range mapped to diaphragm blade curvature")
        .def_property("barrel_clipping", &Camera::GetBarrelClipping, &Camera::SetBarrelClipping,
                      "Optical-vignetting cat-eye strength")
        .def_property("anamorphism", &Camera::GetAnamorphism, &Camera::SetAnamorphism, "Physical camera sensor stretch")
        .def_property("focal_length", &Camera::GetFocalLength, &Camera::SetFocalLength,
                      "Physical camera focal length in millimetres")
        .def_property("sensor_type", &Camera::GetSensorType, &Camera::SetSensorType,
                      "Derived Unity sensor-size preset; Custom leaves the authored size unchanged")
        .def_property("sensor_size", &Camera::GetSensorSize, &Camera::SetSensorSize,
                      "Physical camera sensor size in millimetres (width, height)")
        .def_property("lens_shift", &Camera::GetLensShift, &Camera::SetLensShift,
                      "Physical camera lens shift in normalized sensor units")
        .def_property("gate_fit", &Camera::GetGateFit, &Camera::SetGateFit, "Physical camera film/resolution gate fit")
        .def_property("aspect_ratio", &Camera::GetAspectRatio, &Camera::SetAspectRatio, "Aspect ratio (width/height)")
        .def_property(
            "projection_matrix", [](const Camera &c) { return binding::Matrix4ToPython(c.GetProjectionMatrix()); },
            [](Camera &c, const FloatArray &matrix) {
                c.SetProjectionMatrix(binding::Matrix4FromPython(matrix, "Camera matrix"));
            },
            "Runtime projection as a NumPy (4,4) [row,column] copy; LH, depth [0,1], Y down")
        .def_property(
            "view_matrix", [](const Camera &c) { return binding::Matrix4ToPython(c.GetViewMatrix()); },
            [](Camera &c, const FloatArray &matrix) {
                c.SetViewMatrix(binding::Matrix4FromPython(matrix, "Camera matrix"));
            },
            "Runtime affine world-to-camera matrix copy; +Z forward; independent of Transform until reset")
        .def_property_readonly("camera_to_world_matrix",
                               [](const Camera &c) { return binding::Matrix4ToPython(c.GetCameraToWorldMatrix()); })
        .def_property_readonly("has_custom_view_matrix", &Camera::HasCustomViewMatrix)
        .def("reset_view_matrix", &Camera::ResetViewMatrix)
        .def("reset_history", &Camera::ResetHistory,
             "Discard this camera's accumulated temporal history before its next render; other cameras are unaffected")
        .def_property("invert_culling", &Camera::GetInvertCulling, &Camera::SetInvertCulling,
                      "Camera-local winding inversion; does not affect light-space shadows or other cameras")
        .def_property_readonly("has_custom_projection_matrix", &Camera::HasCustomProjectionMatrix)
        .def("reset_projection_matrix", &Camera::ResetProjectionMatrix)
        .def(
            "calculate_oblique_matrix",
            [](const Camera &c, const std::array<float, 4> &plane) {
                return binding::Matrix4ToPython(
                    c.CalculateObliqueMatrix(glm::vec4(plane[0], plane[1], plane[2], plane[3])));
            },
            py::arg("clip_plane"), "Camera-space plane (nx,ny,nz,d), retaining its positive half-space")
        // Orthographic settings
        .def_property("orthographic_size", &Camera::GetOrthographicSize, &Camera::SetOrthographicSize,
                      "Orthographic half-height (Orthographic mode)")
        // Clipping planes
        .def_property("near_clip", &Camera::GetNearClip, &Camera::SetNearClip, "Near clipping plane distance")
        .def_property("far_clip", &Camera::GetFarClip, &Camera::SetFarClip, "Far clipping plane distance")
        .def("set_clip_planes", &Camera::SetClipPlanes, py::arg("near_clip"), py::arg("far_clip"),
             "Set a finite 0 < near < far pair atomically")
        // Multi-camera support
        .def_property("depth", &Camera::GetDepth, &Camera::SetDepth,
                      "Rendering order for the Game camera stack; lower depth renders first")
        .def_property("culling_mask", &Camera::GetCullingMask, &Camera::SetCullingMask,
                      "Layer culling bitmask (which layers this camera renders)")
        // Clear flags & background color
        .def_property("clear_flags", &Camera::GetClearFlags, &Camera::SetClearFlags,
                      "Camera clear flags (Skybox, SolidColor, DepthOnly, DontClear)")
        .def_property(
            "background_color", [](const Camera &c) -> glm::vec4 { return c.GetBackgroundColor(); },
            [](Camera &c, const glm::vec4 &v) { c.SetBackgroundColor(v); },
            "Background color as vec4f (r, g, b, a) — used when clear_flags == SolidColor")
        .def_property("dithering", &Camera::GetDithering, &Camera::SetDithering,
                      "Apply display-space dithering before output quantization")
        .def_property("stop_nans", &Camera::GetStopNaNs, &Camera::SetStopNaNs,
                      "Replace non-finite output pixels before display encoding")
        // Screen dimensions (read-only, set by renderer)
        .def_property_readonly("pixel_width", &Camera::GetPixelWidth, "Render target width in pixels")
        .def_property_readonly("pixel_height", &Camera::GetPixelHeight, "Render target height in pixels")
        // Coordinate conversion
        .def(
            "screen_to_world_point",
            [](const Camera &c, float x, float y, float depth) { return c.ScreenToWorldPoint(glm::vec2(x, y), depth); },
            py::arg("x"), py::arg("y"), py::arg("depth") = 0.0f,
            "Convert screen coordinates (x, y) + depth [0..1] to world position")
        .def(
            "world_to_screen_point",
            [](const Camera &c, float x, float y, float z) { return c.WorldToScreenPoint(glm::vec3(x, y, z)); },
            py::arg("x"), py::arg("y"), py::arg("z"), "Convert world position to screen coordinates (x, y)")
        .def(
            "screen_point_to_ray",
            [](const Camera &c, float x, float y, const std::optional<float> &viewportWidth,
               const std::optional<float> &viewportHeight) -> py::tuple {
                const bool hasExplicitViewport = viewportWidth.has_value() || viewportHeight.has_value();
                if (hasExplicitViewport && (!viewportWidth.has_value() || !viewportHeight.has_value())) {
                    throw py::value_error("viewport_width and viewport_height must be provided together");
                }
                auto [origin, dir] = hasExplicitViewport
                                         ? c.ScreenPointToRay(glm::vec2(x, y), *viewportWidth, *viewportHeight)
                                         : c.ScreenPointToRay(glm::vec2(x, y));
                return py::make_tuple(origin, dir);
            },
            py::arg("x"), py::arg("y"), py::arg("viewport_width") = py::none(), py::arg("viewport_height") = py::none(),
            "Build a ray from viewport-relative screen coordinates. "
            "Returns (origin_Vector3, direction_Vector3) — origin at near plane and normalised direction.")
        // Serialization
        .def("serialize", &Camera::Serialize, "Serialize Camera to JSON string")
        .def("deserialize", &Camera::Deserialize, py::arg("json_str"), "Deserialize Camera from JSON string");
}

} // namespace infernux::scene_binding
