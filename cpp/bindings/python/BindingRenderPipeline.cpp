#include "BindingRegistration.h"
#include <function/renderer/CommandBuffer.h>
#include <function/renderer/RenderGraphDescription.h>
#include <function/renderer/ScriptableRenderContext.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace infernux;
namespace py = pybind11;

// ============================================================================
// Trampoline for Python-overridable RenderPipelineCallback
// ============================================================================

class PyRenderPipelineCallback : public RenderPipelineCallback
{
  public:
    using RenderPipelineCallback::RenderPipelineCallback;

    bool ShouldRenderView(const RenderViewData &view) override
    {
        py::gil_scoped_acquire gil;
        py::function select = py::get_override(static_cast<const RenderPipelineCallback *>(this), "should_render_view");
        // A callback may retain the read-only selection value after returning.
        return select ? select(std::make_shared<RenderViewData>(view)).cast<bool>()
                      : RenderPipelineCallback::ShouldRenderView(view);
    }
    void Render(ScriptableRenderContext &context, const RenderViewData &view) override
    {
        py::gil_scoped_acquire gil;
        py::function render = py::get_override(static_cast<const RenderPipelineCallback *>(this), "render");
        if (!render)
            throw std::runtime_error("RenderPipelineCallback.render(context, view) must be implemented");
        if (&view != &context.GetView())
            throw std::invalid_argument("RenderPipelineCallback must receive its context's RenderView");
        py::object owner = py::cast(&context, py::return_value_policy::reference);
        py::object exposedView = py::cast(context.GetViewPublication());
        render(owner, exposedView);
    }
    void Dispose() override
    {
        PYBIND11_OVERRIDE_NAME(void, RenderPipelineCallback, "dispose", Dispose);
    }
};

// ============================================================================
// Registration
// ============================================================================

namespace infernux
{
void RegisterRenderPipelineBindings(py::module_ &m)
{
    // An owning read-only value shared by the context and callback. No
    // Camera/GameObject/Transform pointer is exported through this object.
    py::class_<RenderViewData, std::shared_ptr<RenderViewData>>(m, "RenderView")
        .def_readonly("camera_id", &RenderViewData::cameraId)
        .def_readonly("world_id", &RenderViewData::worldId)
        .def_readonly("world_frame", &RenderViewData::completedWorldFrame)
        .def_readonly("simulation_epoch", &RenderViewData::simulationEpoch)
        .def_readonly("name", &RenderViewData::name)
        .def_readonly("scene_name", &RenderViewData::sceneName)
        .def_readonly("scene_id", &RenderViewData::sceneWorldId)
        .def_readonly("near_clip", &RenderViewData::nearClip)
        .def_readonly("far_clip", &RenderViewData::farClip)
        .def_readonly("depth", &RenderViewData::depth)
        .def_readonly("output_width", &RenderViewData::outputWidth)
        .def_readonly("output_height", &RenderViewData::outputHeight)
        .def_readonly("pixel_width", &RenderViewData::pixelWidth)
        .def_readonly("pixel_height", &RenderViewData::pixelHeight)
        .def_readonly("culling_mask", &RenderViewData::cullingMask)
        .def_readonly("clear_flags", &RenderViewData::clearFlags)
        .def_readonly("dithering", &RenderViewData::dithering)
        .def_readonly("stop_nans", &RenderViewData::stopNaNs)
        .def_readonly("invert_culling", &RenderViewData::invertCulling)
        .def_readonly("is_editor", &RenderViewData::editorView)
        .def_readonly("is_main_camera", &RenderViewData::isMainCamera)
        .def_readonly("render_to_texture", &RenderViewData::renderToTexture)
        .def_readonly("render_on_demand", &RenderViewData::renderOnDemand)
        .def_property_readonly("position",
                               [](const RenderViewData &view) {
                                   return std::array<float, 3>{view.position.x, view.position.y, view.position.z};
                               })
        .def_property_readonly("forward",
                               [](const RenderViewData &view) {
                                   return std::array<float, 3>{view.forward.x, view.forward.y, view.forward.z};
                               })
        .def_property_readonly(
            "up", [](const RenderViewData &view) { return std::array<float, 3>{view.up.x, view.up.y, view.up.z}; })
        .def_property_readonly("rect",
                               [](const RenderViewData &view) {
                                   return std::array<float, 4>{view.rect.x, view.rect.y, view.rect.z, view.rect.w};
                               })
        .def_property_readonly("background_color",
                               [](const RenderViewData &view) {
                                   return std::array<float, 4>{view.backgroundColor.r, view.backgroundColor.g,
                                                               view.backgroundColor.b, view.backgroundColor.a};
                               })
        .def_property_readonly("view_matrix",
                               [](const RenderViewData &view) {
                                   std::array<float, 16> values{};
                                   for (size_t column = 0; column < 4; ++column)
                                       for (size_t row = 0; row < 4; ++row)
                                           values[column * 4 + row] = view.view[column][row];
                                   return values;
                               })
        .def_property_readonly("projection_matrix", [](const RenderViewData &view) {
            std::array<float, 16> values{};
            for (size_t column = 0; column < 4; ++column)
                for (size_t row = 0; row < 4; ++row)
                    values[column * 4 + row] = view.projection[column][row];
            return values;
        });
    // ---- CullingResults ----
    py::class_<CullingResults>(m, "CullingResults")
        .def_property_readonly("visible_object_count",
                               [](const CullingResults &self) { return self.visibleObjectCount(); })
        .def_property_readonly("visible_light_count",
                               [](const CullingResults &self) { return self.visibleLightCount(); });

    // ---- ScriptableRenderContext ----
    py::class_<ScriptableRenderContext>(m, "ScriptableRenderContext")
        .def_property_readonly("scene", &ScriptableRenderContext::GetScene, py::return_value_policy::reference,
                               "Get the scene associated with this render context")
        .def_property_readonly("view", &ScriptableRenderContext::GetViewPublication)
        .def_property_readonly("capturing_completed_world", &ScriptableRenderContext::IsCapturingCompletedWorld)
        .def_property_readonly("_has_screen_ui_snapshot",
                               [](const ScriptableRenderContext &self) {
                                   const auto plan = self.CapturePlan();
                                   return plan && static_cast<bool>(plan->ui);
                               })
        .def("capture_screen_ui", &ScriptableRenderContext::CaptureScreenUI, py::arg("renderer"))
        .def("cull", &ScriptableRenderContext::Cull, py::return_value_policy::reference_internal,
             "Cull this context's retained RenderView") // RenderGraph-driven API
        .def("apply_graph", &ScriptableRenderContext::ApplyGraph, py::arg("description"),
             "Apply a Python-defined RenderGraph topology to the scene render graph")
        .def("update_parameter_blocks", &ScriptableRenderContext::UpdateParameterBlocks, py::arg("updates"),
             "Upload changed graph parameter blocks without rebuilding topology")
        .def_property_readonly("graph_instance_id", &ScriptableRenderContext::GetGraphInstanceId,
                               "Stable native graph identity for revisioned upload caches")
        .def_property_readonly("output_samples", &ScriptableRenderContext::GetOutputSamples,
                               "Fixed Camera target samples, or zero for pipeline-owned screen MSAA")
        .def("submit_culling", &ScriptableRenderContext::SubmitCulling, py::arg("culling"),
             "Submit all culling results as full draw calls (filtering done by graph pass callbacks)")
        .def("render_with_graph", &ScriptableRenderContext::RenderWithGraph, py::arg("description"),
             py::arg("updates") = std::vector<GraphParameterBlockUpdate>{},
             "Apply changed topology, publish values and render this camera in one call")
        .def("is_graph_revision_current", &ScriptableRenderContext::IsGraphRevisionCurrent, py::arg("source_revision"),
             "Check whether the active graph already owns a Python artifact revision")
        // CommandBuffer integration
        .def("execute_command_buffer", &ScriptableRenderContext::ExecuteCommandBuffer, py::arg("cmd"),
             "Execute a deferred CommandBuffer (commands are buffered until submit)")
        .def("get_camera_target", &ScriptableRenderContext::GetCameraTarget,
             "Get a handle representing the final camera render target");

    // ---- RenderPipelineCallback (abstract, Python inherits via trampoline) ----
    py::class_<RenderPipelineCallback, PyRenderPipelineCallback, std::shared_ptr<RenderPipelineCallback>>(
        m, "RenderPipelineCallback")
        .def(py::init<>())
        .def("should_render_view", &RenderPipelineCallback::ShouldRenderView, py::arg("view"),
             "Select participation before capturing a new World frame; False is a legal omission")
        .def("render", &RenderPipelineCallback::Render, py::arg("context"), py::arg("view"),
             "Render one camera through its dedicated context and RenderView")
        .def("dispose", &RenderPipelineCallback::Dispose, "Called when the pipeline is being replaced");
}
} // namespace infernux
