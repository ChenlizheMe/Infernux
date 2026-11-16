/**
 * @file BindingRenderGraph.cpp
 * @brief Python bindings for RenderGraph and pass output access
 *
 * Provides Python-side control over the render graph, allowing:
 * - Configuration of render passes
 * - Reading pass output pixels to NumPy arrays
 * - Getting pass texture IDs for ImGui display
 * - Render graph topology definition from Python
 */

#include "BindingRegistration.h"
#include "Infernux.h"
#include "function/renderer/RenderGraphDescription.h"
#include "function/renderer/SceneRenderGraph.h"
#include <function/scene/PythonNativeLifetime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infernux
{

void RegisterRenderGraphBindings(py::module_ &m)
{
    m.def("_watch_graph_events", [](Infernux &engine, const std::string &label) {
        auto *renderer = engine.GetRenderer();
        if (!renderer)
            throw std::logic_error("Render graph events require an active graphical renderer");
        return renderer->WatchGraphEvents(label);
    });
    m.def("_take_graph_events", [](Infernux &engine) { return engine.GetRenderer()->TakeGraphEvents(); });
    m.def("_is_graph_submission_completed",
          [](const Infernux &engine, std::tuple<rhi::DeviceId, uint8_t, rhi::SubmissionSerial> identity) {
              const auto [device, queue, serial] = identity;
              return engine.IsGpuSubmissionCompleted({device, static_cast<rhi::QueueRole>(queue), serial});
          });

    // SceneRenderGraph class
    NativeClass<SceneRenderGraph>(m, "SceneRenderGraph")
        .def_property_readonly("storage_deferred", &SceneRenderGraph::IsGraphStorageDeferred)
        .def("capture_buffer", &SceneRenderGraph::CaptureBuffer, py::arg("name"))
        // Pass configuration
        .def("mark_dirty", &SceneRenderGraph::MarkDirty, "Force rebuild of the render graph on next frame")
        // Render graph topology defined from Python
        .def("apply_python_graph", &SceneRenderGraph::ApplyPythonGraph, py::arg("description"),
             "Apply a render graph topology defined in Python")
        .def("update_parameter_blocks", &SceneRenderGraph::UpdateParameterBlocks, py::arg("updates"),
             "Upload changed runtime parameter blocks without rebuilding topology")
        .def("has_python_graph", &SceneRenderGraph::HasPythonGraph, "Check if a Python graph topology has been applied")
        .def("get_pass_count", &SceneRenderGraph::GetPassCount, "Get number of configured passes")
        .def("get_debug_string", &SceneRenderGraph::GetDebugString, "Get debug visualization of the render graph");

    // Add RenderGraph access methods to Infernux
    // These are defined in BindingInfernux.cpp but we document them here for clarity:
    // - get_scene_render_graph() -> SceneRenderGraph
}

} // namespace infernux
