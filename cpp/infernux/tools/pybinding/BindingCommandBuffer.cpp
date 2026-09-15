/**
 * @file BindingCommandBuffer.cpp
 * @brief pybind11 bindings for CommandBuffer + RenderTargetHandle + enhanced SRC.
 *
 * Part of the deferred command-buffer binding surface.
 *
 * Exposes the deferred-recording CommandBuffer API to Python, allowing
 * users to write custom render pipelines with explicit render-target commands.
 */

#include "MatrixPyBridge.h"
#include <function/renderer/CommandBuffer.h>
#include <function/renderer/RendererSelection.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/scene/GameObject.h>
#include <function/scene/MeshRenderer.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace infernux;
namespace py = pybind11;

namespace infernux
{

void RegisterCommandBufferBindings(py::module_ &m)
{
    py::class_<DrawParameterBlock>(m, "DrawParameterBlock", "Mutable values captured by one explicit draw_mesh command")
        .def(py::init<>())
        .def("set_float", &DrawParameterBlock::SetFloat, py::arg("name"), py::arg("value"))
        .def("set_int", &DrawParameterBlock::SetInt, py::arg("name"), py::arg("value"))
        .def(
            "set_vector2",
            [](DrawParameterBlock &self, const std::string &name, py::sequence value) {
                if (py::len(value) != 2)
                    throw py::value_error("set_vector2 requires exactly 2 numbers");
                self.SetVector2(name, {value[0].cast<float>(), value[1].cast<float>()});
            },
            py::arg("name"), py::arg("value"))
        .def(
            "set_vector3",
            [](DrawParameterBlock &self, const std::string &name, py::sequence value) {
                if (py::len(value) != 3)
                    throw py::value_error("set_vector3 requires exactly 3 numbers");
                self.SetVector3(name, {value[0].cast<float>(), value[1].cast<float>(), value[2].cast<float>()});
            },
            py::arg("name"), py::arg("value"))
        .def(
            "set_vector4",
            [](DrawParameterBlock &self, const std::string &name, py::sequence value) {
                if (py::len(value) != 4)
                    throw py::value_error("set_vector4 requires exactly 4 numbers");
                self.SetVector4(name, {value[0].cast<float>(), value[1].cast<float>(), value[2].cast<float>(),
                                       value[3].cast<float>()});
            },
            py::arg("name"), py::arg("value"))
        .def(
            "set_color",
            [](DrawParameterBlock &self, const std::string &name, py::sequence value) {
                if (py::len(value) != 4)
                    throw py::value_error("set_color requires exactly 4 numbers");
                self.SetColor(name, {value[0].cast<float>(), value[1].cast<float>(), value[2].cast<float>(),
                                     value[3].cast<float>()});
            },
            py::arg("name"), py::arg("value"))
        .def(
            "set_matrix",
            [](DrawParameterBlock &self, const std::string &name, py::handle value) {
                self.SetMatrix(name, binding::Matrix4FromPython(value, "Draw matrix", true));
            },
            py::arg("name"), py::arg("value"))
        .def("set_texture", &DrawParameterBlock::SetTexture, py::arg("name"), py::arg("texture_guid"))
        .def("remove", &DrawParameterBlock::Remove, py::arg("name"))
        .def("clear", &DrawParameterBlock::Clear)
        .def_property_readonly("size", &DrawParameterBlock::Size);

    const auto rendererIdentity = [](const MeshRenderer &renderer) {
        const auto *owner = renderer.GetGameObject();
        if (!owner)
            throw py::value_error("RendererSelection requires a live scene renderer");
        return RenderProxyHandle::FromScene(owner->GetHandle(), renderer.GetHandle());
    };
    py::class_<RendererSelection, std::shared_ptr<RendererSelection>>(m, "RendererSelection")
        .def(py::init<std::shared_ptr<InxMaterial>>(), py::arg("material"))
        .def(
            "set",
            [rendererIdentity](RendererSelection &self, const MeshRenderer &renderer, int submesh,
                               const DrawParameterBlock *parameters) {
                self.Set(rendererIdentity(renderer), submesh, parameters);
            },
            py::arg("renderer"), py::arg("submesh") = -1, py::arg("parameters") = nullptr)
        .def(
            "remove",
            [rendererIdentity](RendererSelection &self, const MeshRenderer &renderer, int submesh) {
                return self.Remove(rendererIdentity(renderer), submesh);
            },
            py::arg("renderer"), py::arg("submesh") = -1)
        .def("clear", &RendererSelection::Clear)
        .def_property_readonly("size", &RendererSelection::Size)
        .def_property_readonly("revision", &RendererSelection::Revision);

    // ---- RenderTargetHandle ----
    py::class_<RenderTargetHandle>(m, "RenderTargetHandle", "Opaque handle to a temporary or persistent render target")
        .def(py::init<>())
        .def_readonly("id", &RenderTargetHandle::id, "Internal handle ID")
        .def("is_valid", &RenderTargetHandle::IsValid, "Check if this handle refers to a valid render target")
        .def("__repr__",
             [](const RenderTargetHandle &h) {
                 return "<RenderTargetHandle id=" + std::to_string(h.id) + (h.IsValid() ? " valid" : " invalid") + ">";
             })
        .def("__eq__", &RenderTargetHandle::operator==)
        .def("__ne__", &RenderTargetHandle::operator!=);

    // Expose the CAMERA_TARGET_HANDLE sentinel
    m.attr("CAMERA_TARGET") = CAMERA_TARGET_HANDLE;

    // ---- CommandBuffer ----
    py::class_<CommandBuffer>(m, "CommandBuffer",
                              "Deferred-recording command buffer for the Scriptable Render Pipeline.\n"
                              "\n"
                              "Commands are recorded but not immediately executed. Call\n"
                              "context.execute_command_buffer(cmd) to schedule execution,\n"
                              "then context.submit() to finalize the frame.\n"
                              "\n"
                              "Example::\n"
                              "\n"
                              "    cmd = CommandBuffer('ForwardRenderer')\n"
                              "    rt = cmd.get_temporary_rt(1920, 1080)\n"
                              "    cmd.set_render_target(rt)\n"
                              "    cmd.clear_render_target(True, True, 0.1, 0.1, 0.1, 1.0)\n"
                              "    cmd.draw_renderers(culling, drawing, filtering)\n"
                              "    cmd.release_temporary_rt(rt)\n"
                              "    context.execute_command_buffer(cmd)\n")
        .def(py::init<const std::string &>(), py::arg("name") = "",
             "Create a CommandBuffer with an optional debug name")

        // ---- Render Target Management ----
        .def("get_temporary_rt", &CommandBuffer::GetTemporaryRT, py::arg("width"), py::arg("height"),
             py::arg("format") = rhi::PixelFormat::RGBA8UNorm, py::arg("samples") = rhi::SampleCount::One,
             "Allocate a temporary render target (lazily created at execution time)")
        .def("release_temporary_rt", &CommandBuffer::ReleaseTemporaryRT, py::arg("handle"),
             "Mark a temporary render target for release (returned to pool at frame end)")
        .def(
            "set_render_target", [](CommandBuffer &self, RenderTargetHandle color) { self.SetRenderTarget(color); },
            py::arg("color"), "Set the active color render target")
        .def(
            "set_render_target_with_depth",
            [](CommandBuffer &self, RenderTargetHandle color, RenderTargetHandle depth) {
                self.SetRenderTarget(color, depth);
            },
            py::arg("color"), py::arg("depth"), "Set active color + depth render targets")
        .def("clear_render_target", &CommandBuffer::ClearRenderTarget, py::arg("clear_color"), py::arg("clear_depth"),
             py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"), py::arg("depth") = 1.0f,
             "Clear the currently-bound render target")
        .def(
            "draw_mesh",
            [](CommandBuffer &self, const std::shared_ptr<InxMesh> &mesh, py::handle matrix,
               const std::shared_ptr<InxMaterial> &material, int submeshIndex, int pass,
               const DrawParameterBlock *parameters) {
                self.DrawMesh(mesh, binding::Matrix4FromPython(matrix, "draw_mesh matrix", true), material,
                              submeshIndex, pass, parameters);
            },
            py::arg("mesh"), py::arg("matrix"), py::arg("material"), py::arg("submesh") = 0, py::arg("pass_index") = 0,
            py::arg("parameters") = nullptr,
            "Record one explicit mesh draw from the material's primary pass; parameter values are captured immediately")

        // ---- Misc ----
        .def("clear", &CommandBuffer::Clear, "Discard all recorded commands (reuse the buffer)")
        .def_property_readonly("name", &CommandBuffer::GetName, "Debug name of this CommandBuffer")
        .def_property_readonly("command_count", &CommandBuffer::GetCommandCount, "Number of recorded commands")
        .def("__repr__", [](const CommandBuffer &self) {
            return "<CommandBuffer '" + self.GetName() + "' commands=" + std::to_string(self.GetCommandCount()) + ">";
        });
}

} // namespace infernux
