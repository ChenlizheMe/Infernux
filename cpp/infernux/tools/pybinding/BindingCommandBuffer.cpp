/**
 * @file BindingCommandBuffer.cpp
 * @brief pybind11 bindings for CommandBuffer + RenderTargetHandle + enhanced SRC.
 *
 * Part of the deferred command-buffer binding surface.
 *
 * Exposes the deferred-recording CommandBuffer API to Python, allowing
 * users to write custom render pipelines with full control over render
 * targets, global shader parameters, and async readback.
 */

#include <function/renderer/CommandBuffer.h>
#include <function/resources/InxMaterial/InxMaterial.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace infernux;
namespace py = pybind11;

namespace infernux
{

void RegisterCommandBufferBindings(py::module_ &m)
{
    // ---- VkFormat enum subset (commonly needed for RT creation) ----
    // NOTE: Must be registered BEFORE CommandBuffer, because VkFormat values
    //       are used as default arguments in get_temporary_rt().
    py::enum_<VkFormat>(m, "VkFormat", "Vulkan image format (subset for render target creation)")
        .value("R8G8B8A8_UNORM", VK_FORMAT_R8G8B8A8_UNORM)
        .value("R8G8B8A8_SRGB", VK_FORMAT_R8G8B8A8_SRGB)
        .value("B8G8R8A8_UNORM", VK_FORMAT_B8G8R8A8_UNORM)
        .value("R16G16B16A16_SFLOAT", VK_FORMAT_R16G16B16A16_SFLOAT)
        .value("R32G32B32A32_SFLOAT", VK_FORMAT_R32G32B32A32_SFLOAT)
        .value("R32_SFLOAT", VK_FORMAT_R32_SFLOAT)
        .value("R8_UNORM", VK_FORMAT_R8_UNORM)
        .value("R8G8_UNORM", VK_FORMAT_R8G8_UNORM)
        .value("R16G16_SFLOAT", VK_FORMAT_R16G16_SFLOAT)
        .value("A2R10G10B10_UNORM_PACK32", VK_FORMAT_A2R10G10B10_UNORM_PACK32)
        .value("R16_SFLOAT", VK_FORMAT_R16_SFLOAT)
        .value("D32_SFLOAT", VK_FORMAT_D32_SFLOAT)
        .value("D24_UNORM_S8_UINT", VK_FORMAT_D24_UNORM_S8_UINT)
        .export_values();

    // ---- VkSampleCountFlagBits enum subset ----
    py::enum_<VkSampleCountFlagBits>(m, "VkSampleCount", "Vulkan MSAA sample count")
        .value("COUNT_1", VK_SAMPLE_COUNT_1_BIT)
        .value("COUNT_2", VK_SAMPLE_COUNT_2_BIT)
        .value("COUNT_4", VK_SAMPLE_COUNT_4_BIT)
        .value("COUNT_8", VK_SAMPLE_COUNT_8_BIT)
        .export_values();

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
             py::arg("format") = VK_FORMAT_R8G8B8A8_UNORM, py::arg("samples") = VK_SAMPLE_COUNT_1_BIT,
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

        // ---- Global Shader Parameters ----
        .def("set_global_texture", &CommandBuffer::SetGlobalTexture, py::arg("name"), py::arg("handle"),
             "Set a global texture shader parameter by name")
        .def("set_global_float", &CommandBuffer::SetGlobalFloat, py::arg("name"), py::arg("value"),
             "Set a global float shader parameter by name")
        .def("set_global_vector", &CommandBuffer::SetGlobalVector, py::arg("name"), py::arg("x"), py::arg("y"),
             py::arg("z"), py::arg("w"), "Set a global vec4 shader parameter by name")
        .def(
            "set_global_matrix",
            [](CommandBuffer &self, const std::string &name, py::list data) {
                if (py::len(data) != 16) {
                    throw std::runtime_error("set_global_matrix requires a list of 16 floats");
                }
                std::array<float, 16> arr;
                for (int i = 0; i < 16; i++)
                    arr[i] = data[i].cast<float>();
                self.SetGlobalMatrix(name, arr);
            },
            py::arg("name"), py::arg("data"),
            "Set a global 4x4 matrix shader parameter (list of 16 floats, column-major)")

        // ---- Async Readback ----
        .def("request_async_readback", &CommandBuffer::RequestAsyncReadback, py::arg("handle"), py::arg("callback_id"),
             "Request an asynchronous GPU→CPU readback of a render target.\n"
             "The result can be retrieved later via the callback ID.")

        // ---- Misc ----
        .def("clear", &CommandBuffer::Clear, "Discard all recorded commands (reuse the buffer)")
        .def_property_readonly("name", &CommandBuffer::GetName, "Debug name of this CommandBuffer")
        .def_property_readonly("command_count", &CommandBuffer::GetCommandCount, "Number of recorded commands")
        .def("__repr__", [](const CommandBuffer &self) {
            return "<CommandBuffer '" + self.GetName() + "' commands=" + std::to_string(self.GetCommandCount()) + ">";
        });
}

} // namespace infernux
