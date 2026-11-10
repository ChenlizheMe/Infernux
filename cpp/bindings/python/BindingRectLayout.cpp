#include "BindingSceneInternal.h"
#include "function/scene/RectLayoutBatch.h"
#include "function/scene/Transform.h"
#include <cstring>

namespace py = pybind11;
namespace infernux::scene_binding
{
namespace
{
template <typename T> py::buffer_info Rows(py::buffer buffer, const RectLayoutBatch &batch, bool writable)
{
    auto info = buffer.request(writable);
    const auto count = static_cast<py::ssize_t>(batch.Size());
    const auto channels = static_cast<py::ssize_t>(batch.Channels());
    const auto values = count * channels;
    const bool flat = info.ndim == 1 && info.size == values && (values <= 1 || info.strides[0] == sizeof(T));
    const bool matrix = info.ndim == 2 && info.shape[0] == count && info.shape[1] == channels &&
                        (!count || ((channels <= 1 || info.strides[1] == sizeof(T)) &&
                                    (count <= 1 || info.strides[0] == channels * sizeof(T))));
    const auto format = py::format_descriptor<T>::format();
    const bool nativeFormat = info.format == format || info.format == "=" + format || info.format == "@" + format;
    if (info.itemsize != sizeof(T) || !nativeFormat || (!flat && !matrix))
        throw std::invalid_argument("Rectangle fields require a matching contiguous typed row buffer");
    return info;
}

template <typename T> void ReadRows(const RectLayoutBatch &batch, py::buffer buffer)
{
    auto info = Rows<T>(buffer, batch, true);
    batch.Validate();
    auto *output = static_cast<char *>(info.ptr);
    for (size_t row = 0; row < batch.Size(); ++row) {
        const auto value = batch.Read(row);
        for (size_t channel = 0; channel < batch.Channels(); ++channel) {
            const T item = static_cast<T>(value[channel]);
            std::memcpy(output + (row * batch.Channels() + channel) * sizeof(T), &item, sizeof(T));
        }
    }
}

template <typename T> void WriteRows(RectLayoutBatch &batch, py::buffer buffer)
{
    auto info = Rows<T>(buffer, batch, false);
    batch.Validate(); // Reject every expired owner before the first scene write.
    const auto *input = static_cast<const char *>(info.ptr);
    for (size_t row = 0; row < batch.Size(); ++row) {
        std::array<float, 3> value{};
        for (size_t channel = 0; channel < batch.Channels(); ++channel) {
            T item;
            std::memcpy(&item, input + (row * batch.Channels() + channel) * sizeof(T), sizeof(T));
            value[channel] = static_cast<float>(item);
        }
        batch.Write(row, value);
    }
}
} // namespace

void BindRectLayout(py::module_ &m)
{
    m.def("_rect_layout_input_revision", [] { return TransformECSStore::Instance().RectLayoutInputRevision(); });
    py::class_<ui::RectTransformState>(m, "_RectLayoutState");
    py::class_<RectTransformLayout>(m, "_RectTransformLayout")
        .def_property("anchor_min", &RectTransformLayout::GetAnchorMin, &RectTransformLayout::SetAnchorMin)
        .def_property("anchor_max", &RectTransformLayout::GetAnchorMax, &RectTransformLayout::SetAnchorMax)
        .def_property("pivot", &RectTransformLayout::GetPivot, &RectTransformLayout::SetPivot)
        .def_property("size_delta", &RectTransformLayout::GetSizeDelta, &RectTransformLayout::SetSizeDelta)
        .def_property("anchored_position", &RectTransformLayout::GetAnchoredPosition,
                      &RectTransformLayout::SetAnchoredPosition)
        .def_property("anchored_position3d", &RectTransformLayout::GetAnchoredPosition3D,
                      &RectTransformLayout::SetAnchoredPosition3D)
        .def_property(
            "_planar_position",
            [](const RectTransformLayout &layout) { return glm::vec3(layout.GetAnchoredPosition(), 0.f); },
            [](RectTransformLayout &layout, glm::vec3 value) { layout.SetAnchoredPosition({value.x, value.y}); })
        .def_property("offset_min", &RectTransformLayout::GetOffsetMin, &RectTransformLayout::SetOffsetMin)
        .def_property("offset_max", &RectTransformLayout::GetOffsetMax, &RectTransformLayout::SetOffsetMax)
        .def_property_readonly("rect", &RectTransformLayout::GetRect)
        .def_property_readonly("_authoring_rect", &RectTransformLayout::GetAuthoringRect)
        .def_property_readonly("_driver", &RectTransformLayout::GetDriver, py::return_value_policy::reference)
        .def("_drive_size", &RectTransformLayout::DriveSize, py::arg("driver").none(false), py::arg("size"),
             py::arg("size_mask") = 3)
        .def("_drive_rect", &RectTransformLayout::DriveRect, py::arg("driver").none(false), py::arg("size"),
             py::arg("local_position"))
        .def("_clear_driver", &RectTransformLayout::ClearDriver, py::arg("driver").none(false))
        .def("_set_authoring_size", &RectTransformLayout::SetAuthoringSizeWithCurrentAnchors)
        .def("_set_authoring_local_position", &RectTransformLayout::SetAuthoringLocalPosition)
        .def_property_readonly("revision", &RectTransformLayout::GetRevision)
        .def_property("_position_scale", &RectTransformLayout::GetPositionScale, &RectTransformLayout::SetPositionScale)
        .def_property("_unit_mode", &RectTransformLayout::GetUnitMode, &RectTransformLayout::SetUnitMode)
        .def("_resolve_parent", &RectTransformLayout::ResolveParent, py::arg("size"), py::arg("pivot"))
        .def("get_local_corners", &RectTransformLayout::GetLocalCorners)
        .def("get_world_corners", &RectTransformLayout::GetWorldCorners)
        .def("set_size_with_current_anchors", &RectTransformLayout::SetSizeWithCurrentAnchors, py::arg("axis"),
             py::arg("size"))
        .def("set_inset_and_size_from_parent_edge", &RectTransformLayout::SetInsetAndSizeFromParentEdge,
             py::arg("edge"), py::arg("inset"), py::arg("size"));
    py::class_<RectLayoutBatch>(m, "_RectLayoutBatch")
        .def(py::init<std::vector<RectTransformLayout>, const std::string &>(), py::arg("layouts"), py::arg("field"))
        .def_property_readonly("size", &RectLayoutBatch::Size)
        .def_property_readonly("channels", &RectLayoutBatch::Channels)
        .def("read_into",
             [](const RectLayoutBatch &batch, py::buffer output) {
                 if (batch.Channels() == 1)
                     ReadRows<double>(batch, output);
                 else
                     ReadRows<float>(batch, output);
             })
        .def("write", [](RectLayoutBatch &batch, py::buffer input) {
            if (batch.Channels() == 1)
                WriteRows<double>(batch, input);
            else
                WriteRows<float>(batch, input);
        });
}
} // namespace infernux::scene_binding
