#include "BindingRegistration.h"
#include <array>
#include <core/math/ImageFill.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

void infernux::RegisterRuntimeNumericsBindings(py::module_ &module)
{
    module.def(
        "_image_fill_geometry",
        [](float amount, int method, int origin, bool clockwise) {
            const auto geometry = imagefill::Build(amount, method, origin, clockwise);
            std::vector<std::vector<std::array<float, 2>>> result;
            result.reserve(geometry.count);
            for (size_t polygonIndex = 0; polygonIndex < geometry.count; ++polygonIndex) {
                const auto &source = geometry.polygons[polygonIndex];
                auto &polygon = result.emplace_back();
                polygon.reserve(source.count);
                for (size_t vertex = 0; vertex < source.count; ++vertex) {
                    const auto point = source.points[vertex];
                    polygon.push_back({point.x, point.y});
                }
            }
            return result;
        },
        py::arg("amount"), py::arg("method"), py::arg("origin"), py::arg("clockwise"),
        py::call_guard<py::gil_scoped_release>());
}
