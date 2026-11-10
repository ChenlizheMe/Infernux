#include "BindingRegistration.h"
#include <core/math/Random.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace infernux
{
void RegisterRandomBindings(py::module_ &module)
{
    py::class_<RandomState>(module, "RandomState")
        .def(py::init<>())
        .def(py::init([](const std::array<uint32_t, 4> &words) { return RandomState{words}; }), py::arg("words"))
        .def_property_readonly("words", [](const RandomState &state) { return state.words; })
        .def("__eq__", &RandomState::operator==, py::is_operator())
        .def(py::pickle([](const RandomState &state) { return py::cast(state.words); },
                        [](const py::object &value) { return RandomState{value.cast<std::array<uint32_t, 4>>()}; }));

    py::class_<Random>(module, "Random")
        .def_static("init_state", &Random::InitState, py::arg("seed"))
        .def_property_static(
            "state", [](const py::object &) { return Random::GetState(); },
            [](const py::object &, const RandomState &state) { Random::SetState(state); })
        .def_property_readonly_static("value", [](const py::object &) { return Random::Value(); })
        .def_static("range", py::overload_cast<int32_t, int32_t>(&Random::Range), py::arg("min_inclusive"),
                    py::arg("max_exclusive"))
        .def_static("range", py::overload_cast<float, float>(&Random::Range), py::arg("min_inclusive"),
                    py::arg("max_inclusive"))
        .def_static("range_many", &Random::RangeMany, py::arg("bounds"), py::call_guard<py::gil_scoped_release>(),
                    "Draw float ranges atomically as one native batch, releasing the GIL.");
}
} // namespace infernux
