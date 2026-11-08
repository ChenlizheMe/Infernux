#include "NumericBuffers.h"
#include "PythonEase.h"
#include "infernux/core/math/VectorTween.h"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>

namespace py = pybind11;
namespace infernux
{
namespace
{
struct ColorChannels
{
    std::array<float, 4> values;
    explicit ColorChannels(std::array<float, 4> input) : values(input)
    {
    }
    float operator[](size_t axis) const
    {
        return values[axis];
    }
};

struct ColorPairChannels
{
    ColorChannels a, b;
    ColorPairChannels(std::array<float, 4> first, std::array<float, 4> second) : a(first), b(second)
    {
    }
    float operator[](size_t axis) const
    {
        return axis < 4 ? a[axis] : b[axis - 4];
    }
    std::array<float, 8> Values() const
    {
        return {a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
    }
    ColorPairChannels Add(const ColorPairChannels &other) const
    {
        ColorPairChannels result(*this);
        for (size_t i = 0; i < 4; ++i) {
            result.a.values[i] += other.a[i];
            result.b.values[i] += other.b[i];
        }
        return result;
    }
    ColorPairChannels Subtract(const ColorPairChannels &other) const
    {
        ColorPairChannels result(*this);
        for (size_t i = 0; i < 4; ++i) {
            result.a.values[i] -= other.a[i];
            result.b.values[i] -= other.b[i];
        }
        return result;
    }
    ColorPairChannels Multiply(float factor) const
    {
        ColorPairChannels result(*this);
        for (size_t i = 0; i < 4; ++i) {
            result.a.values[i] *= factor;
            result.b.values[i] *= factor;
        }
        return result;
    }
    bool Equals(const ColorPairChannels &other) const
    {
        for (size_t i = 0; i < 8; ++i) {
            const float first = (*this)[i], second = other[i];
            if (first != second && !(std::isnan(first) && std::isnan(second)))
                return false;
        }
        return true;
    }
};

template <typename Vector, size_t Channels> struct VectorState
{
    py::object startOwner, changeOwner;
    tween::LiveVectorValues<Vector, Channels> values;
    VectorState(py::object start, py::object change, bool snapping)
        : startOwner(std::move(start)), changeOwner(std::move(change)),
          values(startOwner.cast<const Vector &>(), changeOwner.cast<const Vector &>(), snapping)
    {
    }
};
template <typename Vector, size_t Channels> struct VectorValues
{
    std::shared_ptr<VectorState<Vector, Channels>> state;
};
template <typename Vector, size_t Channels> struct VectorEvaluator
{
    std::shared_ptr<VectorState<Vector, Channels>> state;
    tween::EaseKernel ease;
    double duration;
};
template <typename Vector, size_t Channels> auto ToValue(const std::array<float, Channels> &value)
{
    if constexpr (std::is_same_v<Vector, ColorChannels> || std::is_same_v<Vector, ColorPairChannels>) {
        return value;
    } else {
        Vector result;
        for (size_t axis = 0; axis < Channels; ++axis)
            result[axis] = value[axis];
        return result;
    }
}

template <typename Vector, size_t Channels>
void BindLiveVector(py::module_ &module, const char *snapshotName, const char *evaluatorName)
{
    using Values = VectorValues<Vector, Channels>;
    using Evaluator = VectorEvaluator<Vector, Channels>;
    py::class_<Evaluator>(module, evaluatorName)
        .def(
            "__call__",
            [](const Evaluator &self, double progress, double own) {
                return ToValue<Vector>(self.state->values.Interpolate(self.ease.Sample(progress, self.duration), own));
            },
            py::arg("progress"), py::arg("incremental") = 0.)
        .def("sample_nested",
             [](const Evaluator &self, double progress, double own, double multiplier, double parent) {
                 return ToValue<Vector>(self.state->values.Interpolate(self.ease.Sample(progress, self.duration), own,
                                                                       multiplier, parent));
             })
        .def(
            "sample_into",
            [](const Evaluator &self, py::buffer progress, py::buffer output, double own, double multiplier,
               double parent) {
                const bindings::NumericBuffers<double, float, Channels> buffers(progress, output);
                const auto frozen = self.state->values.Frozen();
                const bindings::NumericWorkScope work(buffers.count);
                for (py::ssize_t i = 0; i < buffers.count; ++i) {
                    const auto value =
                        frozen.Interpolate(self.ease.Sample(buffers.Read(i), self.duration), own, multiplier, parent);
                    buffers.Write(i, value.data());
                }
            },
            py::arg("progress"), py::arg("values"), py::arg("incremental") = 0., py::arg("parent_multiplier") = 1.,
            py::arg("parent_incremental") = 0.);
    py::class_<Values>(module, snapshotName)
        .def(py::init([](py::object start, py::object change, bool snapping) {
                 return Values{
                     std::make_shared<VectorState<Vector, Channels>>(std::move(start), std::move(change), snapping)};
             }),
             py::arg("start"), py::arg("delta"), py::arg("snapping") = false)
        .def("bind", [](const Values &self, const tween::EaseKernel &ease,
                        double duration) { return Evaluator{self.state, ease, duration}; })
        .def("sample",
             [](const Values &self, double progress, double duration, const tween::EaseKernel &ease, double own) {
                 return ToValue<Vector>(self.state->values.Interpolate(ease.Sample(progress, duration), own));
             })
        .def(
            "interpolate",
            [](const Values &self, double factor, double own, double multiplier, double parent) {
                return ToValue<Vector>(self.state->values.Interpolate(factor, own, multiplier, parent));
            },
            py::arg("factor"), py::arg("incremental") = 0., py::arg("parent_multiplier") = 1.,
            py::arg("parent_incremental") = 0.)
        .def(
            "sample_custom",
            [](const Values &self, const py::function &ease, const py::object &progress, double own, double multiplier,
               double parent) {
                const auto frozen = self.state->values.Frozen();
                const double factor = bindings::SamplePythonEase(ease, progress);
                return ToValue<Vector>(frozen.Interpolate(factor, own, multiplier, parent));
            },
            py::arg("ease"), py::arg("progress"), py::arg("incremental") = 0., py::arg("parent_multiplier") = 1.,
            py::arg("parent_incremental") = 0.);
}
} // namespace

void RegisterVectorTweenBindings(py::module_ &module)
{
    py::class_<ColorPairChannels>(module, "_TweenColorPairChannels")
        .def(py::init<const ColorPairChannels &>())
        .def(py::init<std::array<float, 4>, std::array<float, 4>>(), py::arg("ca"), py::arg("cb"))
        .def_property(
            "ca", [](ColorPairChannels &self) -> ColorChannels & { return self.a; },
            [](ColorPairChannels &self, std::array<float, 4> value) { self.a.values = value; },
            py::return_value_policy::reference_internal)
        .def_property(
            "cb", [](ColorPairChannels &self) -> ColorChannels & { return self.b; },
            [](ColorPairChannels &self, std::array<float, 4> value) { self.b.values = value; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("values", &ColorPairChannels::Values)
        .def("add", &ColorPairChannels::Add)
        .def("subtract", &ColorPairChannels::Subtract)
        .def("multiply", &ColorPairChannels::Multiply)
        .def("equals", &ColorPairChannels::Equals);
    py::class_<ColorChannels>(module, "_TweenColorChannels")
        .def(py::init<const ColorChannels &>())
        .def(py::init<std::array<float, 4>>(), py::arg("values"))
        .def_property_readonly("values", [](const ColorChannels &self) { return self.values; })
        .def("set",
             [](ColorChannels &self, py::ssize_t index, float value) {
                 if (index < 0)
                     index += 4;
                 if (index < 0 || index >= 4)
                     throw std::out_of_range("color channel index out of range");
                 self.values[static_cast<size_t>(index)] = value;
                 return value;
             })
        .def("assign", [](ColorChannels &self, std::array<float, 4> values) { self.values = values; })
        .def(
            "__iter__",
            [](const ColorChannels &self) { return py::make_iterator(self.values.begin(), self.values.end()); },
            py::keep_alive<0, 1>());
    BindLiveVector<glm::vec2, 2>(module, "_LiveVector2TweenSnapshot", "_LiveVector2TweenEvaluator");
    BindLiveVector<glm::vec3, 3>(module, "_LiveVector3TweenSnapshot", "_LiveVector3TweenEvaluator");
    BindLiveVector<glm::vec4, 4>(module, "_LiveVector4TweenSnapshot", "_LiveVector4TweenEvaluator");
    BindLiveVector<ColorChannels, 4>(module, "_LiveColorTweenSnapshot", "_LiveColorTweenEvaluator");
    BindLiveVector<ColorPairChannels, 8>(module, "_LiveColor2TweenSnapshot", "_LiveColor2TweenEvaluator");
}
} // namespace infernux
