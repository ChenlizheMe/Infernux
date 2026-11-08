#include "NumericBuffers.h"
#include "PythonEase.h"
#include "infernux/core/math/EulerTween.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infernux
{
namespace
{
using EulerValues = tween::LiveEulerValues<glm::vec3>;
glm::quat Quaternion(const EulerValues::Value &v)
{
    return {v[3], v[0], v[1], v[2]};
}

struct EulerState
{
    py::object startOwner, changeOwner, endOwner, upOwner, pointOwner;
    glm::vec3 &start, &change, &end, &up, &point;
    EulerValues native;
    int constraint;
    bool relative;
    EulerState(py::object first, py::object delta, py::object last, py::object vertical, py::object target, int mode,
               int axis, bool isRelative)
        : startOwner(std::move(first)), changeOwner(std::move(delta)), endOwner(std::move(last)),
          upOwner(std::move(vertical)), pointOwner(std::move(target)), start(startOwner.cast<glm::vec3 &>()),
          change(changeOwner.cast<glm::vec3 &>()), end(endOwner.cast<glm::vec3 &>()), up(upOwner.cast<glm::vec3 &>()),
          point(pointOwner.cast<glm::vec3 &>()), native(start, change, mode), constraint(axis), relative(isRelative)
    {
    }
    void Prepare(const glm::vec3 &position)
    {
        tween::PrepareEulerLookAt(start, end, change, point, up, position, native.Mode(), constraint, relative);
    }
};
struct BoundEulerValues
{
    std::shared_ptr<EulerState> data;
};
struct BoundEulerEvaluator
{
    std::shared_ptr<EulerState> data;
    tween::EaseKernel ease;
    double duration;
};
} // namespace

void RegisterEulerTweenBindings(py::module_ &module)
{
    py::class_<BoundEulerEvaluator>(module, "_EulerTweenEvaluator")
        .def(
            "__call__",
            [](const BoundEulerEvaluator &self, double progress, double own) {
                return Quaternion(self.data->native.Interpolate(self.ease.Sample(progress, self.duration), own));
            },
            py::arg("progress"), py::arg("incremental") = 0.)
        .def("sample_nested",
             [](const BoundEulerEvaluator &self, double progress, double own, double multiplier, double parent) {
                 return Quaternion(
                     self.data->native.Interpolate(self.ease.Sample(progress, self.duration), own, multiplier, parent));
             })
        .def(
            "sample_into",
            [](const BoundEulerEvaluator &self, py::buffer progress, py::buffer values, double own, double multiplier,
               double parent) {
                const bindings::NumericBuffers<double, float, 4> buffers(progress, values);
                const auto frozen = self.data->native.Frozen();
                const bindings::NumericWorkScope work(buffers.count);
                for (py::ssize_t i = 0; i < buffers.count; ++i) {
                    const auto value =
                        frozen.Interpolate(self.ease.Sample(buffers.Read(i), self.duration), own, multiplier, parent);
                    buffers.Write(i, value.data());
                }
            },
            py::arg("progress"), py::arg("values"), py::arg("incremental") = 0., py::arg("parent_multiplier") = 1.,
            py::arg("parent_incremental") = 0.);
    py::class_<BoundEulerValues>(module, "_EulerTweenSnapshot")
        .def(py::init([](py::object start, py::object change, py::object end, py::object up, py::object point, int mode,
                         int constraint, bool relative) {
                 return BoundEulerValues{std::make_shared<EulerState>(std::move(start), std::move(change),
                                                                      std::move(end), std::move(up), std::move(point),
                                                                      mode, constraint, relative)};
             }),
             py::arg("start"), py::arg("change"), py::arg("end"), py::arg("up"), py::arg("point"), py::arg("mode") = 0,
             py::arg("constraint") = 0, py::arg("relative") = false)
        .def("bind", [](const BoundEulerValues &self, const tween::EaseKernel &ease,
                        double duration) { return BoundEulerEvaluator{self.data, ease, duration}; })
        .def("prepare_look_at",
             [](const BoundEulerValues &self, const glm::vec3 &position) { self.data->Prepare(position); })
        .def(
            "interpolate",
            [](const BoundEulerValues &self, double factor, double own, double multiplier, double parent) {
                return Quaternion(self.data->native.Interpolate(factor, own, multiplier, parent));
            },
            py::arg("factor"), py::arg("incremental") = 0., py::arg("parent_multiplier") = 1.,
            py::arg("parent_incremental") = 0.)
        .def(
            "sample_custom",
            [](const BoundEulerValues &self, const py::function &ease, const py::object &progress, double own,
               double multiplier, double parent) {
                const auto frozen = self.data->native.Frozen();
                const double factor = bindings::SamplePythonEase(ease, progress);
                return Quaternion(frozen.Interpolate(factor, own, multiplier, parent));
            },
            py::arg("ease"), py::arg("progress"), py::arg("incremental") = 0., py::arg("parent_multiplier") = 1.,
            py::arg("parent_incremental") = 0.)
        .def(
            "sample_custom_time",
            [](const BoundEulerValues &self, const py::function &ease, double elapsed, double duration, double own,
               double multiplier, double parent) {
                const auto frozen = self.data->native.Frozen();
                const double factor = bindings::SamplePythonEase(ease, elapsed, duration);
                return Quaternion(frozen.Interpolate(factor, own, multiplier, parent));
            },
            py::arg("ease"), py::arg("elapsed"), py::arg("duration"), py::arg("incremental") = 0.,
            py::arg("parent_multiplier") = 1., py::arg("parent_incremental") = 0.);
    module.def("_euler_tween_change", [](const glm::vec3 &start, const glm::vec3 &end, int mode, bool relative) {
        const auto v = tween::EulerChange(start, end, mode, relative);
        return glm::vec3(v[0], v[1], v[2]);
    });
    module.def("_euler_tween_speed",
               [](const glm::vec3 &v, float units) { return std::sqrt((v.x * v.x + v.y * v.y) + v.z * v.z) / units; });
}
} // namespace infernux
