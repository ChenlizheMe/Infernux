#include "NumericBuffers.h"
#include "Vector3Array.h"
#include "infernux/core/math/MotionPath.h"
#include "infernux/core/math/WaypointLengths.h"
#include <glm/vec3.hpp>
#include <iomanip>
#include <locale>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <type_traits>

namespace py = pybind11;
namespace infernux
{
namespace
{
using MutablePath = motion::PathDataSnapshot<glm::vec3>;
using Vectors = std::vector<glm::vec3>;

motion::Point Point(const glm::vec3 &value)
{
    return {value.x, value.y, value.z};
}
glm::vec3 Vector(const motion::Point &value)
{
    return {value[0], value[1], value[2]};
}

Vectors CubicPoints(glm::vec3 start, glm::vec3 startControl, glm::vec3 end, glm::vec3 endControl, int resolution)
{
    const int count = std::max(2, resolution);
    const auto p0 = Point(start), p1 = Point(startControl), p2 = Point(endControl), p3 = Point(end);
    const float step = 1.0f / static_cast<float>(count - 1);
    Vectors points(static_cast<size_t>(count));
    const bindings::NumericWorkScope work(static_cast<size_t>(count));
    for (int index = 0; index < count; ++index)
        points[index] = Vector(motion::EvaluateCubicBezier(p0, p1, p2, p3, step * static_cast<float>(index)));
    return points;
}

size_t Index(py::ssize_t index, size_t size)
{
    if (index < 0)
        index += size;
    if (index < 0 || static_cast<size_t>(index) >= size)
        throw py::index_error();
    return static_cast<size_t>(index);
}

struct ControlPoint
{
    std::shared_ptr<Vectors> values;
    size_t index = 0;
    glm::vec3 &A() const
    {
        return values->at(2 * index);
    }
    glm::vec3 &B() const
    {
        return values->at(2 * index + 1);
    }
    static ControlPoint Copy(const glm::vec3 &a, const glm::vec3 &b)
    {
        return {std::make_shared<Vectors>(Vectors{a, b}), 0};
    }
};

struct ControlArray
{
    std::shared_ptr<Vectors> values;
    size_t Size() const
    {
        return values->size() / 2;
    }
    ControlPoint At(size_t index) const
    {
        return {values, index};
    }
};

struct RecipeView
{
    std::shared_ptr<MutablePath> data;
    size_t Size() const
    {
        return data->RecipeSize();
    }
    glm::vec3 &At(size_t index) const
    {
        return data->RecipePoint(index);
    }
};

template <typename View> struct IndexedIterator
{
    using reference = decltype(std::declval<const View &>().At(0));
    using value_type = std::remove_reference_t<reference>;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using iterator_category = std::forward_iterator_tag;
    View view;
    size_t index;
    reference operator*() const
    {
        return view.At(index);
    }
    IndexedIterator &operator++()
    {
        ++index;
        return *this;
    }
    bool operator==(const IndexedIterator &other) const
    {
        return index == other.index;
    }
    bool operator!=(const IndexedIterator &other) const
    {
        return index != other.index;
    }
};

void BindPathViews(py::module_ &module)
{
    py::class_<Float32Array, std::shared_ptr<Float32Array>>(module, "_Float32Array", py::buffer_protocol())
        .def(py::init([](py::iterable input) {
            auto result = std::make_shared<Float32Array>();
            const auto size = PyObject_LengthHint(input.ptr(), 0);
            if (size < 0)
                throw py::error_already_set();
            result->values.reserve(static_cast<size_t>(size));
            for (const auto &value : input)
                result->values.push_back(py::cast<float>(value));
            return result;
        }))
        .def("__len__", [](const Float32Array &self) { return self.values.size(); })
        .def("__getitem__",
             [](const Float32Array &self, py::ssize_t index) { return self.values[Index(index, self.values.size())]; })
        .def("__setitem__", [](Float32Array &self, py::ssize_t index,
                               float value) { self.values[Index(index, self.values.size())] = value; })
        .def("__copy__", [](const Float32Array &self) { return std::make_shared<Float32Array>(self); })
        .def(
            "__iter__",
            [](const Float32Array &self) { return py::make_iterator(self.values.cbegin(), self.values.cend()); },
            py::keep_alive<0, 1>())
        .def_buffer([](Float32Array &self) {
            return py::buffer_info(self.values.data(), sizeof(float), py::format_descriptor<float>::format(), 1,
                                   {self.values.size()}, {sizeof(float)});
        });
    py::class_<ControlPoint>(module, "_MotionControlPoint")
        .def(py::init([] { return ControlPoint::Copy(glm::vec3(0), glm::vec3(0)); }))
        .def(py::init(&ControlPoint::Copy), py::arg("a"), py::arg("b"))
        .def_property(
            "a", &ControlPoint::A, [](const ControlPoint &self, const glm::vec3 &v) { self.A() = v; },
            py::return_value_policy::reference_internal)
        .def_property(
            "b", &ControlPoint::B, [](const ControlPoint &self, const glm::vec3 &v) { self.B() = v; },
            py::return_value_policy::reference_internal)
        .def(
            "__add__",
            [](const ControlPoint &self, const glm::vec3 &offset) {
                return ControlPoint::Copy(self.A() + offset, self.B() + offset);
            },
            py::is_operator())
        .def("__str__",
             [](const ControlPoint &self) {
                 std::ostringstream text;
                 text.imbue(std::locale::classic());
                 text << std::fixed << std::setprecision(2);
                 const auto component = [&text](float value) {
                     if (std::isnan(value))
                         text << "NaN";
                     else if (std::isinf(value))
                         text << (value < 0 ? "-Infinity" : "Infinity");
                     else
                         text << value;
                 };
                 const auto point = [&text, &component](const glm::vec3 &value) {
                     text << '(';
                     component(value.x);
                     text << ", ";
                     component(value.y);
                     text << ", ";
                     component(value.z);
                     text << ')';
                 };
                 text << '[';
                 point(self.A());
                 text << " | ";
                 point(self.B());
                 text << ']';
                 return text.str();
             })
        .def("__copy__", [](const ControlPoint &self) { return ControlPoint::Copy(self.A(), self.B()); });
    py::class_<ControlArray>(module, "_MotionControlPointArray")
        .def(py::init([](py::iterable input) {
            auto values = std::make_shared<Vectors>();
            for (const auto &item : input) {
                const auto &v = py::cast<const ControlPoint &>(item);
                values->push_back(v.A());
                values->push_back(v.B());
            }
            return ControlArray{std::move(values)};
        }))
        .def("__len__", &ControlArray::Size)
        .def("__getitem__",
             [](const ControlArray &self, py::ssize_t index) { return self.At(Index(index, self.Size())); })
        .def("__setitem__",
             [](const ControlArray &self, py::ssize_t index, const ControlPoint &value) {
                 const glm::vec3 a = value.A(), b = value.B();
                 const auto dest = self.At(Index(index, self.Size()));
                 dest.A() = a;
                 dest.B() = b;
             })
        .def(
            "__iter__",
            [](const ControlArray &self) {
                return py::make_iterator<py::return_value_policy::move>(
                    IndexedIterator<ControlArray>{self, 0}, IndexedIterator<ControlArray>{self, self.Size()});
            },
            py::keep_alive<0, 1>())
        .def("__copy__",
             [](const ControlArray &self) { return ControlArray{std::make_shared<Vectors>(*self.values)}; });
    py::class_<RecipeView>(module, "_MotionPathRecipeView")
        .def("__len__", &RecipeView::Size)
        .def(
            "__getitem__",
            [](const RecipeView &self, py::ssize_t index) -> glm::vec3 & { return self.At(Index(index, self.Size())); },
            py::return_value_policy::reference_internal)
        .def("__setitem__", [](const RecipeView &self, py::ssize_t index,
                               const glm::vec3 &value) { self.At(Index(index, self.Size())) = value; })
        .def(
            "__iter__",
            [](const RecipeView &self) {
                return py::make_iterator<py::return_value_policy::reference_internal>(
                    IndexedIterator<RecipeView>{self, 0}, IndexedIterator<RecipeView>{self, self.Size()});
            },
            py::keep_alive<0, 1>());
}
template <typename Storage, bool Mutable> void BindPath(py::module_ &module, const char *name)
{
    using Native = motion::PathDataSnapshot<Storage>;
    using Stable = std::conditional_t<Mutable, Native, const Native &>;
    const auto translate = [](const Native &self, const glm::vec3 &offset, bool incremental) {
        if (incremental && !self.ControlStorage())
            throw motion::MissingPathControls();
        const motion::Point translation{offset.x, offset.y, offset.z};
        Stable stable = self;
        const bindings::NumericWorkScope work(stable.PointCount());
        if constexpr (Mutable)
            return std::move(stable).TranslateOwned(translation);
        else
            return stable.Translated(translation);
    };
    auto type = py::class_<Native, std::shared_ptr<Native>>(module, name);
    type.def(py::init([](std::vector<Storage> points, int kind, int resolution, bool closed) {
                 size_t workSize = points.size();
                 if (kind != 0 && resolution > 1)
                     workSize *= static_cast<size_t>(resolution);
                 const bindings::NumericWorkScope work(workSize);
                 return std::make_shared<Native>(std::move(points), kind, resolution, closed);
             }),
             py::arg("points"), py::arg("kind"), py::arg("resolution"), py::arg("closed"))
        .def_property_readonly("length", &Native::Length)
        .def_property_readonly("segments", &Native::Segments)
        .def_property_readonly("parameters", &Native::Parameters)
        .def_property_readonly("distances", &Native::Distances)
        .def_property_readonly("waypoint_distances", &Native::WaypointDistances)
        .def_property_readonly("waypoint_lengths", &Native::WaypointLengths)
        .def("parameter_at", &Native::ParameterAt)
        .def("sample",
             [](const Native &self, double progress) {
                 const auto value = self.Sample(progress);
                 return glm::vec3(value[0], value[1], value[2]);
             })
        .def("evaluate",
             [](const Native &self, double parameter) {
                 const auto value = self.Evaluate(parameter);
                 return glm::vec3(value[0], value[1], value[2]);
             })
        .def("translated",
             [translate](const Native &self, const glm::vec3 &offset) { return translate(self, offset, false); })
        .def("incremental_clone",
             [translate](const Native &self, const glm::vec3 &offset) { return translate(self, offset, true); })
        .def("sample_many",
             [](const Native &self, const std::vector<double> &progress) {
                 const Stable stable = self;
                 const bindings::NumericWorkScope work(progress.size());
                 std::vector<glm::vec3> output;
                 output.reserve(progress.size());
                 for (double time : progress) {
                     const auto value = stable.Sample(time);
                     output.emplace_back(value[0], value[1], value[2]);
                 }
                 return output;
             })
        .def("sample_into", [](const Native &self, py::buffer progress, py::buffer positions) {
            const bindings::NumericBuffers<double, float, 3> buffers(progress, positions);
            const Stable stable = self;
            const bindings::NumericWorkScope work(buffers.count);
            for (py::ssize_t index = 0; index < buffers.count; ++index) {
                const auto value = stable.Sample(buffers.Read(index));
                buffers.Write(index, value.data());
            }
        });
    if constexpr (Mutable) {
        type.def_property_readonly("points",
                                   [](const Native &self) { return bindings::Vector3Array(self.PointStorage()); })
            .def_property_readonly("controls",
                                   [](const Native &self) -> py::object {
                                       return self.ControlStorage() ? py::cast(ControlArray{self.ControlStorage()})
                                                                    : py::none();
                                   })
            .def_property_readonly("recipe_points",
                                   [](std::shared_ptr<Native> self) { return RecipeView{std::move(self)}; })
            .def("with_waypoints",
                 [](const Native &self, const bindings::Vector3Array &points) {
                     return self.WithWaypoints(points.values);
                 })
            .def("with_controls", [](const Native &self, py::object controls) {
                return self.WithControls(controls.is_none() ? nullptr : controls.cast<const ControlArray &>().values);
            });
    }
}
} // namespace

void RegisterMotionPathBindings(py::module_ &module)
{
    py::register_exception<motion::MissingPathControls>(module, "_MotionPathMissingControls", PyExc_AttributeError);
    py::register_exception<motion::MissingWaypointLengths>(module, "_MotionPathMissingWaypointLengths",
                                                           PyExc_AttributeError);
    BindPathViews(module);
    module.def(
        "_cubic_bezier_point",
        [](const glm::vec3 &start, const glm::vec3 &startControl, const glm::vec3 &end, const glm::vec3 &endControl,
           float factor) {
            return Vector(
                motion::EvaluateCubicBezier(Point(start), Point(startControl), Point(endControl), Point(end), factor));
        },
        py::arg("start_point"), py::arg("start_control_point"), py::arg("end_point"), py::arg("end_control_point"),
        py::arg("factor"));
    module.def("_cubic_bezier_cloud", &CubicPoints, py::arg("start_point"), py::arg("start_control_point"),
               py::arg("end_point"), py::arg("end_control_point"), py::arg("resolution") = 10);
    module.def(
        "_append_cubic_bezier_cloud",
        [](py::list target, glm::vec3 start, glm::vec3 startControl, glm::vec3 end, glm::vec3 endControl,
           int resolution) {
            const auto points = CubicPoints(start, startControl, end, endControl, resolution);
            for (const auto &point : points)
                target.append(point);
        },
        py::arg("add_to_list"), py::arg("start_point"), py::arg("start_control_point"), py::arg("end_point"),
        py::arg("end_control_point"), py::arg("resolution") = 10);
    module.def("_motion_waypoint_index", [](const std::shared_ptr<Float32Array> &lengths, int pointCount,
                                            float percentage, float length, bool forward) {
        return motion::WaypointIndex(lengths.get(), pointCount, percentage, length, forward);
    });
    module.def("_motion_waypoint_fraction", [](const std::shared_ptr<Float32Array> &lengths, int index, float length) {
        return motion::WaypointFraction(lengths.get(), index, length);
    });
    module.def("_motion_path_speed", [](float length, float units) { return length / units; });
    module.def("_bezier_closing_handles", [](const glm::vec3 &start, const glm::vec3 &end, const glm::vec3 &firstHandle,
                                             const glm::vec3 &lastHandle) {
        const auto controls = motion::BezierClosingHandles({start.x, start.y, start.z}, {end.x, end.y, end.z},
                                                           {firstHandle.x, firstHandle.y, firstHandle.z},
                                                           {lastHandle.x, lastHandle.y, lastHandle.z});
        return std::array<glm::vec3, 2>{glm::vec3(controls[0][0], controls[0][1], controls[0][2]),
                                        glm::vec3(controls[1][0], controls[1][1], controls[1][2])};
    });
    module.def("_evaluate_motion_segment", [](int kind, const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2,
                                              const glm::vec3 &p3, float time) {
        const auto value = motion::EvaluateSegment(kind, {p0.x, p0.y, p0.z}, {p1.x, p1.y, p1.z}, {p2.x, p2.y, p2.z},
                                                   {p3.x, p3.y, p3.z}, time);
        return glm::vec3(value[0], value[1], value[2]);
    });
    BindPath<motion::Point, false>(module, "_MotionPathSnapshot");
    BindPath<glm::vec3, true>(module, "_MotionPathData");
}
} // namespace infernux
