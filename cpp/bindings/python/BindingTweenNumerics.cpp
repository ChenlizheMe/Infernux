#include "NumericBuffers.h"
#include "PythonEase.h"
#include "Vector3Array.h"
#include "infernux/core/math/RectLayout.h"
#include "infernux/core/math/ScreenRect.h"
#include "infernux/core/math/TweenNumerics.h"
#include "infernux/core/math/TweenPlaybackClock.h"
#include <cstring>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>
#include <unordered_map>

namespace py = pybind11;
namespace infernux
{
namespace
{
glm::vec3 Vector(const tween::Point &value)
{
    return {value[0], value[1], value[2]};
}

using bindings::Vector3Array;

py::tuple ProgressTuple(const tween::ProgressSample &sample)
{
    PyObject *rawCycle = PyLong_FromDouble(sample.cycle);
    if (!rawCycle)
        throw py::error_already_set();
    py::object cycle = py::reinterpret_steal<py::object>(rawCycle);
    if (sample.previousCycle)
        cycle = cycle - py::int_(1);
    return py::make_tuple(cycle, sample.progress);
}

struct ClockKey
{
    uint32_t duration, delay;
    int64_t loops;
    bool operator==(const ClockKey &other) const
    {
        return duration == other.duration && delay == other.delay && loops == other.loops;
    }
};

struct ClockKeyHash
{
    size_t operator()(const ClockKey &key) const
    {
        const auto bits = static_cast<uint64_t>(key.duration) << 32 | key.delay;
        const auto timing = std::hash<uint64_t>{}(bits);
        return timing ^ (std::hash<int64_t>{}(key.loops) + 0x9e3779b9u + (timing << 6) + (timing >> 2));
    }
};

struct ClockRecipes
{
    // Only immutable native clocks are retained, never user callbacks. The
    // owning module function releases this per-module cache with the GIL held.
    static constexpr size_t capacity = 512;
    ClockRecipes()
    {
        values.reserve(capacity);
    }
    std::unordered_map<ClockKey, py::tuple, ClockKeyHash> values;
};

template <typename Segment> void BindSegments(py::class_<Segment> &type)
{
    type.def("locate",
             [](const Segment &self, double progress, double duration) { return self.Locate(progress, duration); })
        .def("locate_time",
             [](const Segment &self, double elapsed, double duration) { return self.LocateTime(elapsed, duration); })
        .def("sample_time",
             [](const Segment &self, double elapsed, double duration, const tween::EaseKernel &ease,
                double incremental) { return Vector(self.SampleTime(elapsed, duration, ease, incremental)); })
        .def("sample", [](const Segment &self, double progress, double duration, const tween::EaseKernel &ease,
                          double incremental) { return Vector(self.Sample(progress, duration, ease, incremental)); })
        .def("interpolate", [](const Segment &self, size_t index, double factor,
                               double incremental) { return Vector(self.Interpolate(index, factor, incremental)); })
        .def(
            "sample_into",
            [](const Segment &self, py::buffer progress, py::buffer positions, double duration,
               const tween::EaseKernel &ease, double incremental) {
                const bindings::NumericBuffers<double, float, 3> buffers(progress, positions);
                decltype(auto) snapshot = self.Frozen();
                const bindings::NumericWorkScope work(buffers.count);
                for (py::ssize_t index = 0; index < buffers.count; ++index) {
                    const auto value = snapshot.Sample(buffers.Read(index), duration, ease, incremental);
                    buffers.Write(index, value.data());
                }
            },
            py::arg("progress"), py::arg("positions"), py::arg("duration"), py::arg("ease"),
            py::arg("incremental") = 0);
}

template <typename Snapshot, typename Scalar> struct TypedScalarEvaluator : tween::ValueEvaluator<Snapshot>
{
    using tween::ValueEvaluator<Snapshot>::ValueEvaluator;
};

template <typename Snapshot, typename Scalar>
void BindTypedScalar(py::module_ &module, py::class_<Snapshot> &snapshot, const char *evaluatorName,
                     const char *bindName, const char *interpolateName)
{
    using Evaluator = TypedScalarEvaluator<Snapshot, Scalar>;
    using Incremental = typename Snapshot::Incremental;
    // Preserve the portable standard-library long-long buffer aliases while
    // using fixed-width scalar types for NumPy's C API.
    using Buffer = std::conditional_t<std::is_same_v<Scalar, int64_t>, long long,
                                      std::conditional_t<std::is_same_v<Scalar, uint64_t>, unsigned long long, Scalar>>;
    py::class_<Evaluator>(module, evaluatorName)
        .def("__call__",
             [](const Evaluator &self, double progress, Incremental incremental) {
                 return py::make_scalar(static_cast<Scalar>(self.Sample(progress, incremental)));
             })
        .def("sample_nested",
             [](const Evaluator &self, double progress, Incremental incremental, Incremental multiplier,
                Incremental parent) {
                 return py::make_scalar(
                     static_cast<Scalar>(self.SampleNested(progress, incremental, multiplier, parent)));
             })
        .def(
            "sample_into",
            [](const Evaluator &self, py::buffer progress, py::buffer values, Incremental incremental,
               Incremental multiplier, Incremental parent) {
                const bindings::NumericBuffers<double, Buffer> buffers(progress, values);
                const bindings::NumericWorkScope work(buffers.count);
                for (py::ssize_t index = 0; index < buffers.count; ++index) {
                    const Buffer value =
                        static_cast<Buffer>(self.SampleNested(buffers.Read(index), incremental, multiplier, parent));
                    buffers.Write(index, &value);
                }
            },
            py::arg("progress"), py::arg("values"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
            py::arg("parent_incremental") = 0);
    snapshot
        .def(bindName, [](const Snapshot &self, const tween::EaseKernel &ease,
                          double duration) { return Evaluator(self, ease, duration); })
        .def(
            interpolateName,
            [](const Snapshot &self, double factor, Incremental incremental, Incremental multiplier,
               Incremental parent) {
                return py::make_scalar(static_cast<Scalar>(self.Interpolate(factor, incremental, multiplier, parent)));
            },
            py::arg("factor"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
            py::arg("parent_incremental") = 0);
    const std::string customName =
        "sample_custom" + std::string(interpolateName).substr(std::string("interpolate").size());
    snapshot.def(
        customName.c_str(),
        [](const Snapshot &self, const py::function &ease, double progress, Incremental incremental,
           Incremental multiplier, Incremental parent) {
            const Snapshot value = self;
            const double factor = bindings::SamplePythonEase(ease, progress);
            return py::make_scalar(static_cast<Scalar>(value.Interpolate(factor, incremental, multiplier, parent)));
        },
        py::arg("ease"), py::arg("progress"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
        py::arg("parent_incremental") = 0);
    const std::string timedName =
        "sample_custom_time" + std::string(interpolateName).substr(std::string("interpolate").size());
    snapshot.def(
        timedName.c_str(),
        [](const Snapshot &self, const py::tuple &prepared, double progress, Incremental incremental,
           Incremental multiplier, Incremental parent) {
            const Snapshot value = self;
            const double factor = bindings::SamplePreparedPythonEase(prepared, progress);
            return py::make_scalar(static_cast<Scalar>(value.Interpolate(factor, incremental, multiplier, parent)));
        },
        py::arg("prepared"), py::arg("progress"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
        py::arg("parent_incremental") = 0);
}

template <typename Value> Value RoundValue(Value value, int constraint)
{
    for (int axis = 0; axis < Value::length(); ++axis)
        if (constraint == 0 || constraint == (2 << axis))
            value[axis] = std::nearbyint(value[axis]);
    return value;
}

template <typename Snapshot, typename Convert>
void BindVectorJob(py::module_ &module, const char *snapshotName, const char *evaluatorName, Convert convert)
{
    using Evaluator = tween::ValueEvaluator<Snapshot>;
    using Value = typename Snapshot::Value;
    constexpr size_t channels = std::tuple_size<Value>::value;
    py::class_<Evaluator>(module, evaluatorName)
        .def("__call__", [convert](const Evaluator &self, double progress,
                                   double incremental) { return convert(self.Sample(progress, incremental)); })
        .def("sample_nested",
             [convert](const Evaluator &self, double progress, double incremental, double multiplier, double parent) {
                 return convert(self.SampleNested(progress, incremental, multiplier, parent));
             })
        .def(
            "sample_into",
            [](const Evaluator &self, py::buffer progress, py::buffer values, double incremental, double multiplier,
               double parent) {
                const bindings::NumericBuffers<double, float, channels> buffers(progress, values);
                const bindings::NumericWorkScope work(buffers.count);
                for (py::ssize_t index = 0; index < buffers.count; ++index) {
                    const auto value = self.SampleNested(buffers.Read(index), incremental, multiplier, parent);
                    buffers.Write(index, value.data());
                }
            },
            py::arg("progress"), py::arg("values"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
            py::arg("parent_incremental") = 0);
    py::class_<Snapshot>(module, snapshotName)
        .def(py::init<Value, Value, bool>(), py::arg("start"), py::arg("delta"), py::arg("snapping") = false)
        .def("bind", [](const Snapshot &self, const tween::EaseKernel &ease,
                        double duration) { return Evaluator(self, ease, duration); })
        .def("sample",
             [convert](const Snapshot &self, double progress, double duration, const tween::EaseKernel &ease,
                       double incremental) { return convert(self.Sample(progress, duration, ease, incremental)); })
        .def(
            "interpolate",
            [convert](const Snapshot &self, double factor, double incremental, double multiplier, double parent) {
                return convert(self.Interpolate(factor, incremental, multiplier, parent));
            },
            py::arg("factor"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
            py::arg("parent_incremental") = 0);
}
} // namespace

void RegisterTweenNumericsBindings(py::module_ &module)
{
    py::register_exception<tween::NumericOverflow>(module, "_TweenNumericOverflow", PyExc_OverflowError);
    py::class_<ui::ScreenRect>(module, "_ScreenRectGeometry")
        .def(py::init<ui::ScreenRect::Rect, double, ui::ScreenRect::Point>(), py::arg("rect"), py::arg("rotation"),
             py::arg("pivot") = ui::ScreenRect::Point{.5, .5})
        .def_property_readonly("rect", &ui::ScreenRect::GetRect)
        .def_property_readonly("center", &ui::ScreenRect::Center)
        .def_property_readonly("pivot_position", &ui::ScreenRect::PivotPosition)
        .def_property_readonly("render_rect", &ui::ScreenRect::RenderRect)
        .def_property_readonly("corners", &ui::ScreenRect::Corners)
        .def_property_readonly("visual_rect", &ui::ScreenRect::VisualRect)
        .def("to_local", &ui::ScreenRect::ToLocal)
        .def("to_canvas", &ui::ScreenRect::ToCanvas)
        .def("contains", &ui::ScreenRect::Contains, py::arg("point"), py::arg("tolerance") = 0);
    py::class_<ui::RectLayoutGeometry>(module, "_RectLayoutSnapshot")
        .def(py::init([](const glm::vec2 &parentSize, const glm::vec2 &parentPivot, const glm::vec2 &anchorMin,
                         const glm::vec2 &anchorMax, const glm::vec2 &pivot, const glm::vec2 &sizeDelta,
                         const glm::vec2 &anchoredPosition) {
                 const auto point = [](const glm::vec2 &v) -> ui::RectPoint { return {v.x, v.y}; };
                 return ui::CompileRectLayout({point(parentSize), point(parentPivot), point(anchorMin),
                                               point(anchorMax), point(pivot), point(sizeDelta),
                                               point(anchoredPosition)});
             }),
             py::arg("parent_size"), py::arg("parent_pivot"), py::arg("anchor_min"), py::arg("anchor_max"),
             py::arg("pivot"), py::arg("size_delta"), py::arg("anchored_position"))
        .def_property_readonly("rect",
                               [](const ui::RectLayoutGeometry &self) {
                                   return glm::vec4(self.rect[0], self.rect[1], self.rect[2], self.rect[3]);
                               })
        .def_property_readonly(
            "offset_min",
            [](const ui::RectLayoutGeometry &self) { return glm::vec2(self.offsetMin[0], self.offsetMin[1]); })
        .def_property_readonly(
            "offset_max",
            [](const ui::RectLayoutGeometry &self) { return glm::vec2(self.offsetMax[0], self.offsetMax[1]); })
        .def_property_readonly("local_corners", [](const ui::RectLayoutGeometry &self) {
            std::array<glm::vec3, 4> corners;
            for (size_t i = 0; i < corners.size(); ++i)
                corners[i] = glm::vec3(self.localCorners[i][0], self.localCorners[i][1], self.localCorners[i][2]);
            return corners;
        });

    // Micro calls retain the caller's existing GIL. Releasing it for each
    // short value operation causes lock handoffs to busy Python
    // workers. Larger buffer/vector jobs release it once for their whole loop.
    using tween::EaseKernel;
    using ScalarEvaluator = tween::ValueEvaluator<tween::ScalarSnapshot>;
    module.def("_tween_float32", &tween::Float32, py::arg("value"));
    py::class_<tween::PlaybackClock>(module, "_TweenPlaybackClock")
        .def(py::init<float, float, int64_t, double>())
        .def("seek", &tween::PlaybackClock::Seek)
        .def("advance",
             [](tween::PlaybackClock &clock, float delta, float scale, bool backwards) -> py::object {
                 return clock.Advance(delta, scale, backwards) ? py::cast(clock.Time()) : py::none();
             })
        .def("begin_step",
             [](tween::PlaybackClock &clock, float delta, float scale) -> py::object {
                 float step;
                 return clock.BeginStep(delta, scale, step) ? py::cast(step) : py::none();
             })
        .def("advance_step",
             [](tween::PlaybackClock &clock, float step, bool backwards) {
                 clock.AdvanceStep(step, backwards);
                 return clock.Time();
             })
        .def("begin_callbacks", &tween::PlaybackClock::BeginCallbacks)
        .def("finish_callbacks", &tween::PlaybackClock::FinishCallbacks)
        .def("sample", [](const tween::PlaybackClock &clock, double position, bool yoyo,
                          bool inverted) { return ProgressTuple(clock.Sample(position, yoyo, inverted)); })
        .def("completed_at", &tween::PlaybackClock::CompletedAt)
        .def("complete_at", &tween::PlaybackClock::CompleteAt)
        .def_property_readonly("time", &tween::PlaybackClock::Time)
        .def_property_readonly("position", &tween::PlaybackClock::Position)
        .def_property_readonly("completed_loops", &tween::PlaybackClock::CompletedLoops)
        .def_property_readonly("is_complete", &tween::PlaybackClock::IsComplete)
        .def_property_readonly("was_complete", &tween::PlaybackClock::WasComplete)
        .def_property_readonly("elapsed", &tween::PlaybackClock::Elapsed)
        .def_property_readonly("elapsed_delay", &tween::PlaybackClock::ElapsedDelay);
    module.def(
        "_tween_loop_duration", [](float duration, int64_t loops) { return duration * static_cast<float>(loops); },
        py::arg("duration"), py::arg("loops"));
    py::class_<tween::PreparedTweenClock>(module, "_PreparedTweenClock")
        .def(
            "sample",
            [](const tween::PreparedTweenClock &clock, double position, bool yoyo, bool inverted) -> py::tuple {
                if (clock.IsTerminal(position))
                    return py::make_tuple(clock.loops - 1, clock.TerminalProgress(yoyo, inverted));
                return ProgressTuple(clock.SampleInterior(position, yoyo, inverted));
            },
            py::arg("position"), py::arg("yoyo"), py::arg("inverted"));
    module.def(
        "_compile_tween_clock",
        [recipes = std::make_shared<ClockRecipes>()](float duration, float delay, int64_t loops) -> py::tuple {
            ClockKey key;
            std::memcpy(&key.duration, &duration, sizeof(duration));
            std::memcpy(&key.delay, &delay, sizeof(delay));
            key.loops = loops;
            const auto found = recipes->values.find(key);
            if (found != recipes->values.end())
                return found->second;
            const tween::PreparedTweenClock clock(duration, delay, loops);
            py::tuple recipe =
                py::make_tuple(py::cast(clock).attr("sample"), clock.playbackDuration, clock.fullDuration);
            if (recipes->values.size() == ClockRecipes::capacity)
                recipes->values.clear();
            recipes->values.emplace(key, recipe);
            return recipe;
        },
        py::arg("duration"), py::arg("delay"), py::arg("loops"));
    module.def(
        "_evaluate_custom_ease",
        [](const py::function &ease, float amplitude, float period, float time, float duration) {
            return static_cast<float>(bindings::SamplePythonEase(ease, time, duration, amplitude, period));
        },
        py::arg("ease"), py::arg("amplitude"), py::arg("period"), py::arg("time"), py::arg("duration"));
    module.def("_sample_custom_ease", &bindings::SamplePreparedPythonEase, py::arg("prepared"), py::arg("progress"));
    py::class_<tween::RectOffsetInterpolation>(module, "_RectOffsetInterpolation")
        .def(py::init<>())
        .def("prepare", &tween::RectOffsetInterpolation::Prepare)
        .def("interpolate", &tween::RectOffsetInterpolation::Interpolate)
        .def("sample", &tween::RectOffsetInterpolation::Sample);
    module.def("_rect_offset_speed", &tween::RectOffsetSpeed);
    py::class_<tween::StopMotionClock>(module, "_StopMotionClock")
        .def(py::init<int>(), py::arg("motion_fps"))
        .def(
            "evaluate_builtin",
            [](const tween::StopMotionClock &self, int kind, float time, float duration, float amplitude,
               float period) {
                return static_cast<float>(
                    EaseKernel(kind, amplitude, period).Evaluate(self.Sample(time, duration), duration));
            },
            py::arg("kind"), py::arg("time"), py::arg("duration"), py::arg("amplitude"), py::arg("period"))
        .def(
            "evaluate",
            [](const tween::StopMotionClock &self, const py::function &ease, float time, float duration,
               float amplitude, float period) {
                // Callback ownership remains in Python's partial, visible to its GC.
                return static_cast<float>(
                    bindings::SamplePythonEase(ease, self.Sample(time, duration), duration, amplitude, period));
            },
            py::arg("ease"), py::arg("time"), py::arg("duration"), py::arg("amplitude"), py::arg("period"));
    module.def(
        "_tween_array_duration",
        [](const std::vector<double> &durations) {
            const bindings::NumericWorkScope work(durations.size(), 1u << 18);
            return tween::ArrayDuration(durations);
        },
        py::arg("durations"));
    module.def(
        "_tween_impulse_durations",
        [](float duration, int vibrato, bool fadeOut) {
            const int count = tween::ImpulseIterations(duration, vibrato);
            // Two simple float loops are much cheaper than ease/path evaluation.
            // Tiny recipes must not hand the GIL to a busy worker during startup.
            const bindings::NumericWorkScope work(static_cast<size_t>(count), 1u << 18);
            return tween::ImpulseDurations(duration, count, fadeOut);
        },
        py::arg("duration"), py::arg("vibrato"), py::arg("fade_out"));
    module.def("_advance_tween_clock", &tween::AdvanceClock, py::arg("position"), py::arg("delta"),
               py::arg("time_scale"), py::arg("backwards"), py::arg("full_duration"), py::arg("clamp") = true);
    module.def(
        "_exceptional_tween_clock",
        [](float position, float duration, int loops) {
            const auto sample = tween::SampleExceptionalClock(position, duration, loops);
            return py::make_tuple(sample.position, sample.completedLoops);
        },
        py::arg("position"), py::arg("duration"), py::arg("loops"));
    module.def(
        "_advance_exceptional_tween_clock",
        [](float position, float duration, int loops) {
            const auto sample = tween::AdvanceExceptionalClock(position, duration, loops);
            return py::make_tuple(sample.position, sample.completedLoops);
        },
        py::arg("position"), py::arg("duration"), py::arg("loops"));
    module.def(
        "_tween_progress",
        [](double position, double delay, double duration, bool yoyo, bool inverted, bool clamp) {
            return ProgressTuple(tween::SampleProgress(position, delay, duration, yoyo, inverted, clamp));
        },
        py::arg("position"), py::arg("delay"), py::arg("duration"), py::arg("yoyo"), py::arg("inverted"),
        py::arg("clamp") = true);
    module.def(
        "_round_tween_value", [](double value, int) { return std::nearbyint(value); }, py::arg("value"),
        py::arg("axis_constraint") = 0);
    module.def("_round_tween_value", &RoundValue<glm::vec2>, py::arg("value"), py::arg("axis_constraint") = 0);
    module.def("_round_tween_value", &RoundValue<glm::vec3>, py::arg("value"), py::arg("axis_constraint") = 0);
    module.def("_round_tween_value", &RoundValue<glm::vec4>, py::arg("value"), py::arg("axis_constraint") = 0);
    module.def(
        "_evaluate_builtin_ease",
        [](int kind, float time, float duration, float amplitude, float period) {
            return static_cast<float>(EaseKernel(kind, amplitude, period).Evaluate(time, duration));
        },
        py::arg("kind"), py::arg("time"), py::arg("duration"), py::arg("overshoot_or_amplitude"), py::arg("period"));
    module.def(
        "_tween_point_on_circle",
        [](const glm::vec2 &center, float radius, float degrees) {
            const auto point = tween::CircleGeometry({center.x, center.y}, radius).Position(degrees, false);
            return glm::vec2(point[0], point[1]);
        },
        py::arg("center"), py::arg("radius"), py::arg("degrees"));
    py::class_<EaseKernel>(module, "_EaseKernel")
        .def(py::init<int, double, double>(), py::arg("kind"), py::arg("amplitude") = 1.70158, py::arg("period") = 0)
        .def_property_readonly("is_linear", &EaseKernel::IsLinear)
        .def("__call__", [](const EaseKernel &self, double progress) { return self.Sample(progress); })
        .def("sample", &EaseKernel::Sample, py::arg("progress"), py::arg("duration") = 1)
        .def(
            "evaluate",
            [](const EaseKernel &self, float time, float duration) {
                return static_cast<float>(self.Evaluate(time, duration));
            },
            py::arg("time"), py::arg("duration"))
        .def("solve_clock", &EaseKernel::SolveClock)
        .def(
            "sample_many",
            [](const EaseKernel &self, const std::vector<double> &progress, double duration) {
                const bindings::NumericWorkScope work(progress.size());
                std::vector<double> values(progress.size());
                for (size_t index = 0; index < progress.size(); ++index)
                    values[index] = self.Sample(progress[index], duration);
                return values;
            },
            py::arg("progress"), py::arg("duration") = 1)
        .def(
            "sample_into",
            [](const EaseKernel &self, py::buffer progress, py::buffer values, double duration) {
                const bindings::NumericBuffers<double, double> buffers(progress, values);
                const bindings::NumericWorkScope work(buffers.count);
                for (py::ssize_t index = 0; index < buffers.count; ++index) {
                    const double result = self.Sample(buffers.Read(index), duration);
                    buffers.Write(index, &result);
                }
            },
            py::arg("progress"), py::arg("values"), py::arg("duration") = 1);

    py::class_<ScalarEvaluator>(module, "_ScalarTweenEvaluator")
        .def("__call__", &ScalarEvaluator::Sample)
        .def("sample_nested", &ScalarEvaluator::SampleNested)
        .def(
            "sample_into",
            [](const ScalarEvaluator &self, py::buffer progress, py::buffer values, double incremental,
               double multiplier, double parent) {
                const bindings::NumericBuffers<double, double> buffers(progress, values);
                const bindings::NumericWorkScope work(buffers.count);
                for (py::ssize_t index = 0; index < buffers.count; ++index) {
                    const double result = self.SampleNested(buffers.Read(index), incremental, multiplier, parent);
                    buffers.Write(index, &result);
                }
            },
            py::arg("progress"), py::arg("values"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
            py::arg("parent_incremental") = 0);
    auto scalarSnapshot = py::class_<tween::ScalarSnapshot>(module, "_ScalarTweenSnapshot");
    scalarSnapshot
        .def(py::init<double, double, bool>(), py::arg("start"), py::arg("delta"), py::arg("snapping") = false)
        .def("bind", [](const tween::ScalarSnapshot &self, const EaseKernel &ease,
                        double duration) { return ScalarEvaluator(self, ease, duration); })
        .def("sample", &tween::ScalarSnapshot::Sample)
        .def(
            "sample_custom",
            [](const tween::ScalarSnapshot &self, const py::function &ease, double progress, double incremental,
               double multiplier, double parent) {
                const auto value = self;
                const double factor = bindings::SamplePythonEase(ease, progress);
                return value.Interpolate(factor, incremental, multiplier, parent);
            },
            py::arg("ease"), py::arg("progress"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
            py::arg("parent_incremental") = 0)
        .def(
            "sample_custom_time",
            [](const tween::ScalarSnapshot &self, const py::tuple &prepared, double progress, double incremental,
               double multiplier, double parent) {
                const auto value = self;
                const double factor = bindings::SamplePreparedPythonEase(prepared, progress);
                return value.Interpolate(factor, incremental, multiplier, parent);
            },
            py::arg("prepared"), py::arg("progress"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
            py::arg("parent_incremental") = 0)
        .def("interpolate", &tween::ScalarSnapshot::Interpolate, py::arg("factor"), py::arg("incremental") = 0,
             py::arg("parent_multiplier") = 1, py::arg("parent_incremental") = 0);
    BindTypedScalar<tween::ScalarSnapshot, double>(module, scalarSnapshot, "_Float64TweenEvaluator", "bind_float64",
                                                   "interpolate_float64");
    auto float32Snapshot = py::class_<tween::Float32Snapshot>(module, "_Float32TweenSnapshot");
    float32Snapshot.def(py::init<float, float, bool>(), py::arg("start"), py::arg("delta"),
                        py::arg("snapping") = false);
    BindTypedScalar<tween::Float32Snapshot, float>(module, float32Snapshot, "_Float32TweenEvaluator", "bind",
                                                   "interpolate");
    using IntegerEvaluator = tween::ValueEvaluator<tween::SignedIntegerSnapshot>;
    py::class_<IntegerEvaluator>(module, "_SignedIntegerTweenEvaluator")
        .def("__call__", &IntegerEvaluator::Sample)
        .def("sample_nested", &IntegerEvaluator::SampleNested)
        .def(
            "sample_into",
            [](const IntegerEvaluator &self, py::buffer progress, py::buffer values, int64_t incremental,
               int64_t multiplier, int64_t parent) {
                const bindings::NumericBuffers<double, long long> buffers(progress, values);
                const bindings::NumericWorkScope work(buffers.count);
                for (py::ssize_t index = 0; index < buffers.count; ++index) {
                    const long long result = self.SampleNested(buffers.Read(index), incremental, multiplier, parent);
                    buffers.Write(index, &result);
                }
            },
            py::arg("progress"), py::arg("values"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
            py::arg("parent_incremental") = 0);
    auto integerSnapshot = py::class_<tween::SignedIntegerSnapshot>(module, "_SignedIntegerTweenSnapshot");
    integerSnapshot
        .def(py::init([](py::int_ start, py::int_ end, int bits) {
                 const auto first = PyLong_AsUnsignedLongLongMask(start.ptr());
                 const auto last = PyLong_AsUnsignedLongLongMask(end.ptr());
                 if (PyErr_Occurred())
                     throw py::error_already_set();
                 return tween::SignedIntegerSnapshot(first, last, bits);
             }),
             py::arg("start"), py::arg("end"), py::arg("bits"))
        .def_property_readonly("start_value", &tween::SignedIntegerSnapshot::Start)
        .def_property_readonly("end_value", &tween::SignedIntegerSnapshot::End)
        .def_property_readonly("change_value", &tween::SignedIntegerSnapshot::Change)
        .def("bind", [](const tween::SignedIntegerSnapshot &self, const EaseKernel &ease,
                        double duration) { return IntegerEvaluator(self, ease, duration); })
        .def("interpolate", &tween::SignedIntegerSnapshot::Interpolate, py::arg("factor"), py::arg("incremental") = 0,
             py::arg("parent_multiplier") = 1, py::arg("parent_incremental") = 0);
    integerSnapshot.def(
        "sample_custom",
        [](const tween::SignedIntegerSnapshot &self, const py::function &ease, double progress, int64_t incremental,
           int64_t multiplier, int64_t parent) {
            const auto value = self;
            const double factor = bindings::SamplePythonEase(ease, progress);
            return value.Interpolate(factor, incremental, multiplier, parent);
        },
        py::arg("ease"), py::arg("progress"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
        py::arg("parent_incremental") = 0);
    integerSnapshot.def(
        "sample_custom_time",
        [](const tween::SignedIntegerSnapshot &self, const py::tuple &prepared, double progress, int64_t incremental,
           int64_t multiplier, int64_t parent) {
            const auto value = self;
            const double factor = bindings::SamplePreparedPythonEase(prepared, progress);
            return value.Interpolate(factor, incremental, multiplier, parent);
        },
        py::arg("prepared"), py::arg("progress"), py::arg("incremental") = 0, py::arg("parent_multiplier") = 1,
        py::arg("parent_incremental") = 0);
    BindTypedScalar<tween::SignedIntegerSnapshot, int32_t>(module, integerSnapshot, "_Int32TweenEvaluator",
                                                           "bind_int32", "interpolate_int32");
    BindTypedScalar<tween::SignedIntegerSnapshot, int64_t>(module, integerSnapshot, "_Int64TweenEvaluator",
                                                           "bind_int64", "interpolate_int64");
    auto unsigned32Snapshot = py::class_<tween::Unsigned32Snapshot>(module, "_UInt32TweenSnapshot");
    unsigned32Snapshot.def(py::init<uint32_t, uint32_t, bool>(), py::arg("start"), py::arg("change"),
                           py::arg("negative") = false);
    BindTypedScalar<tween::Unsigned32Snapshot, uint32_t>(module, unsigned32Snapshot, "_UInt32TweenEvaluator", "bind",
                                                         "interpolate");
    auto unsigned64Snapshot = py::class_<tween::Unsigned64Snapshot>(module, "_UInt64TweenSnapshot");
    unsigned64Snapshot.def(py::init<uint64_t, uint64_t>(), py::arg("start"), py::arg("change"));
    BindTypedScalar<tween::Unsigned64Snapshot, uint64_t>(module, unsigned64Snapshot, "_UInt64TweenEvaluator", "bind",
                                                         "interpolate");
    BindVectorJob<tween::Vector2Snapshot>(module, "_Vector2TweenSnapshot", "_Vector2TweenEvaluator",
                                          [](const std::array<float, 2> &v) { return glm::vec2(v[0], v[1]); });
    BindVectorJob<tween::Vector3Snapshot>(module, "_Vector3TweenSnapshot", "_Vector3TweenEvaluator", Vector);
    BindVectorJob<tween::Vector4Snapshot>(
        module, "_Vector4TweenSnapshot", "_Vector4TweenEvaluator",
        [](const std::array<float, 4> &v) { return glm::vec4(v[0], v[1], v[2], v[3]); });
    BindVectorJob<tween::ColorSnapshot>(module, "_ColorTweenSnapshot", "_ColorTweenEvaluator",
                                        [](const std::array<float, 4> &v) { return v; });
    BindVectorJob<tween::RectSnapshot>(
        module, "_RectTweenSnapshot", "_RectTweenEvaluator",
        [](const std::array<float, 4> &v) { return py::make_tuple(v[0], v[1], v[2], v[3]); });
    py::class_<tween::CircleGeometry>(module, "_CircleGeometry")
        .def(py::init<tween::CircleGeometry::Value, float>())
        .def(
            "position",
            [](const tween::CircleGeometry &self, float degrees, bool snapping) {
                const auto v = self.Position(degrees, snapping);
                return glm::vec2(v[0], v[1]);
            },
            py::arg("degrees"), py::arg("snapping") = false)
        .def("interpolate",
             [](const tween::CircleGeometry &self, float start, float change, float factor, float own,
                float parentMultiplier, float parent, bool snapping) {
                 const auto v = self.Interpolate(start, change, factor, own, parentMultiplier, parent, snapping);
                 return glm::vec2(v[0], v[1]);
             })
        .def("sample",
             [](const tween::CircleGeometry &self, double progress, double duration, const EaseKernel &ease,
                float start, float change, float own, float parentMultiplier, float parent, bool snapping) {
                 const auto v =
                     self.Sample(progress, duration, ease, start, change, own, parentMultiplier, parent, snapping);
                 return glm::vec2(v[0], v[1]);
             })
        .def(
            "sample_into",
            [](const tween::CircleGeometry &self, py::buffer progress, py::buffer values, double duration,
               const EaseKernel &ease, float start, float change, float own, float parentMultiplier, float parent,
               bool snapping) {
                const bindings::NumericBuffers<double, float, 2> buffers(progress, values);
                const bindings::NumericWorkScope work(buffers.count);
                for (py::ssize_t index = 0; index < buffers.count; ++index) {
                    const auto value = self.Sample(buffers.Read(index), duration, ease, start, change, own,
                                                   parentMultiplier, parent, snapping);
                    buffers.Write(index, value.data());
                }
            },
            py::arg("progress"), py::arg("values"), py::arg("duration"), py::arg("ease"), py::arg("start"),
            py::arg("change"), py::arg("own_increment") = 0, py::arg("parent_multiplier") = 1,
            py::arg("parent_increment") = 0, py::arg("snapping") = false);
    py::class_<tween::SegmentSample>(module, "_TweenSegmentSample")
        .def_readonly("index", &tween::SegmentSample::index)
        .def_readonly("width", &tween::SegmentSample::width)
        .def_readonly("progress", &tween::SegmentSample::progress);
    auto segment = py::class_<tween::SegmentSnapshot>(module, "_TweenSegmentSnapshot");
    segment.def(py::init<std::vector<tween::Point>, std::vector<tween::Point>, std::vector<double>>());
    BindSegments(segment);
    bindings::BindVector3Array(module);
    using LiveSegments = tween::LiveSegmentSnapshot<glm::vec3>;
    module.def("_tween_segment_speed_durations", [](const Vector3Array &changes, const std::vector<float> &speeds) {
        const auto values = *changes.values;
        const bindings::NumericWorkScope work(values.size());
        return tween::SegmentSpeedDurations(values, speeds);
    });
    auto live = py::class_<LiveSegments>(module, "_TweenLiveSegments");
    live.def(py::init([](const Vector3Array &starts, const Vector3Array &changes, std::vector<double> durations) {
        return LiveSegments(starts.values, changes.values, std::move(durations));
    }));
    BindSegments(live);
}
} // namespace infernux
