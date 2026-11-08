#include "NumericBuffers.h"
#include "infernux/core/math/CurveAuthoring.h"
#include "infernux/core/math/CurveContentHash.h"
#include "infernux/core/math/CurveInterpolation.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
namespace infernux
{
namespace
{
struct CurveKey
{
    float time, value, inTangent, outTangent, inWeight, outWeight;
    unsigned weightedMode;
    int32_t tangentMode;
};

struct CurveSegment
{
    curve::HermitePolynomial polynomial{};
    bool isPolynomial = false;
};

// Immutable data owned by C++, never borrowed Python key objects. A snapshot
// stays valid while another thread edits or replaces the source curve.
class CurveSnapshot
{
  public:
    CurveSnapshot(const std::vector<std::array<double, 7>> &rows, int pre, int post) : pre_(pre), post_(post)
    {
        LoadKeys(rows, nullptr);
    }

    CurveSnapshot(const std::vector<std::array<double, 7>> &rows, int pre, int post,
                  const std::vector<int32_t> &tangentModes)
        : pre_(pre), post_(post)
    {
        if (tangentModes.size() != rows.size())
            throw std::invalid_argument("curve snapshot requires one tangent mode per key");
        LoadKeys(rows, tangentModes.data());
    }

  private:
    void LoadKeys(const std::vector<std::array<double, 7>> &rows, const int32_t *tangentModes)
    {
        // AnimationCurve owns validation and ordering. Consume its canonical
        // rows without repeating authoring work in the numerical snapshot.
        keys_.reserve(rows.size());
        for (size_t index = 0; index < rows.size(); ++index) {
            const auto &r = rows[index];
            keys_.push_back({static_cast<float>(r[0]), static_cast<float>(r[1]), static_cast<float>(r[2]),
                             static_cast<float>(r[3]), static_cast<float>(r[4]), static_cast<float>(r[5]),
                             static_cast<unsigned>(r[6]), tangentModes ? tangentModes[index] : 0});
        }
        if (keys_.size() < 2)
            return;
        segments_.reserve(keys_.size() - 1);
        for (size_t index = 1; index < keys_.size(); ++index) {
            const auto &left = keys_[index - 1];
            const auto &right = keys_[index];
            CurveSegment segment;
            if (!(left.weightedMode & 2) && !(right.weightedMode & 1) && right.time > left.time &&
                !std::isinf(left.outTangent) && !std::isinf(right.inTangent)) {
                segment.polynomial = curve::HermitePolynomial::Compile(left, right);
                segment.isPolynomial = true;
            }
            segments_.push_back(segment);
        }
    }

  public:
    size_t Length() const
    {
        return keys_.size();
    }
    float LastTime() const
    {
        return keys_.empty() ? 0.f : keys_.back().time;
    }

    int32_t GetHashCode() const
    {
        const bindings::NumericWorkScope work(keys_.size());
        return curve::ContentHash(keys_.data(), keys_.size());
    }

    bool Equals(const CurveSnapshot &other) const
    {
        if (this == &other)
            return true;
        if (pre_ != other.pre_ || post_ != other.post_ || keys_.size() != other.keys_.size())
            return false;
        const bindings::NumericWorkScope work(keys_.size());
        for (size_t index = 0; index < keys_.size(); ++index) {
            const auto &a = keys_[index];
            const auto &b = other.keys_[index];
            if (a.time != b.time || a.value != b.value || a.inTangent != b.inTangent || a.outTangent != b.outTangent ||
                a.inWeight != b.inWeight || a.outWeight != b.outWeight || a.weightedMode != b.weightedMode)
                return false;
        }
        return true;
    }

    float Evaluate(double time) const
    {
        if (keys_.empty())
            return 0.f;
        time = static_cast<float>(time);
        const auto &first = keys_.front();
        const auto &last = keys_.back();
        if (keys_.size() == 1)
            return first.value;
        if (!std::isfinite(time)) {
            const int mode = time < 0 ? pre_ : post_;
            return !std::isnan(time) && mode == 4 ? last.value : std::numeric_limits<float>::quiet_NaN();
        }
        const double span = static_cast<double>(last.time) - first.time;
        if (span > 0 && (time < first.time || time >= last.time)) {
            const int mode = time < first.time ? pre_ : post_;
            if (mode == 0 || mode == 2 || mode == 4) {
                const double period = mode == 4 ? span * 2 : span;
                double phase = std::fmod(time - first.time, period);
                if (phase < 0)
                    phase += period;
                if (mode == 4 && phase > span)
                    phase = period - phase;
                if (std::isnan(phase))
                    return std::numeric_limits<float>::quiet_NaN();
                time = first.time + phase;
            }
        }
        if (time >= last.time)
            return last.value;
        if (time < first.time || (time == first.time && keys_[1].time == first.time))
            return first.value;
        const auto right =
            std::upper_bound(keys_.begin(), keys_.end(), time, [](double t, const auto &key) { return t < key.time; });
        const auto &segment = segments_[static_cast<size_t>(right - keys_.begin() - 1)];
        if (segment.isPolynomial)
            return segment.polynomial.Evaluate(static_cast<float>(time) - (right - 1)->time);
        return curve::EvaluateSegment(*(right - 1), *right, static_cast<float>(time));
    }

    float EvaluateMapped(double time, double divisor, double scale) const
    {
        // Keep the division before multiplication, including overflow,
        // underflow and signed zero. Folding scale/divisor changes this map.
        volatile double normalized = time / divisor;
        return Evaluate(normalized * scale);
    }

    template <typename Read, typename Write>
    void EvaluateSamples(size_t count, Read read, Write write, double timeScale, double timeDivisor) const
    {
        if (timeDivisor == 1.0) {
            for (size_t index = 0; index < count; ++index)
                write(index, Evaluate(read(index) * timeScale));
        } else {
            for (size_t index = 0; index < count; ++index)
                write(index, EvaluateMapped(read(index), timeDivisor, timeScale));
        }
    }

    std::vector<float> EvaluateMany(const std::vector<double> &times, double timeScale, double timeDivisor) const
    {
        const bindings::NumericWorkScope work(times.size());
        std::vector<float> values(times.size());
        EvaluateSamples(
            times.size(), [&](size_t index) { return times[index]; },
            [&](size_t index, float value) { values[index] = value; }, timeScale, timeDivisor);
        return values;
    }

  private:
    const int pre_, post_;
    std::vector<CurveKey> keys_;
    std::vector<CurveSegment> segments_;
};
} // namespace

void RegisterAnimationCurveBindings(py::module_ &m)
{
    m.def(
        "_curve_smoothed_tangent",
        [](const std::array<float, 6> &points, float weight, int side) {
            struct Point
            {
                float time, value;
            };
            const Point keys[] = {{points[0], points[1]}, {points[2], points[3]}, {points[4], points[5]}};
            if (side < 0)
                return curve::Slope(keys[1], keys[2]);
            if (side > 0)
                return curve::Slope(keys[0], keys[1]);
            return curve::SmoothedTangent(keys, 3, 1, weight);
        },
        py::arg("points"), py::arg("weight"), py::arg("side"));
    py::class_<CurveSnapshot>(m, "_CurveSnapshot")
        .def(py::init<const std::vector<std::array<double, 7>> &, int, int>(), py::arg("keys"),
             py::arg("pre_wrap_mode"), py::arg("post_wrap_mode"))
        .def(py::init<const std::vector<std::array<double, 7>> &, int, int, const std::vector<int32_t> &>(),
             py::arg("keys"), py::arg("pre_wrap_mode"), py::arg("post_wrap_mode"), py::arg("tangent_modes"))
        .def_property_readonly("length", &CurveSnapshot::Length)
        .def_property_readonly("last_time", &CurveSnapshot::LastTime)
        .def("get_hash_code", &CurveSnapshot::GetHashCode)
        .def("equals", &CurveSnapshot::Equals, py::arg("other"))
        .def("evaluate", &CurveSnapshot::Evaluate, py::arg("time"))
        .def("evaluate_mapped", &CurveSnapshot::EvaluateMapped, py::arg("time"), py::arg("divisor"), py::arg("scale"))
        .def("evaluate_many", &CurveSnapshot::EvaluateMany, py::arg("times"), py::arg("time_scale") = 1.0,
             py::arg("time_divisor") = 1.0)
        .def(
            "evaluate_into",
            [](const CurveSnapshot &self, py::buffer times, py::buffer values, double timeScale, double timeDivisor) {
                const bindings::NumericBuffers<double, double> buffers(times, values);
                const bindings::NumericWorkScope work(buffers.count);
                self.EvaluateSamples(
                    buffers.count, [&](size_t index) { return buffers.Read(index); },
                    [&](size_t index, float value) {
                        const double result = value;
                        buffers.Write(index, &result);
                    },
                    timeScale, timeDivisor);
            },
            py::arg("times"), py::arg("values"), py::arg("time_scale") = 1.0, py::arg("time_divisor") = 1.0);
}
} // namespace infernux
