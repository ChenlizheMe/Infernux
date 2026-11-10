#pragma once

#include "DecimalUInt64.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernux::tween
{
// Keep the managed clock's two observable float stores without allocating
// Python struct buffers. Use IEEE overflow consistently: native-format struct
// packing differed between the Windows and Linux CPython builds.
inline double Float32(double value)
{
    return static_cast<float>(value);
}

struct StopMotionClock
{
    explicit StopMotionClock(int framesPerSecond) : step(1.0f / static_cast<float>(framesPerSecond))
    {
    }

    float Sample(float time, float duration) const
    {
        return time < duration ? time - std::fmod(time, step) : time;
    }

    const float step;
};

inline int ImpulseIterations(float duration, int vibrato)
{
    // The measured managed runtime converts the intermediate product to int
    // before storing a float. Invalid/out-of-range conversions produce Int32.MinValue.
    const double product = static_cast<double>(duration) * vibrato;
    return product >= 2 && product < 2147483648.0 ? static_cast<int>(product) : 2;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
inline std::vector<float>
ImpulseDurations(float duration, int count, bool fadeOut)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    std::vector<float> widths(static_cast<size_t>(count));
    float total = 0;
    for (int index = 0; index < count; ++index) {
        const float fraction = static_cast<float>(index + 1) / static_cast<float>(count);
        const float width = fadeOut ? duration * fraction : duration / static_cast<float>(count);
        widths[index] = width;
        total = total + width;
    }
    const float scale = duration / total;
    for (float &width : widths)
        width = width * scale;
    return widths;
}

inline double AdvanceClock(double position, double delta, double timeScale, bool backwards, double fullDuration,
                           bool clamp = true)
{
    const double step = Float32(delta * timeScale);
    const double next = Float32(position + (backwards ? -step : step));
    return std::min(fullDuration, clamp ? std::max(0.0, next) : next);
}

struct ProgressSample
{
    double cycle, progress;
    bool previousCycle;
};

struct ExceptionalClockSample
{
    double position;
    int completedLoops;
};

inline ExceptionalClockSample SampleExceptionalClock(float requested, float duration, int loops)
{
    if (duration <= 0)
        return {duration, loops};
    const float quotient = requested / duration;
    const int completed = quotient >= -2147483648.0 && quotient < 2147483648.0 ? static_cast<int>(quotient)
                                                                               : std::numeric_limits<int>::min();
    if (loops != -1 && completed >= loops)
        return {duration, loops};
    float position = std::fmod(requested, duration);
    if (position == 0 && completed > 0)
        position = duration;
    return {position, completed};
}

inline ExceptionalClockSample AdvanceExceptionalClock(float requested, float duration, int loops)
{
    if (loops != -1 && requested >= duration * loops)
        return {duration, loops};
    if (std::isnan(duration))
        return {requested, 0};
    return SampleExceptionalClock(requested, duration, loops);
}

// Keep an exact-loop endpoint on the preceding cycle. The Python bridge owns
// integer conversion, including arbitrarily large clocks and nonfinite errors.
inline ProgressSample SampleProgress(double position, double delay, double duration, bool yoyo, bool inverted,
                                     bool clamp = true)
{
    const double elapsed = clamp ? std::max(0.0, position - delay) : position - delay;
    const double quotient = duration != 0 ? elapsed / duration : 0;
    const double floored = std::floor(quotient);
    const double cycle = floored == 0 ? 0.0 : floored;
    double progress = duration != 0 ? quotient - cycle : 1;
    const bool previousCycle = elapsed > 0 && progress == 0;
    if (previousCycle)
        progress = 1;
    // Evaluate parity before subtracting one: a double cannot retain that
    // subtraction for cycles above 2^53, while the public cycle is a Python int.
    if (yoyo && ((std::fmod(cycle, 2) != 0) != previousCycle))
        progress = 1 - progress;
    if (inverted)
        progress = 1 - progress;
    return {cycle, progress, previousCycle};
}

struct PreparedTweenClock
{
    PreparedTweenClock(float duration, float delay, int64_t loops)
        : duration(duration), delay(delay), loops(loops),
          playbackDuration(loops == 1    ? duration
                           : loops == -1 ? std::numeric_limits<double>::infinity()
                                         : duration * static_cast<float>(loops)),
          fullDuration(static_cast<double>(delay) + playbackDuration), hasLoops(loops == -1 || loops > 1)
    {
    }

    bool IsTerminal(double position) const
    {
        return position == fullDuration && duration > 0 && loops > 0;
    }

    double TerminalProgress(bool yoyo, bool inverted) const
    {
        const double progress = yoyo && (loops - 1) % 2 ? 0 : 1;
        return inverted ? 1 - progress : progress;
    }

    ProgressSample SampleInterior(double position, bool yoyo, bool inverted) const
    {
        return SampleProgress(position, delay, duration, yoyo && hasLoops, inverted, position >= 0);
    }

    const float duration, delay;
    const int64_t loops;
    const double playbackDuration, fullDuration;
    const bool hasLoops;
};

class EaseKernel
{
  public:
    bool IsLinear() const
    {
        return kind_ == 1;
    }

    EaseKernel(int kind, double amplitude = 1.70158, double period = 0)
        : kind_(kind == 0 ? 6 : kind), amplitude_(amplitude), period_(period)
    {
        if (kind < 0 || kind > 36)
            throw std::invalid_argument("a numeric ease must be a built-in kind");
        if (kind >= 32 && kind <= 35 && !(std::isfinite(amplitude) && amplitude > 0))
            throw std::invalid_argument("Flash count must be positive and finite");
    }

    float Sample(double t, double duration = 1) const
    {
        constexpr double pi = 3.1415926535897932384626433832795;
        if (kind_ == 1)
            return t;
        if (kind_ == 36)
            return 1;
        if (kind_ >= 32)
            return Flash(t);
        if (kind_ == 28) {
            const double overshoot = amplitude_ * 1.525;
            const double x = t < .5 ? 2 * t : 2 * t - 2;
            return t < .5 ? x * x * ((overshoot + 1) * x - overshoot) / 2
                          : (x * x * ((overshoot + 1) * x + overshoot) + 2) / 2;
        }
        if (kind_ >= 23 && kind_ <= 25) {
            if (t == 0 || t == 1)
                return t;
            const double amplitude = std::max(1.0, amplitude_);
            const double wave = period_ ? period_ / duration : kind_ == 25 ? .45 : .3;
            const double shift = wave / (2 * pi) * std::asin(1 / amplitude);
            if (kind_ == 23)
                return -amplitude * std::pow(2, 10 * (t - 1)) * std::sin((t - 1 - shift) * 2 * pi / wave);
            if (kind_ == 24)
                return amplitude * std::pow(2, -10 * t) * std::sin((t - shift) * 2 * pi / wave) + 1;
            const double x = 2 * t - 1;
            return x < 0 ? -.5 * amplitude * std::pow(2, 10 * x) * std::sin((x - shift) * 2 * pi / wave)
                         : amplitude * std::pow(2, -10 * x) * std::sin((x - shift) * 2 * pi / wave) * .5 + 1;
        }
        const int direction = (kind_ - 2) % 3;
        if (direction == 0)
            return Inward(t);
        if (direction == 1)
            return 1 - Inward(1 - t);
        return t < .5 ? Inward(2 * t) / 2 : 1 - Inward(2 - 2 * t) / 2;
    }

    double Evaluate(double time, double duration) const
    {
        return Sample(time / duration, duration);
    }

    double SolveClock(double time, double duration, double low, double high, bool inverse) const
    {
        if (low > high)
            std::swap(low, high);
        const auto clock = [&](double progress) {
            return (inverse ? 1 - Sample(1 - progress, duration) : Sample(progress, duration)) * duration;
        };
        const bool increasing = clock(low) < clock(high);
        double progress = low;
        for (int step = 0; step < 32; ++step) {
            progress = (low + high) / 2;
            const double value = clock(progress);
            if (value == time)
                break;
            if ((value < time) == increasing)
                low = progress;
            else
                high = progress;
        }
        return progress;
    }

  private:
    static double Bounce(double t)
    {
        if (t < 1 / 2.75)
            return 7.5625 * t * t;
        if (t < 2 / 2.75) {
            t -= 1.5 / 2.75;
            return 7.5625 * t * t + .75;
        }
        if (t < 2.5 / 2.75) {
            t -= 2.25 / 2.75;
            return 7.5625 * t * t + .9375;
        }
        t -= 2.625 / 2.75;
        return 7.5625 * t * t + .984375;
    }

    double Inward(double t) const
    {
        constexpr double pi = 3.1415926535897932384626433832795;
        switch ((kind_ - 2) / 3) {
        case 0:
            return 1 - std::cos(t * pi / 2);
        case 1:
            return t * t;
        case 2:
            return t * t * t;
        case 3:
            return t * t * t * t;
        case 4:
            return t * t * t * t * t;
        case 5:
            return t == 0 ? 0 : std::pow(2, 10 * t - 10);
        case 6:
            return 1 - std::sqrt(1 - t * t);
        case 8:
            return (amplitude_ + 1) * t * t * t - amplitude_ * t * t;
        default:
            return 1 - Bounce(1 - t);
        }
    }

    double Flash(double t) const
    {
        const double count = amplitude_;
        double step = std::ceil(t * count);
        const bool forward = std::fmod(step, 2) != 0;
        double phase = t * count - (step - 1);
        if (!forward)
            phase = 1 - phase;
        if (kind_ == 33)
            phase *= phase;
        else if (kind_ == 34)
            phase *= 2 - phase;
        else if (kind_ == 35)
            phase = phase < .5 ? 2 * phase * phase : 1 - 2 * (1 - phase) * (1 - phase);
        const bool evenCount = std::fmod(std::trunc(count), 2) == 0;
        if (forward == evenCount)
            step += 1;
        double weight = period_, weighted = 0, correction = 0;
        if (weight > 0) {
            double tail = count - std::trunc(count);
            if (!evenCount)
                tail = 1 - tail;
            correction = tail * step / count;
            weighted = phase * (count - step) / count;
        } else if (weight < 0) {
            weight = -weight;
            weighted = phase * step / count;
        }
        const double value = phase + (weighted - phase) * weight + correction;
        return std::min(1.0, value);
    }

    int kind_;
    double amplitude_, period_;
};

template <typename Storage> class ScalarValueSnapshot
{
  public:
    using Incremental = double;
    ScalarValueSnapshot(Storage start, Storage delta, bool snapping = false)
        : start_(start), delta_(delta), snapping_(snapping)
    {
    }
    Storage Interpolate(double factor, double incremental = 0, double parentMultiplier = 1,
                        double parentIncremental = 0) const
    {
        // Scalar expressions retain a wider intermediate until assignment.
        // In particular a parent loop's two products must not acquire the
        // intermediate float stores used by the vector-operator route.
        Storage start = start_;
        if (incremental != 0) {
            const double offset = static_cast<double>(delta_) * static_cast<Storage>(incremental);
            start = static_cast<Storage>(static_cast<double>(start) + offset);
        }
        if (parentIncremental != 0) {
            const double scaled = static_cast<double>(delta_) * static_cast<Storage>(parentMultiplier);
            const double offset = scaled * static_cast<Storage>(parentIncremental);
            start = static_cast<Storage>(static_cast<double>(start) + offset);
        }
        const double change = static_cast<double>(delta_) * static_cast<Storage>(factor);
        const double value = static_cast<double>(start) + change;
        return static_cast<Storage>(snapping_ ? std::nearbyint(value) : value);
    }
    Storage Sample(double progress, double duration, const EaseKernel &ease, double incremental) const
    {
        return Interpolate(ease.Sample(progress, duration), incremental);
    }
    Storage SampleNested(double progress, double duration, const EaseKernel &ease, double incremental,
                         double parentMultiplier, double parentIncremental) const
    {
        return Interpolate(ease.Sample(progress, duration), incremental, parentMultiplier, parentIncremental);
    }

  private:
    Storage start_, delta_;
    bool snapping_;
};
using ScalarSnapshot = ScalarValueSnapshot<double>;
using Float32Snapshot = ScalarValueSnapshot<float>;

class RectSnapshot
{
  public:
    using Incremental = double;
    using Value = std::array<float, 4>;
    RectSnapshot(Value start, Value change, bool snapping = false) : start_(start), change_(change), snapping_(snapping)
    {
    }
    Value Interpolate(double factor, double own = 0, double multiplier = 1, double parent = 0) const
    {
        Value value;
        for (size_t i = 0; i < value.size(); ++i) {
            // Rectangle fields are stored before snapping, unlike scalar
            // floats whose rounding precedes the final narrowing.
            const float channel = Float32Snapshot(start_[i], change_[i]).Interpolate(factor, own, multiplier, parent);
            value[i] = snapping_ ? static_cast<float>(std::nearbyint(channel)) : channel;
        }
        return value;
    }
    Value Sample(double progress, double duration, const EaseKernel &ease, double own) const
    {
        return Interpolate(ease.Sample(progress, duration), own);
    }
    Value SampleNested(double progress, double duration, const EaseKernel &ease, double own, double multiplier,
                       double parent) const
    {
        return Interpolate(ease.Sample(progress, duration), own, multiplier, parent);
    }

  private:
    Value start_, change_;
    bool snapping_;
};

class Unsigned32Snapshot
{
  public:
    using Incremental = int64_t;
    Unsigned32Snapshot(uint32_t start, uint32_t change, bool negative)
        : start_(start), change_(change), negative_(negative), amount_(static_cast<float>(change))
    {
    }
    uint32_t Interpolate(double factor, int64_t own = 0, int64_t multiplier = 1, int64_t parent = 0) const
    {
        const uint32_t offset = change_ * uint32_t(own) + (change_ * uint32_t(multiplier)) * uint32_t(parent);
        const uint32_t start = negative_ ? start_ - offset : start_ + offset;
        const double rounded = std::nearbyint(static_cast<double>(amount_) * static_cast<float>(factor));
        const int64_t converted = rounded >= -9223372036854775808.0 && rounded < 9223372036854775808.0
                                      ? static_cast<int64_t>(rounded)
                                      : INT64_MIN;
        return negative_ ? start - uint32_t(converted) : start + uint32_t(converted);
    }
    uint32_t Sample(double progress, double duration, const EaseKernel &ease, int64_t own) const
    {
        return Interpolate(ease.Sample(progress, duration), own);
    }
    uint32_t SampleNested(double progress, double duration, const EaseKernel &ease, int64_t own, int64_t multiplier,
                          int64_t parent) const
    {
        return Interpolate(ease.Sample(progress, duration), own, multiplier, parent);
    }

  private:
    uint32_t start_, change_;
    bool negative_;
    float amount_;
};

class Unsigned64Snapshot
{
  public:
    using Incremental = int64_t;
    Unsigned64Snapshot(uint64_t start, uint64_t change) : start_(start), change_(change)
    {
    }
    uint64_t Interpolate(double factor, int64_t own = 0, int64_t multiplier = 1, int64_t parent = 0) const
    {
        const uint64_t start = start_ + change_ * uint64_t(own) + (change_ * uint64_t(multiplier)) * uint64_t(parent);
        return detail::DecimalUInt64Interpolate(start, change_, static_cast<float>(factor));
    }
    uint64_t Sample(double progress, double duration, const EaseKernel &ease, int64_t own) const
    {
        return Interpolate(ease.Sample(progress, duration), own);
    }
    uint64_t SampleNested(double progress, double duration, const EaseKernel &ease, int64_t own, int64_t multiplier,
                          int64_t parent) const
    {
        return Interpolate(ease.Sample(progress, duration), own, multiplier, parent);
    }

  private:
    uint64_t start_, change_;
};

class SignedIntegerSnapshot
{
  public:
    using Incremental = int64_t;
    SignedIntegerSnapshot(uint64_t start, uint64_t end, int bits)
        : mask_(bits == 32 ? uint64_t(UINT32_MAX) : UINT64_MAX), start_(start & mask_), delta_((end - start) & mask_)
    {
        if (bits != 32 && bits != 64)
            throw std::invalid_argument("Integer width must be 32 or 64");
        first_ = static_cast<float>(Decode(start_));
        change_ = static_cast<float>(Decode(delta_));
    }
    int64_t Start() const
    {
        return Decode(start_);
    }
    int64_t End() const
    {
        return Decode((start_ + delta_) & mask_);
    }
    int64_t Change() const
    {
        return Decode(delta_);
    }
    int64_t Interpolate(double factor, int64_t incremental = 0, int64_t parentMultiplier = 1,
                        int64_t parentIncremental = 0) const
    {
        double base = first_;
        if (incremental != 0 || parentIncremental != 0) {
            uint64_t start = start_ + delta_ * static_cast<uint64_t>(incremental);
            start += delta_ * static_cast<uint64_t>(parentMultiplier) * static_cast<uint64_t>(parentIncremental);
            base = static_cast<float>(Decode(start & mask_));
        }
        const double change = static_cast<double>(change_) * static_cast<float>(factor);
        const double result = std::nearbyint(base + change);
        const double limit = mask_ == UINT32_MAX ? 2147483648.0 : 9223372036854775808.0;
        // Define the numeric conversion at the range boundary; never cast an
        // out-of-range or nonfinite floating value to a C++ signed integer.
        return result >= -limit && result < limit ? static_cast<int64_t>(result)
               : mask_ == UINT32_MAX              ? int64_t(INT32_MIN)
                                                  : INT64_MIN;
    }
    int64_t Sample(double progress, double duration, const EaseKernel &ease, int64_t incremental) const
    {
        return Interpolate(ease.Sample(progress, duration), incremental);
    }
    int64_t SampleNested(double progress, double duration, const EaseKernel &ease, int64_t incremental,
                         int64_t parentMultiplier, int64_t parentIncremental) const
    {
        return Interpolate(ease.Sample(progress, duration), incremental, parentMultiplier, parentIncremental);
    }

  private:
    int64_t Decode(uint64_t value) const
    {
        const uint64_t maximum = mask_ >> 1;
        return value <= maximum ? static_cast<int64_t>(value) : -1 - static_cast<int64_t>(mask_ - value);
    }
    uint64_t mask_, start_, delta_;
    float first_, change_;
};

class RectOffsetInterpolation
{
  public:
    using Value = std::array<int32_t, 4>;

    void Prepare(const Value &start, const Value &change, int64_t own, int64_t parentMultiplier, int64_t parent)
    {
        const auto iterations =
            static_cast<uint32_t>(own) + static_cast<uint32_t>(parentMultiplier) * static_cast<uint32_t>(parent);
        for (size_t axis = 0; axis < base_.size(); ++axis)
            base_[axis] = static_cast<uint32_t>(start[axis]) + static_cast<uint32_t>(change[axis]) * iterations;
    }

    Value Interpolate(const Value &change, float factor) const
    {
        Value result;
        for (size_t axis = 0; axis < result.size(); ++axis) {
            // Reuse the signed scalar kernel's float conversion and rounding.
            // The change is read after a custom callback, while the base was
            // captured before it. Integer offsets wrap before either conversion.
            const uint32_t end = base_[axis] + static_cast<uint32_t>(change[axis]);
            result[axis] = static_cast<int32_t>(SignedIntegerSnapshot(base_[axis], end, 32).Interpolate(factor));
        }
        return result;
    }

    Value Sample(const Value &start, const Value &change, int64_t own, int64_t multiplier, int64_t parent,
                 const EaseKernel &ease, double time, double duration)
    {
        Prepare(start, change, own, multiplier, parent);
        return Interpolate(change, static_cast<float>(ease.Evaluate(time, duration)));
    }

  private:
    std::array<uint32_t, 4> base_{};
};

inline float RectOffsetSpeed(const RectOffsetInterpolation::Value &change, float unitsPerSecond)
{
    const double right = static_cast<float>(change[1]), bottom = static_cast<float>(change[3]);
    return static_cast<float>(std::sqrt(right * right + bottom * bottom)) / unitsPerSecond;
}

using Point = std::array<float, 3>;
template <size_t Channels> class VectorSnapshot
{
  public:
    using Incremental = double;
    using Value = std::array<float, Channels>;
    VectorSnapshot(Value start, Value delta, bool snapping = false) : start_(start), delta_(delta), snapping_(snapping)
    {
    }
    Value Interpolate(double factor, double incremental = 0, double parentMultiplier = 1,
                      double parentIncremental = 0) const
    {
        const float weight = static_cast<float>(factor);
        Value result;
        for (size_t axis = 0; axis < Channels; ++axis) {
            float start = start_[axis];
            if (incremental != 0) {
                const float offset = delta_[axis] * static_cast<float>(incremental);
                start += offset;
            }
            if (parentIncremental != 0) {
                const float scaled = delta_[axis] * static_cast<float>(parentMultiplier);
                const float offset = scaled * static_cast<float>(parentIncremental);
                start += offset;
            }
            // The measured managed reference rounds vector-operator results
            // between loop offsets, then stores the final component expression
            // from a wider intermediate. Preserve both boundaries explicitly.
            result[axis] = static_cast<float>(static_cast<double>(start) + static_cast<double>(delta_[axis]) * weight);
            if (snapping_)
                result[axis] = std::nearbyint(result[axis]);
        }
        return result;
    }
    Value Sample(double progress, double duration, const EaseKernel &ease, double incremental) const
    {
        return Interpolate(ease.Sample(progress, duration), incremental);
    }
    Value SampleNested(double progress, double duration, const EaseKernel &ease, double incremental,
                       double parentMultiplier, double parentIncremental) const
    {
        return Interpolate(ease.Sample(progress, duration), incremental, parentMultiplier, parentIncremental);
    }

  private:
    Value start_, delta_;
    bool snapping_;
};
using Vector2Snapshot = VectorSnapshot<2>;
using Vector3Snapshot = VectorSnapshot<3>;
using Vector4Snapshot = VectorSnapshot<4>;
// Separate Python binding identity: colors return the engine's RGBA lists,
// while Vector4 returns its native vector type. The numerical loop is shared.
class ColorSnapshot : public Vector4Snapshot
{
  public:
    using Vector4Snapshot::Vector4Snapshot;
};

class CircleGeometry
{
  public:
    using Value = std::array<float, 2>;
    CircleGeometry(Value center, float radius) : center_(center), radius_(radius)
    {
    }
    Value Position(float degrees, bool snapping) const
    {
        // The game-facing convention starts at +Y and turns clockwise.
        const float radians = (90.0f - degrees) * 0.017453292519943295f;
        Value result = {center_[0] + std::cos(radians) * radius_, center_[1] + std::sin(radians) * radius_};
        if (snapping)
            for (auto &component : result)
                component = std::nearbyint(component);
        return result;
    }
    Value Interpolate(float start, float change, float factor, float ownIncrement, float parentMultiplier,
                      float parentIncrement, bool snapping) const
    {
        // Keep the parent's two multiplications and both additions separate:
        // merging these into one factor changes observable float rounding.
        if (ownIncrement != 0)
            start += change * ownIncrement;
        if (parentIncrement != 0)
            start += (change * parentMultiplier) * parentIncrement;
        return Position(start + change * factor, snapping);
    }
    Value Sample(double progress, double duration, const EaseKernel &ease, float start, float change,
                 float ownIncrement, float parentMultiplier, float parentIncrement, bool snapping) const
    {
        return Interpolate(start, change, static_cast<float>(ease.Sample(progress, duration)), ownIncrement,
                           parentMultiplier, parentIncrement, snapping);
    }

  private:
    Value center_;
    float radius_;
};

// Bind stable parameters at configuration time. Frame calls then pass only
// progress and the loop offset, without a Python ease-object conversion.
template <typename Snapshot> class ValueEvaluator
{
  public:
    using Incremental = typename Snapshot::Incremental;
    ValueEvaluator(Snapshot value, EaseKernel ease, double duration)
        : value_(std::move(value)), ease_(ease), duration_(duration)
    {
    }
    auto Sample(double progress, Incremental incremental) const
    {
        return value_.Sample(progress, duration_, ease_, incremental);
    }
    auto SampleNested(double progress, Incremental incremental, Incremental parentMultiplier,
                      Incremental parentIncremental) const
    {
        return value_.SampleNested(progress, duration_, ease_, incremental, parentMultiplier, parentIncremental);
    }

  private:
    Snapshot value_;
    EaseKernel ease_;
    double duration_;
};

struct SegmentSample
{
    size_t index;
    double width, progress;
};

inline float ArrayDuration(const std::vector<double> &durations)
{
    // Segment clocks store each accumulated boundary in single precision.
    // Summing in double and rounding only the result changes loop endpoints.
    float total = 0;
    for (double duration : durations)
        total += static_cast<float>(duration);
    return total;
}

class SegmentClock
{
  public:
    explicit SegmentClock(std::vector<double> durations) : durations_(std::move(durations))
    {
        if (durations_.empty())
            throw std::invalid_argument("array tween requires at least one duration");
        ends_.reserve(durations_.size());
        float total = 0;
        for (size_t index = 0; index < durations_.size(); ++index) {
            ordered_ = ordered_ && std::isfinite(durations_[index]) && durations_[index] >= 0;
            durations_[index] = static_cast<float>(durations_[index]);
            total += static_cast<float>(durations_[index]);
            ends_.push_back(total);
        }
        total_ = total;
    }

    SegmentSample Locate(double progress, double duration) const
    {
        return LocateTime(progress * total_, duration);
    }

    SegmentSample LocateTime(double elapsed, double) const
    {
        const float time = static_cast<float>(elapsed);
        size_t index = 0;
        if (ordered_) {
            const auto upper = std::lower_bound(ends_.begin(), ends_.end(), time);
            index = std::isnan(time) ? ends_.size() : static_cast<size_t>(upper - ends_.begin());
        } else {
            while (index < ends_.size() && !(time <= ends_[index]))
                ++index;
        }
        // A clock with no matching interval selects the first segment and
        // uses the accumulated duration, including a retimed outer timeline.
        if (index == ends_.size())
            return {0, durations_[0], total_ / durations_[0]};
        const double start = index ? ends_[index - 1] : 0;
        const double width = durations_[index];
        const float localTime = time - static_cast<float>(start);
        return {index, width, localTime / width};
    }

    size_t Size() const
    {
        return durations_.size();
    }

  private:
    std::vector<double> durations_, ends_;
    double total_;
    bool ordered_ = true;
};

template <typename Vector>
std::vector<float> SegmentSpeedDurations(const std::vector<Vector> &changes, const std::vector<float> &speeds)
{
    if (changes.size() != speeds.size())
        throw std::invalid_argument("segment speeds must match changes");
    std::vector<float> result;
    result.reserve(changes.size());
    for (size_t i = 0; i < changes.size(); ++i) {
        const auto &v = changes[i];
        const float squared = (v[0] * v[0] + v[1] * v[1]) + v[2] * v[2];
        result.push_back(std::sqrt(squared) / speeds[i]);
    }
    return result;
}

class SegmentSnapshot : public SegmentClock
{
  public:
    SegmentSnapshot(std::vector<Point> starts, std::vector<Point> ends, std::vector<double> durations)
        : SegmentClock(std::move(durations)), starts_(std::move(starts))
    {
        if (starts_.size() != ends.size() || starts_.size() != Size())
            throw std::invalid_argument("array tween segments require matching inputs");
        deltas_.reserve(starts_.size());
        for (size_t index = 0; index < starts_.size(); ++index) {
            Point delta;
            for (int axis = 0; axis < 3; ++axis)
                delta[axis] = ends[index][axis] - starts_[index][axis];
            deltas_.push_back(delta);
        }
        for (int axis = 0; axis < 3; ++axis)
            incremental_[axis] = ends.back()[axis] - starts_.front()[axis];
    }

    Point Interpolate(size_t index, double factor, double incremental) const
    {
        Point result;
        const float weight = static_cast<float>(factor), loops = static_cast<float>(incremental);
        for (int axis = 0; axis < 3; ++axis)
            result[axis] = (starts_[index][axis] + deltas_[index][axis] * weight) + incremental_[axis] * loops;
        return result;
    }
    Point Sample(double progress, double duration, const EaseKernel &ease, double incremental) const
    {
        const auto segment = Locate(progress, duration);
        return Interpolate(segment.index, ease.Sample(segment.progress, segment.width), incremental);
    }
    Point SampleTime(double elapsed, double duration, const EaseKernel &ease, double incremental) const
    {
        const auto segment = LocateTime(elapsed, duration);
        return Interpolate(segment.index, ease.Sample(segment.progress, segment.width), incremental);
    }
    const SegmentSnapshot &Frozen() const
    {
        return *this;
    }

  private:
    std::vector<Point> starts_, deltas_;
    Point incremental_;
};

template <typename Vector> class LiveSegmentSnapshot : public SegmentClock
{
  public:
    using Array = std::vector<Vector>;
    LiveSegmentSnapshot(std::shared_ptr<Array> starts, std::shared_ptr<Array> changes, std::vector<double> durations)
        : SegmentClock(std::move(durations)), starts_(std::move(starts)), changes_(std::move(changes))
    {
        if (starts_->size() != Size() || changes_->size() != Size())
            throw std::invalid_argument("array tween segments require matching inputs");
    }
    SegmentSample Locate(double progress, double duration) const
    {
        return LocateTime(progress * duration, duration);
    }
    Point Interpolate(size_t index, double factor, double incremental) const
    {
        Point result;
        const float weight = static_cast<float>(factor), loops = static_cast<float>(incremental);
        for (int axis = 0; axis < 3; ++axis) {
            const float offset = ((*starts_).back()[axis] + (*changes_).back()[axis]) - (*starts_).front()[axis];
            result[axis] = ((*starts_)[index][axis] + (*changes_)[index][axis] * weight) + offset * loops;
        }
        return result;
    }
    Point Sample(double progress, double duration, const EaseKernel &ease, double incremental) const
    {
        return SampleTime(progress * duration, duration, ease, incremental);
    }
    Point SampleTime(double elapsed, double duration, const EaseKernel &ease, double incremental) const
    {
        if (duration == 0)
            return Interpolate(0, 1., incremental);
        const auto segment = LocateTime(elapsed, duration);
        return Interpolate(segment.index, ease.Sample(segment.progress, segment.width), incremental);
    }
    LiveSegmentSnapshot Frozen() const
    {
        return LiveSegmentSnapshot(*this, std::make_shared<Array>(*starts_), std::make_shared<Array>(*changes_));
    }

  private:
    LiveSegmentSnapshot(const LiveSegmentSnapshot &source, std::shared_ptr<Array> starts,
                        std::shared_ptr<Array> changes)
        : SegmentClock(source), starts_(std::move(starts)), changes_(std::move(changes))
    {
    }
    std::shared_ptr<Array> starts_, changes_;
};
} // namespace infernux::tween
