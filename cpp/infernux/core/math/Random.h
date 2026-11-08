#pragma once

#include "core/FoundationApi.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace infernux
{
struct RandomState
{
    std::array<uint32_t, 4> words{};
    bool operator==(const RandomState &other) const
    {
        return words == other.words;
    }
};

/// Shared gameplay stream. Unsigned arithmetic fixes the state transition on
/// all targets; snapshots and batches own their data and never borrow Python.
class Random
{
  public:
    // A native operation can draw/shuffle under one lock instead of locking
    // once per element. No Python or user callbacks execute in this scope.
    class Scope
    {
      public:
        Scope() : lock_(Shared().mutex), state_(Shared().state)
        {
        }
        int32_t Range(int32_t minimum, int32_t maximum)
        {
            return IntegerRange(state_, minimum, maximum);
        }
        template <typename Value> void Shuffle(Value *values, size_t count)
        {
            for (size_t i = 0; i < count; ++i)
                std::swap(values[i], values[Range(static_cast<int32_t>(i), static_cast<int32_t>(count))]);
        }

      private:
        std::lock_guard<std::mutex> lock_;
        RandomState &state_;
    };

    static void InitState(int32_t seed)
    {
        auto &storage = Shared();
        std::lock_guard<std::mutex> lock(storage.mutex);
        storage.state = Seed(static_cast<uint32_t>(seed));
    }

    static RandomState GetState()
    {
        auto &storage = Shared();
        std::lock_guard<std::mutex> lock(storage.mutex);
        return storage.state;
    }

    static void SetState(const RandomState &state)
    {
        auto &storage = Shared();
        std::lock_guard<std::mutex> lock(storage.mutex);
        storage.state = state;
    }

    static float Value()
    {
        auto &storage = Shared();
        std::lock_guard<std::mutex> lock(storage.mutex);
        return Unit(storage.state);
    }

    static float Range(float minimum, float maximum)
    {
        auto &storage = Shared();
        std::lock_guard<std::mutex> lock(storage.mutex);
        return FloatRange(storage.state, minimum, maximum);
    }

    static int32_t Range(int32_t minimum, int32_t maximum)
    {
        if (minimum == maximum)
            return minimum;
        auto &storage = Shared();
        std::lock_guard<std::mutex> lock(storage.mutex);
        return IntegerRange(storage.state, minimum, maximum);
    }

    static std::vector<float> RangeMany(const std::vector<std::array<float, 2>> &bounds)
    {
        std::vector<float> result(bounds.size());
        auto &storage = Shared();
        std::lock_guard<std::mutex> lock(storage.mutex);
        for (size_t index = 0; index < bounds.size(); ++index)
            result[index] = FloatRange(storage.state, bounds[index][0], bounds[index][1]);
        return result;
    }

  private:
    static int32_t IntegerRange(RandomState &state, int32_t minimum, int32_t maximum)
    {
        if (minimum == maximum)
            return minimum;
        const int64_t span = static_cast<int64_t>(maximum) - minimum;
        const uint32_t distance = Next(state) % static_cast<uint64_t>(span < 0 ? -span : span);
        return static_cast<int32_t>(static_cast<int64_t>(minimum) +
                                    (span < 0 ? -static_cast<int64_t>(distance) : distance));
    }

    static RandomState Seed(uint32_t seed)
    {
        RandomState state{{seed, 0, 0, 0}};
        for (size_t index = 1; index < state.words.size(); ++index)
            state.words[index] = 1812433253u * state.words[index - 1] + 1u;
        return state;
    }

    static uint32_t Next(RandomState &state)
    {
        auto &s = state.words;
        const uint32_t mix = s[0] ^ (s[0] << 11);
        s = {s[1], s[2], s[3], s[3] ^ (s[3] >> 19) ^ mix ^ (mix >> 8)};
        return s[3];
    }

    static float Unit(RandomState &state)
    {
        const volatile float numerator = static_cast<float>(Next(state) & 0x7fffffu);
        const volatile float denominator = 8388607.f;
        return numerator / denominator;
    }

    static float FloatRange(RandomState &state, float minimum, float maximum)
    {
        const float value = Unit(state);
        // Separate rounded products also prevent FMA contraction from changing
        // the recorded range result on a different compiler/processor.
        const volatile float first = minimum * value;
        const volatile float second = maximum * (1.f - value);
        return first + second;
    }

    struct Storage
    {
        std::mutex mutex;
        RandomState state =
            Seed(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    };

    // Defined once in Foundation, so callers in different engine DLLs share
    // one stream rather than each instantiating a header-local singleton.
    INFERNUX_FOUNDATION_API static Storage &Shared();
};
} // namespace infernux
