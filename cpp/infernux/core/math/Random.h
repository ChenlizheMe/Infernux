#pragma once

#include "core/FoundationApi.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
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

/// Owner-selected gameplay stream. Unsigned arithmetic fixes the state transition on
/// all targets; snapshots and batches own their data and never borrow Python.
class Random
{
  public:
    // Internal native context. The public RandomState/pickle stays four words.
    // Scrambling is lazy so worlds without string tweens allocate no tables.
    struct ScrambleState
    {
        std::array<std::u16string, 4> text{u"ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvxyz1234567890",
                                           u"ABCDEFGHIJKLMNOPQRSTUVXYZ", u"abcdefghijklmnopqrstuvxyz", u"1234567890"};
        bool initialized = false;
        int last = 0;
    };

    struct Snapshot
    {
        RandomState state;
        std::optional<ScrambleState> scramble;
    };

    struct Storage
    {
        INFERNUX_FOUNDATION_API Storage();
        std::mutex mutex;
        RandomState state;
        std::optional<ScrambleState> scramble;
    };

    // One authoring/tool context also supports standalone Foundation callers.
    // Engine ownership sets a per-thread default; explicit scopes override it.
    INFERNUX_FOUNDATION_API static std::shared_ptr<Storage> ToolStorage();
    INFERNUX_FOUNDATION_API static void SetThreadDefaultStorage(std::shared_ptr<Storage> storage);
    INFERNUX_FOUNDATION_API static Snapshot CaptureStorage(Storage &storage);
    INFERNUX_FOUNDATION_API static void RestoreStorage(Storage &storage, const Snapshot &snapshot);

    class StorageScope
    {
      public:
        INFERNUX_FOUNDATION_API explicit StorageScope(std::shared_ptr<Storage> storage) noexcept;
        INFERNUX_FOUNDATION_API ~StorageScope();
        StorageScope(const StorageScope &) = delete;
        StorageScope &operator=(const StorageScope &) = delete;

      private:
        std::shared_ptr<Storage> m_storage;
        Storage *m_previous;
    };

    // A native operation can draw/shuffle under one lock instead of locking
    // once per element. No Python or user callbacks execute in this scope.
    class Scope
    {
      public:
        Scope() : storage_(Shared()), lock_(storage_.mutex), state_(storage_.state)
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
        ScrambleState &Scramble()
        {
            if (!storage_.scramble)
                storage_.scramble.emplace();
            return *storage_.scramble;
        }

      private:
        Storage &storage_;
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

    // TLS is defined once in Foundation, so every engine DLL observes the same
    // scoped/default owner on a thread, without a dependency on SceneManager.
    INFERNUX_FOUNDATION_API static Storage &Shared();
};
} // namespace infernux
