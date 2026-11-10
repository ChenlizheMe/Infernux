#pragma once

#include "IoAffinity.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <thread>

namespace infernux::detail
{
inline constexpr size_t kMinimumFileWindow = 64 * 1024;
inline constexpr size_t kMaximumFileWindow = 4 * 1024 * 1024;

struct FileIoWindow
{
    uint64_t bytes;
    uint64_t prediction;
};

inline uint64_t PredictFileIoNanoseconds(uint64_t bytes, double bytesPerNanosecond)
{
    if (!bytes || !bytesPerNanosecond)
        return 0;
    return static_cast<uint64_t>(std::clamp(std::ceil(static_cast<double>(bytes) / bytesPerNanosecond), 1.0,
                                            static_cast<double>((std::numeric_limits<int64_t>::max)())));
}

inline FileIoWindow PlanFileIoWindow(uint64_t remaining, uint64_t allowance, double bytesPerNanosecond,
                                     size_t maximum = kMaximumFileWindow)
{
    constexpr size_t initial = 256 * 1024;
    if (!remaining)
        return {};
    const auto window =
        bytesPerNanosecond
            ? static_cast<size_t>(std::clamp(static_cast<double>(allowance) * bytesPerNanosecond,
                                             static_cast<double>(kMinimumFileWindow), static_cast<double>(maximum)))
            : initial;
    const auto bytes = (std::min<uint64_t>)(remaining, window);
    return {bytes, PredictFileIoNanoseconds(bytes, bytesPerNanosecond)};
}

inline void ObserveFileIoWindow(double &bytesPerNanosecond, uint64_t bytes, uint64_t elapsed,
                                size_t minimumSampleBytes = kMinimumFileWindow)
{
    if (bytes && elapsed && (bytes >= minimumSampleBytes || !bytesPerNanosecond))
        bytesPerNanosecond = static_cast<double>(bytes) / static_cast<double>(elapsed);
}

struct IoStepState
{
    static constexpr uint8_t active = 1, cancel = 2;
    std::string error;
    std::thread::id owner = std::this_thread::get_id();
    std::atomic<size_t> retained{0}, reservation{0};
    std::atomic<uint64_t> executionNs{0}, submissions{0};
    std::atomic<bool> ranOnWorker{false};
    std::atomic<uint8_t> control{0};
};

using IoClock = std::chrono::steady_clock;

inline uint64_t IoElapsedNs(IoClock::time_point start)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(IoClock::now() - start).count());
}

template <class State> void FinishIoStep(State &state, IoClock::time_point started)
{
    state.Snapshot();
    state.executionNs.fetch_add(IoElapsedNs(started), std::memory_order_relaxed);
    // Last access to the cursor on the ordinary path. Cancel marks this
    // callback, or sees idle and hands cleanup to a new IO job.
    uint8_t expected = State::active;
    if (!state.control.compare_exchange_strong(expected, 0, std::memory_order_acq_rel)) {
        const auto cleanupStarted = IoClock::now();
        try {
            state.RetireCancelled();
        } catch (const std::exception &error) {
            state.Fail(error);
        }
        state.Snapshot();
        state.executionNs.fetch_add(IoElapsedNs(cleanupStarted), std::memory_order_relaxed);
        state.control.store(State::cancel, std::memory_order_release);
    }
}

template <class State, class Work> JobHandle DispatchIo(const std::shared_ptr<State> &state, Work work)
{
    state->control.fetch_or(State::active, std::memory_order_acq_rel);
    JobHandle job;
    try {
        job = JobSystem::Get().Schedule(
            [state, work] {
                const auto started = IoClock::now();
                state->ranOnWorker.store(std::this_thread::get_id() != state->owner, std::memory_order_relaxed);
                try {
                    work(*state);
                } catch (const std::exception &error) {
                    state->Fail(error);
                }
                FinishIoStep(*state, started);
            },
            JobDomain::IO, JobPriority::Low, IoJobAffinity);
    } catch (...) {
        state->control.fetch_and(static_cast<uint8_t>(~State::active), std::memory_order_release);
        throw;
    }
    state->submissions.fetch_add(1, std::memory_order_relaxed);
    return job;
}
} // namespace infernux::detail
