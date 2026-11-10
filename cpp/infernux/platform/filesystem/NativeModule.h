#pragma once

#include "core/FoundationApi.h"

#include "IoWork.h"
#include "VerifiedFile.h"
#include <core/threading/JobSystem.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(INX_PLATFORM_WINDOWS) || defined(INX_PLATFORM_LINUX)
namespace infernux
{
/// One target-native code image. Entries retain this owner through actual work.
/// Cold load and the IO task drive the same inspection and mapping operations.
class NativeModule
{
  public:
    INFERNUX_FOUNDATION_API ~NativeModule();
    INFERNUX_FOUNDATION_API static std::shared_ptr<NativeModule> Load(std::shared_ptr<VerifiedFile> file,
                                                                      std::vector<std::string> symbols);
    INFERNUX_FOUNDATION_API uintptr_t Identity() const;
    INFERNUX_FOUNDATION_API size_t ImageBytes() const;
    INFERNUX_FOUNDATION_API uintptr_t Address(const std::string &symbol) const;
    INFERNUX_FOUNDATION_API bool IsPublished() const;
    INFERNUX_FOUNDATION_API void RequirePublished() const;
    // The model loader publishes only after its code and state are complete.
    INFERNUX_FOUNDATION_API void Publish();
    INFERNUX_FOUNDATION_API void Discard();

  private:
    friend class NativeModuleTask;
    NativeModule(std::shared_ptr<VerifiedFile> file, std::vector<std::string> symbols);
    void Inspect(size_t maxBytes);
    bool IsInspected() const;
    bool IsLoaded() const;
    void Map();
    size_t RetainedBytes() const;
    uint64_t ReadBytes() const;
    uint64_t LoadNs() const;
    IoWork DescribeWork(size_t maxBytes) const;
    IoRecipe DescribeRemainingWork() const;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/// Inspect, admit the measured image size, map, then publish or retire privately.
/// The caller supplies admission between inspected and loaded; each Advance
/// uses the original IO domain. OS mapping remains an indivisible operation.
class NativeModuleTask
{
  public:
    INFERNUX_FOUNDATION_API NativeModuleTask(std::shared_ptr<VerifiedFile> file, std::vector<std::string> symbols);
    INFERNUX_FOUNDATION_API ~NativeModuleTask();
    INFERNUX_FOUNDATION_API void Advance(size_t maxBytes);
    INFERNUX_FOUNDATION_API std::shared_ptr<NativeModule> GetModule() const;
    INFERNUX_FOUNDATION_API void Commit();
    INFERNUX_FOUNDATION_API bool Cancel();
    INFERNUX_FOUNDATION_API void Wait();
    INFERNUX_FOUNDATION_API bool IsPending() const;
    INFERNUX_FOUNDATION_API bool IsComplete() const;
    INFERNUX_FOUNDATION_API std::string GetStateName() const;
    INFERNUX_FOUNDATION_API std::string GetError() const;
    INFERNUX_FOUNDATION_API size_t GetRetainedBytes() const;
    INFERNUX_FOUNDATION_API size_t GetReservationBytes() const;
    INFERNUX_FOUNDATION_API size_t GetImageBytes() const;
    INFERNUX_FOUNDATION_API uintptr_t GetModuleIdentity() const;
    INFERNUX_FOUNDATION_API uint64_t GetReadBytes() const;
    INFERNUX_FOUNDATION_API uint64_t GetLoadNs() const;
    INFERNUX_FOUNDATION_API uint64_t GetExecutionNs() const;
    INFERNUX_FOUNDATION_API uint64_t GetSubmissionCount() const;
    INFERNUX_FOUNDATION_API IoWork DescribeWork(size_t maxBytes) const;
    INFERNUX_FOUNDATION_API IoRecipe DescribeRemainingWork() const;
    INFERNUX_FOUNDATION_API IoWork GetSubmittedWork() const;
    INFERNUX_FOUNDATION_API bool RanOnWorker() const;

  private:
    struct State;
    std::shared_ptr<State> m_state;
    mutable std::mutex m_mutex;
    JobHandle m_job;
    IoWork m_submittedWork;
    State &Idle() const;
    void Dispatch(size_t maximum);
};
} // namespace infernux
#endif
