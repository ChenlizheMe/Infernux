#pragma once

#include "TextureArtifact.h"
#include <core/threading/JobSystem.h>

namespace infernux
{
class FrameWorkOwner;
// Measured on the registry owner thread and shared by its texture readers.
struct TextureReadProfile
{
    uint64_t openNanoseconds = 0, closeNanoseconds = 0;
    double bytesPerNanosecond = 0;
};
/// Owner-thread control of a cursor on the original IO worker domain. A busy
/// task exposes only atomic accounting; its output is borrowed until Consume.
/// Cancel withdraws future steps and retires accepted IO on its actual worker.
class TextureReadTask final
{
  public:
    TextureReadTask(std::string path, std::string sourceHash, std::shared_ptr<TextureReadProfile> profile);
    ~TextureReadTask();
    bool IsPending() const noexcept;
    bool IsRetired() const noexcept;
    TextureReadState GetState() const;
    const std::string &GetError() const;
    const TextureArtifactDescription &GetDescription() const;
    const std::vector<uint8_t> &GetOutput() const;
    size_t GetRetainedBytes() const noexcept;
    size_t GetReservationBytes(size_t batchBytes) const;
    uint64_t GetExecutionNanoseconds() const noexcept;
    // Managed admission and real receipt collection use the same IO worker as
    // cold Advance. Poll must run before replacing an accepted step.
    bool CanAdvance(const FrameWorkOwner &owner) const;
    bool TryAdvance(FrameWorkOwner &owner);
    void Poll(FrameWorkOwner &owner);
    void Advance();
    void ConsumeOutput();
    void Cancel();
    void Wait();

  private:
    struct State;
    struct ReadStep
    {
        size_t bytes;
        size_t batchBytes;
        uint64_t prediction;
        uint64_t reservation;
    };
    std::shared_ptr<State> m_state;
    std::shared_ptr<TextureReadProfile> m_profile;
    JobHandle m_job;
    uint64_t m_observedNanoseconds = 0, m_frame = 0, m_grant = 0;
    size_t m_submittedBytes = 0, m_batchBytes = 0;
    TextureReadState m_submittedState = TextureReadState::Opening;
    State &Idle() const;
    [[nodiscard]] std::optional<ReadStep> Plan(const FrameWorkOwner &owner) const;
    void Advance(size_t bytes, size_t batchBytes);
};
} // namespace infernux
