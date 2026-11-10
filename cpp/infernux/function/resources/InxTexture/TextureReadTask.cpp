#include "TextureReadTask.h"
#include <core/threading/FrameWorkOwner.h>
#include <platform/filesystem/IoStep.h>

namespace infernux
{
struct TextureReadTask::State : detail::IoStepState
{
    State(std::string path, std::string hash) : cursor(std::move(path), std::move(hash))
    {
    }
    // Destroy the cursor before returning the claim, even if the IO callback
    // is the final owner of this State.
    std::shared_ptr<FrameCpuStorageClaim> storage;
    TextureReadCursor cursor;
    void Snapshot()
    {
        const auto bytes = sizeof(State) - sizeof(TextureReadCursor) + cursor.GetRetainedBytes();
        retained.store(bytes, std::memory_order_relaxed);
        reservation.store(bytes, std::memory_order_relaxed);
    }
    void Fail(const std::exception &failure)
    {
        error = failure.what();
        cursor.Close();
    }
    void RetireCancelled()
    {
        cursor.Close();
    }
};

TextureReadTask::TextureReadTask(std::string path, std::string sourceHash)
    : m_state(std::make_shared<State>(std::move(path), std::move(sourceHash)))
{
    m_state->Snapshot();
}
TextureReadTask::~TextureReadTask() = default;
bool TextureReadTask::IsPending() const noexcept
{
    return !m_job.IsComplete();
}
bool TextureReadTask::IsRetired() const noexcept
{
    return !IsPending() && ((m_state->control.load(std::memory_order_acquire) & State::cancel) ||
                            !m_state->error.empty() || m_state->cursor.GetState() == TextureReadState::Complete);
}
TextureReadTask::State &TextureReadTask::Idle() const
{
    if (IsPending())
        throw std::logic_error("texture IO step is still running");
    if (!m_state->error.empty())
        throw std::runtime_error(m_state->error);
    if (m_state->control.load(std::memory_order_acquire) & State::cancel)
        throw std::logic_error("texture read was cancelled");
    return *m_state;
}
TextureReadState TextureReadTask::GetState() const
{
    return Idle().cursor.GetState();
}
const std::string &TextureReadTask::GetError() const
{
    if (IsPending())
        throw std::logic_error("texture IO step is still running");
    return m_state->error;
}
const TextureArtifactDescription &TextureReadTask::GetDescription() const
{
    return Idle().cursor.GetDescription();
}
const std::vector<uint8_t> &TextureReadTask::GetOutput() const
{
    return Idle().cursor.GetOutput();
}
size_t TextureReadTask::GetRetainedBytes() const noexcept
{
    return sizeof(*this) + m_state->retained.load(std::memory_order_relaxed);
}
size_t TextureReadTask::GetReservationBytes(size_t batchBytes) const
{
    if (IsPending())
        return sizeof(*this) + m_state->reservation.load(std::memory_order_relaxed);
    if (IsRetired())
        return GetRetainedBytes();
    return sizeof(*this) + sizeof(State) - sizeof(TextureReadCursor) + Idle().cursor.GetReservationBytes(batchBytes);
}
uint64_t TextureReadTask::GetExecutionNanoseconds() const noexcept
{
    return m_state->executionNs.load(std::memory_order_relaxed);
}
void TextureReadTask::Poll(FrameWorkOwner &owner)
{
    if (IsPending())
        return;
    const auto elapsed = GetExecutionNanoseconds();
    const auto duration = elapsed - m_observedNanoseconds;
    owner.ObserveResource(FrameWorkResource::Io, duration, m_frame, m_grant);
    if (m_state->error.empty() && !(m_state->control.load(std::memory_order_acquire) & State::cancel))
        detail::ObserveFileIoWindow(m_bytesPerNanosecond, m_submittedBytes, duration);
    m_submittedBytes = 0;
    m_observedNanoseconds = elapsed;
    m_frame = m_grant = 0;
    if (m_state->storage && IsRetired())
        m_state->storage->ReduceTo(GetRetainedBytes());
}
std::optional<TextureReadTask::ReadStep> TextureReadTask::Plan(const FrameWorkOwner &owner) const
{
    const auto allowance = owner.ResourceRemaining(FrameWorkResource::Io);
    if (!allowance || IsPending() || IsRetired())
        return {};

    const auto &storage = m_state->storage;
    // Keep one small window available until close. Larger windows use only
    // current headroom, so opening more cursors cannot strand all the readers.
    const auto reserved = storage ? storage->Bytes() : GetReservationBytes(0) + detail::kMinimumFileWindow;
    if (!storage && !owner.CanReserveCpuStorage(reserved))
        return {};
    auto step = ReadStep{0, m_batchBytes, 0, reserved};
    const auto &cursor = Idle().cursor;
    if (cursor.GetState() == TextureReadState::Reading) {
        const auto unit = GetTextureFormatLayout(GetDescription().format).bytesPerBlock;
        if (!step.batchBytes) {
            // Admit the whole output window once. Read slices reuse that
            // reservation until the upload consumer takes the batch.
            const auto transfer = owner.ResourceState(FrameWorkResource::Transfer).limit;
            const auto maximum = (std::min<uint64_t>)(detail::kMaximumFileWindow, (std::max<uint64_t>)(unit, transfer));
            auto batch = cursor.DescribeWork(static_cast<size_t>(maximum)).bytes;
            const auto needed = GetReservationBytes(batch);
            if (needed > reserved) {
                const auto capacity = owner.CpuStorageLimit();
                const auto headroom = capacity - (std::min)(capacity, owner.CpuStorageBytes());
                const auto available = storage ? reserved + headroom : (std::max)(reserved, headroom);
                if (needed > available)
                    batch -= (std::min<uint64_t>)(batch, needed - available);
            }
            step.batchBytes = batch - batch % unit;
            if (!step.batchBytes)
                return {};
            step.reservation = (std::max<uint64_t>)(reserved, GetReservationBytes(step.batchBytes));
        }
        const auto remaining = step.batchBytes - cursor.GetBufferedBytes();
        const auto bytes = detail::PlanFileIoWindow(remaining, allowance, m_bytesPerNanosecond).bytes;
        step.bytes = static_cast<size_t>(bytes - bytes % unit);
        step.prediction = detail::PredictFileIoNanoseconds(step.bytes, m_bytesPerNanosecond);
    } else if (cursor.GetState() != TextureReadState::Opening && cursor.GetState() != TextureReadState::Closing) {
        return {};
    }
    if (!owner.CanAdmitResource(FrameWorkResource::Io, step.prediction, true))
        return {};
    return step;
}
bool TextureReadTask::CanAdvance(const FrameWorkOwner &owner) const
{
    return Plan(owner).has_value();
}
bool TextureReadTask::TryAdvance(FrameWorkOwner &owner)
{
    Poll(owner);
    const auto step = Plan(owner);
    if (!step)
        return false;
    auto &storage = m_state->storage;
    if (!storage)
        storage = owner.TryReserveCpuStorage(step->reservation);
    if (!storage || !owner.TryGrowCpuStorage(*storage, step->reservation))
        return false;
    Advance(step->bytes, step->batchBytes);
    m_frame = owner.Opportunity();
    m_grant = owner.RecordAcceptedResource(FrameWorkResource::Io, step->prediction);
    return true;
}
void TextureReadTask::Advance()
{
    const auto &cursor = Idle().cursor;
    if (cursor.GetState() != TextureReadState::Reading) {
        Advance(0, 0);
        return;
    }
    // Explicit cold completion may take over a batch already reserved by
    // managed IO. Finish it within the same allocation.
    const auto limit = m_state->storage ? detail::kMinimumFileWindow : detail::kMaximumFileWindow;
    const auto batch = m_batchBytes ? m_batchBytes : cursor.DescribeWork(limit).bytes;
    Advance(batch - cursor.GetBufferedBytes(), batch);
}
void TextureReadTask::Advance(size_t bytes, size_t batchBytes)
{
    auto &cursor = Idle().cursor;
    m_state->reservation.store(sizeof(State) - sizeof(TextureReadCursor) + cursor.GetReservationBytes(batchBytes),
                               std::memory_order_relaxed);
    m_batchBytes = batchBytes;
    m_submittedBytes = bytes;
    m_job = detail::DispatchIo(m_state, [bytes, batchBytes](State &state) { state.cursor.Advance(bytes, batchBytes); });
}
void TextureReadTask::ConsumeOutput()
{
    Idle().cursor.ConsumeOutput();
    m_batchBytes = 0;
    m_state->Snapshot();
}
void TextureReadTask::Cancel()
{
    const auto previous = m_state->control.fetch_or(State::cancel, std::memory_order_acq_rel);
    if (previous & State::cancel)
        return;
    if (!(previous & State::active)) {
        if (m_state->cursor.GetState() == TextureReadState::Opening ||
            m_state->cursor.GetState() == TextureReadState::Complete ||
            m_state->cursor.GetState() == TextureReadState::Closed) {
            m_state->cursor.Close();
            m_state->Snapshot();
        } else
            m_job = detail::DispatchIo(m_state, [](State &state) { state.cursor.Close(); });
    }
}
void TextureReadTask::Wait()
{
    if (IsPending())
        JobSystem::Get().WaitPassiveCompletion(m_job);
}
} // namespace infernux
