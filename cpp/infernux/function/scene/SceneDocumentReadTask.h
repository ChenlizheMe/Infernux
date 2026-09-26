#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <platform/filesystem/AtomicFile.h>
#include <string>

namespace infernux
{

class SceneDocumentReadTicket
{
  public:
    enum class Status : uint8_t
    {
        Pending,
        Ready,
        Failed,
        Cancelled,
        Consumed,
    };

    SceneDocumentReadTicket() = default;

    [[nodiscard]] bool IsComplete() const noexcept;
    [[nodiscard]] bool IsReady() const noexcept;
    [[nodiscard]] bool RanOnWorker() const noexcept;
    [[nodiscard]] std::string GetStatusName() const;
    [[nodiscard]] std::string GetError() const;
    [[nodiscard]] std::optional<AtomicFileState> GetFileState() const;
    bool Cancel();
    nlohmann::json TakeDocument();

  private:
    struct State
    {
        std::atomic<Status> status{Status::Pending};
        std::atomic<bool> cancelRequested{false};
        std::atomic<uint64_t> callerThread{0};
        std::atomic<uint64_t> workerThread{0};
        mutable std::mutex mutex;
        nlohmann::json document;
        std::optional<AtomicFileState> fileState;
        std::string error;
    };

    explicit SceneDocumentReadTicket(std::shared_ptr<State> state) : m_state(std::move(state))
    {
    }

    std::shared_ptr<State> m_state;

    friend SceneDocumentReadTicket ScheduleSceneDocumentRead(const std::string &path);
};

SceneDocumentReadTicket ScheduleSceneDocumentRead(const std::string &path);

} // namespace infernux
