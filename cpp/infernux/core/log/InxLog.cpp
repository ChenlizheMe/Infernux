#include "InxLog.h"
#include <condition_variable>
#include <exception>
#include <platform/filesystem/InxPath.h>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace infernux
{
namespace detail
{
struct DiagnosticSinkState
{
    std::mutex queueMutex;
    std::condition_variable idle;
    std::deque<OwnedDiagnosticEvent> pending;
    InxLog::DiagnosticCallback callback;
    bool draining = false;
    bool callbackRunning = false;
    std::thread::id callbackThread;
};

struct DiagnosticChannel
{
    // Serializes set/resolve and subscription replay independently of log I/O.
    std::mutex mutex;
    size_t nextId = 1;
    std::unordered_map<size_t, OwnedDiagnosticEvent> active;
    std::vector<std::weak_ptr<DiagnosticSinkState>> sinks;
};

std::vector<std::shared_ptr<DiagnosticSinkState>> QueueLocked(DiagnosticChannel &channel,
                                                              const OwnedDiagnosticEvent &event)
{
    std::vector<std::shared_ptr<DiagnosticSinkState>> sinks;
    for (auto it = channel.sinks.begin(); it != channel.sinks.end();) {
        if (auto sink = it->lock()) {
            {
                std::lock_guard lock(sink->queueMutex);
                sink->pending.push_back(event);
            }
            sinks.push_back(std::move(sink));
            ++it;
        } else {
            it = channel.sinks.erase(it);
        }
    }
    return sinks;
}

void Drain(const std::shared_ptr<DiagnosticSinkState> &sink)
{
    std::unique_lock lock(sink->queueMutex);
    if (sink->draining)
        return;
    sink->draining = true;
    std::exception_ptr failure;
    try {
        while (sink->callback && !sink->pending.empty()) {
            auto callback = sink->callback;
            auto event = std::move(sink->pending.front());
            sink->pending.pop_front();
            sink->callbackRunning = true;
            sink->callbackThread = std::this_thread::get_id();
            lock.unlock();
            // Every callback runs outside both logger and subscription locks.
            try {
                callback(event);
            } catch (...) {
                failure = std::current_exception();
            }
            // A callback capture can own the final source token. Its release
            // may synchronously queue a resolved event, including to this sink.
            callback = {};
            lock.lock();
            sink->callbackRunning = false;
            sink->callbackThread = {};
            sink->idle.notify_all();
            if (failure)
                break;
        }
    } catch (...) {
        failure = std::current_exception();
    }
    // Cleanup precedes propagation, so Reset cannot wait forever after a
    // throwing subscriber. Ordinary delivery does not swallow subscriber errors.
    if (!lock.owns_lock())
        lock.lock();
    sink->callbackRunning = false;
    sink->callbackThread = {};
    sink->draining = false;
    sink->idle.notify_all();
    lock.unlock();
    if (failure)
        std::rethrow_exception(failure);
}

struct OwnedDiagnosticState
{
    std::weak_ptr<DiagnosticChannel> channel;
    size_t id = 0;
    std::string owner;
    std::string key;
    ~OwnedDiagnosticState() noexcept
    {
        try {
            if (auto current = channel.lock()) {
                std::vector<std::shared_ptr<DiagnosticSinkState>> sinks;
                {
                    std::lock_guard lock(current->mutex);
                    const auto found = current->active.find(id);
                    if (found != current->active.end()) {
                        auto event = std::move(found->second);
                        current->active.erase(found);
                        event.resolved = true;
                        sinks = QueueLocked(*current, event);
                    }
                }
                for (const auto &sink : sinks)
                    Drain(sink);
            }
        } catch (...) {
        }
    }
};
} // namespace detail

InxLog::InxLog()
    : logLevel(LOG_INFO), nextSinkId_(0), m_diagnosticChannel(std::make_shared<detail::DiagnosticChannel>())
{
}

OwnedDiagnostic InxLog::CreateOwnedDiagnostic(const std::string &owner, const std::string &key)
{
    if (owner.empty() || key.empty())
        throw std::invalid_argument("Owned diagnostic requires a source owner and key");
    OwnedDiagnostic token;
    auto state = std::make_shared<detail::OwnedDiagnosticState>();
    state->channel = m_diagnosticChannel;
    state->owner = owner;
    state->key = key;
    {
        std::lock_guard lock(m_diagnosticChannel->mutex);
        state->id = m_diagnosticChannel->nextId++;
    }
    token.m_state = std::move(state);
    return token;
}

void OwnedDiagnostic::Set(LogLevel level, const std::string &message, const std::string &sourceFile, int sourceLine)
{
    if (!m_state)
        throw std::logic_error("Owned diagnostic has no source token");
    const auto channel = m_state->channel.lock();
    if (!channel)
        return;
    OwnedDiagnosticEvent event{m_state->owner, m_state->key, level, message, sourceFile, sourceLine, false};
    std::vector<std::shared_ptr<detail::DiagnosticSinkState>> sinks;
    {
        std::lock_guard lock(channel->mutex);
        const auto previous = channel->active.find(m_state->id);
        if (previous != channel->active.end() && previous->second.level == level &&
            previous->second.message == message && previous->second.sourceFile == sourceFile &&
            previous->second.sourceLine == sourceLine)
            return;
        channel->active.insert_or_assign(m_state->id, event);
        sinks = detail::QueueLocked(*channel, event);
    }
    // Keep one changed failure in engine.log without adding a duplicate ordinary
    // Console row. Owned events deliberately bypass ordinary log-level filters.
    InxLog::GetInstance().LogInternal(level, sourceFile.c_str(), sourceLine, message);
    for (const auto &sink : sinks)
        detail::Drain(sink);
}

DiagnosticSubscription InxLog::SubscribeDiagnostics(DiagnosticCallback callback)
{
    DiagnosticSubscription subscription;
    auto sink = std::make_shared<detail::DiagnosticSinkState>();
    sink->callback = std::move(callback);
    {
        std::lock_guard lock(m_diagnosticChannel->mutex);
        auto &sinks = m_diagnosticChannel->sinks;
        sinks.erase(std::remove_if(sinks.begin(), sinks.end(), [](const auto &entry) { return entry.expired(); }),
                    sinks.end());
        m_diagnosticChannel->sinks.emplace_back(sink);
        // Replay and registration are atomic with publication. Any concurrent
        // event is queued behind this initial snapshot, never before it.
        std::lock_guard queueLock(sink->queueMutex);
        for (const auto &[id, event] : m_diagnosticChannel->active) {
            (void)id;
            sink->pending.push_back(event);
        }
    }
    detail::Drain(sink);
    subscription.m_state = std::move(sink);
    return subscription;
}

DiagnosticSubscription::~DiagnosticSubscription()
{
    Reset();
}
DiagnosticSubscription::DiagnosticSubscription(DiagnosticSubscription &&other) noexcept
    : m_state(std::move(other.m_state))
{
}
DiagnosticSubscription &DiagnosticSubscription::operator=(DiagnosticSubscription &&other) noexcept
{
    if (this != &other) {
        Reset();
        m_state = std::move(other.m_state);
    }
    return *this;
}
void DiagnosticSubscription::Reset() noexcept
{
    auto sink = std::move(m_state);
    if (!sink)
        return;
    InxLog::DiagnosticCallback retiredCallback;
    try {
        std::unique_lock lock(sink->queueMutex);
        // Releasing captures can publish source retirement back to this sink.
        // Move ownership under lock, but destroy the callable after unlocking.
        retiredCallback = std::move(sink->callback);
        sink->pending.clear();
        // A copied callback cannot outlive a different thread's subscriber.
        // Self-unsubscription closes delivery without waiting on itself.
        if (sink->callbackRunning && sink->callbackThread != std::this_thread::get_id())
            sink->idle.wait(lock, [&] { return !sink->callbackRunning; });
    } catch (...) {
    }
}

std::string FormatDiagnosticMessage(LogLevel level, const std::string &sourceFile, int sourceLine,
                                    const std::string &message, const std::string &projectRoot)
{
    const char *label = level >= LOG_ERROR ? "Error" : level == LOG_WARN ? "Warning" : "Debug";
    std::string formatted = std::string("[") + label + "]";
    if (!sourceFile.empty()) {
        auto source = ToFsPath(NormalizePortablePath(sourceFile)).lexically_normal();
        if (source.is_absolute() && !projectRoot.empty()) {
            const auto root = ToFsPath(NormalizePortablePath(projectRoot)).lexically_normal();
            auto part = source.begin();
            auto rootPart = root.begin();
            for (; rootPart != root.end() && part != source.end(); ++rootPart, ++part) {
                if (FoldFilesystemPathCase(FromFsPath(*part)) != FoldFilesystemPathCase(FromFsPath(*rootPart)))
                    break;
            }
            if (rootPart == root.end() && part != source.end()) {
                std::filesystem::path relative;
                for (; part != source.end(); ++part)
                    relative /= *part;
                source = std::move(relative);
            }
        }
        formatted += " (" + FromFsPath(source) + ", line " + std::to_string((std::max)(sourceLine, 0)) + ")";
    }
    return formatted + " " + message;
}

InxLog &InxLog::GetInstance()
{
    static InxLog instance;
    return instance;
}

} // namespace infernux
