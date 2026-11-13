#include <core/log/InxLog.h>

#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace infernux;

static bool CheckOwnedDiagnostics(InxLog &log)
{
    std::vector<OwnedDiagnosticEvent> events;
    // The source fails before the editor Console subscribes.
    auto source = log.CreateOwnedDiagnostic("material.shader-domain", "author:1");
    source.Set(LOG_ERROR, "expected Mesh, got Fullscreen", "Assets/Effects/Test.frag.py");
    auto subscription = log.SubscribeDiagnostics([&](const OwnedDiagnosticEvent &event) { events.push_back(event); });
    if (events.size() != 1 || events.back().resolved || events.back().key != "author:1")
        return false;
    source.Set(LOG_ERROR, "expected Mesh, got Fullscreen", "Assets/Effects/Test.frag.py");
    if (events.size() != 1) // Unchanged Host retries do not add rows/events.
        return false;
    auto second = log.CreateOwnedDiagnostic("material.shader-domain", "author:2");
    second.Set(LOG_ERROR, "second source");
    source.Reset();
    if (events.size() != 3 || !events.back().resolved || events.back().key != "author:1")
        return false;
    subscription.Reset();
    const auto beforeClosed = events.size();
    second.Set(LOG_ERROR, "changed second source");
    if (events.size() != beforeClosed)
        return false;
    // Reopened Console gets only the still-active source, with its latest value.
    std::vector<OwnedDiagnosticEvent> reopened;
    auto next = log.SubscribeDiagnostics([&](const OwnedDiagnosticEvent &event) { reopened.push_back(event); });
    if (reopened.size() != 1 || reopened.front().key != "author:2" ||
        reopened.front().message != "changed second source")
        return false;
    {
        auto moved = std::move(second);
    } // Source retirement resolves once.
    if (reopened.size() != 2 || !reopened.back().resolved)
        return false;
    // Callbacks run outside logger locks and can close their own subscription.
    DiagnosticSubscription selfClosing;
    selfClosing = log.SubscribeDiagnostics([&](const OwnedDiagnosticEvent &) { selfClosing.Reset(); });
    auto finalSource = log.CreateOwnedDiagnostic("material.shader-domain", "author:3");
    finalSource.Set(LOG_ERROR, "self unsubscribe");
    // A subscriber may be the last owner of an active diagnostic token.
    // Reset must not release that capture while holding its queue mutex.
    auto captured =
        std::make_shared<OwnedDiagnostic>(log.CreateOwnedDiagnostic("material.shader-domain", "author:captured"));
    captured->Set(LOG_ERROR, "captured source");
    auto capturesSource = log.SubscribeDiagnostics([keep = captured](const OwnedDiagnosticEvent &) {});
    captured.reset();
    const auto beforeRetirement = reopened.size();
    capturesSource.Reset();
    if (reopened.size() != beforeRetirement + 1 || !reopened.back().resolved ||
        reopened.back().key != "author:captured")
        return false;
    // Self-unsubscription drops the sink's owner during invocation; the copied
    // callback is then the last token owner and must retire before relocking.
    DiagnosticSubscription releasesCopy;
    bool releaseOnTrigger = false;
    auto copiedCapture =
        std::make_shared<OwnedDiagnostic>(log.CreateOwnedDiagnostic("material.shader-domain", "author:copied-capture"));
    copiedCapture->Set(LOG_ERROR, "copy owns source");
    releasesCopy = log.SubscribeDiagnostics(
        [keep = copiedCapture, &releasesCopy, &releaseOnTrigger](const OwnedDiagnosticEvent &event) {
            if (releaseOnTrigger && event.key == "author:trigger")
                releasesCopy.Reset();
        });
    copiedCapture.reset();
    releaseOnTrigger = true;
    auto trigger = log.CreateOwnedDiagnostic("material.shader-domain", "author:trigger");
    trigger.Set(LOG_ERROR, "release callback copy");
    if (std::none_of(reopened.begin(), reopened.end(), [](const OwnedDiagnosticEvent &event) {
            return event.key == "author:copied-capture" && event.resolved;
        }))
        return false;
    finalSource.Reset();
    trigger.Reset();
    // A normal Set exposes subscriber failure, but completes its delivery
    // bookkeeping so unsubscribe cannot hang. The source stays authoritative.
    auto throwingSource = log.CreateOwnedDiagnostic("material.shader-domain", "author:throwing");
    auto throwingSubscriber = log.SubscribeDiagnostics(
        [](const OwnedDiagnosticEvent &) { throw std::runtime_error("owned diagnostic subscriber probe"); });
    bool sawSubscriberFailure = false;
    try {
        throwingSource.Set(LOG_ERROR, "retained despite subscriber failure");
    } catch (const std::runtime_error &error) {
        sawSubscriberFailure = std::string(error.what()) == "owned diagnostic subscriber probe";
    }
    if (!sawSubscriberFailure)
        return false;
    throwingSubscriber.Reset();
    std::vector<OwnedDiagnosticEvent> afterFailure;
    auto replayAfterFailure =
        log.SubscribeDiagnostics([&](const OwnedDiagnosticEvent &event) { afterFailure.push_back(event); });
    if (afterFailure.size() != 1 || afterFailure.front().resolved || afterFailure.front().key != "author:throwing" ||
        afterFailure.front().message != "retained despite subscriber failure")
        return false;
    return true;
}

int main()
{
    InxLog &log = InxLog::GetInstance();
    log.SetLogLevel(LOG_DEBUG);

    std::vector<LogLevel> receivedLevels;
    std::vector<bool> receivedInternal;
    std::vector<std::string> receivedMessages;
    const size_t sinkId =
        log.AddSink([&](LogLevel level, const char *, int, const std::string &message, bool internal) {
            receivedLevels.push_back(level);
            receivedInternal.push_back(internal);
            receivedMessages.push_back(message);
        });

    int evaluatedNonErrorArgument = 0;
    INXLOG_DEBUG(++evaluatedNonErrorArgument);
    INXLOG_INFO(++evaluatedNonErrorArgument);
    INXLOG_WARN(++evaluatedNonErrorArgument);
    log.Log(LOG_DEBUG, __FILE__, __LINE__, "direct debug call");
    log.Log(LOG_INFO, __FILE__, __LINE__, "direct info call");
    log.Log(LOG_WARN, __FILE__, __LINE__, "direct warning call");
    INXLOG_ERROR("release policy error probe");
    INXLOG_DIAGNOSTIC("startup geometry probe");
    log.SetLogLevel(LOG_WARN);
    INXLOG_DIAGNOSTIC(++evaluatedNonErrorArgument);

    log.RemoveSink(sinkId);

    if (evaluatedNonErrorArgument != 0)
        return EXIT_FAILURE;
    if (receivedLevels.size() != 2 || receivedLevels.front() != LOG_ERROR || receivedLevels.back() != LOG_INFO ||
        !receivedInternal.back())
        return EXIT_FAILURE;
    if (receivedMessages.front() != "release policy error probe" || receivedMessages.back() != "startup geometry probe")
        return EXIT_FAILURE;
    const auto root = std::filesystem::current_path();
    const auto source = root / "Assets" / "Scripts" / "Player.py";
    if (FormatDiagnosticMessage(LOG_ERROR, source.string(), 35, "failed", root.string()) !=
        "[Error] (Assets/Scripts/Player.py, line 35) failed")
        return EXIT_FAILURE;
    if (FormatDiagnosticMessage(LOG_WARN, "", 0, "warning") != "[Warning] warning" ||
        FormatDiagnosticMessage(LOG_INFO, "", 0, "ready") != "[Debug] ready")
        return EXIT_FAILURE;
    return CheckOwnedDiagnostics(log) ? EXIT_SUCCESS : EXIT_FAILURE;
}
