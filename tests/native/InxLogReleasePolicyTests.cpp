#include <core/log/InxLog.h>

#include <cstdlib>
#include <vector>

using namespace infernux;

int main()
{
    InxLog &log = InxLog::GetInstance();
    log.SetLogLevel(LOG_DEBUG);

    std::vector<LogLevel> receivedLevels;
    std::vector<bool> receivedInternal;
    std::vector<std::string> receivedMessages;
    const size_t sinkId = log.AddSink([&](LogLevel level, const char *, int, const std::string &message, bool internal) {
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
    return EXIT_SUCCESS;
}
