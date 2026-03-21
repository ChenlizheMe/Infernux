#pragma once

#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>

namespace infengine
{
enum LogLevel
{
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3,
    LOG_FATAL = 4
};

class InfLog
{
  public:
    static InfLog &GetInstance()
    {
        static InfLog instance;
        return instance;
    }

    template <typename... Args> void Log(LogLevel level, const char *file, int line, Args &&...args)
    {
        if (logLevel.load(std::memory_order_relaxed) > level)
            return;

        std::ostringstream oss;
        (oss << ... << args);

        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << LogLevelToColor(level) << "[" << LogLevelToString(level) << "] "
                  << "(" << file << ":" << line << ") " << oss.str() << "\033[0m" << std::endl;
    }

    void SetLogLevel(int level)
    {
        logLevel.store(level, std::memory_order_relaxed);
    }

    int GetLogLevel() const
    {
        return logLevel.load(std::memory_order_relaxed);
    }

  private:
    InfLog() : logLevel(LOG_INFO)
    {
    }
    InfLog(const InfLog &) = delete;
    InfLog &operator=(const InfLog &) = delete;

    std::mutex mutex_;

    std::atomic<int> logLevel;

    const char *LogLevelToString(LogLevel level)
    {
        switch (level) {
        case LOG_DEBUG:
            return "DEBUG";
        case LOG_INFO:
            return "INFO";
        case LOG_WARN:
            return "WARN";
        case LOG_ERROR:
            return "ERROR";
        case LOG_FATAL:
            return "FATAL";
        default:
            return "UNKNOWN";
        }
    }

    const char *LogLevelToColor(LogLevel level)
    {
        switch (level) {
        case LOG_DEBUG:
            return "\033[36m"; // Cyan
        case LOG_INFO:
            return "\033[37m"; // White
        case LOG_WARN:
            return "\033[33m"; // Yellow
        case LOG_ERROR:
            return "\033[31m"; // Red
        case LOG_FATAL:
            return "\033[35m"; // Magenta
        default:
            return "\033[0m"; // Default color
        }
    }
};
} // namespace infengine

#define INFLOG_INTERNAL(level, ...)                                                                                    \
    do {                                                                                                               \
        if ((level) >= InfLog::GetInstance().GetLogLevel())                                                            \
            InfLog::GetInstance().Log(static_cast<LogLevel>(level), __FILE__, __LINE__, __VA_ARGS__);                  \
    } while (false)

#define INFLOG_DEBUG(...) INFLOG_INTERNAL(LOG_DEBUG, __VA_ARGS__)
#define INFLOG_INFO(...) INFLOG_INTERNAL(LOG_INFO, __VA_ARGS__)
#define INFLOG_WARN(...) INFLOG_INTERNAL(LOG_WARN, __VA_ARGS__)
#define INFLOG_ERROR(...) INFLOG_INTERNAL(LOG_ERROR, __VA_ARGS__)
#define INFLOG_FATAL(...)                                                                                              \
    do {                                                                                                               \
        INFLOG_INTERNAL(LOG_FATAL, __VA_ARGS__);                                                                       \
        std::abort();                                                                                                  \
    } while (false)

#define INFLOG_SET_LEVEL(level) InfLog::GetInstance().SetLogLevel(level)

#define INFLOG_GET_LEVEL() InfLog::GetInstance().GetLogLevel()