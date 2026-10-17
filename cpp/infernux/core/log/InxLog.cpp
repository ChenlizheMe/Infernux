#include "InxLog.h"
#include <platform/filesystem/InxPath.h>

namespace infernux
{

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
