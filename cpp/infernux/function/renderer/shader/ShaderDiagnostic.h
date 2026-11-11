#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace infernux
{

struct ShaderAuthorRange
{
    std::string path;
    uint32_t line = 0;
    uint32_t column = 0;
    uint32_t endLine = 0;
    uint32_t endColumn = 0;
};

/// Source ranges use one-based lines/columns, with an exclusive end.
/// A locationless compiler/driver diagnostic has a zero range.
struct ShaderDiagnostic
{
    std::string code;
    std::string severity;
    std::string path;
    uint32_t line = 0;
    uint32_t column = 0;
    uint32_t endLine = 0;
    uint32_t endColumn = 0;
    std::string message;
    std::string phase;
    std::string target;

    [[nodiscard]] size_t GetRuntimeMemoryBytes() const noexcept
    {
        return sizeof(*this) + code.capacity() + severity.capacity() + path.capacity() + message.capacity() +
               phase.capacity() + target.capacity();
    }
};

inline std::string FormatShaderDiagnostics(const std::vector<ShaderDiagnostic> &diagnostics)
{
    std::string text;
    for (const auto &diagnostic : diagnostics) {
        if ((diagnostic.severity == "span" || diagnostic.severity == "note-span") && diagnostic.message.empty())
            continue;
        if (!diagnostic.path.empty()) {
            text += diagnostic.path;
            if (diagnostic.line)
                text += ":" + std::to_string(diagnostic.line) + ":" + std::to_string(diagnostic.column);
            text += ": ";
        }
        text += diagnostic.severity + "[" + diagnostic.code + "]: " + diagnostic.message + "\n";
    }
    return text;
}

} // namespace infernux
