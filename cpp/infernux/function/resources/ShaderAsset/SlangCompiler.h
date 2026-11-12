#pragma once

#include "ShaderCompilerApi.h"
#include <function/renderer/shader/ShaderProgramCode.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernux
{

class ShaderCompilationError : public std::runtime_error
{
  public:
    explicit ShaderCompilationError(std::vector<ShaderDiagnostic> diagnostics)
        : std::runtime_error(FormatShaderDiagnostics(diagnostics)), diagnostics(std::move(diagnostics))
    {
    }
    const std::vector<ShaderDiagnostic> diagnostics;
};

/// A generated module. Source paths are retained in compiler diagnostics;
/// module names identify static imports, independently of physical files.
struct ShaderSourceModule
{
    std::string name;
    std::string path;
    std::string source;
};

struct ShaderAuthorSource
{
    ShaderSourceModule source;
    std::vector<ShaderSourceModule> libraries;
    /// Logical fields and graph ports. Target addresses belong to compilation.
    std::string schema;
    /// Original AST statement ranges for generated #line locations.
    std::vector<ShaderAuthorRange> sourceRanges;
};

using ShaderTextReader = std::function<std::optional<std::string>(const std::string &)>;
enum class ShaderAuthorPhase
{
    Metadata,
    Program
};
using ShaderAuthorFrontend =
    std::function<ShaderAuthorSource(const ShaderSourceModule &, const ShaderTextReader &, ShaderAuthorPhase)>;

/// The process owns one compiler and its loaded standard library. Each
/// accepted source revision gets a separate module session, so editing a
/// module never reuses Slang's earlier module with the same name.
class SlangCompiler
{
  public:
    static INFERNUX_SHADER_COMPILER_API SlangCompiler &Get();
    static INFERNUX_SHADER_COMPILER_API const char *GetStandardModuleName() noexcept;
    INFERNUX_SHADER_COMPILER_API void SetStandardModuleDirectory(std::string directory);
    INFERNUX_SHADER_COMPILER_API void SetAuthorFrontend(ShaderAuthorFrontend frontend);
    [[nodiscard]] INFERNUX_SHADER_COMPILER_API ShaderAuthorSource
    PrepareAuthorSource(const ShaderSourceModule &source, const ShaderTextReader &reader,
                        ShaderAuthorPhase phase = ShaderAuthorPhase::Program);

    INFERNUX_SHADER_COMPILER_API ShaderCompilation Compile(const ShaderSourceModule &source,
                                                           const std::vector<ShaderSourceModule> &libraries,
                                                           const std::vector<ShaderCodeTarget> &targets,
                                                           bool precise = false,
                                                           const std::vector<ShaderAuthorRange> &sourceRanges = {});

    [[nodiscard]] INFERNUX_SHADER_COMPILER_API const std::string &GetVersion() const noexcept;

    SlangCompiler(const SlangCompiler &) = delete;
    SlangCompiler &operator=(const SlangCompiler &) = delete;

  private:
    SlangCompiler();
    ~SlangCompiler();

    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace infernux
