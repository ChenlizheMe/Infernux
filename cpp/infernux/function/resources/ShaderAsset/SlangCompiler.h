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
/// Project source identities captured by the exact author/cooker owner.
using ShaderAuthorSurfaceSources = std::vector<std::pair<std::string, std::string>>;
using ShaderAuthorSurfaceReader = std::function<const ShaderAuthorSurfaceSources &()>;
using ShaderAuthorFrontend =
    std::function<ShaderAuthorSource(const ShaderSourceModule &, const ShaderTextReader &, ShaderAuthorPhase,
                                     const ShaderSourceModule &, const ShaderAuthorSurfaceReader &)>;

/// The process owns one compiler and its loaded standard library. Each
/// accepted source revision gets a separate module session, so editing a
/// module never reuses Slang's earlier module with the same name.
class SlangCompiler
{
  public:
    static INFERNUX_SHADER_COMPILER_API SlangCompiler &Get();
    INFERNUX_SHADER_COMPILER_API void SetAuthorFrontend(ShaderAuthorFrontend frontend);
    [[nodiscard]] INFERNUX_SHADER_COMPILER_API ShaderAuthorSource
    PrepareAuthorSource(const ShaderSourceModule &source, const ShaderTextReader &reader,
                        ShaderAuthorPhase phase = ShaderAuthorPhase::Program, const ShaderSourceModule &vertex = {},
                        const ShaderAuthorSurfaceReader &raySurfaces = {});
    /// Compile a frozen author snapshot for the targets its consumer will use.
    [[nodiscard]] INFERNUX_SHADER_COMPILER_API ShaderCompilation
    CompileAuthorProgram(const ShaderAuthorSource &source, const std::vector<ShaderCodeTarget> &targets);

    /// One logical root block: native push constants and the RHI uniform
    /// address on WGSL. Algorithms and their logical fields stay identical.
    [[nodiscard]] INFERNUX_SHADER_COMPILER_API ShaderCompilation
    CompileRootProgram(ShaderSourceModule source, const std::vector<ShaderSourceModule> &libraries,
                       const std::vector<ShaderCodeTarget> &targets, bool precise = false,
                       const std::vector<ShaderAuthorRange> &sourceRanges = {});

    /// Libraries contain the frozen import/include closure with original source
    /// paths. Relative includes use those paths; built-in standard modules
    /// are embedded in the resident compiler.
    INFERNUX_SHADER_COMPILER_API ShaderCompilation Compile(const ShaderSourceModule &source,
                                                           const std::vector<ShaderSourceModule> &libraries,
                                                           const std::vector<ShaderCodeTarget> &targets,
                                                           bool precise = false,
                                                           const std::vector<ShaderAuthorRange> &sourceRanges = {});

    /// Select one entry in a project compute module. The most recent exact
    /// source/target snapshot retains its parsed modules for sibling entries.
    INFERNUX_SHADER_COMPILER_API ShaderCompilation
    CompileComputeEntry(const ShaderSourceModule &source, const std::vector<ShaderSourceModule> &libraries,
                        const std::vector<ShaderCodeTarget> &targets, const std::string &entryPoint, bool precise,
                        const std::vector<ShaderAuthorRange> &sourceRanges, bool &frontendReused);

    [[nodiscard]] INFERNUX_SHADER_COMPILER_API const std::string &GetVersion() const noexcept;
    [[nodiscard]] INFERNUX_SHADER_COMPILER_API static std::string_view GetAuthorContract() noexcept;

    SlangCompiler(const SlangCompiler &) = delete;
    SlangCompiler &operator=(const SlangCompiler &) = delete;

  private:
    SlangCompiler();
    ~SlangCompiler();

    ShaderCompilation CompileProgram(const ShaderSourceModule &source, const std::vector<ShaderSourceModule> &libraries,
                                     const std::vector<ShaderCodeTarget> &targets, bool precise,
                                     const std::vector<ShaderAuthorRange> &sourceRanges, const std::string *entryPoint,
                                     bool *frontendReused);

    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace infernux
