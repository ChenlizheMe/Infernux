#pragma once

#include "ShaderCompilerApi.h"
#include <function/renderer/shader/ShaderProgramCode.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace infernux
{

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
};

using ShaderTextReader = std::function<std::optional<std::string>(const std::string &)>;
using ShaderAuthorFrontend = std::function<ShaderAuthorSource(const ShaderSourceModule &, const ShaderTextReader &)>;

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
    [[nodiscard]] INFERNUX_SHADER_COMPILER_API ShaderAuthorSource PrepareAuthorSource(const ShaderSourceModule &source,
                                                                                      const ShaderTextReader &reader);

    INFERNUX_SHADER_COMPILER_API ShaderCompilation Compile(const ShaderSourceModule &source,
                                                           const std::vector<ShaderSourceModule> &libraries,
                                                           const std::vector<ShaderCodeTarget> &targets,
                                                           bool precise = false);

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
