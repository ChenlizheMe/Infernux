#pragma once

#include "ShaderCompilerApi.h"
#include <function/renderer/rhi/RhiDescriptors.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace infernux
{

enum class ShaderCodeTarget : uint8_t
{
    SpirV,
    Wgsl,
};

/// A generated module. Source paths are retained in compiler diagnostics;
/// module names identify static imports, independently of physical files.
struct ShaderSourceModule
{
    std::string name;
    std::string path;
    std::string source;
};

struct ShaderEntryPointCode
{
    std::string name;
    rhi::ShaderStage stage = rhi::ShaderStage::Compute;
    std::array<uint32_t, 3> workgroupSize{1, 1, 1};
    /// Actual descriptor set/binding pairs retained by this target entry.
    std::vector<std::array<uint32_t, 2>> bindings;
    std::vector<char> code;
};

struct ShaderTargetCode
{
    ShaderCodeTarget target = ShaderCodeTarget::SpirV;
    std::vector<ShaderEntryPointCode> entries;
    /// Target-specific linked layout, including nested fields and strides.
    std::string reflection;
};

struct ShaderCompilation
{
    std::vector<ShaderTargetCode> targets;
    std::string diagnostics;
};

/// The process owns one compiler and its loaded standard library. Each
/// accepted source revision gets a separate module session, so editing a
/// module never reuses Slang's earlier module with the same name.
class SlangCompiler
{
  public:
    static INFERNUX_SHADER_COMPILER_API SlangCompiler &Get();
    static INFERNUX_SHADER_COMPILER_API const char *GetStandardModuleName() noexcept;

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
