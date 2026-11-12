#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace infernux
{

/// Stable scene/GBuffer identity, independent of compiler and asset import order.
[[nodiscard]] constexpr uint32_t ShadingModelId(std::string_view name) noexcept
{
    uint32_t hash = 2166136261u;
    for (const unsigned char character : name) {
        hash ^= character;
        hash *= 16777619u;
    }
    return hash == 0u ? 1u : hash;
}

/// Authored render-state defaults shared by assets and device publication.
struct ShaderRenderMeta
{
    std::string cullMode;
    std::string depthWrite;
    std::string depthTest;
    std::string blend;
    int queue = -1;
    std::string passTag;
    std::string stencil;
    std::string alphaClip;
};

enum class ShaderProgramDomain : unsigned char
{
    Mesh = 0,
    ParticleSprite,
    ScreenUI,
    WorldUI,
    Fullscreen,

    Count,
};

[[nodiscard]] constexpr const char *ShaderProgramDomainName(ShaderProgramDomain domain) noexcept
{
    switch (domain) {
    case ShaderProgramDomain::Mesh:
        return "Mesh";
    case ShaderProgramDomain::ParticleSprite:
        return "ParticleSprite";
    case ShaderProgramDomain::ScreenUI:
        return "ScreenUI";
    case ShaderProgramDomain::WorldUI:
        return "WorldUI";
    case ShaderProgramDomain::Fullscreen:
        return "Fullscreen";
    case ShaderProgramDomain::Count:
        return "Count";
    }
    return "Unknown";
}

/// A rejected author selection is local to a material publication. It never
/// unwinds the editor frame or changes an accepted device program.
class ShaderProgramSelectionError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

class ShaderProgramDomainMismatch : public ShaderProgramSelectionError
{
  public:
    ShaderProgramDomainMismatch(ShaderProgramDomain expected, ShaderProgramDomain actual)
        : ShaderProgramSelectionError("Material shader domain mismatch: expected " +
                                      std::string(ShaderProgramDomainName(expected)) + ", got " +
                                      ShaderProgramDomainName(actual)),
          expected(expected), actual(actual)
    {
    }

    const ShaderProgramDomain expected;
    const ShaderProgramDomain actual;
};

// ============================================================================
// ShaderCompileTarget — identifies which rendering pass variant to compile for.
//
// Shared between InxShaderLoader (compile-time variant generation) and
// InxMaterial (per-pass pipeline storage).  Defined in a lightweight header
// to avoid pulling in heavy shader-compiler includes through InxMaterial.h.
// ============================================================================

enum class ShaderCompileTarget : int
{
    Forward = 0,     // Standard forward rendering (default)
    GBuffer = 1,     // Deferred GBuffer output
    Shadow = 2,      // Depth-only shadow caster
    Depth = 3,       // Camera depth/depth-prepass output
    Picking = 4,     // Stable object identity output
    Motion = 5,      // Motion-vector output
    ForwardPlus = 6, // Tiled/clustered forward lighting
    Normal = 7,      // World-space normal output for screen-space effects
    BaseColor = 8,   // Linear surface albedo + alpha output for the Geometry Stage

    Count // Sentinel — number of targets; must be last
};

[[nodiscard]] constexpr const char *ShaderCompileTargetName(ShaderCompileTarget target) noexcept
{
    switch (target) {
    case ShaderCompileTarget::Forward:
        return "Forward";
    case ShaderCompileTarget::GBuffer:
        return "GBuffer";
    case ShaderCompileTarget::Shadow:
        return "Shadow";
    case ShaderCompileTarget::Depth:
        return "Depth";
    case ShaderCompileTarget::Picking:
        return "Picking";
    case ShaderCompileTarget::Motion:
        return "Motion";
    case ShaderCompileTarget::ForwardPlus:
        return "ForwardPlus";
    case ShaderCompileTarget::Normal:
        return "Normal";
    case ShaderCompileTarget::BaseColor:
        return "BaseColor";
    case ShaderCompileTarget::Count:
        return "Count";
    }
    return "Unknown";
}

/// Geometry variants that consume InstanceAuxBuffer for object identity,
/// transform history or the engine-owned layer mask must upload that stream
/// before drawing. Keep this contract beside the compile target enum so the
/// shader linker and draw path cannot silently diverge.
[[nodiscard]] constexpr bool ShaderCompileTargetUsesInstanceAuxiliary(ShaderCompileTarget target) noexcept
{
    switch (target) {
    case ShaderCompileTarget::Forward:
    case ShaderCompileTarget::ForwardPlus:
    case ShaderCompileTarget::GBuffer:
    case ShaderCompileTarget::Picking:
    case ShaderCompileTarget::Motion:
    case ShaderCompileTarget::Normal:
    case ShaderCompileTarget::BaseColor:
    case ShaderCompileTarget::Depth:
    case ShaderCompileTarget::Shadow:
        return true;
    case ShaderCompileTarget::Count:
        return false;
    }
    return false;
}

} // namespace infernux
