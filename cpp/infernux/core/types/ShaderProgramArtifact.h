#pragma once

#include "core/FoundationApi.h"

#include <core/types/ShaderTypes.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernux
{

struct ShaderTargetCode;
struct ShaderEntryPointCode;
struct ShaderCompilation;
enum class ShaderCodeTarget : uint8_t;

struct ShaderStagePair
{
    // GUIDs for imported assets, registered keys for symbolic built-ins.
    std::string vertexKey;
    std::string fragmentKey;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return !fragmentKey.empty();
    }

    [[nodiscard]] std::string ToString() const
    {
        return vertexKey + "|" + fragmentKey;
    }

    [[nodiscard]] bool UsesShader(std::string_view shaderId) const noexcept
    {
        return vertexKey == shaderId || fragmentKey == shaderId;
    }

    friend bool operator==(const ShaderStagePair &lhs, const ShaderStagePair &rhs) noexcept
    {
        return lhs.vertexKey == rhs.vertexKey && lhs.fragmentKey == rhs.fragmentKey;
    }

    friend bool operator!=(const ShaderStagePair &lhs, const ShaderStagePair &rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

struct ShaderStagePairHash
{
    [[nodiscard]] INFERNUX_FOUNDATION_API size_t operator()(const ShaderStagePair &pair) const noexcept;
};

struct ShaderProgramKey
{
    ShaderStagePair stages;
    uint64_t revision = 0;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return stages.IsValid();
    }

    [[nodiscard]] INFERNUX_FOUNDATION_API std::string ToString() const;

    friend bool operator==(const ShaderProgramKey &lhs, const ShaderProgramKey &rhs) noexcept
    {
        return lhs.stages == rhs.stages && lhs.revision == rhs.revision;
    }

    friend bool operator!=(const ShaderProgramKey &lhs, const ShaderProgramKey &rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

struct ShaderProgramKeyHash
{
    [[nodiscard]] INFERNUX_FOUNDATION_API size_t operator()(const ShaderProgramKey &key) const noexcept;
};

struct ShaderProgramVariantKey
{
    ShaderProgramKey program;
    ShaderCompileTarget target = ShaderCompileTarget::Forward;
    /// Device shader ABI contract. This keeps a program compiled for the
    /// bounded descriptor ABI separate from one compiled with the bindless
    /// set layout and other enabled device contracts.
    uint64_t deviceContract = 0;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return program.IsValid() && target >= ShaderCompileTarget::Forward && target < ShaderCompileTarget::Count;
    }

    [[nodiscard]] INFERNUX_FOUNDATION_API std::string ToString() const;

    friend bool operator==(const ShaderProgramVariantKey &lhs, const ShaderProgramVariantKey &rhs) noexcept
    {
        return lhs.program == rhs.program && lhs.target == rhs.target && lhs.deviceContract == rhs.deviceContract;
    }

    friend bool operator!=(const ShaderProgramVariantKey &lhs, const ShaderProgramVariantKey &rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

struct ShaderProgramVariantKeyHash
{
    [[nodiscard]] INFERNUX_FOUNDATION_API size_t operator()(const ShaderProgramVariantKey &key) const noexcept;
};

enum class ShaderProgramStageMask : uint8_t
{
    None = 0,
    Vertex = 1 << 0,
    Fragment = 1 << 1,
};

constexpr ShaderProgramStageMask operator|(ShaderProgramStageMask lhs, ShaderProgramStageMask rhs) noexcept
{
    return static_cast<ShaderProgramStageMask>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

constexpr bool HasStage(ShaderProgramStageMask value, ShaderProgramStageMask stage) noexcept
{
    return (static_cast<uint8_t>(value) & static_cast<uint8_t>(stage)) != 0;
}

struct ShaderProgramPropertyBinding
{
    std::string name;
    std::string type;
    std::string defaultValue;
    std::string textureDefault;
    ShaderProgramStageMask stages = ShaderProgramStageMask::None;
    bool hdr = false;
    std::optional<std::array<double, 2>> range;
    std::optional<uint32_t> bufferOffset;
    std::optional<uint32_t> textureSlot;
    uint32_t byteSize = 0;
    uint32_t byteAlignment = 0;
    uint32_t arrayCount = 1;

    [[nodiscard]] bool IsTexture() const noexcept
    {
        return textureSlot.has_value();
    }

    [[nodiscard]] bool IsValid(uint32_t materialBufferSize) const noexcept;
};

struct ShaderProgramArtifact
{
    ShaderProgramKey key;
    ShaderProgramDomain domain = ShaderProgramDomain::Mesh;
    std::string shadingModel;
    uint32_t deferredModel = 0;
    bool receivesShadows = true;
    /// Authored, statically evaluable secondary PBR entry; never inferred from parameter names.
    bool hasRaySurface = false;
    bool raySurfaceMayEmit = false;
    uint32_t materialBufferSize = 0;
    std::optional<uint32_t> alphaClipThresholdOffset;
    std::vector<ShaderProgramPropertyBinding> properties;
    uint64_t varyingInterfaceSignature = 0;
    uint64_t materialLayoutSignature = 0;
    uint64_t compatibilitySignature = 0;
    // Scene-depth requirement of a linked particle material.
    bool usesParticleSceneDepthBinding = false;
    // True when surface variants consume the device-global sampled-texture
    // table. The per-domain descriptor set then carries uint texture indices
    // instead of one combined sampler descriptor per property.
    bool usesBindlessTextureABI = false;
    struct PassCode
    {
        // The alias keeps both entry pointers alive when a pass is copied.
        std::shared_ptr<const ShaderTargetCode> module;
        const ShaderEntryPointCode *vertex = nullptr;
        const ShaderEntryPointCode *fragment = nullptr;
    };
    struct PassVariant
    {
        ShaderCompileTarget target = ShaderCompileTarget::Forward;
        uint64_t compatibilitySignature = 0;
        std::vector<PassCode> outputs;
    };
    std::vector<PassVariant> variants;
    /// All target layouts/code of an authored module, retained once. Outputs
    /// that provide native geometry compose its fragment entries directly.
    std::shared_ptr<const ShaderCompilation> compilation;

    [[nodiscard]] INFERNUX_FOUNDATION_API const PassVariant *FindVariant(ShaderCompileTarget target) const noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API const PassCode *FindPassCode(ShaderCompileTarget pass,
                                                                       ShaderCodeTarget target) const noexcept;

    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsValid() const noexcept;
};

[[nodiscard]] INFERNUX_FOUNDATION_API uint64_t ComputeShaderProgramRevision(std::string_view generatedVertexSource,
                                                                            std::string_view generatedFragmentSource,
                                                                            ShaderCompileTarget target,
                                                                            uint64_t compatibilitySignature) noexcept;

/// Computes the runtime identity of the complete, ordered pass-variant set.
/// key.revision is deliberately ignored so this can initialize that field.
[[nodiscard]] INFERNUX_FOUNDATION_API uint64_t
ComputeShaderProgramArtifactRevision(const ShaderProgramArtifact &artifact) noexcept;

} // namespace infernux
