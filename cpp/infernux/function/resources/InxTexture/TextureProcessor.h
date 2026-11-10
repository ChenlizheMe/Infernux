#pragma once

#include "InxTexture.h"

#include <memory>

namespace infernux
{

enum class TextureCompression : uint32_t
{
    None = 0,
    Automatic,
    BC1,
    BC3,
    BC4,
    BC5,
};

enum class TextureCompressionQuality : uint32_t
{
    Fast = 0,
    Normal,
    High,
};

enum class TextureTargetFormat : uint32_t
{
    Automatic = 0,
    Rgba8,
    Rgba4UNorm,
    Rgba16UNorm,
    Rgba16Float,
    Rgba32Float,
};

struct TextureProcessOptions
{
    bool generateMipmaps = true;
    TextureCompression compression = TextureCompression::Automatic;
    TextureCompressionQuality quality = TextureCompressionQuality::Normal;
    TextureTargetFormat targetFormat = TextureTargetFormat::Automatic;
};

/// Deterministic offline processing used by texture importers. The runtime
/// consumes the resulting concrete format and never regenerates mips or
/// recompresses a texture while loading it.
class TextureProcessor final
{
  public:
    // Runtime workspace planning shares this choice with the offline processor.
    [[nodiscard]] static constexpr TextureCompression AutomaticRgba8Compression(TextureSemantic semantic, bool hasAlpha)
    {
        if (semantic == TextureSemantic::Color)
            return hasAlpha ? TextureCompression::BC3 : TextureCompression::BC1;
        if (semantic == TextureSemantic::Normal)
            return TextureCompression::BC5;
        return TextureCompression::None;
    }
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> Process(TextureCpuData source,
                                                                       const TextureProcessOptions &options);
};

} // namespace infernux
