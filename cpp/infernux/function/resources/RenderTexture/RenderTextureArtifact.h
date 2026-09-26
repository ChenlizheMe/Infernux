#pragma once

#include <function/renderer/rhi/RhiRenderTexture.h>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace infernux
{
/// CPU-only asset description; never contains pixels, GPU handles or a source path.
/// The existing RHI RenderTexture is the sole allocation type.
class RenderTextureArtifact final
{
  public:
    static rhi::RenderTextureDesc ParseDocument(const nlohmann::json &document);
    static nlohmann::json SerializeDocument(const rhi::RenderTextureDesc &description);
    static std::vector<std::string> FormatNames(bool depth);
    static std::string Encode(const rhi::RenderTextureDesc &description, std::string_view sourceHash);
    static rhi::RenderTextureDesc Decode(std::string_view bytes);
    static bool HasCurrentHeader(std::string_view bytes) noexcept;
};
} // namespace infernux
