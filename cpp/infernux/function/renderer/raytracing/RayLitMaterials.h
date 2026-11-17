#pragma once

#include "RayTexturePages.h"

#include <function/renderer/shader/ShaderProgramLayout.h>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace infernux
{
struct DrawCallResult;
class InxMaterial;
struct RendererParameterBlock;
} // namespace infernux

namespace infernux::raytracing
{

/// Shader descriptor classes 0 and 1 address the sampled texture pages.
enum class RayLitSolidTexture : uint8_t
{
    White = 2,
    Black = 3,
    FlatNormal = 4,
};

using RayLitTexture = std::variant<RayTexturePageSource, RayLitSolidTexture>;
using RayLitTextureDescriptor = std::variant<uint32_t, RayLitSolidTexture>;
using RayLitTextureReference = std::variant<RayLitSolidTexture, std::string>;

[[nodiscard]] RayLitTextureReference ResolveRayLitTextureReference(const InxMaterial &material,
                                                                   const RendererParameterBlock *parameters,
                                                                   std::string_view property);

struct RayLitMaterialInput
{
    std::vector<std::byte> materials;
    std::vector<RayTexturePageSource> sources;
    /// Material rows index this table. Page-backed descriptors index sources.
    std::vector<RayLitTextureDescriptor> descriptors;
};

/// Resolve the raster draw's exact resident texture or its constant default.
/// An unresolved texture leaves the whole input pending.
using ResolveRayLitTexture =
    std::function<std::optional<RayLitTexture>(const InxMaterial &, const RendererParameterBlock *, std::string_view)>;

/// The reflected per-device Slang layout is the only authority for offsets.
/// Pair Capture and PackDescriptors with the same ordered texture page plan.
class RayLitMaterialLayout final
{
  public:
    explicit RayLitMaterialLayout(const std::vector<ShaderResourceLayout> &resources);
    ~RayLitMaterialLayout();
    [[nodiscard]] std::optional<RayLitMaterialInput> Capture(const DrawCallResult &world,
                                                             const ResolveRayLitTexture &resolve) const;
    [[nodiscard]] std::vector<std::byte> PackDescriptors(const RayLitMaterialInput &input,
                                                         const std::vector<RayTexturePageSlot> &slots) const;

  private:
    struct State;
    std::unique_ptr<State> m_state;
};

} // namespace infernux::raytracing
