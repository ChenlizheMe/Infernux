#pragma once

#include "RayTexturePages.h"

#include <function/renderer/shader/ShaderProgramLayout.h>

#include <cstddef>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

namespace infernux
{
struct DrawCallResult;
class InxMaterial;
struct RendererParameterBlock;
} // namespace infernux

namespace infernux::raytracing
{

struct RayLitMaterialInput
{
    std::vector<std::byte> materials;
    std::vector<RayTexturePageSource> sources;
};

/// Resolve the same effective texture used for the raster draw, with its
/// exact resident source lease. An unresolved texture leaves the whole input pending.
using ResolveRayLitTexture = std::function<std::optional<RayTexturePageSource>(
    const InxMaterial &, const RendererParameterBlock *, std::string_view)>;

/// The reflected per-device Slang layout is the only authority for offsets.
/// Pair Capture and PackDescriptors with the same ordered texture page plan.
class RayLitMaterialLayout final
{
  public:
    explicit RayLitMaterialLayout(const std::vector<ShaderResourceLayout> &resources);
    ~RayLitMaterialLayout();
    [[nodiscard]] std::optional<RayLitMaterialInput> Capture(const DrawCallResult &world,
                                                             const ResolveRayLitTexture &resolve) const;
    [[nodiscard]] std::vector<std::byte> PackDescriptors(const std::vector<RayTexturePageSlot> &slots) const;

  private:
    struct State;
    std::unique_ptr<State> m_state;
};

} // namespace infernux::raytracing
