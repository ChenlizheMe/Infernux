#pragma once

#include <cstdint>

namespace infernux
{
class InxMaterial;
}

namespace infernux::raytracing
{

/// Families implemented by the resident secondary-surface module.
enum class RayMaterialParticipation : uint32_t
{
    Unknown = 0,
    Lit = 1,
    Excluded = 2,
    Unlit = 3,
};

[[nodiscard]] RayMaterialParticipation ClassifyRayMaterial(const InxMaterial &material) noexcept;

} // namespace infernux::raytracing
