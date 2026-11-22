#pragma once

#include <cstdint>

namespace infernux
{
enum class PhysicsParticipantStage : uint32_t
{
    Predict,
    Prepare,
    Exchange,
    Accept,
    Initialize
};

constexpr uint32_t PhysicsParticipantStageBit(PhysicsParticipantStage stage) noexcept
{
    return 1u << static_cast<uint32_t>(stage);
}
} // namespace infernux
