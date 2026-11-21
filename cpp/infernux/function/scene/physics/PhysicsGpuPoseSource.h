// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <memory>

namespace infernux::rhi
{
class ComputeBuffer;
}

namespace infernux
{

struct PhysicsGpuBodyPoseBinding
{
    uint32_t id;
    uint32_t row;
    std::array<float, 3> centerOffset{};
};

/// Rows in a completed GPU publication. Authoring binds solver-neutral local
/// body rows; the exporter addresses the corresponding world in this buffer.
struct PhysicsGpuPoseSource
{
    enum class Layout : uint8_t
    {
        CenterOfMassXYZW,
        OriginWXYZ,
    };

    std::shared_ptr<rhi::ComputeBuffer> states;
    uint64_t firstRow = 0;
    uint32_t bodyCount = 0;
    Layout layout = Layout::CenterOfMassXYZW;
};

} // namespace infernux
