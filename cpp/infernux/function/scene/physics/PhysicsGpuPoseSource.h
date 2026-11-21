// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <vector>

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
    uint32_t rowBytes = 0;
    Layout layout = Layout::CenterOfMassXYZW;
};

struct PhysicsGpuBoxAuthor
{
    glm::vec3 centerOfMassOffset;
    glm::quat localRotation;
    glm::vec3 halfExtents;
    float friction, bounciness;
    uint32_t memberIndex;
    bool trigger;
};

struct PhysicsGpuBoxBinding
{
    uint32_t bodyRow, bodyIndex;
    PhysicsGpuBoxAuthor author;
};

struct PhysicsGpuBoxExport
{
    PhysicsGpuPoseSource poses;
    std::vector<PhysicsGpuBoxBinding> boxes;
};

} // namespace infernux
