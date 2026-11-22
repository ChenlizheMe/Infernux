// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
};

/// Rows in a completed GPU publication. Authoring binds solver-neutral local
/// body rows; the exporter addresses the corresponding world in this buffer.
/// Records hold the body origin, WXYZ rotation, center of mass with its sleep
/// flag, linear velocity, and angular velocity.
struct PhysicsGpuPoseSource
{
    static constexpr uint32_t Position = 0, Rotation = 1, CenterOfMass = 2, LinearVelocity = 3, AngularVelocity = 4;
    static constexpr uint32_t RowWords = 5;
    static constexpr uint32_t RowBytes = RowWords * 4 * sizeof(float);

    std::shared_ptr<rhi::ComputeBuffer> states;
    uint64_t firstRow = 0;
    uint32_t bodyCount = 0;
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
