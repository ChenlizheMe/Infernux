#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <stdexcept>
#include <vector>

namespace infernux
{
struct PhysicsGpuJointMotion
{
    std::array<float, 4> coordinate;
    std::array<float, 4> orientation; // Connected frame quaternion in wxyz order.
};
static_assert(sizeof(PhysicsGpuJointMotion) == 32);

struct PhysicsGpuJointMotionRead
{
    std::shared_ptr<rhi::PreparedComputeBatch> packet;
    std::shared_ptr<rhi::ComputeBuffer> motion;
    uint32_t count = 0;

    [[nodiscard]] std::vector<PhysicsGpuJointMotion> ReadHostMotion() const
    {
        std::vector<PhysicsGpuJointMotion> result(count);
        if (!result.empty())
            packet->Readback(0)->ReadInto(result.data(), uint64_t{count} * sizeof(PhysicsGpuJointMotion));
        const auto finite = [](const auto &values) {
            return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
        };
        for (const auto &row : result)
            if (!finite(row.coordinate) || !finite(row.orientation))
                throw std::runtime_error("GPU joint motion contains an invalid numerical row");
        return result;
    }
};
} // namespace infernux
