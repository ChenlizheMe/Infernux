// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpContactPublication.h"

namespace infernux::physics::warp
{

PhysicsContactPublication::Frame
DecodeWarpContactFrame(std::vector<WarpContactReport> reports,
                       const std::vector<PhysicsContactPublication::Endpoint> &geometry,
                       const PhysicsContactPublication::Policy &policy)
{
    // Narrowphase reserves contact slots on device. Use authored geometry and
    // per-pair feature order for callback selection, independent of slot order.
    std::sort(reports.begin(), reports.end(), [](const auto &a, const auto &b) {
        return std::tie(a.endpoints, a.identity[0]) < std::tie(b.endpoints, b.identity[0]);
    });
    PhysicsContactPublication::Frame result;
    result.manifolds.reserve(reports.size());
    result.points.reserve(reports.size());
    const auto xyz = [](const ModelVector &value) { return glm::vec3(value[0], value[1], value[2]); };
    for (const auto &report : reports) {
        auto a = geometry.at(report.endpoints[0]);
        auto b = geometry.at(report.endpoints[1]);
        if (report.endpoints[2] != NoModelIndex)
            a.subshape = report.endpoints[2];
        if (report.endpoints[3] != NoModelIndex)
            b.subshape = report.endpoints[3];
        const bool trigger = report.identity[1] != 0;
        if (!policy.Observes(a.SurfaceKey(), b.SurfaceKey(), trigger))
            continue;
        result.manifolds.push_back({std::move(a), std::move(b), trigger, result.points.size(), 1});
        result.points.push_back({xyz(report.pointSeparation), xyz(report.normal), xyz(report.relativeVelocity),
                                 xyz(report.impulse), report.identity[0], report.identity[2] != 0,
                                 report.pointSeparation[3]});
    }
    return result;
}

} // namespace infernux::physics::warp
