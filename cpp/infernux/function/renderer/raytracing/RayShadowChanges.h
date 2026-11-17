#pragma once

#include "RayGeometry.h"

#include <function/renderer/RenderLightingSource.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace infernux::raytracing
{

struct RayShadowInstance
{
    uint32_t surfaceId;
    uint32_t layerMask;
    bool castsShadows;
    bool receivesShadows;
    glm::mat4 objectToWorld;
    spatial::Bounds bounds;
};

struct RayShadowState
{
    uint32_t serial = 1;
    uint64_t lightingGeneration = 0;
    std::vector<RayShadowInstance> instances;
    std::vector<std::array<uint32_t, 4>> stamps;
};

/// Tracks the latest potentially invalidating light path change for each
/// secondary surface. Visits and stamp copies consume the geometry CPU budget.
class RayShadowChanges
{
  public:
    RayShadowChanges(const RayShadowState *previous, std::vector<RayShadowInstance> instances,
                     const RayInstancePage &page, const RenderLightingSource &lighting, size_t surfaceCount)
        : m_previous(previous), m_page(page), m_lighting(lighting),
          m_stampCount(std::max(size_t{1}, (surfaceCount + 3) / 4))
    {
        m_state.instances = std::move(instances);
        m_state.lightingGeneration = lighting.generation;
        m_state.stamps.reserve(m_stampCount);
        if (m_previous) {
            const bool wrapped = m_previous->serial == UINT32_MAX;
            m_state.serial = wrapped ? 1u : m_previous->serial + 1;
            m_resetSlots = wrapped || m_previous->lightingGeneration != m_state.lightingGeneration ||
                           m_previous->instances.size() != m_state.instances.size() ||
                           m_previous->stamps.size() != m_stampCount;
            if (!m_resetSlots)
                m_changedCasters.reserve(m_state.instances.size());
        }
    }

    [[nodiscard]] bool Advance(uint32_t &budget)
    {
        while (budget && m_phase != Phase::Done) {
            switch (m_phase) {
            case Phase::CopyStamps:
                if (m_state.stamps.size() == m_stampCount) {
                    m_phase = Phase::Compare;
                    break;
                }
                m_state.stamps.push_back(m_previous && !m_resetSlots ? m_previous->stamps[m_state.stamps.size()]
                                                                     : std::array<uint32_t, 4>{});
                --budget;
                break;
            case Phase::Compare:
                if (m_cursor == m_state.instances.size()) {
                    m_cursor = 0;
                    m_phase = m_resetSlots ? Phase::ResetSlots : m_changedCasters.empty() ? Phase::Done : Phase::Lights;
                    break;
                }
                Compare(m_cursor++);
                --budget;
                break;
            case Phase::ResetSlots:
                if (m_cursor == m_stampCount) {
                    m_phase = Phase::Done;
                    break;
                }
                m_state.stamps[m_cursor++].fill(m_state.serial);
                --budget;
                break;
            case Phase::Lights:
                if (m_cursor == m_lighting.lights.size()) {
                    m_cursor = 0;
                    m_phase = m_lights.empty() || m_page.nodes.empty() ? Phase::Done : Phase::Query;
                    break;
                }
                PrepareLight(m_lighting.lights[m_cursor++]);
                --budget;
                break;
            case Phase::Query:
                if (m_stackSize != 0) {
                    Visit(m_page.nodes[m_stack[--m_stackSize]]);
                } else if (m_nextCaster == m_changedCasters.size()) {
                    m_phase = Phase::Done;
                    break;
                } else {
                    PrepareQuery(m_changedCasters[m_nextCaster], m_lights[m_nextLight]);
                    if (++m_nextLight == m_lights.size()) {
                        m_nextLight = 0;
                        ++m_nextCaster;
                    }
                }
                --budget;
                break;
            case Phase::Done:
                break;
            }
        }
        return m_phase == Phase::Done;
    }

    [[nodiscard]] RayShadowState Take()
    {
        return std::move(m_state);
    }

  private:
    using Vec = glm::dvec3;

    struct Box
    {
        Vec lower;
        Vec upper;
    };

    struct ShadowLight
    {
        uint32_t layers = 0;
        std::array<Vec, 3> axes{};
        std::array<Box, 4> endpoints{};
        uint32_t endpointCount = 0;
        double range = 0;

        [[nodiscard]] bool IsDirectional() const noexcept
        {
            return endpointCount == 0;
        }
    };

    enum class Phase : uint8_t
    {
        CopyStamps,
        Compare,
        ResetSlots,
        Lights,
        Query,
        Done
    };

    static Vec NumericPadding(const Vec &magnitude)
    {
        // Include the ray origin's 256-ULP normal offset, plus error in hit
        // reconstruction and light direction. Pad before projection.
        constexpr double scale = 512.0 * std::numeric_limits<float>::epsilon();
        return glm::max(Vec(2.0 / 65536.0), scale * magnitude);
    }

    static Box PaddedBox(const Vec &lower, const Vec &upper)
    {
        const Vec padding = NumericPadding(glm::max(glm::abs(lower), glm::abs(upper)));
        return {lower - padding, upper + padding};
    }

    static Box PaddedBox(const spatial::Bounds &bounds)
    {
        return PaddedBox(Vec(bounds.minimum), Vec(bounds.maximum));
    }

    static Box PaddedBox(const spatial::BvhNode &node)
    {
        return PaddedBox({node.minimum[0], node.minimum[1], node.minimum[2]},
                         {node.maximum[0], node.maximum[1], node.maximum[2]});
    }

    static Box Project(const Box &box, const std::array<Vec, 3> &axes)
    {
        const Vec center = 0.5 * (box.lower + box.upper);
        const Vec extent = 0.5 * (box.upper - box.lower);
        Box projection;
        for (int axis = 0; axis < 3; ++axis) {
            const double t = glm::dot(axes[axis], center);
            const double r = glm::dot(glm::abs(axes[axis]), extent);
            projection.lower[axis] = t - r;
            projection.upper[axis] = t + r;
        }
        return projection;
    }

    static bool ClipPathTime(double start, double end, double limit, double &first, double &last)
    {
        if (start <= limit && end <= limit)
            return true;
        if (start > limit && end > limit)
            return false;
        const double crossing = (limit - start) / (end - start);
        if (start > limit)
            first = std::max(first, std::nextafter(crossing, 0.0));
        else
            last = std::min(last, std::nextafter(crossing, 1.0));
        return first <= last;
    }

    static bool PathOverlaps(const Box &emitter, const Box &receiver, const Box &caster)
    {
        // At each t the set of possible segment points is the interpolated
        // axis-aligned box. All three axes must overlap at the same t.
        double first = 0.0;
        double last = 1.0;
        for (int axis = 0; axis < 3; ++axis) {
            if (!ClipPathTime(emitter.lower[axis], receiver.lower[axis], caster.upper[axis], first, last) ||
                !ClipPathTime(-emitter.upper[axis], -receiver.upper[axis], -caster.lower[axis], first, last))
                return false;
        }
        return true;
    }

    [[nodiscard]] bool MayReach(const Box &receiver) const
    {
        const auto &light = *m_queryLight;
        if (light.IsDirectional()) {
            const Box projection = Project(receiver, light.axes);
            for (uint32_t side = 0; side < m_queryCasterCount; ++side) {
                const auto &caster = m_queryCasters[side];
                if (projection.lower.x <= caster.upper.x && caster.lower.x <= projection.upper.x &&
                    projection.lower.y <= caster.upper.y && caster.lower.y <= projection.upper.y &&
                    projection.lower.z <= caster.upper.z)
                    return true;
            }
            return false;
        }

        for (uint32_t sample = 0; sample < light.endpointCount; ++sample) {
            const auto &endpoint = light.endpoints[sample];
            const Vec gap =
                glm::max(Vec(0.0), glm::max(receiver.lower - endpoint.upper, endpoint.lower - receiver.upper));
            if (glm::dot(gap, gap) > light.range * light.range)
                continue;
            for (uint32_t side = 0; side < m_queryCasterCount; ++side)
                if (PathOverlaps(endpoint, receiver, m_queryCasters[side]))
                    return true;
        }
        return false;
    }

    [[nodiscard]] uint32_t &Stamp(uint32_t surfaceId)
    {
        return m_state.stamps[surfaceId >> 2u][surfaceId & 3u];
    }

    void Compare(size_t index)
    {
        auto &current = m_state.instances[index];
        current.bounds = m_page.sourceBounds[index];
        if (!m_previous || m_resetSlots)
            return;
        const auto &prior = m_previous->instances[index];
        if (current.surfaceId != prior.surfaceId) {
            m_resetSlots = true;
            return;
        }
        const bool moved = current.objectToWorld != prior.objectToWorld;
        const bool changedLayer = current.layerMask != prior.layerMask;
        if (moved || changedLayer || current.receivesShadows != prior.receivesShadows)
            Stamp(current.surfaceId) = m_state.serial;
        if ((current.castsShadows || prior.castsShadows) &&
            (moved || changedLayer || current.castsShadows != prior.castsShadows))
            m_changedCasters.push_back(index);
    }

    void PrepareLight(const RenderLightSource &source)
    {
        // Matches the geometry domain and shadow selection of inx_transport_light.
        constexpr uint32_t geometryDomain = 1u;
        constexpr uint32_t directional = 0u;
        constexpr uint32_t rectangle = 3u;
        if (!source.shadows || source.shadowStrength <= 0.0f || source.intensity <= 0.0f ||
            !(source.influenceDomains & geometryDomain) || !source.cullingMask ||
            (source.type != directional && source.range <= 0.0f))
            return;

        auto &light = m_lights.emplace_back();
        light.layers = source.cullingMask;
        if (source.type == directional) {
            const Vec towardLight = glm::dot(source.direction, source.direction) > 0.0f
                                        ? -glm::normalize(Vec(source.direction))
                                        : Vec(0.0, 1.0, 0.0);
            const Vec axis = std::abs(towardLight.y) < 0.9 ? Vec(0.0, 1.0, 0.0) : Vec(1.0, 0.0, 0.0);
            const Vec tangent = glm::normalize(glm::cross(towardLight, axis));
            light.axes = {tangent, glm::cross(towardLight, tangent), towardLight};
            return;
        }

        light.range = double(source.range) * (1.0 + 8.0 * std::numeric_limits<float>::epsilon());
        light.endpointCount = 1;
        const Vec position(source.position);
        light.endpoints[0] = PaddedBox(position, position);
        if (source.type == rectangle) {
            const Vec right = glm::normalize(Vec(source.right)) * (0.5 * source.areaSize.x);
            const Vec up = glm::normalize(Vec(source.up)) * (0.5 * source.areaSize.y);
            // Endpoint cancellation can be large even when its final world coordinate is near zero.
            const Vec padding = NumericPadding(glm::abs(position) + glm::abs(right) + glm::abs(up));
            light.endpointCount = 4;
            for (uint32_t i = 0; i < light.endpointCount; ++i) {
                const Vec endpoint = position + ((i & 1u) ? right : -right) + ((i & 2u) ? up : -up);
                light.endpoints[i] = {endpoint - padding, endpoint + padding};
            }
        }
    }

    void PrepareQuery(size_t instanceIndex, const ShadowLight &light)
    {
        m_queryLight = &light;
        m_queryCasterCount = 0;
        const auto add = [&](const RayShadowInstance &caster) {
            if (caster.castsShadows && (caster.layerMask & light.layers) != 0u) {
                const auto bounds = PaddedBox(caster.bounds);
                m_queryCasters[m_queryCasterCount++] = light.IsDirectional() ? Project(bounds, light.axes) : bounds;
            }
        };
        add(m_previous->instances[instanceIndex]);
        add(m_state.instances[instanceIndex]);
        if (m_queryCasterCount)
            m_stack[m_stackSize++] = 0;
    }

    void Visit(const spatial::BvhNode &node)
    {
        if (!MayReach(PaddedBox(node)))
            return;
        if ((node.second & spatial::kBvhLeafBit) == 0) {
            m_stack[m_stackSize++] = node.second;
            m_stack[m_stackSize++] = node.first;
            return;
        }
        const auto count = node.second & ~spatial::kBvhLeafBit;
        for (uint32_t i = node.first; i < node.first + count; ++i) {
            const auto &receiver = m_state.instances[m_page.topology->primitiveOrder[i]];
            auto &stamp = Stamp(receiver.surfaceId);
            if (stamp != m_state.serial && receiver.receivesShadows &&
                (receiver.layerMask & m_queryLight->layers) != 0u && MayReach(PaddedBox(receiver.bounds)))
                stamp = m_state.serial;
        }
    }

    const RayShadowState *m_previous;
    const RayInstancePage &m_page;
    const RenderLightingSource &m_lighting;
    const size_t m_stampCount;
    RayShadowState m_state;
    bool m_resetSlots = false;
    Phase m_phase = Phase::CopyStamps;
    size_t m_cursor = 0;
    std::vector<size_t> m_changedCasters;
    std::vector<ShadowLight> m_lights;
    size_t m_nextCaster = 0;
    size_t m_nextLight = 0;
    const ShadowLight *m_queryLight = nullptr;
    std::array<Box, 2> m_queryCasters{};
    uint32_t m_queryCasterCount = 0;
    std::array<uint32_t, spatial::kBvhMaximumDepth + 1> m_stack{};
    size_t m_stackSize = 0;
};

} // namespace infernux::raytracing
