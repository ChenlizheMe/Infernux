#pragma once

#include <core/math/DepthConvention.h>

#include <glm/glm.hpp>

namespace infernux
{

/**
 * @brief Engine time and camera-local parameters exposed via set 2.
 *
 * Layout matches Slang InxEngineGlobals. Each field occupies one 16-byte row.
 *
 * This is a fixed ABI populated by the engine. Arbitrary name-based
 * "set_global" dictionaries are deliberately not
 * part of the render path;
 * effect-local values belong to the pass payload and authored shader values
 * belong to
 * Material/Renderer set 0.
 *
 * Nine 16-byte rows share one camera-local uniform allocation.
 */
struct EngineGlobalsUBO
{
    // ─── Time ───────────────────────────────────────────────────────
    /// x = time since startup (s), y = sin(time), z = cos(time), w = deltaTime
    alignas(16) glm::vec4 time{0.0f};

    /// x = sin(time/20), y = sin(time/4), z = sin(time*2), w = sin(time*3)
    alignas(16) glm::vec4 sinTime{0.0f};

    /// x = cos(time/20), y = cos(time/4), z = cos(time*2), w = cos(time*3)
    alignas(16) glm::vec4 cosTime{0.0f};

    // ─── Screen ─────────────────────────────────────────────────────
    /// x = width, y = height, z = 1/width, w = 1/height
    alignas(16) glm::vec4 screenParams{1.0f, 1.0f, 1.0f, 1.0f};

    // ─── Camera ─────────────────────────────────────────────────────
    /// xyz = world-space camera position, w = 1.0
    alignas(16) glm::vec4 worldSpaceCameraPos{0.0f, 0.0f, 0.0f, 1.0f};

    /// x = near, y = far, z = 1/far, w = near/far
    alignas(16) glm::vec4 projectionParams{0.01f, 5000.0f, 0.0002f, 0.000002f};

    /// Linearization helpers for reversed-Z depth buffer.
    /// x = far/near - 1, y = 1, z = x/far, w = 1/far
    alignas(16) glm::vec4 zBufferParams = depth::LinearizationParams(0.01f, 5000.0f);

    // ─── Frame ──────────────────────────────────────────────────────
    /// x = frame count, y = smoothDeltaTime, z = 1/deltaTime, w = previous view sample time
    alignas(16) glm::vec4 frameParams{0.0f};

    /// x = dithering, y = stop NaNs, z = opaque display output, w = reserved.
    alignas(16) glm::uvec4 viewOptions{0u};
};

static_assert(sizeof(EngineGlobalsUBO) == 144, "EngineGlobalsUBO must contain nine 16-byte rows");

} // namespace infernux
