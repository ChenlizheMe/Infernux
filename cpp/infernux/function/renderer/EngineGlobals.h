#pragma once

#include <cmath>
#include <core/config/MathConstants.h>
#include <core/math/DepthConvention.h>
#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>

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
 * Ten 16-byte rows share one camera-local uniform allocation.
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

    /// x = view flags, y = last accepted ray geometry, z = packed jitter delta in UV, w = accepted view samples.
    alignas(16) glm::uvec4 viewOptions{0u};

    /// Current projection displacement in top-left raster pixels.
    alignas(16) glm::vec2 rasterJitterPixels{0.0f};
};

static_assert(sizeof(EngineGlobalsUBO) == 160, "EngineGlobalsUBO must contain ten 16-byte rows");

[[nodiscard]] inline glm::uvec4 MakeViewOptions(bool dithering, bool stopNaNs, bool opaqueOutput,
                                                uint32_t previousRayGeometry = 0, uint32_t sampleIndex = 0,
                                                glm::vec2 jitterNdcDelta = {}, bool writesDisplayRgb = false)
{
    const uint32_t flags = static_cast<uint32_t>(dithering) | (static_cast<uint32_t>(stopNaNs) << 1u) |
                           (static_cast<uint32_t>(opaqueOutput) << 2u) |
                           (static_cast<uint32_t>(writesDisplayRgb) << 3u);
    // Both native projections express positive raster Y down. Convert NDC to top-left texture UV.
    return {flags, previousRayGeometry, glm::packHalf2x16(jitterNdcDelta * 0.5f), sampleIndex};
}

/// Frame time is authored once by the native host, then copied into each view.
inline EngineGlobalsUBO BuildEngineFrameGlobals(float time, float delta, float smoothDelta, uint64_t frame)
{
    EngineGlobalsUBO globals{};
    globals.time = glm::vec4(time, std::sin(time), std::cos(time), delta);
    globals.sinTime =
        glm::vec4(std::sin(time / 20.0f), std::sin(time / 4.0f), std::sin(time * 2.0f), std::sin(time * 3.0f));
    globals.cosTime =
        glm::vec4(std::cos(time / 20.0f), std::cos(time / 4.0f), std::cos(time * 2.0f), std::cos(time * 3.0f));
    globals.frameParams = glm::vec4(float(frame), smoothDelta, 1.0f / (delta > kEpsilon ? delta : kEpsilon), 0.0f);
    return globals;
}

} // namespace infernux
