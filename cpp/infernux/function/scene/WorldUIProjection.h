#pragma once

#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

namespace infernux
{

// These two bits are independent, opt-in geometry policies. The same basis
// and pixel scale are used by the vertex shader and the native input batch.
constexpr uint32_t WorldUIBillboard = 1u;
constexpr uint32_t WorldUIConstantScreenSize = 2u;

inline float WorldUIPixelScale(const glm::mat4 &viewProjection, const glm::mat4 &projection, const glm::vec3 &anchor,
                               float viewportHeight)
{
    const float clipW = (viewProjection * glm::vec4(anchor, 1.0f)).w;
    const float denominator = std::abs(projection[1][1]) * viewportHeight;
    return denominator > 1e-6f && clipW > 0.0f ? 2.0f * clipW / denominator : 0.0f;
}

inline glm::mat3 WorldUIBasis(const glm::mat4 &authoredPose, const glm::mat4 &view, bool billboard)
{
    if (!billboard)
        return glm::mat3(authoredPose);
    const glm::mat4 cameraToWorld = glm::inverse(view);
    return glm::mat3(glm::vec3(cameraToWorld[0]), glm::vec3(cameraToWorld[1]),
                     glm::normalize(glm::cross(glm::vec3(cameraToWorld[0]), glm::vec3(cameraToWorld[1]))));
}

} // namespace infernux
