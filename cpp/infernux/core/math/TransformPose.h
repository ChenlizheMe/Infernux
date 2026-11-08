#pragma once

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <algorithm>
#include <cmath>
#include <glm/gtx/matrix_decompose.hpp>
#include <stdexcept>
#include <string>

namespace infernux
{
struct TransformPose
{
    glm::vec3 position, scale;
    glm::quat rotation;
};

/// Import boundary: Transform stores TRS, so never silently discard source shear.
inline TransformPose DecomposeTransformPose(const glm::mat4 &matrix, const std::string &name)
{
    TransformPose pose;
    glm::vec3 skew;
    glm::vec4 perspective;
    if (!glm::decompose(matrix, pose.scale, pose.rotation, pose.position, skew, perspective))
        throw std::invalid_argument("Model node has a non-decomposable transform: " + name);
    pose.rotation = glm::normalize(pose.rotation);
    const glm::mat4 rebuilt = glm::translate(glm::mat4(1), pose.position) * glm::mat4_cast(pose.rotation) *
                              glm::scale(glm::mat4(1), pose.scale);
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            if (!std::isfinite(matrix[column][row]) || !std::isfinite(rebuilt[column][row]) ||
                std::abs(rebuilt[column][row] - matrix[column][row]) >
                    1e-5f * std::max(1.0f, std::abs(matrix[column][row])))
                throw std::invalid_argument("Model node shear or perspective requires baking: " + name);
    return pose;
}
} // namespace infernux
