#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <optional>
#include <variant>

namespace infernux
{
enum class ForceMode;

enum class PhysicsForceKind : uint8_t
{
    Force,
    Torque,
    ForceAtPosition,
};

/// Commands name the stable scene body. Backend slots are resolved during admission.
struct PhysicsBodyCommand
{
    struct Pose
    {
        std::optional<glm::vec3> position;
        std::optional<glm::quat> rotation;
        bool cancelDrive = true;
    };
    struct Velocity
    {
        std::optional<glm::vec3> linear, angular;
    };
    struct Force
    {
        PhysicsForceKind kind;
        ForceMode mode;
        glm::vec3 value;
        glm::vec3 point{0.0f};
    };
    struct ResetMotion
    {
        glm::vec3 position;
        glm::quat rotation;
        glm::vec3 linear, angular;
    };
    struct KinematicMove
    {
        std::optional<glm::vec3> position;
        std::optional<glm::quat> rotation;
        float duration;
        float maxSpeed = 0.0f;
        bool temporaryStatic = false;
    };

    uint32_t bodyId;
    std::variant<Pose, Velocity, Force, ResetMotion, KinematicMove> operation;
};
} // namespace infernux
