#pragma once

#include "PhysicsTargetReference.h"

#include <glm/glm.hpp>

#include <cstdint>

namespace infernux
{

class Collider;
class GameObject;

// ============================================================================
// Collision — Unity-style collision data passed to callbacks
// ============================================================================

/**
 * @brief Data about a collision event (Unity: Collision).
 *
 * Passed to OnCollisionEnter / OnCollisionStay / OnCollisionExit.
 * For trigger events, only collider/gameObject are meaningful.
 */
struct CollisionInfo
{
    Collider *collider = nullptr;      ///< The other collider involved
    GameObject *gameObject = nullptr;  ///< The other GameObject
    PhysicsTargetReference target;     ///< Safe identity when a script retains this value
    PhysicsTargetReference bodyTarget; ///< Body that participated in this event, independent of later reparenting
    glm::vec3 contactPoint{0.0f};      ///< World-space contact point (first contact)
    glm::vec3 contactNormal{0.0f};     ///< Contact normal (points from other → this)
    glm::vec3 relativeVelocity{0.0f};  ///< Relative velocity between the bodies
};

// ============================================================================
// Internal event struct — used between contact publication and dispatch
// ============================================================================

enum class ContactEventType : uint8_t
{
    CollisionEnter,
    CollisionStay,
    CollisionExit,
    TriggerEnter,
    TriggerStay,
    TriggerExit,
};

enum ContactEventInterest : uint8_t
{
    CollisionEnterInterest = 1u << 0u,
    CollisionStayInterest = 1u << 1u,
    CollisionExitInterest = 1u << 2u,
    TriggerEnterInterest = 1u << 3u,
    TriggerStayInterest = 1u << 4u,
    TriggerExitInterest = 1u << 5u,
};

[[nodiscard]] constexpr uint8_t ContactEventInterestBit(ContactEventType type)
{
    switch (type) {
    case ContactEventType::CollisionEnter:
        return CollisionEnterInterest;
    case ContactEventType::CollisionStay:
        return CollisionStayInterest;
    case ContactEventType::CollisionExit:
        return CollisionExitInterest;
    case ContactEventType::TriggerEnter:
        return TriggerEnterInterest;
    case ContactEventType::TriggerStay:
        return TriggerStayInterest;
    case ContactEventType::TriggerExit:
        return TriggerExitInterest;
    }
    return 0;
}

struct ContactEvent
{
    ContactEventType type;
    uint32_t bodyIdA; ///< Opaque body identity for body A
    uint32_t bodyIdB; ///< Opaque body identity for body B
    uint32_t subShapeIdA = 0;
    uint32_t subShapeIdB = 0;
    glm::vec3 contactPoint{0.0f};
    glm::vec3 contactNormal{0.0f}; ///< Contact normal, body A -> body B
    glm::vec3 relativeVelocity{0.0f};
};

} // namespace infernux
