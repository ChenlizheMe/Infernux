#pragma once

/**
 * @file PhysicsLayers.h
 * @brief Jolt Physics layer mapping for Infernux.
 *
 * Maps Infernux's 32 GameObject layers to Jolt object layers while still
 * using a simplified 2-layer broad phase (moving / static).
 */

#include "../TagLayerManager.h"

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>

namespace infernux
{

/// Object layers (Jolt requires uint16_t).
/// We encode 32 user layers for NON_MOVING and 32 user layers for MOVING.
namespace PhysicsObjectLayers
{
static constexpr uint16_t LAYER_COUNT = 32;
static constexpr JPH::ObjectLayer NON_MOVING_BASE = 0;
static constexpr JPH::ObjectLayer MOVING_BASE = LAYER_COUNT;
static constexpr JPH::ObjectLayer MIXED_NON_MOVING = LAYER_COUNT * 2;
static constexpr JPH::ObjectLayer MIXED_MOVING = MIXED_NON_MOVING + 1;
static constexpr JPH::ObjectLayer NUM_LAYERS = MIXED_MOVING + 1;

inline bool IsMixed(JPH::ObjectLayer layer)
{
    return layer == MIXED_NON_MOVING || layer == MIXED_MOVING;
}

inline JPH::ObjectLayer Encode(int gameLayer, bool moving)
{
    uint16_t clamped =
        static_cast<uint16_t>((gameLayer >= 0 && gameLayer < static_cast<int>(LAYER_COUNT)) ? gameLayer : 0);
    return static_cast<JPH::ObjectLayer>((moving ? MOVING_BASE : NON_MOVING_BASE) + clamped);
}

inline int DecodeGameLayer(JPH::ObjectLayer objectLayer)
{
    return static_cast<int>(objectLayer % LAYER_COUNT);
}

inline bool IsMovingLayer(JPH::ObjectLayer objectLayer)
{
    return objectLayer == MIXED_MOVING || (objectLayer >= MOVING_BASE && objectLayer < MIXED_NON_MOVING);
}
} // namespace PhysicsObjectLayers

/// Broad-phase layers (Jolt performance optimisation).
namespace PhysicsBroadPhaseLayers
{
static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
static constexpr JPH::BroadPhaseLayer MOVING(1);
static constexpr uint32_t NUM_LAYERS = 2;
} // namespace PhysicsBroadPhaseLayers

/// Maps object layer → broad-phase layer.
class BPLayerInterface final : public JPH::BroadPhaseLayerInterface
{
  public:
    BPLayerInterface()
    {
        for (uint16_t i = 0; i < PhysicsObjectLayers::LAYER_COUNT; ++i) {
            m_objectToBroadPhase[PhysicsObjectLayers::NON_MOVING_BASE + i] = PhysicsBroadPhaseLayers::NON_MOVING;
            m_objectToBroadPhase[PhysicsObjectLayers::MOVING_BASE + i] = PhysicsBroadPhaseLayers::MOVING;
        }
        m_objectToBroadPhase[PhysicsObjectLayers::MIXED_NON_MOVING] = PhysicsBroadPhaseLayers::NON_MOVING;
        m_objectToBroadPhase[PhysicsObjectLayers::MIXED_MOVING] = PhysicsBroadPhaseLayers::MOVING;
    }

    [[nodiscard]] uint32_t GetNumBroadPhaseLayers() const override
    {
        return PhysicsBroadPhaseLayers::NUM_LAYERS;
    }

    [[nodiscard]] JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override
    {
        return m_objectToBroadPhase[inLayer];
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override
    {
        switch ((JPH::BroadPhaseLayer::Type)inLayer) {
        case (JPH::BroadPhaseLayer::Type)PhysicsBroadPhaseLayers::NON_MOVING:
            return "NON_MOVING";
        case (JPH::BroadPhaseLayer::Type)PhysicsBroadPhaseLayers::MOVING:
            return "MOVING";
        default:
            return "UNKNOWN";
        }
    }
#endif

  private:
    JPH::BroadPhaseLayer m_objectToBroadPhase[PhysicsObjectLayers::NUM_LAYERS];
};

/// Determines object-vs-broadphase layer collision.
class ObjectVsBPLayerFilter final : public JPH::ObjectVsBroadPhaseLayerFilter
{
  public:
    [[nodiscard]] bool ShouldCollide(JPH::ObjectLayer inObjLayer, JPH::BroadPhaseLayer inBPLayer) const override
    {
        if (!PhysicsObjectLayers::IsMovingLayer(inObjLayer)) {
            return inBPLayer == PhysicsBroadPhaseLayers::MOVING;
        }
        return true;
    }
};

/// Determines object-vs-object layer collision.
class ObjectLayerPairFilter final : public JPH::ObjectLayerPairFilter
{
  public:
    [[nodiscard]] bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::ObjectLayer inLayer2) const override
    {
        if (!PhysicsObjectLayers::IsMovingLayer(inLayer1) && !PhysicsObjectLayers::IsMovingLayer(inLayer2)) {
            return false;
        }

        // A compound may contain several game layers. Broad phase admits
        // candidates; the contact listener checks the actual two leaf shapes.
        if (PhysicsObjectLayers::IsMixed(inLayer1) || PhysicsObjectLayers::IsMixed(inLayer2))
            return true;

        const int gameLayer1 = PhysicsObjectLayers::DecodeGameLayer(inLayer1);
        const int gameLayer2 = PhysicsObjectLayers::DecodeGameLayer(inLayer2);
        return TagLayerManager::Instance().GetLayersCollide(gameLayer1, gameLayer2);
    }
};

} // namespace infernux
