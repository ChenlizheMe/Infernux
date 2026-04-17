/**
 * @file PhysicsWorld.cpp
 * @brief Jolt Physics integration — singleton world, body management, raycasts.
 */

// Jolt requires specific defines before including its headers
#include <Jolt/Jolt.h>

// Jolt includes (order matters)
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include "PhysicsContactListener.h"
#include "PhysicsLayers.h"
#include "PhysicsWorld.h"

#include "../Collider.h"
#include "../Component.h"
#include "../GameObject.h"
#include "../Scene.h"
#include "../Transform.h"
#include <core/config/EngineConfig.h>
#include <core/config/MathConstants.h>
#include <core/log/InxLog.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <thread>
#include <unordered_set>

namespace infernux
{

namespace
{

constexpr float kMinQueryDirectionLengthSq = 1e-12f;

static bool NormalizeQueryDirection(const glm::vec3 &direction, float maxDistance, glm::vec3 &outDirection)
{
    if (maxDistance <= 0.0f) {
        return false;
    }

    const float lengthSq = glm::dot(direction, direction);
    if (lengthSq <= kMinQueryDirectionLengthSq) {
        return false;
    }

    outDirection = direction / std::sqrt(lengthSq);
    return true;
}

static int MapMotionQualityMode(int quality)
{
    switch (quality) {
    case 1:
    case 2:
        return 1;
    case 0:
    case 3:
    default:
        return 0;
    }
}

class LayerMaskObjectFilter final : public JPH::ObjectLayerFilter
{
  public:
    explicit LayerMaskObjectFilter(uint32_t layerMask) : m_layerMask(layerMask)
    {
    }

    bool ShouldCollide(JPH::ObjectLayer inLayer) const override
    {
        const int gameLayer = PhysicsObjectLayers::DecodeGameLayer(inLayer);
        return (m_layerMask & (1u << static_cast<uint32_t>(gameLayer))) != 0;
    }

  private:
    uint32_t m_layerMask;
};

static JPH::RefConst<JPH::Shape> BuildShapeForColliderSet(GameObject *go, const Collider *exclude)
{
    if (!go) {
        return nullptr;
    }

    std::vector<std::pair<Collider *, JPH::RefConst<JPH::Shape>>> childShapes;
    auto colliders = go->GetComponents<Collider>();
    childShapes.reserve(colliders.size());

    for (auto *col : colliders) {
        if (!col || col == exclude || !col->IsEnabled()) {
            continue;
        }

        JPH::RefConst<JPH::Shape> child(static_cast<const JPH::Shape *>(col->CreateJoltShapeRaw()));
        if (child) {
            childShapes.emplace_back(col, child);
        }
    }

    if (childShapes.empty()) {
        return nullptr;
    }

    if (childShapes.size() == 1) {
        return childShapes.front().second;
    }

    JPH::StaticCompoundShapeSettings compoundSettings;
    for (size_t i = 0; i < childShapes.size(); ++i) {
        auto *col = childShapes[i].first;
        uint32_t userData = col ? static_cast<uint32_t>(col->GetComponentID()) : 0;
        compoundSettings.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), childShapes[i].second, userData);
    }

    auto result = compoundSettings.Create();
    if (result.HasError()) {
        return nullptr;
    }
    return result.Get();
}

} // namespace

// ============================================================================
// Jolt trace / assert callbacks (required by Jolt)
// ============================================================================

static void JoltTraceImpl(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    INXLOG_DEBUG("[Jolt] ", buffer);
}

#ifdef JPH_ENABLE_ASSERTS
static bool JoltAssertFailed(const char *expression, const char *message, const char *file, unsigned int line)
{
    INXLOG_ERROR("[Jolt Assert] ", file, ":", line, " – ", expression, " – ", message ? message : "");
    // Returning true triggers __debugbreak() inside Jolt's assert macro.
    // In debug builds (debugger attached), this is desirable so we can inspect the state.
    // In release / no-debugger builds, return false to log-and-continue.
#ifdef NDEBUG
    return false;
#else
    return true;
#endif
}
#endif

// ============================================================================
// Layer helpers (must outlive PhysicsSystem)
// ============================================================================

struct PhysicsWorld::LayerInterfaces
{
    BPLayerInterface bpInterface;
    ObjectVsBPLayerFilter objVsBpFilter;
    ObjectLayerPairFilter objPairFilter;
};

// ============================================================================
// Singleton
// ============================================================================

PhysicsWorld &PhysicsWorld::Instance()
{
    static PhysicsWorld instance;
    return instance;
}

PhysicsWorld::~PhysicsWorld()
{
    Shutdown();
}

// ============================================================================
// Init / Shutdown
// ============================================================================

void PhysicsWorld::Initialize()
{
    if (m_initialized)
        return;

    INXLOG_INFO("PhysicsWorld: Initializing Jolt Physics…");

    // Register Jolt allocation hooks (use default malloc)
    JPH::RegisterDefaultAllocator();

    // Install trace / assert callbacks
    JPH::Trace = JoltTraceImpl;
#ifdef JPH_ENABLE_ASSERTS
    JPH::AssertFailed = JoltAssertFailed;
#endif

    // Create factory & register types
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    // -------------------------------------------------------------------------
    // TempAllocatorImpl — shared stack pool for ALL Jolt worker threads.
    //
    // IMPORTANT: this is NOT a per-thread allocator.  All threads push/pop
    // from the same pool concurrently during broadphase + narrowphase +
    // constraint solver.  Running out of space calls std::abort() immediately,
    // so the pool must be sized for PEAK simultaneous usage across all threads.
    //
    // Sizing guide (bodies in scene → recommended pool):
    //   < 256  bodies : 32 MB  is comfortable
    //   < 1024 bodies : 64 MB  recommended
    //   < 4096 bodies : 128 MB recommended
    //   ≥ 4096 bodies : 256 MB or more
    //
    // Increasing this value has negligible real memory cost because the OS
    // only commits pages that are actually touched (virtual memory).
    // -------------------------------------------------------------------------
    auto &cfg = EngineConfig::Get();

    int hwThreads = static_cast<int>(std::thread::hardware_concurrency());
    int numThreads = cfg.physicsMaxWorkerThreads > 0 ? cfg.physicsMaxWorkerThreads : std::clamp(hwThreads - 1, 1, 8);

    m_tempAllocator = std::make_unique<JPH::TempAllocatorImpl>(cfg.physicsTempAllocatorSize);

    m_jobSystem = std::make_unique<JPH::JobSystemThreadPool>(cfg.physicsMaxJobs, cfg.physicsMaxBarriers, numThreads);

    // Layer interfaces
    m_layers = std::make_unique<LayerInterfaces>();

    // Physics system
    m_physicsSystem = std::make_unique<JPH::PhysicsSystem>();
    m_physicsSystem->Init(cfg.physicsMaxBodies, 0, cfg.physicsMaxBodyPairs, cfg.physicsMaxContactConstraints,
                          m_layers->bpInterface, m_layers->objVsBpFilter, m_layers->objPairFilter);

    // Tune physics settings for thin-body stability and precision.
    //  - Penetration slop: 2 mm (default 20 mm). Min BoxShape thickness is
    //    2 × kMinHalfExtent = 0.02 m; 2 mm slop keeps sinking at 10 %.
    //  - Speculative contact: 10 mm (default 20 mm).  Tighter for thin shells.
    //  - Position solver: 3 iterations (default 2). Better stacking stability.
    //  - LinearCast max penetration: 10 % of inner radius (default 25 %).
    //  - Baumgarte: 0.15 (default 0.2). Softer correction prevents violent
    //    pop-out when thin bodies briefly penetrate a surface.
    //  - Max penetration distance: 50 mm (default 200 mm). Limits per-step
    //    correction so thin-body overlaps resolve gradually.
    //  - LinearCast threshold: 50 % (default 75 %). Triggers CCD earlier,
    //    critical for thin bodies whose inner radius is very small.
    JPH::PhysicsSettings settings = m_physicsSystem->GetPhysicsSettings();
    settings.mPenetrationSlop = 0.002f;           // 2 mm  (default 20 mm)
    settings.mSpeculativeContactDistance = 0.01f; // 10 mm (default 20 mm)
    settings.mNumPositionSteps = 3;               // (default 2)
    settings.mLinearCastMaxPenetration = 0.1f;    // (default 0.25)
    settings.mBaumgarte = 0.15f;                  // softer penetration correction (default 0.2)
    settings.mMaxPenetrationDistance = 0.05f;     // limit per-step correction for thin bodies (default 0.2)
    settings.mLinearCastThreshold = 0.5f;         // trigger CCD earlier for thin bodies (default 0.75)
    m_physicsSystem->SetPhysicsSettings(settings);

    // Gravity
    m_physicsSystem->SetGravity(JPH::Vec3(cfg.physicsGravity.x, cfg.physicsGravity.y, cfg.physicsGravity.z));

    // Install contact listener for collision/trigger callbacks
    m_contactListener = std::make_unique<InxContactListener>();
    m_physicsSystem->SetContactListener(m_contactListener.get());

    // Warm up the broadphase / job-system worker threads before the first real
    // Step().  On a cold start with many bodies, the very first Step() generates
    // a burst of jobs that can transiently exceed queue limits; running this
    // no-op pass lets the thread pool spin up and avoids the burst.
    m_physicsSystem->OptimizeBroadPhase();

    m_initialized = true;
    INXLOG_INFO("PhysicsWorld: Jolt Physics initialized.");
}

void PhysicsWorld::Shutdown()
{
    if (!m_initialized)
        return;

    INXLOG_INFO("PhysicsWorld: Shutting down...");
    if (m_contactListener)
        m_contactListener->ClearAll();
    m_bodyToCollider.clear();
    m_contactListener.reset();
    m_physicsSystem.reset();
    m_jobSystem.reset();
    m_tempAllocator.reset();
    m_layers.reset();

    JPH::UnregisterTypes();
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;

    m_initialized = false;
}

// ============================================================================
// Step
// ============================================================================

void PhysicsWorld::Step(float deltaTime)
{
    if (!m_initialized)
        return;

    if (m_contactListener)
        m_contactListener->PreStep();

    // 2 collision steps for better accuracy with fast-moving objects.
    // Each collision step subdivides the interval, giving narrower deltas
    // that prevent tunneling even at high velocities (e.g. objects dropped
    // from large heights).  Cost is ~1.5× one step (broadphase is shared).
    m_physicsSystem->Update(deltaTime, EngineConfig::Get().physicsCollisionSteps, m_tempAllocator.get(),
                            m_jobSystem.get());

    // Resolve raw contact events through pair tracking — suppresses spurious
    // Enter/Exit caused by Jolt's body sleep/wake cycles.
    if (m_contactListener) {
        JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
        m_contactListener->ResolveEvents(bi);
    }
}

// ============================================================================
// Contact event dispatch (Unity-style collision/trigger callbacks)
// ============================================================================

void PhysicsWorld::DispatchContactEvents()
{
    if (!m_contactListener)
        return;

    const auto &events = m_contactListener->GetEvents();
    if (events.empty())
        return;

    std::vector<Component *> receiversA;
    std::vector<Component *> receiversB;
    receiversA.reserve(8);
    receiversB.reserve(8);

    for (const auto &evt : events) {
        Collider *colA = ResolveColliderForSubShape(evt.bodyIdA, evt.subShapeIdA);
        Collider *colB = ResolveColliderForSubShape(evt.bodyIdB, evt.subShapeIdB);
        if (!colA || !colB)
            continue;

        GameObject *goA = colA->GetGameObject();
        GameObject *goB = colB->GetGameObject();
        if (!goA || !goB)
            continue;

        // Reclassify Exit events: OnContactRemoved doesn't have Body refs,
        // so the listener stores CollisionExit. Check trigger flag here.
        ContactEventType type = evt.type;
        if (type == ContactEventType::CollisionExit) {
            if (IsBodySensor(evt.bodyIdA) || IsBodySensor(evt.bodyIdB))
                type = ContactEventType::TriggerExit;
        }

        // Layer filtering for trigger events — sensors bypass Jolt's object layer
        // pair filter, so enforce Infernux's layer collision matrix here.
        bool isTrigger = (type == ContactEventType::TriggerEnter || type == ContactEventType::TriggerStay ||
                          type == ContactEventType::TriggerExit);
        if (isTrigger) {
            int layerA = goA->GetLayer();
            int layerB = goB->GetLayer();
            if (!TagLayerManager::Instance().GetLayersCollide(layerA, layerB))
                continue;
        }

        receiversA.clear();
        receiversB.clear();

        for (const auto &comp : goA->GetAllComponents()) {
            if (!comp || !comp->IsEnabled() || !comp->WantsPhysicsCallbacks())
                continue;
            receiversA.push_back(comp.get());
        }

        for (const auto &comp : goB->GetAllComponents()) {
            if (!comp || !comp->IsEnabled() || !comp->WantsPhysicsCallbacks())
                continue;
            receiversB.push_back(comp.get());
        }

        if (receiversA.empty() && receiversB.empty())
            continue;

        // Build CollisionInfo for each side
        CollisionInfo infoForA;
        infoForA.collider = colB;
        infoForA.gameObject = goB;
        infoForA.contactPoint = evt.contactPoint;
        infoForA.contactNormal = evt.contactNormal;
        infoForA.relativeVelocity = evt.relativeVelocity;
        infoForA.impulse = evt.impulse;

        CollisionInfo infoForB;
        infoForB.collider = colA;
        infoForB.gameObject = goA;
        infoForB.contactPoint = evt.contactPoint;
        infoForB.contactNormal = -evt.contactNormal; // flip for B
        infoForB.relativeVelocity = -evt.relativeVelocity;
        infoForB.impulse = evt.impulse;

        // Dispatch to all components on both GameObjects
        auto dispatchToReceivers = [&](const std::vector<Component *> &receivers, const CollisionInfo &info,
                                       ContactEventType t) {
            for (Component *comp : receivers) {
                switch (t) {
                case ContactEventType::CollisionEnter:
                    comp->OnCollisionEnter(info);
                    break;
                case ContactEventType::CollisionStay:
                    comp->OnCollisionStay(info);
                    break;
                case ContactEventType::CollisionExit:
                    comp->OnCollisionExit(info);
                    break;
                case ContactEventType::TriggerEnter:
                    comp->OnTriggerEnter(info.collider);
                    break;
                case ContactEventType::TriggerStay:
                    comp->OnTriggerStay(info.collider);
                    break;
                case ContactEventType::TriggerExit:
                    comp->OnTriggerExit(info.collider);
                    break;
                }
            }
        };

        dispatchToReceivers(receiversA, infoForA, type);
        // Guard: a callback on side A may have destroyed body B's physics body (and vice-versa).
        // Re-validate both sides before dispatching to B's receivers.
        if (FindColliderByBodyId(evt.bodyIdA) && FindColliderByBodyId(evt.bodyIdB))
            dispatchToReceivers(receiversB, infoForB, type);
    }
}

// ============================================================================
// Body management
// ============================================================================

uint32_t PhysicsWorld::CreateBody(Collider *collider, bool isStatic, bool isTrigger)
{
    if (!m_initialized || !collider)
        return 0xFFFFFFFF;

    auto *go = collider->GetGameObject();
    if (!go)
        return 0xFFFFFFFF;

    auto shape = BuildShapeForColliderSet(go, nullptr);
    if (!shape)
        return 0xFFFFFFFF;

    Transform *tf = go->GetTransform();
    glm::quat rot = tf->GetWorldRotation();
    glm::vec3 pos = tf->GetPosition();

    JPH::EMotionType motionType = isStatic ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic;
    JPH::ObjectLayer objLayer = PhysicsObjectLayers::Encode(go->GetLayer(), !isStatic);

    JPH::BodyCreationSettings settings(shape, JPH::RVec3(pos.x, pos.y, pos.z), JPH::Quat(rot.x, rot.y, rot.z, rot.w),
                                       motionType, objLayer);

    // Allow static bodies to later be switched to dynamic/kinematic
    // (e.g. when a Rigidbody component is added). Without this flag,
    // Jolt does not create MotionProperties for static bodies and
    // SetMotionType() will crash.
    settings.mAllowDynamicOrKinematic = true;
    settings.mIsSensor = isTrigger;

    // Set friction & restitution from the collider group (use max of all enabled colliders)
    float maxFriction = 0.0f;
    float maxBounciness = 0.0f;
    {
        auto allColliders = go->GetComponents<Collider>();
        for (auto *col : allColliders) {
            if (!col || !col->IsEnabled())
                continue;
            const auto &cd = PhysicsECSStore::Instance().GetCollider(col->GetECSHandle());
            maxFriction = std::max(maxFriction, cd.friction);
            maxBounciness = std::max(maxBounciness, cd.bounciness);
        }
    }
    settings.mFriction = maxFriction;
    settings.mRestitution = maxBounciness;

    // Use LinearCast (continuous collision detection) for dynamic bodies
    // to prevent fast-moving objects from tunneling through thin geometry.
    // Static bodies keep Discrete (no motion → no tunneling risk).
    if (!isStatic) {
        settings.mMotionQuality = JPH::EMotionQuality::LinearCast;
    }

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    JPH::Body *body = bodyInterface.CreateBody(settings);
    if (!body) {
        INXLOG_ERROR("PhysicsWorld: Failed to create body.");
        return 0xFFFFFFFF;
    }

    JPH::BodyID bodyId = body->GetID();
    // NOTE: Body is created but NOT added to broadphase here.
    // Collider::OnEnable() calls AddBodyToBroadphase() to add it.

    uint32_t id = bodyId.GetIndexAndSequenceNumber();
    m_bodyToCollider[id] = collider;
    return id;
}

void PhysicsWorld::DestroyBody(Collider *collider)
{
    if (!m_initialized || !collider)
        return;

    uint32_t id = collider->GetBodyId();
    if (id == 0xFFFFFFFF)
        return;

    JPH::BodyID bodyId(id);
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    // NOTE: Caller (Collider::UnregisterBody) must have already called
    // RemoveFromBroadphase() before reaching here.
    bodyInterface.DestroyBody(bodyId);

    m_bodyToCollider.erase(id);
}

void PhysicsWorld::SetBodyPosition(uint32_t bodyId, const glm::vec3 &pos, const glm::quat &rot)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.SetPositionAndRotation(JPH::BodyID(bodyId), JPH::RVec3(pos.x, pos.y, pos.z),
                                         JPH::Quat(rot.x, rot.y, rot.z, rot.w), JPH::EActivation::DontActivate);
}

void PhysicsWorld::UpdateBodyShape(Collider *collider, const Collider *exclude)
{
    if (!m_initialized || !collider)
        return;

    uint32_t id = collider->GetBodyId();
    if (id == 0xFFFFFFFF)
        return;

    auto newShape = BuildShapeForColliderSet(collider->GetGameObject(), exclude);
    if (!newShape)
        return;

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.SetShape(JPH::BodyID(id), newShape, true, JPH::EActivation::Activate);
}

void PhysicsWorld::SetBodyIsSensor(uint32_t bodyId, bool isSensor)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        lock.GetBody().SetIsSensor(isSensor);
    }
}

void PhysicsWorld::InvalidateContactPairsForBody(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;
    if (m_contactListener)
        m_contactListener->InvalidatePairsForBody(bodyId);
}

void PhysicsWorld::AddBodyToBroadphase(uint32_t bodyId, bool isStatic)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.AddBody(JPH::BodyID(bodyId), isStatic ? JPH::EActivation::DontActivate : JPH::EActivation::Activate);
}

void PhysicsWorld::AddBodiesBatch(const std::vector<std::pair<uint32_t, bool>> &bodies)
{
    if (!m_initialized || bodies.empty())
        return;

    // Separate static and dynamic bodies since they need different activation modes.
    std::vector<JPH::BodyID> staticIds;
    std::vector<JPH::BodyID> dynamicIds;
    staticIds.reserve(bodies.size());
    dynamicIds.reserve(bodies.size() / 4); // most spawned bodies are static

    for (auto &[id, isStatic] : bodies) {
        if (id == 0xFFFFFFFF)
            continue;
        if (isStatic)
            staticIds.push_back(JPH::BodyID(id));
        else
            dynamicIds.push_back(JPH::BodyID(id));
    }

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();

    if (!staticIds.empty()) {
        JPH::BodyInterface::AddState state = bi.AddBodiesPrepare(staticIds.data(), static_cast<int>(staticIds.size()));
        bi.AddBodiesFinalize(staticIds.data(), static_cast<int>(staticIds.size()), state,
                             JPH::EActivation::DontActivate);
    }
    if (!dynamicIds.empty()) {
        JPH::BodyInterface::AddState state =
            bi.AddBodiesPrepare(dynamicIds.data(), static_cast<int>(dynamicIds.size()));
        bi.AddBodiesFinalize(dynamicIds.data(), static_cast<int>(dynamicIds.size()), state, JPH::EActivation::Activate);
    }
}

void PhysicsWorld::RemoveBodyFromBroadphase(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.RemoveBody(JPH::BodyID(bodyId));
}

// ============================================================================
// Body dynamics (used by Rigidbody component)
// ============================================================================

void PhysicsWorld::SetBodyMotionType(uint32_t bodyId, int motionType)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::EMotionType mt;
    JPH::ObjectLayer layer;
    switch (motionType) {
    case 0:
        mt = JPH::EMotionType::Static;
        layer = PhysicsObjectLayers::Encode(0, false);
        break;
    case 1:
        mt = JPH::EMotionType::Kinematic;
        layer = PhysicsObjectLayers::Encode(0, true);
        break;
    case 2:
    default:
        mt = JPH::EMotionType::Dynamic;
        layer = PhysicsObjectLayers::Encode(0, true);
        break;
    }

    if (auto it = m_bodyToCollider.find(bodyId);
        it != m_bodyToCollider.end() && it->second && it->second->GetGameObject()) {
        layer = PhysicsObjectLayers::Encode(it->second->GetGameObject()->GetLayer(), motionType != 0);
    }

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetMotionType(JPH::BodyID(bodyId), mt, JPH::EActivation::Activate);
    bi.SetObjectLayer(JPH::BodyID(bodyId), layer);
}

void PhysicsWorld::SetBodyGameLayer(uint32_t bodyId, int gameLayer)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    const JPH::EMotionType motionType = bi.GetMotionType(JPH::BodyID(bodyId));
    const bool moving = motionType != JPH::EMotionType::Static;
    bi.SetObjectLayer(JPH::BodyID(bodyId), PhysicsObjectLayers::Encode(gameLayer, moving));
}

void PhysicsWorld::SetBodyMassProperties(uint32_t bodyId, float mass)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (body.IsDynamic()) {
            JPH::MotionProperties *mp = body.GetMotionProperties();
            if (mp->GetInverseMass() > 0.0f) {
                // Scale mass and inertia proportionally
                mp->ScaleToMass(mass > 0.001f ? mass : 0.001f);
            } else {
                // Body was just switched from static — compute mass from shape
                JPH::MassProperties massProp = body.GetShape()->GetMassProperties();
                massProp.ScaleToMass(mass > 0.001f ? mass : 0.001f);
                mp->SetMassProperties(JPH::EAllowedDOFs::All, massProp);
            }
        }
    }
}

void PhysicsWorld::SetBodyDamping(uint32_t bodyId, float linearDamping, float angularDamping)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (!body.IsStatic()) {
            JPH::MotionProperties *mp = body.GetMotionProperties();
            mp->SetLinearDamping(linearDamping);
            mp->SetAngularDamping(angularDamping);
        }
    }
}

void PhysicsWorld::SetBodyGravityFactor(uint32_t bodyId, float factor)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (!body.IsStatic()) {
            body.GetMotionProperties()->SetGravityFactor(factor);
        }
    }
}

void PhysicsWorld::SetBodyFriction(uint32_t bodyId, float friction)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetFriction(JPH::BodyID(bodyId), friction);
}

void PhysicsWorld::SetBodyRestitution(uint32_t bodyId, float restitution)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetRestitution(JPH::BodyID(bodyId), restitution);
}

glm::vec3 PhysicsWorld::GetBodyLinearVelocity(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::vec3(0.0f);

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::Vec3 v = bi.GetLinearVelocity(JPH::BodyID(bodyId));
    return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
}

void PhysicsWorld::SetBodyLinearVelocity(uint32_t bodyId, const glm::vec3 &vel)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetLinearVelocity(JPH::BodyID(bodyId), JPH::Vec3(vel.x, vel.y, vel.z));
}

glm::vec3 PhysicsWorld::GetBodyAngularVelocity(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::vec3(0.0f);

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::Vec3 v = bi.GetAngularVelocity(JPH::BodyID(bodyId));
    return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
}

void PhysicsWorld::SetBodyAngularVelocity(uint32_t bodyId, const glm::vec3 &vel)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetAngularVelocity(JPH::BodyID(bodyId), JPH::Vec3(vel.x, vel.y, vel.z));
}

void PhysicsWorld::AddBodyForce(uint32_t bodyId, const glm::vec3 &force)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.AddForce(JPH::BodyID(bodyId), JPH::Vec3(force.x, force.y, force.z));
}

void PhysicsWorld::AddBodyImpulse(uint32_t bodyId, const glm::vec3 &impulse)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.AddImpulse(JPH::BodyID(bodyId), JPH::Vec3(impulse.x, impulse.y, impulse.z));
}

void PhysicsWorld::AddBodyTorque(uint32_t bodyId, const glm::vec3 &torque)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.AddTorque(JPH::BodyID(bodyId), JPH::Vec3(torque.x, torque.y, torque.z));
}

void PhysicsWorld::AddBodyAngularImpulse(uint32_t bodyId, const glm::vec3 &impulse)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.AddAngularImpulse(JPH::BodyID(bodyId), JPH::Vec3(impulse.x, impulse.y, impulse.z));
}

// ---- Forces at position ----

void PhysicsWorld::AddBodyForceAtPosition(uint32_t bodyId, const glm::vec3 &force, const glm::vec3 &point)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.AddForce(JPH::BodyID(bodyId), JPH::Vec3(force.x, force.y, force.z), JPH::RVec3(point.x, point.y, point.z));
}

void PhysicsWorld::AddBodyImpulseAtPosition(uint32_t bodyId, const glm::vec3 &impulse, const glm::vec3 &point)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.AddImpulse(JPH::BodyID(bodyId), JPH::Vec3(impulse.x, impulse.y, impulse.z),
                  JPH::RVec3(point.x, point.y, point.z));
}

// ---- Constraints / Motion quality ----

void PhysicsWorld::SetBodyAllowedDOFs(uint32_t bodyId, int allowedDOFs, float mass)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (body.IsDynamic()) {
            JPH::MassProperties massProps = body.GetShape()->GetMassProperties();
            massProps.ScaleToMass(mass > 0.001f ? mass : 0.001f);

            // Thin-body inertia stabilization: ensure no principal axis has
            // less than 10 % of the maximum moment of inertia.  This prevents
            // extreme angular accelerations for flat / thin shapes (e.g.
            // sprites, panels) where one dimension is much smaller than the
            // others.
            JPH::Vec3 diag = massProps.mInertia.GetDiagonal3();
            float maxI = std::max({diag.GetX(), diag.GetY(), diag.GetZ()});
            if (maxI > 1e-10f) {
                float minI = maxI * 0.1f;
                massProps.mInertia.SetDiagonal3(
                    JPH::Vec3(std::max(diag.GetX(), minI), std::max(diag.GetY(), minI), std::max(diag.GetZ(), minI)));
            }

            body.GetMotionProperties()->SetMassProperties(static_cast<JPH::EAllowedDOFs>(allowedDOFs), massProps);
        }
    }
}

void PhysicsWorld::SetBodyMotionQuality(uint32_t bodyId, int quality)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::EMotionQuality mq =
        (MapMotionQualityMode(quality) == 1) ? JPH::EMotionQuality::LinearCast : JPH::EMotionQuality::Discrete;
    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetMotionQuality(JPH::BodyID(bodyId), mq);
}

void PhysicsWorld::SetBodyMaxAngularVelocity(uint32_t bodyId, float maxVel)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (!body.IsStatic()) {
            body.GetMotionProperties()->SetMaxAngularVelocity(maxVel);
        }
    }
}

void PhysicsWorld::SetBodyMaxLinearVelocity(uint32_t bodyId, float maxVel)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        JPH::Body &body = lock.GetBody();
        if (!body.IsStatic()) {
            body.GetMotionProperties()->SetMaxLinearVelocity(maxVel);
        }
    }
}

// ---- Kinematic move ----

void PhysicsWorld::MoveBodyKinematic(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot,
                                     float deltaTime)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.MoveKinematic(JPH::BodyID(bodyId), JPH::RVec3(targetPos.x, targetPos.y, targetPos.z),
                     JPH::Quat(targetRot.x, targetRot.y, targetRot.z, targetRot.w), deltaTime);
}

bool PhysicsWorld::IsBodySleeping(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return true;

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    return !bi.IsActive(JPH::BodyID(bodyId));
}

bool PhysicsWorld::IsBodySensor(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return false;

    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    return lock.Succeeded() ? lock.GetBody().IsSensor() : false;
}

void PhysicsWorld::ActivateBody(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.ActivateBody(JPH::BodyID(bodyId));
}

void PhysicsWorld::ActivateBodiesInAABB(const glm::vec3 &min, const glm::vec3 &max)
{
    if (!m_initialized)
        return;

    JPH::AABox box(JPH::Vec3(min.x, min.y, min.z), JPH::Vec3(max.x, max.y, max.z));
    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.ActivateBodiesInAABox(box, JPH::BroadPhaseLayerFilter(), JPH::ObjectLayerFilter());
}

void PhysicsWorld::WakeBodiesTouchingStatic(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    // Read the body's world-space AABB, then RELEASE the lock before
    // calling ActivateBodiesInAABox (which takes its own internal locks).
    // Holding BodyLockRead while calling the locking BodyInterface causes
    // a deadlock on Jolt's striped mutex.
    JPH::AABox bounds;
    {
        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
        if (!lock.Succeeded())
            return;

        const JPH::Body &body = lock.GetBody();
        bounds = body.GetWorldSpaceBounds();
    } // lock released

    // Expand by a small margin so bodies resting exactly on the surface are caught
    bounds.ExpandBy(JPH::Vec3::sReplicate(0.1f));

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.ActivateBodiesInAABox(bounds, JPH::BroadPhaseLayerFilter(), JPH::ObjectLayerFilter());
}

void PhysicsWorld::DeactivateBody(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.DeactivateBody(JPH::BodyID(bodyId));
}

glm::vec3 PhysicsWorld::GetBodyPosition(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::vec3(0.0f);

    // NoLock: safe because this is called from main thread AFTER Step().
    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::RVec3 p = bi.GetPosition(JPH::BodyID(bodyId));
    return glm::vec3(static_cast<float>(p.GetX()), static_cast<float>(p.GetY()), static_cast<float>(p.GetZ()));
}

glm::quat PhysicsWorld::GetBodyRotation(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::quat(1, 0, 0, 0);

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::Quat q = bi.GetRotation(JPH::BodyID(bodyId));
    return glm::normalize(glm::quat(q.GetW(), q.GetX(), q.GetY(), q.GetZ()));
}

glm::vec3 PhysicsWorld::GetBodyCenterOfMassPosition(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::vec3(0.0f);

    const JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterfaceNoLock();
    JPH::RVec3 p = bi.GetCenterOfMassPosition(JPH::BodyID(bodyId));
    return glm::vec3(static_cast<float>(p.GetX()), static_cast<float>(p.GetY()), static_cast<float>(p.GetZ()));
}

glm::mat3 PhysicsWorld::GetBodyWorldSpaceInertiaTensor(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::mat3(1.0f);

    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (!lock.Succeeded())
        return glm::mat3(1.0f);

    const JPH::Body &body = lock.GetBody();
    if (!body.IsDynamic())
        return glm::mat3(1.0f);

    const JPH::Mat44 invInertia = body.GetInverseInertia();
    const JPH::Vec3 c0 = invInertia.GetColumn3(0);
    const JPH::Vec3 c1 = invInertia.GetColumn3(1);
    const JPH::Vec3 c2 = invInertia.GetColumn3(2);

    const glm::mat3 invTensor(glm::vec3(c0.GetX(), c0.GetY(), c0.GetZ()), glm::vec3(c1.GetX(), c1.GetY(), c1.GetZ()),
                              glm::vec3(c2.GetX(), c2.GetY(), c2.GetZ()));

    const float det = glm::determinant(invTensor);
    if (std::abs(det) < 1e-8f)
        return glm::mat3(1.0f);

    return glm::inverse(invTensor);
}

// ============================================================================
// Raycast
// ============================================================================

bool PhysicsWorld::Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance, RaycastHit &outHit,
                           uint32_t layerMask, bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0)
        return false;

    auto hits = RaycastAll(origin, direction, maxDistance, layerMask, queryTriggers);
    if (hits.empty()) {
        return false;
    }

    outHit = hits.front();
    return true;
}

std::vector<RaycastHit> PhysicsWorld::RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance,
                                                 uint32_t layerMask, bool queryTriggers) const
{
    std::vector<RaycastHit> hits;
    if (!m_initialized || layerMask == 0)
        return hits;

    glm::vec3 dir(0.0f);
    if (!NormalizeQueryDirection(direction, maxDistance, dir)) {
        return hits;
    }

    JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                      JPH::Vec3(dir.x * maxDistance, dir.y * maxDistance, dir.z * maxDistance));

    const JPH::NarrowPhaseQuery &npQuery = m_physicsSystem->GetNarrowPhaseQuery();
    JPH::AllHitCollisionCollector<JPH::CastRayCollector> collector;
    LayerMaskObjectFilter objectFilter(layerMask);
    npQuery.CastRay(ray, JPH::RayCastSettings(), collector, JPH::BroadPhaseLayerFilter(), objectFilter);

    if (!collector.HadHit()) {
        return hits;
    }

    collector.Sort();
    hits.reserve(static_cast<size_t>(collector.mHits.size()));

    for (const JPH::RayCastResult &result : collector.mHits) {
        const uint32_t bodyId = result.mBodyID.GetIndexAndSequenceNumber();
        if (!queryTriggers && IsBodySensor(bodyId)) {
            continue;
        }

        RaycastHit hit;
        hit.distance = result.mFraction * maxDistance;
        hit.point = origin + dir * hit.distance;
        hit.bodyId = bodyId;

        hit.collider = ResolveColliderForSubShape(hit.bodyId, result.mSubShapeID2.GetValue());
        if (hit.collider && hit.collider->GetGameObject()) {
            hit.gameObject = hit.collider->GetGameObject();
        }

        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), result.mBodyID);
        if (lock.Succeeded()) {
            JPH::Vec3 normal = lock.GetBody().GetWorldSpaceSurfaceNormal(
                result.mSubShapeID2, JPH::RVec3(hit.point.x, hit.point.y, hit.point.z));
            hit.normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
        }

        hits.push_back(hit);
    }

    return hits;
}

// ============================================================================
// Shared overlap / shape-cast implementations
// ============================================================================

std::vector<Collider *> PhysicsWorld::OverlapShapeImpl(const JPH::Shape &shape, const glm::vec3 &center,
                                                       uint32_t layerMask, bool queryTriggers) const
{
    std::vector<Collider *> results;
    JPH::CollideShapeSettings settings;
    JPH::RMat44 transform = JPH::RMat44::sTranslation(JPH::RVec3(center.x, center.y, center.z));

    const JPH::NarrowPhaseQuery &npQuery = m_physicsSystem->GetNarrowPhaseQuery();
    JPH::AllHitCollisionCollector<JPH::CollideShapeCollector> collector;
    LayerMaskObjectFilter objectFilter(layerMask);
    npQuery.CollideShape(&shape, JPH::Vec3::sReplicate(1.0f), transform, settings,
                         JPH::RVec3(center.x, center.y, center.z), collector, JPH::BroadPhaseLayerFilter(),
                         objectFilter);

    if (!collector.HadHit())
        return results;

    std::unordered_set<Collider *> seen;
    for (const auto &hit : collector.mHits) {
        uint32_t bodyId = hit.mBodyID2.GetIndexAndSequenceNumber();
        if (!queryTriggers && IsBodySensor(bodyId))
            continue;
        Collider *col = ResolveColliderForSubShape(bodyId, hit.mSubShapeID2.GetValue());
        if (col && seen.insert(col).second) {
            results.push_back(col);
        }
    }
    return results;
}

bool PhysicsWorld::ShapeCastImpl(const JPH::Shape &shape, const glm::vec3 &origin, const glm::vec3 &direction,
                                 float maxDistance, RaycastHit &outHit, uint32_t layerMask, bool queryTriggers) const
{
    glm::vec3 dir(0.0f);
    if (!NormalizeQueryDirection(direction, maxDistance, dir))
        return false;

    JPH::RShapeCast shapeCast = JPH::RShapeCast::sFromWorldTransform(
        &shape, JPH::Vec3::sReplicate(1.0f), JPH::RMat44::sTranslation(JPH::RVec3(origin.x, origin.y, origin.z)),
        JPH::Vec3(dir.x * maxDistance, dir.y * maxDistance, dir.z * maxDistance));

    JPH::ShapeCastSettings castSettings;
    JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector> collector;
    LayerMaskObjectFilter objectFilter(layerMask);

    const JPH::NarrowPhaseQuery &npQuery = m_physicsSystem->GetNarrowPhaseQuery();
    npQuery.CastShape(shapeCast, castSettings, JPH::RVec3(origin.x, origin.y, origin.z), collector,
                      JPH::BroadPhaseLayerFilter(), objectFilter);

    if (!collector.HadHit())
        return false;

    const auto &result = collector.mHit;
    uint32_t bodyId = result.mBodyID2.GetIndexAndSequenceNumber();
    if (!queryTriggers && IsBodySensor(bodyId))
        return false;

    outHit.distance = result.mFraction * maxDistance;
    outHit.point = origin + dir * outHit.distance;
    outHit.bodyId = bodyId;
    outHit.normal =
        glm::vec3(-result.mPenetrationAxis.GetX(), -result.mPenetrationAxis.GetY(), -result.mPenetrationAxis.GetZ());
    float nLen = glm::length(outHit.normal);
    if (nLen > kEpsilon)
        outHit.normal /= nLen;

    outHit.collider = ResolveColliderForSubShape(bodyId, result.mSubShapeID2.GetValue());
    if (outHit.collider && outHit.collider->GetGameObject())
        outHit.gameObject = outHit.collider->GetGameObject();

    return true;
}

// ============================================================================
// Overlap queries (Unity: Physics.OverlapSphere / OverlapBox)
// ============================================================================

std::vector<Collider *> PhysicsWorld::OverlapSphere(const glm::vec3 &center, float radius, uint32_t layerMask,
                                                    bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0)
        return {};
    JPH::SphereShape sphere(radius);
    return OverlapShapeImpl(sphere, center, layerMask, queryTriggers);
}

std::vector<Collider *> PhysicsWorld::OverlapBox(const glm::vec3 &center, const glm::vec3 &halfExtents,
                                                 uint32_t layerMask, bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0)
        return {};
    JPH::BoxShape box(JPH::Vec3(halfExtents.x, halfExtents.y, halfExtents.z));
    return OverlapShapeImpl(box, center, layerMask, queryTriggers);
}

// ============================================================================
// Shape cast queries (Unity: Physics.SphereCast / BoxCast)
// ============================================================================

bool PhysicsWorld::SphereCast(const glm::vec3 &origin, float radius, const glm::vec3 &direction, float maxDistance,
                              RaycastHit &outHit, uint32_t layerMask, bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || radius < 0.0f)
        return false;
    JPH::SphereShape sphere(radius);
    return ShapeCastImpl(sphere, origin, direction, maxDistance, outHit, layerMask, queryTriggers);
}

bool PhysicsWorld::BoxCast(const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction,
                           float maxDistance, RaycastHit &outHit, uint32_t layerMask, bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0)
        return false;

    if (halfExtents.x < 0.0f || halfExtents.y < 0.0f || halfExtents.z < 0.0f)
        return false;

    JPH::BoxShape box(JPH::Vec3(halfExtents.x, halfExtents.y, halfExtents.z));
    return ShapeCastImpl(box, center, direction, maxDistance, outHit, layerMask, queryTriggers);
}

// ============================================================================
// Lookup
// ============================================================================

Collider *PhysicsWorld::FindColliderByBodyId(uint32_t bodyId) const
{
    auto it = m_bodyToCollider.find(bodyId);
    return (it != m_bodyToCollider.end()) ? it->second : nullptr;
}

Collider *PhysicsWorld::ResolveColliderForSubShape(uint32_t bodyId, uint32_t subShapeIdValue) const
{
    Collider *fallback = FindColliderByBodyId(bodyId);
    if (!fallback) {
        return nullptr;
    }

    auto *go = fallback->GetGameObject();
    if (!go || subShapeIdValue == 0) {
        return fallback;
    }

    JPH::BodyID joltBodyId(bodyId);
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), joltBodyId);
    if (!lock.Succeeded()) {
        return fallback;
    }

    const JPH::Shape *shape = lock.GetBody().GetShape();
    if (!shape || shape->GetType() != JPH::EShapeType::Compound) {
        return fallback;
    }

    const auto *compound = static_cast<const JPH::CompoundShape *>(shape);
    JPH::SubShapeID subShapeId;
    subShapeId.SetValue(subShapeIdValue);
    if (!compound->IsSubShapeIDValid(subShapeId)) {
        return fallback;
    }

    JPH::SubShapeID remainder;
    uint32_t subShapeIndex = compound->GetSubShapeIndexFromID(subShapeId, remainder);
    uint32_t componentId = compound->GetCompoundUserData(subShapeIndex);
    if (componentId == 0) {
        return fallback;
    }

    auto colliders = go->GetComponents<Collider>();
    for (auto *col : colliders) {
        if (col && static_cast<uint32_t>(col->GetComponentID()) == componentId) {
            return col;
        }
    }

    return fallback;
}

void PhysicsWorld::RebindBodyCollider(uint32_t bodyId, Collider *collider)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF || !collider) {
        return;
    }
    m_bodyToCollider[bodyId] = collider;
}

void PhysicsWorld::EnsureSceneBodiesRegistered(Scene *scene)
{
    if (!m_initialized || !scene)
        return;

    bool anyRegistered = false;
    auto &store = PhysicsECSStore::Instance();

    // Flush deferred body creation queue first (from Collider::Awake).
    auto pendingBodies = store.ConsumePendingBodyCreations();
    for (auto handle : pendingBodies) {
        if (!store.IsValid(handle))
            continue;
        auto &data = store.GetCollider(handle);
        auto *col = data.owner;
        if (!col || !col->IsEnabled() || data.bodyId != 0xFFFFFFFF)
            continue;
        col->RegisterBody();
        if (data.bodyId != 0xFFFFFFFF) {
            col->AddToBroadphase();
            anyRegistered = true;
        }
    }

    // Walk all alive colliders — register any that still lack a body
    // (shouldn't normally happen after the pending queue flush, but
    // guards against edge cases).
    auto handles = store.GetAliveColliderHandles();
    for (auto handle : handles) {
        auto &data = store.GetCollider(handle);
        auto *col = data.owner;
        if (!col || !col->IsEnabled())
            continue;

        if (col->GetBodyId() == 0xFFFFFFFF) {
            col->RegisterBody();
            col->AddToBroadphase();
            anyRegistered = true;
        }

        col->SyncTransformToPhysics();
    }

    // Flush deferred broadphase additions, then rebuild the BVH tree
    // so raycasts can find newly added static bodies.
    auto pending = store.ConsumePendingBroadphaseAdds();
    for (auto &[bodyId, isStatic] : pending) {
        AddBodyToBroadphase(bodyId, isStatic);
        anyRegistered = true;
    }

    if (anyRegistered) {
        m_physicsSystem->OptimizeBroadPhase();
    }
}

void PhysicsWorld::OptimizeBroadPhase()
{
    if (m_initialized && m_physicsSystem) {
        m_physicsSystem->OptimizeBroadPhase();
    }
}

} // namespace infernux
