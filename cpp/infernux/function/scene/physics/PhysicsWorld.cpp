/**
 * @file PhysicsWorld.cpp
 * @brief Jolt Physics integration — singleton world, body management, raycasts.
 */

// Jolt requires specific defines before including its headers
#include <Jolt/Jolt.h>

// Jolt includes (order matters)
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLockMulti.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseQuery.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <limits>

#include "InfernuxJoltJobSystemAdapter.h"
#include "PhysicsContactListener.h"
#include "PhysicsLayers.h"
#include "PhysicsWorld.h"

#include "core/threading/JobSystem.h"

#include "../BoxCollider.h"
#include "../CapsuleCollider.h"
#include "../Collider.h"
#include "../Component.h"
#include "../CylinderCollider.h"
#include "../GameObject.h"
#include "../MeshCollider.h"
#include "../Rigidbody.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../SphereCollider.h"
#include "../Transform.h"
#include <core/config/EngineConfig.h>
#include <core/config/MathConstants.h>
#include <core/log/InxLog.h>
#include <core/threading/JobSystem.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <glm/gtc/constants.hpp>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{

namespace
{

constexpr float kMinQueryDirectionLengthSq = 1e-12f;

static bool IsFinite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

static bool IsFinite(const glm::quat &value)
{
    return std::isfinite(value.w) && std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           glm::dot(value, value) > kMinQueryDirectionLengthSq;
}

static bool HasPositiveFiniteExtents(const glm::vec3 &value)
{
    return IsFinite(value) && value.x > 0.0f && value.y > 0.0f && value.z > 0.0f;
}

static JPH::Quat ToJoltQuat(const glm::quat &rotation)
{
    const glm::quat normalized = glm::normalize(rotation);
    return JPH::Quat(normalized.x, normalized.y, normalized.z, normalized.w);
}

static glm::quat CapsuleOrientation(const glm::vec3 &point0, const glm::vec3 &point1)
{
    const glm::vec3 axis = glm::normalize(point1 - point0);
    constexpr glm::vec3 up(0.0f, 1.0f, 0.0f);
    const float cosine = glm::dot(up, axis);
    if (cosine > 1.0f - 1e-6f)
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    if (cosine < -1.0f + 1e-6f)
        return glm::quat(0.0f, 1.0f, 0.0f, 0.0f);
    const glm::vec3 rotationAxis = glm::cross(up, axis);
    return glm::normalize(glm::quat(1.0f + cosine, rotationAxis.x, rotationAxis.y, rotationAxis.z));
}

static bool NormalizeQueryDirection(const glm::vec3 &direction, float maxDistance, glm::vec3 &outDirection)
{
    if (!IsFinite(direction) || !std::isfinite(maxDistance) || maxDistance <= 0.0f) {
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

static glm::mat3 InvertAllowedAngularSubspace(const glm::mat3 &inverseTensor, uint8_t allowedDofs)
{
    std::array<int, 3> axes{};
    int axisCount = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if ((allowedDofs & (1u << (axis + 3))) != 0)
            axes[axisCount++] = axis;
    }

    glm::mat3 result(0.0f);
    if (axisCount == 0)
        return result;

    float augmented[3][6]{};
    for (int row = 0; row < axisCount; ++row) {
        for (int column = 0; column < axisCount; ++column)
            augmented[row][column] = inverseTensor[axes[column]][axes[row]];
        augmented[row][axisCount + row] = 1.0f;
    }

    for (int pivotColumn = 0; pivotColumn < axisCount; ++pivotColumn) {
        int pivotRow = pivotColumn;
        for (int row = pivotColumn + 1; row < axisCount; ++row) {
            if (std::abs(augmented[row][pivotColumn]) > std::abs(augmented[pivotRow][pivotColumn]))
                pivotRow = row;
        }
        if (std::abs(augmented[pivotRow][pivotColumn]) < 1e-8f)
            throw std::logic_error("dynamic body has singular inertia on an allowed rotation axis");

        if (pivotRow != pivotColumn) {
            for (int column = 0; column < axisCount * 2; ++column)
                std::swap(augmented[pivotRow][column], augmented[pivotColumn][column]);
        }

        const float inversePivot = 1.0f / augmented[pivotColumn][pivotColumn];
        for (int column = 0; column < axisCount * 2; ++column)
            augmented[pivotColumn][column] *= inversePivot;

        for (int row = 0; row < axisCount; ++row) {
            if (row == pivotColumn)
                continue;
            const float factor = augmented[row][pivotColumn];
            for (int column = 0; column < axisCount * 2; ++column)
                augmented[row][column] -= factor * augmented[pivotColumn][column];
        }
    }

    for (int row = 0; row < axisCount; ++row) {
        for (int column = 0; column < axisCount; ++column)
            result[axes[column]][axes[row]] = augmented[row][axisCount + column];
    }
    return result;
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

static JPH::RefConst<JPH::Shape> BuildShapeForColliderSet(GameObject *go, const Collider *exclude,
                                                          size_t *outShapeCount = nullptr)
{
    if (!go) {
        return nullptr;
    }

    std::vector<std::pair<Collider *, JPH::RefConst<JPH::Shape>>> childShapes;
    auto colliders = go->GetComponents<Collider>();
    childShapes.reserve(colliders.size());
    bool complete = true;

    for (auto *col : colliders) {
        if (!col || col == exclude || !col->IsEnabled()) {
            continue;
        }

        JPH::RefConst<JPH::Shape> child(static_cast<const JPH::Shape *>(col->CreateJoltShapeRaw()));
        if (child) {
            childShapes.emplace_back(col, child);
        } else {
            // A pending or rejected child must not silently disappear from
            // a replacement compound. Visit every child to start its cook.
            complete = false;
        }
    }

    if (!complete || childShapes.empty()) {
        if (outShapeCount)
            *outShapeCount = 0;
        return nullptr;
    }

    if (outShapeCount)
        *outShapeCount = childShapes.size();

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
    // Intentionally leaked: Shutdown() is called explicitly from
    // Infernux::Cleanup(); a static destructor would re-run it after
    // dependent singletons may already be gone.
    static PhysicsWorld *instance = new PhysicsWorld();
    return *instance;
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

    if (!JobSystem::IsAvailable()) {
        throw std::logic_error("PhysicsWorld requires an initialized Infernux JobSystem");
    }
    JobSystem::Get().SetDomainConcurrency(JobDomain::Physics, cfg.physicsMaxConcurrency);

    m_tempAllocator = std::make_unique<JPH::TempAllocatorImpl>(cfg.physicsTempAllocatorSize);

    m_jobSystem = std::make_unique<InfernuxJoltJobSystemAdapter>(cfg.physicsMaxJobs, cfg.physicsMaxBarriers,
                                                                 cfg.physicsMaxConcurrency);

    // Layer interfaces
    m_layers = std::make_unique<LayerInterfaces>();

    // Physics system
    m_physicsSystem = std::make_unique<JPH::PhysicsSystem>();
    m_physicsSystem->Init(cfg.physicsMaxBodies, 0, cfg.physicsMaxBodyPairs, cfg.physicsMaxContactConstraints,
                          m_layers->bpInterface, m_layers->objVsBpFilter, m_layers->objPairFilter);
    // Preserve a stream request made before the world was initialized.
    m_physicsSystem->SetRecordAppliedContactImpulses(m_contactImpulseStreamEnabled);

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
    settings.mPenetrationSlop = cfg.physicsPenetrationSlop;
    settings.mSpeculativeContactDistance = cfg.physicsSpeculativeContactDistance;
    settings.mNumVelocitySteps = cfg.physicsVelocitySteps;
    settings.mNumPositionSteps = cfg.physicsPositionSteps;
    settings.mLinearCastMaxPenetration = cfg.physicsLinearCastMaxPenetration;
    settings.mBaumgarte = cfg.physicsBaumgarte;
    settings.mMaxPenetrationDistance = cfg.physicsMaxPenetrationDistance;
    settings.mLinearCastThreshold = cfg.physicsLinearCastThreshold;
    settings.mMinVelocityForRestitution = cfg.physicsMinVelocityForRestitution;
    settings.mTimeBeforeSleep = cfg.physicsTimeBeforeSleep;
    settings.mPointVelocitySleepThreshold = cfg.physicsPointVelocitySleepThreshold;
    m_physicsSystem->SetPhysicsSettings(settings);

    // Gravity
    m_physicsSystem->SetGravity(JPH::Vec3(cfg.physicsGravity.x, cfg.physicsGravity.y, cfg.physicsGravity.z));

    // Install contact listener for collision/trigger callbacks
    m_contactListener = std::make_unique<InxContactListener>();
    m_physicsSystem->SetContactListener(m_contactListener.get());

    // Warm up the broadphase through the same Physics domain used by real
    // simulation steps. The adapter never owns a worker thread.
    {
        auto warmupGroup = JobSystem::Get().CreateTaskGroup(JobDomain::Physics, JobPriority::Critical);
        m_jobSystem->BeginFrame(warmupGroup);
        m_physicsSystem->OptimizeBroadPhase();
        m_jobSystem->EndFrame();
        warmupGroup.Close();
        JobSystem::Get().Wait(warmupGroup);
    }

    m_initialized = true;
}

void PhysicsWorld::Shutdown()
{
    if (!m_initialized)
        return;

    // Constraints must leave the solver before either referenced body dies.
    while (!m_constraints.empty())
        DestroyConstraint(m_constraints.begin()->first);

    // Step 1: drop contact pair tracking before any body dies, so callbacks
    // racing the teardown can't dereference freed Collider* pointers.
    if (m_contactListener)
        m_contactListener->ClearAll();

    // Step 1b: destroy any bodies that still exist. Scenes should already be
    // gone (SceneManager::Shutdown runs first in Infernux::Cleanup), so a
    // non-empty map here indicates a teardown-ordering bug — sweep it anyway
    // so Jolt's PhysicsSystem is destroyed with zero live bodies.
    if (m_physicsSystem && !m_bodyToCollider.empty()) {
        INXLOG_WARN("PhysicsWorld: ", m_bodyToCollider.size(),
                    " bodies still alive at shutdown — destroying them (check teardown order).");
        JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
        for (const auto &[id, collider] : m_bodyToCollider) {
            (void)collider;
            JPH::BodyID bodyId(id);
            if (bi.IsAdded(bodyId))
                bi.RemoveBody(bodyId);
            bi.DestroyBody(bodyId);
        }
    }
    m_bodyToCollider.clear();
    m_poseReadbackBodyIds.clear();
    m_staticContinuousBodyIds.clear();
    m_continuousBodyIds.clear();
    m_kinematicMoveStates.clear();
    m_nextConstraintId = 1;
    m_lastDynamicCCDSplitCount = 0;

    // Step 2: tear down subsystems in dependency order (newest first).
    m_contactListener.reset();
    m_physicsSystem.reset();
    if (m_jobSystem) {
        m_jobSystem->Shutdown();
        m_jobSystem.reset();
    }
    if (JobSystem::IsAvailable()) {
        JobSystem::Get().SetDomainConcurrency(JobDomain::Physics, 0);
    }
    m_tempAllocator.reset();
    m_layers.reset();

    // Step 3: drop Jolt's process-global state. Safe because every owned
    // unique_ptr above has been released, so no Jolt object outlives the
    // factory.
    JPH::UnregisterTypes();
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;

    m_initialized = false;
}

// ============================================================================
// Step
// ============================================================================

float PhysicsWorld::FindEarliestDynamicCCDFraction(float deltaTime) const
{
    if (m_continuousBodyIds.empty())
        return 1.0f;

    struct BodySnapshot
    {
        JPH::BodyID id;
        JPH::ObjectLayer layer;
        JPH::CollisionGroup collisionGroup;
        JPH::RefConst<JPH::Shape> shape;
        JPH::RMat44 centerOfMassTransform;
        JPH::TransformedShape transformedShape;
        JPH::AABox sweptBounds;
        JPH::Vec3 deltaPosition;
        bool continuousDynamic = false;
    };

    JPH::BodyIDVector activeBodyIds;
    m_physicsSystem->GetActiveBodies(JPH::EBodyType::RigidBody, activeBodyIds);

    std::vector<BodySnapshot> bodies;
    bodies.reserve(activeBodyIds.size());
    for (const JPH::BodyID id : activeBodyIds) {
        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), id);
        if (!lock.Succeeded())
            continue;

        const JPH::Body &body = lock.GetBody();
        if (body.IsStatic() || body.IsSensor() || body.GetMotionPropertiesUnchecked() == nullptr)
            continue;

        BodySnapshot snapshot;
        snapshot.id = id;
        snapshot.layer = body.GetObjectLayer();
        snapshot.collisionGroup = body.GetCollisionGroup();
        snapshot.shape = body.GetShape();
        snapshot.centerOfMassTransform = body.GetCenterOfMassTransform();
        snapshot.transformedShape = body.GetTransformedShape();
        snapshot.deltaPosition = deltaTime * body.GetLinearVelocity();
        snapshot.sweptBounds = body.GetWorldSpaceBounds();
        JPH::AABox endBounds = snapshot.sweptBounds;
        endBounds.Translate(snapshot.deltaPosition);
        snapshot.sweptBounds.Encapsulate(endBounds);
        snapshot.continuousDynamic =
            body.IsDynamic() && body.GetMotionProperties()->GetMotionQuality() == JPH::EMotionQuality::LinearCast;
        bodies.push_back(std::move(snapshot));
    }

    JPH::ShapeCastSettings castSettings;
    castSettings.mUseShrunkenShapeAndConvexRadius = true;
    castSettings.mBackFaceModeTriangles = JPH::EBackFaceMode::IgnoreBackFaces;
    castSettings.mBackFaceModeConvex = JPH::EBackFaceMode::IgnoreBackFaces;

    float earliestFraction = 1.0f;
    for (size_t sourceIndex = 0; sourceIndex < bodies.size(); ++sourceIndex) {
        const BodySnapshot &source = bodies[sourceIndex];
        if (!source.continuousDynamic)
            continue;

        for (size_t targetIndex = 0; targetIndex < bodies.size(); ++targetIndex) {
            if (sourceIndex == targetIndex)
                continue;

            const BodySnapshot &target = bodies[targetIndex];
            if (target.continuousDynamic && source.id > target.id)
                continue;
            if (!source.sweptBounds.Overlaps(target.sweptBounds))
                continue;
            if (!m_layers->objPairFilter.ShouldCollide(source.layer, target.layer) ||
                !source.collisionGroup.CanCollide(target.collisionGroup))
                continue;

            const JPH::Vec3 relativeMotion = source.deltaPosition - target.deltaPosition;
            if (relativeMotion.LengthSq() <= 1e-12f)
                continue;

            JPH::RShapeCast cast(source.shape, JPH::Vec3::sOne(), source.centerOfMassTransform, relativeMotion);
            JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector> collector;
            JPH::ShapeFilter shapeFilter;
            shapeFilter.mBodyID2 = target.id;
            target.transformedShape.CastShape(cast, castSettings, source.centerOfMassTransform.GetTranslation(),
                                              collector, shapeFilter);
            if (!collector.HadHit())
                continue;

            const float fraction = collector.mHit.mFraction;
            if (fraction > 1e-4f && fraction < earliestFraction)
                earliestFraction = fraction;
        }
    }

    return earliestFraction;
}

float PhysicsWorld::FindEarliestStaticCCDFraction(float deltaTime) const
{
    if (m_staticContinuousBodyIds.empty())
        return 1.0f;

    JPH::ShapeCastSettings castSettings;
    castSettings.mUseShrunkenShapeAndConvexRadius = true;
    castSettings.mBackFaceModeTriangles = JPH::EBackFaceMode::IgnoreBackFaces;
    castSettings.mBackFaceModeConvex = JPH::EBackFaceMode::IgnoreBackFaces;

    float earliestFraction = 1.0f;
    for (const uint32_t sourceRawId : m_staticContinuousBodyIds) {
        const JPH::BodyID sourceId(sourceRawId);
        JPH::BodyLockRead sourceLock(m_physicsSystem->GetBodyLockInterface(), sourceId);
        if (!sourceLock.Succeeded())
            continue;
        const JPH::Body &source = sourceLock.GetBody();
        if (!source.IsDynamic() || source.IsSensor() || !source.IsActive())
            continue;

        const JPH::Vec3 motion = deltaTime * source.GetLinearVelocity();
        if (motion.LengthSq() <= 1e-12f)
            continue;

        JPH::AABox sweptBounds = source.GetWorldSpaceBounds();
        JPH::AABox endBounds = sweptBounds;
        endBounds.Translate(motion);
        sweptBounds.Encapsulate(endBounds);
        const JPH::RMat44 sourceTransform = source.GetCenterOfMassTransform();
        const JPH::RShapeCast cast(source.GetShape(), JPH::Vec3::sOne(), sourceTransform, motion);

        for (const auto &[targetRawId, collider] : m_bodyToCollider) {
            (void)collider;
            if (targetRawId == sourceRawId)
                continue;
            const JPH::BodyID targetId(targetRawId);
            JPH::BodyLockRead targetLock(m_physicsSystem->GetBodyLockInterface(), targetId);
            if (!targetLock.Succeeded())
                continue;
            const JPH::Body &target = targetLock.GetBody();
            if (!target.IsStatic() || target.IsSensor() || !sweptBounds.Overlaps(target.GetWorldSpaceBounds()))
                continue;
            if (!m_layers->objPairFilter.ShouldCollide(source.GetObjectLayer(), target.GetObjectLayer()) ||
                !source.GetCollisionGroup().CanCollide(target.GetCollisionGroup()))
                continue;

            JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector> collector;
            JPH::ShapeFilter shapeFilter;
            shapeFilter.mBodyID2 = targetId;
            target.GetTransformedShape().CastShape(cast, castSettings, sourceTransform.GetTranslation(), collector,
                                                   shapeFilter);
            if (collector.HadHit() && collector.mHit.mFraction > 1e-4f)
                earliestFraction = std::min(earliestFraction, collector.mHit.mFraction);
        }
    }
    return earliestFraction;
}

void PhysicsWorld::SettleKinematicMoves()
{
    if (m_kinematicMoveStates.empty())
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    for (auto it = m_kinematicMoveStates.begin(); it != m_kinematicMoveStates.end();) {
        KinematicMoveState &state = it->second;
        if (state.movedThisStep) {
            // A target arrived since the previous step — let this step
            // integrate it, then re-evaluate.
            state.movedThisStep = false;
            ++it;
            continue;
        }

        const JPH::BodyID id(it->first);

        // The body arrived at its last target and no new target came in.
        // Without this, the MoveKinematic velocity persists and the body
        // glides away from its Transform forever.
        if (state.idleSteps == 0)
            bi.SetLinearAndAngularVelocity(id, JPH::Vec3::sZero(), JPH::Vec3::sZero());
        ++state.idleSteps;

        if (!state.restoreStatic) {
            // Genuine kinematic Rigidbody — stopping it is all we owe it.
            it = m_kinematicMoveStates.erase(it);
            continue;
        }

        // Temporarily-kinematic static: keep it kinematic for a few idle
        // steps, since a gizmo drag does not deliver a new pose on every
        // fixed step, then restore. SetMotionType(Static) deactivates the
        // body and zeroes its velocity inside Jolt.
        constexpr int kRestoreAfterIdleSteps = 3;
        if (state.idleSteps >= kRestoreAfterIdleSteps) {
            bool restore = true;
            if (auto found = m_bodyToCollider.find(it->first); found != m_bodyToCollider.end() && found->second) {
                const Rigidbody *rb = found->second->GetCachedRigidbody();
                restore = (rb == nullptr || !rb->IsEnabled());
            }
            if (restore)
                bi.SetMotionType(id, JPH::EMotionType::Static, JPH::EActivation::DontActivate);
            it = m_kinematicMoveStates.erase(it);
            continue;
        }
        ++it;
    }
}

void PhysicsWorld::Step(float deltaTime)
{
    m_poseReadbackBodyIds.clear();
    m_contactImpulses.clear();
    m_lastDynamicCCDSplitCount = 0;
    if (!m_initialized)
        return;

    SettleKinematicMoves();

    JPH::BodyIDVector activeBefore;
    m_physicsSystem->GetActiveBodies(JPH::EBodyType::RigidBody, activeBefore);

    if (m_contactListener) {
        uint8_t eventInterestMask = 0;
        std::unordered_set<GameObject *> visitedObjects;
        visitedObjects.reserve(m_bodyToCollider.size());
        for (const auto &[bodyId, collider] : m_bodyToCollider) {
            (void)bodyId;
            GameObject *gameObject = collider ? collider->GetGameObject() : nullptr;
            if (!gameObject || !gameObject->IsActiveInHierarchy() || !visitedObjects.insert(gameObject).second)
                continue;
            for (const auto &component : gameObject->GetAllComponents()) {
                if (!component || !component->IsEnabled() || !component->WantsPhysicsCallbacks())
                    continue;
                if (component->WantsCollisionEnterCallbacks())
                    eventInterestMask |= CollisionEnterInterest;
                if (component->WantsCollisionStayCallbacks())
                    eventInterestMask |= CollisionStayInterest;
                if (component->WantsCollisionExitCallbacks())
                    eventInterestMask |= CollisionExitInterest;
                if (component->WantsTriggerEnterCallbacks())
                    eventInterestMask |= TriggerEnterInterest;
                if (component->WantsTriggerStayCallbacks())
                    eventInterestMask |= TriggerStayInterest;
                if (component->WantsTriggerExitCallbacks())
                    eventInterestMask |= TriggerExitInterest;
            }
        }
        if (m_contactEventStreamEnabled) {
            eventInterestMask |= CollisionEnterInterest | CollisionStayInterest | CollisionExitInterest;
            if (m_contactEventStreamIncludeTriggers)
                eventInterestMask |= TriggerEnterInterest | TriggerStayInterest | TriggerExitInterest;
        }
        m_contactListener->SetEventInterestMask(eventInterestMask);
        m_contactListener->PreStep();
    }

    // Jolt's LinearCast narrow phase handles relative target motion, but its
    // broad-phase cast only spans the source body's own displacement. Two fast
    // dynamic bodies can therefore meet halfway without either source cast
    // reaching the other's starting AABB. Split at the earliest relative TOI
    // so the following Jolt update starts with the pair touching and lets the
    // regular contact solver own the response.
    constexpr int kMaxDynamicCCDSplits = 8;
    constexpr float kTOIPadding = 1e-4f;
    constexpr float kMinStepDuration = 1e-6f;
    auto physicsGroup = JobSystem::Get().CreateTaskGroup(JobDomain::Physics, JobPriority::Critical);
    m_jobSystem->BeginFrame(physicsGroup);
    float remainingTime = deltaTime;
    int dynamicCCDSplits = 0;
    auto appendContactImpulses = [&]() {
        if (!m_contactImpulseStreamEnabled)
            return;
        const auto &solved = m_physicsSystem->GetAppliedContactImpulses();
        m_contactImpulses.reserve(m_contactImpulses.size() + solved.size());
        for (const auto &value : solved) {
            ContactImpulse impulse;
            impulse.bodyIdA = value.body1ID;
            impulse.bodyIdB = value.body2ID;
            impulse.subShapeIdA = value.subShapeID1;
            impulse.subShapeIdB = value.subShapeID2;
            impulse.contactPoint = {value.contactPoint.x, value.contactPoint.y, value.contactPoint.z};
            impulse.contactNormal = {value.normal.x, value.normal.y, value.normal.z};
            impulse.impulse = {value.impulse.x, value.impulse.y, value.impulse.z};
            m_contactImpulses.push_back(impulse);
        }
    };
    try {
        while (remainingTime > kMinStepDuration) {
            const float hitFraction =
                std::min(FindEarliestStaticCCDFraction(remainingTime), FindEarliestDynamicCCDFraction(remainingTime));
            if (hitFraction >= 1.0f || dynamicCCDSplits >= kMaxDynamicCCDSplits) {
                m_physicsSystem->Update(remainingTime, EngineConfig::Get().physicsCollisionSteps, m_tempAllocator.get(),
                                        m_jobSystem.get());
                appendContactImpulses();
                remainingTime = 0.0f;
                break;
            }

            const float segmentFraction = std::clamp(hitFraction + kTOIPadding, kTOIPadding, 0.999f);
            const float segmentTime = remainingTime * segmentFraction;
            m_physicsSystem->Update(segmentTime, 1, m_tempAllocator.get(), m_jobSystem.get());
            appendContactImpulses();
            remainingTime -= segmentTime;
            ++dynamicCCDSplits;
        }
    } catch (...) {
        m_jobSystem->EndFrame();
        physicsGroup.Close();
        JobSystem::Get().Wait(physicsGroup);
        throw;
    }
    m_jobSystem->EndFrame();
    physicsGroup.Close();
    JobSystem::Get().Wait(physicsGroup);
    m_lastDynamicCCDSplitCount = static_cast<size_t>(dynamicCCDSplits);

    JPH::BodyIDVector activeAfter;
    m_physicsSystem->GetActiveBodies(JPH::EBodyType::RigidBody, activeAfter);
    m_poseReadbackBodyIds.reserve(activeBefore.size() + activeAfter.size());
    for (const JPH::BodyID id : activeBefore)
        m_poseReadbackBodyIds.push_back(id.GetIndexAndSequenceNumber());
    for (const JPH::BodyID id : activeAfter)
        m_poseReadbackBodyIds.push_back(id.GetIndexAndSequenceNumber());
    std::sort(m_poseReadbackBodyIds.begin(), m_poseReadbackBodyIds.end());
    m_poseReadbackBodyIds.erase(std::unique(m_poseReadbackBodyIds.begin(), m_poseReadbackBodyIds.end()),
                                m_poseReadbackBodyIds.end());

    // Resolve raw contact events through pair tracking — suppresses spurious
    // Enter/Exit caused by Jolt's body sleep/wake cycles.
    if (m_contactListener) {
        JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
        m_contactListener->ResolveEvents(bi);
    }
}

static void PublishRaycastSubShape(const JPH::Body &body, const JPH::SubShapeID &subShapeId, RaycastHit &outHit)
{
    outHit.subShapeId = subShapeId.GetValue();
    JPH::SubShapeID remainder;
    const JPH::Shape *leaf = body.GetShape()->GetLeafShape(subShapeId, remainder);
    if (leaf && leaf->GetSubType() == JPH::EShapeSubType::Mesh) {
        outHit.triangleIndex = static_cast<const JPH::MeshShape *>(leaf)->GetTriangleUserData(remainder);
    }
}

void PhysicsWorld::SetContactEventStreamEnabled(bool enabled, bool includeTriggers)
{
    m_contactEventStreamEnabled = enabled;
    m_contactEventStreamIncludeTriggers = enabled && includeTriggers;
}

void PhysicsWorld::SetContactImpulseStreamEnabled(bool enabled)
{
    m_contactImpulseStreamEnabled = enabled;
    if (m_physicsSystem)
        m_physicsSystem->SetRecordAppliedContactImpulses(enabled);
    if (!enabled)
        m_contactImpulses.clear();
}

const std::vector<ContactEvent> &PhysicsWorld::GetContactEvents() const
{
    static const std::vector<ContactEvent> empty;
    return m_contactListener ? m_contactListener->GetEvents() : empty;
}

// ============================================================================
// Contact event dispatch (Unity-style collision/trigger callbacks)
// ============================================================================

size_t PhysicsWorld::DispatchContactEvents()
{
    if (!m_contactListener)
        return 0;

    const auto &events = m_contactListener->GetEvents();
    if (events.empty())
        return 0;
    const size_t eventCount = events.size();

    std::vector<Component *> receiversA;
    std::vector<Component *> receiversB;
    receiversA.reserve(8);
    receiversB.reserve(8);

    std::unordered_map<GameObject *, uint8_t> callbackMasks;
    callbackMasks.reserve(std::min(events.size() * 2u, m_bodyToCollider.size()));

    auto callbackMask = [&](GameObject *go) {
        const auto found = callbackMasks.find(go);
        if (found != callbackMasks.end())
            return found->second;

        uint8_t mask = 0;
        for (const auto &comp : go->GetAllComponents()) {
            if (!comp || !comp->IsEnabled() || !comp->WantsPhysicsCallbacks())
                continue;
            if (comp->WantsCollisionEnterCallbacks())
                mask |= CollisionEnterInterest;
            if (comp->WantsCollisionStayCallbacks())
                mask |= CollisionStayInterest;
            if (comp->WantsCollisionExitCallbacks())
                mask |= CollisionExitInterest;
            if (comp->WantsTriggerEnterCallbacks())
                mask |= TriggerEnterInterest;
            if (comp->WantsTriggerStayCallbacks())
                mask |= TriggerStayInterest;
            if (comp->WantsTriggerExitCallbacks())
                mask |= TriggerExitInterest;
        }
        callbackMasks.emplace(go, mask);
        return mask;
    };

    auto wantsEvent = [](const Component &comp, ContactEventType type) {
        switch (type) {
        case ContactEventType::CollisionEnter:
            return comp.WantsCollisionEnterCallbacks();
        case ContactEventType::CollisionStay:
            return comp.WantsCollisionStayCallbacks();
        case ContactEventType::CollisionExit:
            return comp.WantsCollisionExitCallbacks();
        case ContactEventType::TriggerEnter:
            return comp.WantsTriggerEnterCallbacks();
        case ContactEventType::TriggerStay:
            return comp.WantsTriggerStayCallbacks();
        case ContactEventType::TriggerExit:
            return comp.WantsTriggerExitCallbacks();
        }
        return false;
    };

    for (const auto &evt : events) {
        Collider *colA = ResolveColliderForSubShape(evt.bodyIdA, evt.subShapeIdA);
        Collider *colB = ResolveColliderForSubShape(evt.bodyIdB, evt.subShapeIdB);
        if (!colA || !colB)
            continue;

        GameObject *goA = colA->GetGameObject();
        GameObject *goB = colB->GetGameObject();
        if (!goA || !goB)
            continue;

        ContactEventType type = evt.type;
        const bool triggerPair = colA->IsTrigger() || colB->IsTrigger();
        if (triggerPair) {
            if (type == ContactEventType::CollisionEnter)
                type = ContactEventType::TriggerEnter;
            else if (type == ContactEventType::CollisionStay)
                type = ContactEventType::TriggerStay;
            else if (type == ContactEventType::CollisionExit)
                type = ContactEventType::TriggerExit;
        } else {
            if (type == ContactEventType::TriggerEnter)
                type = ContactEventType::CollisionEnter;
            else if (type == ContactEventType::TriggerStay)
                type = ContactEventType::CollisionStay;
            else if (type == ContactEventType::TriggerExit)
                type = ContactEventType::CollisionExit;
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

        const uint8_t requiredBit = ContactEventInterestBit(type);
        const bool wantsA = (callbackMask(goA) & requiredBit) != 0;
        const bool wantsB = (callbackMask(goB) & requiredBit) != 0;
        if (!wantsA && !wantsB)
            continue;

        receiversA.clear();
        receiversB.clear();

        if (wantsA) {
            for (const auto &comp : goA->GetAllComponents()) {
                if (!comp || !comp->IsEnabled() || !wantsEvent(*comp, type))
                    continue;
                receiversA.push_back(comp.get());
            }
        }

        if (wantsB) {
            for (const auto &comp : goB->GetAllComponents()) {
                if (!comp || !comp->IsEnabled() || !wantsEvent(*comp, type))
                    continue;
                receiversB.push_back(comp.get());
            }
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

        CollisionInfo infoForB;
        infoForB.collider = colA;
        infoForB.gameObject = goA;
        infoForB.contactPoint = evt.contactPoint;
        infoForB.contactNormal = -evt.contactNormal; // flip for B
        infoForB.relativeVelocity = -evt.relativeVelocity;

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
    return eventCount;
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

    size_t shapeCount = 0;
    auto shape = BuildShapeForColliderSet(go, nullptr, &shapeCount);
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
    settings.mUserData = reinterpret_cast<uint64_t>(collider);
    settings.mUseManifoldReduction = shapeCount <= 1;

    // Body-level fallback. Contact callbacks replace these values using the
    // actual compound subshapes and their material combine modes.
    settings.mFriction = collider->GetFriction();
    settings.mRestitution = collider->GetBounciness();

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
    m_queryGeneration.fetch_add(1, std::memory_order_release);
    return id;
}

void PhysicsWorld::DestroyBody(Collider *collider)
{
    if (!m_initialized || !collider)
        return;

    uint32_t id = collider->GetBodyId();
    if (id == 0xFFFFFFFF)
        return;

    std::vector<uint64_t> attachedConstraints;
    for (const auto &[constraintId, record] : m_constraints) {
        if (record.bodyIdA == id || record.bodyIdB == id)
            attachedConstraints.push_back(constraintId);
    }
    for (uint64_t constraintId : attachedConstraints)
        DestroyConstraint(constraintId);

    // Contract: caller (Collider::UnregisterBody) must have already removed
    // this body from the broadphase. See PhysicsWorld.h::DestroyBody for the
    // full ordering invariant.
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.DestroyBody(JPH::BodyID(id));

    m_bodyToCollider.erase(id);
    m_staticContinuousBodyIds.erase(id);
    m_continuousBodyIds.erase(id);
    m_kinematicMoveStates.erase(id);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
    if (m_contactListener)
        m_contactListener->RemoveIgnoredPairsForBody(id);
}

void PhysicsWorld::SetBodyPosition(uint32_t bodyId, const glm::vec3 &pos, const glm::quat &rot)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    // Transform sync runs serially on the main thread before PhysicsSystem::Update.
    // Avoid taking Jolt's striped body lock once per moved static/kinematic body.
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterfaceNoLock();
    bodyInterface.SetPositionAndRotation(JPH::BodyID(bodyId), JPH::RVec3(pos.x, pos.y, pos.z),
                                         JPH::Quat(rot.x, rot.y, rot.z, rot.w), JPH::EActivation::DontActivate);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::SetBodyPositionsBatch(const std::vector<PhysicsBodyPoseUpdate> &updates)
{
    if (!m_initialized || updates.empty())
        return;

    static thread_local JPH::BodyIDVector bodyIds;
    static thread_local std::vector<JPH::RVec3> positions;
    static thread_local std::vector<JPH::Quat> rotations;
    bodyIds.clear();
    positions.clear();
    rotations.clear();
    bodyIds.reserve(updates.size());
    positions.reserve(updates.size());
    rotations.reserve(updates.size());
    for (const auto &update : updates) {
        if (update.bodyId == 0xFFFFFFFF)
            continue;
        bodyIds.emplace_back(update.bodyId);
        positions.emplace_back(update.position.x, update.position.y, update.position.z);
        rotations.emplace_back(update.rotation.x, update.rotation.y, update.rotation.z, update.rotation.w);
    }
    if (bodyIds.empty())
        return;

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterfaceNoLock();
    // Jolt 5.6 removed the legacy batch setter. Keep the update path explicit
    // and use the no-lock interface; the caller already owns the serial sync
    // phase, so this preserves the same ordering without a compatibility shim.
    for (size_t i = 0; i < bodyIds.size(); ++i)
        bodyInterface.SetPositionAndRotationWhenChanged(bodyIds[i], positions[i], rotations[i],
                                                        JPH::EActivation::DontActivate);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::UpdateBodyShape(Collider *collider, const Collider *exclude)
{
    if (!m_initialized || !collider)
        return;

    uint32_t id = collider->GetBodyId();
    if (id == 0xFFFFFFFF)
        return;

    size_t shapeCount = 0;
    auto newShape = BuildShapeForColliderSet(collider->GetGameObject(), exclude, &shapeCount);
    if (!newShape)
        return;

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.SetUseManifoldReduction(JPH::BodyID(id), shapeCount <= 1);
    bodyInterface.SetShape(JPH::BodyID(id), newShape, true, JPH::EActivation::Activate);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::SetBodyIsSensor(uint32_t bodyId, bool isSensor)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (lock.Succeeded()) {
        lock.GetBody().SetIsSensor(isSensor);
        m_queryGeneration.fetch_add(1, std::memory_order_release);
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
    m_queryGeneration.fetch_add(1, std::memory_order_release);
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
    if (!staticIds.empty() || !dynamicIds.empty())
        m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::RemoveBodyFromBroadphase(uint32_t bodyId)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.RemoveBody(JPH::BodyID(bodyId));
    m_queryGeneration.fetch_add(1, std::memory_order_release);
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
    if (mt != JPH::EMotionType::Dynamic) {
        m_staticContinuousBodyIds.erase(bodyId);
        m_continuousBodyIds.erase(bodyId);
    }
    // An explicit motion-type change supersedes any pending drag tracking —
    // never restore this body to Static behind the caller's back.
    m_kinematicMoveStates.erase(bodyId);
    m_queryGeneration.fetch_add(1, std::memory_order_release);
}

void PhysicsWorld::SetBodyGameLayer(uint32_t bodyId, int gameLayer)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    const JPH::EMotionType motionType = bi.GetMotionType(JPH::BodyID(bodyId));
    const bool moving = motionType != JPH::EMotionType::Static;
    bi.SetObjectLayer(JPH::BodyID(bodyId), PhysicsObjectLayers::Encode(gameLayer, moving));
    m_queryGeneration.fetch_add(1, std::memory_order_release);
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

    // An explicit velocity supersedes a pending MoveKinematic arrival stop
    // (e.g. constant-velocity kinematic platforms). Drag-converted statics
    // stay tracked so they are still restored to Static.
    if (auto it = m_kinematicMoveStates.find(bodyId); it != m_kinematicMoveStates.end() && !it->second.restoreStatic)
        m_kinematicMoveStates.erase(it);
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

    if (auto it = m_kinematicMoveStates.find(bodyId); it != m_kinematicMoveStates.end() && !it->second.restoreStatic)
        m_kinematicMoveStates.erase(it);
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

    const int mappedQuality = MapMotionQualityMode(quality);
    // Jolt LinearCast also catches dynamic targets. Keep it only for the
    // ContinuousDynamic contract; plain Continuous is swept against statics by
    // FindEarliestStaticCCDFraction and otherwise remains discrete.
    JPH::EMotionQuality mq =
        (mappedQuality == 1 && quality == 2) ? JPH::EMotionQuality::LinearCast : JPH::EMotionQuality::Discrete;
    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    bi.SetMotionQuality(JPH::BodyID(bodyId), mq);
    if (quality == 1)
        m_staticContinuousBodyIds.insert(bodyId);
    else
        m_staticContinuousBodyIds.erase(bodyId);
    if (quality == 2)
        m_continuousBodyIds.insert(bodyId);
    else
        m_continuousBodyIds.erase(bodyId);
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
                                     float deltaTime, float maxSpeed)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return;

    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();
    const JPH::BodyID id(bodyId);
    const JPH::RVec3 target(targetPos.x, targetPos.y, targetPos.z);
    const JPH::Quat targetQuat = ToJoltQuat(targetRot);

    if (maxSpeed > 0.0f && deltaTime > 0.0f) {
        // Cap the contact velocity the move can impart (matches PhysX's
        // default maxDepenetrationVelocity). Excess displacement — a scripted
        // long-range teleport, or the first sync after un-pausing — is applied
        // as a teleport so the body never lags behind its Transform, and
        // trigger/contact detection still sees the body at its new location
        // in the very next step.
        const JPH::Vec3 delta(target - bi.GetPosition(id));
        const float maxStepDistance = maxSpeed * deltaTime;
        const float distance = delta.Length();
        if (distance > maxStepDistance) {
            const JPH::RVec3 nearTarget = target - JPH::RVec3(delta * (maxStepDistance / distance));
            bi.SetPositionAndRotation(id, nearTarget, targetQuat, JPH::EActivation::Activate);
        }
    }

    bi.MoveKinematic(id, target, targetQuat, deltaTime);

    // Track the move so SettleKinematicMoves() can zero the velocity once the
    // body has arrived — MoveKinematic velocity persists in Jolt otherwise.
    auto &state = m_kinematicMoveStates[bodyId];
    state.movedThisStep = true;
    state.idleSteps = 0;
}

void PhysicsWorld::MoveStaticBodyWithVelocity(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot,
                                              float deltaTime)
{
    if (!m_initialized || bodyId == 0xFFFFFFFF || deltaTime <= 0.0f)
        return;

    const JPH::BodyID id(bodyId);
    JPH::BodyInterface &bi = m_physicsSystem->GetBodyInterface();

    const JPH::EMotionType motionType = bi.GetMotionType(id);
    if (motionType == JPH::EMotionType::Dynamic) {
        // A Rigidbody took ownership of this body mid-drag — plain teleport.
        SetBodyPosition(bodyId, targetPos, targetRot);
        return;
    }

    if (motionType == JPH::EMotionType::Static) {
        // The object layer is intentionally left unchanged (still non-moving):
        // statics never collide with each other, and dynamic bodies pair with
        // the non-moving layer, which is all a drag push needs.
        bi.SetMotionType(id, JPH::EMotionType::Kinematic, JPH::EActivation::Activate);
    }

    MoveBodyKinematic(bodyId, targetPos, targetRot, deltaTime, kMaxTransformDriveSpeed);
    m_kinematicMoveStates[bodyId].restoreStatic = true;
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

PhysicsBodyMotionState PhysicsWorld::GetBodyMotionState(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        throw std::logic_error("Motion state requires an active physics body");
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (!lock.Succeeded() || !lock.GetBody().IsInBroadPhase())
        throw std::logic_error("Motion state requires an active physics body");
    const auto &body = lock.GetBody();
    const auto vector = [](const auto &v) {
        return glm::vec3(static_cast<float>(v.GetX()), static_cast<float>(v.GetY()), static_cast<float>(v.GetZ()));
    };
    PhysicsBodyMotionState state;
    state.position = vector(body.GetPosition());
    const auto rotation = body.GetRotation();
    state.rotation = glm::quat(rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ());
    state.centerOfMass = vector(body.GetCenterOfMassPosition());
    state.linearVelocity = vector(body.GetLinearVelocity());
    state.angularVelocity = vector(body.GetAngularVelocity());
    if (body.IsDynamic()) {
        const auto *motion = body.GetMotionProperties();
        state.inverseMass = vector(motion->LockTranslation(JPH::Vec3::sReplicate(motion->GetInverseMass())));
        const auto inertia = body.GetInverseInertia();
        for (int column = 0; column < 3; ++column)
            state.inverseInertia[column] = vector(inertia.GetColumn3(column));
    }
    return state;
}

uint64_t PhysicsWorld::CreateHingeConstraint(uint32_t bodyIdA, uint32_t bodyIdB, const glm::vec3 &worldAnchor,
                                             const glm::vec3 &worldAxis, bool useLimits, float minimumAngle,
                                             float maximumAngle, bool enableCollision)
{
    if (!m_initialized || !m_physicsSystem)
        throw std::logic_error("hinge creation requires an initialized physics world");
    if (bodyIdA == 0xFFFFFFFF)
        throw std::invalid_argument("hinge body A is unavailable");
    if (bodyIdA == bodyIdB)
        throw std::invalid_argument("hinge cannot connect a body to itself");
    if (!IsFinite(worldAnchor) || !IsFinite(worldAxis) || glm::dot(worldAxis, worldAxis) <= 1e-12f)
        throw std::invalid_argument("hinge anchor and axis must be finite and the axis must be non-zero");
    if (!std::isfinite(minimumAngle) || !std::isfinite(maximumAngle) || minimumAngle > 0.0f || maximumAngle < 0.0f ||
        minimumAngle > maximumAngle || minimumAngle < -glm::pi<float>() || maximumAngle > glm::pi<float>())
        throw std::invalid_argument("hinge limits must be ordered radians within [-pi, pi] and span zero");

    const glm::vec3 axis = glm::normalize(worldAxis);
    const glm::vec3 reference = std::abs(axis.y) < 0.9f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 normal = glm::normalize(glm::cross(axis, reference));

    JPH::HingeConstraintSettings settings;
    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
    settings.mPoint1 = settings.mPoint2 = JPH::RVec3(worldAnchor.x, worldAnchor.y, worldAnchor.z);
    settings.mHingeAxis1 = settings.mHingeAxis2 = JPH::Vec3(axis.x, axis.y, axis.z);
    settings.mNormalAxis1 = settings.mNormalAxis2 = JPH::Vec3(normal.x, normal.y, normal.z);
    settings.mLimitsMin = useLimits ? minimumAngle : -glm::pi<float>();
    settings.mLimitsMax = useLimits ? maximumAngle : glm::pi<float>();

    JPH::BodyID ids[2] = {JPH::BodyID(bodyIdA), JPH::BodyID(bodyIdB)};
    JPH::Constraint *constraint = nullptr;
    if (bodyIdB == 0xFFFFFFFF) {
        JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), ids[0]);
        if (!lock.Succeeded())
            throw std::logic_error("hinge body A no longer exists");
        constraint = settings.Create(JPH::Body::sFixedToWorld, lock.GetBody());
    } else {
        JPH::BodyLockMultiWrite lock(m_physicsSystem->GetBodyLockInterface(), ids, 2);
        JPH::Body *bodyA = lock.GetBody(0);
        JPH::Body *bodyB = lock.GetBody(1);
        if (!bodyA || !bodyB)
            throw std::logic_error("hinge connected body no longer exists");
        constraint = settings.Create(*bodyB, *bodyA);
    }
    if (!constraint)
        throw std::runtime_error("Jolt failed to create hinge constraint");

    m_physicsSystem->AddConstraint(constraint);
    const uint64_t constraintId = m_nextConstraintId++;
    const bool ignoresCollision = bodyIdB != 0xFFFFFFFF && !enableCollision;
    m_constraints.emplace(constraintId,
                          ConstraintRecord{constraint, bodyIdA, bodyIdB, ignoresCollision, ConstraintKind::Hinge});
    if (ignoresCollision)
        SetConstraintPairSuppressed(bodyIdA, bodyIdB, true);
    return constraintId;
}

uint64_t PhysicsWorld::CreateSliderConstraint(uint32_t bodyIdA, uint32_t bodyIdB, const glm::vec3 &worldAnchor,
                                              const glm::vec3 &worldAxis, bool useLimits, float minimumDistance,
                                              float maximumDistance, bool enableCollision)
{
    if (!m_initialized || !m_physicsSystem)
        throw std::logic_error("slider creation requires an initialized physics world");
    if (bodyIdA == 0xFFFFFFFF)
        throw std::invalid_argument("slider body A is unavailable");
    if (bodyIdA == bodyIdB)
        throw std::invalid_argument("slider cannot connect a body to itself");
    if (!IsFinite(worldAnchor) || !IsFinite(worldAxis) || glm::dot(worldAxis, worldAxis) <= 1e-12f)
        throw std::invalid_argument("slider anchor and axis must be finite and the axis must be non-zero");
    if (!std::isfinite(minimumDistance) || !std::isfinite(maximumDistance) || minimumDistance > 0.0f ||
        maximumDistance < 0.0f || minimumDistance > maximumDistance)
        throw std::invalid_argument("slider limits must be ordered metres and span zero");

    const glm::vec3 axis = glm::normalize(worldAxis);
    JPH::SliderConstraintSettings settings;
    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
    settings.mPoint1 = settings.mPoint2 = JPH::RVec3(worldAnchor.x, worldAnchor.y, worldAnchor.z);
    settings.SetSliderAxis(JPH::Vec3(axis.x, axis.y, axis.z));
    settings.mLimitsMin = useLimits ? minimumDistance : -FLT_MAX;
    settings.mLimitsMax = useLimits ? maximumDistance : FLT_MAX;

    JPH::BodyID ids[2] = {JPH::BodyID(bodyIdA), JPH::BodyID(bodyIdB)};
    JPH::Constraint *constraint = nullptr;
    if (bodyIdB == 0xFFFFFFFF) {
        JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), ids[0]);
        if (!lock.Succeeded())
            throw std::logic_error("slider body A no longer exists");
        constraint = settings.Create(JPH::Body::sFixedToWorld, lock.GetBody());
    } else {
        JPH::BodyLockMultiWrite lock(m_physicsSystem->GetBodyLockInterface(), ids, 2);
        JPH::Body *bodyA = lock.GetBody(0);
        JPH::Body *bodyB = lock.GetBody(1);
        if (!bodyA || !bodyB)
            throw std::logic_error("slider connected body no longer exists");
        constraint = settings.Create(*bodyB, *bodyA);
    }
    if (!constraint)
        throw std::runtime_error("Jolt failed to create slider constraint");

    m_physicsSystem->AddConstraint(constraint);
    const uint64_t constraintId = m_nextConstraintId++;
    const bool ignoresCollision = bodyIdB != 0xFFFFFFFF && !enableCollision;
    m_constraints.emplace(constraintId,
                          ConstraintRecord{constraint, bodyIdA, bodyIdB, ignoresCollision, ConstraintKind::Slider});
    if (ignoresCollision)
        SetConstraintPairSuppressed(bodyIdA, bodyIdB, true);
    return constraintId;
}

void PhysicsWorld::DestroyConstraint(uint64_t constraintId)
{
    auto it = m_constraints.find(constraintId);
    if (it == m_constraints.end())
        return;
    ConstraintRecord record = it->second;
    m_constraints.erase(it);
    if (m_physicsSystem && record.constraint)
        m_physicsSystem->RemoveConstraint(record.constraint);
    if (record.ignoresCollision)
        SetConstraintPairSuppressed(record.bodyIdA, record.bodyIdB, false);
}

float PhysicsWorld::GetHingeConstraintAngle(uint64_t constraintId) const
{
    auto it = m_constraints.find(constraintId);
    if (it == m_constraints.end())
        throw std::logic_error("hinge constraint is not active");
    if (it->second.kind != ConstraintKind::Hinge)
        throw std::logic_error("constraint is not a hinge");
    // Avoid cross-DLL RTTI: the record already carries the authoritative kind.
    return static_cast<JPH::HingeConstraint *>(it->second.constraint)->GetCurrentAngle();
}

float PhysicsWorld::GetSliderConstraintPosition(uint64_t constraintId) const
{
    auto it = m_constraints.find(constraintId);
    if (it == m_constraints.end())
        throw std::logic_error("slider constraint is not active");
    if (it->second.kind != ConstraintKind::Slider)
        throw std::logic_error("constraint is not a slider");
    return static_cast<JPH::SliderConstraint *>(it->second.constraint)->GetCurrentPosition();
}

void PhysicsWorld::SetConstraintPairSuppressed(uint32_t bodyIdA, uint32_t bodyIdB, bool suppressed)
{
    if (!m_contactListener || !m_physicsSystem || bodyIdA == 0xFFFFFFFF || bodyIdB == 0xFFFFFFFF)
        return;
    m_contactListener->SetBodyPairIgnored(bodyIdA, bodyIdB, suppressed);
    m_contactListener->InvalidatePairsForBody(bodyIdA);
    m_contactListener->InvalidatePairsForBody(bodyIdB);
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.InvalidateContactCache(JPH::BodyID(bodyIdA));
    bodyInterface.InvalidateContactCache(JPH::BodyID(bodyIdB));
    bodyInterface.ActivateBody(JPH::BodyID(bodyIdA));
    bodyInterface.ActivateBody(JPH::BodyID(bodyIdB));
}

void PhysicsWorld::SetColliderPairIgnored(Collider *colliderA, Collider *colliderB, bool ignored)
{
    if (!m_contactListener || !m_physicsSystem || !colliderA || !colliderB)
        throw std::logic_error("collision ignore requires an initialized physics world and two Colliders");
    m_contactListener->SetColliderPairIgnored(colliderA->GetComponentID(), colliderB->GetComponentID(), ignored);

    const uint32_t bodyIdA = colliderA->GetBodyId();
    const uint32_t bodyIdB = colliderB->GetBodyId();
    if (bodyIdA == 0xFFFFFFFF || bodyIdB == 0xFFFFFFFF || bodyIdA == bodyIdB)
        return;
    m_contactListener->InvalidatePairsForBody(bodyIdA);
    m_contactListener->InvalidatePairsForBody(bodyIdB);
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.InvalidateContactCache(JPH::BodyID(bodyIdA));
    bodyInterface.InvalidateContactCache(JPH::BodyID(bodyIdB));
    bodyInterface.ActivateBody(JPH::BodyID(bodyIdA));
    bodyInterface.ActivateBody(JPH::BodyID(bodyIdB));
}

bool PhysicsWorld::GetColliderPairIgnored(const Collider *colliderA, const Collider *colliderB) const
{
    if (!m_contactListener || !colliderA || !colliderB)
        return false;
    return m_contactListener->GetColliderPairIgnored(colliderA->GetComponentID(), colliderB->GetComponentID());
}

void PhysicsWorld::RemoveIgnoredPairsForCollider(const Collider *collider)
{
    if (m_contactListener && collider)
        m_contactListener->RemoveIgnoredPairsForCollider(collider->GetComponentID());
}

std::optional<PhysicsPenetrationResult> PhysicsWorld::ComputePenetration(const Collider &a, const glm::vec3 &positionA,
                                                                         const glm::quat &rotationA, const Collider &b,
                                                                         const glm::vec3 &positionB,
                                                                         const glm::quat &rotationB) const
{
    if (!m_initialized)
        throw std::logic_error("compute_penetration requires initialized physics");
    const auto shapeForQuery = [](const Collider &collider) -> JPH::RefConst<JPH::Shape> {
        // Primitive construction is read-only. A convex MeshCollider is also
        // valid once its cooked shape is published; do not start an async cook
        // from a query and do not silently approximate a non-convex mesh.
        const auto *mesh = dynamic_cast<const MeshCollider *>(&collider);
        if (!dynamic_cast<const BoxCollider *>(&collider) && !dynamic_cast<const SphereCollider *>(&collider) &&
            !dynamic_cast<const CapsuleCollider *>(&collider) && !dynamic_cast<const CylinderCollider *>(&collider) &&
            !mesh)
            throw std::invalid_argument(
                "compute_penetration supports Box, Sphere, Capsule, Cylinder and convex Mesh colliders");
        if (mesh && !mesh->IsConvex())
            throw std::invalid_argument("compute_penetration requires MeshCollider.convex = true");
        const auto *rawShape = static_cast<const JPH::Shape *>(collider.CreateJoltShapeRaw());
        if (!rawShape) {
            const std::string detail = mesh && !mesh->GetShapeError().empty() ? ": " + mesh->GetShapeError() : "";
            throw std::logic_error("compute_penetration collider geometry is not ready" + detail);
        }
        return rawShape;
    };
    const auto shapeA = shapeForQuery(a), shapeB = shapeForQuery(b);
    // Work relative to A's origin, then restore world coordinates in the result.
    const auto delta = positionB - positionA;
    const auto transformA =
        JPH::Mat44::sRotation(ToJoltQuat(rotationA)) * JPH::Mat44::sTranslation(shapeA->GetCenterOfMass());
    const auto transformB =
        JPH::Mat44::sRotationTranslation(ToJoltQuat(rotationB), JPH::Vec3(delta.x, delta.y, delta.z)) *
        JPH::Mat44::sTranslation(shapeB->GetCenterOfMass());
    JPH::ClosestHitCollisionCollector<JPH::CollideShapeCollector> collector;
    JPH::CollisionDispatch::sCollideShapeVsShape(shapeA, shapeB, JPH::Vec3::sReplicate(1), JPH::Vec3::sReplicate(1),
                                                 transformA, transformB, JPH::SubShapeIDCreator(),
                                                 JPH::SubShapeIDCreator(), JPH::CollideShapeSettings(), collector);
    if (!collector.HadHit() || collector.mHit.mPenetrationDepth <= 0)
        return std::nullopt;
    const auto &hit = collector.mHit;
    const auto normal = -hit.mPenetrationAxis.Normalized();
    PhysicsPenetrationResult result;
    result.direction = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
    result.distance = hit.mPenetrationDepth;
    result.pointA =
        positionA + glm::vec3(hit.mContactPointOn1.GetX(), hit.mContactPointOn1.GetY(), hit.mContactPointOn1.GetZ());
    result.pointB =
        positionA + glm::vec3(hit.mContactPointOn2.GetX(), hit.mContactPointOn2.GetY(), hit.mContactPointOn2.GetZ());
    return result;
}

glm::mat3 PhysicsWorld::GetBodyWorldSpaceInertiaTensor(uint32_t bodyId) const
{
    if (!m_initialized || bodyId == 0xFFFFFFFF)
        return glm::mat3(0.0f);

    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
    if (!lock.Succeeded())
        return glm::mat3(0.0f);

    const JPH::Body &body = lock.GetBody();
    if (!body.IsDynamic())
        return glm::mat3(0.0f);

    const JPH::Mat44 invInertia = body.GetInverseInertia();
    const JPH::Vec3 c0 = invInertia.GetColumn3(0);
    const JPH::Vec3 c1 = invInertia.GetColumn3(1);
    const JPH::Vec3 c2 = invInertia.GetColumn3(2);

    const glm::mat3 invTensor(glm::vec3(c0.GetX(), c0.GetY(), c0.GetZ()), glm::vec3(c1.GetX(), c1.GetY(), c1.GetZ()),
                              glm::vec3(c2.GetX(), c2.GetY(), c2.GetZ()));

    const auto allowedDofs = static_cast<uint8_t>(body.GetMotionProperties()->GetAllowedDOFs());
    return InvertAllowedAngularSubspace(invTensor, allowedDofs);
}

// ============================================================================
// Raycast
// ============================================================================

bool PhysicsWorld::Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance, RaycastHit &outHit,
                           uint32_t layerMask, bool queryTriggers) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    return RaycastCurrent(origin, direction, maxDistance, outHit, layerMask, queryTriggers);
}

void PhysicsWorld::RaycastBatch(const float *originsXYZ, const float *directionsXYZ, size_t count, float maxDistance,
                                RaycastHit *outHits, uint8_t *outHitMask, uint32_t layerMask, bool queryTriggers,
                                uint64_t *outQueryGeneration) const
{
    if ((count != 0 && (!originsXYZ || !directionsXYZ || !outHits || !outHitMask)))
        throw std::invalid_argument("raycast batch requires non-null storage");

    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    const uint64_t queryGeneration = GetQueryGeneration();
    if (outQueryGeneration)
        *outQueryGeneration = queryGeneration;
    // The physics snapshot is immutable for the duration of this call.  Keep
    // one native query boundary, then fan out large batches through the
    // engine JobSystem instead of making the caller serialize thousands of
    // independent narrow-phase casts on the owner thread.  Small batches stay
    // inline: scheduling overhead is larger than the query itself there.
    const auto castOne = [&](uint32_t index) {
        const size_t offset = index * 3;
        const glm::vec3 origin(originsXYZ[offset], originsXYZ[offset + 1], originsXYZ[offset + 2]);
        const glm::vec3 direction(directionsXYZ[offset], directionsXYZ[offset + 1], directionsXYZ[offset + 2]);
        outHits[index] = RaycastHit{};
        outHitMask[index] = RaycastCurrent(origin, direction, maxDistance, outHits[index], layerMask, queryTriggers)
                                ? uint8_t{1}
                                : uint8_t{0};
    };

    constexpr size_t kParallelRaycastThreshold = 256;
    if (count >= kParallelRaycastThreshold && count <= std::numeric_limits<uint32_t>::max() &&
        JobSystem::IsAvailable() && !JobSystem::Get().IsInline() && JobSystem::Get().GetWorkerCount() > 1) {
        // One task per ray makes large batches spend most of their time in
        // queueing, allocation and completion bookkeeping. Keep a few chunks
        // per worker so Jolt's narrow-phase work remains parallel without
        // turning the batch into thousands of tiny scheduler operations.
        const uint32_t workerCount = JobSystem::Get().GetWorkerCount();
        const uint32_t targetWorkerCount = std::min<uint32_t>(workerCount, std::numeric_limits<uint32_t>::max() / 4u);
        const uint32_t targetChunks = std::max<uint32_t>(targetWorkerCount * 4u, 1u);
        const uint32_t chunkSize = std::max<uint32_t>(64, 1u + (static_cast<uint32_t>(count) - 1u) / targetChunks);
        JobSystem::Get().ParallelForChunks(
            static_cast<uint32_t>(count), chunkSize,
            [&](uint32_t begin, uint32_t end) {
                for (uint32_t index = begin; index < end; ++index) {
                    castOne(index);
                }
            },
            JobDomain::Physics, JobPriority::Normal);
        if (GetQueryGeneration() != queryGeneration)
            throw std::logic_error("physics query world changed during raycast batch");
        return;
    }

    for (size_t index = 0; index < count; ++index) {
        castOne(static_cast<uint32_t>(index));
    }
    if (GetQueryGeneration() != queryGeneration)
        throw std::logic_error("physics query world changed during raycast batch");
}

bool PhysicsWorld::RaycastCurrent(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance,
                                  RaycastHit &outHit, uint32_t layerMask, bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(origin))
        return false;

    glm::vec3 normalizedDirection(0.0f);
    if (!NormalizeQueryDirection(direction, maxDistance, normalizedDirection))
        return false;

    const JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                            JPH::Vec3(normalizedDirection.x * maxDistance, normalizedDirection.y * maxDistance,
                                      normalizedDirection.z * maxDistance));
    const JPH::NarrowPhaseQuery &query = m_physicsSystem->GetNarrowPhaseQuery();
    LayerMaskObjectFilter objectFilter(layerMask);

    const auto publish = [&](const JPH::RayCastResult &result, bool filterTriggers) {
        outHit.distance = result.mFraction * maxDistance;
        outHit.point = origin + normalizedDirection * outHit.distance;
        outHit.bodyId = result.mBodyID.GetIndexAndSequenceNumber();
        outHit.collider = nullptr;
        outHit.gameObject = nullptr;

        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), result.mBodyID);
        if (lock.Succeeded()) {
            const JPH::Body &body = lock.GetBody();
            // Resolve the owning collider while the body lock is already held.
            // The previous path acquired a second BodyLockRead for every hit,
            // which made large raycast batches needlessly serialize on Jolt's
            // body-lock table.
            Collider *collider = ResolveColliderForSubShape(body, result.mBodyID.GetIndexAndSequenceNumber(),
                                                            result.mSubShapeID2.GetValue());
            if (filterTriggers && (body.IsSensor() || (collider && collider->IsTrigger()))) {
                return false;
            }
            outHit.collider = collider;
            outHit.gameObject = collider ? collider->GetGameObject() : nullptr;
            const JPH::Vec3 normal = body.GetWorldSpaceSurfaceNormal(
                result.mSubShapeID2, JPH::RVec3(outHit.point.x, outHit.point.y, outHit.point.z));
            outHit.normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
            PublishRaycastSubShape(body, result.mSubShapeID2, outHit);
        }
        return true;
    };

    if (queryTriggers) {
        JPH::ClosestHitCollisionCollector<JPH::CastRayCollector> collector;
        query.CastRay(ray, JPH::RayCastSettings(), collector, JPH::BroadPhaseLayerFilter(), objectFilter);
        if (!collector.HadHit())
            return false;
        return publish(collector.mHit, false);
    }

    JPH::AllHitCollisionCollector<JPH::CastRayCollector> collector;
    query.CastRay(ray, JPH::RayCastSettings(), collector, JPH::BroadPhaseLayerFilter(), objectFilter);
    if (!collector.HadHit())
        return false;
    collector.Sort();
    for (const JPH::RayCastResult &result : collector.mHits) {
        if (publish(result, true))
            return true;
    }
    return false;
}

bool PhysicsWorld::RaycastCollider(const Collider &collider, const glm::vec3 &origin, const glm::vec3 &direction,
                                   float maxDistance, RaycastHit &outHit) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    if (!m_initialized || !IsFinite(origin))
        return false;

    glm::vec3 normalizedDirection(0.0f);
    if (!NormalizeQueryDirection(direction, maxDistance, normalizedDirection))
        return false;

    const uint32_t bodyId = collider.GetBodyId();
    if (bodyId == 0xFFFFFFFF)
        return false;

    const JPH::BodyID joltBodyId(bodyId);
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), joltBodyId);
    if (!lock.Succeeded())
        return false;

    const JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                            JPH::Vec3(normalizedDirection.x * maxDistance, normalizedDirection.y * maxDistance,
                                      normalizedDirection.z * maxDistance));
    const JPH::TransformedShape transformedShape = lock.GetBody().GetTransformedShape();
    JPH::AllHitCollisionCollector<JPH::CastRayCollector> collector;
    transformedShape.CastRay(ray, JPH::RayCastSettings(), collector);
    if (!collector.HadHit())
        return false;

    collector.Sort();
    for (const JPH::RayCastResult &result : collector.mHits) {
        if (ResolveColliderForSubShape(bodyId, result.mSubShapeID2.GetValue()) != &collider)
            continue;

        outHit.distance = result.mFraction * maxDistance;
        outHit.point = origin + normalizedDirection * outHit.distance;
        outHit.bodyId = bodyId;
        outHit.collider = const_cast<Collider *>(&collider);
        outHit.gameObject = collider.GetGameObject();
        const JPH::Vec3 normal = transformedShape.GetWorldSpaceSurfaceNormal(
            result.mSubShapeID2, JPH::RVec3(outHit.point.x, outHit.point.y, outHit.point.z));
        outHit.normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
        PublishRaycastSubShape(lock.GetBody(), result.mSubShapeID2, outHit);
        return true;
    }
    return false;
}

glm::vec3 PhysicsWorld::ClosestPointOnCollider(const Collider &collider, const glm::vec3 &point) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    if (!m_initialized)
        throw std::logic_error("closest_point requires initialized physics");

    const uint32_t bodyId = collider.GetBodyId();
    if (bodyId == 0xFFFFFFFF)
        throw std::logic_error("closest_point requires a registered collider");

    glm::vec3 bodyPosition(0.0f);
    glm::quat bodyRotation(1.0f, 0.0f, 0.0f, 0.0f);
    {
        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(bodyId));
        if (!lock.Succeeded())
            throw std::logic_error("closest_point collider body is unavailable");
        const JPH::Body &body = lock.GetBody();
        const JPH::TransformedShape transformedShape = body.GetTransformedShape();
        JPH::AllHitCollisionCollector<JPH::CollidePointCollector> insideCollector;
        transformedShape.CollidePoint(JPH::RVec3(point.x, point.y, point.z), insideCollector);
        for (const JPH::CollidePointResult &result : insideCollector.mHits) {
            if (ResolveColliderForSubShape(bodyId, result.mSubShapeID2.GetValue()) == &collider)
                return point;
        }
        const JPH::RVec3 position = body.GetPosition();
        bodyPosition = glm::vec3(static_cast<float>(position.GetX()), static_cast<float>(position.GetY()),
                                 static_cast<float>(position.GetZ()));
        const JPH::Quat rotation = body.GetRotation();
        bodyRotation = glm::quat(rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ());
    }

    const auto *meshCollider = dynamic_cast<const MeshCollider *>(&collider);
    if (!dynamic_cast<const BoxCollider *>(&collider) && !dynamic_cast<const SphereCollider *>(&collider) &&
        !dynamic_cast<const CapsuleCollider *>(&collider) && !dynamic_cast<const CylinderCollider *>(&collider) &&
        !meshCollider)
        throw std::invalid_argument(
            "closest_point currently supports Box, Sphere, Capsule, Cylinder and Mesh colliders");

    // Jolt's MeshShape participates in the same point-vs-shape narrow phase
    // as convex shapes.  Do not reject a ready static triangle mesh here: the
    // broad phase and triangle BVH already provide the correct non-convex
    // closest surface.  Dynamic rigidbodies still reject non-convex meshes at
    // MeshCollider::CreateJoltShapeRaw(), so this does not weaken the physics
    // body contract or introduce a convex approximation.

    const auto *rawShape = static_cast<const JPH::Shape *>(collider.CreateJoltShapeRaw());
    if (!rawShape) {
        const auto *mesh = dynamic_cast<const MeshCollider *>(&collider);
        const std::string detail = mesh && !mesh->GetShapeError().empty() ? ": " + mesh->GetShapeError() : "";
        throw std::logic_error("closest_point collider geometry is not ready" + detail);
    }
    const JPH::RefConst<JPH::Shape> targetShape = rawShape;
    constexpr float pointRadius = 1.0e-5f;
    const JPH::SphereShape pointShape(pointRadius);
    const glm::vec3 relativePoint = point - bodyPosition;
    const JPH::Mat44 pointTransform =
        JPH::Mat44::sTranslation(JPH::Vec3(relativePoint.x, relativePoint.y, relativePoint.z));
    const JPH::Mat44 targetTransform =
        JPH::Mat44::sRotation(ToJoltQuat(bodyRotation)) * JPH::Mat44::sTranslation(targetShape->GetCenterOfMass());
    const JPH::AABox bounds = targetShape->GetLocalBounds();
    JPH::CollideShapeSettings settings;
    settings.mMaxSeparationDistance =
        glm::length(relativePoint) + bounds.GetCenter().Length() + bounds.GetExtent().Length() + 1.0f;
    JPH::ClosestHitCollisionCollector<JPH::CollideShapeCollector> collector;
    JPH::CollisionDispatch::sCollideShapeVsShape(
        &pointShape, targetShape, JPH::Vec3::sReplicate(1), JPH::Vec3::sReplicate(1), pointTransform, targetTransform,
        JPH::SubShapeIDCreator(), JPH::SubShapeIDCreator(), settings, collector);
    if (!collector.HadHit())
        throw std::runtime_error("closest_point could not evaluate collider geometry");

    const JPH::Vec3 closest = collector.mHit.mContactPointOn2;
    return bodyPosition + glm::vec3(closest.GetX(), closest.GetY(), closest.GetZ());
}

std::vector<RaycastHit> PhysicsWorld::RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance,
                                                 uint32_t layerMask, bool queryTriggers) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    std::vector<RaycastHit> hits;
    if (!m_initialized || layerMask == 0 || !IsFinite(origin))
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

        RaycastHit hit;
        hit.distance = result.mFraction * maxDistance;
        hit.point = origin + dir * hit.distance;
        hit.bodyId = bodyId;

        hit.collider = ResolveColliderForSubShape(hit.bodyId, result.mSubShapeID2.GetValue());
        if (!queryTriggers && (IsBodySensor(bodyId) || (hit.collider && hit.collider->IsTrigger())))
            continue;
        if (hit.collider && hit.collider->GetGameObject()) {
            hit.gameObject = hit.collider->GetGameObject();
        }

        JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), result.mBodyID);
        if (lock.Succeeded()) {
            const JPH::Body &body = lock.GetBody();
            JPH::Vec3 normal =
                body.GetWorldSpaceSurfaceNormal(result.mSubShapeID2, JPH::RVec3(hit.point.x, hit.point.y, hit.point.z));
            hit.normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
            PublishRaycastSubShape(body, result.mSubShapeID2, hit);
        }

        hits.push_back(hit);
    }

    return hits;
}

// ============================================================================
// Shared overlap / shape-cast implementations
// ============================================================================

std::vector<Collider *> PhysicsWorld::OverlapShapeImpl(const JPH::Shape &shape, const glm::vec3 &center,
                                                       const glm::quat &orientation, uint32_t layerMask,
                                                       bool queryTriggers) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    std::vector<Collider *> results;
    JPH::CollideShapeSettings settings;
    JPH::RMat44 transform =
        JPH::RMat44::sRotationTranslation(ToJoltQuat(orientation), JPH::RVec3(center.x, center.y, center.z));

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
        Collider *col = ResolveColliderForSubShape(bodyId, hit.mSubShapeID2.GetValue());
        if (!queryTriggers && (IsBodySensor(bodyId) || (col && col->IsTrigger())))
            continue;
        if (col && seen.insert(col).second) {
            results.push_back(col);
        }
    }
    return results;
}

bool PhysicsWorld::ShapeCastImpl(const JPH::Shape &shape, const glm::vec3 &origin, const glm::quat &orientation,
                                 const glm::vec3 &direction, float maxDistance, RaycastHit &outHit, uint32_t layerMask,
                                 bool queryTriggers) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    glm::vec3 dir(0.0f);
    if (!NormalizeQueryDirection(direction, maxDistance, dir))
        return false;

    JPH::RShapeCast shapeCast = JPH::RShapeCast::sFromWorldTransform(
        &shape, JPH::Vec3::sReplicate(1.0f),
        JPH::RMat44::sRotationTranslation(ToJoltQuat(orientation), JPH::RVec3(origin.x, origin.y, origin.z)),
        JPH::Vec3(dir.x * maxDistance, dir.y * maxDistance, dir.z * maxDistance));

    JPH::ShapeCastSettings castSettings;
    JPH::AllHitCollisionCollector<JPH::CastShapeCollector> collector;
    LayerMaskObjectFilter objectFilter(layerMask);

    const JPH::NarrowPhaseQuery &npQuery = m_physicsSystem->GetNarrowPhaseQuery();
    npQuery.CastShape(shapeCast, castSettings, JPH::RVec3(origin.x, origin.y, origin.z), collector,
                      JPH::BroadPhaseLayerFilter(), objectFilter);

    if (!collector.HadHit())
        return false;

    collector.Sort();
    const JPH::ShapeCastResult *selected = nullptr;
    Collider *selectedCollider = nullptr;
    for (const auto &candidate : collector.mHits) {
        uint32_t candidateBodyId = candidate.mBodyID2.GetIndexAndSequenceNumber();
        Collider *candidateCollider = ResolveColliderForSubShape(candidateBodyId, candidate.mSubShapeID2.GetValue());
        if (!queryTriggers && (IsBodySensor(candidateBodyId) || (candidateCollider && candidateCollider->IsTrigger())))
            continue;
        selected = &candidate;
        selectedCollider = candidateCollider;
        break;
    }
    if (!selected)
        return false;

    const auto &result = *selected;
    uint32_t bodyId = result.mBodyID2.GetIndexAndSequenceNumber();

    outHit.distance = result.mFraction * maxDistance;
    outHit.point = origin + dir * outHit.distance;
    outHit.bodyId = bodyId;
    outHit.normal =
        glm::vec3(-result.mPenetrationAxis.GetX(), -result.mPenetrationAxis.GetY(), -result.mPenetrationAxis.GetZ());
    float nLen = glm::length(outHit.normal);
    if (nLen > kEpsilon)
        outHit.normal /= nLen;

    outHit.collider = selectedCollider;
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
    if (!m_initialized || layerMask == 0 || !IsFinite(center) || !std::isfinite(radius) || radius <= 0.0f)
        return {};
    JPH::SphereShape sphere(radius);
    return OverlapShapeImpl(sphere, center, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), layerMask, queryTriggers);
}

std::vector<Collider *> PhysicsWorld::OverlapBox(const glm::vec3 &center, const glm::vec3 &halfExtents,
                                                 const glm::quat &orientation, uint32_t layerMask,
                                                 bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(center) || !HasPositiveFiniteExtents(halfExtents) ||
        !IsFinite(orientation))
        return {};
    JPH::BoxShape box(JPH::Vec3(halfExtents.x, halfExtents.y, halfExtents.z));
    return OverlapShapeImpl(box, center, orientation, layerMask, queryTriggers);
}

std::vector<Rigidbody *> PhysicsWorld::QueryRigidbodiesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum,
                                                                uint32_t layerMask, bool queryTriggers) const
{
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    std::vector<Rigidbody *> results;
    if (!m_initialized || layerMask == 0 || !IsFinite(minimum) || !IsFinite(maximum) || minimum.x > maximum.x ||
        minimum.y > maximum.y || minimum.z > maximum.z)
        return results;

    JPH::AllHitCollisionCollector<JPH::CollideShapeBodyCollector> collector;
    LayerMaskObjectFilter objectFilter(layerMask);
    const JPH::AABox bounds(JPH::Vec3(minimum.x, minimum.y, minimum.z), JPH::Vec3(maximum.x, maximum.y, maximum.z));
    m_physicsSystem->GetBroadPhaseQuery().CollideAABox(bounds, collector, JPH::BroadPhaseLayerFilter(), objectFilter);

    results.reserve(static_cast<size_t>(collector.mHits.size()));
    std::unordered_set<Rigidbody *> seen;
    for (const JPH::BodyID &bodyId : collector.mHits) {
        const uint32_t id = bodyId.GetIndexAndSequenceNumber();
        Collider *collider = FindColliderByBodyId(id);
        if (!collider || (!queryTriggers && (IsBodySensor(id) || collider->IsTrigger())))
            continue;
        GameObject *gameObject = collider->GetGameObject();
        Rigidbody *rigidbody = gameObject ? gameObject->GetComponent<Rigidbody>() : nullptr;
        if (rigidbody && rigidbody->IsEnabled() && seen.insert(rigidbody).second)
            results.push_back(rigidbody);
    }
    return results;
}

std::vector<Collider *> PhysicsWorld::OverlapCapsule(const glm::vec3 &point0, const glm::vec3 &point1, float radius,
                                                     uint32_t layerMask, bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(point0) || !IsFinite(point1) || !std::isfinite(radius) ||
        radius <= 0.0f)
        return {};
    const float segmentLength = glm::length(point1 - point0);
    const glm::vec3 center = (point0 + point1) * 0.5f;
    if (segmentLength <= 1e-6f) {
        JPH::SphereShape sphere(radius);
        return OverlapShapeImpl(sphere, center, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), layerMask, queryTriggers);
    }
    JPH::CapsuleShape capsule(segmentLength * 0.5f, radius);
    return OverlapShapeImpl(capsule, center, CapsuleOrientation(point0, point1), layerMask, queryTriggers);
}

// ============================================================================
// Shape cast queries (Unity: Physics.SphereCast / BoxCast)
// ============================================================================

bool PhysicsWorld::SphereCast(const glm::vec3 &origin, float radius, const glm::vec3 &direction, float maxDistance,
                              RaycastHit &outHit, uint32_t layerMask, bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(origin) || !std::isfinite(radius) || radius <= 0.0f)
        return false;
    JPH::SphereShape sphere(radius);
    return ShapeCastImpl(sphere, origin, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), direction, maxDistance, outHit, layerMask,
                         queryTriggers);
}

bool PhysicsWorld::BoxCast(const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction,
                           const glm::quat &orientation, float maxDistance, RaycastHit &outHit, uint32_t layerMask,
                           bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(center) || !HasPositiveFiniteExtents(halfExtents) ||
        !IsFinite(orientation))
        return false;

    JPH::BoxShape box(JPH::Vec3(halfExtents.x, halfExtents.y, halfExtents.z));
    return ShapeCastImpl(box, center, orientation, direction, maxDistance, outHit, layerMask, queryTriggers);
}

bool PhysicsWorld::CapsuleCast(const glm::vec3 &point0, const glm::vec3 &point1, float radius,
                               const glm::vec3 &direction, float maxDistance, RaycastHit &outHit, uint32_t layerMask,
                               bool queryTriggers) const
{
    if (!m_initialized || layerMask == 0 || !IsFinite(point0) || !IsFinite(point1) || !std::isfinite(radius) ||
        radius <= 0.0f)
        return false;
    const float segmentLength = glm::length(point1 - point0);
    const glm::vec3 center = (point0 + point1) * 0.5f;
    if (segmentLength <= 1e-6f) {
        JPH::SphereShape sphere(radius);
        return ShapeCastImpl(sphere, center, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), direction, maxDistance, outHit,
                             layerMask, queryTriggers);
    }
    JPH::CapsuleShape capsule(segmentLength * 0.5f, radius);
    return ShapeCastImpl(capsule, center, CapsuleOrientation(point0, point1), direction, maxDistance, outHit, layerMask,
                         queryTriggers);
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
    if (!go) {
        return fallback;
    }

    JPH::BodyID joltBodyId(bodyId);
    JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), joltBodyId);
    if (!lock.Succeeded()) {
        return fallback;
    }

    return ResolveColliderForSubShape(lock.GetBody(), bodyId, subShapeIdValue);
}

Collider *PhysicsWorld::ResolveColliderForSubShape(const JPH::Body &body, uint32_t bodyId,
                                                   uint32_t subShapeIdValue) const
{
    Collider *fallback = FindColliderByBodyId(bodyId);
    if (!fallback) {
        return nullptr;
    }

    auto *go = fallback->GetGameObject();
    if (!go) {
        return fallback;
    }

    const JPH::Shape *shape = body.GetShape();
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
    m_physicsSystem->GetBodyInterface().SetUserData(JPH::BodyID(bodyId), reinterpret_cast<uint64_t>(collider));
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
        if (!col || !col->IsEnabled() || col->GetBodyId() != 0xFFFFFFFF)
            continue;
        col->RegisterBody();
        if (col->GetBodyId() != 0xFFFFFFFF) {
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
