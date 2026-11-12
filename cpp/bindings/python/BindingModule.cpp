#include "BindingRegistration.h"
#include <core/config/EngineConfig.h>
#include <core/log/InxLog.h>
#include <function/renderer/ProfileConfig.h>

PYBIND11_MODULE(_Infernux, module)
{
    using infernux::EngineConfig;
    // ---- EngineConfig (centralised runtime configuration) ----
    pybind11::class_<EngineConfig>(module, "EngineConfig",
                                   "Centralised engine configuration singleton.\n"
                                   "Modify values BEFORE the corresponding subsystem initializes.\n"
                                   "Access via EngineConfig.get().")
        .def_static("get", &EngineConfig::Get, pybind11::return_value_policy::reference,
                    "Get the singleton EngineConfig instance.")
        // Rendering — Swapchain
        .def_readwrite("max_frames_in_flight", &EngineConfig::maxFramesInFlight)
        // Physics — Jolt Configuration
        .def_readwrite("physics_temp_allocator_size", &EngineConfig::physicsTempAllocatorSize)
        .def_readwrite("physics_max_jobs", &EngineConfig::physicsMaxJobs)
        .def_readwrite("physics_max_barriers", &EngineConfig::physicsMaxBarriers)
        .def_readwrite("physics_max_bodies", &EngineConfig::physicsMaxBodies)
        .def_readwrite("physics_max_body_pairs", &EngineConfig::physicsMaxBodyPairs)
        .def_readwrite("physics_max_contact_constraints", &EngineConfig::physicsMaxContactConstraints)
        .def_readwrite("physics_collision_steps", &EngineConfig::physicsCollisionSteps)
        .def_readwrite("physics_velocity_steps", &EngineConfig::physicsVelocitySteps)
        .def_readwrite("physics_position_steps", &EngineConfig::physicsPositionSteps)
        .def_readwrite("physics_penetration_slop", &EngineConfig::physicsPenetrationSlop)
        .def_readwrite("physics_speculative_contact_distance", &EngineConfig::physicsSpeculativeContactDistance)
        .def_readwrite("physics_linear_cast_max_penetration", &EngineConfig::physicsLinearCastMaxPenetration)
        .def_readwrite("physics_baumgarte", &EngineConfig::physicsBaumgarte)
        .def_readwrite("physics_max_penetration_distance", &EngineConfig::physicsMaxPenetrationDistance)
        .def_readwrite("physics_linear_cast_threshold", &EngineConfig::physicsLinearCastThreshold)
        .def_readwrite("physics_min_velocity_for_restitution", &EngineConfig::physicsMinVelocityForRestitution)
        .def_readwrite("physics_time_before_sleep", &EngineConfig::physicsTimeBeforeSleep)
        .def_readwrite("physics_point_velocity_sleep_threshold", &EngineConfig::physicsPointVelocitySleepThreshold)
        .def_property(
            "physics_gravity", [](EngineConfig &self) { return self.physicsGravity; },
            [](EngineConfig &self, const glm::vec3 &v) { self.physicsGravity = v; },
            "Default gravity vector (applied on physics init)")
        .def_readwrite("physics_max_concurrency", &EngineConfig::physicsMaxConcurrency)
        // Physics — Default Collider Properties
        .def_readwrite("default_collider_friction", &EngineConfig::defaultColliderFriction)
        .def_readwrite("default_collider_bounciness", &EngineConfig::defaultColliderBounciness)
        // Physics — Default Rigidbody Properties
        .def_readwrite("default_rigidbody_mass", &EngineConfig::defaultRigidbodyMass)
        .def_readwrite("default_rigidbody_drag", &EngineConfig::defaultRigidbodyDrag)
        .def_readwrite("default_rigidbody_angular_drag", &EngineConfig::defaultRigidbodyAngularDrag)
        .def_readwrite("default_max_angular_velocity", &EngineConfig::defaultMaxAngularVelocity)
        .def_readwrite("default_max_linear_velocity", &EngineConfig::defaultMaxLinearVelocity)
        // Physics — Layers
        .def_readwrite("default_query_layer_mask", &EngineConfig::defaultQueryLayerMask)
        // Render Queue Ranges
        .def_readwrite("opaque_queue_min", &EngineConfig::opaqueQueueMin)
        .def_readwrite("opaque_queue_max", &EngineConfig::opaqueQueueMax)
        .def_readwrite("transparent_queue_min", &EngineConfig::transparentQueueMin)
        .def_readwrite("transparent_queue_max", &EngineConfig::transparentQueueMax)
        .def_readwrite("shadow_caster_queue_min", &EngineConfig::shadowCasterQueueMin)
        .def_readwrite("shadow_caster_queue_max", &EngineConfig::shadowCasterQueueMax)
        .def_readwrite("component_gizmo_queue_min", &EngineConfig::componentGizmoQueueMin)
        .def_readwrite("component_gizmo_queue_max", &EngineConfig::componentGizmoQueueMax)
        .def_readwrite("editor_gizmo_queue_min", &EngineConfig::editorGizmoQueueMin)
        .def_readwrite("editor_gizmo_queue_max", &EngineConfig::editorGizmoQueueMax)
        .def_readwrite("editor_tools_queue_min", &EngineConfig::editorToolsQueueMin)
        .def_readwrite("editor_tools_queue_max", &EngineConfig::editorToolsQueueMax)
        .def_readwrite("skybox_queue", &EngineConfig::skyboxQueue);

    module.def("_format_log_message", [](int level, const std::string &sourceFile, int sourceLine,
                                         const std::string &message, const std::string &projectRoot) {
        return infernux::FormatDiagnosticMessage(static_cast<infernux::LogLevel>(level), sourceFile, sourceLine,
                                                 message, projectRoot);
    });
    pybind11::register_exception<infernux::InvalidNativeObjectError>(module, "InvalidNativeObjectError",
                                                                     PyExc_RuntimeError);
    module.def(
        "is_frame_profile_enabled", []() { return INFERNUX_FRAME_PROFILE != 0; },
        "Return whether this native module was built with detailed frame profiling enabled.");
    infernux::RegisterSemanticCatalogBindings(module);
    infernux::RegisterRuntimeNumericsBindings(module);
    infernux::RegisterFrameWorkBindings(module);
    infernux::RegisterTextBindings(module);
    infernux::RegisterShaderCompilerBindings(module);
#if defined(INFERNUX_PYBIND_WEB_PLAYER)
    module.attr("__runtime_profile__") = "web-player";
    infernux::RegisterVector2Bindings(module);
    infernux::RegisterAnimationCurveBindings(module);
    infernux::RegisterMotionPathBindings(module);
    infernux::RegisterTweenNumericsBindings(module);
    infernux::RegisterEulerTweenBindings(module);
    infernux::RegisterVectorTweenBindings(module);
    infernux::RegisterStringTweenBindings(module);
    infernux::RegisterVector3Bindings(module);
    infernux::RegisterVec4fBindings(module);
    infernux::RegisterBatchBindings(module);
    infernux::RegisterResourceBindings(module);
    infernux::RegisterAssetRegistryBindings(module);
    infernux::RegisterSceneBindings(module);
    infernux::RegisterTagLayerBindings(module);
    infernux::RegisterInputBindings(module);
    infernux::RegisterPhysicsBindings(module);
    infernux::RegisterAudioBindings(module);
    infernux::RegisterComputeBindings(module);
#else
    module.attr("__runtime_profile__") = "desktop";
    infernux::RegisterInfernuxBindings(module);
    infernux::RegisterGUIBindings(module);
    infernux::RegisterVector2Bindings(module);
    infernux::RegisterAnimationCurveBindings(module);
    infernux::RegisterMotionPathBindings(module);
    infernux::RegisterTweenNumericsBindings(module);
    infernux::RegisterEulerTweenBindings(module);
    infernux::RegisterVectorTweenBindings(module);
    infernux::RegisterStringTweenBindings(module);
    infernux::RegisterVector3Bindings(module);
    infernux::RegisterVec4fBindings(module);
    infernux::RegisterResourceBindings(module);
    infernux::RegisterAssetDatabaseBindings(module);
    infernux::RegisterAssetRegistryBindings(module);
    infernux::RegisterSceneBindings(module);
    infernux::RegisterTagLayerBindings(module);
    infernux::RegisterRhiBindings(module);
    infernux::RegisterRenderGraphBindings(module);
    infernux::RegisterCommandBufferBindings(module);
    infernux::RegisterRenderPipelineBindings(module);
    infernux::RegisterInputBindings(module);
    infernux::RegisterPhysicsBindings(module);
    infernux::RegisterAudioBindings(module);
    infernux::RegisterBatchBindings(module);
#endif
    infernux::RegisterCpuComputeBindings(module);
    infernux::RegisterRandomBindings(module);
}
