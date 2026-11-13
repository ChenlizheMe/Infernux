#include "BindingSceneInternal.h"
#include "ComponentBindingRegistry.h"
#include "JsonPyBridge.h"
#include "function/resources/InxMaterial/InxMaterial.h"
#include "function/scene/Camera.h"
#include "function/scene/ComponentFactory.h"
#include "function/scene/ComponentRecord.h"
#include "function/scene/GameObject.h"
#include "function/scene/MeshRenderer.h"
#include "function/scene/PyComponentProxy.h"
#include "function/scene/Rigidbody.h"
#include "function/scene/Scene.h"
#include "function/scene/SceneManager.h"
#include "function/scene/SkeletonInstance.h"
#include "function/scene/Transform.h"
#include "function/scene/physics/PhysicsWorld.h"
#include <algorithm>
#include <optional>
#include <pybind11/functional.h>
#include <stdexcept>
#include <tuple>
#include <variant>

namespace py = pybind11;

namespace infernux::scene_binding
{

namespace
{

enum class ResultTargetStatus
{
    Ready,
    Waiting,
    Stale,
    Changed
};

using RigidbodyEpoch = std::tuple<uint32_t, uint32_t, uint64_t, uint64_t>;
using ResultTargetEpoch = std::variant<std::monostate, uint64_t, RigidbodyEpoch>;

std::optional<ResultTargetEpoch> ReadResultTargetEpoch(const Component &component, const Rigidbody *body)
{
    if (body) {
        if (auto epoch = body->GetResultTargetEpoch())
            return *epoch;
        return std::nullopt;
    }
    if (const auto *mesh = dynamic_cast<const MeshRenderer *>(&component))
        return mesh->GetInlineMeshVersion();
    if (const auto *skeleton = dynamic_cast<const SkeletonInstance *>(&component)) {
        if (auto epoch = skeleton->GetResultTargetEpoch())
            return *epoch;
        return std::nullopt;
    }
    return std::monostate{};
}

class ResultTargetBatch
{
  public:
    explicit ResultTargetBatch(const std::vector<Component *> &components)
    {
        m_targets.reserve(components.size());
        for (const Component *component : components) {
            if (!component)
                throw std::invalid_argument("result target requires a native component");
            auto epoch = ReadResultTargetEpoch(*component, dynamic_cast<const Rigidbody *>(component));
            if (!epoch)
                throw std::invalid_argument("result target has no bound state");
            m_targets.push_back({component->GetHandle(), *epoch});
        }
    }

    py::tuple Handles() const
    {
        py::tuple handles(m_targets.size());
        for (size_t index = 0; index < m_targets.size(); ++index)
            handles[index] = py::cast(m_targets[index].handle, py::return_value_policy::copy);
        return handles;
    }

    py::tuple Inspect(const SceneManager &manager, bool requireBodyReady, bool collect) const
    {
        std::vector<Component *> resolved;
        if (collect)
            resolved.reserve(m_targets.size());
        bool waiting = false;
        uint64_t worldId = 0;
        Scene *world = nullptr;
        for (const auto &target : m_targets) {
            if (worldId != target.handle.worldId) {
                worldId = target.handle.worldId;
                world = manager.GetSceneByWorldId(worldId);
            }
            Component *component = world ? world->ResolveComponent(target.handle) : nullptr;
            if (!component || world->IsPendingDestroy(component->GetGameObject()))
                return py::make_tuple(ResultTargetStatus::Stale, py::tuple());

            const auto *body = dynamic_cast<const Rigidbody *>(component);
            if (!std::holds_alternative<std::monostate>(target.epoch) &&
                ReadResultTargetEpoch(*component, body) != target.epoch)
                return py::make_tuple(ResultTargetStatus::Changed, py::tuple());
            if (requireBodyReady && body &&
                !PhysicsWorld::Instance().IsBodyInBroadphase(body->GetParticipatingBodyId()))
                waiting = true;
            if (collect)
                resolved.push_back(component);
        }
        if (waiting)
            return py::make_tuple(ResultTargetStatus::Waiting, py::tuple());

        py::tuple targets(resolved.size());
        for (size_t index = 0; index < resolved.size(); ++index)
            targets[index] = ComponentBindingRegistry::Instance().CastToPython(resolved[index]);
        return py::make_tuple(ResultTargetStatus::Ready, targets);
    }

  private:
    struct Target
    {
        ObjectHandle handle;
        ResultTargetEpoch epoch;
    };

    std::vector<Target> m_targets;
};

} // namespace

static void ValidatePrefabNativeRecords(const nlohmann::json &node)
{
    ComponentFactory::ValidateDocument("Transform", node.at("transform"));
    std::vector<std::string> types{"Transform"};
    for (const auto &component : node.at("components")) {
        const auto record = DecodeComponentRecord(component);
        if (record.kind != ComponentRecordKind::Native)
            continue;
        const auto &type = record.nativeTypeName;
        ComponentFactory::ValidateDocument(type, BuildNativeComponentDocument(record));
        const auto blockers = ComponentFactory::GetAttachmentBlockers(type, types);
        if (!blockers.empty())
            throw std::invalid_argument(type + ": " + blockers.front());
        types.push_back(type);
    }
    for (size_t i = 1; i < types.size(); ++i) {
        for (const auto &required : ComponentFactory::GetTypeConstraints(types[i]).requiredTypes) {
            bool found = false;
            for (size_t j = 0; j < types.size() && !found; ++j) {
                if (j == i)
                    continue;
                const auto &type = types[j];
                const auto &aliases = ComponentFactory::GetTypeConstraints(type).satisfiedTypes;
                found = type == required || "native:infernux." + type == required ||
                        std::find(aliases.begin(), aliases.end(), required) != aliases.end();
            }
            if (!found)
                throw std::invalid_argument(types[i] + " requires missing component '" + required + "'");
        }
    }
    for (const auto &child : node.at("children"))
        ValidatePrefabNativeRecords(child);
}

py::dict ToPythonInstantiationIds(const InstantiationIds &ids)
{
    py::dict mapping;
    for (const auto &[source, published] : ids)
        mapping[py::int_(source)] = py::make_tuple(published.object, published.transform, published.components);
    return mapping;
}

static void BindSceneEnvironment(NativeClass<Scene> &cls)
{
    cls.def_property("name", &Scene::GetName, &Scene::SetName)
        .def_property_readonly("is_preview", &Scene::IsPreview)
        .def(
            "get_environment",
            [](const Scene &scene) {
                const SceneEnvironmentSettings &env = scene.GetEnvironment();
                py::dict d;
                d["skybox_material_guid"] = env.skyboxMaterialGuid;
                d["sky_top_color"] = py::make_tuple(env.skyTopColor.r, env.skyTopColor.g, env.skyTopColor.b);
                d["sky_horizon_color"] =
                    py::make_tuple(env.skyHorizonColor.r, env.skyHorizonColor.g, env.skyHorizonColor.b);
                d["sky_ground_color"] =
                    py::make_tuple(env.skyGroundColor.r, env.skyGroundColor.g, env.skyGroundColor.b);
                d["sky_exposure"] = env.skyExposure;
                d["ambient_source"] = env.ambientSource;
                d["ambient_intensity"] = env.ambientIntensity;
                d["ambient_color"] = py::make_tuple(env.ambientColor.r, env.ambientColor.g, env.ambientColor.b);
                d["ambient_sky_color"] =
                    py::make_tuple(env.ambientSkyColor.r, env.ambientSkyColor.g, env.ambientSkyColor.b);
                d["ambient_equator_color"] =
                    py::make_tuple(env.ambientEquatorColor.r, env.ambientEquatorColor.g, env.ambientEquatorColor.b);
                d["ambient_ground_color"] =
                    py::make_tuple(env.ambientGroundColor.r, env.ambientGroundColor.g, env.ambientGroundColor.b);
                return d;
            },
            "Get the scene environment (skybox material + ambient) settings as a dict")
        .def(
            "set_environment",
            [](Scene &scene, const py::dict &d) {
                SceneEnvironmentSettings env = scene.GetEnvironment();
                const auto readColor = [&](const char *key, glm::vec3 &out) {
                    if (!d.contains(key))
                        return;
                    py::sequence seq = d[key].cast<py::sequence>();
                    out = glm::vec3(seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>());
                };
                if (d.contains("skybox_material_guid"))
                    env.skyboxMaterialGuid = d["skybox_material_guid"].cast<std::string>();
                readColor("sky_top_color", env.skyTopColor);
                readColor("sky_horizon_color", env.skyHorizonColor);
                readColor("sky_ground_color", env.skyGroundColor);
                if (d.contains("sky_exposure"))
                    env.skyExposure = glm::clamp(d["sky_exposure"].cast<float>(), 0.0f, 8.0f);
                if (d.contains("ambient_source"))
                    env.ambientSource = glm::clamp(d["ambient_source"].cast<int>(), 0, 2);
                if (d.contains("ambient_intensity"))
                    env.ambientIntensity = glm::clamp(d["ambient_intensity"].cast<float>(), 0.0f, 8.0f);
                readColor("ambient_color", env.ambientColor);
                readColor("ambient_sky_color", env.ambientSkyColor);
                readColor("ambient_equator_color", env.ambientEquatorColor);
                readColor("ambient_ground_color", env.ambientGroundColor);
                scene.SetEnvironment(env);
            },
            py::arg("settings"),
            "Update scene environment settings from a dict (missing keys keep their current value)")
        .def(
            "resolve_skybox_material", [](const Scene &scene) { return scene.ResolveSkyboxMaterial(); },
            "Resolve the active skybox material (environment asset or builtin procedural sky)")
        .def("set_playing", &Scene::SetPlaying, py::arg("playing"), "Set the scene play-state flag");
}

static void BindSceneObjects(NativeClass<Scene> &cls)
{
    cls.def("create_game_object", &Scene::CreateGameObject, py::return_value_policy::reference,
            py::arg("name") = "GameObject", "Create a new empty GameObject in this scene")
        .def(
            "create_primitive",
            [](Scene *scene, PrimitiveType type, const std::string &name) {
                return CreatePrimitiveObject(scene, type, name);
            },
            py::return_value_policy::reference, py::arg("type"), py::arg("name") = "",
            "Create a primitive GameObject (Cube, Sphere, Capsule, Cylinder, Plane)")
        .def(
            "create_primitives_batch",
            [](Scene *scene, PrimitiveType type, size_t count, const std::string &namePrefix, bool withColliders) {
                return CreatePrimitiveObjectsBatch(scene, type, count, namePrefix, withColliders);
            },
            py::arg("type"), py::arg("count"), py::arg("name_prefix") = "", py::arg("with_colliders") = true,
            "Batch-create N primitive GameObjects. Returns a list of GameObjects.")
        .def(
            "create_from_model",
            [](Scene *scene, const std::string &guid, const std::string &name) {
                return CreateModelObject(scene, guid, name);
            },
            py::return_value_policy::reference, py::arg("guid"), py::arg("name") = "",
            "Create a GameObject from a mesh asset GUID")
        .def(
            "get_root_objects",
            [](Scene *scene) {
                std::vector<GameObject *> result;
                for (const auto &obj : scene->GetRootObjects()) {
                    result.push_back(obj.get());
                }
                return result;
            },
            py::return_value_policy::reference, "Get all root-level GameObjects")
        .def("get_all_objects", &Scene::GetAllObjects, py::return_value_policy::reference,
             "Get all GameObjects in the scene")
        .def(
            "find_objects_with_component",
            [](Scene &scene, const std::string &typeName) {
                std::vector<GameObject *> result;
                for (GameObject *object : scene.GetAllObjects()) {
                    if (!object)
                        continue;
                    for (const auto &component : object->GetAllComponents()) {
                        if (component && component->IsComponentType(typeName)) {
                            result.push_back(object);
                            break;
                        }
                    }
                }
                return result;
            },
            py::return_value_policy::reference, py::arg("type_name"),
            "Find GameObjects containing a native component type")
        .def("find", &Scene::Find, py::return_value_policy::reference, py::arg("name"), "Find a GameObject by name")
        .def("find_by_id", &Scene::FindByID, py::return_value_policy::reference, py::arg("id"),
             "Find a GameObject by ID")
        .def("resolve_game_object", &Scene::ResolveGameObject, py::return_value_policy::reference, py::arg("handle"),
             "Resolve a stable GameObject handle, or return None when stale")
        .def(
            "_resolve_native_component",
            [](Scene &scene, const ObjectHandle &handle, py::handle owner) -> Component * {
                Component *component = scene.ResolveComponent(handle);
                if (auto *proxy = dynamic_cast<PyComponentProxy *>(component);
                    proxy && !proxy->GetPyComponent().is(owner))
                    return nullptr;
                return component;
            },
            py::return_value_policy::reference, py::arg("handle"), py::arg("owner"))
        .def(
            "resolve_component",
            [](Scene &scene, const ObjectHandle &handle) -> py::object {
                Component *component = scene.ResolveComponent(handle);
                if (!component)
                    return py::none();
                return ComponentBindingRegistry::Instance().CastToPython(component);
            },
            py::arg("handle"), "Resolve a stable Component or Transform handle, or return None when stale")
        .def("find_object_by_id", &Scene::FindByID, py::return_value_policy::reference, py::arg("id"),
             "Alias for find_by_id. Find a GameObject by ID")
        .def("find_with_tag", &Scene::FindWithTag, py::return_value_policy::reference, py::arg("tag"),
             "Find the first GameObject with a given tag")
        .def("find_game_objects_with_tag", &Scene::FindGameObjectsWithTag, py::return_value_policy::reference,
             py::arg("tag"), "Find all GameObjects with a given tag")
        .def("find_game_objects_in_layer", &Scene::FindGameObjectsInLayer, py::return_value_policy::reference,
             py::arg("layer"), "Find all GameObjects in a given layer")
        .def("destroy_game_object", &Scene::DestroyGameObject, py::arg("game_object"),
             "Destroy a GameObject (will be removed at end of frame)")
        .def("is_pending_destroy", &Scene::IsPendingDestroy, py::arg("game_object"),
             "Check whether an object or one of its parents is queued for destruction or being destroyed")
        .def(
            "_remove_game_object_immediately",
            [](Scene &scene, GameObject *object) {
                if (!object || object->GetScene() != &scene)
                    throw std::invalid_argument("Immediate removal requires an object owned by this Scene");
                scene.RemoveGameObject(object);
            },
            py::arg("game_object"), "Remove one transaction-owned object without draining the Scene destroy queue")
        .def(
            "_clone_game_object",
            [](Scene &scene, GameObject *source, GameObject *parent, bool worldSpace) {
                InstantiationIds ids;
                auto *root = scene.InstantiateGameObject(source, ids, parent, worldSpace);
                return py::make_tuple(py::cast(root, py::return_value_policy::reference),
                                      ToPythonInstantiationIds(ids));
            },
            py::arg("source"), py::arg("parent") = nullptr, py::arg("instantiate_in_world_space") = false,
            "Internal native clone and source-to-published object/component identities")
        .def(
            "_clone_game_objects",
            [](Scene &scene, GameObject *source, const FloatArray &positions,
               const std::optional<FloatArray> &rotations, const std::optional<FloatArray> &scales, GameObject *parent,
               bool instantiateInWorldSpace, bool returnObjects) {
                return InstantiateGameObjectsBatch(scene, source, positions, rotations, scales, parent,
                                                   instantiateInWorldSpace, returnObjects);
            },
            py::arg("source"), py::arg("positions"), py::arg("rotations") = py::none(), py::arg("scales") = py::none(),
            py::arg("parent") = nullptr, py::arg("instantiate_in_world_space") = true, py::arg("return_objects") = true,
            "Internal native bulk subtree clone used by the public Instantiate overload")
        .def(
            "_instantiate_document",
            [](Scene &scene, const PythonDocument &document, GameObject *parent) {
                InstantiationIds ids;
                auto *root = scene.InstantiateFromDocument(document.value, ids, parent);
                return py::make_tuple(py::cast(root, py::return_value_policy::reference),
                                      ToPythonInstantiationIds(ids));
            },
            py::return_value_policy::reference, py::arg("document"), py::arg("parent") = nullptr,
            "Internal native ObjectGraph instantiate; Python callers must preflight first")
        .def("process_pending_destroys", &Scene::ProcessPendingDestroys, "Process pending GameObject destroys")
        .def("is_playing", &Scene::IsPlaying, "Check if the scene is in play mode")
        .def("start", &Scene::Start, "Trigger Awake+Start on all components (idempotent — skipped if already started)")
        .def("awake_object", &Scene::AwakeObject, py::arg("game_object"),
             "Re-run Awake+OnEnable on a GameObject and its descendants (used after undo deserialization)");
}

static void BindSceneState(NativeClass<Scene> &cls)
{
    cls.def_property_readonly("structure_version", &Scene::GetStructureVersion,
                              "Monotonic counter bumped on structural changes (add/remove/reparent)")
        .def_property_readonly("temporal_discontinuity_revision", &Scene::GetTemporalDiscontinuityRevision,
                               "Monotonic counter bumped only by explicit world-time jumps")
        .def_property_readonly("world_id", &Scene::GetWorldId, "Unique identity of this native Scene world")
        // Camera management
        .def_property("main_camera", &Scene::GetMainCamera, &Scene::SetMainCamera, py::return_value_policy::reference,
                      "Get/set the explicitly preferred Camera component for this scene")
        .def_property_readonly(
            "effective_game_camera", [](Scene &scene) { return scene.FindGameCamera(nullptr); },
            py::return_value_policy::reference,
            "Get the explicitly preferred active Camera, or the first active camera by depth")
        .def_property_readonly(
            "active_game_cameras", [](Scene &scene) { return scene.GetActiveGameCameras(nullptr); },
            py::return_value_policy::reference, "Get all active Game cameras in stable depth order");
}

void BindScene(py::module_ &m)
{
    NativeClass<Scene> cls(m, "Scene");
    BindSceneEnvironment(cls);
    BindSceneObjects(cls);
    BindSceneDocuments(cls);
    BindSceneState(cls);
}

void BindSceneManager(py::module_ &m)
{
    py::enum_<SceneManager::RuntimeFrameBarrier>(m, "NativeRuntimeFrameBarrier")
        .value("FIXED_SCRIPT", SceneManager::RuntimeFrameBarrier::FixedScript)
        .value("PHYSICS_POST_SCRIPT", SceneManager::RuntimeFrameBarrier::PhysicsPostScript)
        .value("TRANSFORM_TO_PHYSICS", SceneManager::RuntimeFrameBarrier::TransformToPhysics)
        .value("PHYSICS_SIMULATION", SceneManager::RuntimeFrameBarrier::PhysicsSimulation)
        .value("PHYSICS_TO_TRANSFORM", SceneManager::RuntimeFrameBarrier::PhysicsToTransform)
        .value("TRANSFORM_RESOLVE", SceneManager::RuntimeFrameBarrier::TransformResolve)
        .value("FINAL_TRANSFORM_RESOLVE", SceneManager::RuntimeFrameBarrier::FinalTransformResolve)
        .value("ANIMATION_EVALUATION", SceneManager::RuntimeFrameBarrier::AnimationEvaluation)
        .value("ANIMATION_TIMELINE", SceneManager::RuntimeFrameBarrier::AnimationTimeline)
        .value("RENDER_EXTRACTION", SceneManager::RuntimeFrameBarrier::RenderExtraction)
        .value("RENDER_GRAPH", SceneManager::RuntimeFrameBarrier::RenderGraph)
        .value("SNAPSHOT_PUBLICATION", SceneManager::RuntimeFrameBarrier::SnapshotPublication)
        .value("RENDER_SUBMITTED", SceneManager::RuntimeFrameBarrier::RenderSubmitted)
        .value("RENDER_PRESENTED", SceneManager::RuntimeFrameBarrier::RenderPresented)
        .value("PENDING_DESTROY", SceneManager::RuntimeFrameBarrier::PendingDestroy);

    py::class_<RuntimeWorld::FrameTime>(m, "_FrameTime")
        .def_readonly("time", &RuntimeWorld::FrameTime::time)
        .def_readonly("unscaled_time", &RuntimeWorld::FrameTime::unscaledTime)
        .def_readonly("delta_time", &RuntimeWorld::FrameTime::deltaTime)
        .def_readonly("unscaled_delta_time", &RuntimeWorld::FrameTime::unscaledDeltaTime)
        .def_readonly("smooth_delta_time", &RuntimeWorld::FrameTime::smoothDeltaTime)
        .def_readonly("frame_count", &RuntimeWorld::FrameTime::frameCount);

    py::class_<SceneManager, std::unique_ptr<SceneManager, py::nodelete>>(m, "SceneManager")
        .def_static("instance", &SceneManager::Instance, py::return_value_policy::reference,
                    "Get the singleton SceneManager instance")
        .def_property_readonly("_frame_time", &SceneManager::GetFrameTime, py::return_value_policy::reference_internal)
        .def("get_realtime_since_startup", &SceneManager::GetRealtimeSinceStartup,
             "Monotonic real seconds since the owning engine started")
        .def("create_scene", &SceneManager::CreateScene, py::return_value_policy::reference, py::arg("name"),
             "Create a new empty scene")
        .def("_create_preview_scene", &SceneManager::CreatePreviewScene, py::return_value_policy::reference,
             py::arg("name"))
        .def("_close_preview_scene", &SceneManager::ClosePreviewScene, py::arg("scene"))
        .def("_get_preview_scenes",
             [](SceneManager &manager) {
                 py::list result;
                 for (const auto &scene : manager.GetPreviewScenes())
                     result.append(py::cast(scene.get(), py::return_value_policy::reference));
                 return result;
             })
        .def("unload_scene", &SceneManager::UnloadScene, py::arg("scene"),
             "Unload and destroy a scene, removing all its GameObjects and physics bodies")
        .def("get_active_scene", &SceneManager::GetActiveScene, py::return_value_policy::reference,
             "Get the currently active scene")
        .def("get_runtime_persistent_scene", &SceneManager::GetRuntimePersistentScene,
             py::return_value_policy::reference, "Get the runtime-only DontDestroyOnLoad Scene, or None")
        .def("find_runtime_object_by_id", &SceneManager::FindRuntimeObjectByID, py::return_value_policy::reference,
             py::arg("id"), "Find an object in the active or DontDestroyOnLoad runtime Scene")
        .def("set_active_scene", &SceneManager::SetActiveScene, py::arg("scene"), "Set the active scene")
        .def("prepare_active_scene_replacement", &SceneManager::PrepareActiveSceneReplacement,
             "Move queued DontDestroyOnLoad roots before replacing the active Scene document")
        .def("mark_temporal_discontinuity", &SceneManager::MarkActiveSceneTemporalDiscontinuity,
             "Mark an explicit time jump in the active scene for temporal render effects")
        .def("get_scene", &SceneManager::GetScene, py::return_value_policy::reference, py::arg("name"),
             "Get a scene by name")
        .def("get_scene_by_world_id", &SceneManager::GetSceneByWorldId, py::return_value_policy::reference,
             py::arg("world_id"), "Get a loaded scene by its stable runtime World identity")
        .def("get_scene_at", &SceneManager::GetSceneAt, py::return_value_policy::reference, py::arg("index"),
             "Get a loaded scene by its stable loaded-list index, or None")
        .def("move_scene_adjacent", &SceneManager::MoveSceneAdjacent, py::arg("dragged_world_id"),
             py::arg("target_world_id"), py::arg("after"), "Move a loaded scene before or after another loaded scene")
        .def("find_runtime_object", &SceneManager::FindRuntimeObject, py::return_value_policy::reference,
             py::arg("name"), "Find the first named GameObject across loaded scenes")
        .def("find_runtime_object_with_tag", &SceneManager::FindRuntimeObjectWithTag,
             py::return_value_policy::reference, py::arg("tag"),
             "Find the first tagged GameObject across loaded scenes")
        .def("find_runtime_objects_with_tag", &SceneManager::FindRuntimeObjectsWithTag,
             py::return_value_policy::reference, py::arg("tag"), "Find all tagged GameObjects across loaded scenes")
        .def("find_runtime_objects_in_layer", &SceneManager::FindRuntimeObjectsInLayer,
             py::return_value_policy::reference, py::arg("layer"),
             "Find all GameObjects in a layer across loaded scenes")
        .def("move_game_object_to_scene", &SceneManager::MoveGameObjectToScene, py::arg("game_object"),
             py::arg("destination"), "Move a root GameObject hierarchy to another loaded Scene")
        .def_property_readonly("scene_count", &SceneManager::GetSceneCount, "Number of currently loaded scenes")
        .def("is_playing", &SceneManager::IsPlaying, "Check if in play mode")
        .def("set_runtime_lifecycle_callbacks", &SceneManager::SetRuntimeLifecycleCallbacks, py::arg("begin_frame"),
             py::arg("fixed_update"), py::arg("update"), py::arg("late_update"), py::arg("editor_update"),
             py::arg("end_frame"), "Install the shared Editor/Player runtime lifecycle bridge")
        .def("set_runtime_frame_barrier_callback", &SceneManager::SetRuntimeFrameBarrierCallback, py::arg("callback"),
             "Install the native runtime frame barrier bridge")
        .def("set_runtime_result_consumer_callback", &SceneManager::SetRuntimeResultConsumerCallback,
             py::arg("callback"))
        .def("_set_result_consumers_present", &SceneManager::SetRuntimeResultConsumersPresent)
        .def("set_runtime_lifecycle_plan", &SceneManager::SetRuntimeLifecyclePlan, py::arg("revision"),
             py::arg("fixed_update_count"), py::arg("update_count"), py::arg("animation_update_count"),
             py::arg("late_update_count"),
             "Publish the immutable Python lifecycle phase plan to the native frame driver")
        .def("set_runtime_lifecycle_work_available", &SceneManager::SetRuntimeLifecycleWorkAvailable,
             py::arg("available"), "Enable lifecycle bridge calls only while Python components exist")
        .def("clear_runtime_lifecycle_callbacks", &SceneManager::ClearRuntimeLifecycleCallbacks,
             "Remove the shared runtime lifecycle bridge")
        .def("play", &SceneManager::Play, "Enter play mode")
        .def("_start_active_scene_for_play", &SceneManager::StartActiveSceneForPlay,
             "Internal: publish a transactionally loaded Scene into the current play session")
        .def("_start_scene_for_play", &SceneManager::StartSceneForPlay, py::arg("scene"),
             "Internal: publish one newly loaded additive Scene without resetting World time")
        .def("stop", &SceneManager::Stop, "Stop play mode")
        .def("pause", &SceneManager::Pause, "Pause play mode")
        .def("is_paused", &SceneManager::IsPaused, "Check if paused")
        .def("is_stepping", &SceneManager::IsStepping, "Check if an explicit paused simulation step is in progress")
        .def("_is_executing_step", &SceneManager::IsExecutingStep,
             "Internal lifecycle permission during native Step dispatch")
        .def("_retain_step_frame", &SceneManager::RetainStepFrame)
        .def_property_readonly("_step_permission_index", &SceneManager::GetStepPermissionIndex)
        .def_property_readonly("runtime_frame_count", &SceneManager::GetRuntimeFrameCount,
                               "Completed gameplay frames for the current active runtime scene")
        .def_property_readonly("simulation_epoch", &SceneManager::GetSimulationEpoch,
                               "Native simulation session identity; Fresh Play and Stop advance it")
        .def_property_readonly("fixed_step_index", &SceneManager::GetFixedStepIndex,
                               "Fixed bundles begun in this session, including empty-body and paused steps")
        .def_property_readonly("completed_fixed_step_index", &SceneManager::GetCompletedFixedStepIndex,
                               "Last completed physical integration in this Play session")
        .def_property_readonly("published_fixed_step_index", &SceneManager::GetPublishedFixedStepIndex)
        .def_property_readonly("fixed_step_pending", &SceneManager::HasPendingFixedStep)
        .def("get_fixed_time_step", &SceneManager::GetFixedTimeStep, "Get the fixed physics timestep in seconds")
        .def("set_fixed_time_step", &SceneManager::SetFixedTimeStep, py::arg("value"),
             "Set the fixed physics timestep in seconds")
        .def("get_frame_delta_time", &SceneManager::GetFrameDeltaTime)
        .def("get_max_fixed_delta_time", &SceneManager::GetMaxFixedDeltaTime,
             "Get the max clamped frame delta used by the fixed-step accumulator")
        .def("set_max_fixed_delta_time", &SceneManager::SetMaxFixedDeltaTime, py::arg("value"),
             "Set the max clamped frame delta used by the fixed-step accumulator")
        .def_property("time_scale", &SceneManager::GetTimeScale, &SceneManager::SetTimeScale,
                      "Global scale applied to gameplay and fixed-step simulation")
        .def_property_readonly("fixed_time", &SceneManager::GetFixedTime,
                               "Scaled simulation time at the current fixed step")
        .def_property_readonly("fixed_unscaled_time", &SceneManager::GetFixedUnscaledTime,
                               "Real time represented by completed fixed steps")
        .def("step", &SceneManager::Step, py::arg("delta_time") = 0.016f,
             "Execute exactly one shared fixed-physics bundle plus Update/LateUpdate while paused. "
             "No-op while Play is running; manual stepping cannot advance a second World beside the "
             "automatic accumulator.")
        .def("get_last_collider_sync_candidate_count", &SceneManager::GetLastColliderSyncCandidateCount,
             "Number of dirty collider handles considered by the most recent simulation frame")
        .def("get_last_rigidbody_sync_candidate_count", &SceneManager::GetLastRigidbodySyncCandidateCount,
             "Number of active physics bodies considered for pose readback in the most recent simulation frame")
        .def("get_last_interpolation_candidate_count", &SceneManager::GetLastInterpolationCandidateCount,
             "Number of physics bodies considered for presentation interpolation in the most recent frame")
        .def("get_global_transform_serial", &SceneManager::GetGlobalTransformSerial,
             "Get the monotonic native Transform storage revision")
        .def(
            "get_last_frame_profile",
            [](const SceneManager &manager) {
                const auto &profile = manager.GetLastFrameProfile();
                py::dict result;
                result["editor_camera_ms"] = profile.editorCameraMs;
                result["editor_update_ms"] = profile.editorUpdateMs;
                result["pending_starts_ms"] = profile.pendingStartsMs;
                result["sync_external_moves_ms"] = profile.syncExternalMovesMs;
                result["sync_colliders_ms"] = profile.syncCollidersMs;
                result["fixed_update_ms"] = profile.fixedUpdateMs;
                result["physics_step_ms"] = profile.physicsStepMs;
                result["physics_prepare_ms"] = profile.physicsPrepareMs;
                result["physics_ccd_ms"] = profile.physicsCCDMs;
                result["physics_jolt_ms"] = profile.physicsJoltMs;
                result["physics_finish_ms"] = profile.physicsFinishMs;
                result["physics_events_ms"] = profile.physicsEventsMs;
                result["sync_rigidbodies_ms"] = profile.syncRigidbodiesMs;
                result["interpolation_ms"] = profile.interpolationMs;
                result["gameplay_update_ms"] = profile.gameplayUpdateMs;
                result["late_update_ms"] = profile.lateUpdateMs;
                result["audio_ms"] = profile.audioMs;
                result["end_frame_ms"] = profile.endFrameMs;
                result["fixed_steps"] = profile.fixedSteps;
                result["collider_sync_candidates"] = profile.colliderSyncCandidates;
                result["rigidbody_sync_candidates"] = profile.rigidbodySyncCandidates;
                result["interpolation_candidates"] = profile.interpolationCandidates;
                result["contact_events"] = profile.contactEvents;
                result["dynamic_ccd_splits"] = profile.dynamicCCDSplits;
                return result;
            },
            "Return normalized timing and candidate counters for the most recent frame")
        .def("dont_destroy_on_load", &SceneManager::DontDestroyOnLoad, py::arg("game_object"),
             "Mark a root GameObject so it survives scene switches. Unity: DontDestroyOnLoad()");

    py::enum_<ResultTargetStatus>(m, "_ResultTargetStatus")
        .value("Ready", ResultTargetStatus::Ready)
        .value("Waiting", ResultTargetStatus::Waiting)
        .value("Stale", ResultTargetStatus::Stale)
        .value("Changed", ResultTargetStatus::Changed);
    py::class_<ResultTargetBatch>(m, "_ResultTargetBatch")
        .def(py::init<const std::vector<Component *> &>(), py::arg("components"))
        .def_property_readonly("handles", &ResultTargetBatch::Handles)
        .def("inspect", &ResultTargetBatch::Inspect, py::arg("manager"), py::arg("require_body_ready") = false,
             py::arg("collect") = false);

    // ComponentFactory — query registered native component types
    m.def("get_registered_component_types", &ComponentFactory::GetUserAddableTypeNames,
          "Get user-addable registered native component type names");
    m.def("_validate_prefab_native_records",
          [](const PythonDocument &document) { ValidatePrefabNativeRecords(document.value); });
    m.def("_rewrite_native_component_references", [](const std::string &typeName, const PythonDocument &document,
                                                     const py::function &rewrite) {
        auto candidate = document.value;
        ComponentFactory::RewriteDocumentComponentReferences(typeName, candidate, [&](const nlohmann::json &value) {
            return PythonToJson(rewrite(JsonToPython(value)));
        });
        return JsonToPython(candidate);
    });
}

} // namespace infernux::scene_binding
