/**
 * @file BindingScene.cpp
 * @brief Python bindings for SceneManager, Scene, and GameObject.
 *
 * Exposes the scene hierarchy to Python for editor integration.
 */

// Jolt types are no longer exposed in collider headers — no Jolt include needed here

#include "ComponentBindingRegistry.h"
#include "core/log/InxLog.h"
#include "function/resources/AssetRegistry/AssetRegistry.h"
#include "function/resources/InxMaterial/InxMaterial.h"
#include "function/resources/InxMesh/InxMesh.h"
#include "function/resources/InxResource/InxResourceMeta.h"
#include "function/scene/BoxCollider.h"
#include "function/scene/Camera.h"
#include "function/scene/CapsuleCollider.h"
#include "function/scene/ComponentFactory.h"
#include "function/scene/GameObject.h"
#include "function/scene/Light.h"
#include "function/scene/MeshCollider.h"
#include "function/scene/MeshRenderer.h"
#include "function/scene/PrimitiveMeshes.h"
#include "function/scene/PyComponentProxy.h"
#include "function/scene/Rigidbody.h"
#include "function/scene/SkinnedMeshRenderer.h"
#include "function/scene/Scene.h"
#include "function/scene/SceneManager.h"
#include "function/scene/SphereCollider.h"
#include "function/scene/SpriteRenderer.h"
#include "function/scene/Transform.h"
#include "function/scene/physics/PhysicsECSStore.h"
#include <cctype>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <unordered_map>

namespace py = pybind11;

namespace infernux
{

/// Resolve a Python component type (str, class with _cpp_type_name, or class with __name__)
/// to a C++ type name string. Returns empty string on failure.
static std::string ResolveComponentTypeName(py::object componentType)
{
    if (py::isinstance<py::str>(componentType)) {
        return componentType.cast<std::string>();
    }
    if (py::hasattr(componentType, "_cpp_type_name")) {
        std::string cppName = py::str(componentType.attr("_cpp_type_name"));
        if (!cppName.empty())
            return cppName;
    }
    if (py::hasattr(componentType, "__name__")) {
        return py::str(componentType.attr("__name__")).cast<std::string>();
    }
    return {};
}

/**
 * @brief Coordinate space enum (Unity: Space.Self, Space.World).
 */
enum class CoordinateSpace
{
    Self = 0,
    World = 1
};

/**
 * @brief Enum for primitive types that can be created in the scene.
 */
enum class PrimitiveType
{
    Cube,
    Sphere,
    Capsule,
    Cylinder,
    Plane,
    Quad
};

/**
 * @brief Resolve static primitive mesh data (zero-copy reference).
 */
static void GetPrimitiveMeshData(PrimitiveType type, const std::vector<Vertex> *&outVertices,
                                 const std::vector<uint32_t> *&outIndices, const char *&outDefaultName)
{
    switch (type) {
    case PrimitiveType::Cube:
        outVertices = &PrimitiveMeshes::GetCubeVertices();
        outIndices = &PrimitiveMeshes::GetCubeIndices();
        outDefaultName = "Cube";
        break;
    case PrimitiveType::Sphere:
        outVertices = &PrimitiveMeshes::GetSphereVertices();
        outIndices = &PrimitiveMeshes::GetSphereIndices();
        outDefaultName = "Sphere";
        break;
    case PrimitiveType::Capsule:
        outVertices = &PrimitiveMeshes::GetCapsuleVertices();
        outIndices = &PrimitiveMeshes::GetCapsuleIndices();
        outDefaultName = "Capsule";
        break;
    case PrimitiveType::Cylinder:
        outVertices = &PrimitiveMeshes::GetCylinderVertices();
        outIndices = &PrimitiveMeshes::GetCylinderIndices();
        outDefaultName = "Cylinder";
        break;
    case PrimitiveType::Plane:
        outVertices = &PrimitiveMeshes::GetPlaneVertices();
        outIndices = &PrimitiveMeshes::GetPlaneIndices();
        outDefaultName = "Plane";
        break;
    case PrimitiveType::Quad:
        outVertices = &PrimitiveMeshes::GetQuadVertices();
        outIndices = &PrimitiveMeshes::GetQuadIndices();
        outDefaultName = "Quad";
        break;
    }
}

/**
 * @brief Helper function to create a primitive GameObject.
 * Auto-reserves capacity when rapid creation is detected.
 */
static GameObject *CreatePrimitiveObject(Scene *scene, PrimitiveType type, const std::string &name = "")
{
    const std::vector<Vertex> *vertices = nullptr;
    const std::vector<uint32_t> *indices = nullptr;
    const char *defaultName = "Primitive";
    GetPrimitiveMeshData(type, vertices, indices, defaultName);

    const std::string objectName = name.empty() ? defaultName : name;

    // Auto-reserve: when the ECS store is near capacity, pre-allocate a
    // large chunk so subsequent creates don't trigger per-call reallocation.
    auto &ecs = TransformECSStore::Instance();
    const size_t cap = ecs.Capacity();
    const size_t alive = ecs.AliveCount();
    if (alive + 1 >= cap) {
        // Growing: reserve 2× current or at least 1024 extra slots.
        const size_t newCap = std::max(cap * 2, cap + 1024);
        ecs.Reserve(newCap);
        scene->ReserveCapacity(newCap);
        // Reserve component registry (~3 components per GO)
        Component::ReserveRegistry(newCap * 3);
        // Reserve renderer containers (1 MeshRenderer per primitive)
        SceneManager::Instance().ReserveRendererCapacity(newCap);
        // Reserve physics pools (1 collider per GO when physics is used)
        PhysicsECSStore::Instance().ReserveForBulkCreation(newCap);
    }

    GameObject *obj = scene->CreateGameObject(objectName);
    if (obj) {
        MeshRenderer *renderer = obj->AddComponent<MeshRenderer>();
        if (renderer) {
            renderer->SetSharedPrimitiveMesh(*vertices, *indices, objectName);
        }
    }
    return obj;
}

/**
 * @brief Batch-create N primitive GameObjects with pre-reserved capacity.
 * Returns a Python list of GameObjects.
 */
static py::list CreatePrimitiveObjectsBatch(Scene *scene, PrimitiveType type, size_t count,
                                            const std::string &namePrefix = "")
{
    const std::vector<Vertex> *vertices = nullptr;
    const std::vector<uint32_t> *indices = nullptr;
    const char *defaultName = "Primitive";
    GetPrimitiveMeshData(type, vertices, indices, defaultName);

    const std::string prefix = namePrefix.empty() ? defaultName : namePrefix;

    // Pre-allocate capacity to avoid incremental vector growth.
    scene->ReserveCapacity(count);
    TransformECSStore::Instance().Reserve(TransformECSStore::Instance().Capacity() + count);
    Component::ReserveRegistry(count * 3);
    SceneManager::Instance().ReserveRendererCapacity(count);
    PhysicsECSStore::Instance().ReserveForBulkCreation(count);

    py::list result(count);
    for (size_t i = 0; i < count; ++i) {
        std::string objName = prefix + "_" + std::to_string(i);
        GameObject *obj = scene->CreateGameObject(objName);
        if (obj) {
            MeshRenderer *renderer = obj->AddComponent<MeshRenderer>();
            if (renderer) {
                renderer->SetSharedPrimitiveMesh(*vertices, *indices, prefix);
            }
        }
        result[i] = py::cast(obj, py::return_value_policy::reference);
    }
    return result;
}

/**
 * @brief Helper function to create a GameObject from a mesh asset GUID.
 */
/// Apply FBX-extracted material data to a MeshRenderer's slots.
static void ApplyFbxMaterialData(MeshRenderer *renderer, const std::shared_ptr<InxMesh> &mesh)
{
    if (!renderer || !mesh)
        return;
    const auto &slotData = mesh->GetMaterialSlotData();
    const auto &slotNames = mesh->GetMaterialSlotNames();
    if (slotData.empty())
        return;
    auto defaultMat = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
    if (!defaultMat)
        return;

    uint32_t slotCount = static_cast<uint32_t>(renderer->GetMaterialGuids().size());
    for (uint32_t s = 0; s < slotCount && s < static_cast<uint32_t>(slotData.size()); ++s) {
        // Skip slots that already have an assigned material
        if (renderer->GetMaterial(s))
            continue;
        const auto &sd = slotData[s];
        auto mat = defaultMat->Clone();
        mat->SetColor("baseColor", sd.baseColor);
        mat->SetColor("emissionColor", sd.emissionColor);
        mat->SetFloat("metallic", sd.metallic);
        mat->SetFloat("smoothness", sd.smoothness);
        if (s < slotNames.size() && !slotNames[s].empty())
            mat->SetName(slotNames[s]);
        else
            mat->SetName("EmbeddedMaterial_" + std::to_string(s));
        if (!mesh->GetFilePath().empty())
            mat->SetFilePath(mesh->GetFilePath() + "::submat:" + std::to_string(s));
        renderer->SetMaterial(s, std::move(mat));
    }
}

static std::string TrimCopy(std::string s)
{
    auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
    while (!s.empty() && !notSpace(static_cast<unsigned char>(s.front())))
        s.erase(s.begin());
    while (!s.empty() && !notSpace(static_cast<unsigned char>(s.back())))
        s.pop_back();
    return s;
}

static std::vector<std::string> SplitCommaList(const std::string &csv)
{
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = TrimCopy(item);
        if (!item.empty())
            out.push_back(item);
    }
    return out;
}

static const InxResourceMeta *GetModelMeta(const std::string &guid, const std::shared_ptr<InxMesh> &mesh)
{
    auto *adb = AssetRegistry::Instance().GetAssetDatabase();
    if (!adb)
        return nullptr;
    if (!guid.empty()) {
        if (const auto *meta = adb->GetMetaByGuid(guid))
            return meta;
    }
    if (mesh && !mesh->GetFilePath().empty())
        return adb->GetMetaByPath(mesh->GetFilePath());
    return nullptr;
}

static int TryGetMetaInt(const InxResourceMeta *meta, const std::string &key, int fallback = 0)
{
    if (!meta || !meta->HasKey(key))
        return fallback;
    try {
        return meta->GetDataAs<int>(key);
    } catch (...) {
        return fallback;
    }
}

static std::string TryGetMetaString(const InxResourceMeta *meta, const std::string &key)
{
    if (!meta || !meta->HasKey(key))
        return {};
    try {
        return meta->GetDataAs<std::string>(key);
    } catch (...) {
        return {};
    }
}

static std::vector<std::string> GetAnimationTakeNames(const std::string &guid, const std::shared_ptr<InxMesh> &mesh)
{
    return SplitCommaList(TryGetMetaString(GetModelMeta(guid, mesh), "animation_names_csv"));
}

static bool ShouldUseSkinnedRenderer(const std::string &guid, const std::shared_ptr<InxMesh> &mesh)
{
    const auto *meta = GetModelMeta(guid, mesh);
    if (!meta)
        return false;
    if (TryGetMetaInt(meta, "animation_count", 0) > 0)
        return true;
    return !GetAnimationTakeNames(guid, mesh).empty();
}

static GameObject *CreateModelObject(Scene *scene, const std::string &guid, const std::string &name = "")
{
    auto &registry = AssetRegistry::Instance();

    auto mesh = registry.LoadAsset<InxMesh>(guid, ResourceType::Mesh);
    if (!mesh)
        return nullptr;

    std::string objName = name.empty() ? mesh->GetName() : name;
    if (objName.empty())
        objName = "Mesh Object";

    uint32_t nodeGroupCount = mesh->GetNodeGroupCount();
    const auto &nodeNames = mesh->GetNodeNames();
    const bool useSkinnedRenderer = ShouldUseSkinnedRenderer(guid, mesh);
    const auto animationTakeNames = GetAnimationTakeNames(guid, mesh);

    if (nodeGroupCount <= 1) {
        // Single node — one object with the mesh asset.
        GameObject *obj = scene->CreateGameObject(objName);
        if (!obj)
            return nullptr;
        if (useSkinnedRenderer) {
            auto *renderer = obj->AddComponent<SkinnedMeshRenderer>();
            if (renderer) {
                renderer->SetSourceModelGuid(guid);
                renderer->SetSourceModelPath(mesh->GetFilePath());
                renderer->SetAnimationTakeNames(animationTakeNames);
                renderer->SetMeshAsset(guid, mesh);
                ApplyFbxMaterialData(renderer, mesh);
            }
        } else {
            MeshRenderer *renderer = obj->AddComponent<MeshRenderer>();
            if (renderer) {
                renderer->SetMeshAsset(guid, mesh);
                ApplyFbxMaterialData(renderer, mesh);
            }
        }
        return obj;
    }

    // Multiple nodes — container + one child per node group.
    GameObject *container = scene->CreateGameObject(objName);
    if (!container)
        return nullptr;

    for (uint32_t g = 0; g < nodeGroupCount; ++g) {
        std::string childName =
            (g < nodeNames.size() && !nodeNames[g].empty()) ? nodeNames[g] : "Node_" + std::to_string(g);
        GameObject *child = scene->CreateGameObject(childName);
        if (!child)
            continue;
        child->GetTransform()->SetParent(container->GetTransform());
        if (useSkinnedRenderer) {
            auto *renderer = child->AddComponent<SkinnedMeshRenderer>();
            if (renderer) {
                renderer->SetSourceModelGuid(guid);
                renderer->SetSourceModelPath(mesh->GetFilePath());
                renderer->SetAnimationTakeNames(animationTakeNames);
                renderer->SetMeshAsset(guid, mesh);
                renderer->SetNodeGroup(static_cast<int32_t>(g));
                ApplyFbxMaterialData(renderer, mesh);
            }
        } else {
            MeshRenderer *renderer = child->AddComponent<MeshRenderer>();
            if (renderer) {
                renderer->SetMeshAsset(guid, mesh);
                renderer->SetNodeGroup(static_cast<int32_t>(g));
                ApplyFbxMaterialData(renderer, mesh);
            }
        }
    }

    return container;
}

void RegisterSceneBindings(py::module_ &m)
{
    // ========================================================================
    // PrimitiveType enum
    // ========================================================================
    py::enum_<PrimitiveType>(m, "PrimitiveType")
        .value("Cube", PrimitiveType::Cube)
        .value("Sphere", PrimitiveType::Sphere)
        .value("Capsule", PrimitiveType::Capsule)
        .value("Cylinder", PrimitiveType::Cylinder)
        .value("Plane", PrimitiveType::Plane)
        .value("Quad", PrimitiveType::Quad)
        .export_values();

    // ========================================================================
    // Space enum (Unity: Space.Self, Space.World)
    // ========================================================================
    py::enum_<CoordinateSpace>(m, "Space")
        .value("Self", CoordinateSpace::Self)
        .value("World", CoordinateSpace::World)
        .export_values();

    // ========================================================================
    // Component binding
    // ========================================================================
    py::class_<Component>(m, "Component")
        .def_property_readonly("type_name", &Component::GetTypeName)
        .def_property_readonly("component_id", &Component::GetComponentID)
        .def_property("enabled", &Component::IsEnabled, &Component::SetEnabled)
        .def_property("execution_order", &Component::GetExecutionOrder, &Component::SetExecutionOrder)
        .def_property_readonly(
            "game_object", [](Component *c) { return c->GetGameObject(); }, py::return_value_policy::reference,
            "Get the GameObject this component is attached to")
        .def("serialize", &Component::Serialize, "Serialize component to JSON string")
        .def("deserialize", &Component::Deserialize, py::arg("json_str"), "Deserialize component from JSON string")
        .def_property_readonly("required_component_types", &Component::GetRequiredComponentTypes,
                               "List of type names this component depends on (RequireComponent)")
        .def("is_component_type", &Component::IsComponentType, py::arg("type_name"),
             "Check if this component matches a given type name (including base types)");

    // ========================================================================
    // Transform binding — aligned with Unity convention:
    //   position / euler_angles   → world space
    //   local_position / local_euler_angles / local_scale → local space
    // ========================================================================
    py::class_<Transform, Component>(m, "Transform")
        // ---- World-space properties (Unity: transform.position) ----
        .def_property(
            "position", [](Transform *t) { return t->GetWorldPosition(); },
            [](Transform *t, const glm::vec3 &v) { t->SetWorldPosition(v.x, v.y, v.z); },
            "Position in world space (considering parent hierarchy)")
        .def_property(
            "euler_angles", [](Transform *t) { return t->GetWorldEulerAngles(); },
            [](Transform *t, const glm::vec3 &v) { t->SetWorldEulerAngles(v); },
            "Rotation as Euler angles (degrees) in world space")
        // ---- Local-space properties (Unity: transform.localPosition) ----
        .def_property(
            "local_position", [](Transform *t) { return t->GetLocalPosition(); },
            [](Transform *t, const glm::vec3 &v) { t->SetLocalPosition(v.x, v.y, v.z); },
            "Position in local (parent) space")
        .def_property(
            "local_euler_angles", [](Transform *t) { return t->GetLocalEulerAngles(); },
            [](Transform *t, const glm::vec3 &v) { t->SetLocalEulerAngles(v); },
            "Rotation as Euler angles (degrees) in local space")
        .def_property(
            "local_scale", [](Transform *t) { return t->GetLocalScale(); },
            [](Transform *t, const glm::vec3 &v) { t->SetLocalScale(v.x, v.y, v.z); }, "Scale in local space")
        .def_property_readonly(
            "lossy_scale", [](Transform *t) { return t->GetWorldScale(); },
            "Approximate world-space scale (read-only, like Unity lossyScale)")
        // ---- Direction vectors ----
        .def_property_readonly(
            "forward", [](Transform *t) { return t->GetWorldForward(); },
            "Forward direction in world space (positive Z)")
        .def_property_readonly(
            "right", [](Transform *t) { return t->GetWorldRight(); }, "Right direction in world space (positive X)")
        .def_property_readonly(
            "up", [](Transform *t) { return t->GetWorldUp(); }, "Up direction in world space (positive Y)")
        .def_property_readonly(
            "local_forward", [](Transform *t) { return t->GetLocalForward(); },
            "Forward direction in local space (positive Z)")
        .def_property_readonly(
            "local_right", [](Transform *t) { return t->GetLocalRight(); },
            "Right direction in local space (positive X)")
        .def_property_readonly(
            "local_up", [](Transform *t) { return t->GetLocalUp(); }, "Up direction in local space (positive Y)")
        // ---- Methods ----
        .def(
            "look_at", [](Transform *t, const glm::vec3 &target) { t->LookAt(target); }, py::arg("target"),
            "Rotate to face a world-space target position")
        .def(
            "translate",
            [](Transform *t, const glm::vec3 &delta, int space) {
                if (space == static_cast<int>(CoordinateSpace::Self)) {
                    t->TranslateLocal(delta);
                } else {
                    t->Translate(delta);
                }
            },
            py::arg("delta"), py::arg("space") = static_cast<int>(CoordinateSpace::Self),
            "Translate by delta. space: Space.Self (default, local axes) or Space.World")
        .def(
            "translate_local", [](Transform *t, const glm::vec3 &delta) { t->TranslateLocal(delta); }, py::arg("delta"),
            "Translate in local space (alias for translate(delta, Space.Self))")
        // ---- Quaternion rotation (Unity: transform.rotation / transform.localRotation) ----
        .def_property(
            "rotation", [](Transform *t) { return t->GetWorldRotation(); },
            [](Transform *t, const glm::quat &q) { t->SetWorldRotation(q); }, "World-space rotation as quaternion")
        .def_property(
            "local_rotation", [](Transform *t) { return t->GetLocalRotation(); },
            [](Transform *t, const glm::quat &q) { t->SetLocalRotation(q); }, "Local-space rotation as quaternion")
        // ---- Hierarchy (Unity: transform.parent, transform.root, etc.) ----
        .def_property(
            "parent", [](Transform *t) { return t->GetParent(); },
            [](Transform *t, Transform *parent) { t->SetParent(parent); }, py::return_value_policy::reference,
            "Parent Transform (None if root). Unity: transform.parent")
        .def_property_readonly(
            "root", [](Transform *t) { return t->GetRoot(); }, py::return_value_policy::reference,
            "Topmost Transform in the hierarchy. Unity: transform.root")
        .def_property_readonly(
            "child_count", [](Transform *t) { return t->GetChildCount(); },
            "Number of children. Unity: transform.childCount")
        .def(
            "set_parent",
            [](Transform *t, Transform *parent, bool worldPositionStays) { t->SetParent(parent, worldPositionStays); },
            py::arg("parent"), py::arg("world_position_stays") = true,
            "Set parent Transform. Unity: transform.SetParent(parent, worldPositionStays)")
        .def(
            "get_child", [](Transform *t, int index) { return t->GetChild(static_cast<size_t>(index)); },
            py::return_value_policy::reference, py::arg("index"),
            "Get child Transform by index. Unity: transform.GetChild(index)")
        .def(
            "find", [](Transform *t, const std::string &name) { return t->Find(name); },
            py::return_value_policy::reference, py::arg("name"),
            "Find child Transform by name (non-recursive). Unity: transform.Find(name)")
        .def("detach_children", &Transform::DetachChildren, "Unparent all children. Unity: transform.DetachChildren()")
        .def(
            "is_child_of", [](Transform *t, Transform *parent) { return t->IsChildOf(parent); }, py::arg("parent"),
            "Is this transform a child of parent? Unity: transform.IsChildOf(parent)")
        .def("get_sibling_index", &Transform::GetSiblingIndex, "Get sibling index. Unity: transform.GetSiblingIndex()")
        .def("set_sibling_index", &Transform::SetSiblingIndex, py::arg("index"),
             "Set sibling index. Unity: transform.SetSiblingIndex(index)")
        .def("set_as_first_sibling", &Transform::SetAsFirstSibling,
             "Move to first sibling. Unity: transform.SetAsFirstSibling()")
        .def("set_as_last_sibling", &Transform::SetAsLastSibling,
             "Move to last sibling. Unity: transform.SetAsLastSibling()")
        // ---- Space conversion (Unity: TransformPoint, InverseTransformPoint, etc.) ----
        .def(
            "transform_point", [](Transform *t, const glm::vec3 &p) { return t->TransformPoint(p); }, py::arg("point"),
            "Transform point from local to world space")
        .def(
            "inverse_transform_point", [](Transform *t, const glm::vec3 &p) { return t->InverseTransformPoint(p); },
            py::arg("point"), "Transform point from world to local space")
        .def(
            "transform_direction", [](Transform *t, const glm::vec3 &d) { return t->TransformDirection(d); },
            py::arg("direction"), "Transform direction from local to world space (rotation only)")
        .def(
            "inverse_transform_direction",
            [](Transform *t, const glm::vec3 &d) { return t->InverseTransformDirection(d); }, py::arg("direction"),
            "Transform direction from world to local space (rotation only)")
        .def(
            "transform_vector", [](Transform *t, const glm::vec3 &v) { return t->TransformVector(v); },
            py::arg("vector"), "Transform vector from local to world space (with scale)")
        .def(
            "inverse_transform_vector", [](Transform *t, const glm::vec3 &v) { return t->InverseTransformVector(v); },
            py::arg("vector"), "Transform vector from world to local space (with scale)")
        // ---- Matrices ----
        .def(
            "local_to_world_matrix",
            [](Transform *t) {
                auto m = t->GetLocalToWorldMatrix();
                // Return as list of 16 floats (column-major for GLM)
                py::list result;
                const float *data = &m[0][0];
                for (int i = 0; i < 16; ++i)
                    result.append(data[i]);
                return result;
            },
            "Get the local-to-world transformation matrix (16 floats, column-major)")
        .def(
            "world_to_local_matrix",
            [](Transform *t) {
                auto m = t->GetWorldToLocalMatrix();
                py::list result;
                const float *data = &m[0][0];
                for (int i = 0; i < 16; ++i)
                    result.append(data[i]);
                return result;
            },
            "Get the world-to-local transformation matrix (16 floats, column-major)")
        // ---- Additional rotation methods ----
        .def(
            "rotate",
            [](Transform *t, const glm::vec3 &euler, int space) {
                if (space == static_cast<int>(CoordinateSpace::Self)) {
                    t->Rotate(euler);
                } else {
                    // YXZ intrinsic: q = qY * qX * qZ (Unity convention)
                    glm::vec3 r = glm::radians(euler);
                    float cx = std::cos(r.x * 0.5f), sx = std::sin(r.x * 0.5f);
                    float cy = std::cos(r.y * 0.5f), sy = std::sin(r.y * 0.5f);
                    float cz = std::cos(r.z * 0.5f), sz = std::sin(r.z * 0.5f);
                    glm::quat deltaRot;
                    deltaRot.w = cy * cx * cz + sy * sx * sz;
                    deltaRot.x = cy * sx * cz + sy * cx * sz;
                    deltaRot.y = sy * cx * cz - cy * sx * sz;
                    deltaRot.z = cy * cx * sz - sy * sx * cz;
                    t->SetWorldRotation(deltaRot * t->GetWorldRotation());
                }
            },
            py::arg("euler"), py::arg("space") = static_cast<int>(CoordinateSpace::Self),
            "Rotate by Euler angles (degrees). space: Space.Self (default) or Space.World")
        .def(
            "rotate_around",
            [](Transform *t, const glm::vec3 &point, const glm::vec3 &axis, float angle) {
                t->RotateAround(point, axis, angle);
            },
            py::arg("point"), py::arg("axis"), py::arg("angle"),
            "Rotate around a world-space point. Unity: transform.RotateAround(point, axis, angle)")
        // ---- hasChanged (Unity: transform.hasChanged) ----
        .def_property(
            "has_changed", [](Transform *t) { return t->HasChanged(); },
            [](Transform *t, bool value) { t->SetHasChanged(value); },
            "Has the transform changed since last reset? Unity: transform.hasChanged");

    // ========================================================================
    // MeshRenderer binding
    // ========================================================================
    py::class_<MeshRenderer, Component>(m, "MeshRenderer")
        .def("has_inline_mesh", &MeshRenderer::HasInlineMesh)
        .def_property("inline_mesh_name", &MeshRenderer::GetInlineMeshName, &MeshRenderer::SetInlineMeshName,
                      "Display name for inline (primitive) meshes, e.g. 'Cube', 'Sphere'")
        .def(
            "get_effective_material",
            [](const MeshRenderer &mr, uint32_t slot) { return mr.GetEffectiveMaterial(slot); }, py::arg("slot") = 0,
            "Get the effective material for a given slot (custom or default)")
        // Multi-material API
        .def_property_readonly(
            "material_count",
            [](const MeshRenderer &mr) -> uint32_t { return static_cast<uint32_t>(mr.GetMaterialGuids().size()); },
            "Number of material slots")
        .def(
            "get_material", [](const MeshRenderer &mr, uint32_t slot) { return mr.GetMaterial(slot); }, py::arg("slot"),
            "Get material at slot index")
        .def(
            "set_material",
            [](MeshRenderer &mr, uint32_t slot, py::object material) {
                if (material.is_none()) {
                    mr.SetMaterial(slot, std::string{});
                    return;
                }
                if (py::isinstance<py::str>(material)) {
                    mr.SetMaterial(slot, material.cast<std::string>());
                    return;
                }
                try {
                    mr.SetMaterial(slot, material.cast<std::shared_ptr<InxMaterial>>());
                    return;
                } catch (const py::cast_error &) {
                }
                throw py::type_error("set_material expects a material GUID string, InxMaterial, or None");
            },
            py::arg("slot"), py::arg("material"), "Set material at slot index by GUID, material object, or None")
        .def(
            "get_material_guids", [](const MeshRenderer &mr) { return mr.GetMaterialGuids(); },
            "Get all material slot GUIDs as a list")
        .def(
            "set_materials", [](MeshRenderer &mr, const std::vector<std::string> &guids) { mr.SetMaterials(guids); },
            py::arg("guids"), "Set all material slots from a list of GUIDs")
        .def(
            "set_material_slot_count", [](MeshRenderer &mr, uint32_t count) { mr.SetMaterialSlotCount(count); },
            py::arg("count"), "Set the number of material slots")
        .def("serialize", &MeshRenderer::Serialize, "Serialize MeshRenderer to JSON string")
        .def(
            "set_primitive_mesh",
            [](MeshRenderer &mr, PrimitiveType type) {
                const std::vector<Vertex> *vertices = nullptr;
                const std::vector<uint32_t> *indices = nullptr;
                const char *defaultName = "Primitive";
                GetPrimitiveMeshData(type, vertices, indices, defaultName);
                mr.SetSharedPrimitiveMesh(*vertices, *indices, defaultName);
            },
            py::arg("type"), "Set the mesh to a built-in primitive (Cube, Sphere, Quad, etc.)")

        // ====================================================================
        // Mesh data access for scripting and inspection tools
        // ====================================================================
        .def_property_readonly(
            "vertex_count",
            [](const MeshRenderer &mr) -> size_t {
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetMeshAssetRef().Get();
                    return m ? m->GetVertexCount() : 0;
                }
                return mr.HasInlineMesh() ? mr.GetInlineVertices().size() : 0;
            },
            "Number of vertices in the mesh")
        .def_property_readonly(
            "index_count",
            [](const MeshRenderer &mr) -> size_t {
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetMeshAssetRef().Get();
                    return m ? m->GetIndexCount() : 0;
                }
                return mr.HasInlineMesh() ? mr.GetInlineIndices().size() : 0;
            },
            "Number of indices in the mesh")
        .def(
            "get_positions",
            [](const MeshRenderer &mr) -> py::list {
                py::list result;
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetMeshAssetRef().Get();
                    if (m) {
                        for (const auto &v : m->GetVertices())
                            result.append(py::make_tuple(v.pos.x, v.pos.y, v.pos.z));
                    }
                } else if (mr.HasInlineMesh()) {
                    for (const auto &v : mr.GetInlineVertices())
                        result.append(py::make_tuple(v.pos.x, v.pos.y, v.pos.z));
                }
                return result;
            },
            "Get all vertex positions as a list of (x, y, z) tuples")
        .def(
            "get_normals",
            [](const MeshRenderer &mr) -> py::list {
                py::list result;
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetMeshAssetRef().Get();
                    if (m) {
                        for (const auto &v : m->GetVertices())
                            result.append(py::make_tuple(v.normal.x, v.normal.y, v.normal.z));
                    }
                } else if (mr.HasInlineMesh()) {
                    for (const auto &v : mr.GetInlineVertices())
                        result.append(py::make_tuple(v.normal.x, v.normal.y, v.normal.z));
                }
                return result;
            },
            "Get all vertex normals as a list of (x, y, z) tuples")
        .def(
            "get_uvs",
            [](const MeshRenderer &mr) -> py::list {
                py::list result;
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetMeshAssetRef().Get();
                    if (m) {
                        for (const auto &v : m->GetVertices())
                            result.append(py::make_tuple(v.texCoord.x, v.texCoord.y));
                    }
                } else if (mr.HasInlineMesh()) {
                    for (const auto &v : mr.GetInlineVertices())
                        result.append(py::make_tuple(v.texCoord.x, v.texCoord.y));
                }
                return result;
            },
            "Get all vertex UVs as a list of (u, v) tuples")
        .def(
            "get_indices",
            [](const MeshRenderer &mr) -> py::list {
                py::list result;
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetMeshAssetRef().Get();
                    if (m) {
                        for (uint32_t idx : m->GetIndices())
                            result.append(idx);
                    }
                } else if (mr.HasInlineMesh()) {
                    for (uint32_t idx : mr.GetInlineIndices())
                        result.append(idx);
                }
                return result;
            },
            "Get all indices as a flat list")
        .def_property_readonly(
            "mesh_asset_guid", [](const MeshRenderer &mr) -> std::string { return mr.GetMeshAssetGuid(); },
            "GUID of the mesh asset (empty if using inline mesh)")
        .def_property_readonly(
            "has_mesh_asset", [](const MeshRenderer &mr) -> bool { return mr.HasMeshAsset(); },
            "Whether this renderer uses an asset-managed mesh")
        .def_property_readonly(
            "mesh_name",
            [](const MeshRenderer &mr) -> std::string {
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetMeshAssetRef().Get();
                    if (m)
                        return m->GetName();
                }
                return "";
            },
            "Name of the mesh asset (empty if using inline mesh)")
        .def(
            "get_mesh_asset",
            [](const MeshRenderer &mr) -> std::shared_ptr<InxMesh> {
                if (mr.HasMeshAsset())
                    return mr.GetMeshAssetRef().Get();
                return nullptr;
            },
            "Get the InxMesh asset object (None if no asset mesh)")
        .def_property("casts_shadows", &MeshRenderer::CastsShadows, &MeshRenderer::SetCastShadows,
                      "Whether this renderer casts shadows")
        .def_property("receives_shadows", &MeshRenderer::ReceivesShadows, &MeshRenderer::SetReceivesShadows,
                      "Whether this renderer receives shadows")
        .def_property("submesh_index", &MeshRenderer::GetSubmeshIndex, &MeshRenderer::SetSubmeshIndex,
                      "Submesh index to render (-1 = all, >= 0 = specific submesh)")
        .def_property(
            "mesh_pivot_offset", [](const MeshRenderer &mr) -> glm::vec3 { return mr.GetMeshPivotOffset(); },
            [](MeshRenderer &mr, const glm::vec3 &v) { mr.SetMeshPivotOffset(v); },
            "Pivot offset to re-center submesh geometry around the transform")
        .def(
            "get_world_bounds",
            [](const MeshRenderer &mr) -> py::tuple {
                glm::vec3 outMin, outMax;
                mr.GetWorldBounds(outMin, outMax);
                return py::make_tuple(outMin.x, outMin.y, outMin.z, outMax.x, outMax.y, outMax.z);
            },
            "Get world-space AABB as (min_x, min_y, min_z, max_x, max_y, max_z)");

    // ========================================================================
    // SkinnedMeshRenderer — animated model placeholder, inherits MeshRenderer
    // ========================================================================
    py::class_<SkinnedMeshRenderer, MeshRenderer>(m, "SkinnedMeshRenderer")
        .def(py::init<>())
        .def_property_readonly("source_model_guid", &SkinnedMeshRenderer::GetSourceModelGuid,
                               "GUID of the animated source model asset")
        .def_property_readonly("source_model_path", &SkinnedMeshRenderer::GetSourceModelPath,
                               "Filesystem path of the animated source model")
        .def_property("active_take_name", &SkinnedMeshRenderer::GetActiveTakeName, &SkinnedMeshRenderer::SetActiveTakeName,
                      "Currently selected animation take name")
        .def_property_readonly("has_animation_takes", &SkinnedMeshRenderer::HasAnimationTakes,
                               "Whether this renderer has any imported animation takes")
        .def_property_readonly(
            "animation_take_count",
            [](const SkinnedMeshRenderer &sr) -> size_t { return sr.GetAnimationTakeNames().size(); },
            "Number of imported animation takes on the source model")
        .def(
            "get_animation_take_names",
            [](const SkinnedMeshRenderer &sr) { return sr.GetAnimationTakeNames(); },
            "Get imported animation take names from the source model")
        .def("get_animation_duration_seconds", &SkinnedMeshRenderer::GetAnimationDurationSeconds, py::arg("take_name"),
             "Get imported animation take duration in seconds")
        .def("submit_animation_pose", &SkinnedMeshRenderer::SubmitAnimationPose, py::arg("take_name"),
             py::arg("time_seconds"), py::arg("normalized_time"), py::arg("blend_take_name") = "",
             py::arg("blend_time_seconds") = 0.0f, py::arg("blend_weight") = 0.0f,
             "Submit active and blend animation state in one native call")
        .def_property("runtime_animation_time", &SkinnedMeshRenderer::GetRuntimeAnimationTime,
                      &SkinnedMeshRenderer::SetRuntimeAnimationTime,
                      "Current clip time in seconds (runtime; driven by SkeletalAnimator)")
        .def_property("runtime_animation_normalized_time", &SkinnedMeshRenderer::GetRuntimeAnimationNormalizedTime,
                      &SkinnedMeshRenderer::SetRuntimeAnimationNormalizedTime,
                      "Normalized clip time 0..1 (runtime; driven when duration is known)")
        .def_property("blend_take_name", &SkinnedMeshRenderer::GetBlendTakeName, &SkinnedMeshRenderer::SetBlendTakeName,
                      "Secondary animation take name used for runtime pose blending")
        .def_property("blend_animation_time", &SkinnedMeshRenderer::GetBlendAnimationTime,
                      &SkinnedMeshRenderer::SetBlendAnimationTime,
                      "Secondary animation time in seconds used for runtime pose blending")
        .def_property("blend_weight", &SkinnedMeshRenderer::GetBlendWeight, &SkinnedMeshRenderer::SetBlendWeight,
                      "Runtime pose blend weight from active take to blend take")
        .def("clear_animation_blend", &SkinnedMeshRenderer::ClearAnimationBlend, "Clear runtime pose blending state");

    // ========================================================================
    // SpriteRenderer — inherits MeshRenderer for rendering, adds sprite props
    // ========================================================================
    py::class_<SpriteRenderer, MeshRenderer>(m, "SpriteRenderer")
        .def(py::init<>())
        .def_property("sprite_guid", &SpriteRenderer::GetSpriteGuid, &SpriteRenderer::SetSpriteGuid,
                      "Asset GUID of the sprite texture")
        .def_property("frame_index", &SpriteRenderer::GetFrameIndex, &SpriteRenderer::SetFrameIndex,
                      "Index of the sprite frame to display")
        .def_property(
            "sprite_color",
            [](const SpriteRenderer &sr) -> py::tuple {
                const auto &c = sr.GetColor();
                return py::make_tuple(c.r, c.g, c.b, c.a);
            },
            [](SpriteRenderer &sr, const py::tuple &t) {
                sr.SetColor(glm::vec4(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(), t[3].cast<float>()));
            },
            "Sprite tint color (r, g, b, a)")
        .def_property("flip_x", &SpriteRenderer::GetFlipX, &SpriteRenderer::SetFlipX, "Flip sprite horizontally")
        .def_property("flip_y", &SpriteRenderer::GetFlipY, &SpriteRenderer::SetFlipY, "Flip sprite vertically");

    // ========================================================================
    // LightType enum (matches Unity)
    // ========================================================================
    py::enum_<LightType>(m, "LightType")
        .value("Directional", LightType::Directional)
        .value("Point", LightType::Point)
        .value("Spot", LightType::Spot)
        .value("Area", LightType::Area)
        .export_values();

    py::enum_<LightShadows>(m, "LightShadows")
        .value("NoShadows", LightShadows::None)
        .value("Hard", LightShadows::Hard)
        .value("Soft", LightShadows::Soft)
        .export_values();

    // ========================================================================
    // Light component binding (Unity-like API)
    // ========================================================================
    py::class_<Light, Component>(m, "Light")
        // Light type
        .def_property("light_type", &Light::GetLightType, &Light::SetLightType,
                      "Type of light (Directional, Point, Spot, Area)")

        // Color & intensity (Unity-style)
        .def_property(
            "color", [](Light *l) { return glm::vec3(l->GetColor()); },
            [](Light *l, const glm::vec3 &v) { l->SetColor(v.x, v.y, v.z); }, "Light color (linear RGB)")
        .def_property("intensity", &Light::GetIntensity, &Light::SetIntensity, "Light intensity multiplier")

        // Range (Point/Spot)
        .def_property("range", &Light::GetRange, &Light::SetRange, "Light range (Point/Spot lights)")

        // Spot angle (Spot only)
        .def_property("spot_angle", &Light::GetSpotAngle, &Light::SetSpotAngle, "Inner spot angle in degrees")
        .def_property("outer_spot_angle", &Light::GetOuterSpotAngle, &Light::SetOuterSpotAngle,
                      "Outer spot angle in degrees")

        // Shadows
        .def_property("shadows", &Light::GetShadows, &Light::SetShadows, "Shadow type (None, Hard, Soft)")
        .def_property("shadow_strength", &Light::GetShadowStrength, &Light::SetShadowStrength, "Shadow strength (0-1)")
        .def_property("shadow_bias", &Light::GetShadowBias, &Light::SetShadowBias, "Shadow depth bias")

        // Shadow mapping matrices
        .def("get_light_view_matrix", &Light::GetLightViewMatrix, "Get the light's view matrix for shadow mapping")
        .def("get_light_projection_matrix", &Light::GetLightProjectionMatrix, py::arg("shadow_extent") = 20.0f,
             py::arg("near_plane") = 0.1f, py::arg("far_plane") = 100.0f,
             "Get the light's projection matrix for shadow mapping")

        // Serialization
        .def("serialize", &Light::Serialize, "Serialize Light to JSON string");

    // ========================================================================
    // PyComponentProxy binding (for Python-defined components)
    // ========================================================================
    py::class_<PyComponentProxy, Component>(m, "PyComponentProxy")
        .def("get_py_component", &PyComponentProxy::GetPyComponent, "Get the underlying Python component")
        .def("get_py_type_name", &PyComponentProxy::GetPyTypeName, "Get the Python type name")
        .def("is_valid", &PyComponentProxy::IsValid, "Check if this proxy holds a valid Python component");

    // ========================================================================
    // CameraProjection enum
    // ========================================================================
    py::enum_<CameraProjection>(m, "CameraProjection")
        .value("Perspective", CameraProjection::Perspective)
        .value("Orthographic", CameraProjection::Orthographic)
        .export_values();

    // ========================================================================
    // CameraClearFlags enum
    // ========================================================================
    py::enum_<CameraClearFlags>(m, "CameraClearFlags")
        .value("Skybox", CameraClearFlags::Skybox)
        .value("SolidColor", CameraClearFlags::SolidColor)
        .value("DepthOnly", CameraClearFlags::DepthOnly)
        .value("DontClear", CameraClearFlags::DontClear)
        .export_values();

    // ========================================================================
    // Camera component binding (Unity-like API)
    // ========================================================================
    py::class_<Camera, Component>(m, "Camera")
        // Projection mode
        .def_property("projection_mode", &Camera::GetProjectionMode, &Camera::SetProjectionMode,
                      "Camera projection mode (Perspective or Orthographic)")
        // Perspective settings
        .def_property("field_of_view", &Camera::GetFieldOfView, &Camera::SetFieldOfView,
                      "Field of view in degrees (Perspective mode)")
        .def_property("aspect_ratio", &Camera::GetAspectRatio, &Camera::SetAspectRatio, "Aspect ratio (width/height)")
        // Orthographic settings
        .def_property("orthographic_size", &Camera::GetOrthographicSize, &Camera::SetOrthographicSize,
                      "Orthographic half-height (Orthographic mode)")
        // Clipping planes
        .def_property("near_clip", &Camera::GetNearClip, &Camera::SetNearClip, "Near clipping plane distance")
        .def_property("far_clip", &Camera::GetFarClip, &Camera::SetFarClip, "Far clipping plane distance")
        // Multi-camera support
        .def_property("depth", &Camera::GetDepth, &Camera::SetDepth,
                      "Rendering depth (lower depth renders first, like Unity Camera.depth)")
        .def_property("culling_mask", &Camera::GetCullingMask, &Camera::SetCullingMask,
                      "Layer culling bitmask (which layers this camera renders)")
        // Clear flags & background color
        .def_property("clear_flags", &Camera::GetClearFlags, &Camera::SetClearFlags,
                      "Camera clear flags (Skybox, SolidColor, DepthOnly, DontClear)")
        .def_property(
            "background_color", [](const Camera &c) -> glm::vec4 { return c.GetBackgroundColor(); },
            [](Camera &c, const glm::vec4 &v) { c.SetBackgroundColor(v); },
            "Background color as vec4f (r, g, b, a) — used when clear_flags == SolidColor")
        // Screen dimensions (read-only, set by renderer)
        .def_property_readonly("pixel_width", &Camera::GetPixelWidth, "Render target width in pixels")
        .def_property_readonly("pixel_height", &Camera::GetPixelHeight, "Render target height in pixels")
        // Coordinate conversion
        .def(
            "screen_to_world_point",
            [](const Camera &c, float x, float y, float depth) { return c.ScreenToWorldPoint(glm::vec2(x, y), depth); },
            py::arg("x"), py::arg("y"), py::arg("depth") = 0.0f,
            "Convert screen coordinates (x, y) + depth [0..1] to world position")
        .def(
            "world_to_screen_point",
            [](const Camera &c, float x, float y, float z) { return c.WorldToScreenPoint(glm::vec3(x, y, z)); },
            py::arg("x"), py::arg("y"), py::arg("z"), "Convert world position to screen coordinates (x, y)")
        .def(
            "screen_point_to_ray",
            [](const Camera &c, float x, float y) -> py::tuple {
                auto [origin, dir] = c.ScreenPointToRay(glm::vec2(x, y));
                return py::make_tuple(origin, dir);
            },
            py::arg("x"), py::arg("y"),
            "Build a ray from viewport-relative screen coordinates. "
            "Returns (origin_Vector3, direction_Vector3) — origin at near plane and normalised direction.")
        // Serialization
        .def("serialize", &Camera::Serialize, "Serialize Camera to JSON string")
        .def("deserialize", &Camera::Deserialize, py::arg("json_str"), "Deserialize Camera from JSON string");

    // ========================================================================
    // Register component type casters (auto-dispatch for add/get_component)
    // ========================================================================
    auto &registry = ComponentBindingRegistry::Instance();
    registry.Register("MeshRenderer", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<MeshRenderer *>(c), py::return_value_policy::reference);
    });
    registry.Register("SkinnedMeshRenderer", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<SkinnedMeshRenderer *>(c), py::return_value_policy::reference);
    });
    registry.Register("Light", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<Light *>(c), py::return_value_policy::reference);
    });
    registry.Register("Camera", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<Camera *>(c), py::return_value_policy::reference);
    });
    registry.Register("PyComponentProxy", [](Component *c) -> py::object {
        auto *proxy = dynamic_cast<PyComponentProxy *>(c);
        if (proxy == nullptr) {
            return py::none();
        }
        py::object pyComponent = proxy->GetPyComponent();
        if (pyComponent.is_none()) {
            return py::none();
        }
        return pyComponent;
    });
    registry.Register("BoxCollider", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<BoxCollider *>(c), py::return_value_policy::reference);
    });
    registry.Register("SphereCollider", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<SphereCollider *>(c), py::return_value_policy::reference);
    });
    registry.Register("CapsuleCollider", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<CapsuleCollider *>(c), py::return_value_policy::reference);
    });
    registry.Register("MeshCollider", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<MeshCollider *>(c), py::return_value_policy::reference);
    });
    registry.Register("Rigidbody", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<Rigidbody *>(c), py::return_value_policy::reference);
    });

    // NOTE: AudioSource and AudioListener casters are registered in BindingAudio.cpp
    // because RegisterAudioBindings runs after RegisterSceneBindings.

    // ========================================================================
    // GameObject binding
    // ========================================================================
    py::class_<GameObject>(m, "GameObject")
        .def_property("name", &GameObject::GetName, &GameObject::SetName)
        .def_property("active", &GameObject::IsActive, &GameObject::SetActive)
        .def_property_readonly("active_self", &GameObject::GetActiveSelf,
                               "Local active state (Unity: gameObject.activeSelf)")
        .def_property_readonly("active_in_hierarchy", &GameObject::IsActiveInHierarchy,
                               "Is active in hierarchy? (Unity: gameObject.activeInHierarchy)")
        .def_property_readonly("id", &GameObject::GetID)
        .def_property("tag", &GameObject::GetTag, &GameObject::SetTag, "Tag string for this GameObject")
        .def_property("layer", &GameObject::GetLayer, &GameObject::SetLayer, "Layer index (0-31) for this GameObject")
        .def_property("is_static", &GameObject::IsStatic, &GameObject::SetStatic,
                      "Static flag for this GameObject (Unity: gameObject.isStatic)")
        .def_property_readonly("is_persistent", &GameObject::IsPersistent,
                               "True if this object is marked DontDestroyOnLoad")
        .def_property("prefab_guid", &GameObject::GetPrefabGuid, &GameObject::SetPrefabGuid,
                      "GUID of the source .prefab asset (empty = not a prefab instance)")
        .def_property("prefab_root", &GameObject::IsPrefabRoot, &GameObject::SetPrefabRoot,
                      "True if this object is the root of a prefab instance hierarchy")
        .def_property_readonly("is_prefab_instance", &GameObject::IsPrefabInstance,
                               "True if this object belongs to a prefab instance")
        .def("compare_tag", &GameObject::CompareTag, py::arg("tag"),
             "Returns true if the GameObject's tag matches the given tag")
        .def_property_readonly(
            "transform", [](GameObject *obj) { return obj->GetTransform(); }, py::return_value_policy::reference,
            "Get the Transform component")
        .def_property_readonly(
            "scene", [](GameObject *obj) { return obj->GetScene(); }, py::return_value_policy::reference,
            "Get the Scene this GameObject belongs to (Unity: gameObject.scene)")
        .def(
            "get_transform", [](GameObject *obj) { return obj->GetTransform(); }, py::return_value_policy::reference,
            "Get the Transform component")
        .def(
            "add_component",
            [](GameObject *obj, py::object componentType) -> py::object {
                std::string typeName = ResolveComponentTypeName(componentType);
                if (typeName.empty()) {
                    return py::none();
                }
                // Try native C++ component first.
                Component *comp = obj->AddComponentByTypeName(typeName);
                if (comp) {
                    return ComponentBindingRegistry::Instance().CastToPython(comp);
                }
                // If native creation failed and the argument is a class (not a
                // string), treat it as a Python InxComponent subclass:
                // instantiate and delegate to add_py_component.
                if (!py::isinstance<py::str>(componentType) && py::isinstance<py::type>(componentType)) {
                    try {
                        py::object instance = componentType();
                        return py::cast(obj, py::return_value_policy::reference).attr("add_py_component")(instance);
                    } catch (py::error_already_set &e) {
                        INXLOG_WARN("[Binding] Failed to instantiate Python component '{}': {}", typeName, e.what());
                        return py::none();
                    }
                }
                return py::none();
            },
            py::arg("component_type"), "Add a component by type, type name, or InxComponent subclass")
        .def(
            "remove_component", [](GameObject *obj, Component *component) { return obj->RemoveComponent(component); },
            py::arg("component"), "Remove a component instance (cannot remove Transform or required components)")
        .def(
            "can_remove_component",
            [](GameObject *obj, Component *component) { return obj->CanRemoveComponent(component); },
            py::arg("component"), "Check if a component can be removed (not blocked by RequireComponent)")
        .def(
            "get_remove_component_blockers",
            [](GameObject *obj, Component *component) { return obj->GetRemovalBlockingComponentTypes(component); },
            py::arg("component"), "Get sibling component type names that block removing the specified component")
        .def(
            "get_components",
            [](GameObject *obj) -> py::list {
                py::list result;
                auto &reg = ComponentBindingRegistry::Instance();
                // Include Transform first (it's not in m_components)
                result.append(py::cast(obj->GetTransform(), py::return_value_policy::reference));
                for (const auto &comp : obj->GetAllComponents()) {
                    py::object pythonComponent = reg.CastToPython(comp.get());
                    if (!pythonComponent.is_none()) {
                        result.append(pythonComponent);
                    }
                }
                return result;
            },
            "Get all components (including Transform)")
        .def(
            "get_component",
            [](GameObject *obj, const std::string &typeName) -> py::object {
                auto &reg = ComponentBindingRegistry::Instance();
                if (typeName == "Transform") {
                    return py::cast(obj->GetTransform(), py::return_value_policy::reference);
                }
                for (const auto &comp : obj->GetAllComponents()) {
                    if (auto *proxy = dynamic_cast<PyComponentProxy *>(comp.get())) {
                        if (proxy->GetPyTypeName() == typeName) {
                            py::object pyComponent = proxy->GetPyComponent();
                            if (!pyComponent.is_none()) {
                                return pyComponent;
                            }
                        }
                        continue;
                    }
                    if (comp->GetTypeName() == typeName) {
                        return reg.CastToPython(comp.get());
                    }
                }
                return py::none();
            },
            py::arg("type_name"), "Get a component by type name (e.g., 'Transform', 'MeshRenderer', 'Light')")
        .def(
            "get_cpp_component",
            [](GameObject *obj, const std::string &typeName) -> py::object {
                auto &reg = ComponentBindingRegistry::Instance();
                // Special case for Transform
                if (typeName == "Transform") {
                    return py::cast(obj->GetTransform(), py::return_value_policy::reference);
                }
                // Search in components by type name
                for (const auto &comp : obj->GetAllComponents()) {
                    if (dynamic_cast<PyComponentProxy *>(comp.get())) {
                        continue;
                    }
                    if (comp->GetTypeName() == typeName) {
                        return reg.CastToPython(comp.get());
                    }
                }
                return py::none();
            },
            py::arg("type_name"), "Get a C++ component by type name (e.g., 'Transform', 'MeshRenderer', 'Light')")
        .def(
            "get_cpp_components",
            [](GameObject *obj, const std::string &typeName) -> py::list {
                py::list result;
                auto &reg = ComponentBindingRegistry::Instance();
                // Special case for Transform
                if (typeName == "Transform") {
                    result.append(py::cast(obj->GetTransform(), py::return_value_policy::reference));
                    return result;
                }
                // Search in components by type name
                for (const auto &comp : obj->GetAllComponents()) {
                    if (dynamic_cast<PyComponentProxy *>(comp.get())) {
                        continue;
                    }
                    if (comp->GetTypeName() == typeName) {
                        result.append(reg.CastToPython(comp.get()));
                    }
                }
                return result;
            },
            py::arg("type_name"), "Get all C++ components of a given type name")
        .def(
            "add_py_component",
            [](GameObject *obj, py::object pyComponentInstance) -> py::object {
                auto hasCppComponent = [&](const std::string &typeName) -> bool {
                    if (typeName == "Transform") {
                        return true;
                    }
                    for (const auto &comp : obj->GetAllComponents()) {
                        if (comp && comp->GetTypeName() == typeName) {
                            return true;
                        }
                    }
                    return false;
                };

                // Check for DisallowMultipleComponent
                py::object pyType = pyComponentInstance.attr("__class__");
                std::string cppTypeName;
                if (py::hasattr(pyType, "_cpp_type_name")) {
                    try {
                        cppTypeName = pyType.attr("_cpp_type_name").cast<std::string>();
                    } catch (...) {
                        INXLOG_WARN("[Binding] Failed to read _cpp_type_name from component type");
                        cppTypeName.clear();
                    }
                }
                bool disallowMultiple = false;
                if (py::hasattr(pyType, "_disallow_multiple_")) {
                    try {
                        disallowMultiple = pyType.attr("_disallow_multiple_").cast<bool>();
                    } catch (...) {
                        INXLOG_WARN("[Binding] Failed to read _disallow_multiple_ from component type");
                    }
                }

                if (disallowMultiple) {
                    if (!cppTypeName.empty()) {
                        if (hasCppComponent(cppTypeName)) {
                            std::string typeName = pyType.attr("__name__").cast<std::string>();
                            py::print("[Warning] Cannot add multiple", typeName,
                                      "components - DisallowMultipleComponent is set");
                            return py::none();
                        }
                    }

                    // Check if component of this type already exists
                    for (const auto &comp : obj->GetAllComponents()) {
                        if (auto *proxy = dynamic_cast<PyComponentProxy *>(comp.get())) {
                            py::object existingComp = proxy->GetPyComponent();
                            if (!existingComp.is_none() && py::isinstance(existingComp, pyType)) {
                                // Duplicate detected - return None with warning
                                std::string typeName = pyType.attr("__name__").cast<std::string>();
                                py::print("[Warning] Cannot add multiple", typeName,
                                          "components - DisallowMultipleComponent is set");
                                return py::none();
                            }
                        }
                    }
                }

                // Check for RequireComponent
                if (py::hasattr(pyType, "_require_components_")) {
                    py::list requiredTypes = pyType.attr("_require_components_").cast<py::list>();
                    for (auto reqType : requiredTypes) {
                        bool found = false;

                        if (py::isinstance<py::str>(reqType)) {
                            std::string reqTypeName = reqType.cast<std::string>();
                            if (hasCppComponent(reqTypeName)) {
                                found = true;
                            } else {
                                found = (obj->AddComponentByTypeName(reqTypeName) != nullptr);
                            }
                            if (!found) {
                                py::print("[Warning] Failed to auto-add required native component", reqTypeName);
                            }
                            continue;
                        } else if (py::hasattr(reqType, "_cpp_type_name")) {
                            std::string reqCppTypeName;
                            try {
                                reqCppTypeName = reqType.attr("_cpp_type_name").cast<std::string>();
                            } catch (...) {
                                INXLOG_WARN("[Binding] Failed to read _cpp_type_name from required component type");
                                reqCppTypeName.clear();
                            }

                            if (!reqCppTypeName.empty()) {
                                if (hasCppComponent(reqCppTypeName)) {
                                    found = true;
                                } else if (obj->AddComponentByTypeName(reqCppTypeName) != nullptr) {
                                    found = true;
                                }
                            }

                            if (!found && !reqCppTypeName.empty()) {
                                py::print("[Warning] Failed to auto-add required native component", reqCppTypeName);
                            }
                            continue;
                        }

                        if (found) {
                            continue;
                        }

                        for (const auto &comp : obj->GetAllComponents()) {
                            if (auto *proxy = dynamic_cast<PyComponentProxy *>(comp.get())) {
                                py::object existingComp = proxy->GetPyComponent();
                                if (!existingComp.is_none() && py::isinstance(existingComp, reqType)) {
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (!found) {
                            std::string typeName = pyType.attr("__name__").cast<std::string>();
                            std::string reqTypeName = py::hasattr(reqType, "__name__")
                                                          ? reqType.attr("__name__").cast<std::string>()
                                                          : std::string("Component");
                            py::print("[Warning] Component", typeName, "requires", reqTypeName,
                                      "- adding it automatically");
                            // Auto-add the required component
                            py::object newReqComp = reqType();
                            auto reqProxy = std::make_unique<PyComponentProxy>(newReqComp);
                            Component *reqAdded = obj->AddExistingComponent(std::move(reqProxy));
                            if (reqAdded && py::hasattr(newReqComp, "_bind_native_component")) {
                                try {
                                    newReqComp.attr("_bind_native_component")(
                                        py::cast(reqAdded, py::return_value_policy::reference),
                                        py::cast(obj, py::return_value_policy::reference));
                                } catch (...) {
                                    INXLOG_WARN("[Binding] Failed to bind required component to native proxy");
                                }
                            }
                        }
                    }
                }

                // Create a PyComponentProxy that wraps the Python component
                auto proxy = std::make_unique<PyComponentProxy>(pyComponentInstance);
                Component *added = obj->AddExistingComponent(std::move(proxy));
                if (added) {
                    // Immediately bind the native proxy and owning GameObject.
                    // This makes the C++ proxy the lifecycle authority from the
                    // moment the component is attached, even before Awake().
                    try {
                        if (py::hasattr(pyComponentInstance, "_bind_native_component")) {
                            pyComponentInstance.attr("_bind_native_component")(
                                py::cast(added, py::return_value_policy::reference),
                                py::cast(obj, py::return_value_policy::reference));
                        } else if (py::hasattr(pyComponentInstance, "_set_game_object")) {
                            pyComponentInstance.attr("_set_game_object")(
                                py::cast(obj, py::return_value_policy::reference));
                        }
                    } catch (...) {
                        INXLOG_WARN("[Binding] Failed to bind newly added component to native proxy");
                    }
                    // Return the original Python component
                    return pyComponentInstance;
                }
                return py::none();
            },
            py::arg("component_instance"), "Add a Python InxComponent instance to this GameObject")
        .def(
            "get_py_component",
            [](GameObject *obj, py::object componentType) -> py::object {
                // Find a PyComponentProxy whose Python component is an instance of the given type
                for (const auto &comp : obj->GetAllComponents()) {
                    if (auto *proxy = dynamic_cast<PyComponentProxy *>(comp.get())) {
                        py::object pyComp = proxy->GetPyComponent();
                        if (!pyComp.is_none() && py::isinstance(pyComp, componentType)) {
                            return pyComp;
                        }
                    }
                }
                return py::none();
            },
            py::arg("component_type"), "Get a Python component of the specified type")
        .def(
            "get_py_components",
            [](GameObject *obj) {
                // Return all Python components
                std::vector<py::object> result;
                for (const auto &comp : obj->GetAllComponents()) {
                    if (auto *proxy = dynamic_cast<PyComponentProxy *>(comp.get())) {
                        py::object pyComp = proxy->GetPyComponent();
                        if (!pyComp.is_none()) {
                            result.push_back(pyComp);
                        }
                    }
                }
                return result;
            },
            "Get all Python components attached to this GameObject")
        .def(
            "remove_py_component",
            [](GameObject *obj, py::object pyComponent) {
                // Find the proxy that wraps this Python component and remove it
                for (const auto &comp : obj->GetAllComponents()) {
                    if (auto *proxy = dynamic_cast<PyComponentProxy *>(comp.get())) {
                        py::object pyComp = proxy->GetPyComponent();
                        if (!pyComp.is_none() && pyComp.is(pyComponent)) {
                            return obj->RemoveComponent(proxy);
                        }
                    }
                }
                return false;
            },
            py::arg("component"), "Remove a Python component instance")
        .def("get_parent", &GameObject::GetParent, py::return_value_policy::reference, "Get the parent GameObject")
        .def("set_parent", &GameObject::SetParent, py::arg("parent"), py::arg("world_position_stays") = true,
             "Set the parent GameObject (None for root). world_position_stays preserves world transform.")
        .def(
            "get_children",
            [](GameObject *obj) {
                std::vector<GameObject *> result;
                for (const auto &child : obj->GetChildren()) {
                    result.push_back(child.get());
                }
                return result;
            },
            py::return_value_policy::reference, "Get list of child GameObjects")
        .def("get_child_count", &GameObject::GetChildCount, "Get the number of children")
        .def(
            "get_child", [](GameObject *obj, int index) { return obj->GetChild(static_cast<size_t>(index)); },
            py::return_value_policy::reference, py::arg("index"), "Get child by index")
        .def("find_child", &GameObject::FindChild, py::return_value_policy::reference, py::arg("name"),
             "Find a child by name (non-recursive)")
        .def("find_descendant", &GameObject::FindDescendant, py::return_value_policy::reference, py::arg("name"),
             "Find a descendant by name (recursive)")
        .def("is_active_in_hierarchy", &GameObject::IsActiveInHierarchy,
             "Check if this object and all parents are active")
        .def("serialize", &GameObject::Serialize, "Serialize GameObject to JSON string")
        .def("deserialize", &GameObject::Deserialize, py::arg("json_str"), "Deserialize GameObject from JSON string")
        // ---- Hierarchy component search (Unity: GetComponentInChildren/Parent) ----
        .def(
            "get_component_in_children",
            [](GameObject *obj, py::object componentType, bool includeInactive) -> py::object {
                if (!obj)
                    return py::none();

                std::string typeName = ResolveComponentTypeName(componentType);

                bool isCpp = (typeName == "Transform" || ComponentFactory::IsRegistered(typeName));
                auto &reg = ComponentBindingRegistry::Instance();

                std::function<py::object(GameObject *)> search = [&](GameObject *go) -> py::object {
                    if (!go)
                        return py::none();
                    // Unity: by default, skip inactive objects
                    if (!includeInactive && !go->IsActiveInHierarchy())
                        return py::none();

                    if (isCpp) {
                        if (typeName == "Transform") {
                            return py::cast(go->GetTransform(), py::return_value_policy::reference);
                        }
                        for (const auto &comp : go->GetAllComponents()) {
                            if (!dynamic_cast<PyComponentProxy *>(comp.get()) && comp->GetTypeName() == typeName) {
                                return reg.CastToPython(comp.get());
                            }
                        }
                    } else {
                        for (const auto &comp : go->GetAllComponents()) {
                            if (auto *proxy = dynamic_cast<PyComponentProxy *>(comp.get())) {
                                py::object pyComp = proxy->GetPyComponent();
                                if (!pyComp.is_none()) {
                                    bool match =
                                        py::isinstance<py::str>(componentType)
                                            ? (py::str(pyComp.attr("__class__").attr("__name__")).cast<std::string>() ==
                                               typeName)
                                            : py::isinstance(pyComp, componentType);
                                    if (match)
                                        return pyComp;
                                }
                            }
                        }
                    }

                    for (const auto &child : go->GetChildren()) {
                        py::object r = search(child.get());
                        if (!r.is_none())
                            return r;
                    }
                    return py::none();
                };

                return search(obj);
            },
            py::arg("component_type"), py::arg("include_inactive") = false,
            "Get a component on this or any child GameObject. Unity: GetComponentInChildren<T>()")
        .def(
            "get_component_in_parent",
            [](GameObject *obj, py::object componentType, bool includeInactive) -> py::object {
                if (!obj)
                    return py::none();

                std::string typeName = ResolveComponentTypeName(componentType);

                bool isCpp = (typeName == "Transform" || ComponentFactory::IsRegistered(typeName));
                auto &reg = ComponentBindingRegistry::Instance();

                GameObject *current = obj;
                while (current) {
                    // Unity: by default, skip inactive objects
                    if (!includeInactive && !current->IsActiveInHierarchy()) {
                        current = current->GetParent();
                        continue;
                    }
                    if (isCpp) {
                        if (typeName == "Transform") {
                            return py::cast(current->GetTransform(), py::return_value_policy::reference);
                        }
                        for (const auto &comp : current->GetAllComponents()) {
                            if (!dynamic_cast<PyComponentProxy *>(comp.get()) && comp->GetTypeName() == typeName) {
                                return reg.CastToPython(comp.get());
                            }
                        }
                    } else {
                        for (const auto &comp : current->GetAllComponents()) {
                            if (auto *proxy = dynamic_cast<PyComponentProxy *>(comp.get())) {
                                py::object pyComp = proxy->GetPyComponent();
                                if (!pyComp.is_none()) {
                                    bool match =
                                        py::isinstance<py::str>(componentType)
                                            ? (py::str(pyComp.attr("__class__").attr("__name__")).cast<std::string>() ==
                                               typeName)
                                            : py::isinstance(pyComp, componentType);
                                    if (match)
                                        return pyComp;
                                }
                            }
                        }
                    }
                    current = current->GetParent();
                }
                return py::none();
            },
            py::arg("component_type"), py::arg("include_inactive") = false,
            "Get a component on this or any parent GameObject. Unity: GetComponentInParent<T>()")
        // ---- Static query methods (Unity: GameObject.Find, FindWithTag, etc.) ----
        .def_static(
            "find",
            [](const std::string &name) -> GameObject * {
                Scene *scene = SceneManager::Instance().GetActiveScene();
                return scene ? scene->Find(name) : nullptr;
            },
            py::return_value_policy::reference, py::arg("name"),
            "Find a GameObject by name in the active scene. Unity: GameObject.Find(name)")
        .def_static(
            "find_with_tag",
            [](const std::string &tag) -> GameObject * {
                Scene *scene = SceneManager::Instance().GetActiveScene();
                return scene ? scene->FindWithTag(tag) : nullptr;
            },
            py::return_value_policy::reference, py::arg("tag"),
            "Find the first GameObject with a given tag. Unity: GameObject.FindWithTag(tag)")
        .def_static(
            "find_game_objects_with_tag",
            [](const std::string &tag) -> std::vector<GameObject *> {
                Scene *scene = SceneManager::Instance().GetActiveScene();
                return scene ? scene->FindGameObjectsWithTag(tag) : std::vector<GameObject *>{};
            },
            py::return_value_policy::reference, py::arg("tag"),
            "Find all GameObjects with a given tag. Unity: GameObject.FindGameObjectsWithTag(tag)")
        // ---- Static lifecycle methods (Unity: Object.Instantiate, Object.Destroy) ----
        .def_static(
            "instantiate",
            [](GameObject *original, GameObject *parent) -> GameObject * {
                if (!original)
                    return nullptr;
                Scene *scene = original->GetScene();
                if (!scene)
                    scene = SceneManager::Instance().GetActiveScene();
                return scene ? scene->InstantiateGameObject(original, parent) : nullptr;
            },
            py::return_value_policy::reference, py::arg("original"), py::arg("parent") = nullptr,
            "Clone a GameObject (deep copy). Unity: Object.Instantiate(original)")
        .def_static(
            "destroy",
            [](GameObject *gameObject) {
                if (gameObject && gameObject->GetScene()) {
                    gameObject->GetScene()->DestroyGameObject(gameObject);
                }
            },
            py::arg("game_object"),
            "Destroy a GameObject (removed at end of frame). Unity: Object.Destroy(gameObject)");

    // ========================================================================
    // PendingPyComponent binding (for scene restoration)
    // ========================================================================
    py::class_<Scene::PendingPyComponent>(m, "PendingPyComponent")
        .def_readonly("game_object_id", &Scene::PendingPyComponent::gameObjectId)
        .def_readonly("type_name", &Scene::PendingPyComponent::typeName)
        .def_readonly("script_guid", &Scene::PendingPyComponent::scriptGuid)
        .def_readonly("fields_json", &Scene::PendingPyComponent::fieldsJson)
        .def_readonly("enabled", &Scene::PendingPyComponent::enabled);

    // ========================================================================
    // Scene binding
    // ========================================================================
    py::class_<Scene>(m, "Scene")
        .def_property("name", &Scene::GetName, &Scene::SetName)
        .def("set_playing", &Scene::SetPlaying, py::arg("playing"), "Set the scene play-state flag")
        .def("create_game_object", &Scene::CreateGameObject, py::return_value_policy::reference,
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
            [](Scene *scene, PrimitiveType type, size_t count, const std::string &namePrefix) {
                return CreatePrimitiveObjectsBatch(scene, type, count, namePrefix);
            },
            py::arg("type"), py::arg("count"), py::arg("name_prefix") = "",
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
        .def("find", &Scene::Find, py::return_value_policy::reference, py::arg("name"), "Find a GameObject by name")
        .def("find_by_id", &Scene::FindByID, py::return_value_policy::reference, py::arg("id"),
             "Find a GameObject by ID")
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
        .def("instantiate_game_object", &Scene::InstantiateGameObject, py::return_value_policy::reference,
             py::arg("source"), py::arg("parent") = nullptr,
             "Clone a GameObject (deep copy). Unity: Object.Instantiate()")
        .def("instantiate_from_json", &Scene::InstantiateFromJson, py::return_value_policy::reference,
             py::arg("json_str"), py::arg("parent") = nullptr,
             "Instantiate a GameObject hierarchy from a JSON string (e.g. prefab). Fresh IDs are assigned.")
        .def("process_pending_destroys", &Scene::ProcessPendingDestroys, "Process pending GameObject destroys")
        .def("is_playing", &Scene::IsPlaying, "Check if the scene is in play mode")
        .def("start", &Scene::Start, "Trigger Awake+Start on all components (idempotent — skipped if already started)")
        .def("awake_object", &Scene::AwakeObject, py::arg("game_object"),
             "Re-run Awake+OnEnable on a GameObject and its descendants (used after undo deserialization)")
        .def("serialize", &Scene::Serialize, "Serialize scene to JSON string")
        .def("deserialize", &Scene::Deserialize, py::arg("json_str"), "Deserialize scene from JSON string")
        .def("save_to_file", &Scene::SaveToFile, py::arg("path"), "Save scene to a JSON file")
        .def("load_from_file", &Scene::LoadFromFile, py::arg("path"), "Load scene from a JSON file")
        .def("has_pending_py_components", &Scene::HasPendingPyComponents,
             "Check if there are pending Python components to restore")
        .def("take_pending_py_components", &Scene::TakePendingPyComponents,
             "Get and clear pending Python components for restoration")
        .def_property_readonly("structure_version", &Scene::GetStructureVersion,
                               "Monotonic counter bumped on structural changes (add/remove/reparent)")
        // Camera management
        .def_property("main_camera", &Scene::GetMainCamera, &Scene::SetMainCamera, py::return_value_policy::reference,
                      "Get/set the main Camera component for this scene (used by Game View)");

    // ========================================================================
    // SceneManager binding (singleton - use nodelete to prevent pybind11 from deleting)
    // ========================================================================
    py::class_<SceneManager, std::unique_ptr<SceneManager, py::nodelete>>(m, "SceneManager")
        .def_static("instance", &SceneManager::Instance, py::return_value_policy::reference,
                    "Get the singleton SceneManager instance")
        .def("create_scene", &SceneManager::CreateScene, py::return_value_policy::reference, py::arg("name"),
             "Create a new empty scene")
        .def("unload_scene", &SceneManager::UnloadScene, py::arg("scene"),
             "Unload and destroy a scene, removing all its GameObjects and physics bodies")
        .def("get_active_scene", &SceneManager::GetActiveScene, py::return_value_policy::reference,
             "Get the currently active scene")
        .def("set_active_scene", &SceneManager::SetActiveScene, py::arg("scene"), "Set the active scene")
        .def("get_scene", &SceneManager::GetScene, py::return_value_policy::reference, py::arg("name"),
             "Get a scene by name")
        .def("is_playing", &SceneManager::IsPlaying, "Check if in play mode")
        .def("play", &SceneManager::Play, "Enter play mode")
        .def("stop", &SceneManager::Stop, "Stop play mode")
        .def("pause", &SceneManager::Pause, "Pause play mode")
        .def("is_paused", &SceneManager::IsPaused, "Check if paused")
        .def("get_fixed_time_step", &SceneManager::GetFixedTimeStep, "Get the fixed physics timestep in seconds")
        .def("set_fixed_time_step", &SceneManager::SetFixedTimeStep, py::arg("value"),
             "Set the fixed physics timestep in seconds")
        .def("get_max_fixed_delta_time", &SceneManager::GetMaxFixedDeltaTime,
             "Get the max clamped frame delta used by the fixed-step accumulator")
        .def("set_max_fixed_delta_time", &SceneManager::SetMaxFixedDeltaTime, py::arg("value"),
             "Set the max clamped frame delta used by the fixed-step accumulator")
        .def("step", &SceneManager::Step, py::arg("delta_time") = 0.016f,
             "Execute one frame while paused (Update + LateUpdate + EndFrame). No-op if not paused.")
        .def("dont_destroy_on_load", &SceneManager::DontDestroyOnLoad, py::arg("game_object"),
             "Mark a root GameObject so it survives scene switches. Unity: DontDestroyOnLoad()")
        .def("mark_mesh_renderers_dirty", &SceneManager::MarkMeshRenderersDirtyForAsset, py::arg("mesh_guid"),
             "Mark all MeshRenderers referencing a mesh GUID as needing GPU buffer re-upload");

    // ========================================================================
    // ComponentFactory — query registered native component types
    // ========================================================================
    m.def("get_registered_component_types", &ComponentFactory::GetRegisteredTypeNames,
          "Get list of all registered native component type names");
}

} // namespace infernux
