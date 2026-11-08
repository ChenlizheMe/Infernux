#include "BindingSceneInternal.h"
#include "function/resources/AssetRegistry/AssetRegistry.h"
#include "function/resources/InxMesh/InxMesh.h"
#include "function/resources/InxMesh/ModelMeshReference.h"
#include "function/resources/InxResource/InxResourceMeta.h"
#include "function/scene/BoxCollider.h"
#include "function/scene/CapsuleCollider.h"
#include "function/scene/CylinderCollider.h"
#include "function/scene/GameObject.h"
#include "function/scene/MeshCollider.h"
#include "function/scene/MeshRenderer.h"
#include "function/scene/PrimitiveMeshes.h"
#include "function/scene/Scene.h"
#include "function/scene/SceneManager.h"
#include "function/scene/SkeletonInstance.h"
#include "function/scene/SkinnedMeshRenderer.h"
#include "function/scene/SphereCollider.h"
#include "function/scene/Transform.h"
#include "function/scene/physics/PhysicsECSStore.h"
#include <algorithm>
#include <core/math/TransformPose.h>
#include <glm/gtc/quaternion.hpp>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace py = pybind11;

namespace infernux::scene_binding
{

/**
 * @brief Resolve static primitive mesh data (zero-copy reference).
 */
void GetPrimitiveMeshData(PrimitiveType type, const std::vector<Vertex> *&outVertices,
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

std::shared_ptr<InxMesh> GetBuiltinPrimitiveMeshAsset(const std::string &name)
{
    static std::mutex cacheMutex;
    static std::unordered_map<std::string, std::shared_ptr<InxMesh>> cache;
    std::lock_guard lock(cacheMutex);
    if (const auto found = cache.find(name); found != cache.end())
        return found->second;

    static const std::unordered_map<std::string, PrimitiveType> types = {
        {"Cube", PrimitiveType::Cube},         {"Sphere", PrimitiveType::Sphere}, {"Capsule", PrimitiveType::Capsule},
        {"Cylinder", PrimitiveType::Cylinder}, {"Plane", PrimitiveType::Plane},   {"Quad", PrimitiveType::Quad},
    };
    const auto type = types.find(name);
    if (type == types.end())
        throw std::invalid_argument("unknown built-in primitive Mesh '" + name + "'");

    const std::vector<Vertex> *vertices = nullptr;
    const std::vector<uint32_t> *indices = nullptr;
    const char *defaultName = "Primitive";
    GetPrimitiveMeshData(type->second, vertices, indices, defaultName);
    auto mesh = std::make_shared<InxMesh>(defaultName);
    mesh->SetGuid(BuiltinMeshGuidPrefix + name);
    mesh->SetData(std::vector<Vertex>(vertices->begin(), vertices->end()),
                  std::vector<uint32_t>(indices->begin(), indices->end()), {});
    cache.emplace(name, mesh);
    return mesh;
}

static void AddPrimitiveCollider(GameObject &object, PrimitiveType type)
{
    switch (type) {
    case PrimitiveType::Cube:
        object.AddComponent<BoxCollider>();
        break;
    case PrimitiveType::Sphere:
        object.AddComponent<SphereCollider>();
        break;
    case PrimitiveType::Capsule:
        object.AddComponent<CapsuleCollider>();
        break;
    case PrimitiveType::Cylinder:
        object.AddComponent<CylinderCollider>();
        break;
    case PrimitiveType::Plane:
    case PrimitiveType::Quad:
        object.AddComponent<MeshCollider>();
        break;
    }
}

/**
 * @brief Helper function to create a primitive GameObject.
 * Auto-reserves capacity when rapid creation is detected.
 */
GameObject *CreatePrimitiveObject(Scene *scene, PrimitiveType type, const std::string &name)
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
            // The GameObject's display name is independent from the mesh
            // identity. Keeping the canonical primitive name lets scene
            // serialization store a compact built-in reference.
            renderer->SetSharedPrimitiveMesh(*vertices, *indices, defaultName);
            AddPrimitiveCollider(*obj, type);
        }
    }
    return obj;
}

/**
 * @brief Batch-create N primitive GameObjects with pre-reserved capacity.
 * Returns a Python list of GameObjects.
 */
py::list CreatePrimitiveObjectsBatch(Scene *scene, PrimitiveType type, size_t count, const std::string &namePrefix,
                                     bool withColliders)
{
    const std::vector<Vertex> *vertices = nullptr;
    const std::vector<uint32_t> *indices = nullptr;
    const char *defaultName = "Primitive";
    GetPrimitiveMeshData(type, vertices, indices, defaultName);

    const std::string prefix = namePrefix.empty() ? defaultName : namePrefix;

    // Pre-allocate capacity to avoid incremental vector growth.
    scene->ReserveCapacity(count);
    TransformECSStore::Instance().Reserve(TransformECSStore::Instance().Capacity() + count);
    Component::ReserveRegistry(count * (withColliders ? 3 : 2));
    SceneManager::Instance().ReserveRendererCapacity(count);
    if (withColliders) {
        PhysicsECSStore::Instance().ReserveForBulkCreation(count);
    }

    py::list result(count);
    for (size_t i = 0; i < count; ++i) {
        std::string objName = prefix + "_" + std::to_string(i);
        GameObject *obj = scene->CreateGameObject(objName);
        if (obj) {
            MeshRenderer *renderer = obj->AddComponent<MeshRenderer>();
            if (renderer) {
                renderer->SetSharedPrimitiveMesh(*vertices, *indices, defaultName);
                if (withColliders) {
                    AddPrimitiveCollider(*obj, type);
                }
            }
        }
        result[i] = py::cast(obj, py::return_value_policy::reference);
    }
    return result;
}

class RendererRegistryTransaction final
{
  public:
    RendererRegistryTransaction()
    {
        SceneManager::Instance().BeginRendererRegistryTransaction();
    }
    ~RendererRegistryTransaction()
    {
        SceneManager::Instance().EndRendererRegistryTransaction();
    }

    RendererRegistryTransaction(const RendererRegistryTransaction &) = delete;
    RendererRegistryTransaction &operator=(const RendererRegistryTransaction &) = delete;
};

py::object InstantiateGameObjectsBatch(Scene &scene, GameObject *source, const FloatArray &positions,
                                       const std::optional<FloatArray> &rotations,
                                       const std::optional<FloatArray> &scales, GameObject *parent,
                                       bool instantiateInWorldSpace, bool returnObjects)
{
    if (!source)
        throw std::invalid_argument("Instantiate batch requires a source GameObject");
    if (positions.ndim() != 2 || positions.shape(1) != 3)
        throw std::invalid_argument("Instantiate positions must have shape (N, 3)");

    const py::ssize_t count = positions.shape(0);
    const bool hasRotations = rotations.has_value();
    const bool hasScales = scales.has_value();
    if (hasRotations && (rotations->ndim() != 2 || rotations->shape(0) != count || rotations->shape(1) != 4))
        throw std::invalid_argument("Instantiate rotations must have shape (N, 4) in x, y, z, w order");
    if (hasScales && (scales->ndim() != 2 || scales->shape(0) != count || scales->shape(1) != 3))
        throw std::invalid_argument("Instantiate scales must have shape (N, 3)");

    const float *positionData = positions.data();
    const float *rotationData = hasRotations ? rotations->data() : nullptr;
    const float *scaleData = hasScales ? scales->data() : nullptr;

    const size_t batchCount = static_cast<size_t>(count);
    scene.ReserveCapacity(batchCount);
    auto &transforms = TransformECSStore::Instance();
    transforms.Reserve(transforms.Capacity() + batchCount);
    Component::ReserveRegistry(Component::GetInstanceCount() + batchCount * 3);
    SceneManager::Instance().ReserveRendererCapacity(batchCount);

    RendererRegistryTransaction transaction;
    py::list result;
    std::vector<GameObject *> created;
    created.reserve(batchCount);
    try {
        for (py::ssize_t index = 0; index < count; ++index) {
            InstantiationIds ids;
            auto *object = scene.InstantiateGameObject(source, ids, parent, instantiateInWorldSpace);
            created.push_back(object);
            auto *transform = object->GetTransform();
            const float *position = positionData + index * 3;
            transform->SetWorldPosition(glm::vec3(position[0], position[1], position[2]));
            if (hasRotations) {
                const float *rotation = rotationData + index * 4;
                transform->SetWorldRotation(glm::quat(rotation[3], rotation[0], rotation[1], rotation[2]));
            }
            if (hasScales) {
                const float *scale = scaleData + index * 3;
                transform->SetLocalScale(glm::vec3(scale[0], scale[1], scale[2]));
            }
            if (returnObjects)
                result.append(py::cast(object, py::return_value_policy::reference));
        }
    } catch (...) {
        for (auto item = created.rbegin(); item != created.rend(); ++item)
            scene.RemoveGameObject(*item);
        throw;
    }
    if (returnObjects)
        return std::move(result);
    return py::int_(count);
}

std::string GetModelSubresourceId(const std::shared_ptr<const InxResourceMeta> &meta,
                                  const std::vector<std::string> &path)
{
    if (!meta || path.empty() || !meta->HasKey("model_meshes"))
        return {};
    const auto manifest = nlohmann::json::parse(meta->GetDataAs<std::string>("model_meshes"));
    if (!manifest.is_array())
        throw std::invalid_argument("model_meshes metadata must be an array");
    for (const auto &entry : manifest) {
        if (!entry.is_object() || !entry.contains("path") || !entry["path"].is_array())
            continue;
        if (entry["path"].get<std::vector<std::string>>() != path)
            continue;
        const auto id = entry.value("subresource_id", std::string{});
        if (!id.empty())
            return id;
    }
    return {};
}

GameObject *CreateModelObject(Scene *scene, const std::string &guid, const std::string &name)
{
    auto &registry = AssetRegistry::Instance();

    const auto [sourceGuid, nodePath] = SplitModelMeshReference(guid);
    if (!nodePath.empty()) {
        auto source = registry.LoadAsset<InxMesh>(sourceGuid, ResourceType::Mesh);
        if (!source)
            throw std::invalid_argument("Model mesh source cannot be loaded");
        const int32_t sourceNode = source->RequireModelNode(nodePath);
        const auto &metadata = source->GetImportMetadata();
        const bool generateCollider = !source->HasSkinnedData() && metadata && metadata->HasKey("generate_colliders") &&
                                      metadata->GetDataAs<bool>("generate_colliders");
        auto *object = scene->CreateGameObject(name.empty() ? nodePath.back() : name);
        if (!object)
            return nullptr;
        try {
            auto *renderer = object->AddComponent<MeshRenderer>();
            renderer->SetMeshAsset(sourceGuid, source);
            renderer->SetModelNodePath(nodePath);
            renderer->SetModelSubresourceId(GetModelSubresourceId(metadata, nodePath));
            renderer->SetEnabled(source->GetModelNodes().at(static_cast<size_t>(sourceNode)).visible);
            if (generateCollider)
                object->AddComponent<MeshCollider>();
        } catch (...) {
            scene->DestroyGameObject(object);
            scene->ProcessPendingDestroys();
            throw;
        }
        return object;
    }

    auto mesh = registry.LoadAsset<InxMesh>(guid, ResourceType::Mesh);
    if (!mesh)
        return nullptr;

    std::string objName = name.empty() ? mesh->GetName() : name;
    if (objName.empty())
        objName = "Mesh Object";

    uint32_t nodeGroupCount = mesh->GetNodeGroupCount();
    const auto &nodeNames = mesh->GetNodeNames();
    const bool useSkinnedRenderer = mesh->HasSkinnedData();
    const auto &metadata = mesh->GetImportMetadata();
    const bool generateColliders = !useSkinnedRenderer && metadata && metadata->HasKey("generate_colliders") &&
                                   metadata->GetDataAs<bool>("generate_colliders");

    const auto &nodes = mesh->GetModelNodes();
    if (!useSkinnedRenderer && mesh->GetModelSourceGeometry() && !nodes.empty()) {
        std::vector<TransformPose> poses;
        poses.reserve(nodes.size());
        std::vector<std::vector<std::string>> paths;
        std::set<std::vector<std::string>> geometryPaths;
        // Validate all local transforms before changing the scene. A sheared
        // source cannot be silently approximated by the engine's TRS Transform.
        for (const auto &node : nodes) {
            auto path = node.parentIndex < 0 ? std::vector<std::string>{} : paths[node.parentIndex];
            path.push_back(node.name);
            if (!geometryPaths.insert(path).second)
                throw std::invalid_argument("Model geometry node path is ambiguous: " + node.name);
            paths.push_back(std::move(path));
            poses.push_back(DecomposeTransformPose(node.localTransform, node.name));
        }
        GameObject *container = scene->CreateGameObject(objName);
        if (!container)
            return nullptr;
        container->SetModelSource(guid, {});
        try {
            std::vector<GameObject *> objects;
            objects.reserve(nodes.size());
            for (size_t index = 0; index < nodes.size(); ++index) {
                const auto &node = nodes[index];
                GameObject *child = scene->CreateGameObject(node.name);
                if (!child)
                    throw std::runtime_error("Cannot create imported model node");
                child->SetModelSource(guid, paths[index]);
                child->GetTransform()->SetParent(
                    (node.parentIndex < 0 ? container : objects[node.parentIndex])->GetTransform(), false);
                child->GetTransform()->SetLocalPosition(poses[index].position);
                child->GetTransform()->SetLocalRotation(poses[index].rotation);
                child->GetTransform()->SetLocalScale(poses[index].scale);
                objects.push_back(child);
                if (node.nodeGroup >= 0) {
                    auto *renderer = child->AddComponent<MeshRenderer>();
                    renderer->SetNodeGroup(node.nodeGroup);
                    renderer->SetMeshAsset(guid, mesh);
                    renderer->SetModelNodePath(paths[index]);
                    renderer->SetModelSubresourceId(GetModelSubresourceId(metadata, paths[index]));
                    renderer->SetEnabled(node.visible);
                    if (generateColliders)
                        child->AddComponent<MeshCollider>();
                }
            }
        } catch (...) {
            scene->DestroyGameObject(container);
            scene->ProcessPendingDestroys();
            throw;
        }
        return container;
    }

    // Both single and multi-mesh instances have one owner and one skeleton.
    GameObject *container = scene->CreateGameObject(objName);
    try {
        if (useSkinnedRenderer)
            container->AddComponent<SkeletonInstance>()->SetSourceModelGuid(guid);
        const uint32_t count = std::max(1u, nodeGroupCount);
        for (uint32_t group = 0; group < count; ++group) {
            GameObject *object = container;
            if (count > 1) {
                const auto name = group < nodeNames.size() && !nodeNames[group].empty()
                                      ? nodeNames[group]
                                      : "Node_" + std::to_string(group);
                object = scene->CreateGameObject(name);
                object->GetTransform()->SetParent(container->GetTransform(), false);
            }
            if (useSkinnedRenderer) {
                auto *renderer = object->AddComponent<SkinnedMeshRenderer>();
                renderer->SetSourceModelGuid(guid);
                if (count > 1)
                    renderer->SetNodeGroup(static_cast<int32_t>(group));
            } else {
                auto *renderer = object->AddComponent<MeshRenderer>();
                renderer->SetMeshAsset(guid, mesh);
                if (count > 1)
                    renderer->SetNodeGroup(static_cast<int32_t>(group));
                if (generateColliders)
                    object->AddComponent<MeshCollider>();
            }
        }
    } catch (...) {
        scene->DestroyGameObject(container);
        scene->ProcessPendingDestroys();
        throw;
    }

    return container;
}

} // namespace infernux::scene_binding
