#include "Scene.h"
#include "ComponentFactory.h"
#include "MeshRenderer.h"
#include "SceneManager.h"
#include "TransformECSStore.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <core/log/InxLog.h>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <platform/filesystem/InxPath.h>

using json = nlohmann::json;

namespace infernux
{

Scene::~Scene()
{
    // Explicitly clear root objects to ensure destructors run while Scene members are valid
    m_rootObjects.clear();
}

GameObject *Scene::CreateGameObject(const std::string &name)
{
    auto gameObject = std::make_unique<GameObject>(name);
    gameObject->m_scene = this;

    GameObject *ptr = gameObject.get();
    m_objectsById[ptr->GetID()] = ptr;
    m_rootObjects.push_back(std::move(gameObject));
    ++m_structureVersion;

    return ptr;
}

void Scene::ReserveCapacity(size_t count)
{
    m_rootObjects.reserve(m_rootObjects.size() + count);
    m_objectsById.reserve(m_objectsById.size() + count);
    // Each GO gets ~2-3 components that queue for Start()
    m_pendingStartComponentIds.reserve(m_pendingStartComponentIds.size() + count * 3);
}

void Scene::AddGameObject(std::unique_ptr<GameObject> gameObject)
{
    if (!gameObject)
        return;

    gameObject->m_scene = this;

    GameObject *ptr = gameObject.get();
    m_objectsById[ptr->GetID()] = ptr;

    // If it has no parent, add to root objects
    if (gameObject->GetParent() == nullptr) {
        m_rootObjects.push_back(std::move(gameObject));
    }
}

void Scene::RemoveGameObject(GameObject *gameObject)
{
    if (!gameObject)
        return;

    // 1. Locate and Detach ownership
    std::unique_ptr<GameObject> ownedPtr;

    if (gameObject->GetParent()) {
        ownedPtr = gameObject->GetParent()->DetachChild(gameObject);
    } else {
        ownedPtr = DetachRootObject(gameObject);
    }
    ++m_structureVersion;

    // 2. ownedPtr goes out of scope -> deleted.
}

void Scene::DestroyGameObject(GameObject *gameObject)
{
    if (!gameObject)
        return;

    // Queue for removal at frame-end, not immediate
    const uint64_t id = gameObject->GetID();
    if (m_pendingDestroySet.insert(id).second) {
        m_pendingDestroy.push_back(id);

        // Unity-like behavior: once Destroy() is requested, object is treated as
        // inactive for this frame's remaining callbacks.  This triggers OnDisable
        // immediately for active components; OnDestroy still runs at frame-end.
        if (gameObject->IsActiveInHierarchy()) {
            gameObject->SetActive(false);
        }
    }
    ++m_structureVersion;
}

std::unique_ptr<GameObject> Scene::DetachRootObject(GameObject *gameObject)
{
    auto it = std::find_if(m_rootObjects.begin(), m_rootObjects.end(),
                           [gameObject](const std::unique_ptr<GameObject> &obj) { return obj.get() == gameObject; });

    if (it != m_rootObjects.end()) {
        std::unique_ptr<GameObject> ret = std::move(*it);
        m_rootObjects.erase(it);
        ++m_structureVersion;
        return ret;
    }
    return nullptr;
}

void Scene::AttachRootObject(std::unique_ptr<GameObject> gameObject)
{
    if (!gameObject)
        return;
    gameObject->SetScene(this); // Ensure scene is set
    m_rootObjects.push_back(std::move(gameObject));
    ++m_structureVersion;
}

void Scene::SetRootObjectSiblingIndex(GameObject *gameObject, int newIndex)
{
    int currentIndex = -1;
    for (size_t i = 0; i < m_rootObjects.size(); ++i) {
        if (m_rootObjects[i].get() == gameObject) {
            currentIndex = static_cast<int>(i);
            break;
        }
    }
    if (currentIndex < 0)
        return;
    newIndex = std::max(0, std::min(newIndex, static_cast<int>(m_rootObjects.size()) - 1));
    if (currentIndex == newIndex)
        return;
    auto ptr = std::move(m_rootObjects[currentIndex]);
    m_rootObjects.erase(m_rootObjects.begin() + currentIndex);
    m_rootObjects.insert(m_rootObjects.begin() + newIndex, std::move(ptr));
    ++m_structureVersion;
}

void Scene::UnregisterGameObject(uint64_t id)
{
    m_objectsById.erase(id);
}

void Scene::RegisterGameObject(GameObject *gameObject)
{
    if (!gameObject)
        return;
    m_objectsById[gameObject->GetID()] = gameObject;
}

std::vector<GameObject *> Scene::GetAllObjects() const
{
    std::vector<GameObject *> result;
    result.reserve(m_objectsById.size());

    for (const auto &root : m_rootObjects) {
        CollectAllObjects(root.get(), result);
    }

    return result;
}

void Scene::CollectAllObjects(GameObject *obj, std::vector<GameObject *> &result) const
{
    if (!obj)
        return;

    result.push_back(obj);

    for (const auto &child : obj->GetChildren()) {
        CollectAllObjects(child.get(), result);
    }
}

GameObject *Scene::Find(const std::string &name) const
{
    for (const auto &root : m_rootObjects) {
        if (root->GetName() == name) {
            return root.get();
        }

        // Search in children recursively
        GameObject *found = root->FindDescendant(name);
        if (found)
            return found;
    }
    return nullptr;
}

std::vector<GameObject *> Scene::FindAll(const std::string &name) const
{
    std::vector<GameObject *> result;
    std::vector<GameObject *> allObjects = GetAllObjects();

    for (GameObject *obj : allObjects) {
        if (obj->GetName() == name) {
            result.push_back(obj);
        }
    }

    return result;
}

GameObject *Scene::FindByID(uint64_t id) const
{
    auto it = m_objectsById.find(id);
    if (it != m_objectsById.end()) {
        return it->second;
    }
    return nullptr;
}

GameObject *Scene::FindWithTag(const std::string &tag) const
{
    for (const auto &[id, obj] : m_objectsById) {
        if (obj && obj->GetTag() == tag) {
            return obj;
        }
    }
    return nullptr;
}

std::vector<GameObject *> Scene::FindGameObjectsWithTag(const std::string &tag) const
{
    std::vector<GameObject *> result;
    for (const auto &[id, obj] : m_objectsById) {
        if (obj && obj->GetTag() == tag) {
            result.push_back(obj);
        }
    }
    return result;
}

std::vector<GameObject *> Scene::FindGameObjectsInLayer(int layer) const
{
    std::vector<GameObject *> result;
    for (const auto &[id, obj] : m_objectsById) {
        if (obj && obj->GetLayer() == layer) {
            result.push_back(obj);
        }
    }
    return result;
}

void Scene::Start()
{
    if (m_hasStarted)
        return;

    m_isLoaded = true;
    m_hasStarted = true;

    // ---- Unity-correct 2-pass lifecycle ----
    // Pass 1: Awake + OnEnable on every object/component
    for (size_t i = 0; i < m_rootObjects.size(); ++i) {
        AwakeObject(m_rootObjects[i].get());
    }
    // Pass 2: Start on every enabled component (all Awake calls finished)
    for (size_t i = 0; i < m_rootObjects.size(); ++i) {
        StartObject(m_rootObjects[i].get());
    }
}

void Scene::AwakeObject(GameObject *obj)
{
    if (!obj)
        return;

    // Unity: Awake is only called on GameObjects that are active in the hierarchy.
    // Inactive objects will have Awake deferred until they are first activated
    // (handled by GameObject::HandleActiveStateChanged).
    if (!obj->IsActiveInHierarchy())
        return;

    std::vector<Component *> components = obj->GetComponentsInExecutionOrder();
    for (Component *component : components) {
        if (component) {
            component->CallAwake();
        }
    }

    const auto &children = obj->GetChildren();
    for (size_t i = 0; i < children.size(); ++i) {
        AwakeObject(children[i].get());
    }
}

void Scene::StartObject(GameObject *obj)
{
    if (!obj)
        return;

    const bool activeInHierarchy = obj->IsActiveInHierarchy();

    std::vector<Component *> components = obj->GetComponentsInExecutionOrder();
    for (Component *component : components) {
        if (component && activeInHierarchy && component->IsEnabled()) {
            component->CallStart();
        }
    }

    const auto &children = obj->GetChildren();
    for (size_t i = 0; i < children.size(); ++i) {
        StartObject(children[i].get());
    }
}

void Scene::Update(float deltaTime)
{
    if (!m_isPlaying)
        return;

    TransformECSStore::Instance().SyncSceneWorldMatrices(this);

    // Flush deferred Start() calls for components that were added/enabled
    // during previous callbacks.
    ProcessPendingStarts();

    // Snapshot root count so objects instantiated mid-frame are not updated
    // until the next frame (Unity-style frame consistency).
    const size_t rootCount = m_rootObjects.size();
    for (size_t i = 0; i < rootCount && i < m_rootObjects.size(); ++i) {
        UpdateObject(m_rootObjects[i].get(), deltaTime);
    }
}

void Scene::FixedUpdate(float fixedDeltaTime)
{
    if (!m_isPlaying)
        return;

    TransformECSStore::Instance().SyncSceneWorldMatrices(this);

    const size_t rootCount = m_rootObjects.size();
    for (size_t i = 0; i < rootCount && i < m_rootObjects.size(); ++i) {
        FixedUpdateObject(m_rootObjects[i].get(), fixedDeltaTime);
    }
}

void Scene::TraverseActiveObjects(GameObject *obj, float dt, void (GameObject::*updateMethod)(float))
{
    if (!obj || !obj->IsActiveInHierarchy() || IsPendingDestroy(obj))
        return;

    (obj->*updateMethod)(dt);

    const auto &children = obj->GetChildren();
    const size_t childCount = children.size();
    for (size_t i = 0; i < childCount && i < children.size(); ++i) {
        TraverseActiveObjects(children[i].get(), dt, updateMethod);
    }
}

void Scene::FixedUpdateObject(GameObject *obj, float fixedDeltaTime)
{
    TraverseActiveObjects(obj, fixedDeltaTime, &GameObject::FixedUpdate);
}

void Scene::UpdateObject(GameObject *obj, float deltaTime)
{
    TraverseActiveObjects(obj, deltaTime, &GameObject::Update);
}

void Scene::LateUpdate(float deltaTime)
{
    if (!m_isPlaying)
        return;

    TransformECSStore::Instance().SyncSceneWorldMatrices(this);

    const size_t rootCount = m_rootObjects.size();
    for (size_t i = 0; i < rootCount && i < m_rootObjects.size(); ++i) {
        LateUpdateObject(m_rootObjects[i].get(), deltaTime);
    }
}

void Scene::EditorUpdate(float deltaTime)
{
    if (m_isPlaying)
        return;

    TransformECSStore::Instance().SyncSceneWorldMatrices(this);

    const size_t rootCount = m_rootObjects.size();
    for (size_t i = 0; i < rootCount && i < m_rootObjects.size(); ++i) {
        EditorUpdateObject(m_rootObjects[i].get(), deltaTime);
    }
}

void Scene::LateUpdateObject(GameObject *obj, float deltaTime)
{
    TraverseActiveObjects(obj, deltaTime, &GameObject::LateUpdate);
}

void Scene::EditorUpdateObject(GameObject *obj, float deltaTime)
{
    TraverseActiveObjects(obj, deltaTime, &GameObject::EditorUpdate);
}

// ============================================================================
// Shared JSON → GameObject builder (used by both Deserialize and InstantiateFromJson)
// ============================================================================

// Internal overload operating directly on a parsed json value.
std::unique_ptr<GameObject> Scene::BuildGameObjectFromJsonImpl(const json &objJson, bool preserveIds)
{
    std::string name = objJson.value("name", std::string(preserveIds ? "GameObject" : "Prefab"));
    auto obj = std::make_unique<GameObject>(name);
    obj->m_scene = this;

    // Restore original ID only when deserializing (not cloning)
    if (preserveIds && objJson.contains("id")) {
        obj->m_id = objJson["id"].get<uint64_t>();
        GameObject::EnsureNextID(obj->m_id);
    }

    if (objJson.contains("active"))
        obj->m_active = objJson["active"].get<bool>();
    if (objJson.contains("is_static"))
        obj->m_isStatic = objJson["is_static"].get<bool>();
    if (objJson.contains("tag"))
        obj->m_tag = objJson["tag"].get<std::string>();
    if (objJson.contains("layer")) {
        int l = objJson["layer"].get<int>();
        obj->m_layer = (l >= 0 && l < 32) ? l : 0;
    }
    if (objJson.contains("prefab_guid"))
        obj->m_prefabGuid = objJson["prefab_guid"].get<std::string>();
    obj->m_prefabRoot = objJson.value("prefab_root", false);

    // Transform
    if (objJson.contains("transform")) {
        json tJson = objJson["transform"];
        if (!preserveIds)
            tJson.erase("component_id");
        obj->m_transform.Deserialize(tJson.dump());
    }

    // C++ components (factory-based)
    if (objJson.contains("components") && objJson["components"].is_array()) {
        for (const auto &compJson : objJson["components"]) {
            std::string typeName = compJson.value("type", std::string());
            if (typeName.empty() || typeName == "Transform")
                continue;
            std::unique_ptr<Component> comp = ComponentFactory::Create(typeName);
            if (!comp)
                continue;
            json cJson = compJson;
            if (!preserveIds) {
                cJson.erase("component_id");
                cJson.erase("instance_guid");
            }
            comp->SetGameObject(obj.get());
            comp->Deserialize(cJson.dump());
            obj->m_components.push_back(std::move(comp));
        }
    }

    // Python components — store as pending for Python-side reconstruction
    if (objJson.contains("py_components") && objJson["py_components"].is_array()) {
        uint64_t objId = obj->m_id ? obj->m_id : obj->GetID();
        for (const auto &pyCompJson : objJson["py_components"]) {
            PendingPyComponent pending;
            pending.gameObjectId = objId;
            pending.typeName = pyCompJson.value("py_type_name", std::string("PyComponent"));
            pending.scriptGuid = pyCompJson.value("script_guid", std::string());
            pending.enabled = pyCompJson.value("enabled", true);
            if (pyCompJson.contains("py_fields"))
                pending.fieldsJson = pyCompJson["py_fields"].dump();
            m_pendingPyComponents.push_back(pending);
        }
    }

    // Recurse children
    if (objJson.contains("children") && objJson["children"].is_array()) {
        for (const auto &childJson : objJson["children"]) {
            auto child = BuildGameObjectFromJsonImpl(childJson, preserveIds);
            if (child)
                obj->AttachChild(std::move(child));
        }
    }

    return obj;
}

std::unique_ptr<GameObject> Scene::BuildGameObjectFromJson(const std::string &jsonStr, bool preserveIds)
{
    json objJson = json::parse(jsonStr);
    return BuildGameObjectFromJsonImpl(objJson, preserveIds);
}

void Scene::RegisterObjectSubtree(GameObject *root)
{
    if (!root)
        return;
    RegisterGameObject(root);
    for (const auto &child : root->GetChildren())
        RegisterObjectSubtree(child.get());
}

void Scene::ProcessPendingDestroys()
{
    std::vector<uint64_t> currentPending;
    currentPending.swap(m_pendingDestroy); // To ensure we don't loop forever if destroy triggers destroy

    for (uint64_t id : currentPending) {
        m_pendingDestroySet.erase(id);
        GameObject *obj = FindByID(id);
        if (obj) {
            RemoveGameObject(obj);
        }
    }
}

bool Scene::IsPendingDestroy(const GameObject *obj) const
{
    if (!obj)
        return false;

    const GameObject *current = obj;
    while (current) {
        if (m_pendingDestroySet.find(current->GetID()) != m_pendingDestroySet.end()) {
            return true;
        }
        current = current->GetParent();
    }
    return false;
}

void Scene::QueueComponentStart(Component *component)
{
    if (!component)
        return;

    const uint64_t id = component->GetComponentID();
    if (id == 0)
        return;

    if (m_pendingStartComponentIdSet.insert(id).second) {
        m_pendingStartComponentIds.push_back(id);
    }
}

void Scene::ProcessPendingStarts()
{
    if (m_pendingStartComponentIds.empty())
        return;

    std::vector<uint64_t> pending;
    pending.swap(m_pendingStartComponentIds);
    m_pendingStartComponentIdSet.clear();

    // Build a component-pointer cache so the sort and dispatch each do O(1) lookups.
    std::vector<Component *> comps;
    comps.reserve(pending.size());
    for (uint64_t id : pending) {
        comps.push_back(Component::FindByComponentId(id));
    }

    // Stable-sort by execution order, then by component ID.
    std::vector<size_t> indices(pending.size());
    std::iota(indices.begin(), indices.end(), size_t(0));
    std::stable_sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        Component *a = comps[i];
        Component *b = comps[j];
        if (!a || !b)
            return pending[i] < pending[j];
        if (a->GetExecutionOrder() != b->GetExecutionOrder())
            return a->GetExecutionOrder() < b->GetExecutionOrder();
        return a->GetComponentID() < b->GetComponentID();
    });

    for (size_t idx : indices) {
        Component *component = comps[idx];
        if (!component)
            continue;

        GameObject *go = component->GetGameObject();
        if (!go)
            continue;

        if (!m_isPlaying || !m_hasStarted)
            continue;

        if (component->IsEnabled() && go->IsActiveInHierarchy() && component->HasAwake()) {
            component->CallStart();
        }
    }
}

void Scene::QueueStartObject(GameObject *obj)
{
    if (!obj || !obj->IsActiveInHierarchy())
        return;

    std::vector<Component *> components = obj->GetComponentsInExecutionOrder();
    for (Component *component : components) {
        if (component && component->IsEnabled() && component->HasAwake()) {
            QueueComponentStart(component);
        }
    }

    const auto &children = obj->GetChildren();
    for (size_t i = 0; i < children.size(); ++i) {
        QueueStartObject(children[i].get());
    }
}

Component *Scene::FindComponentByID(uint64_t componentId) const
{
    if (componentId == 0)
        return nullptr;

    return Component::FindByComponentId(componentId);
}

// ============================================================================
// Instantiate (deep clone) — Unity: Object.Instantiate()
// ============================================================================

GameObject *Scene::InstantiateGameObject(GameObject *source, GameObject *parent)
{
    if (!source)
        return nullptr;

    // Native deep clone — no JSON serialization round-trip.
    auto clone = source->Clone(this);
    if (!clone)
        return nullptr;

    // Unity: cloned root object gets " (Clone)" suffix
    clone->SetName(source->GetName() + " (Clone)");

    // Register all cloned objects in scene lookup
    GameObject *ptr = clone.get();
    auto registerAll = [&](auto &&self, GameObject *go) -> void {
        if (!go)
            return;
        RegisterGameObject(go);
        for (const auto &child : go->GetChildren()) {
            self(self, child.get());
        }
    };
    registerAll(registerAll, ptr);

    // Attach to scene hierarchy
    if (parent) {
        parent->AttachChild(std::move(clone));
    } else {
        m_rootObjects.push_back(std::move(clone));
    }

    // Awake C++ components so they register with subsystems
    AwakeObject(ptr);
    if (m_isPlaying && m_hasStarted) {
        QueueStartObject(ptr);
    }

    ++m_structureVersion;

    return ptr;
}

// ============================================================================
// Instantiate from JSON (prefab) — clone from raw JSON string (prefab file)
// ============================================================================

GameObject *Scene::InstantiateFromJson(const std::string &jsonStr, GameObject *parent)
{
    json j;
    try {
        j = json::parse(jsonStr);
    } catch (const std::exception &) {
        return nullptr;
    }

    auto clone = BuildGameObjectFromJsonImpl(j, /*preserveIds=*/false);
    if (!clone)
        return nullptr;

    GameObject *ptr = clone.get();
    RegisterObjectSubtree(ptr);

    if (parent) {
        parent->AttachChild(std::move(clone));
    } else {
        m_rootObjects.push_back(std::move(clone));
    }

    AwakeObject(ptr);
    if (m_isPlaying && m_hasStarted) {
        QueueStartObject(ptr);
    }

    ++m_structureVersion;

    return ptr;
}

std::string Scene::Serialize() const
{
    json j;
    j["schema_version"] = 1;
    j["name"] = m_name;
    j["isPlaying"] = m_isPlaying;

    // Serialize main camera reference via component_id (survives deserialization)
    if (m_mainCamera) {
        j["mainCameraComponentId"] = m_mainCamera->GetComponentID();
    }

    // Serialize all root objects
    json objectsArray = json::array();
    for (const auto &obj : m_rootObjects) {
        try {
            json objJson = json::parse(obj->Serialize());
            objectsArray.push_back(objJson);
        } catch (const std::exception &e) {
            INXLOG_ERROR("[Scene] Failed to serialize object '", obj->GetName(), "': ", e.what());
        }
    }
    j["objects"] = objectsArray;

    return j.dump(2);
}

bool Scene::Deserialize(const std::string &jsonStr)
{
    try {
        json j = json::parse(jsonStr);

        if (!j.contains("schema_version")) {
            INXLOG_ERROR("Scene::Deserialize: missing 'schema_version' field — data predates versioning system");
            return false;
        }

        // Basic properties
        if (j.contains("name")) {
            m_name = j["name"].get<std::string>();
        }
        if (j.contains("isPlaying")) {
            m_isPlaying = j["isPlaying"].get<bool>();
        }

        // Clear existing objects and pending destroys
        m_mainCamera = nullptr; // Reset before clearing objects to avoid dangling pointer

        // Clear physics/renderer registries BEFORE destroying old GameObjects so
        // that no stale component pointers remain in the iteration vectors while
        // destructors run.  Each old component's destructor will find empty
        // registries (safe no-op), and new components will re-register in OnEnable.
        SceneManager::Instance().ClearComponentRegistries();

        m_rootObjects.clear();
        m_objectsById.clear();
        m_pendingDestroy.clear();
        m_pendingDestroySet.clear();
        m_pendingStartComponentIds.clear();
        m_pendingPyComponents.clear();
        m_hasStarted = false;

        if (j.contains("objects") && j["objects"].is_array()) {
            int objectCounter = 0;
            for (const auto &objJson : j["objects"]) {
                auto obj = BuildGameObjectFromJsonImpl(objJson, /*preserveIds=*/true);
                if (obj) {
                    RegisterObjectSubtree(obj.get());
                    m_rootObjects.push_back(std::move(obj));
                }
                // Pump the OS message queue periodically during heavy
                // deserialization to prevent Windows "Not Responding".
                if (++objectCounter % 8 == 0) {
                    SDL_PumpEvents();
                }
            }
        }

        // Always awake all C++ components after deserialization so they initialize
        // correctly regardless of play state.  This populates the MeshRenderer /
        // Rigidbody / Collider registries that CollectRenderables() and the physics
        // system rely on.
        //
        // NOTE: Python proxy components (PyComponentProxy) are NOT present in
        // m_rootObjects at this point — they are stored as PendingPyComponent and
        // added by _restore_py_components() AFTER Deserialize() returns.  So this
        // loop never touches Python lifecycle methods; it only affects C++ components.
        for (const auto &root : m_rootObjects) {
            AwakeObject(root.get());
        }

        // Restore main camera reference from component ID
        if (j.contains("mainCameraComponentId")) {
            uint64_t camCompId = j["mainCameraComponentId"].get<uint64_t>();
            auto camObjects = FindObjectsWithComponent<Camera>();
            for (auto *obj : camObjects) {
                Camera *cam = obj->GetComponent<Camera>();
                if (cam && cam->GetComponentID() == camCompId) {
                    m_mainCamera = cam;
                    break;
                }
            }
        }

        ++m_structureVersion; // Scene was fully rebuilt

        return true;
    } catch (const std::exception &e) {
        return false;
    }
}

bool Scene::SaveToFile(const std::string &path) const
{
    try {
        std::string jsonStr = Serialize();
        std::ofstream file = OpenOutputFile(path, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            return false;
        }
        file << jsonStr;
        file.close();
        return true;
    } catch (const std::exception &e) {
        return false;
    }
}

bool Scene::LoadFromFile(const std::string &path)
{
    try {
        std::ifstream file = OpenInputFile(path);
        if (!file.is_open()) {
            return false;
        }

        std::string jsonStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        return Deserialize(jsonStr);
    } catch (const std::exception &e) {
        return false;
    }
}

Camera *Scene::FindGameCamera(Camera *editorCam)
{
    // Fast path: cached main camera is still valid and active
    if (m_mainCamera && m_mainCamera != editorCam) {
        // Verify the camera's GameObject is still active and the component is enabled
        GameObject *go = m_mainCamera->GetGameObject();
        if (go && go->IsActiveInHierarchy() && m_mainCamera->IsEnabled()) {
            return m_mainCamera;
        }
        // Cached main camera is no longer valid — clear and re-discover
        m_mainCamera = nullptr;
    }

    // Auto-discover: find highest-priority (lowest depth) active Camera component
    auto objects = FindObjectsWithComponent<Camera>();
    Camera *bestCam = nullptr;
    float bestDepth = std::numeric_limits<float>::max();

    for (auto *obj : objects) {
        if (!obj->IsActiveInHierarchy())
            continue;

        Camera *c = obj->GetComponent<Camera>();
        if (!c || !c->IsEnabled() || c == editorCam)
            continue;

        if (c->GetDepth() < bestDepth) {
            bestDepth = c->GetDepth();
            bestCam = c;
        }
    }

    if (bestCam) {
        m_mainCamera = bestCam;
        INXLOG_DEBUG("Game camera auto-assigned from GameObject '", bestCam->GetGameObject()->GetName(),
                     "' (depth=", bestDepth, ")");
    }

    return bestCam;
}

} // namespace infernux
