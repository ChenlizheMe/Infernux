#pragma once

#include "Camera.h"
#include "GameObject.h"
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

/**
 * @brief Scene container that holds all GameObjects.
 *
 * The Scene is the root container for all GameObjects in the game world.
 * It manages object lifecycle, updates, and provides lookup functionality.
 */
class Scene
{
  public:
    Scene() = default;
    explicit Scene(const std::string &name) : m_name(name)
    {
    }
    ~Scene();

    // Non-copyable, movable
    Scene(const Scene &) = delete;
    Scene &operator=(const Scene &) = delete;
    Scene(Scene &&) = default;
    Scene &operator=(Scene &&) = default;

    // ========================================================================
    // Properties
    // ========================================================================

    [[nodiscard]] const std::string &GetName() const
    {
        return m_name;
    }
    void SetName(const std::string &name)
    {
        m_name = name;
    }

    // ========================================================================
    // GameObject management
    // ========================================================================

    /// @brief Create a new empty GameObject in this scene
    GameObject *CreateGameObject(const std::string &name = "GameObject");

    /// @brief Pre-allocate capacity for root objects and id map
    void ReserveCapacity(size_t count);

    /// @brief Add an existing GameObject to this scene (takes ownership)
    void AddGameObject(std::unique_ptr<GameObject> gameObject);

    /// @brief Remove a GameObject from this scene
    void RemoveGameObject(GameObject *gameObject);

    /// @brief Destroy a GameObject (will be removed at end of frame)
    void DestroyGameObject(GameObject *gameObject);

    /// @brief Clone a GameObject (deep copy). Unity: Object.Instantiate()
    /// Creates a full copy including all children and components.
    /// Python components are stored as pending for Python-side reconstruction.
    /// @param source The GameObject to clone
    /// @param parent Optional parent for the clone (nullptr = root level)
    /// @return The cloned GameObject, or nullptr on failure
    GameObject *InstantiateGameObject(GameObject *source, GameObject *parent = nullptr);

    /// @brief Instantiate a GameObject hierarchy from a JSON string (e.g. prefab file).
    /// Creates fresh IDs for all objects. Python components are stored as pending.
    /// @param jsonStr The serialized GameObject JSON (from GameObject::Serialize())
    /// @param parent Optional parent for the new object (nullptr = root level)
    /// @return The root GameObject, or nullptr on failure
    GameObject *InstantiateFromJson(const std::string &jsonStr, GameObject *parent = nullptr);

    /// @brief Internal: Unregister object ID from lookup (called from GameObject dtor)
    void UnregisterGameObject(uint64_t id);

    /// @brief Internal: Register object ID into lookup
    void RegisterGameObject(GameObject *gameObject);

    /// @brief Detach an object from root list (returns ownership)
    std::unique_ptr<GameObject> DetachRootObject(GameObject *gameObject);

    /// @brief Attach an object to root list (takes ownership)
    void AttachRootObject(std::unique_ptr<GameObject> gameObject);

    /// @brief Reorder a root object to a new sibling index
    void SetRootObjectSiblingIndex(GameObject *gameObject, int newIndex);

    /// @brief Get all root GameObjects (objects without parents)
    [[nodiscard]] const std::vector<std::unique_ptr<GameObject>> &GetRootObjects() const
    {
        return m_rootObjects;
    }

    /// @brief Get all GameObjects in the scene (including children)
    [[nodiscard]] std::vector<GameObject *> GetAllObjects() const;

    // ========================================================================
    // Finding objects
    // ========================================================================

    /// @brief Find a GameObject by name (first match)
    [[nodiscard]] GameObject *Find(const std::string &name) const;

    /// @brief Find all GameObjects with a given name
    [[nodiscard]] std::vector<GameObject *> FindAll(const std::string &name) const;

    /// @brief Find a GameObject by ID
    [[nodiscard]] GameObject *FindByID(uint64_t id) const;

    /// @brief Find all GameObjects with a specific component type
    template <typename T> [[nodiscard]] std::vector<GameObject *> FindObjectsWithComponent() const;

    /// @brief Find the first GameObject with a given tag
    [[nodiscard]] GameObject *FindWithTag(const std::string &tag) const;

    /// @brief Find all GameObjects with a given tag
    [[nodiscard]] std::vector<GameObject *> FindGameObjectsWithTag(const std::string &tag) const;

    /// @brief Find all GameObjects in a given layer
    [[nodiscard]] std::vector<GameObject *> FindGameObjectsInLayer(int layer) const;

    // ========================================================================
    // Camera
    // ========================================================================

    /// @brief Get the main camera for this scene
    [[nodiscard]] Camera *GetMainCamera() const
    {
        return m_mainCamera;
    }

    /// @brief Set the main camera for this scene
    void SetMainCamera(Camera *camera)
    {
        m_mainCamera = camera;
    }

    /// @brief Find the best game camera based on depth ordering and active state.
    /// Skips the editor camera. If m_mainCamera is valid and active, returns it.
    /// Otherwise auto-discovers the highest-priority (lowest depth) active Camera
    /// in the scene and caches it as m_mainCamera.
    /// @param editorCam Editor camera to exclude from search
    /// @return The best game camera, or nullptr if none found
    Camera *FindGameCamera(Camera *editorCam);

    // ========================================================================
    // Update loop
    // ========================================================================

    /// @brief Called once at scene start
    void Start();

    /// @brief Called every frame
    void Update(float deltaTime);

    /// @brief Called at a fixed time step (physics / deterministic logic)
    void FixedUpdate(float fixedDeltaTime);

    /// @brief Called every frame after Update
    void LateUpdate(float deltaTime);

    /// @brief Called every frame while not playing; runs edit-mode component updates.
    void EditorUpdate(float deltaTime);

    /// @brief Process pending destroy operations
    void ProcessPendingDestroys();

    /// @brief Queue a component for deferred Start() (runtime add/enable path).
    /// Start will execute before the next simulation/update pass.
    void QueueComponentStart(class Component *component);

    /// @brief Flush queued Start() calls for components that became active.
    void ProcessPendingStarts();

    // ========================================================================
    // Scene state
    // ========================================================================

    [[nodiscard]] bool IsLoaded() const
    {
        return m_isLoaded;
    }
    [[nodiscard]] bool IsPlaying() const
    {
        return m_isPlaying;
    }

    [[nodiscard]] bool HasStarted() const
    {
        return m_hasStarted;
    }

    void SetPlaying(bool playing)
    {
        m_isPlaying = playing;
    }

    /// @brief Monotonically increasing counter bumped whenever the scene
    ///        structure changes (object add/remove/reparent or component
    ///        add/remove).
    ///        Python caches can compare this to their last-seen value to decide
    ///        whether to re-query cached object/component lists.
    [[nodiscard]] uint64_t GetStructureVersion() const
    {
        return m_structureVersion;
    }

    /// @brief Bump structure version (public so that external mutators can signal changes)
    void BumpStructureVersion()
    {
        ++m_structureVersion;
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    /// @brief Serialize scene to JSON string
    /// @return JSON string representation of the scene
    [[nodiscard]] std::string Serialize() const;

    /// @brief Deserialize scene from JSON string
    /// @param jsonStr JSON string to deserialize from
    /// @return true if successful
    bool Deserialize(const std::string &jsonStr);

    /// @brief Save scene to file
    /// @param path File path to save to
    /// @return true if successful
    bool SaveToFile(const std::string &path) const;

    /// @brief Load scene from file
    /// @param path File path to load from
    /// @return true if successful
    bool LoadFromFile(const std::string &path);

    // ========================================================================
    // Pending Python Components (for deserialization)
    // ========================================================================

    /**
     * @brief Info about a Python component that needs to be recreated
     * after scene deserialization. The actual component creation is done
     * by Python code, as C++ cannot directly instantiate Python classes.
     */
    struct PendingPyComponent
    {
        uint64_t gameObjectId = 0; // Which GameObject this belongs to
        std::string typeName;      // Python class name
        std::string scriptGuid;    // GUID for the script asset
        std::string fieldsJson;    // Serialized field values as JSON
        bool enabled = true;
    };

    /// @brief Get pending Python components to be restored (and clear the list)
    [[nodiscard]] std::vector<PendingPyComponent> TakePendingPyComponents()
    {
        std::vector<PendingPyComponent> result;
        result.swap(m_pendingPyComponents);
        return result;
    }

    /// @brief Check if there are pending Python components
    [[nodiscard]] bool HasPendingPyComponents() const
    {
        return !m_pendingPyComponents.empty();
    }

    /// @brief Push a pending Python component (used by native clone to avoid JSON round-trip).
    void AddPendingPyComponent(PendingPyComponent pc)
    {
        m_pendingPyComponents.push_back(std::move(pc));
    }

    /// @brief Re-run Awake+OnEnable on a GameObject and its descendants.
    /// Used after undo-driven deserialization to initialise newly-created
    /// C++ components (e.g. MeshRenderer registration).
    void AwakeObject(GameObject *obj);

  private:
    void CollectAllObjects(GameObject *obj, std::vector<GameObject *> &result) const;
    void QueueStartObject(GameObject *obj);
    void StartObject(GameObject *obj);

    /// @brief Shared recursive traversal for all update variants.
    /// @param updateMethod Pointer-to-member on GameObject (e.g. &GameObject::Update).
    void TraverseActiveObjects(GameObject *obj, float dt, void (GameObject::*updateMethod)(float));

    void UpdateObject(GameObject *obj, float deltaTime);
    void FixedUpdateObject(GameObject *obj, float fixedDeltaTime);
    void LateUpdateObject(GameObject *obj, float deltaTime);
    void EditorUpdateObject(GameObject *obj, float deltaTime);
    class Component *FindComponentByID(uint64_t componentId) const;
    bool IsPendingDestroy(const GameObject *obj) const;

    /// @brief Shared recursive GameObject builder from JSON string.
    /// @param preserveIds If true, restores original IDs (Deserialize); otherwise generates new ones (Instantiate).
    std::unique_ptr<GameObject> BuildGameObjectFromJson(const std::string &jsonStr, bool preserveIds);

    /// @brief Internal overload operating on an already-parsed JSON value.
    std::unique_ptr<GameObject> BuildGameObjectFromJsonImpl(const nlohmann::json &objJson, bool preserveIds);

    /// @brief Recursively register all objects in a subtree with Scene's lookup map.
    void RegisterObjectSubtree(GameObject *root);

    std::string m_name = "Untitled Scene";

    // Root-level game objects (objects without parents)
    std::vector<std::unique_ptr<GameObject>> m_rootObjects;

    // Quick lookup by ID
    std::unordered_map<uint64_t, GameObject *> m_objectsById;

    // GameObjects pending destruction (IDs)
    std::vector<uint64_t> m_pendingDestroy;
    std::unordered_set<uint64_t> m_pendingDestroySet;

    // Components pending first Start() (stored by stable component ID)
    std::vector<uint64_t> m_pendingStartComponentIds;

    // Python components pending recreation after deserialize
    std::vector<PendingPyComponent> m_pendingPyComponents;

    // Main camera reference
    Camera *m_mainCamera = nullptr;

    // State flags
    bool m_isLoaded = false;
    bool m_isPlaying = false;
    bool m_hasStarted = false;

    // Structure version counter (bumped on add/remove/reparent)
    uint64_t m_structureVersion = 0;
};

// ============================================================================
// Template implementations
// ============================================================================

template <typename T> std::vector<GameObject *> Scene::FindObjectsWithComponent() const
{
    std::vector<GameObject *> result;
    std::vector<GameObject *> allObjects = GetAllObjects();

    for (GameObject *obj : allObjects) {
        if (obj->GetComponent<T>() != nullptr) {
            result.push_back(obj);
        }
    }

    return result;
}

} // namespace infernux
