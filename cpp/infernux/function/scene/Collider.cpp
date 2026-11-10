/**
 * @file Collider.cpp
 * @brief Base Collider implementation — body registration, transform sync.
 */

#include "Collider.h"
#include "ComponentFactory.h"

#include "GameObject.h"
#include "MeshRenderer.h"
#include "Rigidbody.h"
#include "SceneManager.h"
#include "Transform.h"
#include "physics/PhysicsECSStore.h"
#include "physics/PhysicsWorld.h"
#include <core/config/EngineConfig.h>
#include <core/log/InxLog.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace infernux
{

SemanticTypeDescriptor Collider::DescribeSemanticType(std::string typeName, std::string readableId,
                                                      std::string displayName)
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux." + typeName;
    type.readableId = std::move(readableId);
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = std::move(displayName);
    type.runtimeProfiles = {"editor", "player", "headless"};
    AddSemanticField(type, "center", "VEC3", nlohmann::json::array({0.0, 0.0, 0.0}), "collider.center",
                     "collider.tooltip.center");
    AddSemanticField(type, "is_trigger", "BOOL", false, "collider.is_trigger", "collider.tooltip.is_trigger");
    auto &material = AddSemanticField(type, "physic_material", "ASSET",
                                      {{"$type", "asset_ref"}, {"asset_type", "PhysicMaterial"}, {"guid", ""}},
                                      "collider.physic_material", "collider.tooltip.physic_material");
    material["serialized_name"] = "physic_material_guid";
    material["asset_type"] = "PhysicMaterial";
    material["nullable"] = true;
    // The Python-facing reference is a typed asset document while the native
    // scene document stores its stable GUID. The property setter owns that
    // one explicit representation boundary.
    material["setter_owns_document_shape"] = true;
    return type;
}

nlohmann::json &Collider::AddSemanticField(SemanticTypeDescriptor &type, std::string fieldId, std::string valueType,
                                           nlohmann::json defaultValue, std::string displayNameKey, std::string tooltip)
{
    const auto separator = type.typeGuid.find_last_of('.');
    const std::string typeName = separator == std::string::npos ? type.typeGuid : type.typeGuid.substr(separator + 1);
    type.fields.push_back({typeName + "." + fieldId,
                           "FieldType." + std::move(valueType),
                           false,
                           {{"field_id", fieldId},
                            {"serialized_name", fieldId},
                            {"serialized", true},
                            {"hidden", false},
                            {"nullable", false},
                            {"storage_kind", "native_property"},
                            {"display_name_key", std::move(displayNameKey)},
                            {"tooltip", std::move(tooltip)},
                            {"default", std::move(defaultValue)}}});
    return type.fields.back().attributes;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

Collider::Collider()
{
    m_ecsHandle = PhysicsECSStore::Instance().AllocateCollider(this);
}

Collider::~Collider()
{
    if (const auto material = m_physicMaterial.Get())
        material->RemoveColliderBinding();
    if (m_physicMaterial.HasGuid())
        AssetDependencyGraph::Instance().RemoveRuntimeDependency(GetInstanceGuid(), m_physicMaterial.GetGuid());
    DetachActor();
    PhysicsECSStore::Instance().ReleaseCollider(m_ecsHandle);
}

void Collider::EnsureActor()
{
    auto &store = PhysicsECSStore::Instance();
    auto &data = DataMut();
    if (store.IsValid(data.actorHandle))
        return;
    auto *gameObject = GetGameObject();
    if (!gameObject)
        throw std::logic_error("Collider must be attached to a GameObject before accessing its physics actor");
    store.AttachCollider(this);
}

void Collider::OnHierarchyPublished()
{
    PhysicsECSStore::Instance().AttachCollider(this);
}

PhysicsActorData &Collider::ActorMut()
{
    EnsureActor();
    return PhysicsECSStore::Instance().GetActor(Data().actorHandle);
}

const PhysicsActorData &Collider::Actor() const
{
    const_cast<Collider *>(this)->EnsureActor();
    return PhysicsECSStore::Instance().GetActor(Data().actorHandle);
}

uint32_t Collider::GetBodyId() const
{
    auto &store = PhysicsECSStore::Instance();
    if (!store.IsValid(Data().actorHandle)) {
        if (!GetGameObject() || IsBeingDestroyed() || IsDestroyed())
            return 0xFFFFFFFF;
        const_cast<Collider *>(this)->EnsureActor();
    }
    return store.GetActor(Data().actorHandle).bodyId;
}

uint64_t Collider::GetShapeRevision() const
{
    return Actor().shapeRevision;
}

Rigidbody *Collider::GetAttachedRigidbody() const
{
    return Actor().rigidbody;
}

uint32_t Collider::GetSharedBodyId(const GameObject *gameObject)
{
    auto &store = PhysicsECSStore::Instance();
    const auto handle = store.FindActor(gameObject);
    return store.IsValid(handle) ? store.GetActor(handle).bodyId : 0xFFFFFFFF;
}

// ============================================================================
// Lifecycle
// ============================================================================

void Collider::Awake()
{
    EnsureActor();
    if (!Data().deserialized)
        AutoFitToMesh();
    PhysicsECSStore::Instance().QueueBodyCreation(Data().actorHandle);
}

void Collider::OnEnable()
{
    EnsureActor();
    PhysicsECSStore::Instance().RefreshActor(Data().actorHandle);
}

void Collider::OnDisable()
{
    auto &store = PhysicsECSStore::Instance();
    if (!IsBeingDestroyed() && store.IsValid(Data().actorHandle))
        store.RefreshActor(Data().actorHandle);
}

void Collider::OnDestroy()
{
    PhysicsWorld::Instance().RemoveIgnoredPairsForCollider(this);
    DetachActor();
}

void Collider::SetIsTrigger(bool trigger)
{
    auto &d = DataMut();
    if (d.isTrigger == trigger)
        return;
    d.isTrigger = trigger;
    PhysicsECSStore::Instance().NotifyCollisionSceneChanged();

    const uint32_t bodyId = GetBodyId();
    if (bodyId != 0xFFFFFFFF) {
        auto &physics = PhysicsWorld::Instance();
        physics.RefreshBodyTriggerState(bodyId);

        // Wake dynamic bodies that may be resting on (or overlapping) this
        // collider so they re-evaluate the changed sensor state.
        physics.WakeBodiesTouchingStatic(bodyId);

        // Clear stale contact-pair tracking so the listener produces fresh
        // Enter events (Trigger or Collision) on the next physics step
        // instead of suppressing them as wake-from-sleep duplicates.
        physics.InvalidateContactPairsForBody(bodyId);
    }
}

void Collider::SetCenter(const glm::vec3 &center)
{
    if (!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(center.z))
        throw std::invalid_argument("collider center must contain finite values");
    auto &d = DataMut();
    if (d.center == center)
        return;
    d.center = center;
    RebuildShape();
}

// ---- Friction / Bounciness helpers ----

/// @brief Update Jolt's body-level fallback material from the primary enabled collider.
/// Actual contacts are combined per subshape by InxContactListener.
static void ApplyMaterialToBody(Collider *self)
{
    const uint32_t bodyId = self->GetBodyId();
    if (bodyId == 0xFFFFFFFF)
        return;

    auto &world = PhysicsWorld::Instance();
    auto *primary = world.FindColliderByBodyId(bodyId);
    if (primary) {
        world.SetBodyFriction(bodyId, primary->GetFriction());
        world.SetBodyRestitution(bodyId, primary->GetBounciness());
    }
}

std::shared_ptr<PhysicMaterial> Collider::GetPhysicMaterial() const
{
    return m_physicMaterial.Get();
}

static std::shared_ptr<PhysicMaterial> CopySurfaceMaterial(const std::shared_ptr<PhysicMaterial> &source)
{
    auto instance = std::make_shared<PhysicMaterial>();
    if (source) {
        instance->CopyValuesFrom(*source);
    } else {
        instance->SetFriction(EngineConfig::Get().defaultColliderFriction);
        instance->SetBounciness(EngineConfig::Get().defaultColliderBounciness);
    }
    return instance;
}

std::shared_ptr<PhysicMaterial> Collider::GetMaterial()
{
    if (!m_materialIsInstance || m_physicMaterial.Get()->GetColliderBindingCount() > 1) {
        if (m_physicMaterial.HasGuid() && !m_physicMaterial.Get())
            throw std::runtime_error("Collider material asset is not resolved");
        SetPhysicMaterial(CopySurfaceMaterial(m_physicMaterial.Get()), true);
    }
    return m_physicMaterial.Get();
}

const std::string &Collider::GetPhysicMaterialGuid() const
{
    return m_physicMaterial.GetGuid();
}

void Collider::SetPhysicMaterial(std::shared_ptr<PhysicMaterial> material, bool instance)
{
    const std::string guid = material ? material->GetGuid() : std::string();
    if (instance && (!material || !guid.empty()))
        throw std::invalid_argument("Collider private material must be a runtime surface instance");
    if (!guid.empty()) {
        SetPhysicMaterialGuid(guid);
        return;
    }
    SetMaterialBinding(AssetRef<PhysicMaterial>(guid, std::move(material), 0), instance);
}

void Collider::SetMaterialBinding(AssetRef<PhysicMaterial> material, bool instance)
{
    m_materialIsInstance = instance;
    if (m_physicMaterial.Get() == material.Get() && m_physicMaterial.GetGuid() == material.GetGuid() &&
        m_physicMaterial.GetCachedVersion() == material.GetCachedVersion())
        return;
    auto &graph = AssetDependencyGraph::Instance();
    if (m_physicMaterial.HasGuid() && m_physicMaterial.GetGuid() != material.GetGuid())
        graph.RemoveRuntimeDependency(GetInstanceGuid(), m_physicMaterial.GetGuid());
    if (const auto old = m_physicMaterial.Get())
        old->RemoveColliderBinding();
    m_physicMaterial = std::move(material);
    if (const auto current = m_physicMaterial.Get())
        current->AddColliderBinding();
    if (m_physicMaterial.HasGuid())
        graph.AddRuntimeDependency(GetInstanceGuid(), m_physicMaterial.GetGuid());
    ApplyMaterialToBody(this);
    PhysicsECSStore::Instance().NotifyCollisionSceneChanged();
}

void Collider::SetPhysicMaterialGuid(const std::string &guid)
{
    if (guid.empty()) {
        ClearPhysicMaterial();
        return;
    }
    AssetRef<PhysicMaterial> material(guid);
    AssetRegistry::Instance().Resolve(material, ResourceType::PhysicMaterial);
    SetMaterialBinding(std::move(material), false);
}

void Collider::ClearPhysicMaterial()
{
    SetPhysicMaterial(nullptr);
}

void Collider::OnPhysicMaterialAssetEvent(AssetEvent event)
{
    if (event == AssetEvent::Deleted) {
        SetMaterialBinding(AssetRef<PhysicMaterial>(m_physicMaterial.GetGuid()), false);
        return;
    }
    if (event == AssetEvent::Modified) {
        SetPhysicMaterialGuid(m_physicMaterial.GetGuid());
    }
}

float Collider::GetFriction() const
{
    const auto material = m_physicMaterial.Get();
    return material ? material->GetFriction() : EngineConfig::Get().defaultColliderFriction;
}

float Collider::GetBounciness() const
{
    const auto material = m_physicMaterial.Get();
    return material ? material->GetBounciness() : EngineConfig::Get().defaultColliderBounciness;
}

PhysicsMaterialCombine Collider::GetFrictionCombine() const
{
    const auto material = m_physicMaterial.Get();
    return material ? material->GetFrictionCombine() : PhysicsMaterialCombine::Average;
}

PhysicsMaterialCombine Collider::GetBounceCombine() const
{
    const auto material = m_physicMaterial.Get();
    return material ? material->GetBounceCombine() : PhysicsMaterialCombine::Average;
}

// ============================================================================
// Helpers
// ============================================================================

glm::vec3 Collider::GetWorldScale() const
{
    if (auto *go = GetGameObject()) {
        if (auto *tf = go->GetTransform()) {
            return glm::abs(tf->GetWorldScale());
        }
    }
    return glm::vec3(1.0f);
}

glm::vec3 Collider::GetCenterInRigidSpace() const
{
    if (auto *object = GetGameObject()) {
        const auto *transform = object->GetTransform();
        const auto offset = glm::vec3(transform->GetWorldMatrix() * glm::vec4(GetCenter(), 0.0f));
        return glm::inverse(transform->GetWorldRotation()) * offset;
    }
    return GetCenter();
}

void Collider::AutoFitToMesh()
{
    // Default no-op. Derived classes override to set size/radius/height
    // from sibling MeshRenderer bounds.
}

// ============================================================================
// Jolt body management
// ============================================================================

void Collider::DetachActor()
{
    auto &store = PhysicsECSStore::Instance();
    const auto handle = Data().actorHandle;
    DataMut().actorHandle = {};
    if (store.IsValid(handle))
        store.ReleaseActor(handle, this);
}

void Collider::RebuildShape()
{
    if (!GetGameObject())
        return; // Prepared component values have no actor to refresh yet.
    EnsureActor();
    PhysicsECSStore::Instance().RefreshActor(Data().actorHandle);
}

// ============================================================================
// Serialization
// ============================================================================

nlohmann::json Collider::SerializeDocument() const
{
    const auto &d = Data();
    auto j = Component::SerializeDocument();
    j["is_trigger"] = d.isTrigger;
    j["center"] = {d.center.x, d.center.y, d.center.z};
    j["physic_material_guid"] = m_physicMaterial.GetGuid();
    const auto material = m_physicMaterial.Get();
    j["physic_material_instance"] =
        material && !m_physicMaterial.HasGuid() ? material->SerializeDocument() : nlohmann::json(nullptr);
    return j;
}

bool Collider::DeserializeDocument(const nlohmann::json &j)
{
    try {
        ColliderECSData staged = Data();
        staged.isTrigger = j.at("is_trigger").get<bool>();
        const auto &center = j.at("center");
        staged.center = glm::vec3(center[0].get<float>(), center[1].get<float>(), center[2].get<float>());
        const std::string materialGuid = j.at("physic_material_guid").get<std::string>();
        std::shared_ptr<PhysicMaterial> materialInstance;
        if (const auto instance = j.find("physic_material_instance"); instance != j.end() && !instance->is_null()) {
            if (!materialGuid.empty())
                throw std::invalid_argument("Collider material cannot be both an asset and an inline instance");
            materialInstance = std::make_shared<PhysicMaterial>();
            materialInstance->DeserializeDocument(*instance);
        }
        if (!Component::DeserializeDocument(j))
            return false;
        staged.deserialized = true;
        DataMut() = staged;
        if (materialInstance) {
            SetPhysicMaterial(std::move(materialInstance), true);
        } else if (!materialGuid.empty()) {
            SetPhysicMaterialGuid(materialGuid);
        } else {
            ClearPhysicMaterial();
        }
        // NOTE: RebuildShape() is called by derived classes after their own
        // fields are deserialized (so both base + derived changes are applied
        // in a single shape rebuild).
        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Collider::Deserialize failed: ", e.what());
        return false;
    }
}

void Collider::CloneBaseColliderData(Collider &target) const
{
    target.m_enabled = m_enabled;
    target.m_executionOrder = m_executionOrder;
    const auto &src = Data();
    auto &dst = target.DataMut();
    dst.isTrigger = src.isTrigger;
    dst.center = src.center;
    // Clone already has an authored shape, just like a deserialized collider.
    // Awake must not replace it with a new-component fit to renderer bounds.
    dst.deserialized = true;
    if (m_materialIsInstance) {
        target.SetPhysicMaterial(CopySurfaceMaterial(m_physicMaterial.Get()), true);
    } else if (m_physicMaterial.HasGuid())
        target.SetPhysicMaterialGuid(m_physicMaterial.GetGuid());
    else
        target.SetPhysicMaterial(m_physicMaterial.Get());
}

} // namespace infernux
