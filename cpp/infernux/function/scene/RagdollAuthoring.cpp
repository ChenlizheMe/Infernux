#include "BoxCollider.h"
#include "GameObject.h"
#include "Ragdoll.h"
#include "Rigidbody.h"
#include "SkeletonInstance.h"
#include "SwingTwistJoint.h"
#include "Transform.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace infernux
{
bool Ragdoll::OwnsObject(const GameObject *object) const
{
    const auto *root = GetGameObject();
    for (; object; object = object->GetParent()) {
        if (object == root)
            return true;
        if (object->GetComponent<Ragdoll>())
            return false;
    }
    return false;
}

Component *Ragdoll::ResolveOwnedComponent(uint64_t id) const
{
    auto *component = Component::FindByComponentId(id);
    auto *object = component ? component->GetGameObject() : nullptr;
    return object && OwnsObject(object) ? component : nullptr;
}

SkeletonInstance *Ragdoll::GetSkeleton() const
{
    if (auto *skeleton = dynamic_cast<SkeletonInstance *>(ResolveOwnedComponent(m_skeletonComponentId)))
        return skeleton;
    auto *root = GetGameObject();
    if (!root)
        return nullptr;
    SkeletonInstance *found = nullptr;
    const auto visit = [&](auto &&self, GameObject *object) -> void {
        if (object != root && object->GetComponent<Ragdoll>())
            return;
        if (auto *skeleton = object->GetComponent<SkeletonInstance>()) {
            if (found)
                throw std::invalid_argument(
                    "Attach Ragdoll to one character root, not a container of multiple skeletons");
            found = skeleton;
            return; // A nested character is not part of this skeleton.
        }
        for (const auto &child : object->GetChildren())
            self(self, child.get());
    };
    visit(visit, root);
    return found;
}

void Ragdoll::OnSkeletonHierarchyReady(SkeletonInstance &skeleton)
{
    if (GetSkeleton() != &skeleton)
        return;
    m_skeletonComponentId = skeleton.GetComponentID();
    for (auto *bone : skeleton.GetBones())
        if (bone->GetGameObject()->GetComponent<Rigidbody>())
            return; // Existing authoring values are authoritative.
    GeneratePhysics();
}

void Ragdoll::OnSkeletonDefinitionChanging(SkeletonInstance &skeleton, const InxSkinnedMesh *definition)
{
    if (m_skeletonComponentId != skeleton.GetComponentID())
        return;
    std::set<std::vector<std::string>> retained;
    if (definition) {
        const auto paths = definition->skeleton.GetNodePaths();
        retained.insert(paths.begin(), paths.end());
    }
    std::unordered_set<Component *> retiring;
    for (auto id : m_generatedComponentIds)
        if (auto *component = ResolveOwnedComponent(id))
            if (!definition || component->GetGameObject()->GetModelSourceGuid() != definition->guid ||
                !retained.count(component->GetGameObject()->GetModelSourcePath()))
                retiring.insert(component);
    // A user-added component may now depend on a generated Rigidbody. Diagnose
    // that authoring conflict before changing the model or removing anything.
    for (auto *component : retiring) {
        auto *object = component->GetGameObject();
        for (const auto &type : object->GetRemovalBlockingComponentTypes(component))
            for (const auto &other : object->GetAllComponents())
                if (type == other->GetTypeName() && !retiring.count(other.get()))
                    throw std::logic_error("Remove the authored dependency before changing this ragdoll skeleton");
    }
    if (!retiring.empty())
        Release();
    for (auto item = m_generatedComponentIds.rbegin(); item != m_generatedComponentIds.rend(); ++item)
        if (auto *component = ResolveOwnedComponent(*item); component && retiring.count(component))
            if (!component->GetGameObject()->RemoveComponent(component))
                throw std::logic_error("Failed to retire generated ragdoll physics");
    m_generatedComponentIds.erase(std::remove_if(m_generatedComponentIds.begin(), m_generatedComponentIds.end(),
                                                 [&](auto id) { return !ResolveOwnedComponent(id); }),
                                  m_generatedComponentIds.end());
}

namespace
{
struct BoneFit
{
    glm::vec3 minimum{std::numeric_limits<float>::infinity()};
    glm::vec3 maximum{-std::numeric_limits<float>::infinity()};
    bool hasVertices = false;
    void Include(glm::vec3 point)
    {
        minimum = glm::min(minimum, point);
        maximum = glm::max(maximum, point);
        hasVertices = true;
    }
};

bool IsDeformationBone(const SkinnedRuntimeBone &bone)
{
    // This reserved importer record makes unweighted mesh attachments follow
    // their node. It does not describe an extra physical bone.
    return bone.nodeIndex >= 0 && bone.name != "__mesh_node_fallback_" + std::to_string(bone.nodeIndex);
}
} // namespace

size_t Ragdoll::GeneratePhysics(float totalMass)
{
    auto *skeleton = GetSkeleton();
    if (!skeleton || !skeleton->HasBoundHierarchy())
        throw std::logic_error("Automatic ragdoll configuration requires a bound SkeletonInstance");
    const auto model = skeleton->GetModelSnapshot();
    if (!model->IsValid())
        throw std::logic_error("Automatic ragdoll fitting requires skinned geometry");
    const auto transforms = skeleton->GetBones();
    for (auto *bone : transforms)
        if (bone->GetGameObject()->GetComponent<Rigidbody>())
            return 0;
    const auto &nodes = model->skeleton.nodes;
    std::set<int> selected;
    if (model->humanoid.IsValid()) {
        static const std::set<std::string> slots{"hips",
                                                 "spine",
                                                 "head",
                                                 "left_upper_arm",
                                                 "left_lower_arm",
                                                 "left_hand",
                                                 "right_upper_arm",
                                                 "right_lower_arm",
                                                 "right_hand",
                                                 "left_upper_leg",
                                                 "left_lower_leg",
                                                 "left_foot",
                                                 "right_upper_leg",
                                                 "right_lower_leg",
                                                 "right_foot"};
        for (const auto &[name, index] : model->humanoid.bones)
            if (slots.count(name))
                selected.insert(index);
    } else {
        for (const auto &influence : model->influences)
            for (size_t slot = 0; slot < kMaxSkinInfluences; ++slot)
                if (influence.weight[slot] > 0) {
                    const auto &bone = model->skeleton.bones.at(influence.boneIndex[slot]);
                    if (IsDeformationBone(bone))
                        selected.insert(bone.nodeIndex);
                }
    }
    if (selected.empty())
        throw std::logic_error("The model has no deformation bones for automatic ragdoll fitting");
    if (!std::isfinite(totalMass) || totalMass < .001f * selected.size())
        throw std::invalid_argument("Total ragdoll mass must allow at least 0.001 kg per physical bone");

    std::vector<int> physicalOwner(nodes.size(), -1), physicalParent(nodes.size(), -1);
    auto bind = skeleton->GetBindLocalMatrices();
    for (size_t index = 0; index < nodes.size(); ++index) {
        const int parent = nodes[index].parent;
        if (parent >= 0)
            bind[index] = bind[parent] * bind[index];
        physicalParent[index] = parent < 0 ? -1 : physicalOwner[parent];
        physicalOwner[index] =
            selected.count(static_cast<int>(index)) ? static_cast<int>(index) : physicalParent[index];
    }
    std::vector<glm::mat4> inverseBind(nodes.size());
    std::vector<BoneFit> dominant(nodes.size()), support(nodes.size());
    for (int index : selected) {
        const auto *object = transforms[index]->GetGameObject();
        if (!OwnsObject(object))
            throw std::logic_error("A selected physical bone belongs to a nested Ragdoll");
        if (!object->CanAddComponentByTypeName("Rigidbody") || object->GetComponent<Collider>() ||
            object->GetComponent<Joint>())
            throw std::logic_error("Remove existing bone physics before generating a ragdoll");
        if (glm::determinant(bind[index]) == 0.0f)
            throw std::invalid_argument("Cannot fit physics to a collapsed skeleton bind transform");
        inverseBind[index] = glm::inverse(bind[index]);
    }
    const auto vertices = model->SampleVertices(SkinnedSampleRequest{});
    for (size_t vertex = 0; vertex < vertices.size(); ++vertex) {
        const auto &influence = model->influences.at(vertex);
        std::array<int, kMaxSkinInfluences> owners{-1, -1, -1, -1};
        std::array<float, kMaxSkinInfluences> weights{};
        for (size_t slot = 0; slot < kMaxSkinInfluences; ++slot) {
            if (influence.weight[slot] <= 0)
                continue;
            const auto &bone = model->skeleton.bones.at(influence.boneIndex[slot]);
            if (!IsDeformationBone(bone))
                continue;
            const int owner = physicalOwner[bone.nodeIndex];
            if (owner < 0)
                continue;
            size_t group = 0;
            while (owners[group] != -1 && owners[group] != owner)
                ++group;
            owners[group] = owner;
            weights[group] += influence.weight[slot];
        }
        int winner = -1;
        float maximumWeight = 0;
        for (size_t group = 0; group < kMaxSkinInfluences && owners[group] >= 0; ++group) {
            const int owner = owners[group];
            const auto point = glm::vec3(inverseBind[owner] * glm::vec4(vertices[vertex].pos, 1));
            support[owner].Include(point);
            if (weights[group] > maximumWeight) {
                maximumWeight = weights[group];
                winner = owner;
            }
        }
        if (winner >= 0)
            dominant[winner].Include(glm::vec3(inverseBind[winner] * glm::vec4(vertices[vertex].pos, 1)));
    }

    struct Part
    {
        GameObject *object;
        int parent;
        std::unique_ptr<Rigidbody> body;
        std::unique_ptr<BoxCollider> collider;
        std::unique_ptr<SwingTwistJoint> joint;
        double volume;
    };
    std::vector<Part> parts;
    std::unordered_map<int, size_t> partByNode;
    double volumeSum = 0;
    for (int index : selected) {
        auto fit = dominant[index].hasVertices ? dominant[index] : support[index];
        if (!fit.hasVertices) {
            fit.Include(glm::vec3(0));
            for (size_t child = index + 1; child < nodes.size(); ++child)
                if (physicalParent[child] == index && selected.count(static_cast<int>(child)))
                    fit.Include(glm::vec3(inverseBind[index] * bind[child][3]));
            // An unmapped leaf has no geometric volume to fit. Its parent
            // segment supplies an explicit initial physical size.
            if (fit.minimum == fit.maximum) {
                const int parent = physicalParent[index];
                if (parent < 0)
                    throw std::logic_error("A physical root has neither skin vertices nor a bone segment");
                const float length = glm::length(glm::vec3(inverseBind[index] * bind[parent][3]));
                fit.minimum = glm::vec3(-length * .15f);
                fit.maximum = -fit.minimum;
            }
        }
        const auto size = glm::max(fit.maximum - fit.minimum, glm::vec3(.001f));
        Part part{transforms[index]->GetGameObject(),
                  physicalParent[index],
                  std::make_unique<Rigidbody>(),
                  std::make_unique<BoxCollider>(),
                  nullptr,
                  double(size.x) * size.y * size.z * std::abs(glm::determinant(glm::dmat3(bind[index])))};
        part.collider->SetCenter((fit.minimum + fit.maximum) * .5f);
        part.collider->SetSize(size);
        if (part.parent >= 0) {
            part.joint = std::make_unique<SwingTwistJoint>();
            part.joint->SetConnectedBody(parts.at(partByNode.at(part.parent)).body.get());
            glm::vec3 axis{0};
            for (size_t child = index + 1; child < nodes.size(); ++child)
                if (physicalParent[child] == index && selected.count(static_cast<int>(child))) {
                    const auto segment = glm::vec3(inverseBind[index] * bind[child][3]);
                    if (glm::dot(segment, segment) > glm::dot(axis, axis))
                        axis = segment;
                }
            if (glm::dot(axis, axis) == 0)
                axis = -glm::vec3(inverseBind[index] * bind[part.parent][3]);
            if (glm::dot(axis, axis) == 0) {
                int dimension = size.y > size.x ? 1 : 0;
                if (size.z > size[dimension])
                    dimension = 2;
                axis[dimension] = 1;
            }
            axis = glm::normalize(axis);
            const auto reference = std::abs(axis.z) < .9f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
            part.joint->SetFrame(axis, glm::normalize(glm::cross(axis, reference)));
        }
        volumeSum += part.volume;
        partByNode.emplace(index, parts.size());
        parts.push_back(std::move(part));
    }
    for (auto &part : parts)
        part.body->SetMass(.001f + static_cast<float>((totalMass - .001f * parts.size()) * part.volume / volumeSum));

    // Every part is prepared before publication. Actual component state is the
    // only physics configuration, saved normally with its Scene or Prefab.
    m_skeletonComponentId = skeleton->GetComponentID();
    std::vector<uint64_t> created;
    const auto attach = [&](GameObject &object, std::unique_ptr<Component> component) {
        const auto id = component->GetComponentID();
        created.push_back(id);
        if (!object.AddExistingComponent(std::move(component)))
            throw std::logic_error("Could not publish generated bone physics");
    };
    try {
        for (auto &part : parts)
            attach(*part.object, std::move(part.body));
        for (auto &part : parts)
            attach(*part.object, std::move(part.collider));
        for (auto &part : parts)
            if (part.joint)
                attach(*part.object, std::move(part.joint));
    } catch (...) {
        for (auto item = created.rbegin(); item != created.rend(); ++item)
            if (auto *component = ResolveOwnedComponent(*item))
                component->GetGameObject()->RemoveComponent(component);
        throw;
    }
    m_generatedComponentIds.insert(m_generatedComponentIds.end(), created.begin(), created.end());
    Rebuild();
    return parts.size();
}

float Ragdoll::GetTotalMass() const
{
    double total = 0;
    for (auto *body : GetBodies())
        total += body->GetMass();
    return static_cast<float>(total);
}

void Ragdoll::SetTotalMass(float totalMass)
{
    const float previous = GetTotalMass();
    if (!std::isfinite(totalMass) || totalMass <= 0 || previous <= 0)
        throw std::invalid_argument(
            "Set ragdoll mass after configuring its physical bones, with a positive finite mass");
    const auto bodies = GetBodies();
    std::vector<float> masses;
    for (auto *body : bodies) {
        const float mass = static_cast<float>(double(body->GetMass()) * totalMass / previous);
        if (!std::isfinite(mass) || mass < .001f)
            throw std::invalid_argument("Total ragdoll mass must keep every physical bone at or above 0.001 kg");
        body->ValidateMass(mass);
        masses.push_back(mass);
    }
    for (size_t index = 0; index < bodies.size(); ++index)
        bodies[index]->SetMass(masses[index]);
}
} // namespace infernux
