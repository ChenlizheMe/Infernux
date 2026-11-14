#include "GameObject.h"
#include "Rigidbody.h"
#include "Scene.h"
#include "SkeletonInstance.h"
#include "SkeletonPhysicsPoseRead.h"
#include "Transform.h"
#include "physics/PhysicsWorld.h"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{
TransformECSStore &SkeletonPhysicsPoseRead::Prepare(const std::vector<Transform *> &transforms, Space space)
{
    auto &store = TransformECSStore::Instance();
    if (!store.IsPhysicsPoseReadSuppressed())
        PhysicsWorld::Instance().PrepareCpuPoseReads(transforms, space == Space::World);
    return store;
}

std::vector<Transform *> SkeletonInstance::GetPhysicsControlledTransforms() const
{
    auto *root = GetGameObject();
    std::vector<Transform *> transforms;
    if (!m_model || !root || !root->GetScene() || !root->GetScene()->IsPlaying())
        return transforms;
    const auto visit = [&](auto &&self, GameObject *object) -> void {
        if (!object->IsActiveInHierarchy())
            return;
        if (auto *body = object->GetComponent<Rigidbody>();
            body && body->IsEnabled() && !body->IsKinematic() && body->GetParticipatingBodyId() != 0xFFFFFFFF)
            transforms.push_back(object->GetTransform());
        for (const auto &child : object->GetChildren())
            self(self, child.get());
    };
    visit(visit, root);
    return transforms;
}

bool SkeletonInstance::PhysicsControlsPose() const
{
    return !GetPhysicsControlledTransforms().empty();
}

void SkeletonInstance::CaptureAnimationPose()
{
    EnsureBindings();
    SkeletonPhysicsPoseRead reading(GetBones(), SkeletonPhysicsPoseRead::Space::Local);
    m_animationPose.clear();
    m_animationPose.reserve(m_nodes.size());
    for (size_t index = 0; index < m_nodes.size(); ++index) {
        const auto *bone = ResolveBone(index);
        m_animationPose.push_back({bone->GetLocalPosition(), bone->GetLocalRotation(), bone->GetLocalScale()});
    }
}

void SkeletonInstance::ApplySampledPose(const EvaluatedSkinPose &pose)
{
    EnsureBindings();
    if (pose.localPoses.size() != m_nodes.size())
        throw std::invalid_argument("Sampled pose must contain this skeleton's complete node layout");
    m_animationPose = pose.localPoses;
    for (auto &local : m_animationPose)
        local.translation *= m_model->scaleFactor;
    ApplyAnimationPose();
}

void SkeletonInstance::ApplyAnimationPose()
{
    EnsureBindings();
    if (!HasAnimationPose())
        throw std::logic_error("Skeleton has no animation target; sample or capture a pose first");

    const auto controlledTransforms = GetPhysicsControlledTransforms();
    SkeletonPhysicsPoseRead reading(controlledTransforms, SkeletonPhysicsPoseRead::Space::World);
    struct PhysicalPose
    {
        Transform *transform;
        glm::vec3 position;
        glm::quat rotation;
    };
    std::vector<PhysicalPose> physical;
    std::unordered_set<Transform *> controlled, physicalScale;
    // Capture the already presented world pose, including sleeping bodies and
    // frames without a solver step. This also retains explicit earlier script
    // edits. Animation must not turn these values into a teleport or a wake-up.
    for (auto *transform : controlledTransforms) {
        physical.push_back({transform, transform->GetPosition(), transform->GetWorldRotation()});
        controlled.insert(transform);
        for (auto *object = transform->GetGameObject(); object; object = object->GetParent()) {
            physicalScale.insert(object->GetTransform());
            if (object == GetGameObject())
                break;
        }
    }
    for (size_t index = 0; index < m_nodes.size(); ++index) {
        auto *bone = ResolveBone(index);
        const auto &local = m_animationPose[index];
        if (!controlled.count(bone)) {
            bone->SetLocalPosition(local.translation);
            bone->SetLocalRotation(local.rotation);
        }
        // Physical dimensions remain authored. An animated helper scale must
        // not silently resize descendant collision shapes or their inertia.
        if (!physicalScale.count(bone))
            bone->SetLocalScale(local.scale);
    }
    // Hierarchy traversal above is parent-first. Restore full world rotations
    // even for FreezeRotation bodies; locking the solver is not animation ownership.
    for (const auto &pose : physical)
        pose.transform->ApplyWorldPoseFromPhysics(pose.position, pose.rotation, true);
}

std::vector<glm::mat4> SkeletonInstance::GetAnimationWorldMatrices() const
{
    EnsureBindings();
    if (!HasAnimationPose())
        throw std::logic_error("Skeleton has no animation target; sample or capture a pose first");
    std::vector<glm::mat4> matrices;
    matrices.reserve(m_nodes.size());
    const auto root = GetGameObject()->GetTransform()->GetWorldMatrix();
    for (size_t index = 0; index < m_nodes.size(); ++index) {
        const auto &pose = m_animationPose[index];
        const auto local = glm::translate(glm::mat4(1), pose.translation) * glm::mat4_cast(pose.rotation) *
                           glm::scale(glm::mat4(1), pose.scale);
        const int parent = m_model->skeleton.nodes[index].parent;
        matrices.push_back((parent < 0 ? root : matrices[parent]) * local);
    }
    return matrices;
}

std::vector<glm::quat> SkeletonInstance::GetAnimationWorldRotations() const
{
    EnsureBindings();
    if (!HasAnimationPose())
        throw std::logic_error("Skeleton has no animation target; sample or capture a pose first");
    std::vector<glm::quat> rotations;
    rotations.reserve(m_nodes.size());
    const auto root = GetGameObject()->GetTransform()->GetWorldRotation();
    for (size_t index = 0; index < m_nodes.size(); ++index) {
        const int parent = m_model->skeleton.nodes[index].parent;
        rotations.push_back(glm::normalize((parent < 0 ? root : rotations[parent]) * m_animationPose[index].rotation));
    }
    return rotations;
}

void SkeletonInstance::AlignAnimationRoot(Transform *physicalRoot, const glm::vec3 &physicalPosition,
                                          const glm::quat &physicalRotation)
{
    const auto bones = GetBones();
    const auto found = std::find(bones.begin(), bones.end(), physicalRoot);
    if (found == bones.end() || !HasAnimationPose())
        return;
    // An independent dynamic owner keeps its own pose authority. Otherwise
    // relocating the origin is an ordinary authored move, including any root collider.
    if (auto *body = GetGameObject()->GetComponent<Rigidbody>();
        body && body->IsEnabled() && !body->IsKinematic() && body->GetParticipatingBodyId() != 0xFFFFFFFF)
        return;
    const auto controlledTransforms = GetPhysicsControlledTransforms();
    auto dependencies = controlledTransforms;
    dependencies.push_back(physicalRoot);
    dependencies.push_back(GetGameObject()->GetTransform());
    SkeletonPhysicsPoseRead reading(dependencies, SkeletonPhysicsPoseRead::Space::World);
    // Use the actually applied target, including authored scales preserved for
    // independent physical attachments, rather than the unconstrained sample.
    const auto delta = glm::normalize(physicalRotation * glm::conjugate(physicalRoot->GetWorldRotation()));
    auto *owner = GetGameObject()->GetTransform();
    const auto rotation = glm::normalize(delta * owner->GetWorldRotation());
    // Moving the character origin must not move independent dynamic attachments
    // or nested characters. Preserve their presented poses before changing the owner.
    struct PhysicalPose
    {
        Transform *transform;
        glm::vec3 position;
        glm::quat rotation;
    };
    std::vector<PhysicalPose> physical;
    for (auto *transform : controlledTransforms)
        physical.push_back({transform, transform->GetPosition(), transform->GetWorldRotation()});
    owner->SetRotation(rotation);
    // Evaluate the real hierarchy after rotation: under a nonuniformly scaled
    // ancestor its point offsets do not transform as a free rigid body.
    owner->SetPosition(owner->GetPosition() + physicalPosition - physicalRoot->GetPosition());
    for (const auto &pose : physical)
        pose.transform->ApplyWorldPoseFromPhysics(pose.position, pose.rotation, true);
}
} // namespace infernux
