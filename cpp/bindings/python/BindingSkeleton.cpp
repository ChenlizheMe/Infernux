#include "BindingSceneInternal.h"
#include "ComponentBindingRegistry.h"
#include "MatrixPyBridge.h"
#include "function/scene/SkeletonInstance.h"
#include "function/scene/Transform.h"

namespace py = pybind11;
namespace infernux::scene_binding
{
void BindSkeletonInstance(py::module_ &m)
{
    NativeClass<SkeletonInstance, Component>(m, "SkeletonInstance")
        .def_property("model", &SkeletonInstance::GetSourceModelGuid, &SkeletonInstance::SetSourceModelGuid)
        .def_property_readonly("root", &SkeletonInstance::GetRootBone, py::return_value_policy::reference)
        .def_property_readonly("bones", &SkeletonInstance::GetBones, py::return_value_policy::reference)
        .def_property_readonly("bone_names", &SkeletonInstance::GetBoneNames)
        .def_property_readonly("parent_indices", &SkeletonInstance::GetParentIndices)
        .def_property_readonly("physics_controls_pose", &SkeletonInstance::PhysicsControlsPose)
        .def_property_readonly("has_animation_pose", &SkeletonInstance::HasAnimationPose)
        .def("capture_animation_pose", &SkeletonInstance::CaptureAnimationPose)
        .def("apply_animation_pose", &SkeletonInstance::ApplyAnimationPose)
        .def("get_animation_world_matrices",
             [](const SkeletonInstance &s) { return binding::Matrix4ArrayToPython(s.GetAnimationWorldMatrices()); })
        .def("get_bone_transform", &SkeletonInstance::GetBoneTransform, py::return_value_policy::reference)
        .def("get_human_bone_transform", &SkeletonInstance::GetHumanBoneTransform, py::return_value_policy::reference)
        .def("get_bind_local_matrices",
             [](const SkeletonInstance &s) { return binding::Matrix4ArrayToPython(s.GetBindLocalMatrices()); })
        .def("get_local_matrices",
             [](const SkeletonInstance &s) { return binding::Matrix4ArrayToPython(s.GetLocalMatrices()); })
        .def("get_root_space_matrices",
             [](const SkeletonInstance &s) { return binding::Matrix4ArrayToPython(s.GetRootSpaceMatrices()); })
        .def("get_world_matrices",
             [](const SkeletonInstance &s) { return binding::Matrix4ArrayToPython(s.GetWorldMatrices()); })
        .def("rebuild_hierarchy", &SkeletonInstance::RebuildHierarchy)
        .def("reset_to_bind_pose", &SkeletonInstance::ResetToBindPose);
    ComponentBindingRegistry::Instance().Register("SkeletonInstance", [](Component *component) {
        return py::cast(static_cast<SkeletonInstance *>(component), py::return_value_policy::reference);
    });
}
} // namespace infernux::scene_binding
