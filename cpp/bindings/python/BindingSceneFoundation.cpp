#include "BindingSceneInternal.h"
#include "ComponentBindingRegistry.h"
#include "JsonPyBridge.h"
#include "MatrixPyBridge.h"
#include "function/scene/BoxCollider.h"
#include "function/scene/Camera.h"
#include "function/scene/CapsuleCollider.h"
#include "function/scene/ComponentFactory.h"
#include "function/scene/CylinderCollider.h"
#include "function/scene/GameObject.h"
#include "function/scene/Light.h"
#include "function/scene/LineRenderer.h"
#include "function/scene/MeshCollider.h"
#include "function/scene/MeshRenderer.h"
#include "function/scene/ObjectHandle.h"
#include "function/scene/PyComponentProxy.h"
#include "function/scene/Rigidbody.h"
#include "function/scene/Scene.h"
#include "function/scene/SkinnedMeshRenderer.h"
#include "function/scene/SphereCollider.h"
#include "function/scene/Transform.h"
#include "function/scene/UITransformDependencies.h"

namespace py = pybind11;

namespace infernux::scene_binding
{

void BindUITransformDependencies(py::module_ &m)
{
    py::class_<UITransformDependencies>(m, "_UITransformDependencies")
        .def(py::init<const std::vector<GameObject *> &, const std::vector<GameObject *> &>(), py::arg("screen"),
             py::arg("world"))
        .def("poll", &UITransformDependencies::Poll)
        .def_property_readonly("changed_entries", &UITransformDependencies::GetChangedEntries)
        .def("project_world_ray", &UITransformDependencies::ProjectWorldRay, py::arg("origin"), py::arg("direction"),
             py::arg("layer_mask"))
        .def(
            "project_world_ray_with_policies",
            [](UITransformDependencies &self, const glm::vec3 &origin, const glm::vec3 &direction, uint32_t layerMask,
               const std::vector<uint8_t> &policies, py::object view, py::object projection, float viewportHeight) {
                return self.ProjectWorldRayWithPolicies(
                    origin, direction, layerMask, policies, binding::Matrix4FromPython(view, "view"),
                    binding::Matrix4FromPython(projection, "projection"), viewportHeight);
            },
            py::arg("origin"), py::arg("direction"), py::arg("layer_mask"), py::arg("policies"), py::arg("view"),
            py::arg("projection"), py::arg("viewport_height"));
}

void BindPrimitiveTypes(py::module_ &m)
{
    // PrimitiveType enum
    py::enum_<PrimitiveType>(m, "PrimitiveType")
        .value("Cube", PrimitiveType::Cube)
        .value("Sphere", PrimitiveType::Sphere)
        .value("Capsule", PrimitiveType::Capsule)
        .value("Cylinder", PrimitiveType::Cylinder)
        .value("Plane", PrimitiveType::Plane)
        .value("Quad", PrimitiveType::Quad)
        .export_values();

    m.def("get_builtin_primitive_mesh", &GetBuiltinPrimitiveMeshAsset, py::arg("name"),
          "Return an immutable engine built-in primitive Mesh resource");

    // Space enum (Unity: Space.Self, Space.World)
    py::enum_<CoordinateSpace>(m, "Space")
        .value("Self", CoordinateSpace::Self)
        .value("World", CoordinateSpace::World)
        .export_values();
}

void BindObjectHandle(py::module_ &m)
{
    py::class_<ObjectHandle>(m, "ObjectHandle")
        .def(py::init<>())
        .def(py::init<uint64_t, uint64_t, uint64_t>(), py::arg("id"), py::arg("generation"), py::arg("world_id"))
        .def_readonly("id", &ObjectHandle::id)
        .def_readonly("generation", &ObjectHandle::generation)
        .def_property_readonly("world_id", [](const ObjectHandle &handle) { return handle.worldId; })
        .def_property_readonly("is_valid", &ObjectHandle::IsValid)
        .def("__bool__", &ObjectHandle::IsValid)
        .def("__eq__", [](const ObjectHandle &lhs, const ObjectHandle &rhs) { return lhs == rhs; })
        .def("__hash__", [](const ObjectHandle &handle) {
            return py::hash(py::make_tuple(handle.id, handle.generation, handle.worldId));
        });
}

void BindComponent(py::module_ &m)
{
    NativeClass<Component>(m, "Component")
        .def_property_readonly("type_name", &Component::GetTypeName)
        .def_property_readonly("component_id", &Component::GetComponentID)
        .def_property_readonly("handle", &Component::GetHandle)
        .def_property("_prefab_source_id", &Component::GetPrefabSourceID, &Component::SetPrefabSourceID)
        .def("_set_component_id", &Component::SetComponentID, py::arg("component_id"),
             "Internal transactional restore hook")
        .def_property("enabled", &Component::IsEnabled, &Component::SetEnabled)
        .def_property("execution_order", &Component::GetExecutionOrder, &Component::SetExecutionOrder)
        .def_property_readonly(
            "game_object", [](Component *c) { return c->GetGameObject(); }, py::return_value_policy::reference,
            "Get the GameObject this component is attached to")
        .def("get_transform", &Component::GetTransform, py::return_value_policy::reference,
             "Get the Transform attached to this component's owner")
        .def("serialize", &Component::Serialize, "Serialize component to JSON string")
        .def("deserialize", &Component::Deserialize, py::arg("json_str"), "Deserialize component from JSON string")
        .def(
            "serialize_document",
            [](const Component &component) { return JsonToPython(component.SerializeDocument()); },
            "Serialize component to a Python document")
        .def(
            "validate_document",
            [](const Component &component, const PythonDocument &document) {
                ComponentFactory::ValidateDocument(component.GetTypeName(), document.value);
            },
            py::arg("document"), "Validate a complete candidate document without modifying the component")
        .def(
            "deserialize_document",
            [](Component &component, const PythonDocument &document) {
                return component.DeserializeDocument(document.value);
            },
            py::arg("document"), "Deserialize component from a Python document")
        .def_property_readonly("required_component_types", &Component::GetRequiredComponentTypes,
                               "List of type names this component depends on (RequireComponent)")
        .def("is_component_type", &Component::IsComponentType, py::arg("type_name"),
             "Check if this component matches a given type name (including base types)");
}

void BindTransform(py::module_ &m)
{
    NativeClass<Transform, Component>(m, "Transform")
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
        // Combined local TRS setter — single boundary crossing + one subtree
        // invalidation. Per-frame animation fast path (TimelineAction etc.).
        .def(
            "set_local_trs",
            [](Transform *t, float px, float py, float pz, float rx, float ry, float rz, float sx, float sy, float sz) {
                t->SetLocalTRS(glm::vec3(px, py, pz), glm::vec3(rx, ry, rz), glm::vec3(sx, sy, sz));
            },
            py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("rx"), py::arg("ry"), py::arg("rz"), py::arg("sx"),
            py::arg("sy"), py::arg("sz"),
            "Set local position+euler(deg)+scale in one call (single dirty/invalidate). "
            "Fast path for per-frame animation; avoids 3 separate setters + Vector3 allocations.")
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
}

void RegisterSceneComponentCasters()
{
    auto &registry = ComponentBindingRegistry::Instance();
    registry.Register("MeshRenderer", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<MeshRenderer *>(c), py::return_value_policy::reference);
    });
    registry.Register("LineRenderer", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<LineRenderer *>(c), py::return_value_policy::reference);
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
    registry.Register("CylinderCollider", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<CylinderCollider *>(c), py::return_value_policy::reference);
    });
    registry.Register("MeshCollider", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<MeshCollider *>(c), py::return_value_policy::reference);
    });
    registry.Register("Rigidbody", [](Component *c) -> py::object {
        return py::cast(dynamic_cast<Rigidbody *>(c), py::return_value_policy::reference);
    });

    // NOTE: AudioSource and AudioListener casters are registered in BindingAudio.cpp
    // because RegisterAudioBindings runs after RegisterSceneBindings.
}

void BindPyComponentProxy(py::module_ &m)
{
    NativeClass<PyComponentProxy, Component>(m, "PyComponentProxy")
        .def("get_py_component", &PyComponentProxy::GetPyComponent, "Get the underlying Python component")
        .def("get_py_type_name", &PyComponentProxy::GetPyTypeName, "Get the Python type name")
        .def("is_valid", &PyComponentProxy::IsValid, "Check if this proxy holds a valid Python component")
        .def("refresh_python_lifecycle_dispatch", &PyComponentProxy::RefreshPythonLifecycleDispatch,
             "Refresh cached non-frame lifecycle wrappers and physics callback gates")
        .def("_reset_lifecycle_for_play", &PyComponentProxy::ResetLifecycleForPlay,
             "Internal Edit-to-Play lifecycle reset for a fresh Python mirror");
}

} // namespace infernux::scene_binding
