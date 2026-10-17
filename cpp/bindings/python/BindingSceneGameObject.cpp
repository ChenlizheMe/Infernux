#include "BindingSceneInternal.h"
#include "ComponentBindingRegistry.h"
#include "JsonPyBridge.h"
#include "core/log/InxLog.h"
#include "function/scene/ComponentFactory.h"
#include "function/scene/GameObject.h"
#include "function/scene/PyComponentProxy.h"
#include "function/scene/Scene.h"
#include "function/scene/SceneManager.h"
#include "function/scene/Transform.h"
#include <functional>

namespace py = pybind11;

namespace infernux::scene_binding
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

/// Match Python component types across body-reload generations.
///
/// ``isinstance`` remains the fast path for ordinary inheritance. A published
/// script revision creates a new Python class object, however, so a caller may
/// legitimately hold the previous class object while the live proxy owns the
/// replacement. In that case the asset-stable component type GUID is the
/// authoritative identity.
static bool MatchesPythonComponentType(const PyComponentProxy &proxy, const py::object &component,
                                       const py::object &requestedType, const std::string &requestedName)
{
    if (component.is_none())
        return false;
    if (py::isinstance<py::str>(requestedType))
        return py::str(component.attr("__class__").attr("__name__")).cast<std::string>() == requestedName;
    if (py::isinstance(component, requestedType))
        return true;
    if (!py::hasattr(requestedType, "_get_type_guid"))
        return false;

    const py::object value = requestedType.attr("_get_type_guid")();
    if (value.is_none())
        return false;
    const std::string requestedGuid = value.cast<std::string>();
    return !requestedGuid.empty() && requestedGuid == proxy.GetTypeGuid();
}

static void BindObjectIdentity(NativeClass<GameObject> &cls)
{
    cls.def_property("name", &GameObject::GetName, &GameObject::SetName)
        .def_property("active", &GameObject::IsActive, &GameObject::SetActive)
        .def_property_readonly("active_self", &GameObject::GetActiveSelf,
                               "Local active state (Unity: gameObject.activeSelf)")
        .def_property_readonly("active_in_hierarchy", &GameObject::IsActiveInHierarchy,
                               "Is active in hierarchy? (Unity: gameObject.activeInHierarchy)")
        .def_property_readonly("is_destroying", &GameObject::IsDestroying,
                               "True while the native GameObject is being retired")
        .def_property_readonly("id", &GameObject::GetID)
        .def_property_readonly("handle", &GameObject::GetHandle)
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
        .def_property("prefab_source_id", &GameObject::GetPrefabSourceID, &GameObject::SetPrefabSourceID,
                      "Stable asset-local Prefab node identity (zero for unlinked nodes)")
        .def_property_readonly("_model_source_guid", &GameObject::GetModelSourceGuid)
        .def_property_readonly("_model_source_path", &GameObject::GetModelSourcePath)
        .def("_set_model_source", &GameObject::SetModelSource)
        .def_property(
            "_prefab_source_document",
            [](const GameObject &object) { return JsonToPython(object.GetPrefabSourceDocument()); },
            [](GameObject &object, const PythonDocument &document) { object.SetPrefabSourceDocument(document.value); })
        .def_static("_reserve_document_ids",
                    [](size_t objectCount, size_t componentCount) {
                        std::vector<uint64_t> objects(objectCount), components(componentCount);
                        for (auto &id : objects)
                            id = GameObject::ReserveDocumentID();
                        for (auto &id : components)
                            id = Component::ReserveDocumentID();
                        return py::make_tuple(objects, components);
                    })
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
            "Get the Transform component");
}

static void BindNativeComponentAccess(NativeClass<GameObject> &cls)
{
    cls.def(
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
               // A registered native type that was rejected by its component
               // constraints must never fall through into Python attachment.
               if (ComponentFactory::IsRegistered(typeName)) {
                   return py::none();
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
        .def("can_add_component", &GameObject::CanAddComponentByTypeName, py::arg("type_name"),
             "Check authoritative component registry constraints before adding a native component")
        .def("get_add_component_blockers", &GameObject::GetAddComponentBlockers, py::arg("type_name"),
             "Get authoritative registry reasons that reject adding a native component")
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
        .def("get_component_order", &GameObject::GetComponentOrder,
             "Get stable component IDs in serialized/Inspector order (Transform excluded)")
        .def("set_component_order", &GameObject::SetComponentOrder, py::arg("component_ids"),
             "Atomically reorder attached components using an exact stable-ID permutation")
        .def(
            "get_component_default_document",
            [](const GameObject &obj, Component *component) {
                return JsonToPython(obj.GetDefaultComponentDocument(component));
            },
            py::arg("component"), "Build a default document while preserving component identity")
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
            py::arg("type_name"), "Get all C++ components of a given type name");
}

static void BindPythonComponentAttachment(NativeClass<GameObject> &cls)
{
    cls.def(
        "add_py_component",
        [](GameObject *obj, py::object pyComponentInstance) -> py::object {
            std::vector<Component *> autoAddedDependencies;
            const auto rollbackDependencies = [&]() {
                for (auto it = autoAddedDependencies.rbegin(); it != autoAddedDependencies.rend(); ++it) {
                    if (*it && (*it)->GetGameObject() == obj && !obj->RemoveComponent(*it))
                        INXLOG_ERROR("[Binding] Failed to roll back auto-added component '", (*it)->GetTypeName(), "'");
                }
                autoAddedDependencies.clear();
            };
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

            py::object pyType = pyComponentInstance.attr("__class__");
            py::object constraints =
                py::module_::import("infernux.components.registry").attr("get_component_constraints")(pyType);
            std::string cppTypeName;
            if (py::hasattr(pyType, "_cpp_type_name")) {
                try {
                    cppTypeName = pyType.attr("_cpp_type_name").cast<std::string>();
                } catch (...) {
                    INXLOG_WARN("[Binding] Failed to read _cpp_type_name from component type");
                    cppTypeName.clear();
                }
            }
            const bool disallowMultiple = !constraints.attr("allow_multiple").cast<bool>();

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

            const py::tuple requiredTypes = constraints.attr("required_types").cast<py::tuple>();
            bool dependencyFailed = false;
            if (requiredTypes.size() > 0) {
                for (auto reqType : requiredTypes) {
                    const std::string requiredToken = py::module_::import("infernux.components.registry")
                                                          .attr("component_constraint_type_id")(reqType)
                                                          .cast<std::string>();
                    bool found = obj->GetTransform()->IsComponentType(requiredToken);
                    for (const auto &component : obj->GetAllComponents()) {
                        if (component && component->IsComponentType(requiredToken)) {
                            found = true;
                            break;
                        }
                    }
                    if (found)
                        continue;

                    std::string nativeTypeName;
                    if (py::hasattr(reqType, "_cpp_type_name"))
                        nativeTypeName = reqType.attr("_cpp_type_name").cast<std::string>();
                    else if (py::isinstance<py::str>(reqType)) {
                        const std::string declaredName = reqType.cast<std::string>();
                        if (ComponentFactory::IsRegistered(declaredName))
                            nativeTypeName = declaredName;
                    }
                    if (!nativeTypeName.empty()) {
                        Component *required = obj->AddComponentByTypeName(nativeTypeName);
                        found = required != nullptr;
                        if (required)
                            autoAddedDependencies.push_back(required);
                    } else if (py::isinstance<py::type>(reqType)) {
                        py::object newReqComp = reqType();
                        auto reqProxy = std::make_unique<PyComponentProxy>(newReqComp);
                        Component *reqAdded = obj->AddExistingComponent(std::move(reqProxy));
                        found = reqAdded != nullptr;
                        if (reqAdded)
                            autoAddedDependencies.push_back(reqAdded);
                    }
                    if (!found) {
                        py::print("[Warning] Failed to satisfy required component", requiredToken);
                        dependencyFailed = true;
                    }
                }
            }
            if (dependencyFailed) {
                rollbackDependencies();
                return py::none();
            }

            // Create a PyComponentProxy that wraps the Python component
            auto proxy = std::make_unique<PyComponentProxy>(pyComponentInstance);
            Component *added = obj->AddExistingComponent(std::move(proxy));
            if (added) {
                // Return the original Python component
                return pyComponentInstance;
            }
            rollbackDependencies();
            return py::none();
        },
        py::arg("component_instance"), "Add a Python InxComponent instance to this GameObject");
}

static void BindPythonComponentAccess(NativeClass<GameObject> &cls)
{
    cls.def(
           "_attach_prepared_py_component",
           [](GameObject *obj, py::object instance, size_t componentIndex, uint64_t componentId) -> py::object {
               if (!py::hasattr(instance, "_bind_native_component"))
                   throw py::type_error("prepared Python component requires _bind_native_component");
               auto proxy = std::make_unique<PyComponentProxy>(instance);
               if (componentId)
                   proxy->SetComponentID(componentId);
               obj->AddPreparedPythonComponent(std::move(proxy), componentIndex);
               return instance;
           },
           py::arg("component_instance"), py::arg("component_index"), py::arg("component_id") = 0,
           "Internal deferred Python component publication hook")
        .def("_activate_prepared_py_component", &GameObject::ActivatePreparedPythonComponent,
             py::arg("native_component"), "Internal prepared Python component activation hook")
        .def("_remove_prepared_py_component", &GameObject::RemovePreparedPythonComponent, py::arg("native_component"),
             "Internal prepared Python component rollback hook")
        .def(
            "get_py_component",
            [](GameObject *obj, py::object componentType) -> py::object {
                const std::string typeName = ResolveComponentTypeName(componentType);
                for (const auto &comp : obj->GetAllComponents()) {
                    if (auto *proxy = dynamic_cast<PyComponentProxy *>(comp.get())) {
                        py::object pyComp = proxy->GetPyComponent();
                        if (MatchesPythonComponentType(*proxy, pyComp, componentType, typeName)) {
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
                if (obj == nullptr || obj->IsDestroying())
                    return result;
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
        .def(
            "replace_py_component",
            [](GameObject *obj, py::object oldComponent, py::object newComponent) -> py::object {
                if (!py::hasattr(newComponent, "_bind_native_component")) {
                    throw py::type_error("replacement Python component requires _bind_native_component");
                }

                for (const auto &component : obj->GetAllComponents()) {
                    auto *oldProxy = dynamic_cast<PyComponentProxy *>(component.get());
                    if (!oldProxy) {
                        continue;
                    }
                    py::object attached = oldProxy->GetPyComponent();
                    if (attached.is_none() || !attached.is(oldComponent)) {
                        continue;
                    }

                    auto replacement = std::make_unique<PyComponentProxy>(newComponent);
                    Component *published = obj->ReplacePythonComponent(oldProxy, std::move(replacement));
                    if (!published) {
                        throw std::runtime_error("attached Python component replacement failed");
                    }

                    if (py::hasattr(oldComponent, "_detach_native_binding_for_replacement")) {
                        oldComponent.attr("_detach_native_binding_for_replacement")();
                    } else if (py::hasattr(oldComponent, "_invalidate_native_binding")) {
                        oldComponent.attr("_invalidate_native_binding")();
                    }

                    return newComponent;
                }
                return py::none();
            },
            py::arg("old_component"), py::arg("new_component"),
            "Replace an attached Python component without changing native identity or invoking on_destroy");
}

static void BindObjectHierarchy(NativeClass<GameObject> &cls)
{
    cls.def("get_parent", &GameObject::GetParent, py::return_value_policy::reference, "Get the parent GameObject")
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
        .def(
            "serialize_document", [](const GameObject &object) { return JsonToPython(object.SerializeDocument()); },
            "Serialize GameObject to a Python document")
        .def(
            "_commit_document",
            [](GameObject &object, const PythonDocument &document, bool preserveDocumentIds) {
                return object.DeserializeDocument(document.value, preserveDocumentIds);
            },
            py::arg("document"), py::arg("preserve_document_ids") = true,
            "Internal native subtree commit; Python callers must preflight first");
}

static void BindHierarchyComponentSearch(NativeClass<GameObject> &cls)
{
    cls.def(
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
                                   const bool match =
                                       MatchesPythonComponentType(*proxy, pyComp, componentType, typeName);
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
                                    const bool match =
                                        MatchesPythonComponentType(*proxy, pyComp, componentType, typeName);
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
            "Get a component on this or any parent GameObject. Unity: GetComponentInParent<T>()");
}

static void BindObjectRuntimeQueries(NativeClass<GameObject> &cls)
{
    cls.def_static(
           "find",
           [](const std::string &name) -> GameObject * { return SceneManager::Instance().FindRuntimeObject(name); },
           py::return_value_policy::reference, py::arg("name"),
           "Find a GameObject by name across loaded scenes. Unity: GameObject.Find(name)")
        .def_static(
            "find_with_tag",
            [](const std::string &tag) -> GameObject * {
                return SceneManager::Instance().FindRuntimeObjectWithTag(tag);
            },
            py::return_value_policy::reference, py::arg("tag"),
            "Find the first GameObject with a given tag. Unity: GameObject.FindWithTag(tag)")
        .def_static(
            "find_game_objects_with_tag",
            [](const std::string &tag) -> std::vector<GameObject *> {
                return SceneManager::Instance().FindRuntimeObjectsWithTag(tag);
            },
            py::return_value_policy::reference, py::arg("tag"),
            "Find all GameObjects with a given tag. Unity: GameObject.FindGameObjectsWithTag(tag)")
        // ---- Static lifecycle methods (Unity: Object.Destroy) ----
        .def_static(
            "destroy",
            [](GameObject *gameObject) {
                if (gameObject && gameObject->GetScene()) {
                    gameObject->GetScene()->DestroyGameObject(gameObject);
                }
            },
            py::arg("game_object"),
            "Destroy a GameObject (removed at end of frame). Unity: Object.Destroy(gameObject)");
}

void BindGameObject(py::module_ &m)
{
    NativeClass<GameObject> cls(m, "GameObject");
    BindObjectIdentity(cls);
    BindNativeComponentAccess(cls);
    BindPythonComponentAttachment(cls);
    BindPythonComponentAccess(cls);
    BindObjectHierarchy(cls);
    BindHierarchyComponentSearch(cls);
    BindObjectRuntimeQueries(cls);
}

} // namespace infernux::scene_binding
