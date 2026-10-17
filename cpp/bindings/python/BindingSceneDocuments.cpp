#include "BindingSceneInternal.h"
#include "JsonPyBridge.h"
#include "core/threading/JobSystem.h"
#include "function/scene/GameObject.h"
#include "function/scene/Scene.h"
#include "function/scene/SceneDocumentReadTask.h"
#include "function/scene/SceneResourceDependencyPreflight.h"
#include <string_view>

namespace py = pybind11;

namespace infernux
{

class ScenePlayModeSnapshot
{
  public:
    explicit ScenePlayModeSnapshot(nlohmann::json document) : m_document(std::move(document))
    {
    }

    [[nodiscard]] py::tuple GetPythonComponentRecords() const
    {
        static constexpr std::string_view kNativeTypePrefix = "native:infernux.";
        py::set objectIds;
        py::set nativeTypes;
        py::list descriptors;

        const auto visit = [&](const auto &self, const nlohmann::json &object) -> void {
            if (!object.is_object())
                return;

            const auto idIt = object.find("id");
            if (idIt == object.end() || !idIt->is_number_unsigned())
                return;
            const uint64_t objectId = idIt->get<uint64_t>();
            objectIds.add(py::int_(objectId));

            const auto componentsIt = object.find("components");
            // Component references may target objects with no Python script.
            // Keep their native identities while extracting only Python data.
            if (componentsIt != object.end() && componentsIt->is_array()) {
                const auto transformIt = object.find("transform");
                if (transformIt != object.end() && transformIt->is_object()) {
                    const auto typeIt = transformIt->find("type");
                    if (typeIt != transformIt->end() && typeIt->is_string()) {
                        nativeTypes.add(py::make_tuple(objectId, typeIt->get<std::string>()));
                    }
                }

                size_t componentIndex = 0;
                for (const auto &component : *componentsIt) {
                    const auto typeIt = component.find("type_id");
                    if (typeIt != component.end() && typeIt->is_string()) {
                        const std::string &typeId = typeIt->get_ref<const std::string &>();
                        if (typeId.rfind(kNativeTypePrefix, 0) == 0) {
                            nativeTypes.add(py::make_tuple(objectId, typeId.substr(kNativeTypePrefix.size())));
                        } else if (typeId.rfind("python:", 0) == 0) {
                            descriptors.append(py::make_tuple(objectId,
                                                              "snapshot.objects[id=" + std::to_string(objectId) +
                                                                  "].components[" + std::to_string(componentIndex) +
                                                                  "]",
                                                              JsonToPython(component)));
                        }
                    }
                    ++componentIndex;
                }
            }

            const auto childrenIt = object.find("children");
            if (childrenIt != object.end() && childrenIt->is_array()) {
                for (const auto &child : *childrenIt)
                    self(self, child);
            }
        };

        const auto objectsIt = m_document.find("objects");
        if (objectsIt != m_document.end() && objectsIt->is_array()) {
            for (const auto &object : *objectsIt)
                visit(visit, object);
        }
        return py::make_tuple(std::move(objectIds), std::move(nativeTypes), std::move(descriptors));
    }

    void PreflightResourceDependencies() const
    {
        PreflightSceneResourceDependencies(m_document);
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> ResourceDependencies() const
    {
        return CollectSceneResourceDependencies(m_document);
    }

    [[nodiscard]] std::shared_ptr<SceneCommitToken> CommitRetainingWorld(Scene &scene) const
    {
        return scene.CommitDocumentRetainingCurrentWorld(m_document);
    }

  private:
    nlohmann::json m_document;
};

namespace scene_binding
{

void BindSceneDocumentTypes(py::module_ &m)
{
    py::class_<Scene::PendingPyComponent>(m, "PendingPyComponent")
        .def_readonly("game_object_id", &Scene::PendingPyComponent::gameObjectId)
        .def_readonly("type_name", &Scene::PendingPyComponent::typeName)
        .def_readonly("script_guid", &Scene::PendingPyComponent::scriptGuid)
        .def_readonly("type_guid", &Scene::PendingPyComponent::typeGuid)
        .def_property_readonly(
            "fields_document",
            [](const Scene::PendingPyComponent &pending) { return JsonToPython(pending.fieldsDocument); })
        .def_readonly("enabled", &Scene::PendingPyComponent::enabled)
        .def_readonly("execution_order", &Scene::PendingPyComponent::executionOrder)
        .def_readonly("component_index", &Scene::PendingPyComponent::componentIndex);

    py::class_<SceneDocumentReadTicket>(m, "_SceneDocumentReadTicket")
        .def_property_readonly("is_complete", &SceneDocumentReadTicket::IsComplete)
        .def_property_readonly("is_ready", &SceneDocumentReadTicket::IsReady)
        .def_property_readonly("ran_on_worker", &SceneDocumentReadTicket::RanOnWorker)
        .def_property_readonly("status", &SceneDocumentReadTicket::GetStatusName)
        .def_property_readonly("error", &SceneDocumentReadTicket::GetError)
        .def_property_readonly("file_state", &SceneDocumentReadTicket::GetFileState)
        .def("cancel", &SceneDocumentReadTicket::Cancel)
        .def(
            "_take_document", [](SceneDocumentReadTicket &ticket) { return JsonToPython(ticket.TakeDocument()); },
            "Consume and return the validated scene document");
    m.def("_schedule_scene_document_read", &ScheduleSceneDocumentRead, py::arg("path"),
          "Schedule scene file IO and structural validation on the native JobSystem");
    m.def(
        "_validate_resolved_scene_document",
        [](py::handle document) { ValidateResolvedSceneDocument(PythonToJson(document)); }, py::arg("document"),
        "Validate a resolved Scene before publishing its cooked artifact");
    m.def(
        "_encode_scene_authoring_document",
        [](py::handle document) {
            const auto snapshot = PythonToJson(document);
            const auto identities = DeserializeSceneAuthoringIdentity(snapshot.at("authoring_identity"));
            return JsonToPython(EncodeSceneAuthoringDocument(snapshot, identities));
        },
        py::arg("document"), "Encode a runtime snapshot as a GUID authoring asset");
    m.def(
        "_decode_scene_authoring_document",
        [](py::handle document) {
            SceneAuthoringIdentity identities;
            return JsonToPython(DecodeSceneAuthoringDocument(PythonToJson(document), identities));
        },
        py::arg("document"), "Decode a GUID authoring asset to a runtime snapshot");
    m.def(
        "_pump_inline_jobs",
        [](uint32_t maxJobs) {
            if (!JobSystem::IsAvailable() || !JobSystem::Get().IsInline())
                return 0u;
            return JobSystem::Get().RunPendingJobs(maxJobs);
        },
        py::arg("max_jobs") = 64, "Run queued native jobs only when the JobSystem uses its owner-thread executor");
    m.def(
        "_preflight_scene_resource_dependencies",
        [](py::handle document) { PreflightSceneResourceDependencies(PythonToJson(document)); }, py::arg("document"),
        "Validate native Scene resource GUIDs and embedded resource documents on the owner thread");
    m.def(
        "_collect_scene_resource_dependencies",
        [](py::handle document) { return CollectSceneResourceDependencies(PythonToJson(document)); },
        py::arg("document"), "Return the typed transitive resource dependencies of a Scene document");

    // Scene binding
    py::class_<SceneCommitToken, std::shared_ptr<SceneCommitToken>>(m, "_SceneCommitToken")
        .def_property_readonly("is_active", [](const SceneCommitToken &token) { return token.IsActive(); })
        .def_property_readonly("object_id_remap", &SceneCommitToken::GetObjectIdRemap)
        .def_property_readonly("component_id_remap", &SceneCommitToken::GetComponentIdRemap)
        .def("rollback", &SceneCommitToken::Rollback)
        .def("finalize", &SceneCommitToken::Finalize);

    py::class_<ScenePlayModeSnapshot, std::shared_ptr<ScenePlayModeSnapshot>>(m, "_ScenePlayModeSnapshot")
        .def("_python_component_records", &ScenePlayModeSnapshot::GetPythonComponentRecords)
        .def("_preflight_resource_dependencies", &ScenePlayModeSnapshot::PreflightResourceDependencies)
        .def("_resource_dependencies", &ScenePlayModeSnapshot::ResourceDependencies);
}

void BindSceneDocuments(NativeClass<Scene> &cls)
{
    cls.def("serialize", &Scene::Serialize, "Serialize scene to JSON string")
        .def("serialize_asset", &Scene::SerializeAuthoring, "Serialize a GUID Scene authoring file")
        .def(
            "serialize_asset_document",
            [](const Scene &scene) { return JsonToPython(scene.SerializeAuthoringDocument()); },
            "Serialize a GUID Scene authoring document without runtime allocation state")
        .def(
            "serialize_document", [](const Scene &scene) { return JsonToPython(scene.SerializeDocument()); },
            "Serialize scene to a Python document")
        .def(
            "_capture_play_mode_snapshot",
            [](const Scene &scene) { return std::make_shared<ScenePlayModeSnapshot>(scene.SerializeDocument()); },
            "Capture an opaque native scene document for Play Mode restoration")
        .def(
            "_commit_document",
            [](Scene &scene, const PythonDocument &document) { return scene.DeserializeDocument(document.value); },
            py::arg("document"), "Internal native staging commit; Python callers must preflight first")
        .def(
            "_commit_document_retaining_world",
            [](Scene &scene, const PythonDocument &document) {
                return scene.CommitDocumentRetainingCurrentWorld(document.value);
            },
            py::arg("document"), "Commit a candidate and retain the previous native world until finalized")
        .def(
            "_commit_play_mode_snapshot_retaining_world",
            [](Scene &scene, const std::shared_ptr<ScenePlayModeSnapshot> &snapshot) {
                if (!snapshot)
                    return std::shared_ptr<SceneCommitToken>{};
                return snapshot->CommitRetainingWorld(scene);
            },
            py::arg("snapshot"), "Commit an opaque Play Mode snapshot without crossing the Python JSON bridge")
        .def("save_to_file", &Scene::SaveToFile, py::arg("path"), "Save scene to a JSON file")
        .def("has_pending_py_components", &Scene::HasPendingPyComponents,
             "Check if there are pending Python components to restore")
        .def("get_pending_py_components", &Scene::GetPendingPyComponents,
             "Get a read-only snapshot of pending Python component descriptors")
        .def("take_pending_py_components", &Scene::TakePendingPyComponents,
             "Get and clear pending Python components for restoration");
}

} // namespace scene_binding

} // namespace infernux
