#include "BindingRegistration.h"
#include "JsonPyBridge.h"
#include <algorithm>
#include <core/reflection/SemanticTypeRegistry.h>
#include <function/scene/ComponentFactory.h>
#include <pybind11/stl.h>

namespace infernux
{
namespace
{
SemanticTypeDescriptor ParseType(const nlohmann::json &document)
{
    SemanticTypeDescriptor type;
    type.typeGuid = document.at("type_guid").get<std::string>();
    type.readableId = document.at("readable_id").get<std::string>();
    type.owner = document.at("owner").get<std::string>();
    type.origin = document.at("origin").get<std::string>();
    type.displayName = document.at("display_name").get<std::string>();
    type.baseTypeGuid = document.at("base_type_guid").get<std::string>();
    type.constructible = document.at("constructible").get<bool>();
    type.serializable = document.at("serializable").get<bool>();
    type.runtimeAvailable = document.at("runtime_available").get<bool>();
    type.runtimeProfiles = document.at("runtime_profiles").get<std::vector<std::string>>();
    type.lifecycle = document.at("lifecycle").get<std::vector<std::string>>();
    const auto &fields = document.at("fields");
    if (!fields.is_array())
        throw py::value_error("semantic fields must be an array");
    for (const auto &field : fields) {
        type.fields.push_back({field.at("property_path").get<std::string>(), field.at("value_type").get<std::string>(),
                               field.at("read_only").get<bool>(), field.at("attributes")});
    }
    return type;
}

nlohmann::json TypeDocument(const SemanticTypeDescriptor &type)
{
    auto fields = nlohmann::json::array();
    for (const auto &field : type.fields)
        fields.push_back({{"property_path", field.propertyPath},
                          {"value_type", field.valueType},
                          {"read_only", field.readOnly},
                          {"attributes", field.attributes}});
    return {{"type_guid", type.typeGuid},
            {"readable_id", type.readableId},
            {"owner", type.owner},
            {"origin", type.origin},
            {"revision", type.revision},
            {"display_name", type.displayName},
            {"base_type_guid", type.baseTypeGuid},
            {"constructible", type.constructible},
            {"serializable", type.serializable},
            {"runtime_available", type.runtimeAvailable},
            {"runtime_profiles", type.runtimeProfiles},
            {"lifecycle", type.lifecycle},
            {"fields", std::move(fields)}};
}

struct SnapshotView
{
    SemanticTypeRegistry::Snapshot snapshot;
};
} // namespace

void RegisterSemanticCatalogBindings(py::module_ &module)
{
    ComponentFactory::PublishSemanticTypes();
    py::class_<SnapshotView>(module, "_SemanticCatalogSnapshot")
        .def_property_readonly("revision", [](const SnapshotView &view) { return view.snapshot->revision; })
        .def_property_readonly("type_guids",
                               [](const SnapshotView &view) {
                                   std::vector<std::string> ids;
                                   ids.reserve(view.snapshot->types.size());
                                   for (const auto &[guid, type] : view.snapshot->types)
                                       ids.push_back(guid);
                                   std::sort(ids.begin(), ids.end());
                                   return ids;
                               })
        .def("type_document", [](const SnapshotView &view, const std::string &guid) {
            const auto found = view.snapshot->types.find(guid);
            if (found == view.snapshot->types.end())
                throw py::key_error(guid);
            return JsonToPython(TypeDocument(*found->second));
        });
    py::class_<SemanticTypeRegistry::PreparedPublication>(module, "_SemanticCatalogPublication")
        .def_property_readonly("candidate",
                               [](const SemanticTypeRegistry::PreparedPublication &publication) {
                                   return SnapshotView{publication.Candidate()};
                               })
        .def("publish", [](const SemanticTypeRegistry::PreparedPublication &publication) {
            return SemanticTypeRegistry::Instance().Publish(publication);
        });
    module.def("_semantic_catalog_snapshot", [] { return SnapshotView{SemanticTypeRegistry::Instance().Read()}; });
    module.def("_semantic_catalog_prepare", [](py::list edits) {
        std::vector<SemanticOwnerEdit> owners;
        for (const auto &document : PythonToJson(edits)) {
            SemanticOwnerEdit edit;
            edit.owner = document.at("owner").get<std::string>();
            const auto &types = document.at("types");
            if (!types.is_array())
                throw py::value_error("semantic types must be an array");
            for (const auto &type : types)
                edit.types.push_back(ParseType(type));
            owners.push_back(std::move(edit));
        }
        return SemanticTypeRegistry::Instance().Prepare(owners);
    });
}
} // namespace infernux
