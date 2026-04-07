/**
 * @file BindingTagLayer.cpp
 * @brief Python bindings for TagLayerManager.
 *
 * Exposes the project-wide Tag & Layer management singleton to Python.
 */

#include "function/scene/TagLayerManager.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infernux
{

void RegisterTagLayerBindings(py::module_ &m)
{
    py::class_<TagLayerManager, std::unique_ptr<TagLayerManager, py::nodelete>>(m, "TagLayerManager",
                                                                                "Manages project-wide tags and layers")
        .def_static("instance", &TagLayerManager::Instance, py::return_value_policy::reference,
                    "Get the singleton TagLayerManager instance")

        // --- Tags ---
        .def("get_tag", &TagLayerManager::GetTag, py::arg("index"), "Get tag string by index")
        .def("get_tag_index", &TagLayerManager::GetTagIndex, py::arg("tag"), "Get tag index by name (-1 if not found)")
        .def("add_tag", &TagLayerManager::AddTag, py::arg("tag"), "Add a custom tag. Returns its index.")
        .def("remove_tag", &TagLayerManager::RemoveTag, py::arg("tag"),
             "Remove a custom tag. Built-in tags cannot be removed.")
        .def("get_all_tags", &TagLayerManager::GetAllTags, py::return_value_policy::copy,
             "Get all tags (built-in + custom)")
        .def("is_builtin_tag", &TagLayerManager::IsBuiltinTag, py::arg("tag"), "Check if a tag is built-in")

        // --- Layers ---
        .def("get_layer_name", &TagLayerManager::GetLayerName, py::arg("layer"), "Get layer name by index (0-31)")
        .def("get_layer_by_name", &TagLayerManager::GetLayerByName, py::arg("name"),
             "Get layer index by name (-1 if not found)")
        .def("set_layer_name", &TagLayerManager::SetLayerName, py::arg("layer"), py::arg("name"),
             "Set a layer name (built-in layers cannot be renamed)")
        .def("get_all_layers", &TagLayerManager::GetAllLayers, py::return_value_policy::copy, "Get all 32 layer names")
        .def("is_builtin_layer", &TagLayerManager::IsBuiltinLayer, py::arg("layer"), "Check if a layer is built-in")
        .def("get_layer_collision_mask", &TagLayerManager::GetLayerCollisionMask, py::arg("layer"),
             "Get the 32-bit physics collision mask for a layer")
        .def("set_layer_collision_mask", &TagLayerManager::SetLayerCollisionMask, py::arg("layer"), py::arg("mask"),
             "Set the full 32-bit physics collision mask for a layer")
        .def("get_layers_collide", &TagLayerManager::GetLayersCollide, py::arg("layer_a"), py::arg("layer_b"),
             "Check whether two layers collide in physics")
        .def("set_layers_collide", &TagLayerManager::SetLayersCollide, py::arg("layer_a"), py::arg("layer_b"),
             py::arg("should_collide"), "Enable/disable collision between two layers")

        // --- Layer mask ---
        .def_static("layer_to_mask", &TagLayerManager::LayerToMask, py::arg("layer"),
                    "Create a layer mask from a single layer index")
        .def("get_mask", &TagLayerManager::GetMask, py::arg("layer_names"), "Create a mask from multiple layer names")

        // --- Serialization ---
        .def("serialize", &TagLayerManager::Serialize, "Serialize to JSON string")
        .def("deserialize", &TagLayerManager::Deserialize, py::arg("json_str"), "Deserialize from JSON string")
        .def("save_to_file", &TagLayerManager::SaveToFile, py::arg("path"), "Save tag/layer settings to file")
        .def("load_from_file", &TagLayerManager::LoadFromFile, py::arg("path"), "Load tag/layer settings from file");
}

} // namespace infernux
