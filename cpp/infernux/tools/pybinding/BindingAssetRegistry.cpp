#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infernux
{

void RegisterAssetRegistryBindings(py::module_ &m)
{
    // ── InxMesh — read-only runtime mesh asset ───────────────────────────
    py::class_<InxMesh, std::shared_ptr<InxMesh>>(m, "InxMesh")
        .def_property_readonly("name", &InxMesh::GetName, "Mesh asset name")
        .def_property_readonly("guid", &InxMesh::GetGuid, "Mesh asset GUID")
        .def_property_readonly("file_path", &InxMesh::GetFilePath, "Source file path")
        .def_property_readonly("vertex_count", &InxMesh::GetVertexCount, "Total vertex count")
        .def_property_readonly("index_count", &InxMesh::GetIndexCount, "Total index count")
        .def_property_readonly("submesh_count", &InxMesh::GetSubMeshCount, "Number of submeshes")
        .def_property_readonly("material_slot_count", &InxMesh::GetMaterialSlotCount, "Number of material slots")
        .def_property_readonly("material_slot_names", &InxMesh::GetMaterialSlotNames,
                               "Material slot names from model file")
        .def(
            "get_material_slot_data",
            [](const InxMesh &self) -> py::list {
                py::list result;
                for (const auto &sd : self.GetMaterialSlotData()) {
                    py::dict d;
                    d["base_color"] = py::make_tuple(sd.baseColor.r, sd.baseColor.g, sd.baseColor.b, sd.baseColor.a);
                    d["emission_color"] =
                        py::make_tuple(sd.emissionColor.r, sd.emissionColor.g, sd.emissionColor.b, sd.emissionColor.a);
                    d["metallic"] = sd.metallic;
                    d["smoothness"] = sd.smoothness;
                    d["opacity"] = sd.opacity;
                    result.append(d);
                }
                return result;
            },
            "Get per-slot material data extracted from model file")
        .def(
            "get_bounds",
            [](const InxMesh &self) -> py::tuple {
                const auto &bmin = self.GetBoundsMin();
                const auto &bmax = self.GetBoundsMax();
                return py::make_tuple(bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);
            },
            "Get AABB as (minX, minY, minZ, maxX, maxY, maxZ)")
        .def(
            "get_submesh_info",
            [](const InxMesh &self, uint32_t index) -> py::dict {
                const auto &sub = self.GetSubMesh(index);
                py::dict d;
                d["name"] = sub.name;
                d["index_start"] = sub.indexStart;
                d["index_count"] = sub.indexCount;
                d["vertex_start"] = sub.vertexStart;
                d["vertex_count"] = sub.vertexCount;
                d["material_slot"] = sub.materialSlot;
                d["bounds_min"] = py::make_tuple(sub.boundsMin.x, sub.boundsMin.y, sub.boundsMin.z);
                d["bounds_max"] = py::make_tuple(sub.boundsMax.x, sub.boundsMax.y, sub.boundsMax.z);
                return d;
            },
            py::arg("index"), "Get submesh info as dict (name, index_start, index_count, ...)")
        .def("__repr__", [](const InxMesh &self) {
            return "<InxMesh '" + self.GetName() + "' " + std::to_string(self.GetVertexCount()) + " verts, " +
                   std::to_string(self.GetSubMeshCount()) + " submesh(es)>";
        });

    // ── AssetRegistry — unified asset cache (singleton) ─────────────────
    py::class_<AssetRegistry, std::unique_ptr<AssetRegistry, py::nodelete>>(m, "AssetRegistry")
        .def_static("instance", &AssetRegistry::Instance, py::return_value_policy::reference,
                    "Get the AssetRegistry singleton")
        .def("is_initialized", &AssetRegistry::IsInitialized, "Check if the registry is initialized")
        .def("get_asset_database", &AssetRegistry::GetAssetDatabase, py::return_value_policy::reference,
             "Get the owned AssetDatabase (may be None before InitRenderer)")

        // Material convenience wrappers (type-safe, avoids exposing void* to Python)
        .def(
            "load_material",
            [](AssetRegistry &self, const std::string &path) {
                return self.LoadAssetByPath<InxMaterial>(path, ResourceType::Material);
            },
            py::arg("path"), "Load a material by file path (GUID resolved internally)")
        .def(
            "load_material_by_guid",
            [](AssetRegistry &self, const std::string &guid) {
                return self.LoadAsset<InxMaterial>(guid, ResourceType::Material);
            },
            py::arg("guid"), "Load a material by its GUID")
        .def(
            "get_material",
            [](AssetRegistry &self, const std::string &guid) { return self.GetAsset<InxMaterial>(guid); },
            py::arg("guid"), "Get a cached material by GUID (returns None if not loaded)")
        .def("get_builtin_material", &AssetRegistry::GetBuiltinMaterial, py::arg("key"),
             "Get a built-in material by key (e.g. 'DefaultLit', 'ErrorMaterial')")
        .def("load_builtin_material_from_file", &AssetRegistry::LoadBuiltinMaterialFromFile, py::arg("key"),
             py::arg("mat_file_path"), "Load/replace a builtin material from a .mat file (e.g. key='DefaultLit')")

        // Mesh convenience wrappers
        .def(
            "load_mesh",
            [](AssetRegistry &self, const std::string &path) {
                return self.LoadAssetByPath<InxMesh>(path, ResourceType::Mesh);
            },
            py::arg("path"), "Load a mesh by file path (.fbx, .obj, .gltf, …)")
        .def(
            "load_mesh_by_guid",
            [](AssetRegistry &self, const std::string &guid) {
                return self.LoadAsset<InxMesh>(guid, ResourceType::Mesh);
            },
            py::arg("guid"), "Load a mesh by its GUID")
        .def(
            "get_mesh", [](AssetRegistry &self, const std::string &guid) { return self.GetAsset<InxMesh>(guid); },
            py::arg("guid"), "Get a cached mesh by GUID (returns None if not loaded)")

        // Hot-reload / invalidation
        .def("reload_asset", &AssetRegistry::ReloadAsset, py::arg("guid"),
             "Reload an asset in-place from disk (preserves shared_ptr identity)")
        .def("invalidate_asset", &AssetRegistry::InvalidateAsset, py::arg("guid"),
             "Evict an asset from cache so next load re-reads from disk")
        .def("remove_asset", &AssetRegistry::RemoveAsset, py::arg("guid"),
             "Fully remove an asset record (e.g. when file is deleted)")

        // File-event hooks
        .def("on_asset_modified", &AssetRegistry::OnAssetModified, py::arg("path"),
             "Notify that a file was modified — reloads if cached")
        .def("on_asset_moved", &AssetRegistry::OnAssetMoved, py::arg("old_path"), py::arg("new_path"),
             "Notify that a file was moved/renamed")
        .def("on_asset_deleted", &AssetRegistry::OnAssetDeleted, py::arg("path"),
             "Notify that a file was deleted — evicts from cache")

        // Queries
        .def("is_loaded", &AssetRegistry::IsLoaded, py::arg("guid"), "Check if an asset is currently cached");
}

} // namespace infernux
