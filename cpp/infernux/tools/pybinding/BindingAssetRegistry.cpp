#include "JsonPyBridge.h"
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshArtifact.h>
#include <function/resources/InxMesh/MeshImportSettings.h>
#include <function/resources/InxMesh/ModelMeshReference.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <function/scene/MeshRenderer.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace py = pybind11;

namespace infernux
{
namespace
{
py::array_t<float> MeshVec3Array(const std::vector<Vertex> &vertices, const glm::vec3 Vertex::*member)
{
    py::array_t<float> result({static_cast<py::ssize_t>(vertices.size()), py::ssize_t{3}});
    auto output = result.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < static_cast<py::ssize_t>(vertices.size()); ++row) {
        const auto &value = vertices[static_cast<size_t>(row)].*member;
        output(row, 0) = value.x;
        output(row, 1) = value.y;
        output(row, 2) = value.z;
    }
    return result;
}

py::dict MeshVertexData(const InxMesh &mesh)
{
    mesh.RequireCpuReadable("Mesh.vertex_buffer");
    const auto &vertices = mesh.GetVertices();
    py::dict result;
    result["positions"] = MeshVec3Array(vertices, &Vertex::pos);
    result["normals"] = MeshVec3Array(vertices, &Vertex::normal);
    result["colors"] = MeshVec3Array(vertices, &Vertex::color);

    py::array_t<float> tangents({static_cast<py::ssize_t>(vertices.size()), py::ssize_t{4}});
    py::array_t<float> uvs({static_cast<py::ssize_t>(vertices.size()), py::ssize_t{2}});
    py::array_t<float> uvs1({static_cast<py::ssize_t>(vertices.size()), py::ssize_t{2}});
    auto tangentOutput = tangents.mutable_unchecked<2>();
    auto uvOutput = uvs.mutable_unchecked<2>();
    auto uv1Output = uvs1.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < static_cast<py::ssize_t>(vertices.size()); ++row) {
        const auto &vertex = vertices[static_cast<size_t>(row)];
        tangentOutput(row, 0) = vertex.tangent.x;
        tangentOutput(row, 1) = vertex.tangent.y;
        tangentOutput(row, 2) = vertex.tangent.z;
        tangentOutput(row, 3) = vertex.tangent.w;
        uvOutput(row, 0) = vertex.texCoord.x;
        uvOutput(row, 1) = vertex.texCoord.y;
        uv1Output(row, 0) = vertex.texCoord1.x;
        uv1Output(row, 1) = vertex.texCoord1.y;
    }
    result["tangents"] = std::move(tangents);
    result["uvs"] = std::move(uvs);
    result["uvs1"] = std::move(uvs1);
    return result;
}

void RequireFinite(float value, const char *name)
{
    if (!std::isfinite(value))
        throw py::value_error(std::string(name) + " must contain finite values");
}

template <size_t Columns, typename Assign>
void ApplyOptionalRows(const py::object &value, size_t rows, const char *name, Assign assign)
{
    if (value.is_none())
        return;
    const auto array = value.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    if (array.ndim() != 2 || array.shape(0) != static_cast<py::ssize_t>(rows) ||
        array.shape(1) != static_cast<py::ssize_t>(Columns))
        throw py::value_error(std::string(name) + " must have shape (N, " + std::to_string(Columns) + ")");
    const auto input = array.unchecked<2>();
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < Columns; ++column)
            RequireFinite(input(static_cast<py::ssize_t>(row), static_cast<py::ssize_t>(column)), name);
        assign(row, input, static_cast<py::ssize_t>(row));
    }
}

std::vector<Vertex> DecodeMeshVertices(const py::array_t<float, py::array::c_style | py::array::forcecast> &positions,
                                       const py::object &normals, const py::object &uvs, const py::object &tangents,
                                       const py::object &colors, bool &generatedNormals, bool &generatedTangents)
{
    if (positions.ndim() != 2 || positions.shape(1) != 3)
        throw py::value_error("positions must have shape (N, 3)");
    const size_t count = static_cast<size_t>(positions.shape(0));
    std::vector<Vertex> vertices(count);
    const auto positionData = positions.unchecked<2>();
    for (size_t row = 0; row < count; ++row) {
        const auto index = static_cast<py::ssize_t>(row);
        for (py::ssize_t column = 0; column < 3; ++column)
            RequireFinite(positionData(index, column), "positions");
        vertices[row].pos = {positionData(index, 0), positionData(index, 1), positionData(index, 2)};
    }
    ApplyOptionalRows<3>(normals, count, "normals", [&](size_t row, const auto &data, py::ssize_t index) {
        vertices[row].normal = {data(index, 0), data(index, 1), data(index, 2)};
    });
    ApplyOptionalRows<2>(uvs, count, "uvs", [&](size_t row, const auto &data, py::ssize_t index) {
        vertices[row].texCoord = {data(index, 0), data(index, 1)};
    });
    ApplyOptionalRows<4>(tangents, count, "tangents", [&](size_t row, const auto &data, py::ssize_t index) {
        vertices[row].tangent = {data(index, 0), data(index, 1), data(index, 2), data(index, 3)};
    });
    ApplyOptionalRows<3>(colors, count, "colors", [&](size_t row, const auto &data, py::ssize_t index) {
        vertices[row].color = {data(index, 0), data(index, 1), data(index, 2)};
    });
    generatedNormals = normals.is_none();
    generatedTangents = tangents.is_none();
    return vertices;
}

std::vector<uint32_t> DecodeMeshIndices(const py::array_t<uint32_t, py::array::c_style | py::array::forcecast> &indices,
                                        size_t vertexCount)
{
    if (indices.ndim() != 1)
        throw py::value_error("indices must have shape (M,)");
    if (indices.shape(0) % 3 != 0)
        throw py::value_error("indices must contain complete triangles");
    std::vector<uint32_t> result(static_cast<size_t>(indices.shape(0)));
    const auto input = indices.unchecked<1>();
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = input(static_cast<py::ssize_t>(index));
        if (result[index] >= vertexCount)
            throw py::value_error("indices contain a vertex index outside positions");
    }
    return result;
}

std::vector<SubMesh> DecodeSubMeshes(const py::object &descriptions, const std::vector<uint32_t> &indices,
                                     const std::vector<Vertex> &vertices)
{
    const size_t vertexCount = vertices.size();
    if (descriptions.is_none()) {
        if (indices.empty())
            return {};
        SubMesh whole;
        whole.indexCount = static_cast<uint32_t>(indices.size());
        whole.vertexCount = static_cast<uint32_t>(vertexCount);
        whole.name = "Mesh";
        whole.boundsMin = glm::vec3(std::numeric_limits<float>::max());
        whole.boundsMax = glm::vec3(-std::numeric_limits<float>::max());
        for (uint32_t index : indices) {
            whole.boundsMin = glm::min(whole.boundsMin, vertices[index].pos);
            whole.boundsMax = glm::max(whole.boundsMax, vertices[index].pos);
        }
        return {std::move(whole)};
    }
    std::vector<SubMesh> result;
    for (const py::handle item : descriptions.cast<py::iterable>()) {
        if (!py::isinstance<py::dict>(item))
            throw py::type_error("submeshes must contain dictionaries");
        const auto description = py::reinterpret_borrow<py::dict>(item);
        SubMesh subMesh;
        subMesh.indexStart = description["index_start"].cast<uint32_t>();
        subMesh.indexCount = description["index_count"].cast<uint32_t>();
        subMesh.vertexStart = description.contains("vertex_start") ? description["vertex_start"].cast<uint32_t>() : 0;
        subMesh.vertexCount = description.contains("vertex_count") ? description["vertex_count"].cast<uint32_t>()
                                                                   : static_cast<uint32_t>(vertexCount);
        subMesh.materialSlot =
            description.contains("material_slot") ? description["material_slot"].cast<uint32_t>() : 0;
        subMesh.nodeGroup = description.contains("node_group") ? description["node_group"].cast<uint32_t>() : 0;
        subMesh.name = description.contains("name") ? description["name"].cast<std::string>() : "SubMesh";
        if (subMesh.indexStart > indices.size() || subMesh.indexCount > indices.size() - subMesh.indexStart ||
            subMesh.indexCount % 3 != 0 || subMesh.vertexStart > vertexCount ||
            subMesh.vertexCount > vertexCount - subMesh.vertexStart)
            throw py::value_error("submeshes contain an invalid vertex or triangle range");
        if (subMesh.indexCount > 0) {
            subMesh.boundsMin = glm::vec3(std::numeric_limits<float>::max());
            subMesh.boundsMax = glm::vec3(-std::numeric_limits<float>::max());
            for (size_t offset = 0; offset < subMesh.indexCount; ++offset) {
                const auto &position = vertices[indices[subMesh.indexStart + offset]].pos;
                subMesh.boundsMin = glm::min(subMesh.boundsMin, position);
                subMesh.boundsMax = glm::max(subMesh.boundsMax, position);
            }
        }
        result.push_back(std::move(subMesh));
    }
    return result;
}
} // namespace

void RegisterAssetRegistryBindings(py::module_ &m)
{
    m.def("make_model_mesh_reference", &MakeModelMeshReference);
    m.def("split_model_mesh_reference", &SplitModelMeshReference);
    m.def("_mesh_import_settings_schema", [] { return JsonToPython(MeshImportSettings::Schema()); });
#if defined(INFERNUX_PYBIND_WEB_PLAYER)
    // The browser Player owns this database through AssetRegistry. Expose only
    // immutable runtime queries; authoring, refresh, and mutation APIs stay out
    // of the Web profile.
    py::class_<AssetDatabase>(m, "AssetDatabase")
        .def("contains_guid", &AssetDatabase::ContainsGuid, py::arg("guid"))
        .def("contains_path", &AssetDatabase::ContainsPath, py::arg("path"))
        .def("get_guid_from_path", &AssetDatabase::GetGuidFromPath, py::arg("path"))
        .def("get_path_from_guid", &AssetDatabase::GetPathFromGuid, py::arg("guid"))
        .def("get_all_guids", &AssetDatabase::GetAllGuids, py::call_guard<py::gil_scoped_release>(),
             "Get all cooked Player GUIDs in one published generation")
        .def("get_resource_type", &AssetDatabase::GetResourceTypeForPath, py::arg("file_path"))
        .def("get_runtime_artifact_path", &AssetDatabase::GetRuntimeArtifactPath, py::arg("guid"), py::arg("type"))
        .def_property_readonly("asset_count", &AssetDatabase::GetAssetCount)
        .def_property_readonly("project_root", &AssetDatabase::GetProjectRoot);
#endif

    py::class_<AssetLoadTicket, std::shared_ptr<AssetLoadTicket>>(m, "AssetLoadTicket")
        .def_property_readonly("guid", &AssetLoadTicket::GetGuid)
        .def_property_readonly("resource_type", &AssetLoadTicket::GetResourceType)
        .def_property_readonly("complete", &AssetLoadTicket::IsComplete)
        .def_property_readonly("committed", &AssetLoadTicket::IsCommitted)
        .def_property_readonly("produced_on_worker", &AssetLoadTicket::WasProducedOnWorker);

    py::class_<AssetResidencyRecord>(m, "AssetResidencyRecord")
        .def_readonly("guid", &AssetResidencyRecord::guid)
        .def_readonly("resource_type", &AssetResidencyRecord::type)
        .def_readonly("runtime_type_name", &AssetResidencyRecord::runtimeTypeName)
        .def_readonly("runtime_version", &AssetResidencyRecord::runtimeVersion)
        .def_readonly("cpu_bytes", &AssetResidencyRecord::cpuBytes)
        .def_readonly("last_access_serial", &AssetResidencyRecord::lastAccessSerial)
        .def_readonly("explicit_pin_count", &AssetResidencyRecord::explicitPinCount)
        .def_readonly("external_reference_count", &AssetResidencyRecord::externalReferenceCount)
        .def_readonly("evictable", &AssetResidencyRecord::evictable);

    // ── InxMesh — read-only runtime mesh asset ───────────────────────────
    py::class_<InxMesh, std::shared_ptr<InxMesh>>(m, "InxMesh")
        .def_property_readonly("name", &InxMesh::GetName, "Mesh asset name")
        .def("create_model_node_copy", &InxMesh::CreateModelNodeCopy)
        .def("require_model_node", &InxMesh::RequireModelNode)
        .def("get_model_node_path", &InxMesh::GetModelNodePath)
        .def_property_readonly("guid", &InxMesh::GetGuid, "Mesh asset GUID")
        .def_property_readonly("file_path", &InxMesh::GetFilePath, "Source file path")
        .def_property_readonly("vertex_count", &InxMesh::GetVertexCount, "Total vertex count")
        .def_property_readonly("index_count", &InxMesh::GetIndexCount, "Total index count")
        .def_property_readonly("submesh_count", &InxMesh::GetSubMeshCount, "Number of submeshes")
        .def_property_readonly("material_slot_count", &InxMesh::GetMaterialSlotCount, "Number of material slots")
        .def_property_readonly("material_slot_names", &InxMesh::GetMaterialSlotNames,
                               "Material slot names from model file")
        .def(
            "get_model_nodes",
            [](const InxMesh &mesh) {
                py::list result;
                for (const auto &node : mesh.GetModelNodes()) {
                    py::dict item;
                    item["name"] = node.name;
                    item["parent_index"] = node.parentIndex;
                    item["node_group"] = node.nodeGroup;
                    item["visible"] = node.visible;
                    py::list rows;
                    for (glm::length_t row = 0; row < 4; ++row) {
                        py::list values;
                        for (glm::length_t column = 0; column < 4; ++column)
                            values.append(node.localTransform[column][row]);
                        rows.append(values);
                    }
                    item["local_matrix"] = rows;
                    result.append(item);
                }
                return result;
            },
            "Copy source nodes in parent-before-child order; matrices are rows, indices are import-local, not stable "
            "IDs")
        .def_property_readonly("has_skinned_data", &InxMesh::HasSkinnedData,
                               "Whether this Mesh carries immutable skin/animation data")
        .def_property_readonly("generation", &InxMesh::GetGeneration, "Monotonic immutable geometry generation")
        .def_property_readonly("skinned_bone_count",
                               [](const InxMesh &mesh) {
                                   const auto &skinned = mesh.GetSkinnedData();
                                   return skinned ? skinned->skeleton.bones.size() : size_t{0};
                               })
        .def_property_readonly(
            "skeleton_definition",
            [](const InxMesh &mesh) -> py::dict {
                py::dict result;
                const auto &skinned = mesh.GetSkinnedData();
                if (!skinned)
                    return result;
                result["guid"] = skinned->skeletonDefinitionGuid;
                result["subresource_id"] = skinned->skeletonDefinitionId;
                result["root_node_index"] = skinned->skeletonRootNodeIndex;
                return result;
            },
            "GUID-backed skeleton definition identity")
        .def_property_readonly(
            "exposed_skeleton_nodes",
            [](const InxMesh &mesh) {
                std::vector<std::string> result;
                const auto &skinned = mesh.GetSkinnedData();
                if (!skinned)
                    return result;
                for (const int index : skinned->exposedSkeletonNodeIndices)
                    if (index >= 0 && static_cast<size_t>(index) < skinned->skeleton.nodes.size())
                        result.push_back(skinned->skeleton.nodes[static_cast<size_t>(index)].name);
                return result;
            },
            "Published script-visible skeleton attachment nodes")
        .def_property_readonly(
            "humanoid_rig",
            [](const InxMesh &mesh) -> py::dict {
                py::dict result;
                const auto &skinned = mesh.GetSkinnedData();
                if (!skinned || !skinned->humanoid.enabled)
                    return result;
                py::dict mapping;
                for (const auto &[slot, nodeIndex] : skinned->humanoid.bones)
                    mapping[py::str(slot)] = skinned->skeleton.nodes.at(static_cast<size_t>(nodeIndex)).name;
                py::list issues;
                for (const auto &issue : skinned->humanoid.issues) {
                    py::dict item;
                    item["code"] = issue.code;
                    item["bone"] = issue.bone;
                    item["detail"] = issue.detail;
                    issues.append(std::move(item));
                }
                result["valid"] = skinned->humanoid.IsValid();
                result["required_bones_valid"] = skinned->humanoid.requiredBonesValid;
                result["hierarchy_valid"] = skinned->humanoid.hierarchyValid;
                result["reference_pose_valid"] = skinned->humanoid.referencePoseValid;
                result["mapping"] = std::move(mapping);
                result["issues"] = std::move(issues);
                return result;
            },
            "Resolved humanoid mapping and import-time validity report")
        .def_property_readonly("skinned_animation_count",
                               [](const InxMesh &mesh) {
                                   const auto &skinned = mesh.GetSkinnedData();
                                   return skinned ? skinned->animations.size() : size_t{0};
                               })
        .def_property_readonly("skinned_animation_names",
                               [](const InxMesh &mesh) {
                                   std::vector<std::string> names;
                                   const auto &skinned = mesh.GetSkinnedData();
                                   if (!skinned)
                                       return names;
                                   names.reserve(skinned->animations.size());
                                   for (const auto &animation : skinned->animations)
                                       names.push_back(animation.name);
                                   return names;
                               })
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
                    d["alpha_mode"] = sd.alphaMode == ModelAlphaMode::Mask    ? "mask"
                                      : sd.alphaMode == ModelAlphaMode::Blend ? "blend"
                                                                              : "opaque";
                    d["alpha_cutoff"] = sd.alphaCutoff;
                    d["double_sided"] = sd.doubleSided;
                    d["source_id"] = sd.sourceId;
                    d["material_guid"] = sd.materialGuid;
                    constexpr const char *keys[] = {"base_color_texture_guid", "normal_texture_guid",
                                                    "metallic_texture_guid",   "roughness_texture_guid",
                                                    "occlusion_texture_guid",  "emission_texture_guid"};
                    constexpr const char *uvKeys[] = {"base_color_uv_set", "normal_uv_set",    "metallic_uv_set",
                                                      "roughness_uv_set",  "occlusion_uv_set", "emission_uv_set"};
                    constexpr const char *samplerKeys[] = {"base_color_sampler", "normal_sampler",
                                                           "metallic_sampler",   "roughness_sampler",
                                                           "occlusion_sampler",  "emission_sampler"};
                    for (size_t index = 0; index < ModelTextureCount; ++index) {
                        d[keys[index]] = sd.textureGuids[index];
                        d[uvKeys[index]] = sd.textureUvSets[index];
                        const auto &sampler = sd.textureSamplers[index];
                        py::dict samplerData;
                        samplerData["min_filter"] = static_cast<uint8_t>(sampler.minFilter);
                        samplerData["mag_filter"] = static_cast<uint8_t>(sampler.magFilter);
                        samplerData["mip_filter"] = static_cast<uint8_t>(sampler.mipFilter);
                        samplerData["address_u"] = static_cast<uint8_t>(sampler.addressU);
                        samplerData["address_v"] = static_cast<uint8_t>(sampler.addressV);
                        samplerData["address_w"] = static_cast<uint8_t>(sampler.addressW);
                        d[samplerKeys[index]] = std::move(samplerData);
                    }
                    d["normal_scale"] = sd.normalScale;
                    d["occlusion_strength"] = sd.occlusionStrength;
                    d["packed_metallic_roughness"] = sd.packedMetallicRoughness;
                    result.append(d);
                }
                return result;
            },
            "Get per-slot material data extracted from model file")
        .def("create_material_copy", &InxMesh::CreateMaterialCopy, py::arg("slot"),
             "Create an independent material instance from one imported model slot; does not save a file")
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
                d["node_group"] = sub.nodeGroup;
                d["bounds_min"] = py::make_tuple(sub.boundsMin.x, sub.boundsMin.y, sub.boundsMin.z);
                d["bounds_max"] = py::make_tuple(sub.boundsMax.x, sub.boundsMax.y, sub.boundsMax.z);
                return d;
            },
            py::arg("index"), "Get submesh info as dict (name, index_start, index_count, ...)")
        .def(
            "_particle_sampling_data",
            [](const InxMesh &self) -> py::dict {
                const auto &vertices = self.GetVertices();
                const auto &indices = self.GetIndices();
                py::array_t<float> positions({static_cast<py::ssize_t>(vertices.size()), py::ssize_t{3}});
                auto positionView = positions.mutable_unchecked<2>();
                for (py::ssize_t index = 0; index < static_cast<py::ssize_t>(vertices.size()); ++index) {
                    const auto &position = vertices[static_cast<size_t>(index)].pos;
                    positionView(index, 0) = position.x;
                    positionView(index, 1) = position.y;
                    positionView(index, 2) = position.z;
                }
                py::array_t<uint32_t> encodedIndices(indices.size());
                auto indexView = encodedIndices.mutable_unchecked<1>();
                for (py::ssize_t index = 0; index < static_cast<py::ssize_t>(indices.size()); ++index)
                    indexView(index) = indices[static_cast<size_t>(index)];
                py::dict result;
                result["positions"] = std::move(positions);
                result["indices"] = std::move(encodedIndices);
                return result;
            },
            "Internal immutable geometry snapshot for particle CPU sampling")
        .def("get_vertex_data", &MeshVertexData, "Copy CPU-readable vertex streams into NumPy arrays")
        .def(
            "get_index_data",
            [](const InxMesh &self) {
                const auto &indices = self.GetIndices();
                py::array_t<uint32_t> result(indices.size());
                if (!indices.empty())
                    std::memcpy(result.mutable_data(), indices.data(), indices.size() * sizeof(uint32_t));
                return result;
            },
            "Copy the triangle index stream into a uint32 NumPy array")
        .def(
            "serialize_source", [](const InxMesh &self) { return py::bytes(MeshArtifact::SerializeSource(self)); },
            "Encode a static .inxmesh source. Does not write or import an asset.")
        .def("__repr__", [](const InxMesh &self) {
            return "<InxMesh '" + self.GetName() + "' " + std::to_string(self.GetVertexCount()) + " verts, " +
                   std::to_string(self.GetSubMeshCount()) + " submesh(es)>";
        });

    py::class_<InxTexture, std::shared_ptr<InxTexture>>(m, "InxTexture")
        .def_property_readonly("name", &InxTexture::GetName)
        .def_property_readonly("guid", &InxTexture::GetGuid)
        .def_property_readonly("file_path", &InxTexture::GetFilePath)
        .def_property_readonly("generation", &InxTexture::GetGeneration)
        .def_property_readonly("mip_count", [](const InxTexture &self) { return self.GetMipCount(); })
        .def_property_readonly("pixel_width", [](const InxTexture &self) { return self.GetPixelWidth(); })
        .def_property_readonly("pixel_height", [](const InxTexture &self) { return self.GetPixelHeight(); })
        .def_property_readonly("pixel_depth", [](const InxTexture &self) { return self.GetPixelDepth(); })
        .def_property_readonly(
            "dimension",
            [](const InxTexture &self) { return self.GetDimension() == TextureDimension::Texture3D ? "3d" : "2d"; })
        .def_property_readonly("semantic",
                               [](const InxTexture &self) {
                                   switch (self.GetSemantic()) {
                                   case TextureSemantic::Color:
                                       return "color";
                                   case TextureSemantic::Normal:
                                       return "normal";
                                   case TextureSemantic::Data:
                                       return "data";
                                   case TextureSemantic::UserInterface:
                                       return "user_interface";
                                   case TextureSemantic::Sprite:
                                       return "sprite";
                                   case TextureSemantic::VectorField:
                                       return "vector_field";
                                   case TextureSemantic::SignedDistanceField:
                                       return "signed_distance_field";
                                   }
                                   return "unknown";
                               })
        .def_property_readonly("srgb", &InxTexture::IsSrgb)
        .def_property_readonly("pixel_format",
                               [](const InxTexture &self) { return std::string(TextureFormatName(self.GetFormat())); })
        .def_property_readonly("pixel_storage",
                               [](const InxTexture &self) {
                                   const TextureFormat format = self.GetFormat();
                                   if (format == TextureFormat::Rgba32Float)
                                       return std::string("rgba32_float");
                                   if (format == TextureFormat::Rgba4UNormPack16)
                                       return std::string("rgba4_unorm_pack16");
                                   if (format == TextureFormat::Rgba16UNorm)
                                       return std::string("rgba16_unorm");
                                   if (format == TextureFormat::Rgba16Float)
                                       return std::string("rgba16_float");
                                   return TextureFormatIsBlockCompressed(format) ? std::string("block_compressed")
                                                                                 : std::string("rgba8");
                               })
        .def_property_readonly("bake_basis", [](const InxTexture &self) { return self.GetBakeBasis(); })
        .def_property_readonly("value_min", [](const InxTexture &self) { return self.GetValueMin(); })
        .def_property_readonly("value_max", [](const InxTexture &self) { return self.GetValueMax(); });

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

        .def(
            "load_physic_material",
            [](AssetRegistry &self, const std::string &path) {
                return self.LoadAssetByPath<PhysicMaterial>(path, ResourceType::PhysicMaterial);
            },
            py::arg("path"), "Load a PhysicMaterial by file path")
        .def(
            "load_physic_material_by_guid",
            [](AssetRegistry &self, const std::string &guid) {
                return self.LoadAsset<PhysicMaterial>(guid, ResourceType::PhysicMaterial);
            },
            py::arg("guid"), "Load a PhysicMaterial by GUID")
        .def(
            "get_physic_material",
            [](AssetRegistry &self, const std::string &guid) { return self.GetAsset<PhysicMaterial>(guid); },
            py::arg("guid"), "Get a cached PhysicMaterial by GUID")

        // Mesh convenience wrappers
        .def("create_runtime_mesh", &AssetRegistry::CreateRuntimeMesh, py::arg("name") = "Mesh",
             "Create a transient Mesh in the versioned AssetRegistry")
        .def("copy_mesh", &AssetRegistry::CloneRuntimeMesh, py::arg("guid"), py::arg("name") = "",
             "Create an independently versioned transient copy of a loaded Mesh")
        .def("destroy_runtime_mesh", &AssetRegistry::DestroyRuntimeMesh, py::arg("guid"),
             "Destroy a transient Mesh and invalidate its live renderer references")
        .def(
            "set_mesh_data",
            [](AssetRegistry &self, const std::string &guid,
               const py::array_t<float, py::array::c_style | py::array::forcecast> &positions,
               const py::array_t<uint32_t, py::array::c_style | py::array::forcecast> &indices,
               const py::object &normals, const py::object &uvs, const py::object &tangents, const py::object &colors,
               const py::object &submeshes, const std::vector<std::string> &materialSlots) {
                auto current = self.GetAsset<InxMesh>(guid);
                if (!current)
                    throw py::value_error("set_mesh_data requires a loaded Mesh GUID");
                bool generatedNormals = false;
                bool generatedTangents = false;
                auto vertices =
                    DecodeMeshVertices(positions, normals, uvs, tangents, colors, generatedNormals, generatedTangents);
                auto encodedIndices = DecodeMeshIndices(indices, vertices.size());
                auto decodedSubMeshes = DecodeSubMeshes(submeshes, encodedIndices, vertices);
                uint32_t requiredSlots = 0;
                for (const auto &subMesh : decodedSubMeshes) {
                    if (subMesh.materialSlot == std::numeric_limits<uint32_t>::max())
                        throw py::value_error("submesh material slot is out of range");
                    requiredSlots = std::max(requiredSlots, subMesh.materialSlot + 1);
                }
                if (requiredSlots > decodedSubMeshes.size())
                    throw py::value_error("submesh material slots must be contiguous from zero");
                std::vector<std::string> slotNames = materialSlots;
                while (slotNames.size() < requiredSlots)
                    slotNames.push_back("Material_" + std::to_string(slotNames.size()));
                if (generatedNormals)
                    RecalculateMeshNormals(vertices, encodedIndices);
                if (generatedTangents)
                    RecalculateMeshTangents(vertices, encodedIndices);
                InxMesh replacement = *current;
                replacement.SetData(std::move(vertices), std::move(encodedIndices), std::move(decodedSubMeshes));
                replacement.SetMaterialSlotNames(std::move(slotNames));
                replacement.SetMaterialSlotData(std::vector<MaterialSlotData>(replacement.GetMaterialSlotCount()));
                replacement.SetNodeNames({});
                replacement.SetModelNodes({});
                replacement.SetSkinnedData(nullptr);
                self.PublishMesh(guid, std::move(replacement));
            },
            py::arg("guid"), py::arg("positions"), py::arg("indices"), py::arg("normals") = py::none(),
            py::arg("uvs") = py::none(), py::arg("tangents") = py::none(), py::arg("colors") = py::none(),
            py::arg("submeshes") = py::none(), py::arg("material_slots") = std::vector<std::string>{},
            "Atomically replace all Mesh geometry and layout; omitted normals/tangents are derived")
        .def(
            "update_mesh_vertices",
            [](AssetRegistry &self, const std::string &guid, size_t first, const py::object &positions,
               const py::object &normals, const py::object &uvs, const py::object &tangents, const py::object &colors) {
                auto current = self.GetAsset<InxMesh>(guid);
                if (!current)
                    throw py::value_error("update_mesh_vertices requires a loaded Mesh GUID");
                const py::object values[] = {positions, normals, uvs, tangents, colors};
                size_t count = 0;
                bool found = false;
                for (const auto &value : values) {
                    if (value.is_none())
                        continue;
                    const auto array = value.cast<py::array>();
                    if (array.ndim() != 2)
                        throw py::value_error("Mesh vertex streams must be two-dimensional");
                    if (!found) {
                        count = static_cast<size_t>(array.shape(0));
                        found = true;
                    } else if (array.shape(0) != static_cast<py::ssize_t>(count)) {
                        throw py::value_error("Mesh vertex streams must have the same row count");
                    }
                }
                if (!found)
                    throw py::value_error("update_mesh_vertices requires at least one vertex stream");
                const auto &source = current->GetVertices();
                if (first > source.size() || count > source.size() - first)
                    throw py::value_error("Mesh vertex update exceeds the existing vertex range");
                std::vector<Vertex> vertices(source.begin() + first, source.begin() + first + count);
                ApplyOptionalRows<3>(positions, count, "positions",
                                     [&](size_t row, const auto &data, py::ssize_t index) {
                                         vertices[row].pos = {data(index, 0), data(index, 1), data(index, 2)};
                                     });
                ApplyOptionalRows<3>(normals, count, "normals", [&](size_t row, const auto &data, py::ssize_t index) {
                    vertices[row].normal = {data(index, 0), data(index, 1), data(index, 2)};
                });
                ApplyOptionalRows<2>(uvs, count, "uvs", [&](size_t row, const auto &data, py::ssize_t index) {
                    vertices[row].texCoord = {data(index, 0), data(index, 1)};
                });
                ApplyOptionalRows<4>(tangents, count, "tangents", [&](size_t row, const auto &data, py::ssize_t index) {
                    vertices[row].tangent = {data(index, 0), data(index, 1), data(index, 2), data(index, 3)};
                });
                ApplyOptionalRows<3>(colors, count, "colors", [&](size_t row, const auto &data, py::ssize_t index) {
                    vertices[row].color = {data(index, 0), data(index, 1), data(index, 2)};
                });
                InxMesh replacement = *current;
                replacement.UpdateVertexRange(first, vertices);
                self.PublishMesh(guid, std::move(replacement));
            },
            py::arg("guid"), py::arg("first"), py::arg("positions") = py::none(), py::arg("normals") = py::none(),
            py::arg("uvs") = py::none(), py::arg("tangents") = py::none(), py::arg("colors") = py::none(),
            "Atomically replace matching CPU vertex-stream ranges while preserving topology")
        .def(
            "recalculate_mesh_normals",
            [](AssetRegistry &self, const std::string &guid) {
                auto current = self.GetAsset<InxMesh>(guid);
                if (!current)
                    throw py::value_error("normal recalculation requires a loaded Mesh GUID");
                auto vertices = current->GetVertices();
                RecalculateMeshNormals(vertices, current->GetIndices());
                InxMesh replacement = *current;
                replacement.UpdateVertexRange(0, vertices);
                self.PublishMesh(guid, std::move(replacement));
            },
            py::arg("guid"), "Recalculate and publish all Mesh normals")
        .def(
            "recalculate_mesh_tangents",
            [](AssetRegistry &self, const std::string &guid) {
                auto current = self.GetAsset<InxMesh>(guid);
                if (!current)
                    throw py::value_error("tangent recalculation requires a loaded Mesh GUID");
                auto vertices = current->GetVertices();
                RecalculateMeshTangents(vertices, current->GetIndices());
                InxMesh replacement = *current;
                replacement.UpdateVertexRange(0, vertices);
                self.PublishMesh(guid, std::move(replacement));
            },
            py::arg("guid"), "Recalculate and publish all Mesh tangents")
        .def(
            "update_mesh_positions",
            [](AssetRegistry &self, const std::string &guid, size_t first,
               const py::array_t<float, py::array::c_style | py::array::forcecast> &positions,
               const py::object &normals) {
                if (positions.ndim() != 2 || positions.shape(1) != 3)
                    throw py::value_error("positions must have shape (N, 3)");
                const auto data = positions.unchecked<2>();
                std::vector<glm::vec3> values(static_cast<size_t>(positions.shape(0)));
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = {data(i, 0), data(i, 1), data(i, 2)};
                std::optional<std::vector<glm::vec3>> normalValues;
                if (!normals.is_none()) {
                    const auto array = normals.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                    if (array.ndim() != 2 || array.shape(1) != 3 || array.shape(0) != positions.shape(0))
                        throw py::value_error("normals must have shape (N, 3) and match positions");
                    const auto normalData = array.unchecked<2>();
                    normalValues.emplace(values.size());
                    for (size_t i = 0; i < values.size(); ++i)
                        (*normalValues)[i] = {normalData(i, 0), normalData(i, 1), normalData(i, 2)};
                }
                self.UpdateMeshPositions(guid, first, values, normalValues);
            },
            py::arg("guid"), py::arg("first"), py::arg("positions"), py::arg("normals") = py::none(),
            "Publish positions and optional normals together; preserve topology, without collision recooking.")
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
        .def(
            "begin_load_mesh_by_guid",
            [](AssetRegistry &self, const std::string &guid) { return self.BeginLoadAsset(guid, ResourceType::Mesh); },
            py::arg("guid"), "Schedule Assimp CPU mesh preparation on JobSystem")
        .def(
            "begin_load_material_by_guid",
            [](AssetRegistry &self, const std::string &guid) {
                return self.BeginLoadAsset(guid, ResourceType::Material);
            },
            py::arg("guid"), "Schedule material parsing on JobSystem")
        .def(
            "begin_load_physic_material_by_guid",
            [](AssetRegistry &self, const std::string &guid) {
                return self.BeginLoadAsset(guid, ResourceType::PhysicMaterial);
            },
            py::arg("guid"), "Schedule physics material parsing on JobSystem")
        .def(
            "begin_load_shader_by_guid",
            [](AssetRegistry &self, const std::string &guid) {
                return self.BeginLoadAsset(guid, ResourceType::Shader);
            },
            py::arg("guid"), "Schedule authored shader compilation on JobSystem")
        .def(
            "begin_load_audio_by_guid",
            [](AssetRegistry &self, const std::string &guid) { return self.BeginLoadAsset(guid, ResourceType::Audio); },
            py::arg("guid"), "Schedule audio decode on JobSystem")
        .def(
            "load_texture_by_guid",
            [](AssetRegistry &self, const std::string &guid) {
                return self.LoadAsset<InxTexture>(guid, ResourceType::Texture);
            },
            py::arg("guid"), "Load a decoded texture CPU artifact by GUID")
        .def(
            "get_texture_asset",
            [](AssetRegistry &self, const std::string &guid) { return self.GetAsset<InxTexture>(guid); },
            py::arg("guid"), "Get a cached decoded texture asset by GUID")
        .def(
            "begin_load_texture_by_guid",
            [](AssetRegistry &self, const std::string &guid) {
                return self.BeginLoadAsset(guid, ResourceType::Texture);
            },
            py::arg("guid"), "Schedule texture artifact load/decode on JobSystem")
        .def("try_commit_asset_load", &AssetRegistry::TryCommitAssetLoad, py::arg("ticket"),
             py::arg("allow_stale_if_unloaded") = false,
             "Publish a completed typed CPU payload; false means still pending")

        // Hot-reload / invalidation
        .def("reload_asset", &AssetRegistry::ReloadAsset, py::arg("guid"),
             "Reload an asset in-place from disk (preserves shared_ptr identity)")
        .def("invalidate_asset", &AssetRegistry::InvalidateAsset, py::arg("guid"),
             "Evict an asset from cache so next load re-reads from disk")
        .def("remove_asset", &AssetRegistry::RemoveAsset, py::arg("guid"),
             "Fully remove an asset record (e.g. when file is deleted)")

        .def("update_loaded_asset_path", &AssetRegistry::UpdateLoadedAssetPath, py::arg("guid"), py::arg("new_path"),
             "Update path-bearing cached state by GUID after a catalog move")

        // Queries
        .def("is_loaded", &AssetRegistry::IsLoaded, py::arg("guid"), "Check if an asset is currently cached")
        .def("get_asset_version", &AssetRegistry::GetAssetVersion, py::arg("guid"),
             "Get the last successfully published runtime generation, or zero before first publication")
        .def("get_asset_runtime_type_name", &AssetRegistry::GetAssetRuntimeTypeName, py::arg("guid"),
             "Get the validated native payload type name, or an empty string")
        .def("get_asset_residency", &AssetRegistry::GetAssetResidency, py::arg("guid"))
        .def("get_all_asset_residency", &AssetRegistry::GetAllAssetResidency)
        .def_property("cpu_budget_bytes", &AssetRegistry::GetCpuBudgetBytes, &AssetRegistry::SetCpuBudgetBytes)
        .def_property_readonly("total_cpu_bytes", &AssetRegistry::GetTotalCpuBytes)
        .def_property_readonly("cpu_eviction_count", &AssetRegistry::GetCpuEvictionCount)
        .def("trim_cpu_budget", &AssetRegistry::TrimCpuBudget)
        .def("pin_asset", &AssetRegistry::PinAsset, py::arg("guid"))
        .def("unpin_asset", &AssetRegistry::UnpinAsset, py::arg("guid"));
}

} // namespace infernux
