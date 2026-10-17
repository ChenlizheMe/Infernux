#include "BindingSceneInternal.h"
#include "MatrixPyBridge.h"
#include "function/renderer/rhi/RhiComputeBuffer.h"
#include "function/resources/AssetRegistry/AssetRegistry.h"
#include "function/resources/InxMaterial/InxMaterial.h"
#include "function/resources/InxMesh/InxMesh.h"
#include "function/resources/InxResource/InxResourceMeta.h"
#include "function/scene/MeshRenderer.h"
#include <cstring>

namespace py = pybind11;

namespace infernux::scene_binding
{

static void BindRendererMaterials(NativeClass<MeshRenderer, Component> &cls)
{
    cls.def("has_inline_mesh", &MeshRenderer::HasInlineMesh)
        .def_property("inline_mesh_name", &MeshRenderer::GetInlineMeshName, &MeshRenderer::SetInlineMeshName,
                      "Display name for inline (primitive) meshes, e.g. 'Cube', 'Sphere'")
        .def(
            "get_effective_material",
            [](const MeshRenderer &mr, uint32_t slot) { return mr.GetEffectiveMaterial(slot); }, py::arg("slot") = 0,
            "Get the effective material for a given slot (custom or default)")
        // Multi-material API
        .def_property_readonly(
            "material_count",
            [](const MeshRenderer &mr) -> uint32_t { return static_cast<uint32_t>(mr.GetMaterialGuids().size()); },
            "Number of material slots")
        .def(
            "get_material", [](const MeshRenderer &mr, uint32_t slot) { return mr.GetMaterial(slot); }, py::arg("slot"),
            "Get material at slot index")
        .def(
            "set_material",
            [](MeshRenderer &mr, uint32_t slot, py::object material) {
                if (material.is_none()) {
                    mr.SetMaterial(slot, std::string{});
                    return;
                }
                if (py::isinstance<py::str>(material)) {
                    mr.SetMaterial(slot, material.cast<std::string>());
                    return;
                }
                try {
                    mr.SetMaterial(slot, material.cast<std::shared_ptr<InxMaterial>>());
                    return;
                } catch (const py::cast_error &) {
                }
                throw py::type_error("set_material expects a material GUID string, InxMaterial, or None");
            },
            py::arg("slot"), py::arg("material"), "Set material at slot index by GUID, material object, or None")
        .def(
            "get_material_guids", [](const MeshRenderer &mr) { return mr.GetMaterialGuids(); },
            "Get all material slot GUIDs as a list")
        .def("set_materials", &MeshRenderer::SetMaterialSlots, py::arg("guids"),
             "Replace all slots with GUIDs, InxMaterial instances, or None")
        .def(
            "set_material_slot_count", [](MeshRenderer &mr, uint32_t count) { mr.SetMaterialSlotCount(count); },
            py::arg("count"), "Set the number of material slots");
}

static void BindRendererParameters(NativeClass<MeshRenderer, Component> &cls)
{
    cls.def(
           "set_parameter",
           [](py::object renderer, const std::string &name, py::object value, uint32_t materialSlot, bool persistent,
              const std::string &owner) {
               auto material = renderer.cast<MeshRenderer &>().GetEffectiveMaterial(materialSlot);
               if (!material)
                   throw py::value_error("set_parameter requires an effective material");
               const MaterialProperty *property = material->GetProperty(name);
               if (!property)
                   throw py::key_error("material shader has no parameter named '" + name + "'");

               MaterialPropertyValue nativeValue;
               switch (property->type) {
               case MaterialPropertyType::Float:
                   if ((!py::isinstance<py::float_>(value) && !py::isinstance<py::int_>(value)) ||
                       py::isinstance<py::bool_>(value))
                       throw py::type_error("float renderer parameters require one number");
                   nativeValue = value.cast<float>();
                   break;
               case MaterialPropertyType::Int:
                   if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value))
                       throw py::type_error("int renderer parameters require one integer");
                   nativeValue = value.cast<int>();
                   break;
               case MaterialPropertyType::Float2:
               case MaterialPropertyType::Float3:
               case MaterialPropertyType::Float4:
               case MaterialPropertyType::Color: {
                   if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value))
                       throw py::type_error("vector and color renderer parameters require a numeric sequence");
                   py::sequence sequence = value.cast<py::sequence>();
                   const size_t expected = property->type == MaterialPropertyType::Float2
                                               ? 2
                                               : (property->type == MaterialPropertyType::Float3 ? 3 : 4);
                   if (static_cast<size_t>(py::len(sequence)) != expected)
                       throw py::value_error("renderer parameter vector has the wrong number of channels");
                   if (expected == 2)
                       nativeValue = glm::vec2(sequence[0].cast<float>(), sequence[1].cast<float>());
                   else if (expected == 3)
                       nativeValue =
                           glm::vec3(sequence[0].cast<float>(), sequence[1].cast<float>(), sequence[2].cast<float>());
                   else
                       nativeValue = glm::vec4(sequence[0].cast<float>(), sequence[1].cast<float>(),
                                               sequence[2].cast<float>(), sequence[3].cast<float>());
                   break;
               }
               case MaterialPropertyType::Mat4: {
                   nativeValue = binding::Matrix4FromPython(value, "Renderer matrix");
                   break;
               }
               case MaterialPropertyType::Texture2D:
                   if (py::isinstance<py::str>(value))
                       nativeValue = value.cast<std::string>();
                   else if (py::hasattr(value, "guid"))
                       nativeValue = value.attr("guid").cast<std::string>();
                   else
                       throw py::type_error("texture renderer parameters require a texture GUID or Texture object");
                   break;
               case MaterialPropertyType::FloatArray:
                   if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value))
                       throw py::type_error("float array renderer parameters require a numeric sequence");
                   nativeValue = value.cast<std::vector<float>>();
                   break;
               case MaterialPropertyType::Float4Array: {
                   if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value))
                       throw py::type_error("float4 array renderer parameters require a vector sequence");
                   std::vector<glm::vec4> values;
                   for (py::handle item : value.cast<py::sequence>()) {
                       py::sequence vector = py::reinterpret_borrow<py::sequence>(item);
                       if (py::len(vector) != 4)
                           throw py::value_error("float4 array renderer parameters require four-component vectors");
                       values.emplace_back(vector[0].cast<float>(), vector[1].cast<float>(), vector[2].cast<float>(),
                                           vector[3].cast<float>());
                   }
                   nativeValue = std::move(values);
                   break;
               }
               }
               // Custom sequence/texture conversion above can retire the
               // renderer. Resolve the original wrapper immediately before use.
               renderer.cast<MeshRenderer &>().SetParameter(materialSlot, name, std::move(nativeValue), persistent,
                                                            owner);
           },
           py::arg("name"), py::arg("value"), py::arg("material_slot") = 0, py::arg("persistent") = false,
           py::arg("owner") = "script",
           "Set one shader-reflected parameter on this renderer without mutating its shared material")
        .def(
            "get_parameter",
            [](const MeshRenderer &renderer, const std::string &name, uint32_t materialSlot, bool persistentOnly,
               const std::string &owner) -> py::object {
                const MaterialProperty *property = renderer.GetParameter(materialSlot, name, persistentOnly, owner);
                if (!property)
                    return py::none();
                switch (property->type) {
                case MaterialPropertyType::Float:
                    return py::float_(std::get<float>(property->value));
                case MaterialPropertyType::Int:
                    return py::int_(std::get<int>(property->value));
                case MaterialPropertyType::Float2: {
                    const auto value = std::get<glm::vec2>(property->value);
                    return py::make_tuple(value.x, value.y);
                }
                case MaterialPropertyType::Float3: {
                    const auto value = std::get<glm::vec3>(property->value);
                    return py::make_tuple(value.x, value.y, value.z);
                }
                case MaterialPropertyType::Float4:
                case MaterialPropertyType::Color: {
                    const auto value = std::get<glm::vec4>(property->value);
                    return py::make_tuple(value.x, value.y, value.z, value.w);
                }
                case MaterialPropertyType::Mat4: {
                    const auto value = std::get<glm::mat4>(property->value);
                    py::tuple result(16);
                    for (int index = 0; index < 16; ++index)
                        result[index] = value[index / 4][index % 4];
                    return result;
                }
                case MaterialPropertyType::Texture2D:
                    return py::str(std::get<std::string>(property->value));
                case MaterialPropertyType::FloatArray:
                    return py::cast(std::get<std::vector<float>>(property->value));
                case MaterialPropertyType::Float4Array: {
                    py::list values;
                    for (const auto &value : std::get<std::vector<glm::vec4>>(property->value))
                        values.append(py::make_tuple(value.x, value.y, value.z, value.w));
                    return values;
                }
                }
                return py::none();
            },
            py::arg("name"), py::arg("material_slot") = 0, py::arg("persistent_only") = false, py::arg("owner") = "")
        .def(
            "remove_parameter",
            [](MeshRenderer &renderer, const std::string &name, uint32_t materialSlot, bool persistent,
               const std::string &owner) { return renderer.RemoveParameter(materialSlot, name, persistent, owner); },
            py::arg("name"), py::arg("material_slot") = 0, py::arg("persistent") = false, py::arg("owner") = "script")
        .def(
            "clear_parameters",
            [](MeshRenderer &renderer, uint32_t materialSlot, bool persistent, const std::string &owner) {
                renderer.ClearParameters(materialSlot, persistent, owner);
            },
            py::arg("material_slot") = 0, py::arg("persistent") = false, py::arg("owner") = "script")
        .def("serialize", &MeshRenderer::Serialize, "Serialize MeshRenderer to JSON string");
}

static void BindRendererGeometry(NativeClass<MeshRenderer, Component> &cls)
{
    cls.def(
           "set_primitive_mesh",
           [](MeshRenderer &mr, PrimitiveType type) {
               const std::vector<Vertex> *vertices = nullptr;
               const std::vector<uint32_t> *indices = nullptr;
               const char *defaultName = "Primitive";
               GetPrimitiveMeshData(type, vertices, indices, defaultName);
               mr.SetSharedPrimitiveMesh(*vertices, *indices, defaultName);
           },
           py::arg("type"), "Set the mesh to a built-in primitive (Cube, Sphere, Quad, etc.)")
        .def(
            "set_inline_mesh_data",
            [](MeshRenderer &mr, const py::array_t<float, py::array::c_style | py::array::forcecast> &positions,
               const std::optional<FloatArray> &normals,
               const py::array_t<float, py::array::c_style | py::array::forcecast> &uvs,
               const py::array_t<uint32_t, py::array::c_style | py::array::forcecast> &indices, const std::string &name,
               const std::optional<FloatArray> &tangents) {
                if (positions.ndim() != 2 || positions.shape(1) != 3)
                    throw py::value_error("positions must have shape (N, 3)");
                if (uvs.ndim() != 2 || uvs.shape(1) != 2 || uvs.shape(0) != positions.shape(0))
                    throw py::value_error("uvs must have shape (N, 2) and match positions");
                if (indices.ndim() != 1)
                    throw py::value_error("indices must have shape (M,)");
                if (!normals && indices.shape(0) % 3 != 0)
                    throw py::value_error("Normal generation requires complete triangles");

                const size_t vertexCount = static_cast<size_t>(positions.shape(0));
                std::vector<Vertex> vertices(vertexCount);
                const auto positionData = positions.unchecked<2>();
                const auto uvData = uvs.unchecked<2>();
                for (size_t i = 0; i < vertexCount; ++i) {
                    auto &vertex = vertices[i];
                    const auto row = static_cast<py::ssize_t>(i);
                    vertex.pos = {positionData(row, 0), positionData(row, 1), positionData(row, 2)};
                    vertex.normal = glm::vec3(0.0f);
                    vertex.texCoord = {uvData(row, 0), uvData(row, 1)};
                }
                if (normals) {
                    const auto &normalArray = *normals;
                    if (normalArray.ndim() != 2 || normalArray.shape(1) != 3 ||
                        normalArray.shape(0) != positions.shape(0))
                        throw py::value_error("normals must have shape (N, 3) and match positions");
                    const auto normalData = normalArray.unchecked<2>();
                    for (py::ssize_t row = 0; row < normalArray.shape(0); ++row)
                        vertices[row].normal = {normalData(row, 0), normalData(row, 1), normalData(row, 2)};
                }
                if (tangents) {
                    const auto &tangentArray = *tangents;
                    if (tangentArray.ndim() != 2 || tangentArray.shape(1) != 4 ||
                        tangentArray.shape(0) != positions.shape(0))
                        throw py::value_error("tangents must have shape (N, 4) and match positions");
                    const auto tangentData = tangentArray.unchecked<2>();
                    for (py::ssize_t row = 0; row < tangentArray.shape(0); ++row)
                        vertices[row].tangent = {tangentData(row, 0), tangentData(row, 1), tangentData(row, 2),
                                                 tangentData(row, 3)};
                }

                std::vector<uint32_t> encodedIndices(static_cast<size_t>(indices.shape(0)));
                const auto indexData = indices.unchecked<1>();
                for (size_t i = 0; i < encodedIndices.size(); ++i) {
                    const uint32_t index = indexData(static_cast<py::ssize_t>(i));
                    if (index >= vertexCount)
                        throw py::value_error("indices contain a vertex index outside positions");
                    encodedIndices[i] = index;
                }

                if (!normals)
                    RecalculateMeshNormals(vertices, encodedIndices);
                if (!tangents && encodedIndices.size() % 3 == 0)
                    RecalculateMeshTangents(vertices, encodedIndices);

                mr.SetProceduralMesh(std::move(vertices), std::move(encodedIndices));
                mr.SetInlineMeshName(name.empty() ? "Procedural Mesh" : name);
            },
            py::arg("positions"), py::arg("normals"), py::arg("uvs"), py::arg("indices"),
            py::arg("name") = "Procedural Mesh", py::arg("tangents") = py::none(),
            "Copy NumPy geometry; omitted normals/tangents are derived. Does not recook colliders.")
        .def(
            "get_vertex_buffer_data",
            [](const MeshRenderer &mr) {
                if (!mr.HasInlineMesh())
                    throw py::value_error("Vertex-buffer data requires an inline mesh");
                const auto &vertices = mr.GetInlineVertices();
                constexpr py::ssize_t words = static_cast<py::ssize_t>(sizeof(Vertex) / sizeof(float));
                static_assert(sizeof(Vertex) % sizeof(float) == 0);
                py::array_t<float> result({static_cast<py::ssize_t>(vertices.size()), words});
                if (!vertices.empty())
                    std::memcpy(result.mutable_data(), vertices.data(), vertices.size() * sizeof(Vertex));
                return result;
            },
            "Return canonical interleaved Vertex bytes as float32 (N, 25) storage")
        .def(
            "get_tangents",
            [](const MeshRenderer &mr) -> py::list {
                py::list result;
                if (mr.HasMeshAsset()) {
                    const auto geometry = mr.GetAssetGeometry();
                    if (geometry) {
                        for (const auto &vertex : geometry->vertices)
                            result.append(
                                py::make_tuple(vertex.tangent.x, vertex.tangent.y, vertex.tangent.z, vertex.tangent.w));
                    }
                } else if (mr.HasInlineMesh()) {
                    for (const auto &vertex : mr.GetInlineVertices())
                        result.append(
                            py::make_tuple(vertex.tangent.x, vertex.tangent.y, vertex.tangent.z, vertex.tangent.w));
                }
                return result;
            },
            "Get tangents in the same geometry space as positions and normals, as (x, y, z, handedness) tuples")
        .def("recalculate_normals", &MeshRenderer::RecalculateInlineNormals,
             "Rebuild CPU inline normals from the authored triangle topology")
        .def("recalculate_tangents", &MeshRenderer::RecalculateInlineTangents,
             "Rebuild CPU inline tangent frames from positions, normals, and UVs")
        .def("recalculate_bounds", &MeshRenderer::RecalculateInlineBounds,
             "Rebuild CPU inline local bounds from positions")
        .def(
            "set_vertex_buffer",
            [](MeshRenderer &mr, const std::shared_ptr<rhi::ComputeBuffer> &buffer, const glm::vec3 &boundsMin,
               const glm::vec3 &boundsMax,
               bool worldSpace) { mr.SetVertexBuffer(buffer, boundsMin, boundsMax, worldSpace); },
            py::arg("buffer"), py::arg("bounds_min"), py::arg("bounds_max"), py::arg("world_space") = false,
            "Bind a resident inx.buffer as the canonical interleaved vertex stream")
        .def("clear_vertex_buffer", &MeshRenderer::ClearVertexBuffer,
             "Return rendering to the authored CPU vertex stream")
        .def_property_readonly("vertex_buffer_capacity", &MeshRenderer::GetVertexBufferCapacity,
                               "Allocated canonical Vertex slots in the bound resident stream")
        .def(
            "set_mesh_asset_guid",
            [](MeshRenderer &mr, const std::string &guid) {
                if (guid.empty()) {
                    mr.ClearMeshAsset();
                    return;
                }
                auto mesh = AssetRegistry::Instance().LoadAsset<InxMesh>(guid, ResourceType::Mesh);
                if (mesh) {
                    mr.SetMeshAsset(guid, mesh);
                    return;
                }
                mr.SetMeshAssetGuid(guid);
            },
            py::arg("guid"), "Assign a model/mesh asset by GUID")
        .def("clear_mesh_asset", &MeshRenderer::ClearMeshAsset, "Clear the assigned asset mesh")
        .def_property_readonly("model_node_path", &MeshRenderer::GetModelNodePath)
        .def_property_readonly("model_subresource_id", &MeshRenderer::GetModelSubresourceId)
        .def_property_readonly("model_node_group", &MeshRenderer::GetNodeGroup)
        .def(
            "set_model_mesh",
            [](MeshRenderer &renderer, const std::string &guid, const std::vector<std::string> &path) {
                auto mesh = AssetRegistry::Instance().LoadAsset<InxMesh>(guid, ResourceType::Mesh);
                if (!mesh)
                    throw std::invalid_argument("Model mesh source cannot be loaded");
                (void)mesh->RequireModelNode(path);
                renderer.SetMeshAsset(guid, mesh);
                renderer.SetModelNodePath(path);
                renderer.SetModelSubresourceId(GetModelSubresourceId(GetModelMeta(guid, mesh), path));
            },
            py::arg("guid"), py::arg("node_path"));
}

static void BindRendererMeshQueries(NativeClass<MeshRenderer, Component> &cls)
{
    cls.def_property_readonly(
           "vertex_count",
           [](const MeshRenderer &mr) -> size_t {
               if (mr.HasMeshAsset()) {
                   auto m = mr.GetMeshAssetRef().Get();
                   return m ? m->GetVertexCount() : 0;
               }
               return mr.HasInlineMesh() ? mr.GetInlineVertices().size() : 0;
           },
           "Number of vertices in the mesh")
        .def_property_readonly(
            "index_count",
            [](const MeshRenderer &mr) -> size_t {
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetMeshAssetRef().Get();
                    return m ? m->GetIndexCount() : 0;
                }
                return mr.HasInlineMesh() ? mr.GetInlineIndices().size() : 0;
            },
            "Number of indices in the mesh")
        .def_property_readonly("inline_mesh_version", &MeshRenderer::GetInlineMeshVersion,
                               "Monotonic generation of the authored/runtime inline geometry")
        .def(
            "get_positions",
            [](const MeshRenderer &mr) -> py::list {
                py::list result;
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetAssetGeometry();
                    if (m) {
                        for (const auto &v : m->vertices)
                            result.append(py::make_tuple(v.pos.x, v.pos.y, v.pos.z));
                    }
                } else if (mr.HasInlineMesh()) {
                    for (const auto &v : mr.GetInlineVertices())
                        result.append(py::make_tuple(v.pos.x, v.pos.y, v.pos.z));
                }
                return result;
            },
            "Get all vertex positions as a list of (x, y, z) tuples")
        .def(
            "get_normals",
            [](const MeshRenderer &mr) -> py::list {
                py::list result;
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetAssetGeometry();
                    if (m) {
                        for (const auto &v : m->vertices)
                            result.append(py::make_tuple(v.normal.x, v.normal.y, v.normal.z));
                    }
                } else if (mr.HasInlineMesh()) {
                    for (const auto &v : mr.GetInlineVertices())
                        result.append(py::make_tuple(v.normal.x, v.normal.y, v.normal.z));
                }
                return result;
            },
            "Get all vertex normals as a list of (x, y, z) tuples")
        .def(
            "get_uvs",
            [](const MeshRenderer &mr) -> py::list {
                py::list result;
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetAssetGeometry();
                    if (m) {
                        for (const auto &v : m->vertices)
                            result.append(py::make_tuple(v.texCoord.x, v.texCoord.y));
                    }
                } else if (mr.HasInlineMesh()) {
                    for (const auto &v : mr.GetInlineVertices())
                        result.append(py::make_tuple(v.texCoord.x, v.texCoord.y));
                }
                return result;
            },
            "Get all vertex UVs as a list of (u, v) tuples")
        .def(
            "get_indices",
            [](const MeshRenderer &mr) -> py::list {
                py::list result;
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetAssetGeometry();
                    if (m) {
                        for (uint32_t idx : m->indices)
                            result.append(idx);
                    }
                } else if (mr.HasInlineMesh()) {
                    for (uint32_t idx : mr.GetInlineIndices())
                        result.append(idx);
                }
                return result;
            },
            "Get all indices as a flat list")
        .def_property_readonly(
            "mesh_asset_guid", [](const MeshRenderer &mr) -> std::string { return mr.GetMeshAssetGuid(); },
            "GUID of the mesh asset (empty if using inline mesh)")
        .def_property_readonly(
            "has_mesh_asset", [](const MeshRenderer &mr) -> bool { return mr.HasMeshAsset(); },
            "Whether this renderer uses an asset-managed mesh")
        .def_property_readonly(
            "mesh_name",
            [](const MeshRenderer &mr) -> std::string {
                if (mr.HasMeshAsset()) {
                    auto m = mr.GetMeshAssetRef().Get();
                    if (m)
                        return m->GetName();
                }
                return "";
            },
            "Name of the mesh asset (empty if using inline mesh)")
        .def(
            "get_mesh_asset",
            [](const MeshRenderer &mr) -> std::shared_ptr<InxMesh> {
                if (mr.HasMeshAsset())
                    return mr.GetMeshAssetRef().Get();
                return nullptr;
            },
            "Get the InxMesh asset object (None if no asset mesh)")
        .def_property("casts_shadows", &MeshRenderer::CastsShadows, &MeshRenderer::SetCastShadows,
                      "Whether this renderer casts shadows")
        .def_property("receives_shadows", &MeshRenderer::ReceivesShadows, &MeshRenderer::SetReceivesShadows,
                      "Whether this renderer receives shadows")
        .def_property("submesh_index", &MeshRenderer::GetSubmeshIndex, &MeshRenderer::SetSubmeshIndex,
                      "Submesh index to render (-1 = all, >= 0 = specific submesh)")
        .def_property(
            "mesh_pivot_offset", [](const MeshRenderer &mr) -> glm::vec3 { return mr.GetMeshPivotOffset(); },
            [](MeshRenderer &mr, const glm::vec3 &v) { mr.SetMeshPivotOffset(v); },
            "Pivot offset to re-center submesh geometry around the transform")
        .def(
            "get_world_bounds",
            [](const MeshRenderer &mr) -> py::tuple {
                glm::vec3 outMin, outMax;
                mr.GetWorldBounds(outMin, outMax);
                return py::make_tuple(outMin.x, outMin.y, outMin.z, outMax.x, outMax.y, outMax.z);
            },
            "Get world-space AABB as (min_x, min_y, min_z, max_x, max_y, max_z)");
}

void BindMeshRenderer(py::module_ &m)
{
    NativeClass<MeshRenderer, Component> cls(m, "MeshRenderer");
    BindRendererMaterials(cls);
    BindRendererParameters(cls);
    BindRendererGeometry(cls);
    BindRendererMeshQueries(cls);
}

} // namespace infernux::scene_binding
