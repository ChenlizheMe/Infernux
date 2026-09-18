#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshArtifact.h>
#include <function/resources/InxMesh/MeshImportSettings.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/resources/InxMesh/ModelVertexBasis.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxSkinnedMesh/SkinnedModelImporter.h>
#include <function/resources/InxSkinnedMesh/SkinnedMeshArtifact.h>
#include <assimp/scene.h>
#include <platform/filesystem/InxPath.h>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <string>

#ifndef INFERNUX_SOURCE_DIR
#error "INFERNUX_SOURCE_DIR must be supplied by the CMake test target"
#endif

static void TestSkinWeightImport()
{
    aiScene scene;
    scene.mRootNode = new aiNode("Root");
    scene.mRootNode->mNumMeshes = 1;
    scene.mRootNode->mMeshes = new unsigned int[1]{0};
    scene.mNumMeshes = 1;
    scene.mMeshes = new aiMesh *[1]{new aiMesh()};
    auto &mesh = *scene.mMeshes[0];
    mesh.mName = aiString("WeightedTriangle");
    mesh.mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
    mesh.mNumVertices = 3;
    mesh.mVertices = new aiVector3D[3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.mNumFaces = 1;
    mesh.mFaces = new aiFace[1];
    mesh.mFaces[0].mNumIndices = 3;
    mesh.mFaces[0].mIndices = new unsigned int[3]{0, 1, 2};
    mesh.mNumBones = 6;
    mesh.mBones = new aiBone *[6];
    scene.mRootNode->mNumChildren = 6;
    scene.mRootNode->mChildren = new aiNode *[6];
    for (unsigned int i = 0; i < 6; ++i) {
        const std::string name = "Bone" + std::to_string(i);
        scene.mRootNode->mChildren[i] = new aiNode(name);
        scene.mRootNode->mChildren[i]->mParent = scene.mRootNode;
        auto &bone = *(mesh.mBones[i] = new aiBone());
        bone.mName = aiString(name);
        bone.mNumWeights = 3;
        bone.mWeights = new aiVertexWeight[3];
        for (unsigned int v = 0; v < 3; ++v)
            bone.mWeights[v] = aiVertexWeight(v, float(i + 1) / 21.0f);
    }
    const auto convert = [&](int limit, float threshold) {
        return infernux::SkinnedModelImporter::ConvertScene(scene, "weights-guid", "weights.gltf", 1.0f,
                                                           false, limit, threshold);
    };
    for (int limit = 1; limit <= 4; ++limit) {
        const auto imported = convert(limit, 0.0f);
        const auto loaded = infernux::SkinnedMeshArtifact::Deserialize(
            infernux::SkinnedMeshArtifact::Serialize(*imported, "weights"), "weights");
        for (const auto &model : {imported, loaded}) {
            assert(model && model->influences.size() == 3);
            const float sum = float(limit * (13 - limit)) / 2.0f;
            for (size_t v = 0; v < 3; ++v) {
                std::set<uint32_t> selected;
                for (unsigned int i = 0; i < infernux::kMaxSkinInfluences; ++i) {
                    const auto &influence = model->influences[v];
                    assert(model->baseVertices[v].boneWeights[i] == influence.weight[i]);
                    if (influence.weight[i] > 0) {
                        selected.insert(influence.boneIndex[i]);
                        assert(std::abs(influence.weight[i] - float(influence.boneIndex[i] + 1) / sum) < 1e-6f);
                    }
                }
                assert(selected.size() == static_cast<size_t>(limit));
                assert(*selected.begin() == static_cast<uint32_t>(6 - limit));
                assert(*selected.rbegin() == 5);
            }
            // The renderer's palette/CPU consumer reads those same weights.
            for (size_t i = 0; i < 6; ++i)
                model->skeleton.nodes[i + 1].bindLocal[3].x = float(i + 1);
            const auto vertices = model->SampleVertices({});
            float expected = 0;
            for (int i = 7 - limit; i <= 6; ++i)
                expected += float(i * i) / sum;
            assert(std::abs(vertices[0].pos.x - expected) < 1e-5f);
        }
    }
    const auto thresholded = convert(4, 0.2f);
    for (const auto &influence : thresholded->influences) {
        unsigned int count = 0;
        for (const float weight : influence.weight)
            count += weight > 0;
        assert(count == 2);
    }
    bool rejected = false;
    try { (void)convert(4, 0.9f); }
    catch (const std::runtime_error &error) {
        rejected = std::string(error.what()).find("min_bone_weight") != std::string::npos;
    }
    assert(rejected);
    // Truly unweighted geometry follows its mesh node, unlike author weights
    // accidentally eliminated by a threshold.
    for (unsigned int i = 0; i < 6; ++i)
        mesh.mBones[i]->mWeights[0].mWeight = 0.0f;
    const auto unweighted = convert(4, 0.0f);
    assert(unweighted->influences[0].weight[0] == 1.0f);
    // Positive tiny weights still normalize to a valid palette contribution.
    for (unsigned int i = 0; i < 6; ++i)
        mesh.mBones[i]->mWeights[0].mWeight = 1e-10f;
    const auto tiny = convert(4, 0.0f);
    for (const float weight : tiny->influences[0].weight)
        assert(weight == 0.25f);
}

static void TestMikkMirroredSeamPreservesVertexChannels()
{
    aiScene scene;
    scene.mRootNode = new aiNode("Root");
    scene.mRootNode->mNumMeshes = 1;
    scene.mRootNode->mMeshes = new unsigned int[1]{0};
    scene.mNumMeshes = 1;
    scene.mMeshes = new aiMesh *[1]{new aiMesh()};
    auto &mesh = *scene.mMeshes[0];
    mesh.mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
    mesh.mNumVertices = 4;
    mesh.mVertices = new aiVector3D[4]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    mesh.mNormals = new aiVector3D[4]{{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    mesh.mTextureCoords[0] = new aiVector3D[4]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    mesh.mNumUVComponents[0] = 2;
    mesh.mColors[1] = new aiColor4D[4];
    for (unsigned int i = 0; i < 4; ++i)
        mesh.mColors[1][i] = aiColor4D(float(i), 0, 0, 1);
    mesh.mNumFaces = 2;
    mesh.mFaces = new aiFace[2];
    mesh.mFaces[0].mNumIndices = mesh.mFaces[1].mNumIndices = 3;
    mesh.mFaces[0].mIndices = new unsigned int[3]{0, 1, 2};
    mesh.mFaces[1].mIndices = new unsigned int[3]{1, 3, 2};
    mesh.mNumBones = 1;
    mesh.mBones = new aiBone *[1]{new aiBone()};
    mesh.mBones[0]->mName = aiString("Root");
    mesh.mBones[0]->mNumWeights = 4;
    mesh.mBones[0]->mWeights = new aiVertexWeight[4]{{0, 1}, {1, 1}, {2, 1}, {3, 1}};
    mesh.mNumAnimMeshes = 1;
    mesh.mAnimMeshes = new aiAnimMesh *[1]{new aiAnimMesh()};
    auto &morph = *mesh.mAnimMeshes[0];
    morph.mNumVertices = 4;
    morph.mVertices = new aiVector3D[4];
    for (unsigned int i = 0; i < 4; ++i)
        morph.mVertices[i] = mesh.mVertices[i] + aiVector3D(0, 0, 0.1f);
    infernux::MeshImportSettings settings;
    settings.tangentMode = "calculate";
    infernux::BuildModelVertexBasis(scene, settings);
    assert(mesh.mNumVertices == 6 && morph.mNumVertices == 6);
    assert(mesh.mBones[0]->mNumWeights == 6);
    const unsigned int source[] = {0, 1, 2, 1, 3, 2};
    std::set<unsigned int> influenced;
    for (unsigned int i = 0; i < 6; ++i) {
        assert(mesh.mColors[1][i].r == float(source[i]));
        assert((morph.mVertices[i] - mesh.mVertices[i] - aiVector3D(0, 0, 0.1f)).Length() < 1.e-6f);
        const auto expected = i < 3 ? aiVector3D(1, 0, 0) : aiVector3D(0, -1, 0);
        assert((mesh.mTangents[i] - expected).Length() < 1.e-6f);
        const float sign = (mesh.mNormals[i] ^ mesh.mTangents[i]) * mesh.mBitangents[i];
        assert(std::abs(sign - (i < 3 ? 1.0f : -1.0f)) < 1.e-6f);
        influenced.insert(mesh.mBones[0]->mWeights[i].mVertexId);
        assert(mesh.mBones[0]->mWeights[i].mWeight == 1.0f);
    }
    assert(influenced.size() == 6);
    const auto skin = infernux::SkinnedModelImporter::ConvertScene(scene, "mikk-skin", "mikk-skin", settings.scaleFactor);
    assert(skin && skin->baseVertices.size() == 6 && skin->influences.size() == 6);
    for (const auto &influence : skin->influences)
        assert(influence.weight[0] == 1.0f);
}

int main(int argc, char **argv)
{
    TestSkinWeightImport();
    TestMikkMirroredSeamPreservesVertexChannels();
    {
        using Settings = infernux::MeshImportSettings;
        infernux::InxResourceMeta candidate;
        Settings::EnsureDefaults(candidate);
        const auto defaults = Settings::Read(candidate);
        const auto schema = Settings::Schema();
        assert(schema.at("fields").size() ==
               Settings::Flags.size() + Settings::Scalars.size() + Settings::BasisModes.size() + 7);
        assert(defaults.materialImportMode == "description");
        assert(defaults.materialRemaps.empty());
        Settings::ApplyPatch(candidate, {{"material_remaps", {{"material/Body", "abcdabcdabcdabcdabcdabcdabcdabcd"}}}});
        assert(candidate.GetMetadata().at("material_remaps").first == "json_object");
        infernux::InxResourceMeta roundTrip;
        roundTrip.DeserializeDocument(candidate.SerializeDocument());
        assert(Settings::Read(roundTrip).materialRemaps == Settings::Read(candidate).materialRemaps);
        assert(schema.at("fields")[0].at("default").get<float>() == defaults.scaleFactor);
        for (size_t index = 0; index < Settings::Flags.size(); ++index) {
            const auto &flag = Settings::Flags[index];
            assert(schema.at("fields")[index + Settings::Scalars.size()].at("name") == flag.name);
            assert(schema.at("fields")[index + Settings::Scalars.size()].at("default").get<bool>() ==
                   defaults.*(flag.member));
        }
        Settings::ApplyPatch(candidate, {{"scale_factor", 2.0}, {"weld_vertices", false}});
        Settings::EnsureDefaults(candidate);
        assert(Settings::Read(candidate).scaleFactor == 2.0f);
        assert(!Settings::Read(candidate).weldVertices);
        const auto reject = [&](const nlohmann::json &patch) {
            bool rejected = false;
            try {
                Settings::ApplyPatch(candidate, patch);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            assert(rejected);
            assert(Settings::Read(candidate).scaleFactor == 2.0f);
            assert(!Settings::Read(candidate).weldVertices);
        };
        reject({{"scale_factor", 3.0}, {"weld_vertices", "true"}});
        reject({{"scale_factor", 3.0}, {"unknown_setting", true}});
        reject({{"scale_factor", false}});
        reject({{"scale_factor", 0.0}});
        reject({{"scale_factor", -1.0}});
        reject({{"scale_factor", std::numeric_limits<double>::infinity()}});
        reject({{"scale_factor", std::numeric_limits<double>::quiet_NaN()}});
        reject({{"normal_smoothing_angle", -0.1}});
        reject({{"max_bones_per_vertex", 0}});
        reject({{"max_bones_per_vertex", 5}});
        reject({{"max_bones_per_vertex", 1.5}});
        reject({{"max_bones_per_vertex", true}});
        reject({{"min_bone_weight", -0.1}});
        reject({{"min_bone_weight", 1.1}});
        reject({{"normal_smoothing_angle", 175.1}});
        reject({{"normal_smoothing_angle", true}});
        reject({{"normal_mode", true}});
        reject({{"normal_mode", "auto"}});
        reject({{"tangent_mode", 0}});
        reject({{"normal_weighting", "auto"}});
        reject({{"tangent_algorithm", "fallback"}});
        assert(defaults.tangentAlgorithm == "mikktspace");
        for (const bool generate : {true, false}) {
            infernux::InxResourceMeta legacy;
            legacy.AddMetadata("generate_normals", generate);
            legacy.AddMetadata("generate_tangents", generate);
            assert(Settings::Read(legacy).normalMode == (generate ? "import" : "source_only"));
            assert(Settings::Read(legacy).tangentAlgorithm == "assimp");
            Settings::EnsureDefaults(legacy);
            assert(!legacy.HasKey("generate_normals") && !legacy.HasKey("generate_tangents"));
            assert(Settings::Read(legacy).normalMode == (generate ? "import" : "source_only"));
            assert(Settings::Read(legacy).tangentMode == Settings::Read(legacy).normalMode);
            Settings::ApplyPatch(legacy, {{"normal_mode", "none"}});
            Settings::EnsureDefaults(legacy);
            assert(Settings::Read(legacy).normalMode == "none");
        }
        reject(nlohmann::json::array());
        reject({{"material_remaps", nlohmann::json::array()}});
        reject({{"material_remaps", {{"slot/0", "guid"}}}});
        reject({{"rig_type", "humanoid"}});
        reject({{"rig_type", true}});
        reject({{"import_animations", 1}});
        reject({{"material_import_mode", "legacy"}});
        reject({{"material_import_mode", false}});
        infernux::InxResourceMeta legacy;
        legacy.AddMetadata("scale_factor", 0.5f);
        assert(Settings::Read(legacy).weldVertices);
        assert(Settings::Read(legacy).materialImportMode == "description");
    }
    const std::filesystem::path sourceRoot = INFERNUX_SOURCE_DIR;
    const auto animatedFbx = sourceRoot / "external/assimp/test/models/FBX/animation_with_skeleton.fbx";
    infernux::InxResourceMeta animationSettings;
    const auto full =
        infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(animatedFbx), "animated", animationSettings);
    assert(full.skinnedMesh && !full.skinnedMesh->animations.empty() && !full.boneNames.empty());
    for (const std::string mode : {"none", "description"}) {
        infernux::InxResourceMeta materialSettings;
        infernux::MeshImportSettings::ApplyPatch(materialSettings, {{"material_import_mode", mode}});
        const auto variant = infernux::MeshLoader::ImportSourceDetailed(
            infernux::FromFsPath(animatedFbx), "animated", materialSettings);
        assert(variant.vertexCount == full.vertexCount && variant.indexCount == full.indexCount);
        assert(variant.materialSlots == full.materialSlots);
        assert(variant.skinnedMesh && variant.boneNames == full.boneNames);
        assert(variant.mesh->GetMaterialSlotData().empty() == (mode == "none"));
        const auto restored = infernux::MeshArtifact::Deserialize(
            infernux::MeshArtifact::Serialize(*variant.mesh, "fixture"), "fixture");
        assert(restored->GetMaterialSlotNames() == full.materialSlots);
        assert(restored->GetMaterialSlotData().empty() == (mode == "none"));
    }
    for (const bool animations : {false, true}) {
        for (const std::string rig : {"none", "generic"}) {
            infernux::MeshImportSettings::ApplyPatch(animationSettings,
                                                     {{"rig_type", rig}, {"import_animations", animations}});
            const auto variant = infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(animatedFbx),
                                                                            "animated", animationSettings);
            assert(variant.vertexCount == full.vertexCount && variant.indexCount == full.indexCount);
            assert(variant.boneNames == full.boneNames && variant.animationNames == full.animationNames);
            if (rig == "none") {
                assert(!variant.skinnedMesh);
                continue;
            }
            assert(variant.skinnedMesh);
            assert(variant.skinnedMesh->skeleton.bones.size() == full.skinnedMesh->skeleton.bones.size());
            assert(variant.skinnedMesh->influences.size() == full.skinnedMesh->influences.size());
            assert(variant.skinnedMesh->animations.size() == (animations ? full.skinnedMesh->animations.size() : 0));
        }
    }
    const auto blendPath = sourceRoot / "external" / "assimp" / "test" / "models" / "BLEND" / "CubeHierarchy_248.blend";
    assert(std::filesystem::is_regular_file(blendPath));

    infernux::InxResourceMeta metadata;
    const auto imported =
        infernux::MeshLoader::ImportSourceDetailed(blendPath.string(), "0123456789abcdef0123456789abcdef", metadata);

    // A .blend is a composite source, but the loader must expose the same
    // engine-owned mesh contract as FBX/GLTF: binary-ready geometry, stable
    // submesh/material slots, and the authored node hierarchy.
    assert(imported.mesh);
    assert(imported.meshCount > 0);
    assert(imported.vertexCount > 0);
    assert(imported.indexCount > 0);
    assert(imported.mesh->GetSubMeshCount() == imported.meshCount);
    assert(!imported.mesh->GetNodeNames().empty());
    assert(imported.materialSlots.size() == imported.mesh->GetMaterialSlotNames().size());
    assert(imported.mesh->GetGuid() == "0123456789abcdef0123456789abcdef");

    // Assimp's memory importer uses a synthetic OBJ root.  It is an
    // implementation detail and must never leak into the authored hierarchy
    // shown by the editor (or become a persisted model-node identity).
    const auto syntheticObjPath = sourceRoot / "cpp/tests/fixtures/model_smoothing.obj";
    const auto objImport = infernux::MeshLoader::ImportSourceDetailed(
        infernux::FromFsPath(syntheticObjPath), "obj-root-guid", metadata);
    assert(objImport.mesh && !objImport.mesh->GetModelNodes().empty());
    assert(objImport.mesh->GetModelNodes().front().name == "model_smoothing.obj");
    assert(objImport.mesh->GetModelNodes().front().name != "$$$___magic___$$$.obj");

    // A real tree is distinct from the former flat list of mesh-node names.
    // Transform-only parents, nonuniform mirrored scale and local rotation
    // must survive import and the same binary artifact used by Player.
    const auto hierarchyPath = sourceRoot / "cpp/tests/fixtures/model_hierarchy.gltf";
    metadata.AddMetadata("scale_factor", 2.0f);
    const auto hierarchyImport = infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(hierarchyPath),
                                                                            "hierarchy-test-guid", metadata);
    const auto hierarchy = hierarchyImport.mesh;
    assert(hierarchy);
    assert(hierarchyImport.meshCount == hierarchy->GetSubMeshCount());
    assert(hierarchyImport.vertexCount == hierarchy->GetVertexCount());
    assert(hierarchyImport.indexCount == hierarchy->GetIndexCount());
    assert(hierarchyImport.materialSlots == hierarchy->GetMaterialSlotNames());
    assert(hierarchy->GetModelSourceGeometry());
    const auto &nodes = hierarchy->GetModelNodes();
    const auto findNode = [](const auto &modelNodes, const std::string &name) -> size_t {
        for (size_t index = 0; index < modelNodes.size(); ++index)
            if (modelNodes[index].name == name)
                return index;
        assert(false && "expected imported node is missing");
        return 0;
    };
    const auto root = findNode(nodes, "Assembly");
    const auto pivot = findNode(nodes, "Empty pivot");
    const auto upper = findNode(nodes, "Upper");
    const auto lower = findNode(nodes, "Lower");
    assert(nodes[pivot].parentIndex == root);
    assert(nodes[upper].parentIndex == pivot && nodes[lower].parentIndex == pivot);
    assert(nodes[root].nodeGroup == -1 && nodes[pivot].nodeGroup == -1);
    assert(nodes[upper].nodeGroup >= 0 && nodes[lower].nodeGroup >= 0);
    assert(nodes[upper].nodeGroup != nodes[lower].nodeGroup);
    assert(glm::vec3(nodes[root].localTransform[3]) == glm::vec3(4, 6, 8));
    assert(glm::vec3(nodes[pivot].localTransform[3]) == glm::vec3(0, 10, 0));
    assert(nodes[pivot].localTransform[0][0] == -2.0f);
    assert(nodes[pivot].localTransform[1][1] == 3.0f);
    assert(std::abs(nodes[lower].localTransform[0][1] - 1.0f) < 1.e-5f);
    for (const auto &subMesh : hierarchy->GetSubMeshes()) {
        if (subMesh.nodeGroup != nodes[upper].nodeGroup)
            continue;
        // Existing combined geometry remains in model space, not transformed twice.
        assert(subMesh.boundsMin == glm::vec3(-4, 16, 8));
        assert(subMesh.boundsMax == glm::vec3(0, 22, 8));
    }
    const auto bytes = infernux::MeshArtifact::Serialize(*hierarchy, "hierarchy-source");
    const auto cooked = infernux::MeshArtifact::Deserialize(bytes, "hierarchy-source");
    assert(cooked->GetModelSourceGeometry());
    assert(cooked->GetModelSourceGeometry()->vertices.size() == hierarchy->GetVertexCount());
    for (size_t index = 0; index < hierarchy->GetVertexCount(); ++index) {
        assert(cooked->GetModelSourceGeometry()->vertices[index].pos ==
               hierarchy->GetModelSourceGeometry()->vertices[index].pos);
        assert(cooked->GetVertices()[index].pos == hierarchy->GetVertices()[index].pos);
    }
    assert(infernux::MeshArtifact::Serialize(*cooked, "hierarchy-source") == bytes);
    assert(cooked->GetModelNodes().size() == nodes.size());
    for (size_t index = 0; index < nodes.size(); ++index) {
        const auto &restored = cooked->GetModelNodes()[index];
        assert(restored.name == nodes[index].name);
        assert(restored.parentIndex == nodes[index].parentIndex);
        assert(restored.nodeGroup == nodes[index].nodeGroup);
        assert(restored.localTransform == nodes[index].localTransform);
    }

    // Author options must affect the actual imported geometry, not only meta.
    const auto duplicatePath = sourceRoot / "cpp/tests/fixtures/model_duplicate_vertices.obj";
    infernux::InxResourceMeta unweldedSettings;
    unweldedSettings.AddMetadata("weld_vertices", false);
    unweldedSettings.AddMetadata("optimize_mesh", false);
    const auto unwelded = infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(duplicatePath),
                                                                     "unwelded-guid", unweldedSettings);
    infernux::InxResourceMeta weldedSettings;
    weldedSettings.AddMetadata("weld_vertices", true);
    weldedSettings.AddMetadata("optimize_mesh", true);
    const auto welded =
        infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(duplicatePath), "welded-guid", weldedSettings);
    assert(unwelded.vertexCount == 6);
    assert(welded.vertexCount == 4);
    assert(unwelded.indexCount == 6 && welded.indexCount == 6);
    assert(unwelded.mesh->GetModelNodes().size() == welded.mesh->GetModelNodes().size());

    // The source deliberately has a mirrored, non-uniform transform and two
    // UV sets with opposite handedness. Swapping must affect both consumers
    // and recompute the basis, even when the file supplied old tangents.
    const auto uvPath = sourceRoot / "cpp/tests/fixtures/model_uv_basis.gltf";
    for (const bool swap : {false, true}) {
        for (const bool flip : {false, true}) {
            infernux::InxResourceMeta uvSettings;
            uvSettings.AddMetadata("tangent_algorithm", std::string("assimp"));
            uvSettings.AddMetadata("swap_uv_channels", swap);
            uvSettings.AddMetadata("flip_uvs", flip);
            const auto result =
                infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(uvPath), "uv-basis-guid", uvSettings);
            assert(result.skinnedMesh && result.skinnedMesh->IsValid());
            const auto &vertices = result.mesh->GetVertices();
            const auto &skinned = result.skinnedMesh->baseVertices;
            assert(vertices.size() == 3 && skinned.size() == vertices.size());
            const glm::vec3 expectedLocalTangent =
                swap ? glm::normalize(glm::vec3(1, 1, -2)) : glm::normalize(glm::vec3(1, -1, 0));
            const glm::vec3 expectedTangent = glm::normalize(glm::vec3(-2, 3, 0.5) * expectedLocalTangent);
            // Exchanging the axes and reversing V each reverse handedness;
            // the mirrored node contributes one more reversal.
            const float expectedSign = swap != flip ? 1.0f : -1.0f;
            for (size_t index = 0; index < vertices.size(); ++index) {
                const auto &vertex = vertices[index];
                assert(glm::length(vertex.pos - skinned[index].pos) < 1.e-5f);
                assert(glm::length(vertex.normal - skinned[index].normal) < 1.e-5f);
                assert(glm::length(vertex.tangent - skinned[index].tangent) < 1.e-5f);
                assert(glm::length(vertex.texCoord - skinned[index].texCoord) < 1.e-5f);
                assert(glm::length(glm::vec3(vertex.tangent) - expectedTangent) < 1.e-5f);
                assert(std::abs(glm::dot(vertex.normal, glm::vec3(vertex.tangent))) < 1.e-5f);
                assert(vertex.tangent.w == expectedSign);
            }
            const auto serialized = infernux::MeshArtifact::Serialize(*result.mesh, "uv-basis-guid");
            const auto restored = infernux::MeshArtifact::Deserialize(serialized, "uv-basis-guid");
            for (size_t index = 0; index < vertices.size(); ++index) {
                assert(glm::length(restored->GetVertices()[index].tangent - vertices[index].tangent) < 1.e-5f);
                assert(restored->GetVertices()[index].texCoord == vertices[index].texCoord);
            }
        }
    }

    for (const bool swap : {false, true}) {
        for (const bool flip : {false, true}) {
            infernux::InxResourceMeta settings;
            settings.AddMetadata("tangent_mode", std::string("calculate"));
            settings.AddMetadata("tangent_algorithm", std::string("mikktspace"));
            settings.AddMetadata("swap_uv_channels", swap);
            settings.AddMetadata("flip_uvs", flip);
            const auto result = infernux::MeshLoader::ImportSourceDetailed(
                infernux::FromFsPath(uvPath), "mikk-uv-basis", settings);
            const auto &vertices = result.mesh->GetVertices();
            const auto e1 = vertices[1].pos - vertices[0].pos;
            const auto e2 = vertices[2].pos - vertices[0].pos;
            const auto uv1 = vertices[1].texCoord - vertices[0].texCoord;
            const auto uv2 = vertices[2].texCoord - vertices[0].texCoord;
            const auto determinant = uv1.x * uv2.y - uv1.y * uv2.x;
            const auto derivativeU = glm::normalize((e1 * uv2.y - e2 * uv1.y) / determinant);
            const auto derivativeV = (e2 * uv1.x - e1 * uv2.x) / determinant;
            for (size_t i = 0; i < vertices.size(); ++i) {
                const auto &vertex = vertices[i];
                assert(glm::length(glm::vec3(vertex.tangent) - derivativeU) < 1.e-5f);
                const float sign = glm::dot(glm::cross(vertex.normal, derivativeU), derivativeV) < 0 ? -1.f : 1.f;
                assert(vertex.tangent.w == sign);
                assert(glm::length(vertex.tangent - result.skinnedMesh->baseVertices[i].tangent) < 1.e-5f);
            }
        }
    }
    {
        const auto smoothingPath = sourceRoot / "cpp/tests/fixtures/model_smoothing.obj";
        infernux::InxResourceMeta smoothSettings, hardSettings;
        smoothSettings.AddMetadata("normal_smoothing_angle", 175.0f);
        hardSettings.AddMetadata("normal_smoothing_angle", 30.0f);
        const auto smooth = infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(smoothingPath),
                                                                       "smooth-guid", smoothSettings);
        const auto hard =
            infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(smoothingPath), "hard-guid", hardSettings);
        assert(smooth.vertexCount == 4);
        assert(hard.vertexCount == 6);
        bool foundSmoothedCorner = false;
        for (const auto &vertex : smooth.mesh->GetVertices())
            if (glm::length(vertex.pos) < 1.e-5f) {
                assert(glm::length(vertex.normal - glm::normalize(glm::vec3(0, 1, 1))) < 1.e-5f);
                foundSmoothedCorner = true;
            }
        assert(foundSmoothedCorner);
        for (const auto &vertex : hard.mesh->GetVertices())
            assert(vertex.normal == glm::vec3(0, 1, 0) || vertex.normal == glm::vec3(0, 0, 1));
        // Smoothing settings never replace authored normals under Import.
        const auto authoredSmooth = infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(uvPath),
                                                                               "authored-smooth-guid", smoothSettings);
        const auto authoredHard = infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(uvPath),
                                                                             "authored-hard-guid", hardSettings);
        for (size_t index = 0; index < authoredSmooth.vertexCount; ++index)
            assert(authoredSmooth.mesh->GetVertices()[index].normal == authoredHard.mesh->GetVertices()[index].normal);
    }
    {
        const auto authoredPath = sourceRoot / "cpp/tests/fixtures/model_authored_normals.obj";
        const auto missingPath = sourceRoot / "cpp/tests/fixtures/model_smoothing.obj";
        for (const auto *mode : {"import", "source_only", "calculate", "none"}) {
            infernux::InxResourceMeta settings;
            settings.AddMetadata("normal_mode", std::string(mode));
            settings.AddMetadata("normal_smoothing_angle", 30.0f);
            const auto authored = infernux::MeshLoader::ImportSourceDetailed(
                infernux::FromFsPath(authoredPath), "authored-basis", settings);
            const auto missing = infernux::MeshLoader::ImportSourceDetailed(
                infernux::FromFsPath(missingPath), "missing-basis", settings);
            for (const auto &vertex : authored.mesh->GetVertices()) {
                if (std::string_view(mode) == "none") {
                    assert(vertex.normal == glm::vec3(0));
                    assert(vertex.tangent == glm::vec4(0));
                } else if (std::string_view(mode) == "calculate")
                    assert(vertex.normal == glm::vec3(0, 1, 0) || vertex.normal == glm::vec3(0, 0, 1));
                else
                    assert(vertex.normal == glm::vec3(1, 0, 0));
            }
            for (const auto &vertex : missing.mesh->GetVertices()) {
                const bool generated = std::string_view(mode) == "import" || std::string_view(mode) == "calculate";
                assert(std::abs(glm::length(vertex.normal) - (generated ? 1.0f : 0.0f)) < 1.e-5f);
            }
        }
        // Both consumers and their cooked binaries use the same basis policy.
        for (const auto *normalMode : {"import", "calculate", "none", "source_only"}) {
            for (const auto *tangentMode : {"import", "calculate", "none", "source_only"}) {
                infernux::InxResourceMeta settings;
                settings.AddMetadata("normal_mode", std::string(normalMode));
                settings.AddMetadata("tangent_mode", std::string(tangentMode));
                const auto result = infernux::MeshLoader::ImportSourceDetailed(
                    infernux::FromFsPath(uvPath), "mode-basis", settings);
                const auto mesh = infernux::MeshArtifact::Deserialize(
                    infernux::MeshArtifact::Serialize(*result.mesh, "mode-basis"), "mode-basis");
                const auto skin = infernux::SkinnedMeshArtifact::Deserialize(
                    infernux::SkinnedMeshArtifact::Serialize(*result.skinnedMesh, "mode-basis"), "mode-basis");
                const bool normalPresent = std::string_view(normalMode) != "none";
                const bool tangentPresent = normalPresent && std::string_view(tangentMode) != "none" &&
                    !(std::string_view(normalMode) == "calculate" && std::string_view(tangentMode) == "source_only");
                for (size_t i = 0; i < mesh->GetVertices().size(); ++i) {
                    const auto &vertex = mesh->GetVertices()[i];
                    assert(glm::length(vertex.normal - skin->baseVertices[i].normal) < 1.e-5f);
                    assert(glm::length(vertex.tangent - skin->baseVertices[i].tangent) < 1.e-5f);
                    assert(std::abs(glm::length(vertex.normal) - (normalPresent ? 1.0f : 0.0f)) < 1.e-5f);
                    assert(std::abs(glm::length(glm::vec3(vertex.tangent)) - (tangentPresent ? 1.0f : 0.0f)) < 1.e-5f);
                    if (tangentPresent)
                        assert(std::abs(glm::dot(vertex.normal, glm::vec3(vertex.tangent))) < 1.e-5f);
                    else
                        assert(vertex.tangent == glm::vec4(0));
                }
            }
        }
    }
    {
        const auto path = sourceRoot / "cpp/tests/fixtures/model_weighted_normals.obj";
        for (const auto &[weighting, ratio] : std::array<std::pair<const char *, float>, 4>{
                 {{"unweighted", 1.0f}, {"area", 2.0f}, {"angle", 2.0f}, {"area_angle", 4.0f}}}) {
            infernux::InxResourceMeta settings;
            settings.AddMetadata("normal_weighting", std::string(weighting));
            const auto result = infernux::MeshLoader::ImportSourceDetailed(
                infernux::FromFsPath(path), "weighted-normal", settings);
            bool found = false;
            for (const auto &vertex : result.mesh->GetVertices())
                if (vertex.pos == glm::vec3(0)) {
                    assert(glm::length(vertex.normal - glm::normalize(glm::vec3(0, 1, ratio))) < 1.e-5f);
                    found = true;
                }
            assert(found);
            settings.AddMetadata("normal_smoothing_angle", 30.0f);
            const auto hard = infernux::MeshLoader::ImportSourceDetailed(
                infernux::FromFsPath(path), "weighted-hard", settings);
            for (const auto &vertex : hard.mesh->GetVertices())
                assert(vertex.normal == glm::vec3(0, 1, 0) || vertex.normal == glm::vec3(0, 0, 1));
        }
    }
    // Optional modern Blender-generated GLB supplied by an integration run.
    // This is additional evidence, never a replacement for the fixed fixture.
    if (argc > 1) {
        const auto modernImport = infernux::MeshLoader::ImportSourceDetailed(argv[1], "modern-blend-guid", metadata);
        const auto modern = modernImport.mesh;
        assert(modern);
        const auto &modernNodes = modern->GetModelNodes();
        const auto assembly = findNode(modernNodes, "Assembly");
        const auto hinge = findNode(modernNodes, "Hinge");
        assert(modernNodes[hinge].parentIndex == assembly);
        assert(modernNodes[findNode(modernNodes, "Upper")].parentIndex == hinge);
        assert(modernNodes[findNode(modernNodes, "Lower")].parentIndex == hinge);
        assert(modern->GetMaterialSlotCount() == 2);
        assert(modernImport.materialSlots == modern->GetMaterialSlotNames());
    }

    // Composite model sources must expose their regular external textures as
    // authoring paths so the AssetDatabase can publish GUID-only edges. The
    // OBJ fixture deliberately uses Windows-style relative texture tokens;
    // this keeps the importer contract exercised independently of a project
    // scan.
    const auto objPath = sourceRoot / "external" / "assimp" / "test" / "models" / "OBJ" / "spider.obj";
    const auto textures = infernux::MeshLoader::ScanExternalTexturePaths(objPath.string());
    assert(textures.size() == 5);
    for (const char *name :
         {"wal67ar_small.jpg", "wal69ar_small.jpg", "SpiderTex.jpg", "drkwood2.jpg", "engineflare1.jpg"}) {
        assert(textures.count(infernux::FromFsPath(objPath.parent_path() / name)) == 1);
    }
    {
        infernux::InxResourceMeta metadata;
        infernux::MeshImportSettings::EnsureDefaults(metadata);
        const auto source = infernux::MeshLoader::ImportSourceDetailed(
            infernux::FromFsPath(sourceRoot / "external/assimp/test/models/FBX/animation_with_skeleton.fbx"),
            "animation-clips-guid", metadata);
        assert(source.skinnedMesh && !source.skinnedMesh->animations.empty());
        const auto &take = source.skinnedMesh->animations.front();
        const double duration = take.durationTicks / take.ticksPerSecond;
        infernux::MeshImportSettings settings;
        settings.customAnimationClips = true;
        const std::string id(32, 'a');
        settings.animationClips = {{{"id", id}, {"name", "Middle"}, {"source_take", take.name},
                                    {"start", duration * .25}, {"end", duration * .75}}};
        auto model = *source.skinnedMesh;
        infernux::SkinnedModelImporter::ApplyAnimationClips(model, settings);
        assert(model.animations.size() == 1);
        const auto binary = infernux::SkinnedMeshArtifact::Serialize(model, "clips");
        auto loaded = infernux::SkinnedMeshArtifact::Deserialize(binary, "clips");
        assert(loaded->FindAnimation(id) && loaded->FindAnimation("Middle"));
        assert(std::abs(loaded->GetAnimationDurationSeconds(id) - duration * .5) < 1e-5);
        for (double phase : {0.0, .1, .5, 1.0}) {
            infernux::SkinnedSampleRequest original, sliced;
            original.takeName = take.name;
            original.timeSeconds = static_cast<float>(duration * (.25 + .5 * phase));
            original.loop = false;
            sliced.takeName = id;
            sliced.timeSeconds = static_cast<float>(duration * .5 * phase);
            sliced.loop = false;
            const auto expected = source.skinnedMesh->SampleVertices(original);
            const auto actual = loaded->SampleVertices(sliced);
            assert(expected.size() == actual.size());
            float maxError = 0, maxPosition = 0;
            for (size_t i = 0; i < expected.size(); ++i) {
                // This FBX uses coordinates in the thousands. Rebased float
                // playback times/SLERP introduce a few ULPs, not a fixed metre error.
                maxError = std::max(maxError, glm::length(expected[i].pos - actual[i].pos));
                maxPosition = std::max(maxPosition, glm::length(expected[i].pos));
            }
            assert(maxError < 1e-5f + 2e-6f * maxPosition);
        }
        settings.animationClips[0]["name"] = "Renamed";
        model = *source.skinnedMesh;
        infernux::SkinnedModelImporter::ApplyAnimationClips(model, settings);
        assert(model.FindAnimation(id)->name == "Renamed");
        auto invalid = settings;
        invalid.animationClips[0]["end"] = duration + 1;
        model = *source.skinnedMesh;
        bool rejected = false;
        try { infernux::SkinnedModelImporter::ApplyAnimationClips(model, invalid); }
        catch (const std::invalid_argument &error) {
            rejected = std::string(error.what()).find("exceeds source take duration") != std::string::npos;
        }
        assert(rejected && model.animations[0].id == take.id && !model.FindAnimation(id));
        // Analytic sparse curve: both cut points fall between source keys.
        infernux::InxSkinnedMesh analytic;
        infernux::SkinnedRuntimeAnimation animation;
        animation.name = "Linear";
        animation.durationTicks = 2;
        animation.ticksPerSecond = 1;
        infernux::SkinnedRuntimeTrack track;
        track.positions = {{0, glm::vec3(0)}, {2, glm::vec3(2, 4, 6)}};
        track.scales = {{0, glm::vec3(1)}};
        track.rotations = {{0, glm::quat(1, 0, 0, 0)},
                           {2, glm::angleAxis(glm::radians(90.0f), glm::vec3(0, 1, 0))}};
        animation.tracks.push_back(track);
        analytic.animations.push_back(animation);
        settings.animationClips[0]["source_take"] = "Linear";
        settings.animationClips[0]["start"] = .5;
        settings.animationClips[0]["end"] = 1.5;
        infernux::SkinnedModelImporter::ApplyAnimationClips(analytic, settings);
        const auto &cut = analytic.animations[0].tracks[0];
        assert(cut.positions.size() == 2 && cut.positions[0].first == 0 && cut.positions[1].first == 1);
        assert(glm::length(cut.positions[0].second - glm::vec3(.5, 1, 1.5)) < 1e-6f);
        assert(glm::length(cut.positions[1].second - glm::vec3(1.5, 3, 4.5)) < 1e-6f);
        assert(glm::length(cut.scales[0].second - glm::vec3(1)) < 1e-6f);
        const auto expectedRotation = glm::angleAxis(glm::radians(22.5f), glm::vec3(0, 1, 0));
        assert(std::abs(glm::dot(expectedRotation, cut.rotations[0].second)) > 1 - 1e-6f);
        analytic.animations = {animation};
        std::reverse(analytic.animations[0].tracks[0].positions.begin(),
                     analytic.animations[0].tracks[0].positions.end());
        rejected = false;
        try { infernux::SkinnedModelImporter::ApplyAnimationClips(analytic, settings); }
        catch (const std::invalid_argument &error) {
            rejected = std::string(error.what()).find("ordered by time") != std::string::npos;
        }
        assert(rejected && analytic.animations[0].name == "Linear");
    }
    return 0;
}
