/**
 * @file MeshLoader.cpp
 * @brief Assimp-based mesh loading for Infernux's AssetRegistry.
 *
 * Converts Assimp's aiScene into the engine's InxMesh representation.
 * All aiMesh nodes are collected via a recursive scene-graph traversal
 * and merged into a single vertex/index buffer with per-submesh offsets.
 */

#include "MeshLoader.h"
#include "InxMesh.h"
#include "MeshArtifact.h"
#include "MeshImportSettings.h"
#include "ModelVertexBasis.h"

#include <core/config/MathConstants.h>
#include <core/log/InxLog.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxSkinnedMesh/SkinnedModelImporter.h>

#include <assimp/Importer.hpp>
#include <assimp/GltfMaterial.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <platform/filesystem/InxPath.h>
#include <unordered_map>
#include <unordered_set>

namespace infernux
{

// ============================================================================
// Import-setting helpers
// ============================================================================

static unsigned int BuildAssimpFlags(const MeshImportSettings &settings)
{
    unsigned int flags = 0;
    if (settings.tangentAlgorithm == "assimp" && settings.normalMode != "none" &&
        (settings.tangentMode == "import" || settings.tangentMode == "calculate"))
        flags |= aiProcess_CalcTangentSpace;
    // NOTE: aiProcess_OptimizeMeshes and aiProcess_OptimizeGraph are intentionally
    // omitted — they merge meshes across different Assimp nodes, destroying the
    // per-object hierarchy needed for correct scene object splitting.
    if (settings.optimizeMesh)
        flags |= aiProcess_ImproveCacheLocality;
    if (settings.weldVertices)
        flags |= aiProcess_JoinIdenticalVertices;

    flags |= aiProcess_SortByPType; // Separate points/lines from triangles

    return flags;
}

static void PrepareVertexBasis(const aiScene &scene, const MeshImportSettings &settings)
{
    // Change the primary channel before tangent generation and welding, so
    // static geometry and its skinned companion consume the same vertex basis.
    const auto swapChannels = [](auto &mesh) {
        std::swap(mesh.mTextureCoords[0], mesh.mTextureCoords[1]);
        delete[] mesh.mTangents;
        delete[] mesh.mBitangents;
        mesh.mTangents = nullptr;
        mesh.mBitangents = nullptr;
    };
    const auto flipImportedBasis = [](auto &mesh) {
        // Assimp's FlipUVs changes coordinates but leaves authored tangents
        // untouched. Preserve T and reverse B when the V direction reverses.
        if (mesh.HasTextureCoords(0) && mesh.mBitangents)
            for (unsigned int vertex = 0; vertex < mesh.mNumVertices; ++vertex)
                mesh.mBitangents[vertex] *= -1.0f;
    };
    const auto prepare = [&](auto &mesh) {
        if (settings.normalMode == "none" || settings.normalMode == "calculate") {
            delete[] mesh.mNormals;
            mesh.mNormals = nullptr;
        }
        // Tangents belong to a normal/UV basis. Never retain authored tangents
        // after replacing that basis; generation, when requested, follows below.
        if (!mesh.mNormals || settings.tangentMode == "none" || settings.tangentMode == "calculate") {
            delete[] mesh.mTangents;
            delete[] mesh.mBitangents;
            mesh.mTangents = nullptr;
            mesh.mBitangents = nullptr;
        }
    };
    for (unsigned int index = 0; index < scene.mNumMeshes; ++index) {
        auto &mesh = *scene.mMeshes[index];
        prepare(mesh);
        for (unsigned int morph = 0; morph < mesh.mNumAnimMeshes; ++morph)
            prepare(*mesh.mAnimMeshes[morph]);
        // With no secondary channel there is nothing to exchange.
        if (settings.swapUVChannels && mesh.HasTextureCoords(1)) {
            swapChannels(mesh);
            std::swap(mesh.mNumUVComponents[0], mesh.mNumUVComponents[1]);
            if (mesh.mTextureCoordsNames)
                std::swap(mesh.mTextureCoordsNames[0], mesh.mTextureCoordsNames[1]);
            for (unsigned int morph = 0; morph < mesh.mNumAnimMeshes; ++morph)
                swapChannels(*mesh.mAnimMeshes[morph]);
        }
        if (settings.flipUVs) {
            flipImportedBasis(mesh);
            for (unsigned int morph = 0; morph < mesh.mNumAnimMeshes; ++morph)
                flipImportedBasis(*mesh.mAnimMeshes[morph]);
        }
    }
}

// ============================================================================
// Scene traversal — collect all meshes from the node hierarchy
// ============================================================================

/**
 * @brief Recursively traverse the Assimp node tree and collect mesh indices
 *        along with their source-node geometry group.
 *
 * Each node in an Assimp scene has a local transform and references zero or
 * more meshes by index. The source tree owns transforms; vertex data stays
 * local to its node; InxMesh derives the
 * merged model-space view once.
 */
struct CollectedMesh
{
    uint32_t meshIndex; ///< Index into aiScene::mMeshes
    uint32_t nodeGroup; ///< Source node group (for per-object splitting)
};

static glm::mat4 AiToGlm(const aiMatrix4x4 &m)
{
    // Assimp stores column-major internally, but the aiMatrix4x4 API
    // exposes row-major accessors.  GLM is column-major.
    return glm::mat4(m.a1, m.b1, m.c1, m.d1, // col 0
                     m.a2, m.b2, m.c2, m.d2, // col 1
                     m.a3, m.b3, m.c3, m.d3, // col 2
                     m.a4, m.b4, m.c4, m.d4  // col 3
    );
}

static void CollectMeshes(const aiNode *node, std::vector<CollectedMesh> &outMeshes,
                          std::vector<std::string> &outNodeNames, std::vector<ImportedModelNode> &outNodes,
                          int32_t parentIndex, float scale)
{
    const glm::mat4 localTransform = AiToGlm(node->mTransformation);
    const int32_t nodeIndex = static_cast<int32_t>(outNodes.size());
    ImportedModelNode importedNode;
    importedNode.name = node->mName.C_Str();
    importedNode.parentIndex = parentIndex;
    importedNode.localTransform = localTransform;
    // Unit conversion is applied to translations once, not as a scale at
    // every ancestor. Geometry below remains in the existing model space.
    importedNode.localTransform[3] = glm::vec4(glm::vec3(localTransform[3]) * scale, 1.0f);

    if (node->mNumMeshes > 0) {
        uint32_t group = static_cast<uint32_t>(outNodeNames.size());
        importedNode.nodeGroup = static_cast<int32_t>(group);
        outNodeNames.push_back(node->mName.C_Str());
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            outMeshes.push_back({node->mMeshes[i], group});
        }
    }

    outNodes.push_back(std::move(importedNode));

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        CollectMeshes(node->mChildren[i], outMeshes, outNodeNames, outNodes, nodeIndex, scale);
    }
}

// ============================================================================
// Core conversion: aiScene → InxMesh
// ============================================================================

static std::shared_ptr<InxMesh> ConvertScene(const aiScene *scene, const MeshImportSettings &settings,
                                             const std::string &name,
                                             std::vector<MeshSourceImportResult::TextureSource> &textureSources)
{
    auto mesh = std::make_shared<InxMesh>(name);

    // Collect all mesh instances with their transforms and node grouping
    std::vector<CollectedMesh> collectedMeshes;
    std::vector<std::string> nodeNames;
    std::vector<ImportedModelNode> modelNodes;
    collectedMeshes.reserve(scene->mNumMeshes);
    CollectMeshes(scene->mRootNode, collectedMeshes, nodeNames, modelNodes, -1, settings.scaleFactor);

    if (collectedMeshes.empty()) {
        mesh->SetModelNodes(std::move(modelNodes));
        INXLOG_WARN("MeshLoader: scene '", name, "' contains no meshes");
        return mesh;
    }

    // Pre-calculate total counts for a single allocation
    uint32_t totalVertices = 0;
    uint32_t totalIndices = 0;
    for (const auto &cm : collectedMeshes) {
        const aiMesh *aiM = scene->mMeshes[cm.meshIndex];
        totalVertices += aiM->mNumVertices;
        for (unsigned int f = 0; f < aiM->mNumFaces; ++f)
            totalIndices += aiM->mFaces[f].mNumIndices;
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SubMesh> subMeshes;
    vertices.reserve(totalVertices);
    indices.reserve(totalIndices);
    subMeshes.reserve(collectedMeshes.size());

    // Deduplicate material slot assignments:
    // Assimp's material indices are per-aiScene.  Multiple aiMeshes can
    // share the same material index → same slot.
    std::unordered_map<unsigned int, uint32_t> aiMatToSlot;
    std::vector<std::string> materialSlotNames;
    std::vector<MaterialSlotData> materialSlotDataVec;
    std::unordered_map<std::string, unsigned> materialNameCounts;
    for (unsigned mi = 0; mi < scene->mNumMaterials; ++mi) {
        aiString name;
        scene->mMaterials[mi]->Get(AI_MATKEY_NAME, name);
        ++materialNameCounts[name.C_Str()];
    }

    uint32_t currentVertexOffset = 0;
    uint32_t currentIndexOffset = 0;

    const float scale = settings.scaleFactor;
    const bool applyScale = std::abs(scale - 1.0f) > kEpsilon;

    for (const auto &cm : collectedMeshes) {
        const aiMesh *aiM = scene->mMeshes[cm.meshIndex];

        // Skip non-triangle primitives (points, lines)
        if (!(aiM->mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
            continue;

        const bool hasNormals = aiM->HasNormals();
        const bool hasTangents = aiM->HasTangentsAndBitangents();
        const bool hasUVs = aiM->HasTextureCoords(0);
        const bool hasColors = aiM->HasVertexColors(0);

        // ── Vertices ────────────────────────────────────────────────
        for (unsigned int v = 0; v < aiM->mNumVertices; ++v) {
            Vertex vert{};

            // Retain authored local geometry. Unit conversion is shared with
            // node translations, never baked a second time at each parent.
            vert.pos = glm::vec3(aiM->mVertices[v].x, aiM->mVertices[v].y, aiM->mVertices[v].z);
            if (applyScale)
                vert.pos *= scale;

            // Normal
            if (hasNormals) {
                glm::vec3 n(aiM->mNormals[v].x, aiM->mNormals[v].y, aiM->mNormals[v].z);
                vert.normal = n;
            } else
                vert.normal = glm::vec3(0.0f);

            // Tangent + bitangent handedness
            if (hasTangents) {
                glm::vec3 t(aiM->mTangents[v].x, aiM->mTangents[v].y, aiM->mTangents[v].z);

                glm::vec3 b(aiM->mBitangents[v].x, aiM->mBitangents[v].y, aiM->mBitangents[v].z);

                // Compute handedness: sign of dot(cross(N,T), B)
                float handedness = (glm::dot(glm::cross(vert.normal, t), b) < 0.0f) ? -1.0f : 1.0f;
                vert.tangent = glm::vec4(t, handedness);
            } else {
                vert.tangent = glm::vec4(0.0f);
            }

            // UV (channel 0 only for now)
            if (hasUVs) {
                vert.texCoord = glm::vec2(aiM->mTextureCoords[0][v].x, aiM->mTextureCoords[0][v].y);
            } else {
                // Auto-generate UV via triplanar-dominant-axis projection
                // for meshes that have no texture coordinates at all.
                const glm::vec3 &p = vert.pos;
                const glm::vec3 n = hasNormals ? vert.normal : glm::vec3(0.0f, 1.0f, 0.0f);
                const glm::vec3 absN = glm::abs(n);
                if (absN.x >= absN.y && absN.x >= absN.z)
                    vert.texCoord = glm::vec2(p.z, p.y); // project along X
                else if (absN.y >= absN.x && absN.y >= absN.z)
                    vert.texCoord = glm::vec2(p.x, p.z); // project along Y
                else
                    vert.texCoord = glm::vec2(p.x, p.y); // project along Z
            }

            // Vertex colour
            if (hasColors) {
                vert.color = glm::vec3(aiM->mColors[0][v].r, aiM->mColors[0][v].g, aiM->mColors[0][v].b);
            } else {
                vert.color = glm::vec3(1.0f);
            }

            vertices.push_back(vert);
        }

        // ── Indices ─────────────────────────────────────────────────
        uint32_t submeshIndexStart = currentIndexOffset;
        for (unsigned int f = 0; f < aiM->mNumFaces; ++f) {
            const aiFace &face = aiM->mFaces[f];
            for (unsigned int idx = 0; idx < face.mNumIndices; ++idx) {
                indices.push_back(face.mIndices[idx] + currentVertexOffset);
            }
            currentIndexOffset += face.mNumIndices;
        }

        // ── Material slot mapping ───────────────────────────────────
        uint32_t slot = 0;
        auto it = aiMatToSlot.find(aiM->mMaterialIndex);
        if (it != aiMatToSlot.end()) {
            slot = it->second;
        } else {
            slot = static_cast<uint32_t>(materialSlotNames.size());
            aiMatToSlot[aiM->mMaterialIndex] = slot;

            // Extract material name from Assimp
            std::string matName;
            MaterialSlotData slotData;
            if (aiM->mMaterialIndex < scene->mNumMaterials) {
                const aiMaterial *aiMat = scene->mMaterials[aiM->mMaterialIndex];
                aiString aiName;
                aiMat->Get(AI_MATKEY_NAME, aiName);
                matName = aiName.C_Str();
                if (!matName.empty() && materialNameCounts.at(matName) == 1)
                    slotData.sourceId = "material/" + matName;

                if (settings.materialImportMode != "none" && !settings.materialRemaps.contains(slotData.sourceId)) {
                    const auto readTexture = [&](aiTextureType semantic, ModelTexture channel) {
                        if (!aiMat->GetTextureCount(semantic))
                            return;
                        aiString texturePath;
                        unsigned int uvChannel = 0;
                        if (aiMat->GetTexture(semantic, 0, &texturePath, nullptr, &uvChannel) != AI_SUCCESS)
                            throw std::runtime_error("model material texture could not be read");
                        if (texturePath.length) {
                            if (uvChannel != 0)
                                throw std::invalid_argument("model material texture currently requires UV channel 0");
                            int32_t embeddedIndex = -1;
                            if (const auto *image = scene->GetEmbeddedTexture(texturePath.C_Str())) {
                                for (unsigned int index = 0; index < scene->mNumTextures; ++index)
                                    if (scene->mTextures[index] == image) {
                                        embeddedIndex = static_cast<int32_t>(index);
                                        break;
                                    }
                            }
                            textureSources.push_back({slot, texturePath.C_Str(), static_cast<uint32_t>(channel), embeddedIndex});
                        }
                    };
                    readTexture(aiMat->GetTextureCount(aiTextureType_BASE_COLOR)
                                    ? aiTextureType_BASE_COLOR : aiTextureType_DIFFUSE, ModelTexture::BaseColor);
                    readTexture(aiTextureType_NORMALS, ModelTexture::Normal);
                    readTexture(aiTextureType_METALNESS, ModelTexture::Metallic);
                    readTexture(aiTextureType_DIFFUSE_ROUGHNESS, ModelTexture::Roughness);
                    readTexture(aiMat->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION)
                                    ? aiTextureType_AMBIENT_OCCLUSION : aiTextureType_LIGHTMAP, ModelTexture::Occlusion);
                    readTexture(aiTextureType_EMISSIVE, ModelTexture::Emission);
                    aiString packedTexture;
                    slotData.packedMetallicRoughness =
                        aiMat->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE,
                                          &packedTexture) == AI_SUCCESS;
                    aiMat->Get(AI_MATKEY_GLTF_TEXTURE_SCALE(aiTextureType_NORMALS, 0), slotData.normalScale);
                    aiMat->Get(AI_MATKEY_GLTF_TEXTURE_STRENGTH(aiTextureType_LIGHTMAP, 0), slotData.occlusionStrength);
                }

                // Diffuse / base colour
                aiColor4D diffuse;
                if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == AI_SUCCESS) {
                    slotData.baseColor = glm::vec4(diffuse.r, diffuse.g, diffuse.b, diffuse.a);
                }
                // Try PBR base color as override (glTF workflow)
                aiColor4D pbrBase;
                if (aiMat->Get(AI_MATKEY_BASE_COLOR, pbrBase) == AI_SUCCESS) {
                    slotData.baseColor = glm::vec4(pbrBase.r, pbrBase.g, pbrBase.b, pbrBase.a);
                }
                // Emission colour
                aiColor4D emission;
                if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emission) == AI_SUCCESS) {
                    slotData.emissionColor = glm::vec4(emission.r, emission.g, emission.b, emission.a);
                }
                // Metallic factor
                float metallic = 0.0f;
                if (aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == AI_SUCCESS) {
                    slotData.metallic = metallic;
                }
                // Roughness → smoothness
                float roughness = 0.5f;
                if (aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) == AI_SUCCESS) {
                    slotData.smoothness = 1.0f - roughness;
                }
                // glTF exposes the SAME factor as baseColor.a and OPACITY.
                // Do not multiply that factor twice.
                aiString alphaMode;
                const bool explicitAlpha = aiMat->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == AI_SUCCESS;
                float opacity = 1.0f;
                if (aiMat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
                    slotData.opacity = opacity;
                    if (!explicitAlpha)
                        slotData.baseColor.a *= opacity;
                }
                if (explicitAlpha) {
                    const std::string mode = alphaMode.C_Str();
                    if (mode == "MASK")
                        slotData.alphaMode = ModelAlphaMode::Mask;
                    else if (mode == "BLEND")
                        slotData.alphaMode = ModelAlphaMode::Blend;
                    else if (mode != "OPAQUE")
                        throw std::invalid_argument("unsupported source material alpha mode: " + mode);
                    aiMat->Get(AI_MATKEY_GLTF_ALPHACUTOFF, slotData.alphaCutoff);
                } else if (slotData.baseColor.a < 1.0f) {
                    slotData.alphaMode = ModelAlphaMode::Blend;
                }
                int doubleSided = 0;
                aiMat->Get(AI_MATKEY_TWOSIDED, doubleSided);
                slotData.doubleSided = doubleSided != 0;
            }
            if (matName.empty())
                matName = "Material_" + std::to_string(slot);
            materialSlotNames.push_back(matName);
            materialSlotDataVec.push_back(slotData);
        }

        // ── SubMesh ─────────────────────────────────────────────────
        SubMesh sub;
        sub.indexStart = submeshIndexStart;
        sub.indexCount = currentIndexOffset - submeshIndexStart;
        sub.vertexStart = currentVertexOffset;
        sub.vertexCount = aiM->mNumVertices;
        sub.materialSlot = slot;
        sub.nodeGroup = cm.nodeGroup;
        sub.name = aiM->mName.C_Str();

        // Compute per-submesh AABB from the vertices we just added
        if (sub.vertexCount > 0) {
            constexpr float INF = std::numeric_limits<float>::max();
            sub.boundsMin = glm::vec3(INF);
            sub.boundsMax = glm::vec3(-INF);
            for (uint32_t vi = sub.vertexStart; vi < sub.vertexStart + sub.vertexCount; ++vi) {
                sub.boundsMin = glm::min(sub.boundsMin, vertices[vi].pos);
                sub.boundsMax = glm::max(sub.boundsMax, vertices[vi].pos);
            }
        }

        subMeshes.push_back(std::move(sub));
        currentVertexOffset += aiM->mNumVertices;
    }

    // Slot names/layout remain geometry information in None mode. No source
    // materials or remapped material references enter the runtime artifact.
    if (settings.materialImportMode == "none") {
        materialSlotDataVec.clear();
    } else {
        for (const auto &[sourceId, guid] : settings.materialRemaps.items()) {
            auto found = std::find_if(materialSlotDataVec.begin(), materialSlotDataVec.end(),
                                      [&](const auto &slot) { return slot.sourceId == sourceId; });
            if (found == materialSlotDataVec.end())
                throw std::invalid_argument("model material remap source is missing or ambiguous: " + sourceId);
            found->materialGuid = guid.get<std::string>();
        }
    }
    mesh->SetMaterialSlotNames(std::move(materialSlotNames));
    mesh->SetMaterialSlotData(std::move(materialSlotDataVec));
    mesh->SetNodeNames(std::move(nodeNames));
    mesh->SetModelData(std::move(vertices), std::move(indices), std::move(subMeshes), std::move(modelNodes));

    return mesh;
}

// ============================================================================
// IAssetLoader interface
// ============================================================================

MeshSourceImportResult MeshLoader::ImportSourceDetailed(const std::string &filePath, const std::string &guid,
                                                        const InxResourceMeta &metadata)
{
    auto fsPath = ToFsPath(filePath);
    if (!std::filesystem::is_regular_file(fsPath))
        throw std::runtime_error("MeshLoader source file not found: " + filePath);

    // Read file into memory to avoid Assimp's narrow-string path issues on Windows
    std::ifstream file(fsPath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("MeshLoader cannot open source file: " + filePath);
    auto fileSize = file.tellg();
    if (fileSize <= 0)
        throw std::runtime_error("MeshLoader source file is empty or unreadable: " + filePath);
    std::vector<char> fileData(static_cast<size_t>(fileSize));
    file.seekg(0);
    if (!file.read(fileData.data(), fileSize))
        throw std::runtime_error("MeshLoader failed to read source file: " + filePath);

    MeshImportSettings settings = MeshImportSettings::Read(metadata);
    unsigned int flags = BuildAssimpFlags(settings);

    // Derive extension hint for Assimp (e.g. "fbx")
    std::string ext = FromFsPath(fsPath.extension());
    if (!ext.empty() && ext[0] == '.')
        ext = ext.substr(1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

    if (ext == "inxmesh") {
        if (settings.materialImportMode != "description")
            throw std::invalid_argument("material import mode requires a source model, not an authored .inxmesh");
        if (!settings.materialRemaps.empty())
            throw std::invalid_argument("material import remaps require a source model, not an authored .inxmesh");
        MeshSourceImportResult result;
        result.mesh = MeshArtifact::DeserializeSource(std::string_view(fileData.data(), fileData.size()));
        result.mesh->SetGuid(guid);
        result.mesh->SetFilePath(filePath);
        result.meshCount = result.mesh->GetSubMeshCount();
        result.vertexCount = result.mesh->GetVertexCount();
        result.indexCount = result.mesh->GetIndexCount();
        result.materialSlots = result.mesh->GetMaterialSlotNames();
        return result;
    }

    Assimp::Importer importer;
    // Validate external input once, before touching its channel pointers.
    const aiScene *scene =
        importer.ReadFileFromMemory(fileData.data(), fileData.size(), aiProcess_ValidateDataStructure, ext.c_str());

    // OBJ has no authored scene root: Assimp derives it from the input file
    // name. Memory IO supplies a synthetic filename, not an author node name.
    // Restore the real filename at this boundary; never rename OBJ o/g nodes.
    if (scene && scene->mRootNode && ext == "obj" &&
        std::string_view(scene->mRootNode->mName.C_Str()) == "$$$___magic___$$$.obj")
        scene->mRootNode->mName.Set(FromFsPath(fsPath.filename()));

    if (scene) {
        PrepareVertexBasis(*scene, settings);
        scene = importer.ApplyPostProcessing(aiProcess_Triangulate | (settings.flipUVs ? aiProcess_FlipUVs : 0));
        if (scene) {
            BuildModelVertexBasis(*scene, settings);
            scene = importer.ApplyPostProcessing(flags);
        }
    }

    const bool animationOnlyScene = scene && scene->mRootNode && scene->mNumMeshes == 0 && scene->mNumAnimations > 0;
    if (!scene || !scene->mRootNode || ((scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) && !animationOnlyScene))
        throw std::runtime_error("MeshLoader Assimp import failed for '" + filePath +
                                 "': " + importer.GetErrorString());

    std::string name = FromFsPath(fsPath.stem());
    MeshSourceImportResult result;
    auto mesh = ConvertScene(scene, settings, name, result.textureSources);
    result.embeddedImages.resize(scene->mNumTextures);
    std::unordered_map<std::string, size_t> imageNameCounts;
    for (unsigned int index = 0; index < scene->mNumTextures; ++index)
        ++imageNameCounts[scene->mTextures[index]->mFilename.C_Str()];
    for (unsigned int index = 0; index < scene->mNumTextures; ++index) {
        auto &output = result.embeddedImages[index];
        const auto &input = *scene->mTextures[index];
        output.name = input.mFilename.C_Str();
        // Named images survive source reordering. An unnamed source only has
        // its explicit import-local ordinal; do not invent content-hash identity.
        output.key = !output.name.empty() && imageNameCounts.at(output.name) == 1
                         ? "name/" + output.name : "index/" + std::to_string(index);
        if (output.name.empty())
            output.name = "Texture " + std::to_string(index);
        output.width = input.mWidth;
        output.height = input.mHeight;
        if (!input.mHeight) {
            const auto *begin = reinterpret_cast<const unsigned char *>(input.pcData);
            output.bytes.assign(begin, begin + input.mWidth);
        } else {
            output.bytes.resize(static_cast<size_t>(input.mWidth) * input.mHeight * 4);
            for (size_t pixel = 0; pixel < output.bytes.size() / 4; ++pixel) {
                const auto &texel = input.pcData[pixel];
                output.bytes[pixel * 4] = texel.r;
                output.bytes[pixel * 4 + 1] = texel.g;
                output.bytes[pixel * 4 + 2] = texel.b;
                output.bytes[pixel * 4 + 3] = texel.a;
            }
        }
    }
    mesh->SetGuid(guid);
    mesh->SetFilePath(filePath);

    result.mesh = std::move(mesh);
    // Report the published geometry/slot layout, including node instances,
    // rather than unused Assimp materials or uninstanced source mesh counts.
    result.meshCount = result.mesh->GetSubMeshCount();
    result.vertexCount = result.mesh->GetVertexCount();
    result.indexCount = result.mesh->GetIndexCount();
    result.materialSlots = result.mesh->GetMaterialSlotNames();

    std::unordered_set<std::string> seenBones;
    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex) {
        const aiMesh *sourceMesh = scene->mMeshes[meshIndex];
        if (!sourceMesh)
            throw std::runtime_error("MeshLoader Assimp scene contains a null mesh");
        for (unsigned int boneIndex = 0; boneIndex < sourceMesh->mNumBones; ++boneIndex) {
            const aiBone *bone = sourceMesh->mBones[boneIndex];
            if (!bone)
                throw std::runtime_error("MeshLoader Assimp scene contains a null bone");
            std::string boneName = bone->mName.C_Str();
            if (!boneName.empty() && seenBones.insert(boneName).second)
                result.boneNames.push_back(std::move(boneName));
        }
    }

    result.animationNames.reserve(scene->mNumAnimations);
    for (unsigned int animationIndex = 0; animationIndex < scene->mNumAnimations; ++animationIndex) {
        const aiAnimation *animation = scene->mAnimations[animationIndex];
        if (!animation)
            throw std::runtime_error("MeshLoader Assimp scene contains a null animation");
        std::string animationName = animation->mName.C_Str();
        if (animationName.empty())
            animationName = "Anim_" + std::to_string(animationIndex);
        const double rate = std::isfinite(animation->mTicksPerSecond) && animation->mTicksPerSecond > 0
                                ? animation->mTicksPerSecond : 25.0;
        result.sourceAnimations.push_back({{"name", animationName},
                                          {"duration", std::isfinite(animation->mDuration)
                                              ? std::max(0.0, animation->mDuration) / rate : 0.0},
                                          {"sample_rate", rate}});
        result.animationNames.push_back(std::move(animationName));
    }
    // Animation-only FBX files are first-class sources: their skeleton and
    // tracks can drive a compatible render model even when they have no mesh.
    if (settings.rigType != "none" && SkinnedModelImporter::HasSkinningData(*scene, settings.importAnimations))
        result.skinnedMesh =
            SkinnedModelImporter::ConvertScene(*scene, guid, filePath, settings.scaleFactor, settings.importAnimations,
                                              settings.maxBonesPerVertex, settings.minBoneWeight);
    if (result.skinnedMesh) {
        SkinnedModelImporter::ApplyAnimationClips(*result.skinnedMesh, settings);
        if (!result.skinnedMesh->IsAssetPayloadValid())
            result.skinnedMesh.reset();
    }
    return result;
}

std::shared_ptr<InxMesh> MeshLoader::ImportSource(const std::string &filePath, const std::string &guid,
                                                  const InxResourceMeta &metadata)
{
    auto imported = ImportSourceDetailed(filePath, guid, metadata);
    imported.mesh->SetSkinnedData(std::move(imported.skinnedMesh));
    return imported.mesh;
}

// =============================================================================
// CreateMeta — mesh/binary asset .meta creation
// =============================================================================

void MeshLoader::CreateMeta(const char *content, size_t contentSize, const std::string &filePath,
                            InxResourceMeta &metaData) const
{
    metaData.Init(content, contentSize, filePath, ResourceType::Mesh);

    std::filesystem::path path = ToFsPath(filePath);
    std::string extension = FromFsPath(path.extension());

    metaData.AddMetadata("file_type", std::string("mesh"));
    metaData.AddMetadata("file_extension", extension);
    metaData.AddMetadata("is_readable", false);

    metaData.AddMetadata("file_size", contentSize);
}

} // namespace infernux
