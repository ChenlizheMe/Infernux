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

#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <platform/filesystem/InxPath.h>
#include <unordered_map>
#include <unordered_set>

namespace infernux
{

namespace
{
glm::vec4 FallbackTangent(const glm::vec3 &normal)
{
    const float normalLengthSquared = glm::dot(normal, normal);
    if (normalLengthSquared <= kEpsilon * kEpsilon)
        return glm::vec4(0.0f);
    const glm::vec3 unitNormal = normal / std::sqrt(normalLengthSquared);
    const glm::vec3 reference =
        std::abs(unitNormal.y) < 0.999f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    return glm::vec4(glm::normalize(glm::cross(reference, unitNormal)), 1.0f);
}

MaterialSamplerAddress ConvertAddressMode(aiTextureMapMode mode)
{
    switch (mode) {
    case aiTextureMapMode_Wrap:
        return MaterialSamplerAddress::Repeat;
    case aiTextureMapMode_Clamp:
        return MaterialSamplerAddress::Clamp;
    case aiTextureMapMode_Mirror:
        return MaterialSamplerAddress::Mirror;
    default:
        return MaterialSamplerAddress::Inherit;
    }
}

void ApplyGltfFilter(int value, MaterialTextureSampler &sampler, bool magnification)
{
    if (magnification) {
        if (value == 9728)
            sampler.magFilter = MaterialSamplerFilter::Nearest;
        else if (value == 9729)
            sampler.magFilter = MaterialSamplerFilter::Linear;
        else
            throw std::invalid_argument("unsupported glTF magnification filter: " + std::to_string(value));
        return;
    }
    switch (value) {
    case 9728:
        sampler.minFilter = MaterialSamplerFilter::Nearest;
        break;
    case 9729:
        sampler.minFilter = MaterialSamplerFilter::Linear;
        break;
    case 9984:
        sampler.minFilter = MaterialSamplerFilter::Nearest;
        sampler.mipFilter = MaterialSamplerFilter::Nearest;
        break;
    case 9985:
        sampler.minFilter = MaterialSamplerFilter::Linear;
        sampler.mipFilter = MaterialSamplerFilter::Nearest;
        break;
    case 9986:
        sampler.minFilter = MaterialSamplerFilter::Nearest;
        sampler.mipFilter = MaterialSamplerFilter::Linear;
        break;
    case 9987:
        sampler.minFilter = MaterialSamplerFilter::Linear;
        sampler.mipFilter = MaterialSamplerFilter::Linear;
        break;
    default:
        throw std::invalid_argument("unsupported glTF minification filter: " + std::to_string(value));
    }
}
} // namespace

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
        if (settings.normalMode == "none" ||
            (settings.normalMode == "calculate" && settings.normalSmoothingSource != "source")) {
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
    uint32_t meshIndex;          ///< Index into aiScene::mMeshes
    uint32_t nodeGroup;          ///< Source node group (for per-object splitting)
    glm::mat4 sourceBasis{1.0f}; ///< Root basis baked into root-owned geometry only
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

static float ResolveSourceUnitScale(const aiScene &scene, std::string_view extension,
                                    const MeshImportSettings &settings)
{
    if (!settings.convertUnits || extension != "fbx" || !scene.mMetaData)
        return 1.0f;
    double centimeters = 0.0;
    bool found = false;
    for (unsigned int index = 0; index < scene.mMetaData->mNumProperties; ++index) {
        if (std::string_view(scene.mMetaData->mKeys[index].C_Str()) != "UnitScaleFactor")
            continue;
        const auto &entry = scene.mMetaData->mValues[index];
        if (entry.mType == AI_FLOAT) {
            centimeters = *static_cast<const float *>(entry.mData);
            found = true;
        } else if (entry.mType == AI_DOUBLE) {
            centimeters = *static_cast<const double *>(entry.mData);
            found = true;
        }
        break;
    }
    if (!found)
        throw std::runtime_error("FBX model does not declare UnitScaleFactor");
    if (!std::isfinite(centimeters) || centimeters <= 0.0)
        throw std::runtime_error("FBX model declares an invalid UnitScaleFactor");
    const double meters = centimeters / 100.0;
    if (!std::isfinite(meters) || meters <= 0.0 || meters > std::numeric_limits<float>::max())
        throw std::runtime_error("FBX model unit conversion is outside the supported range");
    return static_cast<float>(meters);
}

static bool ReadNodeVisibility(const aiNode &node)
{
    if (!node.mMetaData)
        return true;
    for (unsigned int index = 0; index < node.mMetaData->mNumProperties; ++index) {
        const std::string_view key = node.mMetaData->mKeys[index].C_Str();
        if (key != "infernux_source_visible" && key != "Visibility" && key != "visibility")
            continue;
        const aiMetadataEntry &entry = node.mMetaData->mValues[index];
        switch (entry.mType) {
        case AI_BOOL:
            return *static_cast<const bool *>(entry.mData);
        case AI_INT32:
            return *static_cast<const int32_t *>(entry.mData) != 0;
        case AI_UINT32:
            return *static_cast<const uint32_t *>(entry.mData) != 0;
        case AI_INT64:
            return *static_cast<const int64_t *>(entry.mData) != 0;
        case AI_UINT64:
            return *static_cast<const uint64_t *>(entry.mData) != 0;
        case AI_FLOAT:
            return *static_cast<const float *>(entry.mData) != 0.0f;
        case AI_DOUBLE:
            return *static_cast<const double *>(entry.mData) != 0.0;
        default:
            return true;
        }
    }
    return true;
}

static std::unordered_set<const aiNode *> NonModelNodes(const aiScene &scene)
{
    std::unordered_set<const aiNode *> nodes;
    nodes.reserve(scene.mNumCameras + scene.mNumLights);
    for (unsigned int index = 0; index < scene.mNumCameras; ++index)
        if (scene.mCameras[index]) {
            if (const auto *node = scene.mRootNode->FindNode(scene.mCameras[index]->mName))
                nodes.insert(node);
        }
    for (unsigned int index = 0; index < scene.mNumLights; ++index)
        if (scene.mLights[index]) {
            if (const auto *node = scene.mRootNode->FindNode(scene.mLights[index]->mName))
                nodes.insert(node);
        }
    return nodes;
}

static void CollectMeshes(const aiNode *node, std::vector<CollectedMesh> &outMeshes,
                          std::vector<std::string> &outNodeNames, std::vector<ImportedModelNode> &outNodes,
                          int32_t parentIndex, float scale, const MeshImportSettings &settings,
                          const std::unordered_set<const aiNode *> &nonModelNodes,
                          const glm::mat4 &inheritedTransform = glm::mat4(1.0f), bool inheritedVisible = true,
                          bool sourceRoot = true)
{
    const glm::mat4 authoredTransform = AiToGlm(node->mTransformation);
    const bool bakeRootBasis = sourceRoot && settings.bakeAxisConversion;
    const glm::mat4 localTransform = bakeRootBasis ? glm::mat4(1.0f) : inheritedTransform * authoredTransform;
    const bool visible = inheritedVisible && (!settings.importVisibility || ReadNodeVisibility(*node));
    // Cameras and lights are scene-authoring objects, not model nodes in the
    // current Infernux product boundary. If a DCC placed model descendants
    // below one, retain their world-equivalent local transform while removing
    // only the unsupported scene node.
    if (node->mNumMeshes == 0 && nonModelNodes.find(node) != nonModelNodes.end()) {
        for (unsigned int index = 0; index < node->mNumChildren; ++index)
            CollectMeshes(node->mChildren[index], outMeshes, outNodeNames, outNodes, parentIndex, scale, settings,
                          nonModelNodes, localTransform, visible, false);
        return;
    }
    const int32_t nodeIndex = static_cast<int32_t>(outNodes.size());
    ImportedModelNode importedNode;
    importedNode.name = node->mName.C_Str();
    importedNode.parentIndex = parentIndex;
    importedNode.localTransform = localTransform;
    importedNode.visible = visible;
    // Unit conversion is applied to translations once, not as a scale at
    // every ancestor. Geometry below remains in the existing model space.
    importedNode.localTransform[3] = glm::vec4(glm::vec3(localTransform[3]) * scale, 1.0f);

    if (node->mNumMeshes > 0) {
        uint32_t group = static_cast<uint32_t>(outNodeNames.size());
        importedNode.nodeGroup = static_cast<int32_t>(group);
        outNodeNames.push_back(node->mName.C_Str());
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            outMeshes.push_back({node->mMeshes[i], group, bakeRootBasis ? authoredTransform : glm::mat4(1.0f)});
        }
    }

    outNodes.push_back(std::move(importedNode));

    std::vector<const aiNode *> children;
    children.reserve(node->mNumChildren);
    for (unsigned int index = 0; index < node->mNumChildren; ++index)
        children.push_back(node->mChildren[index]);
    if (settings.sortHierarchyByName)
        std::stable_sort(children.begin(), children.end(), [](const aiNode *left, const aiNode *right) {
            return std::string_view(left->mName.C_Str()) < std::string_view(right->mName.C_Str());
        });
    for (const aiNode *child : children)
        CollectMeshes(child, outMeshes, outNodeNames, outNodes, nodeIndex, scale, settings, nonModelNodes,
                      bakeRootBasis ? authoredTransform : glm::mat4(1.0f), visible, false);
}

// ============================================================================
// Core conversion: aiScene → InxMesh
// ============================================================================

static std::shared_ptr<InxMesh>
ConvertScene(const aiScene *scene, const MeshImportSettings &settings, const std::string &name,
             std::vector<MeshSourceImportResult::TextureSource> &textureSources,
             std::vector<MeshSourceImportResult::MaterialDiagnostic> &materialDiagnostics)
{
    auto mesh = std::make_shared<InxMesh>(name);

    // Collect all mesh instances with their transforms and node grouping
    std::vector<CollectedMesh> collectedMeshes;
    std::vector<std::string> nodeNames;
    std::vector<ImportedModelNode> modelNodes;
    collectedMeshes.reserve(scene->mNumMeshes);
    const auto nonModelNodes = NonModelNodes(*scene);
    CollectMeshes(scene->mRootNode, collectedMeshes, nodeNames, modelNodes, -1, settings.scaleFactor, settings,
                  nonModelNodes);

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
        if (!(aiM->mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
            continue;
        totalVertices += aiM->mNumVertices;
        for (unsigned int f = 0; f < aiM->mNumFaces; ++f)
            totalIndices += aiM->mFaces[f].mNumIndices;
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SubMesh> subMeshes;
    std::vector<MeshMorphTarget> morphTargets;
    std::unordered_map<std::string, size_t> morphTargetByName;
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
        const glm::mat3 sourceLinear(cm.sourceBasis);
        const float sourceOrientation = glm::determinant(sourceLinear) < 0.0f ? -1.0f : 1.0f;
        const glm::mat3 sourceNormals(glm::cross(sourceLinear[1], sourceLinear[2]) * sourceOrientation,
                                      glm::cross(sourceLinear[2], sourceLinear[0]) * sourceOrientation,
                                      glm::cross(sourceLinear[0], sourceLinear[1]) * sourceOrientation);
        const auto normalizeOrZero = [](const glm::vec3 &value) {
            const float lengthSquared = glm::dot(value, value);
            return std::isfinite(lengthSquared) && lengthSquared > 1.e-12f ? value * glm::inversesqrt(lengthSquared)
                                                                           : glm::vec3(0.0f);
        };

        // ── Vertices ────────────────────────────────────────────────
        for (unsigned int v = 0; v < aiM->mNumVertices; ++v) {
            Vertex vert{};

            // Retain authored local geometry. Unit conversion is shared with
            // node translations, never baked a second time at each parent.
            vert.pos = glm::vec3(cm.sourceBasis *
                                 glm::vec4(aiM->mVertices[v].x, aiM->mVertices[v].y, aiM->mVertices[v].z, 1.0f));
            if (applyScale)
                vert.pos *= scale;

            // Normal
            if (hasNormals) {
                glm::vec3 n(aiM->mNormals[v].x, aiM->mNormals[v].y, aiM->mNormals[v].z);
                vert.normal = normalizeOrZero(sourceNormals * n);
            } else
                vert.normal = glm::vec3(0.0f);

            // Tangent + bitangent handedness
            if (hasTangents) {
                glm::vec3 t(aiM->mTangents[v].x, aiM->mTangents[v].y, aiM->mTangents[v].z);

                glm::vec3 b(aiM->mBitangents[v].x, aiM->mBitangents[v].y, aiM->mBitangents[v].z);

                // Compute handedness: sign of dot(cross(N,T), B)
                const glm::vec3 sourceNormal(aiM->mNormals[v].x, aiM->mNormals[v].y, aiM->mNormals[v].z);
                float handedness = (glm::dot(glm::cross(sourceNormal, t), b) < 0.0f) ? -1.0f : 1.0f;
                vert.tangent = glm::vec4(normalizeOrZero(sourceLinear * t), handedness * sourceOrientation);
            } else if (settings.tangentMode != "none" && settings.tangentMode != "source_only" &&
                       settings.normalMode != "none") {
                // A source can legitimately omit UVs and tangents while still
                // using a Lit material with the built-in flat normal texture.
                // A zero tangent makes the shader's TBN basis undefined and
                // turns otherwise valid geometry black.  Publish a stable
                // orthogonal frame; authored/calculated tangents still win.
                vert.tangent = FallbackTangent(vert.normal);
            } else {
                vert.tangent = glm::vec4(0.0f);
            }

            // Preserve the authored primary and secondary channels. UV1 is
            // the runtime lightmap-coordinate contract; missing data remains
            // explicit zero rather than being guessed from positions.
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
            if (aiM->HasTextureCoords(1))
                vert.texCoord1 = glm::vec2(aiM->mTextureCoords[1][v].x, aiM->mTextureCoords[1][v].y);
            if (!std::isfinite(vert.texCoord.x) || !std::isfinite(vert.texCoord.y) ||
                !std::isfinite(vert.texCoord1.x) || !std::isfinite(vert.texCoord1.y))
                throw std::runtime_error("model contains non-finite UV coordinates");

            // Vertex colour
            if (hasColors) {
                vert.color = glm::vec3(aiM->mColors[0][v].r, aiM->mColors[0][v].g, aiM->mColors[0][v].b);
            } else {
                vert.color = glm::vec3(1.0f);
            }

            vertices.push_back(vert);
        }

        if (settings.importBlendShapes) {
            std::unordered_set<std::string> meshTargetNames;
            for (unsigned int morphIndex = 0; morphIndex < aiM->mNumAnimMeshes; ++morphIndex) {
                const aiAnimMesh *sourceTarget = aiM->mAnimMeshes[morphIndex];
                if (!sourceTarget || sourceTarget->mNumVertices != aiM->mNumVertices)
                    throw std::runtime_error("model blend shape has an invalid vertex domain");
                if (!std::isfinite(sourceTarget->mWeight))
                    throw std::runtime_error("model blend shape has a non-finite default weight");
                std::string targetName = sourceTarget->mName.C_Str();
                if (targetName.empty())
                    targetName = nodeNames.at(cm.nodeGroup) + "/" + std::string(aiM->mName.C_Str()) + "/Morph_" +
                                 std::to_string(morphIndex);
                if (!meshTargetNames.insert(targetName).second)
                    throw std::runtime_error("model mesh contains duplicate blend shape names: " + targetName);
                auto [found, inserted] = morphTargetByName.emplace(targetName, morphTargets.size());
                if (inserted) {
                    MeshMorphTarget target;
                    target.name = targetName;
                    target.defaultWeight = static_cast<float>(sourceTarget->mWeight);
                    target.positionDeltas.resize(totalVertices, glm::vec3(0.0f));
                    morphTargets.push_back(std::move(target));
                } else if (std::abs(morphTargets[found->second].defaultWeight -
                                    static_cast<float>(sourceTarget->mWeight)) > kEpsilon) {
                    throw std::runtime_error("model blend shape parts disagree on the default weight: " + targetName);
                }
                auto &target = morphTargets[found->second];
                if (sourceTarget->mNormals && target.normalDeltas.empty())
                    target.normalDeltas.resize(totalVertices, glm::vec3(0.0f));
                if (sourceTarget->mTangents && target.tangentDeltas.empty())
                    target.tangentDeltas.resize(totalVertices, glm::vec3(0.0f));
                for (unsigned int vertexIndex = 0; vertexIndex < aiM->mNumVertices; ++vertexIndex) {
                    const size_t outputIndex = static_cast<size_t>(currentVertexOffset) + vertexIndex;
                    if (sourceTarget->mVertices) {
                        const glm::vec3 targetPosition(sourceTarget->mVertices[vertexIndex].x,
                                                       sourceTarget->mVertices[vertexIndex].y,
                                                       sourceTarget->mVertices[vertexIndex].z);
                        const glm::vec3 basePosition(aiM->mVertices[vertexIndex].x, aiM->mVertices[vertexIndex].y,
                                                     aiM->mVertices[vertexIndex].z);
                        target.positionDeltas[outputIndex] = sourceLinear * (targetPosition - basePosition) * scale;
                    }
                    if (sourceTarget->mNormals) {
                        const glm::vec3 targetNormal(sourceTarget->mNormals[vertexIndex].x,
                                                     sourceTarget->mNormals[vertexIndex].y,
                                                     sourceTarget->mNormals[vertexIndex].z);
                        const glm::vec3 baseNormal =
                            aiM->mNormals ? glm::vec3(aiM->mNormals[vertexIndex].x, aiM->mNormals[vertexIndex].y,
                                                      aiM->mNormals[vertexIndex].z)
                                          : glm::vec3(0.0f);
                        target.normalDeltas[outputIndex] =
                            normalizeOrZero(sourceNormals * targetNormal) - normalizeOrZero(sourceNormals * baseNormal);
                    }
                    if (sourceTarget->mTangents) {
                        const glm::vec3 targetTangent(sourceTarget->mTangents[vertexIndex].x,
                                                      sourceTarget->mTangents[vertexIndex].y,
                                                      sourceTarget->mTangents[vertexIndex].z);
                        const glm::vec3 baseTangent =
                            aiM->mTangents ? glm::vec3(aiM->mTangents[vertexIndex].x, aiM->mTangents[vertexIndex].y,
                                                       aiM->mTangents[vertexIndex].z)
                                           : glm::vec3(0.0f);
                        target.tangentDeltas[outputIndex] =
                            normalizeOrZero(sourceLinear * targetTangent) - normalizeOrZero(sourceLinear * baseTangent);
                    }
                }
            }
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
                    const auto reportMaterialProperty = [&](std::string code, std::string property,
                                                            std::string detail) {
                        materialDiagnostics.push_back({std::move(code),
                                                       matName.empty() ? "Material_" + std::to_string(slot) : matName,
                                                       std::move(property), std::move(detail)});
                    };
                    const auto readTexture = [&](aiTextureType semantic, ModelTexture channel) {
                        if (!aiMat->GetTextureCount(semantic))
                            return;
                        aiString texturePath;
                        aiTextureMapping mapping = aiTextureMapping_UV;
                        unsigned int uvChannel = 0;
                        ai_real blend = 1.0f;
                        aiTextureOp operation = aiTextureOp_Multiply;
                        aiTextureMapMode addressModes[3] = {aiTextureMapMode_Wrap, aiTextureMapMode_Wrap,
                                                            aiTextureMapMode_Wrap};
                        if (aiMat->GetTexture(semantic, 0, &texturePath, &mapping, &uvChannel, &blend, &operation,
                                              addressModes) != AI_SUCCESS)
                            throw std::runtime_error("model material texture could not be read");
                        const std::string property = std::string("texture/") + aiTextureTypeToString(semantic);
                        if (mapping != aiTextureMapping_UV) {
                            reportMaterialProperty("unsupported_texture_mapping", property,
                                                   std::to_string(static_cast<int>(mapping)));
                            return;
                        }
                        if (uvChannel > 1) {
                            reportMaterialProperty("unsupported_uv_set", property, std::to_string(uvChannel));
                            return;
                        }
                        if (std::abs(blend - 1.0f) > kEpsilon)
                            reportMaterialProperty("unsupported_texture_blend", property, std::to_string(blend));
                        if (operation != aiTextureOp_Multiply)
                            reportMaterialProperty("unsupported_texture_operation", property,
                                                   std::to_string(static_cast<int>(operation)));
                        if (addressModes[0] == aiTextureMapMode_Decal || addressModes[1] == aiTextureMapMode_Decal ||
                            addressModes[2] == aiTextureMapMode_Decal)
                            reportMaterialProperty("unsupported_sampler_address", property,
                                                   std::to_string(static_cast<int>(addressModes[0])) + "," +
                                                       std::to_string(static_cast<int>(addressModes[1])) + "," +
                                                       std::to_string(static_cast<int>(addressModes[2])));
                        aiUVTransform transform;
                        if (aiMat->Get(AI_MATKEY_UVTRANSFORM(semantic, 0), transform) == AI_SUCCESS &&
                            (glm::length(glm::vec2(transform.mTranslation.x, transform.mTranslation.y)) > kEpsilon ||
                             glm::length(glm::vec2(transform.mScaling.x - 1.0f, transform.mScaling.y - 1.0f)) >
                                 kEpsilon ||
                             std::abs(transform.mRotation) > kEpsilon))
                            reportMaterialProperty("unsupported_uv_transform", property, "source transform");
                        for (unsigned int layer = 1; layer < aiMat->GetTextureCount(semantic); ++layer)
                            reportMaterialProperty("unsupported_texture_layer", property, std::to_string(layer));
                        if (texturePath.length) {
                            MaterialTextureSampler sampler;
                            sampler.addressU = ConvertAddressMode(addressModes[0]);
                            sampler.addressV = ConvertAddressMode(addressModes[1]);
                            sampler.addressW = ConvertAddressMode(addressModes[2]);
                            int filter = 0;
                            if (aiMat->Get(AI_MATKEY_GLTF_MAPPINGFILTER_MAG(semantic, 0), filter) == AI_SUCCESS)
                                ApplyGltfFilter(filter, sampler, true);
                            if (aiMat->Get(AI_MATKEY_GLTF_MAPPINGFILTER_MIN(semantic, 0), filter) == AI_SUCCESS)
                                ApplyGltfFilter(filter, sampler, false);
                            int32_t embeddedIndex = -1;
                            if (const auto *image = scene->GetEmbeddedTexture(texturePath.C_Str())) {
                                for (unsigned int index = 0; index < scene->mNumTextures; ++index)
                                    if (scene->mTextures[index] == image) {
                                        embeddedIndex = static_cast<int32_t>(index);
                                        break;
                                    }
                            }
                            textureSources.push_back({slot, texturePath.C_Str(), static_cast<uint32_t>(channel),
                                                      embeddedIndex, static_cast<uint8_t>(uvChannel), sampler});
                        }
                    };
                    readTexture(aiMat->GetTextureCount(aiTextureType_BASE_COLOR) ? aiTextureType_BASE_COLOR
                                                                                 : aiTextureType_DIFFUSE,
                                ModelTexture::BaseColor);
                    readTexture(aiMat->GetTextureCount(aiTextureType_NORMALS) ? aiTextureType_NORMALS
                                                                              : aiTextureType_NORMAL_CAMERA,
                                ModelTexture::Normal);
                    readTexture(aiMat->GetTextureCount(aiTextureType_METALNESS) ? aiTextureType_METALNESS
                                                                                : aiTextureType_GLTF_METALLIC_ROUGHNESS,
                                ModelTexture::Metallic);
                    readTexture(aiMat->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS)
                                    ? aiTextureType_DIFFUSE_ROUGHNESS
                                    : aiTextureType_GLTF_METALLIC_ROUGHNESS,
                                ModelTexture::Roughness);
                    readTexture(aiMat->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION)
                                    ? aiTextureType_AMBIENT_OCCLUSION
                                    : aiTextureType_LIGHTMAP,
                                ModelTexture::Occlusion);
                    readTexture(aiMat->GetTextureCount(aiTextureType_EMISSION_COLOR) ? aiTextureType_EMISSION_COLOR
                                                                                     : aiTextureType_EMISSIVE,
                                ModelTexture::Emission);
                    const std::unordered_set<unsigned int> mappedTextureTypes = {aiTextureType_DIFFUSE,
                                                                                 aiTextureType_EMISSIVE,
                                                                                 aiTextureType_LIGHTMAP,
                                                                                 aiTextureType_BASE_COLOR,
                                                                                 aiTextureType_NORMALS,
                                                                                 aiTextureType_NORMAL_CAMERA,
                                                                                 aiTextureType_EMISSION_COLOR,
                                                                                 aiTextureType_METALNESS,
                                                                                 aiTextureType_DIFFUSE_ROUGHNESS,
                                                                                 aiTextureType_AMBIENT_OCCLUSION,
                                                                                 aiTextureType_GLTF_METALLIC_ROUGHNESS};
                    for (unsigned int textureType = 1; textureType <= AI_TEXTURE_TYPE_MAX; ++textureType) {
                        if (mappedTextureTypes.find(textureType) != mappedTextureTypes.end())
                            continue;
                        const auto semantic = static_cast<aiTextureType>(textureType);
                        for (unsigned int layer = 0; layer < aiMat->GetTextureCount(semantic); ++layer)
                            reportMaterialProperty("unsupported_texture_semantic",
                                                   std::string("texture/") + aiTextureTypeToString(semantic),
                                                   std::to_string(layer));
                    }
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

        for (const auto &texture : textureSources) {
            // UV0 has a defined zero-filled representation in Vertex and has
            // historically been legal on unwrapped meshes. Secondary UVs are
            // opt-in authored data and must exist when a material selects one.
            if (texture.materialSlot == slot && texture.uvSet != 0 && !aiM->HasTextureCoords(texture.uvSet))
                throw std::invalid_argument("model material selects UV" + std::to_string(texture.uvSet) +
                                            " but mesh '" + std::string(aiM->mName.C_Str()) +
                                            "' does not provide that channel");
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
        for (const auto &remap : settings.materialRemaps.items()) {
            const auto &sourceId = remap.key();
            const auto &guid = remap.value();
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
    mesh->SetModelData(std::move(vertices), std::move(indices), std::move(subMeshes), std::move(modelNodes),
                       std::move(morphTargets));

    return mesh;
}

// ============================================================================
// IAssetLoader interface
// ============================================================================

MeshSourceImportResult MeshLoader::ImportSourceDetailed(const std::string &filePath, const std::string &guid,
                                                        const InxResourceMeta &metadata,
                                                        const InxSkinnedMesh *copiedDefinition)
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
    const MeshCompression compression = settings.meshCompression == "low"      ? MeshCompression::Low
                                        : settings.meshCompression == "medium" ? MeshCompression::Medium
                                        : settings.meshCompression == "high"   ? MeshCompression::High
                                                                               : MeshCompression::Off;
    const MeshIndexFormat indexFormat = settings.indexFormat == "uint16"   ? MeshIndexFormat::UInt16
                                        : settings.indexFormat == "uint32" ? MeshIndexFormat::UInt32
                                                                           : MeshIndexFormat::Auto;
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
        (void)ResolveMeshIndexFormat(indexFormat, result.mesh->GetVertexCount(), result.mesh->GetIndices());
        result.mesh->SetIndexFormat(indexFormat);
        result.mesh->SetCompression(compression);
        result.mesh->SetCpuReadable(settings.isReadable);
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

    const float sourceUnitScale = ResolveSourceUnitScale(*scene, ext, settings);
    MeshImportSettings conversionSettings = settings;
    conversionSettings.scaleFactor *= sourceUnitScale;
    if (!std::isfinite(conversionSettings.scaleFactor) || conversionSettings.scaleFactor <= 0.0f)
        throw std::runtime_error("model effective import scale is outside the supported range");

    std::string name = FromFsPath(fsPath.stem());
    MeshSourceImportResult result;
    result.sourceUnitScale = sourceUnitScale;
    result.effectiveScale = conversionSettings.scaleFactor;
    auto mesh = ConvertScene(scene, conversionSettings, name, result.textureSources, result.materialDiagnostics);
    mesh->SetCompression(compression);
    mesh->SetCpuReadable(settings.isReadable);
    // Validate explicit UInt16 atomically at import publication. Auto remains
    // a policy until Cook/GPU publication resolves it from this geometry.
    (void)ResolveMeshIndexFormat(indexFormat, mesh->GetVertexCount(), mesh->GetIndices());
    mesh->SetIndexFormat(indexFormat);
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
        output.key = !output.name.empty() && imageNameCounts.at(output.name) == 1 ? "name/" + output.name
                                                                                  : "index/" + std::to_string(index);
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
                                ? animation->mTicksPerSecond
                                : 25.0;
        result.sourceAnimations.push_back(
            {{"name", animationName},
             {"duration", std::isfinite(animation->mDuration) ? std::max(0.0, animation->mDuration) / rate : 0.0},
             {"sample_rate", rate}});
        result.animationNames.push_back(std::move(animationName));
    }
    // Animation-only FBX files are first-class sources: their skeleton and
    // tracks can drive a compatible render model even when they have no mesh.
    if (settings.rigType != "none" && SkinnedModelImporter::HasSkinningData(*scene, settings.importAnimations))
        result.skinnedMesh = SkinnedModelImporter::ConvertScene(
            *scene, guid, filePath, conversionSettings.scaleFactor, settings.importAnimations,
            settings.maxBonesPerVertex, settings.minBoneWeight,
            settings.tangentMode != "none" && settings.normalMode != "none" &&
                !(settings.normalMode == "calculate" && settings.tangentMode == "source_only"),
            settings.importBlendShapes, settings.bakeAxisConversion);
    if (result.skinnedMesh) {
        result.skinnedMesh->compression = compression;
        (void)ResolveMeshIndexFormat(indexFormat, result.skinnedMesh->baseVertices.size(), result.skinnedMesh->indices);
        result.skinnedMesh->indexFormat = indexFormat;
        SkinnedModelImporter::ApplyAnimationClips(*result.skinnedMesh, settings);
        SkinnedModelImporter::ApplyRigSettings(*result.skinnedMesh, settings, copiedDefinition);
        SkinnedModelImporter::ApplyAnimationSettings(*result.skinnedMesh, settings);
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
