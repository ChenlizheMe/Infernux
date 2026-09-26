#include "ModelVertexBasis.h"
#include "MeshImportSettings.h"

#include <assimp/SpatialSort.h>
#include <assimp/scene.h>
#include <mikktspace.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace infernux
{
namespace
{
template <typename T> void RemapArray(T *&data, const std::vector<unsigned int> &source)
{
    if (!data)
        return;
    auto remapped = std::make_unique<T[]>(source.size());
    for (size_t i = 0; i < source.size(); ++i)
        remapped[i] = data[source[i]];
    delete[] data;
    data = remapped.release();
}

template <typename Mesh> void RemapVertexChannels(Mesh &mesh, const std::vector<unsigned int> &source)
{
    RemapArray(mesh.mVertices, source);
    RemapArray(mesh.mNormals, source);
    RemapArray(mesh.mTangents, source);
    RemapArray(mesh.mBitangents, source);
    for (auto &channel : mesh.mTextureCoords)
        RemapArray(channel, source);
    for (auto &channel : mesh.mColors)
        RemapArray(channel, source);
    mesh.mNumVertices = static_cast<unsigned int>(source.size());
}

void ExpandCorners(aiMesh &mesh)
{
    size_t count = 0;
    for (unsigned int face = 0; face < mesh.mNumFaces; ++face)
        count += mesh.mFaces[face].mNumIndices;
    if (count > std::numeric_limits<unsigned int>::max())
        throw std::runtime_error("Model basis corner count exceeds the mesh index range");
    std::vector<unsigned int> source;
    source.reserve(count);
    for (unsigned int face = 0; face < mesh.mNumFaces; ++face) {
        const auto &indices = mesh.mFaces[face];
        source.insert(source.end(), indices.mIndices, indices.mIndices + indices.mNumIndices);
    }
    // Preserve every skin weight on every split copy, without scanning the full
    // vertex buffer for each bone. Morph/UV/color channels use the same mapping.
    const auto end = std::numeric_limits<unsigned int>::max();
    std::vector<unsigned int> heads(mesh.mNumVertices, end), next(source.size(), end);
    for (unsigned int i = 0; i < source.size(); ++i) {
        next[i] = heads[source[i]];
        heads[source[i]] = i;
    }
    for (unsigned int bone = 0; bone < mesh.mNumBones; ++bone) {
        auto &binding = *mesh.mBones[bone];
        std::vector<aiVertexWeight> weights;
        for (unsigned int i = 0; i < binding.mNumWeights; ++i) {
            const auto &weight = binding.mWeights[i];
            for (auto copy = heads[weight.mVertexId]; copy != end; copy = next[copy])
                weights.emplace_back(copy, weight.mWeight);
        }
        auto remapped = std::make_unique<aiVertexWeight[]>(weights.size());
        std::copy(weights.begin(), weights.end(), remapped.get());
        delete[] binding.mWeights;
        binding.mWeights = remapped.release();
        binding.mNumWeights = static_cast<unsigned int>(weights.size());
    }
    RemapVertexChannels(mesh, source);
    for (unsigned int morph = 0; morph < mesh.mNumAnimMeshes; ++morph)
        RemapVertexChannels(*mesh.mAnimMeshes[morph], source);
    unsigned int corner = 0;
    for (unsigned int face = 0; face < mesh.mNumFaces; ++face)
        for (unsigned int i = 0; i < mesh.mFaces[face].mNumIndices; ++i)
            mesh.mFaces[face].mIndices[i] = corner++;
}

void CalculateNormals(aiMesh &mesh, const MeshImportSettings &settings)
{
    std::vector<aiVector3D> sourceNormals;
    if (settings.normalSmoothingSource == "source") {
        if (!mesh.HasNormals())
            throw std::runtime_error("source smoothing requires authored source normals");
        sourceNormals.assign(mesh.mNormals, mesh.mNormals + mesh.mNumVertices);
        for (auto &normal : sourceNormals) {
            if (!std::isfinite(normal.x) || !std::isfinite(normal.y) || !std::isfinite(normal.z) ||
                normal.SquareLength() <= std::numeric_limits<ai_real>::epsilon())
                throw std::runtime_error("source smoothing requires finite non-zero authored normals");
            normal.Normalize();
        }
    }
    std::vector<aiVector3D> directions(mesh.mNumVertices);
    std::vector<double> weights(mesh.mNumVertices, 0.0);
    const bool area = settings.normalWeighting == "area" || settings.normalWeighting == "area_angle";
    const bool angle = settings.normalWeighting == "angle" || settings.normalWeighting == "area_angle";
    for (unsigned int faceIndex = 0; faceIndex < mesh.mNumFaces; ++faceIndex) {
        const auto &face = mesh.mFaces[faceIndex];
        if (face.mNumIndices != 3)
            continue;
        const auto &a = mesh.mVertices[face.mIndices[0]];
        const auto &b = mesh.mVertices[face.mIndices[1]];
        const auto &c = mesh.mVertices[face.mIndices[2]];
        auto cross = (b - a) ^ (c - a);
        const double twiceArea = cross.Length();
        if (twiceArea == 0.0)
            continue;
        const auto direction = cross / static_cast<ai_real>(twiceArea);
        for (unsigned int corner = 0; corner < 3; ++corner) {
            const auto index = face.mIndices[corner];
            double weight = area ? twiceArea : 1.0;
            if (angle) {
                const auto first = mesh.mVertices[face.mIndices[(corner + 1) % 3]] - mesh.mVertices[index];
                const auto second = mesh.mVertices[face.mIndices[(corner + 2) % 3]] - mesh.mVertices[index];
                weight *=
                    std::atan2(static_cast<double>((first ^ second).Length()), static_cast<double>(first * second));
            }
            directions[index] = direction;
            weights[index] = weight;
        }
    }
    aiVector3D minimum = mesh.mVertices[0], maximum = minimum;
    for (unsigned int i = 1; i < mesh.mNumVertices; ++i) {
        const auto &v = mesh.mVertices[i];
        minimum = {std::min(minimum.x, v.x), std::min(minimum.y, v.y), std::min(minimum.z, v.z)};
        maximum = {std::max(maximum.x, v.x), std::max(maximum.y, v.y), std::max(maximum.z, v.z)};
    }
    // Same source-position tolerance as the previous importer, including UV
    // seams. Do not merge by UV or material values when calculating normals.
    const auto epsilon = (maximum - minimum).Length() * ai_real(1.e-4);
    Assimp::SpatialSort spatial(mesh.mVertices, mesh.mNumVertices, sizeof(aiVector3D));
    auto normals = std::make_unique<aiVector3D[]>(mesh.mNumVertices);
    const bool smoothAll = settings.normalSmoothingAngle >= 175.0f;
    const auto limit = std::cos(settings.normalSmoothingAngle * 0.017453292519943295);
    std::vector<unsigned int> neighbors;
    std::vector<bool> assigned(mesh.mNumVertices, false);
    for (unsigned int i = 0; i < mesh.mNumVertices; ++i) {
        if (smoothAll && assigned[i])
            continue;
        if (epsilon > 0)
            spatial.FindPositions(mesh.mVertices[i], epsilon, neighbors);
        else
            spatial.FindIdenticalPositions(mesh.mVertices[i], neighbors);
        aiVector3D sum;
        for (const auto other : neighbors) {
            const bool sameSourceGroup = sourceNormals.empty() || sourceNormals[i] * sourceNormals[other] >= 0.99999f;
            const bool withinAngle = smoothAll || other == i || directions[i] * directions[other] >= limit;
            if (sameSourceGroup && (settings.normalSmoothingSource == "source" || withinAngle))
                sum += directions[other] * static_cast<ai_real>(weights[other]);
        }
        sum.NormalizeSafe();
        normals[i] = sum;
        if (smoothAll && settings.normalSmoothingSource != "source")
            for (const auto other : neighbors) {
                normals[other] = sum;
                assigned[other] = true;
            }
    }
    delete[] mesh.mNormals;
    mesh.mNormals = normals.release();
}

struct MikkMesh
{
    aiMesh &mesh;
    std::vector<unsigned int> triangles;
    static MikkMesh &Get(const SMikkTSpaceContext *context)
    {
        return *static_cast<MikkMesh *>(context->m_pUserData);
    }
    unsigned int Vertex(int face, int corner) const
    {
        return mesh.mFaces[triangles[face]].mIndices[corner];
    }
};

void CalculateMikkTangents(aiMesh &mesh)
{
    MikkMesh input{mesh, {}};
    for (unsigned int face = 0; face < mesh.mNumFaces; ++face)
        if (mesh.mFaces[face].mNumIndices == 3)
            input.triangles.push_back(face);
    if (input.triangles.empty())
        return;
    if (input.triangles.size() > std::numeric_limits<int>::max())
        throw std::runtime_error("MikkTSpace triangle count exceeds its index range");
    mesh.mTangents = new aiVector3D[mesh.mNumVertices];
    mesh.mBitangents = new aiVector3D[mesh.mNumVertices];
    SMikkTSpaceInterface interface{};
    interface.m_getNumFaces = [](const SMikkTSpaceContext *c) {
        return static_cast<int>(MikkMesh::Get(c).triangles.size());
    };
    interface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext *, int) { return 3; };
    interface.m_getPosition = [](const SMikkTSpaceContext *c, float out[], int f, int v) {
        const auto &m = MikkMesh::Get(c);
        const auto &p = m.mesh.mVertices[m.Vertex(f, v)];
        out[0] = p.x;
        out[1] = p.y;
        out[2] = p.z;
    };
    interface.m_getNormal = [](const SMikkTSpaceContext *c, float out[], int f, int v) {
        const auto &m = MikkMesh::Get(c);
        const auto &n = m.mesh.mNormals[m.Vertex(f, v)];
        out[0] = n.x;
        out[1] = n.y;
        out[2] = n.z;
    };
    interface.m_getTexCoord = [](const SMikkTSpaceContext *c, float out[], int f, int v) {
        const auto &m = MikkMesh::Get(c);
        const auto &uv = m.mesh.mTextureCoords[0][m.Vertex(f, v)];
        out[0] = uv.x;
        out[1] = uv.y;
    };
    interface.m_setTSpaceBasic = [](const SMikkTSpaceContext *c, const float tangent[], float sign, int f, int v) {
        auto &m = MikkMesh::Get(c);
        const auto index = m.Vertex(f, v);
        m.mesh.mTangents[index] = {tangent[0], tangent[1], tangent[2]};
        m.mesh.mBitangents[index] = (m.mesh.mNormals[index] ^ m.mesh.mTangents[index]) * sign;
    };
    SMikkTSpaceContext context{&interface, &input};
    if (!genTangSpaceDefault(&context))
        throw std::runtime_error("MikkTSpace could not calculate the model tangent basis");
}
} // namespace

void BuildModelVertexBasis(const aiScene &scene, const MeshImportSettings &settings)
{
    for (unsigned int index = 0; index < scene.mNumMeshes; ++index) {
        auto &mesh = *scene.mMeshes[index];
        if (!mesh.mNumVertices || !(mesh.mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
            continue;
        const bool wantsNormals = settings.normalMode == "import" || settings.normalMode == "calculate";
        if (wantsNormals && settings.normalSmoothingSource == "source" && !mesh.HasNormals())
            throw std::runtime_error("source smoothing requires authored source normals");
        const bool normals = settings.normalMode == "calculate" || (!mesh.HasNormals() && wantsNormals);
        const bool tangents = settings.tangentAlgorithm == "mikktspace" && !mesh.HasTangentsAndBitangents() &&
                              (mesh.HasNormals() || normals) && mesh.HasTextureCoords(0) &&
                              (settings.tangentMode == "import" || settings.tangentMode == "calculate");
        if (!normals && !tangents)
            continue;
        ExpandCorners(mesh);
        if (normals)
            CalculateNormals(mesh, settings);
        if (tangents)
            CalculateMikkTangents(mesh);
    }
}
} // namespace infernux
