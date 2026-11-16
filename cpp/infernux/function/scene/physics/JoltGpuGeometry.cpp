#include "JoltGpuGeometry.h"
#include <Jolt/Jolt.h>
#include <Jolt/Geometry/Plane.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
struct GeometryMemo
{
    std::mutex mutex;
    std::unordered_map<const JPH::Shape *, std::weak_ptr<const JoltGpuGeometryRecord>> records;
};
std::shared_ptr<GeometryMemo> RecordMemo()
{
    static auto memo = std::make_shared<GeometryMemo>();
    return memo;
}
UInt4 Word(Float4 value)
{
    UInt4 result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
Float4 Value(UInt4 word)
{
    Float4 result;
    std::memcpy(&result, &word, sizeof(result));
    return result;
}
UInt4 Word(JPH::Vec3Arg point)
{
    return Word(Float4{point.GetX(), point.GetY(), point.GetZ(), 0});
}
bool Finite(Float4 value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) && std::isfinite(value.w);
}
uint32_t Lane(UInt4 word, uint32_t lane)
{
    return lane == 0 ? word.x : lane == 1 ? word.y : lane == 2 ? word.z : word.w;
}
void SetLane(UInt4 &word, uint32_t lane, uint32_t value)
{
    if (lane == 0)
        word.x = value;
    else if (lane == 1)
        word.y = value;
    else if (lane == 2)
        word.z = value;
    else
        word.w = value;
}

// Match ConvexHullShape construction's selected incidences, in native face order.
// This is author-data extraction, not a runtime collision/support approximation.
JPH::Vec3 CorePoint(const JPH::ConvexHullShape &hull, uint32_t vertex, const std::vector<std::vector<uint32_t>> &loops)
{
    const auto position = hull.GetPoint(vertex);
    const float radius = hull.GetConvexRadius();
    if (radius == 0)
        return position;
    const auto &planes = hull.GetPlanes();
    std::vector<uint32_t> faces;
    for (uint32_t i = 0; i < loops.size(); ++i)
        if (std::find(loops[i].begin(), loops[i].end(), vertex) != loops[i].end())
            faces.push_back(i);
    if (faces.size() < 2)
        throw std::invalid_argument("Final convex hull vertex has fewer than two incident faces");
    float biggestVolume = 0.05f;
    float smallestDot = JPH::Cos(JPH::DegreesToRadians(1.0f));
    int best3[3] = {-1, -1, -1}, best2[2] = {-1, -1};
    for (uint32_t i = 0; i < faces.size(); ++i) {
        const auto n1 = planes[faces[i]].GetNormal();
        for (uint32_t j = i + 1; j < faces.size(); ++j) {
            const auto n2 = planes[faces[j]].GetNormal();
            const auto cross = n1.Cross(n2);
            const float dot = n1.Dot(n2);
            if (dot < smallestDot) {
                smallestDot = dot;
                best2[0] = faces[i];
                best2[1] = faces[j];
            }
            for (uint32_t k = j + 1; k < faces.size(); ++k) {
                const float volume = std::abs(cross.Dot(planes[faces[k]].GetNormal()));
                if (volume > biggestVolume) {
                    biggestVolume = volume;
                    best3[0] = faces[i];
                    best3[1] = faces[j];
                    best3[2] = faces[k];
                }
            }
        }
    }
    if (best3[0] == -1 && best2[0] == -1)
        return position - planes[faces[0]].GetNormal() * radius;
    const auto p1 = planes[best3[0] != -1 ? best3[0] : best2[0]].Offset(-radius);
    const auto p2 = planes[best3[0] != -1 ? best3[1] : best2[1]].Offset(-radius);
    const auto p3 = best3[0] != -1 ? planes[best3[2]].Offset(-radius)
                                   : JPH::Plane::sFromPointAndNormal(position, p1.GetNormal().Cross(p2.GetNormal()));
    JPH::Vec3 core;
    // Exactly the same native ExcludeConvexRadius construction for singular planes.
    if (!JPH::Plane::sIntersectPlanes(p1, p2, p3, core))
        core = position - p1.GetNormal() * radius;
    return core;
}
} // namespace

std::shared_ptr<const JoltGpuGeometryRecord> EncodeJoltGpuGeometry(const JPH::Shape &shape,
                                                                   const std::array<JPH::Vec3, 3> *meshRayVertices)
{
    if (shape.GetSubType() != JPH::EShapeSubType::ConvexHull)
        return {};
    // Mesh recipes are already memoized by MeshShape. Their uncentered vertices
    // are separate input, so a hull-only memo cannot identify this record.
    auto memo = meshRayVertices ? nullptr : RecordMemo();
    if (memo) {
        std::lock_guard lock(memo->mutex);
        if (const auto found = memo->records.find(&shape); found != memo->records.end())
            if (auto existing = found->second.lock())
                return existing;
    }
    const auto &hull = static_cast<const JPH::ConvexHullShape &>(shape);
    const uint32_t vertices = hull.GetNumPoints(), faces = hull.GetNumFaces();
    if (vertices == 0 || vertices > JPH::ConvexHullShape::cMaxPointsInHull || faces == 0 ||
        hull.GetPlanes().size() != faces)
        throw std::invalid_argument("Invalid final native convex hull counts");
    std::vector<std::vector<uint32_t>> loops(faces);
    uint64_t indexCount = 0;
    for (uint32_t face = 0; face < faces; ++face) {
        auto &loop = loops[face];
        loop.resize(hull.GetNumVerticesInFace(face));
        if (loop.size() < 3 || loop.size() > vertices ||
            hull.GetFaceVertices(face, static_cast<uint32_t>(loop.size()), loop.data()) != loop.size())
            throw std::invalid_argument("Invalid native convex hull face loop");
        for (const auto vertex : loop)
            if (vertex >= vertices)
                throw std::invalid_argument("Native convex hull face index exceeds final vertex count");
        indexCount += loop.size();
    }
    const bool hasCore = hull.GetConvexRadius() != 0;
    const uint64_t full = ConvexGeometryHeaderWords;
    const uint64_t core = hasCore ? full + vertices : full;
    const uint64_t plane = full + vertices + (hasCore ? vertices : 0);
    const uint64_t directory = plane + faces;
    const uint64_t indices = directory + faces;
    const uint64_t loopEnd = indices + (indexCount + 3) / 4;
    if (meshRayVertices && (vertices != 3 || faces != 2 || hasCore))
        throw std::invalid_argument("Mesh ray vertices require an exact zero-margin planar triangle");
    const uint64_t total = loopEnd + (meshRayVertices ? 3 : 0);
    if (total > UINT32_MAX || indexCount > UINT32_MAX || total > SIZE_MAX / sizeof(UInt4))
        throw std::length_error("Convex geometry record exceeds typed word address range");
    const std::weak_ptr<GeometryMemo> weak = memo;
    auto record = std::shared_ptr<JoltGpuGeometryRecord>(
        new JoltGpuGeometryRecord, [weak, key = &shape](JoltGpuGeometryRecord *value) {
            if (auto memo = weak.lock()) {
                std::lock_guard lock(memo->mutex);
                const auto found = memo->records.find(key);
                if (found != memo->records.end() && found->second.expired())
                    memo->records.erase(found);
            }
            delete value;
        });
    record->shape = &shape;
    record->convexRadius = hull.GetConvexRadius();
    record->words.resize(static_cast<size_t>(total)); // All header/index padding is initialized.
    auto &words = record->words;
    words[0] = {ShapeConvexHull, static_cast<uint32_t>(total), vertices, faces};
    words[1] = {static_cast<uint32_t>(full), static_cast<uint32_t>(core), static_cast<uint32_t>(plane),
                static_cast<uint32_t>(directory)};
    words[2] = {static_cast<uint32_t>(indices), static_cast<uint32_t>(indexCount),
                meshRayVertices ? static_cast<uint32_t>(loopEnd) : 0u, 0};
    if (meshRayVertices)
        for (size_t i = 0; i < meshRayVertices->size(); ++i)
            words[loopEnd + i] = Word((*meshRayVertices)[i]);
    const auto bounds = hull.GetLocalBounds();
    words[3] = Word(bounds.mMin);
    words[4] = Word(bounds.mMax);
    for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
        words[full + vertex] = Word(hull.GetPoint(vertex));
        if (hasCore)
            words[core + vertex] = Word(CorePoint(hull, vertex, loops));
    }
    uint32_t firstIndex = 0;
    for (uint32_t face = 0; face < faces; ++face) {
        const auto &p = hull.GetPlanes()[face];
        const auto n = p.GetNormal();
        words[plane + face] = Word(Float4{n.GetX(), n.GetY(), n.GetZ(), p.GetConstant()});
        words[directory + face] = {firstIndex, static_cast<uint32_t>(loops[face].size()), face, 0};
        for (const auto vertex : loops[face]) {
            SetLane(words[indices + firstIndex / 4], firstIndex % 4, vertex);
            ++firstIndex;
        }
    }
    ValidateJoltGpuGeometry(*record);
    if (memo) {
        std::lock_guard lock(memo->mutex);
        auto &entry = memo->records[&shape];
        if (auto existing = entry.lock())
            return existing;
        entry = record;
    }
    return record;
}

void ValidateJoltGpuGeometry(const JoltGpuGeometryRecord &record)
{
    const auto &w = record.words;
    if (!record.shape || record.shape->GetSubType() != JPH::EShapeSubType::ConvexHull ||
        w.size() < ConvexGeometryHeaderWords || w.size() > UINT32_MAX)
        throw std::invalid_argument("Convex geometry lacks its final native owner/header");
    const auto count = w[0], offsets = w[1], loops = w[2];
    const auto &hull = static_cast<const JPH::ConvexHullShape &>(*record.shape);
    if (!std::isfinite(record.convexRadius) || record.convexRadius != hull.GetConvexRadius())
        throw std::invalid_argument("Convex geometry margin differs from final native hull");
    const bool hasCore = hull.GetConvexRadius() != 0;
    const uint64_t plane = uint64_t(ConvexGeometryHeaderWords) + count.z * uint64_t(hasCore ? 2 : 1);
    const uint64_t loopEnd = uint64_t(loops.x) + (uint64_t(loops.y) + 3) / 4;
    const bool hasMeshRay = loops.z != 0;
    if (count.x != ShapeConvexHull || count.y != w.size() || count.z != hull.GetNumPoints() ||
        count.w != hull.GetNumFaces() || offsets.x != ConvexGeometryHeaderWords ||
        offsets.y != (hasCore ? offsets.x + count.z : offsets.x) || offsets.z != plane ||
        offsets.w != plane + count.w || loops.x != plane + uint64_t(count.w) * 2 || loops.w ||
        loopEnd + (hasMeshRay ? 3u : 0u) != w.size() ||
        (hasMeshRay && (loops.z != loopEnd || hasCore || count.z != 3 || count.w != 2)))
        throw std::invalid_argument("Convex geometry regions do not match final native counts");
    for (uint32_t vertex = 0; vertex < count.z; ++vertex) {
        const auto expected = Word(hull.GetPoint(vertex));
        if (std::memcmp(&w[offsets.x + vertex], &expected, sizeof(UInt4)) != 0)
            throw std::invalid_argument("Convex full point differs from the final native hull");
        for (const auto base : {offsets.x, offsets.y}) {
            const auto point = Value(w[base + vertex]);
            if (!Finite(point) || point.w)
                throw std::invalid_argument("Convex full/core vertex is not finite with zero padding");
        }
    }
    const auto minimum = Value(w[3]), maximum = Value(w[4]);
    const auto bounds = hull.GetLocalBounds();
    const auto expectedMinimum = Word(bounds.mMin), expectedMaximum = Word(bounds.mMax);
    if (std::memcmp(&w[3], &expectedMinimum, sizeof(UInt4)) || std::memcmp(&w[4], &expectedMaximum, sizeof(UInt4)))
        throw std::invalid_argument("Convex bounds differ from the final native hull");
    if (!Finite(minimum) || !Finite(maximum) || minimum.w || maximum.w || minimum.x > maximum.x ||
        minimum.y > maximum.y || minimum.z > maximum.z)
        throw std::invalid_argument("Convex local bounds are invalid");
    uint64_t end = 0;
    for (uint32_t face = 0; face < count.w; ++face) {
        const auto dir = w[offsets.w + face];
        const auto planeValue = Value(w[offsets.z + face]);
        const auto &nativePlane = hull.GetPlanes()[face];
        const auto n = nativePlane.GetNormal();
        const auto expectedPlane = Word(Float4{n.GetX(), n.GetY(), n.GetZ(), nativePlane.GetConstant()});
        std::vector<uint32_t> nativeLoop(hull.GetNumVerticesInFace(face));
        (void)hull.GetFaceVertices(face, static_cast<uint32_t>(nativeLoop.size()), nativeLoop.data());
        if (std::memcmp(&w[offsets.z + face], &expectedPlane, sizeof(UInt4)) != 0)
            throw std::invalid_argument("Convex plane differs from the final native hull");
        if (dir.x != end || dir.y != hull.GetNumVerticesInFace(face) || dir.z != face || dir.w || !Finite(planeValue) ||
            uint64_t(dir.x) + dir.y > loops.y)
            throw std::invalid_argument("Convex face directory/plane is invalid");
        for (uint32_t i = 0; i < dir.y; ++i)
            if (Lane(w[loops.x + (dir.x + i) / 4], (dir.x + i) % 4) != nativeLoop[i])
                throw std::invalid_argument("Convex loop index differs from final native topology");
        end += dir.y;
    }
    if (end != loops.y)
        throw std::invalid_argument("Convex loops do not cover the declared index region");
    if (loops.y % 4)
        for (uint32_t i = loops.y % 4; i < 4; ++i)
            if (Lane(w[loopEnd - 1], i))
                throw std::invalid_argument("Convex packed-index padding is not zero");
    if (hasMeshRay) {
        uint32_t matched = 0;
        for (uint32_t i = 0; i < 3; ++i) {
            const auto vertex = Value(w[loops.z + i]);
            if (!Finite(vertex) || vertex.w != 0)
                throw std::invalid_argument("Mesh ray vertex is not finite with zero padding");
            const auto centered = JPH::Vec3(vertex.x, vertex.y, vertex.z) - hull.GetCenterOfMass();
            for (uint32_t v = 0; v < 3; ++v)
                if (centered == hull.GetPoint(v))
                    matched |= 1u << v;
        }
        if (matched != 7)
            throw std::invalid_argument("Mesh ray vertices do not match the native triangle hull");
    }
}
} // namespace infernux
