#include "RayGeometry.h"

#include <function/renderer/InxRenderStruct.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace infernux::raytracing
{
namespace
{
using spatial::Bounds;
using spatial::BvhBuild;
using spatial::BvhNode;
using spatial::kBvhLeafBit;

bool IsFinite(const glm::vec3 &value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

uint32_t PageOffset(size_t size, size_t count)
{
    // The leaf tag occupies the upper bit of a node word.
    if (count > kBvhLeafBit - size)
        throw std::length_error("Ray geometry page exceeds its index range");
    return static_cast<uint32_t>(size);
}

Bounds TransformBounds(const BvhNode &root, const glm::mat4 &matrix)
{
    // Accumulate interval endpoints in double precision, then round outwards for the GPU slab test.
    Bounds result;
    for (unsigned row = 0; row < 3; ++row) {
        double low = matrix[3][row], high = low;
        for (unsigned axis = 0; axis < 3; ++axis) {
            const double coefficient = matrix[axis][row];
            const double a = coefficient * root.minimum[axis];
            const double b = coefficient * root.maximum[axis];
            low += std::min(a, b);
            high += std::max(a, b);
        }
        result.minimum[row] = std::nextafter(static_cast<float>(low), -std::numeric_limits<float>::infinity());
        result.maximum[row] = std::nextafter(static_cast<float>(high), std::numeric_limits<float>::infinity());
    }
    return result;
}

RayInstance EncodeInstance(const RayMeshInstance &instance, uint32_t root)
{
    const glm::dmat4 model(instance.objectToWorld);
    if (model[0].w != 0.0 || model[1].w != 0.0 || model[2].w != 0.0 || model[3].w != 1.0)
        throw std::invalid_argument("Ray geometry instance requires an affine transform");
    const glm::dmat3 inverseLinear = glm::inverse(glm::dmat3(model));
    const glm::dvec3 inverseTranslation = -(inverseLinear * glm::dvec3(model[3]));
    const auto row = [&](unsigned i) {
        std::array<float, 4> components{
            static_cast<float>(inverseLinear[0][i]), static_cast<float>(inverseLinear[1][i]),
            static_cast<float>(inverseLinear[2][i]), static_cast<float>(inverseTranslation[i])};
        if (!std::all_of(components.begin(), components.end(), [](float v) { return std::isfinite(v); }))
            throw std::invalid_argument("Ray geometry instance requires a finite inverse transform");
        return components;
    };
    return {row(0), row(1), row(2), root, instance.surfaceId, instance.layerMask};
}

BvhNode RelocateNode(BvhNode node, uint32_t root, uint32_t leafOffset)
{
    if ((node.second & kBvhLeafBit) != 0) {
        node.first += leafOffset;
    } else {
        node.first += root;
        node.second += root;
    }
    return node;
}

template <typename Visit> void VisitWindow(size_t &cursor, size_t count, uint32_t &budget, Visit visit)
{
    const auto visits = std::min<size_t>(count - cursor, budget);
    budget -= static_cast<uint32_t>(visits);
    const auto end = cursor + visits;
    for (; cursor != end; ++cursor)
        visit(cursor);
}
} // namespace

struct RayMeshBuild::State
{
    State(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, uint32_t indexStart,
          uint32_t indexCount, int32_t vertexStart)
        : vertices(vertices), indices(indices), indexStart(indexStart), vertexStart(vertexStart),
          triangleCount(indexCount / 3)
    {
        if (indexCount % 3 != 0 || indexStart > indices.size() || indexCount > indices.size() - indexStart)
            throw std::invalid_argument("Ray mesh requires a complete indexed triangle slice");
        bounds.reserve(triangleCount);
        triangles.reserve(triangleCount);
        mesh.positions.reserve(std::min<size_t>(vertices.size(), indexCount));
    }

    void Gather(uint32_t primitive)
    {
        const size_t first = static_cast<size_t>(indexStart) + primitive * 3;
        uint32_t corners[3];
        Bounds triangleBounds;
        for (unsigned corner = 0; corner < 3; ++corner) {
            const int64_t vertex = static_cast<int64_t>(indices[first + corner]) + vertexStart;
            if (vertex < 0 || static_cast<uint64_t>(vertex) >= vertices.size() || vertex > UINT32_MAX)
                throw std::invalid_argument("Ray mesh index falls outside its vertex stream");
            const glm::vec3 &position = vertices[static_cast<size_t>(vertex)].pos;
            const auto source = static_cast<uint32_t>(vertex);
            auto [page, inserted] = positionIndices.try_emplace(source / RemapPageSize);
            if (inserted)
                page->second.fill(kInvalidRayIndex);
            auto &mapped = page->second[source % RemapPageSize];
            if (mapped == kInvalidRayIndex) {
                if (!IsFinite(position))
                    throw std::invalid_argument("Ray mesh requires finite vertex positions");
                mapped = static_cast<uint32_t>(mesh.positions.size());
                mesh.positions.push_back({position.x, position.y, position.z});
            }
            corners[corner] = mapped;
            triangleBounds.Include({position, position});
        }
        bounds.push_back(triangleBounds);
        triangles.push_back({corners[0], corners[1], corners[2], primitive});
    }

    const std::vector<Vertex> &vertices;
    const std::vector<uint32_t> &indices;
    uint32_t indexStart;
    int32_t vertexStart;
    uint32_t triangleCount;
    size_t cursor = 0;
    RayMesh mesh;
    std::vector<Bounds> bounds;
    std::vector<RayTriangle> triangles;
    static constexpr uint32_t RemapPageSize = 32;
    std::map<uint32_t, std::array<uint32_t, RemapPageSize>> positionIndices;
    std::optional<BvhBuild> builder;
    std::vector<uint32_t> order;
    enum class Phase : uint8_t
    {
        Gather,
        Build,
        Reorder,
        Done,
    } phase = Phase::Gather;
};

RayMeshBuild::RayMeshBuild(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices,
                           uint32_t indexStart, uint32_t indexCount, int32_t vertexStart)
    : m_state(std::make_unique<State>(vertices, indices, indexStart, indexCount, vertexStart))
{
}

RayMeshBuild::~RayMeshBuild() = default;

bool RayMeshBuild::Advance(uint32_t budget)
{
    auto &state = *m_state;
    using Phase = State::Phase;
    // Each stage owns the window. In particular the BVH builder accounts for
    // every SAH visit internally, including repeated visits to the same range.
    switch (state.phase) {
    case Phase::Gather:
        VisitWindow(state.cursor, state.triangleCount, budget,
                    [&](size_t index) { state.Gather(static_cast<uint32_t>(index)); });
        if (state.cursor == state.triangleCount) {
            state.builder.emplace(state.bounds);
            state.phase = Phase::Build;
        }
        return false;
    case Phase::Build: {
        if (!state.builder->Advance(budget))
            return false;
        auto hierarchy = state.builder->Take();
        state.builder.reset();
        state.mesh.nodes = std::move(hierarchy.nodes);
        state.order = std::move(hierarchy.primitiveOrder);
        state.mesh.triangles.reserve(state.order.size());
        state.cursor = 0;
        state.phase = Phase::Reorder;
        return false;
    }
    case Phase::Reorder:
        VisitWindow(state.cursor, state.order.size(), budget,
                    [&](size_t index) { state.mesh.triangles.push_back(state.triangles[state.order[index]]); });
        if (state.cursor != state.order.size())
            return false;
        state.phase = Phase::Done;
        return true;
    case Phase::Done:
        return true;
    }
    std::terminate();
}

RayMesh RayMeshBuild::Take()
{
    return std::move(m_state->mesh);
}

struct RayGeometryPageBuild::State
{
    struct MeshRange
    {
        const RayMesh *mesh;
        uint32_t node;
        uint32_t position;
        uint32_t triangle;
    };

    explicit State(const std::vector<RayMeshInstance> &instances) : instances(instances)
    {
        bounds.reserve(instances.size());
        packed.reserve(instances.size());
        ranges.reserve(instances.size());
    }

    void AddInstance(const RayMeshInstance &instance)
    {
        if (instance.mesh.nodes.empty())
            return;
        const auto [entry, inserted] = roots.try_emplace(&instance.mesh, nodeCount);
        if (inserted) {
            const auto &mesh = instance.mesh;
            ranges.push_back({&mesh, PageOffset(nodeCount, mesh.nodes.size()),
                              PageOffset(positionCount, mesh.positions.size()),
                              PageOffset(triangleCount, mesh.triangles.size())});
            nodeCount += static_cast<uint32_t>(mesh.nodes.size());
            positionCount += static_cast<uint32_t>(mesh.positions.size());
            triangleCount += static_cast<uint32_t>(mesh.triangles.size());
        }
        bounds.push_back(TransformBounds(instance.mesh.nodes.front(), instance.objectToWorld));
        packed.push_back(EncodeInstance(instance, entry->second));
    }

    bool AppendMeshes(uint32_t budget)
    {
        for (; meshIndex < ranges.size() && budget; ++meshIndex) {
            const auto &range = ranges[meshIndex];
            const auto &mesh = *range.mesh;
            const auto positions = mesh.nodes.size();
            const auto triangles = positions + mesh.positions.size();
            const auto end = triangles + mesh.triangles.size();
            if (cursor < positions)
                VisitWindow(cursor, positions, budget, [&](size_t i) {
                    page.nodes.push_back(RelocateNode(mesh.nodes[i], range.node, range.triangle));
                });
            if (cursor < triangles)
                VisitWindow(cursor, triangles, budget,
                            [&](size_t i) { page.positions.push_back(mesh.positions[i - positions]); });
            if (cursor < end)
                VisitWindow(cursor, end, budget, [&](size_t i) {
                    const auto &tri = mesh.triangles[i - triangles];
                    page.triangles.push_back(
                        {tri.a + range.position, tri.b + range.position, tri.c + range.position, tri.primitive});
                });
            if (cursor != end)
                return false;
            cursor = 0;
        }
        return meshIndex == ranges.size();
    }

    const std::vector<RayMeshInstance> &instances;
    size_t cursor = 0;
    size_t meshIndex = 0;
    uint32_t nodeCount = 0, positionCount = 0, triangleCount = 0;
    RayGeometryPage page;
    std::map<const RayMesh *, uint32_t> roots;
    std::vector<MeshRange> ranges;
    std::vector<Bounds> bounds;
    std::vector<RayInstance> packed;
    std::optional<BvhBuild> builder;
    spatial::Bvh hierarchy;
    enum class Phase : uint8_t
    {
        Gather,
        Build,
        AppendMeshes,
        AppendTop,
        Reorder,
        Done,
    } phase = Phase::Gather;
};

RayGeometryPageBuild::RayGeometryPageBuild(const std::vector<RayMeshInstance> &instances)
    : m_state(std::make_unique<State>(instances))
{
}

RayGeometryPageBuild::~RayGeometryPageBuild() = default;

bool RayGeometryPageBuild::Advance(uint32_t budget)
{
    auto &state = *m_state;
    using Phase = State::Phase;
    switch (state.phase) {
    case Phase::Gather:
        VisitWindow(state.cursor, state.instances.size(), budget,
                    [&](size_t index) { state.AddInstance(state.instances[index]); });
        if (state.cursor == state.instances.size()) {
            state.builder.emplace(state.bounds);
            state.phase = Phase::Build;
        }
        return false;
    case Phase::Build:
        if (!state.builder->Advance(budget))
            return false;
        state.hierarchy = state.builder->Take();
        state.builder.reset();
        if (state.hierarchy.nodes.empty()) {
            state.phase = Phase::Done;
            return true;
        }
        state.page.root = PageOffset(state.nodeCount, state.hierarchy.nodes.size());
        state.page.nodes.reserve(state.nodeCount + state.hierarchy.nodes.size());
        state.page.positions.reserve(state.positionCount);
        state.page.triangles.reserve(state.triangleCount);
        state.page.instances.reserve(state.packed.size());
        state.cursor = 0;
        state.phase = Phase::AppendMeshes;
        return false;
    case Phase::AppendMeshes:
        if (state.AppendMeshes(budget))
            state.phase = Phase::AppendTop;
        return false;
    case Phase::AppendTop:
        VisitWindow(state.cursor, state.hierarchy.nodes.size(), budget, [&](size_t index) {
            state.page.nodes.push_back(RelocateNode(state.hierarchy.nodes[index], state.page.root, 0));
        });
        if (state.cursor == state.hierarchy.nodes.size()) {
            state.cursor = 0;
            state.phase = Phase::Reorder;
        }
        return false;
    case Phase::Reorder:
        VisitWindow(state.cursor, state.packed.size(), budget, [&](size_t index) {
            state.page.instances.push_back(state.packed[state.hierarchy.primitiveOrder[index]]);
        });
        if (state.cursor != state.packed.size())
            return false;
        state.phase = Phase::Done;
        return true;
    case Phase::Done:
        return true;
    }
    std::terminate();
}

RayGeometryPage RayGeometryPageBuild::Take()
{
    return std::move(m_state->page);
}

RayMesh BuildRayMesh(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, uint32_t indexStart,
                     uint32_t indexCount, int32_t vertexStart)
{
    RayMeshBuild build(vertices, indices, indexStart, indexCount, vertexStart);
    while (!build.Advance(UINT32_MAX)) {
    }
    return build.Take();
}

RayGeometryPage BuildRayGeometryPage(const std::vector<RayMeshInstance> &instances)
{
    RayGeometryPageBuild build(instances);
    while (!build.Advance(UINT32_MAX)) {
    }
    return build.Take();
}

RayGeometryPageLayout::RayGeometryPageLayout(const RayGeometryPage &page, RayGeometryStatus status) : m_page(page)
{
    const uint64_t nodes = 2;
    const uint64_t positions = nodes + uint64_t{page.nodes.size()} * (sizeof(BvhNode) / sizeof(RayGeometryWord));
    const uint64_t triangles = positions + page.positions.size();
    const uint64_t instances = triangles + page.triangles.size();
    const uint64_t size = instances + uint64_t{page.instances.size()} * (sizeof(RayInstance) / sizeof(RayGeometryWord));
    if (size > kInvalidRayIndex)
        throw std::length_error("Ray geometry exceeds the uvec4 address range");
    header[0] = {page.root, static_cast<uint32_t>(nodes), static_cast<uint32_t>(positions),
                 static_cast<uint32_t>(triangles)};
    header[1] = {static_cast<uint32_t>(instances), static_cast<uint32_t>(status), 0u, 0u};
    bytes = size * sizeof(RayGeometryWord);
}

std::array<RayGeometryBytes, 5> RayGeometryPageLayout::Sections() const
{
    const auto section = [](const auto &values) -> RayGeometryBytes {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        static_assert(sizeof(Value) % sizeof(RayGeometryWord) == 0);
        return {values.data(), values.size() * sizeof(Value)};
    };
    return {section(header), section(m_page.nodes), section(m_page.positions), section(m_page.triangles),
            section(m_page.instances)};
}

std::vector<RayGeometryWord> EncodeRayGeometryPage(const RayGeometryPage &page, RayGeometryStatus status)
{
    const RayGeometryPageLayout layout(page, status);
    std::vector<RayGeometryWord> storage(static_cast<size_t>(layout.bytes / sizeof(RayGeometryWord)));
    auto *destination = reinterpret_cast<std::byte *>(storage.data());
    for (const auto &section : layout.Sections()) {
        if (section.bytes)
            std::memcpy(destination, section.data, section.bytes);
        destination += section.bytes;
    }
    return storage;
}

} // namespace infernux::raytracing
