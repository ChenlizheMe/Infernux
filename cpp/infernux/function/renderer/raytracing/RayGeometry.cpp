#include "RayGeometry.h"

#include <function/renderer/InxRenderStruct.h>

#include <algorithm>
#include <cmath>
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

Bounds TransformBounds(const Bounds &root, const glm::mat4 &matrix)
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

RayInstance EncodeInstance(const RayMeshInstance &instance)
{
    const glm::dmat4 model(instance.objectToWorld);
    if (model[0].w != 0.0 || model[1].w != 0.0 || model[2].w != 0.0 || model[3].w != 1.0)
        throw std::invalid_argument("Ray geometry instance requires an affine transform");
    const glm::dmat3 linear(model);
    const glm::dmat3 inverseLinear = glm::inverse(linear);
    const glm::dvec3 inverseTranslation = -(inverseLinear * glm::dvec3(model[3]));
    const auto row = [&](unsigned i) {
        std::array<float, 4> components{
            static_cast<float>(inverseLinear[0][i]), static_cast<float>(inverseLinear[1][i]),
            static_cast<float>(inverseLinear[2][i]), static_cast<float>(inverseTranslation[i])};
        if (!std::all_of(components.begin(), components.end(), [](float v) { return std::isfinite(v); }))
            throw std::invalid_argument("Ray geometry instance requires a finite inverse transform");
        return components;
    };
    const auto orientation = glm::determinant(linear) < 0.0 ? -1.0f : 1.0f;
    return {row(0), row(1), row(2), instance.mesh.root, instance.surfaceId, instance.layerMask, orientation};
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
        const auto vertexCapacity = std::min<size_t>(vertices.size(), indexCount);
        mesh.positions.reserve(vertexCapacity);
        mesh.attributes.reserve(vertexCapacity);
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
            const auto &sourceVertex = vertices[static_cast<size_t>(vertex)];
            const auto &position = sourceVertex.pos;
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
                const auto &uv = sourceVertex.texCoord;
                const auto &uv1 = sourceVertex.texCoord1;
                const auto &normal = sourceVertex.normal;
                const auto &tangent = sourceVertex.tangent;
                const auto &color = sourceVertex.color;
                mesh.attributes.push_back({{uv.x, uv.y, uv1.x, uv1.y},
                                           {normal.x, normal.y, normal.z, 0.0f},
                                           {tangent.x, tangent.y, tangent.z, tangent.w},
                                           {color.r, color.g, color.b, 1.0f}});
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

struct RayMeshPageBuild::State
{
    explicit State(const std::vector<RayMesh> &meshes) : meshes(meshes)
    {
        page.placements.reserve(meshes.size());
    }

    const std::vector<RayMesh> &meshes;
    RayMeshPage page;
    size_t nextSize = 0;
    bool allocated = false;
    size_t nodeCount = 0, vertexCount = 0, triangleCount = 0;
    size_t meshIndex = 0;
    size_t cursor = 0;
    uint32_t root = 0;
    uint32_t positionBase = 0;
    uint32_t triangleBase = 0;
};

RayMeshPageBuild::RayMeshPageBuild(const std::vector<RayMesh> &meshes) : m_state(std::make_unique<State>(meshes))
{
}

RayMeshPageBuild::~RayMeshPageBuild() = default;

bool RayMeshPageBuild::Advance(uint32_t budget)
{
    auto &s = *m_state;
    if (!s.allocated) {
        VisitWindow(s.nextSize, s.meshes.size(), budget, [&](size_t index) {
            const auto &mesh = s.meshes[index];
            s.nodeCount += mesh.nodes.size();
            s.vertexCount += mesh.positions.size();
            s.triangleCount += mesh.triangles.size();
        });
        if (s.nextSize != s.meshes.size())
            return false;
        s.page.nodes.reserve(s.nodeCount);
        s.page.positions.reserve(s.vertexCount);
        s.page.attributes.reserve(s.vertexCount);
        s.page.triangles.reserve(s.triangleCount);
        s.allocated = true;
    }
    for (; s.meshIndex < s.meshes.size() && budget; ++s.meshIndex) {
        const auto &mesh = s.meshes[s.meshIndex];
        if (mesh.nodes.empty()) {
            s.page.placements.emplace_back();
            --budget;
            continue;
        }
        if (!s.cursor) {
            s.root = PageOffset(s.page.nodes.size(), mesh.nodes.size());
            s.positionBase = PageOffset(s.page.positions.size(), mesh.positions.size());
            s.triangleBase = PageOffset(s.page.triangles.size(), mesh.triangles.size());
            const auto &bounds = mesh.nodes.front();
            s.page.placements.push_back({{
                                             {bounds.minimum[0], bounds.minimum[1], bounds.minimum[2]},
                                             {bounds.maximum[0], bounds.maximum[1], bounds.maximum[2]},
                                         },
                                         s.root});
        }
        const size_t nodes = mesh.nodes.size();
        const size_t vertices = nodes + mesh.positions.size();
        const size_t end = vertices + mesh.triangles.size();
        if (s.cursor < nodes)
            VisitWindow(s.cursor, nodes, budget,
                        [&](size_t i) { s.page.nodes.push_back(RelocateNode(mesh.nodes[i], s.root, s.triangleBase)); });
        if (s.cursor < vertices)
            VisitWindow(s.cursor, vertices, budget, [&](size_t i) {
                const auto vertex = i - nodes;
                s.page.positions.push_back(mesh.positions[vertex]);
                s.page.attributes.push_back(mesh.attributes[vertex]);
            });
        if (s.cursor < end)
            VisitWindow(s.cursor, end, budget, [&](size_t i) {
                const auto &tri = mesh.triangles[i - vertices];
                s.page.triangles.push_back(
                    {tri.a + s.positionBase, tri.b + s.positionBase, tri.c + s.positionBase, tri.primitive});
            });
        if (s.cursor != end)
            return false;
        s.cursor = 0;
    }
    return s.meshIndex == s.meshes.size();
}

RayMeshPage RayMeshPageBuild::Take()
{
    return std::move(m_state->page);
}

struct RayInstancePageBuild::State
{
    State(const std::vector<RayMeshInstance> &instances, std::shared_ptr<const spatial::BvhTopology> topology)
        : instances(instances), previous(std::move(topology))
    {
        bounds.reserve(instances.size());
        packed.reserve(instances.size());
    }

    const std::vector<RayMeshInstance> &instances;
    std::vector<Bounds> bounds;
    std::vector<RayInstance> packed;
    std::shared_ptr<const spatial::BvhTopology> previous;
    std::optional<spatial::BvhRefit> refit;
    std::optional<BvhBuild> builder;
    std::shared_ptr<spatial::BvhTopology> captured;
    RayInstancePage page;
    double buildCost = 0.0;
    size_t cursor = 0;
    enum class Phase : uint8_t
    {
        Gather,
        Refit,
        Build,
        CaptureTopology,
        Reorder,
        Done
    } phase = Phase::Gather;

    void StartBuild()
    {
        builder.emplace(bounds);
        phase = Phase::Build;
    }

    void Reorder(std::shared_ptr<const spatial::BvhTopology> topology)
    {
        page.topology = std::move(topology);
        page.instances.reserve(packed.size());
        cursor = 0;
        phase = Phase::Reorder;
    }
};

RayInstancePageBuild::RayInstancePageBuild(const std::vector<RayMeshInstance> &instances,
                                           std::shared_ptr<const spatial::BvhTopology> topology)
    : m_state(std::make_unique<State>(instances, std::move(topology)))
{
}

RayInstancePageBuild::~RayInstancePageBuild() = default;

bool RayInstancePageBuild::Advance(uint32_t budget)
{
    auto &s = *m_state;
    using Phase = State::Phase;
    switch (s.phase) {
    case Phase::Gather:
        VisitWindow(s.cursor, s.instances.size(), budget, [&](size_t index) {
            const auto &instance = s.instances[index];
            s.bounds.push_back(TransformBounds(instance.mesh.bounds, instance.objectToWorld));
            s.packed.push_back(EncodeInstance(instance));
        });
        if (s.cursor == s.instances.size()) {
            if (s.previous) {
                s.refit.emplace(*s.previous, s.bounds);
                s.phase = Phase::Refit;
            } else {
                s.StartBuild();
            }
        }
        return false;
    case Phase::Refit: {
        if (!s.refit->Advance(budget))
            return false;
        // Relative SAH preserves a well-partitioned tree under uniform motion or scale.
        constexpr double rebuildCostRatio = 1.5;
        if (s.refit->Cost() <= s.previous->buildCost * rebuildCostRatio) {
            s.page.nodes = s.refit->Take();
            s.Reorder(std::move(s.previous));
        } else {
            s.StartBuild();
        }
        s.refit.reset();
        return false;
    }
    case Phase::Build: {
        if (!s.builder->Advance(budget))
            return false;
        auto hierarchy = s.builder->Take();
        s.builder.reset();
        s.page.nodes = std::move(hierarchy.nodes);
        s.captured = std::make_shared<spatial::BvhTopology>();
        s.captured->primitiveOrder = std::move(hierarchy.primitiveOrder);
        s.captured->links.reserve(s.page.nodes.size());
        s.cursor = 0;
        s.phase = Phase::CaptureTopology;
        return false;
    }
    case Phase::CaptureTopology:
        VisitWindow(s.cursor, s.page.nodes.size(), budget, [&](size_t index) {
            const auto &node = s.page.nodes[index];
            s.captured->links.push_back({node.first, node.second});
            s.buildCost += spatial::BvhNodeCost(node);
        });
        if (s.cursor == s.page.nodes.size()) {
            s.captured->buildCost = spatial::NormalizeBvhCost(s.buildCost, s.page.nodes);
            s.Reorder(std::move(s.captured));
        }
        return false;
    case Phase::Reorder:
        VisitWindow(s.cursor, s.packed.size(), budget, [&](size_t index) {
            s.page.instances.push_back(s.packed[s.page.topology->primitiveOrder[index]]);
        });
        if (s.cursor != s.packed.size())
            return false;
        s.phase = Phase::Done;
        return true;
    case Phase::Done:
        return true;
    }
    std::terminate();
}

RayInstancePage RayInstancePageBuild::Take()
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

namespace
{
template <typename T> RayGeometryBytes MeshSection(const std::vector<T> &values)
{
    static_assert(sizeof(T) % sizeof(RayGeometryWord) == 0);
    return {values.data(), values.size() * sizeof(T)};
}

uint32_t PageWordOffset(uint64_t bytes)
{
    const auto words = bytes / sizeof(RayGeometryWord);
    if (words > kInvalidRayIndex)
        throw std::length_error("Ray geometry exceeds the uvec4 address range");
    return static_cast<uint32_t>(words);
}
} // namespace

RayPageUpload::RayPageUpload(const RayMeshPage &page)
    : m_sections{MeshSection(page.nodes), MeshSection(page.positions), MeshSection(page.triangles),
                 MeshSection(page.attributes)}
{
    bytes = sizeof(m_header);
    for (size_t index = 0; index < m_sections.size(); ++index) {
        m_header[index] = PageWordOffset(bytes);
        bytes += m_sections[index].bytes;
    }
    (void)PageWordOffset(bytes);
}

RayPageUpload::RayPageUpload(const RayInstancePage &page, RayGeometryStatus status)
    : m_sections{MeshSection(page.nodes), MeshSection(page.instances)}
{
    m_header = {page.nodes.empty() ? kInvalidRayIndex : 0u, 1u, PageWordOffset(sizeof(m_header) + m_sections[0].bytes),
                static_cast<uint32_t>(status)};
    bytes = sizeof(m_header) + m_sections[0].bytes + m_sections[1].bytes;
    (void)PageWordOffset(bytes);
}

RayGeometryBytes RayPageUpload::Section(size_t index) const noexcept
{
    return index ? m_sections[index - 1] : RayGeometryBytes{m_header.data(), sizeof(m_header)};
}

} // namespace infernux::raytracing
