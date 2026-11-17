#pragma once

#include "RayGeometry.h"

#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <function/renderer/shader/DeviceShaderProgram.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace infernux::raytracing
{
static_assert(sizeof(Vertex) == 100 && offsetof(Vertex, normal) == 12 && offsetof(Vertex, tangent) == 24 &&
              offsetof(Vertex, color) == 40 && offsetof(Vertex, texCoord) == 52 && offsetof(Vertex, texCoord1) == 60);

struct RayDeformedMesh
{
    uint32_t surfaceId;
    uint32_t meshIndex;
    uint32_t root = 0;
    uint32_t vertexBase = 0;
    uint32_t nodeCount = 0;
    std::vector<uint32_t> sources;
    std::vector<uint32_t> leaves;
    std::vector<std::vector<uint32_t>> levels;
};

struct RayRefitRange
{
    uint32_t offset;
    uint32_t count;
};

struct RayRefitMesh
{
    uint32_t surfaceId;
    uint32_t root;
    uint32_t vertexBase;
    uint32_t vertices;
    uint32_t nodes;
    uint32_t sources;
    RayRefitRange leaves;
    std::vector<RayRefitRange> levels;
};

struct RayRefitPlan
{
    std::vector<RayRefitMesh> meshes;
    std::shared_ptr<const rhi::BufferResource> buffer;
    mutable uint64_t predictedNanoseconds = 100'000;
};

inline RayRefitRange AppendRayRefitIndices(std::vector<uint32_t> &words, const std::vector<uint32_t> &indices)
{
    const auto start = static_cast<uint32_t>(words.size());
    words.insert(words.end(), indices.begin(), indices.end());
    return {start, static_cast<uint32_t>(indices.size())};
}

inline std::shared_ptr<rhi::BufferResource> AllocateRayRefitBuffer(rhi::Device &device, uint64_t bytes,
                                                                   rhi::BufferMemory memory)
{
    rhi::BufferDesc desc;
    desc.byteSize = bytes;
    desc.usage = rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::TransferDestination;
    desc.memory = memory;
    desc.queueAccess =
        rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute | rhi::QueueAccessFlags::Transfer;
    if (!device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
        throw std::length_error("Dynamic ray geometry exceeds the device storage range");
    const auto buffer = device.CreateBuffer(desc);
    if (!buffer.IsValid())
        throw std::runtime_error("Could not allocate dynamic ray geometry");
    return std::make_shared<rhi::BufferResource>(device, buffer, bytes);
}

inline RayRefitPlan UploadRayRefitPlan(rhi::Device &device, std::vector<RayDeformedMesh> meshes)
{
    RayRefitPlan plan;
    std::vector<uint32_t> indices;
    for (auto &mesh : meshes) {
        RayRefitMesh target{mesh.surfaceId,
                            mesh.root,
                            mesh.vertexBase,
                            static_cast<uint32_t>(mesh.sources.size()),
                            mesh.nodeCount,
                            AppendRayRefitIndices(indices, mesh.sources).offset,
                            AppendRayRefitIndices(indices, mesh.leaves),
                            {}};
        for (auto level = mesh.levels.rbegin(); level != mesh.levels.rend(); ++level)
            if (!level->empty())
                target.levels.push_back(AppendRayRefitIndices(indices, *level));
        plan.meshes.push_back(std::move(target));
    }
    if (!indices.empty()) {
        auto buffer = AllocateRayRefitBuffer(device, indices.size() * sizeof(uint32_t), rhi::BufferMemory::Upload);
        if (!device.WriteBuffer(buffer->GetBuffer(), 0, indices.data(), buffer->GetByteSize()))
            throw std::runtime_error("Could not publish the dynamic ray topology");
        plan.buffer = std::move(buffer);
    }
    return plan;
}

class RayDeformationProgram final
{
  public:
    RayDeformationProgram(rhi::Device &device, const ShaderCompilation &compiled) : m_device(device)
    {
        m_shader = std::make_unique<DeviceShaderProgram>(device, compiled.targets.front(), std::vector<uint32_t>{0});
        for (const auto &resource : m_shader->GetResources()) {
            if (resource.name == "vertex_source")
                m_bindings[0] = resource.binding.binding;
            else if (resource.name == "deformation_plan")
                m_bindings[1] = resource.binding.binding;
            else if (resource.name == "mesh_page")
                m_bindings[2] = resource.binding.binding;
            else if (resource.name == "geometry_page")
                m_bindings[3] = resource.binding.binding;
        }
        rhi::ComputePipelineDesc desc;
        m_shader->Apply(desc);
        m_pipeline = device.CreateComputePipeline(desc);
        if (!m_pipeline.IsValid())
            throw std::runtime_error("Could not create the dynamic ray geometry pipeline");
    }
    ~RayDeformationProgram()
    {
        m_device.Release(m_pipeline);
    }

    struct Constants
    {
        glm::uvec4 work;
        glm::uvec4 destination;
        glm::uvec4 mesh;
    };
    static_assert(sizeof(Constants) == 48);

    rhi::BindGroupHandle Bind(rhi::BufferHandle vertices, rhi::BufferHandle plan, rhi::BufferHandle meshes,
                              rhi::BufferHandle output) const
    {
        const rhi::BufferHandle buffers[]{vertices, plan, meshes, output};
        rhi::BindGroupDesc desc;
        desc.layout = m_shader->GetBindingLayouts().front();
        desc.bufferCount = 4;
        for (uint32_t i = 0; i < 4; ++i)
            desc.buffers[i] = {m_bindings[i], rhi::BindingType::StorageBuffer, buffers[i], 0, 0};
        const auto group = m_device.CreateBindGroup(desc);
        if (!group.IsValid())
            throw std::runtime_error("Could not bind the frozen ray deformation streams");
        return group;
    }

    void Record(const rhi::ComputeCommandEncoder &encoder, rhi::BindGroupHandle group, const Constants &constants) const
    {
        encoder.BindPipeline(m_pipeline);
        encoder.BindGroup(m_pipeline, 0, group);
        const uint64_t capacity = uint64_t{m_device.GetCapabilities().limits.maxComputeWorkgroupCount[0]} * 64u;
        for (uint64_t begin = 0; begin < constants.work.y; begin += capacity) {
            auto packet = constants;
            packet.mesh.w = static_cast<uint32_t>(begin);
            const auto count = std::min<uint64_t>(constants.work.y - begin, capacity);
            encoder.PushConstants(m_pipeline, sizeof(Constants), &packet);
            encoder.Dispatch(static_cast<uint32_t>((count + 63u) / 64u));
        }
    }

  private:
    rhi::Device &m_device;
    std::unique_ptr<DeviceShaderProgram> m_shader;
    rhi::ComputePipelineHandle m_pipeline;
    uint32_t m_bindings[4]{};
};

/// This candidate owns private output through its GPU receipt. The queue retains
/// it after acceptance, including when the requesting view is cancelled.
class RayDeformation final : public std::enable_shared_from_this<RayDeformation>
{
  public:
    RayDeformation(rhi::Device &device, std::shared_ptr<RayDeformationProgram> program, const RayRefitPlan &plan,
                   const RayInstancePage &instances, const std::vector<DrawCall> &draws, const RayPageUpload &prefix,
                   std::shared_ptr<const rhi::BufferResource> meshes)
        : m_device(device), m_program(std::move(program)), m_meshes(std::move(meshes)), m_plan(plan.buffer)
    {
        std::map<uint32_t, std::shared_ptr<rhi::ComputeBuffer>> sources;
        for (const auto &draw : draws)
            if (draw.meshVertexBuffer)
                sources.emplace(draw.surfaceId, draw.meshVertexBuffer);
        std::vector<RayGeometryWord> templateWords(static_cast<size_t>(prefix.bytes / 16));
        size_t offset = 0;
        for (size_t i = 0; i < RayPageUpload::sectionCount; ++i) {
            const auto section = prefix.Section(i);
            if (section.bytes)
                std::memcpy(reinterpret_cast<char *>(templateWords.data()) + offset, section.data, section.bytes);
            offset += section.bytes;
        }
        const auto tlasNodes = templateWords[0][1];
        const auto instanceOffset = templateWords[0][2];
        std::map<uint32_t, uint32_t> roots;
        uint32_t nextWord = static_cast<uint32_t>(templateWords.size());
        for (const auto &mesh : plan.meshes) {
            const auto header = nextWord;
            const auto nodes = header + 1u;
            const auto positions = nodes + 2u * mesh.nodes;
            const auto attributes = positions + mesh.vertices;
            const auto end = attributes + 4u * mesh.vertices;
            if (end >= spatial::kBvhLeafBit)
                throw std::length_error("Dynamic ray geometry exceeds tagged addresses");
            m_headers.push_back({header, {nodes, positions, attributes, mesh.vertexBase}});
            nextWord = end;
            roots.emplace(mesh.surfaceId, header | spatial::kBvhLeafBit);
            auto source = sources.at(mesh.surfaceId);
            m_sources.push_back(std::move(source));
            RayDeformationProgram::Constants constants{{0u, mesh.vertices, mesh.sources, 0u},
                                                       {nodes, positions, attributes, mesh.vertexBase},
                                                       {mesh.root, tlasNodes, instanceOffset, 0u}};
            const auto sourceIndex = static_cast<uint32_t>(m_sources.size() - 1);
            m_dispatches.push_back({sourceIndex, constants});
            constants.work = {1u, mesh.leaves.count, mesh.sources, mesh.leaves.offset};
            m_dispatches.push_back({sourceIndex, constants});
            for (const auto &level : mesh.levels) {
                constants.work = {2u, level.count, mesh.sources, level.offset};
                m_dispatches.push_back({sourceIndex, constants});
            }
        }
        for (size_t i = 0; i < instances.instances.size(); ++i) {
            const auto found = roots.find(instances.instances[i].surfaceId);
            if (found != roots.end())
                templateWords[instanceOffset + 4u * i + 3u][0] = found->second;
        }
        std::vector<uint32_t> topIndices;
        std::vector<std::vector<uint32_t>> topLevels;
        std::vector<uint32_t> depths(instances.nodes.size());
        for (size_t i = 0; i < instances.nodes.size(); ++i) {
            const auto &node = instances.nodes[i];
            if (node.second & spatial::kBvhLeafBit)
                topIndices.push_back(static_cast<uint32_t>(i));
            else {
                const auto depth = depths[i];
                if (topLevels.size() <= depth)
                    topLevels.resize(depth + 1);
                topLevels[depth].push_back(static_cast<uint32_t>(i));
                depths[node.first] = depths[node.second] = depth + 1;
            }
        }
        std::vector<RayRefitRange> ranges;
        ranges.push_back({0u, static_cast<uint32_t>(topIndices.size())});
        for (auto level = topLevels.rbegin(); level != topLevels.rend(); ++level)
            if (!level->empty())
                ranges.push_back(AppendRayRefitIndices(topIndices, *level));
        m_topPlan = AllocateRayRefitBuffer(device, topIndices.size() * 4u, rhi::BufferMemory::Upload);
        if (!device.WriteBuffer(m_topPlan->GetBuffer(), 0, topIndices.data(), m_topPlan->GetByteSize()))
            throw std::runtime_error("Could not upload the ray instance refit order");
        for (size_t i = 0; i < ranges.size(); ++i)
            m_dispatches.push_back(
                {UINT32_MAX,
                 {{i == 0 ? 3u : 4u, ranges[i].count, 0u, ranges[i].offset}, {}, {0u, tlasNodes, instanceOffset, 0u}}});
        m_output = AllocateRayRefitBuffer(device, uint64_t{nextWord} * 16u, rhi::BufferMemory::DeviceLocal);
        m_prefix = std::move(templateWords);
        for (const auto &source : m_sources)
            m_groups.push_back(m_program->Bind(source->GetBuffer(), m_plan->GetBuffer(), m_meshes->GetBuffer(),
                                               m_output->GetBuffer()));
        m_groups.push_back(m_program->Bind(m_sources.front()->GetBuffer(), m_topPlan->GetBuffer(),
                                           m_meshes->GetBuffer(), m_output->GetBuffer()));
    }

    ~RayDeformation()
    {
        for (auto group : m_groups)
            m_device.Release(group);
    }

    bool Submit()
    {
        auto &host = m_sources.front()->GetHost();
        rhi::ComputeSubmissionResources resources;
        resources.retained = shared_from_this();
        resources.receipt = m_receipt;
        const auto ticket = host.Queue().TrySubmit(
            [this](rhi::ComputeRecordingContext &context) {
                for (const auto &source : m_sources)
                    context.Require(source->GetLastWriteSubmission());
                auto transfer = context.Transfer();
                if (!transfer.UpdateBuffer(m_output->GetBuffer(), 0, m_prefix.data(), m_prefix.size() * 16u))
                    throw std::runtime_error("Could not initialize the private ray instance prefix");
                for (const auto &header : m_headers)
                    if (!transfer.UpdateBuffer(m_output->GetBuffer(), uint64_t{header.first} * 16u,
                                               header.second.data(), 16u))
                        throw std::runtime_error("Could not initialize the dynamic ray addresses");
                context.PipelineBarrier(rhi::PipelineStage::AllCommands, rhi::Access::MemoryWrite,
                                        rhi::PipelineStage::ComputeShader,
                                        rhi::Access::ShaderRead | rhi::Access::ShaderWrite);
                for (const auto &dispatch : m_dispatches) {
                    const auto group = dispatch.source == UINT32_MAX ? m_groups.back() : m_groups[dispatch.source];
                    m_program->Record(context.Compute(), group, dispatch.constants);
                    context.PipelineBarrier(rhi::PipelineStage::ComputeShader, rhi::Access::ShaderWrite,
                                            rhi::PipelineStage::ComputeShader,
                                            rhi::Access::ShaderRead | rhi::Access::ShaderWrite);
                }
                context.PipelineBarrier(rhi::PipelineStage::ComputeShader, rhi::Access::ShaderWrite,
                                        rhi::PipelineStage::AllCommands, rhi::Access::MemoryRead);
                return true;
            },
            resources);
        if (!ticket)
            return false;
        host.Queue().RecordDispatches(m_dispatches.size());
        host.Queue().RecordUpload(TransferBytes());
        m_submission = *ticket;
        return true;
    }

    bool IsComplete()
    {
        return m_submission.IsValid() && m_sources.front()->GetHost().Queue().IsComplete(m_submission);
    }
    uint64_t Elapsed() const noexcept
    {
        return m_receipt->deviceNanoseconds ? static_cast<uint64_t>(*m_receipt->deviceNanoseconds) : 0u;
    }
    uint64_t TransferBytes() const noexcept
    {
        return m_prefix.size() * 16u + m_headers.size() * 16u + m_topPlan->GetByteSize();
    }
    const auto &Output() const noexcept
    {
        return m_output;
    }
    rhi::SubmissionTicket Submission() const noexcept
    {
        return m_submission;
    }

  private:
    struct Dispatch
    {
        uint32_t source;
        RayDeformationProgram::Constants constants;
    };
    rhi::Device &m_device;
    std::shared_ptr<RayDeformationProgram> m_program;
    std::shared_ptr<const rhi::BufferResource> m_meshes, m_plan;
    std::shared_ptr<rhi::BufferResource> m_topPlan, m_output;
    std::vector<std::shared_ptr<rhi::ComputeBuffer>> m_sources;
    std::vector<rhi::BindGroupHandle> m_groups;
    std::vector<Dispatch> m_dispatches;
    std::vector<RayGeometryWord> m_prefix;
    std::vector<std::pair<uint32_t, RayGeometryWord>> m_headers;
    std::shared_ptr<rhi::ComputeExecutionReceipt> m_receipt = std::make_shared<rhi::ComputeExecutionReceipt>();
    rhi::SubmissionTicket m_submission;
};
} // namespace infernux::raytracing
