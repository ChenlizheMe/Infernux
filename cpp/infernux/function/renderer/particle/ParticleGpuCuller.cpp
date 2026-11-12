#include "ParticleGpuCuller.h"
#include "ParticleComputePipeline.h"

#include <array>
#include <string>

namespace infernux::particle
{

namespace
{

constexpr std::string_view CommonBindings = R"slang(
struct ParticleVisibilityInstance {
    float4 position_radius;
};
struct ParticleRenderInstance {
    float4 position_size;
    float4 color;
    float4 rotation_custom;
    float4 scale_custom;
    uint4 ribbon_data;
    float4 custom_data;
    float4 previous_position_history;
};
[[vk::binding(0, 0)]] StructuredBuffer<ParticleVisibilityInstance> visibility;
struct SourceIndirectArgumentsData {
    uint source_vertex_count;
    uint source_instance_count;
    uint source_first_vertex;
    uint source_first_instance;
};
[[vk::binding(1, 0)]] StructuredBuffer<SourceIndirectArgumentsData> source_indirect_arguments;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> visible_indices;
struct DrawIndirectArgumentsData {
    uint draw_vertex_count;
    Atomic<uint> draw_instance_count;
    uint draw_first_vertex;
    uint draw_first_instance;
};
[[vk::binding(3, 0)]] RWStructuredBuffer<DrawIndirectArgumentsData> draw_indirect_arguments;
struct SortDispatchArgumentsData {
    uint sort_group_count_x;
    uint sort_group_count_y;
    uint sort_group_count_z;
    uint stats_source_count;
    uint stats_flags;
};
[[vk::binding(4, 0)]] INX_CULL_ARGUMENTS<SortDispatchArgumentsData> sort_dispatch_arguments;
struct BoundsData {
    uint bounds_min_x;
    uint bounds_min_y;
    uint bounds_min_z;
    uint bounds_max_x;
    uint bounds_max_y;
    uint bounds_max_z;
    uint bounds_valid;
    uint bounds_reserved;
};
[[vk::binding(5, 0)]] StructuredBuffer<BoundsData> bounds;
struct SimulationControlData {
    Atomic<uint> any_view_visible;
    uint simulation_allowed;
    uint offscreen_policy;
    uint simulation_control_reserved;
};
[[vk::binding(6, 0)]] RWStructuredBuffer<SimulationControlData> simulation_control;
[[vk::binding(7, 0)]] StructuredBuffer<uint> source_indices;
[[vk::binding(8, 0)]] StructuredBuffer<ParticleRenderInstance> ribbon_instances;
struct CullConstants {
    float4 frustum_planes[6];
    uint capacity;
    uint vertex_count;
    uint mode;
    uint view_enabled;
};
INX_ROOT_CONSTANTS ConstantBuffer<CullConstants> pc;
)slang";

std::string BuildShader(std::string_view body, bool writableArguments = true)
{
    std::string result = writableArguments ? "#define INX_CULL_ARGUMENTS RWStructuredBuffer\n"
                                           : "#define INX_CULL_ARGUMENTS StructuredBuffer\n";
    result += CommonBindings;
    result += body;
    return result;
}

} // namespace

std::string_view GpuParticleCullShaderSources::Reset() noexcept
{
    static const std::string Source = BuildShader(R"slang([shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    bool ribbon_segments = pc.mode == 1u;
    draw_indirect_arguments[0].draw_vertex_count = ribbon_segments ? 6u : pc.vertex_count;
    draw_indirect_arguments[0].draw_instance_count.store(0u);
    draw_indirect_arguments[0].draw_first_vertex = source_indirect_arguments[0].source_first_vertex;
    draw_indirect_arguments[0].draw_first_instance = source_indirect_arguments[0].source_first_instance;
    uint source_count = ribbon_segments
        ? min(source_indirect_arguments[0].source_vertex_count / 6u, pc.capacity > 0u ? pc.capacity - 1u : 0u)
        : min(source_indirect_arguments[0].source_instance_count, pc.capacity);
    sort_dispatch_arguments[0].stats_source_count = source_count;
    sort_dispatch_arguments[0].stats_flags = (bounds[0].bounds_valid != 0u ? 1u : 0u) | (ribbon_segments ? 4u : 0u);
    bool bounds_visible = true;
    bool bounds_fully_inside = false;
    if (bounds[0].bounds_valid != 0u) {
        uint ordered_min[3] = {bounds[0].bounds_min_x, bounds[0].bounds_min_y, bounds[0].bounds_min_z};
        uint ordered_max[3] = {bounds[0].bounds_max_x, bounds[0].bounds_max_y, bounds[0].bounds_max_z};
        float3 lower;
        float3 upper;
        for (uint axis = 0u; axis < 3u; ++axis) {
            uint min_bits = (ordered_min[axis] & 0x80000000u) != 0u
                                ? (ordered_min[axis] ^ 0x80000000u)
                                : ~ordered_min[axis];
            uint max_bits = (ordered_max[axis] & 0x80000000u) != 0u
                                ? (ordered_max[axis] ^ 0x80000000u)
                                : ~ordered_max[axis];
            lower[axis] = asfloat(min_bits);
            upper[axis] = asfloat(max_bits);
        }
        bounds_fully_inside = true;
        for (uint plane_index = 0u; plane_index < 6u; ++plane_index) {
            float4 plane = pc.frustum_planes[plane_index];
            float3 positive_vertex = select(plane.xyz >= float3(0.0), upper, lower);
            if (dot(plane.xyz, positive_vertex) + plane.w < 0.0)
                bounds_visible = false;
            float3 negative_vertex = select(plane.xyz < float3(0.0), upper, lower);
            if (dot(plane.xyz, negative_vertex) + plane.w < 0.0)
                bounds_fully_inside = false;
        }
    }
    bool visible_in_view = pc.view_enabled != 0u && (bounds[0].bounds_valid == 0u || bounds_visible);
    if (visible_in_view) sort_dispatch_arguments[0].stats_flags |= 2u;
    // Reset publishes immutable culling flags separately from the counter.
    // All workgroups consume the same flag while atomically appending indices.
    if (bounds_fully_inside) sort_dispatch_arguments[0].stats_flags |= 8u;
    if (visible_in_view) simulation_control[0].any_view_visible.or(1u);
    sort_dispatch_arguments[0].sort_group_count_x = visible_in_view ? (source_count + 255u) / 256u : 0u;
    sort_dispatch_arguments[0].sort_group_count_y = 1u;
    sort_dispatch_arguments[0].sort_group_count_z = 1u;
}
)slang");
    return Source;
}

std::string_view GpuParticleCullShaderSources::Cull() noexcept
{
    static const std::string Source = BuildShader(R"slang(
groupshared Atomic<uint> local_visible_count;
groupshared uint global_base;

bool inx_visible_sphere(float3 center, float radius) {
    for (uint plane_index = 0u; plane_index < 6u; ++plane_index) {
        float4 plane = pc.frustum_planes[plane_index];
        if (dot(plane.xyz, center) + plane.w < -radius) return false;
    }
    return true;
}
[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    bool bounds_fully_inside = (sort_dispatch_arguments[0].stats_flags & 8u) != 0u;
    if (group_index == 0u) {
        local_visible_count.store(0u);
        global_base = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint source_index = dispatch_id.x;
    bool ribbon_segments = pc.mode == 1u;
    uint source_count = ribbon_segments
        ? min(source_indirect_arguments[0].source_vertex_count / 6u, pc.capacity > 0u ? pc.capacity - 1u : 0u)
        : min(source_indirect_arguments[0].source_instance_count, pc.capacity);
    bool visible = false;
    uint output_value = 0u;
    if (ribbon_segments) {
        if (source_index < source_count) {
            uint first_index = source_indices[source_index];
            uint second_index = source_indices[source_index + 1u];
            if (first_index < pc.capacity && second_index < pc.capacity) {
                ParticleRenderInstance first_metadata = ribbon_instances[first_index];
                ParticleRenderInstance second_metadata = ribbon_instances[second_index];
                ParticleVisibilityInstance first = visibility[first_index];
                ParticleVisibilityInstance second = visibility[second_index];
                if (first_metadata.ribbon_data.x == second_metadata.ribbon_data.x &&
                    (second_metadata.ribbon_data.z & 1u) == 0u) {
                    visible = bounds_fully_inside || inx_visible_sphere(
                                                          (first.position_radius.xyz + second.position_radius.xyz) * 0.5,
                                                          length(second.position_radius.xyz - first.position_radius.xyz) * 0.5 +
                                                              max(abs(first.position_radius.w), abs(second.position_radius.w)) *
                                                                  0.5);
                    output_value = source_index;
                }
            }
        }
    } else {
        if (source_index < source_count) {
            uint particle_index = source_indices[source_index];
            if (particle_index < pc.capacity) {
                ParticleVisibilityInstance instance = visibility[particle_index];
                visible = bounds_fully_inside || inx_visible_sphere(
                                                   instance.position_radius.xyz,
                                                   abs(instance.position_radius.w));
                output_value = particle_index;
            }
        }
    }

    uint local_output_index = 0u;
    if (visible) local_output_index = local_visible_count.add(1u);
    GroupMemoryBarrierWithGroupSync();
    if (group_index == 0u) global_base = draw_indirect_arguments[0].draw_instance_count.add(local_visible_count.load());
    GroupMemoryBarrierWithGroupSync();

    if (visible && global_base < pc.capacity && local_output_index < pc.capacity - global_base) {
        visible_indices[global_base + local_output_index] = output_value;
    }
}
)slang",
                                                  false);
    return Source;
}

std::string_view GpuParticleCullShaderSources::Finalize() noexcept
{
    static const std::string Source = BuildShader(R"slang([shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    // A skipped (empty or coarse-rejected) Cull retains Reset's zero count.
    uint visible_count = min(draw_indirect_arguments[0].draw_instance_count.load(), pc.capacity);
    draw_indirect_arguments[0].draw_instance_count.store(visible_count);
    sort_dispatch_arguments[0].sort_group_count_x = (visible_count + 255u) / 256u;
    sort_dispatch_arguments[0].sort_group_count_y = 1u;
    sort_dispatch_arguments[0].sort_group_count_z = 1u;
}
)slang");
    return Source;
}

bool GpuParticleCullProgram::IsValid() const noexcept
{
    return IsCompiledParticleShaderValid(reset) && IsCompiledParticleShaderValid(cull) &&
           IsCompiledParticleShaderValid(finalize);
}

bool GpuParticleCullProgramStorage::Assign(const GpuParticleCullProgram &program)
{
    if (!program.IsValid())
        return false;
    const std::array<CompiledParticleShader, 3> sources = {program.reset, program.cull, program.finalize};
    std::array<CompiledParticleShader, 3> candidate;
    for (size_t index = 0; index < sources.size(); ++index)
        candidate[index] = sources[index];
    shaders = std::move(candidate);
    return true;
}

bool GpuParticleCullProgramStorage::IsValid() const noexcept
{
    return View().IsValid();
}

GpuParticleCullProgram GpuParticleCullProgramStorage::View() const noexcept
{
    return {
        shaders[0],
        shaders[1],
        shaders[2],
    };
}

ParticleGpuCuller::ParticleGpuCuller() = default;

ParticleGpuCuller::~ParticleGpuCuller()
{
    Destroy();
}

bool ParticleGpuCuller::Create(rhi::Device &device, const GpuParticleCullerDesc &desc)
{
    Destroy();
    if (desc.capacity == 0 || desc.vertexCount == 0 || !desc.visibility.IsValid() || !desc.ribbonInstances.IsValid() ||
        !desc.sourceIndirectArguments.IsValid() || !desc.sourceIndices.IsValid() || !desc.bounds.IsValid() ||
        !desc.simulationControl.IsValid() || !desc.program.IsValid()) {
        return false;
    }

    m_device = &device;
    m_capacity = desc.capacity;
    m_vertexCount = desc.vertexCount;
    m_visibility = desc.visibility;
    m_ribbonInstances = desc.ribbonInstances;
    m_sourceIndirectArguments = desc.sourceIndirectArguments;
    m_sourceIndices = desc.sourceIndices;
    m_bounds = desc.bounds;
    m_simulationControl = desc.simulationControl;
    m_mode = desc.mode;
    const auto storage = rhi::BufferUsageFlags::Storage;
    const auto createSharedStorage = [&](uint64_t bytes, rhi::BufferUsageFlags usage) {
        rhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = bytes;
        bufferDesc.usage = usage;
        bufferDesc.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
        return device.CreateBuffer(bufferDesc);
    };
    m_visibleIndices = createSharedStorage(static_cast<uint64_t>(desc.capacity) * sizeof(uint32_t), storage);
    m_drawIndirectArguments =
        createSharedStorage(16, storage | rhi::BufferUsageFlags::Indirect | rhi::BufferUsageFlags::TransferSource);
    m_sortDispatchArguments =
        createSharedStorage(sizeof(GpuParticleCullDispatchState),
                            storage | rhi::BufferUsageFlags::Indirect | rhi::BufferUsageFlags::TransferSource);
    if (!m_visibleIndices.IsValid() || !m_drawIndirectArguments.IsValid() || !m_sortDispatchArguments.IsValid()) {
        Destroy();
        return false;
    }

    const std::array<CompiledParticleShader, 3> shaders = {desc.program.reset, desc.program.cull,
                                                           desc.program.finalize};
    std::array<std::unique_ptr<ParticleComputePipeline> *, 3> pipelines = {&m_resetPipeline, &m_cullPipeline,
                                                                           &m_finalizePipeline};
    for (size_t index = 0; index < shaders.size(); ++index) {
        auto program = std::make_unique<ParticleComputePipeline>(device, *shaders[index]);
        program->AddBindings(ParticleBufferBindings({m_visibility, m_sourceIndirectArguments, m_visibleIndices,
                                                     m_drawIndirectArguments, m_sortDispatchArguments, m_bounds,
                                                     m_simulationControl, m_sourceIndices, m_ribbonInstances}));
        *pipelines[index] = std::move(program);
    }
    return true;
}

void ParticleGpuCuller::Destroy() noexcept
{
    m_resetPipeline.reset();
    m_cullPipeline.reset();
    m_finalizePipeline.reset();

    if (m_device) {
        m_device->Release(m_sortDispatchArguments);
        m_device->Release(m_drawIndirectArguments);
        m_device->Release(m_visibleIndices);
    }
    m_device = nullptr;
    m_capacity = 0;
    m_vertexCount = 0;
    m_visibility = {};
    m_ribbonInstances = {};
    m_sourceIndirectArguments = {};
    m_sourceIndices = {};
    m_bounds = {};
    m_simulationControl = {};
    m_mode = GpuParticleCullMode::Instances;
    m_visibleIndices = {};
    m_drawIndirectArguments = {};
    m_sortDispatchArguments = {};
    m_resetPipeline = {};
    m_cullPipeline = {};
    m_finalizePipeline = {};
}

bool ParticleGpuCuller::IsValid() const noexcept
{
    return m_device && m_resetPipeline && m_cullPipeline && m_finalizePipeline;
}

void ParticleGpuCuller::RecordReset(const rhi::ComputeCommandEncoder &encoder,
                                    const std::array<float, PlaneCount * 4> &frustumPlanes, bool viewEnabled) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    GpuParticleCullConstants constants;
    constants.frustumPlanes = frustumPlanes;
    constants.capacity = m_capacity;
    constants.vertexCount = m_vertexCount;
    constants.mode = m_mode;
    constants.viewEnabled = viewEnabled ? 1u : 0u;
    Record(encoder, *m_resetPipeline, constants, 1);
}

void ParticleGpuCuller::RecordCull(const rhi::ComputeCommandEncoder &encoder,
                                   const std::array<float, PlaneCount * 4> &frustumPlanes) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    GpuParticleCullConstants constants;
    constants.frustumPlanes = frustumPlanes;
    constants.capacity = m_capacity;
    constants.vertexCount = m_vertexCount;
    constants.mode = m_mode;
    m_cullPipeline->Bind(encoder, constants);
    encoder.DispatchIndirect(m_sortDispatchArguments);
}

void ParticleGpuCuller::RecordFinalize(const rhi::ComputeCommandEncoder &encoder) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    GpuParticleCullConstants constants;
    constants.capacity = m_capacity;
    constants.vertexCount = m_vertexCount;
    constants.mode = m_mode;
    Record(encoder, *m_finalizePipeline, constants, 1);
}

void ParticleGpuCuller::Record(const rhi::ComputeCommandEncoder &encoder, const ParticleComputePipeline &pipeline,
                               const GpuParticleCullConstants &constants, uint32_t groups) const
{
    pipeline.Bind(encoder, constants);
    encoder.Dispatch(groups, 1, 1);
}

} // namespace infernux::particle
