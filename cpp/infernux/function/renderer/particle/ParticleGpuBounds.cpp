#include "ParticleGpuBounds.h"
#include "ParticleComputePipeline.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

namespace infernux::particle
{

namespace
{

constexpr std::string_view CommonBindings = R"slang(
struct ParticleVisibilityInstance {
    float4 position_radius;
};
[[vk::binding(0, 0)]] StructuredBuffer<ParticleVisibilityInstance> visibility;
struct SourceIndirectArgumentsData {
    uint source_vertex_count;
    uint source_instance_count;
    uint source_first_vertex;
    uint source_first_instance;
};
[[vk::binding(1, 0)]] StructuredBuffer<SourceIndirectArgumentsData> source_indirect_arguments;
struct BoundsData {
    Atomic<uint> min_x;
    Atomic<uint> min_y;
    Atomic<uint> min_z;
    Atomic<uint> max_x;
    Atomic<uint> max_y;
    Atomic<uint> max_z;
    uint bounds_valid;
    uint bounds_reserved;
};
[[vk::binding(2, 0)]] RWStructuredBuffer<BoundsData> bounds;
struct DispatchArgumentsData {
    uint dispatch_group_count_x;
    uint dispatch_group_count_y;
    uint dispatch_group_count_z;
};
[[vk::binding(3, 0)]] RWStructuredBuffer<DispatchArgumentsData> dispatch_arguments;
struct SimulationControlData {
    Atomic<uint> any_view_visible;
    uint simulation_allowed;
    uint offscreen_policy;
    uint simulation_control_reserved;
};
[[vk::binding(4, 0)]] RWStructuredBuffer<SimulationControlData> simulation_control;
[[vk::binding(5, 0)]] StructuredBuffer<uint> source_indices;
struct BoundsConstants {
    float4 manual_lower;
    float4 manual_upper;
    uint capacity;
    uint bounds_mode;
    uint requested_offscreen_policy;
    uint force_simulation;
};
INX_ROOT_CONSTANTS ConstantBuffer<BoundsConstants> pc;
)slang";

std::string BuildShader(std::string_view body)
{
    std::string result(CommonBindings);
    result += body;
    return result;
}

} // namespace

std::string_view GpuParticleBoundsShaderSources::Prepare() noexcept
{
    static const std::string Source = BuildShader(R"slang([shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    // Always-simulate emitters do not consume visibility feedback. Leaving
    // that word to the graphics culler removes a cross-queue read/write race
    // while async simulation overlaps the previous frame's rendering.
    uint was_visible = pc.requested_offscreen_policy == 0u ? 1u : simulation_control[0].any_view_visible.exchange(0u);
    simulation_control[0].offscreen_policy = pc.requested_offscreen_policy;
    simulation_control[0].simulation_allowed = pc.requested_offscreen_policy == 0u || was_visible != 0u || pc.force_simulation != 0u
                             ? 1u
                             : 0u;
    simulation_control[0].simulation_control_reserved = 0u;
}
)slang");
    return Source;
}

std::string_view GpuParticleBoundsShaderSources::Reset() noexcept
{
    static const std::string Source = BuildShader(R"slang(
uint inx_ordered_float(float value) {
    uint bits = asuint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}
[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint source_count = min(source_indirect_arguments[0].source_instance_count, pc.capacity);
    if (pc.bounds_mode == 1u) {
        bounds[0].min_x.store(inx_ordered_float(pc.manual_lower.x));
        bounds[0].min_y.store(inx_ordered_float(pc.manual_lower.y));
        bounds[0].min_z.store(inx_ordered_float(pc.manual_lower.z));
        bounds[0].max_x.store(inx_ordered_float(pc.manual_upper.x));
        bounds[0].max_y.store(inx_ordered_float(pc.manual_upper.y));
        bounds[0].max_z.store(inx_ordered_float(pc.manual_upper.z));
        bounds[0].bounds_valid = 1u;
        dispatch_arguments[0].dispatch_group_count_x = 0u;
    } else if (simulation_control[0].simulation_allowed != 0u) {
        bounds[0].min_x.store(0xffffffffu);
        bounds[0].min_y.store(0xffffffffu);
        bounds[0].min_z.store(0xffffffffu);
        bounds[0].max_x.store(0u);
        bounds[0].max_y.store(0u);
        bounds[0].max_z.store(0u);
        bounds[0].bounds_valid = source_count > 0u ? 1u : 0u;
        dispatch_arguments[0].dispatch_group_count_x = (source_count + 255u) / 256u;
    } else {
        dispatch_arguments[0].dispatch_group_count_x = 0u;
    }
    bounds[0].bounds_reserved = 0u;
    dispatch_arguments[0].dispatch_group_count_y = 1u;
    dispatch_arguments[0].dispatch_group_count_z = 1u;
}
)slang");
    return Source;
}

std::string_view GpuParticleBoundsShaderSources::Reduce() noexcept
{
    static const std::string Source = BuildShader(R"slang(
uint inx_ordered_float(float value) {
    uint bits = asuint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}
bool inx_is_finite(float value) {
    return !isnan(value) && !isinf(value);
}
bool inx_is_finite_vec3(float3 value) {
    return inx_is_finite(value.x) && inx_is_finite(value.y) && inx_is_finite(value.z);
}
groupshared uint3 shared_lower[256];
groupshared uint3 shared_upper[256];
[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint local_index = group_thread_id.x;
    uint3 lower_bits = uint3(0xffffffffu);
    uint3 upper_bits = uint3(0u);
    uint index = dispatch_id.x;
    uint source_count = min(source_indirect_arguments[0].source_instance_count, pc.capacity);
    if (simulation_control[0].simulation_allowed != 0u && index < source_count) {
        uint particle_index = source_indices[index];
        if (particle_index < pc.capacity) {
            ParticleVisibilityInstance instance = visibility[particle_index];
            float3 position = instance.position_radius.xyz;
            float radius = instance.position_radius.w;
            bool valid = inx_is_finite_vec3(position) && inx_is_finite(radius);
            if (valid) {
                valid = radius >= 0.0;
                if (valid) {
                    float3 bounds_lower = position - float3(radius);
                    float3 bounds_upper = position + float3(radius);
                    valid = inx_is_finite_vec3(bounds_lower) && inx_is_finite_vec3(bounds_upper);
                    if (valid) {
                        lower_bits = uint3(inx_ordered_float(bounds_lower.x), inx_ordered_float(bounds_lower.y),
                                           inx_ordered_float(bounds_lower.z));
                        upper_bits = uint3(inx_ordered_float(bounds_upper.x), inx_ordered_float(bounds_upper.y),
                                           inx_ordered_float(bounds_upper.z));
                    }
                }
            }
        }
    }

    shared_lower[local_index] = lower_bits;
    shared_upper[local_index] = upper_bits;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_index < stride) {
            shared_lower[local_index] = min(shared_lower[local_index], shared_lower[local_index + stride]);
            shared_upper[local_index] = max(shared_upper[local_index], shared_upper[local_index + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (simulation_control[0].simulation_allowed != 0u && local_index == 0u) {
        bounds[0].min_x.min(shared_lower[0].x);
        bounds[0].min_y.min(shared_lower[0].y);
        bounds[0].min_z.min(shared_lower[0].z);
        bounds[0].max_x.max(shared_upper[0].x);
        bounds[0].max_y.max(shared_upper[0].y);
        bounds[0].max_z.max(shared_upper[0].z);
    }
}
)slang");
    return Source;
}

bool GpuParticleBoundsProgram::IsValid() const noexcept
{
    return IsCompiledParticleShaderValid(prepare) && IsCompiledParticleShaderValid(reset) &&
           IsCompiledParticleShaderValid(reduce);
}

bool GpuParticleBoundsProgramStorage::Assign(const GpuParticleBoundsProgram &program)
{
    if (!program.IsValid())
        return false;
    const std::array<CompiledParticleShader, 3> sources = {program.prepare, program.reset, program.reduce};
    std::array<CompiledParticleShader, 3> candidate;
    for (size_t index = 0; index < sources.size(); ++index)
        candidate[index] = sources[index];
    shaders = std::move(candidate);
    return true;
}

bool GpuParticleBoundsProgramStorage::IsValid() const noexcept
{
    return View().IsValid();
}

GpuParticleBoundsProgram GpuParticleBoundsProgramStorage::View() const noexcept
{
    return {
        shaders[0],
        shaders[1],
        shaders[2],
    };
}

ParticleGpuBounds::ParticleGpuBounds() = default;

ParticleGpuBounds::~ParticleGpuBounds()
{
    Destroy();
}

bool ParticleGpuBounds::Create(rhi::Device &device, const GpuParticleBoundsDesc &desc)
{
    Destroy();
    if (desc.capacity == 0 || !desc.visibility.IsValid() || !desc.sourceIndices.IsValid() ||
        !desc.sourceIndirectArguments.IsValid() || !desc.simulationControl.IsValid() || !desc.program.IsValid()) {
        return false;
    }

    m_device = &device;
    m_capacity = desc.capacity;
    m_visibility = desc.visibility;
    m_sourceIndices = desc.sourceIndices;
    m_sourceIndirectArguments = desc.sourceIndirectArguments;
    m_simulationControl = desc.simulationControl;
    const auto storage = rhi::BufferUsageFlags::Storage;
    const auto createSharedStorage = [&](uint64_t bytes, rhi::BufferUsageFlags usage) {
        rhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = bytes;
        bufferDesc.usage = usage;
        // Prepare/Reset run inside GpuParticle/Simulation on the independent
        // compute family. A Graphics-only exclusive dispatch buffer hangs that
        // queue the first time Game-only preview creates a new emitter.
        bufferDesc.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
        return device.CreateBuffer(bufferDesc);
    };
    m_bounds = createSharedStorage(BoundsBufferBytes, storage | rhi::BufferUsageFlags::TransferSource);
    m_dispatchArguments = createSharedStorage(DispatchBufferBytes, storage | rhi::BufferUsageFlags::Indirect);
    if (!m_bounds.IsValid() || !m_dispatchArguments.IsValid()) {
        Destroy();
        return false;
    }

    const std::array<CompiledParticleShader, 3> shaders = {desc.program.prepare, desc.program.reset,
                                                           desc.program.reduce};
    std::array<std::unique_ptr<ParticleComputePipeline> *, 3> pipelines = {&m_preparePipeline, &m_resetPipeline,
                                                                           &m_reducePipeline};
    for (size_t index = 0; index < shaders.size(); ++index) {
        auto program = std::make_unique<ParticleComputePipeline>(device, *shaders[index]);
        program->AddBindings(ParticleBufferBindings({m_visibility, m_sourceIndirectArguments, m_bounds,
                                                     m_dispatchArguments, m_simulationControl, m_sourceIndices}));
        *pipelines[index] = std::move(program);
    }
    return true;
}

void ParticleGpuBounds::Destroy() noexcept
{
    m_preparePipeline.reset();
    m_resetPipeline.reset();
    m_reducePipeline.reset();

    if (m_device) {
        m_device->Release(m_dispatchArguments);
        m_device->Release(m_bounds);
    }
    m_device = nullptr;
    m_capacity = 0;
    m_visibility = {};
    m_sourceIndices = {};
    m_sourceIndirectArguments = {};
    m_simulationControl = {};
    m_bounds = {};
    m_dispatchArguments = {};
    m_preparePipeline = {};
    m_resetPipeline = {};
    m_reducePipeline = {};
    m_controlPrepared = false;
    m_preparedPolicy = GpuParticleOffscreenPolicy::AlwaysSimulate;
    m_preparedForceSimulation = false;
}

bool ParticleGpuBounds::IsValid() const noexcept
{
    return m_device && m_preparePipeline && m_resetPipeline && m_reducePipeline;
}

void ParticleGpuBounds::RecordPrepare(const rhi::ComputeCommandEncoder &encoder, GpuParticleOffscreenPolicy policy,
                                      bool forceSimulation) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    if (m_controlPrepared && policy == GpuParticleOffscreenPolicy::AlwaysSimulate &&
        m_preparedPolicy == GpuParticleOffscreenPolicy::AlwaysSimulate && m_preparedForceSimulation == forceSimulation)
        return;
    GpuParticleBoundsConstants constants;
    constants.capacity = m_capacity;
    constants.offscreenPolicy = policy;
    constants.forceSimulation = forceSimulation ? 1u : 0u;
    Bind(encoder, *m_preparePipeline, constants);
    encoder.Dispatch(1, 1, 1);
    m_controlPrepared = true;
    m_preparedPolicy = policy;
    m_preparedForceSimulation = forceSimulation;
}

void ParticleGpuBounds::RecordReset(const rhi::ComputeCommandEncoder &encoder, GpuParticleBoundsMode mode,
                                    const std::array<float, 3> &manualLower,
                                    const std::array<float, 3> &manualUpper) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    GpuParticleBoundsConstants constants;
    constants.capacity = m_capacity;
    constants.boundsMode = mode;
    std::copy(manualLower.begin(), manualLower.end(), constants.manualLower.begin());
    std::copy(manualUpper.begin(), manualUpper.end(), constants.manualUpper.begin());
    Bind(encoder, *m_resetPipeline, constants);
    encoder.Dispatch(1, 1, 1);
}

void ParticleGpuBounds::RecordReduce(const rhi::ComputeCommandEncoder &encoder) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    GpuParticleBoundsConstants constants;
    constants.capacity = m_capacity;
    Bind(encoder, *m_reducePipeline, constants);
    encoder.DispatchIndirect(m_dispatchArguments);
}

void ParticleGpuBounds::Bind(const rhi::ComputeCommandEncoder &encoder, const ParticleComputePipeline &pipeline,
                             const GpuParticleBoundsConstants &constants) const
{
    pipeline.Bind(encoder, constants);
}

} // namespace infernux::particle
