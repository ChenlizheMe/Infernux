#include "ParticleGpuRibbonTopology.h"
#include "ParticleComputePipeline.h"
#include "ParticleGpuSortModule.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace infernux::particle
{

namespace
{

constexpr std::string_view CommonBindings = R"slang(
struct ParticleRenderInstance {
    float4 position_size;
    float4 color;
    float4 rotation_custom;
    float4 scale_custom;
    uint4 ribbon_data;
    float4 custom_data;
    float4 previous_position_history;
};
[[vk::binding(0, 0)]] StructuredBuffer<ParticleRenderInstance> instances;
[[vk::binding(1, 0)]] StructuredBuffer<uint> source_indices;
struct SourceIndirectArgumentsData {
    uint source_vertex_count;
    uint source_instance_count;
    uint source_first_vertex;
    uint source_first_instance;
};
[[vk::binding(2, 0)]] StructuredBuffer<SourceIndirectArgumentsData> source_indirect_arguments;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> input_indices;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> output_indices;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> histograms;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> block_offsets;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> global_offsets;
struct DispatchArgumentsData {
    uint dispatch_group_count_x;
    uint dispatch_group_count_y;
    uint dispatch_group_count_z;
};
[[vk::binding(8, 0)]] RWStructuredBuffer<DispatchArgumentsData> dispatch_arguments;
struct RibbonIndirectArgumentsData {
    uint ribbon_vertex_count;
    uint ribbon_instance_count;
    uint ribbon_first_vertex;
    uint ribbon_first_instance;
};
[[vk::binding(9, 0)]] RWStructuredBuffer<RibbonIndirectArgumentsData> ribbon_indirect_arguments;
struct SimulationControlData {
    uint any_view_visible;
    uint simulation_allowed;
    uint offscreen_policy;
    uint simulation_control_reserved;
};
[[vk::binding(10, 0)]] StructuredBuffer<SimulationControlData> simulation_control;
struct RibbonConstants {
    uint capacity;
    uint block_count;
    uint key_field;
    uint digit_shift;
};
INX_ROOT_CONSTANTS ConstantBuffer<RibbonConstants> pc;
uint inx_live_count() { return min(source_indirect_arguments[0].source_instance_count, pc.capacity); }
uint inx_key(uint particle_index) {
    uint4 data = instances[particle_index].ribbon_data;
    return pc.key_field == 0u ? data.x : (pc.key_field == 1u ? data.y : data.w);
}
)slang";

std::string BuildShader(std::string_view body)
{
    std::string result(CommonBindings);
    result += body;
    return result;
}

uint32_t DivideRoundUp(uint32_t value, uint32_t divisor) noexcept
{
    return value == 0 ? 0 : 1u + (value - 1u) / divisor;
}

} // namespace

std::string_view GpuParticleRibbonShaderSources::Reset() noexcept
{
    static const std::string Source = BuildShader(R"slang([shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    // Reset is also the draw-safety GroupMemoryBarrierWithGroupSync. A newly created or paused
    // emitter must publish zero indirect arguments instead of leaving the
    // buffers with stale or uninitialized GPU contents.
    uint live_count = simulation_control[0].simulation_allowed != 0u ? inx_live_count() : 0u;
    dispatch_arguments[0].dispatch_group_count_x = (live_count + 255u) / 256u;
    dispatch_arguments[0].dispatch_group_count_y = 1u;
    dispatch_arguments[0].dispatch_group_count_z = 1u;
    ribbon_indirect_arguments[0].ribbon_vertex_count = live_count > 1u ? (live_count - 1u) * 6u : 0u;
    ribbon_indirect_arguments[0].ribbon_instance_count = live_count > 1u ? 1u : 0u;
    ribbon_indirect_arguments[0].ribbon_first_vertex = 0u;
    ribbon_indirect_arguments[0].ribbon_first_instance = 0u;
}
)slang");
    return Source;
}

std::string_view GpuParticleRibbonShaderSources::Initialize() noexcept
{
    static const std::string Source = BuildShader(R"slang([shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    if (simulation_control[0].simulation_allowed == 0u) return;
    uint index = dispatch_id.x;
    if (index < inx_live_count()) input_indices[index] = source_indices[index];
}
)slang");
    return Source;
}

std::string_view GpuParticleRibbonShaderSources::Histogram() noexcept
{
    static const std::string Source = BuildShader(R"slang(
groupshared Atomic<uint> local_histogram[16];
[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint local_index = group_thread_id.x;
    if (local_index < 16u) local_histogram[local_index].store(0u);
    GroupMemoryBarrierWithGroupSync();
    uint index = dispatch_id.x;
    uint live_count = simulation_control[0].simulation_allowed != 0u ? inx_live_count() : 0u;
    if (index < live_count) {
        uint digit = (inx_key(input_indices[index]) >> pc.digit_shift) & 15u;
        local_histogram[digit].add(1u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_index < 16u)
        histograms[group_id.x * 16u + local_index] = local_histogram[local_index].load();
}
)slang");
    return Source;
}

std::string_view GpuParticleRibbonShaderSources::Scan() noexcept
{
    static const std::string Source = BuildShader(R"slang(
groupshared uint totals[16];
[shader("compute")] [numthreads(16, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint bin = group_thread_id.x;
    uint running = 0u;
    uint active_block_count = simulation_control[0].simulation_allowed != 0u
                                 ? min(dispatch_arguments[0].dispatch_group_count_x, pc.block_count)
                                 : 0u;
    for (uint block = 0u; block < active_block_count; ++block) {
        uint offset = block * 16u + bin;
        uint count = histograms[offset];
        block_offsets[offset] = running;
        running += count;
    }
    totals[bin] = running;
    GroupMemoryBarrierWithGroupSync();
    uint base = 0u;
    for (uint prior = 0u; prior < bin; ++prior) base += totals[prior];
    global_offsets[bin] = base;
}
)slang");
    return Source;
}

std::string_view GpuParticleRibbonShaderSources::Scatter() noexcept
{
    static const std::string Source =
        BuildShader(std::string(SlangParticleRadixRank) + R"slang([shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint source_index = dispatch_id.x;
    bool active = simulation_control[0].simulation_allowed != 0u && source_index < inx_live_count();
    uint particle_index = active ? input_indices[source_index] : 0u;
    uint digit = active ? ((inx_key(particle_index) >> pc.digit_shift) & 15u) : 0u;
    uint local_rank = inx_radix_rank(group_index, digit, active);
    if (!active) return;
    uint block_offset = group_id.x * 16u + digit;
    uint destination = global_offsets[digit] + block_offsets[block_offset] + local_rank;
    output_indices[destination] = particle_index;
}
)slang");
    return Source;
}

bool GpuParticleRibbonProgram::IsValid() const noexcept
{
    return IsCompiledParticleShaderValid(reset) && IsCompiledParticleShaderValid(initialize) &&
           IsCompiledParticleShaderValid(histogram) && IsCompiledParticleShaderValid(scan) &&
           IsCompiledParticleShaderValid(scatter);
}

bool GpuParticleRibbonProgramStorage::Assign(const GpuParticleRibbonProgram &program)
{
    if (!program.IsValid())
        return false;
    const std::array<CompiledParticleShader, 5> sources = {program.reset, program.initialize, program.histogram,
                                                           program.scan, program.scatter};
    std::array<CompiledParticleShader, 5> candidate;
    for (size_t index = 0; index < sources.size(); ++index)
        candidate[index] = sources[index];
    shaders = std::move(candidate);
    return true;
}

bool GpuParticleRibbonProgramStorage::IsValid() const noexcept
{
    return View().IsValid();
}

GpuParticleRibbonProgram GpuParticleRibbonProgramStorage::View() const noexcept
{
    return {
        shaders[0], shaders[1], shaders[2], shaders[3], shaders[4],
    };
}

ParticleGpuRibbonTopology::ParticleGpuRibbonTopology() = default;

ParticleGpuRibbonTopology::~ParticleGpuRibbonTopology()
{
    Destroy();
}

bool ParticleGpuRibbonTopology::Create(rhi::Device &device, const GpuParticleRibbonDesc &desc)
{
    Destroy();
    if (desc.capacity == 0 || desc.capacity > std::numeric_limits<uint32_t>::max() / 6u || !desc.instances.IsValid() ||
        !desc.sourceIndices.IsValid() || !desc.sourceIndirectArguments.IsValid() || !desc.simulationControl.IsValid() ||
        !desc.program.IsValid())
        return false;

    m_device = &device;
    m_capacity = desc.capacity;
    m_blockCount = DivideRoundUp(desc.capacity, WorkgroupSize);
    m_instances = desc.instances;
    m_sourceIndices = desc.sourceIndices;
    m_sourceIndirectArguments = desc.sourceIndirectArguments;
    m_simulationControl = desc.simulationControl;

    const uint64_t elementBytes = static_cast<uint64_t>(m_capacity) * sizeof(uint32_t);
    const uint64_t blockBytes = static_cast<uint64_t>(m_blockCount) * Radix * sizeof(uint32_t);
    const auto storage = rhi::BufferUsageFlags::Storage;
    const auto storageIndirect = storage | rhi::BufferUsageFlags::Indirect;
    const auto createRenderExport = [&](uint64_t bytes, rhi::BufferUsageFlags usage) {
        rhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = bytes;
        bufferDesc.usage = usage;
        bufferDesc.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
        return device.CreateBuffer(bufferDesc);
    };
    for (auto &buffer : m_indices)
        buffer = createRenderExport(elementBytes, storage);
    m_drawIndirectArguments = createRenderExport(16, storageIndirect);
    m_dispatchArguments = createRenderExport(12, storageIndirect);
    m_histograms = createRenderExport(blockBytes, storage);
    m_blockOffsets = createRenderExport(blockBytes, storage);
    m_globalOffsets = createRenderExport(Radix * sizeof(uint32_t), storage);
    if (!std::all_of(m_indices.begin(), m_indices.end(), [](auto value) { return value.IsValid(); }) ||
        !m_drawIndirectArguments.IsValid() || !m_dispatchArguments.IsValid() || !m_histograms.IsValid() ||
        !m_blockOffsets.IsValid() || !m_globalOffsets.IsValid()) {
        Destroy();
        return false;
    }

    const std::array<CompiledParticleShader, 5> shaders = {
        desc.program.reset, desc.program.initialize, desc.program.histogram, desc.program.scan, desc.program.scatter};
    std::array<std::unique_ptr<ParticleComputePipeline> *, 5> pipelines = {
        &m_resetPipeline, &m_initializePipeline, &m_histogramPipeline, &m_scanPipeline, &m_scatterPipeline};
    for (size_t index = 0; index < shaders.size(); ++index) {
        auto program = std::make_unique<ParticleComputePipeline>(device, *shaders[index]);
        for (uint32_t pingPong = 0; pingPong < 2; ++pingPong) {
            const uint32_t output = 1u - pingPong;
            program->AddBindings(
                ParticleBufferBindings({m_instances, m_sourceIndices, m_sourceIndirectArguments, m_indices[pingPong],
                                        m_indices[output], m_histograms, m_blockOffsets, m_globalOffsets,
                                        m_dispatchArguments, m_drawIndirectArguments, m_simulationControl}));
        }
        *pipelines[index] = std::move(program);
    }
    return true;
}

void ParticleGpuRibbonTopology::Destroy() noexcept
{
    m_resetPipeline.reset();
    m_initializePipeline.reset();
    m_histogramPipeline.reset();
    m_scanPipeline.reset();
    m_scatterPipeline.reset();

    if (m_device) {
        m_device->Release(m_globalOffsets);
        m_device->Release(m_blockOffsets);
        m_device->Release(m_histograms);
        m_device->Release(m_dispatchArguments);
        m_device->Release(m_drawIndirectArguments);
        for (auto buffer : m_indices)
            m_device->Release(buffer);
    }
    m_device = nullptr;
    m_capacity = 0;
    m_blockCount = 0;
    m_instances = {};
    m_sourceIndices = {};
    m_sourceIndirectArguments = {};
    m_simulationControl = {};
    m_indices.fill({});
    m_drawIndirectArguments = {};
    m_dispatchArguments = {};
    m_histograms = {};
    m_blockOffsets = {};
    m_globalOffsets = {};
    m_resetPipeline = {};
    m_initializePipeline = {};
    m_histogramPipeline = {};
    m_scanPipeline = {};
    m_scatterPipeline = {};
}

bool ParticleGpuRibbonTopology::IsValid() const noexcept
{
    return m_device && m_resetPipeline && m_initializePipeline && m_histogramPipeline && m_scanPipeline &&
           m_scatterPipeline;
}

GpuParticleRibbonConstants ParticleGpuRibbonTopology::Constants(uint32_t passIndex) const noexcept
{
    GpuParticleRibbonConstants constants;
    constants.capacity = m_capacity;
    constants.blockCount = m_blockCount;
    const uint32_t keyPass = std::min(passIndex / PassesPerKey, KeyCount - 1u);
    constants.keyField = (KeyCount - 1u) - keyPass;
    constants.digitShift = (passIndex % PassesPerKey) * 4u;
    return constants;
}

void ParticleGpuRibbonTopology::RecordReset(const rhi::ComputeCommandEncoder &encoder) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    RecordDirect(encoder, *m_resetPipeline, 0u, Constants(), 1);
}

void ParticleGpuRibbonTopology::RecordInitialize(const rhi::ComputeCommandEncoder &encoder) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    RecordIndirect(encoder, *m_initializePipeline, 0u, Constants());
}

void ParticleGpuRibbonTopology::RecordHistogram(const rhi::ComputeCommandEncoder &encoder, uint32_t passIndex) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    RecordIndirect(encoder, *m_histogramPipeline, passIndex % 2u, Constants(passIndex));
}

void ParticleGpuRibbonTopology::RecordScan(const rhi::ComputeCommandEncoder &encoder, uint32_t passIndex) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    RecordDirect(encoder, *m_scanPipeline, passIndex % 2u, Constants(passIndex), 1);
}

void ParticleGpuRibbonTopology::RecordScatter(const rhi::ComputeCommandEncoder &encoder, uint32_t passIndex) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    RecordIndirect(encoder, *m_scatterPipeline, passIndex % 2u, Constants(passIndex));
}

void ParticleGpuRibbonTopology::RecordDirect(const rhi::ComputeCommandEncoder &encoder,
                                             const ParticleComputePipeline &pipeline, uint32_t variant,
                                             const GpuParticleRibbonConstants &constants, uint32_t groups) const
{
    pipeline.Bind(encoder, constants, variant);
    encoder.Dispatch(groups, 1, 1);
}

void ParticleGpuRibbonTopology::RecordIndirect(const rhi::ComputeCommandEncoder &encoder,
                                               const ParticleComputePipeline &pipeline, uint32_t variant,
                                               const GpuParticleRibbonConstants &constants) const
{
    pipeline.Bind(encoder, constants, variant);
    encoder.DispatchIndirect(m_dispatchArguments);
}

} // namespace infernux::particle
