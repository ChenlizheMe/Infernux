#include "ParticleGpuSorter.h"
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
struct ParticleVisibilityInstance {
    float4 position_radius;
};
[[vk::binding(0, 0)]] StructuredBuffer<ParticleVisibilityInstance> visibility;
struct IndirectArgumentsData {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};
[[vk::binding(1, 0)]] StructuredBuffer<IndirectArgumentsData> indirect_args;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> input_keys;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> input_indices;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> output_keys;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> output_indices;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> histograms;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> block_offsets;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> global_offsets;
[[vk::binding(9, 0)]] StructuredBuffer<uint> source_indices;
struct DispatchArgumentsData {
    uint dispatch_group_count_x;
    uint dispatch_group_count_y;
    uint dispatch_group_count_z;
};
[[vk::binding(10, 0)]] StructuredBuffer<DispatchArgumentsData> dispatch_arguments;
struct SortConstants {
    column_major float4x4 view;
    uint capacity;
    uint block_count;
    uint digit_shift;
    uint descending;
};
INX_ROOT_CONSTANTS ConstantBuffer<SortConstants> pc;
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

std::string_view GpuParticleSortShaderSources::Small() noexcept
{
    static const std::string Source = BuildShader(R"slang(
static const uint INX_SMALL_SORT_CAPACITY = 1024u;
static const uint INX_SMALL_SORT_LANES = 256u;
static const uint INX_SMALL_SORT_VALUES_PER_LANE = 4u;
groupshared uint2 shared_keys[INX_SMALL_SORT_CAPACITY];
groupshared uint shared_indices[INX_SMALL_SORT_CAPACITY];

uint inx_ordered_float(float value) {
    uint bits = asuint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

bool inx_greater(uint2 left_key, uint left_index, uint2 right_key, uint right_index) {
    return left_key.x > right_key.x ||
           (left_key.x == right_key.x &&
            (left_key.y > right_key.y || (left_key.y == right_key.y && left_index > right_index)));
}

bool inx_less(uint2 left_key, uint left_index, uint2 right_key, uint right_index) {
    return inx_greater(right_key, right_index, left_key, left_index);
}

uint2 inx_particle_sort_key(float view_depth, uint particle_id) {
    uint depth_key = inx_ordered_float(view_depth);
    depth_key = pc.descending != 0u ? ~depth_key : depth_key;
    return uint2(depth_key, particle_id);
}

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint lane = group_thread_id.x;
    uint live_count = min(indirect_args[0].instance_count, pc.capacity);
    for (uint item = 0u; item < INX_SMALL_SORT_VALUES_PER_LANE; ++item) {
        uint index = lane + item * INX_SMALL_SORT_LANES;
        uint particle_index = index < live_count ? source_indices[index] : 0xffffffffu;
        uint2 key = uint2(0xffffffffu, 0xffffffffu);
        if (index < live_count) {
            float4 view_position = mul(pc.view, float4(visibility[particle_index].position_radius.xyz, 1.0));
            key = inx_particle_sort_key(view_position.z, particle_index);
        }
        shared_keys[index] = key;
        shared_indices[index] = particle_index;
    }
    GroupMemoryBarrierWithGroupSync();

    // Sort the full power-of-two workspace. Invalid lanes carry the maximum
    // tuple and therefore remain at the tail; every compare/exchange pair is
    // handled by exactly one lane before the GroupMemoryBarrierWithGroupSync.
    for (uint k = 2u; k <= INX_SMALL_SORT_CAPACITY; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint item = 0u; item < INX_SMALL_SORT_VALUES_PER_LANE; ++item) {
                uint index = lane + item * INX_SMALL_SORT_LANES;
                uint partner = index ^ j;
                if (partner > index) {
                    const bool ascending = (index & k) == 0u;
                    const bool swap = ascending
                                          ? inx_greater(shared_keys[index], shared_indices[index],
                                                        shared_keys[partner], shared_indices[partner])
                                          : inx_less(shared_keys[index], shared_indices[index],
                                                     shared_keys[partner], shared_indices[partner]);
                    if (swap) {
                        uint2 key = shared_keys[index];
                        uint particle_index = shared_indices[index];
                        shared_keys[index] = shared_keys[partner];
                        shared_indices[index] = shared_indices[partner];
                        shared_keys[partner] = key;
                        shared_indices[partner] = particle_index;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    for (uint item = 0u; item < INX_SMALL_SORT_VALUES_PER_LANE; ++item) {
        uint index = lane + item * INX_SMALL_SORT_LANES;
        if (index < live_count)
            input_indices[index] = shared_indices[index];
    }
}
)slang");
    return Source;
}

std::string_view GpuParticleSortShaderSources::Generate() noexcept
{
    static const std::string Source = BuildShader(R"slang(
uint inx_ordered_float(float value) {
    uint bits = asuint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}
uint inx_particle_sort_key(float view_depth, uint particle_id) {
    uint depth_key = inx_ordered_float(view_depth);
    depth_key = pc.descending != 0u ? ~depth_key : depth_key;
    return (depth_key & 0xffffff00u) | (particle_id & 0xffu);
}
[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint index = dispatch_id.x;
    uint live_count = min(indirect_args[0].instance_count, pc.capacity);
    if (index >= live_count) return;
    uint particle_index = source_indices[index];
    float4 view_position = mul(pc.view, float4(visibility[particle_index].position_radius.xyz, 1.0));
    input_keys[index] = inx_particle_sort_key(view_position.z, particle_index);
    input_indices[index] = particle_index;
}
)slang");
    return Source;
}

std::string_view GpuParticleSortShaderSources::Histogram() noexcept
{
    static const std::string Source = BuildShader(R"slang(
groupshared Atomic<uint> local_histogram[16];
uint inx_key_digit(uint key) {
    return (key >> pc.digit_shift) & 15u;
}
[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint local_index = group_thread_id.x;
    if (local_index < 16u) local_histogram[local_index].store(0u);
    GroupMemoryBarrierWithGroupSync();
    uint index = dispatch_id.x;
    uint live_count = min(indirect_args[0].instance_count, pc.capacity);
    if (index < live_count) {
        uint digit = inx_key_digit(input_keys[index]);
        local_histogram[digit].add(1u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_index < 16u)
        histograms[group_id.x * 16u + local_index] = local_histogram[local_index].load();
}
)slang");
    return Source;
}

std::string_view GpuParticleSortShaderSources::Scan() noexcept
{
    static const std::string Source = BuildShader(R"slang(
groupshared uint totals[16];
[shader("compute")] [numthreads(16, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint bin = group_thread_id.x;
    uint running = 0u;
    uint active_block_count = min(dispatch_arguments[0].dispatch_group_count_x, pc.block_count);
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

std::string_view GpuParticleSortShaderSources::Scatter() noexcept
{
    static const std::string Source =
        BuildShader(std::string(SlangParticleRadixRank) + R"slang([shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint source_index = dispatch_id.x;
    uint live_count = min(indirect_args[0].instance_count, pc.capacity);
    bool active = source_index < live_count;
    uint key = active ? input_keys[source_index] : 0u;
    uint digit = (key >> pc.digit_shift) & 15u;
    uint local_rank = inx_radix_rank(group_index, digit, active);
    if (!active) return;
    uint block_offset = group_id.x * 16u + digit;
    uint destination = global_offsets[digit] + block_offsets[block_offset] + local_rank;
    output_keys[destination] = key;
    output_indices[destination] = input_indices[source_index];
}
)slang");
    return Source;
}

bool GpuParticleSortProgram::IsValid() const noexcept
{
    return IsCompiledParticleShaderValid(small) && IsCompiledParticleShaderValid(generate) &&
           IsCompiledParticleShaderValid(histogram) && IsCompiledParticleShaderValid(scan) &&
           IsCompiledParticleShaderValid(scatter);
}

bool GpuParticleSortProgramStorage::Assign(const GpuParticleSortProgram &program)
{
    if (!program.IsValid())
        return false;
    const std::array<CompiledParticleShader, 5> sources = {program.small, program.generate, program.histogram,
                                                           program.scan, program.scatter};
    std::array<CompiledParticleShader, 5> candidate;
    for (size_t index = 0; index < sources.size(); ++index)
        candidate[index] = sources[index];
    shaders = std::move(candidate);
    return true;
}

bool GpuParticleSortProgramStorage::IsValid() const noexcept
{
    return View().IsValid();
}

GpuParticleSortProgram GpuParticleSortProgramStorage::View() const noexcept
{
    return {
        shaders[0], shaders[1], shaders[2], shaders[3], shaders[4],
    };
}

ParticleGpuSorter::ParticleGpuSorter() = default;

ParticleGpuSorter::~ParticleGpuSorter()
{
    Destroy();
}

bool ParticleGpuSorter::Create(rhi::Device &device, const GpuParticleSorterDesc &desc)
{
    Destroy();
    if (desc.capacity == 0 || !desc.visibility.IsValid() || !desc.indirectArguments.IsValid() ||
        !desc.sourceIndices.IsValid() || !desc.dispatchArguments.IsValid() || !desc.program.IsValid())
        return false;

    const uint64_t elementBytes = static_cast<uint64_t>(desc.capacity) * sizeof(uint32_t);
    const uint64_t keyBytes = static_cast<uint64_t>(desc.capacity) * PackedKeyStride;
    const uint32_t blockCount = DivideRoundUp(desc.capacity, WorkgroupSize);
    if (blockCount == 0 || blockCount > std::numeric_limits<uint32_t>::max() / Radix)
        return false;
    const uint64_t blockBytes = static_cast<uint64_t>(blockCount) * Radix * sizeof(uint32_t);
    const auto storage = rhi::BufferUsageFlags::Storage;
    const auto createSharedStorage = [&](uint64_t bytes) {
        rhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = bytes;
        bufferDesc.usage = storage;
        bufferDesc.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
        return device.CreateBuffer(bufferDesc);
    };

    m_device = &device;
    m_capacity = desc.capacity;
    m_blockCount = blockCount;
    m_visibility = desc.visibility;
    m_indirectArguments = desc.indirectArguments;
    m_sourceIndices = desc.sourceIndices;
    m_dispatchArguments = desc.dispatchArguments;
    for (auto &buffer : m_keys)
        buffer = createSharedStorage(keyBytes);
    for (auto &buffer : m_indices)
        buffer = createSharedStorage(elementBytes);
    m_histograms = createSharedStorage(blockBytes);
    m_blockOffsets = createSharedStorage(blockBytes);
    m_globalOffsets = createSharedStorage(Radix * sizeof(uint32_t));
    if (!std::all_of(m_keys.begin(), m_keys.end(), [](auto value) { return value.IsValid(); }) ||
        !std::all_of(m_indices.begin(), m_indices.end(), [](auto value) { return value.IsValid(); }) ||
        !m_histograms.IsValid() || !m_blockOffsets.IsValid() || !m_globalOffsets.IsValid()) {
        Destroy();
        return false;
    }

    const std::array<CompiledParticleShader, 5> shaders = {
        desc.program.small, desc.program.generate, desc.program.histogram, desc.program.scan, desc.program.scatter};
    std::array<std::unique_ptr<ParticleComputePipeline> *, 5> pipelines = {
        &m_smallPipeline, &m_generatePipeline, &m_histogramPipeline, &m_scanPipeline, &m_scatterPipeline};
    for (size_t index = 0; index < shaders.size(); ++index) {
        auto program = std::make_unique<ParticleComputePipeline>(device, *shaders[index]);
        for (uint32_t pingPong = 0; pingPong < 2; ++pingPong) {
            const uint32_t output = 1u - pingPong;
            program->AddBindings(
                ParticleBufferBindings({m_visibility, m_indirectArguments, m_keys[pingPong], m_indices[pingPong],
                                        m_keys[output], m_indices[output], m_histograms, m_blockOffsets,
                                        m_globalOffsets, m_sourceIndices, m_dispatchArguments}));
        }
        *pipelines[index] = std::move(program);
    }
    return true;
}

void ParticleGpuSorter::Destroy() noexcept
{
    m_smallPipeline.reset();
    m_generatePipeline.reset();
    m_histogramPipeline.reset();
    m_scanPipeline.reset();
    m_scatterPipeline.reset();

    if (m_device) {
        m_device->Release(m_globalOffsets);
        m_device->Release(m_blockOffsets);
        m_device->Release(m_histograms);
        for (auto buffer : m_indices)
            m_device->Release(buffer);
        for (auto buffer : m_keys)
            m_device->Release(buffer);
    }
    m_device = nullptr;
    m_capacity = 0;
    m_blockCount = 0;
    m_visibility = {};
    m_indirectArguments = {};
    m_sourceIndices = {};
    m_dispatchArguments = {};
    m_keys.fill({});
    m_indices.fill({});
    m_histograms = {};
    m_blockOffsets = {};
    m_globalOffsets = {};
    m_smallPipeline = {};
    m_generatePipeline = {};
    m_histogramPipeline = {};
    m_scanPipeline = {};
    m_scatterPipeline = {};
}

bool ParticleGpuSorter::IsValid() const noexcept
{
    return m_device && m_smallPipeline && m_generatePipeline && m_histogramPipeline && m_scanPipeline &&
           m_scatterPipeline;
}

GpuParticleSortConstants ParticleGpuSorter::Constants(uint32_t passIndex) const noexcept
{
    GpuParticleSortConstants constants;
    constants.capacity = m_capacity;
    constants.blockCount = m_blockCount;
    constants.digitShift = (passIndex % PassCount) * 4u;
    return constants;
}

void ParticleGpuSorter::RecordSmall(const rhi::ComputeCommandEncoder &encoder, const std::array<float, 16> &view,
                                    ParticleSortMode mode) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    auto constants = Constants();
    constants.view = view;
    constants.descending = mode == ParticleSortMode::BackToFront ? 1u : 0u;
    RecordDirect(encoder, *m_smallPipeline, 0u, constants, 1);
}

void ParticleGpuSorter::RecordGenerate(const rhi::ComputeCommandEncoder &encoder, const std::array<float, 16> &view,
                                       ParticleSortMode mode) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    auto constants = Constants();
    constants.view = view;
    constants.descending = mode == ParticleSortMode::BackToFront ? 1u : 0u;
    RecordIndirect(encoder, *m_generatePipeline, 0u, constants);
}

void ParticleGpuSorter::RecordHistogram(const rhi::ComputeCommandEncoder &encoder, uint32_t passIndex) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    const auto constants = Constants(passIndex);
    RecordIndirect(encoder, *m_histogramPipeline, passIndex % 2u, constants);
}

void ParticleGpuSorter::RecordScan(const rhi::ComputeCommandEncoder &encoder, uint32_t passIndex) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    const auto constants = Constants(passIndex);
    RecordDirect(encoder, *m_scanPipeline, passIndex % 2u, constants, 1);
}

void ParticleGpuSorter::RecordScatter(const rhi::ComputeCommandEncoder &encoder, uint32_t passIndex) const
{
    if (!IsValid() || !encoder.IsValid())
        return;
    const auto constants = Constants(passIndex);
    RecordIndirect(encoder, *m_scatterPipeline, passIndex % 2u, constants);
}

void ParticleGpuSorter::RecordDirect(const rhi::ComputeCommandEncoder &encoder, const ParticleComputePipeline &pipeline,
                                     uint32_t variant, const GpuParticleSortConstants &constants, uint32_t groups) const
{
    pipeline.Bind(encoder, constants, variant);
    encoder.Dispatch(groups, 1, 1);
}

void ParticleGpuSorter::RecordIndirect(const rhi::ComputeCommandEncoder &encoder,
                                       const ParticleComputePipeline &pipeline, uint32_t variant,
                                       const GpuParticleSortConstants &constants) const
{
    pipeline.Bind(encoder, constants, variant);
    encoder.DispatchIndirect(m_dispatchArguments);
}

} // namespace infernux::particle
