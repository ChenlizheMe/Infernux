#include "ParticleGpuMigrator.h"
#include "ParticleComputePipeline.h"

#include <algorithm>
#include <limits>

namespace infernux::particle
{

namespace
{

constexpr std::string_view ResetShader = R"slang([[vk::binding(0, 0)]] StructuredBuffer<uint> source_states;
struct SourceCountersData {
    uint source_free_count;
    uint source_visible_count;
    uint source_dropped_count;
    uint source_reserved_count;
    uint source_collision_hit_count;
    uint source_collision_response_count;
    uint source_collision_trigger_count;
    uint source_collision_enter_count;
    uint source_collision_stay_count;
    uint source_collision_exit_count;
    uint source_collision_max_outward_speed_bits;
    uint source_collision_max_tangent_speed_bits;
    uint source_collision_candidate_overflow_count;
};
[[vk::binding(1, 0)]] StructuredBuffer<SourceCountersData> source_counters;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> destination_states;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> destination_free_slots;
struct DestinationCountersData {
    Atomic<uint> destination_free_count;
    uint destination_visible_count;
    Atomic<uint> destination_dropped_count;
    uint destination_reserved_count;
    uint destination_collision_hit_count;
    uint destination_collision_response_count;
    uint destination_collision_trigger_count;
    uint destination_collision_enter_count;
    uint destination_collision_stay_count;
    uint destination_collision_exit_count;
    uint destination_collision_max_outward_speed_bits;
    uint destination_collision_max_tangent_speed_bits;
    uint destination_collision_candidate_overflow_count;
};
[[vk::binding(4, 0)]] RWStructuredBuffer<DestinationCountersData> destination_counters;
[[vk::binding(5, 0)]] StructuredBuffer<uint4> copy_ranges;
[[vk::binding(6, 0)]] StructuredBuffer<uint> default_state;
[[vk::binding(8, 0)]] RWStructuredBuffer<Atomic<uint>> destination_alive_counts;
struct MigrationConstants {
    uint source_capacity;
    uint destination_capacity;
    uint source_stride_words;
    uint destination_stride_words;
    uint copy_range_count;
    uint invocation_count;
    uint reserved0;
    uint reserved1;
};
INX_ROOT_CONSTANTS ConstantBuffer<MigrationConstants> pc;

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    destination_alive_counts[0].store(0u);
    destination_alive_counts[1].store(0u);
    destination_counters[0].destination_free_count.store(0u);
    destination_counters[0].destination_visible_count = 0u;
    destination_counters[0].destination_dropped_count.store(source_counters[0].source_dropped_count);
    destination_counters[0].destination_reserved_count = source_counters[0].source_reserved_count;
    destination_counters[0].destination_collision_hit_count = source_counters[0].source_collision_hit_count;
    destination_counters[0].destination_collision_response_count = source_counters[0].source_collision_response_count;
    destination_counters[0].destination_collision_trigger_count = source_counters[0].source_collision_trigger_count;
    destination_counters[0].destination_collision_enter_count = source_counters[0].source_collision_enter_count;
    destination_counters[0].destination_collision_stay_count = source_counters[0].source_collision_stay_count;
    destination_counters[0].destination_collision_exit_count = source_counters[0].source_collision_exit_count;
    destination_counters[0].destination_collision_max_outward_speed_bits = source_counters[0].source_collision_max_outward_speed_bits;
    destination_counters[0].destination_collision_max_tangent_speed_bits = source_counters[0].source_collision_max_tangent_speed_bits;
    destination_counters[0].destination_collision_candidate_overflow_count = source_counters[0].source_collision_candidate_overflow_count;
}
)slang";

constexpr std::string_view MigrateShader = R"slang([[vk::binding(0, 0)]] StructuredBuffer<uint> source_states;
struct SourceCountersData {
    uint source_free_count;
    uint source_visible_count;
    uint source_dropped_count;
    uint source_reserved_count;
    uint source_collision_hit_count;
    uint source_collision_response_count;
    uint source_collision_trigger_count;
    uint source_collision_enter_count;
    uint source_collision_stay_count;
    uint source_collision_exit_count;
    uint source_collision_max_outward_speed_bits;
    uint source_collision_max_tangent_speed_bits;
    uint source_collision_candidate_overflow_count;
};
[[vk::binding(1, 0)]] StructuredBuffer<SourceCountersData> source_counters;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> destination_states;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> destination_free_slots;
struct DestinationCountersData {
    Atomic<uint> destination_free_count;
    uint destination_visible_count;
    Atomic<uint> destination_dropped_count;
    uint destination_reserved_count;
    uint destination_collision_hit_count;
    uint destination_collision_response_count;
    uint destination_collision_trigger_count;
    uint destination_collision_enter_count;
    uint destination_collision_stay_count;
    uint destination_collision_exit_count;
    uint destination_collision_max_outward_speed_bits;
    uint destination_collision_max_tangent_speed_bits;
    uint destination_collision_candidate_overflow_count;
};
[[vk::binding(4, 0)]] RWStructuredBuffer<DestinationCountersData> destination_counters;
[[vk::binding(5, 0)]] StructuredBuffer<uint4> copy_ranges;
[[vk::binding(6, 0)]] StructuredBuffer<uint> default_state;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> destination_alive_slots;
[[vk::binding(8, 0)]] RWStructuredBuffer<Atomic<uint>> destination_alive_counts;
struct MigrationConstants {
    uint source_capacity;
    uint destination_capacity;
    uint source_stride_words;
    uint destination_stride_words;
    uint copy_range_count;
    uint invocation_count;
    uint reserved0;
    uint reserved1;
};
INX_ROOT_CONSTANTS ConstantBuffer<MigrationConstants> pc;

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
    uint index = dispatch_id.x;
    if (index >= pc.invocation_count) return;

    bool source_alive = false;
    uint source_base = index * pc.source_stride_words;
    if (index < pc.source_capacity) source_alive = (source_states[source_base] & 1u) != 0u;
    if (index >= pc.destination_capacity) {
        if (source_alive) destination_counters[0].destination_dropped_count.add(1u);
        return;
    }

    uint destination_base = index * pc.destination_stride_words;
    for (uint word = 0u; word < pc.destination_stride_words; ++word)
        destination_states[destination_base + word] = default_state[word];

    if (source_alive) {
        // Migrated particles have already completed Init.
        destination_states[destination_base] = source_states[source_base] | 2u;
        destination_states[destination_base + 1u] = source_states[source_base + 1u];
        uint alive_index = destination_alive_counts[0].add(1u);
        destination_alive_slots[alive_index] = index;
        for (uint range_index = 0u; range_index < pc.copy_range_count; ++range_index) {
            uint4 range = copy_ranges[range_index];
            for (uint word = 0u; word < range.z; ++word)
                destination_states[destination_base + range.y + word] =
                    source_states[source_base + range.x + word];
        }
    } else {
        uint free_index = destination_counters[0].destination_free_count.add(1u);
        destination_free_slots[free_index] = index;
    }
}
)slang";

uint32_t GroupCount(uint32_t count) noexcept
{
    return count == 0 ? 0 : 1 + (count - 1) / ParticleGpuMigrator::WorkgroupSize;
}

} // namespace

bool GpuParticleMigrationProgram::IsValid() const noexcept
{
    return IsCompiledParticleShaderValid(reset) && IsCompiledParticleShaderValid(migrate);
}

bool GpuParticleMigrationProgramStorage::Assign(const GpuParticleMigrationProgram &program)
{
    if (!program.IsValid())
        return false;
    reset = program.reset;
    migrate = program.migrate;
    return true;
}

bool GpuParticleMigrationProgramStorage::IsValid() const noexcept
{
    return View().IsValid();
}

GpuParticleMigrationProgram GpuParticleMigrationProgramStorage::View() const noexcept
{
    return {reset, migrate};
}

std::string_view GpuParticleMigrationShaderSources::Reset() noexcept
{
    return ResetShader;
}

std::string_view GpuParticleMigrationShaderSources::Migrate() noexcept
{
    return MigrateShader;
}

ParticleGpuMigrator::ParticleGpuMigrator() = default;

ParticleGpuMigrator::~ParticleGpuMigrator()
{
    Destroy();
}

bool ParticleGpuMigrator::Create(rhi::Device &device, const GpuParticleMigrationDesc &desc)
{
    Destroy();
    if (desc.sourceCapacity == 0 || desc.destinationCapacity == 0 || desc.sourceStride == 0 ||
        desc.destinationStride == 0 || desc.sourceStride % sizeof(uint32_t) != 0 ||
        desc.destinationStride % sizeof(uint32_t) != 0 ||
        desc.sourceCounterByteSize < ParticleGpuRuntime::BaseCounterWordCount * sizeof(uint32_t) ||
        !desc.sourceStates.IsValid() || !desc.sourceCounters.IsValid() || !desc.destinationStates.IsValid() ||
        !desc.destinationFreeList.IsValid() || !desc.destinationCounters.IsValid() ||
        !desc.destinationAliveIndices.IsValid() || !desc.destinationAliveControl.IsValid() ||
        desc.defaultStateWords.empty() || desc.defaultStateWords.size() != desc.destinationStride / sizeof(uint32_t) ||
        !desc.program.IsValid() || desc.copyRanges.size() > std::numeric_limits<uint32_t>::max())
        return false;

    const uint32_t sourceStrideWords = desc.sourceStride / sizeof(uint32_t);
    const uint32_t destinationStrideWords = desc.destinationStride / sizeof(uint32_t);
    for (const auto &range : desc.copyRanges) {
        if (range.wordCount == 0 || range.sourceOffsetWords < 2 || range.destinationOffsetWords < 2 ||
            range.sourceOffsetWords >= sourceStrideWords || range.destinationOffsetWords >= destinationStrideWords ||
            range.wordCount > sourceStrideWords - range.sourceOffsetWords ||
            range.wordCount > destinationStrideWords - range.destinationOffsetWords)
            return false;
    }

    m_device = &device;
    m_sourceStates = desc.sourceStates;
    m_sourceCounters = desc.sourceCounters;
    m_sourceCounterByteSize = desc.sourceCounterByteSize;
    m_destinationStates = desc.destinationStates;
    m_destinationFreeList = desc.destinationFreeList;
    m_destinationCounters = desc.destinationCounters;
    m_destinationAliveIndices = desc.destinationAliveIndices;
    m_destinationAliveControl = desc.destinationAliveControl;
    m_constants.sourceCapacity = desc.sourceCapacity;
    m_constants.destinationCapacity = desc.destinationCapacity;
    m_constants.sourceStrideWords = sourceStrideWords;
    m_constants.destinationStrideWords = destinationStrideWords;
    m_constants.copyRangeCount = static_cast<uint32_t>(desc.copyRanges.size());
    m_constants.invocationCount = std::max(desc.sourceCapacity, desc.destinationCapacity);

    rhi::BufferDesc upload;
    upload.usage = rhi::BufferUsageFlags::Storage;
    upload.memory = rhi::BufferMemory::Upload;
    upload.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
    upload.byteSize = std::max<size_t>(desc.copyRanges.size(), 1) * sizeof(GpuParticleMigrationRange);
    m_copyRanges = device.CreateBuffer(upload);
    upload.byteSize = desc.defaultStateWords.size() * sizeof(uint32_t);
    m_defaultState = device.CreateBuffer(upload);
    const GpuParticleMigrationRange emptyRange{};
    const void *copyRangeData = desc.copyRanges.empty() ? static_cast<const void *>(&emptyRange)
                                                        : static_cast<const void *>(desc.copyRanges.data());
    const size_t copyRangeBytes = std::max<size_t>(desc.copyRanges.size(), 1) * sizeof(GpuParticleMigrationRange);
    if (!m_copyRanges.IsValid() || !m_defaultState.IsValid() ||
        !device.WriteBuffer(m_copyRanges, 0, copyRangeData, copyRangeBytes) ||
        !device.WriteBuffer(m_defaultState, 0, desc.defaultStateWords.data(),
                            desc.defaultStateWords.size() * sizeof(uint32_t))) {
        Destroy();
        return false;
    }

    const std::array<CompiledParticleShader, 2> shaders = {desc.program.reset, desc.program.migrate};
    std::array<std::unique_ptr<ParticleComputePipeline> *, 2> pipelines = {&m_resetPipeline, &m_migratePipeline};
    for (size_t index = 0; index < shaders.size(); ++index) {
        auto program = std::make_unique<ParticleComputePipeline>(device, *shaders[index]);
        auto bindings = ParticleBufferBindings({m_sourceStates, m_sourceCounters, m_destinationStates,
                                                m_destinationFreeList, m_destinationCounters, m_copyRanges,
                                                m_defaultState, m_destinationAliveIndices, m_destinationAliveControl});
        bindings[1].byteSize = m_sourceCounterByteSize;
        program->AddBindings(bindings);
        *pipelines[index] = std::move(program);
    }
    return true;
}

void ParticleGpuMigrator::Destroy() noexcept
{
    m_resetPipeline.reset();
    m_migratePipeline.reset();

    if (m_device) {
        m_device->Release(m_defaultState);
        m_device->Release(m_copyRanges);
    }
    m_device = nullptr;
    m_sourceStates = {};
    m_sourceCounters = {};
    m_sourceCounterByteSize = 0;
    m_destinationStates = {};
    m_destinationFreeList = {};
    m_destinationCounters = {};
    m_destinationAliveIndices = {};
    m_destinationAliveControl = {};
    m_copyRanges = {};
    m_defaultState = {};
    m_resetPipeline = {};
    m_migratePipeline = {};
    m_constants = {};
    m_resetRecorded = false;
    m_recorded = false;
}

bool ParticleGpuMigrator::IsValid() const noexcept
{
    return m_device && m_resetPipeline && m_migratePipeline;
}

void ParticleGpuMigrator::RecordReset(const rhi::ComputeCommandEncoder &encoder)
{
    if (!IsValid() || !encoder.IsValid() || m_recorded)
        return;
    m_resetPipeline->Bind(encoder, m_constants);
    encoder.Dispatch(1, 1, 1);
    m_resetRecorded = true;
}

void ParticleGpuMigrator::RecordMigrate(const rhi::ComputeCommandEncoder &encoder)
{
    if (!IsValid() || !encoder.IsValid() || !m_resetRecorded || m_recorded)
        return;
    m_migratePipeline->Bind(encoder, m_constants);
    encoder.Dispatch(GroupCount(m_constants.invocationCount), 1, 1);
    m_recorded = true;
}

} // namespace infernux::particle
