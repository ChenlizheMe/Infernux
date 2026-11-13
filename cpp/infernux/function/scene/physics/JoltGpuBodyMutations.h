#pragma once

#include "JoltGpuPrograms.h"

namespace infernux
{

/// Sparse state changes using the common compute service. Author model uploads
/// and these dispatches form one accepted transaction. The owner must complete
/// its current logical tick before preparing a mutation; no CPU state is read.
class JoltGpuBodyMutations
{
  public:
    JoltGpuBodyMutations(rhi::ComputeHost &host, const ShaderTargetCode &target);

    /// Initialization is explicit, for newly owned or deliberately reset slots
    /// within the existing capacity, clearing those slots' complete targets.
    /// Other bodies keep their resident state. COM changes preserve origin
    /// using the actual device rotation and rebase any live target's root COM.
    /// Repeated slots, including across the two lists, are not allowed.
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate>
    Prepare(JoltGpuStorage &storage, std::vector<JoltGpuBodyModelUpdate> bodies,
            std::vector<JoltGpuShapeModelUpdate> shapes,
            const std::vector<JPH::GPU::BodyInitialization> &initializations = {},
            const std::vector<JPH::GPU::CenterOfMassUpdate> &centersOfMass = {});
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate>
    PrepareReplacement(JoltGpuStorage &storage, JoltGpuModel replacement,
                       const std::vector<JPH::GPU::BodyInitialization> &initializations = {},
                       const std::vector<JPH::GPU::CenterOfMassUpdate> &centersOfMass = {});
    /// Initialize one contiguous new-capacity tail without reading old state.
    /// The caller uploads StageWork{{first, count, 0, 0}} to workUniform in the
    /// same ordered packet. Each call covers at most 2048 complete body slots.
    [[nodiscard]] rhi::ComputeDispatchDesc
    PrepareCapacityInitialization(const JoltGpuBodyStatePorts &ports, uint32_t first, uint32_t count,
                                  const std::shared_ptr<rhi::ComputeBuffer> &workUniform) const;

  private:
    struct Entry
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<uint8_t> ports;
    };
    struct MutationPacket
    {
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
    };
    [[nodiscard]] MutationPacket PreparePacket(JoltGpuStorage &storage, const std::vector<JPH::GPU::BodyModel> &models,
                                               const std::vector<JoltGpuBodyModelUpdate> &bodyChanges,
                                               const std::vector<JPH::GPU::BodyInitialization> &initializations,
                                               const std::vector<JPH::GPU::CenterOfMassUpdate> &centersOfMass);
    rhi::ComputeHost m_host;
    Entry m_initialize, m_centerOfMass, m_capacity;
};

} // namespace infernux
