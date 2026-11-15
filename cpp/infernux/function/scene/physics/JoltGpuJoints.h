#pragma once

#include "JoltGpuScene.h"
#include <array>
#include <vector>

namespace infernux
{

struct JoltGpuJointReplacement
{
    std::shared_ptr<rhi::ComputeBuffer> stream;
    std::vector<rhi::ComputeBufferUpdate> updates;
    std::vector<rhi::ComputeDispatchDesc> dispatches;
};

struct JoltGpuJointMotionResult
{
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    std::shared_ptr<rhi::PreparedComputeBatch> packet;
    uint32_t count = 0;
    uint32_t kind = JPH::GPU::JointFixed;
    /// Consume one completed readback, validate every row, then write output.
    void CopyMotion(float *output, size_t valueCount);

  private:
    bool materialized = false;
    std::vector<JPH::GPU::JointReadback> values;
};

struct JoltGpuPreparedJointMotion
{
    std::shared_ptr<JoltGpuJointMotionResult> result;
    JoltGpuControlWork work;
};

/// Joint topology initialization and motion reads use the same compiled
/// artifact and shared services as the world's solver. Preparation never submits.
class JoltGpuJoints final
{
  public:
    JoltGpuJoints(rhi::ComputeHost &host, const ShaderTargetCode &target);
    /// An empty stream means managed capacity deferred this unaccepted
    /// transaction. Validation errors still throw. No private source survives
    /// a deferred result, so the same author transaction may be prepared again.
    [[nodiscard]] JoltGpuJointReplacement
    PrepareReplacement(const std::vector<JPH::GPU::JointAuthor> &authors, const std::vector<uint32_t> &previousRows,
                       const std::shared_ptr<rhi::ComputeBuffer> &oldStream,
                       std::shared_ptr<const JoltGpuJointLayout> oldLayout,
                       const std::shared_ptr<rhi::ComputeBuffer> &bodyStates) const;
    /// Author settings and coefficient-dependent warm/rotor resets share one
    /// model acceptance boundary. Reference frames and unrelated runtime survive.
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate>
    PrepareUpdate(JoltGpuStorage &storage, std::vector<JoltGpuJointModelUpdate> changes, bool &deferred) const;
    /// Existing state update entry; does not upload/reinitialize joint authors.
    [[nodiscard]] JoltGpuJointReplacement PrepareMobilityReset(JoltGpuStorage &storage, uint32_t bodySlot,
                                                               bool &deferred) const;
    [[nodiscard]] JoltGpuPreparedJointMotion
    PrepareReadMotion(std::shared_ptr<const JoltGpuScene::Publication> publication, const std::vector<uint32_t> &rows,
                      uint32_t kind) const;
    /// Captured stable identities are resolved by the source before preparation.
    [[nodiscard]] JoltGpuControlWork PreparePoseDrive(JoltGpuStorage &storage,
                                                      std::vector<JPH::GPU::JointPoseDrive> requests) const;
    [[nodiscard]] JoltGpuControlWork PrepareStatus(std::shared_ptr<const JoltGpuScene::Publication> publication) const;

  private:
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<uint8_t> ports;
    };
    [[nodiscard]] JoltGpuControlWork
    PreparePoseDriveChunk(JoltGpuStorage &storage,
                          std::shared_ptr<const std::vector<JPH::GPU::JointPoseDrive>> requests, size_t first) const;
    rhi::ComputeHost m_host;
    std::array<Program, 5> m_programs;
};

} // namespace infernux
