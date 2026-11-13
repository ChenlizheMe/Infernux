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
    /// Consume one completed readback, validate every row, then write output.
    void CopyMotion(float *output, size_t valueCount);

  private:
    bool materialized = false;
    std::vector<JPH::GPU::QueryRecordWord> values;
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
    [[nodiscard]] JoltGpuJointReplacement
    PrepareReplacement(const std::vector<JPH::GPU::JointAuthor> &authors, const std::vector<uint32_t> &previousRows,
                       const std::shared_ptr<rhi::ComputeBuffer> &oldStream, uint32_t oldCount,
                       const std::shared_ptr<rhi::ComputeBuffer> &bodyStates) const;
    /// Author settings and coefficient-dependent warm/rotor resets share one
    /// model acceptance boundary. Reference frames and unrelated runtime survive.
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate> PrepareUpdate(JoltGpuStorage &storage,
                                                                    std::vector<JoltGpuJointModelUpdate> changes) const;
    [[nodiscard]] JoltGpuPreparedJointMotion
    PrepareReadMotion(std::shared_ptr<const JoltGpuScene::Publication> publication,
                      const std::vector<uint32_t> &rows) const;
    [[nodiscard]] JoltGpuControlWork PrepareStatus(std::shared_ptr<const JoltGpuScene::Publication> publication) const;

  private:
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<uint8_t> ports;
    };
    rhi::ComputeHost m_host;
    std::array<Program, 4> m_programs;
};

} // namespace infernux
