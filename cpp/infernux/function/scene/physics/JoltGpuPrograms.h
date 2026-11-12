#pragma once

#include <Jolt/GPU/PhysicsStages.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/shader/ShaderProgramCode.h>

#include <array>
#include <memory>
#include <vector>

namespace infernux
{

/// Device records and immutable geometry for the first GPU rigid-body slice.
/// No CPU PhysicsSystem or Jolt Body memory is uploaded through this interface.
struct JoltGpuModel
{
    std::vector<JPH::GPU::BodyModel> bodies;
    std::vector<JPH::GPU::ShapeModel> shapes;
    std::vector<JPH::GPU::WorldRange> worlds;
};

/// Physics-owned resident values. Compilation, submission, budgets, fences and
/// resource retirement remain services of the shared engine compute runtime.
class JoltGpuStorage
{
  public:
    JoltGpuStorage(rhi::ComputeHost &host, JoltGpuModel model);

    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareInitialization(const std::vector<JPH::GPU::BodyState> &states,
                          const std::vector<JPH::GPU::BodyForces> &forces);
    /// Updating step parameters never overwrites a device-produced force input.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch> PrepareParameters(JPH::GPU::StepParams parameters);
    /// Explicit CPU input path. A GPU controller can instead write the resident
    /// BodyForces port directly; the normal compute dependency orders its use.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareForces(const std::vector<JPH::GPU::BodyForces> &forces);
    // All Prepare methods return HasWork()==false for an empty model. The
    // caller completes that empty phase without attempting a GPU submission.

    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Get(JPH::GPU::Resource resource) const;
    [[nodiscard]] uint32_t GetBodyCount() const noexcept
    {
        return m_bodyCount;
    }
    [[nodiscard]] uint32_t GetWorldCount() const noexcept
    {
        return m_worldCount;
    }
    [[nodiscard]] uint32_t GetContactCount() const noexcept
    {
        return m_contactCount;
    }
    [[nodiscard]] uint32_t GetMaxBodiesPerWorld() const noexcept
    {
        return m_maxBodiesPerWorld;
    }
    [[nodiscard]] const rhi::ComputeHost &GetHost() const noexcept
    {
        return m_host;
    }

  private:
    rhi::ComputeHost m_host;
    const JoltGpuModel m_model;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 8> m_buffers;
    uint32_t m_bodyCount = 0, m_worldCount = 0, m_contactCount = 0, m_maxBodiesPerWorld = 0;
};

/// A consumer of the common compiler's typed target artifact. This class does
/// not load sources, own a compiler/cache/device, or submit a second GPU queue.
class JoltGpuPrograms
{
  public:
    JoltGpuPrograms(rhi::ComputeHost &host, const ShaderTargetCode &target);

    /// Prepare exactly one legal numerical stage. The host may admit several
    /// ready stages in one service period, or resume at the next stage later.
    /// Preparing/accepting a batch does not publish a physics tick.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch> PrepareStage(JPH::GPU::Stage stage,
                                                                          const JoltGpuStorage &storage);

  private:
    struct Entry
    {
        JPH::GPU::StageDefinition definition;
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<JPH::GPU::Resource> resources;
    };
    rhi::ComputeHost m_host;
    std::vector<Entry> m_entries;
};

} // namespace infernux
