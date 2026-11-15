#pragma once

#include "JoltGpuBodyMutations.h"
#include "JoltGpuTick.h"

namespace infernux
{
/// A cold layout transaction under the shared arena writer. Address metadata
/// comes from the scene's sole range builder; numerical state stays on device.
class JoltGpuPackedLayout final
{
  public:
    /// previousBodySlots[target] is an old global slot or UINT32_MAX for a new
    /// capacity slot. Every layout joint retains an accepted stable identity;
    /// new joint initialization belongs to its ordered topology group.
    JoltGpuPackedLayout(JoltGpuStorage &storage, JoltGpuBodyMutations &mutations, FrameWorkOwner &work,
                        JoltGpuModel replacement, std::vector<uint32_t> previousBodySlots,
                        std::vector<uint32_t> previousJointRows);
    [[nodiscard]] JoltGpuControlWork Prepare();
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate> Commit();
    /// After this owner's accepted source reads settle, replay only the joint
    /// runtime copies whose warm/rotor representation changed. No author remap,
    /// body copy, initialization or force input is repeated.
    void InvalidateJointRuntimeCopies() noexcept;

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
} // namespace infernux
