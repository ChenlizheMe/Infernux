#pragma once

#include "JoltGpuTick.h"

namespace infernux
{
class JoltGpuBodyMutations;

/// Preserves one accepted body prefix and initializes only its new capacity tail.
/// Both phases belong to the scene's existing source. Preparation never submits
/// or waits; Commit is legal once, after the complete preparation receipt chain.
class JoltGpuBodyGrowth final
{
  public:
    JoltGpuBodyGrowth(JoltGpuStorage &storage, JoltGpuBodyMutations &mutations, uint32_t expectedFromCapacity,
                      uint32_t targetCapacity);

    [[nodiscard]] JoltGpuControlWork Prepare();
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate> Commit();

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace infernux
