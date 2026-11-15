#pragma once

#include "RhiCommand.h"

#include <cstdint>
#include <functional>
#include <utility>

namespace infernux::rhi
{

/// Borrowed commands for a declared buffer compute pass. There is deliberately
/// no native command buffer or graphics interface. The containing graph owns
/// resource dependencies; its recorder owns submission and completion.
class ComputePassContext
{
  public:
    using DispatchObserver = std::function<void(uint32_t, uint32_t, uint32_t, uint64_t, bool)>;

    ComputePassContext(ComputeCommandEncoder compute, TransferCommandEncoder transfer, DispatchObserver observer = {})
        : m_compute(compute), m_transfer(transfer), m_observer(std::move(observer))
    {
    }

    [[nodiscard]] ComputeCommandEncoder &GetComputeCommandEncoder() noexcept
    {
        return m_compute;
    }

    [[nodiscard]] TransferCommandEncoder &GetTransferCommandEncoder() noexcept
    {
        return m_transfer;
    }

    /// Report actual commands, keeping an unknown indirect input count unknown.
    void RecordComputeDispatch(uint32_t x, uint32_t y, uint32_t z, uint64_t inputs, bool indirect = false) const
    {
        if (m_observer)
            m_observer(x, y, z, inputs, indirect);
    }

  private:
    ComputeCommandEncoder m_compute;
    TransferCommandEncoder m_transfer;
    DispatchObserver m_observer;
};

using ComputePassExecuteCallback = std::function<void(ComputePassContext &)>;

} // namespace infernux::rhi
