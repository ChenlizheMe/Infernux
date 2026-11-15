#pragma once

#include "JoltGpuGeometry.h"
#include <core/math/Intervals.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>

namespace infernux
{
struct JoltGpuControlWork;
struct JoltGpuGeometryHandle
{
    JPH::GPU::UInt4 reference{};
    std::shared_ptr<const JoltGpuGeometryRecord> record;
};
using JoltGpuGeometryHandles = std::vector<std::shared_ptr<const JoltGpuGeometryHandle>>;

/// Immutable ranges in one compatible, initialized buffer generation.
/// The registry alone writes unleased ranges; no consumer may write/Freeze it.
/// Buffer allocation retention and exact immutable record retention travel together.
struct JoltGpuGeometryLease
{
    std::shared_ptr<rhi::ComputeBuffer> buffer;
    uint32_t wordCount = 0;
    intervals::Set wordRanges; // Initialized half-open UInt4 WORD ranges, including sentinel [0,1).
    JoltGpuGeometryHandles records;
    [[nodiscard]] bool Contains(JPH::GPU::UInt4 reference) const noexcept;
};

/// One cold geometry registry for the shared packed PhysicsWorld ComputeHost.
/// Stable word offsets survive growth; Worlds/models retain only small handles.
/// Intern directories are weak; actual model/publication/job/receipt owners
/// hold the range. Last owner release returns its words to the free intervals.
/// Old leases retain exact CPU records and GPU allocations. No per-World copy.
class JoltGpuGeometryStore final : public std::enable_shared_from_this<JoltGpuGeometryStore>
{
  public:
    explicit JoltGpuGeometryStore(rhi::ComputeHost &host);
    [[nodiscard]] std::shared_ptr<const JoltGpuGeometryHandle>
    Intern(std::shared_ptr<const JoltGpuGeometryRecord> record);
    [[nodiscard]] std::shared_ptr<const JoltGpuGeometryHandle> Find(JPH::GPU::UInt4 reference) const;
    /// Native preparation only. Busy allocation is retained as an ordinary
    /// no-packet continuation. Upload/copy packets use the caller's existing
    /// JoltGpuControlWork admission/traffic accounting/receipts, never Submit().
    /// Captured native records must all originate from this store.
    [[nodiscard]] JoltGpuControlWork
    PrepareCapture(JoltGpuGeometryHandles records,
                   std::function<void(std::shared_ptr<const JoltGpuGeometryLease>)> accepted);
    [[nodiscard]] uint64_t GetReservedWordCount() const noexcept
    {
        return m_highWaterWords;
    }

  private:
    struct Upload;
    friend struct Upload;
    [[nodiscard]] uint32_t RequireRecords(const JoltGpuGeometryHandles &records) const;
    [[nodiscard]] std::shared_ptr<const JoltGpuGeometryLease> Lease(JoltGpuGeometryHandles records) const;
    [[nodiscard]] uint32_t ReserveWords(uint32_t count);
    void FreeWords(uint32_t first, uint32_t count) noexcept;
    void Release(uint32_t first, uint32_t count, size_t fingerprint, const JoltGpuGeometryRecord *recipe) noexcept;
    rhi::ComputeHost m_host;
    uint32_t m_maxWords;
    uint32_t m_highWaterWords = 1;
    intervals::Set m_free;
    std::unordered_multimap<size_t, std::weak_ptr<const JoltGpuGeometryHandle>> m_intern;
    std::unordered_map<const JoltGpuGeometryRecord *, std::weak_ptr<const JoltGpuGeometryHandle>> m_recipes;
    std::map<uint32_t, std::weak_ptr<const JoltGpuGeometryHandle>> m_offsets;
    std::map<uint32_t, std::weak_ptr<const JoltGpuGeometryHandle>> m_available;
    std::shared_ptr<rhi::ComputeBuffer> m_buffer;
    uint32_t m_capacityWords = 0;
    bool m_sentinelAccepted = false;
    std::weak_ptr<Upload> m_upload;
};
} // namespace infernux
