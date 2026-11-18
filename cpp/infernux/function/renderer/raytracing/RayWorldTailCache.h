#pragma once

#include <function/renderer/rhi/RenderViewContext.h>
#include <function/renderer/rhi/RhiQuery.h>
#include <function/renderer/rhi/RhiSubmission.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

namespace infernux
{
class FrameWorkOwner;
struct FrameWorkFacts;
class RenderWorldFrame;
} // namespace infernux

namespace infernux::rhi
{
class ComputeCommandEncoder;
class Device;
class BufferResource;
} // namespace infernux::rhi

namespace infernux::raytracing
{
struct RayWorldMaterialPublication;
struct RayWorldGeometryBinding;
class RayTailCacheRequest;
class RayTailCacheProgram;
class RayWorldTailCache;

struct RayTailCacheConfig
{
    uint32_t capacity = 1024;
    uint32_t batchSize = 64;
    float cellSize = 1.0f;
    uint32_t directionGrid = 4;
    float maxRayDistance = 1000.0f;
    uint64_t residentBytes = 8 * 1024 * 1024;
};

/// Eight exactly aligned uint4/float4 blocks, shared with inx_ray_tail_cache.
struct alignas(16) RayTailCacheEntry
{
    std::array<uint32_t, 32> words{};
};
static_assert(sizeof(RayTailCacheEntry) == 128);

struct RayTailCachePublication
{
    std::shared_ptr<const rhi::BufferResource> entries;
    uint32_t generation = 0;
    rhi::SubmissionTicket producer;
};

/// A single private generation slice. The original graphics submission owns
/// its buffers and bind groups until its real completion receipt is observed.
class RayTailCacheBatch final
{
  public:
    ~RayTailCacheBatch();
    [[nodiscard]] bool Allows(uint32_t gate) const noexcept;
    [[nodiscard]] bool Accepted() const noexcept;
    [[nodiscard]] const std::shared_ptr<rhi::WorkCompletionReceipt> &Receipt() const noexcept;
    [[nodiscard]] const std::array<std::shared_ptr<const rhi::BufferResource>, 4> &Buffers() const noexcept;
    void SetGpuTiming(std::function<std::optional<rhi::GpuTimestampFrame>()> timing);
    uint32_t RecordPrepare(const rhi::ComputeCommandEncoder &commands,
                           const std::array<rhi::TextureViewHandle, 2> &requests);
    uint32_t RecordMerge(const rhi::ComputeCommandEncoder &commands, rhi::TextureViewHandle samples);

  private:
    friend class RayWorldTailCache;
    struct State;
    explicit RayTailCacheBatch(std::unique_ptr<State> state);
    std::unique_ptr<State> m_state;
};

struct RayTailCacheBinding
{
    /// read, private write, requests, control. No pending bit: fresh D never waits.
    std::array<std::shared_ptr<const rhi::BufferResource>, 4> buffers;
    std::shared_ptr<const RayTailCachePublication> publication;
    std::shared_ptr<RayTailCacheBatch> batch;
    std::array<uint64_t, 4> byteSizes{};
};

/// World-shared, bounded, fixed-depth I1 radiance storage. CPU code controls
/// identities and leases only; all entries and request metadata remain on GPU.
class RayWorldTailCache final
{
  public:
    RayWorldTailCache(FrameWorkOwner &owner, rhi::Device &device, RayTailCacheConfig config = {});
    ~RayWorldTailCache();
    [[nodiscard]] RayTailCacheBinding Acquire(const std::shared_ptr<const RenderWorldFrame> &world,
                                              const RayWorldGeometryBinding &geometry,
                                              const std::shared_ptr<const RayWorldMaterialPublication> &materials,
                                              rhi::RenderViewId view, std::shared_ptr<RayTailCacheRequest> &request);
    /// The submitter first assigns the receipt its actual queue ticket.
    [[nodiscard]] bool AcceptBatch(const std::shared_ptr<RayTailCacheBatch> &batch) noexcept;

  private:
    void Observe();
    [[nodiscard]] FrameWorkFacts Describe();
    bool Advance();
    [[nodiscard]] std::shared_ptr<RayTailCacheBatch> PrepareBatch(const std::shared_ptr<RayTailCacheRequest> &request,
                                                                  rhi::RenderViewId view);
    [[nodiscard]] uint64_t AvailableBytes();
    struct State;
    std::unique_ptr<State> m_state;
};
} // namespace infernux::raytracing
