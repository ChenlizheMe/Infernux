#pragma once

#include "RayLitMaterials.h"

#include <core/threading/FrameWorkOwner.h>
#include <function/renderer/rhi/RhiUpload.h>

#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace infernux
{
class RenderWorldFrame;
namespace rhi
{
class BufferResource;
}
} // namespace infernux

namespace infernux::raytracing
{

class RayWorldMaterialRequest;
class RayWorldMaterialBuild;

struct RayWorldMaterialPublication
{
    std::shared_ptr<const rhi::BufferResource> materials;
    std::shared_ptr<const rhi::BufferResource> descriptors;
    std::array<std::shared_ptr<const rhi::TextureGpuView>, 2> pages;
    uint64_t atlasBytes = 0;
};

struct RayWorldMaterialBinding
{
    std::shared_ptr<const RayWorldMaterialPublication> value;
    bool pending = false;
};

using ResolveRayPlaceholderTexture = std::function<std::shared_ptr<const rhi::TextureGpuView>(RayTexturePageClass)>;

/// A reusable one-texel array for a class used only by constant descriptors.
[[nodiscard]] rhi::TextureUploadRequest RayPlaceholderTextureUpload(RayTexturePageClass pageClass) noexcept;

/// Owns material preparation and buffer uploads on the frame-work owner.
/// The backend records image copies and observes their actual submission receipts.
class RayWorldMaterials final
{
  public:
    RayWorldMaterials(FrameWorkOwner &owner, rhi::Device &device, ResolveRayLitTexture resolver,
                      ResolveRayPlaceholderTexture placeholders, uint64_t atlasByteBudget);
    ~RayWorldMaterials();

    RayWorldMaterials(const RayWorldMaterials &) = delete;
    RayWorldMaterials &operator=(const RayWorldMaterials &) = delete;

    [[nodiscard]] RayWorldMaterialBinding Acquire(const std::shared_ptr<const RenderWorldFrame> &world,
                                                  const std::shared_ptr<const RayLitMaterialLayout> &layout,
                                                  uint64_t textureContentGeneration,
                                                  std::shared_ptr<const RayWorldMaterialRequest> &request);
    [[nodiscard]] std::shared_ptr<RayTexturePageBatch> PrepareCopyBatch();
    /// Submitters retain the batch through its accepted GPU work and give its receipt the real queue ticket.
    [[nodiscard]] bool AcceptCopyBatch(const RayTexturePageBatch &batch) noexcept;
    /// Report a synchronous recording or submission failure for the batch just prepared by this owner.
    void FailCopyBatch() noexcept;

  private:
    void Observe();
    [[nodiscard]] FrameWorkFacts Describe();
    bool Advance();
    bool AdvanceCpu(const std::shared_ptr<RayWorldMaterialRequest> &request);
    [[nodiscard]] uint64_t AvailableAtlasBytes();

    FrameWorkOwner &m_owner;
    rhi::Device &m_device;
    ResolveRayLitTexture m_resolve;
    ResolveRayPlaceholderTexture m_placeholder;
    uint64_t m_atlasByteBudget;
    FrameWorkOwner::SourceId m_source = 0;
    uint64_t m_revision = 0;
    std::vector<std::weak_ptr<RayWorldMaterialRequest>> m_requests;
    std::deque<std::weak_ptr<RayWorldMaterialRequest>> m_waiting;
    std::vector<std::weak_ptr<const RayWorldMaterialPublication>> m_published;
    std::array<std::shared_ptr<const rhi::TextureGpuView>, 2> m_emptyPages;
    std::unique_ptr<RayWorldMaterialBuild> m_active;
};

} // namespace infernux::raytracing
