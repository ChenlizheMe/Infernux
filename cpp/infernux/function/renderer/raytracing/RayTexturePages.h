#pragma once

#include <function/renderer/rhi/RhiTexture.h>

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace infernux::rhi
{
struct DeviceCaps;
} // namespace infernux::rhi

namespace infernux::raytracing
{

enum class RayTexturePageClass : uint8_t
{
    Color,
    Data,
};

struct RayTexturePageSource
{
    std::shared_ptr<const rhi::TextureGpuView> view;
    rhi::TextureDesc allocation;
    uint64_t contentGeneration = 0;
    RayTexturePageClass pageClass = RayTexturePageClass::Color;
};

/// Freeze a resident texture from a device that already exposes its allocation descriptor.
template <class Device>
[[nodiscard]] RayTexturePageSource
CaptureRayTexturePageSource(const Device &device, std::shared_ptr<const rhi::TextureGpuView> view,
                            uint64_t contentGeneration, RayTexturePageClass pageClass)
{
    RayTexturePageSource source{std::move(view), {}, contentGeneration, pageClass};
    if (source.view && !device.GetTextureDesc(source.view->GetTexture(), source.allocation))
        source.allocation = {};
    return source;
}

struct RayTexturePageSlot
{
    RayTexturePageClass pageClass;
    uint32_t layer;
};

struct RayTexturePageLayer
{
    RayTexturePageSource source;
    RayTexturePageSlot slot;
};

struct RayTexturePagePlan
{
    rhi::DeviceId device = rhi::InvalidDeviceId;
    /// Empty classes allocate no page. Nonempty classes put one selected mip chain in each array layer.
    std::array<std::optional<rhi::TextureDesc>, 2> pages;
    std::array<uint64_t, 2> pageBytes{};
    std::vector<RayTexturePageLayer> layers;
    /// One entry per original request, including multiple samplers of the same texture.
    std::vector<RayTexturePageSlot> slots;
    uint64_t byteSize = 0;
};

enum class RayTexturePageIssue : uint8_t
{
    None,
    SourceUnavailable,
    UnsupportedSource,
    DeviceLimit,
    ByteBudget,
};

struct RayTexturePagePreparation
{
    std::optional<RayTexturePagePlan> plan;
    RayTexturePageIssue issue = RayTexturePageIssue::None;
    std::size_t sourceIndex = 0;
};

/// Select one mip chain per class at a resolution that fits the device, storage, and row budgets.
[[nodiscard]] RayTexturePagePreparation PrepareRayTexturePages(rhi::DeviceId device, const rhi::DeviceCaps &caps,
                                                               uint64_t byteBudget, uint64_t maxRowBytes,
                                                               const std::vector<RayTexturePageSource> &sources);

class RayTexturePageDraw;
class RayTexturePageRenderer;
class RayTexturePageCandidate;

struct RayTexturePageSlice
{
    std::shared_ptr<const RayTexturePageDraw> draw;
    rhi::TextureHandle destination;
    rhi::SubmissionTicket sourceUpload;
    RayTexturePageClass pageClass;
    uint32_t mip, layer;
    uint32_t width, height;
    uint32_t y, rowCount;

    [[nodiscard]] bool FirstStripe() const noexcept
    {
        return y == 0;
    }
    [[nodiscard]] bool LastStripe() const noexcept
    {
        return y + rowCount == height;
    }
    [[nodiscard]] uint64_t PixelCount() const noexcept
    {
        return uint64_t{width} * rowCount;
    }
};

/// Position in the frozen destination page layout.
class RayTexturePageCursor final
{
  public:
    explicit RayTexturePageCursor(const RayTexturePagePlan &plan) noexcept : m_plan(plan)
    {
    }
    [[nodiscard]] bool Done() const noexcept;
    [[nodiscard]] uint64_t MinimumBytes() const noexcept;

  private:
    friend class RayTexturePageCandidate;
    [[nodiscard]] uint32_t Width() const noexcept;
    [[nodiscard]] uint32_t Height() const noexcept;
    void Advance(uint32_t rows) noexcept;

    const RayTexturePagePlan &m_plan;
    std::size_t m_layer = 0;
    uint32_t m_mip = 0;
    uint32_t m_row = 0;
};

/// Retain the batch through its accepted submission. The candidate only retains the receipt.
class RayTexturePageBatch final
{
  public:
    RayTexturePageBatch(const RayTexturePageBatch &) = delete;
    RayTexturePageBatch &operator=(const RayTexturePageBatch &) = delete;
    [[nodiscard]] const std::vector<RayTexturePageSlice> &Slices() const noexcept
    {
        return m_slices;
    }
    [[nodiscard]] const std::shared_ptr<rhi::WorkCompletionReceipt> &Receipt() const noexcept
    {
        return m_receipt;
    }
    [[nodiscard]] const std::array<uint64_t, 2> &PageBytes() const noexcept
    {
        return m_pageBytes;
    }
    [[nodiscard]] uint64_t ByteSize() const noexcept
    {
        return m_pageBytes[0] + m_pageBytes[1];
    }

    /// Call after the backend has recorded all stripes in order.
    void MarkRecorded() noexcept
    {
        m_recorded = true;
    }

  private:
    friend class RayTexturePageCandidate;
    RayTexturePageBatch(std::shared_ptr<RayTexturePageCandidate> owner, uint64_t offset);

    std::shared_ptr<RayTexturePageCandidate> m_owner;
    std::shared_ptr<rhi::WorkCompletionReceipt> m_receipt;
    std::vector<RayTexturePageSlice> m_slices;
    std::array<uint64_t, 2> m_pageBytes{};
    uint64_t m_offset;
    bool m_recorded = false;
};

struct RayTexturePagePublication
{
    uint64_t revision;
    std::array<std::shared_ptr<const rhi::TextureGpuView>, 2> pages;
    std::array<uint64_t, 2> pageBytes;
    std::vector<RayTexturePageSlot> slots;
    rhi::SubmissionTicket completion;
};

/// Candidate image lifetime follows the owner thread and existing graphics submission receipts.
class RayTexturePageCandidate final : public std::enable_shared_from_this<RayTexturePageCandidate>
{
  public:
    RayTexturePageCandidate(const RayTexturePageCandidate &) = delete;
    RayTexturePageCandidate &operator=(const RayTexturePageCandidate &) = delete;

    [[nodiscard]] static std::shared_ptr<RayTexturePageCandidate> Create(rhi::Device &device, RayTexturePagePlan plan,
                                                                         std::string sourceId, uint64_t revision);

    /// Preview one bounded batch. Discarding a rejected recording leaves it eligible for the next opportunity.
    [[nodiscard]] std::shared_ptr<RayTexturePageBatch> PrepareBatch(RayTexturePageRenderer &renderer,
                                                                    uint64_t byteAllowance, uint32_t maxSlices);
    /// Call after the original queue has assigned the batch receipt its accepted ticket.
    [[nodiscard]] bool Accept(const RayTexturePageBatch &batch) noexcept;
    /// Collect completion the caller observed on the same submission domain.
    [[nodiscard]] bool Observe();
    /// Report an unrecoverable recording or submission failure.
    void Fail() noexcept;
    [[nodiscard]] bool Failed() const noexcept
    {
        return m_failed;
    }
    [[nodiscard]] const std::shared_ptr<const RayTexturePagePublication> &Publication() const noexcept
    {
        return m_publication;
    }

  private:
    RayTexturePageCandidate(rhi::Device &device, RayTexturePagePlan plan, std::string sourceId, uint64_t revision);
    void Publish();

    RayTexturePagePlan m_plan;
    RayTexturePageCursor m_cursor;
    std::string m_sourceId;
    uint64_t m_revision;
    std::array<std::shared_ptr<rhi::TextureResource>, 2> m_resources;
    std::shared_ptr<rhi::WorkCompletionReceipt> m_pending;
    std::shared_ptr<const RayTexturePagePublication> m_publication;
    /// Reuse the prepared view, bind group, and uniforms when the next stripe continues this subresource.
    std::shared_ptr<const RayTexturePageDraw> m_continuedDraw;
    rhi::SubmissionTicket m_completion;
    uint64_t m_acceptedBytes = 0;
    bool m_failed = false;
};

} // namespace infernux::raytracing
