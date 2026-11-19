#pragma once

#include "RenderGraphDescription.h"
#include "rhi/RhiComputeBuffer.h"

namespace infernux
{
struct RenderViewAccess;
namespace rhi
{
class ComputeBufferAllocation;
class BufferResource;
class ComputeHost;
class ComputeWorkInstance;
struct ComputeGraphReads;
} // namespace rhi
namespace raytracing
{
struct RayWorldMaterialPublication;
}

/// Allocations and immutable dispatch bindings prepared for one graph.
/// The submitter retains them through the last GPU use.
struct RenderGraphWorkStorage
{
    std::unordered_map<std::string, std::shared_ptr<rhi::ComputeWorkInstance>> instances;
    /// Pass-local read publications share the recipe's private allocations.
    std::unordered_map<std::string, std::shared_ptr<rhi::ComputeWorkInstance>> passInstances;
    std::unordered_map<std::string, std::shared_ptr<rhi::ComputeBuffer>> buffers;

    void Publish() const;
};

[[nodiscard]] std::optional<RenderGraphWorkStorage> PrepareRenderGraphWork(rhi::ComputeHost &host,
                                                                           const RenderGraphDescription &graph,
                                                                           const RenderGraphWorkStorage &previous);

/// Resolve active reflected inputs once during resource preparation. An
/// unavailable World publication defers this execution without allocating
/// another set of Work outputs.
[[nodiscard]] bool BindRenderGraphWorkReads(const RenderGraphDescription &graph, RenderGraphWorkStorage &storage,
                                            const rhi::ComputeGraphReads &publications);
[[nodiscard]] rhi::ComputeGraphReads
WorldRayGraphReadPublications(const RenderGraphDescription &graph,
                              const std::array<std::shared_ptr<const rhi::BufferResource>, 2> &geometry,
                              const std::shared_ptr<const raytracing::RayWorldMaterialPublication> &materials);

/// Names for complete Work slots. Aliases retain their own authored ranges.
[[nodiscard]] std::vector<std::string> GraphWorkSlotNames(const RenderGraphDescription &graph,
                                                          const GraphWorkBindingDesc &work);

/// Captured input requirements decide admission. Report a cycle to the
/// authored input so the producing program can retire the request.
[[nodiscard]] bool AppendPendingGraphBufferResults(RenderViewAccess &access,
                                                   const std::vector<const GraphBufferDesc *> &inputs);

struct GraphBufferWrite
{
    std::shared_ptr<rhi::ComputeBufferAllocation> allocation;
    intervals::Set initialized;
    intervals::Set written;

    void Publish(rhi::SubmissionTicket ticket) noexcept;
};

/// Only proven, exported values enter the external compute timeline.
[[nodiscard]] std::vector<GraphBufferWrite> CaptureGraphBufferWrites(const RenderGraphDescription &graph,
                                                                     const RenderGraphWorkStorage &storage,
                                                                     const GraphBufferContents &contents,
                                                                     const std::unordered_set<std::string> &writes);

class GraphFramePublication
{
  public:
    struct Capture
    {
        std::shared_ptr<GraphBufferCapture> request;
        std::shared_ptr<rhi::ComputeBuffer> value;
    };
    std::vector<GraphBufferWrite> writes;
    std::vector<Capture> captures;
    GraphEventQueue events;
    ~GraphFramePublication();
    void Accept(rhi::SubmissionTicket completion, GraphEventQueue &acceptedEvents) noexcept;
};

std::shared_ptr<GraphEventInterest> FindActiveGraphEventInterest(const GraphEventInterests &interests,
                                                                 const std::string &label);

/// Pending captures and occurrence numbering belong to the view. A host
/// publishes the prepared packet after accepting its corresponding commands.
class RenderGraphResults
{
  public:
    [[nodiscard]] bool HasPendingCaptures() const;
    void Cancel(const std::string &reason);
    void Reset(const std::string &reason);
    [[nodiscard]] std::shared_ptr<GraphBufferCapture> CaptureBuffer(const RenderGraphDescription &graph,
                                                                    rhi::RenderViewId view, const std::string &name);
    [[nodiscard]] std::shared_ptr<GraphFramePublication>
    Prepare(const RenderGraphDescription &graph, const RenderGraphWorkStorage &storage,
            const GraphBufferContents &contents, const std::unordered_set<std::string> &writes,
            const std::vector<uint32_t> &eventPasses, const GraphEventInterests &interests, rhi::RenderViewId view,
            uint64_t historyEpoch, const std::function<std::shared_ptr<GraphEventViewSample>()> &sample,
            const std::shared_ptr<const GameScreenAdmission> &screenAdmission = {},
            const rhi::ComputeBufferBindings *recordedReads = nullptr);

  private:
    [[nodiscard]] std::shared_ptr<GraphBufferCapture> MakeCapture(const GraphBufferDesc &buffer,
                                                                  rhi::RenderViewId view) const;
    std::vector<std::weak_ptr<GraphBufferCapture>> m_captures;
    uint64_t m_generation = 0;
    uint64_t m_occurrence = 0;
};

} // namespace infernux
