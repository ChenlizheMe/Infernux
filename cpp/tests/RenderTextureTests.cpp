#include <function/renderer/rhi/RhiRenderTexture.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

using namespace infernux::rhi;

namespace
{
// Allocation failures are injected at each native construction boundary.
class TestDevice final : public Device
{
  public:
    DeviceCaps caps;
    std::shared_ptr<DeviceLifetime> lifetime = std::make_shared<DeviceLifetime>();
    std::set<uint32_t> live;
    std::vector<TextureDesc> images;
    std::vector<SamplerDesc> samplers;
    uint32_t next = 1;
    int failAt = 0;
    TestDevice()
    {
        caps.limits.maxTextureDimension2D = 4096;
        for (size_t i = 1; i < caps.formats.size(); ++i) {
            caps.formats[i] = {
                static_cast<PixelFormat>(i),
                FormatFeature::Sampled | FormatFeature::FilterLinear | FormatFeature::ColorAttachment |
                    FormatFeature::DepthStencilAttachment | FormatFeature::Storage | FormatFeature::TransferSource |
                    FormatFeature::TransferDestination,
                static_cast<SampleCountMask>(SampleCountBit(SampleCount::One) | SampleCountBit(SampleCount::Four))};
        }
    }
    const DeviceCaps &GetCapabilities() const noexcept override
    {
        return caps;
    }
    std::shared_ptr<DeviceLifetime> GetLifetime() const noexcept override
    {
        return lifetime;
    }
    template <class H> H Allocate()
    {
        if (failAt > 0 && --failAt == 0)
            return {};
        live.insert(next);
        return H{next++, ComposeHandleGeneration(1, 1)};
    }
    template <class H> void Free(H h)
    {
        if (h.IsValid())
            assert(live.erase(h.index) == 1);
    }
    TextureHandle CreateTexture(const TextureDesc &d) override
    {
        images.push_back(d);
        return Allocate<TextureHandle>();
    }
    TextureViewHandle CreateTextureView(const TextureViewDesc &) override
    {
        return Allocate<TextureViewHandle>();
    }
    SamplerHandle CreateSampler(const SamplerDesc &d) override
    {
        samplers.push_back(d);
        return Allocate<SamplerHandle>();
    }
    BufferHandle CreateBuffer(const BufferDesc &) override
    {
        assert(false);
        return {};
    }
    ShaderModuleHandle CreateShaderModule(const ShaderModuleDesc &) override
    {
        assert(false);
        return {};
    }
    BindingLayoutHandle CreateBindingLayout(const BindingLayoutDesc &) override
    {
        assert(false);
        return {};
    }
    BindGroupHandle CreateBindGroup(const BindGroupDesc &) override
    {
        assert(false);
        return {};
    }
    GraphicsPipelineHandle CreateGraphicsPipeline(const GraphicsPipelineDesc &) override
    {
        assert(false);
        return {};
    }
    ComputePipelineHandle CreateComputePipeline(const ComputePipelineDesc &) override
    {
        assert(false);
        return {};
    }
    bool WriteBuffer(BufferHandle, uint64_t, const void *, uint64_t) override
    {
        assert(false);
        return false;
    }
    void Release(TextureHandle h) noexcept override
    {
        Free(h);
    }
    void Release(TextureViewHandle h) noexcept override
    {
        Free(h);
    }
    void Release(SamplerHandle h) noexcept override
    {
        Free(h);
    }
    void Release(BufferHandle) noexcept override
    {
    }
    void Release(ShaderModuleHandle) noexcept override
    {
    }
    void Release(BindingLayoutHandle) noexcept override
    {
    }
    void Release(BindGroupHandle) noexcept override
    {
    }
    void Release(GraphicsPipelineHandle) noexcept override
    {
    }
    void Release(ComputePipelineHandle) noexcept override
    {
    }
};

template <class F> void Reject(F &&f)
{
    bool rejected = false;
    try {
        f();
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
}
} // namespace

int main()
{
    TestDevice device;
    RenderTextureDesc desc;
    desc.width = 64;
    desc.height = 32;
    {
        RenderTexture target(device, "render-texture-test", desc);
        auto first = target.Acquire();
        const auto slot = target.GetSampledColorSlot();
        assert(slot && slot->Acquire() == first->sampledColor);
        assert(first->width == 64 && first->height == 32 && first->revision == 1);
        assert(first->sampledColor->IsValid());
        assert(first->sampledColor->GetSourceId() == "render-texture-test");
        assert(&first->ColorAttachment() == first->color.get());
        assert(!first->depth && !first->multisampleColor && device.live.size() == 3);
        assert(first->GetResidentBytes() == 64 * 32 * 4);
        assert(device.samplers.back().addressU == AddressMode::ClampToEdge);
        assert(!target.Reconfigure(desc, 1920, 1080)); // Absolute size ignores viewport.
        assert(target.Acquire() == first && device.images.size() == 1);

        auto msaa = desc;
        msaa.samples = SampleCount::Four;
        msaa.depthFormat = PixelFormat::D32SFloat;
        msaa.colorFormat = PixelFormat::RGBA16SFloat;
        // Color image/view/sampler, MSAA image/view, then depth image/view.
        for (int failure = 1; failure <= 7; ++failure) {
            device.failAt = failure;
            Reject([&] { target.Reconfigure(msaa); });
            assert(target.Acquire() == first && device.live.size() == 3);
            assert(slot->Acquire() == first->sampledColor);
        }
        assert(target.Reconfigure(msaa));
        auto second = target.Acquire();
        assert(target.GetSampledColorSlot() == slot && slot->Acquire() == second->sampledColor);
        assert(second->revision == 2 && first->revision == 1);
        assert(second->sampledColor->GetSourceId() == first->sampledColor->GetSourceId());
        assert(second->color->GetTexture() != first->color->GetTexture());
        assert(second->multisampleColor->IsValid() && second->depth->IsValid());
        assert(!second->multisampleColor->GetSampler().IsValid() && !second->depth->GetSampler().IsValid());
        assert(second->GetResidentBytes() == 64 * 32 * (8 + 8 * 4 + 4 * 4));
        assert(device.live.size() == 10);
        auto sampled = first->sampledColor;
        first.reset();
        assert(device.live.size() == 10); // Independent sampling retains old color.
        sampled.reset();
        assert(device.live.size() == 7);

        auto relative = desc;
        relative.sizeMode = RenderTextureSizeMode::Relative;
        relative.widthScale = 0.5f;
        relative.heightScale = 0.25f;
        assert(target.Reconfigure(relative, 101, 99));
        assert(target.Acquire()->width == 51 && target.Acquire()->height == 25);
        assert(!target.Reconfigure(relative, 101, 99));
        assert(target.Reconfigure(relative, 103, 101));
        assert(target.Acquire()->width == 52 && target.Acquire()->height == 26);
        Reject([&] { target.Reconfigure(relative); });
        relative.widthScale = std::numeric_limits<float>::infinity();
        Reject([&] { target.Reconfigure(relative, 1, 1); });
        relative.widthScale = 0;
        Reject([&] { target.Reconfigure(relative, 1, 1); });

        auto invalid = desc;
        invalid.width = 0;
        Reject([&] { target.Reconfigure(invalid); });
        invalid.width = 4097;
        Reject([&] { target.Reconfigure(invalid); });
        invalid = desc;
        invalid.colorFormat = PixelFormat::D32SFloat;
        Reject([&] { target.Reconfigure(invalid); });
        invalid.colorFormat = PixelFormat::BC7UNorm;
        Reject([&] { target.Reconfigure(invalid); });
        invalid = desc;
        invalid.samples = SampleCount::Eight;
        Reject([&] { target.Reconfigure(invalid); });
        device.caps.formats[static_cast<size_t>(desc.colorFormat)].optimalTiling = FormatFeature::ColorAttachment;
        Reject([&] { target.Reconfigure(desc); });
    }
    assert(device.live.empty());
    {
        TestDevice batchDevice;
        RenderTextureDesc relative;
        relative.sizeMode = RenderTextureSizeMode::Relative;
        relative.widthScale = relative.heightScale = 0.5f;
        auto a = std::make_shared<RenderTexture>(batchDevice, "relative/a", relative, 101, 99);
        auto b = std::make_shared<RenderTexture>(batchDevice, "relative/b", relative, 101, 99);
        const std::vector targets{a, b};
        const auto oldA = a->Acquire(), oldB = b->Acquire();
        const auto imageCount = batchDevice.images.size();
        RenderTexture::ReconfigureReferenceSize(targets, 101, 99);
        assert(batchDevice.images.size() == imageCount);
        assert(a->Acquire() == oldA && b->Acquire() == oldB);
        // A has fully allocated its candidate when B's image-view creation
        // fails. Neither target or sampled publication may change.
        batchDevice.failAt = 5;
        Reject([&] { RenderTexture::ReconfigureReferenceSize(targets, 203, 107); });
        assert(a->Acquire() == oldA && b->Acquire() == oldB);
        assert(a->GetSampledColorSlot()->Acquire() == oldA->sampledColor);
        assert(b->GetSampledColorSlot()->Acquire() == oldB->sampledColor);
        assert(batchDevice.live.size() == 6);
        RenderTexture::ReconfigureReferenceSize(targets, 203, 107);
        assert(a->Acquire()->width == 102 && b->Acquire()->height == 54);
        assert(a->Acquire()->revision == 2 && b->Acquire()->revision == 2);
        assert(oldA->width == 51 && oldB->height == 50);
        assert(batchDevice.live.size() == 12); // recorded consumers retain old attachments
        RenderTexture::ReconfigureReferenceSize(targets, 1, 1);
        assert(a->Acquire()->width == 1 && b->Acquire()->height == 1);
    }
    {
        TestDevice sampled;
        auto sampledDesc = desc;
        sampledDesc.sampledDepth = true;
        Reject([&] { RenderTexture invalid(sampled, "missing-depth", sampledDesc); });
        sampledDesc.depthFormat = PixelFormat::D32SFloat;
        sampledDesc.samples = SampleCount::Four;
        RenderTexture target(sampled, "sampled-depth", sampledDesc);
        assert(HasTextureUsage(sampled.images[1].usage, TextureUsageFlags::TransferSource));
        assert(HasTextureUsage(sampled.images[1].usage, TextureUsageFlags::Sampled));
        assert(HasTextureUsage(sampled.images[2].usage, TextureUsageFlags::Sampled));
        assert(!target.Acquire()->depth->GetSampler().IsValid());
        // Turning sampled depth off is an explicit new allocation contract.
        sampledDesc.sampledDepth = false;
        assert(target.Reconfigure(sampledDesc));
        assert(!HasTextureUsage(sampled.images.back().usage, TextureUsageFlags::Sampled));
        sampled.caps.formats[static_cast<size_t>(PixelFormat::D32SFloat)].optimalTiling =
            FormatFeature::DepthStencilAttachment;
        sampledDesc.sampledDepth = true;
        Reject([&] { target.Reconfigure(sampledDesc); });
    }
    // Device shutdown destroys native allocations; retained wrappers cannot
    // dereference the device after that boundary, nor recreate resources.
    TestDevice stopped;
    auto target = std::make_unique<RenderTexture>(stopped, "shutdown-test", desc);
    auto borrowed = target->Acquire();
    stopped.lifetime->alive.store(false);
    assert(!borrowed->color->IsValid());
    Reject([&] { target->Reconfigure(desc); });
    stopped.live.clear();
    target.reset();
    borrowed.reset();
}
