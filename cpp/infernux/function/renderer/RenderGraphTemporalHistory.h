#pragma once

#include "RenderGraphDescription.h"
#include "rhi/RhiRenderTexture.h"

#include <array>
#include <iterator>

namespace infernux
{

struct GraphTemporalHistory
{
    std::array<std::shared_ptr<rhi::RenderTexture>, 2> targets{};
    std::string readName;
    std::string writeName;
    rhi::PixelFormat format = rhi::PixelFormat::Undefined;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t residentBytes = 0;
    uint32_t readIndex = 0;
    bool valid = false;

    [[nodiscard]] auto Read() const
    {
        return targets[readIndex]->Acquire();
    }
    [[nodiscard]] auto Write() const
    {
        return targets[readIndex ^ 1u]->Acquire();
    }
    void Invalidate() noexcept
    {
        readIndex = 0;
        valid = false;
    }
    void Accept() noexcept
    {
        readIndex ^= 1u;
        valid = true;
    }
};

/// Both devices retain the logical pair across graph rebuilds. The encoder
/// binds the selected generation and initializes invalid history to zero.
template <class HistoryMap>
void PrepareGraphTemporalHistory(HistoryMap &histories, rhi::Device &device, const RenderGraphDescription &graph,
                                 uint32_t width, uint32_t height)
{
    struct Pair
    {
        const GraphTextureDesc *read = nullptr;
        const GraphTextureDesc *write = nullptr;
    };
    std::unordered_map<std::string, Pair> requested;
    for (const auto &texture : graph.textures) {
        if (texture.role == GraphTextureRole::TemporalRead)
            requested[texture.temporalKey].read = &texture;
        else if (texture.role == GraphTextureRole::TemporalWrite)
            requested[texture.temporalKey].write = &texture;
    }
    for (auto it = histories.begin(); it != histories.end();)
        it = requested.count(it->first) ? std::next(it) : histories.erase(it);
    for (const auto &[key, pair] : requested) {
        const auto &desc = *pair.read;
        const auto [targetWidth, targetHeight] = desc.ResolveExtent(width, height);
        auto &history = histories[key];
        if (!history.targets[0] || history.width != targetWidth || history.height != targetHeight ||
            history.format != desc.format) {
            rhi::RenderTextureDesc allocation;
            allocation.width = targetWidth;
            allocation.height = targetHeight;
            allocation.colorFormat = desc.format;
            if (rhi::IsIntegerFormat(desc.format))
                allocation.filter = rhi::FilterMode::Nearest;
            std::array<std::shared_ptr<rhi::RenderTexture>, 2> targets;
            for (uint32_t index = 0; index < targets.size(); ++index)
                targets[index] = std::make_shared<rhi::RenderTexture>(
                    device, "History/" + key + "/" + std::to_string(index), allocation);
            history.residentBytes =
                targets[0]->Acquire()->GetResidentBytes() + targets[1]->Acquire()->GetResidentBytes();
            history.targets = std::move(targets);
            history.width = targetWidth;
            history.height = targetHeight;
            history.format = desc.format;
            history.Invalidate();
        }
        history.readName = pair.read->name;
        history.writeName = pair.write->name;
    }
}

} // namespace infernux
