#pragma once

#include "RhiRenderTexture.h"
#include <algorithm>
#include <atomic>
#include <optional>
#include <unordered_map>

namespace infernux::rhi
{

/// Device-independent ownership for runtime targets and imported descriptions.
/// Registration is weak: neither asset lookup nor relative sizing pins storage.
/// Hosts supply their World's output extent, independently of the last camera.
class RenderTextureRegistry final
{
  public:
    std::shared_ptr<RenderTexture> Create(Device &device, const RenderTextureDesc &description, uint64_t world,
                                          uint32_t width, uint32_t height, const std::string &guid = {})
    {
        static std::atomic<uint64_t> nextIdentity{1};
        auto target = std::make_shared<RenderTexture>(
            device, "runtime/render-texture/" + std::to_string(nextIdentity.fetch_add(1, std::memory_order_relaxed)),
            description, width, height, guid);
        TrackRelative(target, world);
        return target;
    }

    std::shared_ptr<RenderTexture> Load(Device &device, const std::string &guid,
                                        std::shared_ptr<RenderTextureDesc> description, uint64_t world, uint32_t width,
                                        uint32_t height)
    {
        if (const auto found = m_assets.find(guid); found != m_assets.end()) {
            if (auto owner = found->second.lock()) {
                if (owner->description != description) {
                    Reconfigure(guid, *description, world, width, height);
                    owner->description = std::move(description);
                }
                return {owner, owner->texture.get()};
            }
        }
        auto texture = Create(device, *description, world, width, height, guid);
        auto owner = std::make_shared<Imported>(Imported{std::move(description), std::move(texture)});
        for (auto it = m_assets.begin(); it != m_assets.end();)
            it = it->second.expired() ? m_assets.erase(it) : std::next(it);
        m_assets[guid] = owner;
        // The same lease retains the CPU description and GPU target, so asset
        // residency cannot replace one while consumers still use the other.
        return {owner, owner->texture.get()};
    }

    [[nodiscard]] std::shared_ptr<RenderTexture> Find(const std::string &guid) const
    {
        const auto found = m_assets.find(guid);
        if (found != m_assets.end())
            if (auto owner = found->second.lock())
                return {owner, owner->texture.get()};
        return {};
    }

    void Reconfigure(const std::string &guid, const RenderTextureDesc &description, uint64_t world, uint32_t width,
                     uint32_t height)
    {
        auto texture = Find(guid);
        if (!texture)
            return; // Reimport of an unused asset does not allocate storage.
        if (description.sizeMode == RenderTextureSizeMode::Relative)
            m_relative.reserve(m_relative.size() + 1);
        texture->Reconfigure(description, width, height);
        TrackRelative(texture, world);
    }

    void Resize(std::optional<uint64_t> world, uint32_t width, uint32_t height)
    {
        std::vector<std::shared_ptr<RenderTexture>> live;
        live.reserve(m_relative.size());
        m_relative.erase(std::remove_if(m_relative.begin(), m_relative.end(),
                                        [&](const auto &entry) {
                                            auto target = entry.texture.lock();
                                            if (!target)
                                                return true;
                                            if (!world || entry.world == *world)
                                                live.push_back(std::move(target));
                                            return false;
                                        }),
                         m_relative.end());
        RenderTexture::ReconfigureReferenceSize(live, width, height);
    }

  private:
    struct Imported
    {
        std::shared_ptr<RenderTextureDesc> description;
        std::shared_ptr<RenderTexture> texture;
    };
    struct Relative
    {
        uint64_t world;
        std::weak_ptr<RenderTexture> texture;
    };
    void TrackRelative(const std::shared_ptr<RenderTexture> &texture, uint64_t world)
    {
        m_relative.erase(std::remove_if(m_relative.begin(), m_relative.end(),
                                        [&](const auto &entry) {
                                            const auto target = entry.texture.lock();
                                            return !target || target == texture;
                                        }),
                         m_relative.end());
        if (texture->Acquire()->description.sizeMode == RenderTextureSizeMode::Relative)
            m_relative.push_back({world, texture});
    }
    std::unordered_map<std::string, std::weak_ptr<Imported>> m_assets;
    std::vector<Relative> m_relative;
};

} // namespace infernux::rhi
