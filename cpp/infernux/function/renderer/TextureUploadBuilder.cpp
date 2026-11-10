#include "TextureUploadBuilder.h"

#include <limits>
#include <stdexcept>

namespace infernux
{

TextureUploadBatch::TextureUploadBatch(const TextureCpuData &cpuData, const rhi::SamplerDesc &sampler)
{
    if (!cpuData.IsValid() || cpuData.mipLevels.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("texture upload requires valid imported mip data");
    Build(cpuData.dimension, cpuData.format, cpuData.mipLevels, sampler);
    for (size_t index = 0; index < cpuData.mipLevels.size(); ++index) {
        const auto &mip = cpuData.mipLevels[index];
        if (mip.byteOffset > cpuData.bytes.size() || mip.byteSize > cpuData.bytes.size() - mip.byteOffset)
            throw std::invalid_argument("texture upload mip byte range is outside imported storage");
        m_subresources[index].data = cpuData.bytes.data() + mip.byteOffset;
    }
}

TextureUploadBatch::TextureUploadBatch(const TextureArtifactDescription &description, const rhi::SamplerDesc &sampler)
{
    if (!description.IsValid() || description.mipLevels.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("texture upload requires a valid imported description");
    Build(description.dimension, description.format, description.mipLevels, sampler);
}

void TextureUploadBatch::Build(TextureDimension dimension, TextureFormat format,
                               const std::vector<TextureMipLevel> &mips, const rhi::SamplerDesc &sampler)
{
    const auto &baseMip = mips.front();
    m_request.texture.dimension =
        dimension == TextureDimension::Texture3D ? rhi::TextureDimension::Texture3D : rhi::TextureDimension::Texture2D;
    m_request.texture.width = baseMip.width;
    m_request.texture.height = baseMip.height;
    m_request.texture.depthOrLayers = dimension == TextureDimension::Texture3D ? baseMip.depth : 1U;
    m_request.texture.mipLevels = static_cast<uint32_t>(mips.size());
    m_request.texture.format = GetTextureFormatInfo(format).pixelFormat;
    if (m_request.texture.format == rhi::PixelFormat::Undefined)
        throw std::invalid_argument("texture upload received an unsupported imported format");
    // Resident assets can be copied/reused or read back without decoding and
    // re-uploading another CPU copy of the same texture.
    m_request.texture.usage = rhi::TextureUsageFlags::Sampled | rhi::TextureUsageFlags::TransferDestination |
                              rhi::TextureUsageFlags::TransferSource;
    m_request.texture.mutableFormat = rhi::IsSrgbFormat(m_request.texture.format);
    m_request.view.dimension = dimension == TextureDimension::Texture3D ? rhi::TextureViewDimension::Texture3D
                                                                        : rhi::TextureViewDimension::Texture2D;
    m_request.view.format = m_request.texture.format;
    m_request.view.mipCount = m_request.texture.mipLevels;
    m_request.sampler = sampler;

    m_subresources.reserve(mips.size());
    for (uint32_t mipLevel = 0; mipLevel < mips.size(); ++mipLevel) {
        const auto &mip = mips[mipLevel];
        m_subresources.push_back({nullptr, static_cast<size_t>(mip.byteSize), mipLevel, 0, 1, mip.width, mip.height,
                                  mip.depth, static_cast<size_t>(mip.rowPitch), static_cast<size_t>(mip.slicePitch)});
    }
}

} // namespace infernux
