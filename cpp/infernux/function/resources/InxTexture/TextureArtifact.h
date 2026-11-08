#pragma once

#include "InxTexture.h"
#include <platform/filesystem/IoWork.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace infernux
{

class TextureArtifact final
{
  public:
    [[nodiscard]] static bool HasCurrentHeader(std::string_view bytes) noexcept;
    [[nodiscard]] static TextureArtifactDescription ReadDescription(const std::string &artifactPath,
                                                                    std::string_view expectedSourceContentHash);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> Read(const std::string &artifactPath,
                                                                    std::string_view expectedSourceContentHash);
    [[nodiscard]] static std::string Serialize(const TextureCpuData &texture, std::string_view sourceContentHash);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> Deserialize(std::string_view bytes,
                                                                           std::string_view expectedSourceContentHash);
};

enum class TextureReadState : uint8_t
{
    Opening,
    Reading,
    Output,
    Closing,
    Complete,
    Closed
};

/// One imported-file snapshot and one output batch filled by admitted IO steps.
/// Output stays unchanged until its consumer releases it.
class TextureReadCursor final
{
  public:
    TextureReadCursor(std::string artifactPath, std::string sourceContentHash);
    ~TextureReadCursor();
    TextureReadState GetState() const noexcept;
    IoWork DescribeWork(size_t maximum) const;
    void Advance(size_t maximum, size_t batchBytes);
    const TextureArtifactDescription &GetDescription() const;
    const std::vector<uint8_t> &GetOutput() const;
    void ConsumeOutput();
    void Close();
    size_t GetBufferedBytes() const noexcept;
    size_t GetRetainedBytes() const noexcept;
    size_t GetReservationBytes(size_t batchBytes) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
