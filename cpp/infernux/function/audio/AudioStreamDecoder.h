#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace infernux
{
/// File-backed decoder; owned by one loader/worker, never by the audio callback.
class AudioStreamDecoder
{
  public:
    explicit AudioStreamDecoder(const std::string &path);
    ~AudioStreamDecoder();
    AudioStreamDecoder(const AudioStreamDecoder &) = delete;
    AudioStreamDecoder &operator=(const AudioStreamDecoder &) = delete;
    uint64_t FrameCount() const;
    int SampleRate() const;
    int Channels() const;
    size_t Read(float *interleaved, size_t frames);
    void Seek(uint64_t frame);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
}
