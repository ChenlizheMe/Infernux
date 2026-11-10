#pragma once

#include "AudioRuntimeApi.h"
#include <cstdint>
#include <memory>
#include <string>

namespace infernux
{
/// File-backed decoder; owned by one loader/worker, never by the audio callback.
class AudioStreamDecoder
{
  public:
    INFERNUX_AUDIO_RUNTIME_API explicit AudioStreamDecoder(const std::string &path);
    INFERNUX_AUDIO_RUNTIME_API ~AudioStreamDecoder();
    AudioStreamDecoder(const AudioStreamDecoder &) = delete;
    AudioStreamDecoder &operator=(const AudioStreamDecoder &) = delete;
    INFERNUX_AUDIO_RUNTIME_API uint64_t FrameCount() const;
    INFERNUX_AUDIO_RUNTIME_API int SampleRate() const;
    INFERNUX_AUDIO_RUNTIME_API int Channels() const;
    INFERNUX_AUDIO_RUNTIME_API size_t Read(float *interleaved, size_t frames);
    INFERNUX_AUDIO_RUNTIME_API void Seek(uint64_t frame);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace infernux
