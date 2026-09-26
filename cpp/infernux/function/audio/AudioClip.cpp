#include "AudioClip.h"
#include "AudioStreamBuffer.h"

#include <core/log/InxLog.h>
#include <function/resources/InxResource/InxResourceMeta.h>

#include <platform/filesystem/InxPath.h>

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>

namespace infernux
{
size_t AudioClip::GetRuntimeMemoryBytes() const
{
    std::lock_guard<std::mutex> lock(m_playbackMutex);
    const size_t playbackBytes = m_playbackPcm ? m_playbackPcm->stereoFrames.capacity() * sizeof(float) : 0;
    return sizeof(*this) + m_filePath.capacity() + m_name.capacity() + m_guid.capacity() + m_data.capacity() +
           playbackBytes;
}

AudioClip::~AudioClip()
{
    Unload();
}

AudioClip::AudioClip(AudioClip &&other) noexcept
    : m_loaded(other.m_loaded), m_streaming(other.m_streaming), m_forceMono(other.m_forceMono),
      m_frameCount(other.m_frameCount), m_filePath(std::move(other.m_filePath)), m_name(std::move(other.m_name)),
      m_guid(std::move(other.m_guid)), m_spec(other.m_spec), m_data(std::move(other.m_data)),
      m_dataLength(other.m_dataLength), m_playbackPcm(std::move(other.m_playbackPcm))
{
    other.m_loaded = false;
    other.m_spec = {};
    other.m_dataLength = 0;
}

AudioClip &AudioClip::operator=(AudioClip &&other) noexcept
{
    if (this != &other) {
        Unload();
        m_loaded = other.m_loaded;
        m_streaming = other.m_streaming;
        m_forceMono = other.m_forceMono;
        m_frameCount = other.m_frameCount;
        m_filePath = std::move(other.m_filePath);
        m_name = std::move(other.m_name);
        m_guid = std::move(other.m_guid);
        m_spec = other.m_spec;
        m_data = std::move(other.m_data);
        m_dataLength = other.m_dataLength;
        m_playbackPcm = std::move(other.m_playbackPcm);

        other.m_loaded = false;
        other.m_spec = {};
        other.m_dataLength = 0;
    }
    return *this;
}

bool AudioClip::LoadFromFile(const std::string &filePath)
{
    Unload();
    try {
        InxResourceMeta meta;
        const auto metaPath = InxResourceMeta::GetMetaFilePath(filePath);
        if (std::filesystem::exists(ToFsPath(metaPath))) {
            if (!meta.LoadFromFile(metaPath))
                throw std::runtime_error("Cannot read audio import metadata");
            if (meta.HasKey("force_mono"))
                m_forceMono = meta.GetDataAs<bool>("force_mono");
            const auto mode =
                meta.HasKey("load_type") ? meta.GetDataAs<std::string>("load_type") : "decompress_on_load";
            if (mode != "decompress_on_load" && mode != "streaming")
                throw std::invalid_argument("Audio load_type must be decompress_on_load or streaming");
            m_streaming = mode == "streaming";
        }
        AudioStreamDecoder decoder(filePath);
        m_frameCount = static_cast<uint32_t>(decoder.FrameCount());
        m_spec = {SDL_AUDIO_F32, decoder.Channels(), decoder.SampleRate()};
        if (!m_streaming) {
            const uint64_t bytes = decoder.FrameCount() * decoder.Channels() * sizeof(float);
            if (bytes > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("Audio PCM exceeds resident size limit; use Streaming");
            m_data.resize(static_cast<size_t>(bytes));
            size_t read = 0;
            while (read < m_frameCount) {
                const auto got = decoder.Read(reinterpret_cast<float *>(m_data.data()) + read * m_spec.channels,
                                              std::min<size_t>(8192, m_frameCount - read));
                if (!got)
                    throw std::runtime_error("Audio source ended before its declared frame count");
                read += got;
            }
            m_dataLength = static_cast<uint32_t>(bytes);
            if (m_forceMono)
                ConvertToMono();
        } else if (m_forceMono) {
            m_spec.channels = 1;
        }
        m_filePath = filePath;
        m_name = FromFsPath(ToFsPath(filePath).stem());
        m_loaded = true;
        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Failed to load audio '", filePath, "': ", e.what());
        Unload();
        return false;
    }
}

std::unique_ptr<AudioStreamBuffer> AudioClip::CreateStream(uint64_t firstFrame) const
{
    if (!m_loaded || !m_streaming)
        throw std::logic_error("AudioClip is not a loaded stream");
    auto decoder = std::make_unique<AudioStreamDecoder>(m_filePath);
    if (decoder->FrameCount() != m_frameCount || decoder->SampleRate() != m_spec.freq)
        throw std::runtime_error("Audio source changed; reimport before creating a voice");
    return std::make_unique<AudioStreamBuffer>(std::move(decoder), m_forceMono, firstFrame);
}

void AudioClip::ConvertToMono()
{
    if (m_spec.channels <= 1)
        return;
    const int channels = m_spec.channels;
    auto *samples = reinterpret_cast<float *>(m_data.data());
    for (size_t frame = 0; frame < m_frameCount; ++frame) {
        float sum = 0;
        for (int channel = 0; channel < channels; ++channel)
            sum += samples[frame * channels + channel];
        samples[frame] = sum / channels;
    }
    m_data.resize(static_cast<size_t>(m_frameCount) * sizeof(float));
    m_data.shrink_to_fit();
    m_dataLength = static_cast<uint32_t>(m_data.size());
    m_spec.channels = 1;
}

void AudioClip::Unload()
{
    {
        std::lock_guard<std::mutex> lock(m_playbackMutex);
        m_playbackPcm.reset();
    }
    m_data.clear();
    m_data.shrink_to_fit();
    m_spec = {};
    m_dataLength = 0;
    m_loaded = false;
    m_streaming = false;
    m_forceMono = false;
    m_frameCount = 0;
}

std::shared_ptr<const AudioPlaybackPcm> AudioClip::AcquirePlaybackPcm(int sampleRate) const
{
    if (!m_loaded || m_data.empty() || sampleRate <= 0)
        return nullptr;

    std::lock_guard<std::mutex> lock(m_playbackMutex);
    if (m_playbackPcm && m_playbackPcm->sampleRate == sampleRate)
        return m_playbackPcm;

    SDL_AudioSpec playbackSpec = {};
    playbackSpec.format = SDL_AUDIO_F32;
    playbackSpec.channels = 2;
    playbackSpec.freq = sampleRate;

    Uint8 *convertedData = nullptr;
    int convertedLength = 0;
    if (!SDL_ConvertAudioSamples(&m_spec, m_data.data(), static_cast<int>(m_data.size()), &playbackSpec, &convertedData,
                                 &convertedLength)) {
        INXLOG_ERROR("Failed to prepare audio clip for playback: ", SDL_GetError());
        return nullptr;
    }

    auto playback = std::make_shared<AudioPlaybackPcm>();
    playback->stereoFrames.resize(static_cast<size_t>(convertedLength) / sizeof(float));
    std::memcpy(playback->stereoFrames.data(), convertedData, static_cast<size_t>(convertedLength));
    playback->frameCount = playback->stereoFrames.size() / 2;
    playback->sampleRate = sampleRate;
    SDL_free(convertedData);
    m_playbackPcm = playback;
    return playback;
}

float AudioClip::GetDuration() const
{
    if (!m_loaded || m_spec.freq == 0 || m_spec.channels == 0) {
        return 0.0f;
    }

    const int bytesPerSample = SDL_AUDIO_BYTESIZE(m_spec.format);
    if (bytesPerSample == 0) {
        return 0.0f;
    }

    return static_cast<float>(m_frameCount) / static_cast<float>(m_spec.freq);
}

uint32_t AudioClip::GetSampleCount() const
{
    if (!m_loaded || m_spec.channels == 0) {
        return 0;
    }

    const int bytesPerSample = SDL_AUDIO_BYTESIZE(m_spec.format);
    if (bytesPerSample == 0) {
        return 0;
    }

    return m_frameCount;
}

} // namespace infernux
