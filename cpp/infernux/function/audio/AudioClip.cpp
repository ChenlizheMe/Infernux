#include "AudioClip.h"

#include <core/log/InxLog.h>
#include <function/resources/InxResource/InxResourceMeta.h>

#include <platform/filesystem/InxPath.h>

#include <SDL3/SDL.h>

#include <algorithm>
#include <filesystem>
#include <utility>
#include <cstring>

namespace infernux
{

namespace
{

bool DecodeWaveFile(const std::string &filePath, SDL_AudioSpec &spec, std::vector<uint8_t> &data)
{
    Uint8 *audioBuffer = nullptr;
    Uint32 audioLength = 0;

    if (!SDL_LoadWAV(filePath.c_str(), &spec, &audioBuffer, &audioLength)) {
        INXLOG_ERROR("Failed to load WAV file '", filePath, "': ", SDL_GetError());
        return false;
    }

    data.assign(audioBuffer, audioBuffer + audioLength);
    SDL_free(audioBuffer);
    return true;
}

bool DecodeAudioFile(const std::string &filePath, SDL_AudioSpec &spec, std::vector<uint8_t> &data)
{
    std::string extension = ToFsPath(filePath).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == ".wav") {
        return DecodeWaveFile(filePath, spec, data);
    }

    INXLOG_ERROR("Unsupported audio file format: ", filePath);
    return false;
}

} // namespace

AudioClip::~AudioClip()
{
    Unload();
}

AudioClip::AudioClip(AudioClip &&other) noexcept
    : m_loaded(other.m_loaded), m_filePath(std::move(other.m_filePath)), m_name(std::move(other.m_name)),
      m_spec(other.m_spec), m_data(std::move(other.m_data)), m_dataLength(other.m_dataLength)
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
        m_filePath = std::move(other.m_filePath);
        m_name = std::move(other.m_name);
        m_spec = other.m_spec;
        m_data = std::move(other.m_data);
        m_dataLength = other.m_dataLength;

        other.m_loaded = false;
        other.m_spec = {};
        other.m_dataLength = 0;
    }
    return *this;
}

bool AudioClip::LoadFromFile(const std::string &filePath)
{
    if (m_loaded) {
        Unload();
    }

    if (!DecodeAudioFile(filePath, m_spec, m_data)) {
        return false;
    }

    m_dataLength = static_cast<uint32_t>(m_data.size());
    m_filePath = filePath;
    m_name = FromFsPath(ToFsPath(filePath).stem());
    m_loaded = true;

    ApplyImportSettings();

    INXLOG_DEBUG("AudioClip loaded: '", m_name, "' (", m_spec.freq, " Hz, ", m_spec.channels, " ch, ", m_dataLength,
                 " bytes)");
    return true;
}

void AudioClip::ApplyImportSettings()
{
    std::string metaPath = InxResourceMeta::GetMetaFilePath(m_filePath);
    InxResourceMeta meta;
    if (!meta.LoadFromFile(metaPath)) {
        return;
    }

    if (meta.HasKey("force_mono")) {
        bool forceMono = false;
        try {
            forceMono = meta.GetDataAs<bool>("force_mono");
        } catch (...) {
            INXLOG_WARN("[AudioClip] Invalid 'force_mono' metadata in: ", metaPath);
        }

        if (forceMono && m_spec.channels > 1) {
            ConvertToMono();
        }
    }
}

void AudioClip::ConvertToMono()
{
    if (m_spec.channels <= 1) {
        return;
    }

    const int bytesPerSample = SDL_AUDIO_BYTESIZE(m_spec.format);
    if (bytesPerSample == 0) {
        return;
    }

    const int channels = m_spec.channels;
    const uint32_t frameCount = m_dataLength / (bytesPerSample * channels);
    std::vector<uint8_t> monoData(frameCount * bytesPerSample);

    const bool isFloat = SDL_AUDIO_ISFLOAT(m_spec.format);
    const bool isSigned = SDL_AUDIO_ISSIGNED(m_spec.format);

    for (uint32_t frame = 0; frame < frameCount; ++frame) {
        if (isFloat && bytesPerSample == 4) {
            float sum = 0.0f;
            for (int channel = 0; channel < channels; ++channel) {
                float sample = 0.0f;
                std::memcpy(&sample, &m_data[(frame * channels + channel) * sizeof(float)], sizeof(float));
                sum += sample;
            }
            const float mono = sum / static_cast<float>(channels);
            std::memcpy(&monoData[frame * sizeof(float)], &mono, sizeof(float));
        } else if (bytesPerSample == 2 && isSigned) {
            int32_t sum = 0;
            for (int channel = 0; channel < channels; ++channel) {
                int16_t sample = 0;
                std::memcpy(&sample, &m_data[(frame * channels + channel) * sizeof(int16_t)], sizeof(int16_t));
                sum += sample;
            }
            const int16_t mono = static_cast<int16_t>(sum / channels);
            std::memcpy(&monoData[frame * sizeof(int16_t)], &mono, sizeof(int16_t));
        } else if (bytesPerSample == 1) {
            int32_t sum = 0;
            for (int channel = 0; channel < channels; ++channel) {
                sum += m_data[frame * channels + channel];
            }
            monoData[frame] = static_cast<uint8_t>(sum / channels);
        } else {
            INXLOG_WARN("AudioClip: unsupported format for mono conversion, skipping");
            return;
        }
    }

    m_data = std::move(monoData);
    m_dataLength = static_cast<uint32_t>(m_data.size());
    m_spec.channels = 1;
}

void AudioClip::Unload()
{
    m_data.clear();
    m_data.shrink_to_fit();
    m_spec = {};
    m_dataLength = 0;
    m_loaded = false;
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

    const uint32_t totalFrames = m_dataLength / (bytesPerSample * m_spec.channels);
    return static_cast<float>(totalFrames) / static_cast<float>(m_spec.freq);
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

    return m_dataLength / (bytesPerSample * m_spec.channels);
}

} // namespace infernux
