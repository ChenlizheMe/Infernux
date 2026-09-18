#include "AudioStreamDecoder.h"
#include <platform/filesystem/InxPath.h>
#include <dr_flac.h>
#include <dr_mp3.h>
#include <dr_wav.h>
#define STB_VORBIS_HEADER_ONLY
#include <stb/stb_vorbis.c>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <stdexcept>

namespace infernux
{
struct AudioStreamDecoder::Impl
{
    drwav wav{};
    drmp3 mp3{};
    drflac *flac = nullptr;
    stb_vorbis *vorbis = nullptr;
    enum class Kind { None, Wav, Mp3, Flac, Vorbis } kind = Kind::None;
    uint64_t frames = 0;
    int rate = 0, channels = 0;
    ~Impl()
    {
        if (kind == Kind::Wav) drwav_uninit(&wav);
        if (kind == Kind::Mp3) drmp3_uninit(&mp3);
        if (flac) drflac_close(flac);
        if (vorbis) stb_vorbis_close(vorbis);
    }
};

AudioStreamDecoder::AudioStreamDecoder(const std::string &path) : m_impl(std::make_unique<Impl>())
{
    auto &d = *m_impl;
    const auto file = ToFsPath(path);
    auto ext = FromFsPath(file.extension());
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (ext == ".wav") {
#ifdef _WIN32
        const bool opened = drwav_init_file_w(&d.wav, file.c_str(), nullptr);
#else
        const bool opened = drwav_init_file(&d.wav, file.c_str(), nullptr);
#endif
        if (!opened) throw std::runtime_error("Cannot open WAV: " + path);
        d.kind = Impl::Kind::Wav;
        d.frames = d.wav.totalPCMFrameCount; d.rate = d.wav.sampleRate; d.channels = d.wav.channels;
    } else if (ext == ".mp3") {
#ifdef _WIN32
        const bool opened = drmp3_init_file_w(&d.mp3, file.c_str(), nullptr);
#else
        const bool opened = drmp3_init_file(&d.mp3, file.c_str(), nullptr);
#endif
        if (!opened) throw std::runtime_error("Cannot open MP3: " + path);
        d.kind = Impl::Kind::Mp3;
        d.frames = drmp3_get_pcm_frame_count(&d.mp3); d.rate = d.mp3.sampleRate; d.channels = d.mp3.channels;
        if (!drmp3_seek_to_pcm_frame(&d.mp3, 0)) throw std::runtime_error("Cannot rewind MP3: " + path);
    } else if (ext == ".flac") {
#ifdef _WIN32
        d.flac = drflac_open_file_w(file.c_str(), nullptr);
#else
        d.flac = drflac_open_file(file.c_str(), nullptr);
#endif
        if (!d.flac) throw std::runtime_error("Cannot open FLAC: " + path);
        d.kind = Impl::Kind::Flac;
        d.frames = d.flac->totalPCMFrameCount; d.rate = d.flac->sampleRate; d.channels = d.flac->channels;
    } else if (ext == ".ogg") {
#ifdef _WIN32
        FILE *input = _wfopen(file.c_str(), L"rb");
#else
        FILE *input = std::fopen(file.c_str(), "rb");
#endif
        if (!input) throw std::runtime_error("Cannot open Ogg: " + path);
        int error = 0;
        d.vorbis = stb_vorbis_open_file(input, true, &error, nullptr);
        if (!d.vorbis) throw std::runtime_error("Cannot decode Ogg: " + path);
        d.kind = Impl::Kind::Vorbis;
        const auto info = stb_vorbis_get_info(d.vorbis);
        d.frames = stb_vorbis_stream_length_in_samples(d.vorbis); d.rate = info.sample_rate; d.channels = info.channels;
    } else {
        throw std::invalid_argument("Unsupported audio format: " + ext);
    }
    if (!d.frames || d.frames > UINT32_MAX || d.rate <= 0 || d.channels <= 0 || d.channels > 8)
        throw std::runtime_error("Unsupported audio duration/channel layout: " + path);
}
AudioStreamDecoder::~AudioStreamDecoder() = default;
uint64_t AudioStreamDecoder::FrameCount() const { return m_impl->frames; }
int AudioStreamDecoder::SampleRate() const { return m_impl->rate; }
int AudioStreamDecoder::Channels() const { return m_impl->channels; }
size_t AudioStreamDecoder::Read(float *out, size_t frames)
{
    auto &d = *m_impl;
    switch (d.kind) {
    case Impl::Kind::Wav: return drwav_read_pcm_frames_f32(&d.wav, frames, out);
    case Impl::Kind::Mp3: return drmp3_read_pcm_frames_f32(&d.mp3, frames, out);
    case Impl::Kind::Flac: return drflac_read_pcm_frames_f32(d.flac, frames, out);
    case Impl::Kind::Vorbis:
        return stb_vorbis_get_samples_float_interleaved(d.vorbis, d.channels, out,
            static_cast<int>(std::min(frames, size_t(INT_MAX / d.channels))) * d.channels);
    default: throw std::logic_error("Audio decoder not open");
    }
}
void AudioStreamDecoder::Seek(uint64_t frame)
{
    auto &d = *m_impl;
    if (frame >= d.frames) throw std::out_of_range("Audio seek outside source");
    bool ok = false;
    switch (d.kind) {
    case Impl::Kind::Wav: ok = drwav_seek_to_pcm_frame(&d.wav, frame); break;
    case Impl::Kind::Mp3: ok = drmp3_seek_to_pcm_frame(&d.mp3, frame); break;
    case Impl::Kind::Flac: ok = drflac_seek_to_pcm_frame(d.flac, frame); break;
    case Impl::Kind::Vorbis: ok = stb_vorbis_seek(d.vorbis, static_cast<unsigned>(frame)); break;
    default: break;
    }
    if (!ok) throw std::runtime_error("Audio decoder seek failed");
}
}
