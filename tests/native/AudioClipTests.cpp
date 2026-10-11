#include <function/audio/AudioClip.h>
#include <function/audio/AudioStreamDecoder.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace
{

template <typename T> void Write(std::ofstream &stream, const T &value)
{
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void WriteStereoWave(const std::filesystem::path &path)
{
    constexpr uint16_t channelCount = 2;
    constexpr uint32_t sampleRate = 22050;
    constexpr uint16_t bitsPerSample = 16;
    constexpr std::array<int16_t, 8> samples{1000, -1000, 2000, -2000, 3000, -3000, 4000, -4000};
    constexpr uint32_t dataBytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    constexpr uint32_t riffBytes = 36 + dataBytes;
    constexpr uint32_t formatBytes = 16;
    constexpr uint16_t pcmFormat = 1;
    constexpr uint32_t byteRate = sampleRate * channelCount * bitsPerSample / 8;
    constexpr uint16_t blockAlign = channelCount * bitsPerSample / 8;

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write("RIFF", 4);
    Write(stream, riffBytes);
    stream.write("WAVEfmt ", 8);
    Write(stream, formatBytes);
    Write(stream, pcmFormat);
    Write(stream, channelCount);
    Write(stream, sampleRate);
    Write(stream, byteRate);
    Write(stream, blockAlign);
    Write(stream, bitsPerSample);
    stream.write("data", 4);
    Write(stream, dataBytes);
    stream.write(reinterpret_cast<const char *>(samples.data()), dataBytes);
    assert(stream.good());
}

} // namespace

int main()
{
    const auto path = std::filesystem::temp_directory_path() / "infernux_audio_clip_shared_pcm.wav";
    WriteStereoWave(path);

    infernux::AudioClip clip;
    assert(clip.LoadFromFile(infernux::FromFsPath(path)));
    auto first = clip.AcquirePlaybackPcm(44100);
    auto second = clip.AcquirePlaybackPcm(44100);
    assert(first && first == second);
    assert(first->sampleRate == 44100);
    assert(first->frameCount > 0);
    assert(first->stereoFrames.size() == first->frameCount * 2);

    auto differentRate = clip.AcquirePlaybackPcm(22050);
    assert(differentRate && differentRate != first);
    assert(first->frameCount > 0);

    clip.Unload();
    assert(!clip.AcquirePlaybackPcm(44100));
    assert(first->frameCount > 0);
    for (bool mono : {false, true}) {
        for (bool streaming : {false, true}) {
            infernux::InxResourceMeta metadata;
            metadata.AddMetadata("force_mono", mono);
            metadata.AddMetadata("load_type", std::string(streaming ? "streaming" : "decompress_on_load"));
            assert(clip.LoadFromFile(infernux::FromFsPath(path), &metadata));
            assert(clip.IsStreaming() == streaming && clip.GetChannels() == (mono ? 1 : 2));
            if (!streaming) {
                const auto pcm = clip.AcquirePlaybackPcm(22050);
                assert(pcm);
                if (mono)
                    assert(std::all_of(pcm->stereoFrames.begin(), pcm->stereoFrames.end(), [](float f) { return f == 0; }));
            }
        }
    }
    clip.Unload();
    std::filesystem::remove(path);
    // Original PCM supports partial reads, backward seeks and EOF without a cook.
    constexpr uint32_t frames = 25003;
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write("RIFF", 4);
        Write(stream, uint32_t(36 + frames * 4));
        stream.write("WAVEfmt ", 8);
        Write(stream, uint32_t(16));
        Write(stream, uint16_t(1)); Write(stream, uint16_t(2));
        Write(stream, uint32_t(48000)); Write(stream, uint32_t(48000 * 4));
        Write(stream, uint16_t(4)); Write(stream, uint16_t(16));
        stream.write("data", 4); Write(stream, uint32_t(frames * 4));
        for (uint32_t i = 0; i < frames; ++i) {
            const auto sample = static_cast<int>(i % 2000) - 1000;
            Write(stream, static_cast<int16_t>(sample));
            Write(stream, static_cast<int16_t>(-sample));
        }
    }
    {
        infernux::AudioStreamDecoder decoder(infernux::FromFsPath(path));
        for (const uint32_t start : {0u, 8189u, 16400u, 24000u, 1u, frames - 1}) {
            decoder.Seek(start);
            std::vector<float> samples(9000 * 2);
            const auto count = decoder.Read(samples.data(), 9000);
            assert(count == std::min(9000u, frames - start));
            for (size_t i = 0; i < count; ++i) {
                const float expected = (static_cast<int>((start + i) % 2000) - 1000) / 32768.0f;
                assert(samples[i * 2] == expected && samples[i * 2 + 1] == -expected);
            }
            if (start + count == frames)
                assert(decoder.Read(samples.data(), 1) == 0);
        }
        bool rejected = false;
        try { decoder.Seek(frames); }
        catch (const std::out_of_range &) { rejected = true; }
        assert(rejected);
    }
    std::filesystem::remove(path);
    return 0;
}
