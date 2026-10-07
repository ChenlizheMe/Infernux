#include <function/audio/AudioClip.h>
#include <platform/filesystem/InxPath.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>

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
    std::filesystem::remove(path);
    return 0;
}
