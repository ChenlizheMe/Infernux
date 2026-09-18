#include <SDL3/SDL.h>
#include <function/audio/AudioClip.h>
#include <function/audio/AudioEngine.h>
#include <function/audio/AudioSource.h>
#include <function/audio/AudioStreamBuffer.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
template <typename T> void Write(std::ofstream &stream, T value)
{
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void WriteWave(const std::filesystem::path &path)
{
    constexpr uint32_t rate = 44100;
    std::vector<int16_t> samples(rate * 3 * 2);
    for (size_t i = 0; i < samples.size() / 2; ++i) {
        // Each second has a different value so a seek can detect old queued
        // samples. Opposite stereo signs also catch accidental downmixing.
        samples[i * 2] = static_cast<int16_t>(1000 * (1 + i / rate));
        samples[i * 2 + 1] = -samples[i * 2];
    }
    const uint32_t bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    std::ofstream stream(path, std::ios::binary);
    stream.write("RIFF", 4);
    Write<uint32_t>(stream, bytes + 36);
    stream.write("WAVEfmt ", 8);
    Write<uint32_t>(stream, 16);
    Write<uint16_t>(stream, 1);
    Write<uint16_t>(stream, 2);
    Write<uint32_t>(stream, rate);
    Write<uint32_t>(stream, rate * 4);
    Write<uint16_t>(stream, 4);
    Write<uint16_t>(stream, 16);
    stream.write("data", 4);
    Write<uint32_t>(stream, bytes);
    stream.write(reinterpret_cast<const char *>(samples.data()), bytes);
    assert(stream.good());
}

template <typename Predicate> void WaitForAt(Predicate predicate, unsigned line)
{
    const auto deadline = SDL_GetTicks() + 2000;
    while (!predicate() && SDL_GetTicks() < deadline)
        SDL_Delay(5);
    if (!predicate()) {
        const auto &audio = infernux::AudioEngine::Instance();
        std::fprintf(stderr, "Audio wait failed at line %u: time=%f peak=%f saturated=%llu\n", line,
                     audio.GetOutputTime(), audio.GetOutputPeak(),
                     static_cast<unsigned long long>(audio.GetSaturatedSampleCount()));
        assert(false);
    }
}
} // namespace

#define WaitFor(predicate) WaitForAt(predicate, __LINE__)

int main()
{
    assert(SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "dummy"));
    auto &engine = infernux::AudioEngine::Instance();
    assert(engine.Initialize());
    const auto path = std::filesystem::temp_directory_path() /
                      ("infernux-audio-playback-" + std::to_string(SDL_GetPerformanceCounter()) + ".wav");
    WriteWave(path);
    auto clip = std::make_shared<infernux::AudioClip>();
    assert(clip->LoadFromFile(infernux::FromFsPath(path)));

    // The device runs its own clock even when the game thread never updates.
    infernux::AudioSource source;
    source.SetPlayOnAwake(false);
    source.SetSpatialBlend(0.0f);
    source.SetTrackCount(2);
    source.SetTrackClip(0, clip);
    source.SetTrackClip(1, clip);
    source.SetLoop(true);
    source.Play(0);
    source.Play(1);
    WaitFor([&] { return source.GetTrackTime(0) > 0.02 && source.GetTrackTime(1) > 0.02; });
    source.Pause(0);
    const double frozen = source.GetTrackTime(0);
    const double moving = source.GetTrackTime(1);
    WaitFor([&] { return source.GetTrackTime(1) > moving + 0.03; });
    assert(source.GetTrackTime(0) == frozen);

    source.SetTrackTime(0, 1.25);
    SDL_Delay(30);
    assert(source.IsTrackPaused(0));
    assert(std::abs(source.GetTrackTime(0) - 1.25) < 1e-6);
    const auto authored = source.SerializeDocument();
    source.SetTrackTime(0, 1.5);
    assert(source.SerializeDocument() == authored);
    source.SetTrackTime(0, 1.25);
    source.UnPause(0);
    WaitFor([&] { return source.GetTrackTime(0) > 1.28; });

    // Creating or unpausing a voice must not release global device pause.
    engine.PauseAll();
    source.Play(1);
    source.Pause(0);
    source.UnPause(0);
    const double globalFrozen = source.GetTrackTime(0);
    SDL_Delay(60);
    assert(engine.IsPaused());
    assert(source.GetTrackTime(0) == globalFrozen);
    assert(source.GetTrackTime(1) == 0.0);
    source.Pause(0);
    engine.ResumeAll();
    WaitFor([&] { return source.GetTrackTime(1) > 0.03; });
    assert(source.GetTrackTime(0) == globalFrozen);

    // Re-enabling resumes only tracks paused by disable, not manually paused
    // tracks. The disable reason must survive Pause()'s manual-state reset.
    source.OnDisable();
    const double disabledTime = source.GetTrackTime(1);
    SDL_Delay(30);
    assert(source.GetTrackTime(1) == disabledTime);
    source.OnEnable();
    WaitFor([&] { return source.GetTrackTime(1) > disabledTime + 0.03; });
    assert(source.IsTrackPaused(0));
    assert(source.GetTrackTime(0) == globalFrozen);

    source.Stop(0);
    assert(source.GetTrackTime(0) == 0.0);
    source.SetTrackTime(0, 2.0);
    assert(!source.IsTrackPlaying(0));
    engine.PauseAll();
    source.Play(0);
    assert(std::abs(source.GetTrackTime(0) - 2.0) < 1e-6);
    source.Stop(0);
    source.SetTrackTime(0, 2.0);
    source.SetTrackClip(0, clip);
    assert(source.GetTrackTime(0) == 0.0);
    for (double invalid :
         {-1.0, 3.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected = false;
        try {
            source.SetTrackTime(0, invalid);
        } catch (const std::exception &) {
            rejected = true;
        }
        assert(rejected);
    }
    source.StopAll();

    // Pull through the actual SDL callback while unbound, deterministically.
    SDL_AudioStream *voice = engine.CreateVoice(nullptr, clip.get());
    assert(voice);
    engine.UpdateVoiceMix(voice, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, false);
    std::array<float, 2048> output{};
    assert(SDL_GetAudioStreamData(voice, output.data(), sizeof(output)) > 0);
    assert(engine.GetVoiceTime(voice) > 0.0);
    engine.SetVoiceTime(voice, 2.0);
    assert(SDL_GetAudioStreamData(voice, output.data(), sizeof(output)) > 0);
    assert(std::abs(output[1800] - 3000.0f / 32768.0f) < 0.001f);
    assert(std::abs(output[1801] + 3000.0f / 32768.0f) < 0.001f);
    engine.SetVoiceTime(voice, 0.0);
    assert(SDL_GetAudioStreamData(voice, output.data(), sizeof(output)) > 0);
    const double normalStep = engine.GetVoiceTime(voice);
    engine.SetVoiceTime(voice, 0.0);
    engine.UpdateVoiceMix(voice, 1.0f, 1.0f, 0.0f, 0.0f, 2.0f, false);
    assert(SDL_GetAudioStreamData(voice, output.data(), sizeof(output)) > 0);
    assert(std::abs(engine.GetVoiceTime(voice) - normalStep * 2.0) < 1e-6);
    engine.SetVoiceTime(voice, 3.0);
    assert(SDL_GetAudioStreamData(voice, output.data(), sizeof(output)) == 0);
    assert(engine.HasVoiceFinished(voice));
    assert(SDL_PutAudioStreamData(voice, output.data(), sizeof(output)));
    assert(SDL_FlushAudioStream(voice));
    assert(!engine.HasVoiceFinished(voice));
    assert(SDL_GetAudioStreamData(voice, output.data(), sizeof(output)) > 0);
    assert(engine.HasVoiceFinished(voice));
    engine.UpdateVoiceMix(voice, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, true);
    assert(SDL_GetAudioStreamData(voice, output.data(), sizeof(output)) > 0);
    assert(engine.GetVoiceTime(voice) < 0.1);
    assert(!engine.HasVoiceFinished(voice));
    engine.DestroyVoice(voice);
    assert(engine.GetActiveVoiceCount() == 0);

    // Group automation runs on the actual audio thread, not Update(deltaTime).
    // With a constant-valued stereo clip the output meter proves the gain is
    // applied to samples, not only to a main-thread status field.
    engine.SetMasterVolume(0.5f);
    engine.SetBusVolume("Music", 0.5f);
    source.SetOutputBus("Music");
    source.SetTrackCount(1);
    source.SetTrackTime(0, 1.0);
    source.Play(0);
    const double pausedClock = engine.GetOutputTime();
    engine.FadeBusVolume("Music", 0.25f, 0.1f);
    engine.Update(100.0f);
    SDL_Delay(30);
    assert(engine.GetOutputTime() == pausedClock);
    assert(engine.GetBusVolume("Music") == 0.5f);
    assert(engine.IsBusFading("Music"));
    engine.CancelBusFade("Music");
    engine.ResumeAll();
    WaitFor([&] { return engine.GetOutputTime() > pausedClock + 0.1; });
    assert(std::abs(engine.GetOutputPeak() - (2000.0f / 32768.0f * 0.25f)) < 0.001f);
    engine.FadeBusVolume("Music", 0.0f, 0.1f);
    WaitFor([&] { return !engine.IsBusFading("Music") && engine.GetOutputPeak() < 0.00001f; });
    assert(source.IsTrackPlaying(0));
    const double afterFade = source.GetTrackTime(0);
    engine.SetBusMuted("Music", true);
    engine.FadeBusVolume("Music", 1.0f, 0.1f);
    WaitFor([&] { return !engine.IsBusFading("Music"); });
    assert(engine.GetBusVolume("Music") == 1.0f);
    assert(engine.GetOutputPeak() < 0.00001f);
    assert(source.GetTrackTime(0) > afterFade);
    source.SetOutputBus("SFX");
    WaitFor([&] { return engine.GetOutputPeak() > 0.025f; });
    source.SetOutputBus("Music");
    WaitFor([&] { return engine.GetOutputPeak() < 0.00001f; });
    engine.SetBusMuted("Music", false);
    WaitFor([&] { return engine.GetOutputPeak() > 0.025f; });
    engine.FadeBusVolume("Master", 0.0f, 0.1f);
    WaitFor([&] { return !engine.IsBusFading("Master") && engine.GetOutputPeak() < 0.00001f; });
    source.StopAll();
    engine.SetMasterVolume(1.0f);
    const double silentClock = engine.GetOutputTime();
    WaitFor([&] { return engine.GetOutputTime() > silentClock + 0.03; });

    voice = engine.CreateVoice(nullptr, clip.get());
    engine.UpdateVoiceMix(voice, 64.0f, 1.0f, 0.0f, 0.0f, 1.0f, true);
    engine.SetVoicePaused(voice, false);
    WaitFor([&] { return engine.GetOutputPeak() == 1.0f && engine.GetSaturatedSampleCount() > 0; });
    engine.DestroyVoice(voice);
    assert(engine.GetActiveVoiceCount() == 0);
    // Shutdown also owns raw voices; callback userdata cannot be freed before
    // SDL finishes its last callback. A fresh device gets a fresh clock.
    voice = engine.CreateVoice(nullptr, clip.get());
    engine.UpdateVoiceMix(voice, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, true);
    engine.SetVoicePaused(voice, false);
    engine.Shutdown();
    assert(engine.GetActiveVoiceCount() == 0);
    assert(engine.GetOutputTime() == 0.0);
    assert(engine.GetSaturatedSampleCount() == 0);
    assert(engine.Initialize());
    WaitFor([&] { return engine.GetOutputTime() > 0.03; });

    engine.SetMaxRealVoices(4);
    std::vector<SDL_AudioStream *> crowd;
    const auto clipBytes = clip->GetRuntimeMemoryBytes();
    for (int index = 0; index < 512; ++index) {
        auto *stream = engine.CreateVoice(nullptr, clip.get());
        assert(stream);
        engine.UpdateVoiceMix(stream, 0.1f, 1.0f, 0.0f, 0.0f, 1.0f, true, "Master", index == 511 ? 0 : 128);
        engine.SetVoicePaused(stream, false);
        crowd.push_back(stream);
        assert(engine.GetRealVoiceCount() <= 4);
    }
    engine.Update(0.0f);
    assert(engine.GetRealVoiceCount() == 4);
    assert(engine.GetVirtualVoiceCount() == 508);
    assert(!engine.IsVoiceVirtual(crowd[0]));
    assert(engine.IsVoiceVirtual(crowd[3]));
    assert(!engine.IsVoiceVirtual(crowd[511]));
    assert(clip->GetRuntimeMemoryBytes() == clipBytes);
    assert(SDL_GetAudioStreamAvailable(crowd[10]) == 0);
    std::vector<double> schedulingMicros;
    for (int sample = 0; sample < 128; ++sample) {
        const uint64_t before = SDL_GetPerformanceCounter();
        engine.Update(0.0f);
        schedulingMicros.push_back(1.0e6 * (SDL_GetPerformanceCounter() - before) / SDL_GetPerformanceFrequency());
    }
    std::sort(schedulingMicros.begin(), schedulingMicros.end());
    std::printf("INFERNUX_AUDIO_SCHEDULER logical=512 real=4 p50_us=%.3f p95_us=%.3f\n", schedulingMicros[64],
                schedulingMicros[121]);

    engine.PauseAll();
    const double virtualStart = engine.GetVoiceTime(crowd[10]);
    const double virtualClock = engine.GetOutputTime();
    engine.ResumeAll();
    WaitFor([&] { return engine.GetOutputTime() > virtualClock + 0.05; });
    engine.PauseAll();
    assert(std::abs(engine.GetVoiceTime(crowd[10]) - virtualStart - (engine.GetOutputTime() - virtualClock)) < 0.00001);
    const double virtualPaused = engine.GetVoiceTime(crowd[10]);
    SDL_Delay(30);
    assert(engine.GetVoiceTime(crowd[10]) == virtualPaused);

    engine.UpdateVoiceMix(crowd[10], 0.1f, 1.0f, 0.0f, 0.0f, 2.0f, true);
    const double pitchClock = engine.GetOutputTime();
    engine.ResumeAll();
    WaitFor([&] { return engine.GetOutputTime() > pitchClock + 0.05; });
    engine.PauseAll();
    assert(std::abs(engine.GetVoiceTime(crowd[10]) - virtualPaused - 2.0 * (engine.GetOutputTime() - pitchClock)) <
           0.00001);

    engine.UpdateVoiceMix(crowd[3], 0.1f, 1.0f, 0.0f, 0.0f, 1.0f, true, "Master", 32);
    engine.Update(0.0f);
    assert(!engine.IsVoiceVirtual(crowd[3]));
    assert(engine.IsVoiceVirtual(crowd[2]));
    assert(engine.GetRealVoiceCount() == 4);
    engine.SetVoicePaused(crowd[12], true);
    const double manuallyPaused = engine.GetVoiceTime(crowd[12]);
    engine.SetVoiceTime(crowd[10], 2.99);
    engine.UpdateVoiceMix(crowd[10], 0.1f, 1.0f, 0.0f, 0.0f, 1.0f, false);
    engine.ResumeAll();
    WaitFor([&] { return engine.HasVoiceFinished(crowd[10]); });
    engine.PauseAll();
    const double promotionTime = engine.GetVoiceTime(crowd[11]);
    engine.SetMaxRealVoices(512);
    assert(!engine.IsVoiceVirtual(crowd[11]));
    assert(engine.GetVoiceTime(crowd[11]) == promotionTime);
    assert(engine.GetRealVoiceCount() == 510);
    assert(engine.GetVirtualVoiceCount() == 0);
    assert(engine.GetVoiceTime(crowd[12]) == manuallyPaused);
    engine.SetMaxRealVoices(4);
    assert(engine.GetRealVoiceCount() == 4);
    for (auto *stream : crowd)
        engine.DestroyVoice(stream);
    assert(engine.GetRealVoiceCount() == 0);
    assert(engine.GetVirtualVoiceCount() == 0);
    engine.SetMaxRealVoices(64);

    source.SetOneShotPoolSize(2);
    source.PlayOneShot(clip, 0.9f);
    source.PlayOneShot(clip, 0.7f);
    const auto shots = source.GetActiveStreams();
    assert(shots.size() == 2);
    engine.SetVoiceTime(shots[1], 1.0);
    source.PlayOneShot(clip, 0.2f);
    assert(source.GetRejectedOneShotCount() == 1);
    assert(source.GetActiveStreams() == shots);
    source.PlayOneShot(clip, 0.8f);
    const auto replaced = source.GetActiveStreams();
    assert(replaced.size() == 2);
    assert(replaced[0] == shots[0]);
    assert(engine.GetVoiceTime(replaced[1]) == 0.0);
    assert(source.GetRejectedOneShotCount() == 1);
    source.StopOneShots();
    assert(engine.GetRealVoiceCount() == 0);

    // A distant 3D source is virtual even with spare physical capacity. Its
    // loop runs on device time; returning to audible range resumes that time
    // rather than replaying the initial PCM or a stale pre-virtualization tail.
    auto *distant = engine.CreateVoice(nullptr, clip.get());
    engine.UpdateVoiceMix(distant, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, true, "SFX", 0);
    engine.SetVoicePaused(distant, false);
    assert(engine.IsVoiceVirtual(distant));
    assert(engine.GetRealVoiceCount() == 0);
    engine.SetVoiceTime(distant, 2.98);
    const double loopClock = engine.GetOutputTime();
    engine.ResumeAll();
    WaitFor([&] { return engine.GetOutputTime() > loopClock + 0.05; });
    engine.PauseAll();
    assert(std::abs(engine.GetVoiceTime(distant) - std::fmod(2.98 + engine.GetOutputTime() - loopClock, 3.0)) <
           0.00001);
    assert(SDL_GetAudioStreamAvailable(distant) == 0);
    engine.SetVoiceTime(distant, 2.2);
    engine.UpdateVoiceMix(distant, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, true, "SFX", 0);
    // The test WAV's opposite L/R values cancel in a mono point source, so
    // switch to the non-spatial stereo path to inspect the resumed samples.
    engine.UpdateVoiceMix(distant, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, true, "SFX", 0);
    engine.Update(0.0f);
    assert(!engine.IsVoiceVirtual(distant));
    assert(std::abs(engine.GetVoiceTime(distant) - 2.2) < 0.00001);
    engine.ResumeAll();
    WaitFor([&] { return std::abs(engine.GetOutputPeak() - 3000.0f / 32768.0f) < 0.001f; });

    engine.SetBusMuted("SFX", true);
    WaitFor([&] { return engine.GetOutputPeak() < 0.00001f; });
    engine.Update(0.0f);
    assert(engine.IsVoiceVirtual(distant));
    assert(engine.GetRealVoiceCount() == 0);
    const double mutedTime = engine.GetVoiceTime(distant);
    const double mutedClock = engine.GetOutputTime();
    WaitFor([&] { return engine.GetOutputTime() > mutedClock + 0.03; });
    assert(engine.GetVoiceTime(distant) > mutedTime);
    engine.SetBusMuted("SFX", false);
    WaitFor([&] {
        engine.Update(0.0f);
        return !engine.IsVoiceVirtual(distant);
    });
    WaitFor([&] { return engine.GetOutputPeak() > 0.08f; });
    engine.PauseAll();

    // Priority precedes loudness. At equal priority the louder source wins;
    // exact ties use stable start order rather than hash-map iteration order.
    auto *foreground = engine.CreateVoice(nullptr, clip.get());
    engine.UpdateVoiceMix(foreground, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, true, "SFX", 128);
    engine.SetVoicePaused(foreground, false);
    engine.SetMaxRealVoices(1);
    assert(!engine.IsVoiceVirtual(distant));
    assert(engine.IsVoiceVirtual(foreground));
    engine.UpdateVoiceMix(distant, 0.1f, 1.0f, 0.0f, 0.0f, 1.0f, true, "SFX", 128);
    engine.Update(0.0f);
    assert(engine.IsVoiceVirtual(distant));
    assert(!engine.IsVoiceVirtual(foreground));
    engine.UpdateVoiceMix(distant, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, true, "SFX", 128);
    engine.Update(0.0f);
    assert(!engine.IsVoiceVirtual(distant));
    assert(engine.IsVoiceVirtual(foreground));
    engine.DestroyVoice(foreground);
    engine.DestroyVoice(distant);
    engine.SetMaxRealVoices(64);
    // Stream the same source with bounded storage, through the very same voice
    // scheduler/device. Seek/loop and independent playheads must not share PCM.
    infernux::InxResourceMeta meta;
    meta.AddMetadata("load_type", std::string("streaming"));
    const auto metaPath = infernux::InxResourceMeta::GetMetaFilePath(infernux::FromFsPath(path));
    assert(meta.SaveToFile(metaPath));
    infernux::AudioClip streamed;
    assert(streamed.LoadFromFile(infernux::FromFsPath(path)));
    assert(streamed.IsStreaming() && streamed.GetSampleCount() == 44100 * 3);
    assert(streamed.GetData().empty() && streamed.GetRuntimeMemoryBytes() < 4096);
    assert(!streamed.AcquirePlaybackPcm(44100));
    {
        auto pages = streamed.CreateStream(0);
        for (uint64_t frame : {uint64_t(0), uint64_t(100000), uint64_t(44100), uint64_t(132299), uint64_t(0)}) {
            pages->Request(frame);
            float left = 0, right = 0;
            WaitFor([&] { return pages->ReadFrame(frame, left, right); });
            const float expected = (1000.0f * (1 + frame / 44100)) / 32768.0f;
            assert(std::abs(left - expected) < 0.00001f && std::abs(right + expected) < 0.00001f);
        }
    }
    engine.SetBusVolume("Master", 1);
    engine.SetBusVolume("Music", 1);
    auto *music = engine.CreateVoice(nullptr, &streamed, 1.1);
    auto *otherMusic = engine.CreateVoice(nullptr, &streamed, 0.1);
    assert(music && otherMusic);
    assert(std::abs(engine.GetVoiceTime(otherMusic) - .1) < .00001);
    engine.UpdateVoiceMix(music, 1, 1, 0, 0, 1, true, "Music", 0);
    engine.SetVoicePaused(music, false);
    engine.ResumeAll();
    WaitFor([&] { return std::abs(engine.GetOutputPeak() - 2000.0f / 32768) < .001; });
    engine.PauseAll();
    engine.SetVoiceTime(music, 2.99);
    engine.ResumeAll();
    WaitFor([&] { return engine.GetVoiceTime(music) < 0.3; });
    WaitFor([&] { return std::abs(engine.GetOutputPeak() - 1000.0f / 32768) < .001; });
    engine.SetVoicePaused(music, true);
    const auto pausedTime = engine.GetVoiceTime(music);
    SDL_Delay(30);
    assert(engine.GetVoiceTime(music) == pausedTime);
    engine.SetVoiceTime(music, 2.1);
    engine.SetVoicePaused(music, false);
    WaitFor([&] { return std::abs(engine.GetOutputPeak() - 3000.0f / 32768) < .001; });
    engine.DestroyVoice(music);
    engine.DestroyVoice(otherMusic);
    std::filesystem::remove(infernux::ToFsPath(metaPath));
    engine.Shutdown();
    std::filesystem::remove(path);
}
