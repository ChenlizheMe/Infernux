#pragma once

#include "AudioBusAutomation.h"

#include <cstddef>

#include <SDL3/SDL_audio.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

// Forward declarations
class AudioClip;
class AudioSource;
class AudioListener;

/**
 * @brief Core audio engine managing SDL3 audio device and stream mixing.
 *
 * Singleton that owns the SDL3 audio device, manages active voices
 * (AudioStreams), and performs per-frame mixing with 3D spatialization.
 *
 * Design:
 * - One logical SDL3 audio device for all playback
 * - Each playing AudioSource owns SDL_AudioStream voices bound to the device
 * - Per-frame Update() adjusts stream gain/panning for 3D positioning
 * - Thread-safe: SDL3 audio runs its own mixing thread; we only push data
 */
class AudioEngine
{
  public:
    /// @brief Get the singleton instance
    static AudioEngine &Instance();

    /// @brief Initialize the audio subsystem (call once at engine startup)
    /// @return true on success
    bool Initialize();

    /// @brief Shutdown the audio subsystem (call at engine cleanup)
    void Shutdown();

    /// @brief Whether the audio engine is initialized
    [[nodiscard]] bool IsInitialized() const
    {
        return m_initialized;
    }

    /// @brief Per-frame update: recalculate 3D spatialization for all active sources
    /// @param deltaTime Time since last frame
    void Update(float deltaTime);

    // ========================================================================
    // Voice management (called by AudioSource)
    // ========================================================================

    /// @brief Create an unbound SDL_AudioStream for a source
    /// @param source The AudioSource requesting playback
    /// @param clip The AudioClip to play
    /// @return Stream handle, or nullptr on failure
    /// Starts unbound: configure the mix, then call SetVoicePaused(false).
    SDL_AudioStream *CreateVoice(AudioSource *source, AudioClip *clip, double startSeconds = 0.0);

    /// @brief Destroy a voice stream and unbind it from the device
    /// @param stream The stream to destroy
    void DestroyVoice(SDL_AudioStream *stream);

    /// @brief Update per-voice spatial mix and playback parameters
    void UpdateVoiceMix(SDL_AudioStream *stream, float gain, float spatialGain, float pan, float spatialBlend,
                        float pitch, bool loop, const std::string &busName = "Master", int priority = 128);

    /// @brief Pause or resume a specific voice at the stream level
    void SetVoicePaused(SDL_AudioStream *stream, bool paused);

    /// Source position of the next frame to mix, in seconds. Hardware output
    /// can lag this sample clock by the device buffering latency.
    [[nodiscard]] double GetVoiceTime(SDL_AudioStream *stream) const;
    /// Discard queued old samples and seek atomically with the audio callback.
    void SetVoiceTime(SDL_AudioStream *stream, double seconds);

    /// @brief Query whether a voice has reached the end of its clip
    [[nodiscard]] bool HasVoiceFinished(SDL_AudioStream *stream) const;
    [[nodiscard]] bool IsVoiceVirtual(SDL_AudioStream *stream) const;

    // ========================================================================
    // Editor asset preview
    // ========================================================================

    /// Play one non-spatial clip through the shared audio device.
    bool PlayPreview(const std::string &filePath);
    void StopPreview();
    [[nodiscard]] bool IsPreviewPlaying(const std::string &filePath = {}) const;
    [[nodiscard]] std::string GetPreviewPath() const;

    // ========================================================================
    // Source registration (for spatial audio)
    // ========================================================================

    /// @brief Register an AudioSource for per-frame spatial updates
    void RegisterSource(AudioSource *source);

    /// @brief Unregister an AudioSource (called on disable/destroy)
    void UnregisterSource(AudioSource *source);

    // ========================================================================
    // Listener management
    // ========================================================================

    /// @brief Register an AudioListener so it can become active or standby
    void RegisterListener(AudioListener *listener);

    /// @brief Unregister an AudioListener and promote a standby listener if needed
    void UnregisterListener(AudioListener *listener);

    /// @brief Register an AudioListener as active
    void SetActiveListener(AudioListener *listener);

    /// @brief Get the currently active listener
    [[nodiscard]] AudioListener *GetActiveListener() const
    {
        return m_activeListener;
    }

    /// Stable world-wide listener identity used by diagnostics and tooling.
    [[nodiscard]] uint64_t GetActiveListenerGameObjectId() const;

    // ========================================================================
    // Global audio settings
    // ========================================================================

    /// @brief Set master volume (0.0 = silence, 1.0 = full)
    void SetMasterVolume(float volume);

    /// @brief Get master volume
    [[nodiscard]] float GetMasterVolume() const
    {
        return GetBusVolume("Master");
    }

    /// @brief Configure one of Master/Music/SFX/Ambience/UI.
    void SetBusVolume(const std::string &busName, float volume);
    [[nodiscard]] float GetBusVolume(const std::string &busName) const;
    /// Move one bus to a target volume over audio-device time without rebuilding or
    /// seeking any playing voice. A direct SetBusVolume cancels this envelope.
    void FadeBusVolume(const std::string &busName, float volume, float durationSeconds);
    void CancelBusFade(const std::string &busName);
    [[nodiscard]] bool IsBusFading(const std::string &busName) const;
    void SetBusMuted(const std::string &busName, bool muted);
    [[nodiscard]] bool GetBusMuted(const std::string &busName) const;
    [[nodiscard]] float GetBusGain(const std::string &busName) const;
    [[nodiscard]] static bool IsValidBusName(const std::string &busName);

    /// Mixed output time, not the hardware DAC playhead. Pausing the device
    /// freezes this clock; it advances without game frames or playing voices.
    [[nodiscard]] double GetOutputTime() const
    {
        return m_outputTime.load(std::memory_order_relaxed);
    }
    /// Peak magnitude of the latest mixed device block (after SDL's clamp).
    [[nodiscard]] float GetOutputPeak() const
    {
        return m_outputPeak.load(std::memory_order_relaxed);
    }
    /// Output samples at full scale since device initialization. Indicates
    /// saturation, not the magnitude lost to SDL's earlier mixing clamp.
    [[nodiscard]] uint64_t GetSaturatedSampleCount() const
    {
        return m_saturatedSamples.load(std::memory_order_relaxed);
    }

    /// @brief Pause all audio playback
    void PauseAll();

    /// @brief Resume all audio playback
    void ResumeAll();

    /// @brief Whether all audio is paused
    [[nodiscard]] bool IsPaused() const
    {
        return m_globalPaused;
    }

    /// Logical playback streams, including paused and virtual voices.
    [[nodiscard]] size_t GetActiveVoiceCount() const;
    /// Physical mixing work is bounded; excess/inaudible voices retain only
    /// logical playback on the device clock. Configured by the engine owner;
    /// this is not a cap on logical voices or their total memory usage.
    void SetMaxRealVoices(size_t count);
    [[nodiscard]] size_t GetMaxRealVoices() const
    {
        return m_maxRealVoices;
    }
    [[nodiscard]] size_t GetRealVoiceCount() const
    {
        return m_realVoiceCount;
    }
    [[nodiscard]] size_t GetVirtualVoiceCount() const;

    /// @brief Get the output sample rate
    [[nodiscard]] int GetSampleRate() const
    {
        return m_deviceSpec.freq;
    }

    /// @brief Get the output channel count
    [[nodiscard]] int GetChannelCount() const
    {
        return m_deviceSpec.channels;
    }

    ~AudioEngine();

  private:
    struct AudioVoiceState;

    AudioEngine() = default;

    AudioEngine(const AudioEngine &) = delete;
    AudioEngine &operator=(const AudioEngine &) = delete;

    static void SDLCALL FeedVoiceStream(void *userdata, SDL_AudioStream *stream, int additional_amount,
                                        int total_amount);
    static void SDLCALL ProcessOutput(void *userdata, const SDL_AudioSpec *spec, float *buffer, int length);

    std::shared_ptr<AudioVoiceState> GetVoiceState(SDL_AudioStream *stream) const;
    double VoiceCursorAt(const AudioVoiceState &state, double time) const;
    void CommitVirtualCursor(AudioVoiceState &state, double time);
    void SetVoiceReal(AudioVoiceState &state, bool real);
    void RebalanceVoices();
    float VoiceAudibility(const AudioVoiceState &state) const;
    bool VoiceFinished(const AudioVoiceState &state) const;
    AudioListener *FindBestListenerLocked(AudioListener *exclude = nullptr) const;

    bool m_initialized = false;
    bool m_globalPaused = false;
    // Authored automation belongs to the owner thread; only complete snapshots
    // cross into the device callback. Master is included once in each bus gain,
    // before SDL sums/clamps voices, preserving output headroom.
    audio_mixer::BusEnvelopes m_busEnvelopes{};
    audio_mixer::BusEnvelopeMailbox m_busMailbox;
    std::array<std::atomic<float>, 5> m_audioBusGains{1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::atomic<double> m_outputTime{0.0};
    std::atomic<float> m_outputPeak{0.0f};
    std::atomic<uint64_t> m_saturatedSamples{0};

    SDL_AudioDeviceID m_deviceId = 0;
    SDL_AudioSpec m_deviceSpec = {};

    AudioListener *m_activeListener = nullptr;

    /// Logical voices, owned by the engine thread. Device callbacks only read
    /// their stable userdata; mutations of scheduling state stay on the owner.
    mutable std::mutex m_streamsMutex;
    std::unordered_map<SDL_AudioStream *, std::shared_ptr<AudioVoiceState>> m_voiceStates;
    std::vector<AudioVoiceState *> m_voiceCandidates;
    size_t m_maxRealVoices = 64;
    size_t m_realVoiceCount = 0;
    uint64_t m_nextVoiceOrder = 1;

    mutable std::mutex m_previewMutex;
    std::shared_ptr<AudioClip> m_previewClip;
    SDL_AudioStream *m_previewStream = nullptr;
    std::string m_previewPath;

    /// Registered AudioSources for spatial updates
    std::unordered_set<AudioSource *> m_registeredSources;
    mutable std::mutex m_sourcesMutex;

    /// Registered AudioListeners. Only one is active; the rest are standby listeners.
    std::unordered_set<AudioListener *> m_registeredListeners;
    mutable std::mutex m_listenersMutex;
};

} // namespace infernux
