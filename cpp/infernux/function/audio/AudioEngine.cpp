#include "AudioEngine.h"

#include "AudioClip.h"
#include "AudioListener.h"
#include "AudioMixer.h"
#include "AudioSource.h"

#include <core/log/InxLog.h>
#include <function/scene/GameObject.h>
#include <function/scene/Transform.h>

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <glm/glm.hpp>
#include <limits>
#include <stdexcept>

namespace infernux
{
namespace
{

int AudioBusIndex(const std::string &busName)
{
    if (busName == "Master")
        return -1;
    if (busName == "Music")
        return 0;
    if (busName == "SFX")
        return 1;
    if (busName == "Ambience")
        return 2;
    if (busName == "UI")
        return 3;
    throw std::invalid_argument("Unknown audio bus '" + busName + "'");
}

size_t AudioBusSlot(const std::string &busName)
{
    return static_cast<size_t>(AudioBusIndex(busName) + 1);
}

} // namespace

struct AudioEngine::AudioVoiceState
{
    AudioEngine *engine = nullptr;
    std::atomic<size_t> busSlot{0};
    std::shared_ptr<const AudioPlaybackPcm> pcm;
    double cursor = 0.0;
    float currentGain = 0.0f;
    float currentSpatialGain = 1.0f;
    float currentPan = 0.0f;
    float currentSpatialBlend = 0.0f;
    std::atomic<float> gain = 1.0f;
    std::atomic<float> spatialGain = 1.0f;
    std::atomic<float> pan = 0.0f;
    std::atomic<float> spatialBlend = 1.0f;
    std::atomic<float> pitch = 1.0f;
    std::atomic<bool> loop = false;
    std::atomic<bool> destroyed = false;
    std::atomic<bool> finished = false;
    bool paused = true; // Owner-thread binding state; callback never writes it.
    SDL_AudioStream *stream = nullptr;
    bool bound = false;
    bool virtualized = false;
    bool selected = false;
    double virtualSince = 0.0;
    uint64_t order = 0;
    int priority = 128; // Lower values are more important.
    float audibility = 1.0f;
};

AudioEngine &AudioEngine::Instance()
{
    // Intentionally leaked — shut down explicitly in Infernux::Cleanup().
    static AudioEngine *instance = new AudioEngine();
    return *instance;
}

AudioEngine::~AudioEngine()
{
    Shutdown();
}

bool AudioEngine::Initialize()
{
    if (m_initialized) {
        INXLOG_WARN("AudioEngine already initialized");
        return true;
    }

    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        INXLOG_ERROR("Failed to initialize SDL audio subsystem: ", SDL_GetError());
        return false;
    }

    SDL_AudioSpec requestedSpec = {};
    requestedSpec.format = SDL_AUDIO_F32;
    requestedSpec.channels = 2;
    requestedSpec.freq = 44100;

    m_deviceId = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &requestedSpec);
    if (m_deviceId == 0) {
        // A Player can legitimately run on a machine without an output device
        // (CI, a server, Remote Desktop, or a temporarily disconnected headset).
        // Keep the runtime silent and retryable instead of classifying this
        // device-compatibility boundary as a fatal engine error.
        INXLOG_WARN("No audio output device is available; continuing silently: ", SDL_GetError());
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        return false;
    }

    SDL_AudioSpec actualSpec = {};
    int sampleFrames = 0;
    if (SDL_GetAudioDeviceFormat(m_deviceId, &actualSpec, &sampleFrames)) {
        m_deviceSpec = actualSpec;
    } else {
        m_deviceSpec = requestedSpec;
        INXLOG_WARN("Could not query device format, using requested spec");
    }

    m_busMailbox.Publish(m_busEnvelopes);
    if (!SDL_SetAudioPostmixCallback(m_deviceId, &AudioEngine::ProcessOutput, this) ||
        !SDL_ResumeAudioDevice(m_deviceId)) {
        INXLOG_ERROR("Failed to resume audio device: ", SDL_GetError());
        SDL_CloseAudioDevice(m_deviceId);
        m_deviceId = 0;
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        return false;
    }

    m_initialized = true;

    // Browser audio devices cannot be opened until a trusted user gesture.
    // Sources may therefore have completed Start() before the device exists.
    // Replay only the play-on-awake request that could not create a voice;
    // already-playing sources and inactive components remain untouched.
    std::vector<AudioSource *> deferredSources;
    {
        std::lock_guard<std::mutex> lock(m_sourcesMutex);
        deferredSources.assign(m_registeredSources.begin(), m_registeredSources.end());
    }
    for (AudioSource *source : deferredSources) {
        if (!source || !source->GetPlayOnAwake() || source->IsPlaying() || !source->IsEnabled())
            continue;
        GameObject *owner = source->GetGameObject();
        if (owner && owner->IsActiveInHierarchy())
            source->Play(0);
    }

    return true;
}

void AudioEngine::Shutdown()
{
    if (!m_initialized) {
        return;
    }

    StopPreview();

    std::vector<AudioSource *> sources;
    {
        std::lock_guard<std::mutex> lock(m_sourcesMutex);
        sources.assign(m_registeredSources.begin(), m_registeredSources.end());
        m_registeredSources.clear();
    }
    for (AudioSource *source : sources) {
        if (source) {
            source->NotifyAudioEngineShutdown();
        }
    }

    decltype(m_voiceStates) retiringVoices;
    {
        std::lock_guard<std::mutex> lock(m_streamsMutex);
        // Callback userdata must stay alive until SDL has stopped every
        // stream callback, including voices not owned by an AudioSource.
        retiringVoices.swap(m_voiceStates);
    }

    for (const auto &[stream, state] : retiringVoices) {
        if (!stream) {
            continue;
        }
        SDL_LockAudioStream(stream);
        SDL_SetAudioStreamGetCallback(stream, nullptr, nullptr);
        SDL_UnlockAudioStream(stream);
        SDL_UnbindAudioStream(stream);
        SDL_DestroyAudioStream(stream);
    }

    if (m_deviceId != 0) {
        SDL_CloseAudioDevice(m_deviceId);
        m_deviceId = 0;
    }

    SDL_QuitSubSystem(SDL_INIT_AUDIO);

    {
        std::lock_guard<std::mutex> lock(m_listenersMutex);
        m_registeredListeners.clear();
    }

    m_activeListener = nullptr;
    m_globalPaused = false;
    m_initialized = false;
    m_realVoiceCount = 0;
    m_voiceCandidates.clear();
    m_nextVoiceOrder = 1;
    // Freeze author values at shutdown. A fresh device starts a new clock;
    // no fade from the previous engine session survives into that timeline.
    const double stoppedAt = GetOutputTime();
    for (auto &bus : m_busEnvelopes) {
        bus.start = bus.target = bus.VolumeAt(stoppedAt);
        bus.startedAt = bus.duration = 0.0;
    }
    m_outputTime.store(0.0, std::memory_order_relaxed);
    m_outputPeak.store(0.0f, std::memory_order_relaxed);
    m_saturatedSamples.store(0, std::memory_order_relaxed);
}

void SDLCALL AudioEngine::ProcessOutput(void *userdata, const SDL_AudioSpec *spec, float *buffer, int length)
{
    auto &engine = *static_cast<AudioEngine *>(userdata);
    const auto &buses = engine.m_busMailbox.Consume();
    const int frames = length / (static_cast<int>(sizeof(float)) * spec->channels);
    const double begin = engine.GetOutputTime();
    const double step = 1.0 / spec->freq;
    const double end = begin + frames * step;
    // Sub-bus targets become visible to the next voice mixing block. Voice
    // ramps smooth this device-block update without depending on game Update.
    const float masterGain = buses[0].muted ? 0.0f : buses[0].VolumeAt(end);
    engine.m_audioBusGains[0].store(masterGain, std::memory_order_relaxed);
    for (size_t slot = 1; slot < buses.size(); ++slot)
        engine.m_audioBusGains[slot].store(buses[slot].muted ? 0.0f : masterGain * buses[slot].VolumeAt(end),
                                           std::memory_order_relaxed);
    float peak = 0.0f;
    uint64_t saturated = 0;
    for (int sample = 0; sample < frames * spec->channels; ++sample) {
        const float magnitude = std::abs(buffer[sample]);
        peak = std::max(peak, magnitude);
        saturated += magnitude >= 1.0f;
    }
    engine.m_outputPeak.store(peak, std::memory_order_relaxed);
    engine.m_saturatedSamples.fetch_add(saturated, std::memory_order_relaxed);
    engine.m_outputTime.store(end, std::memory_order_relaxed);
}

void SDLCALL AudioEngine::FeedVoiceStream(void *userdata, SDL_AudioStream *stream, int additional_amount,
                                          int total_amount)
{
    (void)total_amount;

    auto *voice = static_cast<AudioVoiceState *>(userdata);
    if (!voice || additional_amount <= 0) {
        return;
    }

    const int bytesPerFrame = static_cast<int>(sizeof(float) * 2);
    int remainingFrames = std::max(1, additional_amount / bytesPerFrame);
    if (voice->destroyed.load(std::memory_order_acquire) || !voice->pcm || voice->pcm->stereoFrames.empty() ||
        voice->pcm->frameCount == 0) {
        return;
    }

    const float targetGain =
        voice->gain.load(std::memory_order_relaxed) *
        voice->engine->m_audioBusGains[voice->busSlot.load(std::memory_order_relaxed)].load(std::memory_order_relaxed);
    const float targetSpatialGain = voice->spatialGain.load(std::memory_order_relaxed);
    const float targetPan = voice->pan.load(std::memory_order_relaxed);
    const float targetSpatialBlend = voice->spatialBlend.load(std::memory_order_relaxed);
    const double step = std::max(0.01, static_cast<double>(voice->pitch.load(std::memory_order_relaxed)));
    const bool loop = voice->loop.load(std::memory_order_relaxed);
    const float smoothingStep = 1.0f / static_cast<float>(std::max(1, voice->pcm->sampleRate / 200));

    bool finished = false;
    const auto &pcmFrames = voice->pcm->stereoFrames;
    const size_t frameCount = voice->pcm->frameCount;
    std::array<float, 2048> output{};
    while (remainingFrames > 0 && !finished) {
        const int chunkFrames = std::min(remainingFrames, static_cast<int>(output.size() / 2));
        int writtenFrames = 0;
        for (; writtenFrames < chunkFrames; ++writtenFrames) {
            if (!loop && voice->cursor >= static_cast<double>(frameCount)) {
                finished = true;
                break;
            }

            double frameCursor = voice->cursor;
            while (loop && frameCursor >= static_cast<double>(frameCount)) {
                frameCursor -= static_cast<double>(frameCount);
                voice->cursor = frameCursor;
            }

            const size_t index0 = std::min(static_cast<size_t>(frameCursor), frameCount - 1);
            const size_t index1 = loop ? (index0 + 1) % frameCount : std::min(index0 + 1, frameCount - 1);
            const float fraction = static_cast<float>(frameCursor - static_cast<double>(index0));
            const float left = pcmFrames[index0 * 2] + (pcmFrames[index1 * 2] - pcmFrames[index0 * 2]) * fraction;
            const float right =
                pcmFrames[index0 * 2 + 1] + (pcmFrames[index1 * 2 + 1] - pcmFrames[index0 * 2 + 1]) * fraction;
            voice->currentGain = audio_mixer::ApproachParameter(voice->currentGain, targetGain, smoothingStep);
            voice->currentSpatialGain =
                audio_mixer::ApproachParameter(voice->currentSpatialGain, targetSpatialGain, smoothingStep);
            voice->currentPan = audio_mixer::ApproachParameter(voice->currentPan, targetPan, smoothingStep);
            voice->currentSpatialBlend =
                audio_mixer::ApproachParameter(voice->currentSpatialBlend, targetSpatialBlend, smoothingStep);
            const auto mixed = audio_mixer::MixStereoFrame(left, right, voice->currentGain, voice->currentSpatialGain,
                                                           voice->currentPan, voice->currentSpatialBlend);
            output[static_cast<size_t>(writtenFrames) * 2] = mixed.left;
            output[static_cast<size_t>(writtenFrames) * 2 + 1] = mixed.right;
            voice->cursor += step;
        }
        if (writtenFrames > 0 && !SDL_PutAudioStreamData(stream, output.data(), writtenFrames * bytesPerFrame)) {
            INXLOG_WARN("AudioEngine: failed to push streamed audio data: ", SDL_GetError());
            break;
        }
        remainingFrames -= writtenFrames;
    }
    if (finished)
        SDL_FlushAudioStream(stream);
    voice->finished.store(finished, std::memory_order_release);
}

std::shared_ptr<AudioEngine::AudioVoiceState> AudioEngine::GetVoiceState(SDL_AudioStream *stream) const
{
    if (!stream) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(m_streamsMutex);
    auto it = m_voiceStates.find(stream);
    return it != m_voiceStates.end() ? it->second : nullptr;
}

AudioListener *AudioEngine::FindBestListenerLocked(AudioListener *exclude) const
{
    AudioListener *best = nullptr;
    uint64_t bestId = (std::numeric_limits<uint64_t>::max)();
    for (AudioListener *candidate : m_registeredListeners) {
        if (!candidate || candidate == exclude || candidate->IsDestroyed() || !candidate->IsEnabled()) {
            continue;
        }

        auto *gameObject = candidate->GetGameObject();
        if (!gameObject || !gameObject->IsActiveInHierarchy()) {
            continue;
        }

        const uint64_t candidateId = candidate->GetGameObjectId();
        if (candidateId != 0 && candidateId < bestId) {
            best = candidate;
            bestId = candidateId;
        }
    }
    return best;
}

void AudioEngine::Update(float deltaTime)
{
    (void)deltaTime;
    if (!m_initialized) {
        return;
    }

    SDL_AudioStream *finishedPreview = nullptr;
    {
        std::lock_guard<std::mutex> previewLock(m_previewMutex);
        if (m_previewStream && HasVoiceFinished(m_previewStream))
            finishedPreview = m_previewStream;
    }
    if (finishedPreview)
        StopPreview();

    glm::vec3 listenerPos(0.0f);
    glm::vec3 listenerRight(1.0f, 0.0f, 0.0f);
    bool hasListener = false;

    if (m_activeListener) {
        auto *listenerGO = m_activeListener->GetGameObject();
        if (listenerGO) {
            auto *listenerTr = listenerGO->GetTransform();
            if (listenerTr) {
                listenerPos = listenerTr->GetWorldPosition();
                listenerRight = listenerTr->GetRight();
                hasListener = true;
            }
        }
    }

    std::lock_guard<std::mutex> lock(m_sourcesMutex);
    for (auto *source : m_registeredSources) {
        if (!source) {
            continue;
        }

        if (!source->HasActiveVoices()) {
            continue;
        }

        if (hasListener) {
            auto *sourceGO = source->GetGameObject();
            auto *sourceTr = sourceGO ? sourceGO->GetTransform() : nullptr;
            if (sourceTr) {
                const glm::vec3 sourcePos = sourceTr->GetWorldPosition();
                const float distance = glm::length(sourcePos - listenerPos);
                const float spatialGain =
                    audio_mixer::DistanceAttenuation(distance, source->GetMinDistance(), source->GetMaxDistance());
                const glm::vec3 toSource = distance > 0.001f ? (sourcePos - listenerPos) / distance : glm::vec3(0.0f);
                const float pan = glm::dot(toSource, listenerRight);
                source->SetComputedSpatialGain(spatialGain);
                source->SetComputedPan(pan);
            }
        } else {
            source->SetComputedSpatialGain(1.0f);
            source->SetComputedPan(0.0f);
        }

        source->ApplyAllTrackGains();
    }
    RebalanceVoices();
}

SDL_AudioStream *AudioEngine::CreateVoice(AudioSource * /*source*/, AudioClip *clip, double startSeconds)
{
    if (!m_initialized || !clip || !clip->IsLoaded()) {
        return nullptr;
    }

    SDL_AudioSpec playbackSpec = {};
    playbackSpec.format = SDL_AUDIO_F32;
    playbackSpec.channels = 2;
    playbackSpec.freq = m_deviceSpec.freq > 0 ? m_deviceSpec.freq : 44100;

    auto pcm = clip->AcquirePlaybackPcm(playbackSpec.freq);
    if (!pcm || pcm->frameCount == 0) {
        INXLOG_ERROR("Failed to acquire prepared audio clip for playback");
        return nullptr;
    }

    SDL_AudioStream *stream = SDL_CreateAudioStream(&playbackSpec, &m_deviceSpec);
    if (!stream) {
        INXLOG_ERROR("Failed to create audio stream: ", SDL_GetError());
        return nullptr;
    }

    auto voiceState = std::make_shared<AudioVoiceState>();
    voiceState->engine = this;
    voiceState->stream = stream;
    voiceState->order = m_nextVoiceOrder++;
    voiceState->pcm = std::move(pcm);
    voiceState->cursor =
        std::clamp(startSeconds * voiceState->pcm->sampleRate, 0.0, static_cast<double>(voiceState->pcm->frameCount));

    if (!SDL_SetAudioStreamGetCallback(stream, &AudioEngine::FeedVoiceStream, voiceState.get())) {
        INXLOG_ERROR("Failed to register audio stream callback: ", SDL_GetError());
        SDL_DestroyAudioStream(stream);
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(m_streamsMutex);
        m_voiceStates.emplace(stream, std::move(voiceState));
        if (m_voiceStates.size() > m_voiceCandidates.capacity())
            m_voiceCandidates.reserve(std::max(m_voiceStates.size(), m_voiceCandidates.capacity() * 2));
    }

    return stream;
}

void AudioEngine::DestroyVoice(SDL_AudioStream *stream)
{
    if (!stream) {
        return;
    }

    std::shared_ptr<AudioVoiceState> state;
    {
        std::lock_guard<std::mutex> lock(m_streamsMutex);
        auto it = m_voiceStates.find(stream);
        if (it != m_voiceStates.end()) {
            state = it->second;
            m_voiceStates.erase(it);
        }
    }

    if (state) {
        state->destroyed.store(true, std::memory_order_release);
        if (state->bound)
            --m_realVoiceCount;
    }

    SDL_LockAudioStream(stream);
    SDL_SetAudioStreamGetCallback(stream, nullptr, nullptr);
    SDL_UnlockAudioStream(stream);
    SDL_UnbindAudioStream(stream);
    SDL_DestroyAudioStream(stream);
}

void AudioEngine::UpdateVoiceMix(SDL_AudioStream *stream, float gain, float spatialGain, float pan, float spatialBlend,
                                 float pitch, bool loop, const std::string &busName, int priority)
{
    auto state = GetVoiceState(stream);
    if (!state) {
        return;
    }

    if (state->virtualized &&
        (pitch != state->pitch.load(std::memory_order_relaxed) || loop != state->loop.load(std::memory_order_relaxed)))
        CommitVirtualCursor(*state, GetOutputTime());
    state->priority = priority;
    state->busSlot.store(AudioBusSlot(busName), std::memory_order_relaxed);
    state->gain.store(std::max(0.0f, gain), std::memory_order_relaxed);
    state->spatialGain.store(std::clamp(spatialGain, 0.0f, 1.0f), std::memory_order_relaxed);
    state->pan.store(std::clamp(pan, -1.0f, 1.0f), std::memory_order_relaxed);
    state->spatialBlend.store(std::clamp(spatialBlend, 0.0f, 1.0f), std::memory_order_relaxed);
    state->pitch.store(std::clamp(pitch, 0.1f, 3.0f), std::memory_order_relaxed);
    state->loop.store(loop, std::memory_order_relaxed);
    if (loop && state->finished.load(std::memory_order_acquire))
        state->finished.store(false, std::memory_order_release);
}

void AudioEngine::SetVoicePaused(SDL_AudioStream *stream, bool paused)
{
    auto state = GetVoiceState(stream);
    if (!state || state->paused == paused) {
        return;
    }

    if (paused) {
        if (state->virtualized)
            CommitVirtualCursor(*state, GetOutputTime());
        if (state->bound) {
            // A manual pause preserves the queued tail, unlike virtualization.
            SDL_UnbindAudioStream(stream);
            state->bound = false;
            --m_realVoiceCount;
        }
        state->paused = true;
    } else {
        state->virtualSince = GetOutputTime();
        state->paused = false;
        SetVoiceReal(*state, m_realVoiceCount < m_maxRealVoices && VoiceAudibility(*state) > 0.0f);
    }
}

double AudioEngine::GetVoiceTime(SDL_AudioStream *stream) const
{
    auto state = GetVoiceState(stream);
    if (!state)
        return 0.0;
    SDL_LockAudioStream(stream);
    const double frames = VoiceCursorAt(*state, GetOutputTime());
    const double seconds = frames / state->pcm->sampleRate;
    SDL_UnlockAudioStream(stream);
    return seconds;
}

void AudioEngine::SetVoiceTime(SDL_AudioStream *stream, double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0)
        throw std::invalid_argument("Audio time must be finite and non-negative");
    auto state = GetVoiceState(stream);
    if (!state)
        return;
    SDL_LockAudioStream(stream);
    SDL_ClearAudioStream(stream);
    state->cursor = std::min(seconds * state->pcm->sampleRate, static_cast<double>(state->pcm->frameCount));
    state->virtualSince = GetOutputTime();
    state->finished.store(false, std::memory_order_release);
    state->currentGain = 0.0f; // Reuse the normal gain ramp after a discontinuity.
    SDL_UnlockAudioStream(stream);
}

bool AudioEngine::HasVoiceFinished(SDL_AudioStream *stream) const
{
    auto state = GetVoiceState(stream);
    return !state || VoiceFinished(*state);
}

bool AudioEngine::VoiceFinished(const AudioVoiceState &state) const
{
    if (state.virtualized)
        return !state.loop.load(std::memory_order_relaxed) &&
               VoiceCursorAt(state, GetOutputTime()) >= static_cast<double>(state.pcm->frameCount);

    // Producing the last sample does not mean SDL has consumed the tail yet.
    // Keep the voice alive until all flushed output has reached the device.
    return state.finished.load(std::memory_order_acquire) && SDL_GetAudioStreamAvailable(state.stream) == 0;
}

double AudioEngine::VoiceCursorAt(const AudioVoiceState &state, double time) const
{
    double cursor = state.cursor;
    if (state.virtualized && !state.paused)
        cursor += std::max(0.0, time - state.virtualSince) * state.pcm->sampleRate *
                  state.pitch.load(std::memory_order_relaxed);
    const double count = static_cast<double>(state.pcm->frameCount);
    return state.loop.load(std::memory_order_relaxed) ? std::fmod(cursor, count) : std::min(cursor, count);
}

void AudioEngine::CommitVirtualCursor(AudioVoiceState &state, double time)
{
    state.cursor = VoiceCursorAt(state, time);
    state.virtualSince = time;
}

float AudioEngine::VoiceAudibility(const AudioVoiceState &state) const
{
    const float blend = state.spatialBlend.load(std::memory_order_relaxed);
    return state.gain.load(std::memory_order_relaxed) *
           (1.0f - blend + blend * state.spatialGain.load(std::memory_order_relaxed)) *
           m_audioBusGains[state.busSlot.load(std::memory_order_relaxed)].load(std::memory_order_relaxed);
}

void AudioEngine::SetVoiceReal(AudioVoiceState &state, bool real)
{
    if (real == state.bound && (real || state.virtualized))
        return;
    if (!real) {
        if (state.bound) {
            SDL_UnbindAudioStream(state.stream);
            state.bound = false;
            --m_realVoiceCount;
        }
        SDL_LockAudioStream(state.stream);
        SDL_ClearAudioStream(state.stream);
        state.virtualSince = GetOutputTime();
        state.virtualized = true;
        SDL_UnlockAudioStream(state.stream);
        return;
    }
    if (state.virtualized) {
        CommitVirtualCursor(state, GetOutputTime());
        state.currentGain = 0.0f;
    }
    if (!SDL_BindAudioStream(m_deviceId, state.stream))
        throw std::runtime_error(std::string("Failed to resume audio voice: ") + SDL_GetError());
    state.virtualized = false;
    state.bound = true;
    ++m_realVoiceCount;
}

void AudioEngine::RebalanceVoices()
{
    m_voiceCandidates.clear();
    for (const auto &[stream, owned] : m_voiceStates) {
        auto &state = *owned;
        state.selected = false;
        if (state.paused || VoiceFinished(state))
            continue;
        state.audibility = VoiceAudibility(state);
        if (state.audibility > 0.0f)
            m_voiceCandidates.push_back(&state);
    }
    const size_t count = std::min(m_maxRealVoices, m_voiceCandidates.size());
    std::partial_sort(m_voiceCandidates.begin(), m_voiceCandidates.begin() + count, m_voiceCandidates.end(),
                      [](const AudioVoiceState *a, const AudioVoiceState *b) {
                          if (a->priority != b->priority)
                              return a->priority < b->priority;
                          if (a->audibility != b->audibility)
                              return a->audibility > b->audibility;
                          return a->order < b->order;
                      });
    for (size_t i = 0; i < count; ++i)
        m_voiceCandidates[i]->selected = true;
    // Demote before promoting so the physical budget is never exceeded.
    for (const auto &[stream, state] : m_voiceStates)
        if (!state->paused && !state->selected)
            SetVoiceReal(*state, false);
    for (size_t i = 0; i < count; ++i)
        SetVoiceReal(*m_voiceCandidates[i], true);
}

void AudioEngine::SetMaxRealVoices(size_t count)
{
    if (count == 0)
        throw std::invalid_argument("Audio real voice budget must be positive");
    m_maxRealVoices = count;
    RebalanceVoices();
}

bool AudioEngine::IsVoiceVirtual(SDL_AudioStream *stream) const
{
    const auto state = GetVoiceState(stream);
    return state && !state->paused && state->virtualized && !VoiceFinished(*state);
}

size_t AudioEngine::GetVirtualVoiceCount() const
{
    size_t count = 0;
    for (const auto &[stream, state] : m_voiceStates)
        count += !state->paused && state->virtualized && !VoiceFinished(*state);
    return count;
}

bool AudioEngine::PlayPreview(const std::string &filePath)
{
    if (!m_initialized || filePath.empty())
        return false;

    StopPreview();
    auto clip = std::make_shared<AudioClip>();
    if (!clip->LoadFromFile(filePath))
        return false;
    SDL_AudioStream *stream = CreateVoice(nullptr, clip.get());
    if (!stream)
        return false;
    UpdateVoiceMix(stream, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, false);
    SetVoicePaused(stream, false);

    std::lock_guard<std::mutex> lock(m_previewMutex);
    m_previewClip = std::move(clip);
    m_previewStream = stream;
    m_previewPath = filePath;
    return true;
}

void AudioEngine::StopPreview()
{
    SDL_AudioStream *stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_previewMutex);
        stream = m_previewStream;
        m_previewStream = nullptr;
        m_previewClip.reset();
        m_previewPath.clear();
    }
    if (stream)
        DestroyVoice(stream);
}

bool AudioEngine::IsPreviewPlaying(const std::string &filePath) const
{
    SDL_AudioStream *stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_previewMutex);
        if (!filePath.empty() && filePath != m_previewPath)
            return false;
        stream = m_previewStream;
    }
    return stream && !HasVoiceFinished(stream);
}

std::string AudioEngine::GetPreviewPath() const
{
    std::lock_guard<std::mutex> lock(m_previewMutex);
    return m_previewPath;
}

void AudioEngine::RegisterSource(AudioSource *source)
{
    if (!source) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_sourcesMutex);
    m_registeredSources.insert(source);
}

void AudioEngine::UnregisterSource(AudioSource *source)
{
    if (!source) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_sourcesMutex);
    m_registeredSources.erase(source);
}

void AudioEngine::RegisterListener(AudioListener *listener)
{
    if (!listener) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_listenersMutex);
    const bool inserted = m_registeredListeners.insert(listener).second;

    if (!m_activeListener || m_activeListener == listener || m_activeListener->IsDestroyed() ||
        !m_activeListener->IsEnabled() || !m_activeListener->GetGameObject() ||
        !m_activeListener->GetGameObject()->IsActiveInHierarchy()) {
        m_activeListener = listener;
        return;
    }

    if (!inserted) {
        return;
    }

    const char *activeName =
        m_activeListener->GetGameObject() ? m_activeListener->GetGameObject()->GetName().c_str() : "<unknown>";
    const char *standbyName = listener->GetGameObject() ? listener->GetGameObject()->GetName().c_str() : "<unknown>";
    INXLOG_INFO("AudioListener: only one listener can be active. Keeping '", activeName, "' active; listener '",
                standbyName, "' is on standby.");
}

void AudioEngine::UnregisterListener(AudioListener *listener)
{
    if (!listener) {
        return;
    }

    AudioListener *promoted = nullptr;
    bool lostActiveListener = false;
    {
        std::lock_guard<std::mutex> lock(m_listenersMutex);
        m_registeredListeners.erase(listener);

        if (m_activeListener == listener) {
            lostActiveListener = true;
            promoted = FindBestListenerLocked(listener);
            m_activeListener = promoted;
        }
    }

    if (promoted) {
        const char *promotedName =
            promoted->GetGameObject() ? promoted->GetGameObject()->GetName().c_str() : "<unknown>";
        INXLOG_INFO("AudioListener: active listener changed. Promoting standby listener '", promotedName, "'.");
    } else if (lostActiveListener) {
        INXLOG_INFO("AudioListener: there is currently no active listener in the scene.");
    }
}

void AudioEngine::SetActiveListener(AudioListener *listener)
{
    if (!listener) {
        std::lock_guard<std::mutex> lock(m_listenersMutex);
        m_activeListener = FindBestListenerLocked();
        return;
    }

    std::lock_guard<std::mutex> lock(m_listenersMutex);
    m_registeredListeners.insert(listener);
    m_activeListener = listener;
}

uint64_t AudioEngine::GetActiveListenerGameObjectId() const
{
    std::lock_guard<std::mutex> lock(m_listenersMutex);
    return m_activeListener ? m_activeListener->GetGameObjectId() : 0;
}

void AudioEngine::SetMasterVolume(float volume)
{
    SetBusVolume("Master", volume);
}

bool AudioEngine::IsValidBusName(const std::string &busName)
{
    return busName == "Master" || busName == "Music" || busName == "SFX" || busName == "Ambience" || busName == "UI";
}

void AudioEngine::SetBusVolume(const std::string &busName, float volume)
{
    if (!std::isfinite(volume))
        throw std::invalid_argument("Audio bus volume must be finite");
    const size_t slot = AudioBusSlot(busName);
    auto &bus = m_busEnvelopes[slot];
    bus.start = bus.target = std::clamp(volume, 0.0f, 1.0f);
    bus.duration = 0.0;
    m_busMailbox.Publish(m_busEnvelopes);
}

float AudioEngine::GetBusVolume(const std::string &busName) const
{
    return m_busEnvelopes[AudioBusSlot(busName)].VolumeAt(GetOutputTime());
}

void AudioEngine::FadeBusVolume(const std::string &busName, float volume, float durationSeconds)
{
    if (!std::isfinite(volume))
        throw std::invalid_argument("Audio bus volume must be finite");
    if (!std::isfinite(durationSeconds) || durationSeconds <= 0.0f)
        throw std::invalid_argument("Audio bus fade duration must be positive and finite");
    const size_t slot = AudioBusSlot(busName);
    auto &fade = m_busEnvelopes[slot];
    const double now = GetOutputTime();
    fade.start = fade.VolumeAt(now);
    fade.target = std::clamp(volume, 0.0f, 1.0f);
    fade.duration = durationSeconds;
    fade.startedAt = now;
    m_busMailbox.Publish(m_busEnvelopes);
}

void AudioEngine::CancelBusFade(const std::string &busName)
{
    SetBusVolume(busName, GetBusVolume(busName));
}

bool AudioEngine::IsBusFading(const std::string &busName) const
{
    return m_busEnvelopes[AudioBusSlot(busName)].IsFading(GetOutputTime());
}

void AudioEngine::SetBusMuted(const std::string &busName, bool muted)
{
    m_busEnvelopes[AudioBusSlot(busName)].muted = muted;
    m_busMailbox.Publish(m_busEnvelopes);
}

bool AudioEngine::GetBusMuted(const std::string &busName) const
{
    return m_busEnvelopes[AudioBusSlot(busName)].muted;
}

float AudioEngine::GetBusGain(const std::string &busName) const
{
    const int index = AudioBusIndex(busName);
    if (index < 0)
        return 1.0f;
    return GetBusMuted(busName) ? 0.0f : GetBusVolume(busName);
}

void AudioEngine::PauseAll()
{
    if (m_deviceId != 0) {
        SDL_PauseAudioDevice(m_deviceId);
        m_globalPaused = true;
    }
}

void AudioEngine::ResumeAll()
{
    if (m_deviceId != 0) {
        SDL_ResumeAudioDevice(m_deviceId);
        m_globalPaused = false;
    }
}

size_t AudioEngine::GetActiveVoiceCount() const
{
    std::lock_guard<std::mutex> lock(m_streamsMutex);
    return m_voiceStates.size();
}

} // namespace infernux
