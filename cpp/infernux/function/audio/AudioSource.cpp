#include "AudioSource.h"
#include "AudioEngine.h"
#include <core/log/InxLog.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/scene/ComponentDocumentValidation.h>
#include <function/scene/ComponentFactory.h>
#include <function/scene/GameObject.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infernux
{

namespace
{
SemanticTypeDescriptor DescribeAudioSource()
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux.AudioSource";
    type.readableId = "infernux.component.audio-source";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "Audio Source";
    type.runtimeProfiles = {"editor", "player", "headless"};
    const auto add = [&](const char *name, const char *kind, json initial) -> json & {
        type.fields.push_back({std::string("AudioSource.") + name,
                               std::string("FieldType.") + kind,
                               false,
                               {{"field_id", name},
                                {"serialized_name", name},
                                {"serialized", true},
                                {"hidden", false},
                                {"nullable", false},
                                {"storage_kind", "native_property"},
                                {"display_name_key", std::string("audio_source.") + name},
                                {"tooltip", std::string("audio_source.tooltip.") + name},
                                {"default", std::move(initial)}}});
        return type.fields.back().attributes;
    };

    auto &trackCount = add("track_count", "INT", 1);
    trackCount["range"] = {1, 16};
    // The setter resizes the coupled tracks document atomically. A generic
    // preflight must not patch track_count while leaving the old array shape.
    trackCount["setter_owns_document_shape"] = true;
    add("volume", "FLOAT", 1.0)["range"] = {0.0, 1.0};
    add("priority", "INT", 128)["range"] = {0, 255};
    add("pitch", "FLOAT", 1.0)["range"] = {0.1, 3.0};
    add("mute", "BOOL", false);
    add("loop", "BOOL", false);
    add("play_on_awake", "BOOL", true);
    add("spatial_blend", "FLOAT", 0.0)["range"] = {0.0, 1.0};
    type.fields.back().attributes["header"] = "audio_source.section.spatial";
    add("min_distance", "FLOAT", 1.0)["range"] = {0.001, 500.0};
    add("max_distance", "FLOAT", 500.0)["range"] = {0.001, 10000.0};
    add("output_bus", "STRING", "Master")["header"] = "audio_source.section.routing";
    auto &pool = add("one_shot_pool_size", "INT", 8);
    pool["hidden"] = true;
    return type;
}

const bool registeredAudioSource = ComponentFactory::Register(
    "AudioSource", [] { return std::make_unique<AudioSource>(); }, AudioSource::ValidateSerializedDocument,
    AudioSource::GetTypeConstraints(), DescribeAudioSource);
} // namespace

AudioSource::AudioSource()
{
    // Default: 1 track
    m_tracks.resize(1);
    m_oneShotVoices.resize(static_cast<size_t>(m_oneShotPoolSize));
}

AudioSource::~AudioSource()
{
    StopAll();
    AssetDependencyGraph::Instance().ClearRuntimeDependenciesOf(GetInstanceGuid());
    AudioEngine::Instance().UnregisterSource(this);
}

void AudioSource::Awake()
{
    // Register with AudioEngine for spatial updates
    AudioEngine::Instance().RegisterSource(this);
}

void AudioSource::Start()
{
    const auto clip = m_tracks.empty() ? nullptr : m_tracks[0].GetClip();
    // The Web host cannot open its audio device until a trusted user gesture.
    // Awake() has already registered this source, so AudioEngine::Initialize()
    // owns the one authoritative deferred play-on-awake transition.
    if (AudioEngine::Instance().IsInitialized() && m_playOnAwake && !IsPlaying() && clip && clip->IsLoaded()) {
        Play(0);
    }
}

void AudioSource::OnEnable()
{
    AudioEngine::Instance().RegisterSource(this);

    for (int i = 0; i < static_cast<int>(m_tracks.size()); ++i) {
        auto &track = m_tracks[i];
        if (track.stream && track.isPlaying && track.isPaused && track.pauseRequestedByDisable) {
            track.pauseRequestedByDisable = false;
            track.isPaused = false;
            AudioEngine::Instance().SetVoicePaused(track.stream, false);
            ApplyTrackGain(i);
        }
    }

    for (int i = 0; i < static_cast<int>(m_oneShotVoices.size()); ++i) {
        auto &voice = m_oneShotVoices[i];
        if (voice.stream && voice.isPaused && voice.pauseRequestedByDisable) {
            voice.pauseRequestedByDisable = false;
            voice.isPaused = false;
            AudioEngine::Instance().SetVoicePaused(voice.stream, false);
            ApplyOneShotGain(i);
        }
    }
}

void AudioSource::OnDisable()
{
    // Pause all playing tracks when component is disabled
    for (int i = 0; i < static_cast<int>(m_tracks.size()); ++i) {
        if (m_tracks[i].isPlaying && !m_tracks[i].isPaused) {
            Pause(i);
            m_tracks[i].pauseRequestedByDisable = true;
        }
    }

    for (int i = 0; i < static_cast<int>(m_oneShotVoices.size()); ++i) {
        auto &voice = m_oneShotVoices[i];
        if (voice.stream && !voice.isPaused) {
            voice.pauseRequestedByDisable = true;
            voice.isPaused = true;
            AudioEngine::Instance().SetVoicePaused(voice.stream, true);
        }
    }

    AudioEngine::Instance().UnregisterSource(this);
}

void AudioSource::OnDestroy()
{
    StopAll();
    AudioEngine::Instance().UnregisterSource(this);
}

void AudioSource::Update(float /*deltaTime*/)
{
    for (int i = 0; i < static_cast<int>(m_tracks.size()); ++i) {
        auto &track = m_tracks[i];
        if (!track.isPlaying || track.isPaused || !track.stream) {
            continue;
        }

        if (!m_loop && AudioEngine::Instance().HasVoiceFinished(track.stream)) {
            StopVoice(i);
        }
    }

    for (int i = 0; i < static_cast<int>(m_oneShotVoices.size()); ++i) {
        auto &voice = m_oneShotVoices[i];
        if (!voice.stream || voice.isPaused) {
            continue;
        }

        if (AudioEngine::Instance().HasVoiceFinished(voice.stream)) {
            StopOneShotVoice(i);
        }
    }
}

// ============================================================================
// Serialization
// ============================================================================

nlohmann::json AudioSource::SerializeDocument() const
{
    json j = Component::SerializeDocument();
    j["volume"] = m_volume;
    j["priority"] = m_priority;
    j["pitch"] = m_pitch;
    j["loop"] = m_loop;
    j["play_on_awake"] = m_playOnAwake;
    j["mute"] = m_mute;
    j["spatial_blend"] = m_spatialBlend;
    j["min_distance"] = m_minDistance;
    j["max_distance"] = m_maxDistance;
    j["one_shot_pool_size"] = m_oneShotPoolSize;
    j["output_bus"] = m_outputBus;
    j["track_count"] = static_cast<int>(m_tracks.size());

    // Serialize per-track data
    json tracksJson = json::array();
    for (const auto &track : m_tracks) {
        json tj;
        tj["volume"] = track.volume;
        if (track.clipAsset.HasGuid())
            tj["clip_guid"] = track.clipAsset.GetGuid();
        tracksJson.push_back(tj);
    }
    j["tracks"] = tracksJson;

    return j;
}

void AudioSource::ValidateSerializedDocument(const nlohmann::json &j)
{
    using namespace component_document_validation;
    ValidateComponentDocument(j, "AudioSource",
                              {"volume", "pitch", "loop", "play_on_awake", "mute", "min_distance", "max_distance",
                               "one_shot_pool_size", "output_bus", "track_count", "tracks"},
                              {"spatial_blend", "priority"});
    if (j.contains("priority")) {
        const int priority = RequireInteger(j, "priority", "AudioSource");
        if (priority < 0 || priority > 255)
            throw std::invalid_argument("AudioSource.priority must be in [0, 255]");
    }
    const float volume = RequireFiniteFloat(j, "volume", "AudioSource");
    const float pitch = RequireFiniteFloat(j, "pitch", "AudioSource");
    RequireBoolean(j, "loop", "AudioSource");
    RequireBoolean(j, "play_on_awake", "AudioSource");
    RequireBoolean(j, "mute", "AudioSource");
    const float spatialBlend =
        j.contains("spatial_blend") ? RequireFiniteFloat(j, "spatial_blend", "AudioSource") : 1.0f;
    const float minDistance = RequireFiniteFloat(j, "min_distance", "AudioSource");
    const float maxDistance = RequireFiniteFloat(j, "max_distance", "AudioSource");
    const int poolSize = RequireInteger(j, "one_shot_pool_size", "AudioSource");
    const std::string &outputBus = RequireString(j, "output_bus", "AudioSource");
    const int trackCount = RequireInteger(j, "track_count", "AudioSource");
    const auto &tracks = j["tracks"];
    if (!tracks.is_array() || trackCount < 1 || trackCount > 16 || tracks.size() != static_cast<size_t>(trackCount))
        throw std::invalid_argument("AudioSource.tracks must match track_count in [1, 16]");
    if (volume < 0.0f || volume > 1.0f || pitch < 0.1f || pitch > 3.0f || spatialBlend < 0.0f || spatialBlend > 1.0f)
        throw std::invalid_argument("AudioSource volume, pitch, or spatial_blend is out of range");
    if (minDistance <= 0.0f || maxDistance < minDistance || poolSize < 1)
        throw std::invalid_argument("AudioSource distance or pool settings are invalid");
    if (!AudioEngine::IsValidBusName(outputBus))
        throw std::invalid_argument("AudioSource.output_bus is not a supported bus");

    for (size_t index = 0; index < tracks.size(); ++index) {
        const auto &track = tracks[index];
        if (!track.is_object())
            throw std::invalid_argument("AudioSource.tracks[" + std::to_string(index) + "] must be an object");
        for (const auto &[key, value] : track.items()) {
            (void)value;
            if (key != "volume" && key != "clip_guid")
                throw std::invalid_argument("AudioSource.tracks[" + std::to_string(index) +
                                            "] contains unknown field '" + key + "'");
        }
        if (!track.contains("volume") || !track["volume"].is_number())
            throw std::invalid_argument("AudioSource track volume is required and must be numeric");
        const double trackVolume = track["volume"].get<double>();
        if (!std::isfinite(trackVolume) || trackVolume < 0.0 || trackVolume > 1.0)
            throw std::invalid_argument("AudioSource track volume is out of range");
        if (track.contains("clip_guid")) {
            if (!track["clip_guid"].is_string() || track["clip_guid"].get_ref<const std::string &>().empty())
                throw std::invalid_argument("AudioSource track clip_guid must be a non-empty string");
        }
    }
}

bool AudioSource::DeserializeDocument(const nlohmann::json &j)
{
    try {
        ValidateSerializedDocument(j);

        auto &registry = AssetRegistry::Instance();
        const auto &tracksJson = j["tracks"];
        std::vector<std::string> stagedGuids(tracksJson.size());
        std::vector<std::shared_ptr<AudioClip>> stagedClips(tracksJson.size());
        for (size_t index = 0; index < tracksJson.size(); ++index) {
            if (!tracksJson[index].contains("clip_guid"))
                continue;
            stagedGuids[index] = tracksJson[index]["clip_guid"].get<std::string>();
            stagedClips[index] = registry.LoadAsset<AudioClip>(stagedGuids[index], ResourceType::Audio);
        }

        if (!Component::DeserializeDocument(j))
            return false;

        m_volume = j["volume"].get<float>();
        m_priority = j.value("priority", 128);
        m_pitch = j["pitch"].get<float>();
        m_loop = j["loop"].get<bool>();
        m_playOnAwake = j["play_on_awake"].get<bool>();
        m_mute = j["mute"].get<bool>();
        m_spatialBlend = j.value("spatial_blend", 1.0f);
        m_minDistance = j["min_distance"].get<float>();
        m_maxDistance = j["max_distance"].get<float>();
        m_oneShotPoolSize = j["one_shot_pool_size"].get<int>();
        SetOutputBus(j["output_bus"].get<std::string>());

        // Deserialize tracks
        const int trackCount = j["track_count"].get<int>();
        for (int i = 0; i < static_cast<int>(m_tracks.size()); ++i)
            AssignTrackClipReference(i, {}, nullptr);
        SetTrackCount(trackCount);

        for (int i = 0; i < trackCount; ++i) {
            const auto &tj = tracksJson[i];
            m_tracks[i].volume = tj["volume"].get<float>();
            AssignTrackClipReference(i, stagedGuids[i], std::move(stagedClips[i]));
        }

        SetVolume(m_volume);
        SetPitch(m_pitch);
        SetSpatialBlend(m_spatialBlend);
        SetMinDistance(m_minDistance);
        SetMaxDistance(m_maxDistance);
        SetOneShotPoolSize(m_oneShotPoolSize);
        for (int i = 0; i < static_cast<int>(m_tracks.size()); ++i) {
            SetTrackVolume(i, m_tracks[i].volume);
        }

        return true;
    } catch (const std::exception &e) {
        INXLOG_WARN("AudioSource::Deserialize failed: ", e.what());
        return false;
    }
}

// ============================================================================
// Track management
// ============================================================================

void AudioSource::SetTrackCount(int count)
{
    const int clamped = std::clamp(count, 1, 16);
    if (count != clamped) {
        INXLOG_WARN("AudioSource::SetTrackCount: track_count must be in [1, 16]. Clamping ", count, " to ", clamped,
                    ".");
        count = clamped;
    }
    int oldCount = static_cast<int>(m_tracks.size());

    // Stop voices for tracks that are being removed
    for (int i = count; i < oldCount; ++i) {
        StopVoice(i);
        if (m_tracks[i].clipAsset.HasGuid())
            AssetDependencyGraph::Instance().RemoveRuntimeDependency(GetInstanceGuid(),
                                                                     m_tracks[i].clipAsset.GetGuid());
    }

    m_tracks.resize(count);
}

void AudioSource::SetTrackClip(int trackIndex, std::shared_ptr<AudioClip> clip)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        INXLOG_WARN("AudioSource::SetTrackClip: track index ", trackIndex, " out of range");
        return;
    }

    const std::string guid = clip ? clip->GetGuid() : std::string{};
    AssignTrackClipReference(trackIndex, guid, std::move(clip));
}

std::shared_ptr<AudioClip> AudioSource::GetTrackClip(int trackIndex) const
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return nullptr;
    }
    return m_tracks[trackIndex].GetClip();
}

void AudioSource::SetTrackClipGuid(int trackIndex, const std::string &guid)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        INXLOG_WARN("AudioSource::SetTrackClipGuid: track index ", trackIndex, " out of range");
        return;
    }

    std::shared_ptr<AudioClip> clip;
    if (!guid.empty())
        clip = AssetRegistry::Instance().LoadAsset<AudioClip>(guid, ResourceType::Audio);
    AssignTrackClipReference(trackIndex, guid, std::move(clip));
}

std::string AudioSource::GetTrackClipGuid(int trackIndex) const
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size()))
        return {};
    return m_tracks[trackIndex].clipAsset.GetGuid();
}

void AudioSource::AssignTrackClipReference(int trackIndex, const std::string &guid, std::shared_ptr<AudioClip> clip)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size()))
        return;

    auto &track = m_tracks[trackIndex];
    if (track.isPlaying)
        StopVoice(trackIndex);
    track.startTime = 0.0;

    auto &graph = AssetDependencyGraph::Instance();
    const std::string oldGuid = track.clipAsset.GetGuid();
    if (!oldGuid.empty())
        graph.RemoveRuntimeDependency(GetInstanceGuid(), oldGuid);

    track.clipAsset.Clear();
    track.transientClip.reset();
    if (guid.empty()) {
        track.transientClip = std::move(clip);
        return;
    }

    track.clipAsset.SetGuid(guid);
    if (clip) {
        const uint64_t version = AssetRegistry::Instance().GetAssetVersion(guid);
        if (version != 0)
            track.clipAsset.SetCached(std::move(clip), version);
        else
            track.transientClip = std::move(clip);
    }
    graph.AddRuntimeDependency(GetInstanceGuid(), guid);
}

void AudioSource::OnAudioClipAssetEvent(const std::string &guid, AssetEvent event)
{
    for (int index = 0; index < static_cast<int>(m_tracks.size()); ++index) {
        auto &track = m_tracks[index];
        if (track.clipAsset.GetGuid() != guid)
            continue;
        if (track.isPlaying)
            StopVoice(index);
        track.clipAsset.Invalidate();
        track.transientClip.reset();
        if (event == AssetEvent::Modified)
            AssetRegistry::Instance().Resolve(track.clipAsset, ResourceType::Audio);
    }
}

void AudioSource::SetTrackVolume(int trackIndex, float volume)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return;
    }
    const float clamped = std::clamp(volume, 0.0f, 1.0f);
    if (clamped != volume) {
        INXLOG_DEBUG("AudioSource::SetTrackVolume: volume must be in [0, 1]. Clamping ", volume, " to ", clamped, ".");
    }
    m_tracks[trackIndex].volume = clamped;
    ApplyTrackGain(trackIndex);
}

float AudioSource::GetTrackVolume(int trackIndex) const
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return 0.0f;
    }
    return m_tracks[trackIndex].volume;
}

// ============================================================================
// Playback control
// ============================================================================

void AudioSource::Play(int trackIndex)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        INXLOG_WARN("AudioSource::Play: track index ", trackIndex, " out of range");
        return;
    }

    auto &track = m_tracks[trackIndex];
    const auto clip = track.GetClip();
    if (!clip || !clip->IsLoaded()) {
        INXLOG_WARN("AudioSource::Play: no clip loaded on track ", trackIndex);
        return;
    }

    const double startTime = track.isPlaying ? 0.0 : track.startTime;
    // Stop any existing playback on this track
    StopVoice(trackIndex);
    track.startTime = startTime;

    // Start new voice
    StartVoice(trackIndex);
}

void AudioSource::Stop(int trackIndex)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return;
    }
    StopVoice(trackIndex);
}

void AudioSource::Pause(int trackIndex)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return;
    }
    auto &track = m_tracks[trackIndex];
    if (track.isPlaying && !track.isPaused && track.stream) {
        track.isPaused = true;
        track.pauseRequestedByDisable = false;
        AudioEngine::Instance().SetVoicePaused(track.stream, true);
    }
}

void AudioSource::UnPause(int trackIndex)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return;
    }
    auto &track = m_tracks[trackIndex];
    if (track.isPlaying && track.isPaused && track.stream) {
        AudioEngine::Instance().SetVoicePaused(track.stream, false);
        track.pauseRequestedByDisable = false;
        track.isPaused = false;
        ApplyTrackGain(trackIndex);
    }
}

void AudioSource::StopAll()
{
    for (int i = 0; i < static_cast<int>(m_tracks.size()); ++i) {
        StopVoice(i);
    }
    StopOneShots();
}

double AudioSource::GetTrackTime(int trackIndex) const
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size()))
        throw std::out_of_range("Audio track index out of range");
    const auto &track = m_tracks[trackIndex];
    return track.stream ? AudioEngine::Instance().GetVoiceTime(track.stream) : track.startTime;
}

void AudioSource::SetTrackTime(int trackIndex, double seconds)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size()))
        throw std::out_of_range("Audio track index out of range");
    if (!std::isfinite(seconds) || seconds < 0.0)
        throw std::invalid_argument("Audio time must be finite and non-negative");
    auto &track = m_tracks[trackIndex];
    const auto clip = track.GetClip();
    if (!clip || !clip->IsLoaded())
        throw std::runtime_error("Cannot seek an audio track without a loaded clip");
    const double duration = static_cast<double>(clip->GetSampleCount()) / clip->GetSampleRate();
    if (seconds > duration)
        throw std::out_of_range("Audio time exceeds clip duration");
    track.startTime = seconds;
    if (track.stream)
        AudioEngine::Instance().SetVoiceTime(track.stream, seconds);
}

void AudioSource::PlayOneShot(std::shared_ptr<AudioClip> clip, float volumeScale)
{
    if (!clip || !clip->IsLoaded()) {
        INXLOG_WARN("AudioSource::PlayOneShot: clip is null or not loaded.");
        return;
    }

    const float clampedVolumeScale = std::clamp(volumeScale, 0.0f, 1.0f);
    if (clampedVolumeScale != volumeScale) {
        INXLOG_DEBUG("AudioSource::PlayOneShot: volumeScale must be in [0, 1]. Clamping ", volumeScale, " to ",
                     clampedVolumeScale, ".");
    }

    int selectedVoice = -1;
    uint64_t oldestPlayOrder = std::numeric_limits<uint64_t>::max();
    float quietest = std::numeric_limits<float>::infinity();
    bool available = false;
    for (int i = 0; i < static_cast<int>(m_oneShotVoices.size()); ++i) {
        const auto &voice = m_oneShotVoices[i];
        if (!voice.stream || AudioEngine::Instance().HasVoiceFinished(voice.stream)) {
            selectedVoice = i;
            available = true;
            break;
        }

        if (voice.volumeScale < quietest || (voice.volumeScale == quietest && voice.playOrder < oldestPlayOrder)) {
            quietest = voice.volumeScale;
            oldestPlayOrder = voice.playOrder;
            selectedVoice = i;
        }
    }

    // The pool always has at least one slot. Capacity pressure is normal:
    // reject a quieter incoming shot, otherwise replace the quietest (oldest
    // for ties). Across sources, physical scheduling also respects priority.
    if (!available && clampedVolumeScale < quietest) {
        ++m_rejectedOneShotCount;
        return;
    }

    if (m_oneShotVoices[selectedVoice].stream) {
        StopOneShotVoice(selectedVoice);
    }

    StartOneShotVoice(selectedVoice, std::move(clip), clampedVolumeScale);
}

void AudioSource::StopOneShots()
{
    for (int i = 0; i < static_cast<int>(m_oneShotVoices.size()); ++i) {
        StopOneShotVoice(i);
    }
}

bool AudioSource::IsTrackPlaying(int trackIndex) const
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return false;
    }
    return m_tracks[trackIndex].isPlaying && !m_tracks[trackIndex].isPaused;
}

bool AudioSource::IsTrackPaused(int trackIndex) const
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return false;
    }
    return m_tracks[trackIndex].isPaused;
}

bool AudioSource::IsTrackVirtual(int trackIndex) const
{
    return trackIndex >= 0 && trackIndex < static_cast<int>(m_tracks.size()) && m_tracks[trackIndex].stream &&
           AudioEngine::Instance().IsVoiceVirtual(m_tracks[trackIndex].stream);
}

// ============================================================================
// Source-level properties
// ============================================================================

void AudioSource::SetVolume(float volume)
{
    const float clamped = std::clamp(volume, 0.0f, 1.0f);
    if (clamped != volume) {
        INXLOG_DEBUG("AudioSource::SetVolume: volume must be in [0, 1]. Clamping ", volume, " to ", clamped, ".");
    }
    m_volume = clamped;
    ApplyAllTrackGains();
}

void AudioSource::SetPitch(float pitch)
{
    const float clamped = std::clamp(pitch, 0.1f, 3.0f);
    if (clamped != pitch) {
        INXLOG_DEBUG("AudioSource::SetPitch: pitch must be in [0.1, 3.0]. Clamping ", pitch, " to ", clamped, ".");
    }
    m_pitch = clamped;
    ApplyAllTrackGains();
}

void AudioSource::SetMute(bool mute)
{
    m_mute = mute;
    ApplyAllTrackGains();
}

void AudioSource::SetPriority(int priority)
{
    if (priority < 0 || priority > 255)
        throw std::invalid_argument("AudioSource.priority must be in [0, 255]");
    m_priority = priority;
    ApplyAllTrackGains();
}

void AudioSource::SetSpatialBlend(float blend)
{
    m_spatialBlend = std::clamp(blend, 0.0f, 1.0f);
    ApplyAllTrackGains();
}

void AudioSource::SetOutputBus(const std::string &busName)
{
    if (!AudioEngine::IsValidBusName(busName))
        throw std::invalid_argument("Unknown AudioSource output bus '" + busName + "'");
    m_outputBus = busName;
    ApplyAllTrackGains();
}

void AudioSource::SetMinDistance(float dist)
{
    const float clamped = std::max(0.001f, dist);
    if (clamped != dist) {
        INXLOG_DEBUG("AudioSource::SetMinDistance: min_distance must be > 0. Clamping ", dist, " to ", clamped, ".");
    }
    m_minDistance = clamped;
    if (m_maxDistance < m_minDistance) {
        INXLOG_DEBUG(
            "AudioSource::SetMinDistance: max_distance was smaller than min_distance. Raising max_distance to ",
            m_minDistance, ".");
        m_maxDistance = m_minDistance;
    }
    ApplyAllTrackGains();
}

void AudioSource::SetMaxDistance(float dist)
{
    const float clamped = std::max(dist, m_minDistance);
    if (clamped != dist) {
        INXLOG_DEBUG("AudioSource::SetMaxDistance: max_distance must be >= min_distance. Clamping ", dist, " to ",
                     clamped, ".");
    }
    m_maxDistance = clamped;
    ApplyAllTrackGains();
}

void AudioSource::SetOneShotPoolSize(int size)
{
    if (size < 1) {
        INXLOG_DEBUG("AudioSource::SetOneShotPoolSize: pool size must be >= 1. Clamping ", size, " to 1.");
        size = 1;
    }

    const int oldSize = static_cast<int>(m_oneShotVoices.size());
    for (int i = size; i < oldSize; ++i) {
        StopOneShotVoice(i);
    }

    m_oneShotPoolSize = size;
    m_oneShotVoices.resize(static_cast<size_t>(m_oneShotPoolSize));
}

uint64_t AudioSource::GetGameObjectId() const
{
    auto *go = GetGameObject();
    return go ? go->GetID() : 0;
}

// ============================================================================
// Spatial audio helpers
// ============================================================================

std::vector<SDL_AudioStream *> AudioSource::GetActiveStreams() const
{
    std::vector<SDL_AudioStream *> streams;
    for (const auto &track : m_tracks) {
        if (track.stream && track.isPlaying) {
            streams.push_back(track.stream);
        }
    }
    for (const auto &voice : m_oneShotVoices) {
        if (voice.stream) {
            streams.push_back(voice.stream);
        }
    }
    return streams;
}

void AudioSource::ApplyAllTrackGains()
{
    for (int i = 0; i < static_cast<int>(m_tracks.size()); ++i) {
        ApplyTrackGain(i);
    }
    for (int i = 0; i < static_cast<int>(m_oneShotVoices.size()); ++i) {
        ApplyOneShotGain(i);
    }
}

bool AudioSource::HasActiveVoices() const
{
    return std::any_of(m_tracks.begin(), m_tracks.end(), [](const AudioTrack &track) { return track.stream; }) ||
           std::any_of(m_oneShotVoices.begin(), m_oneShotVoices.end(),
                       [](const AudioOneShotVoice &voice) { return voice.stream; });
}

void AudioSource::ApplyTrackGain(int trackIndex)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return;
    }
    auto &track = m_tracks[trackIndex];
    if (!track.stream || !track.isPlaying) {
        return;
    }
    if (track.isPaused) {
        AudioEngine::Instance().UpdateVoiceMix(track.stream, 0.0f, m_spatialGain, m_pan, m_spatialBlend, m_pitch,
                                               m_loop, m_outputBus, m_priority);
        return;
    }

    const float gain = m_mute ? 0.0f : (m_volume * track.volume);
    AudioEngine::Instance().UpdateVoiceMix(track.stream, gain, m_spatialGain, m_pan, m_spatialBlend, m_pitch, m_loop,
                                           m_outputBus, m_priority);
}

void AudioSource::ApplyOneShotGain(int voiceIndex)
{
    if (voiceIndex < 0 || voiceIndex >= static_cast<int>(m_oneShotVoices.size())) {
        return;
    }

    auto &voice = m_oneShotVoices[voiceIndex];
    if (!voice.stream) {
        return;
    }

    if (voice.isPaused) {
        AudioEngine::Instance().UpdateVoiceMix(voice.stream, 0.0f, m_spatialGain, m_pan, m_spatialBlend, m_pitch, false,
                                               m_outputBus, m_priority);
        return;
    }

    const float gain = m_mute ? 0.0f : (m_volume * voice.volumeScale);
    AudioEngine::Instance().UpdateVoiceMix(voice.stream, gain, m_spatialGain, m_pan, m_spatialBlend, m_pitch, false,
                                           m_outputBus, m_priority);
}

// ============================================================================
// Internal voice management
// ============================================================================

void AudioSource::StartVoice(int trackIndex)
{
    auto &engine = AudioEngine::Instance();
    if (!engine.IsInitialized()) {
        INXLOG_WARN("AudioSource::StartVoice: AudioEngine not initialized");
        return;
    }

    auto &track = m_tracks[trackIndex];
    const auto clip = track.GetClip();
    track.stream = engine.CreateVoice(this, clip.get(), track.startTime);
    if (!track.stream) {
        INXLOG_ERROR("AudioSource::StartVoice: failed to create voice for track ", trackIndex);
        return;
    }

    track.isPlaying = true;
    track.isPaused = false;
    track.pauseRequestedByDisable = false;
    ApplyTrackGain(trackIndex);
    AudioEngine::Instance().SetVoicePaused(track.stream, false);
}

void AudioSource::StopVoice(int trackIndex)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(m_tracks.size())) {
        return;
    }
    auto &track = m_tracks[trackIndex];
    if (track.stream) {
        if (AudioEngine::Instance().IsInitialized()) {
            AudioEngine::Instance().DestroyVoice(track.stream);
        }
        track.stream = nullptr;
    }
    track.isPlaying = false;
    track.isPaused = false;
    track.pauseRequestedByDisable = false;
    track.startTime = 0.0;
}

void AudioSource::StartOneShotVoice(int voiceIndex, std::shared_ptr<AudioClip> clip, float volumeScale)
{
    auto &engine = AudioEngine::Instance();
    if (!engine.IsInitialized()) {
        INXLOG_WARN("AudioSource::StartOneShotVoice: AudioEngine not initialized");
        return;
    }

    if (voiceIndex < 0 || voiceIndex >= static_cast<int>(m_oneShotVoices.size())) {
        return;
    }

    auto &voice = m_oneShotVoices[voiceIndex];
    voice.clip = std::move(clip);
    voice.volumeScale = volumeScale;
    voice.playOrder = m_nextOneShotPlayOrder++;
    voice.stream = engine.CreateVoice(this, voice.clip.get());
    if (!voice.stream) {
        INXLOG_ERROR("AudioSource::StartOneShotVoice: failed to create pooled voice ", voiceIndex);
        voice.clip.reset();
        voice.volumeScale = 1.0f;
        voice.playOrder = 0;
        return;
    }

    voice.isPaused = false;
    voice.pauseRequestedByDisable = false;
    ApplyOneShotGain(voiceIndex);
    engine.SetVoicePaused(voice.stream, false);
}

void AudioSource::StopOneShotVoice(int voiceIndex)
{
    if (voiceIndex < 0 || voiceIndex >= static_cast<int>(m_oneShotVoices.size())) {
        return;
    }

    auto &voice = m_oneShotVoices[voiceIndex];
    if (voice.stream) {
        if (AudioEngine::Instance().IsInitialized()) {
            AudioEngine::Instance().DestroyVoice(voice.stream);
        }
        voice.stream = nullptr;
    }

    voice.clip.reset();
    voice.isPaused = false;
    voice.pauseRequestedByDisable = false;
    voice.volumeScale = 1.0f;
    voice.playOrder = 0;
}

void AudioSource::CheckLooping(int trackIndex)
{
    (void)trackIndex;
}

void AudioSource::NotifyAudioEngineShutdown()
{
    for (auto &track : m_tracks) {
        track.stream = nullptr;
        track.isPlaying = false;
        track.isPaused = false;
        track.pauseRequestedByDisable = false;
        track.startTime = 0.0;
    }

    for (auto &voice : m_oneShotVoices) {
        voice.stream = nullptr;
        voice.clip.reset();
        voice.isPaused = false;
        voice.pauseRequestedByDisable = false;
        voice.volumeScale = 1.0f;
        voice.playOrder = 0;
    }
}

} // namespace infernux
