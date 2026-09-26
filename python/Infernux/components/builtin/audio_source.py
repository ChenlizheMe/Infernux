"""
AudioSource — Python BuiltinComponent wrapper for C++ AudioSource.

Infernux's AudioSource supports **multi-track** playback: a single
AudioSource can hold N tracks, each with its own AudioClip and per-track
volume.  All tracks share the source-level settings (volume, mute, pitch,
3D spatial attenuation). In addition, each AudioSource owns a pooled
one-shot voice bank for transient SFX, so you keep the multi-track advantage
while still getting a Unity-style convenience path.

Example::

    from Infernux.components.builtin import AudioSource, AudioListener
    from Infernux.core.audio_clip import AudioClip

    class MusicPlayer(InxComponent):
        def start(self):
            source = self.game_object.get_component(AudioSource)
            source.track_count = 2

            bgm = AudioClip.load("Assets/Audio/bgm.wav")
            sfx = AudioClip.load("Assets/Audio/sfx.wav")

            source.set_track_clip(0, bgm)
            source.set_track_clip(1, sfx)
            source.set_track_volume(1, 0.5)

            source.play(0)  # play BGM
            source.play(1)  # play SFX simultaneously
            source.play_one_shot(sfx, 0.8)  # pooled transient playback
"""

from __future__ import annotations

from typing import Optional

from Infernux.components.builtin_component import BuiltinComponent, CppProperty
class AudioSource(BuiltinComponent):
    """Python wrapper for the C++ AudioSource component.

    Properties delegate to the C++ ``AudioSource`` via CppProperty.
    Playback methods delegate directly to the C++ component.

    Multi-track:
        - ``track_count`` — number of tracks (default 1)
        - ``set_track_clip(i, clip)`` / ``get_track_clip(i)``
        - ``set_track_volume(i, vol)`` / ``get_track_volume(i)``
        - ``play(i)`` / ``stop(i)`` / ``pause(i)`` / ``un_pause(i)``
        - ``stop_all()`` — stop all tracks
        - ``play_one_shot(clip, volume_scale)`` — pooled transient playback
    """

    _cpp_type_name = "AudioSource"
    _component_category_ = "Audio"

    # ---- Track Count ----
    track_count = CppProperty.from_native("AudioSource", "track_count")

    # ---- Volume / Pitch / Mute (source-level) ----
    volume = CppProperty.from_native("AudioSource", "volume")
    pitch = CppProperty.from_native("AudioSource", "pitch")
    mute = CppProperty.from_native("AudioSource", "mute")

    # ---- Loop / PlayOnAwake ----
    loop = CppProperty.from_native("AudioSource", "loop")
    play_on_awake = CppProperty.from_native("AudioSource", "play_on_awake")

    # ---- 2D / 3D Spatial ----
    spatial_blend = CppProperty.from_native("AudioSource", "spatial_blend")
    min_distance = CppProperty.from_native("AudioSource", "min_distance")
    max_distance = CppProperty.from_native("AudioSource", "max_distance")
    output_bus = CppProperty.from_native("AudioSource", "output_bus")

    # ------------------------------------------------------------------
    # Properties (not shown in inspector, accessible via script)
    # ------------------------------------------------------------------

    one_shot_pool_size = CppProperty.from_native("AudioSource", "one_shot_pool_size")
    priority = CppProperty.from_native("AudioSource", "priority")

    # ------------------------------------------------------------------
    # Track management (delegate methods)
    # ------------------------------------------------------------------

    def set_track_clip(self, track_index: int, clip) -> None:
        """Assign an AudioClip to a specific track.

        Args:
            track_index: 0-based track index.
            clip: An ``AudioClip`` (Python wrapper) or C++ native clip.
        """
        cpp = self._cpp_component
        if cpp is None:
            return
        # Accept both Python AudioClip wrapper and raw C++ clip
        native_clip = getattr(clip, "native", clip) if clip is not None else None
        cpp.set_track_clip(track_index, native_clip)

    def get_track_clip(self, track_index: int):
        """Get the AudioClip on a specific track."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_track_clip(track_index)
        return None

    def get_track_clip_guid(self, track_index: int) -> str:
        """Get the GUID of the AudioClip on a specific track."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_track_clip_guid(track_index)
        return ""

    def set_track_clip_by_guid(self, track_index: int, guid: str) -> None:
        """Set the AudioClip on a track by asset GUID.

        Args:
            track_index: 0-based track index.
            guid: Asset GUID of the AudioClip. Empty string clears the clip.
        """
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_track_clip_by_guid(track_index, guid)

    def set_track_volume(self, track_index: int, volume: float) -> None:
        """Set per-track volume (0.0–1.0)."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_track_volume(track_index, volume)

    def get_track_volume(self, track_index: int) -> float:
        """Get per-track volume."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_track_volume(track_index)
        return 0.0

    # ------------------------------------------------------------------
    # Playback control (delegate methods)
    # ------------------------------------------------------------------

    def play(self, track_index: int = 0) -> None:
        """Start playing a track (default: track 0)."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.play(track_index)

    def stop(self, track_index: int = 0) -> None:
        """Stop a track (default: track 0)."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.stop(track_index)

    def play_one_shot(self, clip, volume_scale: float = 1.0) -> None:
        """Play a transient clip using the source's pooled one-shot voices."""
        cpp = self._cpp_component
        if cpp is not None:
            native_clip = getattr(clip, "native", clip) if clip is not None else None
            cpp.play_one_shot(native_clip, volume_scale)

    def stop_one_shots(self) -> None:
        """Stop all pooled one-shot voices."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.stop_one_shots()

    def pause(self, track_index: int = 0) -> None:
        """Pause a track (default: track 0)."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.pause(track_index)

    def un_pause(self, track_index: int = 0) -> None:
        """Resume a paused track (default: track 0)."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.un_pause(track_index)

    def stop_all(self) -> None:
        """Stop all tracks and all pooled one-shot voices."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.stop_all()

    def get_track_time(self, track_index: int = 0) -> float:
        """Clip seconds of the next mixed sample, independent of game frames.

        The hardware can lag this position by its output buffering latency.
        Individual/global pause freezes the sample cursor.
        """
        cpp = self._cpp_component
        return cpp.get_track_time(track_index) if cpp is not None else 0.0

    def set_track_time(self, track_index: int, seconds: float) -> None:
        """Seek without changing whether the track is playing or paused.

        A stopped track starts here on its next ``play``; ``stop`` resets it.
        Position is runtime state and is not saved into Scene or Prefab assets.
        A loaded clip and a finite time in [0, clip duration] are required.
        """
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_track_time(track_index, seconds)

    def is_track_playing(self, track_index: int) -> bool:
        """Whether a specific track is currently playing."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.is_track_playing(track_index)
        return False

    def is_track_paused(self, track_index: int) -> bool:
        """Whether a specific track is paused."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.is_track_paused(track_index)
        return False

    def is_track_virtual(self, track_index: int = 0) -> bool:
        """True when playback advances without occupying a physical mixing voice."""
        cpp = self._cpp_component
        return cpp.is_track_virtual(track_index) if cpp is not None else False

    @property
    def rejected_one_shot_count(self) -> int:
        """Quieter incoming one-shots rejected by this source's full pool."""
        cpp = self._cpp_component
        return cpp.rejected_one_shot_count if cpp is not None else 0

    @property
    def is_playing(self) -> bool:
        """Whether track 0 is currently playing."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.is_track_playing(0)
        return False

    @property
    def is_paused(self) -> bool:
        """Whether track 0 is paused."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.is_track_paused(0)
        return False

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def game_object_id(self) -> int:
        """Owning GameObject ID (for Wwise gameObjectId mapping)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.game_object_id
        return 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        """Serialize AudioSource to JSON string (delegates to C++)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.serialize()
        return "{}"

    def deserialize(self, json_str: str) -> bool:
        """Deserialize AudioSource from JSON string (delegates to C++)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.deserialize(json_str)
        return False
