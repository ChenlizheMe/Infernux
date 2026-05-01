from __future__ import annotations

from typing import Any

from Infernux.components.builtin_component import BuiltinComponent

class AudioSource(BuiltinComponent):
    """Multi-track audio playback component.

    Infernux does not expose Unity's single ``clip`` field. Instead one
    AudioSource owns ``track_count`` tracks. Assign each track with
    ``set_track_clip(index, clip)`` or ``set_track_clip_by_guid(index, guid)``,
    then call ``play(index)``. ``play_on_awake`` only auto-plays track 0.

    For transient SFX, prefer ``play_one_shot(clip, volume_scale)`` rather than
    creating temporary AudioSource objects. Sources are spatialized; for "2D"
    audio, place the AudioSource on/near the AudioListener's GameObject.
    """

    _cpp_type_name: str
    _component_category_: str

    # ---- CppProperty fields as properties ----

    @property
    def track_count(self) -> int:
        """Number of audio tracks on this source, valid range 1..16."""
        ...
    @track_count.setter
    def track_count(self, value: int) -> None: ...

    @property
    def volume(self) -> float:
        """The overall volume of the audio source."""
        ...
    @volume.setter
    def volume(self, value: float) -> None: ...

    @property
    def pitch(self) -> float:
        """The pitch multiplier of the audio source."""
        ...
    @pitch.setter
    def pitch(self, value: float) -> None: ...

    @property
    def mute(self) -> bool:
        """Whether the audio source is muted."""
        ...
    @mute.setter
    def mute(self, value: bool) -> None: ...

    @property
    def loop(self) -> bool:
        """Whether all tracks loop when they reach the end."""
        ...
    @loop.setter
    def loop(self, value: bool) -> None: ...

    @property
    def play_on_awake(self) -> bool:
        """Whether track 0 plays automatically during component start."""
        ...
    @play_on_awake.setter
    def play_on_awake(self, value: bool) -> None: ...

    @property
    def min_distance(self) -> float:
        """Distance where 3D attenuation begins."""
        ...
    @min_distance.setter
    def min_distance(self, value: float) -> None: ...

    @property
    def max_distance(self) -> float:
        """Distance where 3D attenuation reaches minimum volume."""
        ...
    @max_distance.setter
    def max_distance(self, value: float) -> None: ...

    @property
    def one_shot_pool_size(self) -> int:
        """The maximum number of concurrent one-shot sounds."""
        ...
    @one_shot_pool_size.setter
    def one_shot_pool_size(self, value: int) -> None: ...

    @property
    def output_bus(self) -> str:
        """Output mixer/audio bus name. Currently script-only, not inspector field."""
        ...
    @output_bus.setter
    def output_bus(self, value: str) -> None: ...

    # ---- Track management ----

    def set_track_clip(self, track_index: int, clip: Any) -> None:
        """Assign an AudioClip wrapper or native clip to a zero-based track.

        ``clip`` can be ``Infernux.core.audio_clip.AudioClip`` or a native
        ``Infernux.lib.AudioClip``. Keep the clip loaded for as long as the
        source may play it.
        """
        ...
    def get_track_clip(self, track_index: int) -> Any:
        """Return the audio clip assigned to the specified track."""
        ...
    def get_track_clip_guid(self, track_index: int) -> str:
        """Return the asset GUID of the clip on the specified track."""
        ...
    def set_track_clip_by_guid(self, track_index: int, guid: str) -> None:
        """Assign an audio clip to a track by asset GUID.

        Requires the AssetRegistry/AssetDatabase to be initialized. If GUID
        resolution is unavailable, load by file path and call ``set_track_clip``.
        """
        ...
    def set_track_volume(self, track_index: int, volume: float) -> None:
        """Set the volume of the specified track."""
        ...
    def get_track_volume(self, track_index: int) -> float:
        """Return the volume of the specified track."""
        ...

    # ---- Playback control ----

    def play(self, track_index: int = ...) -> None:
        """Start playback on the specified zero-based track."""
        ...
    def stop(self, track_index: int = ...) -> None:
        """Stop playback on the specified zero-based track."""
        ...
    def play_one_shot(self, clip: Any, volume_scale: float = ...) -> None:
        """Play a transient clip using the source's pooled one-shot voices."""
        ...
    def stop_one_shots(self) -> None:
        """Stop all currently playing one-shot sounds."""
        ...
    def pause(self, track_index: int = ...) -> None:
        """Pause playback on the specified track."""
        ...
    def un_pause(self, track_index: int = ...) -> None:
        """Resume playback on the specified track."""
        ...
    def stop_all(self) -> None:
        """Stop playback on all tracks and pooled one-shot voices."""
        ...
    def is_track_playing(self, track_index: int) -> bool:
        """Return whether the specified track is currently playing."""
        ...
    def is_track_paused(self, track_index: int) -> bool:
        """Return whether the specified track is currently paused."""
        ...

    @property
    def is_playing(self) -> bool:
        """Whether track 0 is currently playing (convenience)."""
        ...
    @property
    def is_paused(self) -> bool:
        """Whether track 0 is currently paused (convenience)."""
        ...

    # ---- Read-only properties ----

    @property
    def game_object_id(self) -> int:
        """The ID of the GameObject this component is attached to."""
        ...

    # ---- Serialization ----

    def serialize(self) -> str:
        """Serialize the component to a JSON string."""
        ...
    def deserialize(self, json_str: str) -> bool:
        """Deserialize the component from a JSON string."""
        ...
