# AudioSource

<div class="class-info">
class in <b>InfEngine.components.builtin</b>
</div>

**Inherits from:** [BuiltinComponent](Component.md)

## Description

Plays audio clips in the scene.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| track_count | `int` | The number of audio tracks on this source. |
| volume | `float` | The overall volume of the audio source. |
| pitch | `float` | The pitch multiplier of the audio source. |
| mute | `bool` | Whether the audio source is muted. |
| loop | `bool` | Whether the audio clip loops when it reaches the end. |
| play_on_awake | `bool` | Whether the audio source plays automatically on awake. |
| min_distance | `float` | The minimum distance for 3D audio attenuation. |
| max_distance | `float` | The maximum distance for 3D audio attenuation. |
| one_shot_pool_size | `int` | The maximum number of concurrent one-shot sounds. |
| output_bus | `str` | The name of the audio mixer bus to route output to. |
| is_playing | `bool` | Whether any track is currently playing. *(read-only)* |
| is_paused | `bool` | Whether any track is currently paused. *(read-only)* |
| game_object_id | `int` | The ID of the GameObject this component is attached to. *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `set_track_clip(track_index: int, clip: Any) → None` | Assign an audio clip to the specified track. |
| `get_track_clip(track_index: int) → Any` | Return the audio clip assigned to the specified track. |
| `get_track_clip_guid(track_index: int) → str` | Return the asset GUID of the clip on the specified track. |
| `set_track_clip_by_guid(track_index: int, guid: str) → None` | Assign an audio clip to a track by its asset GUID. |
| `set_track_volume(track_index: int, volume: float) → None` | Set the volume of the specified track. |
| `get_track_volume(track_index: int) → float` | Return the volume of the specified track. |
| `play(track_index: int = ...) → None` | Start playback on the specified track. |
| `stop(track_index: int = ...) → None` | Stop playback on the specified track. |
| `play_one_shot(clip: Any, volume_scale: float = ...) → None` | Play an audio clip as a one-shot sound. |
| `stop_one_shots() → None` | Stop all currently playing one-shot sounds. |
| `pause(track_index: int = ...) → None` | Pause playback on the specified track. |
| `un_pause(track_index: int = ...) → None` | Resume playback on the specified track. |
| `stop_all() → None` | Stop playback on all tracks and one-shots. |
| `is_track_playing(track_index: int) → bool` | Return whether the specified track is currently playing. |
| `is_track_paused(track_index: int) → bool` | Return whether the specified track is currently paused. |
| `serialize() → str` | Serialize the component to a JSON string. |
| `deserialize(json_str: str) → bool` | Deserialize the component from a JSON string. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for AudioSource
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
