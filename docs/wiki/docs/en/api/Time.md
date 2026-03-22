# Time

<div class="class-info">
class in <b>InfEngine.timing</b>
</div>

## Description

Provides access to time information for the current frame.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| time | `float` | The time in seconds since the start of the game. *(read-only)* |
| delta_time | `float` | The time in seconds since the last frame. *(read-only)* |
| unscaled_time | `float` | The unscaled time in seconds since the start of the game. *(read-only)* |
| unscaled_delta_time | `float` | The unscaled time in seconds since the last frame. *(read-only)* |
| fixed_delta_time | `float` | The interval in seconds at which physics and fixed updates are performed. *(read-only)* |
| fixed_time | `float` | The time since the last fixed update. *(read-only)* |
| fixed_unscaled_time | `float` | The unscaled time since the last fixed update. *(read-only)* |
| time_scale | `float` | The scale at which time passes (1.0 = normal speed). *(read-only)* |
| frame_count | `int` | The total number of frames rendered since the start of the game. *(read-only)* |
| realtime_since_startup | `float` | The real time in seconds since the application started. *(read-only)* |
| maximum_delta_time | `float` | The maximum time a frame can take before delta_time is clamped. *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for Time
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
