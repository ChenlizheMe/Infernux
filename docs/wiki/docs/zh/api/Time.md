# Time

<div class="class-info">
类位于 <b>InfEngine.timing</b>
</div>

## 描述

时间管理器。掌管每一帧的时间节奏——引擎的心跳。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| time | `float` | 自游戏启动以来的时间（秒）。 *(只读)* |
| delta_time | `float` | 上一帧耗时（秒）。写游戏逻辑离不开它。 *(只读)* |
| unscaled_time | `float` | 不受 time_scale 影响的时间。 *(只读)* |
| unscaled_delta_time | `float` | 不受 time_scale 影响的帧耗时。 *(只读)* |
| fixed_delta_time | `float` | 固定更新时间间隔（秒）。 *(只读)* |
| fixed_time | `float` | The time since the last fixed update. *(只读)* |
| fixed_unscaled_time | `float` | The unscaled time since the last fixed update. *(只读)* |
| time_scale | `float` | 时间缩放。设为 0 暂停，设为 2 双倍速。 *(只读)* |
| frame_count | `int` | 自启动以来的帧数。 *(只读)* |
| realtime_since_startup | `float` | 自启动以来的真实时间（秒）。 *(只读)* |
| maximum_delta_time | `float` | The maximum time a frame can take before delta_time is clamped. *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for Time
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
