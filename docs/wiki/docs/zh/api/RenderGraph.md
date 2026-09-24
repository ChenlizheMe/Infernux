# RenderGraph

<div class="class-info">
类位于 <b>Infernux.rendergraph</b>
</div>

## 描述

声明式渲染图。用 Pass 描述你想怎么画，引擎帮你调度。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 构造函数

| 签名 | 描述 |
|------|------|
| `RenderGraph.__init__(name: str = ..., output_samples: int = 0) → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| name | `str` | The name of this render graph. *(只读)* |
| pass_count | `int` | Number of render passes in the graph. *(只读)* |
| texture_count | `int` | Number of texture resources in the graph. *(只读)* |
| buffer_count | `int` | Number of buffer resources in the graph. *(只读)* |
| topology_sequence | `List[Tuple[str, str]]` | Ordered list of (pass_name, type) entries defining the execution order. *(只读)* |
| injection_points | `list` | List of injection points for pass extension. *(只读)* |
| effect_stages | `List[EffectStage]` | Pipeline-declared user attachment stages in topology order. *(只读)* |
| current_effect_resources | `Mapping[str, TextureHandle]` |  *(只读)* |
| current_pass_result | `PassResult | None` |  *(只读)* |
| geometry_buffer_requirements | `` |  *(只读)* |
| pass_results | `Mapping[str, PassResult]` |  *(只读)* |
| latest_pass_result | `PassResult | None` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `set_temporal_jitter(enabled: bool = True) → None` |  |
| `set_msaa_samples(samples: int) → int` | Set screen MSAA preference; return the effective Camera target sample count. |
| `create_texture(name: str, format: Format = ..., camera_target: bool = ..., size: Optional[Tuple[int, int]] = ..., size_divisor: int = ..., samples: Optional[int] = ...) → TextureHandle` | 创建临时纹理。 |
| `get_texture(name: str) → Optional[TextureHandle]` | Get a texture handle by name, or None if not found. |
| `import_texture(name: str, texture: RenderTexture | InxTexture, attachment: str = 'color') → TextureHandle` | 导入持久 RenderTexture 附件或 GUID 管理的 2D/3D 纹理资产。资产纹理只读采样，重导入后无需重开场景即可更新。 |
| `create_temporal_history(name: str, format: Format = Format.RGBA16_SFLOAT, size: Optional[Tuple[int, int]] = None, size_divisor: int = 0) → Tuple[TextureHandle, TextureHandle]` |  |
| `name_scope(prefix: str) → AbstractContextManager[RenderGraph]` |  |
| `effect_resources(resources: Mapping[str, TextureHandle]) → AbstractContextManager[RenderGraph]` |  |
| `pass_result(result: PassResult) → AbstractContextManager[RenderGraph]` |  |
| `replace_current_pass_result(result: PassResult) → None` |  |
| `resolve_effect_route_policy(stages)` |  |
| `create_buffer(name: str, byte_size: int, storage: bool = ..., indirect: bool = ..., transfer_source: bool = ..., transfer_destination: bool = ...) → BufferHandle` | Declare a transient buffer resource in the render graph. |
| `get_buffer(name: str) → Optional[BufferHandle]` | Get a buffer handle by name, or None if not found. |
| `has_pass(name: str) → bool` | Check if a render pass with the given name exists. |
| `has_injection_point(name: str) → bool` | Check if an injection point with the given name exists. |
| `has_effect_stage(stable_id: str) → bool` | Check for an exact declared stage ID. |
| `injection_point(name: str, display_name: str = ..., resources: Optional[set] = ...) → None` | Declare an injection point where external passes can be inserted. |
| `effect_stage(stable_id: str, scope: EffectScope | str = ..., display_name: str = ..., inputs: Optional[set[str]] = ..., outputs: Optional[set[str]] = ..., capabilities: Optional[set[str]] = ...) → EffectStage` | Declare a stable user-facing RenderEffect attachment stage. |
| `effects(stable_id: str) → EffectStage` | Pipeline-author shorthand for ``effect_stage``. |
| `screen_ui_section(resources: set | None = ..., world_ui_layer_mask: int = ...) → None` | Declare a screen UI section in the graph topology. |
| `set_geometry_buffer_requirements(requirements) → None` |  |
| `require_geometry_buffers(requirements) → None` |  |
| `needs_geometry_buffer(semantic: str) → bool` |  |
| `publish_pass_result(source: str, buffers, materialize = ...) → PassResult` |  |
| `derive_pass_result(source: str, parent: PassResult, overrides) → PassResult` |  |
| `write_buffer(source: str, parent: PassResult, name: str, texture: TextureHandle) → PassResult` |  |
| `get_pass_result(source: str) → PassResult | None` |  |
| `camera_ui_section(resources: set | None = ..., world_ui_layer_mask: int = ...) → None` | Draw Camera UI and declare the after-camera-UI effect stage. |
| `screen_ui_overlay_section(resources: set | None = ...) → None` | Encode for display, draw Screen UI, and declare its effect stage. |
| `add_pass(name: str) → RenderPassBuilder` | 添加一个渲染 Pass。 |
| `add_copy_pass(name: str) → RenderPassBuilder` | Add a transfer-domain texture or buffer copy pass. |
| `add_present_pass(name: str) → RenderPassBuilder` | Add a final graph export pass. |
| `remove_pass(name: str) → RenderPassBuilder | None` | Remove a render pass by name. |
| `append_pass(builder: RenderPassBuilder) → None` | Append an existing RenderPassBuilder to the graph. |
| `set_output(texture: str | TextureHandle) → None` | Set the final output texture of the render graph. |
| `validate_no_ip_before_first_pass() → None` | Validate that no user extension point appears before the first pass. |
| `get_debug_string() → str` | Return a human-readable summary of the graph for debugging. |
| `build() → RenderGraphDescription` | Compile the graph into a RenderGraphDescription for the backend. |

<!-- USER CONTENT START --> public_methods

导入 `InxTexture` 时，资源必须具有已发布的 GUID 和明确的 `2d` 或 `3d`
维度元数据，shader 资源维度必须与之匹配。资产纹理不能作为渲染图输出、
Copy 目标、颜色/深度附件或 Resolve 目标。冷加载资产不会阻塞编辑器帧；资源
准备完成后，渲染图会重新尝试发布。

声明 `Requires [Lighting]` 的 fullscreen shader 会通过描述符 set 1 获得当前
Camera/View 独立的 Lighting UBO、Forward+ 灯光列表、阴影图集和相机位置。
阴影纹理仍必须作为显式 `shadow_map` Pass 依赖声明，保证调度关系可见。
声明式 Deferred route 在开启阴影时，会在每个 GBuffer 几何 Pass 上声明
`set_texture("shadowMap", shadow_map)`。这会将同一 View 的阴影图像发布给
per-view 灯光描述符，并保证阴影投射 Pass 排在光照之前。未开启阴影时，
route 不声明 `shadowMap` 绑定。

默认 Deferred 管线的 `gbuffer_normal` 是私有纹理：RGB 存编码后的世界法线，
alpha 存供延迟光照使用的 smoothness。仅当消费者请求公开 `normal` 时，
GBuffer 结果才生成独立的全分辨率 RGBA16 纹理：RGB 存当前 View 的编码法线，
可见 DeferredCompatible 几何处 alpha 为 1，其余清空区域为 0。Forward+ 不兼容
材质补绘之后，`opaque_lighting` 结果先复制当前公开 normal（包括
`after_gbuffer` 效果的改写），再依据最终只读深度覆盖可见的
DeferredUnsupported 几何；后续阶段继承这一结果。替换 `normal` 的效果必须
保持 alpha 的 coverage 约定。

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example
> **示例状态：** 当前尚未为此符号验证示例。请以上方签名为准；不要根据其他引擎中的同名 API 推测行为。
<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
