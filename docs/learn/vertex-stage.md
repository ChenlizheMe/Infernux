<!-- language:en -->

<span class="mini-tag">Custom Rendering · Chapter 2</span>

# Vertex assets and shared deformation

Create `*.vert.py` for local-space vertex deformation. This is an independent static GPU asset. A Material selects its vertex and fragment in the Inspector, and Slang compiles the selected pair. Its class body never executes as a game script. Ordinary surfaces already have the engine's instanced/skinned vertex path and need no vertex file.

## A first deformation {#minimal-stage}

Save this as `wave.vert.py`:

```python
import infernux.shader as shader


class Wave(shader.vert):
    shader_id = "example/vertex/wave"
    amplitude = shader.material.float(default=0.1, range=(0.0, 0.5), label="Amplitude")
    frequency = shader.material.float(default=4.0)
    speed = shader.material.float(default=2.0)

    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        phase = vertex.position_os.x * self.frequency + ctx.time * self.speed
        vertex.position_os.y += shader.sin(phase) * self.amplitude
        return vertex
```

Create an independent `painted.frag.py`:

```python
import infernux.shader as shader


class Painted(shader.surface):
    shader_id = "example/materials/wave"
    base_color = shader.material.color(default=(0.2, 0.6, 1.0, 1.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=self.base_color.rgb, normal_ws=ctx.normal_ws)
```

Select the vertex and fragment independently in the Material Inspector, or set the vertex ID after creating a Material from its fragment. A matching filename never changes a material:

```python
import infernux as inx

material = inx.Material.from_shader("example/materials/wave", "Water")
material.vert_shader_name = "example/vertex/wave"
material.set_float("amplitude", 0.2)
renderer.set_material(material)
```

Vertex fields appear in the same Material Inspector as fragment fields. Authors do not declare locations, descriptor bindings, a second parameter block or a second material. The ID locates an asset; its GUID owns the program identity. Renaming a class does not change that identity.

## Geometry values {#vertex-layout}

The entry is `deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData`. Modify and return the input. Deformation runs before skinning and the object transform.

| VertexData field | Meaning |
| --- | --- |
| `position_os` | Local position, vec3 |
| `normal_os` | Local normal, vec3 |
| `tangent_os` | Local tangent XYZ and handedness W, vec4 |
| `color` | Vertex color, vec3 |
| `uv0`, `uv1` | Mesh UV sets, vec2 |

| VertexContext field | Meaning |
| --- | --- |
| `time` | Time of this geometry sample, seconds |
| `delta_time` | Current engine frame interval, seconds |
| `model_matrix` | Object matrix for this sample, mat4 |
| `object_id` | Object identity, uvec2 |
| `layer_mask` | Object lighting layer mask, uint |

Bone indices, weights, instance addressing, projection and clipping remain engine data. Returning VertexData preserves those streams. Update normals and tangents when a deformation changes surface orientation; moving positions alone does not derive a new normal.

The standard entry preserves the engine's mesh-to-fragment contract. Arbitrary user varyings and custom clip-space exits are still being extended. Geometry and tessellation stages are separate future capabilities.

## One material layout {#vertex-hook}

The selected vertex and fragment fields merge before compilation. A field declared in both files must have the same type, default and presentation metadata. It then has one value and one target layout shared by both stages. Conflicts identify the field and original source. A rejected selection leaves the previous complete program active. Vertex-only fields use normal Material APIs and serialization.

ShaderLib imports work as they do in surfaces. For example, save this as `bend.shaderlib.py`, then use `from .bend import bend` in the vertex asset:

```python
import infernux.shader as shader


def bend(position: shader.vec3, time: float, amplitude: float) -> shader.vec3:
    return position + shader.vec3(0.0, shader.sin(position.x * 4.0 + time) * amplitude, 0.0)
```

Dependencies and diagnostics follow the consuming surface. Use `shader.sample_level(image, uv, lod)` for vertex texture sampling: a vertex stage has no fragment derivatives for implicit LOD. GPU imports use `infernux.shader` and static ShaderLib declarations, without NumPy or host code.

## Every geometry pass {#geometry-data}

The same deformation feeds visible color, supported GBuffer output, shadow casting, depth, picking, normals, base color and motion. Surface alpha supplies their common coverage rule. The engine applies the appropriate camera or light projection.

Motion evaluates deformation at the previous sample, with the previous object transform and skeletal pose. Each camera graph retains its last submitted sample time; skipped views do not assume a fixed frame interval. A new or invalidated view does not claim valid previous motion. Each evaluation receives its own `ctx.time`; `ctx.delta_time` still describes the current frame.

Geometry evaluation currently uses current material values. Previous parameter values and independently changing displacement textures require an explicit history contract for exact motion. This entry does not record arbitrary GPU state.

Deformation changes GPU vertices, not CPU collision shapes or mesh bounds. Keep conservative bounds over the displacement range, or the mesh may be culled before the vertex program runs. Rigidbody geometry remains separate gameplay data. A static renderer is also a shadow-cache promise: time-varying deformation needs a dynamic renderer.

## Creation and diagnostics {#reuse}

Project creation generates a snake_case `.vert.py` with a `shader.vert` class. A surface may declare its shading model; vertex selection belongs to the Material. Change either stage in the Inspector without editing the other source. Standalone fragment creation starts with the standard vertex.

```python
report = inx.Shader.check("Assets/Shaders/wave.vert.py")
schema = inx.Shader.get_schema("example/materials/wave")
```

Standalone checking verifies the typed definition and libraries. The surface verifies the complete graphics recipe and target entries. Vertex errors retain their original file and line. Successful hot edits publish dependent programs; already submitted work retains its resources.

<!-- language:zh -->

<span class="mini-tag">自定义渲染 · 第 2 章</span>

# 顶点资产与共享变形

需要局部空间顶点变形时，创建 `*.vert.py`。它是独立的静态 GPU 资产，材质在 Inspector 分别选择顶点与片元，再由 Slang 编译所选组合，类体不会作为游戏脚本执行。普通 Surface 已有引擎的实例化、蒙皮和标准变换，不必再写顶点文件。

## 第一个变形 {#minimal-stage}

保存为 `wave.vert.py`：

```python
import infernux.shader as shader


class Wave(shader.vert):
    shader_id = "example/vertex/wave"
    amplitude = shader.material.float(default=0.1, range=(0.0, 0.5), label="幅度")
    frequency = shader.material.float(default=4.0)
    speed = shader.material.float(default=2.0)

    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        phase = vertex.position_os.x * self.frequency + ctx.time * self.speed
        vertex.position_os.y += shader.sin(phase) * self.amplitude
        return vertex
```

另建独立的 `painted.frag.py`：

```python
import infernux.shader as shader


class Painted(shader.surface):
    shader_id = "example/materials/wave"
    base_color = shader.material.color(default=(0.2, 0.6, 1.0, 1.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=self.base_color.rgb, normal_ws=ctx.normal_ws)
```

在材质 Inspector 中分别选择顶点与片元，也可以由片元创建材质后设置顶点 ID。引擎不会因目录里出现同名文件就暗中替换阶段：

```python
import infernux as inx

material = inx.Material.from_shader("example/materials/wave", "水面")
material.vert_shader_name = "example/vertex/wave"
material.set_float("amplitude", 0.2)
renderer.set_material(material)
```

顶点与片元的字段进入同一个 Material Inspector，无须声明 location、绑定号、第二份参数块或第二份材质。ID 用于定位，GUID 持有程序资产身份；修改类名不会改变身份。

## 几何数据 {#vertex-layout}

入口为 `deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData`。修改并返回传入值；引擎先执行变形，再做蒙皮和对象变换。

| VertexData 字段 | 含义 |
| --- | --- |
| `position_os` | 局部位置，vec3 |
| `normal_os` | 局部法线，vec3 |
| `tangent_os` | 局部切线 XYZ 与手性 W，vec4 |
| `color` | 顶点色，vec3 |
| `uv0`、`uv1` | 两组 UV，vec2 |

| VertexContext 字段 | 含义 |
| --- | --- |
| `time` | 本次几何采样时间，秒 |
| `delta_time` | 当前引擎帧间隔，秒 |
| `model_matrix` | 本次采样的对象矩阵，mat4 |
| `object_id` | 对象标识，uvec2 |
| `layer_mask` | 对象光照层掩码，uint |

骨骼索引、权重、实例寻址、投影和裁剪由引擎持有，返回 VertexData 不会丢掉这些数据。变形改变表面朝向时，需要同步更新法线和切线，只移动位置不会自动推导解析法线。

当前标准入口保留引擎既定的网格到片元约定。任意自定义 varying、直接裁剪空间输出仍在扩展；Geometry 和 Tessellation 是独立的后续能力。

## 一份材质布局 {#vertex-hook}

编译前会合并所选两个阶段的材质字段。同名字段的类型、默认值与展示元数据必须一致，合并后共享一个值和一份目标布局。冲突会指出字段与源码位置，拒绝本次替换，保留上一套完整程序。只在顶点声明的字段同样使用正常的 Material API 与序列化。

顶点使用与 Surface 相同的静态 ShaderLib。例如保存以下函数为 `bend.shaderlib.py`，在顶点文件中 `from .bend import bend`：

```python
import infernux.shader as shader


def bend(position: shader.vec3, time: float, amplitude: float) -> shader.vec3:
    return position + shader.vec3(0.0, shader.sin(position.x * 4.0 + time) * amplitude, 0.0)
```

依赖与错误位置跟随使用它的 Surface。顶点采样使用 `shader.sample_level(image, uv, lod)`，因为该阶段没有片元导数来选择隐式 LOD。GPU 文件只引入 `infernux.shader` 与静态 ShaderLib，不引入 NumPy 或宿主代码。

## 各几何 Pass 共用 {#geometry-data}

可见颜色、受支持的 GBuffer、阴影、深度、拾取、法线、反照率和运动矢量都使用同一个变形。Surface 的 alpha 决定共同覆盖，引擎分别应用相机或光源投影。

运动矢量会在上一采样时间重新求值，使用上一对象矩阵和骨骼姿态。每个相机图保存上一次真正提交的时间，视图跳帧时不会假设固定帧间隔。新视图或历史失效后的首帧不声称有上一运动数据。两次求值分别得到自己的 `ctx.time`；`ctx.delta_time` 仍表示当前帧间隔。

几何求值目前使用当前材质值。前一时刻的参数、独立变化的位移纹理，需要明确的历史约定才能产生精确运动；这个入口不会替用户记录任意 GPU 状态。

变形修改 GPU 顶点，不会修改 CPU 碰撞体或网格包围盒。包围盒应覆盖完整位移范围，否则物体可能在顶点程序运行前被剔除。刚体属于独立的玩法数据。声明静态还意味着允许阴影缓存，随时间变形的物体应使用动态 Renderer。

## 创建与诊断 {#reuse}

Project 会生成 snake_case 的 `.vert.py` 与 `shader.vert` 类。Surface 可以声明光照模型；顶点选择属于材质。在 Inspector 更换任何一端时，无须修改另一端源码。单独由片元创建材质时，初始顶点为标准顶点。

```python
report = inx.Shader.check("Assets/Shaders/wave.vert.py")
schema = inx.Shader.get_schema("example/materials/wave")
```

独立检查会核对顶点定义与库，Surface 会验证完整配方与目标入口。顶点错误保留原文件和行号。热改成功后发布依赖程序，已经提交的工作继续持有自己的资源。
