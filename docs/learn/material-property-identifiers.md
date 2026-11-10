# 材质属性名称与 ID

材质属性可以用字符串名称，也可以用 `Shader.property_to_id` 返回的整数 ID。
同一名称在当前进程中始终得到同一 ID，不同材质共享这份名称表。

```python
from infernux.core.material import Material
from infernux.core.shader import Shader

metallic = Shader.property_to_id("metallic")
material = Material.create_lit()
material.set_float(metallic, 0.5)
assert material.get_float("metallic") == material.get_float(metallic)
```

浮点、整数、向量、颜色、矩阵、数组和纹理的原生属性入口共用同一种名称解析。
`has_property`、`get_property` 和 `remove_property` 也接受 ID。纹理偏移和缩放仍写入
原有的 `<名称>_ST` 属性，不创建第二份 UV 状态。

名称表属于进程，不属于某个材质。删除属性、重新载入材质或创建材质副本不会改变已登记
名称的含义。ID 不保存到材质文件，也不跨进程或跨运行复用；保存名称，在启动时取得 ID。
任意未登记整数不能充当名称，解析时报 `KeyError`。已经登记但不在某个材质中的名称继续
遵守原来的缺失属性规则，例如 `has_property` 返回 `False`，`get_property` 返回 `None`。

`Material.property_to_id(name)` 与 `Shader.property_to_id(name)` 返回相同 ID。
`Material.property_name(name_or_id)` 可在配置时取得名称。

补间接入时应在创建任务时解析 ID，后续采样复用准备好的属性名称。属性 ID 不缓存材质内部
元素的地址，因此不会因为删除属性、替换文档或资源重新载入而留下失效指针。原有字符串
调用直接进入字符串转换，不访问 ID 名称表。名称表的登记与整数解析在原生层同步，普通
数值批量计算的 GIL 释放策略保持独立。
