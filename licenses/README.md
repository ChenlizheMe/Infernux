# 许可证、版权声明与签名政策

这里集中维护 Infernux 自有的第三方声明和公开签名政策，并索引依赖随附的原始授权文件。引擎自身的 [MIT 许可证](../LICENSE) 保留在仓库根目录。

这些文件属于源码和发行说明，不是项目资产，不参与资源导入，也不复制到项目的 `Assets` 或 `Library`。发行包中随依赖附带的声明由相应的打包流程保留。

## 代码签名

SignPath 的服务署名、适用范围和签名流程见 [英文政策](signing/CODE_SIGNING_POLICY.md)及[中文政策](signing/CODE_SIGNING_POLICY-zh.md)。代码签名政策说明发行身份，不改变引擎或第三方软件的许可证。

签名操作配置仍由 [release 脚本](../scripts/release/README.md)维护；此目录只存放公开说明。

## 原生依赖与编译器

上游声明保留在对应依赖中，随 Git 子模块版本一起更新。下面直接指向仓库实际使用的源码，避免另外维护一套容易过时的副本。

| 组件 | 原始声明与随附说明 |
| --- | --- |
| MikkTSpace | [项目随附声明](MikkTSpace.txt)、[源码内原始声明](../external/MikkTSpace/mikktspace.h) |
| SDL | [LICENSE](../external/sdl_for_infernux/LICENSE.txt) |
| Assimp | [LICENSE](../external/assimp/LICENSE) |
| Jolt Physics | [LICENSE](../external/joltphysics_for_infernux/LICENSE) |
| Dear ImGui | [LICENSE](../external/imgui_for_infernux/LICENSE.txt) |
| GLM | [copying.txt](../external/glm/copying.txt) |
| glslang | [LICENSE](../external/glslang/LICENSE.txt) |
| Vulkan Memory Allocator | [LICENSE](../external/VulkanMemoryAllocator/LICENSE.txt) |
| stb | [LICENSE](../external/stb/LICENSE) |
| LunaSVG / PlutoVG | [来源、修改与子依赖说明](svg/NOTICE.txt)、[MIT](svg/LunaSVG-PlutoVG.txt)、[FreeType](svg/FTL.txt)；静态链接，声明随 wheel 打包 |
| dr_libs | [引入说明](../external/dr_libs/README.infernux.md)；[WAV](../external/dr_libs/dr_wav.h)、[MP3](../external/dr_libs/dr_mp3.h)、[FLAC](../external/dr_libs/dr_flac.h) 文件末尾的完整授权声明 |
| nlohmann/json | [头文件内版权及许可证标识](../external/nlohmann/json.hpp) |
| Taichi | [LICENSE](../external/taichi_for_infernux/LICENSE)、[NOTICE](../external/taichi_for_infernux/NOTICE) |
| llvmlite | [LICENSE](../external/llvmlite_for_infernux/LICENSE)、[第三方声明](../external/llvmlite_for_infernux/LICENSE.thirdparty)、[运行时说明](../external/llvmlite_for_infernux/NOTICE.runtime)、[Windows runtime](../external/llvmlite_for_infernux/LICENSE.windows-runtime)、[zlib](../external/llvmlite_for_infernux/LICENSE.zlib)、[zstd](../external/llvmlite_for_infernux/LICENSE.zstd) |

依赖目录内更细粒度的文件头声明、第三方子依赖声明仍以原文件为准；这张表不是对其授权条款的替代。

## 网站字体、图标与动画库

网站部署需要携带这些文件，因此原件保留在 `docs/assets/vendor-licenses`，从此处统一索引。

| 组件 | 随附声明 |
| --- | --- |
| Noto Sans SC | [OFL](../docs/assets/vendor-licenses/Noto-Sans-SC-OFL.txt) |
| Inter | [OFL](../docs/assets/vendor-licenses/Inter-OFL.txt) |
| Space Grotesk | [OFL](../docs/assets/vendor-licenses/Space-Grotesk-OFL.txt) |
| JetBrains Mono | [OFL](../docs/assets/vendor-licenses/JetBrains-Mono-OFL.txt) |
| Font Awesome | [LICENSE](../docs/assets/vendor-licenses/Font-Awesome-LICENSE.txt) |
| GSAP 3.13.0 | [随附声明](../docs/assets/vendor-licenses/GSAP-3.13.0.txt) |

新增素材的授权说明应同时记录素材路径、作者、来源及原始授权文本，并更新本索引；不能只记录一个许可证名称而丢失来源。
