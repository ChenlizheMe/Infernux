# 平台 Player 随引擎版本发布

Windows、Linux、Web 和 Android 平台插件是引擎制品的一部分。它们的源码仍在各自的子仓库中维护，二进制由引擎仓库统一构建。普通编辑器插件继续使用独立版本和发布流程。

引擎版本、热修号、Python ABI 和目标平台必须匹配。首次发布 `0.5.0` 时，各平台安装包使用 `0.5.0`；后续 `0.5.0-v2` 对应 `0.5.0.post2`。载荷清单保留准确的 `engine_release` 和目标 ABI。包版本由 `python/infernux/version.py` 自动生成，不再手工调整各个平台的包版本。这样共享缓存和项目锁文件不会混用不同热修的 Player。

平台构建统一登记在 `python/infernux/resources/platform-players.json`。制品收集器、发布工具和重建器使用这份声明；新增平台也进入同样的版本和完整性检查。插件安装器不识别特殊的平台名称，所有插件都按 `engine` 兼容范围选择可用版本。

## 安装和升级规则

引擎兼容版本包括热修号，例如界面上的 `0.5.0-v2` 对应兼容版本 `0.5.0.post2`。普通插件可以声明 `>=0.5.0,<0.6`，平台插件声明 `==0.5.0.post2`。精确的 `==0.5.0` 只匹配首次发布，不代表全部热修。

公共目录保留每个插件的历史版本。安装器先按当前引擎兼容范围过滤，再按插件版本排序，选择最高的可用版本。不能只查 GitHub 的 latest：最新插件可能只支持更高版本引擎。

刷新目录更新发现结果，不在编辑器启动时自动替换项目里的插件。用户执行安装请求时使用当前目录中的最新兼容版本；已安装同版本直接复用。修改过的项目文件继续由更新事务检查，不能因为自动选择版本就丢失用户修改。明确指定的版本仍需通过相同的兼容检查。

## 发布顺序

1. 完成引擎版本、热修号和子模块提交的调整，提交源码。
2. 运行引擎发布流程。`engine-platform-plugins.yml` 使用同一引擎提交及其固定的子模块，构建 Windows、Linux、Web，以及 Android 的 ARM64/x64 载荷。Web 同时收集 Windows/Linux 着色器工具。
3. 统一打包工具给四个平台包写入对应的安装版本，生成摘要、文件大小和引擎源码提交记录。
4. 收集阶段要求登记的平台全部齐全、源码提交相同、引擎版本相同，并校验每个包的摘要。发布前还要与实际 Windows/Linux SDK wheel 的来源记录和 Python ABI 对照；任一包属于另一份源码、SDK 含未提交源码修改或 ABI 不同，都不能进入引擎发布阶段。
5. `publish_engine_plugins.py` 在独立克隆中给各个平台仓库写入包版本和精确 `engine` 约束，只提交这两个字段的变化，推送默认分支及 `v<package_version>` 标签，再在各自仓库发布已验证的 `.inxpkg` 和清单。若对应仓库代码已偏离编译时固定的提交，拒绝把新代码冒充为已测试制品。重复运行复用已有标签和相同制品，不强制推送。
6. 平台包和清单也作为附件保存在引擎 GitHub Release。Cloudflare R2 使用 `platform-plugins/<engine_release>/` 保存它们，先上传全部包，最后上传该版本的 `registry.json` 和包含全部历史版本的公共发现目录；新版本不会覆盖旧版本的入口。

SDK 内置的官方目录在 wheel 组装时生成，引用当前引擎版本的制品地址。在线刷新读取公共历史目录，采用与普通插件相同的兼容版本筛选。GitHub 备用下载使用目录声明的插件源码仓库及插件标签，不能临时拼接另一版本的地址。

跨仓库同步需要在引擎发布仓库配置 `PLATFORM_RELEASE_TOKEN`，它需要各个平台仓库的 contents:write 权限。引擎自己的 `GITHUB_TOKEN` 不能替代跨仓库凭据。发布脚本复用该凭据更新源码和 Release，构建、检查及本地刷新命令不需要它。CI 的正式发布和普通插件目录发布串行进行，避免同时改写公共历史目录。

首次使用新目录前，需要部署更新后的 `scripts/release/r2-upload-worker.mjs`。该上传服务接受受限的引擎版本目录及标准平台包文件名，继续使用现有认证。实际发布集合来自平台注册声明，增加平台不需要再修改服务端的固定平台名称列表。修改本地工作流不会自动改变线上服务。

## 本地构建和收集

在引擎仓库根目录，使用带相应工具链的 Python 3.13 环境运行：

```powershell
cmake --preset platform-plugins -DPython3_EXECUTABLE=<Python路径>
cmake --build --preset platform-plugins
```

这个轻量预设调用 `refresh_platform_plugins.py`，默认刷新本机桌面平台：Windows 上刷新 Windows，Linux 上刷新 Linux。脚本调用统一构建器重新配置、编译、生成当前 Python ABI 的运行包，再验证和覆盖对应子模块的 `player` 目录，清除旧载荷中的废弃文件并保留仍存在资源的 GUID。最后输出版本化 `.inxpkg` 和刷新记录。源码包版本声明直到正式 publication 才提交、推送，本地刷新不会推送仓库。

需要其它已配置工具链的目标，可以在配置时设置 `-DINFERNUX_PLATFORM_PLUGIN_TARGETS="windows;android;web"`；Web 还需要 `-DINFERNUX_PLATFORM_PLUGIN_TOOLS=<着色器工具目录>`。Windows/Linux 的本地编译仍需要各自宿主，`all` 不会跳过缺少的工具链或把旧产物当作成功；完整四平台集合交给 CI 构建并收集。

已有桌面 Release 构建目录也可以执行 `cmake --build --preset windows-msvc-platform-plugins` 或 `linux-clang-platform-plugins`，复用其增量编译目录。`windows-release`、`linux-release` 工作流预设已经把这个步骤放在 wheel 打包之前。

只构建版本化安装包、不覆盖子模块载荷时，可以直接调用基础构建器：

```powershell
python scripts/release/build_platform_plugins.py build --targets windows
```

Linux 构建机可以批量构建其支持的目标：

```bash
python scripts/release/build_platform_plugins.py build --targets linux android web --tools /absolute/path/to/web-tools
```

Windows/Linux 桌面目标分别需要对应宿主编译器。一个 Windows 命令不能直接替代 Linux 原生工具链；跨宿主的完整集合由 CI 收集。Android 需要 `ANDROID_SDK_ROOT`、`INFERNUX_ANDROID_PYTHON_PREFIX_ARM64` 和 `INFERNUX_ANDROID_PYTHON_PREFIX_X86_64`。Web 需要 `INFERNUX_EMSDK_ROOT`、`INFERNUX_WEB_CPYTHON_ROOT`、`INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST`、`INFERNUX_WEB_NUMPY_PAYLOAD_ROOT`，以及同时包含两个宿主工具的 `--tools` 目录。这些是源码构建条件，匹配的预编译插件不要求用户重新配置它们。

已有正确版本的原生载荷，可以只执行最终打包：

```powershell
python scripts/release/build_platform_plugins.py package --targets windows --payload <player目录> --output <收集目录>
```

把四个平台生成的 `.inxpkg` 和 `.release.json` 放入同一个目录后检查：

```powershell
python scripts/release/build_platform_plugins.py assemble --output <收集目录>
```

默认输出是 `dist/platform-plugins/<engine_release>/`。上述构建、打包和收集命令不会上传制品、推送 Git 或修改项目 Python runtime。默认手动刷新编译缓存位于 `out/build/platform-plugins/<engine_release>/<python_abi>/`。普通桌面 Release 预设复用已有目录并重新配置构建；最终载荷仍按版本、源码和 ABI 检查，不能只靠缓存目录名判断兼容。安装包只包含插件目录和目标载荷，不包含仓库外层的 CMake、CI、测试和构建缓存。

## 已安装插件不兼容时

构建检查只报告需要重建，不在面板检查阶段执行耗时编译。用户开始构建后，如果已安装平台插件的载荷确实不匹配，构建器会使用当前 SDK 对应的引擎源码重新编译。源码优先使用开发中的引擎仓库；安装版 SDK 使用打包时记录的准确提交，不选择最新主干。未提交源码的开发 SDK 需要通过 `INFERNUX_ENGINE_SOURCE_ROOT` 指向其原始源码，以免把另一份同版本源码误当作它。

编译缓存和日志位于项目的 `Library/PlayerRuntime/`。候选载荷在插件所在磁盘上准备，验证通过后替换已安装插件的 `player` 目录，保留仍存在文件的资源 GUID。失败或取消不会发布半成品；下一次构建直接使用已经修复的插件。引擎 SDK、游戏脚本和其他插件不会因此被替换。缺少编译工具链时会报告具体条件及构建日志，不会把不兼容载荷改标签后继续打包。

Windows 已验证实际重编译、覆盖、再次复用、安装和游戏运行。其他平台的发布作业和版本检查需要在对应 CI 工具链中执行，不能把本地 Windows 验证视为它们的实机验证。

## 增加平台

在统一登记文件中添加平台 ID、插件 reference、源码位置、构建目标、载荷类型和 CI 作业名。现有桌面、Android、Web 载荷复用对应检查；其它原生载荷可以声明 `required_files` 和 `build_commands`，命令参数支持 `{python}`、`{source}`、`{build_root}`、`{payload}` 和环境变量映射 `{env[NAME]}`，不经过 Shell 拼接。

新后端仍需要自己的工具链和 CI 编译作业，并加入收集作业的依赖。工作流测试会核对登记的 `ci_job`，收集器会要求该平台的包和清单实际存在；漏接作业或遗漏制品会使发布失败。版本生成、SDK 对照、目录、下载缓存和普通插件的隔离规则无需为新平台各写一套。
