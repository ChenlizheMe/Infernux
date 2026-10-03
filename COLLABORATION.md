# 多人协作与项目同步约定

Infernux 项目由 Git 同步源码，由引擎同步运行时状态。两者的边界必须保持清楚：
Git 保存创作输入和稳定身份，引擎在每台机器上重新生成 `Library`、导入缓存和运行时。
不要把某台机器生成的缓存当成项目内容提交。

## 哪些内容进入 Git

以下内容是项目的可移植输入，应提交并参与代码审查：

- `Assets/` 下的场景、Prefab、材质、特效、脚本、模型、纹理、音频和数据资产。
- `Packages/` 下的已安装包内容，以及每个文件对应的 `.meta` sidecar。
- `ProjectSettings/` 下的项目设置、`BuildSettings.json`、`InxPlugins.json`、`InxPackages.lock.json` 和 `PythonRuntime.json`。
- `.infernux-version`、`.gitignore`、`.gitattributes` 和项目文档。

以下内容是本机生成状态，不应提交：

- `Library/`、`Temp/`、`Logs/`、`Cache/`。
- `.runtime/`、`.venv/`、`Build/`、`Builds/`、`Dist/`、`Export/`、`Exports/`。
- `Packages/.staging/`、`Packages/.cache/`、项目锁文件和编辑器缓存。

`.runtime` 不需要同步。项目通过 `.infernux-version` 和
`ProjectSettings/PythonRuntime.json` 声明精确的引擎版本和 Python ABI；Hub 在本机找到
对应版本后从共享运行时重建项目运行时。没有对应版本时项目应保持不可启动，而不是偷偷
使用另一版本。

## GUID 和 `.meta`

`.meta` 是资产身份的权威来源。移动资产必须通过 Editor 的 Project 操作完成，这样会
移动内容和 sidecar 并保留 GUID。不要删除 `.meta` 让引擎“重新生成”，也不要手工复制
GUID；这会把场景中的持久引用变成另一个资产。

`.meta` 中的 `metadata.file_path` 只允许使用相对于项目根的可移植路径，例如
`Assets/Characters/Ship.fbx` 或 `Packages/acme/ships/runtime.py`。它是路径提示，不是
持久引用；场景、Prefab、材质和设置中的资产引用必须保存 GUID，路径只可作为界面提示。
引擎内存中仍会解析出本机绝对路径供加载器使用，但不会把该路径写回项目文件。
模型子资源的嵌套元数据也遵循此规则。本机 `last_modified` 不写入 sidecar；更换克隆目录
不会改变共享身份或因为本机时间不同而改写元数据。旧绝对路径 sidecar 在正常刷新时保留
原 GUID，并转换为当前项目相对路径。
工具显式临时导入项目外资源时，sidecar 省略无法相对项目表示的路径提示，使用调用方指定的
物理路径加载；它不会把作者的绝对路径写进 `.meta`。正式协作资产仍需放入 `Assets` 或
`Packages` 后再提交。

提交前运行：

```powershell
conda activate infernux
python scripts/maintenance/audit_project_sync.py C:\Projects\MyGame --tracked
```

审计会检查 sidecar 是否存在、GUID 是否重复、结构化资产是否含有本机绝对路径、包注册表
是否引用可移植路径，以及项目的忽略和属性策略是否完整。CI 对跨平台 Player fixture
执行同一审计。
`--tracked` 还会检查 Git 索引：资产与 sidecar 是否一起提交、是否错误追踪生成状态。
刚创建但尚未暂存的项目可以先省略这个参数。旧项目需采用引擎资源中的
`project_templates/project.gitignore.txt`、`project_templates/project.gitattributes.txt`，并保留
团队自己的附加规则；引擎不会在启动时覆盖已有的 Git 策略。

## 场景和结构化资产如何合并

场景、Prefab、材质、特效、粒子图和项目设置虽然是文本/JSON，但它们是有身份和引用语义
的结构化文档。逐行自动合并可能得到合法 JSON，却同时产生重复对象 ID、断开的组件引用或
错误的资源 GUID。因此项目 `.gitattributes` 对这些文件保留 LF，同时设置 `merge=binary`：
Git 遇到并发编辑会明确停下来，不会悄悄生成摇摇晃晃的中间结果。

推荐的协作流程是：

1. 一个文档在一个短时间窗口内只由一个人编辑；大型场景按目录或子场景拆分所有权。
2. 开始工作前先同步目标分支并在 Editor 中刷新资源；不要同时用文本编辑器和 Editor 写同一文档。
3. 出现结构化文档冲突时，保留一个完整版本，再在 Editor 中重新应用另一方的意图；不要
   选择 `union`、批量删除冲突标记或手工拼接对象数组。
4. 解决后重新打开场景/Prefab，检查 GUID 引用、Hierarchy、Inspector 和 Build Settings，
   再运行同步审计及对应回归测试。
5. 二进制模型、纹理和音频无法做语义合并。团队需要为这类文件建立明确的文件所有者；大型
   二进制仓库可额外启用 Git LFS，但不能把 LFS 指针当成引擎资源本身。

## Packages 的同步边界

`ProjectSettings/InxPlugins.json` 保存包的版本、文件 GUID、路径提示和来源；它是项目包
状态的锁定记录。安装包后，`Packages/<scope>/<name>/` 中的项目文件和 `.meta` 必须一起
提交。`Library/Plugins` 只是 Hub 的共享下载缓存，不能作为协作输入，也不能把其中的
机器绝对路径写入注册表。

远程/官方包通过注册表的精确 `reference + version` 获取；本地开发包应把源代码作为项目
内包提交，或使用团队约定的可访问仓库。不要只提交 `InxPlugins.json` 而漏掉 `Packages/`
内容，也不要只提交一个本机路径下的 `.inxpkg`。缺失已登记文件时包状态明确报错，Player
启动会拒绝不完整的包检出；提交前的同步审计同样会发现缺失文件，不能用另一版本补位。

安装包的 `package_path` 与 Hub 来源的 `cache_location` 相对共享缓存根目录保存，换用户后
按当前用户缓存重新定位。正常运行依赖已检出的 `Packages`，无需原始下载档案；更新需要
对应旧档案作为编辑基线，缺失时会报冲突而不会猜测。只有本地档案的包需要在新机器安装
同版本档案；远程来源应保留团队可访问的仓库/发布地址。

Python 依赖声明保存在共享注册表，普通项目 pip 依赖和已启用插件依赖会在新建运行时后
重新检查；项目 requirements 的版本约束与平台 marker 也会被执行。安装输出、Python
可执行文件路径和卸载前版本只保存在 `Library/Plugins/PythonEnvironment.json`，不随 Git
同步，避免把原作者的卸载基线套到协作者的环境。包锁文件不含安装时间、事务 ID或本机日志。

## 分支与提交

提交保持单一目的：引擎代码、项目格式、同步策略和测试放在同一个可审查变更中；生成的
`Library`、运行时和构建产物永远不进入提交。合并前至少执行项目同步审计、受影响的
`packaging/tests`/`python/test`，以及对应平台的 CMake/CTest 验证。远程分支改名或删除时，
先确认工作树和 CI 结果，再更新远端默认分支保护规则与协作者文档。
