# 代码签名策略（Code signing policy）

本策略适用于从
[`ChenlizheMe/Infernux`](https://github.com/ChenlizheMe/Infernux)
仓库发布的 Infernux 官方制品，不覆盖社区插件、第三方包、下游构建或用户使用引擎导出的游戏。
如中英文内容存在差异，以[英文策略](CODE_SIGNING_POLICY.md)为准。

**Free code signing provided by [SignPath.io](https://signpath.io/), certificate by [SignPath Foundation](https://signpath.org/).**

## 团队职责

- 提交者及审核者（Committers and reviewers）：[Lizhe Chen（`@ChenlizheMe`）](https://github.com/ChenlizheMe)
- 签名批准者：[Lizhe Chen（`@ChenlizheMe`）](https://github.com/ChenlizheMe)

没有提交权限的人所提交的改动，必须先由维护者审核才能合并。每一个发布签名请求都需要在
SignPath 中单独进行人工批准。

## 可以签名的内容

只有由已关联、可验证来源的 GitHub Actions 流程从本仓库构建的 Infernux 官方可执行文件和安装器，
才可以获得发布签名。构建定义与源码一起接受版本控制。发布制品必须将产品标识为 Infernux，
并在一次构建中使用一致的产品版本。

Infernux 软件包可以携带上游开源依赖，但不会把这些依赖重新签名成 Infernux 自己生产的软件。
测试签名不会作为受信任的发布签名对外分发。

## 联网与隐私说明

引擎和编辑器默认不会向 Infernux 服务上传项目文件、场景、脚本、资产或玩法内容。

自动检查更新默认关闭。用户启用该选项后，安装版 InfernuxHub 会在启动时向
`infernux-engine.com` 请求公开的 Hub 更新目录与发布通知；手动检查更新会请求更新目录。用户打开引擎安装流程时，
Hub 还会向 PyPI 和 GitHub 请求公开的 Release 元数据。这些服务会收到普通的
HTTPS 请求元数据，例如客户端 IP 地址、User-Agent 和所请求的网址，但请求中不包含项目内容。
相关元数据分别适用 [GitHub 隐私声明](https://docs.github.com/zh/site-policy/privacy-policies/github-general-privacy-statement)
和 [PyPI 隐私声明](https://policies.python.org/pypi.org/Privacy-Notice/)。

只有用户选择相应的安装或更新操作后，Hub 才会下载引擎、构建支持和插件。打开在线文档、
社区链接或 Release 页面时，会连接用户所选择的网站。能够联网的插件、MCP 连接、导出的游戏
及其他第三方软件遵循各自的配置与隐私策略。

## 安装与卸载

安装 InfernuxHub 是用户显式发起的操作。安装器会创建应用文件和当前用户的卸载入口；
只有用户安装对应内容时，Hub 才会创建托管的引擎及构建支持数据。Windows 用户可以从
**设置 > 应用 > 已安装的应用**卸载 InfernuxHub；Windows 和 Linux 也可以使用 `--uninstall`
启动 Hub。卸载 Hub 会保留项目与共享项目记录。

## 报告问题

如发现 Infernux 签名被滥用或本策略被违反，请联系
[`chenlizheme@outlook.com`](mailto:chenlizheme@outlook.com)。涉及安全的内容可以通过仓库的
[私密漏洞报告](https://github.com/ChenlizheMe/Infernux/security/advisories/new)提交。
与 SignPath Foundation 证书有关的问题也可以发送至
[`support@signpath.io`](mailto:support@signpath.io)。

最后更新：2026-09-12。
