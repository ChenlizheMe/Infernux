# Code signing policy

This policy applies to official Infernux release artifacts published from the
[`ChenlizheMe/Infernux`](https://github.com/ChenlizheMe/Infernux) repository.
It does not cover community plugins, third-party packages, downstream builds,
or games exported by engine users.

**Free code signing provided by [SignPath.io](https://signpath.io/), certificate by [SignPath Foundation](https://signpath.org/).**

## Team roles

- Committers and reviewers: [Lizhe Chen (`@ChenlizheMe`)](https://github.com/ChenlizheMe)
- Approvers: [Lizhe Chen (`@ChenlizheMe`)](https://github.com/ChenlizheMe)

Changes submitted by people without commit access require maintainer review
before they are merged. Signing approval is a separate, manual action performed
for each release request in SignPath.

## What may be signed

Only official Infernux executables and installers built from this repository by
the linked, origin-verifiable GitHub Actions build may receive the release
signature. Build definitions are versioned alongside the source. Release
artifacts must identify the product as Infernux and use one consistent product
version throughout the build.

Dependencies from upstream open-source projects may be distributed with an
Infernux package, but they are not re-signed as if they were produced by the
Infernux project. Test signatures are not published as trusted release
signatures.

## Network and privacy disclosure

The engine and editor do not upload project files, scenes, scripts, assets, or
gameplay content to Infernux services by default.

Automatic update checks are disabled by default. If the user enables them, the
installed InfernuxHub requests the public Hub update catalog and release notices
from `infernux-engine.com` at startup. A manual update check requests the catalog.
The Hub requests public release metadata from PyPI and GitHub when the user
opens the engine installation workflow. These services
receive ordinary HTTPS request metadata, such as the client's IP address, user
agent, and requested URL; no project content is included. Their handling of
that metadata is governed by the [GitHub Privacy Statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement)
and the [PyPI privacy notice](https://policies.python.org/pypi.org/Privacy-Notice/).

Downloads of engines, build support, and plugins occur only after the user
chooses the corresponding install or update operation. Opening online
documentation, community links, or release pages contacts the selected site.
Network-capable plugins, MCP connections, exported games, and other third-party
software operate under their own configuration and privacy policies.

## Installation and removal

Installing InfernuxHub is an explicit user action. The installer creates the
application files and a per-user uninstall entry; the Hub creates managed engine
and build-support data only when the user installs those items. On Windows,
remove InfernuxHub through **Settings > Apps > Installed apps**. The Hub can also
be started with `--uninstall` on Windows or Linux. Uninstalling the Hub preserves
projects and shared project records.

## Reporting concerns

Report suspected abuse of an Infernux signature or a policy violation to
[`chenlizheme@outlook.com`](mailto:chenlizheme@outlook.com). Security-sensitive
reports may use the repository's
[private vulnerability reporting](https://github.com/ChenlizheMe/Infernux/security/advisories/new).
Reports concerning a SignPath Foundation certificate may also be sent to
[`support@signpath.io`](mailto:support@signpath.io).

Last updated: 2026-09-12.
