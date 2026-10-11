# Hub installation diagnostics

## Logs

Open **Settings → Open Hub Logs**. `hub.log` records catalog requests and full
installation exceptions. Logs rotate at 4 MiB and retain three previous files.

Default locations:

- Windows: `%LOCALAPPDATA%\InfernuxHub\Logs\hub.log`
- Linux: `${XDG_DATA_HOME:-$HOME/.local/share}/InfernuxHub/Logs/hub.log`
- If `INFERNUX_DATA_ROOT` is set: `<INFERNUX_DATA_ROOT>/Logs/hub.log`

Include the failed operation, Hub version, OS/architecture, and the corresponding
exception when reporting a problem. Review logs for private local paths before
posting publicly. This logging is available in builds containing this change;
the original 0.4.1 Hub did not write these persistent diagnostics.

## No available engine versions

The catalog combines PyPI and GitHub Releases. A failed request is logged with
its source; when neither source nor a cached catalog is available, the dialog
shows an error rather than claiming no versions exist. Error text is selectable
and rendered literally, including `<urlopen error ...>`.
After correcting the connection, use **Retry** without restarting the Hub.

The exception distinguishes TLS certificate verification, proxy/connection
failures, HTTP failures, and filesystem permissions. Switching KDE/other desktop
environments is not a diagnosis of these failures. Do not disable TLS verification.
The install page also accepts a locally downloaded compatible engine wheel.

A fresh installation does not need the network for its first project: release
installers carry the engine wheel of their own release in
`<Hub app dir>/InfernuxHubData/engines`. On start, once the managed Python runtime
is installed, the Hub copies it into the `Engines` cache (validating it like a
local wheel import). Each bundled wheel is seeded only once and recorded in
`Engines/_bundled_engines_seeded.json`, so an engine removed from Installs is not
reinstalled on the next start. Locally built installers without a release wheel
skip this step and print a build warning.

## Windows selected a manylinux wheel / CPython ABI mismatch

An engine version and the Hub application version are different. Installing
engine `0.4.1-v3` does not update Hub itself. In particular, old Hubs using
`%USERPROFILE%\.infernux\versions` and `.runtime/python312` selected the first
GitHub wheel and the first cached ZIP without filtering OS or Python ABI.
With a release containing Linux and CPython 3.13 artifacts, that can produce
both `manylinux ... is not a supported wheel` and a 3.12/3.13 mismatch.

Close the old Hub, install and launch the current **Windows Hub installer**,
then install the engine's required Python runtime and the exact engine release
shown in Installs. A manually downloaded Windows `cp313` wheel is not enough
to fix an old Hub that still creates Python 3.12 projects. For a failed new
project creation, retry creation from the updated Hub. Do not delete an existing
project's Assets or ProjectSettings to repair a runtime cache.

The current wheel cache is under the shared resource directory shown in Hub
settings: `Engines/<package-version>/<complete-wheel-filename>`. The catalog
file `_releases_cache.json` contains all platforms; Linux entries in that JSON
are normal. Each selection filters the running OS/architecture and exact release
again. Downloads verify wheel metadata and published size/SHA-256 when supplied;
cached catalog digests are also used on offline reads. Downloads and catalog
updates are published atomically. Project installation separately checks the
wheel's ABI against both the project pin and the actual project interpreter.

For a new report, collect the **Hub version and executable location**, engine
release, full wheel filename, project Python path and Hub log. Keep the suspect
cache long enough to compare it against the release digest. A CDN problem cannot
be diagnosed from the engine version alone: engine wheels come from PyPI/GitHub,
while the Hub installer/update archive can come from the Cloudflare mirror.

Download/runtime compatibility is not the same as the project-template contract.
The Hub locates each template by its unique filename inside the selected wheel,
without assuming its directory. Missing required or ambiguous templates are
errors. The engine package's exact spelling comes from the installed wheel's
metadata in the child process, so both `infernux` and historical `Infernux` entry
points work without importing an engine into Hub. Wheels without authored scene
templates let their own Editor create the initial scene; Hub never injects newer
component serialization into an older engine. Test project creation and actual
launch for each supported wheel, not just download/platform selection.

## CMake versions

Installing an engine wheel and creating a Hub project do not require CMake.
For source builds, the minimum is CMake 3.25 because the checked-in schema-6
presets include workflows. There is no maximum-version check; CMake 4.x is not
excluded. In `cmake_minimum_required(VERSION min...policy_max)`, the second number
controls policy behavior, not the newest allowed executable. Do not lower the
minimum or force every dependency to the newest policy merely to hide a warning.

Windows source builds intentionally leave the generator open in the
`windows-msvc-*` presets. CMake selects the newest compatible Visual Studio
generator available on the machine; if that generator cannot be created, use a
CMake build that lists it in `cmake --help` or set `CMAKE_GENERATOR` to another
listed generator and configure again. The Hub's MSVC discovery pairs `vswhere`
with the selected generator, so side-by-side Visual Studio installations do not
silently mix compiler environments.

The original Linux 0.4.1 Hub bundled OpenSSL with a build-machine conda certificate
path. On a clean machine this can cause `CERTIFICATE_VERIFY_FAILED` for both the
engine catalog and Blender downloads. The rebuilt Hub uses the distribution's
system CA bundle instead. Explicit `SSL_CERT_FILE` / `SSL_CERT_DIR` settings remain
authoritative. Install the distribution's `ca-certificates` package if absent.

For the original Linux Hub, launching it with
`SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt "/path/to/Infernux Hub"` uses
Ubuntu/Kubuntu's system trust store without disabling certificate verification.

## Blender authoring support

This is an optional Blender installation used by the Editor to convert `.blend`
assets, not an engine component or Player dependency. The importer currently
supports Blender 5.2; the Hub-managed release is 5.2.2.

The Editor selects its executable in this order:

1. Explicit Blender path in Editor preferences.
2. Existing Hub-managed executable supplied by `INFERNUX_BLENDER_EXECUTABLE`.
3. The operating system's default `.blend` application.

Windows uses the Shell association API, including the user's default application.
Linux queries `xdg-mime` for `application/x-blender` and resolves the desktop entry
using XDG data-directory precedence. Quoted native executable paths are supported.
Launchers requiring extra arguments (for example `flatpak run ...`) cannot be used
as a native executable: select a native Blender 5.2 executable in preferences.
Clearing the explicit preference restores automatic selection. An explicit invalid
selection is not silently replaced; the import error must be corrected.

Installation distinguishes download from extraction. Windows extraction uses
extended paths, without requiring a system-wide long-path policy change. A verified
download is retained if installation fails, and cleanup errors do not replace the
original failure. Its path is recorded in the log. Downloads live under the shared
resources directory shown in Hub settings, in `Cache/Downloads/Blender`.

For an immediate workaround with an older Hub, extract the official Blender 5.2
archive into a writable, short directory and set its `blender.exe` (Windows) or
native `blender` executable (Linux) in Editor preferences. If extraction still
fails, collect the exact filename/error and check destination permissions and free
space; do not infer the cause solely from the desktop environment.

Linux and Windows installation reports must be investigated separately. A Linux
TLS or HTTP error happens before extraction; Windows reporting a completed download
and failed extraction needs the filesystem exception. Blender's download server
may also return HTTP 403 with a browser challenge on some networks. Hub cannot
complete an interactive browser challenge: use the official browser download and
select the extracted native executable. This is distinct from a certificate error.
