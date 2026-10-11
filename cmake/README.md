# CMake layout

The top-level `CMakeLists.txt` is the only project entry point. It loads the
build policy, dependency layout, source lists, native targets, packaging, and
developer tools in that order.

## Dependency ownership

`external/` contains source dependencies that are linked into the editor,
runtime, tests, or wheel. The submodule revision is pinned by the superproject
gitlink. `external/CMakeLists.txt` configures those dependencies and exposes
their CMake targets; it does not publish plugin packages.

The engine-maintained forks are named `<library>_for_infernux` and use the
single `infernux-support` branch. The branch name controls intentional updates;
normal builds still use the gitlink pinned by the main repository.

`external/plugins/` contains official plugin checkouts. They are packaging
inputs, not editor link dependencies. Its CMake target builds the catalog and
the bundled MCP package into the build tree. Platform plugins own their Player
release workflows. A platform native CMake file is included only by the
platform build that needs it (for example, the Android export path).

`cmake/InfernuxDependencyLayout.cmake` is the canonical path and ownership
map. New CMake code should use its dependency helper instead of reconstructing
`external/...` or `external/plugins/...` paths. It also fails early when a
required submodule is not initialized.

Set `INFERNUX_REFRESH_EDITOR_PLUGIN_RESOURCES=OFF` for a strictly out-of-source
package build. The default remains `ON` for editor development, where the
freshly generated MCP package and catalog are also refreshed in the checkout's
live editor resources.

## Output ownership

Generated files belong under the build tree (`official-plugins`, `_deps`, and
`out/stage`). The wheel install component consumes those staged outputs. The
editor source checkout may receive the bundled MCP/catalog refresh as an
explicit convenience side effect of the official-plugin target; platform
release artifacts remain in their plugin repositories and are not built by the
ordinary editor target.

Useful target groups are:

| Target | Responsibility |
| --- | --- |
| `infernux_official_plugins` | Build catalog metadata and the bundled MCP package (explicit/package builds) |
| `stage_python_package` | Assemble the wheel source tree |
| `package_python` | Build and verify the wheel |
| `prebuild_player_runtime` | Publish a platform Player payload when explicitly requested |
| `package_cpu_jit_dependency` | Build the pinned llvmlite fork separately |

Do not add a plugin with `add_subdirectory()` merely because it is a Git
submodule. Add a target only when the plugin's platform export actually needs
to participate in that build.
