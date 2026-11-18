# CMake layout

The top-level `CMakeLists.txt` is the only project entry point. It loads the
build policy, dependency layout, source lists, native targets, packaging, and
developer tools in that order.

## Dependency ownership

`external/` contains source dependencies that are linked into the editor,
runtime, tests, or wheel. The submodule revision is pinned by the superproject
gitlink. `external/CMakeLists.txt` configures those dependencies and exposes
their CMake targets; it does not publish plugin packages.

Prepare a checkout with `git submodule update --init --recursive` before
taking it offline, or copy an already populated checkout. Gitlinks select the
exact sources. CMake compiles those sources locally, including Slang and its
host generators. Native and Web bindings share the pinned pybind11 headers.
Asset import and screenshot encoding share the pinned zlib target used by
the Web Player. Header-only libraries expose include targets.

The operating system and its SDK provide Vulkan, the C/C++ runtime and the
native window system. CPython and its Python packages come from the prepared
Python environment or platform kit. Prepare these tools and the checkout once
before building offline. The Hub Android kit includes the Gradle artifacts
needed for the first APK or App Bundle export.

`external/plugins/` contains official plugin checkouts. They are packaging
inputs, not editor link dependencies. Its CMake target builds the catalog and
the bundled MCP package into the build tree. Platform plugins own their Player
release workflows. A platform native CMake file is included only by the
platform build that needs it (for example, the Android export path).

`cmake/InfernuxDependencyLayout.cmake` publishes the engine and plugin source
roots and checks the files required by each build. Each platform plugin owns
its private libraries under its own `external/`. The Web plugin uses the
pinned local Emdawnwebgpu port to build its Player.

Set `INFERNUX_REFRESH_EDITOR_PLUGIN_RESOURCES=OFF` for a strictly out-of-source
package build. The default remains `ON` for editor development, where the
freshly generated MCP package and catalog are also refreshed in the checkout's
live editor resources.

## Output ownership

Generated files belong under the build tree (`official-plugins` and
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
| `package_cpu_jit_dependency` | Build the pinned llvmlite fork separately. Windows x64 links only the native LLVM target for the Player. |

Do not add a plugin with `add_subdirectory()` merely because it is a Git
submodule. Add a target only when the plugin's platform export actually needs
to participate in that build.
