#!/usr/bin/env bash
set -euo pipefail

EMSCRIPTEN_VERSION="4.0.10"
EMSDK_REVISION="e5bd3d0874e302a18f13c5b41f5bacf9a40c8e59"
CPYTHON_VERSION="3.13.15"
CPYTHON_SHA256="1e66a7945a48390ee4c2a4268a0e4185884059a13c4aab6d148aa208deea4a76"
CPYTHON_URL="https://www.python.org/ftp/python/${CPYTHON_VERSION}/Python-${CPYTHON_VERSION}.tar.xz"

usage() {
    echo "Usage: $0 <toolchain-root>" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "The Web toolchain setup requires a Linux host or WSL2." >&2
    exit 2
fi
for command in cmake curl git make ninja python3 sha256sum tar /usr/bin/pkg-config; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command is unavailable: $command" >&2
        exit 2
    fi
done
# Conda's wrapper appends native libraries to Emscripten's target-only search.
# The system implementation respects the SDK's PKG_CONFIG_LIBDIR boundary.
export PKG_CONFIG=/usr/bin/pkg-config

toolchain_root="$(realpath -m "$1")"
downloads="$toolchain_root/downloads"
sources="$toolchain_root/src"
builds="$toolchain_root/build"
emsdk_root="$toolchain_root/emsdk"
cpython_root="$sources/Python-${CPYTHON_VERSION}"
mkdir -p "$downloads" "$sources" "$builds"

fetch_and_verify() {
    local url="$1"
    local sha256="$2"
    local destination="$3"
    if [[ ! -f "$destination" ]]; then
        curl -Lf --retry 5 --retry-all-errors --output "$destination" "$url"
    fi
    echo "$sha256  $destination" | sha256sum --check --status || {
        echo "Downloaded file does not match its pinned SHA-256: $destination" >&2
        exit 1
    }
}

if [[ ! -d "$emsdk_root/.git" ]]; then
    git clone https://github.com/emscripten-core/emsdk.git "$emsdk_root"
fi
if [[ "$(git -C "$emsdk_root" rev-parse HEAD)" != "$EMSDK_REVISION" ]]; then
    if [[ -n "$(git -C "$emsdk_root" status --porcelain)" ]]; then
        echo "The Emscripten SDK checkout contains local changes: $emsdk_root" >&2
        exit 1
    fi
    git -C "$emsdk_root" fetch origin "$EMSDK_REVISION"
    git -C "$emsdk_root" checkout --detach "$EMSDK_REVISION"
fi
"$emsdk_root/emsdk" install "$EMSCRIPTEN_VERSION"
"$emsdk_root/emsdk" activate "$EMSCRIPTEN_VERSION"
export EMSDK_QUIET=1
# shellcheck disable=SC1091
source "$emsdk_root/emsdk_env.sh"
emscripten_root="$emsdk_root/upstream/emscripten"
binaryen_root="$emsdk_root/upstream"
node_path="$(command -v node)"
if [[ ! -x "$emscripten_root/emconfigure" || ! -x "$binaryen_root/bin/wasm-opt" || ! -x "$node_path" ]]; then
    echo "The activated Emscripten SDK is incomplete: $emsdk_root" >&2
    exit 1
fi
cpython_em_config="$builds/cpython-emscripten-config.py"
python3 - "$cpython_em_config" "$emscripten_root" "$binaryen_root" "$node_path" <<'PY'
import sys
from pathlib import Path

config_path, emscripten_root, binaryen_root, node_path = map(Path, sys.argv[1:])
config_path.write_text(
    f"LLVM_ROOT = {str(binaryen_root / 'bin')!r}\n"
    f"EMSCRIPTEN_ROOT = {str(emscripten_root)!r}\n"
    f"BINARYEN_ROOT = {str(binaryen_root)!r}\n"
    f"NODE_JS = {str(node_path)!r}\n",
    encoding="utf-8",
)
PY
export EM_CONFIG="$cpython_em_config"
# Populate the same system ports used to compile and link the production Player.
embuilder build sysroot zlib bzip2 sqlite3

cpython_archive="$downloads/Python-${CPYTHON_VERSION}.tar.xz"
fetch_and_verify "$CPYTHON_URL" "$CPYTHON_SHA256" "$cpython_archive"
if [[ ! -f "$cpython_root/Tools/wasm/wasm_build.py" ]]; then
    tar -xf "$cpython_archive" -C "$sources"
fi
cpython_wasm_config="$cpython_root/Tools/wasm/config.site-wasm32-emscripten"
cpython_wasm_config_changed=0
if ! grep -q '^ac_cv_func_pthread_kill=no$' "$cpython_wasm_config"; then
    printf '\n# The browser runtime is single-threaded.\nac_cv_func_pthread_kill=no\n' \
        >> "$cpython_wasm_config"
    cpython_wasm_config_changed=1
fi
(
    cd "$cpython_root"
    if [[ "$cpython_wasm_config_changed" == "1" ]]; then
        python3 Tools/wasm/wasm_build.py emscripten-browser cleanall
    fi
    python3 Tools/wasm/wasm_build.py emscripten-browser
)

wasm_root="$cpython_root/builddir/emscripten-browser"
for path in \
    "$emsdk_root/upstream/emscripten/emcc" \
    "$wasm_root/python.wasm" \
    "$wasm_root/python.data" \
    "$wasm_root/libpython3.13.a"; do
    if [[ ! -f "$path" ]]; then
        echo "Web toolchain output is missing: $path" >&2
        exit 1
    fi
done

python3 - "$toolchain_root/infernux-web-toolchain.json" \
    "$emsdk_root" "$cpython_root" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    manifest_path,
    emsdk_root,
    cpython_root,
) = map(Path, sys.argv[1:])
files = {
    "emcc": emsdk_root / "upstream/emscripten/emcc",
    "python_wasm": cpython_root / "builddir/emscripten-browser/python.wasm",
    "python_data": cpython_root / "builddir/emscripten-browser/python.data",
    "python_library": cpython_root / "builddir/emscripten-browser/libpython3.13.a",
}
payload = {
    "schema": "infernux.web_toolchain",
    "kind": "infernux-web-toolchain",
    "emscripten": {"version": "4.0.10", "revision": "e5bd3d0874e302a18f13c5b41f5bacf9a40c8e59"},
    "cpython": {
        "version": "3.13.15",
        "source_url": "https://www.python.org/ftp/python/3.13.15/Python-3.13.15.tar.xz",
        "source_sha256": "1e66a7945a48390ee4c2a4268a0e4185884059a13c4aab6d148aa208deea4a76",
    },
    "files": {},
}
for name, path in files.items():
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    payload["files"][name] = {"sha256": digest.hexdigest(), "size": path.stat().st_size}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, manifest_path)
PY

echo "Web toolchain ready: $toolchain_root"
echo "  INFERNUX_EMSDK_ROOT=$emsdk_root"
echo "  INFERNUX_WEB_CPYTHON_ROOT=$cpython_root"
