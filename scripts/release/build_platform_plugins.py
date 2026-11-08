"""Build the four engine-owned Player plugins after the release identity is final.

This stdlib-only entry point is shared by CI and local release batches. Native
toolchains remain platform-specific; collection and publication are host-neutral.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = runpy.run_path(str(ROOT / "python/infernux/version.py"))
RELEASE = IDENTITY["ENGINE_RELEASE"]
VERSION = IDENTITY["ENGINE_VERSION"]
BUILD_NUMBER = IDENTITY["ENGINE_BUILD_NUMBER"]
PACKAGE_VERSION = VERSION if BUILD_NUMBER == 1 else f"{VERSION}.post{BUILD_NUMBER}"
BASE_REGISTRY = ROOT / "python/infernux/resources/official_packages/official-registry.json"
PLATFORM_DECLARATIONS = ROOT / "python/infernux/resources/platform-players.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


_declarations = read(PLATFORM_DECLARATIONS)
PYTHON_ABI = _declarations["python_abi"]
PLATFORM_PLAYERS = {item["id"]: item for item in _declarations["platforms"]}
PLATFORMS = tuple(PLATFORM_PLAYERS)
PLATFORM_REFERENCES = frozenset(item["reference"] for item in PLATFORM_PLAYERS.values())


def catalog():
    document = read(BASE_REGISTRY)
    entries = {item["reference"]: item for item in document["packages"]}
    for descriptor in PLATFORM_PLAYERS.values():
        reference = descriptor["reference"]
        item = entries.get(reference)
        if item is None:
            metadata = read(ROOT / descriptor["repository_path"] / "package/inx_package.json")
            item = {"reference": reference, "name": metadata["name"], "category": "platform_build",
                    "repository": descriptor["repository"]}
            for field in ("intro", "intros", "dependencies"):
                if field in metadata:
                    item[field] = metadata[field]
            document["packages"].append(item)
        artifact = reference.replace("/", ".") + ".inxpkg"
        item.update(version=PACKAGE_VERSION, engine="==" + PACKAGE_VERSION, engine_release=RELEASE,
                    artifact=artifact, targets=descriptor["targets"], release_tag="v" + PACKAGE_VERSION)
        item["source"] = {
            "type": "url",
            "location": f"https://downloads.infernux-engine.com/platform-plugins/{RELEASE}/{artifact}",
        }
    return document


def _git_revision(directory):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()


def _validate_payload(platform, payload, tools=None, *, source_revision=None):
    descriptor = PLATFORM_PLAYERS[platform]
    payload_format = descriptor["payload_format"]
    source_revision = source_revision or _git_revision(ROOT)
    roots = [payload / abi for abi in descriptor["abis"]] if payload_format == "android" else [payload]
    for root in roots:
        document = read(root / "Player.inxmanifest")
        expected = {"engine_release": RELEASE, "engine_source_revision": source_revision,
                    "python_abi": PYTHON_ABI, "platform": descriptor["native_platform"]}
        for key, value in expected.items():
            if document.get(key) != value:
                raise ValueError(f"{root}: {key} must be {value}, found {document.get(key)!r}")
        engine_field = "engine" if payload_format == "web" else "engine_version"
        if document.get(engine_field) != VERSION:
            raise ValueError(f"{root}: engine version does not match {RELEASE}")
        if payload_format == "desktop":
            machine = str(document.get("machine", "")).casefold().replace("amd64", "x86_64")
            if machine != descriptor["machine"]:
                raise ValueError(f"{root}: desktop machine must be {descriptor['machine']}, found {machine!r}")
            required = ("Runtime.inxrt", "Parallel.inxmod", "InfernuxPlayerHost.exe" if platform == "windows" else "InfernuxPlayerHost")
        elif payload_format == "android":
            if document.get("abi") != root.name or document.get("configuration") != "Release":
                raise ValueError(f"{root}: Android ABI/Release mismatch")
            required = ["jniLibs/" + name for name in document["native_libraries"]]
            if not (payload / "java/org/libsdl/app/SDLActivity.java").is_file():
                raise ValueError("Android Player is missing its SDL Java host")
        elif payload_format == "web":
            if tools is None:
                raise ValueError("Web Player packaging requires Windows and Linux shader tools")
            contract_path = ROOT / descriptor["repository_path"] / "package/editor/infernux_web/native_payload.py"
            contract = runpy.run_path(str(contract_path))
            contract["inspect_player_payload"](payload, engine=VERSION, engine_release=RELEASE,
                                               source_revision=source_revision)
            for host in contract["HOSTS"]:
                contract["inspect_shader_tools"](tools, host=host)
            required = contract["PLAYER_FILES"]
        else:
            required = descriptor["required_files"]
        for name in required:
            if not (root / name).is_file() or not (root / name).stat().st_size:
                raise FileNotFoundError(root / name)


def package(platform, payload, output, *, tools=None):
    payload = payload.resolve()
    descriptor = PLATFORM_PLAYERS[platform]
    repository = ROOT / descriptor["repository_path"]
    revision = _git_revision(ROOT)
    plugin_revision = _git_revision(repository)
    _validate_payload(platform, payload, tools, source_revision=revision)
    reference = descriptor["reference"]
    stem = reference.replace("/", ".")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="infernux-platform-package-") as folder:
        stage = Path(folder) / "package"
        shutil.copytree(repository / "package", stage,
                        ignore=shutil.ignore_patterns("player", "tools", "__pycache__"))
        plugin = next(path for path in (stage / "editor").iterdir() if (path / "__init__.py").is_file())
        shutil.copytree(payload, plugin / "player", ignore=shutil.ignore_patterns("*.meta", "__pycache__"))
        if tools is not None:
            shutil.copytree(tools, plugin / "tools", ignore=shutil.ignore_patterns("*.meta"))
        metadata = read(stage / "inx_package.json")
        metadata.update(version=PACKAGE_VERSION, engine="==" + PACKAGE_VERSION)
        write(stage / "inx_package.json", metadata)
        spec = importlib.util.spec_from_file_location("_platform_package_builder", repository / "package.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        artifact = builder.build(output / (stem + ".inxpkg"), package_root=stage)
    document = {"$schema": "infernux.plugin_release", "reference": reference,
        "version": PACKAGE_VERSION, "engine": "==" + PACKAGE_VERSION, "engine_release": RELEASE,
        "source_revision": revision, "plugin_source_revision": plugin_revision,
        "python_abi": PYTHON_ABI, "release_tag": "v" + PACKAGE_VERSION,
        "artifact": {"name": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                     "bytes": artifact.stat().st_size}}
    write(output / (stem + ".release.json"), document)
    print(artifact)
    return artifact


def run(command):
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def build(platform, args):
    build_root = (args.build_root / platform).resolve()
    payload = build_root / "publish/player"
    if platform in {"windows", "linux"}:
        if ((platform == "windows" and sys.platform != "win32")
                or (platform == "linux" and not sys.platform.startswith("linux"))):
            raise ValueError(f"Run the {platform} native build on its matching host; CI collects all four hosts")
        preset = "windows-msvc-release" if platform == "windows" else "linux-clang-release"
        run(["cmake", "--preset", preset, "-B", build_root,
             f"-DPython3_EXECUTABLE={sys.executable}",
             f"-DINFERNUX_PYTHON_SYNC_DIR={build_root / 'python-sync'}",
             f"-DINFERNUX_PLATFORM_PLAYER_OUTPUT_DIR={payload}", "-DINFERNUX_PLAYER_PACKAGE_PLUGIN=OFF"])
        run(["cmake", "--build", build_root, "--config", "Release", "--target", "prebuild_player_runtime", "--parallel", "4"])
    elif platform == "android":
        for abi, suffix in (("arm64-v8a", "ARM64"), ("x86_64", "X86_64")):
            prefix = os.environ["INFERNUX_ANDROID_PYTHON_PREFIX_" + suffix]
            run([sys.executable, ROOT / "external/plugins/infernux_android/native/build.py",
                 "--sdk", os.environ["ANDROID_SDK_ROOT"], "--python-prefix", prefix,
                 "--abi", abi, "--build-root", build_root / abi, "--player-output", payload])
    else:
        run([sys.executable, ROOT / "external/plugins/infernux_web/native/build.py",
             "--emsdk", os.environ["INFERNUX_EMSDK_ROOT"],
             "--cpython-source", os.environ["INFERNUX_WEB_CPYTHON_ROOT"],
             "--python-runtime-manifest", os.environ["INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST"],
             "--numpy-payload-root", os.environ["INFERNUX_WEB_NUMPY_PAYLOAD_ROOT"],
             "--host-python", sys.executable, "--build-root", build_root, "--player-output", payload])
    return package(platform, payload, args.output, tools=args.tools if platform == "web" else None)


def assemble(directory, *, sdk_wheels=(), source_revision=None):
    revisions = set()
    for platform in PLATFORMS:
        reference = PLATFORM_PLAYERS[platform]["reference"]
        stem = reference.replace("/", ".")
        document = read(directory / (stem + ".release.json"))
        artifact = directory / (stem + ".inxpkg")
        if (document["engine_release"] != RELEASE or document["version"] != PACKAGE_VERSION
                or document["python_abi"] != PYTHON_ABI or document["reference"] != reference
                or document["engine"] != "==" + PACKAGE_VERSION or document["artifact"]["name"] != artifact.name):
            raise ValueError(f"{platform} package does not belong to {RELEASE}")
        if (hashlib.sha256(artifact.read_bytes()).hexdigest() != document["artifact"]["sha256"]
                or artifact.stat().st_size != document["artifact"]["bytes"]):
            raise ValueError(f"{platform} package digest mismatch")
        revisions.add(document["source_revision"])
    if len(revisions) != 1 or not all(revisions):
        raise ValueError("Platform plugins must come from the same engine source revision")
    revision = revisions.pop()
    if source_revision is not None and revision != source_revision:
        raise ValueError(f"Platform plugins came from {revision}, expected engine source revision {source_revision}")
    for wheel in sdk_wheels:
        try:
            with zipfile.ZipFile(wheel) as archive:
                provenance = json.loads(archive.read("infernux/resources/player-build-source.json"))
        except (KeyError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError(f"SDK wheel and platform plugins must share valid provenance: {wheel}") from exc
        if (not isinstance(provenance, dict) or provenance.get("engine_release") != RELEASE
                or provenance.get("revision") != revision or provenance.get("dirty") is not False):
            raise ValueError(f"SDK wheel and platform plugins must come from the same clean release source: {wheel}")
    full = catalog()
    write(directory / "official-registry.json", full)
    scoped = copy.deepcopy(full)
    scoped["packages"] = [item for item in full["packages"] if item["reference"] in PLATFORM_REFERENCES]
    write(directory / "registry.json", scoped)
    print(f"Validated complete platform plugin set: {RELEASE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "package", "assemble", "catalog"))
    parser.add_argument("--targets", nargs="+", choices=PLATFORMS)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--tools", type=Path)
    parser.add_argument("--build-root", type=Path, default=ROOT / "out/build/platform-plugins" / RELEASE)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/platform-plugins" / RELEASE)
    parser.add_argument("--source-revision")
    parser.add_argument("--sdk-wheels", type=Path, nargs="+", default=())
    args = parser.parse_args()
    if args.operation == "catalog":
        write(args.output, catalog())
    elif args.operation == "assemble":
        assemble(args.output, sdk_wheels=args.sdk_wheels, source_revision=args.source_revision)
    else:
        if not args.targets:
            parser.error("--targets is required")
        if args.operation == "package" and (len(args.targets) != 1 or args.payload is None):
            parser.error("package requires one target and --payload")
        for platform in args.targets:
            if args.operation == "build":
                build(platform, args)
            else:
                package(platform, args.payload, args.output, tools=args.tools)


if __name__ == "__main__":
    main()
