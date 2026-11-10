"""Cold native-code production from the existing proved CPU range entries."""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from infernux.cpu_programs import library_directory, library_filename, native_cpu_target, stage_library


def _object(program, symbol):
    from llvmlite import binding as llvm

    entry = program._native_entry()
    module = llvm.parse_assembly(entry.inspect_llvm())
    module.get_function(entry.native_name).name = symbol
    for function in module.functions:
        if not function.is_declaration and function.name != symbol:
            function.linkage = "internal"
    for value in module.global_variables:
        if not value.is_declaration and not value.name.startswith("llvm."):
            value.linkage = "internal"
    machine = llvm.Target.from_triple(module.triple).create_target_machine(
        cpu="x86-64", features="", reloc="pic", codemodel="small")
    module.data_layout = str(machine.target_data)
    # Optimize the complete standalone call graph after internalization. Merely
    # removing unused globals leaves exception adapters whose callees cannot
    # raise, retaining Python references in otherwise self-contained row code.
    with llvm.create_pipeline_tuning_options(speed_level=2) as tuning:
        with llvm.create_pass_builder(machine, tuning) as builder:
            with builder.getModulePassManager() as passes:
                passes.run(module, builder)
    module.verify()
    # A JIT pointer or Python/NRT dependency cannot become a standalone entry.
    if re.search(r"\binttoptr\b", str(module)) or any(value.is_declaration for value in module.global_variables):
        raise TypeError("CPU program retains a process-local reference")
    # The native linker resolves the actual CRT dependencies. Do not keep a
    # second, partial math-symbol whitelist that rejects valid emitted code.
    return machine.emit_object(module)


def _link(output, target, objects, symbols):
    environment, creationflags = os.environ, 0
    if target == "windows-x86_64":
        from infernux.engine.nuitka_builder import _ensure_windows_msvc_environment

        # Toolchain discovery uses the current build's temporary directory.
        scratch = str(output.parent)
        environment = _ensure_windows_msvc_environment({**os.environ, "TEMP": scratch, "TMP": scratch})
        linker = shutil.which("link.exe", path=environment["PATH"])
        if linker is None:
            raise RuntimeError("native CPU production requires the configured MSVC linker")
        exports = ["/EXPORT:" + symbol for symbol in symbols]
        arguments = [linker, "/nologo", "/DLL", "/NOENTRY", "/NODEFAULTLIB", "/MACHINE:X64", "/Brepro",
                     "/OUT:" + str(output), *exports, *objects, "ucrt.lib", "libvcruntime.lib", "libcmt.lib"]
        creationflags = subprocess.CREATE_NO_WINDOW
    else:
        linker = shutil.which("cc")
        if linker is None:
            raise RuntimeError("native CPU production requires a Linux C toolchain")
        arguments = [linker, "-shared", "-nostartfiles", "-Wl,-z,defs", "-o", str(output), *objects, "-lm"]
    completed = subprocess.run(arguments, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               creationflags=creationflags)
    if completed.returncode:
        raise RuntimeError("native CPU link failed:\n" + completed.stdout.decode("utf-8", errors="replace"))


def build(programs, directory):
    """Build the declared whole/continuation closure for the current host."""
    from infernux.nn._primitives import program_family
    from infernux.engine.project_context import get_project_root
    from infernux._compiler.cache import compiler_cache_root

    target = native_cpu_target()
    if target is None:
        raise RuntimeError("native CPU production is unavailable for this host")
    directory = library_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)
    project = get_project_root()
    scratch_root = directory if project is None else compiler_cache_root() / "CPU" / "Native"
    scratch_root.mkdir(parents=True, exist_ok=True)
    entries, recipes = {}, {}
    for program in programs:
        for variant in program_family(program):
            identity = variant.cost_identity
            entries[identity] = variant
        recipes[program.cost_identity] = dict(recipe=program._nn_recipe, whole=program.cost_identity,
            variants={kind: value.cost_identity for kind, value in program._inner_variants.items()})
    record = dict(device="cpu", entries=[], recipes=list(recipes.values()))
    with ExitStack() as cleanup:
        scratch = Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix="nn-build-", dir=scratch_root)))
        if target == "windows-x86_64":
            from infernux.engine.nuitka_builder import _remove_windows_build_alias, _windows_ascii_build_alias

            alias = _windows_ascii_build_alias(str(scratch))
            cleanup.callback(_remove_windows_build_alias, alias)
            if alias:
                scratch = Path(alias)
        objects = []
        extension = ".obj" if target == "windows-x86_64" else ".o"
        for identity, program in sorted(entries.items()):
            symbol = "inx_cpu_" + identity
            path = scratch / (identity + extension)
            path.write_bytes(_object(program, symbol))
            objects.append(str(path))
            record["entries"].append(dict(identity=identity, symbol=symbol,
                parameter_names=program.parameter_names, parameter_kinds=program.parameter_kinds,
                domain_parameter=program.domain_parameter, continuation=asdict(program.continuation_plan)))
        output = scratch / library_filename(target, "build")
        _link(output, target, objects, [entry["symbol"] for entry in record["entries"]])
        with output.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        filename = library_filename(target, digest)
        record["library"] = dict(target=target, file=filename, sha256=digest, bytes=output.stat().st_size)
        cached = stage_library(output, directory, record["library"])
    return record, (("programs/"+filename, str(cached)),)
