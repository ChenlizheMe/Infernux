"""Resolve PE imports from this build's targets, or audit the contents of a wheel."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


# The Windows/driver and interpreter ABIs the engine imports. A developer's
# System32, Python install or PATH can contain third-party DLLs as well.
HOST_DLLS = frozenset('''
    advapi32.dll bcrypt.dll comctl32.dll comdlg32.dll d3d11.dll d3d12.dll dxgi.dll
    gdi32.dll imm32.dll kernel32.dll kernelbase.dll ntdll.dll ole32.dll oleaut32.dll
    opengl32.dll setupapi.dll shell32.dll shlwapi.dll ucrtbase.dll user32.dll
    version.dll vulkan-1.dll winmm.dll ws2_32.dll python3.dll
'''.split()) | {f'python{sys.version_info.major}{sys.version_info.minor}.dll'}


def imports(binary, dumpbin):
    output = subprocess.check_output([dumpbin, '/nologo', '/dependents', binary])
    # Both the ordinary and delay-load tables print one indented DLL name per row.
    for line in output.splitlines():
        name = line.strip()
        if line[:1].isspace() and name.lower().endswith(b'.dll'):
            yield os.fsdecode(name).casefold()


def resolve(roots, candidates, dumpbin, developer_crt=False):
    producers = {}
    for path in candidates:
        producers.setdefault(path.name.casefold(), set()).add(path)

    pending = list(roots)
    visited = set()
    runtime = set()
    while pending:
        binary = pending.pop()
        if binary in visited:
            continue
        visited.add(binary)
        for name in imports(binary, dumpbin):
            if name in producers:
                choices = [path for path in producers[name] if path.is_file()]
                if len(choices) != 1:
                    raise ValueError(f'{binary.name} imports {name}, expected one built DLL in {producers[name]}')
                dependency, = choices
                runtime.add(dependency)
                pending.append(dependency)
            elif name.startswith(('api-ms-', 'ext-ms-')) or name in HOST_DLLS:
                continue
            elif developer_crt and name.endswith('d.dll') and name.startswith(
                ('msvcp', 'vcruntime', 'concrt', 'ucrtbase', 'python')
            ):
                continue
            else:
                raise ValueError(f'{binary.name} imports {name}, missing from the supplied runtime')
    return sorted(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dumpbin', type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--targets', type=Path)
    source.add_argument('--wheel-lib', type=Path)
    parser.add_argument('--developer-crt', action='store_true')
    parser.add_argument('roots', nargs='*', type=Path)
    args = parser.parse_intermixed_args()
    if args.targets:
        candidates = map(Path, args.targets.read_text(encoding='utf-8').splitlines())
        for path in resolve(args.roots, candidates, args.dumpbin, args.developer_crt):
            print(path.as_posix())
    else:
        binaries = [p for p in args.wheel_lib.iterdir() if p.suffix.lower() in {'.dll', '.pyd', '.exe'}]
        resolve(binaries, binaries, args.dumpbin)
        print(f'Checked native imports of {len(binaries)} packaged Windows binaries')


if __name__ == '__main__':
    try:
        main()
    except ValueError as error:
        sys.exit(str(error))
