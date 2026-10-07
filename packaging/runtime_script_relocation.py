"""Executed by the candidate Python, including from a compiled Hub.

Use pip's launcher writer rather than patching executable bytes. No project or
package entry point is imported or executed while reconstructing its launcher.
"""

RELOCATE_RUNTIME_SCRIPTS = r"""
import csv
import importlib.metadata
import os
from pathlib import Path
import shlex
import sys
import sysconfig
from pip._vendor.distlib.scripts import ScriptMaker, enquote_executable

class RuntimeScriptMaker(ScriptMaker):
    def _build_shebang(self, executable, post_interp):
        if os.name == "nt":
            return super()._build_shebang(executable, post_interp)
        # Always use a shell-quoted trampoline: a plain shebang cannot quote
        # paths, and double quotes would expand $ or backticks in a POSIX path.
        return (b"#!/bin/sh\n'''exec' " + executable + post_interp
                + b' "$0" "$@"\n' + b"' '''\n")

root = Path(sys.prefix).resolve()
site = Path(sysconfig.get_path("purelib")).resolve()
scripts = Path(sysconfig.get_path("scripts")).resolve()
if not site.is_relative_to(root) or not scripts.is_relative_to(root):
    raise RuntimeError("Runtime script paths must belong to the candidate prefix")
scripts.mkdir(parents=True, exist_ok=True)
for info in sorted(site.glob("*.dist-info")):
    distribution = importlib.metadata.Distribution.at(info)
    entries = [entry for entry in distribution.entry_points
               if entry.group in {"console_scripts", "gui_scripts"}]
    if not entries:
        continue
    record = info / "RECORD"
    with record.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    rewritten = {}
    removed = set()
    for entry in entries:
        if Path(entry.name).name != entry.name or any(c in entry.name for c in '/\\:'):
            raise RuntimeError("Invalid runtime entry point name: " + entry.name)
        names = [entry.name]
        # pip installs versioned aliases in addition to its metadata entry.
        if distribution.metadata["Name"].lower() == "pip" and entry.name == "pip":
            names += [f"pip{sys.version_info.major}", f"pip{sys.version_info.major}.{sys.version_info.minor}"]
        maker = RuntimeScriptMaker(None, str(scripts))
        maker.executable = enquote_executable(sys.argv[1]) if os.name == "nt" else shlex.quote(sys.argv[1])
        maker.clobber = True
        maker.variants = {""}
        maker.set_mode = True
        for name in names:
            outputs = maker.make(name + " = " + entry.value, {"gui": entry.group == "gui_scripts"})
            for output in outputs:
                path = Path(output)
                relative = os.path.relpath(path, site).replace(os.sep, "/")
                rewritten[relative] = [relative, "", str(path.stat().st_size)]
                # Previous --target installs put these in site-packages/bin.
                obsolete = site / "bin" / path.name
                if obsolete.is_file() and obsolete != path:
                    obsolete.unlink()
                    removed.add(os.path.relpath(obsolete, site).replace(os.sep, "/"))
    rows = [row for row in rows if row[0].replace("\\", "/") not in rewritten.keys() | removed]
    rows.extend(rewritten.values())
    with record.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
"""
