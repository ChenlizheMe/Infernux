"""Publish MSVC's structured UTF-8 header dependencies to Ninja."""

import json
from pathlib import Path
import subprocess
import sys


def main():
    command = [argument for argument in sys.argv[1:] if argument.casefold() != "/showincludes"]
    output = next(argument[3:] for argument in command if argument[:3].casefold() == "/fo")
    dependencies = Path(output + ".dependencies.json")
    result = subprocess.run([*command, "/sourceDependencies", str(dependencies)])
    if result.returncode:
        return result.returncode
    document = json.loads(dependencies.read_text(encoding="utf-8-sig"))
    for header in document["Data"]["Includes"]:
        sys.stdout.buffer.write(("Note: including file: " + header + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
    dependencies.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
