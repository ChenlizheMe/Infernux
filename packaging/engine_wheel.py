"""Version-independent Hub boundaries for engine-owned files and entry points.

Do not import an engine into the Hub process. Wheel contents and the project's
installed distribution determine the layout, not the Hub's own release number.
"""

from pathlib import PurePosixPath
import zipfile


def read_project_template(wheel: str, name: str, *, required: bool = True) -> bytes | None:
    if not name or PurePosixPath(name).name != name or "\\" in name:
        raise ValueError(f"Expected a template filename, got {name!r}")
    with zipfile.ZipFile(wheel) as archive:
        matches = [entry for entry in archive.infolist()
                   if not entry.is_dir() and PurePosixPath(entry.filename).name == name]
        if not matches and not required:
            return None
        if len(matches) != 1:
            locations = ", ".join(entry.filename for entry in matches)
            raise RuntimeError(
                f"Infernux wheel must contain exactly one project template '{name}', "
                f"found {len(matches)}" + (f": {locations}" if locations else "")
            )
        # Read bytes, never extract/execute the template or choose the first
        # filename match when multiple directories contain it.
        return archive.read(matches[0])


def runtime_package_script(*, installed: bool) -> str:
    """Resolve the exact package spelling in the child, without speculative imports."""
    if not installed:
        return "import importlib\n_engine_package = 'infernux'\n"
    return (
        "import importlib, importlib.metadata\n"
        "_distribution = importlib.metadata.distribution('infernux')\n"
        "_engine_roots = {p.parts[0] for p in (_distribution.files or ()) "
        "if len(p.parts) == 2 and p.parts[0].casefold() == 'infernux' "
        "and p.parts[1] == '__init__.py'}\n"
        "if len(_engine_roots) != 1:\n"
        "    raise RuntimeError('Installed Infernux distribution has no unique engine package')\n"
        "_engine_package = next(iter(_engine_roots))\n"
    )


def editor_launch_script(*, installed: bool) -> str:
    return runtime_package_script(installed=installed) + (
        "import sys\n"
        "_engine = importlib.import_module(_engine_package + '.engine')\n"
        "_lib = importlib.import_module(_engine_package + '.lib')\n"
        "_engine.release_engine(engine_log_level=_lib.LogLevel.Info, project_path=sys.argv[1])\n"
    )
