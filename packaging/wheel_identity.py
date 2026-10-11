"""Validate distribution identity at the wheel publication boundary."""
from email import policy
from email.parser import BytesParser
from pathlib import Path
import zipfile

from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def validate_wheel_identity(path: str, *, filename: str | None = None) -> None:
    """Require filename, distribution metadata, build and ABI to agree.

    This reads only the wheel's two identity documents. It neither executes
    package code nor hashes its payload. ``filename`` supplies the final name
    while a download/import still owns a uniquely named temporary file.
    """
    name = filename or Path(path).name
    try:
        distribution, version, build, tags = parse_wheel_filename(name)
        if distribution != "infernux":
            raise ValueError("expected the infernux distribution")
        with zipfile.ZipFile(path) as archive:
            documents = [entry for entry in archive.namelist() if entry.endswith(".dist-info/METADATA")]
            if len(documents) != 1:
                raise ValueError("expected exactly one distribution metadata document")
            directory = documents[0].rsplit("/", 1)[0]
            if directory.casefold() != f"infernux-{version}.dist-info".casefold():
                raise ValueError("distribution directory does not match the filename")
            parser = BytesParser(policy=policy.default)
            metadata = parser.parsebytes(archive.read(documents[0]))
            wheel = parser.parsebytes(archive.read(f"{directory}/WHEEL"))
        names = metadata.get_all("Name", [])
        versions = metadata.get_all("Version", [])
        if len(names) != 1 or canonicalize_name(names[0]) != distribution:
            raise ValueError("distribution name does not match the filename")
        if len(versions) != 1 or Version(versions[0]) != version:
            raise ValueError("distribution version does not match the filename")
        expected_build = [f"{build[0]}{build[1]}"] if build else []
        if wheel.get_all("Build", []) != expected_build:
            raise ValueError("wheel revision does not match the filename")
        declared_tags = frozenset(tag for value in wheel.get_all("Tag", []) for tag in parse_tag(value))
        if declared_tags != tags:
            raise ValueError("wheel target tags do not match the filename")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Wheel identity mismatch for {name}: {exc}") from exc


def has_matching_wheel_identity(path: str) -> bool:
    try:
        validate_wheel_identity(path)
    except ValueError:
        return False
    return True
