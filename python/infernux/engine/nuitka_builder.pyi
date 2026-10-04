"""NuitkaBuilder — Nuitka-based standalone executable compiler.

Used by :class:`GameBuilder` to compile the player bootstrap script
into a native executable.

Example::

    builder = NuitkaBuilder(
        boot_script="boot.py",
        project_path="/path/to/project",
        output_dir="/path/to/dist",
    )
    builder.build()
"""

from __future__ import annotations

from typing import Callable, List, Optional


class NuitkaBuilder:
    """Low-level Nuitka compilation wrapper."""

    entry_script: str
    output_dir: str
    output_filename: str
    product_name: str
    file_version: str
    icon_path: Optional[str]
    console_mode: str
    extra_include_packages: List[str]
    extra_include_data: List[str]
    extra_requirements_files: List[str]
    raw_copy_packages: List[str]
    lto: bool
    runtime_pack_cache: bool

    def __init__(
        self,
        entry_script: str,
        output_dir: str,
        *,
        output_filename: str = ...,
        product_name: str = ...,
        file_version: str = ...,
        icon_path: Optional[str] = None,
        extra_include_packages: Optional[List[str]] = None,
        extra_include_data: Optional[List[str]] = None,
        extra_requirements_files: Optional[List[str]] = None,
        raw_copy_packages: Optional[List[str]] = None,
        console_mode: str = ...,
        lto: bool = ...,
        runtime_pack_cache: bool = ...,
    ) -> None: ...

    def build(self) -> str:
        """Run Nuitka compilation and post-processing.

        Returns:
            Path to the ``dist`` directory containing the compiled output.

        Raises:
            RuntimeError: If Nuitka is not installed or compilation fails.
        """
        ...
