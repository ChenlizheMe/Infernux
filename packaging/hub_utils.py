"""Utility helpers shared across the Hub codebase."""

import os
import shutil
import sys
from enum import Enum
import infernux_project_lock as project_lock


def remove_directory_tree(path: str | os.PathLike[str]) -> None:
    """Remove one explicitly owned directory through literal filesystem paths.

    The Hub ships without the engine package, so it owns this stdlib boundary.
    Linked roots are rejected; shutil does not follow links inside the tree.
    Missing targets are already clean. Other failures must reach the caller.
    """
    if not path:
        raise ValueError("Directory cleanup requires an explicit path")
    target = os.path.abspath(path)
    if target == os.path.dirname(target):
        raise ValueError("Directory cleanup cannot remove a filesystem root")
    try:
        os.lstat(target)
    except FileNotFoundError:
        return
    if os.path.islink(target) or os.path.isjunction(target):
        raise ValueError(f"Directory cleanup cannot remove a linked root: {target}")
    shutil.rmtree(target)


class HubLaunchContext(Enum):
    """Explicit policy boundary between source and installed Hub launches."""

    SOURCE = "source"
    INSTALLED = "installed"

    @classmethod
    def current(cls) -> "HubLaunchContext":
        return cls.INSTALLED if is_frozen() else cls.SOURCE

    @property
    def uses_installed_versions(self) -> bool:
        return self is HubLaunchContext.INSTALLED


def is_frozen() -> bool:
    """Return *True* inside a Nuitka compiled application."""
    # Nuitka defines ``__compiled__`` on the executable's main module.  It is
    # not guaranteed to copy that marker into imported modules such as this
    # one, so checking only ``globals()`` makes a standalone Hub look like a
    # source checkout.  The launcher then skips packaged-only startup work,
    # including the automatic update check.
    if "__compiled__" in globals():
        return True
    main_module = sys.modules.get("__main__")
    return bool(main_module and "__compiled__" in vars(main_module))


def get_bundle_dir() -> str:
    """Return the directory containing bundled data files."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def get_app_dir() -> str:
    """Return the executable directory for the running Hub."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def get_inner_dir() -> str:
    """Return the Hub private data directory.

    In packaged builds this resolves next to the Hub executable. In source mode
    it resolves under packaging/InfernuxHubData so dev runs behave the same way.
    """
    return get_hub_data_dir()


def get_hub_data_dir() -> str:
    """Return the Hub application data directory next to the executable."""
    return os.path.join(get_app_dir(), "InfernuxHubData")


def get_hub_shared_data_dir(app_dir: str | None = None) -> str:
    """Mutable, reusable content beside the Hub, not in its application payload."""
    if app_dir is None:
        configured = os.environ.get("INFERNUX_SHARED_DATA_ROOT", "").strip()
        if configured:
            return os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
        app_dir = get_app_dir()
    return os.path.abspath(os.path.join(app_dir, "InfernuxHubData", "Shared"))


def get_hub_user_data_dir() -> str:
    """Return the per-user Hub data root shared by source and installed launches."""
    configured = os.environ.get("INFERNUX_DATA_ROOT", "").strip()
    if configured:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if not local_app_data:
            raise RuntimeError("Infernux Hub requires LOCALAPPDATA on Windows")
        return os.path.join(local_app_data, "InfernuxHub")
    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg_data_home:
        return os.path.join(os.path.expanduser(xdg_data_home), "InfernuxHub")
    return os.path.expanduser("~/.local/share/InfernuxHub")


def get_project_lock_path(project_path: str) -> str:
    """Return the lock-file path that marks a project as opened by the engine."""
    return project_lock.lock_path(project_path)


def is_pid_running(pid: int) -> bool:
    """Return True if *pid* currently exists."""
    return project_lock.is_pid_running(pid)


def read_project_lock(project_path: str) -> dict | None:
    """Return active lock metadata for *project_path*, removing stale locks automatically."""
    return project_lock.read_lock(project_path, probe=is_pid_running)


def is_project_open(project_path: str) -> bool:
    """Return True if the project currently has a live engine process."""
    return read_project_lock(project_path) is not None


def write_project_lock(project_path: str, pid: int, token: str, mode: str, state: str) -> str:
    """Reserve project preparation or transfer that reservation to its child."""
    return project_lock.reserve(project_path, pid, token, mode, state, probe=is_pid_running)


def merge_child_env_utf8(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for subprocesses: inherit current env and prefer UTF-8 on Windows."""
    merged = {**os.environ, **(extra or {})}
    merged.setdefault("INFERNUX_DATA_ROOT", get_hub_user_data_dir())
    merged.setdefault("INFERNUX_SHARED_DATA_ROOT", get_hub_shared_data_dir())
    merged.setdefault(
        "INFERNUX_PACKAGE_CACHE_ROOT",
        os.path.join(merged["INFERNUX_SHARED_DATA_ROOT"], "Library", "Plugins"),
    )
    if not merged.get("PIP_CACHE_DIR", "").strip():
        merged["PIP_CACHE_DIR"] = os.path.join(
            merged["INFERNUX_SHARED_DATA_ROOT"], "Cache", "Python", "Pip"
        )
    if sys.platform == "win32":
        merged.setdefault("PYTHONUTF8", "1")
        merged.setdefault("PYTHONIOENCODING", "utf-8")
    return merged


def remove_project_lock(project_path: str, token: str | None = None) -> None:
    """Release only this launch's own reservation or its stopped child."""
    if project_path:
        project_lock.remove_lock(get_project_lock_path(project_path), token, probe=is_pid_running)
