"""Root-confined access to loose project and mod files.

This module is intentionally independent from the managed asset database.  A
``SandboxPath`` is a capability for one path below the active process root; it
is never an asset identity and is never serialized into an asset reference.
"""

from __future__ import annotations

import errno
import fnmatch
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from Infernux.application import Application
from Infernux.engine.path_utils import (
    is_lexical_path_within,
    lexical_path,
    lexical_path_key,
    portable_path,
    relative_path,
    resolved_path,
)


def _portable_glob_match(value: str, pattern: str) -> bool:
    """Match a portable glob with ``**/`` representing zero or more levels."""
    candidates = {pattern}
    collapsed = pattern
    while "**/" in collapsed:
        collapsed = collapsed.replace("**/", "", 1)
        candidates.add(collapsed)
    for candidate in candidates:
        if not fnmatch.fnmatchcase(value, candidate):
            continue
        if "**" in candidate or "/" not in candidate:
            return True
        if value.count("/") == candidate.count("/"):
            return True
    return False


def _sandbox_root() -> str:
    if Application.is_editor():
        project_root = Application.data_path()
        if not project_root:
            raise RuntimeError("Loose-file access requires an active Editor project")
        root = os.path.join(project_root, "Assets")
    elif Application.is_player():
        if sys.platform == "emscripten" or os.environ.get("INFERNUX_WEB_RUNTIME") == "1":
            # Web's executable image and cooked assets live in MEMFS and must
            # remain immutable. Loose files use their own browser-persistent
            # subtree; the Web bootstrap creates it before project code runs.
            root = os.path.join(Application.persistent_data_path(), "loose")
        else:
            root = Application._player_executable_directory()
    else:
        raise RuntimeError("Loose-file access is available only in Editor or Player")
    lexical_root = lexical_path(root)
    if _has_reparse_point(lexical_root):
        raise PermissionError("loose-file sandbox root cannot be a link or junction")
    root = resolved_path(lexical_root)
    if lexical_path_key(root) != lexical_path_key(lexical_root):
        raise PermissionError("loose-file sandbox root cannot traverse a link or junction")
    if not os.path.isdir(root):
        raise RuntimeError(f"Loose-file sandbox root is unavailable: {root}")
    return root


def _has_reparse_point(path: str) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _directory_identity(path: str) -> tuple[int, int]:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or _has_reparse_point(path):
        raise PermissionError("loose-file sandbox root cannot be a link or junction")
    return int(info.st_dev), int(info.st_ino)


def _windows_handle_path(handle: int) -> str:
    """Return the DOS path of an already-open Windows handle."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GetFinalPathNameByHandleW
    function.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    function.restype = wintypes.DWORD
    required = int(function(handle, None, 0, 0))
    if not required:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(required + 1)
    written = int(function(handle, buffer, len(buffer), 0))
    if not written or written >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value


def _windows_handle_attributes(handle: int) -> int:
    import ctypes
    from ctypes import wintypes

    class _FileAttributeTagInfo(ctypes.Structure):
        _fields_ = (
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GetFileInformationByHandleEx
    function.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    function.restype = wintypes.BOOL
    information = _FileAttributeTagInfo()
    if not function(
        handle,
        9,  # FileAttributeTagInfo
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(information.file_attributes)


def _windows_create_handle(path: str, desired_access: int, disposition: int) -> int:
    import ctypes
    from ctypes import wintypes

    file_share_all = 0x00000001 | 0x00000002 | 0x00000004
    file_attribute_normal = 0x00000080
    file_flag_open_reparse_point = 0x00200000
    invalid_handle = ctypes.c_void_p(-1).value
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        path,
        desired_access,
        file_share_all,
        None,
        disposition,
        file_attribute_normal | file_flag_open_reparse_point,
        None,
    )
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _windows_close_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.CloseHandle
    function.argtypes = (wintypes.HANDLE,)
    function.restype = wintypes.BOOL
    if not function(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_mark_delete(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = (("delete_file", wintypes.BOOL),)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.SetFileInformationByHandle
    function.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    function.restype = wintypes.BOOL
    information = _FileDispositionInfo(True)
    if not function(
        handle,
        4,  # FileDispositionInfo
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _relative_token(value: str | os.PathLike[str], *, allow_empty: bool) -> str:
    raw = os.fspath(value)
    if not isinstance(raw, str):
        raise TypeError("sandbox paths must be text paths")
    raw = raw.strip()
    if not raw:
        if allow_empty:
            return ""
        raise ValueError("sandbox path cannot be empty")
    # Reject both platform spellings on every host. A Windows drive/UNC path
    # must not turn into an innocuous-looking relative filename in a Linux
    # Player, and a POSIX absolute path must not be reinterpreted on Windows.
    if (
        os.path.isabs(raw)
        or PurePosixPath(raw).is_absolute()
        or bool(PureWindowsPath(raw).anchor)
    ):
        raise ValueError("sandbox paths must be relative")
    normalized = portable_path(raw).strip("/")
    parts = normalized.split("/") if normalized else []
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("sandbox paths cannot contain '.', '..', or empty segments")
    return "/".join(parts)


def _resolve_confined(
    root: str,
    relative: str,
    *,
    allow_root: bool = False,
    allow_missing_leaf: bool = False,
) -> str:
    candidate = os.path.join(root, *relative.split("/")) if relative else root
    candidate = lexical_path(candidate)
    if not is_lexical_path_within(candidate, root, allow_root=allow_root):
        raise ValueError("sandbox path escapes its root")

    # Reject every existing redirecting filesystem object, including Windows
    # junctions/mount-point reparse records.  Resolve is checked as a second
    # line of defence for platforms whose lstat exposes redirects differently.
    current = root
    for part in relative.split("/") if relative else ():
        current = os.path.join(current, part)
        if not os.path.lexists(current):
            break
        if _has_reparse_point(current):
            raise PermissionError("sandbox paths cannot traverse links or junctions")
    if os.path.lexists(candidate) or not allow_missing_leaf:
        canonical = resolved_path(candidate)
        if not is_lexical_path_within(canonical, root, allow_root=allow_root):
            raise PermissionError("sandbox path resolves outside its root")
    return candidate


@dataclass(frozen=True, slots=True, init=False)
class SandboxPath:
    """One loose file confined to the active immutable sandbox root."""

    _root: str
    _root_identity: tuple[int, int]
    relative_path: str

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Acquire one handle below the active Editor/Player sandbox root.

        The root is deliberately not a constructor argument. Accepting it
        would let gameplay code manufacture a handle for an arbitrary system
        directory and bypass the confinement performed by ``create``.
        """
        root = _sandbox_root()
        relative = _relative_token(path, allow_empty=True)
        candidate = _resolve_confined(
            root,
            relative,
            allow_root=True,
            allow_missing_leaf=True,
        )
        if os.path.isdir(candidate):
            raise IsADirectoryError(relative or ".")
        if os.path.lexists(candidate) and not os.path.isfile(candidate):
            raise ValueError("sandbox handles can represent only regular files")
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_root_identity", _directory_identity(root))
        object.__setattr__(self, "relative_path", relative)

    @classmethod
    def create(cls, path: str | os.PathLike[str]) -> "SandboxPath":
        return cls(path)

    @classmethod
    def _from_active_root(cls, root: str, relative: str) -> "SandboxPath":
        """Construct a listed entry while retaining the already-checked root."""
        active_root = _sandbox_root()
        if lexical_path_key(root) != lexical_path_key(active_root):
            raise PermissionError("loose-file handles must use the active sandbox root")
        normalized = _relative_token(relative, allow_empty=True)
        candidate = _resolve_confined(
            active_root,
            normalized,
            allow_root=True,
            allow_missing_leaf=True,
        )
        if not os.path.isfile(candidate):
            raise ValueError("sandbox query results must be regular files")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_root", active_root)
        object.__setattr__(
            instance, "_root_identity", _directory_identity(active_root)
        )
        object.__setattr__(instance, "relative_path", normalized)
        return instance

    @property
    def name(self) -> str:
        return self.relative_path.rsplit("/", 1)[-1] if self.relative_path else ""

    @property
    def is_file(self) -> bool:
        return os.path.isfile(self._checked_path(allow_root=True))

    def _checked_path(
        self, *, allow_root: bool = False, allow_missing_leaf: bool = False
    ) -> str:
        active_root = self._active_root()
        return _resolve_confined(
            active_root,
            self.relative_path,
            allow_root=allow_root,
            allow_missing_leaf=allow_missing_leaf,
        )

    def _active_root(self) -> str:
        active_root = _sandbox_root()
        if (
            lexical_path_key(self._root) != lexical_path_key(active_root)
            or _directory_identity(active_root) != self._root_identity
        ):
            raise PermissionError("loose-file handle belongs to a different sandbox root")
        return active_root

    def _open_regular_fd(
        self,
        flags: int,
        *,
        allow_missing_leaf: bool = False,
        exclusive_create: bool = False,
    ) -> tuple[int, str]:
        """Open one file and prove the handle still names the checked entry."""
        if os.name != "nt":
            parent_fd, leaf = self._open_posix_parent_fd()
            open_flags = flags | int(getattr(os, "O_BINARY", 0))
            open_flags |= self._required_posix_flag("O_NOFOLLOW")
            open_flags |= int(getattr(os, "O_CLOEXEC", 0))
            try:
                if exclusive_create:
                    fd = os.open(
                        leaf,
                        open_flags | os.O_CREAT | os.O_EXCL,
                        0o666,
                        dir_fd=parent_fd,
                    )
                elif allow_missing_leaf:
                    try:
                        fd = os.open(
                            leaf, open_flags, 0o666, dir_fd=parent_fd
                        )
                    except FileNotFoundError:
                        try:
                            fd = os.open(
                                leaf,
                                open_flags | os.O_CREAT | os.O_EXCL,
                                0o666,
                                dir_fd=parent_fd,
                            )
                        except FileExistsError:
                            fd = os.open(
                                leaf, open_flags, 0o666, dir_fd=parent_fd
                            )
                else:
                    fd = os.open(leaf, open_flags, 0o666, dir_fd=parent_fd)
            except OSError as exc:
                self._raise_posix_redirect_error(exc)
                raise
            finally:
                os.close(parent_fd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError("sandbox handles can represent only regular files")
                return fd, self.relative_path
            except BaseException:
                os.close(fd)
                raise

        path = self._checked_path(allow_missing_leaf=allow_missing_leaf)
        fd = self._open_windows_fd(path, flags, exclusive_create=exclusive_create)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("sandbox handles can represent only regular files")
            return fd, path
        except BaseException:
            os.close(fd)
            raise

    def _open_windows_fd(
        self,
        path: str,
        flags: int,
        *,
        exclusive_create: bool,
    ) -> int:
        """Open a Windows leaf, then validate the opened object before I/O."""
        import msvcrt

        generic_read = 0x80000000
        generic_write = 0x40000000
        delete_access = 0x00010000
        file_read_attributes = 0x00000080
        create_new = 1
        open_existing = 3

        access_mode = flags & (os.O_WRONLY | os.O_RDWR)
        desired_access = generic_read if not access_mode else generic_write
        desired_access |= file_read_attributes
        if exclusive_create:
            desired_access |= delete_access

        handle = _windows_create_handle(
            path,
            desired_access,
            create_new if exclusive_create else open_existing,
        )

        converted = False
        try:
            attributes = _windows_handle_attributes(handle)
            if attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
                raise PermissionError(
                    "sandbox paths cannot traverse links or junctions"
                )
            if attributes & int(getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0x10)):
                raise ValueError("sandbox handles can represent only regular files")
            final_path = _windows_handle_path(handle)
            self._active_root()
            if not is_lexical_path_within(
                final_path,
                self._root,
                allow_root=False,
            ):
                raise PermissionError("sandbox path resolves outside its root")
            crt_flags = flags | int(getattr(os, "O_BINARY", 0))
            fd = msvcrt.open_osfhandle(int(handle), crt_flags)
            if fd == -1:
                raise OSError("failed to convert Windows sandbox file handle")
            converted = True
            return fd
        except BaseException:
            if exclusive_create:
                try:
                    _windows_mark_delete(handle)
                except OSError:
                    pass
            raise
        finally:
            if not converted:
                _windows_close_handle(handle)

    @staticmethod
    def _required_posix_flag(name: str) -> int:
        value = int(getattr(os, name, 0))
        if not value:
            raise RuntimeError(f"POSIX loose-file sandbox requires {name}")
        return value

    @staticmethod
    def _raise_posix_redirect_error(exc: OSError) -> None:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise PermissionError(
                "sandbox paths cannot traverse links or junctions"
            ) from exc

    def _open_posix_parent_fd(self) -> tuple[int, str]:
        """Open the root and every parent without following redirectors."""

        active_root = self._active_root()
        parts = self.relative_path.split("/")
        if not parts or not parts[-1]:
            raise IsADirectoryError(self.relative_path or ".")
        directory_flags = (
            os.O_RDONLY
            | self._required_posix_flag("O_DIRECTORY")
            | self._required_posix_flag("O_NOFOLLOW")
            | int(getattr(os, "O_CLOEXEC", 0))
        )
        try:
            current_fd = os.open(active_root, directory_flags)
        except OSError as exc:
            self._raise_posix_redirect_error(exc)
            raise
        try:
            root_entry = os.stat(active_root, follow_symlinks=False)
            if (
                not os.path.samestat(os.fstat(current_fd), root_entry)
                or (int(root_entry.st_dev), int(root_entry.st_ino))
                != self._root_identity
            ):
                raise PermissionError("loose-file sandbox root changed while opening")
            for part in parts[:-1]:
                try:
                    child_fd = os.open(
                        part,
                        directory_flags,
                        dir_fd=current_fd,
                    )
                except OSError as exc:
                    self._raise_posix_redirect_error(exc)
                    raise
                os.close(current_fd)
                current_fd = child_fd
            return current_fd, parts[-1]
        except BaseException:
            os.close(current_fd)
            raise

    def read_bytes(self) -> bytes:
        try:
            fd, _path = self._open_regular_fd(os.O_RDONLY)
        except FileNotFoundError:
            raise FileNotFoundError(self.relative_path) from None
        with os.fdopen(fd, "rb") as stream:
            return stream.read()

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.read_bytes().decode(encoding)

    def write_bytes(self, data: bytes | bytearray | memoryview) -> int:
        payload = bytes(data)
        if os.name != "nt":
            fd, _path = self._open_regular_fd(
                os.O_WRONLY,
                allow_missing_leaf=True,
            )
            try:
                os.ftruncate(fd, 0)
            except BaseException:
                os.close(fd)
                raise
            with os.fdopen(fd, "wb") as stream:
                return stream.write(payload)

        path = self._checked_path(allow_missing_leaf=True)
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            raise FileNotFoundError(portable_path(os.path.dirname(self.relative_path)))
        # Existing files are opened without truncation, verified against the
        # checked directory entry, and only then replaced. Missing leaves use
        # exclusive creation so a concurrently inserted redirect cannot win.
        try:
            fd, _path = self._open_regular_fd(os.O_WRONLY)
        except FileNotFoundError:
            try:
                fd, _path = self._open_regular_fd(
                    os.O_WRONLY,
                    allow_missing_leaf=True,
                    exclusive_create=True,
                )
            except FileExistsError:
                fd, _path = self._open_regular_fd(os.O_WRONLY)
        try:
            os.ftruncate(fd, 0)
        except BaseException:
            os.close(fd)
            raise
        with os.fdopen(fd, "wb") as stream:
            return stream.write(payload)

    def write_text(self, text: str, encoding: str = "utf-8") -> int:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        return self.write_bytes(text.encode(encoding))

    def delete(self) -> None:
        if os.name != "nt":
            parent_fd, leaf = self._open_posix_parent_fd()
            try:
                flags = (
                    os.O_RDONLY
                    | self._required_posix_flag("O_NOFOLLOW")
                    | int(getattr(os, "O_CLOEXEC", 0))
                )
                try:
                    fd = os.open(leaf, flags, dir_fd=parent_fd)
                except OSError as exc:
                    self._raise_posix_redirect_error(exc)
                    raise
                try:
                    opened = os.fstat(fd)
                    if not stat.S_ISREG(opened.st_mode):
                        raise ValueError(
                            "sandbox handles can represent only regular files"
                        )
                    current = os.stat(
                        leaf,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISREG(current.st_mode) or not os.path.samestat(
                        opened, current
                    ):
                        raise PermissionError(
                            "sandbox file changed before deletion"
                        )
                    os.unlink(leaf, dir_fd=parent_fd)
                finally:
                    os.close(fd)
            finally:
                os.close(parent_fd)
            return

        self._delete_windows_file()

    def _delete_windows_file(self) -> None:
        """Delete the already-validated Windows object by handle, not path."""
        path = self._checked_path()
        delete_access = 0x00010000
        file_read_attributes = 0x00000080
        open_existing = 3
        handle = _windows_create_handle(
            path,
            delete_access | file_read_attributes,
            open_existing,
        )
        try:
            attributes = _windows_handle_attributes(handle)
            if attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
                raise PermissionError(
                    "sandbox paths cannot traverse links or junctions"
                )
            if attributes & int(getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0x10)):
                raise ValueError("sandbox handles can represent only regular files")
            final_path = _windows_handle_path(handle)
            self._active_root()
            if not is_lexical_path_within(
                final_path,
                self._root,
                allow_root=False,
            ):
                raise PermissionError("sandbox path resolves outside its root")
            _windows_mark_delete(handle)
        finally:
            _windows_close_handle(handle)


def find_sandbox_paths(pattern: str | os.PathLike[str]) -> list[SandboxPath]:
    """Return deterministic loose-file matches without following redirects."""

    root = _sandbox_root()
    normalized = _relative_token(pattern, allow_empty=True)
    try:
        directory = _resolve_confined(
            root,
            normalized,
            allow_root=True,
            allow_missing_leaf=False,
        )
    except PermissionError:
        # A query never follows a redirect. Treat the redirected subtree as
        # absent; direct handle acquisition remains fail-fast.
        return []
    if os.path.isdir(directory):
        matches = []
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda value: value.name.casefold()):
                if entry.is_symlink() or _has_reparse_point(entry.path):
                    continue
                if entry.is_file(follow_symlinks=False):
                    relative = relative_path(entry.path, root)
                    matches.append(SandboxPath._from_active_root(root, relative))
        return matches
    normalized = normalized or "*"
    matches: list[SandboxPath] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        safe_directories: list[str] = []
        for name in sorted(directory_names):
            child = os.path.join(current, name)
            if not _has_reparse_point(child):
                safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in sorted(file_names):
            candidate = os.path.join(current, name)
            if _has_reparse_point(candidate) or not os.path.isfile(candidate):
                continue
            relative = relative_path(candidate, root)
            if _portable_glob_match(relative, normalized):
                matches.append(SandboxPath._from_active_root(root, relative))
    matches.sort(key=lambda item: item.relative_path.casefold())
    return matches


__all__ = ["SandboxPath", "find_sandbox_paths"]
