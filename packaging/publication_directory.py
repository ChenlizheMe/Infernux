"""Staging directories whose contents will become a permanent installation."""

from pathlib import Path
import ctypes
import os
import stat
import sys
import tempfile
import uuid

from python_execution import windows_native_path


def restore_windows_inherited_access_tree(root: Path) -> None:
    """Restore parent access on owned files and directories from old staging.

    Preserve explicit permissions and never traverse junctions or symlinks.
    Changing the top-level ACL alone cannot repair independently protected
    Python runtime trees. This runs during installation or one-time migration.
    """
    if sys.platform != "win32":
        return
    from ctypes import wintypes

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_security = advapi.GetNamedSecurityInfoW
    get_security.argtypes = [wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
                            ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)]
    get_security.restype = wintypes.DWORD
    get_control = advapi.GetSecurityDescriptorControl
    get_control.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.WORD),
                           ctypes.POINTER(wintypes.DWORD)]
    get_control.restype = wintypes.BOOL
    set_security = advapi.SetNamedSecurityInfoW
    set_security.argtypes = [wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
                            ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.c_void_p]
    set_security.restype = wintypes.DWORD
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p

    pending = [Path(root)]
    while pending:
        path = pending.pop()
        info = path.stat(follow_symlinks=False)
        if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            continue
        descriptor = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        native_path = windows_native_path(str(path))
        result = get_security(native_path, 1, 4, None, None, ctypes.byref(dacl),
                              None, ctypes.byref(descriptor))
        if result:
            error = ctypes.WinError(result)
            error.filename = str(path)
            raise error
        try:
            control, revision = wintypes.WORD(), wintypes.DWORD()
            if not get_control(descriptor, ctypes.byref(control), ctypes.byref(revision)):
                raise ctypes.WinError(ctypes.get_last_error())
            if control.value & 0x1000:  # SE_DACL_PROTECTED
                # DACL_SECURITY_INFORMATION | UNPROTECTED_DACL_SECURITY_INFORMATION
                result = set_security(native_path, 1, 4 | 0x20000000, None, None, dacl, None)
                if result:
                    error = ctypes.WinError(result)
                    error.filename = str(path)
                    raise error
        finally:
            kernel.LocalFree(descriptor)
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)


def clear_windows_readonly_tree(root: Path) -> None:
    """Clear DOS read-only attributes on an owned installation tree once.

    This does not grant access or change ACLs. Reparse points are excluded so
    installing or repairing Hub never changes files outside its own tree.
    """
    if sys.platform != "win32":
        return
    pending = [Path(root)]
    while pending:
        path = pending.pop()
        info = path.stat(follow_symlinks=False)
        attributes = getattr(info, "st_file_attributes", 0)
        if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            continue
        if attributes & stat.FILE_ATTRIBUTE_READONLY:
            path.chmod(stat.S_IREAD | stat.S_IWRITE)
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)


def create_publication_directory(parent: Path, prefix: str) -> Path:
    if sys.platform != "win32":
        return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    # Python 3.13's mkdtemp uses a protected OWNER RIGHTS ACL. With an
    # elevated token its owner is Administrators, so renaming the directory
    # into place leaves ordinary users unable to access the installation.
    # A permanent tree must inherit its destination parent's permissions.
    directory = parent / (prefix + uuid.uuid4().hex)
    directory.mkdir(mode=0o777)
    return directory
