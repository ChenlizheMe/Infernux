"""Observe file identity and journal revision, or current directory entries.

Windows timestamps can collide even when ChangeTime is included. Read models
use the file's USN to distinguish writes without reading or hashing content.
Directory journals do not describe every child edit, so directories observe
their names directly. Filesystems without file USNs cannot reuse file models.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import struct
from typing import NamedTuple


class _FileInformation(ctypes.Structure):
    _fields_ = [
        ("FileAttributes", wintypes.DWORD),
        ("CreationTime", wintypes.FILETIME),
        ("LastAccessTime", wintypes.FILETIME),
        ("LastWriteTime", wintypes.FILETIME),
        ("VolumeSerialNumber", wintypes.DWORD),
        ("FileSizeHigh", wintypes.DWORD), ("FileSizeLow", wintypes.DWORD),
        ("NumberOfLinks", wintypes.DWORD),
        ("FileIndexHigh", wintypes.DWORD), ("FileIndexLow", wintypes.DWORD),
    ]


class FileStamp(NamedTuple):
    metadata: tuple[int, ...]
    revision: tuple[int, bytes, int] | None
    entries: tuple[str, ...] | None


_kernel = ctypes.WinDLL("kernel32", use_last_error=True)
_open = _kernel.CreateFileW
_open.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
_open.restype = wintypes.HANDLE
_query = _kernel.GetFileInformationByHandle
_query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_FileInformation)]
_query.restype = wintypes.BOOL
_ioctl = _kernel.DeviceIoControl
_ioctl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID,
                   wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                   ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
_ioctl.restype = wintypes.BOOL
_close = _kernel.CloseHandle
_close.argtypes = [wintypes.HANDLE]
_close.restype = wintypes.BOOL

_INVALID_HANDLE = wintypes.HANDLE(-1).value
_FSCTL_READ_FILE_USN_DATA = 0x900EB
_FILE_ATTRIBUTE_DIRECTORY = 0x10
# Documented capability absence, not permission, sharing or arbitrary I/O errors.
_NO_JOURNAL = frozenset((1, 50, 1178, 1179))


def _journal_revision(handle) -> tuple[int, bytes, int] | None:
    versions = (wintypes.WORD * 2)(2, 3)
    output = ctypes.create_string_buffer(4096)
    returned = wintypes.DWORD()
    if not _ioctl(handle, _FSCTL_READ_FILE_USN_DATA, versions, ctypes.sizeof(versions),
                  output, ctypes.sizeof(output), ctypes.byref(returned), None):
        error = ctypes.get_last_error()
        if error in _NO_JOURNAL:
            return None
        raise ctypes.WinError(error)
    raw = output.raw
    length, major, _ = struct.unpack_from("<IHH", raw)
    if major not in (2, 3) or length > returned.value:
        raise OSError("Unsupported Windows file journal record")
    identity_end, revision_offset = (16, 24) if major == 2 else (24, 40)
    revision = struct.unpack_from("<q", raw, revision_offset)[0]
    return (major, raw[8:identity_end], revision) if revision else None


def stamp_reusable(stamp: FileStamp | None) -> bool:
    return stamp is None or stamp.revision is not None or stamp.entries is not None


class FileProbe:
    """Retain a query path, never a handle, resolved target or shared buffer.

    Each query follows the lexical path anew. This preserves junction retargets
    and permits concurrent queries through the same immutable probe.
    """

    __slots__ = ("_path", "_name")

    def __init__(self, path: str):
        self._path = path
        if path.startswith("\\\\?\\"):
            native = path
        elif path.startswith("\\\\"):
            native = "\\\\?\\UNC\\" + path[2:]
        else:
            native = "\\\\?\\" + path
        self._name = ctypes.create_unicode_buffer(native)

    def __call__(self) -> FileStamp | None:
        # Attribute access, all sharing modes, existing file/directory only.
        handle = _open(self._name, 0x80, 7, None, 3, 0x02000000, None)
        if handle == _INVALID_HANDLE:
            error = ctypes.get_last_error()
            if error in (2, 3, 267):
                return None
            raise ctypes.WinError(error)
        try:
            info = _FileInformation()
            if not _query(handle, ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            metadata = (
                info.VolumeSerialNumber, (info.FileIndexHigh << 32) | info.FileIndexLow,
                (info.CreationTime.dwHighDateTime << 32) | info.CreationTime.dwLowDateTime,
                (info.LastWriteTime.dwHighDateTime << 32) | info.LastWriteTime.dwLowDateTime,
                (info.FileSizeHigh << 32) | info.FileSizeLow, info.FileAttributes,
            )
            if info.FileAttributes & _FILE_ATTRIBUTE_DIRECTORY:
                try:
                    with os.scandir(self._name.value) as children:
                        entries = tuple(sorted(child.name for child in children))
                except (FileNotFoundError, NotADirectoryError):
                    return None
                return FileStamp(metadata, None, entries)
            return FileStamp(metadata, _journal_revision(handle), None)
        finally:
            _close(handle)


def file_stamp(path: str) -> FileStamp | None:
    return FileProbe(path)()
