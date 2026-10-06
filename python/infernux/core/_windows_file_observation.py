"""One Windows metadata query, including ChangeTime rather than birth time.

NtQueryFullAttributesFile returns the FILE_NETWORK_OPEN_INFORMATION of the
followed path without reading its contents. In-place copies can preserve size
and LastWriteTime; ChangeTime still observes those ordinary filesystem writes.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes


class _UnicodeString(ctypes.Structure):
    _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                ("Buffer", wintypes.LPWSTR)]


class _ObjectAttributes(ctypes.Structure):
    _fields_ = [("Length", wintypes.ULONG), ("RootDirectory", wintypes.HANDLE),
                ("ObjectName", ctypes.POINTER(_UnicodeString)), ("Attributes", wintypes.ULONG),
                ("SecurityDescriptor", wintypes.LPVOID), ("SecurityQualityOfService", wintypes.LPVOID)]


class _NetworkOpenInformation(ctypes.Structure):
    _fields_ = [("CreationTime", ctypes.c_longlong), ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong), ("ChangeTime", ctypes.c_longlong),
                ("AllocationSize", ctypes.c_longlong), ("EndOfFile", ctypes.c_longlong),
                ("FileAttributes", wintypes.ULONG)]


_ntdll = ctypes.WinDLL("ntdll")
_query = _ntdll.NtQueryFullAttributesFile
_query.argtypes = [ctypes.POINTER(_ObjectAttributes), ctypes.POINTER(_NetworkOpenInformation)]
_query.restype = wintypes.LONG
_dos_error = _ntdll.RtlNtStatusToDosError
_dos_error.argtypes = [wintypes.LONG]
_dos_error.restype = wintypes.ULONG


class FileProbe:
    """Reuse only the immutable query path, never the observed file state.

    A bounded FileReadCache owns these probes. No open handles, global path
    cache, resolved junction targets or shared output buffers are retained.
    """

    __slots__ = ("_path", "_name", "_unicode_name", "_attributes")

    def __init__(self, path: str):
        self._path = path
        # Preserve extended drive/UNC paths and follow the lexical path anew
        # on every call, including after a junction has changed its target.
        if path.startswith("\\\\?\\"):
            native = "\\??\\" + path[4:]
        elif path.startswith("\\\\"):
            native = "\\??\\UNC\\" + path[2:]
        else:
            native = "\\??\\" + path
        self._name = ctypes.create_unicode_buffer(native)
        byte_count = ctypes.sizeof(self._name)
        if byte_count > 65535:
            raise OSError(36, "Filesystem observation path is too long", path)
        self._unicode_name = _UnicodeString(
            byte_count - ctypes.sizeof(ctypes.c_wchar), byte_count,
            ctypes.cast(self._name, wintypes.LPWSTR),
        )
        self._attributes = _ObjectAttributes(
            ctypes.sizeof(_ObjectAttributes), None,
            ctypes.pointer(self._unicode_name), 0x40, None, None,
        )

    def __call__(self):
        info = _NetworkOpenInformation()
        status = _query(ctypes.byref(self._attributes), ctypes.byref(info))
        if status < 0:
            error = _dos_error(status)
            if error in (2, 3, 267):  # absent file/parent, non-directory parent
                return None
            raise ctypes.WinError(
                error, f"Cannot observe {self._path!r}: {ctypes.FormatError(error)}",
            )
        return (info.CreationTime, info.LastWriteTime, info.ChangeTime,
                info.EndOfFile, info.FileAttributes)


def file_stamp(path: str):
    return FileProbe(path)()
