"""Cross-platform file / folder / error dialog helpers.

Consolidates Win32 COM + tkinter fallback dialog code that was previously
duplicated in ``build_settings_panel.py`` and ``scene_manager.py``.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from Infernux.debug import Debug


# ---------------------------------------------------------------------------
# Win32 low-level dialog implementations
# ---------------------------------------------------------------------------

def _win32_pick_folder(title: str) -> Optional[str]:
    import ctypes
    import ctypes.wintypes as wt

    COINIT_APARTMENTTHREADED = 0x2
    BIF_RETURNONLYFSDIRS = 0x00000001
    BIF_NEWDIALOGSTYLE = 0x00000040
    MAX_PATH = 260

    class BROWSEINFOW(ctypes.Structure):
        _fields_ = [
            ("hwndOwner", wt.HWND),
            ("pidlRoot", ctypes.c_void_p),
            ("pszDisplayName", wt.LPWSTR),
            ("lpszTitle", wt.LPCWSTR),
            ("ulFlags", wt.UINT),
            ("lpfn", ctypes.c_void_p),
            ("lParam", ctypes.c_void_p),
            ("iImage", ctypes.c_int),
        ]

    display_name = ctypes.create_unicode_buffer(MAX_PATH)
    browse = BROWSEINFOW()
    browse.pszDisplayName = ctypes.cast(display_name, wt.LPWSTR)
    browse.lpszTitle = title
    browse.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE

    shell32 = ctypes.windll.shell32
    ole32 = ctypes.windll.ole32
    user32 = ctypes.windll.user32
    shell32.SHBrowseForFolderW.argtypes = [ctypes.POINTER(BROWSEINFOW)]
    shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
    shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, wt.LPWSTR]
    shell32.SHGetPathFromIDListW.restype = wt.BOOL
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wt.DWORD]
    ole32.CoInitializeEx.restype = ctypes.HRESULT
    ole32.CoUninitialize.argtypes = []
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree.restype = None

    browse.hwndOwner = user32.GetActiveWindow()

    coinit_hr = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    should_uninitialize = coinit_hr in (0, 1)
    try:
        pidl = shell32.SHBrowseForFolderW(ctypes.byref(browse))
        if not pidl:
            return None
        path_buf = ctypes.create_unicode_buffer(MAX_PATH)
        if shell32.SHGetPathFromIDListW(pidl, path_buf):
            return path_buf.value
        return None
    finally:
        if 'pidl' in locals() and pidl:
            ole32.CoTaskMemFree(pidl)
        if should_uninitialize:
            ole32.CoUninitialize()


def _win32_pick_file(title: str, filter_text: str) -> Optional[str]:
    import ctypes
    import ctypes.wintypes as wt

    COINIT_APARTMENTTHREADED = 0x2
    OFN_FILEMUSTEXIST = 0x00001000
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_NOCHANGEDIR = 0x00000008
    OFN_EXPLORER = 0x00080000
    MAX_PATH = 4096

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wt.DWORD),
            ("hwndOwner", wt.HWND),
            ("hInstance", wt.HINSTANCE),
            ("lpstrFilter", wt.LPCWSTR),
            ("lpstrCustomFilter", wt.LPWSTR),
            ("nMaxCustFilter", wt.DWORD),
            ("nFilterIndex", wt.DWORD),
            ("lpstrFile", wt.LPWSTR),
            ("nMaxFile", wt.DWORD),
            ("lpstrFileTitle", wt.LPWSTR),
            ("nMaxFileTitle", wt.DWORD),
            ("lpstrInitialDir", wt.LPCWSTR),
            ("lpstrTitle", wt.LPCWSTR),
            ("Flags", wt.DWORD),
            ("nFileOffset", wt.WORD),
            ("nFileExtension", wt.WORD),
            ("lpstrDefExt", wt.LPCWSTR),
            ("lCustData", ctypes.c_void_p),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", wt.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", wt.DWORD),
            ("FlagsEx", wt.DWORD),
        ]

    buf = ctypes.create_unicode_buffer(MAX_PATH)
    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.lpstrFilter = filter_text
    ofn.lpstrFile = ctypes.cast(buf, wt.LPWSTR)
    ofn.nMaxFile = MAX_PATH
    ofn.lpstrTitle = title
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER
    ofn.hwndOwner = ctypes.windll.user32.GetActiveWindow()

    comdlg32 = ctypes.windll.comdlg32
    ole32 = ctypes.windll.ole32
    comdlg32.GetOpenFileNameW.argtypes = [ctypes.POINTER(OPENFILENAMEW)]
    comdlg32.GetOpenFileNameW.restype = wt.BOOL
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wt.DWORD]
    ole32.CoInitializeEx.restype = ctypes.HRESULT
    ole32.CoUninitialize.argtypes = []

    coinit_hr = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    should_uninitialize = coinit_hr in (0, 1)
    try:
        if comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return buf.value
        return None
    finally:
        if should_uninitialize:
            ole32.CoUninitialize()


# ---------------------------------------------------------------------------
# Cross-platform public API
# ---------------------------------------------------------------------------

def pick_folder_dialog(title: str) -> Optional[str]:
    if sys.platform == "win32":
        try:
            return _win32_pick_folder(title)
        except Exception as exc:
            Debug.log_warning(f"Win32 folder dialog failed: {exc}")
            return None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askdirectory(parent=root, title=title)
    finally:
        root.destroy()


def pick_file_dialog(title: str, win32_filter: str = "",
                     tk_filetypes: list | None = None) -> Optional[str]:
    if sys.platform == "win32" and win32_filter:
        try:
            return _win32_pick_file(title, win32_filter)
        except Exception as exc:
            Debug.log_warning(f"Win32 open-file dialog failed: {exc}")
            return None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askopenfilename(
            parent=root,
            title=title,
            filetypes=tk_filetypes or [("All Files", "*.*")],
        )
    finally:
        root.destroy()


def show_system_error_dialog(title: str, message: str) -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10 | 0x0)
        return

    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        messagebox.showerror(title, message, parent=root)
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# Save-file dialog  (cross-platform, synchronous)
# ---------------------------------------------------------------------------

def _win32_save_file(title: str, filter_text: str,
                     initial_dir: str, default_filename: str,
                     default_ext: str) -> Optional[str]:
    """Win32 GetSaveFileNameW via ctypes."""
    import ctypes
    import ctypes.wintypes as wt

    OFN_OVERWRITEPROMPT = 0x00000002
    OFN_NOCHANGEDIR     = 0x00000008
    OFN_EXPLORER        = 0x00080000
    MAX_PATH = 1024

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize",       wt.DWORD),
            ("hwndOwner",         wt.HWND),
            ("hInstance",         wt.HINSTANCE),
            ("lpstrFilter",       wt.LPCWSTR),
            ("lpstrCustomFilter", wt.LPWSTR),
            ("nMaxCustFilter",    wt.DWORD),
            ("nFilterIndex",      wt.DWORD),
            ("lpstrFile",         wt.LPWSTR),
            ("nMaxFile",          wt.DWORD),
            ("lpstrFileTitle",    wt.LPWSTR),
            ("nMaxFileTitle",     wt.DWORD),
            ("lpstrInitialDir",   wt.LPCWSTR),
            ("lpstrTitle",        wt.LPCWSTR),
            ("Flags",             wt.DWORD),
            ("nFileOffset",       wt.WORD),
            ("nFileExtension",    wt.WORD),
            ("lpstrDefExt",       wt.LPCWSTR),
            ("lCustData",         ctypes.c_void_p),
            ("lpfnHook",          ctypes.c_void_p),
            ("lpTemplateName",    wt.LPCWSTR),
            ("pvReserved",        ctypes.c_void_p),
            ("dwReserved",        wt.DWORD),
            ("FlagsEx",           wt.DWORD),
        ]

    for ch in '<>:"/\\|?*':
        default_filename = default_filename.replace(ch, '_')

    default_target = os.path.join(initial_dir, default_filename)

    buf = ctypes.create_unicode_buffer(MAX_PATH)
    buf.value = default_target
    ofn = OPENFILENAMEW()
    ofn.lStructSize     = ctypes.sizeof(OPENFILENAMEW)
    ofn.lpstrFilter     = filter_text
    ofn.lpstrFile       = ctypes.cast(buf, wt.LPWSTR)
    ofn.nMaxFile        = MAX_PATH
    ofn.lpstrInitialDir = initial_dir
    ofn.lpstrTitle      = title
    ofn.Flags           = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR | OFN_EXPLORER
    ofn.lpstrDefExt     = default_ext
    ofn.hwndOwner       = ctypes.windll.user32.GetActiveWindow()

    if ctypes.windll.comdlg32.GetSaveFileNameW(ctypes.byref(ofn)):
        return buf.value
    return None


def save_file_dialog(
    title: str = "Save File",
    win32_filter: str = "All files (*.*)\0*.*\0\0",
    initial_dir: str = ".",
    default_filename: str = "Untitled",
    default_ext: str = "",
    tk_filetypes: list | None = None,
) -> Optional[str]:
    """Show a native save-file dialog.  Returns the chosen path or ``None``.

    This is the **single unified entry-point** for all save dialogs in the
    editor.  Platform back-ends can be swapped here without touching callers.
    """
    if sys.platform == "win32":
        try:
            return _win32_save_file(title, win32_filter,
                                    initial_dir, default_filename, default_ext)
        except Exception as exc:
            Debug.log_warning(f"Win32 save dialog failed: {exc}")
            return None

    # Fallback: tkinter
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.asksaveasfilename(
            parent=root,
            title=title,
            initialdir=initial_dir,
            initialfile=default_filename,
            defaultextension=f".{default_ext}" if default_ext else "",
            filetypes=tk_filetypes or [("All Files", "*.*")],
        )
    finally:
        root.destroy()
