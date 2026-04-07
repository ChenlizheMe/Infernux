"""
Pure utility functions and constants for the Project panel.

These have no dependency on ``ProjectPanel`` instance state.
"""

import os
from Infernux.debug import Debug

# File extensions to hide
HIDDEN_EXTENSIONS = {'.meta', '.pyc', '.pyo', '.tmp'}
HIDDEN_PREFIXES = {'.', '__'}
HIDDEN_FILES = {'imgui.ini'}


def should_show(name: str) -> bool:
    """Check if a file or folder should be shown (filters hidden files)."""
    if name in HIDDEN_FILES:
        return False
    for prefix in HIDDEN_PREFIXES:
        if name.startswith(prefix):
            return False
    _, ext = os.path.splitext(name)
    if ext.lower() in HIDDEN_EXTENSIONS:
        return False
    return True


def _find_vscode_executable() -> str | None:
    """Locate the VS Code CLI executable on the current platform.

    On Windows ``code.cmd`` is often *not* on the system PATH even when
    VS Code is installed. This helper checks (in order):

    1. ``shutil.which('code')`` — works if the user ticked "Add to PATH"
       during install.
    2. Common installation directories (User & System installs).
    3. The Windows Registry ``App Paths`` key that VS Code registers.

    Returns the full path to the executable/script, or *None* if VS Code
    cannot be found.
    """
    import shutil
    import platform

    # Fast path: already on PATH
    found = shutil.which('code') or shutil.which('code.cmd') or shutil.which('code.exe')
    if found:
        return found

    if platform.system() != 'Windows':
        return None  # macOS/Linux typically have `code` symlinked

    # --- Windows-specific search -------------------------------------------

    # Common install locations (User install → System install)
    candidates = []
    local = os.environ.get('LOCALAPPDATA', '')
    if local:
        candidates.append(os.path.join(local, 'Programs', 'Microsoft VS Code', 'bin', 'code.cmd'))
        candidates.append(os.path.join(local, 'Programs', 'Microsoft VS Code', 'Code.exe'))
    program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
    candidates.append(os.path.join(program_files, 'Microsoft VS Code', 'bin', 'code.cmd'))
    candidates.append(os.path.join(program_files, 'Microsoft VS Code', 'Code.exe'))
    program_files_x86 = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
    candidates.append(os.path.join(program_files_x86, 'Microsoft VS Code', 'bin', 'code.cmd'))
    candidates.append(os.path.join(program_files_x86, 'Microsoft VS Code', 'Code.exe'))

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Last resort: check Windows Registry (App Paths)
    import winreg
    registry_roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
    registry_paths = (
        r'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\code.exe',
        r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{EA457B21-F73E-494C-ACAB-524FDE069978}_is1',
    )
    for root in registry_roots:
        for key_path in registry_paths:
            try:
                key = winreg.OpenKey(root, key_path)
            except OSError as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                continue
            try:
                exe_path, _ = winreg.QueryValueEx(key, '')
            except OSError:
                try:
                    install_location, _ = winreg.QueryValueEx(key, 'InstallLocation')
                    exe_path = os.path.join(install_location, 'Code.exe')
                except OSError:
                    exe_path = None
            finally:
                winreg.CloseKey(key)
            if exe_path and os.path.isfile(exe_path):
                return exe_path

    return None


def open_in_vscode(file_path: str, line: int = 0, project_root: str = "") -> bool:
    """Open a file in VS Code and optionally jump to a line.

    Returns ``True`` when a VS Code launch was attempted successfully.
    """
    import platform
    import subprocess

    if not file_path:
        return False

    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        return False

    code_exe = _find_vscode_executable()
    if not code_exe:
        return False

    target = f"{file_path}:{max(int(line), 1)}" if line and int(line) > 0 else file_path

    cmd = []
    if project_root:
        project_root = os.path.abspath(project_root)
        if os.path.isdir(project_root):
            cmd.append(project_root)
    cmd.extend(['--goto', target])

    try:
        if platform.system() == 'Windows' and code_exe.lower().endswith('.cmd'):
            subprocess.Popen(
                ['cmd.exe', '/c', code_exe, *cmd],
                shell=False,
                creationflags=0x08000000,
            )
        else:
            subprocess.Popen(
                [code_exe, *cmd],
                shell=False,
                creationflags=(0x08000000 if platform.system() == 'Windows' else 0),
            )
        return True
    except (OSError, subprocess.SubprocessError) as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return False


def open_file_with_system(file_path: str, project_root: str = ""):
    """
    Open *file_path* with the OS default application.

    For ``.py``, ``.vert``, ``.frag``, ``.glsl``, ``.hlsl``, ``.json``,
    ``.txt``, and ``.md`` files, open in VS Code with the *project_root*
    as the workspace folder — so that the project's Python runtime
    interpreter and type stubs are automatically picked up by Pylance.
    """
    import subprocess
    import platform

    CODE_EXTENSIONS = {
        '.py', '.vert', '.frag', '.glsl', '.hlsl',
        '.json', '.txt', '.md', '.yaml', '.yml', '.xml',
        '.lua', '.cs', '.cpp', '.c', '.h',
    }

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # For code files, try to open in VS Code with the project workspace
    if ext in CODE_EXTENSIONS and project_root:
        if open_in_vscode(file_path, project_root=project_root):
            return
        print("[ProjectPanel] VS Code not found, falling back to system default")

    # Fallback: open with OS default application
    system = platform.system()
    if system == 'Windows':
        os.startfile(file_path)
    elif system == 'Darwin':
        subprocess.run(['open', file_path], check=True)
    else:
        subprocess.run(['xdg-open', file_path], check=True)


def get_file_type(filename: str) -> str:
    """Return a short type tag string (e.g. ``[PY]``, ``[IMG]``) based on extension."""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    types = {
        '.png': '[IMG]', '.jpg': '[IMG]', '.jpeg': '[IMG]', '.bmp': '[IMG]',
        '.tga': '[IMG]', '.gif': '[IMG]',
        '.py': '[PY]', '.lua': '[LUA]', '.cs': '[CS]', '.cpp': '[CPP]',
        '.h': '[H]', '.c': '[C]',
        '.vert': '[VERT]', '.frag': '[FRAG]', '.glsl': '[GLSL]', '.hlsl': '[HLSL]',
        '.mat': '[MAT]',
        '.fbx': '[3D]', '.obj': '[3D]', '.gltf': '[3D]', '.glb': '[3D]',
        '.wav': '[SND]',
        '.json': '[JSON]', '.yaml': '[CFG]', '.yml': '[CFG]', '.xml': '[XML]',
        '.txt': '[TXT]', '.md': '[MD]',
        '.ttf': '[FNT]', '.otf': '[FNT]',
    }
    return types.get(ext, '[FILE]')


def update_material_name_in_file(mat_path: str, new_name: str):
    """Rewrite the ``"name"`` key in a ``.mat`` JSON file."""
    import json
    with open(mat_path, 'r', encoding='utf-8') as f:
        mat_data = json.load(f)
    mat_data['name'] = new_name
    with open(mat_path, 'w', encoding='utf-8') as f:
        json.dump(mat_data, f, indent=2)


def reveal_in_file_explorer(path: str):
    """Open the system file explorer and highlight *path*.

    On Windows uses ``explorer /select,<path>``.
    On macOS uses ``open -R``.
    On Linux falls back to ``xdg-open`` on the parent directory.
    """
    import platform
    import subprocess

    path = os.path.abspath(path)
    system = platform.system()
    if system == 'Windows':
        # /select highlights the file/folder in Explorer
        subprocess.Popen(['explorer', '/select,', path])
    elif system == 'Darwin':
        subprocess.Popen(['open', '-R', path])
    else:
        # xdg-open on parent dir
        parent = os.path.dirname(path) if os.path.isfile(path) else path
        subprocess.Popen(['xdg-open', parent])
