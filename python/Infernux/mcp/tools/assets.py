"""Asset/resource creation MCP tools."""

from __future__ import annotations

import os
import shutil

from Infernux.mcp.tools.common import (
    get_asset_database,
    main_thread,
    notify_asset_changed,
    resolve_project_dir,
    resolve_project_path,
)


def register_asset_tools(mcp, project_path: str) -> None:
    @mcp.tool(name="asset.create_builtin_resource")
    def asset_create_builtin_resource(
        kind: str,
        name: str,
        directory: str = "Assets",
        shader_type: str = "frag",
    ) -> dict:
        """Create a built-in resource type: folder, script, material, shader, or scene."""

        def _create():
            return _create_builtin(project_path, kind, name, directory, shader_type)

        return main_thread("asset.create_builtin_resource", _create)

    @mcp.tool(name="asset.create_script")
    def asset_create_script(name: str, directory: str = "Assets") -> dict:
        """Create a Python component script resource from the editor template."""
        return main_thread(
            "asset.create_script",
            lambda: _create_builtin(project_path, "script", name, directory, "frag"),
        )

    @mcp.tool(name="asset.create_material")
    def asset_create_material(name: str, directory: str = "Assets") -> dict:
        """Create a material resource from the editor template."""
        return main_thread(
            "asset.create_material",
            lambda: _create_builtin(project_path, "material", name, directory, "frag"),
        )

    @mcp.tool(name="asset.list")
    def asset_list(
        directory: str = "Assets",
        recursive: bool = True,
        include_meta: bool = False,
        limit: int = 500,
    ) -> dict:
        """List files and directories under the project."""

        def _list():
            root = resolve_project_path(project_path, directory or "Assets")
            entries = []
            max_count = max(int(limit), 1)
            adb = get_asset_database()

            def _entry(path: str) -> dict:
                rel = os.path.relpath(path, project_path).replace("\\", "/")
                data = {
                    "path": rel,
                    "name": os.path.basename(path),
                    "directory": os.path.isdir(path),
                }
                if include_meta and adb and os.path.isfile(path):
                    try:
                        data["guid"] = adb.get_guid_from_path(path) or ""
                    except Exception:
                        data["guid"] = ""
                return data

            if recursive:
                for base, dirs, files in os.walk(root):
                    dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
                    for name in sorted(dirs + files):
                        entries.append(_entry(os.path.join(base, name)))
                        if len(entries) >= max_count:
                            return {"root": os.path.relpath(root, project_path).replace("\\", "/"), "entries": entries}
            else:
                for name in sorted(os.listdir(root)):
                    entries.append(_entry(os.path.join(root, name)))
                    if len(entries) >= max_count:
                        break
            return {"root": os.path.relpath(root, project_path).replace("\\", "/"), "entries": entries}

        return main_thread("asset.list", _list)

    @mcp.tool(name="asset.search")
    def asset_search(query: str, directory: str = "Assets", extensions: list[str] | None = None, limit: int = 100) -> dict:
        """Search asset paths by filename substring and optional extensions."""

        def _search():
            root = resolve_project_path(project_path, directory or "Assets")
            needle = (query or "").lower()
            exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in (extensions or [])}
            matches = []
            for base, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
                for name in files:
                    if needle and needle not in name.lower():
                        continue
                    if exts and os.path.splitext(name)[1].lower() not in exts:
                        continue
                    path = os.path.join(base, name)
                    matches.append(os.path.relpath(path, project_path).replace("\\", "/"))
                    if len(matches) >= max(int(limit), 1):
                        return {"matches": matches}
            return {"matches": matches}

        return main_thread("asset.search", _search)

    @mcp.tool(name="asset.read_text")
    def asset_read_text(path: str, max_bytes: int = 262144) -> dict:
        """Read a UTF-8 text file inside the project."""

        def _read():
            file_path = resolve_project_path(project_path, path)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {path}")
            size = os.path.getsize(file_path)
            if size > max(int(max_bytes), 1):
                raise ValueError(f"File is too large ({size} bytes).")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return {
                "path": os.path.relpath(file_path, project_path).replace("\\", "/"),
                "size": size,
                "text": text,
            }

        return main_thread("asset.read_text", _read)

    @mcp.tool(name="asset.write_text")
    def asset_write_text(path: str, text: str, overwrite: bool = True) -> dict:
        """Write a UTF-8 text file inside the project and notify AssetDatabase."""

        def _write():
            file_path = resolve_project_path(project_path, path)
            existed = os.path.exists(file_path)
            if existed and not overwrite:
                raise FileExistsError(f"File already exists: {path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text or "")
            notify_asset_changed(file_path, "modified" if existed else "created")
            return {
                "path": os.path.relpath(file_path, project_path).replace("\\", "/"),
                "bytes": os.path.getsize(file_path),
                "created": not existed,
            }

        return main_thread("asset.write_text", _write)

    @mcp.tool(name="asset.edit_text")
    def asset_edit_text(path: str, old_text: str, new_text: str, count: int = 1) -> dict:
        """Replace text in a UTF-8 file inside the project."""

        def _edit():
            file_path = resolve_project_path(project_path, path)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {path}")
            with open(file_path, "r", encoding="utf-8") as f:
                original = f.read()
            occurrences = original.count(old_text)
            if occurrences == 0:
                raise ValueError("old_text was not found.")
            replace_count = -1 if int(count) <= 0 else int(count)
            updated = original.replace(old_text, new_text, replace_count)
            with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(updated)
            notify_asset_changed(file_path, "modified")
            return {
                "path": os.path.relpath(file_path, project_path).replace("\\", "/"),
                "occurrences": occurrences,
                "replaced": occurrences if replace_count < 0 else min(occurrences, replace_count),
                "bytes": os.path.getsize(file_path),
            }

        return main_thread("asset.edit_text", _edit)

    @mcp.tool(name="asset.delete")
    def asset_delete(path: str) -> dict:
        """Delete a file or directory inside the project."""

        def _delete():
            target = resolve_project_path(project_path, path)
            if os.path.abspath(target) == os.path.abspath(project_path):
                raise ValueError("Refusing to delete the project root.")
            if not os.path.exists(target):
                raise FileNotFoundError(f"Path not found: {path}")
            is_dir = os.path.isdir(target)
            try:
                from Infernux.engine.ui.project_file_ops import delete_item
                delete_item(target, get_asset_database())
            except Exception:
                if is_dir:
                    shutil.rmtree(target)
                else:
                    os.remove(target)
                notify_asset_changed(target, "deleted")
            return {"deleted": True, "path": os.path.relpath(target, project_path).replace("\\", "/"), "directory": is_dir}

        return main_thread("asset.delete", _delete)

    @mcp.tool(name="asset.refresh")
    def asset_refresh() -> dict:
        """Refresh the AssetDatabase."""

        def _refresh():
            adb = get_asset_database()
            if adb is None:
                raise RuntimeError("AssetDatabase is not available.")
            adb.refresh()
            return {"refreshed": True}

        return main_thread("asset.refresh", _refresh)

    @mcp.tool(name="asset.resolve")
    def asset_resolve(path: str = "", guid: str = "") -> dict:
        """Resolve between asset path and GUID."""

        def _resolve():
            adb = get_asset_database()
            if adb is None:
                raise RuntimeError("AssetDatabase is not available.")
            resolved_path = ""
            resolved_guid = ""
            if guid:
                resolved_path = adb.get_path_from_guid(guid) or ""
                resolved_guid = guid
            elif path:
                file_path = resolve_project_path(project_path, path)
                resolved_guid = adb.get_guid_from_path(file_path) or ""
                resolved_path = file_path
            else:
                raise ValueError("Provide path or guid.")
            return {
                "path": os.path.relpath(resolved_path, project_path).replace("\\", "/") if resolved_path else "",
                "guid": resolved_guid,
            }

        return main_thread("asset.resolve", _resolve)



def _create_builtin(project_path: str, kind: str, name: str, directory: str, shader_type: str) -> dict:
    from Infernux.engine.ui import project_file_ops as ops

    target_dir = resolve_project_dir(project_path, directory)
    adb = get_asset_database()
    normalized = kind.strip().lower()
    if normalized == "folder":
        success, message = ops.create_folder(target_dir, name)
        path = os.path.join(target_dir, name.strip())
    elif normalized == "script":
        success, message = ops.create_script(target_dir, name, adb)
        file_name = name if name.endswith(".py") else name + ".py"
        path = os.path.join(target_dir, file_name)
    elif normalized == "material":
        success, message = ops.create_material(target_dir, name, adb)
        base = name[:-4] if name.endswith(".mat") else name
        path = os.path.join(target_dir, base + ".mat")
    elif normalized == "shader":
        success, message = ops.create_shader(target_dir, name, shader_type, adb)
        base = name
        for ext in (".vert", ".frag", ".glsl"):
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
        path = os.path.join(target_dir, base + "." + shader_type)
    elif normalized == "scene":
        success, message = ops.create_scene(target_dir, name, adb)
        path = message if success else ""
    else:
        raise ValueError("kind must be one of: folder, script, material, shader, scene")

    if not success:
        raise RuntimeError(message or f"Failed to create {kind}.")

    guid = ""
    if adb and path and os.path.isfile(path):
        try:
            guid = adb.get_guid_from_path(path) or ""
        except Exception:
            guid = ""
    return {
        "kind": normalized,
        "name": name,
        "path": path,
        "guid": guid,
    }
