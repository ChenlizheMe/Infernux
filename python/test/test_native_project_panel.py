"""Tests for native ProjectPanel."""
import os
import tempfile
import pytest
from Infernux.lib import ProjectPanel


class TestProjectPanelCreation:

    def test_creation(self):
        pp = ProjectPanel()
        assert pp is not None

    def test_is_editor_panel(self):
        from Infernux.lib import EditorPanel
        pp = ProjectPanel()
        assert isinstance(pp, EditorPanel)

    def test_window_id(self):
        pp = ProjectPanel()
        assert pp.get_window_id() == "project"

    def test_default_open(self):
        pp = ProjectPanel()
        assert pp.is_open()


class TestProjectPanelPaths:

    def test_set_root_path(self):
        pp = ProjectPanel()
        with tempfile.TemporaryDirectory() as d:
            pp.set_root_path(d)
            # No crash

    def test_get_set_current_path(self):
        pp = ProjectPanel()
        with tempfile.TemporaryDirectory() as d:
            pp.set_root_path(d)
            pp.set_current_path(d)
            assert pp.get_current_path() == d

    def test_set_current_path_empty(self):
        pp = ProjectPanel()
        pp.set_current_path("")
        assert pp.get_current_path() == ""

    def test_set_icons_directory(self):
        pp = ProjectPanel()
        with tempfile.TemporaryDirectory() as d:
            pp.set_icons_directory(d)
            # No crash


class TestProjectPanelCallbacks:

    def test_translate_callback(self):
        pp = ProjectPanel()
        pp.translate = lambda key: f"[{key}]"
        assert pp.translate("project.create_folder") == "[project.create_folder]"

    def test_on_file_selected_callback(self):
        pp = ProjectPanel()
        received = []
        pp.on_file_selected = lambda path: received.append(path)
        assert pp.on_file_selected is not None

    def test_on_empty_area_clicked_callback(self):
        pp = ProjectPanel()
        called = []
        pp.on_empty_area_clicked = lambda: called.append(True)
        assert pp.on_empty_area_clicked is not None

    def test_on_state_changed_callback(self):
        pp = ProjectPanel()
        called = []
        pp.on_state_changed = lambda: called.append(True)
        assert pp.on_state_changed is not None

    def test_create_folder_callback(self):
        pp = ProjectPanel()
        results = []
        pp.create_folder = lambda cur, name: (
            results.append((cur, name)) or (True, "")
        )
        ok, err = pp.create_folder("/path", "NewFolder")
        assert results == [("/path", "NewFolder")]

    def test_create_script_callback(self):
        pp = ProjectPanel()
        pp.create_script = lambda cur, name: (True, "")
        ok, err = pp.create_script("/path", "MyScript")
        assert ok is True

    def test_create_shader_callback(self):
        pp = ProjectPanel()
        pp.create_shader = lambda cur, name, typ: (True, "")
        ok, err = pp.create_shader("/path", "MyShader", "unlit")
        assert ok is True

    def test_create_material_callback(self):
        pp = ProjectPanel()
        pp.create_material = lambda cur, name: (True, "")
        ok, err = pp.create_material("/path", "MyMat")
        assert ok is True

    def test_create_scene_callback(self):
        pp = ProjectPanel()
        pp.create_scene = lambda cur, name: (True, "")
        ok, err = pp.create_scene("/path", "Main")
        assert ok is True

    def test_delete_items_callback(self):
        pp = ProjectPanel()
        deleted = []
        pp.delete_items = lambda paths: deleted.extend(paths)
        pp.delete_items(["/a", "/b"])
        assert deleted == ["/a", "/b"]

    def test_do_rename_callback(self):
        pp = ProjectPanel()
        pp.do_rename = lambda old, new_name: f"/dir/{new_name}"
        result = pp.do_rename("/dir/old.txt", "new.txt")
        assert result == "/dir/new.txt"

    def test_get_unique_name_callback(self):
        pp = ProjectPanel()
        pp.get_unique_name = lambda cur, base, ext: f"{base}_1{ext}"
        result = pp.get_unique_name("/dir", "File", ".txt")
        assert result == "File_1.txt"

    def test_move_item_to_directory_callback(self):
        pp = ProjectPanel()
        pp.move_item_to_directory = lambda item, dest: f"{dest}/moved"
        result = pp.move_item_to_directory("/a/b.txt", "/c")
        assert result == "/c/moved"

    def test_open_file_callback(self):
        pp = ProjectPanel()
        opened = []
        pp.open_file = lambda path: opened.append(path)
        pp.open_file("/test.py")
        assert opened == ["/test.py"]

    def test_open_scene_callback(self):
        pp = ProjectPanel()
        opened = []
        pp.open_scene = lambda path: opened.append(path)
        pp.open_scene("/test.scene")
        assert opened == ["/test.scene"]

    def test_open_prefab_mode_callback(self):
        pp = ProjectPanel()
        opened = []
        pp.open_prefab_mode = lambda path: opened.append(path)
        pp.open_prefab_mode("/test.prefab")
        assert opened == ["/test.prefab"]

    def test_reveal_in_explorer_callback(self):
        pp = ProjectPanel()
        revealed = []
        pp.reveal_in_explorer = lambda path: revealed.append(path)
        pp.reveal_in_explorer("/dir")
        assert revealed == ["/dir"]

    def test_validate_script_component_callback(self):
        pp = ProjectPanel()
        pp.validate_script_component = lambda path: path.endswith(".py")
        assert pp.validate_script_component("/test.py") is True
        assert pp.validate_script_component("/test.txt") is False

    def test_guid_callbacks(self):
        pp = ProjectPanel()
        pp.get_guid_from_path = lambda path: "guid-123" if path else ""
        pp.get_path_from_guid = lambda guid: "/test.txt" if guid else ""

        assert pp.get_guid_from_path("/test.txt") == "guid-123"
        assert pp.get_path_from_guid("guid-123") == "/test.txt"

    def test_invalidate_asset_inspector_callback(self):
        pp = ProjectPanel()
        invalidated = []
        pp.invalidate_asset_inspector = lambda path: invalidated.append(path)
        pp.invalidate_asset_inspector("/asset.mat")
        assert invalidated == ["/asset.mat"]

    def test_create_prefab_from_hierarchy_callback(self):
        pp = ProjectPanel()
        created = []
        pp.create_prefab_from_hierarchy = lambda oid, path: created.append((oid, path))
        pp.create_prefab_from_hierarchy(42, "/Assets")
        assert created == [(42, "/Assets")]


class TestProjectPanelPublicAPI:

    def test_clear_selection(self):
        pp = ProjectPanel()
        pp.clear_selection()  # No crash

    def test_set_selected_file(self):
        pp = ProjectPanel()
        pp.set_selected_file("/tmp/test.mat")
        # No crash — used by selection undo replay

    def test_invalidate_material_thumbnail(self):
        pp = ProjectPanel()
        pp.invalidate_material_thumbnail("/path/to/mat.mat")
        # No crash — clears internal thumbnail cache entry

    def test_set_open(self):
        pp = ProjectPanel()
        pp.set_open(False)
        assert not pp.is_open()
        pp.set_open(True)
        assert pp.is_open()
