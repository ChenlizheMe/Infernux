"""Delete/recreate a model while the native Project thumbnail owns work."""
import os
from pathlib import Path

from infernux.core.assets import AssetManager
from infernux.lib import AssetRegistry
from tests.acceptance.asset_load_retirement import triangle


class MeshPreviewRetirementCase:
    def __init__(self, native, directory, stage, recreate):
        assert stage in ('queued','loading','ready')
        self.native, self.stage, self.recreate = native, stage, recreate
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.path = self.directory/'Triangle.obj'
        self.path.write_text(triangle(1),encoding='utf-8')
        result = AssetManager.import_asset(str(self.path))
        assert result, result.error
        self.old_guid = result.guid
        self.new_guid = ''
        self.key = 'mesh|'+self.path.as_posix()
        if os.name == 'nt':
            self.key = self.key.lower()
        native.query_or_schedule_mesh_preview(self.key,str(self.path),1)
        self.retired = False
        self.frames_after_retirement = 0
        if stage == 'queued':
            self.retire()

    def retire(self):
        self.path.unlink()
        assert AssetManager.delete_asset(str(self.path))
        assert self.native.get_mesh_preview_texture_id(self.key) == 0
        state = next(r for r in self.native.preview_task_snapshots if r['resource_key'] == self.key)
        assert not state['in_flight'] and not state['has_render_ticket'] and not state['pending_upload_version'], state
        self.retired = True
        if self.recreate:
            self.path.write_text(triangle(3),encoding='utf-8')
            result = AssetManager.import_asset(str(self.path))
            assert result and result.guid != self.old_guid
            self.new_guid = result.guid
            self.native.query_or_schedule_mesh_preview(self.key,str(self.path),2)

    def tick(self):
        self.native.request_full_speed_frame()
        self.native.pump_preview_tasks()
        if not self.retired:
            if self.stage == 'loading' or self.native.get_mesh_preview_texture_id(self.key):
                self.retire()
        else:
            self.frames_after_retirement += 1
            registry = AssetRegistry.instance()
            assert registry.get_mesh(self.old_guid) is None
            texture = self.native.get_mesh_preview_texture_id(self.key)
            if not self.recreate:
                assert texture == 0
            if self.frames_after_retirement >= 20 and (not self.recreate or texture):
                if self.recreate:
                    bounds = registry.get_mesh(self.new_guid).get_bounds()
                    assert abs(bounds[3]-bounds[0]-3.) < 1.e-6
                return dict(done=True,stage=self.stage,recreate=self.recreate,texture=texture,
                            old_guid=self.old_guid,new_guid=self.new_guid)
        return dict(done=False)

    def close(self):
        if self.path.exists():
            self.path.unlink()
            assert AssetManager.delete_asset(str(self.path))
        for path in self.directory.iterdir():
            assert path.name == 'Triangle.obj.meta', path
            path.unlink()
        self.directory.rmdir()
