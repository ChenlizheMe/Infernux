"""Real worker preparation, source publication and GPU preview observations."""
from pathlib import Path
import os
import time
import uuid

from infernux.core.assets import AssetManager
from infernux.lib import AssetRegistry, InxMaterial


class ShaderPreparationCase:
    modes = ('new_ticket', 'reload', 'cancel', 'unchanged', 'recreate', 'replace_reference', 'cached', 'old_error', 'include')

    def __init__(self, native, directory, mode):
        assert mode in self.modes
        self.native, self.mode = native, mode
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.database = native.get_asset_database()
        self.name = 'Preparation' + uuid.uuid4().hex
        self.fragment = self.directory / 'Color.frag'
        self.material_path = self.directory / 'Color.mat'
        self.assets = [self.fragment, self.material_path]
        self.source = ('#version 450\nShaderInfo {\n Name "' + self.name + '"\n ShadingModel Unlit\n Queue 2000\n}\n'
                       'void surface(out SurfaceData s) { s = InitSurfaceData(); s.albedo = vec3(1.0,0.0,0.0); s.alpha = 1.0; }\n')
        if mode == 'include':
            self.library = self.directory / 'Color.glsl'
            self.assets.insert(0, self.library)
            self.library_source = 'ShaderInfo { Name "' + self.name + 'Library" }\nvec3 color(){return vec3(1.0,0.0,0.0);}\n'
            self.library.write_text(self.library_source, encoding='utf-8')
            assert self.database.import_asset(str(self.library))
            self.source = self.source.replace(' Queue 2000', ' Imports ["' + self.name + 'Library"]\n Queue 2000')
            self.source = self.source.replace('vec3(1.0,0.0,0.0)', 'color()')
        initial = self.source.replace('vec3(1.0,0.0,0.0)', 'notAFunction()') if mode == 'old_error' else self.source
        self.fragment.write_text(initial, encoding='utf-8')
        imported = self.database.import_asset(str(self.fragment))
        assert imported, imported.error
        self.shader_guid = imported.guid
        material = InxMaterial.create_default_unlit()
        material.frag_shader_reference = dict(guid=imported.guid, shader_id=self.name, path_hint='')
        assert material.save_to(str(self.material_path))
        imported = self.database.import_asset(str(self.material_path))
        assert imported, imported.error
        self.material_guid = imported.guid
        self.material = AssetRegistry.instance().load_material(str(self.material_path))
        self.old = native.begin_prepare_linked_shader_programs([self.material_guid])
        self.step = 'warm' if mode == 'cached' else 'old'
        self.deadline = time.monotonic() + 45
        self.proof = dict(mode=mode)
        self.preview_keys = []

    def _change_source(self):
        if self.mode == 'unchanged':
            return
        if self.mode == 'include':
            self.library.write_text(self.library_source.replace('1.0,0.0,0.0', '0.0,0.0,1.0'), encoding='utf-8')
            imported = AssetManager.reimport_asset(str(self.library), database=self.database)
            assert imported, imported.error
            return
        source = self.source.replace('vec3(1.0,0.0,0.0)', 'vec3(0.0,0.0,1.0)')
        if self.mode == 'recreate':
            self.fragment.unlink()
            assert self.database.delete_asset(str(self.fragment))
            self.fragment.write_text(source, encoding='utf-8')
            imported = self.database.import_asset(str(self.fragment))
            assert imported and imported.guid != self.shader_guid, imported.error
            self.material.frag_shader_reference = dict(guid=imported.guid, shader_id=self.name, path_hint='')
        elif self.mode == 'replace_reference':
            self.fragment = self.directory / 'Replacement.frag'
            self.assets.append(self.fragment)
            source = source.replace(self.name, self.name + 'Replacement')
            self.fragment.write_text(source, encoding='utf-8')
            imported = self.database.import_asset(str(self.fragment))
            assert imported, imported.error
            self.material.frag_shader_reference = dict(guid=imported.guid, shader_id=self.name + 'Replacement', path_hint='')
        else:
            self.fragment.write_text(source, encoding='utf-8')
            imported = self.database.reimport_asset(str(self.fragment))
            assert imported, imported.error
        if self.mode in ('recreate', 'replace_reference'):
            # Preview reads the authored material file. Publish its changed
            # reference as well as the in-memory material used by preparation.
            assert self.material.save_to(str(self.material_path))
            imported = self.database.reimport_asset(str(self.material_path))
            assert imported and imported.guid == self.material_guid, imported.error
        self.proof['current_source'] = source

    def _preview(self, suffix):
        key = 'mat|' + self.material_path.as_posix() + '::shader-proof-' + suffix
        if os.name == 'nt':
            key = key.lower()
        if key not in self.preview_keys:
            self.preview_keys.append(key)
        self.native.query_or_schedule_material_preview(key, str(self.material_path), '', 1)
        self.native.pump_preview_tasks()
        self.native.poll_gpu_completions()
        row = next((dict(r) for r in self.native.preview_task_snapshots if r['resource_key'] == key), None)
        if row and row['texture_id'] and row['ready_generation'] == row['generation'] and not row['in_flight']:
            assert row['non_transparent_pixel_count'] > 100, row
            return row
        return None

    def tick(self):
        self.native.request_full_speed_frame()
        assert time.monotonic() < self.deadline, (self.step, self.proof)
        if self.step == 'warm':
            if not self.old.complete:
                return dict(done=False)
            assert self.native.try_commit_linked_shader_programs(self.old)
            self.old = self.native.begin_prepare_linked_shader_programs([self.material_guid])
            assert not self.old.produced_on_worker
            self.step = 'old'
        if self.step == 'old':
            if not self.old.complete:
                return dict(done=False)
            assert self.mode == 'cached' or self.old.produced_on_worker
            assert not self.old.committed
            self._change_source()
            if self.mode in ('reload', 'include'):
                if self.mode == 'reload':
                    imported = AssetManager.reimport_asset(str(self.fragment), database=self.database)
                    assert imported, imported.error
                self.step = 'new_pixels'
            else:
                self.new = self.native.begin_prepare_linked_shader_programs([self.material_guid])
                self.step = 'new'
        if self.step == 'new':
            if not self.new.complete:
                return dict(done=False)
            assert self.new.produced_on_worker, 'A changed source must not reuse a cached old program'
            assert self.native.try_commit_linked_shader_programs(self.new)
            self.step = 'new_pixels'
        if self.step == 'new_pixels':
            row = self._preview('current')
            if row is None:
                return dict(done=False)
            self.proof['current_pixels'] = row
            if self.mode == 'cancel':
                self.old.cancel()
            self.proof['old_commit'] = self.native.try_commit_linked_shader_programs(self.old)
            self.proof['old_committed'] = self.old.committed
            self.step = 'after_pixels'
        if self.step == 'after_pixels':
            row = self._preview('after-old')
            if row is None:
                return dict(done=False)
            self.proof['after_pixels'] = row
            assert row['pixel_hash'] == self.proof['current_pixels']['pixel_hash'], self.proof
            if self.mode not in ('unchanged', 'cached'):
                assert not self.proof['old_commit'] and not self.old.committed, self.proof
                if self.mode != 'cancel':
                    assert self.old.superseded
            assert self.fragment.read_text(encoding='utf-8') == self.proof.get('current_source', self.source)
            return dict(done=True, **self.proof)
        return dict(done=False)

    def close(self):
        self.old.cancel()
        if hasattr(self, 'new'):
            self.new.cancel()
        for path in reversed(self.assets):
            self.native.release_asset_preview_tasks(str(path))
            if path.exists():
                path.unlink()
                assert self.database.delete_asset(str(path))
        for path in self.directory.iterdir():
            assert path.suffix == '.meta', path
            path.unlink()
        self.directory.rmdir()
