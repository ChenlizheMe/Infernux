"""Imported model/submesh previews with real material and texture publications."""
from pathlib import Path
import copy
import os

from PIL import Image

from infernux.core.assets import AssetManager
from infernux.core.asset_types import read_mesh_import_settings
from infernux.engine.ui.asset_resource_preview import _try_get_cpp_mesh_preview
from infernux.engine.undo import MaterialDocumentCommand, UndoManager
from infernux.lib import AssetRegistry, InxMaterial
from infernux.lib._Infernux import make_model_mesh_reference


class MeshPreviewDependencyCase:
    modes = ('edit', 'history', 'material_import', 'texture_import', 'unrelated', 'snapshot')

    def __init__(self, native, directory, mode, consumer='python'):
        assert mode in self.modes and consumer in ('python', 'native', 'project')
        self.native, self.mode, self.consumer = native, mode, consumer
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.material_path = self.directory / 'Paint.mat'
        self.source = self.directory / 'Triangle.obj'
        self.mtl = self.directory / 'Triangle.mtl'
        self.texture = self.directory / 'Color.png'
        self.other = self.directory / 'Unrelated.mat'
        self.assets = []
        self.mtl.write_text('newmtl Paint\nKd 1 0 0\n', encoding='ascii')
        self.source.write_text('mtllib Triangle.mtl\no Panel\nv -1.0 -1.0 0.0\nv 1.0 -1.0 0.0\nv 0.0 1.0 0.0\n'
            'vt 0.0 0.0\nvt 1.0 0.0\nvt 0.5 1.0\nvn 0.0 0.0 1.0\nusemtl Paint\nf 1/1/1 2/2/1 3/3/1\n', encoding='ascii')
        material = InxMaterial.create_default_unlit()
        if mode == 'texture_import':
            Image.new('RGB', (8,8), 'red').save(self.texture)
            imported = AssetManager.import_asset(str(self.texture))
            assert imported, imported.error
            self.assets.append(self.texture)
            material.set_texture_guid('texSampler', imported.guid)
        else:
            material.set_vector4('baseColor', 1., 0., 0., 1.)
        assert material.save_to(str(self.material_path))
        imported = AssetManager.import_asset(str(self.material_path))
        assert imported, imported.error
        self.assets.append(self.material_path)
        self.material_guid = imported.guid
        if mode == 'unrelated':
            assert material.save_to(str(self.other))
            assert AssetManager.import_asset(str(self.other))
            self.assets.append(self.other)
        imported = AssetManager.import_asset(str(self.source))
        assert imported, imported.error
        self.assets.append(self.source)
        self.registry = AssetRegistry.instance()
        mesh = self.registry.load_mesh(str(self.source))
        settings = read_mesh_import_settings(str(self.source))
        settings.material_remaps = {slot['source_id']: self.material_guid for slot in mesh.get_material_slot_data()}
        imported = AssetManager.reimport_asset(str(self.source), import_settings=settings.to_dict())
        assert imported, imported.error
        mesh = self.registry.load_mesh(str(self.source))
        self.geometry_generation = mesh.generation
        index = next(i for i, node in enumerate(mesh.get_model_nodes()) if node['node_group'] >= 0)
        self.paths = dict(model=self.source.as_posix(), submesh=make_model_mesh_reference(self.source.as_posix(), mesh.get_model_node_path(index)))
        self.keys = {name: ('mesh|' + path).lower() if os.name == 'nt' else 'mesh|' + path for name,path in self.paths.items()}
        self.material = self.registry.load_material(str(self.material_path))
        self.history = UndoManager()
        self.step, self.frames = 'baseline', 0
        self.phase = 0
        self.observations = []
        self.write_ticket = None

    def change(self):
        if self.mode == 'history':
            if self.phase == 0:
                old = self.material.serialize_document()
                new = copy.deepcopy(old)
                new['properties']['baseColor']['value'] = [0., 0., 1., 1.]
                assert self.history.execute(MaterialDocumentCommand(self.material, old, new), raise_errors=True)
            elif self.phase == 1:
                self.history.undo()
            else:
                self.history.redo()
        elif self.mode == 'texture_import':
            Image.new('RGB', (8,8), 'blue').save(self.texture)
            result = AssetManager.reimport_asset(str(self.texture))
            assert result, result.error
        elif self.mode == 'material_import':
            authored = InxMaterial.create_default_unlit()
            authored.set_vector4('baseColor', 0., 0., 1., 1.)
            assert authored.save_to(str(self.material_path))
            result = AssetManager.reimport_asset(str(self.material_path))
            assert result, result.error
        else:
            target = self.registry.load_material(str(self.other)) if self.mode == 'unrelated' else self.material
            target.set_vector4('baseColor', 0., 0., 1., 1.)
            if self.mode == 'snapshot':
                self.write_ticket = AssetManager._submit_document_snapshot(str(self.material_path), target.serialize())
            else:
                AssetManager.note_asset_edit(str(self.other if self.mode == 'unrelated' else self.material_path), material_json=target.serialize())

    def tick(self):
        AssetManager.poll_pending_asset_writes()
        self.native.request_full_speed_frame()
        for name,path in self.paths.items():
            if self.consumer == 'python':
                _try_get_cpp_mesh_preview(self.native, path)
            elif self.consumer == 'native':
                self.native.query_or_schedule_mesh_preview(self.keys[name], path, 0)
            # Project mode is requested only by actual Project grid rendering.
        self.native.pump_preview_tasks()
        self.native.poll_gpu_completions()
        snapshots = {r['resource_key']: dict(r) for r in self.native.preview_task_snapshots}
        rows = {name: snapshots.get(key) for name,key in self.keys.items()}
        if not all(r and r['texture_id'] and r['generation'] == r['ready_generation'] and not r['in_flight'] for r in rows.values()):
            return dict(done=False, step=self.step)
        self.frames += 1
        if self.step == 'baseline' and self.frames >= 5:
            assert all(r['non_transparent_pixel_count'] > 100 for r in rows.values())
            assert rows['model']['pixel_hash'] == rows['submesh']['pixel_hash'], rows
            self.before = rows
            self.observations.append(rows)
            self.change()
            self.frames, self.step = 0, 'changed'
        elif self.step == 'changed' and self.frames >= 25:
            assert rows['model']['pixel_hash'] == rows['submesh']['pixel_hash'], rows
            for name,row in rows.items():
                before = self.before[name]
                if self.mode == 'unrelated':
                    assert (row['generation'],row['pixel_hash']) == (before['generation'],before['pixel_hash']), (name,rows,self.before)
                else:
                    assert row['generation'] > before['generation'] and row['pixel_hash'] != before['pixel_hash'], (name,rows,self.before)
            assert self.registry.load_mesh(str(self.source)).generation == self.geometry_generation
            self.observations.append(rows)
            self.before = rows
            self.phase += 1
            self.frames = 0
            if self.mode == 'history' and self.phase < 3:
                self.change()
            else:
                self.step = 'stable'
        elif self.step == 'stable' and self.frames >= 30:
            if self.write_ticket is not None and not self.write_ticket.is_complete:
                return dict(done=False,step='waiting-for-document-write')
            if self.write_ticket is not None:
                assert self.write_ticket.status == 'succeeded', self.write_ticket.error
            assert all(rows[n]['generation'] == self.before[n]['generation'] and rows[n]['pixel_hash'] == self.before[n]['pixel_hash'] for n in rows)
            if self.mode == 'history':
                for n in rows:
                    assert self.observations[0][n]['pixel_hash'] == self.observations[2][n]['pixel_hash']
                    assert self.observations[1][n]['pixel_hash'] == self.observations[3][n]['pixel_hash']
            return dict(done=True, mode=self.mode, consumer=self.consumer, observations=self.observations)
        return dict(done=False, step=self.step)

    def close(self):
        if self.write_ticket is not None:
            self.write_ticket.wait()
            AssetManager.poll_pending_asset_writes()
        self.history.clear()
        for path in reversed(self.assets):
            self.native.release_asset_preview_tasks(str(path))
            path.unlink()
            assert AssetManager.delete_asset(str(path))
        self.mtl.unlink()
        for path in self.directory.iterdir():
            assert path.suffix == '.meta', path
            path.unlink()
        self.directory.rmdir()
