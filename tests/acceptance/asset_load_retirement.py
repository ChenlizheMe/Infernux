"""Real Assimp worker publication across project mutations, reusable by MCP."""
from pathlib import Path

from infernux.core.assets import AssetManager
from infernux.lib import AssetRegistry


def triangle(width):
    return f'o Triangle\nv 0 0 0\nv {width} 0 0\nv 0 1 0\nf 1 2 3\n'


class MeshLoadRetirementCase:
    def __init__(self, native, directory):
        self.native = native
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.path = self.directory / 'Triangle.obj'
        self.path.write_text(triangle(1), encoding='utf-8')
        self.database = native.get_asset_database()
        self.registry = AssetRegistry.instance()
        imported = AssetManager.import_asset(str(self.path), database=self.database)
        assert imported, imported.error
        self.guid = imported.guid
        assert not self.registry.is_loaded(self.guid)
        self.ticket = self.registry.begin_load_mesh_by_guid(self.guid)

    def ready(self):
        return self.ticket.complete

    def exercise(self, mutation, allow_stale):
        assert self.ticket.complete and self.ticket.produced_on_worker, (mutation, self.ticket.complete, self.ticket.produced_on_worker)
        meta = Path(str(self.path) + '.meta')
        saved_meta = meta.read_bytes()
        if mutation in ('delete', 'restore'):
            self.path.unlink()
            result = AssetManager.delete_asset(str(self.path), database=self.database)
            assert result, result.error
            assert not self.registry.is_loaded(self.guid), ('delete resident', self.guid)
            assert not self.database.contains_guid(self.guid), ('delete catalog', self.guid)
            if mutation == 'restore':
                self.path.write_text(triangle(3), encoding='utf-8')
                meta.write_bytes(saved_meta)
                result = AssetManager.import_asset(str(self.path), database=self.database)
                assert result and result.guid == self.guid, ('restore identity', result.guid, self.guid, result.error)
        elif mutation in ('reimport', 'native-reload'):
            self.path.write_text(triangle(3), encoding='utf-8')
            if mutation == 'native-reload':
                result = self.database.reimport_asset(str(self.path))
                assert not self.registry.is_loaded(self.guid)
                self.native.reload_mesh(str(self.path))
            else:
                result = AssetManager.reimport_asset(str(self.path), database=self.database)
            assert result and result.guid == self.guid
        elif mutation == 'invalidate':
            self.registry.invalidate_asset(self.guid)
        else:
            assert mutation == 'unchanged'

        version = self.registry.get_asset_version(self.guid)
        resident = self.registry.get_mesh(self.guid)
        rejected = False
        try:
            committed = self.registry.try_commit_asset_load(self.ticket, allow_stale_if_unloaded=allow_stale)
        except RuntimeError as error:
            assert 'stale' in str(error), str(error)
            rejected, committed = True, False
        if mutation == 'unchanged':
            assert committed and not rejected and self.ticket.committed
        else:
            assert rejected and not committed and not self.ticket.committed, (mutation,allow_stale,rejected,committed,self.ticket.committed)
            assert self.registry.get_asset_version(self.guid) == version, ('version changed',self.guid,version)
            assert self.registry.get_mesh(self.guid) is resident, ('resident changed',self.guid)
        if mutation == 'delete':
            assert self.registry.get_mesh(self.guid) is None
            assert not self.database.contains_guid(self.guid)
            width = None
        else:
            current = self.registry.load_mesh_by_guid(self.guid)
            assert current is not None
            bounds = current.get_bounds()
            width = float(bounds[3] - bounds[0])
            expected = 3. if mutation in ('restore','reimport','native-reload') else 1.
            assert abs(width - expected) < 1.e-6, (width, expected)
        return dict(mutation=mutation,allow_stale=allow_stale,rejected=rejected,
                    committed=committed,guid=self.guid,width=width,worker=self.ticket.produced_on_worker)

    def close(self):
        if self.path.exists():
            self.path.unlink()
        if self.database.contains_guid(self.guid):
            result = AssetManager.delete_asset(str(self.path), database=self.database)
            assert result, result.error
        for path in self.directory.iterdir():
            assert path.name == 'Triangle.obj.meta', path
            path.unlink()
        self.directory.rmdir()
