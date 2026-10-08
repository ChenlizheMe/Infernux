"""Real imported resources and native particles for publication acceptance."""
import json
from pathlib import Path

from infernux.components import ParticleSystem
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.graph import AssetReference, GraphDocument, GraphNodeRecord, GraphLinkRecord, PortKind
from infernux.lib import AssetRegistry
from infernux.particle import ParticleGraphAsset, ParticleEmitterAsset, ParticleParameter, EmitterSettings
from infernux.particle.artifact import ParticleArtifactRegistry


def rendering(kind, properties):
    return GraphDocument('particle.rendering', nodes=(
        GraphNodeRecord('root.rendering', 'particle.root.rendering'),
        GraphNodeRecord('output', 'particle.output.' + kind, properties=properties),
    ), links=(GraphLinkRecord('render', 'root.rendering', 'out', 'output', 'in', PortKind.EXEC),))


class ParticlePublicationCase:
    def __init__(self, native, scene, directory):
        self.native, self.scene = native, scene
        self.database = native.get_asset_database()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mesh = self.directory / 'PublicationShard.obj'
        self.mesh_source = 'o Shard\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'
        self.mesh.write_text(self.mesh_source, encoding='ascii')
        imported = AssetManager.import_asset(str(self.mesh), database=self.database)
        assert imported.succeeded, imported.error
        self.mesh_guid = imported.guid
        self.mesh_meta = Path(str(self.mesh) + '.meta').read_bytes()
        assert AssetRegistry.instance().load_mesh_by_guid(self.mesh_guid) is not None
        self.settings = EmitterSettings(capacity=8, spawn_rate=0.0)
        self.keep = ParticleParameter(stable_id='keep', name='Keep', default=1.0)
        self.first = ParticleGraphAsset(stable_id='publication-atomicity', name='Publication',
            parameters=(self.keep, ParticleParameter(stable_id='removed', name='Removed', default=2.0)),
            emitters=(
                ParticleEmitterAsset(stable_id='emitter-keep', name='Keep', settings=self.settings,
                    rendering=rendering('sprite', {'shader.baseColor':[1.,0.,0.,1.]})),
                ParticleEmitterAsset(stable_id='emitter-removed', name='Removed', settings=self.settings),
            ))
        self.source = self.directory / 'Publication.particlegraph'
        self.first.save(str(self.source))
        imported = AssetManager.import_asset(str(self.source), database=self.database)
        assert imported.succeeded, imported.error
        self.guid = imported.guid
        self.owner = scene.create_game_object('Particle Publication Owner')
        self.component = self.owner.add_py_component(ParticleSystem())
        self.component.graph = ParticleGraphRef(guid=self.guid)
        assert self.component.editor_preview_begin(), self.component.last_compile_error
        self.component.set_parameter('Removed', 7.0)
        self.component.set_parameter('Keep', 5.0)
        assert self.component.set_emitter_options('emitter-removed', enabled=False)
        self.material_key = ('emitter-keep', 'output')

    def snapshot(self):
        component = self.component
        material = component._output_materials[self.material_key]
        return dict(parameter_json=component._parameter_overrides_json,
            emitter_json=component._emitter_overrides_json,
            parameter_cache=component._serialized_parameter_overrides_cache,
            emitter_cache=component._serialized_emitter_overrides_cache,
            parameters=[item.stable_id for item in component._particle_metadata.parameters],
            emitters=[item.stable_id for item in component._particle_metadata.emitters],
            artifact_revision=component._artifact_revision,
            native_revisions={str(i):self.native._gpu_particle_artifact_revision(i) for i in component._gpu_emitter_ids},
            color=list(material.native.get_color('baseColor')),
            controller_ids=[id(item) for item in component._gpu_controllers],
            material_id=id(material), metadata_id=id(component._particle_metadata),
            kernel_id=id(component._particle_kernel),
            prewarm=sorted(component._prewarm_pending_emitters), seek=dict(component._pending_seek_seconds))

    def replacement(self, *, empty_parameters=False):
        self.second = ParticleGraphAsset(stable_id=self.first.stable_id, name=self.first.name,
            parameters=() if empty_parameters else (self.keep,), emitters=(
                ParticleEmitterAsset(stable_id='emitter-keep', name='Keep', settings=self.settings,
                    rendering=rendering('sprite', {'shader.baseColor':[0.,1.,0.,1.]})),
                ParticleEmitterAsset(stable_id='emitter-new', name='Mesh', settings=self.settings,
                    rendering=rendering('mesh', {'mesh':AssetReference(guid=self.mesh_guid).to_dict()})),
            ))
        ParticleArtifactRegistry.save_graph_asset(self.second, str(self.source), guid=self.guid)
        artifact = ParticleArtifactRegistry.get(guid=self.guid)
        assert artifact is not None and artifact.revision > self.component._artifact_revision
        return artifact.revision

    def delete_dependency(self):
        result = AssetManager.delete_asset(str(self.mesh), database=self.database)
        assert result.succeeded, result.error
        assert AssetRegistry.instance().load_mesh_by_guid(self.mesh_guid) is None

    def restore_dependency(self):
        self.mesh.write_text(self.mesh_source, encoding='ascii')
        Path(str(self.mesh) + '.meta').write_bytes(self.mesh_meta)
        result = AssetManager.import_asset(str(self.mesh), database=self.database)
        assert result.succeeded and result.guid == self.mesh_guid, result.error
        assert AssetRegistry.instance().load_mesh_by_guid(self.mesh_guid) is not None

    def close(self):
        self.component.editor_preview_end()
        self.component._remove_native_batch()
        self.scene.destroy_game_object(self.owner)
        for path in (self.source, self.mesh):
            if path.is_file():
                assert self.database.delete_asset(str(path))
