from infernux.engine.ui.node_graph_editor_panel import NodeGraphEditorPanel
from infernux.particle.asset import ParticleGraphAsset


class ParticleGraphEditorPanel(NodeGraphEditorPanel):
    window_id: str
    @property
    def asset(self) -> ParticleGraphAsset: ...
    def open_document_resource_immediate(self, file_path: str) -> bool: ...
