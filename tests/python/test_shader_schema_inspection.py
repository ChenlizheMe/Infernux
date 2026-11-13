"""Reading logical fields must not compile or load a runtime GPU program."""

from pathlib import Path

from infernux.core.assets import AssetManager
from infernux.core.shader import Shader
from infernux.lib import AssetRegistry


def test_schema_query_keeps_the_runtime_shader_unloaded(engine, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(AssetManager, "_registry", AssetRegistry.instance())
    path = Path(database.assets_root) / "SchemaWithoutProgram.frag.py"
    path.write_text(
        'import infernux.shader as shader\n'
        'class SchemaWithoutProgram(shader.surface):\n'
        '    shader_id = "inspection/schema_without_program"\n'
        '    amount = shader.material.float(default=0.1, range=(0.0, 1.0))\n'
        '    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:\n'
        '        return shader.SurfaceData(albedo=shader.vec3(self.amount))\n',
        encoding="utf-8",
    )
    imported = AssetManager.import_asset(str(path))
    assert imported, imported.error
    try:
        schema = Shader.get_schema(imported.guid)
        field = next(item for item in schema["parameters"] if item["name"] == "amount")
        assert field["kind"] == "float" and field["default"] == 0.1
        assert not any(record.guid == imported.guid and record.cpu_resident for record in engine.asset_runtime_records)
    finally:
        AssetManager.delete_asset(str(path))
