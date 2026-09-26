import json

from Infernux.engine.ui import asset_details_renderer as renderer


def test_material_import_report_formats_authoritative_source_context(monkeypatch):
    monkeypatch.setattr(renderer, "t", lambda key: "{material}|{property}|{detail}|" + key)
    report = [{
        "code": "unsupported_uv_set",
        "material": "Body",
        "property": "texture/BaseColor",
        "detail": "1",
    }]

    assert renderer._model_material_diagnostic_messages({
        "model_material_diagnostics": json.dumps(report)
    }) == ("Body|texture/BaseColor|1|asset.model_material_diagnostic_unsupported_uv_set",)


def test_material_import_report_is_empty_when_every_source_property_maps():
    assert renderer._model_material_diagnostic_messages({
        "model_material_diagnostics": "[]"
    }) == ()
