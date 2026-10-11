"""Package naming changes must not change authored component identities."""

import uuid

from infernux.components.component_identity import component_type_guid, intrinsic_script_guid


def test_builtin_package_rename_preserves_shipped_scene_guids():
    old_module = "Infernux.renderstack.render_stack"
    new_module = "infernux.renderstack.render_stack"
    historical_script = uuid.uuid5(uuid.UUID("594f85cc-9c3a-4ea9-93ed-65a26f77e3a4"), old_module).hex
    historical_type = uuid.uuid5(uuid.UUID("41934666-ab60-4a29-b7ae-c8e15faf83c2"), old_module + ":RenderStack").hex
    assert intrinsic_script_guid(new_module) == historical_script
    assert component_type_guid(new_module, "RenderStack") == historical_type


def test_asset_backed_and_external_component_identities_remain_distinct():
    asset = "08a9c2fbf4915945823b1e55aad53cf1"
    namespace = uuid.UUID("41934666-ab60-4a29-b7ae-c8e15faf83c2")
    assert component_type_guid(asset, "TargetReporter") == uuid.uuid5(namespace, asset + ":TargetReporter").hex
    assert component_type_guid("myplugin.render_stack", "RenderStack") == uuid.uuid5(namespace, "myplugin.render_stack:RenderStack").hex
    assert component_type_guid("infernux_plugin.render_stack", "RenderStack") != component_type_guid("infernux.render_stack", "RenderStack")
