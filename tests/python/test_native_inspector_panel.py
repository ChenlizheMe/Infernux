"""Inspector selection and native authoring payload contracts."""

from infernux.lib import (
    InspectorAddComponentEntry,
    InspectorComponentInfo,
    InspectorObjectInfo,
    InspectorPanel,
    InspectorPrefabInfo,
    InspectorTransformData,
)


def test_inspector_switches_between_object_asset_and_detail_selection():
    panel = InspectorPanel()
    panel.set_selected_object_id(42)
    assert panel.get_selected_object_id() == 42

    panel.set_detail_file("Assets/Selected.mat", "material")
    assert panel.get_selected_object_id() == 42
    assert panel.get_selected_file() == "Assets/Selected.mat"

    panel.set_selected_file("Assets/Other.mat", "material")
    assert panel.get_selected_object_id() == 0
    assert panel.get_selected_file() == "Assets/Other.mat"
    panel.clear_selected_file()
    assert panel.get_selected_file() == ""
    panel.set_selected_object_id(99)
    panel.clear_selected_object()
    assert panel.get_selected_object_id() == 0


def test_native_authoring_payload_preserves_identity_transform_and_prefab_state():
    component = InspectorComponentInfo()
    component.type_name = "MeshRenderer"
    component.component_id = 123
    component.enabled = True
    component.is_native = True
    component.is_script = False
    component.is_broken = False

    owner = InspectorObjectInfo()
    owner.name = "Cube"
    owner.active = True
    owner.tag = "Player"
    owner.layer = 3
    owner.transform_component_id = 91
    owner.driven_transform_properties = 5

    transform = InspectorTransformData()
    transform.px, transform.py_, transform.pz = 1.0, 2.0, 3.0
    transform.ry = 90.0
    transform.sx = transform.sy = transform.sz = 1.0

    entry = InspectorAddComponentEntry()
    entry.display_name = "Light"
    entry.category = "Built-in"
    entry.is_native = True
    entry.script_path = ""
    prefab = InspectorPrefabInfo()
    prefab.override_count = 3
    prefab.is_readonly = True
    prefab.is_transform_readonly = False

    assert (component.type_name, component.component_id, component.enabled) == (
        "MeshRenderer", 123, True,
    )
    assert component.is_native and not component.is_script and not component.is_broken
    assert (owner.name, owner.tag, owner.layer) == ("Cube", "Player", 3)
    assert (owner.transform_component_id, owner.driven_transform_properties) == (91, 5)
    assert (transform.px, transform.py_, transform.pz, transform.ry) == (1, 2, 3, 90)
    assert (transform.sx, transform.sy, transform.sz) == (1, 1, 1)
    assert (entry.display_name, entry.category, entry.is_native) == ("Light", "Built-in", True)
    assert prefab.override_count == 3 and prefab.is_readonly
    assert not prefab.is_transform_readonly


def test_repeated_serialized_group_labels_have_independent_fold_state(monkeypatch):
    from infernux.components import SerializableObject, serialized_field
    from infernux.engine.ui import inspector_declarative as inspector
    from infernux.engine.ui import field_draw

    class Settings(SerializableObject):
        first: float = serialized_field(default=1.0, group="Effects")
        middle: float = serialized_field(default=2.0, group="Playback")
        last: float = serialized_field(default=3.0, group="Effects")

    headers, drawn = [], []
    def header(ctx, label, *, header_id, **kwargs):
        headers.append((label, header_id))
        return header_id != "settings.group.first"

    monkeypatch.setattr(inspector, "render_compact_section_header", header)
    monkeypatch.setattr(inspector, "max_label_w", lambda *args: 80)
    monkeypatch.setattr(field_draw, "draw_field",
                        lambda ctx, meta, value, **kwargs: drawn.append(value) or False)
    monkeypatch.setattr(inspector, "record_inspector_component_item", lambda *args, **kwargs: None)
    settings = Settings()
    control = inspector.InspectorSerializedTarget("settings", lambda: settings, owner=None)
    inspector._render_serialized_target(None, control)
    assert len(headers) == len({key for _, key in headers}) == 3
    assert drawn == [2.0, 3.0]


def test_grouped_serialized_columns_match_native_vector_columns(engine):
    """Nested Inspector groups must not add ImGui's GroupOffset a second time."""
    from infernux import lib, resources
    from infernux.engine.ui.inspector_utils import field_label
    from infernux.engine.ui.theme import ImGuiWindowFlags
    from infernux.engine.ui.field_draw import draw_field_value
    from infernux.components.fields import FieldMetadata, FieldType

    engine.set_gui_fonts(resources.engine_font_path, resources.engine_font_cjk_path,
                         resources.engine_mono_font_path, 13.0)
    observed, failures = [], []

    class Probe(lib.InxGUIRenderable):
        def on_render(self, ctx):
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            ctx.set_next_window_size(640.0, 420.0, 0)
            visible = ctx.begin_window("Fields###field_column_contract", True,
                                       ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoTitleBar)
            try:
                if not visible:
                    return
                column = ctx.inspector_label_width()
                ctx.vector3("Native", 1.0, 2.0, 3.0, 0.1, column)
                native_x = ctx.get_item_rect_min_x()
                untouched_float = draw_field_value(
                    ctx, "##unmodified_float", "Float precision",
                    FieldMetadata("float_precision", FieldType.FLOAT, 0.1), 0.1, column,
                )
                assert untouched_float == 0.1
                ctx.begin_group()
                try:
                    ctx.begin_group()
                    try:
                        ctx.vector3("Serialized vector", 1.0, 2.0, 3.0, 0.1, column)
                        vector_x = ctx.get_item_rect_min_x()
                        field_label(ctx, "Serialized scalar", column)
                        ctx.drag_float("##scalar", 4.0, 0.1, 0.0, 0.0)
                        scalar_x = ctx.get_item_rect_min_x()
                    finally:
                        ctx.end_group()
                finally:
                    ctx.end_group()
                observed.append((native_x, vector_x, scalar_x))
            except Exception as error:
                failures.append(error)
            finally:
                ctx.end_window()

    frames = 0

    def tick(_delta):
        nonlocal frames
        frames += 1
        if frames >= 6 or failures:
            engine.exit()

    probe = Probe()
    engine.register_gui_renderable("test.field_column_contract", probe)
    try:
        engine.set_pre_scene_update_callback(tick)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.field_column_contract")
    assert not failures, failures
    assert observed
    for native_x, vector_x, scalar_x in observed:
        assert abs(native_x - vector_x) < 0.1
        assert abs(native_x - scalar_x) < 0.1
