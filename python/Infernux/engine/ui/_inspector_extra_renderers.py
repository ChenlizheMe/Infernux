"""Extra Inspector renderers for specific component types (AudioSource, MeshRenderer)."""

from Infernux.debug import Debug
from Infernux.lib import InxGUIContext
from Infernux.engine.i18n import t
from .inspector_utils import max_label_w, field_label, float_close as _float_close
from .theme import Theme, ImGuiCol
from ._inspector_undo import (
    _notify_scene_modified, _record_track_volume, _record_material_slot,
)
from ._inspector_references import (
    _picker_assets, render_object_field,
)


# ============================================================================
# AudioSource extra renderer (per-track section only)
# ============================================================================


def _render_audio_source_extra(ctx: InxGUIContext, comp):
    """Extra Inspector section for AudioSource: per-track clip & volume.

    Source-level properties (volume, pitch, mute, spatial, etc.) are handled
    by the generic CppProperty renderer.  This function only renders the
    dynamic per-track section that cannot be expressed as CppProperty.
    """
    from Infernux.engine.play_mode import PlayModeManager, PlayModeState

    track_count = comp.track_count

    ctx.separator()
    ctx.label("Tracks")

    track_labels = ["Clip", "Volume"]
    track_lw = max_label_w(ctx, track_labels)

    for i in range(track_count):
        ctx.set_next_item_open(True)
        if ctx.collapsing_header(f"Track {i}"):
            # Track clip
            clip = comp.get_track_clip(i)
            clip_name = "None"
            if clip is not None:
                try:
                    clip_name = clip.name or "None"
                except (RuntimeError, AttributeError):
                    clip_name = "None"

            field_label(ctx, "Clip", track_lw)
            render_object_field(
                ctx,
                f"audio_track_clip_{i}",
                clip_name,
                "AudioClip",
                accept_drag_type="AUDIO_FILE",
                on_drop_callback=lambda payload, _c=comp, _i=i: _apply_track_audio_clip_drop(_c, _i, payload),
            )

            # Track volume
            tv = comp.get_track_volume(i)
            field_label(ctx, "Volume", track_lw)
            new_tv = ctx.float_slider(f"##track_vol_{i}", float(tv), 0.0, 1.0)
            if not _float_close(float(new_tv), float(tv)):
                comp.set_track_volume(i, float(new_tv))
                _record_track_volume(comp, i, float(tv), float(new_tv))

            # Play / Stop buttons (only in play mode for feedback)
            pm = PlayModeManager.instance()
            if pm and pm.state != PlayModeState.EDIT:
                is_playing = comp.is_track_playing(i)
                if is_playing:
                    if ctx.button(f"Stop##track_stop_{i}"):
                        comp.stop(i)
                else:
                    if ctx.button(f"Play##track_play_{i}"):
                        comp.play(i)
                ctx.same_line()
                status = "Playing" if is_playing else ("Paused" if comp.is_track_paused(i) else "Stopped")
                ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
                ctx.label(status)
                ctx.pop_style_color(1)


def _apply_track_audio_clip_drop(comp, track_index: int, payload):
    """Handle an AUDIO_FILE drag-drop onto a track clip field."""
    try:
        file_path = str(payload) if not isinstance(payload, str) else payload

        # Try GUID-based loading via AssetRegistry
        from Infernux.lib import AssetRegistry
        registry = AssetRegistry.instance()
        adb = registry.get_asset_database()
        if adb:
            guid = adb.get_guid_from_path(file_path)
            if guid and hasattr(comp, 'set_track_clip_by_guid'):
                comp.set_track_clip_by_guid(track_index, guid)
                _notify_scene_modified()
                return

        # Fallback: load from file path directly
        from Infernux.core.audio_clip import AudioClip as PyAudioClip

        clip = PyAudioClip.load(file_path)
        if clip is None:
            return

        comp.set_track_clip(track_index, clip.native)
        _notify_scene_modified()
    except Exception as e:
        Debug.log_error(f"Audio clip drop failed: {e}")


# ============================================================================
# MeshRenderer extra renderer (material slots)
# ============================================================================


def _render_mesh_renderer_materials(ctx: InxGUIContext, comp):
    """Render material slot fields after MeshRenderer CppProperty fields."""
    from Infernux.components.builtin_component import BuiltinComponent

    # Ensure we have the Python wrapper
    if not isinstance(comp, BuiltinComponent):
        wrapper_cls = BuiltinComponent._builtin_registry.get(getattr(comp, "type_name", "")) \
            or BuiltinComponent._builtin_registry.get("MeshRenderer")
        go = getattr(comp, 'game_object', None)
        if wrapper_cls and go is not None:
            comp = wrapper_cls._get_or_create_wrapper(comp, go)
        else:
            return

    # Mesh info
    if comp.has_inline_mesh():
        inline_name = getattr(comp, 'inline_mesh_name', '') or ''
        mesh_display = inline_name if inline_name else "(Primitive)"
    elif getattr(comp, 'has_mesh_asset', False):
        mesh_display = getattr(comp, 'mesh_name', '') or 'Mesh'
    else:
        mesh_display = "None"

    ctx.separator()
    labels = [t("inspector.mesh"), "Materials", "Element 0"]
    lw = max_label_w(ctx, labels)

    field_label(ctx, t("inspector.mesh"), lw)
    render_object_field(ctx, "mesh_field", mesh_display, "Mesh", clickable=False)

    # Material slots
    mat_count = getattr(comp, 'material_count', 0) or 1
    material_guids = comp.get_material_guids() if hasattr(comp, 'get_material_guids') else []
    slot_names = comp.get_material_slot_names() if hasattr(comp, 'get_material_slot_names') else []

    field_label(ctx, "Materials", lw)
    ctx.label(f"Size: {mat_count}")

    for slot_idx in range(mat_count):
        # Determine slot label
        if slot_idx < len(slot_names) and slot_names[slot_idx]:
            slot_label = f"{slot_names[slot_idx]} (Slot {slot_idx})"
        else:
            slot_label = f"Element {slot_idx}"

        # Determine display name
        mat = None
        try:
            mat = comp.get_effective_material(slot_idx)
        except (RuntimeError, IndexError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
        mat_path = getattr(mat, "file_path", "") if mat else ""
        is_embedded = isinstance(mat_path, str) and "::submat:" in mat_path
        is_default = ((slot_idx >= len(material_guids)) or (not material_guids[slot_idx])) and not is_embedded
        mat_name = getattr(mat, 'name', 'None') if mat else 'None'
        display_name = mat_name + (" (Default)" if is_default else "")

        def _make_on_drop(s, _comp=comp):
            def _on_drop(mat_path):
                from Infernux.lib import AssetRegistry
                registry = AssetRegistry.instance()
                adb = registry.get_asset_database()
                if not adb:
                    return
                guid = adb.get_guid_from_path(str(mat_path))
                if not guid:
                    return
                old_guid = ""
                guids = _comp.get_material_guids()
                if s < len(guids):
                    old_guid = guids[s] or ""
                _comp.set_material(s, guid)
                _record_material_slot(_comp, s, old_guid, guid,
                                     f"Set Material Slot {s}")
            return _on_drop

        def _make_on_pick(s, _comp=comp):
            def _on_pick(picked_path):
                _make_on_drop(s, _comp)(str(picked_path))
            return _on_pick

        def _make_on_clear(s, _comp=comp):
            def _on_clear():
                old_guid = ""
                guids = _comp.get_material_guids()
                if s < len(guids):
                    old_guid = guids[s] or ""
                _comp.set_material(s, "")
                _record_material_slot(_comp, s, old_guid, "",
                                     f"Clear Material Slot {s}")
            return _on_clear

        field_label(ctx, slot_label, lw)
        render_object_field(
            ctx, f"mat_{slot_idx}", display_name, "Material",
            clickable=False,
            accept_drag_type="MATERIAL_FILE",
            on_drop_callback=_make_on_drop(slot_idx),
            picker_asset_items=lambda filt: _picker_assets(filt, "*.mat"),
            on_pick=_make_on_pick(slot_idx),
            on_clear=_make_on_clear(slot_idx),
        )
