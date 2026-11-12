"""GPU shader schema queries and runtime program publication."""

from __future__ import annotations

import json
from pathlib import Path
from infernux._source_kinds import SHADER_SOURCE_KINDS, source_extension


class Shader:
    """Query published shaders and reimport edited GPU author assets."""

    _runtime_revision = 0
    _runtime_revisions: dict[str, int] = {}
    _surface_metadata: dict[str, tuple] = {}
    _surface_schemas: dict[str, tuple[tuple, int]] = {}

    @staticmethod
    def _fragment_key(material) -> str:
        from infernux.core.assets import AssetManager

        reference = material.native.frag_shader_reference
        if reference["guid"]:
            return reference["guid"]
        database = AssetManager._asset_database
        if database is None:
            return reference["shader_id"]
        path = database.find_shader_path_by_id(reference["shader_id"], "fragment")
        return database.get_guid_from_path(path)

    @staticmethod
    def _read_surface_metadata(meta) -> tuple:
        return (
            meta.get_string("shader_cull_mode"),
            meta.get_string("shader_depth_write"),
            meta.get_string("shader_depth_test"),
            meta.get_string("shader_blend"),
            meta.get_int("shader_queue"),
            meta.get_string("shader_pass_tag"),
            meta.get_string("shader_stencil"),
            meta.get_string("shader_alpha_test"),
        )

    @staticmethod
    def _read_surface_schema(meta) -> tuple:
        # File coordinates describe diagnostics, not particle output ports.
        properties = json.loads(meta.get_string("properties") or "[]")
        return tuple(
            (item["name"], item["type"], item.get("default"), bool(item.get("internal")))
            for item in properties
        )

    @classmethod
    def _record_runtime_publication(cls, database, paths) -> None:
        if database is None:
            return
        for path in paths:
            extension = source_extension(path)
            kind = SHADER_SOURCE_KINDS.get(extension)
            stage = kind[0] if kind else {".vert": "vertex", ".frag": "fragment"}.get(extension)
            if stage not in {"vertex", "fragment", "compute"}:
                continue
            meta = database.get_meta_by_path(path)
            if meta is None or not meta.has_key("shader_id"):
                continue
            shader_id = meta.get_guid()
            cls._runtime_revision += 1
            cls._runtime_revisions[shader_id] = cls._runtime_revision
            if stage == "fragment" and kind is None:
                cls._surface_metadata[shader_id] = cls._read_surface_metadata(meta)
                schema = cls._read_surface_schema(meta)
                if schema != cls._surface_schemas.get(shader_id, ((), 0))[0]:
                    cls._surface_schemas[shader_id] = (schema, cls._runtime_revision)

    @classmethod
    def _apply_published_surface(cls, material) -> None:
        from infernux.core.assets import AssetManager

        database = AssetManager._asset_database
        if database is None:
            return
        shader_id = cls._fragment_key(material)
        if shader_id not in cls._surface_metadata:
            meta = database.get_meta_by_guid(shader_id)
            if meta is not None and meta.has_key("shader_queue"):
                cls._surface_metadata[shader_id] = cls._read_surface_metadata(meta)
                cls._surface_schemas[shader_id] = (cls._read_surface_schema(meta), 0)
        state = cls._surface_metadata.get(shader_id)
        if state is not None:
            material.native.apply_shader_render_meta(*state)

    @classmethod
    def _clear_runtime_publications(cls) -> None:
        cls._runtime_revisions.clear()
        cls._surface_metadata.clear()
        cls._surface_schemas.clear()

    @staticmethod
    def property_to_id(name: str) -> int:
        """Intern a property name for this process; persist the name, not its ID."""
        from infernux.lib import InxMaterial

        return InxMaterial.property_to_id(name)

    @staticmethod
    def _author_source(asset: str):
        from infernux.core.assets import AssetManager

        database = AssetManager.require_asset_database()
        selector = str(asset)
        path = database.get_path_from_guid(selector)
        if path:
            return database, path, selector
        candidate = Path(selector)
        if not candidate.is_absolute():
            candidate = Path(database.project_root) / candidate
        path = str(candidate.resolve())
        return database, path, database.get_guid_from_path(path)

    @staticmethod
    def check(asset: str, *, source: str | None = None) -> dict:
        """Check a GPU file or unsaved draft with the same native preparation service.

        Returns ``valid``, the logical ``schema``, and source diagnostics for both
        GPU targets. It does not publish an asset or create a device pipeline.
        Shader source is statically parsed; host imports and statements never run.
        """
        from infernux.lib import _Infernux

        _, path, _ = Shader._author_source(asset)
        if source_extension(path) not in SHADER_SOURCE_KINDS:
            raise ValueError(f"Expected a GPU author source file: {path}")
        report = _Infernux._check_shader_source(path, source)
        report["schema"] = json.loads(report["schema"]) if report["schema"] else None
        return report

    @staticmethod
    def get_schema(asset: str) -> dict:
        """Read a static GPU asset's logical fields and graph input ports.

        Accepts an asset GUID or registered project path. The native asset
        owner prepares the source revision; querying fields creates no PSO.
        """
        from infernux.lib import _Infernux

        _, _, guid = Shader._author_source(asset)
        if not guid:
            raise FileNotFoundError(f"GPU author asset is not registered: {asset}")
        return json.loads(_Infernux._shader_source_schema(guid))

    @staticmethod
    def _validate_shader_type(shader_type: str) -> str:
        normalized = str(shader_type).strip().lower()
        if normalized not in {"vertex", "fragment"}:
            raise ValueError(
                "shader_type must be 'vertex' or 'fragment'"
            )
        return normalized

    @classmethod
    def is_loaded(cls, name: str, shader_type: str = "vertex") -> bool:
        """Query GPU publication by asset GUID, imported path or built-in name.

        Returns False before startup, after shutdown, or without a renderer.
        Static author assets, standalone stages and linked programs are included.
        This does not claim the last source edit compiled successfully.
        """
        from infernux.core.assets import AssetManager

        stage = cls._validate_shader_type(shader_type)
        native = AssetManager._native_engine()
        return native is not None and native.is_shader_loaded(str(name), stage)

    @classmethod
    def reload(cls, shader_id: str, shader_type: str | None = None) -> bool:
        """Reimport a program name or registered GPU author source path.

        Editor/headless authoring only; frozen Player assets are read-only.
        A Name shared by vertex and fragment stages reloads both unless a stage
        is specified. Duplicate Names within a stage require an explicit path.
        Returns True after every selected reimport succeeds; import/compiler
        errors raise RuntimeError instead of silently claiming success.
        """
        from infernux.application import Application
        from infernux.core.assets import AssetManager

        stage = cls._validate_shader_type(shader_type) if shader_type is not None else None
        if Application.is_player():
            raise RuntimeError("Shader.reload requires authoring assets; frozen Player shaders are read-only")
        database = AssetManager.require_asset_database()
        selector = str(shader_id)
        candidate = Path(selector)
        if not candidate.is_absolute():
            candidate = Path(database.project_root) / candidate
        direct_guid = database.get_guid_from_path(str(candidate))
        paths = []
        if direct_guid:
            path = database.get_path_from_guid(direct_guid)
            extension = source_extension(path)
            kind = SHADER_SOURCE_KINDS.get(extension)
            path_stage = kind[0] if kind else {".vert": "vertex", ".frag": "fragment"}.get(extension)
            if path_stage is None or (stage is not None and path_stage != stage):
                raise ValueError(f"Not a matching ShaderInfo stage asset: {selector}")
            paths.append(path)
        else:
            by_stage = {}
            for guid in database.get_all_guids():
                meta = database.get_meta_by_guid(guid)
                if meta is None or not meta.has_key("shader_id") or not meta.has_key("type"):
                    continue
                current_stage = meta.get_string("type")
                if current_stage not in {"vertex", "fragment"} or (stage is not None and current_stage != stage):
                    continue
                if meta.get_string("shader_id") != selector:
                    continue
                if current_stage in by_stage:
                    raise ValueError(f"Ambiguous ShaderInfo Name {selector!r}; use an asset path")
                by_stage[current_stage] = database.get_path_from_guid(guid)
            paths = [by_stage[key] for key in ("vertex", "fragment") if key in by_stage]
        if not paths:
            raise FileNotFoundError(f"ShaderInfo asset is not registered: {selector}")
        for path in paths:
            result = AssetManager.reimport_asset(path, database=database)
            if not result:
                raise RuntimeError(f"Shader reload failed for {path}: {result.error}")
        return True
