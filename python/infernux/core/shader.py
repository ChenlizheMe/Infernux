"""ShaderInfo queries and runtime shader publication."""

from __future__ import annotations

import json
from pathlib import Path


class Shader:
    """Query published shaders and reimport edited ShaderInfo source assets."""

    _runtime_revision = 0
    _runtime_revisions: dict[str, int] = {}
    _surface_metadata: dict[str, tuple] = {}
    _surface_schemas: dict[str, tuple[tuple, int]] = {}
    _surface_catalog_generation = None

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
            stage = Path(path).suffix.lower()
            if stage not in {".vert", ".frag"}:
                continue
            meta = database.get_meta_by_path(path)
            if meta is None or not meta.has_key("shader_id"):
                continue
            shader_id = meta.get_string("shader_id")
            cls._runtime_revision += 1
            cls._runtime_revisions[shader_id] = cls._runtime_revision
            if stage == ".frag":
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
        shader_id = material.frag_shader_name
        generation = database.query_generation
        if shader_id not in cls._surface_metadata and cls._surface_catalog_generation != generation:
            for path in database.get_all_asset_paths():
                if not path.lower().endswith(".frag"):
                    continue
                meta = database.get_meta_by_path(path)
                if meta is not None and meta.has_key("shader_id") and meta.has_key("shader_queue"):
                    name = meta.get_string("shader_id")
                    if name not in cls._surface_metadata:
                        cls._surface_metadata[name] = cls._read_surface_metadata(meta)
                        cls._surface_schemas[name] = (cls._read_surface_schema(meta), 0)
            cls._surface_catalog_generation = generation
        state = cls._surface_metadata.get(shader_id)
        if state is not None:
            material.native.apply_shader_render_meta(*state)

    @classmethod
    def _clear_runtime_publications(cls) -> None:
        cls._runtime_revisions.clear()
        cls._surface_metadata.clear()
        cls._surface_schemas.clear()
        cls._surface_catalog_generation = None

    @staticmethod
    def property_to_id(name: str) -> int:
        """Intern a property name for this process; persist the name, not its ID."""
        from infernux.lib import InxMaterial

        return InxMaterial.property_to_id(name)

    @staticmethod
    def _validate_shader_type(shader_type: str) -> str:
        normalized = str(shader_type).strip().lower()
        if normalized not in {"vertex", "fragment"}:
            raise ValueError(
                "shader_type must be 'vertex' or 'fragment'; "
                "compute shaders are not supported (use an external parallel backend)"
            )
        return normalized

    @classmethod
    def is_loaded(cls, name: str, shader_type: str = "vertex") -> bool:
        """Query GPU publication by ShaderInfo Name and stage, without loading.

        Returns False before startup, after shutdown, or without a renderer.
        Both standalone stages and linked material programs are included.
        This does not claim the last source edit compiled successfully.
        """
        from infernux.core.assets import AssetManager

        stage = cls._validate_shader_type(shader_type)
        native = AssetManager._native_engine()
        return native is not None and native.is_shader_loaded(str(name), stage)

    @classmethod
    def reload(cls, shader_id: str, shader_type: str | None = None) -> bool:
        """Reimport a ShaderInfo Name or registered .vert/.frag asset path.

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
            path_stage = {".vert": "vertex", ".frag": "fragment"}.get(Path(path).suffix.lower())
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
