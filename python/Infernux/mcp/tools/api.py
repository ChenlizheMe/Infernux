"""Subsystem API and shader/audio knowledge tools for agents."""

from __future__ import annotations

import inspect
import ast
import os
from typing import Any

from Infernux.mcp.tools.common import issue_knowledge_token, ok, register_tool_metadata, serialize_value


PROPERTY_TYPES = {
    "Float": "material.set_float(name, value)",
    "Float2": "material.set_vector2(name, x, y)",
    "Float3": "material.set_vector3(name, x, y, z)",
    "Float4": "material.set_vector4(name, x, y, z, w)",
    "Color": "material.set_color(name, r, g, b, a)",
    "Int": "material.set_int(name, value)",
    "Mat4": "material.set_param(name, matrix_value)",
    "Texture2D": "material.set_texture(name, texture_or_guid_or_path)",
}


SUBSYSTEM_GUIDES: dict[str, dict[str, Any]] = {
    "api": {
        "summary": "Agent-facing API discovery policy for a young, fast-moving engine.",
        "concepts": [
            "Do not guess unfamiliar Infernux APIs from Unity or other engines.",
            "For Python-layer APIs, call api.search(query), then api.get(symbol_or_module). The index is generated from .pyi stubs first, then .py source.",
            "For components, call component.list_types and component.describe_type before setting fields.",
            "For shader authoring, call shader.guide, shader.catalog, and shader.describe because shader behavior is C++/compiler-backed and annotation-driven.",
            "For MCP tools, call mcp.catalog.search or mcp.catalog.recommend before choosing tools.",
        ],
        "workflow": [
            "1. Search: api.search('audio source play one shot') or mcp.catalog.recommend(intent).",
            "2. Inspect: api.get('AudioSource') or component.describe_type('AudioSource').",
            "3. Act: use scene/component/asset tools with the exact fields and signatures returned.",
            "4. Validate: use runtime.read_errors, console.read, or subsystem-specific describe/report tools.",
        ],
        "symbols": ["api.search", "api.get", "mcp.catalog.search", "component.describe_type", "shader.guide"],
    },
    "shader": {
        "summary": "Three-layer shader authoring model: surface fragment/vertex shaders, shading models, and GLSL libraries/templates.",
        "concepts": [
            "Shader authoring is NOT a normal Python-reflection API. It is parsed and compiled by the C++ shader/material pipeline, so this guide is manually curated.",
            "Fragment surface shaders are .frag files with @shader_id, @shading_model, optional @queue, and @property annotations.",
            "Vertex shaders are .vert files with @shader_id. Materials may set vert_shader_name and frag_shader_name separately.",
            "Shading models are .shadingmodel files with @shader_id and @target blocks such as forward or gbuffer.",
            "Library files are .glsl files imported with @import; do not assign them directly as Material shaders.",
            "Material properties are declared in the first 50 lines using @property: name, Type, default[, HDR].",
            "Surface shaders implement void surface(out SurfaceData s). Start with s = InitSurfaceData().",
            "Common built-in varyings include v_TexCoord, v_Color, v_WorldPos, and normal/tangent data supplied by the standard vertex path.",
            "RenderGraph fullscreen effects use fullscreen_triangle.vert automatically and bind fragment shader IDs through p.fullscreen_quad(shader_id).",
        ],
        "workflow": [
            "Call shader.catalog to discover built-in shader IDs and examples.",
            "Call shader.describe(shader_id, kind='fragment') before binding a material to a custom fragment shader.",
            "Create .frag/.vert through asset.create_builtin_resource(kind='shader') or asset.write_text.",
            "After editing shader files, call asset.refresh and use Shader.reload(shader_id) from scripts if runtime reload is needed.",
            "For materials, use Material.create_lit/create_unlit, then set vert_shader_name/frag_shader_name or set_shader for matching IDs.",
        ],
        "common_mistakes": [
            "Do not put @shader_id only in comments; the parser expects lines that start with @shader_id: near the top.",
            "Do not assign a .shadingmodel or .glsl library as Material.frag_shader_name.",
            "Do not invent property names in Material unless the shader declares them or intentionally uses dynamic properties.",
            "Texture2D defaults are symbolic names such as white; material values should be texture GUID/path/wrapper, not a vec4.",
            "Do not assume Shader.reload is fully bound for every runtime path; prefer asset.refresh and material/pipeline refresh tools when available.",
        ],
        "rules": {
            "file_kinds": {
                ".frag": "Surface fragment shader. Requires @shader_id and usually @shading_model. Declares material @property annotations.",
                ".vert": "Vertex shader. Requires @shader_id. Use for custom vertex deformation or varyings.",
                ".shadingmodel": "Lighting/evaluation model. Referenced by @shading_model; not assigned directly to Material.",
                ".glsl": "Shared library code. Imported with @import; not assigned directly to Material.",
            },
            "annotations": {
                "@shader_id": "Unique shader id used by materials/render graph. Keep it near the top; Inspector/Python catalog scans only early lines.",
                "@shading_model": "Fragment surface shader only; names a .shadingmodel shader id such as pbr or unlit. It resolves through the C++ shader id map.",
                "@queue": "Render queue integer, e.g. 2000 opaque, 3000 transparent.",
                "@property": "Material property declaration: @property: name, Type, default[, HDR].",
                "@import": "Import a .glsl or shading helper id. Imports use shader_id, not filenames; .shadingmodel ids are internally namespaced.",
                "@target": "Shading model target block such as forward or gbuffer. Used inside .shadingmodel files.",
                "@hidden": "Hide internal shader from normal shader selection catalogs.",
                "@surface_type": "opaque or transparent. transparent defaults queue/blend/depth/pass_tag when unset.",
                "@blend/@blend_mode": "off, alpha, additive, or premultiply.",
                "@alpha_test/@alpha_clip": "Enables cutout behavior and affects shadow variants.",
            },
            "entry_points": {
                "surface": "Preferred fragment workflow: void surface(out SurfaceData s). If no main() exists, engine injects templates, varyings, outputs, and shading-model evaluate().",
                "vertex": "Optional vertex deformation hook: void vertex(inout VertexInput v).",
                "main": "Advanced path. If author supplies main() or explicit layout(location=...), auto interface/template injection may be skipped; the author owns bindings and IO.",
            },
            "builtins": {
                "surface_data": "Use s = InitSurfaceData(); then set fields such as albedo, alpha, emission, normalWS, metallic, smoothness, occlusion.",
                "common_varyings": ["v_TexCoord", "v_Color", "v_WorldPos", "v_Normal", "v_Tangent", "v_ViewDepth"],
                "common_uniforms": ["material", "_Globals", "lighting"],
                "material_ubo": "MaterialProperties is exposed as global 'material'. Texture2D properties become sampler2D bindings managed by the compiler.",
            },
            "reload_limitations": [
                "File watcher reload primarily handles .vert and .frag.",
                "Editing .glsl, .shadingmodel, or _templates/*.glsl may require touching/reloading a dependent .vert/.frag or restarting/invalidation.",
                "Shader.reload in Python is not a complete substitute for C++ file reload paths in every build.",
            ],
            "material_binding": [
                "Use material.vert_shader_name for vertex shader id.",
                "Use material.frag_shader_name for fragment shader id.",
                "Do not bind .glsl libraries or .shadingmodel files directly as material shaders.",
                "Use material.get_properties MCP to inspect shader.vertex, shader.fragment, render_queue, and synced properties.",
            ],
            "property_types": PROPERTY_TYPES,
        },
        "symbols": ["Shader", "Material", "shader.catalog", "shader.describe", "shader.guide"],
    },
    "audio": {
        "summary": "Audio uses AudioListener for the ears, AudioSource for multi-track playback, and AudioClip for WAV assets.",
        "concepts": [
            "AudioListener should usually be attached to the main camera. Only one listener is active at a time.",
            "AudioSource is multi-track: set track_count, assign clips per track, then play(track_index).",
            "AudioClip.load(path) returns an AudioClip wrapper or None; pass clip or clip.native to AudioSource methods.",
            "Use play_one_shot(clip, volume_scale) for transient SFX rather than creating many temporary AudioSource objects.",
        ],
        "workflow": [
            "Ensure a GameObject has AudioSource and the camera has AudioListener.",
            "Load WAV assets with AudioClip.load('Assets/Audio/name.wav').",
            "Assign clips with source.set_track_clip(index, clip) or set_track_clip_by_guid(index, guid).",
            "Use source.volume/pitch/mute/loop/play_on_awake for source-level behavior.",
            "Use source.set_track_volume(index, value), play(index), pause(index), stop(index), stop_all().",
        ],
        "common_mistakes": [
            "Do not use source.clip = clip; this engine exposes per-track clips instead.",
            "Track indices are zero-based and must be below source.track_count.",
            "AudioClip.load currently documents WAV loading; avoid assuming MP3/OGG unless the asset importer says so.",
            "Attach one AudioListener to the main camera instead of adding listeners to many objects.",
        ],
        "symbols": ["AudioSource", "AudioListener", "AudioClip", "audio.guide"],
    },
    "component": {
        "summary": "Python components inherit InxComponent; built-ins expose CppProperty fields and delegate methods.",
        "concepts": [
            "Use component.list_types and component.describe_type for exact component fields.",
            "Use serialized_field for script fields that should persist and appear in the inspector.",
            "Use lifecycle methods awake/start/update/late_update/on_enable/on_disable/on_destroy.",
        ],
        "symbols": ["InxComponent", "serialized_field", "component.describe_type", "component.list_types"],
    },
    "material": {
        "summary": "Material wraps InxMaterial and stores shader selection, render state, and typed shader properties.",
        "concepts": [
            "Material.create_lit uses default_lit; Material.create_unlit uses default_unlit.",
            "Use vert_shader_name and frag_shader_name when vertex/fragment shader IDs differ.",
            "Use set_color/set_float/set_int/set_vector*/set_texture based on shader @property declarations.",
        ],
        "symbols": ["Material", "shader.describe", "material.create", "material.set_property"],
    },
    "ui": {
        "summary": "Screen-space UI uses UICanvas plus UIText, UIImage, UIButton, pointer events, and persistent UIEventEntry bindings.",
        "concepts": [
            "Create a UICanvas root, then child UI elements such as UIText, UIImage, and UIButton.",
            "Positions and sizes are in canvas design pixels, scaled by UICanvas.compute_scale for the Game View.",
            "Use api.get('UICanvas'), api.get('UIText'), api.get('UIButton'), and api.get('InxUIScreenComponent') before scripting UI.",
            "UIButton.on_click is a runtime UIEvent; persistent callbacks use on_click_entries.",
        ],
        "symbols": ["UICanvas", "UIText", "UIImage", "UIButton", "UIEvent", "PointerEventData"],
    },
}


_API_INDEX: dict[str, Any] | None = None


def register_api_tools(mcp) -> None:
    _register_metadata()

    @mcp.tool(name="api.subsystems")
    def api_subsystems() -> dict:
        """List documented engine subsystems available to agents."""
        return ok({
            "agent_guidance": _agent_api_guidance(),
            "subsystems": [
                {"name": name, "summary": guide["summary"], "symbols": guide.get("symbols", [])}
                for name, guide in sorted(SUBSYSTEM_GUIDES.items())
            ],
            "python_api": _api_index_summary(),
        })

    @mcp.tool(name="api.get")
    def api_get(name: str) -> dict:
        """Return a subsystem guide or symbol API page."""
        key = str(name or "").strip()
        guide = SUBSYSTEM_GUIDES.get(key.lower())
        if guide is not None:
            payload = {"kind": "subsystem", "name": key.lower(), **guide}
            if key.lower() in {"shader", "audio", "ui", "material"}:
                lock_scope = "shader" if key.lower() == "material" else key.lower()
                payload["knowledge_lock"] = issue_knowledge_token(lock_scope, source_tool=f"api.get:{key.lower()}")
            return ok(payload)
        symbol = _symbol_doc(key)
        if symbol:
            return ok({"kind": "symbol", **symbol})
        module_doc = _module_doc(key)
        if module_doc:
            return ok({"kind": "module", **module_doc})
        index = _api_index()
        return ok({
            "found": False,
            "available_subsystems": sorted(SUBSYSTEM_GUIDES),
            "available_symbols": sorted(index["symbols"]),
            "available_modules": sorted(index["modules"]),
        })

    @mcp.tool(name="api.search")
    def api_search(query: str, limit: int = 20) -> dict:
        """Search subsystem guides, symbols, shader IDs, and component names."""
        query_l = str(query or "").lower()
        matches = []
        for name, guide in SUBSYSTEM_GUIDES.items():
            haystack = " ".join([
                name,
                guide.get("summary", ""),
                " ".join(guide.get("concepts", [])),
                " ".join(guide.get("symbols", [])),
            ]).lower()
            score = _score(query_l, haystack)
            if score:
                matches.append({"kind": "subsystem", "name": name, "summary": guide["summary"], "score": score})
        index = _api_index()
        for name in index["symbols"]:
            doc = _symbol_doc(name)
            haystack = " ".join([
                name,
                doc.get("doc", ""),
                " ".join(m["name"] for m in doc.get("methods", [])),
                " ".join(p["name"] for p in doc.get("properties", [])),
                " ".join(a["name"] for a in doc.get("attributes", [])),
            ]).lower()
            score = _score(query_l, haystack)
            if score:
                matches.append({
                    "kind": "symbol",
                    "name": name,
                    "module": doc.get("module", ""),
                    "summary": doc.get("doc", "").splitlines()[0] if doc.get("doc") else "",
                    "score": score,
                })
        for name, module in index["modules"].items():
            haystack = " ".join([name, module.get("path", ""), " ".join(module.get("symbols", []))]).lower()
            score = _score(query_l, haystack)
            if score:
                matches.append({"kind": "module", "name": name, "summary": module.get("path", ""), "score": score})
        for shader in _scan_shaders():
            haystack = " ".join([shader["shader_id"], shader["kind"], shader.get("path", ""), " ".join(shader.get("imports", []))]).lower()
            score = _score(query_l, haystack)
            if score:
                matches.append({"kind": "shader", "name": shader["shader_id"], "shader_kind": shader["kind"], "summary": shader.get("path", ""), "score": score})
        matches.sort(key=lambda item: (-item["score"], item["kind"], item["name"]))
        return ok({"query": query, "matches": matches[: max(int(limit or 20), 1)]})

    @mcp.tool(name="shader.guide")
    def shader_guide(topic: str = "") -> dict:
        """Return shader authoring rules, property annotation syntax, and examples."""
        guide = dict(SUBSYSTEM_GUIDES["shader"])
        guide["property_types"] = PROPERTY_TYPES
        guide["examples"] = _shader_examples()
        guide["knowledge_lock"] = issue_knowledge_token("shader", source_tool="shader.guide")
        if topic:
            guide["topic"] = topic
        return ok(guide)

    @mcp.tool(name="shader.catalog")
    def shader_catalog(kind: str = "", include_hidden: bool = False) -> dict:
        """List shader IDs from project and built-in shader roots."""
        shaders = [
            item
            for item in _scan_shaders()
            if (not kind or item["kind"] == kind or item["extension"].lstrip(".") == kind)
            and (include_hidden or not item.get("hidden"))
        ]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in shaders:
            grouped.setdefault(item["kind"], []).append(item)
        return ok({"shaders": shaders, "grouped": grouped, "property_types": PROPERTY_TYPES})

    @mcp.tool(name="shader.describe")
    def shader_describe(shader_id: str, kind: str = "") -> dict:
        """Describe a shader file, annotations, material properties, and usage."""
        shader_id_l = str(shader_id or "").strip().lower()
        candidates = [
            item for item in _scan_shaders()
            if item["shader_id"].lower() == shader_id_l and (not kind or item["kind"] == kind or item["extension"].lstrip(".") == kind)
        ]
        if not candidates:
            return ok({"found": False, "shader_id": shader_id, "available": [item["shader_id"] for item in _scan_shaders()]})
        return ok({"shader_id": shader_id, "matches": candidates, "usage": _shader_usage(candidates)})

    @mcp.tool(name="audio.guide")
    def audio_guide(topic: str = "") -> dict:
        """Return AudioSource/AudioListener/AudioClip usage guidance."""
        guide = dict(SUBSYSTEM_GUIDES["audio"])
        guide["examples"] = _audio_examples()
        guide["symbols"] = [_symbol_doc(name) for name in ("AudioSource", "AudioListener", "AudioClip")]
        guide["knowledge_lock"] = issue_knowledge_token("audio", source_tool="audio.guide")
        if topic:
            guide["topic"] = topic
        return ok(guide)


def _symbol_doc(symbol: str) -> dict[str, Any]:
    key = str(symbol or "").strip()
    index = _api_index()
    entry = index["symbols"].get(key)
    candidates = _symbol_candidates(index, key)
    if "." not in key and len(candidates) > 1:
        preferred = _preferred_symbol_candidate(candidates)
        if preferred is not None:
            entry = preferred
    if entry is None and "." in key:
        module_name, _, attr = key.rpartition(".")
        entry = index["symbols"].get(attr)
        if entry is not None and entry.get("module") != module_name:
            entry = None
    if entry is None:
        lowered = key.lower()
        matches = [item for name, item in index["symbols"].items() if name.lower() == lowered or item.get("qualname", "").lower() == lowered]
        if len(matches) == 1:
            entry = matches[0]
    if entry is None:
        return {}
    doc = dict(entry)
    if "." not in key and len(candidates) > 1:
        doc["ambiguous_short_name"] = True
        doc["alternatives"] = [
            {"name": item.get("name", ""), "qualname": item.get("qualname", ""), "module": item.get("module", ""), "source": item.get("source", "")}
            for item in candidates
        ]
        doc["recommendation"] = "Use api.get with the module-qualified name from alternatives for deterministic results."
    runtime = _runtime_symbol_doc(doc.get("module", ""), doc.get("name", ""))
    if runtime:
        doc.setdefault("runtime_doc", runtime.get("doc", ""))
        doc.setdefault("runtime_module", runtime.get("module", ""))
        if not doc.get("doc"):
            doc["doc"] = runtime.get("doc", "")
        if not doc.get("methods"):
            doc["methods"] = runtime.get("methods", [])
        if not doc.get("properties"):
            doc["properties"] = runtime.get("properties", [])
    return doc


def _symbol_candidates(index: dict[str, Any], key: str) -> list[dict[str, Any]]:
    lowered = str(key or "").lower()
    seen: set[str] = set()
    candidates = []
    for item in index["symbols"].values():
        qualname = str(item.get("qualname", ""))
        if qualname in seen:
            continue
        if str(item.get("name", "")).lower() == lowered or qualname.lower() == lowered:
            candidates.append(item)
            seen.add(qualname)
    return candidates


def _preferred_symbol_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    prefixes = (
        "Infernux.components.builtin.",
        "Infernux.components.",
        "Infernux.ui.",
        "Infernux.core.",
        "Infernux.renderstack.",
    )
    for prefix in prefixes:
        for item in candidates:
            if str(item.get("module", "")).startswith(prefix):
                return item
    return candidates[0] if candidates else None


def _runtime_symbol_doc(module_name: str, attr: str) -> dict[str, Any]:
    try:
        module = __import__(module_name, fromlist=[attr])
        obj = getattr(module, attr)
    except Exception:
        return {}
    methods = []
    properties = []
    for name, member in inspect.getmembers(obj):
        if name.startswith("_"):
            continue
        if isinstance(member, property):
            properties.append({"name": name, "doc": inspect.getdoc(member) or ""})
        elif inspect.isfunction(member) or inspect.ismethod(member):
            try:
                signature = str(inspect.signature(member))
            except Exception:
                signature = "(...)"
            methods.append({"name": name, "signature": signature, "doc": inspect.getdoc(member) or ""})
    return {
        "name": attr,
        "module": module_name,
        "doc": inspect.getdoc(obj) or "",
        "methods": methods,
        "properties": properties,
    }


def _module_doc(name: str) -> dict[str, Any]:
    index = _api_index()
    module = index["modules"].get(str(name or "").strip())
    if module is None:
        lowered = str(name or "").strip().lower()
        matches = [item for key, item in index["modules"].items() if key.lower() == lowered]
        if len(matches) == 1:
            module = matches[0]
    return dict(module) if module else {}


def _api_index_summary() -> dict[str, int]:
    index = _api_index()
    return {
        "module_count": len(index["modules"]),
        "symbol_count": len(index["symbols"]),
        "stub_symbol_count": sum(1 for item in index["symbols"].values() if item.get("source") == "stub"),
    }


def _api_index() -> dict[str, Any]:
    global _API_INDEX
    if _API_INDEX is not None:
        return _API_INDEX
    modules: dict[str, Any] = {}
    symbols: dict[str, Any] = {}
    roots = _python_api_roots()
    seen_stems: set[str] = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in {"__pycache__", ".mypy_cache"} and not name.startswith(".")]
            for filename in filenames:
                if not filename.endswith(".pyi"):
                    continue
                path = os.path.join(dirpath, filename)
                module_name = _module_name_for_path(path)
                seen_stems.add(os.path.splitext(path)[0])
                _merge_module_api(modules, symbols, module_name, path, source="stub")
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in {"__pycache__", ".mypy_cache"} and not name.startswith(".")]
            for filename in filenames:
                if not filename.endswith(".py") or filename.startswith("_"):
                    continue
                path = os.path.join(dirpath, filename)
                if os.path.splitext(path)[0] in seen_stems:
                    continue
                module_name = _module_name_for_path(path)
                _merge_module_api(modules, symbols, module_name, path, source="python")
    _API_INDEX = {"modules": modules, "symbols": symbols}
    return _API_INDEX


def _python_api_roots() -> list[str]:
    return [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))]


def _module_name_for_path(path: str) -> str:
    root = _python_api_roots()[0]
    package_parent = os.path.dirname(root)
    rel = os.path.relpath(os.path.abspath(path), package_parent)
    module = os.path.splitext(rel)[0].replace(os.sep, ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def _merge_module_api(
    modules: dict[str, Any],
    symbols: dict[str, Any],
    module_name: str,
    path: str,
    *,
    source: str,
) -> None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read(), filename=path)
    except Exception:
        return
    module_doc = ast.get_docstring(tree) or ""
    module_symbols = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            entry = _class_entry(module_name, node, path, source)
            _store_symbol(symbols, entry)
            module_symbols.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            entry = _function_entry(module_name, node, path, source)
            _store_symbol(symbols, entry)
            module_symbols.append(node.name)
    modules[module_name] = {
        "name": module_name,
        "path": _project_rel(path),
        "source": source,
        "doc": module_doc,
        "symbols": sorted(module_symbols),
    }


def _store_symbol(symbols: dict[str, Any], entry: dict[str, Any]) -> None:
    name = entry["name"]
    existing = symbols.get(name)
    if (
        existing is None
        or _symbol_priority(entry) > _symbol_priority(existing)
        or (existing.get("source") != "stub" and entry.get("source") == "stub")
    ):
        symbols[name] = entry
    symbols[entry["qualname"]] = entry


def _symbol_priority(entry: dict[str, Any]) -> int:
    module = str(entry.get("module", ""))
    if module.startswith("Infernux.components.builtin."):
        return 50
    if module.startswith("Infernux.components."):
        return 40
    if module.startswith("Infernux.ui."):
        return 35
    if module.startswith("Infernux.core."):
        return 30
    if module.startswith("Infernux.renderstack."):
        return 25
    if module.startswith("Infernux.lib."):
        return 10
    return 0


def _class_entry(module_name: str, node: ast.ClassDef, path: str, source: str) -> dict[str, Any]:
    methods = []
    properties = []
    attributes = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and not item.target.id.startswith("_"):
            attributes.append({
                "name": item.target.id,
                "type": _expr_to_source(item.annotation),
                "doc": _inline_attribute_doc(node.body, item),
            })
            continue
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    attributes.append({
                        "name": target.id,
                        "type": "",
                        "doc": _inline_attribute_doc(node.body, item),
                    })
            continue
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or item.name.startswith("_"):
            continue
        decorators = [_decorator_name(dec) for dec in item.decorator_list]
        if "property" in decorators:
            properties.append({"name": item.name, "type": _return_annotation(item), "doc": ast.get_docstring(item) or ""})
        elif any(dec.endswith(".setter") for dec in decorators):
            continue
        else:
            methods.append({"name": item.name, "signature": _signature_from_ast(item), "doc": ast.get_docstring(item) or ""})
    return {
        "name": node.name,
        "qualname": f"{module_name}.{node.name}",
        "module": module_name,
        "kind": "class",
        "source": source,
        "path": _project_rel(path),
        "doc": ast.get_docstring(node) or "",
        "bases": [_expr_to_source(base) for base in node.bases],
        "attributes": attributes,
        "methods": methods,
        "properties": properties,
    }


def _function_entry(module_name: str, node: ast.FunctionDef | ast.AsyncFunctionDef, path: str, source: str) -> dict[str, Any]:
    return {
        "name": node.name,
        "qualname": f"{module_name}.{node.name}",
        "module": module_name,
        "kind": "function",
        "source": source,
        "path": _project_rel(path),
        "doc": ast.get_docstring(node) or "",
        "signature": _signature_from_ast(node),
        "attributes": [],
        "methods": [],
        "properties": [],
    }


def _inline_attribute_doc(body: list[ast.stmt], item: ast.stmt) -> str:
    try:
        index = body.index(item)
    except ValueError:
        return ""
    if index + 1 >= len(body):
        return ""
    next_item = body[index + 1]
    if isinstance(next_item, ast.Expr) and isinstance(next_item.value, ast.Constant) and isinstance(next_item.value.value, str):
        return str(next_item.value.value)
    return ""


def _signature_from_ast(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(positional, defaults):
        args.append(_format_arg(arg, default))
    if node.args.vararg:
        args.append("*" + _format_arg(node.args.vararg, None))
    elif node.args.kwonlyargs:
        args.append("*")
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        args.append(_format_arg(arg, default))
    if node.args.kwarg:
        args.append("**" + _format_arg(node.args.kwarg, None))
    ret = _return_annotation(node)
    return f"({', '.join(args)})" + (f" -> {ret}" if ret else "")


def _format_arg(arg: ast.arg, default: ast.expr | None) -> str:
    text = arg.arg
    if arg.annotation is not None:
        text += f": {_expr_to_source(arg.annotation)}"
    if default is not None:
        text += f" = {_expr_to_source(default)}"
    return text


def _return_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return _expr_to_source(node.returns) if node.returns is not None else ""


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _decorator_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _expr_to_source(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def _scan_shaders() -> list[dict[str, Any]]:
    from Infernux.engine.ui import inspector_shader_utils as shader_utils

    results = []
    for root in shader_utils._get_shader_search_roots():
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirs, filenames in os.walk(root):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in {".vert", ".frag", ".glsl", ".shadingmodel"}:
                    continue
                path = os.path.join(dirpath, filename)
                annotations = _parse_shader_annotations(path)
                shader_id = annotations.get("shader_id") or os.path.splitext(filename)[0]
                kind = {
                    ".vert": "vertex",
                    ".frag": "fragment",
                    ".glsl": "library",
                    ".shadingmodel": "shading_model",
                }[ext]
                results.append({
                    "shader_id": shader_id,
                    "kind": kind,
                    "extension": ext,
                    "path": _project_rel(path),
                    "hidden": annotations.get("hidden", False),
                    "properties": shader_utils.parse_shader_properties(path) if ext in {".vert", ".frag"} else [],
                    "imports": annotations.get("imports", []),
                    "targets": annotations.get("targets", []),
                    "shading_model": annotations.get("shading_model", ""),
                    "queue": annotations.get("queue", ""),
                })
    results.sort(key=lambda item: (item["kind"], item["shader_id"], item["path"]))
    return results


def _parse_shader_annotations(path: str) -> dict[str, Any]:
    annotations: dict[str, Any] = {"imports": [], "targets": []}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 120:
                    break
                stripped = line.strip().lstrip("/ ")
                if stripped == "@hidden":
                    annotations["hidden"] = True
                elif stripped.startswith("@shader_id:"):
                    annotations["shader_id"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("@import:"):
                    annotations.setdefault("imports", []).append(stripped.split(":", 1)[1].strip())
                elif stripped.startswith("@target:"):
                    annotations.setdefault("targets", []).append(stripped.split(":", 1)[1].strip())
                elif stripped.startswith("@shading_model:"):
                    annotations["shading_model"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("@queue:"):
                    annotations["queue"] = stripped.split(":", 1)[1].strip()
    except OSError:
        pass
    return annotations


def _shader_usage(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = {item["kind"] for item in candidates}
    usage = {
        "material_binding": [],
        "notes": [],
        "next_tools": ["shader.catalog", "asset.create_builtin_resource", "asset.write_text", "asset.refresh", "material.create"],
    }
    if "vertex" in kinds:
        usage["material_binding"].append("material.vert_shader_name = '<shader_id>'")
    if "fragment" in kinds:
        usage["material_binding"].append("material.frag_shader_name = '<shader_id>'")
    if "shading_model" in kinds:
        usage["notes"].append("Reference this from a .frag surface shader with @shading_model: <shader_id>; do not bind it directly as a material fragment shader.")
    if "library" in kinds:
        usage["notes"].append("Import this from .frag/.vert/.shadingmodel code with @import; do not bind it directly to Material.")
    return usage


def _shader_examples() -> dict[str, str]:
    return {
        "surface_fragment": (
            "#version 450\n\n"
            "@shader_id: my_unlit\n"
            "@shading_model: unlit\n"
            "@queue: 2000\n"
            "@property: baseColor, Color, [1.0, 0.8, 0.4, 1.0]\n"
            "@property: texSampler, Texture2D, white\n\n"
            "void surface(out SurfaceData s) {\n"
            "    s = InitSurfaceData();\n"
            "    vec4 texColor = texture(texSampler, v_TexCoord);\n"
            "    s.albedo = texColor.rgb * material.baseColor.rgb;\n"
            "    s.alpha = texColor.a * material.baseColor.a;\n"
            "}\n"
        ),
        "material_binding": (
            "from Infernux.core.material import Material\n"
            "mat = Material.create_unlit('MyMat')\n"
            "mat.vert_shader_name = 'standard'\n"
            "mat.frag_shader_name = 'my_unlit'\n"
            "mat.set_color('baseColor', 1.0, 0.8, 0.4, 1.0)\n"
        ),
        "shading_model": (
            "@shader_id: my_lighting\n"
            "@import: lighting\n\n"
            "@target: forward\n"
            "void evaluate(in SurfaceData s, out vec4 color) {\n"
            "    color = vec4(s.albedo + s.emission, s.alpha);\n"
            "}\n"
        ),
        "fullscreen_effect_fragment": (
            "#version 450\n\n"
            "@shader_id: my_post_fx\n"
            "@property: intensity, Float, 0.5\n\n"
            "// Use with RenderGraph pass builder: p.fullscreen_quad('my_post_fx')\n"
        ),
    }


def _audio_examples() -> dict[str, str]:
    return {
        "multi_track": (
            "from Infernux.components.builtin import AudioSource, AudioListener\n"
            "from Infernux.core.audio_clip import AudioClip\n\n"
            "source = self.game_object.get_component(AudioSource)\n"
            "source.track_count = 2\n"
            "bgm = AudioClip.load('Assets/Audio/bgm.wav')\n"
            "sfx = AudioClip.load('Assets/Audio/click.wav')\n"
            "source.set_track_clip(0, bgm)\n"
            "source.set_track_clip(1, sfx)\n"
            "source.loop = True\n"
            "source.play(0)\n"
            "source.play_one_shot(sfx, 0.8)\n"
        ),
        "listener": "Attach AudioListener to the main camera GameObject; do not create multiple active listeners.",
    }


def _score(query: str, haystack: str) -> int:
    tokens = [token for token in query.split() if token]
    if not tokens:
        return 1
    return sum(1 for token in tokens if token in haystack)


def _agent_api_guidance() -> list[str]:
    return [
        "Infernux is new and changes quickly. Do not infer unknown APIs from Unity; query them first.",
        "Use api.search(query) for Python/stub-backed APIs, then api.get(symbol_or_module) for signatures and docstrings.",
        "Use component.describe_type(component_type) before component.set_field/component.set_fields.",
        "Use shader.guide, shader.catalog, and shader.describe for shader authoring because shader behavior is C++/annotation/compiler-backed.",
        "When a guide returns data.knowledge_lock.token, pass that token as knowledge_token to gated write tools for that subsystem.",
        "Use mcp.catalog.search or mcp.catalog.recommend before selecting MCP tools for unfamiliar tasks.",
    ]


def _project_rel(path: str) -> str:
    try:
        from Infernux.engine.project_context import get_project_root
        root = get_project_root()
        if root:
            return os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace("\\", "/")
    except Exception:
        pass
    return str(path).replace("\\", "/")


def _register_metadata() -> None:
    for name, summary, category, tags in [
        ("api.subsystems", "List documented engine subsystems and API entry points.", "foundation/api", ["api", "subsystem", "docs"]),
        ("api.get", "Return a subsystem guide or symbol API page.", "foundation/api", ["api", "docs", "symbol"]),
        ("api.search", "Search subsystem guides, symbols, shader IDs, and component names.", "foundation/api", ["api", "search"]),
        ("shader.guide", "Return shader authoring rules and examples.", "shader/guide", ["shader", "guide", "glsl"]),
        ("shader.catalog", "List project and built-in shader IDs.", "shader/catalog", ["shader", "catalog", "vertex", "fragment", "shadingmodel"]),
        ("shader.describe", "Describe shader annotations, properties, and material binding usage.", "shader/catalog", ["shader", "properties", "material"]),
        ("audio.guide", "Return AudioSource, AudioListener, and AudioClip usage guidance.", "audio/guide", ["audio", "guide", "script"]),
    ]:
        register_tool_metadata(
            name,
            summary=summary,
            category=category,
            tags=tags,
            level="foundation" if name.startswith("api.") else "semantic",
            aliases=["script api", "engine api", "how to use", "查询API"] + tags,
            next_suggested_tools=["api.search", "api.get", "component.describe_type"],
        )
