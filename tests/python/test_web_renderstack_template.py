"""The Hub's default RenderStack is a portable, typed authoring document."""

import json
import ast
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _web_settings_functions():
    source = ROOT / "external/plugins/infernux_web/native/bootstrap.py"
    names = {
        "infernux_web_render_settings",
        "_iter_web_render_effects",
        "_default_web_render_settings",
        "_apply_web_render_effect",
        "_WebRenderSettingsState",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"))
    namespace = {"Any": Any, "json": json, "_prepare_player_asset_contract": lambda: None}
    exec(
        compile(
            ast.Module(
                [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
                ],
                [],
            ),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace
