"""Export the theme GitHub template without engine binaries or build outputs."""

import argparse
import json
import importlib.util
from pathlib import Path
import shutil


def export(output):
    repo = Path(__file__).resolve().parents[2]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo / "scripts/templates/theme_plugin", output, dirs_exist_ok=True)
    shutil.copyfile(repo / "scripts/plugins/package.py", output / "package.py")
    shutil.copyfile(repo / "LICENSE", output / "LICENSE")
    schema_path = repo / "python/infernux/theme_data.py"
    shutil.copyfile(schema_path, output / "theme_schema.py")
    spec = importlib.util.spec_from_file_location("infernux_template_theme_schema", schema_path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    table = (repo / "cpp/infernux/function/editor/EditorThemeTable.inl").read_text(encoding="utf-8")
    tokens = schema.theme_token_names(table)
    (output / "theme-tokens.json").write_text(json.dumps(tokens, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(export(parser.parse_args().output))
