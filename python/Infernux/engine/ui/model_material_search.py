"""Explicit, read-only model material matching against the authoring catalog."""

from pathlib import Path

from Infernux.engine.path_utils import resolved_path


def find_material_candidates(model_path, slots, paths, assets_root, *, scope="local", naming="material"):
    """Return every exact-name candidate; never assign or break a tie.

    Local includes the model directory's subtree. Upwards chooses the nearest
    ancestor subtree with matches for each source, stopping at Assets. Project
    searches all of Assets. No filesystem walk, material load or import occurs.
    """
    if scope not in {"local", "upwards", "project"}:
        raise ValueError(f"Unknown material search scope: {scope}")
    if naming not in {"material", "model_material"}:
        raise ValueError(f"Unknown material naming rule: {naming}")
    root = Path(resolved_path(assets_root))
    def absolute(path):
        value = Path(path)
        return Path(resolved_path(value if value.is_absolute() else root.parent / value))
    model = absolute(model_path)
    by_name = {}
    for raw in paths:
        path = absolute(raw)
        if path.suffix.lower() == ".mat" and path.is_relative_to(root):
            by_name.setdefault(path.stem.casefold(), set()).add(path)
    directories = [root] if scope == "project" else [model.parent]
    if scope == "upwards" and model.is_relative_to(root):
        while directories[-1] != root:
            directories.append(directories[-1].parent)
    result = {}
    for slot in slots:
        source = slot["source_id"]
        if not source:
            continue  # Duplicate/unnamed sources cannot be remapped by name.
        name = source.removeprefix("material/")
        if naming == "model_material":
            name = f"{model.stem}-{name}"
        candidates = by_name.get(name.casefold(), ())
        result[source] = ()
        for directory in directories:
            matches = tuple(sorted((p for p in candidates if p.is_relative_to(directory)),
                                   key=lambda p: (str(p).casefold(), str(p))))
            if matches:
                result[source] = tuple(str(p) for p in matches)
                break
    return result
