"""Retire a model test's catalog entries and its owned source directory together."""
from pathlib import Path

from infernux.lib import AssetRegistry


def remove_model_test_folder(database, folder):
    folder = Path(folder).resolve()
    assert folder.parent == Path(database.assets_root).resolve()
    registry = AssetRegistry.instance()
    # A fixture may import/extract textures, materials and clips or rename its
    # model. Clean its whole owned directory, including those new identities.
    for path in list(folder.rglob("*")):
        if path.is_file() and path.suffix != ".meta":
            guid = database.get_guid_from_path(str(path))
            if guid:
                registry.invalidate_asset(guid)
                assert database.delete_asset(str(path))
    for path in sorted(folder.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)
    folder.rmdir()
