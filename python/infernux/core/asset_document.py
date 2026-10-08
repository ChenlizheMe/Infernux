"""Known asset documents: JSON while authoring, cooked CBOR in Players."""
from pathlib import Path
import json


def read_asset_document(path: str):
    if Path(path).suffix.casefold() == ".inxdoc":
        from infernux.lib import _Infernux as native

        return native._read_asset_document(str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode_asset_document(document) -> bytes:
    from infernux.lib import _Infernux as native

    return native._encode_asset_document(document)
