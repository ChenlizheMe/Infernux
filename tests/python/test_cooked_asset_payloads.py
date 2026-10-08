"""Cooked asset document codec contract."""
import pytest
from infernux.core.asset_document import encode_asset_document, read_asset_document
from infernux.lib import _Infernux as native


def test_cooked_document_roundtrip_and_bad_payload(tmp_path):
    document = {"name": "材质", "parameters": [True, None, 42, -2, .375], "nested": {"guid": "a" * 32}}
    payload = encode_asset_document(document)
    assert payload.startswith(b"INXDOCUMENT")
    assert encode_asset_document(document) == payload
    path = tmp_path / "asset.inxdoc"
    path.write_bytes(payload)
    assert read_asset_document(str(path)) == document
    for raw in (payload[:-1], b"{}", b"INXDOCUMENT"):
        with pytest.raises(RuntimeError):
            native._decode_asset_document(raw)
