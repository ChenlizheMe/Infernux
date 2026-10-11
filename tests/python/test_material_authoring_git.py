"""Independent material edits remain independent in ordinary Git merges."""
import copy
import json
import subprocess

import pytest

from infernux.engine.interaction.material_authoring import MaterialAuthoringSnapshot, dump_material_document
from infernux.engine.ui.project_file_ops import _new_material_document
from infernux.lib import InxMaterial


def _git(path, *args):
    result = subprocess.run(['git', '-c', 'user.name=Material Author', '-c', 'user.email=author@example.invalid',
                             '-c', 'commit.gpgsign=false', '-C', str(path), *map(str, args)],
                            capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _source():
    document = _new_material_document('Shared Bronze')
    document['properties']['baseColor']['value'] = [.26, .17, .075, 1.0]
    document['properties']['metallic']['value'] = .65
    return document


def _native(document):
    native = InxMaterial('Shared Bronze', 'Unlit')
    assert native.deserialize_document(document)
    return native


def test_first_independent_material_edits_merge_without_float_or_layout_churn(tmp_path):
    source = tmp_path/'origin'
    source.mkdir()
    path = source/'Bronze.mat'
    document = _source()
    path.write_text(dump_material_document(document), encoding='utf-8')
    _git(source, 'init', '-b', 'main')
    _git(source, 'add', 'Bronze.mat')
    _git(source, 'commit', '-m', 'New material from the editor template')
    authors = [tmp_path/'author-a', tmp_path/'author-b']
    for index, author in enumerate(authors):
        _git(source, 'clone', '--no-local', source, author)
        target = author/'Bronze.mat'
        material = _native(json.loads(target.read_text()))
        before = material.serialize_document()
        # Confirm real native float32 hydration differs from the source spelling.
        assert before['properties']['metallic']['value'] != .65
        snapshot = MaterialAuthoringSnapshot(str(target), before)
        after = copy.deepcopy(before)
        if index == 0:
            after['properties']['metallic']['value'] = .55
        else:
            after['properties']['baseColor']['value'] = [.32,.2,.1,1]
        assert material.deserialize_document(after)
        target.write_text(snapshot.dump(material.serialize_document()), encoding='utf-8')
        _git(author, 'add', 'Bronze.mat')
        _git(author, 'commit', '-m', 'Edit a distinct material property')
    _git(authors[0], 'fetch', authors[1], 'main')
    _git(authors[0], 'merge', '--no-edit', 'FETCH_HEAD')
    merged = json.loads((authors[0]/'Bronze.mat').read_text())
    assert merged['properties']['metallic']['value'] == pytest.approx(.55)
    assert merged['properties']['baseColor']['value'] == pytest.approx([.32,.2,.1,1])
    assert _native(merged).serialize_document()['properties']['metallic']['value'] == pytest.approx(.55)


def test_material_undo_returns_authored_values_and_omits_resolved_defaults(tmp_path):
    path = tmp_path/'Bronze.mat'
    original = _source()
    path.write_text(dump_material_document(original), encoding='utf-8')
    before = _native(original).serialize_document()
    before['properties']['ResolvedOnly'] = {'type':0, 'value':.125}
    before['_shader_property_order'] = ['metallic','baseColor']
    snapshot = MaterialAuthoringSnapshot(str(path), before)
    after = copy.deepcopy(before)
    after['properties']['metallic']['value'] = .55
    edited = json.loads(snapshot.dump(after))
    assert edited['properties']['baseColor'] == original['properties']['baseColor']
    assert 'ResolvedOnly' not in edited['properties']
    assert '_shader_property_order' not in edited
    assert snapshot.dump(before) == dump_material_document(original)
    # This object never reads a later external edit as a merge input.
    path.write_text('{"external": true}', encoding='utf-8')
    assert json.loads(snapshot.dump(after)) == edited


def test_material_property_addition_removal_and_new_asset_first_publication(tmp_path):
    path = tmp_path/'New.mat'
    before = {'properties': {'old': {'type':0,'value':1}}}
    snapshot = MaterialAuthoringSnapshot(str(path), before)
    after = {'properties': {'new': {'type':0,'value':2}}}
    assert json.loads(snapshot.dump(after)) == after
    assert json.loads(snapshot.dump(before)) == before
