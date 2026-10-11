"""Scene publication preserves the physical intent of every resident World."""
from __future__ import annotations

import pytest

from infernux.lib import Physics, SceneManager, Vector3, _Infernux as native
from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction


def _hit(x):
    hit = Physics.raycast(Vector3(x, 3, 0), Vector3(0, -1, 0), 6)
    return None if hit is None else (int(hit.body_id), float(hit.distance))


def _queue_change(owner, operation, x):
    owner.transform.position = Vector3(x, 0, 0)
    collider = owner.add_component("BoxCollider")
    if operation == "create_compound":
        collider.enabled = False
        owner.add_component("SphereCollider")
    elif operation in ("disable", "reenable", "move"):
        assert _hit(x) is not None
        if operation in ("disable", "reenable"):
            collider.enabled = False
            if operation == "reenable":
                Physics.sync_transforms()
                assert _hit(x) is None
                collider.enabled = True
        else:
            owner.transform.position = Vector3(x + 4, 0, 0)


@pytest.mark.parametrize("operation", ["create_single", "create_compound", "disable", "reenable", "move"])
@pytest.mark.parametrize("boundary", ["native_commit", "python_additive", "python_rollback"])
@pytest.mark.parametrize("persistent", [False, True])
def test_other_scene_pending_physics_survives_publication(scene, operation, boundary, persistent):
    manager = SceneManager.instance()
    target = manager.create_scene("ReplacementTarget")
    owner = scene.create_game_object("UnaffectedOwner")
    if persistent:
        manager.play()
        manager.pause()
        manager.dont_destroy_on_load(owner)
        manager.prepare_active_scene_replacement()
        assert owner.scene is manager.get_runtime_persistent_scene()
    owner_world = owner.scene
    _queue_change(owner, operation, 20.0)
    document = target.serialize_document()

    if boundary == "native_commit":
        native._preflight_scene_resource_dependencies(document)
        assert target._commit_document(document)
    else:
        def after_publish():
            if boundary == "python_rollback":
                raise RuntimeError("intentional scene publication rejection")

        transaction = SceneDocumentTransaction(
            target, document=document, clear_registries=False, after_publish=after_publish,
        )
        assert transaction.run_to_completion(raise_on_failure=False) is (boundary != "python_rollback")
        if boundary == "python_rollback":
            assert transaction.rolled_back and not transaction.rollback_error

    Physics.sync_transforms()
    assert owner.scene is owner_world
    assert (_hit(20.0) is not None) is (operation not in ("disable", "move"))
    if operation == "move":
        assert _hit(24.0) is not None


@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("operation", ["create_single", "create_compound", "disable", "reenable"])
def test_retained_world_resolves_pending_bodies_before_suspension(scene, operation, rollback):
    # Capture the real empty document before creating the graph to be retained.
    empty = scene.serialize_document()
    owner = scene.create_game_object("RetainedOwner")
    _queue_change(owner, operation, 40.0)
    native._preflight_scene_resource_dependencies(empty)
    token = scene._commit_document_retaining_world(empty)
    assert token is not None
    try:
        # A suspended graph must not re-enter the broadphase through a stale
        # deferred creation/add, even while the rollback token keeps it alive.
        Physics.sync_transforms()
        assert _hit(40.0) is None
        if rollback:
            assert token.rollback()
            assert scene.find("RetainedOwner") is owner
        else:
            token.finalize()
            assert scene.find("RetainedOwner") is None
        Physics.sync_transforms()
        assert (_hit(40.0) is not None) is (rollback and operation != "disable")
    finally:
        if token.is_active:
            token.rollback()


def test_single_prepare_preserves_pending_persistent_reenable(scene):
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    owner = scene.create_game_object("PersistentSingle")
    manager.dont_destroy_on_load(owner)
    manager.prepare_active_scene_replacement()
    _queue_change(owner, "reenable", 60.0)
    transaction = SceneDocumentTransaction(
        scene, document=scene.serialize_document(), clear_registries=True,
        before_commit=manager.prepare_active_scene_replacement,
    )
    assert transaction.run_to_completion(), transaction.error
    Physics.sync_transforms()
    assert owner.scene is manager.get_runtime_persistent_scene()
    assert _hit(60.0) is not None
