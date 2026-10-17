from __future__ import annotations

from test_integration_scene import (
    AssetRegistry,
    ComponentRef,
    GameObjectRef,
    InxComponent,
    InxMaterial,
    Path,
    PythonComponentRestoreError,
    SceneDocumentTransaction,
    SceneDocumentTransactionState,
    SceneFileManager,
    SceneManager,
    Vector3,
    _AdditiveSceneComponent,
    _CameraCleanupSceneComponent,
    _ObjectGraphRefsComponent,
    _PlayLifecycleResetComponent,
    _RequiresRigidbodyComponent,
    _SingleInstanceSceneComponent,
    _StrictSceneComponent,
    _capture_exception,
    _cds_alive_count,
    _native_record,
    _python_records,
    _strip_prefab_runtime_fields,
    deserialize_game_object_document_transactionally,
    deserialize_scene_document_transactionally,
    encode_scene_document,
    instantiate_game_object_document_transactionally,
    json,
    preflight_scene_python_components,
    pytest,
    replace_scene_python_components_for_play,
    threading,
)


class TestSceneSerialization:
    def test_native_play_snapshot_restores_scene_without_python_document(self, scene):
        source = scene.create_game_object("NativeSnapshotSource")
        source_id = source.id
        source.transform.position = Vector3(1.0, 2.0, 3.0)
        snapshot = scene._capture_play_mode_snapshot()

        source.name = "NativeSnapshotMutated"
        source.transform.position = Vector3(9.0, 8.0, 7.0)
        transaction = SceneDocumentTransaction(
            scene,
            document=snapshot,
            borrow_document=True,
        )

        assert transaction.run_to_completion() is True
        restored = scene.find_by_id(source_id)
        assert restored is not None
        assert restored.name == "NativeSnapshotSource"
        position = restored.transform.position
        assert (position.x, position.y, position.z) == pytest.approx((1.0, 2.0, 3.0))
        assert transaction.phase_timings_ms["resources"] >= 0.0
        assert transaction.phase_timings_ms["native_commit"] >= 0.0

    def test_path_scene_transaction_reads_and_validates_on_worker(self, scene, tmp_path):
        scene.create_game_object("WorkerReadSource")
        document = scene.serialize_document()
        document["objects"][0]["name"] = "WorkerReadTarget"
        path = tmp_path / "worker-read.scene"
        path.write_text(json.dumps(encode_scene_document(document)), encoding="utf-8")
        transaction = SceneDocumentTransaction(scene, path=path)

        assert transaction.run_to_completion(raise_on_failure=False) is True
        assert transaction.ran_on_worker is True
        assert transaction.state is SceneDocumentTransactionState.COMPLETED
        assert scene.find("WorkerReadTarget") is not None

    def test_worker_structural_failure_preserves_live_scene(self, scene, tmp_path):
        existing = scene.create_game_object("WorkerFailureExisting")
        original_document = scene.serialize_document()
        candidate = json.loads(json.dumps(original_document))
        candidate["unexpected"] = True
        path = tmp_path / "invalid-worker.scene"
        path.write_text(json.dumps(encode_scene_document(candidate)), encoding="utf-8")
        transaction = SceneDocumentTransaction(scene, path=path)

        assert transaction.run_to_completion(raise_on_failure=False) is False
        assert transaction.ran_on_worker is True
        assert transaction.state is SceneDocumentTransactionState.FAILED
        assert "unknown field 'unexpected'" in transaction.error
        assert scene.serialize_document() == original_document
        assert scene.find("WorkerFailureExisting") is existing

    @pytest.mark.parametrize("corruption", ["rigidbody_mass"])
    def test_worker_type_validator_rejects_physics_document_before_publish(
        self,
        scene,
        tmp_path,
        corruption,
    ):
        existing = scene.create_game_object("WorkerTypedValidationExisting")
        component_type = None
        if corruption == "rigidbody_mass":
            component_type = "Rigidbody"
        elif corruption == "box_size":
            component_type = "BoxCollider"
        elif corruption == "sphere_radius":
            component_type = "SphereCollider"
        elif corruption == "capsule_direction":
            component_type = "CapsuleCollider"
        elif corruption == "cylinder_direction":
            component_type = "CylinderCollider"
        elif corruption == "mesh_convex":
            component_type = "MeshCollider"
        component = existing.transform if component_type is None else existing.add_component(component_type)
        original_document = scene.serialize_document()
        candidate = json.loads(json.dumps(original_document))
        object_document = candidate["objects"][0]

        if corruption == "transform_missing_scale":
            object_document["transform"].pop("scale")
            expected_path = "Scene.objects[0].transform"
        else:
            component_document = object_document["components"][0]["data"]
            expected_path = "Scene.objects[0].components[0]"
            if corruption == "rigidbody_mass":
                component_document["mass"] = "heavy"
            elif corruption == "box_size":
                component_document["size"] = [1.0, 0.0, 1.0]
            elif corruption == "sphere_radius":
                component_document["radius"] = 0.0
            elif corruption == "capsule_direction":
                component_document["direction"] = 9
            elif corruption == "cylinder_direction":
                component_document["direction"] = 9
            else:
                component_document["convex"] = "yes"

        path = tmp_path / f"invalid-{corruption}.scene"
        path.write_text(json.dumps(encode_scene_document(candidate)), encoding="utf-8")
        transaction = SceneDocumentTransaction(scene, path=path)

        assert transaction.run_to_completion(raise_on_failure=False) is False
        assert transaction.ran_on_worker is True
        assert transaction.state is SceneDocumentTransactionState.FAILED
        assert transaction.document is None
        assert expected_path in transaction.error
        assert scene.serialize_document() == original_document
        assert scene.find("WorkerTypedValidationExisting") is existing
        if component_type is None:
            assert existing.transform is component
        else:
            assert existing.get_component(component_type) is component

    @pytest.mark.parametrize("failure", ["wrong_type", "embedded_material"])
    def test_resource_preflight_failure_preserves_live_scene(self, scene, tmp_path, failure):
        existing = scene.create_game_object("ResourcePreflightExisting")
        renderer = existing.add_component("MeshRenderer")
        original_document = scene.serialize_document()
        candidate = json.loads(json.dumps(original_document))
        renderer_document = candidate["objects"][0]["components"][0]["data"]

        if failure == "wrong_type":
            asset_database = AssetRegistry.instance().get_asset_database()
            asset_path = Path(asset_database.assets_root) / f"{tmp_path.name}-wrong-type.physicMaterial"
            asset_path.write_text(
                json.dumps(
                    {
                        "friction": 0.5,
                        "bounciness": 0.0,
                        "friction_combine": 0,
                        "bounce_combine": 0,
                    }
                ),
                encoding="utf-8",
            )
            renderer_document["meshAssetGuid"] = asset_database.import_asset(str(asset_path)).guid
        else:
            renderer_document["materials"] = [{"material": {}}]

        path = tmp_path / f"resource-{failure}.scene"
        path.write_text(json.dumps(encode_scene_document(candidate)), encoding="utf-8")
        transaction = SceneDocumentTransaction(scene, path=path)

        assert transaction.run_to_completion(raise_on_failure=False) is False
        assert transaction.ran_on_worker is True
        assert transaction.state is SceneDocumentTransactionState.FAILED
        if failure == "embedded_material":
            assert transaction.document is None
        else:
            assert transaction.document is not None
        assert "Scene.objects[0].components[0]" in transaction.error
        assert scene.serialize_document() == original_document
        assert scene.find("ResourcePreflightExisting") is existing
        assert existing.get_component("MeshRenderer") is renderer

    def test_scene_resource_preflight_restores_registered_missing_and_cleared_references_together(
        self, scene, tmp_path
    ):
        scene.create_game_object("MissingResourceReference").add_component("MeshRenderer")
        renderer = scene.create_game_object("ClearedEmbeddedTexture").add_component("MeshRenderer")
        embedded = InxMaterial.create_default_unlit()
        embedded.set_texture("texSampler", "white")
        embedded.clear_texture("texSampler")
        renderer.material = embedded
        scene.create_game_object("ResourcePreflightSuccess").add_component("BoxCollider")
        database = AssetRegistry.instance().get_asset_database()
        asset_path = Path(database.assets_root) / f"{tmp_path.name}-valid.physicMaterial"
        asset_path.write_text(
            json.dumps({"friction": 0.65, "bounciness": 0.1, "friction_combine": 2, "bounce_combine": 3}),
            encoding="utf-8",
        )
        guid = database.import_asset(str(asset_path)).guid
        texture_path = asset_path.with_suffix(".ppm")
        texture_path.write_text("P3\n1 1\n255\n255 0 0\n", encoding="ascii")
        texture_guid = database.import_asset(str(texture_path)).guid
        texture_artifact = Path(database.assets_root).parent / "Library/Artifacts/Texture" / f"{texture_guid}.inxtex"
        assert texture_artifact.is_file()
        texture_artifact.write_bytes(b"invalid texture artifact")
        textured = scene.create_game_object("UnreadableTexture").add_component("MeshRenderer")
        textured.material = InxMaterial.create_default_unlit()
        textured.material.set_texture("texSampler", texture_guid)
        try:
            document = scene.serialize_document()
            by_name = {record["name"]: record for record in document["objects"]}
            data = by_name["MissingResourceReference"]["components"][0]["data"]
            data["meshAssetGuid"], data["materials"] = "missing-scene-mesh-guid", ["missing-scene-material-guid"]
            by_name["ResourcePreflightSuccess"]["components"][0]["data"]["physic_material_guid"] = guid
            path = tmp_path / "resource-references.scene"
            path.write_text(json.dumps(encode_scene_document(document)), encoding="utf-8")
            transaction = SceneDocumentTransaction(scene, path=path)
            assert transaction.run_to_completion(raise_on_failure=False)
            assert transaction.state is SceneDocumentTransactionState.COMPLETED
            assert (
                scene.find("UnreadableTexture").get_component("MeshRenderer").material.get_texture("texSampler")
                == texture_guid
            )
            missing = scene.find("MissingResourceReference").get_component("MeshRenderer").serialize_document()
            assert missing["meshAssetGuid"] == "missing-scene-mesh-guid"
            assert missing["materials"] == ["missing-scene-material-guid"]
            assert (
                scene.find("ResourcePreflightSuccess")
                .get_component("BoxCollider")
                .serialize_document()["physic_material_guid"]
                == guid
            )
            assert (
                scene.find("ClearedEmbeddedTexture").get_component("MeshRenderer").material.get_texture("texSampler")
                == ""
            )
            assert SceneDocumentTransaction(scene, document=scene.serialize_document()).run_to_completion(
                raise_on_failure=False
            )
            assert (
                scene.find("ClearedEmbeddedTexture").get_component("MeshRenderer").material.get_texture("texSampler")
                == ""
            )
        finally:
            database.delete_asset(str(asset_path))
            database.delete_asset(str(texture_path))

    def test_scene_navigation_undo_redo_restores_both_session_scenes(
        self,
        scene,
        monkeypatch,
    ):
        from infernux.engine.interaction import (
            ContextRestoreStatus,
            DocumentOpenStatus,
            DocumentKind,
            EditorInteractionCore,
        )
        from infernux.engine.undo import UndoManager

        previous_core = EditorInteractionCore._instance
        previous_manager = SceneFileManager._instance
        previous_undo = UndoManager._instance
        core = EditorInteractionCore()
        manager = SceneFileManager()
        undo = UndoManager(core.action_journal)
        try:
            monkeypatch.setattr(manager, "_prepare_native_scene_swap", lambda: None)
            monkeypatch.setattr(manager, "_restore_camera_state", lambda _path: None)
            monkeypatch.setattr(manager, "sync_all_prefab_instances", lambda _scene: None)
            core.document_open.register(
                DocumentKind.SCENE,
                manager.restore_document_locator,
            )

            def restore_context(context, _phase):
                if context.scene is None:
                    return ContextRestoreStatus.READY
                result = core.document_open.resolve_or_open(context.scene)
                return (
                    ContextRestoreStatus.READY
                    if result.status is DocumentOpenStatus.READY
                    else ContextRestoreStatus.FAILED
                )

            undo.set_context_hooks(core.capture_context, restore_context)
            scene.create_game_object("SceneAOnly")
            manager.mark_dirty()
            scene_a_id = manager.document_id

            manager._stage_scene_navigation()
            assert manager._do_new_scene() is True
            scene.create_game_object("SceneBOnly")
            manager.mark_dirty()
            scene_b_id = manager.document_id
            assert scene_b_id != scene_a_id
            assert undo.undo_description == "New Scene"

            undo.undo()
            assert scene.find("SceneAOnly") is not None
            assert scene.find("SceneBOnly") is None

            undo.redo()
            assert scene.find("SceneAOnly") is None
            assert scene.find("SceneBOnly") is not None
        finally:
            core.shutdown()
            EditorInteractionCore._instance = previous_core
            SceneFileManager._instance = previous_manager
            UndoManager._instance = previous_undo

    def test_scene_transaction_cancel_and_owner_thread_guard(self, scene):
        existing = scene.create_game_object("CancelledTransactionExisting")
        transaction = SceneDocumentTransaction(scene, document=scene.serialize_document())
        transaction.start()
        errors = []

        thread = threading.Thread(target=lambda: _capture_exception(transaction.poll, errors))
        thread.start()
        thread.join(timeout=2.0)

        assert len(errors) == 1
        assert "owner thread" in str(errors[0])
        assert transaction.cancel() is True
        assert transaction.state is SceneDocumentTransactionState.CANCELLED
        assert transaction.poll() is True
        assert scene.find("CancelledTransactionExisting") is existing

    def test_scene_transaction_cancel_discards_prepared_component_slots(self, scene):
        owner = scene.create_game_object("CancelPreparedGraph")
        owner.add_py_component(_StrictSceneComponent())
        alive_before = _cds_alive_count(_StrictSceneComponent)
        transaction = SceneDocumentTransaction(scene, document=scene.serialize_document())

        transaction.start()
        assert transaction.poll() is False
        assert transaction.state is SceneDocumentTransactionState.RESOURCES_READY
        assert transaction.poll() is False
        assert transaction.state is SceneDocumentTransactionState.READY_TO_COMMIT
        assert _cds_alive_count(_StrictSceneComponent) == alive_before + 1

        assert transaction.cancel() is True
        assert transaction.state is SceneDocumentTransactionState.CANCELLED
        assert _cds_alive_count(_StrictSceneComponent) == alive_before

    def test_scene_creation_atomic_save_and_reopen_keep_current_canonical_document(self, scene, tmp_path):
        from infernux.engine.scene_manager import _empty_scene_document

        empty = _empty_scene_document("BlankEditorScene")
        assert set(empty) == {"name", "isPlaying", "objects"}
        assert scene._commit_document(empty)
        assert scene.name == "BlankEditorScene" and scene.get_root_objects() == []
        persistent = scene.create_game_object("Persistent")
        persistent.transform.position = Vector3(42, 0, 0)
        for index in range(511):
            scene.create_game_object(f"Dense_{index}")
        serialized = scene.serialize()
        assert "Persistent" in serialized and len(json.loads(serialized)["objects"]) == 512
        path = tmp_path / "canonical.scene"
        path.write_text("old scene content", encoding="utf-8")
        assert scene.save_to_file(str(path))
        content = path.read_text(encoding="utf-8")
        document = json.loads(content)
        assert len(document["objects"]) == 512 and document["objects"][0]["name"] == "Persistent"
        assert content == json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        assert all(line.strip().startswith('"name": "Dense_') for line in content.splitlines() if '"Dense_' in line)
        original = path.read_bytes()
        assert b"\r" not in original and list(tmp_path.glob("canonical.scene.tmp.*")) == []
        assert scene.save_to_file(str(path)) and path.read_bytes() == original
        manager = SceneManager.instance()
        loaded = manager.create_scene("loaded_scene")
        manager.set_active_scene(loaded)
        assert SceneDocumentTransaction(loaded, path=path).run_to_completion()
        assert loaded.find("Persistent").transform.position.x == pytest.approx(42)
        assert len(loaded.get_all_objects()) == 512

    def test_runtime_scene_publish_rebuilds_scaled_collider_from_current_world_transform(self, scene):
        from infernux.lib import Physics

        platform = scene.create_game_object("RuntimeScaledPlatform")
        platform.transform.position = Vector3(0.0, 0.4, 16.0)
        platform.transform.local_scale = Vector3(5.0, 0.8, 18.0)
        platform.add_component("BoxCollider")
        document = scene.serialize_document()

        sm = SceneManager.instance()
        sm.play()
        transaction = SceneDocumentTransaction(scene, document=document)
        assert transaction.run_to_completion() is True

        sm._start_active_scene_for_play()

        edge_hits = Physics.overlap_box(
            Vector3(0.0, 0.4, 24.0),
            Vector3(0.1, 0.1, 0.1),
        )
        assert any(collider.game_object.name == "RuntimeScaledPlatform" for collider in edge_hits)

    def test_retained_world_restores_exact_instances_when_native_commit_fails(self, scene):
        existing = scene.create_game_object("RetainedNativeFailure")
        rigidbody = existing.add_component("Rigidbody")
        collider = existing.add_component("BoxCollider")
        transform = existing.transform
        document = scene.serialize_document()
        document["unknown"] = 1

        assert scene._commit_document_retaining_world(document) is None
        assert scene.find("RetainedNativeFailure") is existing
        assert existing.transform is transform
        assert existing.get_component("Rigidbody") is rigidbody
        assert existing.get_component("BoxCollider") is collider

    @pytest.mark.parametrize(
        "corruption", ["duplicate_object_id", "invalid_component_field", "invalid_python_descriptor"]
    )
    def test_transactional_deserialize_preserves_live_graph(self, scene, corruption):
        first = scene.create_game_object("TransactionFirst")
        second = scene.create_game_object("TransactionSecond")
        first_rb = first.add_component("Rigidbody")
        first.add_py_component(_StrictSceneComponent())
        second.add_component("Rigidbody")
        first_rb.mass = 6.5

        original_document = scene.serialize_document()
        candidate = json.loads(json.dumps(original_document))
        first_doc, second_doc = candidate["objects"]

        if corruption == "duplicate_object_id":
            second_doc["id"] = first_doc["id"]
        elif corruption == "duplicate_component_id":
            second_doc["transform"]["component_id"] = first_doc["transform"]["component_id"]
        elif corruption == "invalid_component_field":
            _native_record(first_doc, "Rigidbody")["data"]["mass"] = "heavy"
        elif corruption == "invalid_main_camera":
            candidate["mainCameraComponentId"] = first_doc["components"][0]["component_id"]
        elif corruption == "unknown_scene_field":
            candidate["unexpected"] = True
        elif corruption == "unknown_object_field":
            first_doc["unexpected"] = True
        elif corruption == "invalid_layer":
            first_doc["layer"] = 32
        elif corruption == "invalid_python_descriptor":
            _python_records(first_doc)[0]["unexpected"] = True
        elif corruption == "empty_python_script_guid":
            descriptor = _python_records(first_doc)[0]
            descriptor["type_id"] = descriptor["type_id"].replace("python:", "python::", 1)

        assert scene._commit_document(candidate) is False
        assert scene.serialize_document() == original_document
        assert scene.find("TransactionFirst") is first
        assert scene.find("TransactionSecond") is second
        assert first.get_component("Rigidbody") is first_rb
        assert first_rb.mass == pytest.approx(6.5)

    def test_python_field_preflight_rejects_invalid_value_without_replacing_scene(self, scene):
        existing = scene.create_game_object("PythonPreflightExisting")
        component = _StrictSceneComponent()
        component.value = 19
        existing.add_py_component(component)
        original_document = scene.serialize_document()
        alive_before = _cds_alive_count(_StrictSceneComponent)
        candidate = json.loads(json.dumps(original_document))
        candidate["objects"][0]["name"] = "RejectedPythonCandidate"
        _python_records(candidate["objects"][0])[0]["data"]["value"] = "not-an-int"

        with pytest.raises(PythonComponentRestoreError, match="INT field requires an integer"):
            deserialize_scene_document_transactionally(scene, candidate)
        assert scene.serialize_document() == original_document
        assert scene.find("PythonPreflightExisting") is existing
        assert existing.get_py_component(_StrictSceneComponent) is component
        assert component.value == 19
        assert scene.find("RejectedPythonCandidate") is None
        assert not scene.has_pending_py_components()
        assert _cds_alive_count(_StrictSceneComponent) == alive_before

    def test_after_publish_hook_failure_rolls_back_complete_candidate_graph(self, scene):
        existing = scene.create_game_object("AfterPublishSource")
        original_document = scene.serialize_document()
        candidate = json.loads(json.dumps(original_document))
        candidate["objects"][0]["name"] = "AfterPublishCandidate"

        def fail_after_publish():
            raise RuntimeError("intentional after_publish failure")

        transaction = SceneDocumentTransaction(
            scene,
            document=candidate,
            after_publish=fail_after_publish,
        )

        assert transaction.run_to_completion(raise_on_failure=False) is False
        assert transaction.rolled_back is True
        assert "intentional after_publish failure" in transaction.error
        assert scene.serialize_document() == original_document
        assert scene.find("AfterPublishSource") is existing
        assert scene.find("AfterPublishCandidate") is None

    def test_play_snapshot_preserves_native_required_component_type_names(self, scene):
        owner = scene.create_game_object("PlaySnapshotRequiredComponent")
        owner.add_component("Rigidbody")
        owner.add_py_component(_RequiresRigidbodyComponent())

        prepared = preflight_scene_python_components(scene._capture_play_mode_snapshot())

        assert len(prepared.components) == 1
        assert prepared.components[0].type_name == "_RequiresRigidbodyComponent"
        prepared.discard()

    @pytest.mark.parametrize("pre_resolve", [False, True])
    def test_scene_replacement_camera_reference_is_null_during_retired_disable(self, scene, pre_resolve):
        camera_object = scene.create_game_object("RetiredCamera")
        camera_object.add_component("Camera")
        owner = scene.create_game_object("CameraCleanupOwner")
        probe = owner.add_py_component(_CameraCleanupSceneComponent())
        probe.target_component = ComponentRef(go_id=camera_object.id, component_type="Camera")
        if pre_resolve:
            assert probe.target_component.is_valid
        _CameraCleanupSceneComponent._cleanup_results = []
        document = scene.serialize_document()

        assert deserialize_scene_document_transactionally(scene, document)
        assert _CameraCleanupSceneComponent._cleanup_results == ["missing"]
        restored = scene.find("CameraCleanupOwner").get_py_component(_CameraCleanupSceneComponent)
        assert restored.target_component.is_valid

    @pytest.mark.parametrize("native_type,nested", [("Light", True), ("Rigidbody", False)])
    def test_play_snapshot_preserves_references_to_native_only_objects(self, scene, native_type, nested):
        owner = scene.create_game_object("ReferenceOwner")
        target = scene.create_game_object("NativeOnlyTarget")
        if nested:
            target.set_parent(scene.create_game_object("NativeOnlyParent"))
        if native_type != "Transform":
            target.add_component(native_type)
        refs = owner.add_py_component(_ObjectGraphRefsComponent())
        refs.target_component = ComponentRef(go_id=target.id, component_type=native_type)
        snapshot = scene._capture_play_mode_snapshot()
        object_ids, native_types, _ = snapshot._python_component_records()
        assert target.id in object_ids
        assert (target.id, native_type) in native_types

        assert replace_scene_python_components_for_play(scene, snapshot)
        restored = owner.get_py_component(_ObjectGraphRefsComponent)
        assert restored is not refs
        assert restored.target_component is not None
        assert restored.target_component.game_object.id == target.id
        assert scene.find_by_id(target.id) is target

    def test_python_preflight_rejects_duplicate_disallow_multiple_component(self, scene):
        owner = scene.create_game_object("StrictDisallowMultiple")
        owner.add_py_component(_SingleInstanceSceneComponent())
        original_document = scene.serialize_document()
        candidate = json.loads(json.dumps(original_document))
        duplicate = json.loads(json.dumps(_python_records(candidate["objects"][0])[0]))
        duplicate_id = duplicate["component_id"] + 100000
        duplicate["component_id"] = duplicate_id
        candidate["objects"][0]["components"].append(duplicate)

        transaction = SceneDocumentTransaction(scene, document=candidate)

        assert transaction.run_to_completion(raise_on_failure=False) is False
        assert "disallows multiple instances" in transaction.error
        assert scene.serialize_document() == original_document
        assert scene.find("StrictDisallowMultiple") is owner

    def test_mixed_scene_restore_preserves_python_identity_order_and_current_field_defaults(self, scene):
        root = scene.create_game_object("MixedCurrentScene")
        root.add_component("BoxCollider")
        strict = root.add_py_component(_StrictSceneComponent())
        strict.value, strict.execution_order = 19, 37
        root.add_component("Light")
        additive = root.add_py_component(_AdditiveSceneComponent())
        additive.value, additive.label = 23, "authored"
        old_id = strict.component_id
        document = json.loads(json.dumps(scene.serialize_document()))
        records = document["objects"][0]["components"]
        types = [record["type_id"] for record in records]
        assert types[0] == "native:infernux.BoxCollider" and types[2] == "native:infernux.Light"
        assert records[1]["execution_order"] == 37
        descriptor = records[1]
        script_guid, type_guid, module_name, qualified_name = descriptor["type_id"][len("python:") :].split(":")
        assert script_guid == strict._script_guid and len(script_guid) == 32
        assert type_guid == type(strict)._get_type_guid() and len(type_guid) == 32
        assert (module_name, qualified_name) == (type(strict).__module__, type(strict).__qualname__)
        descriptor["data"]["removed_field"] = 1
        records[3]["data"].pop("label")
        assert deserialize_scene_document_transactionally(scene, document)
        restored_root = scene.find("MixedCurrentScene")
        restored = restored_root.get_py_component(_StrictSceneComponent)
        current = restored_root.get_py_component(_AdditiveSceneComponent)
        assert restored_root is not root and restored is not strict
        assert restored.component_id == old_id and restored.value == 19 and restored.execution_order == 37
        assert current.value == 23 and current.label == "default"
        assert "removed_field" not in restored._serialize_fields_document()
        assert [record["type_id"] for record in restored_root.serialize_document()["components"]] == types
        assert scene.has_pending_py_components() is False

    def test_play_domain_replacement_resets_lifecycle_before_scene_start(self, scene):
        _PlayLifecycleResetComponent.awake_calls = 0
        _PlayLifecycleResetComponent.start_calls = 0
        _PlayLifecycleResetComponent.destroy_calls = 0
        _PlayLifecycleResetComponent.active_instance = None
        root = scene.create_game_object("PlayLifecycleReset")
        edit_component = root.add_py_component(_PlayLifecycleResetComponent())
        snapshot = scene._capture_play_mode_snapshot()

        assert edit_component._awake_called is True
        assert replace_scene_python_components_for_play(scene, snapshot) is True

        play_component = root.get_py_component(_PlayLifecycleResetComponent)
        assert play_component is not edit_component
        assert play_component._awake_called is False
        assert play_component._has_started is False
        assert not hasattr(play_component, "_runtime_token")
        assert _PlayLifecycleResetComponent.destroy_calls == 1
        assert _PlayLifecycleResetComponent.active_instance is None

        scene.set_playing(True)
        scene.start()

        assert play_component._runtime_token == "initialized-in-awake"
        assert play_component._awake_called is True
        assert play_component._has_started is True
        assert _PlayLifecycleResetComponent.awake_calls == 2
        assert _PlayLifecycleResetComponent.start_calls == 1
        assert _PlayLifecycleResetComponent.destroy_calls == 1
        assert _PlayLifecycleResetComponent.active_instance is play_component

    def test_play_mode_isolates_and_restores_every_resident_scene(self, scene, tmp_path, monkeypatch):
        from infernux.engine.interaction import DocumentRegistry
        from infernux.engine.play_mode import PlayModeManager
        from infernux.engine.project_context import get_project_root, set_project_root

        previous_root = get_project_root()
        previous_scene_files = SceneFileManager._instance
        project_root = tmp_path / "Project"
        assets = project_root / "Assets"
        assets.mkdir(parents=True)
        primary_path = assets / "Primary.scene"
        additive_path = assets / "Additive.scene"
        native = SceneManager.instance()
        additive = native.create_scene("AdditivePlayIsolation")

        try:
            set_project_root(str(project_root))
            scene_files = SceneFileManager()
            monkeypatch.setattr(scene_files, "_restore_camera_state", lambda _path: None)
            monkeypatch.setattr(scene_files, "_remember_last_scene", lambda _path: None)
            scene_files._current_scene_path = str(primary_path)
            scene_files._replace_scene_document(
                kind="scene",
                resource_path=str(primary_path),
                title="Primary",
                dirty=False,
            )
            primary_document_id = scene_files.document_id
            additive_document_id = scene_files.register_loaded_scene(
                additive,
                str(additive_path),
            )

            primary_object = scene.create_game_object("PrimaryPlayObject")
            additive_object = additive.create_game_object("AdditivePlayObject")
            primary_edit = primary_object.add_py_component(_StrictSceneComponent())
            additive_edit = additive_object.add_py_component(_StrictSceneComponent())
            primary_edit.value = 11
            additive_edit.value = 22

            registry = DocumentRegistry.instance()
            registry.mark_changed(primary_document_id)
            registry.mark_changed(additive_document_id)
            primary_revision = registry.require(primary_document_id).revision
            additive_revision = registry.require(additive_document_id).revision
            assert scene_files.activate_loaded_scene(additive)

            play = PlayModeManager()
            play._save_scene_state()
            assert {scene, additive}.issubset({item.scene for item in play._scene_backups})
            assert next(item for item in play._scene_backups if item.was_active).scene is additive

            assert play._prepare_loaded_scenes_for_play()
            primary_play = primary_object.get_py_component(_StrictSceneComponent)
            additive_play = additive_object.get_py_component(_StrictSceneComponent)
            assert primary_play is not primary_edit
            assert additive_play is not additive_edit
            primary_object.name = "PrimaryRuntimeMutation"
            additive_object.name = "AdditiveRuntimeMutation"
            primary_play.value = 101
            additive_play.value = 202

            assert play._restore_loaded_scenes_after_play()
            restored_primary = scene.find("PrimaryPlayObject")
            restored_additive = additive.find("AdditivePlayObject")
            assert restored_primary is not None
            assert restored_additive is not None
            assert restored_primary.get_py_component(_StrictSceneComponent).value == 11
            assert restored_additive.get_py_component(_StrictSceneComponent).value == 22
            assert native.get_active_scene() is additive
            assert scene_files.document_id == additive_document_id
            assert registry.require(primary_document_id).revision == primary_revision
            assert registry.require(additive_document_id).revision == additive_revision
        finally:
            native.set_active_scene(scene)
            native.unload_scene(additive)
            SceneFileManager._instance = previous_scene_files
            set_project_root(previous_root)

    def test_stop_recreates_authored_scene_set_after_runtime_single_replacement(self, scene, tmp_path, monkeypatch):
        from infernux.engine.interaction import DocumentRegistry
        from infernux.engine.play_mode import PlayModeManager
        from infernux.engine.project_context import get_project_root, set_project_root

        previous_root = get_project_root()
        previous_scene_files = SceneFileManager._instance
        project_root = tmp_path / "Project"
        assets = project_root / "Assets"
        assets.mkdir(parents=True)
        primary_path = assets / "Primary.scene"
        additive_path = assets / "Additive.scene"
        native = SceneManager.instance()
        additive = native.create_scene("AuthoredAdditive")

        try:
            set_project_root(str(project_root))
            scene_files = SceneFileManager()
            monkeypatch.setattr(scene_files, "_restore_camera_state", lambda _path: None)
            monkeypatch.setattr(scene_files, "_remember_last_scene", lambda _path: None)
            scene_files._current_scene_path = str(primary_path)
            scene_files._replace_scene_document(
                kind="scene",
                resource_path=str(primary_path),
                title="Primary",
                dirty=False,
            )
            primary_document_id = scene_files.document_id
            additive_document_id = scene_files.register_loaded_scene(
                additive,
                str(additive_path),
            )
            primary_object = scene.create_game_object("PrimaryBeforePlay")
            additive_object = additive.create_game_object("AdditiveBeforePlay")
            primary_object.add_py_component(_StrictSceneComponent()).value = 31
            additive_object.add_py_component(_StrictSceneComponent()).value = 47
            assert scene_files.activate_loaded_scene(scene)

            registry = DocumentRegistry.instance()
            registry.mark_changed(primary_document_id)
            registry.mark_changed(additive_document_id)
            revisions = {
                primary_document_id: registry.require(primary_document_id).revision,
                additive_document_id: registry.require(additive_document_id).revision,
            }

            play = PlayModeManager()
            play._save_scene_state()
            assert play._prepare_loaded_scenes_for_play()
            additive_world_id = int(additive.world_id)

            # A Play-time Single load retains the active native Scene and
            # destroys every additive Scene. A later Additive load may also
            # leave runtime-only worlds resident at the Stop boundary.
            native.unload_scene(additive)
            runtime_only = native.create_scene("RuntimeOnly")
            runtime_only.create_game_object("MustDisappearOnStop")
            runtime_only_world_id = int(runtime_only.world_id)

            assert play._restore_loaded_scenes_after_play()

            restored_additive = scene_files.scene_for_document(additive_document_id)
            assert restored_additive is not None
            assert int(restored_additive.world_id) != additive_world_id
            assert restored_additive.find("AdditiveBeforePlay") is not None
            assert restored_additive.find("AdditiveBeforePlay").get_py_component(_StrictSceneComponent).value == 47
            assert scene.find("PrimaryBeforePlay") is not None
            assert native.get_scene_by_world_id(runtime_only_world_id) is None
            assert native.get_active_scene() is scene
            assert scene_files.document_id == primary_document_id
            assert scene_files.document_id_for_scene(restored_additive) == additive_document_id
            for document_id, revision in revisions.items():
                assert registry.require(document_id).revision == revision
        finally:
            native.set_active_scene(scene)
            for index in range(int(native.scene_count) - 1, -1, -1):
                candidate = native.get_scene_at(index)
                if candidate is not None and candidate is not scene:
                    native.unload_scene(candidate)
            SceneFileManager._instance = previous_scene_files
            set_project_root(previous_root)

    def test_python_component_replacement_rejects_new_registry_constraints(self, scene):
        class ReloadSource(InxComponent):
            pass

        class ReloadTarget(InxComponent):
            _incompatible_components_ = ("MeshRenderer",)

        root = scene.create_game_object("ConstraintAwareReplacement")
        assert root.add_component("MeshRenderer") is not None
        source = root.add_py_component(ReloadSource())

        with pytest.raises(RuntimeError, match="replacement failed"):
            root.replace_py_component(source, ReloadTarget())

        assert root.get_py_components() == [source]
        assert source._cpp_component is not None

    def test_scene_restore_remaps_python_component_id_owned_by_another_scene(self, scene):
        source = scene.create_game_object("LiveSourceSceneObject")
        source_component = _StrictSceneComponent()
        source.add_py_component(source_component)
        document = json.loads(json.dumps(scene.serialize_document()))
        manager = SceneManager.instance()
        target = manager.create_scene("PythonIdCollisionTarget")
        target.create_game_object("TargetState")
        try:
            assert deserialize_scene_document_transactionally(target, document) is True

            restored = target.find("LiveSourceSceneObject")
            restored_component = restored.get_py_component(_StrictSceneComponent)
            assert restored.id != source.id
            assert restored_component.component_id != source_component.component_id
            assert source.get_py_component(_StrictSceneComponent) is source_component
        finally:
            manager.unload_scene(target)

    def test_game_object_restore_rejects_mismatched_python_component_id(self, scene):
        root = scene.create_game_object("MismatchedPythonComponent")
        original = _StrictSceneComponent()
        root.add_py_component(original)
        original_document = root.serialize_document()
        candidate = json.loads(json.dumps(original_document))
        _python_records(candidate)[0]["data"]["__component_id__"] = 999999

        with pytest.raises(PythonComponentRestoreError, match="exact ComponentRecord|reserved"):
            deserialize_game_object_document_transactionally(root, candidate)

        assert root.serialize_document() == original_document
        assert root.get_py_component(_StrictSceneComponent) is original

    def test_idless_object_graph_remaps_internal_python_references(self, scene):
        source = scene.create_game_object("LocalReferenceRoot")
        child = scene.create_game_object("LocalReferenceChild")
        child.set_parent(source)
        child.add_py_component(_StrictSceneComponent())
        refs = _ObjectGraphRefsComponent()
        refs.target_object = GameObjectRef(child)
        refs.target_component = ComponentRef(
            go_id=child.id,
            component_type="_StrictSceneComponent",
        )
        source.add_py_component(refs)
        document = json.loads(json.dumps(source.serialize_document()))
        _strip_prefab_runtime_fields(document)

        created = instantiate_game_object_document_transactionally(scene, document)

        created_child = created.get_child(0)
        restored = created.get_py_component(_ObjectGraphRefsComponent)
        assert restored.target_object is created_child
        assert restored.target_component is created_child.get_py_component(_StrictSceneComponent)

    @pytest.mark.parametrize("collision", ["game_object_id", "component_id", "python_component_id"])
    def test_game_object_commit_rejects_ids_owned_outside_subtree(self, scene, collision):
        root = scene.create_game_object("CollisionCheckedRoot")
        rigidbody = root.add_component("Rigidbody")
        outside = scene.create_game_object("OutsideOwner")
        outside_collider = outside.add_component("BoxCollider")
        root_python = _StrictSceneComponent()
        outside_python = _StrictSceneComponent()
        root.add_py_component(root_python)
        outside.add_py_component(outside_python)
        original_document = root.serialize_document()
        candidate = json.loads(json.dumps(original_document))

        if collision == "game_object_id":
            candidate["id"] = outside.id
        elif collision == "component_id":
            candidate["components"][0]["component_id"] = outside_collider.component_id
        else:
            _python_records(candidate)[0]["component_id"] = outside_python.component_id

        if collision == "python_component_id":
            assert deserialize_game_object_document_transactionally(root, candidate) is False
        else:
            assert root._commit_document(candidate) is False
        assert root.serialize_document() == original_document
        assert scene.find("CollisionCheckedRoot") is root
        assert root.get_component("Rigidbody") is rigidbody
        assert scene.find("OutsideOwner") is outside
        assert outside.get_component("BoxCollider") is outside_collider

    def test_camera_stack_depth_and_explicit_preference_follow_enable_transitions(self, scene):
        high_owner = scene.create_game_object("HighCamera")
        high = high_owner.add_component("Camera")
        high.depth = 20.0
        low = scene.create_game_object("LowCamera").add_component("Camera")
        low.depth = -5.0
        middle_owner = scene.create_game_object("MiddleCamera")
        middle = middle_owner.add_component("Camera")
        middle.depth = 3.0
        assert scene.main_camera is None
        assert [camera.component_id for camera in scene.active_game_cameras] == [
            low.component_id,
            middle.component_id,
            high.component_id,
        ]
        assert scene.effective_game_camera.component_id == low.component_id
        high.depth = -10.0
        assert scene.effective_game_camera.component_id == high.component_id and scene.main_camera is None
        scene.main_camera = middle
        assert scene.effective_game_camera.component_id == middle.component_id
        middle.enabled = False
        assert scene.main_camera.component_id == middle.component_id
        assert scene.effective_game_camera.component_id == high.component_id
        assert [camera.component_id for camera in scene.active_game_cameras] == [high.component_id, low.component_id]
        middle.enabled = True
        middle_owner.active = False
        assert [camera.component_id for camera in scene.active_game_cameras] == [high.component_id, low.component_id]
        middle_owner.active = True
        assert scene.effective_game_camera.component_id == middle.component_id
        high.enabled = low.enabled = middle.enabled = False
        assert scene.effective_game_camera is None

    def test_main_camera_rejects_cross_scene_reference(self, scene):
        other = SceneManager.instance().create_scene("OtherCameraScene")
        owner = other.create_game_object("ForeignCamera")
        foreign = owner.add_component("Camera")

        with pytest.raises(ValueError, match="owned by this Scene"):
            scene.main_camera = foreign
