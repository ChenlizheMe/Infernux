# Writing Editor tools

`inx.editor` is the public entry point for project-specific Editor tools.
Its operations use the same Hierarchy, Prefab, save-ticket and Undo services as
the Editor UI. They require an active Editor session and are unavailable in Play
Mode. They are not gameplay APIs and do not belong in a Player script.

Place this script in `Assets/Editor/CreateObstacle.py`:

```python
import infernux as inx


class ObstacleTools(inx.InxPreload):
    def preload(self, context):
        inx.editor.EditorCommandRegistry.instance().register(
            inx.editor.EditorCommand(
                "my_game.create_obstacle",
                self.create_obstacle,
                display_name="Create obstacle",
            )
        )
        inx.editor.ShortcutRouter.instance().register(
            inx.editor.ShortcutBinding(
                "my_game.create_obstacle",
                inx.editor.KeyChord("F9"),
                binding_id="my_game.create_obstacle.f9",
            )
        )

    def create_obstacle(self, context):
        with inx.editor.edit_scene("Create obstacle"):
            root = inx.editor.create_game_object("Obstacle")
            inx.editor.create_game_object(
                "Body", kind="primitive.cube", parent=root,
            )
            prefab_path = inx.editor.create_prefab(root, "Assets")
            placed = inx.editor.instantiate_prefab(prefab_path)
        return placed.id
```

Press F9 or find **Create obstacle** in the command palette. One Undo removes the
created hierarchy, asset and placed instance; Redo restores the recorded action.
Register commands and shortcuts inside `preload`: their owner is then removed on
reload, unload or project close. Do not register them each frame.

## Creation and editing

- `create_game_object` returns the created GameObject. `kind` uses the Hierarchy
  creation catalog, for example `empty`, `primitive.sphere` or `ui.button`.
- Its optional `configure(obj)` callback runs before the creation snapshot is
  recorded. Use it for initial components or Transform values that must survive
  Undo/Redo. An exception cancels that creation.
- `edit_scene(description)` groups **journal-aware operations**. It does not
  intercept arbitrary Python assignments or roll back an entire script on error.
- `create_prefab(obj, directory)` creates a uniquely named asset and links its
  source hierarchy. The second argument is a **directory**, not an output filename;
  it does not overwrite an existing Prefab.
- `instantiate_prefab(path, parent=...)` returns the new GameObject.
- `apply_prefab(obj)` publishes instance overrides to the linked asset;
  `revert_prefab(obj)` restores source-owned values. Both use the global journal.

## Scene documents and history

`save_scene("Assets/Scenes/MyLevel.scene")` saves the active scene to an explicit
project-relative path. `save_scene()` saves its current document, or requests a
Save As dialog if it has no path. The result is a `DocumentActionResult`:

- `APPLIED` / `NO_OP`: the operation completed / nothing needed saving.
- `PENDING`: a dialog or asynchronous operation remains outstanding.
- `REJECTED` / `FAILED`: inspect `message`; do not report success.

`open_scene(path)` and `new_scene()` retain the normal unsaved-changes confirmation
and deferred scene-switch rules. Returning from a request is **not** proof that
the new scene is loaded. Do not create objects for the destination immediately
after requesting a scene switch.

`undo()` and `redo()` defer replay to the Editor safe point by default. If a
command invokes them, register that command with `creates_user_action=False`:
replaying history is not a new edit. `defer=False` is for an already safe,
caller-controlled non-rendering host, not a GUI callback.

Prefab components retain source identities independently of their runtime IDs.
Deleting or reordering same-type components does not retarget the surviving
components during Apply/Revert, Undo/Redo, or scene synchronization. Newly added
components remain private to an instance until applied. Deleted source IDs are
not reused by later additions.

Old scenes without component source identities are adopted once using their
saved baseline and type order. This cannot recover the identity of a same-type
component that was already deleted before the scene acquired that metadata.

Component fields store the exact component ID as well as its owning object and
type. Assign a component instance directly; the Inspector picker lists each
component separately. Dropping a GameObject binds its first matching component.
Deleting that target resolves the reference to `None`, not to another component
of the same type. Clear missing references before saving a Prefab asset.
Legacy type-only references remain readable and acquire an exact ID when saved.
Copies, Prefab operations and scene loading remap internal references to the new
identities; explicit references to objects outside the copied graph stay external.

Reparenting a node inside a Prefab preserves its identity. Source hierarchy edits
can merge with instance field edits, and an instance parent override survives
unrelated source updates. If concurrent edits form a parent cycle, Apply rejects
the conflict before changing the asset file; resolve the hierarchy and apply again.

## Nested Prefabs

Place a Prefab instance under another hierarchy and save the outer hierarchy as
a Prefab. The inner instance retains its own asset link. Repeated instances of
the same inner asset have separate outer identities; object and component
references continue to target their respective instances.

- Applying an inner instance updates its inner source. Applying the outer root
  records those changes in the outer asset instead.
- Outer Apply/Revert, Undo/Redo and Prefab Mode saves preserve inner links.
  Copying an inner instance creates a new outer member rather than reusing the
  original member's identity.
- Unpacking the outer instance preserves inner Prefab links. It does not unpack
  every nesting level recursively.
- Scene reopening, fresh instantiation and build resolution read the current
  inner sources even when the outer asset file has not changed. Cyclic nesting
  is rejected rather than expanded indefinitely.

These APIs are not the complete Unity Editor SDK. Moving nested roots between
different outer instances and migrating the project's complete authoring tools
remain under validation; do not treat those workflows as stable yet.
