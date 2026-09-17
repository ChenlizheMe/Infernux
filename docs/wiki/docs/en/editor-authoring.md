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

## Data assets and build scenes

Declare runtime configuration with `DataAsset` and `serialized_field` in
`Assets/Scripts/LevelConfig.py`. Editor tools import it as
`from Scripts.LevelConfig import LevelConfig`: `Assets` is the project module
root, not a Python package name.

```python
from Scripts.LevelConfig import LevelConfig

with inx.editor.edit_scene("Create level configuration"):
    inx.editor.create_folder("Assets/NewLevel")
    path = inx.editor.create_data_asset(
        LevelConfig(title="Garden"), "Assets/NewLevel/Level.inxdata",
    )
config = inx.DataAsset.load(path)
```

`create_data_asset` saves an independent copy through the asset system, which
owns its GUID and import results. The input object's persistent identity is
unchanged. The parent directory must exist and the target must not exist.
Undo removes the asset; Redo restores its original GUID and fields. Use the
Inspector document workflow to edit existing assets rather than overwriting them.

Authoring scripts can use that same document without opening an Inspector:

```python
level = inx.editor.load_data_asset("Assets/Data/Level01.inxdata")
with inx.editor.edit_scene("Update level configuration"):
    inx.editor.set_data_asset_fields(level, title="First level", difficulty=3)
result = inx.editor.save_data_asset(level)
```

`load_data_asset` reads the live Editor document, including unsaved changes.
Do not use the disk-reading `DataAsset.load` to observe an Undo immediately.
`set_data_asset_fields` validates every declared field on an independent copy
before recording one undoable edit. Unknown or read-only fields reject the whole
call; unchanged values return `False`. A group can include several assets.
`save_data_asset` uses the existing save transaction; inspect its
`APPLIED/PENDING/FAILED` result rather than assuming that a request has finished.

Declare an asset-reference list with
`inx.list_field(element_type=inx.FieldType.ASSET, asset_type="DataAsset")`.
The list stores references, not embedded copies of each asset.

To attach a component in an authoring tool, use
`inx.editor.add_component(obj, "MyComponent", configure=initialize)`.
It resolves the published script and asset identity through the same service as
Inspector, and records initialized fields for Undo/Redo. Constructing an instance
from an old imported Python class and calling `add_py_component` directly bypasses
that authoring service.

The regular `obj.add_component(MyComponent)` and `obj.get_component(MyComponent)`
also resolve retained class handles to the currently published script type. This
includes a class imported by a preload before its first asset-GUID publication;
matching uses the owning module and qualified name, not a short name shared by
unrelated scripts. Directly constructing `MyComponent()` and passing that instance
to `add_py_component` is a lower-level path and does not perform this resolution.
For undoable edits on existing objects, keep using `inx.editor.add_component`.

To load an existing material, resolve its project path with
`inx.Application.asset_path("Assets/Materials/Example.mat")` before passing it to
`inx.AssetManager.load`. Tools should not depend on the Editor process's working
directory or hard-code asset GUIDs.

`get_build_scenes()` returns a detached ordered list of project-relative paths.
`set_build_scenes(paths)` edits the shared Build Settings document as one Undo
operation. Entries must be existing `.scene` files beneath `Assets`; an empty
list clears the selection, and an unchanged list creates no history. Normal
Editor autosave applies. `save_project_settings()` also provides explicit saving
with the same status contract as scene saving: `PENDING` is not a completed write.

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

## Revert one property override

```python
inx.editor.revert_property_override(component, "amount")
```

Restores one declared serialized field from the nearest Prefab source and records
one undoable edit. Other field overrides, object identities, and component
references stay intact. The source asset is not modified. A `False` result means
there was no change to record.

Use the public Python field name, including native properties such as a
collider's `size` or Transform's `local_position`. Explicitly reverting a root
Transform field restores its source value; whole-instance Revert instead
preserves root placement. Collection fields are restored as a whole. Nested
property paths, added instance components, and read-only fields are not supported
by this entry point. If a source reference targets a removed instance member,
restore that member first; the reference will not bind an unrelated object.

## Query property overrides

```python
if inx.editor.is_property_override(component, "amount"):
    inx.editor.revert_property_override(component, "amount")

for change in inx.editor.get_property_modifications(instance):
    print(change.object_id, change.component_id, change.property_path,
          change.source_value, change.instance_value)
```

Queries use the same source identities and typed reference comparison as Prefab
Apply/Revert. Same-named objects and same-type components remain distinct.
`is_property_override` takes a public Python field name, including read-only
serialized fields. An undeclared field is an error; an ordinary scene object or
an added component has no source field override.

`get_property_modifications` returns a detached tuple for the nearest containing
Prefab and its descendants. Records contain scene object/component IDs and the
corresponding asset-local `source_object_id`/`source_component_id`. Component
fields use `data.<serialized_name>`; Transform fields use `position`, `rotation`,
and `scale`. Values follow the serialized value codec, including typed references:
source references use asset-local IDs, instance references use scene IDs.
Collections are compared as whole fields, not separate array elements.

Root placement/organization differences are included with `is_default_override`
set, reflecting the existing whole-instance Apply/Revert exclusions. Added or
removed objects/components and ordering changes are structural overrides, not
property records. These are current differences against the source, not a log of
past edits or unused historical Unity overrides. Queries do not save, add Undo,
or advance the merge baseline; modifying returned values does not modify assets.

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
- Moving an inner instance to a different outer instance makes it a new member
  of that outer instance while preserving its inner asset link. Undo restores
  its previous ownership; Redo applies the move again. Moving within the same
  outer instance preserves its existing member identity.
- Moving an ordinary source-owned child out of its instance removes that source
  membership. Use sibling ordering to reorder children, not detach/reattach;
  use Undo to restore a detached child's original membership.
- Scene reopening, fresh instantiation and build resolution read the current
  inner sources even when the outer asset file has not changed. Cyclic nesting
  is rejected rather than expanded indefinitely.

These APIs are not the complete Unity Editor SDK. Migrating the project's
complete authoring tools remains a separate acceptance task.
