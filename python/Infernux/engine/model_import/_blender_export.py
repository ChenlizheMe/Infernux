"""Run only in the managed Blender process; do not import from the engine."""
import sys


def main():
    import bpy

    if bpy.app.version[:2] != (5, 2):
        raise RuntimeError(f"Infernux model import requires Blender 5.2, got {bpy.app.version_string}")
    source, destination = sys.argv[sys.argv.index("--") + 1:]
    # Never execute embedded project scripts or replace the importer's UI.
    bpy.ops.wm.open_mainfile(filepath=source, load_ui=False, use_scripts=False)
    result = bpy.ops.export_scene.gltf(
        filepath=destination,
        export_format="GLB",
        export_yup=True,
        export_cameras=True,
        export_lights=True,
        export_animations=True,
        export_skins=True,
        export_morph=True,
        export_apply=False,
        export_extras=False,
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"Blender glTF export did not finish: {result}")


if __name__ == "__main__":
    main()
