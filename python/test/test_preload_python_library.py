"""Bundled compiler libraries have a preload lifetime, not component reload."""
import json
from Infernux.engine.project_context import (
    register_preload_python_library, release_preload_python_libraries,
    is_project_component_script,
)


def test_library_boundary_does_not_hide_author_scripts_or_other_projects(tmp_path):
    project=tmp_path/'project'
    package=project/'Packages/example/compiler'
    vendor=package/'runtime/vendor'
    vendor.mkdir(parents=True)
    (package/'inx_package.json').write_text(json.dumps({'reference':'example/compiler'}))
    library=vendor/'implementation.py'
    author=package/'runtime/Component.py'
    library.write_text('')
    author.write_text('')
    assert is_project_component_script(str(library),str(project))
    try:
        register_preload_python_library('owner',str(vendor))
        assert not is_project_component_script(str(library),str(project))
        assert is_project_component_script(str(author),str(project))
        # Meta/asset inclusion is unaffected; only component-script discovery changes.
        from Infernux.engine.project_context import package_script_reference
        assert package_script_reference(str(library),str(project))=='example/compiler'
    finally:
        release_preload_python_libraries('owner')
    assert is_project_component_script(str(library),str(project))
