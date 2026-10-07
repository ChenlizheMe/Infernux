"""Project declarations resolve by exact Name across Assets and Packages."""
from pathlib import Path

import pytest

from infernux.core.assets import AssetManager
from infernux.core.shader import Shader
from test_shader_dependency_publication import dependency_sources


@pytest.mark.parametrize('root_name', ('Assets', 'Packages'))
def test_nested_library_is_discovered_outside_the_consumer_directory(
        engine, dependency_sources, root_name):
    prefix, create, registry = dependency_sources
    directory = Path(engine.get_asset_database().project_root) / root_name / prefix / 'nested'
    directory.mkdir(parents=True)
    library = directory / 'shared.glsl'
    library_name = prefix + ' Shared'
    library.write_text(f'ShaderInfo {{ Name "{library_name}" }}\n'
                       'vec4 discoveredColor(){return vec4(0.4,0.3,0.2,1);}\n', encoding='utf-8')
    try:
        root, guid = create('Consumer', '.frag',
                            f'#version 450\nShaderInfo {{ Name "{prefix} Consumer" '
                            f'Capabilities [Fullscreen] Imports ["{library_name}"] '
                            'Outputs { Float4 outColor } }\n'
                            'void main(){outColor=discoveredColor();}\n')
        assert Shader.reload(str(root))
        assert registry.get_asset_version(guid) > 0
        assert engine.is_shader_loaded(prefix + ' Consumer', 'fragment')
    finally:
        library.unlink()
        directory.rmdir()
        directory.parent.rmdir()


def test_project_library_overrides_the_matching_builtin_declaration(engine, dependency_sources):
    import infernux

    prefix, create, registry = dependency_sources
    original = (Path(infernux.__file__).parent / 'resources/shaders/math.glsl').read_text(encoding='utf-8')
    function = prefix + '_project_only'
    override, _ = create('MathOverride', '.glsl', original + f'\nvec4 {function}(){{return vec4(1);}}\n')
    root_source = (f'#version 450\nShaderInfo {{ Name "{prefix} Consumer" '
                   'Capabilities [Fullscreen] Imports ["Math"] Outputs { Float4 outColor } }\n'
                   f'void main(){{outColor={function}();}}\n')
    root, guid = create('Consumer', '.frag', root_source)
    assert Shader.reload(str(root))
    assert registry.get_asset_version(guid) > 0
    # After removing only the authored extra function, resolving the same
    # built-in Name must compile its ordinary library body without a duplicate.
    override.write_text(original, encoding='utf-8')
    rejected = AssetManager.reimport_asset(str(override))
    assert not rejected and function in rejected.error
    repaired = root_source.replace(function + '()', 'vec4(1)')
    root.write_text(repaired, encoding='utf-8')
    accepted = AssetManager.reimport_asset(str(root))
    assert accepted, accepted.error


@pytest.mark.parametrize('rejection', ('case-mismatch', 'template-directory', 'duplicate-library', 'duplicate-model'))
def test_invalid_declaration_discovery_rejects_before_publication_and_recovers(
        engine, dependency_sources, rejection):
    prefix, create, registry = dependency_sources
    name = prefix + ' Shared'
    root_source = (f'#version 450\nShaderInfo {{ Name "{prefix} Consumer" '
                   'Capabilities [Fullscreen] Outputs { Float4 outColor } }\n'
                   'void main(){outColor=vec4(1);}\n')
    root, guid = create('Consumer', '.frag', root_source)
    assert Shader.reload(str(root))
    version = registry.get_asset_version(guid)
    directory = Path(engine.get_asset_database().project_root) / 'Packages' / prefix
    directory.mkdir(parents=True)
    written = []
    try:
        if rejection == 'template-directory':
            directory = directory / '_templates'
            directory.mkdir()
        extension = '.shadingmodel' if rejection == 'duplicate-model' else '.glsl'
        declaration = (f'ShadingModelInfo {{ Name "{name}" }}\n'
                       'void shading(in SurfaceData s,out vec4 color){color=vec4(s.albedo,1);}\n'
                       if rejection == 'duplicate-model' else
                       f'ShaderInfo {{ Name "{name}" }}\n'
                       'float found(){return 1.0;}\n')
        first = directory / ('first' + extension)
        first.write_text(declaration, encoding='utf-8')
        written.append(first)
        if rejection.startswith('duplicate'):
            second = root.parent / (prefix + 'duplicate' + extension)
            second.write_text(declaration, encoding='utf-8')
            written.append(second)
        imported_name = name.lower() if rejection == 'case-mismatch' else name
        changed = (root_source if rejection == 'duplicate-model' else
                   root_source.replace('Capabilities [Fullscreen]',
                                       f'Capabilities [Fullscreen] Imports ["{imported_name}"]'))
        root.write_text(changed, encoding='utf-8')
        rejected = AssetManager.reimport_asset(str(root))
        assert not rejected and rejected.database_committed
        assert registry.get_asset_version(guid) == version
        if rejection.startswith('duplicate'):
            assert 'Duplicate' in rejected.error and name in rejected.error, rejected.error
            assert all(path.as_posix() in rejected.error.replace('\\', '/') for path in written), rejected.error
        else:
            assert 'shader import not found: ' + imported_name in rejected.error, rejected.error
        for path in written:
            path.unlink()
        written.clear()
        root.write_text(root_source, encoding='utf-8')
        accepted = AssetManager.reimport_asset(str(root))
        assert accepted, accepted.error
        assert registry.get_asset_version(guid) == version + 1
        assert engine.is_shader_loaded(prefix + ' Consumer', 'fragment')
    finally:
        for path in written:
            path.unlink()
        directory.rmdir()
        if rejection == 'template-directory':
            directory.parent.rmdir()
