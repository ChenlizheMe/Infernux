"""A discoverable module is insufficient when a new major removes its API."""
from pathlib import Path
import shutil
import subprocess
import sys

import packaging
from packaging.requirements import Requirement
import pytest

from runtime_requirements import runtime_packages, runtime_probe_code


def test_managed_mcp_requirements_match_the_default_plugin():
    root = Path(__file__).resolve().parents[2]
    declared = [Requirement(line) for line in
                (root/'external/plugins/infernux_mcp/package/requirements.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
    managed = {req.name: req for req in map(Requirement, runtime_packages())}
    for requirement in declared:
        assert managed[requirement.name].specifier == requirement.specifier


@pytest.mark.parametrize('mcp_version,client_version,ready', [
    ('1.24.0','3.0.0',True),
    ('1.28.0','3.5.0',True),
    ('2.0.0','3.5.0',False),
    ('1.28.0','4.1.0',False),
    ('1.23.0','3.0.0',False),
    ('1.28.0','2.14.0',False),
])
def test_runtime_rejects_importable_but_incompatible_sdk(tmp_path,mcp_version,client_version,ready):
    # Run against real distribution metadata in an isolated Python process.
    # SDK v2 leaves an importable top-level mcp module, despite removing FastMCP.
    shutil.copytree(Path(packaging.__file__).parent,tmp_path/'packaging')
    for name,version in (('mcp',mcp_version),('fastmcp',client_version)):
        (tmp_path/f'{name}.py').write_text('')
        metadata = tmp_path/f'{name}-{version}.dist-info'
        metadata.mkdir()
        (metadata/'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
    code = f'import sys; sys.path.insert(0,{str(tmp_path)!r})\n'+runtime_probe_code(('mcp','fastmcp'))
    result = subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=20,check=True)
    assert result.stdout.strip() == str(int(ready))
