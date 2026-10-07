from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import urllib.error
import xml.etree.ElementTree as ET
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/release' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pe_pair(magic=0x20b, length=515):
    original = bytearray(length)
    original[:2] = b'MZ'
    struct.pack_into('<I', original, 60, 128)
    original[128:132] = b'PE\0\0'
    struct.pack_into('<H', original, 152, magic)
    original[440:445] = b'CODE!'
    security = 152 + (112 if magic == 0x20b else 96) + 32
    signed = bytearray(original)
    offset = (length + 7) & ~7
    signed += bytes(offset - length) + b'fakecert'
    struct.pack_into('<I', signed, 216, 12345)
    struct.pack_into('<II', signed, security, offset, 8)
    return bytes(original), bytes(signed), security


@pytest.mark.parametrize('magic', [0x10b, 0x20b])
@pytest.mark.parametrize('length', [512, 515])
def test_signing_preserves_executable_except_certificate_and_checksum(magic, length):
    before, after, _ = pe_pair(magic, length)
    load('verify_signed_payload').verify(before, after)


@pytest.mark.parametrize('change', ['code', 'padding', 'trailing', 'unsigned', 'dos', 'header'])
def test_reject_modified_or_unsigned_signing_output(change):
    before, after, _ = pe_pair()
    modified = bytearray(after)
    if change == 'code':
        modified[442] ^= 1
    elif change == 'padding':
        modified[515] = 1
    elif change == 'trailing':
        modified += b'garbage'
    elif change == 'unsigned':
        modified = before
    elif change == 'dos':
        modified[0] = 0
    else:
        modified[152] = 0
    with pytest.raises(ValueError):
        load('verify_signed_payload').verify(before, modified)


def test_hub_installer_payload_and_update_are_identical(tmp_path):
    module = load('verify_hub_payloads')
    (tmp_path / 'Infernux Hub.exe').write_bytes(b'signed hub')
    archives = [tmp_path / 'update.zip', tmp_path / 'installer.zip']
    for path in archives:
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('Infernux Hub.exe', b'signed hub')
    module.verify(tmp_path, *archives)
    with zipfile.ZipFile(archives[1], 'w') as archive:
        archive.writestr('Infernux Hub.exe', b'unsigned hub')
    with pytest.raises(ValueError, match='differs'):
        module.verify(tmp_path, *archives)


@pytest.mark.parametrize('kind,name', [('hub', 'Infernux Hub.exe'), ('installer', 'InfernuxHubInstaller-${hubVersion}-windows-x64.exe')])
def test_signpath_rules_only_sign_the_named_infernux_binary(kind, name):
    ns = {'s': 'http://signpath.io/artifact-configuration/v1'}
    root = ET.parse(ROOT / 'scripts/release/signpath' / f'{kind}.xml').getroot()
    binary = root.find('s:zip-file/s:pe-file', ns)
    assert binary.attrib == {'path': name, 'product-name': 'Infernux', 'product-version': '${version}'}
    assert len(root.findall('.//s:authenticode-sign', ns)) == 1
    assert {p.attrib['name'] for p in root.findall('s:parameters/s:parameter', ns)} == {'version', 'hubVersion'}


def workflow(path):
    return yaml.safe_load((ROOT / path).read_text(encoding='utf-8'))


def test_sign_inside_out_and_never_rebuild_the_signed_hub():
    steps = workflow('.github/actions/build-windows-hub/action.yml')['runs']['steps']
    names = [s.get('name', '') for s in steps]
    assert names.index('Build unsigned Hub and private Python runtime') < names.index('Sign Hub before embedding it')
    assert names.index('Sign Hub before embedding it') < names.index('Package the same Hub in both distribution formats') < names.index('Sign the completed installer')
    package = next(s['run'] for s in steps if s.get('name') == 'Package the same Hub in both distribution formats')
    assert '--target installer' in package
    assert 'cmake --build' not in package
    sign = workflow('.github/actions/sign-windows-file/action.yml')['runs']['steps']
    verify = sign[-1]['run']
    assert verify.index('verify_signed_payload.py') < verify.index('Copy-Item')
    assert verify.index('verify_windows_signature.ps1') < verify.index('Copy-Item')


def test_test_signatures_cannot_enter_release_publication():
    test = workflow('.github/workflows/test-windows-signing.yml')
    assert test['permissions'] == {'contents': 'read', 'actions': 'read'}
    text = (ROOT / '.github/workflows/test-windows-signing.yml').read_text()
    assert 'test-signing' in text and 'NOT-FOR-RELEASE' in text
    assert 'publish-desktop-release.yml' not in text
    publish = workflow('.github/workflows/publish-desktop-release.yml')
    assert publish['jobs']['publish']['needs'] == 'verify'
    validation = str(publish['jobs']['verify'])
    assert 'SIGNPATH_RELEASE_CERTIFICATE_THUMBPRINT' in validation
    assert "source.conclusion -ne 'success'" in publish['jobs']['verify']['steps'][0]['run']
    assert 'test-windows-signing.yml' not in validation
    assert 'github-action-submit-signing-request' not in str(publish)
    assert "*-desktop-distribution-${{ inputs.run_id }}-${{ needs.verify.outputs.attempt }}" in str(publish)


def test_test_certificate_trust_is_noninteractive_and_disposable_only():
    source = (ROOT / 'scripts/release/verify_windows_signature.ps1').read_text()
    store = "::new('Root', 'LocalMachine')"
    assert store in source
    assert "::new('Root', 'CurrentUser')" not in source
    assert source.index("$env:RUNNER_ENVIRONMENT -ne 'github-hosted'") < source.index(store)
    assert source.index('WindowsBuiltInRole]::Administrator') < source.index(store)
    assert '$store.Add($signerCertificate)' in source
    assert 'if ($added) { $store.Remove($signerCertificate) }' in source


@pytest.mark.parametrize('test_certificate', [False, True])
def test_signature_verifier_keeps_the_certificate_separate_from_its_switch(tmp_path, test_certificate):
    shell = shutil.which('pwsh')
    if shell is None:
        pytest.skip('PowerShell 7 is required to execute the signature verifier')
    probe = tmp_path / 'signature-probe.ps1'
    probe.write_text(r'''
$ErrorActionPreference = 'Stop'
$rsa = [System.Security.Cryptography.RSA]::Create(2048)
$request = [System.Security.Cryptography.X509Certificates.CertificateRequest]::new(
    'CN=Infernux regression only', $rsa,
    [System.Security.Cryptography.HashAlgorithmName]::SHA256,
    [System.Security.Cryptography.RSASignaturePadding]::Pkcs1)
$certificate = $request.CreateSelfSigned([DateTimeOffset]::UtcNow.AddDays(-1), [DateTimeOffset]::UtcNow.AddDays(1))
try {
    # Exercise the actual script and PowerShell's typed parameter semantics.
    # No file is signed and no machine/user certificate store is modified.
    function Get-Item {
        param([string]$LiteralPath)
        [pscustomobject]@{
            Name = 'probe.exe'; FullName = $LiteralPath
            VersionInfo = [pscustomobject]@{ProductName = 'Infernux'; ProductVersion = '0.4.1.3'}
        }
    }
    function Get-AuthenticodeSignature {
        param([string]$LiteralPath)
        [pscustomobject]@{
            SignerCertificate = $certificate; Status = 'Valid'
            TimeStamperCertificate = $certificate
        }
    }
    $useTestCertificate = $env:INFERNUX_TEST_CERTIFICATE -eq 'true'
    & $env:INFERNUX_SIGNATURE_SCRIPT -Path 'probe.exe' -Thumbprint $certificate.Thumbprint `
        -ProductVersion '0.4.1.3' -TestCertificate:$useTestCertificate
} finally {
    $certificate.Dispose()
    $rsa.Dispose()
}
''', encoding='utf-8')
    result = subprocess.run(
        [shell, '-NoProfile', '-NonInteractive', '-File', str(probe)],
        env={
            **os.environ,
            'GITHUB_ACTIONS': 'false',
            'RUNNER_ENVIRONMENT': 'self-hosted',
            'INFERNUX_SIGNATURE_SCRIPT': str(ROOT / 'scripts/release/verify_windows_signature.ps1'),
            'INFERNUX_TEST_CERTIFICATE': str(test_certificate).lower(),
        },
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
    )
    if test_certificate:
        assert result.returncode != 0
        assert 'Test certificate trust is restricted to disposable' in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'Verified probe.exe' in result.stdout


def test_github_digest_must_match_exact_final_files(tmp_path):
    module = load('verify_publication')
    (tmp_path / 'Hub.exe').write_bytes(b'signed')
    item = {'name': 'Hub.exe', 'size': 6, 'digest': 'sha256:' + hashlib.sha256(b'signed').hexdigest()}
    module.verify_github(tmp_path, {'assets': [item]})
    item['digest'] = 'sha256:' + '0' * 64
    with pytest.raises(ValueError, match='differs'):
        module.verify_github(tmp_path, {'assets': [item]})
    with pytest.raises(ValueError, match='inventory'):
        module.verify_github(tmp_path, {'assets': []})


def test_pypi_existing_wheel_mismatch_is_not_skip_existing(tmp_path, monkeypatch):
    module = load('verify_publication')
    for name in ('windows.whl', 'linux.whl'):
        (tmp_path / name).write_bytes(b'wheel')
    files = [{'filename': n, 'size': 5, 'digests': {'sha256': hashlib.sha256(b'wheel').hexdigest()}} for n in ('windows.whl', 'linux.whl')]
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps({'urls': files}).encode()))
    module.verify_pypi(tmp_path, '1.2.3')
    files[0]['digests']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='differs'):
        module.verify_pypi(tmp_path, '1.2.3', allow_missing=True)
    files.clear()
    module.verify_pypi(tmp_path, '1.2.3', allow_missing=True)
    with pytest.raises(ValueError, match='missing'):
        module.verify_pypi(tmp_path, '1.2.3')


def test_r2_recovery_skips_identical_but_refuses_changed_public_objects(tmp_path, monkeypatch):
    module = load('publish_hub_objects')
    source = tmp_path / 'hub.zip'
    source.write_bytes(b'original')
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'original'))
    publisher = module.Publisher('secret')
    monkeypatch.setattr(publisher, '_request', lambda *args: pytest.fail('must not overwrite existing object'))
    publisher.upload(source, 'hub/1.2.3/build-1/hub.zip')
    source.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='differs'):
        publisher.upload(source, 'hub/1.2.3/build-1/hub.zip')


def test_r2_checks_the_download_after_upload(tmp_path, monkeypatch):
    module = load('publish_hub_objects')
    source = tmp_path / 'hub.zip'
    source.write_bytes(b'original')
    publisher = module.Publisher('secret')
    checks = []
    def verify(key, expected, *, allow_missing):
        checks.append(allow_missing)
        assert expected == hashlib.sha256(b'original').hexdigest()
        return not allow_missing
    monkeypatch.setattr(publisher, '_verify_public_object', verify)
    monkeypatch.setattr(publisher, '_request', lambda method, path, body: {'uploadId': 'id'} if path == '/start' else {'size': 8} if path == '/complete' else {'partNumber': 1, 'etag': 'tag'})
    publisher.upload(source, 'key')
    assert checks == [True, False]


def test_r2_validates_all_files_before_any_upload(tmp_path, monkeypatch):
    module = load('publish_hub_objects')
    (tmp_path / module.release_assets('1.2.3')[0]).write_bytes(b'file')
    monkeypatch.setattr(module, 'Publisher', lambda token: pytest.fail('all files must be present first'))
    with pytest.raises(FileNotFoundError):
        module.publish(tmp_path, '1.2.3', 1, 'secret')
