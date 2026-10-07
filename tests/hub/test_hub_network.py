from types import SimpleNamespace

import pytest
import hub_network


@pytest.fixture
def linux_without_build_machine_certificates(monkeypatch):
    monkeypatch.setattr(hub_network.sys, "platform", "linux")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setattr(hub_network.ssl, "get_default_verify_paths", lambda: SimpleNamespace(cafile=None, capath=None))


def test_portable_linux_hub_uses_system_certificates(monkeypatch, linux_without_build_machine_certificates):
    bundle = "/etc/ssl/certs/ca-certificates.crt"
    monkeypatch.setattr(hub_network.Path, "is_file", lambda path: path.as_posix() == bundle)
    hub_network.configure_system_certificates()
    assert hub_network.Path(hub_network.os.environ["SSL_CERT_FILE"]).as_posix() == bundle


@pytest.mark.parametrize("setting", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_explicit_trust_configuration_is_preserved(monkeypatch, linux_without_build_machine_certificates, setting):
    monkeypatch.setenv(setting, "/company/trust")
    hub_network.configure_system_certificates()
    assert hub_network.os.environ[setting] == "/company/trust"


def test_missing_trust_store_remains_an_error(monkeypatch, caplog, linux_without_build_machine_certificates):
    monkeypatch.setattr(hub_network.Path, "is_file", lambda path: False)
    hub_network.configure_system_certificates()
    assert "SSL_CERT_FILE" not in hub_network.os.environ
    assert "ca-certificates" in caplog.text
