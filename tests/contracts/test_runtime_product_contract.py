"""Build/Player contract checks that work before a native engine is built."""

import itertools
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "python"
FEATURES = list(itertools.product([(False, False), (True, False), (True, True)], [False, True]))


def _document(contract, flavor, features):
    return {
        "$schema": contract.PLAYER_MANIFEST_SCHEMA,
        "product": {"flavor": flavor.value},
        "features": features.to_manifest(),
        "runtime_policy": contract.runtime_policy_for(flavor).to_manifest(),
        "services": contract.player_manifest_service_section(flavor, features),
    }


@pytest.mark.parametrize("flavor_name", ["EDITOR_DEVELOPMENT", "PLAYER_DEBUG", "PLAYER_RELEASE"])
@pytest.mark.parametrize("compute,splash", FEATURES)
def test_product_graph_and_manifest_contract(service_contract, flavor_name, compute, splash):
    c = service_contract
    flavor = c.RuntimeFlavor[flavor_name]
    jit, parallel = compute
    features = c.RuntimeFeatureSet(jit=jit, parallel=parallel, optional_subsystems=("splash",) if splash else ())
    graph = c.runtime_service_graph_for(flavor, features)

    seen = set()
    for service in graph.services:
        assert service.service_id not in seen
        assert set(service.dependencies) <= seen
        seen.add(service.service_id)
        # Detect stale registrations after removing or moving a source module.
        if service.service_id == "parallel_module":
            # Built Numba/LLVM payload, not a Python source file.
            assert service.module == "Modules/Parallel.inxmod"
        else:
            assert service.module.endswith(".pyc")
            assert (SOURCE / service.module[:-1]).is_file(), service.module

    assert graph.contains("player_runtime_session") == flavor.is_player
    for editor_service in ("editor_resources", "editor_selection", "editor_undo"):
        assert graph.contains(editor_service) == (not flavor.is_player)
    if flavor.is_player:
        assert not any(service.authoring for service in graph.services)
        assert graph.contains("player_control_debug") == (flavor == c.RuntimeFlavor.PLAYER_DEBUG)
        assert graph.contains("jit_runtime_support") == jit
        assert graph.contains("parallel_module") == parallel
        assert graph.contains("splash_player") == splash
        manifest = c.RuntimeProductManifest.from_document(_document(c, flavor, features))
        assert manifest.flavor == flavor
        for service in graph.services:
            assert manifest.require_service(service.service_id) == service


@pytest.mark.parametrize("features,error,message", [
    ({"parallel": True}, ValueError, "requires the JIT"),
    ({"optional_subsystems": ("editor_preview",)}, ValueError, "unknown runtime subsystems"),
    ({"jit": 1}, TypeError, "must be booleans"),
    ({"parallel": "false"}, TypeError, "must be booleans"),
])
def test_invalid_features_are_rejected(service_contract, features, error, message):
    with pytest.raises(error, match=message):
        service_contract.RuntimeFeatureSet(**features)


@pytest.mark.parametrize("drift,message", [
    ("module", "authoritative runtime product graph"),
    ("declared", "declared-service"),
    ("policy", "runtime policy"),
])
def test_manifest_rejects_contract_drift(service_contract, drift, message):
    c = service_contract
    document = _document(c, c.RuntimeFlavor.PLAYER_RELEASE, c.RuntimeFeatureSet())
    if drift == "module":
        document["services"]["graph"][0]["module"] = "infernux/engine/undo/_manager.pyc"
    elif drift == "declared":
        document["services"]["declared"].append("player_control_debug")
    else:
        document["runtime_policy"]["profiling"] = "available"
    with pytest.raises(RuntimeError, match=message):
        c.RuntimeProductManifest.from_document(document)
