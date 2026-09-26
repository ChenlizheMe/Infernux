from __future__ import annotations

from scripts.acceptance.audio_report_gate import validate_audio_report


def _report(audio: dict) -> dict:
    return {
        "schema": "infernux.windows_player_smoke",
        "status": "passed",
        "result": {"audio": audio},
    }


def test_dummy_audio_is_never_hardware_evidence() -> None:
    errors = validate_audio_report(
        _report(
            {
                "verification": "simulated",
                "hardware_verified": False,
                "driver": "dummy",
            }
        )
    )
    assert any("simulated" in error for error in errors)
    assert any("hardware" in error for error in errors)


def test_missing_audio_evidence_fails_release_gate() -> None:
    errors = validate_audio_report(
        {"schema": "infernux.linux_player_smoke", "status": "passed"}
    )
    assert errors == [
        "Player report has no audio evidence; software or dummy audio tests "
        "cannot satisfy the hardware gate"
    ]


def test_complete_hardware_probe_passes() -> None:
    errors = validate_audio_report(
        _report(
            {
                "verification": "hardware",
                "hardware_verified": True,
                "driver": "wasapi",
                "device_name": "Headphones (USB DAC)",
                "sample_rate": 48000,
                "channels": 2,
                "captured_at": "2026-09-26T02:00:00Z",
            }
        )
    )
    assert errors == []


def test_diagnostics_mode_still_rejects_simulated_driver() -> None:
    errors = validate_audio_report(
        _report({"verification": "unverified", "driver": "dummy"}),
        require_hardware=False,
    )
    assert errors == ["SDL audio driver 'dummy' is simulated"]
