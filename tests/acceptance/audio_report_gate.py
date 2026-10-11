"""Validate the audio evidence contract used by Player release acceptance.

The native audio tests run with SDL's dummy backend in hosted CI. They prove
the mixer and device-clock semantics, but do not prove that a physical output
device rendered audio. This gate keeps that distinction explicit when an
acceptance report is promoted to release evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_SIMULATED_DRIVERS = frozenset({"dummy", "disk", "null", "off", "disabled"})
_REQUIRED_HARDWARE_FIELDS = ("driver", "device_name", "sample_rate", "channels")


def _audio_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return audio evidence from either Player report layout."""

    candidates: list[dict[str, Any]] = []
    for container in (payload, payload.get("result")):
        if not isinstance(container, dict):
            continue
        for key in ("audio", "audio_evidence"):
            value = container.get(key)
            if isinstance(value, dict):
                candidates.append(value)
    return candidates


def validate_audio_report(
    payload: dict[str, Any], *, require_hardware: bool = True
) -> list[str]:
    """Return actionable contract violations for one acceptance report."""

    errors: list[str] = []
    if payload.get("status") != "passed":
        errors.append("Player report status is not 'passed'")

    candidates = _audio_candidates(payload)
    if not candidates:
        errors.append(
            "Player report has no audio evidence; software or dummy audio tests "
            "cannot satisfy the hardware gate"
        )
        return errors

    evidence = candidates[0]
    verification = str(evidence.get("verification", "")).strip().casefold()
    driver = str(evidence.get("driver", "")).strip().casefold()
    hardware_verified = evidence.get("hardware_verified") is True

    if driver in _SIMULATED_DRIVERS:
        errors.append(f"SDL audio driver '{driver}' is simulated")
    if verification in {"simulated", "dummy", "software"}:
        errors.append(
            f"audio verification scope '{verification}' is not physical-device evidence"
        )

    if not require_hardware:
        return errors

    if verification != "hardware":
        errors.append("audio verification must be exactly 'hardware'")
    if not hardware_verified:
        errors.append("audio.hardware_verified must be true for release evidence")
    for field in _REQUIRED_HARDWARE_FIELDS:
        value = evidence.get(field)
        if field in {"sample_rate", "channels"}:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"audio.{field} must be a positive integer")
        elif not isinstance(value, str) or not value.strip():
            errors.append(f"audio.{field} must be a non-empty string")
    captured_at = evidence.get("captured_at")
    if not isinstance(captured_at, str) or not captured_at.strip():
        errors.append("audio.captured_at must identify when the device probe ran")
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Player acceptance JSON report; repeat for each target in the matrix",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Only reject simulated drivers; useful for non-release CI diagnostics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary: list[dict[str, Any]] = []
    failed = False
    for raw_path in args.report:
        path = Path(raw_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value is not an object")
            errors = validate_audio_report(
                payload, require_hardware=not args.allow_unverified
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors = [f"cannot read report: {type(exc).__name__}: {exc}"]
        failed = failed or bool(errors)
        summary.append(
            {
                "report": str(path),
                "status": "failed" if errors else "passed",
                "errors": errors,
            }
        )

    print(
        json.dumps(
            {
                "schema": "infernux.audio_report_gate",
                "status": "failed" if failed else "passed",
                "reports": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
