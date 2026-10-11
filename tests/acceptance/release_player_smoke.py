"""Run the cooked Release fixture using public gameplay APIs and normal exit.

Invoke with the candidate wheel installed. The child uses its own packaged
runtime; it neither imports the editor checkout nor opens PlayerDebug control.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def require_release_manifest(manifest: dict) -> None:
    contract = manifest.get('runtime_contract', {})
    if (manifest.get('debug_build') is not False or
            contract.get('product', {}).get('flavor') != 'PlayerRelease' or
            contract.get('runtime_policy', {}).get('player_control') != 'disabled'):
        raise RuntimeError('Release acceptance requires a PlayerRelease package with control disabled')


def verify_evidence(evidence: dict, capture: Path) -> None:
    import numpy as np
    from PIL import Image

    if evidence['status'] != 'passed' or evidence['fixed_steps'] < 90:
        raise RuntimeError(f'Release fixture did not complete gameplay: {evidence}')
    start, end = evidence['initial_position'], evidence['final_position']
    if len(start) != 3 or len(end) != 3 or not all(math.isfinite(value) for value in (*start, *end)):
        raise RuntimeError('Release physics evidence contains an invalid position')
    if end[2] - start[2] < .5 or end[1] < 0 or evidence['trail_points'] < 3:
        raise RuntimeError('Release physics or LineRenderer evidence is invalid')
    if evidence['cpu_jit_result'] != 84 or not evidence['renderer']['submission_ready']:
        raise RuntimeError('Release CPU compiler or renderer evidence is invalid')
    if evidence['package_text'] != 'Package resource reached UIText on every Player target.':
        raise RuntimeError('Release package resource did not reach the UI')
    if evidence['managed_guid'] != '8b7148eba8303c90b0589315c16f7cba':
        raise RuntimeError('Release managed asset GUID evidence is invalid')
    pixels = np.asarray(Image.open(capture).convert('RGB'), dtype=np.int16)
    if min(pixels.shape[:2]) < 64 or np.unique(pixels.reshape(-1, 3), axis=0).shape[0] < 16:
        raise RuntimeError('Release screenshot is empty or has no rendered scene')
    marker = (pixels[..., 1] > pixels[..., 0] + 40) & (pixels[..., 1] > pixels[..., 2] + 10)
    if np.count_nonzero(marker) < 20:
        raise RuntimeError('Release screenshot is missing the green Screen UI fixture marker')


def run(player: Path, artifacts: Path, timeout: float) -> dict:
    # Capture the clean caller environment before importing the installed
    # engine, which can adjust native-library lookup in its own process.
    environment = dict(os.environ)
    import infernux
    from infernux.lib import _Infernux
    from infernux.engine.platform_player_bootstrap import read_player_build_manifest

    checkout = Path(__file__).resolve().parents[2]
    origins = [Path(infernux.__file__).resolve(), Path(_Infernux.__file__).resolve()]
    if any(origin.is_relative_to(checkout) for origin in origins):
        raise RuntimeError('Release acceptance requires an installed wheel outside the source checkout')
    data = player.with_name(player.stem + '_Data')
    manifest = read_player_build_manifest(data)
    require_release_manifest(manifest)
    state = artifacts / 'state'
    state.mkdir(parents=True, exist_ok=False)
    for name in tuple(environment):
        if name.startswith('_INFERNUX_PLAYER_') or name in (
                'PYTHONHOME', 'PYTHONPATH', 'INFERNUX_NATIVE_MODULE_DIR',
                'INFERNUX_GPU_JIT_VENDOR_DIR', 'LD_LIBRARY_PATH'):
            del environment[name]
    environment.update(LOCALAPPDATA=str(state), XDG_STATE_HOME=str(state),
                       _INFERNUX_FIXTURE_RELEASE_ACCEPTANCE='1', SDL_AUDIODRIVER='dummy')
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    started = time.monotonic()
    with (artifacts / 'outer.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen([str(player)], cwd=player.parent, env=environment,
                              stdout=log, stderr=subprocess.STDOUT, creationflags=flags) as process:
            try:
                result = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError('Release Player failed to complete and exit within its deadline') from None
    if result != 0:
        raise RuntimeError(f'Release Player exited with code {result}; see {artifacts}')
    persistent = state / 'Infernux/Players' / player.stem / 'Data'
    evidence = json.loads((persistent / 'release-acceptance.json').read_text(encoding='utf-8'))
    verify_evidence(evidence, persistent / 'release-acceptance.png')
    fatal_patterns = ('Validation Error', 'VUID-', 'Traceback (most recent call last)',
                      '[ERROR]', 'VK_ERROR_DEVICE_LOST', 'Segmentation fault')
    errors = []
    for log in artifacts.rglob('*.log'):
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
            if any(pattern in line for pattern in fatal_patterns):
                errors.append(f'{log.name}: {line}')
    if errors:
        raise RuntimeError(f'Release Player emitted errors: {errors[:20]}')
    report = dict(status='passed', player=str(player), installed_origins=list(map(str, origins)),
                  elapsed_seconds=time.monotonic()-started, debug_control='disabled',
                  exit_code=result, evidence=evidence)
    (artifacts / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('player', type=Path)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    result = run(args.player.resolve(), args.artifacts.resolve(), args.timeout)
    print(json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()
