"""Real CTest pass/skip/failure reports must not produce a false GPU gate pass."""
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tests.acceptance.software_vulkan_gate import run_gate, require_executed_results


@pytest.mark.parametrize('outcome', ['pass', 'skip', 'fail', 'missing'])
def test_gate_consumes_actual_ctest_result(tmp_path, outcome):
    source = tmp_path / 'source'
    source.mkdir()
    build = tmp_path / 'build'
    script = source / 'result.py'
    script.write_text(f'raise SystemExit({dict(skip=77, fail=9).get(outcome, 0)})\n')
    test_name = 'fixture.absent' if outcome == 'missing' else 'fixture.gpu'
    (source / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.25)\nproject(Gate NONE)\nenable_testing()\n'
        f'add_test(NAME {test_name} COMMAND "{Path(sys.executable).as_posix()}" "{script.as_posix()}")\n'
        f'set_tests_properties({test_name} PROPERTIES SKIP_RETURN_CODE 77)\n', encoding='utf-8')
    subprocess.run(['cmake', '-S', str(source), '-B', str(build)], check=True, capture_output=True)
    args = (['--test-dir', str(build), '-C', 'Debug'], tmp_path / 'evidence', ('fixture.gpu',))
    if outcome == 'pass':
        run_gate(*args)
    else:
        error = 'qualification set' if outcome == 'missing' else 'failed or were skipped'
        with pytest.raises(RuntimeError, match=error):
            run_gate(*args)


@pytest.mark.parametrize('body', [
    '',
    '<testcase name="gpu"/><testcase name="gpu"/>',
    '<testcase name="other"/>',
    '<testcase name="gpu" status="notrun"/>',
    '<testcase name="gpu"><error/></testcase>',
])
def test_incomplete_or_duplicate_evidence_is_rejected(tmp_path, body):
    report = tmp_path / 'results.xml'
    report.write_text(f'<testsuite>{body}</testsuite>', encoding='utf-8')
    with pytest.raises(RuntimeError):
        require_executed_results(report, ('gpu',))


def test_both_desktop_jobs_require_gpu_gate_and_upload_its_evidence():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / '.github/workflows/ci.yml').read_text(encoding='utf-8'))
    for platform, preset in (('windows', 'windows-msvc-release'), ('linux', 'linux-clang-release')):
        steps = workflow['jobs'][platform + '-desktop']['steps']
        gate = next(step for step in steps if 'software_vulkan_gate.py' in step.get('run', ''))
        assert '--preset ' + preset in gate['run']
        assert '--output out/test-results/gpu-' + platform in gate['run']
        assert 'if' not in gate and not gate.get('continue-on-error', False)
        assert steps.index(gate) < next(i for i, step in enumerate(steps) if 'Install complete' in step.get('name', ''))
        evidence = next(step for step in steps if
                        f'out/test-results/gpu-{platform}/' in step.get('with', {}).get('path', ''))
        assert evidence['if'] == 'always()'
