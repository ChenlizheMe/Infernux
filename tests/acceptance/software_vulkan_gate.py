"""Required GPU correctness coverage on the CI software Vulkan device.

This is an explicit qualification set, not a substitute for the hardware GPU
release suite. In particular, 2x/8x MSAA is not supported by both CI drivers.
Missing tests and skipped tests are failures, even when CTest returns zero.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


REQUIRED_TESTS = tuple('infernux.' + name for name in (
    'material_texture_readiness_vulkan',
    'compute_dispatch_limits_vulkan',
    'vulkan_submission_executor',
    'render_graph_compute_indirect',
    'render_texture_vulkan',
    'screen_ui_vulkan',
    'render_graph_queue_replay',
    'buffer_upload_consumer.async_graphics',
    'buffer_upload_consumer.async_compute',
    'image_readback.graphics',
    'image_readback.graphics_cancelled',
    'image_readback.graphics_drain',
    'gpu_upload',
    'mesh_index_format_gpu',
    'material_first_shadow_gpu',
    'default_material_first_shadow_gpu',
    'material_raster_state_gpu',
    'surface_gbuffer_scalars_gpu',
    'deformed_pbr_lighting_gpu',
    'skinned_deformed_shadow_gpu',
    'skinned_motion_blur_gpu',
    'effect_stage_context_gpu',
    'renderstack_recovery_gpu',
    'render_effect_group_order_gpu',
    'camera_commit_grade_screen_gpu',
    'camera_commit_grade_asset_gpu',
    'shared_camera_forward_gpu',
    'shared_camera_forward_plus_gpu',
    'shared_camera_deferred_gpu',
    'view_history_gpu',
    'material_transparency_forward_1_gpu',
    'material_transparency_forward_4_gpu',
    'material_transparency_forward_plus_1_gpu',
    'material_transparency_forward_plus_4_gpu',
    'material_transparency_deferred_1_gpu',
    'material_transparency_deferred_4_gpu',
    'fullscreen_depth_1_gpu',
    'fullscreen_depth_4_gpu',
    'mesh_reject_ParticleSprite_forward_gpu',
    'mesh_reject_ParticleSprite_forward_plus_gpu',
    'mesh_reject_ParticleSprite_deferred_gpu',
    'texture_import_semantics_gpu',
    'mixed_deferred_coverage_gpu',
    'bloom_chain_forward_layer_gpu',
    'bloom_chain_forward_route_gpu',
    'bloom_chain_forward_plus_layer_gpu',
    'bloom_chain_forward_plus_route_gpu',
    'bloom_chain_deferred_layer_gpu',
    'bloom_chain_deferred_route_gpu',
    'compute_buffer_gpu',
    'compute_kernel_gpu',
    'compute_mesh_gpu',
    'compute_gizmo_gpu',
    'compute_physics_exchange_gpu',
    'mesh_publication_gpu',
    'mesh_publication_gpu_continuous',
    'runtime_mesh_retirement_gpu',
    'runtime_mesh_retirement_multi_camera_gpu',
    'runtime_mesh_resource_gpu',
    'scene_residency_soak',
))


def require_executed_results(junit: Path, required: tuple[str, ...]) -> None:
    cases = list(ET.parse(junit).getroot().iter('testcase'))
    actual = Counter(case.get('name') for case in cases)
    if actual != Counter(required):
        raise RuntimeError(f'GPU evidence does not match the required tests: {actual}')
    rejected = [case.get('name') for case in cases if
                case.get('status') not in (None, 'run') or
                any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped'))]
    if rejected:
        raise RuntimeError(f'Required GPU tests failed or were skipped: {rejected}')


def run_gate(ctest_args: list[str], output: Path, required=REQUIRED_TESTS) -> None:
    output.mkdir(parents=True, exist_ok=True)
    expression = '^(' + '|'.join(re.escape(name) for name in required) + ')$'
    command = ['ctest', *ctest_args, '-R', expression]
    inventory = subprocess.run([*command, '--show-only=json-v1'], check=True,
                               capture_output=True, text=True)
    (output / 'software-vulkan-inventory.json').write_text(inventory.stdout, encoding='utf-8')
    tests = json.loads(inventory.stdout)['tests']
    actual = Counter(test['name'] for test in tests)
    if actual != Counter(required):
        raise RuntimeError(f'Configured GPU tests differ from the qualification set: {actual}')
    junit = (output / 'software-vulkan.xml').resolve()
    # Never accept a previous invocation's report if CTest fails before writing.
    junit.unlink(missing_ok=True)
    with (output / 'software-vulkan.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen([*command, '--parallel', '1', '--output-on-failure',
                               '--output-junit', str(junit)], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            result = process.wait()
    require_executed_results(junit, required)
    if result:
        raise RuntimeError(f'GPU CTest process failed with exit code {result}')
    print(f'Software Vulkan gate: {len(required)} executed, passed; zero skips.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--preset')
    source.add_argument('--build-dir', type=Path)
    parser.add_argument('--configuration')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    driver = os.environ.get('VK_DRIVER_FILES') or os.environ.get('VK_ICD_FILENAMES')
    if not driver or not Path(driver).is_file():
        parser.error('Set VK_DRIVER_FILES to the qualified software Vulkan ICD manifest.')
    command = ['--preset', args.preset] if args.preset else ['--test-dir', str(args.build_dir)]
    if args.configuration:
        command += ['-C', args.configuration]
    run_gate(command, args.output)


if __name__ == '__main__':
    main()
