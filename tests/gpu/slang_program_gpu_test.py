"""Slang programs run through the engine's real Vulkan compute lane."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile

import numpy as np

import infernux as inx
from infernux._source_kinds import SHADER_SOURCE_KINDS
from infernux.lib import _Infernux as native_api


SOURCE = """
struct Parameters { uint count; float scale; float bias; uint begin; };
ConstantBuffer<Parameters> params;
StructuredBuffer<float> inputs;
RWStructuredBuffer<float> outputs;

[Differentiable]
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

[shader("compute")]
[numthreads(32, 1, 1)]
void affine(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < params.count) {
        uint i = tid.x + params.begin;
        outputs[i] = inputs[i] * params.scale + params.bias;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void sigmoid_vjp(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < params.count) {
        uint i = tid.x + params.begin;
        DifferentialPair<float> x = diffPair(inputs[i], 0.0);
        bwd_diff(sigmoid)(x, params.scale);
        outputs[i] = x.d;
    }
}
"""


def main() -> None:
    compiled = native_api._compile_shader_program(SOURCE, "slang_probe.compute.py", "slang_probe")
    spirv = next(target for target in compiled["targets"] if target["target"] == "spirv")
    reflection = json.loads(spirv["reflection"])
    parameters = sorted(reflection["parameters"], key=lambda item: item["binding"]["index"])
    bindings = [(item["binding"]["index"],
                 "uniform" if item["type"]["kind"] == "constantBuffer" else
                 "read_only_storage" if item["type"].get("access", "read") == "read" else "storage")
                for item in parameters]
    count, begin = 255, 1
    initial = np.linspace(-4.0, 4.0, 257, dtype=np.float32)
    constants = np.frombuffer(struct.pack("IffI", count, 2.5, 0.75, begin), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="infernux-slang-gpu-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        buffers = {}
        kernels = []
        host = None
        try:
            frontend.init_renderer(64, 64, str(project))
            database = native.get_asset_database()
            author_path = project / "Assets" / "NativeSource.frag.py"
            author_path.write_text('''import infernux.shader as shader
from infernux.shader.lib.color import luminance

class NativeSource(shader.fullscreen):
    image = shader.input("scene", shader.Color2D, domain="scene_linear_hdr")
    strength = shader.parameter.float(default=0.75, label="强度", range=(0.0, 1.0))

    def main(self, uv: shader.vec2) -> shader.vec4:
        color = shader.sample_level(self.image, uv, 0.0)
        return shader.vec4(shader.vec3(luminance(color.rgb)) * self.strength, color.a)
''', encoding="utf-8-sig")
            imported_author = database.import_asset(str(author_path))
            assert imported_author, imported_author.error
            schema = inx.Shader.get_schema(str(author_path))
            assert schema["name"] == "NativeSource"
            assert schema["parameters"][0]["label"] == "强度"
            assert schema["inputs"][0]["domain"] == "scene_linear_hdr"
            for suffix, (stage, _, _) in SHADER_SOURCE_KINDS.items():
                path = project / "Assets" / ("Static" + suffix.upper())
                path.write_text('raise AssertionError("GPU source must not execute on the host")', encoding="utf-8")
                imported = database.import_asset(str(path))
                assert imported, imported.error
                metadata = database.get_meta_by_path(str(path))
                assert metadata.get_resource_type() == native_api.ResourceType.Shader
                assert metadata.get_string("type") == stage
                assert metadata.get_string("file_extension") == suffix
            host = native._acquire_compute_host()
            buffers["params"] = inx.buffer(shape=4, dtype=np.float32, device="gpu", data=constants)
            buffers["inputs"] = inx.buffer(shape=257, dtype=np.float32, device="gpu", data=initial)
            buffers["outputs"] = inx.buffer(shape=257, dtype=np.float32, device="gpu",
                                            data=np.full(257, -37.0, dtype=np.float32))
            ordered = [buffers[item["name"]]._native for item in parameters]
            input_serials = {name: buffers[name]._native.last_write_serial for name in ("params", "inputs")}
            for entry in spirv["entries"]:
                kernel = host._create_kernel_from_compiler(
                    entry["code"], bindings, entry_point=entry["name"], target="spirv",
                )
                kernels.append(kernel)
                kernel.dispatch(ordered, group_count_x=(count + 31) // 32)
                result = buffers["outputs"].get_data().numpy()
                expected = np.full(257, -37.0, dtype=np.float32)
                active = initial[begin:begin + count]
                if entry["name"] == "affine":
                    expected[begin:begin + count] = active * 2.5 + 0.75
                else:
                    sigmoid = 1.0 / (1.0 + np.exp(-active.astype(np.float64)))
                    expected[begin:begin + count] = 2.5 * sigmoid * (1.0 - sigmoid)
                np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-7)
                assert result[0] == result[-1] == -37.0
                assert {name: buffers[name]._native.last_write_serial for name in input_serials} == input_serials
                print(f"{entry['name']}: real GPU output and range sentinels passed")
            print(f"Slang {native_api._slang_version()}: engine Vulkan execution passed")
        finally:
            for kernel in kernels:
                kernel.wait()
            kernels.clear()
            ordered = None
            kernel = None
            for buffer in buffers.values():
                buffer.close()
            buffers.clear()
            host = None
            inx.compute._release_engine_resources()
            native.cleanup()


if __name__ == "__main__":
    main()
