"""Real fullscreen white-balance fixture, also driven by the editor MCP probe."""
import numpy as np

from infernux.rendergraph import Format
from infernux.renderstack import RenderPipeline, WhiteBalanceEffect
from infernux.renderstack.resource_bus import ResourceBus


COLORS = np.array([
    (0, 0, 0, 0), (.18, .18, .18, .125), (1, 1, 1, .5), (4, 2, 1, .875),
    (1, 0, 0, 1), (0, 1, 0, .25), (0, 0, 1, .75), (.001, .1, .4, .625),
], dtype=np.float32)
PHASES = [(False, 0., 0.), (True, 0., 0.)] + [
    (True, temperature, tint) for temperature, tint in (
        (-100., -100.), (-100., 0.), (-100., 100.),
        (0., -100.), (0., 100.), (100., -100.), (100., 0.), (100., 100.),
        (-25., 0.), (25., 0.), (0., -25.), (0., 25.),
        (-.001, 0.), (.001, 0.), (0., -.001), (0., .001), (0., 0.),
    )
] + [(False, 0., 0.)]

INPUT_SHADER = '''#version 450
ShaderInfo {
    Name "White Balance Acceptance Input"
    Hidden On
    Capabilities [Fullscreen]
    Inputs { Float2 inUV }
    Outputs { Float4 outColor }
}
void main() {
    const vec4 colors[8] = vec4[8](
        vec4(0,0,0,0), vec4(.18,.18,.18,.125), vec4(1,1,1,.5), vec4(4,2,1,.875),
        vec4(1,0,0,1), vec4(0,1,0,.25), vec4(0,0,1,.75), vec4(.001,.1,.4,.625));
    outColor = colors[min(int(inUV.x * 8.0), 7)];
}
'''

LIBRARY_SHADER = '''#version 450
ShaderInfo {
    Name "White Balance Acceptance Library"
    Hidden On
    Imports ["Lib Color"]
    Capabilities [Fullscreen]
    Resources { Texture2D _SourceTex }
    PushConstants pc { Float temperature Float tint Float _pad0 Float _pad1 }
    Inputs { Float2 inUV }
    Outputs { Float4 outColor }
}
void main() {
    vec4 source = texture(_SourceTex, inUV);
    outColor = vec4(whiteBalance(source.rgb, pc.temperature, pc.tint), source.a);
}
'''

MONITOR_SHADER = '''#version 450
ShaderInfo {
    Name "White Balance Acceptance Monitor"
    Hidden On
    Capabilities [Fullscreen]
    Resources { Texture2D _SourceTex }
    Inputs { Float2 inUV }
    Outputs { Float4 outColor }
}
void main() {
    vec4 value = texture(_SourceTex, inUV);
    outColor = vec4(value.rgb / 32.0, value.a);
}
'''


def expected_linear(temperature, tint):
    """Independent XYZ/CAT02 reference; neutral is the input-color contract."""
    colors = COLORS.astype(np.float16).astype(np.float64)
    if temperature == tint == 0:
        return colors
    # sRGB/D65 primaries to XYZ; apply chromatic adaptation there instead of
    # copying the production shader's combined RGB-to-LMS/inverse constants.
    rgb_to_xyz = np.array([[.4124564, .3575761, .1804375],
                           [.2126729, .7151522, .0721750],
                           [.0193339, .1191920, .9503041]])
    cat02 = np.array([[.7328, .4296, -.1624], [-.7036, 1.6975, .0061], [.003, .0136, .9834]])
    x0, y0 = .31271, .32902
    t = temperature / 65.
    x = x0 - t * (.1 if t < 0 else .05)
    y = y0 + (x - x0) * (2.87 - 3 * (x + x0)) + tint / 65. * .05
    reference = cat02 @ [x0 / y0, 1., (1. - x0 - y0) / y0]
    white = cat02 @ [x / y, 1., (1. - x - y) / y]
    adaptation = np.linalg.inv(cat02) @ np.diag(reference / white) @ cat02
    transform = np.linalg.inv(rgb_to_xyz) @ adaptation @ rgb_to_xyz
    colors[:, :3] = np.maximum(colors[:, :3] @ transform.T, 0.)
    return colors


def display_values(linear):
    result = linear.copy()
    # Keep the HDR range observable through the camera's LDR display encoding.
    values = result[:, :3] / 32.
    result[:, :3] = np.where(values <= .0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - .055)
    return result


class WhiteBalancePipeline(RenderPipeline):
    name = 'White Balance Acceptance'

    def __init__(self, *, library=False):
        super().__init__()
        self.effect = WhiteBalanceEffect()
        self.library = library
        self.use_effect = False

    def change(self, phase):
        self.use_effect, self.effect.temperature, self.effect.tint = phase
        self.dispose()

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        source = graph.create_texture('input', format=Format.RGBA16_SFLOAT, samples=1)
        depth = graph.create_texture('depth', format=Format.D32_SFLOAT, samples=1)
        graph.add_pass('Color patches').write_color(source).write_depth(depth).set_clear(depth=1.).fullscreen_quad('White Balance Acceptance Input')
        bus = ResourceBus({'color': source}, graph=graph)
        if self.use_effect:
            if self.library:
                self.effect.apply_single_source_effect(graph, bus, output_name='_library_white_balance',
                    pass_name='Library White Balance', shader_name='White Balance Acceptance Library',
                    format=Format.RGBA16_SFLOAT,
                    params={'temperature':self.effect.temperature, 'tint':self.effect.tint})
            else:
                self.effect.setup_passes(graph, bus)
        target = graph.create_texture('color', camera_target=True)
        graph.add_pass('Commit').write_color(target).set_texture('_SourceTex', bus.get('color')).fullscreen_quad('White Balance Acceptance Monitor')
        graph.screen_ui_section(resources={'color'})
        graph.set_output(target)


def check_pixels(pixels, phase):
    rows, columns = pixels.shape[:2]
    samples = np.array([pixels[rows // 2, int((i + .5) * columns / 8)] for i in range(8)], dtype=np.float32)
    enabled, temperature, tint = phase
    expected = expected_linear(temperature, tint) if enabled else COLORS.astype(np.float16).astype(np.float64)
    expected = display_values(expected)
    assert np.isfinite(samples).all(), samples
    np.testing.assert_allclose(samples[:, :3], expected[:, :3], atol=.001, rtol=.003, err_msg=str(phase))
    np.testing.assert_allclose(samples[:, 3], COLORS[:, 3], atol=.0006, rtol=0., err_msg=str(phase))
    np.testing.assert_array_equal(samples[0], np.zeros(4))
    return samples.tolist()
