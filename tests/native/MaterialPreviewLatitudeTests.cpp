#include <function/renderer/gui/MaterialPreviewRenderer.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace infernux;

namespace
{
PreviewTexture MakeTexture(bool striped)
{
    PreviewTexture texture;
    texture.width = 8;
    texture.height = 128;
    texture.pixels.resize(8 * 128 * 4);
    for (int y = 0; y < 128; ++y) {
        for (int x = 0; x < 8; ++x) {
            const size_t offset = size_t(y * 8 + x) * 4;
            const int channel = striped ? (y < 51 ? 0 : (y < 77 ? 1 : 2)) : 1;
            texture.pixels[offset + channel] = 255;
            texture.pixels[offset + 3] = 255;
        }
    }
    return texture;
}

std::array<unsigned char, 4> Pixel(const std::vector<unsigned char> &pixels, int y)
{
    const size_t offset = size_t(y * 128 + 64) * 4;
    return {pixels.at(offset), pixels.at(offset + 1), pixels.at(offset + 2), pixels.at(offset + 3)};
}
} // namespace

int main()
{
    try {
        for (bool striped : {false, true}) {
            const auto texture = MakeTexture(striped);
            PreviewMaterialParams params;
            params.baseColor = glm::vec3(0);
            params.metallic = 1;
            params.specularHighlights = 0;
            params.ambientOcclusion = 0;
            params.emissionColor = glm::vec3(1);
            params.emissionTex = &texture;
            std::vector<unsigned char> actual;
            MaterialPreviewRenderer::RenderPreview(params, 128, actual);
            for (int y : {16, 32, 64, 96, 112}) {
                // Latitude oracle chooses a uniform emission color, independently
                // of the production UV calculation and texture sampling.
                const double normalY = 1.0 - 2.0 * (double(y) + .5) / 128.0;
                const double v = .5 - std::asin(normalY) / std::acos(-1.0);
                const int channel = striped ? (v < .4 ? 0 : (v < .6 ? 1 : 2)) : 1;
                PreviewMaterialParams reference = params;
                reference.emissionTex = nullptr;
                reference.emissionColor = glm::vec3(0);
                reference.emissionColor[channel] = 1;
                std::vector<unsigned char> expected;
                MaterialPreviewRenderer::RenderPreview(reference, 128, expected);
                if (Pixel(actual, y) != Pixel(expected, y))
                    throw std::runtime_error("Material sphere samples the wrong latitude at y=" + std::to_string(y));
            }
        }
        std::cout << "10 material sphere latitude checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
