#include <algorithm>
#include <function/resources/InxTexture/SvgRasterizer.h>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
void Check(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}
infernux::InxTextureData Decode(const std::string &svg, int size)
{
    return infernux::RasterizeSvg(reinterpret_cast<const unsigned char *>(svg.data()), svg.size(), size);
}
const std::string solid =
    R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 2"><rect width="4" height="2" fill="#ff0040" fill-opacity="0.5"/></svg>)";
} // namespace

int main()
try {
    for (const int size : {16, 32, 128, 512}) {
        const auto image = Decode(solid, size);
        Check(image.width == size && image.height == size / 2, "SVG must rasterize directly at requested size");
        Check(image.pixels.size() == static_cast<size_t>(size) * size * 2, "RGBA byte count");
        Check(image.pixels[0] >= 253 && image.pixels[1] == 0 && image.pixels[2] >= 62 && image.pixels[2] <= 66 &&
                  image.pixels[3] >= 126 && image.pixels[3] <= 129,
              "SVG channels/straight alpha must match PNG");
    }
    // Exercise the real cap without allocating three full 8192-square images.
    for (const int size : {0, 8192, 65536}) {
        const auto image = Decode("<svg viewBox='0 0 8192 1'><rect width='8192' height='1'/></svg>", size);
        Check(image.width == 8192 && image.height == 1, "unlimited SVG must resolve to 8192");
    }
    const auto portrait = Decode("<svg width='2' height='4'><rect width='2' height='4'/></svg>", 32);
    Check(portrait.width == 16 && portrait.height == 32, "portrait aspect ratio");
    const auto translated =
        Decode("<svg viewBox='10 20 4 2'><rect x='10' y='20' width='4' height='2' fill='red'/></svg>", 32);
    Check(translated.pixels[0] == 255 && translated.pixels[3] == 255, "nonzero viewBox origin");
    const auto blank = Decode("<svg viewBox='0 0 4 2'/>", 32);
    const auto documentImage = infernux::RasterizeSvg(
        reinterpret_cast<const unsigned char *>(solid.data()), solid.size(), 0, true);
    Check(documentImage.width == 4 && documentImage.height == 2, "documentation must not upscale its viewport");
    const std::string filter = "<svg><!-- <filter/> --><defs><filter id='blur'/></defs></svg>";
    Check(infernux::SvgUsesFilters(reinterpret_cast<const unsigned char *>(filter.data()), filter.size()),
          "unsupported SVG filters need diagnostics");
    const std::string comment = "<svg><!-- <filter/> --><![CDATA[<filter/>]]></svg>";
    Check(!infernux::SvgUsesFilters(reinterpret_cast<const unsigned char *>(comment.data()), comment.size()),
          "comments and text are not filters");
    Check(std::all_of(blank.pixels.begin(), blank.pixels.end(), [](auto p) { return p == 0; }),
          "empty SVG is transparent");
    const auto gradient = Decode(
        R"svg(<svg viewBox="0 0 100 100"><defs><linearGradient id="g"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><circle cx="50" cy="50" r="45" fill="url(#g)"/></svg>)svg",
        100);
    Check(gradient.pixels[3] == 0, "transparent corners");
    Check(gradient.pixels[(50 * 100 + 20) * 4] > gradient.pixels[(50 * 100 + 80) * 4], "gradient rendering");
    for (const std::string source :
         {"not svg", "<svg><", "<html/>", "<svg width='0' height='4'/>", "<svg width='4' height='0'/>"}) {
        bool failed = false;
        try {
            (void)Decode(source, 32);
        } catch (const std::exception &) {
            failed = true;
        }
        Check(failed, "invalid SVG must fail explicitly");
    }
    const std::string withProlog = "\xef\xbb\xbf<?xml version='1.0'?><!-- test -->" + solid;
    Check(infernux::IsSvgSource(reinterpret_cast<const unsigned char *>(withProlog.data()), withProlog.size()),
          "SVG signature with BOM/prolog");
    Check(Decode(withProlog, 32).width == 32, "BOM/prolog decode");
    auto worker = std::async(std::launch::async, [] { return Decode(solid, 64); });
    Check(Decode(solid, 32).width == 32 && worker.get().width == 64, "concurrent import/preview");
    std::cout << "SVG_RASTERIZER_OK\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
