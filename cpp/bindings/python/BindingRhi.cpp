#include "BindingRegistration.h"
#include <function/renderer/rhi/RhiCommand.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/renderer/rhi/RhiTypes.h>
#include <function/resources/RenderTexture/RenderTextureArtifact.h>
#include <nlohmann/json.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace infernux
{

void RegisterRhiBindings(py::module_ &m)
{
    py::enum_<rhi::PixelFormat>(m, "PixelFormat", "Backend-neutral pixel format")
        .value("UNDEFINED", rhi::PixelFormat::Undefined)
        .value("R8_UNORM", rhi::PixelFormat::R8UNorm)
        .value("RG8_UNORM", rhi::PixelFormat::RG8UNorm)
        .value("RGBA8_UNORM", rhi::PixelFormat::RGBA8UNorm)
        .value("RGBA8_SRGB", rhi::PixelFormat::RGBA8Srgb)
        .value("BGRA8_UNORM", rhi::PixelFormat::BGRA8UNorm)
        .value("R16_SFLOAT", rhi::PixelFormat::R16SFloat)
        .value("RG16_SFLOAT", rhi::PixelFormat::RG16SFloat)
        .value("RGBA16_SFLOAT", rhi::PixelFormat::RGBA16SFloat)
        .value("R32_SFLOAT", rhi::PixelFormat::R32SFloat)
        .value("RG32_UINT", rhi::PixelFormat::RG32UInt)
        .value("RGBA32_UINT", rhi::PixelFormat::RGBA32UInt)
        .value("RGBA32_SFLOAT", rhi::PixelFormat::RGBA32SFloat)
        .value("RGB10A2_UNORM", rhi::PixelFormat::RGB10A2UNorm)
        .value("D32_SFLOAT", rhi::PixelFormat::D32SFloat)
        .value("D24_UNORM_S8_UINT", rhi::PixelFormat::D24UNormS8UInt)
        .value("D32_SFLOAT_S8_UINT", rhi::PixelFormat::D32SFloatS8UInt)
        .def_property_readonly("is_depth", [](rhi::PixelFormat format) { return rhi::IsDepthFormat(format); })
        .def_property_readonly("buffer_copy_texel_bytes", &rhi::TextureBufferCopyRegion::TexelBytes,
                               "Bytes per texel for a packed buffer copy, or zero for unsupported formats");

    py::enum_<rhi::SampleCount>(m, "SampleCount", "Backend-neutral MSAA sample count")
        .value("COUNT_1", rhi::SampleCount::One)
        .value("COUNT_2", rhi::SampleCount::Two)
        .value("COUNT_4", rhi::SampleCount::Four)
        .value("COUNT_8", rhi::SampleCount::Eight);

    py::class_<rhi::RenderTextureDesc>(m, "_RenderTextureDesc")
        .def(py::init<>())
        .def_property(
            "relative_size",
            [](const rhi::RenderTextureDesc &desc) { return desc.sizeMode == rhi::RenderTextureSizeMode::Relative; },
            [](rhi::RenderTextureDesc &desc, bool relative) {
                desc.sizeMode = relative ? rhi::RenderTextureSizeMode::Relative : rhi::RenderTextureSizeMode::Absolute;
            })
        .def_readwrite("width_scale", &rhi::RenderTextureDesc::widthScale)
        .def_readwrite("height_scale", &rhi::RenderTextureDesc::heightScale)
        .def_readwrite("width", &rhi::RenderTextureDesc::width)
        .def_readwrite("height", &rhi::RenderTextureDesc::height)
        .def_readwrite("color_format", &rhi::RenderTextureDesc::colorFormat)
        .def_readwrite("depth_format", &rhi::RenderTextureDesc::depthFormat)
        .def_readwrite("samples", &rhi::RenderTextureDesc::samples)
        .def_readwrite("storage", &rhi::RenderTextureDesc::storage)
        .def_readwrite("sampled_depth", &rhi::RenderTextureDesc::sampledDepth)
        .def_property(
            "linear_filter", [](const rhi::RenderTextureDesc &desc) { return desc.filter == rhi::FilterMode::Linear; },
            [](rhi::RenderTextureDesc &desc, bool linear) {
                desc.filter = linear ? rhi::FilterMode::Linear : rhi::FilterMode::Nearest;
            });
    m.def("_render_texture_description_from_json", [](const std::string &source) {
        return RenderTextureArtifact::ParseDocument(nlohmann::json::parse(source));
    });
    m.def("_render_texture_format_names", &RenderTextureArtifact::FormatNames, py::arg("depth"));
    m.def("_render_texture_description_to_json", [](const rhi::RenderTextureDesc &description) {
        return RenderTextureArtifact::SerializeDocument(description).dump(2);
    });
    m.def("_encode_render_texture_artifact", [](const rhi::RenderTextureDesc &description, const std::string &hash) {
        return py::bytes(RenderTextureArtifact::Encode(description, hash));
    });
    m.def("_decode_render_texture_artifact", [](const py::bytes &bytes) {
        const std::string data = bytes;
        return RenderTextureArtifact::Decode(data);
    });
    py::class_<rhi::RenderTexture, std::shared_ptr<rhi::RenderTexture>>(m, "_RenderTexture")
        .def_property_readonly("asset_guid", &rhi::RenderTexture::GetAssetGuid)
        .def_property_readonly("width", [](const rhi::RenderTexture &texture) { return texture.Acquire()->width; })
        .def_property_readonly("height", [](const rhi::RenderTexture &texture) { return texture.Acquire()->height; })
        .def_property_readonly("description",
                               [](const rhi::RenderTexture &texture) { return texture.Acquire()->description; })
        .def_property_readonly("revision",
                               [](const rhi::RenderTexture &texture) { return texture.Acquire()->revision; })
        .def_property_readonly("is_valid",
                               [](const rhi::RenderTexture &texture) { return texture.Acquire()->color->IsValid(); })
        .def_property_readonly("resident_bytes",
                               [](const rhi::RenderTexture &texture) { return texture.Acquire()->GetResidentBytes(); })
        .def(
            "resize",
            [](rhi::RenderTexture &texture, uint32_t width, uint32_t height) {
                auto description = texture.Acquire()->description;
                if (description.sizeMode == rhi::RenderTextureSizeMode::Relative)
                    throw std::invalid_argument(
                        "Relative RenderTexture dimensions are owned by the Game output resolution");
                description.width = width;
                description.height = height;
                return texture.Reconfigure(description);
            },
            py::arg("width"), py::arg("height"));
}

} // namespace infernux
