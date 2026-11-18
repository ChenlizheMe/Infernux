#pragma once

#include "MaterialPassPipeline.h"
#include <function/resources/InxMaterial/MaterialRenderState.h>

namespace infernux
{

/// Pass policy belongs to the renderer, not to Vulkan or WebGPU. Depth
/// comparisons below remain logical near/far until converted into the RHI.
inline RenderState ResolveMaterialPassState(const RenderState &authored, const MaterialPassPipelineDescriptor &pass)
{
    auto state = authored;
    if (pass.invertCulling)
        state.frontFace = state.frontFace == MaterialFrontFace::Clockwise ? MaterialFrontFace::CounterClockwise
                                                                          : MaterialFrontFace::Clockwise;
    if (pass.target != ShaderCompileTarget::Forward && !IsForwardPlusMaterialPass(pass.target)) {
        state.blendEnable = false;
        state.depthTestEnable = pass.depthFormat != rhi::PixelFormat::Undefined;
        state.depthWriteEnable = state.depthTestEnable;
    }
    if (pass.depthReadOnly)
        state.depthWriteEnable = false;
    if (pass.target == ShaderCompileTarget::ThinSurface)
        state.depthCompareOp = MaterialCompareOp::Less;
    if (pass.target == ShaderCompileTarget::Motion || pass.target == ShaderCompileTarget::Normal ||
        pass.target == ShaderCompileTarget::BaseColor)
        state.depthCompareOp = MaterialCompareOp::LessOrEqual;
    if (!rhi::IsStencilFormat(pass.depthFormat))
        state.stencilTestEnable = false;
    return state;
}

inline rhi::GraphicsPipelineDesc MaterialPipelineDescription(const RenderState &authored,
                                                             const MaterialPassPipelineDescriptor &pass)
{
    const auto state = ResolveMaterialPassState(authored, pass);
    rhi::GraphicsPipelineDesc result;
    pass.ApplyRenderingContract(result);
    result.samples = pass.samples;
    constexpr std::array topologies{rhi::PrimitiveTopology::PointList, rhi::PrimitiveTopology::LineList,
                                    rhi::PrimitiveTopology::LineStrip, rhi::PrimitiveTopology::TriangleList,
                                    rhi::PrimitiveTopology::TriangleStrip};
    result.topology = topologies[static_cast<size_t>(state.topology)];
    result.raster.cullMode = static_cast<rhi::CullMode>(state.cullMode);
    result.raster.frontFace =
        state.frontFace == MaterialFrontFace::Clockwise ? rhi::FrontFace::Clockwise : rhi::FrontFace::CounterClockwise;
    result.raster.polygonMode = static_cast<rhi::PolygonMode>(state.polygonMode);
    result.raster.lineWidth = state.lineWidth;
    if (pass.target == ShaderCompileTarget::Shadow) {
        result.raster.depthBias = kShadowCasterDepthBias;
        result.raster.depthBiasSlope = kShadowCasterSlopeBias;
    } else if (state.depthBiasEnable) {
        result.raster.depthBias = -state.depthBiasConstantFactor;
        result.raster.depthBiasSlope = -state.depthBiasSlopeFactor;
        result.raster.depthBiasClamp = -state.depthBiasClamp;
    }
    result.depth = {state.depthTestEnable, state.depthWriteEnable,
                    rhi::ReverseDepthCompare(static_cast<rhi::CompareFunction>(state.depthCompareOp))};
    const auto face = [](const MaterialStencilOpState &value) -> rhi::StencilFaceState {
        return {static_cast<rhi::StencilOperation>(value.failOp),
                static_cast<rhi::StencilOperation>(value.passOp),
                static_cast<rhi::StencilOperation>(value.depthFailOp),
                static_cast<rhi::CompareFunction>(value.compareOp),
                value.compareMask,
                value.writeMask,
                value.reference};
    };
    result.stencil = {state.stencilTestEnable, face(state.stencilFront), face(state.stencilBack)};
    result.colorTargetCount = static_cast<uint32_t>(pass.colorFormats.size());
    for (uint32_t index = 0; index < result.colorTargetCount; ++index) {
        auto &color = result.colorTargets[index];
        color.format = pass.colorFormats[index];
        color.blendEnabled = state.blendEnable;
        color.colorBlend = {static_cast<rhi::BlendFactor>(state.srcColorBlendFactor),
                            static_cast<rhi::BlendFactor>(state.dstColorBlendFactor),
                            static_cast<rhi::BlendOperation>(state.colorBlendOp)};
        color.alphaBlend = {static_cast<rhi::BlendFactor>(state.srcAlphaBlendFactor),
                            static_cast<rhi::BlendFactor>(state.dstAlphaBlendFactor),
                            static_cast<rhi::BlendOperation>(state.alphaBlendOp)};
    }
    return result;
}

} // namespace infernux
