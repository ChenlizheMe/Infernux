#include "BindingRegistration.h"
#include "MatrixPyBridge.h"
#include "ShaderParameterPyBridge.h"
#include <function/renderer/MaterialPassPipeline.h>
#include <function/renderer/RenderGraphDescription.h>
#include <function/renderer/RendererSelection.h>
#include <function/renderer/raytracing/RayWorldTailCache.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace infernux
{
void RegisterRenderGraphDescriptionBindings(py::module_ &m)
{
    m.def("_ray_tail_batch_size", [] { return infernux::raytracing::RayTailCacheConfig{}.batchSize; });
    m.def("_gbuffer_formats", [] { return kGBufferColorFormats; });
    // ========================================================================
    // RenderGraph topology binding types
    // ========================================================================

    py::enum_<ShaderCompileTarget>(m, "MaterialPassType", "Material shader program selected by a render pass")
        .value("FORWARD", ShaderCompileTarget::Forward)
        .value("FORWARD_PLUS", ShaderCompileTarget::ForwardPlus)
        .value("GBUFFER", ShaderCompileTarget::GBuffer)
        .value("SHADOW", ShaderCompileTarget::Shadow)
        .value("DEPTH", ShaderCompileTarget::Depth)
        .value("PICKING", ShaderCompileTarget::Picking)
        .value("MOTION", ShaderCompileTarget::Motion)
        .value("NORMAL", ShaderCompileTarget::Normal)
        .value("BASE_COLOR", ShaderCompileTarget::BaseColor)
        .value("RAY_SURFACE", ShaderCompileTarget::RaySurface)
        .value("TRANSPARENT_BEHIND", ShaderCompileTarget::TransparentBehind)
        .value("TRANSPARENT_FRONT", ShaderCompileTarget::TransparentFront)
        .value("THIN_SURFACE", ShaderCompileTarget::ThinSurface);

    py::enum_<rhi::CompareFunction>(m, "DepthCompare", "Depth comparison against a graph attachment")
        .value("NEVER", rhi::CompareFunction::Never)
        .value("LESS", rhi::CompareFunction::Less)
        .value("EQUAL", rhi::CompareFunction::Equal)
        .value("LESS_EQUAL", rhi::CompareFunction::LessEqual)
        .value("GREATER", rhi::CompareFunction::Greater)
        .value("NOT_EQUAL", rhi::CompareFunction::NotEqual)
        .value("GREATER_EQUAL", rhi::CompareFunction::GreaterEqual)
        .value("ALWAYS", rhi::CompareFunction::Always);

    py::enum_<GraphCommandType>(m, "GraphCommandType", "Backend-neutral command recorded by a graph pass")
        .value("DRAW_RENDERERS", GraphCommandType::DrawRenderers)
        .value("DRAW_SKYBOX", GraphCommandType::DrawSkybox)
        .value("DRAW_SHADOW_CASTERS", GraphCommandType::DrawShadowCasters)
        .value("DRAW_WORLD_UI", GraphCommandType::DrawWorldUI)
        .value("DRAW_SCREEN_UI", GraphCommandType::DrawScreenUI)
        .value("FULLSCREEN_QUAD", GraphCommandType::FullscreenQuad)
        .value("COPY_TEXTURE", GraphCommandType::CopyTexture)
        .value("COPY_BUFFER", GraphCommandType::CopyBuffer)
        .value("COPY_TEXTURE_TO_BUFFER", GraphCommandType::CopyTextureToBuffer)
        .value("COPY_BUFFER_TO_TEXTURE", GraphCommandType::CopyBufferToTexture)
        .value("COMPUTE_WORK", GraphCommandType::ComputeWork)
        .value("RAY_TAIL_PREPARE", GraphCommandType::RayTailPrepare)
        .value("RAY_TAIL_MERGE", GraphCommandType::RayTailMerge)
        .value("PRESENT", GraphCommandType::Present);

    py::enum_<GraphPassType>(m, "GraphPassType", "Execution domain of a graph pass")
        .value("RASTER", GraphPassType::Raster)
        .value("COMPUTE", GraphPassType::Compute)
        .value("COPY", GraphPassType::Copy)
        .value("PRESENT", GraphPassType::Present);

    py::enum_<GraphMaterialFilter>(m, "GraphMaterialFilter", "Shading-model compatibility filter for renderer lists")
        .value("ALL", GraphMaterialFilter::All)
        .value("DEFERRED_COMPATIBLE", GraphMaterialFilter::DeferredCompatible)
        .value("DEFERRED_UNSUPPORTED", GraphMaterialFilter::DeferredUnsupported)
        .value("THIN_SHEET", GraphMaterialFilter::ThinSheet)
        .value("ORDINARY_TRANSPARENT", GraphMaterialFilter::OrdinaryTransparent);

    py::enum_<GraphBufferUsage>(m, "GraphBufferUsage", py::arithmetic(), "Allowed uses of a graph buffer")
        .value("NONE", GraphBufferUsage::None)
        .value("STORAGE", GraphBufferUsage::Storage)
        .value("INDIRECT", GraphBufferUsage::Indirect)
        .value("TRANSFER_SOURCE", GraphBufferUsage::TransferSource)
        .value("TRANSFER_DESTINATION", GraphBufferUsage::TransferDestination)
        .value("UNIFORM", GraphBufferUsage::Uniform);

    py::enum_<GraphBufferAccessType>(m, "GraphBufferAccessType", "Per-pass graph buffer access")
        .value("STORAGE_READ", GraphBufferAccessType::StorageRead)
        .value("UNIFORM_READ", GraphBufferAccessType::UniformRead)
        .value("STORAGE_WRITE", GraphBufferAccessType::StorageWrite)
        .value("INDIRECT_READ", GraphBufferAccessType::IndirectRead)
        .value("TRANSFER_READ", GraphBufferAccessType::TransferRead)
        .value("TRANSFER_WRITE", GraphBufferAccessType::TransferWrite);

    py::enum_<GraphTextureRole>(m, "GraphTextureRole", "Lifetime role of a graph texture")
        .value("TRANSIENT", GraphTextureRole::Transient)
        .value("TEMPORAL_READ", GraphTextureRole::TemporalRead)
        .value("TEMPORAL_WRITE", GraphTextureRole::TemporalWrite)
        .value("PERSISTENT", GraphTextureRole::Persistent)
        .value("ASSET", GraphTextureRole::Asset)
        .value("CAMERA", GraphTextureRole::Camera)
        .value("WORLD_RAY_COLOR_PAGES", GraphTextureRole::WorldRayColorPages)
        .value("WORLD_RAY_DATA_PAGES", GraphTextureRole::WorldRayDataPages);

    py::enum_<GraphTextureAttachment>(m, "GraphTextureAttachment")
        .value("COLOR", GraphTextureAttachment::Color)
        .value("DEPTH", GraphTextureAttachment::Depth)
        .value("RESOLVE", GraphTextureAttachment::Resolve);

    py::class_<GraphCommandDesc>(m, "GraphCommandDesc", "Typed command in the Python-defined graph IR")
        .def(py::init<>())
        .def_readwrite("type", &GraphCommandDesc::type)
        .def_readwrite("material_pass", &GraphCommandDesc::shaderTarget)
        .def_readwrite("material_filter", &GraphCommandDesc::materialFilter)
        .def_readwrite("queue_min", &GraphCommandDesc::queueMin)
        .def_readwrite("queue_max", &GraphCommandDesc::queueMax)
        .def_readwrite("sort_mode", &GraphCommandDesc::sortMode)
        .def_readwrite("pass_tag", &GraphCommandDesc::passTag)
        .def_readwrite("override_material", &GraphCommandDesc::overrideMaterial)
        .def_readwrite("renderer_selection", &GraphCommandDesc::rendererSelection)
        .def_readwrite("light_index", &GraphCommandDesc::lightIndex)
        .def_readwrite("screen_ui_list", &GraphCommandDesc::screenUIList)
        .def_readwrite("world_ui_layer_mask", &GraphCommandDesc::worldUILayerMask)
        .def_readwrite("shader_name", &GraphCommandDesc::shaderName)
        .def_readwrite("depth_test", &GraphCommandDesc::depthTest)
        .def_readwrite("depth_write", &GraphCommandDesc::depthWrite)
        .def_readwrite("depth_compare", &GraphCommandDesc::depthCompare)
        .def_readwrite("alpha_blend", &GraphCommandDesc::alphaBlend)
        .def_readwrite("parameter_block", &GraphCommandDesc::parameterBlock)
        .def_readwrite("parameters", &GraphCommandDesc::parameters)
        .def_readwrite("input_bindings", &GraphCommandDesc::inputBindings)
        .def_readwrite("source_resource", &GraphCommandDesc::sourceResource)
        .def_readwrite("destination_resource", &GraphCommandDesc::destinationResource)
        .def_readwrite("copy_bytes", &GraphCommandDesc::copyBytes)
        .def_readwrite("copy_width", &GraphCommandDesc::copyWidth)
        .def_readwrite("copy_height", &GraphCommandDesc::copyHeight)
        .def_readwrite("copy_buffer_offset", &GraphCommandDesc::copyBufferOffset)
        .def_readwrite("work_binding", &GraphCommandDesc::workBinding);

    py::class_<GraphParameterBlockUpdate>(m, "GraphParameterBlockUpdate",
                                          "Revisioned runtime values for a graph parameter block")
        .def(py::init<>())
        .def_readwrite("id", &GraphParameterBlockUpdate::id)
        .def_readwrite("revision", &GraphParameterBlockUpdate::revision)
        .def_readwrite("values", &GraphParameterBlockUpdate::values);

    // GraphTextureDesc struct
    py::class_<GraphTextureDesc>(m, "GraphTextureDesc", "Description of a texture resource in the Python-defined graph")
        .def(py::init<>())
        .def(py::init<const GraphTextureDesc &>())
        .def("__eq__", &GraphTextureDesc::operator==, py::is_operator())
        .def("_bind_render_texture", &GraphTextureDesc::BindRenderTexture, py::arg("owner"), py::arg("attachment"),
             py::arg("require_current_write") = py::none())
        .def_readwrite("name", &GraphTextureDesc::name, "Unique resource name")
        .def_readwrite("format", &GraphTextureDesc::format, "Backend-neutral pixel format")
        .def_property_readonly("is_camera_target", &GraphTextureDesc::IsViewAttachment)
        .def_property_readonly("_is_sampled_import", &GraphTextureDesc::IsSampledImport)
        .def_property_readonly("is_depth", &GraphTextureDesc::IsDepth)
        .def_readwrite("width", &GraphTextureDesc::width, "Custom width (0 = use scene target size)")
        .def_readwrite("height", &GraphTextureDesc::height, "Custom height (0 = use scene target size)")
        .def_readwrite("size_divisor", &GraphTextureDesc::sizeDivisor,
                       "Size divisor relative to scene target (>0: actual = scene / divisor)")
        .def_readwrite("samples", &GraphTextureDesc::samples, "Sample count (0=inherit frame MSAA, otherwise 1/2/4/8)")
        .def_readwrite("role", &GraphTextureDesc::role, "Ownership and lifetime of this texture")
        .def_readwrite("temporal_key", &GraphTextureDesc::temporalKey, "Stable identity shared by a temporal pair")
        .def_readwrite("render_texture", &GraphTextureDesc::renderTexture, "Persistent RenderTexture owner")
        .def_readwrite("attachment", &GraphTextureDesc::attachment)
        .def_readwrite("require_current_write", &GraphTextureDesc::requireCurrentWrite)
        .def_readwrite("asset_guid", &GraphTextureDesc::assetGuid, "GUID of a read-only imported Texture asset")
        .def_readwrite("depth", &GraphTextureDesc::depth, "Imported Texture3D depth")
        .def_readwrite("is_volume", &GraphTextureDesc::isVolume, "Whether the imported asset is a Texture3D");

    py::class_<GraphWorkBindingDesc>(m, "GraphWorkBindingDesc")
        .def(py::init<>())
        .def_readwrite("name", &GraphWorkBindingDesc::name)
        .def_readwrite("recipe", &GraphWorkBindingDesc::recipe)
        .def_readwrite("input_bindings", &GraphWorkBindingDesc::inputBindings);

    py::class_<GraphBufferInput, std::shared_ptr<GraphBufferInput>>(m, "_GraphBufferInput")
        .def(py::init<const std::shared_ptr<rhi::ComputeBuffer> &, uint64_t, uint64_t>())
        .def("publish", &GraphBufferInput::Publish)
        .def("_set_requirement", &GraphBufferInput::SetRequirement)
        .def_property_readonly("_result_cycle", &GraphBufferInput::HasResultCycle)
        .def_property_readonly("value", &GraphBufferInput::GetValue);

    py::enum_<GraphFrameBuffer>(m, "_GraphFrameBuffer")
        .value("NONE", GraphFrameBuffer::None)
        .value("VIEW_LIGHTS", GraphFrameBuffer::ViewLights)
        .value("WORLD_SURFACES", GraphFrameBuffer::WorldSurfaces)
        .value("WORLD_RAY_GEOMETRY", GraphFrameBuffer::WorldRayGeometry)
        .value("WORLD_RAY_MESHES", GraphFrameBuffer::WorldRayMeshes)
        .value("WORLD_RAY_LIT_MATERIALS", GraphFrameBuffer::WorldRayLitMaterials)
        .value("WORLD_RAY_TEXTURE_DESCRIPTORS", GraphFrameBuffer::WorldRayTextureDescriptors)
        .value("WORLD_RAY_TAIL_READ", GraphFrameBuffer::WorldRayTailRead)
        .value("WORLD_RAY_TAIL_WRITE", GraphFrameBuffer::WorldRayTailWrite)
        .value("WORLD_RAY_TAIL_REQUESTS", GraphFrameBuffer::WorldRayTailRequests)
        .value("WORLD_RAY_TAIL_CONTROL", GraphFrameBuffer::WorldRayTailControl);

    py::class_<GraphBufferDesc>(m, "GraphBufferDesc", "Description of a buffer resource in the graph")
        .def(py::init<>())
        .def_readwrite("name", &GraphBufferDesc::name)
        .def_readwrite("byte_size", &GraphBufferDesc::byteSize)
        .def_readwrite("byte_offset", &GraphBufferDesc::byteOffset)
        .def_readwrite("usage", &GraphBufferDesc::usage)
        .def_readwrite("compute_buffer", &GraphBufferDesc::computeBuffer)
        .def_readwrite("input", &GraphBufferDesc::input)
        .def_readwrite("frame_buffer", &GraphBufferDesc::frameBuffer)
        .def_readwrite("work_binding", &GraphBufferDesc::workBinding)
        .def_readwrite("work_slot", &GraphBufferDesc::workSlot)
        .def_readwrite("export_layout", &GraphBufferDesc::exportLayout)
        .def_readwrite("export_shape", &GraphBufferDesc::exportShape);

    py::class_<GraphBufferAccessDesc>(m, "GraphBufferAccessDesc", "Typed buffer access declared by a graph pass")
        .def(py::init<>())
        .def_readwrite("resource", &GraphBufferAccessDesc::resource)
        .def_readwrite("type", &GraphBufferAccessDesc::type);

    py::class_<GraphEventDesc>(m, "GraphEventDesc")
        .def(py::init<>())
        .def_readwrite("label", &GraphEventDesc::label)
        .def_readwrite("metadata", &GraphEventDesc::metadata)
        .def_readwrite("buffers", &GraphEventDesc::buffers);

    // GraphPassDesc struct
    py::class_<GraphPassDesc>(m, "GraphPassDesc", "Description of a single render pass in the Python-defined graph")
        .def(py::init<>())
        .def_readwrite("name", &GraphPassDesc::name, "Pass name (must be unique)")
        .def_readwrite("type", &GraphPassDesc::type, "Pass execution domain")
        .def_readwrite("read_textures", &GraphPassDesc::readTextures, "Names of textures this pass reads")
        .def_readwrite("write_colors", &GraphPassDesc::writeColors,
                       "MRT color outputs: list of (slot, texture_name) pairs")
        .def_readwrite("write_depth", &GraphPassDesc::writeDepth, "Name of depth output texture")
        .def_readwrite("resolve_color", &GraphPassDesc::resolveColor,
                       "Optional single-sample resolve target for color slot 0")
        .def_readwrite("buffer_accesses", &GraphPassDesc::bufferAccesses, "Typed buffer accesses")
        .def_readwrite("side_effect", &GraphPassDesc::sideEffect, "Retain the pass for externally observable work")
        .def_readwrite("ray_tail_batch", &GraphPassDesc::rayTailBatch)
        .def_readwrite("events", &GraphPassDesc::events)
        .def_readwrite("clear_color", &GraphPassDesc::clearColor, "Whether to clear color buffer")
        .def_readwrite("clear_depth", &GraphPassDesc::clearDepth, "Whether to clear depth buffer")
        .def_readwrite("clear_color_r", &GraphPassDesc::clearColorR, "Clear color red")
        .def_readwrite("clear_color_g", &GraphPassDesc::clearColorG, "Clear color green")
        .def_readwrite("clear_color_b", &GraphPassDesc::clearColorB, "Clear color blue")
        .def_readwrite("clear_color_a", &GraphPassDesc::clearColorA, "Clear color alpha")
        .def_readwrite("clear_depth_value", &GraphPassDesc::clearDepthValue, "Depth clear value")
        .def_readwrite("commands", &GraphPassDesc::commands, "Typed commands recorded by this pass");

    // RenderGraphDescription struct
    py::class_<RenderGraphDescription>(m, "RenderGraphDescription", "Complete render graph topology defined by Python")
        .def(py::init<>())
        .def("_render_texture_layouts_current", &RenderGraphDescription::RenderTextureLayoutsCurrent)
        .def_readwrite("name", &RenderGraphDescription::name, "Graph name for debugging")
        .def_readwrite("source_revision", &RenderGraphDescription::sourceRevision,
                       "Monotonic Python graph artifact revision")
        .def_readwrite("textures", &RenderGraphDescription::textures, "All texture resources")
        .def_readwrite("buffers", &RenderGraphDescription::buffers, "All buffer resources")
        .def_readwrite("work_bindings", &RenderGraphDescription::workBindings)
        .def_readwrite("passes", &RenderGraphDescription::passes, "All passes in declaration order")
        .def_readwrite("output_texture", &RenderGraphDescription::outputTexture, "Name of the final output texture")
        .def_readwrite("output_display_encoded", &RenderGraphDescription::outputDisplayEncoded)
        .def_readwrite("linear_output_texture", &RenderGraphDescription::linearOutputTexture)
        .def_readwrite("linear_output_pass_count", &RenderGraphDescription::linearOutputPassCount)
        .def_readwrite("temporal_jitter", &RenderGraphDescription::temporalJitter)
        .def_readwrite("msaa_samples", &RenderGraphDescription::msaaSamples,
                       "MSAA sample count (0=no change, 1=off, 2, 4, 8)");

    py::class_<GraphBufferCapture, std::shared_ptr<GraphBufferCapture>>(m, "GraphBufferCapture")
        .def_property_readonly(
            "ready",
            [](const GraphBufferCapture &capture) { return capture.status == GraphBufferCapture::Status::Accepted; })
        .def_property_readonly(
            "done",
            [](const GraphBufferCapture &capture) { return capture.status != GraphBufferCapture::Status::Pending; })
        .def_readonly("error", &GraphBufferCapture::error)
        .def_readonly("shape", &GraphBufferCapture::shape)
        .def_property_readonly("dtype",
                               [](const GraphBufferCapture &capture) {
                                   static const char *names[] = {"float32", "int32", "uint32", "int64"};
                                   return names[static_cast<size_t>(capture.scalarType)];
                               })
        .def_readonly("byte_offset", &GraphBufferCapture::byteOffset)
        .def_readonly("storage_byte_size", &GraphBufferCapture::storageByteSize)
        .def_readonly("storage_claim", &GraphBufferCapture::storageClaim)
        .def_readonly("storage_buffer", &GraphBufferCapture::storageBuffer)
        .def_readonly("view", &GraphBufferCapture::view)
        .def_readonly("generation", &GraphBufferCapture::generation)
        .def_readonly("occurrence", &GraphBufferCapture::occurrence)
        .def_readonly("name", &GraphBufferCapture::name)
        .def_property_readonly("value",
                               [](const GraphBufferCapture &capture) {
                                   if (capture.status != GraphBufferCapture::Status::Accepted)
                                       throw std::logic_error("Graph buffer capture has no accepted value");
                                   return capture.value;
                               })
        .def("cancel", [](GraphBufferCapture &capture) { capture.Cancel("capture was cancelled"); });

    py::class_<GraphEventInterest, std::shared_ptr<GraphEventInterest>>(m, "_GraphEventInterest")
        .def_readwrite("active", &GraphEventInterest::active);
    py::class_<GraphEventViewSample, std::shared_ptr<GraphEventViewSample>>(m, "_GraphEventViewSample")
        .def_property_readonly(
            "extent", [](const GraphEventViewSample &sample) { return py::make_tuple(sample.width, sample.height); })
        .def_property_readonly(
            "jitter_ndc",
            [](const GraphEventViewSample &sample) { return py::make_tuple(sample.jitterNdc.x, sample.jitterNdc.y); })
        .def_property_readonly("view_matrix",
                               [](const GraphEventViewSample &sample) { return binding::Matrix4ToPython(sample.view); })
        .def_property_readonly(
            "projection_matrix",
            [](const GraphEventViewSample &sample) { return binding::Matrix4ToPython(sample.projection); })
        .def_property_readonly(
            "unjittered_projection_matrix",
            [](const GraphEventViewSample &sample) { return binding::Matrix4ToPython(sample.unjitteredProjection); })
        .def_property_readonly("previous_view_projection", [](const GraphEventViewSample &sample) -> py::object {
            if (sample.previousViewProjection)
                return binding::Matrix4ToPython(*sample.previousViewProjection);
            return py::none();
        });
    py::class_<GraphEventRecord>(m, "_GraphEventRecord")
        .def_readonly("label", &GraphEventRecord::label)
        .def_readonly("metadata", &GraphEventRecord::metadata)
        .def_readonly("pass_name", &GraphEventRecord::pass)
        .def_readonly("ordinal", &GraphEventRecord::ordinal)
        .def_readonly("view", &GraphEventRecord::view)
        .def_readonly("generation", &GraphEventRecord::generation)
        .def_readonly("occurrence", &GraphEventRecord::occurrence)
        .def_readonly("history_epoch", &GraphEventRecord::historyEpoch)
        .def_readonly("view_sample", &GraphEventRecord::viewSample)
        .def_readonly("sources", &GraphEventRecord::sources)
        .def_property_readonly("interest", [](const GraphEventRecord &event) { return event.interest.lock(); })
        .def_property_readonly("submission", [](const GraphEventRecord &event) {
            return py::make_tuple(event.submission.device, static_cast<uint32_t>(event.submission.queue),
                                  event.submission.serial);
        });
}
} // namespace infernux
