/**
 * @file CommandBuffer.cpp
 * @brief Implementation of the deferred-recording CommandBuffer.
 *
 * Each public method simply pushes a RenderCommand variant onto the internal
 * command list.  Actual GPU work is deferred to
 * ScriptableRenderContext::ExecuteCommandBuffer().
 */

#include "CommandBuffer.h"
#include <atomic>
#include <core/log/InxLog.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <stdexcept>

namespace infernux
{

namespace
{
std::atomic<uint64_t> g_nextExplicitDrawId{1};

bool ParameterTypeMatches(MaterialPropertyType declared, MaterialPropertyType supplied)
{
    return declared == supplied ||
           ((declared == MaterialPropertyType::Color || declared == MaterialPropertyType::Float4) &&
            (supplied == MaterialPropertyType::Color || supplied == MaterialPropertyType::Float4));
}
} // namespace

void DrawParameterBlock::Set(MaterialProperty property)
{
    if (property.name.empty())
        throw std::invalid_argument("draw parameter name cannot be empty");
    m_properties[property.name] = std::move(property);
}

void DrawParameterBlock::SetFloat(const std::string &name, float value)
{
    Set({name, MaterialPropertyType::Float, value});
}

void DrawParameterBlock::SetVector2(const std::string &name, const glm::vec2 &value)
{
    Set({name, MaterialPropertyType::Float2, value});
}

void DrawParameterBlock::SetVector3(const std::string &name, const glm::vec3 &value)
{
    Set({name, MaterialPropertyType::Float3, value});
}

void DrawParameterBlock::SetVector4(const std::string &name, const glm::vec4 &value)
{
    Set({name, MaterialPropertyType::Float4, value});
}

void DrawParameterBlock::SetColor(const std::string &name, const glm::vec4 &value)
{
    Set({name, MaterialPropertyType::Color, value});
}

void DrawParameterBlock::SetInt(const std::string &name, int value)
{
    Set({name, MaterialPropertyType::Int, value});
}

void DrawParameterBlock::SetMatrix(const std::string &name, const glm::mat4 &value)
{
    Set({name, MaterialPropertyType::Mat4, value});
}

void DrawParameterBlock::SetTexture(const std::string &name, const std::string &textureGuid)
{
    Set({name, MaterialPropertyType::Texture2D, textureGuid});
}

bool DrawParameterBlock::Remove(const std::string &name)
{
    return m_properties.erase(name) != 0;
}

void DrawParameterBlock::Clear()
{
    m_properties.clear();
}

std::shared_ptr<const RendererParameterBlock> DrawParameterBlock::Capture(const InxMaterial &material) const
{
    if (m_properties.empty())
        return nullptr;
    auto result = std::make_shared<RendererParameterBlock>();
    for (const auto &[name, supplied] : m_properties) {
        const MaterialProperty *declared = material.GetProperty(name);
        if (!declared)
            throw std::invalid_argument("material shader has no parameter named '" + name + "'");
        if (!ParameterTypeMatches(declared->type, supplied.type))
            throw std::invalid_argument("draw parameter '" + name + "' does not match the reflected shader type");
        MaterialProperty captured = supplied;
        captured.type = declared->type;
        captured.hdr = declared->hdr;
        captured.range = declared->range;
        if (captured.type == MaterialPropertyType::Texture2D)
            captured.value = InxMaterial::RequireTextureGuid(std::get<std::string>(captured.value));
        result->properties.emplace(name, std::move(captured));
    }
    result->revision = g_nextExplicitDrawId.fetch_add(1, std::memory_order_relaxed);
    return result;
}

// ============================================================================
// Construction / Clear
// ============================================================================

CommandBuffer::CommandBuffer(const std::string &name) : m_name(name)
{
    m_commands.reserve(32); // Typical small pipeline: ~10-20 commands
}

void CommandBuffer::Clear()
{
    m_commands.clear();
    // NOTE: m_nextHandleId is intentionally NOT reset.
    // Handle IDs are unique across the lifetime of this CommandBuffer object
    // to prevent aliasing after a clear-then-reuse cycle.
}

// ============================================================================
// Render Target Management
// ============================================================================

RenderTargetHandle CommandBuffer::GetTemporaryRT(int width, int height, rhi::PixelFormat format,
                                                 rhi::SampleCount samples)
{
    if (width <= 0 || height <= 0) {
        INXLOG_WARN("CommandBuffer '", m_name, "': GetTemporaryRT with invalid size (", width, "x", height, ")");
        return RenderTargetHandle{UINT32_MAX};
    }

    RenderTargetHandle handle{m_nextHandleId++};

    GetTemporaryRTParams params;
    params.handleId = handle.id;
    params.width = width;
    params.height = height;
    params.format = format;
    params.samples = samples;

    m_commands.push_back({RenderCommandType::GetTemporaryRT, params});
    return handle;
}

void CommandBuffer::ReleaseTemporaryRT(RenderTargetHandle handle)
{
    if (!handle.IsValid()) {
        INXLOG_WARN("CommandBuffer '", m_name, "': ReleaseTemporaryRT with invalid handle");
        return;
    }

    ReleaseTemporaryRTParams params;
    params.handleId = handle.id;

    m_commands.push_back({RenderCommandType::ReleaseTemporaryRT, params});
}

void CommandBuffer::SetRenderTarget(RenderTargetHandle colorTarget)
{
    SetRenderTargetParams params;
    params.colorHandleId = colorTarget.id;
    params.depthHandleId = UINT32_MAX; // no explicit depth

    m_commands.push_back({RenderCommandType::SetRenderTarget, params});
}

void CommandBuffer::SetRenderTarget(RenderTargetHandle colorTarget, RenderTargetHandle depthTarget)
{
    SetRenderTargetParams params;
    params.colorHandleId = colorTarget.id;
    params.depthHandleId = depthTarget.id;

    m_commands.push_back({RenderCommandType::SetRenderTarget, params});
}

void CommandBuffer::ClearRenderTarget(bool clearColor, bool clearDepth, float r, float g, float b, float a, float depth)
{
    ClearRenderTargetParams params;
    params.clearColor = clearColor;
    params.clearDepth = clearDepth;
    params.r = r;
    params.g = g;
    params.b = b;
    params.a = a;
    params.depth = depth;

    m_commands.push_back({RenderCommandType::ClearRenderTarget, params});
}

void CommandBuffer::DrawMesh(const std::shared_ptr<InxMesh> &mesh, const glm::mat4 &worldMatrix,
                             const std::shared_ptr<InxMaterial> &material, int submeshIndex, int pass,
                             const DrawParameterBlock *parameters)
{
    if (!mesh)
        throw std::invalid_argument("draw_mesh requires a mesh");
    if (!material)
        throw std::invalid_argument("draw_mesh requires a material");
    if (submeshIndex < 0 ||
        (mesh->GetSubMeshCount() == 0 ? submeshIndex != 0
                                      : static_cast<uint32_t>(submeshIndex) >= mesh->GetSubMeshCount()))
        throw std::out_of_range("draw_mesh submesh index does not exist");
    if (pass != 0)
        throw std::out_of_range("draw_mesh currently exposes the material's primary pass only");

    DrawMeshParams params;
    params.geometry = mesh->GetGeometrySnapshot();
    params.vertices = &params.geometry->vertices;
    params.indices = &params.geometry->indices;
    params.worldMatrix = worldMatrix;
    params.material = material;
    params.parameterBlock = parameters ? parameters->Capture(*material) : nullptr;
    params.worldBounds = AABB(params.geometry->boundsMin, params.geometry->boundsMax).Transform(worldMatrix);
    params.objectId = 0xF000000000000000ULL | g_nextExplicitDrawId.fetch_add(1, std::memory_order_relaxed);
    params.meshGuid = mesh->GetGuid();
    params.meshGeneration = mesh->GetGeneration();
    params.submeshIndex = submeshIndex;
    params.pass = pass;
    m_commands.push_back({RenderCommandType::DrawMesh, std::move(params)});
}

} // namespace infernux
