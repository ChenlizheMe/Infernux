/**
 * @file CommandBuffer.cpp
 * @brief Implementation of the deferred-recording CommandBuffer.
 *
 * Each public method simply pushes a RenderCommand variant onto the internal
 * command list.  Actual GPU work is deferred to
 * ScriptableRenderContext::ExecuteCommandBuffer().
 */

#include "CommandBuffer.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <core/log/InxLog.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <function/renderer/shader/ShaderProgram.h>
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

bool PropertyValueIsFinite(const MaterialProperty &property)
{
    const auto finiteVector = [](const auto &value) {
        for (glm::length_t component = 0; component < value.length(); ++component) {
            if (!std::isfinite(value[component]))
                return false;
        }
        return true;
    };
    switch (property.type) {
    case MaterialPropertyType::Float:
        return std::isfinite(std::get<float>(property.value));
    case MaterialPropertyType::Float2:
        return finiteVector(std::get<glm::vec2>(property.value));
    case MaterialPropertyType::Float3:
        return finiteVector(std::get<glm::vec3>(property.value));
    case MaterialPropertyType::Float4:
    case MaterialPropertyType::Color:
        return finiteVector(std::get<glm::vec4>(property.value));
    case MaterialPropertyType::Mat4: {
        const auto &matrix = std::get<glm::mat4>(property.value);
        for (glm::length_t column = 0; column < matrix.length(); ++column) {
            if (!finiteVector(matrix[column]))
                return false;
        }
        return true;
    }
    case MaterialPropertyType::Int:
    case MaterialPropertyType::Texture2D:
        return true;
    case MaterialPropertyType::FloatArray: {
        const auto &values = std::get<std::vector<float>>(property.value);
        return !values.empty() &&
               std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
    }
    case MaterialPropertyType::Float4Array: {
        const auto &values = std::get<std::vector<glm::vec4>>(property.value);
        return !values.empty() && std::all_of(values.begin(), values.end(), finiteVector);
    }
    }
    return false;
}

bool SamePropertyValue(const MaterialProperty &lhs, const MaterialProperty &rhs)
{
    if (lhs.name != rhs.name || lhs.type != rhs.type || lhs.hdr != rhs.hdr || lhs.range != rhs.range)
        return false;
    const auto sameVector = [](const auto &a, const auto &b) {
        for (glm::length_t component = 0; component < a.length(); ++component) {
            if (a[component] != b[component])
                return false;
        }
        return true;
    };
    switch (lhs.type) {
    case MaterialPropertyType::Float:
        return std::get<float>(lhs.value) == std::get<float>(rhs.value);
    case MaterialPropertyType::Float2:
        return sameVector(std::get<glm::vec2>(lhs.value), std::get<glm::vec2>(rhs.value));
    case MaterialPropertyType::Float3:
        return sameVector(std::get<glm::vec3>(lhs.value), std::get<glm::vec3>(rhs.value));
    case MaterialPropertyType::Float4:
    case MaterialPropertyType::Color:
        return sameVector(std::get<glm::vec4>(lhs.value), std::get<glm::vec4>(rhs.value));
    case MaterialPropertyType::Int:
        return std::get<int>(lhs.value) == std::get<int>(rhs.value);
    case MaterialPropertyType::Mat4: {
        const auto &a = std::get<glm::mat4>(lhs.value);
        const auto &b = std::get<glm::mat4>(rhs.value);
        for (glm::length_t column = 0; column < a.length(); ++column) {
            if (!sameVector(a[column], b[column]))
                return false;
        }
        return true;
    }
    case MaterialPropertyType::Texture2D:
        return std::get<std::string>(lhs.value) == std::get<std::string>(rhs.value);
    case MaterialPropertyType::FloatArray:
        return std::get<std::vector<float>>(lhs.value) == std::get<std::vector<float>>(rhs.value);
    case MaterialPropertyType::Float4Array: {
        const auto &a = std::get<std::vector<glm::vec4>>(lhs.value);
        const auto &b = std::get<std::vector<glm::vec4>>(rhs.value);
        return a.size() == b.size() &&
               std::equal(a.begin(), a.end(), b.begin(),
                          [&](const auto &left, const auto &right) { return sameVector(left, right); });
    }
    }
    return false;
}
} // namespace

void DrawParameterBlock::Set(MaterialProperty property)
{
    if (property.name.empty())
        throw std::invalid_argument("draw parameter name cannot be empty");
    if (m_buffers.find(property.name) != m_buffers.end())
        throw std::invalid_argument("draw binding '" + property.name +
                                    "' is already assigned as a storage buffer; remove it before assigning a value");
    if (!PropertyValueIsFinite(property))
        throw std::invalid_argument("draw parameter '" + property.name + "' must contain only finite values");
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

void DrawParameterBlock::SetFloatArray(const std::string &name, const std::vector<float> &values)
{
    Set({name, MaterialPropertyType::FloatArray, values});
}

void DrawParameterBlock::SetVector4Array(const std::string &name, const std::vector<glm::vec4> &values)
{
    Set({name, MaterialPropertyType::Float4Array, values});
}

void DrawParameterBlock::SetTexture(const std::string &name, const std::string &textureGuid)
{
    Set({name, MaterialPropertyType::Texture2D, textureGuid});
}

void DrawParameterBlock::SetBuffer(const std::string &name, std::shared_ptr<rhi::ComputeBuffer> buffer)
{
    if (name.empty())
        throw std::invalid_argument("draw buffer name cannot be empty");
    if (m_properties.find(name) != m_properties.end())
        throw std::invalid_argument("draw binding '" + name +
                                    "' is already assigned as a value; remove it before assigning a storage buffer");
    if (!buffer)
        throw std::invalid_argument("draw buffer '" + name + "' cannot be null");
    m_buffers[name] = std::move(buffer);
}

bool DrawParameterBlock::Remove(const std::string &name)
{
    const bool removedProperty = m_properties.erase(name) != 0;
    const bool removedBuffer = m_buffers.erase(name) != 0;
    return removedProperty || removedBuffer;
}

void DrawParameterBlock::Clear()
{
    m_properties.clear();
    m_buffers.clear();
}

std::shared_ptr<const RendererParameterBlock> DrawParameterBlock::Capture(const InxMaterial &material) const
{
    if (m_properties.empty() && m_buffers.empty())
        return nullptr;

    const ShaderProgram *program = material.GetPassShaderProgram(ShaderCompileTarget::Forward);
    const auto validateBuffer = [program](const std::string &name, const std::shared_ptr<rhi::ComputeBuffer> &buffer) {
        if (!buffer)
            throw std::invalid_argument("draw buffer '" + name + "' cannot be null");
        // A newly-authored material may not have published its first Forward
        // program yet. Preserve the binding and validate it when the renderer
        // resolves the actual pass; never make Start-order determine whether
        // a legitimate material/buffer configuration can be authored.
        if (!program)
            return;
        const auto binding =
            std::find_if(program->GetDescriptorBindings().begin(), program->GetDescriptorBindings().end(),
                         [&](const MergedDescriptorBinding &candidate) {
                             return candidate.set == 0 && candidate.name == name &&
                                    candidate.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                         });
        if (binding == program->GetDescriptorBindings().end())
            throw std::invalid_argument("material shader has no storage buffer named '" + name + "'");
    };

    const auto normalize = [&material](const std::string &name, const MaterialProperty &supplied) {
        const MaterialProperty *declared = material.GetProperty(name);
        if (!declared)
            throw std::invalid_argument("material shader has no parameter named '" + name + "'");
        if (!ParameterTypeMatches(declared->type, supplied.type))
            throw std::invalid_argument("draw parameter '" + name + "' does not match the reflected shader type");
        if (declared->type == MaterialPropertyType::FloatArray &&
            std::get<std::vector<float>>(declared->value).size() != std::get<std::vector<float>>(supplied.value).size())
            throw std::invalid_argument("draw parameter '" + name + "' does not match the reflected array length");
        if (declared->type == MaterialPropertyType::Float4Array &&
            std::get<std::vector<glm::vec4>>(declared->value).size() !=
                std::get<std::vector<glm::vec4>>(supplied.value).size())
            throw std::invalid_argument("draw parameter '" + name + "' does not match the reflected array length");
        MaterialProperty captured = supplied;
        captured.type = declared->type;
        captured.hdr = declared->hdr;
        captured.range = declared->range;
        if (captured.type == MaterialPropertyType::Texture2D)
            captured.value = InxMaterial::RequireTextureGuid(std::get<std::string>(captured.value));
        return captured;
    };

    // Validate against the current shader reflection on every submission. If
    // the normalized payload is unchanged, retain its immutable identity so
    // consecutive explicit draws can still be instanced.
    if (const auto cached = m_cachedPublication.lock();
        cached && cached->properties.size() == m_properties.size() && cached->buffers.size() == m_buffers.size()) {
        bool unchanged = true;
        for (const auto &[name, supplied] : m_properties) {
            const MaterialProperty captured = normalize(name, supplied);
            const auto previous = cached->properties.find(name);
            if (previous == cached->properties.end() || !SamePropertyValue(previous->second, captured)) {
                unchanged = false;
                break;
            }
        }
        if (unchanged) {
            for (const auto &[name, buffer] : m_buffers) {
                validateBuffer(name, buffer);
                const auto previous = cached->buffers.find(name);
                if (previous == cached->buffers.end() || previous->second != buffer) {
                    unchanged = false;
                    break;
                }
            }
        }
        if (unchanged)
            return cached;
    }

    auto result = std::make_shared<RendererParameterBlock>();
    for (const auto &[name, supplied] : m_properties) {
        result->properties.emplace(name, normalize(name, supplied));
    }
    for (const auto &[name, buffer] : m_buffers) {
        validateBuffer(name, buffer);
        result->buffers.emplace(name, buffer);
    }
    result->revision = g_nextExplicitDrawId.fetch_add(1, std::memory_order_relaxed);
    m_cachedPublication = result;
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
    params.meshIndexFormat = mesh->GetIndexFormat();
    params.submeshIndex = submeshIndex;
    params.pass = pass;
    m_commands.push_back({RenderCommandType::DrawMesh, std::move(params)});
}

} // namespace infernux
