/**
 * @file CommandBuffer.h
 * @brief Deferred-recording command buffer for the Scriptable Render Pipeline.
 *
 * Provides the deferred command-buffer API used by ScriptableRenderContext.
 *
 * Design: Python calls (e.g. cmd.draw_renderers()) do NOT immediately operate
 * on Vulkan.  Instead, commands are recorded into a std::vector<RenderCommand>.
 * When ScriptableRenderContext::ExecuteCommandBuffer() is called, the commands
 * are translated into real Vulkan operations during the current frame's command
 * buffer recording.
 *
 * Why deferred?
 *   1. Python calls happen before a valid VkCommandBuffer is available.
 *   2. Deferred mode allows C++ to batch-optimize barriers and state changes.
 *   3. Transient render targets can be allocated lazily at execution time.
 */

#pragma once

#include "InxRenderStruct.h"
#include "RendererParameterBlock.h"
#include "rhi/RhiTypes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace infernux
{

class InxMaterial;
class InxMesh;
struct MeshGeometry;

// ============================================================================
// RenderTargetHandle — lightweight opaque identifier
// ============================================================================

/**
 * @brief Opaque handle to a temporary or persistent render target.
 *
 * Created by CommandBuffer::GetTemporaryRT() or
 * ScriptableRenderContext::CreateRenderTarget().
 * Internally maps to a TransientResourcePool slot.
 */
struct RenderTargetHandle
{
    uint32_t id = UINT32_MAX;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return id != UINT32_MAX;
    }

    bool operator==(const RenderTargetHandle &o) const noexcept
    {
        return id == o.id;
    }
    bool operator!=(const RenderTargetHandle &o) const noexcept
    {
        return id != o.id;
    }
};

/// @brief Sentinel representing the "camera target" (the final scene render target).
inline constexpr RenderTargetHandle CAMERA_TARGET_HANDLE{0xFFFFFFFE};

// ============================================================================
// Render Command Types
// ============================================================================

enum class RenderCommandType : uint8_t
{
    GetTemporaryRT,
    ReleaseTemporaryRT,
    SetRenderTarget,
    ClearRenderTarget,
    DrawMesh,
};

// ---- Per-command parameter structs ----

struct GetTemporaryRTParams
{
    uint32_t handleId = UINT32_MAX;
    int width = 0;
    int height = 0;
    rhi::PixelFormat format = rhi::PixelFormat::RGBA8UNorm;
    rhi::SampleCount samples = rhi::SampleCount::One;
};

struct ReleaseTemporaryRTParams
{
    uint32_t handleId = UINT32_MAX;
};

struct SetRenderTargetParams
{
    uint32_t colorHandleId = UINT32_MAX;
    uint32_t depthHandleId = UINT32_MAX;
};

struct ClearRenderTargetParams
{
    bool clearColor = true;
    bool clearDepth = true;
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 1.0f;
    float depth = 1.0f;
};

struct DrawMeshParams
{
    std::shared_ptr<const MeshGeometry> geometry;
    const std::vector<Vertex> *vertices = nullptr;
    const std::vector<uint32_t> *indices = nullptr;
    glm::mat4 worldMatrix{1.0f};
    std::shared_ptr<InxMaterial> material;
    std::shared_ptr<const RendererParameterBlock> parameterBlock;
    AABB worldBounds;
    uint64_t objectId = 0;
    std::string meshGuid;
    uint64_t meshGeneration = 0;
    int submeshIndex = 0;
    int pass = 0;
};

// ---- Variant-based command storage ----

using RenderCommandData = std::variant<GetTemporaryRTParams, ReleaseTemporaryRTParams, SetRenderTargetParams,
                                       ClearRenderTargetParams, DrawMeshParams>;

struct RenderCommand
{
    RenderCommandType type;
    RenderCommandData data;
};

// ============================================================================
// CommandBuffer
// ============================================================================

/**
 * @brief Deferred-recording command buffer (Unity CommandBuffer equivalent).
 *
 * Usage from Python:
 * @code
 *   cmd = CommandBuffer("ForwardRenderer")
 *   scene_rt = cmd.get_temporary_rt(w, h)
 *   cmd.set_render_target(scene_rt)
 *   cmd.clear_render_target(True, True, 0.1, 0.1, 0.1, 1.0)
 *   cmd.draw_renderers(culling, opaque_settings, opaque_filter)
 *   cmd.release_temporary_rt(scene_rt)
 *   context.execute_command_buffer(cmd)
 * @endcode
 */
class CommandBuffer
{
  public:
    explicit CommandBuffer(const std::string &name = "");
    ~CommandBuffer() = default;

    // Movable, non-copyable
    CommandBuffer(CommandBuffer &&) noexcept = default;
    CommandBuffer &operator=(CommandBuffer &&) noexcept = default;
    CommandBuffer(const CommandBuffer &) = delete;
    CommandBuffer &operator=(const CommandBuffer &) = delete;

    // ====================================================================
    // Render Target Management
    // ====================================================================

    /// @brief Allocate a temporary render target.
    /// The actual GPU resource is created at execution time by TransientResourcePool.
    RenderTargetHandle GetTemporaryRT(int width, int height, rhi::PixelFormat format = rhi::PixelFormat::RGBA8UNorm,
                                      rhi::SampleCount samples = rhi::SampleCount::One);

    /// @brief Mark a temporary RT for release (returned to pool at frame end).
    void ReleaseTemporaryRT(RenderTargetHandle handle);

    /// @brief Set the active color render target.
    void SetRenderTarget(RenderTargetHandle colorTarget);

    /// @brief Set active color + depth render targets.
    void SetRenderTarget(RenderTargetHandle colorTarget, RenderTargetHandle depthTarget);

    /// @brief Clear the currently-bound render target.
    void ClearRenderTarget(bool clearColor, bool clearDepth, float r, float g, float b, float a, float depth = 1.0f);

    /// @brief Record one explicit mesh draw and capture parameter values now.
    void DrawMesh(const std::shared_ptr<InxMesh> &mesh, const glm::mat4 &worldMatrix,
                  const std::shared_ptr<InxMaterial> &material, int submeshIndex = 0, int pass = 0,
                  const DrawParameterBlock *parameters = nullptr);

    // ====================================================================
    // Accessors
    // ====================================================================

    /// @brief Get all recorded commands (for execution by SRC).
    [[nodiscard]] const std::vector<RenderCommand> &GetCommands() const noexcept
    {
        return m_commands;
    }

    /// @brief Get the command buffer's name (for debugging / profiling).
    [[nodiscard]] const std::string &GetName() const noexcept
    {
        return m_name;
    }

    /// @brief Get number of recorded commands.
    [[nodiscard]] size_t GetCommandCount() const noexcept
    {
        return m_commands.size();
    }

    /// @brief Discard all recorded commands (reuse the buffer).
    void Clear();

  private:
    std::string m_name;
    std::vector<RenderCommand> m_commands;
    uint32_t m_nextHandleId = 1; // 0 is reserved, 0xFFFFFFFE = camera target
};

} // namespace infernux
