#pragma once

#include "Component.h"
#include "Transform.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <optional>
#include <utility>

namespace infernux
{
namespace rhi
{
class RenderTexture;
}

/**
 * @brief Camera projection mode
 */
enum class CameraProjection
{
    Perspective,
    Orthographic
};

/** Physical camera film/resolution gate matching Unity's Gate Fit modes. */
enum class PhysicalGateFit
{
    None,
    Vertical,
    Horizontal,
    Fill,
    Overscan
};

/** Unity Camera Inspector sensor-size presets. Custom is derived, not serialized. */
enum class CameraSensorType
{
    Film8mm,
    Super8mm,
    Film16mm,
    Super16mm,
    Film35mm2Perf,
    Film35mmAcademy,
    Super35,
    Film35mmTVProjection,
    Film35mmFullAperture,
    Film35mm185Projection,
    Film35mmAnamorphic,
    Film65mmAlexa,
    Film70mm,
    Film70mmImax,
    Custom
};

/**
 * @brief Camera clear flags (Unity URP-style)
 *
 * Controls how a camera clears the render target before rendering.
 */
enum class CameraClearFlags
{
    Skybox,     ///< Clear color+depth, then draw skybox (default)
    SolidColor, ///< Clear color+depth with backgroundColor
    DepthOnly,  ///< Clear depth only, preserve color (for Overlay cameras)
    DontClear   ///< No clearing (accumulative rendering)
};

/**
 * @brief Camera component for viewing the scene.
 *
 * Provides view and projection matrices for rendering.
 * Supports both perspective and orthographic projection.
 */
class Camera : public Component
{
  public:
    Camera() = default;
    ~Camera() override;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "Camera";
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

    // ========================================================================
    // Projection settings
    // ========================================================================

    [[nodiscard]] CameraProjection GetProjectionMode() const
    {
        return m_projectionMode;
    }
    void SetProjectionMode(CameraProjection mode);

    // Perspective settings
    [[nodiscard]] float GetFieldOfView() const
    {
        return m_fov;
    }
    void SetFieldOfView(float fov);

    // Physical camera settings (Unity-compatible photographic model).
    [[nodiscard]] bool GetUsePhysicalProperties() const
    {
        return m_usePhysicalProperties;
    }
    void SetUsePhysicalProperties(bool enabled);
    [[nodiscard]] int GetIso() const
    {
        return m_iso;
    }
    void SetIso(int value);
    [[nodiscard]] float GetShutterSpeed() const
    {
        return m_shutterSpeed;
    }
    void SetShutterSpeed(float value);
    [[nodiscard]] float GetAperture() const
    {
        return m_aperture;
    }
    void SetAperture(float value);
    [[nodiscard]] float GetFocusDistance() const
    {
        return m_focusDistance;
    }
    void SetFocusDistance(float value);
    [[nodiscard]] int GetBladeCount() const
    {
        return m_bladeCount;
    }
    void SetBladeCount(int value);
    [[nodiscard]] glm::vec2 GetCurvature() const
    {
        return m_curvature;
    }
    void SetCurvature(const glm::vec2 &value);
    [[nodiscard]] float GetBarrelClipping() const
    {
        return m_barrelClipping;
    }
    void SetBarrelClipping(float value);
    [[nodiscard]] float GetAnamorphism() const
    {
        return m_anamorphism;
    }
    void SetAnamorphism(float value);
    [[nodiscard]] float GetFocalLength() const
    {
        return m_focalLength;
    }
    void SetFocalLength(float value);
    [[nodiscard]] CameraSensorType GetSensorType() const;
    void SetSensorType(CameraSensorType value);
    [[nodiscard]] glm::vec2 GetSensorSize() const
    {
        return m_sensorSize;
    }
    void SetSensorSize(const glm::vec2 &value);
    [[nodiscard]] glm::vec2 GetLensShift() const
    {
        return m_lensShift;
    }
    void SetLensShift(const glm::vec2 &value);
    [[nodiscard]] PhysicalGateFit GetGateFit() const
    {
        return m_gateFit;
    }
    void SetGateFit(PhysicalGateFit value);

    [[nodiscard]] float GetAspectRatio() const
    {
        return m_aspectRatio;
    }
    void SetAspectRatio(float aspect);

    // Orthographic settings
    [[nodiscard]] float GetOrthographicSize() const
    {
        return m_orthoSize;
    }
    void SetOrthographicSize(float size);

    // Clipping planes
    [[nodiscard]] float GetNearClip() const
    {
        return m_nearClip;
    }
    void SetNearClip(float nearClip)
    {
        SetClipPlanes(nearClip, m_farClip);
    }

    [[nodiscard]] float GetFarClip() const
    {
        return m_farClip;
    }
    void SetFarClip(float farClip)
    {
        SetClipPlanes(m_nearClip, farClip);
    }
    void SetClipPlanes(float nearClip, float farClip);

    // ========================================================================
    // Multi-camera support (depth ordering, layer culling)
    // ========================================================================

    /// @brief Camera rendering depth (lower depth renders first, like Unity)
    [[nodiscard]] float GetDepth() const
    {
        return m_depth;
    }
    void SetDepth(float depth);

    /// @brief Culling mask — which layers this camera renders (bitmask)
    [[nodiscard]] uint32_t GetCullingMask() const
    {
        return m_cullingMask;
    }
    void SetCullingMask(uint32_t mask)
    {
        m_cullingMask = mask;
    }

    /// Imported outputs persist by GUID; anonymous runtime allocations do not.
    /// An unresolved authored target is not a request for screen output.
    [[nodiscard]] const std::shared_ptr<rhi::RenderTexture> &GetTargetTexture() const
    {
        return m_targetTexture;
    }
    void SetTargetTexture(std::shared_ptr<rhi::RenderTexture> target);
    [[nodiscard]] const std::string &GetTargetTextureGuid() const
    {
        return m_targetTextureGuid;
    }
    void SetTargetTextureGuid(const std::string &guid);
    [[nodiscard]] bool HasTargetTexture() const
    {
        return m_targetTexture || !m_targetTextureGuid.empty();
    }
    void OnTargetTextureAssetChanged(bool deleted);

    // ========================================================================
    // Clear flags & background color
    // ========================================================================

    [[nodiscard]] CameraClearFlags GetClearFlags() const
    {
        return m_clearFlags;
    }
    void SetClearFlags(CameraClearFlags flags);

    [[nodiscard]] glm::vec4 GetBackgroundColor() const
    {
        return m_backgroundColor;
    }
    void SetBackgroundColor(const glm::vec4 &color);

    [[nodiscard]] bool GetDithering() const
    {
        return m_dithering;
    }
    void SetDithering(bool enabled)
    {
        m_dithering = enabled;
    }

    [[nodiscard]] bool GetStopNaNs() const
    {
        return m_stopNaNs;
    }
    void SetStopNaNs(bool enabled)
    {
        m_stopNaNs = enabled;
    }

    // ========================================================================
    // Screen dimensions used by ScreenToWorld / WorldToScreen
    // ========================================================================

    [[nodiscard]] uint32_t GetPixelWidth() const
    {
        return m_screenWidth;
    }
    [[nodiscard]] uint32_t GetPixelHeight() const
    {
        return m_screenHeight;
    }
    void SetScreenDimensions(uint32_t width, uint32_t height)
    {
        m_screenWidth = width;
        m_screenHeight = height;
    }

    // ========================================================================
    // Matrices
    // ========================================================================

    /// @brief Get view matrix (inverse of camera transform)
    [[nodiscard]] glm::mat4 GetViewMatrix() const;
    /// Runtime affine world-to-camera override. Does not edit the Transform.
    void SetViewMatrix(const glm::mat4 &view);
    void ResetViewMatrix();
    /// Invalidate accumulated history on the next render of each view of this camera.
    /// Runtime-only: independent of pose, projection and the saved scene document.
    void ResetHistory()
    {
        ++m_temporalHistoryRevision;
    }
    [[nodiscard]] uint64_t GetTemporalHistoryRevision() const
    {
        return m_temporalHistoryRevision;
    }
    [[nodiscard]] bool HasCustomViewMatrix() const
    {
        return m_viewOverride.has_value();
    }
    [[nodiscard]] glm::mat4 GetCameraToWorldMatrix() const;
    /// Camera-local raster state, independent of light-space shadow passes.
    /// A reflection consumer toggles this when reflecting its source view.
    [[nodiscard]] bool GetInvertCulling() const
    {
        return m_invertCulling;
    }
    void SetInvertCulling(bool invert)
    {
        m_invertCulling = invert;
    }

    /// @brief Get projection matrix
    [[nodiscard]] glm::mat4 GetProjectionMatrix() const;
    /// Runtime override in engine clip space: left-handed, depth [0,1], Y down.
    /// Authoring FOV/clip/aspect settings resume when the override is reset.
    void SetProjectionMatrix(const glm::mat4 &projection);
    void ResetProjectionMatrix();
    [[nodiscard]] bool HasCustomProjectionMatrix() const
    {
        return m_projectionOverride.has_value();
    }
    /// Return, without applying, a projection whose near plane is clipPlane in
    /// camera space. The positive half-space is retained.
    [[nodiscard]] glm::mat4 CalculateObliqueMatrix(const glm::vec4 &clipPlane) const;

    /// @brief Get view-projection matrix
    [[nodiscard]] glm::mat4 GetViewProjectionMatrix() const
    {
        return GetProjectionMatrix() * GetViewMatrix();
    }

    // ========================================================================
    // Utility
    // ========================================================================

    /// @brief Convert screen coordinates to world ray
    [[nodiscard]] glm::vec3 ScreenToWorldPoint(const glm::vec2 &screenPos, float depth = 0.0f) const;

    /// @brief Convert world position to screen coordinates
    [[nodiscard]] glm::vec2 WorldToScreenPoint(const glm::vec3 &worldPos) const;

    /// @brief Build a ray from screen coordinates (Unity: Camera.ScreenPointToRay).
    ///        Returns (origin, direction) as a pair of vec3.
    [[nodiscard]] std::pair<glm::vec3, glm::vec3> ScreenPointToRay(const glm::vec2 &screenPos) const;

    /// @brief Overload that uses explicit viewport dimensions instead of m_screenWidth/m_screenHeight.
    ///        Used by the editor scene picker where the viewport may differ from the camera's render target.
    [[nodiscard]] std::pair<glm::vec3, glm::vec3> ScreenPointToRay(const glm::vec2 &screenPos, float viewportWidth,
                                                                   float viewportHeight) const;

  private:
    void AssignTargetTexture(std::shared_ptr<rhi::RenderTexture> target, const std::string &guid);
    void InvalidateOutput();
    void UpdateProjectionMatrix() const;
    [[nodiscard]] glm::mat4 BuildProjectionMatrix(float aspect) const;
    [[nodiscard]] const glm::dmat4 &GetInverseRayProjection(float aspect) const;

    CameraProjection m_projectionMode = CameraProjection::Perspective;

    // Perspective
    float m_fov = 60.0f; // Field of view in degrees
    float m_aspectRatio = 16.0f / 9.0f;
    bool m_usePhysicalProperties = false;
    int m_iso = 200;
    float m_shutterSpeed = 0.005f; // seconds (1/200)
    float m_aperture = 16.0f;      // f-stop
    float m_focusDistance = 10.0f;
    float m_focalLength = 50.0f; // millimetres
    int m_bladeCount = 5;
    glm::vec2 m_curvature{2.0f, 11.0f};
    float m_barrelClipping = 0.25f;
    float m_anamorphism = 0.0f;
    glm::vec2 m_sensorSize{36.0f, 24.0f}; // millimetres (width, height)
    glm::vec2 m_lensShift{0.0f, 0.0f};    // normalized sensor offsets
    PhysicalGateFit m_gateFit = PhysicalGateFit::Horizontal;

    // Orthographic
    float m_orthoSize = 5.0f; // Half-height of the view

    // Clipping - use large range for editor camera
    float m_nearClip = 0.01f;
    float m_farClip = 5000.0f;

    // Multi-camera: depth ordering (lower = rendered first)
    float m_depth = 0.0f;

    // Layer culling mask (all layers by default)
    uint32_t m_cullingMask = 0xFFFFFFFF;
    std::shared_ptr<rhi::RenderTexture> m_targetTexture;
    std::string m_targetTextureGuid;

    // Clear flags
    CameraClearFlags m_clearFlags = CameraClearFlags::Skybox;
    glm::vec4 m_backgroundColor{0.1f, 0.1f, 0.1f, 1.0f};
    bool m_dithering = false;
    bool m_stopNaNs = false;

    // Screen dimensions updated by InxRenderer
    uint32_t m_screenWidth = 1920;
    uint32_t m_screenHeight = 1080;

    // Cached projection matrix
    mutable glm::mat4 m_cachedProjection{1.0f};
    mutable bool m_projectionDirty = true;
    std::optional<glm::mat4> m_projectionOverride;
    std::optional<glm::mat4> m_viewOverride;
    glm::mat4 m_cameraToWorldOverride{1.0f};
    bool m_invertCulling = false;
    uint64_t m_temporalHistoryRevision = 0;
    mutable glm::dmat4 m_inverseRayProjection{1.0};
    mutable float m_rayProjectionAspect = 0.0f;
    mutable bool m_inverseRayProjectionDirty = true;
};

} // namespace infernux
