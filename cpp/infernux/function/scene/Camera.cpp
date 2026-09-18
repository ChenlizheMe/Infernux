#include "Camera.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "Scene.h"
#include <algorithm>
#include <cmath>
#include <core/log/InxLog.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/resources/AssetDependencyGraph.h>
#include <limits>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infernux
{

namespace
{
void RequireProjectionMode(int mode)
{
    if (mode < static_cast<int>(CameraProjection::Perspective) ||
        mode > static_cast<int>(CameraProjection::Physical))
        throw std::invalid_argument("Camera.projectionMode is unsupported");
}

void RequireClearFlags(int flags)
{
    if (flags < static_cast<int>(CameraClearFlags::Skybox) || flags > static_cast<int>(CameraClearFlags::DontClear))
        throw std::invalid_argument("Camera.clearFlags is unsupported");
}

void RequireFieldOfView(float value)
{
    if (!std::isfinite(value) || value <= 0.0f || value >= 180.0f)
        throw std::invalid_argument("Camera.fov must be finite and in (0, 180)");
}

void RequirePositive(float value, const char *field)
{
    if (!std::isfinite(value) || value <= 0.0f)
        throw std::invalid_argument(std::string("Camera.") + field + " must be finite and positive");
}

void RequireClipPlanes(float nearClip, float farClip)
{
    if (!std::isfinite(nearClip) || !std::isfinite(farClip) || nearClip <= 0.0f || farClip <= nearClip)
        throw std::invalid_argument("Camera clip planes require finite 0 < near < far");
}

SemanticTypeDescriptor DescribeCamera()
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux.Camera";
    type.readableId = "infernux.component.camera";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "Camera";
    type.runtimeProfiles = {"editor", "player", "headless"};
    const auto add = [&](const char *name, const char *stored, const char *kind, json initial) -> json & {
        type.fields.push_back({std::string("Camera.") + name,
                               std::string("FieldType.") + kind,
                               false,
                               {{"field_id", name},
                                {"serialized_name", stored},
                                {"serialized", true},
                                {"hidden", false},
                                {"nullable", false},
                                {"storage_kind", "native_property"},
                                {"display_name_key", std::string("camera.") + name},
                                {"tooltip", std::string("camera.tooltip.") + name},
                                {"default", std::move(initial)}}});
        return type.fields.back().attributes;
    };
    const auto enumeration = [&](const char *name, const char *stored, const char *enumName,
                                 std::initializer_list<const char *> names, json labels) {
        auto members = json::array();
        for (const auto *member : names)
            members.push_back({{"name", member}, {"value", members.size()}});
        auto &attributes =
            add(name, stored, "ENUM", {{"$type", "enum"}, {"enum_type", enumName}, {"name", *names.begin()}});
        attributes["enum"] = {{"type_id", std::string("native:infernux.") + enumName},
                              {"members", std::move(members)},
                              {"labels", std::move(labels)}};
    };
    enumeration("projection_mode", "projectionMode", "CameraProjection", {"Perspective", "Orthographic", "Physical"},
                {"camera.projection.perspective", "camera.projection.orthographic", "camera.projection.physical"});
    add("field_of_view", "fov", "FLOAT", 60.0)["range"] = {1.0, 179.0};
    add("focal_length", "focalLength", "FLOAT", 50.0)["range"] = {1.0, 1000.0};
    add("sensor_size", "sensorSize", "VEC2", {36.0, 24.0});
    add("lens_shift", "lensShift", "VEC2", {0.0, 0.0});
    add("aspect_ratio", "aspectRatio", "FLOAT", 16.0 / 9.0);
    add("orthographic_size", "orthoSize", "FLOAT", 5.0);
    add("near_clip", "nearClip", "FLOAT", 0.01)["header"] = "camera.section.clipping";
    add("far_clip", "farClip", "FLOAT", 5000.0);
    add("depth", "depth", "FLOAT", 0.0);
    add("culling_mask", "cullingMask", "INT", 0xffffffffu)["range"] = {0u, 0xffffffffu};
    enumeration(
        "clear_flags", "clearFlags", "CameraClearFlags", {"Skybox", "SolidColor", "DepthOnly", "DontClear"},
        {"camera.clear.skybox", "camera.clear.solid_color", "camera.clear.depth_only", "camera.clear.dont_clear"});
    type.fields.back().attributes["header"] = "camera.section.clear";
    add("background_color", "backgroundColor", "COLOR", {0.1, 0.1, 0.1, 1.0});
    add("dithering", "dithering", "BOOL", false);
    add("stop_nans", "stopNaNs", "BOOL", false)["header"] = "camera.section.output";
    auto &target = add("target_texture", "targetTextureGuid", "ASSET",
                       {{"$type", "asset_ref"}, {"asset_type", "RenderTexture"}, {"guid", ""}, {"path_hint", ""}});
    target["asset_type"] = "RenderTexture";
    target["nullable"] = true;
    target["setter_owns_document_shape"] = true;
    return type;
}

const bool registeredCamera = ComponentFactory::Register(
    "Camera", [] { return std::make_unique<Camera>(); }, Camera::ValidateSerializedDocument,
    Camera::GetTypeConstraints(), DescribeCamera);
} // namespace

Camera::~Camera()
{
    if (!m_targetTextureGuid.empty())
        AssetDependencyGraph::Instance().RemoveRuntimeDependency(GetInstanceGuid(), m_targetTextureGuid);
    if (m_targetTexture)
        m_targetTexture->ReleaseDepthAttachment();
    if (GameObject *owner = GetGameObject()) {
        if (Scene *scene = owner->GetScene(); scene && scene->GetMainCamera() == this)
            scene->SetMainCamera(nullptr);
    }
}

void Camera::SetDepth(float depth)
{
    if (!std::isfinite(depth))
        throw std::invalid_argument("Camera.depth must be finite");
    if (m_depth == depth)
        return;

    m_depth = depth;
    if (GameObject *owner = GetGameObject()) {
        if (Scene *scene = owner->GetScene())
            scene->BumpStructureVersion();
    }
}

void Camera::SetProjectionMode(CameraProjection mode)
{
    RequireProjectionMode(static_cast<int>(mode));
    m_projectionMode = mode;
    m_projectionDirty = true;
}

void Camera::SetFieldOfView(float fov)
{
    RequireFieldOfView(fov);
    m_fov = fov;
    m_projectionDirty = true;
}

void Camera::SetFocalLength(float value)
{
    RequirePositive(value, "focalLength");
    m_focalLength = value;
    m_projectionDirty = true;
}

void Camera::SetSensorSize(const glm::vec2 &value)
{
    RequirePositive(value.x, "sensorSize.x");
    RequirePositive(value.y, "sensorSize.y");
    m_sensorSize = value;
    m_projectionDirty = true;
}

void Camera::SetLensShift(const glm::vec2 &value)
{
    if (!std::isfinite(value.x) || !std::isfinite(value.y))
        throw std::invalid_argument("Camera.lensShift must be finite");
    m_lensShift = value;
    m_projectionDirty = true;
}

void Camera::SetAspectRatio(float aspect)
{
    RequirePositive(aspect, "aspectRatio");
    m_aspectRatio = std::max(0.01f, aspect);
    m_projectionDirty = true;
}

void Camera::SetOrthographicSize(float size)
{
    RequirePositive(size, "orthoSize");
    m_orthoSize = size;
    m_projectionDirty = true;
}

void Camera::SetClipPlanes(float nearClip, float farClip)
{
    RequireClipPlanes(nearClip, farClip);
    m_nearClip = nearClip;
    m_farClip = farClip;
    m_projectionDirty = true;
}

void Camera::SetClearFlags(CameraClearFlags flags)
{
    RequireClearFlags(static_cast<int>(flags));
    m_clearFlags = flags;
}

void Camera::SetBackgroundColor(const glm::vec4 &color)
{
    for (unsigned i = 0; i < 4; ++i)
        if (!std::isfinite(color[i]))
            throw std::invalid_argument("Camera.backgroundColor must be finite");
    m_backgroundColor = color;
}

// ============================================================================
// Serialization
// ============================================================================

nlohmann::json Camera::SerializeDocument() const
{
    json j = Component::SerializeDocument();

    j["projectionMode"] = static_cast<int>(m_projectionMode);
    j["fov"] = m_fov;
    j["focalLength"] = m_focalLength;
    j["sensorSize"] = {m_sensorSize.x, m_sensorSize.y};
    j["lensShift"] = {m_lensShift.x, m_lensShift.y};
    j["aspectRatio"] = m_aspectRatio;
    j["orthoSize"] = m_orthoSize;
    j["nearClip"] = m_nearClip;
    j["farClip"] = m_farClip;
    j["depth"] = m_depth;
    j["cullingMask"] = m_cullingMask;
    j["clearFlags"] = static_cast<int>(m_clearFlags);
    j["backgroundColor"] = {m_backgroundColor.r, m_backgroundColor.g, m_backgroundColor.b, m_backgroundColor.a};
    j["dithering"] = m_dithering;
    j["stopNaNs"] = m_stopNaNs;
    j["targetTextureGuid"] = m_targetTextureGuid;

    return j;
}

void Camera::ValidateSerializedDocument(const nlohmann::json &j)
{
    using namespace component_document_validation;
    ValidateComponentDocument(j, "Camera",
                              {"projectionMode", "fov", "aspectRatio", "orthoSize", "nearClip", "farClip", "depth",
                               "cullingMask", "clearFlags", "backgroundColor"},
                              {"focalLength", "sensorSize", "lensShift", "dithering", "stopNaNs", "targetTextureGuid"});
    const int projectionMode = RequireInteger(j, "projectionMode", "Camera");
    const float fov = RequireFiniteFloat(j, "fov", "Camera");
    const float focalLength = j.contains("focalLength") ? RequireFiniteFloat(j, "focalLength", "Camera") : 50.0f;
    const float aspectRatio = RequireFiniteFloat(j, "aspectRatio", "Camera");
    const float orthoSize = RequireFiniteFloat(j, "orthoSize", "Camera");
    const float nearClip = RequireFiniteFloat(j, "nearClip", "Camera");
    const float farClip = RequireFiniteFloat(j, "farClip", "Camera");
    RequireFiniteFloat(j, "depth", "Camera");
    const uint64_t cullingMask = RequireUnsignedInteger(j, "cullingMask", "Camera");
    const int clearFlags = RequireInteger(j, "clearFlags", "Camera");
    RequireFiniteVector(j, "backgroundColor", 4, "Camera");
    if (j.contains("dithering"))
        RequireBoolean(j, "dithering", "Camera");
    if (j.contains("stopNaNs"))
        RequireBoolean(j, "stopNaNs", "Camera");
    if (j.contains("targetTextureGuid"))
        RequireString(j, "targetTextureGuid", "Camera");

    RequireProjectionMode(projectionMode);
    RequireFieldOfView(fov);
    RequirePositive(focalLength, "focalLength");
    if (j.contains("sensorSize"))
        RequireFiniteVector(j, "sensorSize", 2, "Camera");
    if (j.contains("lensShift"))
        RequireFiniteVector(j, "lensShift", 2, "Camera");
    RequirePositive(orthoSize, "orthoSize");
    if (aspectRatio < 0.01f)
        throw std::invalid_argument("Camera.aspectRatio must be at least 0.01");
    RequireClipPlanes(nearClip, farClip);
    if (cullingMask > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Camera.cullingMask exceeds 32 bits");
    RequireClearFlags(clearFlags);
}

bool Camera::DeserializeDocument(const nlohmann::json &j)
{
    try {
        ValidateSerializedDocument(j);
        if (!Component::DeserializeDocument(j))
            return false;

        m_projectionMode = static_cast<CameraProjection>(j["projectionMode"].get<int>());
        m_fov = j["fov"].get<float>();
        m_focalLength = j.value("focalLength", 50.0f);
        if (j.contains("sensorSize"))
            m_sensorSize = glm::vec2(j["sensorSize"][0].get<float>(), j["sensorSize"][1].get<float>());
        if (j.contains("lensShift"))
            m_lensShift = glm::vec2(j["lensShift"][0].get<float>(), j["lensShift"][1].get<float>());
        m_aspectRatio = j["aspectRatio"].get<float>();
        m_orthoSize = j["orthoSize"].get<float>();
        m_nearClip = j["nearClip"].get<float>();
        m_farClip = j["farClip"].get<float>();
        SetDepth(j["depth"].get<float>());
        m_cullingMask = j["cullingMask"].get<uint32_t>();
        m_clearFlags = static_cast<CameraClearFlags>(j["clearFlags"].get<int>());
        const auto &background = j["backgroundColor"];
        m_backgroundColor = glm::vec4(background[0].get<float>(), background[1].get<float>(),
                                      background[2].get<float>(), background[3].get<float>());
        m_dithering = j.value("dithering", false);
        m_stopNaNs = j.value("stopNaNs", false);
        SetTargetTextureGuid(j.value("targetTextureGuid", std::string{}));
        m_projectionDirty = true;

        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Camera::Deserialize failed: ", e.what());
        return false;
    }
}

// ============================================================================
// Matrices
// ============================================================================

glm::mat4 Camera::GetViewMatrix() const
{
    if (m_viewOverride)
        return *m_viewOverride;
    if (!m_gameObject) {
        return glm::mat4{1.0f};
    }

    const Transform *transform = m_gameObject->GetTransform();

    // Use world-space position and orientation so that the camera
    // correctly follows parent transform hierarchy (e.g. camera
    // attached as a child of a moving character).
    glm::vec3 position = transform->GetWorldPosition();
    glm::vec3 forward = transform->GetWorldForward();
    glm::vec3 up = transform->GetWorldUp();

    return glm::lookAt(position, position + forward, up);
}

glm::mat4 Camera::GetCameraToWorldMatrix() const
{
    if (m_viewOverride)
        return m_cameraToWorldOverride;
    if (!m_gameObject)
        return glm::mat4(1.0f);
    const auto *transform = m_gameObject->GetTransform();
    return glm::mat4(glm::vec4(transform->GetWorldRight(), 0.0f), glm::vec4(transform->GetWorldUp(), 0.0f),
                     glm::vec4(transform->GetWorldForward(), 0.0f), glm::vec4(transform->GetWorldPosition(), 1.0f));
}

void Camera::SetViewMatrix(const glm::mat4 &view)
{
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            if (!std::isfinite(view[column][row]))
                throw std::invalid_argument("Camera view matrix must be finite");
    // Perspective belongs to projection_matrix. An affine view has a unique
    // world-space eye and direction basis, shared by rays, lighting and culling.
    if (view[0][3] != 0.0f || view[1][3] != 0.0f || view[2][3] != 0.0f || view[3][3] != 1.0f)
        throw std::invalid_argument("Camera view matrix must be affine (last row 0,0,0,1)");
    const auto precise = glm::dmat4(view);
    const double determinant = glm::determinant(precise);
    if (!std::isfinite(determinant) || determinant == 0.0)
        throw std::invalid_argument("Camera view matrix must be invertible");
    const glm::mat4 inverse(glm::inverse(precise));
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            if (!std::isfinite(inverse[column][row]))
                throw std::invalid_argument("Camera inverse view matrix must fit finite float32 values");
    m_viewOverride = view;
    m_cameraToWorldOverride = inverse;
}

void Camera::ResetViewMatrix()
{
    m_viewOverride.reset();
}

glm::mat4 Camera::GetProjectionMatrix() const
{
    if (m_projectionOverride)
        return *m_projectionOverride;
    if (m_projectionDirty) {
        UpdateProjectionMatrix();
    }
    return m_cachedProjection;
}

void Camera::UpdateProjectionMatrix() const
{
    m_cachedProjection = BuildProjectionMatrix(m_aspectRatio);
    m_projectionDirty = false;
    m_inverseRayProjectionDirty = true;
}

glm::mat4 Camera::BuildProjectionMatrix(float aspect) const
{
    glm::mat4 projection;
    if (m_projectionMode == CameraProjection::Perspective) {
        float fovRad = glm::radians(m_fov);
        projection = glm::perspective(fovRad, aspect, m_nearClip, m_farClip);
    } else if (m_projectionMode == CameraProjection::Orthographic) {
        float halfWidth = m_orthoSize * aspect;
        float halfHeight = m_orthoSize;
        projection = glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, m_nearClip, m_farClip);
    } else {
        // Unity-style physical camera: focal length and sensor width determine
        // the perspective projection. Sensor height is retained for portrait
        // and lens-shift authoring; aspect selects the active gate.
        const float sensorWidth = std::max(m_sensorSize.x, 0.001f);
        const float fovRad = 2.0f * std::atan(sensorWidth / (2.0f * m_focalLength));
        projection = glm::perspective(fovRad, aspect, m_nearClip, m_farClip);
        projection[2][0] += m_lensShift.x * 2.0f;
        projection[2][1] += m_lensShift.y * 2.0f;
    }
    projection[1][1] *= -1.0f;
    return projection;
}

void Camera::SetProjectionMatrix(const glm::mat4 &projection)
{
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            if (!std::isfinite(projection[column][row]))
                throw std::invalid_argument("Camera projection matrix must be finite");
    const double determinant = glm::determinant(glm::dmat4(projection));
    if (!std::isfinite(determinant) || determinant == 0.0)
        throw std::invalid_argument("Camera projection matrix must be invertible");
    m_projectionOverride = projection;
    m_inverseRayProjectionDirty = true;
}

void Camera::ResetProjectionMatrix()
{
    m_projectionOverride.reset();
    m_inverseRayProjectionDirty = true;
}

const glm::dmat4 &Camera::GetInverseRayProjection(float aspect) const
{
    const auto projection = GetProjectionMatrix();
    if (m_inverseRayProjectionDirty || m_rayProjectionAspect != aspect) {
        // Explicit viewport dimensions change the default aspect only; an
        // authored asymmetric/oblique override remains authoritative.
        m_inverseRayProjection = glm::inverse(
            glm::dmat4(!m_projectionOverride && aspect != m_aspectRatio ? BuildProjectionMatrix(aspect) : projection));
        m_rayProjectionAspect = aspect;
        m_inverseRayProjectionDirty = false;
    }
    return m_inverseRayProjection;
}

glm::mat4 Camera::CalculateObliqueMatrix(const glm::vec4 &clipPlane) const
{
    const glm::dvec4 plane(clipPlane);
    for (int i = 0; i < 4; ++i)
        if (!std::isfinite(plane[i]))
            throw std::invalid_argument("Camera oblique clip plane must be finite");
    if (glm::length(glm::dvec3(plane)) == 0.0)
        throw std::invalid_argument("Camera oblique clip plane requires a nonzero normal");
    auto projection = GetProjectionMatrix();
    const auto inverse = glm::inverse(glm::dmat4(projection));
    // Pick the retained far corner in homogeneous clip space. This handles
    // Y-down, off-axis and orthographic projections without GL [-1,1] formulas.
    double maximum = 0.0;
    for (double x : {-1.0, 1.0})
        for (double y : {-1.0, 1.0})
            maximum = std::max(maximum, glm::dot(plane, inverse * glm::dvec4(x, y, 1.0, 1.0)));
    if (maximum <= 0.0)
        throw std::invalid_argument("Camera oblique clip plane excludes the complete far plane");
    const glm::vec4 nearRow(plane / maximum);
    for (int column = 0; column < 4; ++column)
        projection[column][2] = nearRow[column]; // z >= 0, not z >= -w.
    return projection;
}

glm::vec3 Camera::ScreenToWorldPoint(const glm::vec2 &screenPos, float depth) const
{
    if (m_screenWidth == 0 || m_screenHeight == 0) {
        return glm::vec3(0.0f);
    }

    // Normalise screen coords to NDC [-1, 1]
    // Vulkan uses top-left origin → invert Y in projection (already done),
    // so we keep standard NDC conversion here.
    float ndcX = (2.0f * screenPos.x / static_cast<float>(m_screenWidth)) - 1.0f;
    float ndcY = (2.0f * screenPos.y / static_cast<float>(m_screenHeight)) - 1.0f;

    // Vulkan depth range [0, 1] — depth parameter is already in this range
    float ndcZ = depth;

    glm::vec4 clipPos(ndcX, ndcY, ndcZ, 1.0f);

    // Inverse VP (note: our projection already has Vulkan Y-flip baked in)
    glm::mat4 invVP = glm::inverse(GetProjectionMatrix() * GetViewMatrix());
    glm::vec4 worldPos = invVP * clipPos;

    if (std::abs(worldPos.w) < 1e-8f) {
        return glm::vec3(0.0f);
    }

    return glm::vec3(worldPos) / worldPos.w;
}

glm::vec2 Camera::WorldToScreenPoint(const glm::vec3 &worldPos) const
{
    if (m_screenWidth == 0 || m_screenHeight == 0) {
        return glm::vec2(0.0f);
    }

    glm::vec4 clipPos = GetProjectionMatrix() * GetViewMatrix() * glm::vec4(worldPos, 1.0f);

    if (std::abs(clipPos.w) < 1e-8f) {
        return glm::vec2(0.0f);
    }

    // Perspective divide → NDC
    glm::vec3 ndc = glm::vec3(clipPos) / clipPos.w;

    // NDC → top-left screen coordinates (projection already contains Y-flip).
    float screenX = (ndc.x + 1.0f) * 0.5f * static_cast<float>(m_screenWidth);
    float screenY = (ndc.y + 1.0f) * 0.5f * static_cast<float>(m_screenHeight);

    return glm::vec2(screenX, screenY);
}

std::pair<glm::vec3, glm::vec3> Camera::ScreenPointToRay(const glm::vec2 &screenPos) const
{
    return ScreenPointToRay(screenPos, static_cast<float>(m_screenWidth), static_cast<float>(m_screenHeight));
}

std::pair<glm::vec3, glm::vec3> Camera::ScreenPointToRay(const glm::vec2 &screenPos, float viewportWidth,
                                                         float viewportHeight) const
{
    RequirePositive(viewportWidth, "ray viewport width");
    RequirePositive(viewportHeight, "ray viewport height");
    if (!m_gameObject)
        throw std::logic_error("Camera ray requires an attached camera");
    const auto &inverse = GetInverseRayProjection(viewportWidth / viewportHeight);
    const double x = 2.0 * screenPos.x / viewportWidth - 1.0;
    const double y = 2.0 * screenPos.y / viewportHeight - 1.0;
    const glm::dvec4 nearH = inverse * glm::dvec4(x, y, 0.0, 1.0);
    const glm::dvec4 middleH = inverse * glm::dvec4(x, y, 0.5, 1.0);
    if (nearH.w == 0.0 || middleH.w == 0.0)
        throw std::invalid_argument("Camera ray has no finite origin in this projection");
    const auto nearPoint = glm::dvec3(nearH) / nearH.w;
    // Never unproject z=1 to build the ray: far can legitimately be at infinity.
    const auto direction = glm::normalize(glm::dvec3(middleH) / middleH.w - nearPoint);
    const auto cameraToWorld = GetCameraToWorldMatrix();
    return {glm::vec3(cameraToWorld * glm::vec4(glm::vec3(nearPoint), 1.0f)),
            glm::normalize(glm::mat3(cameraToWorld) * glm::vec3(direction))};
}

void Camera::SetTargetTexture(std::shared_ptr<rhi::RenderTexture> target)
{
    const std::string guid = target ? target->GetAssetGuid() : std::string{};
    AssignTargetTexture(std::move(target), guid);
}

void Camera::SetTargetTextureGuid(const std::string &guid)
{
    if (!guid.empty() && guid == m_targetTextureGuid)
        return;
    // Scene loading/cooking records identity only. The renderer owns GPU creation.
    AssignTargetTexture(nullptr, guid);
}

void Camera::AssignTargetTexture(std::shared_ptr<rhi::RenderTexture> target, const std::string &guid)
{
    if (target == m_targetTexture && guid == m_targetTextureGuid)
        return;
    if (target) {
        const auto generation = target->Acquire();
        if (!generation || !generation->color->IsValid() || !generation->depth)
            throw std::invalid_argument("Camera target_texture requires a live RenderTexture with a depth attachment");
    }
    if (target)
        target->RetainDepthAttachment();
    if (m_targetTexture)
        m_targetTexture->ReleaseDepthAttachment();
    if (guid != m_targetTextureGuid) {
        auto &graph = AssetDependencyGraph::Instance();
        if (!m_targetTextureGuid.empty())
            graph.RemoveRuntimeDependency(GetInstanceGuid(), m_targetTextureGuid);
        if (!guid.empty())
            graph.AddRuntimeDependency(GetInstanceGuid(), guid);
        m_targetTextureGuid = guid;
    }
    m_targetTexture = std::move(target);
    InvalidateOutput();
}

void Camera::OnTargetTextureAssetChanged(bool deleted)
{
    if (deleted)
        AssignTargetTexture(nullptr, m_targetTextureGuid);
    // Reimport/Undo reconnects missing targets at the next camera publication.
    // Live targets are reconfigured in place by the resource loader.
    if (!m_targetTexture)
        InvalidateOutput();
}

void Camera::InvalidateOutput()
{
    // Switching screen/offscreen changes which Camera composites screen UI.
    if (GameObject *owner = GetGameObject())
        if (Scene *scene = owner->GetScene())
            scene->BumpStructureVersion();
}

std::unique_ptr<Component> Camera::Clone() const
{
    auto clone = std::make_unique<Camera>();
    clone->m_enabled = m_enabled;
    clone->m_executionOrder = m_executionOrder;
    clone->m_projectionMode = m_projectionMode;
    clone->m_fov = m_fov;
    clone->m_focalLength = m_focalLength;
    clone->m_sensorSize = m_sensorSize;
    clone->m_lensShift = m_lensShift;
    clone->m_aspectRatio = m_aspectRatio;
    clone->m_orthoSize = m_orthoSize;
    clone->m_nearClip = m_nearClip;
    clone->m_farClip = m_farClip;
    clone->m_depth = m_depth;
    clone->m_cullingMask = m_cullingMask;
    clone->AssignTargetTexture(m_targetTexture, m_targetTextureGuid);
    clone->m_projectionOverride = m_projectionOverride;
    clone->m_viewOverride = m_viewOverride;
    clone->m_cameraToWorldOverride = m_cameraToWorldOverride;
    clone->m_invertCulling = m_invertCulling;
    clone->m_clearFlags = m_clearFlags;
    clone->m_backgroundColor = m_backgroundColor;
    clone->m_dithering = m_dithering;
    clone->m_stopNaNs = m_stopNaNs;
    clone->m_screenWidth = m_screenWidth;
    clone->m_screenHeight = m_screenHeight;
    return clone;
}

} // namespace infernux
