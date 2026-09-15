#include "EditorTools.h"
#include <algorithm>
#include <core/log/InxLog.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/scene/MeshRenderer.h>
#include <function/scene/Scene.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace infernux
{

// ============================================================================
// Axis colors
// ============================================================================

static constexpr glm::vec3 COLOR_X_DEFAULT{0.92f, 0.30f, 0.30f};
static constexpr glm::vec3 COLOR_Y_DEFAULT{0.30f, 0.92f, 0.38f};
static constexpr glm::vec3 COLOR_Z_DEFAULT{0.32f, 0.48f, 0.94f};
static constexpr glm::vec3 COLOR_HIGHLIGHT{1.0f, 0.92f, 0.18f};
// Rect Tool mirrors the editor's neutral text hierarchy and amber interaction
// state.  Its frame stays quiet over scene content, handles carry one step more
// contrast, and only the part under interaction receives the warm highlight.
static constexpr glm::vec3 COLOR_RECT_EDGE{0.55f, 0.55f, 0.55f};
static constexpr glm::vec3 COLOR_RECT_POINT{0.84f, 0.84f, 0.84f};
static constexpr glm::vec3 COLOR_RECT_HOVER{1.0f, 0.718f, 0.302f};
static constexpr glm::vec3 COLOR_RECT_CENTER_HOVER{0.70f, 0.62f, 0.46f};

// ============================================================================
// Construction
// ============================================================================

EditorTools::EditorTools()
{
    RebuildActiveMeshes();
}

// ============================================================================
// SetToolMode — switch modes and rebuild geometry
// ============================================================================

void EditorTools::SetToolMode(ToolMode mode)
{
    if (mode == m_mode)
        return;
    m_mode = mode;
    m_meshesBuilt = false;
    m_highlightedAxis = HandleAxis::None;
    RebuildActiveMeshes();
}

// ============================================================================
// Geometry helpers — cylinder & cone (shared by translate + scale)
// ============================================================================

void EditorTools::BuildCylinder(std::vector<Vertex> &verts, std::vector<uint32_t> &inds, float radius, float length,
                                int segments, const glm::vec3 &color)
{
    uint32_t baseIdx = static_cast<uint32_t>(verts.size());

    for (int i = 0; i <= segments; ++i) {
        float angle = static_cast<float>(i) / static_cast<float>(segments) * 2.0f * static_cast<float>(M_PI);
        float cx = std::cos(angle) * radius;
        float cz = std::sin(angle) * radius;
        glm::vec3 normal = glm::normalize(glm::vec3(cx, 0.0f, cz));
        verts.push_back(Vertex::Create(glm::vec3(cx, 0.0f, cz), normal, glm::vec2(0.0f), color));
        verts.push_back(Vertex::Create(glm::vec3(cx, length, cz), normal, glm::vec2(0.0f), color));
    }

    for (int i = 0; i < segments; ++i) {
        uint32_t bl = baseIdx + static_cast<uint32_t>(i) * 2;
        uint32_t tl = bl + 1;
        uint32_t br = bl + 2;
        uint32_t tr = bl + 3;
        inds.push_back(bl);
        inds.push_back(br);
        inds.push_back(tl);
        inds.push_back(tl);
        inds.push_back(br);
        inds.push_back(tr);
    }

    // Bottom cap
    uint32_t bottomCenter = static_cast<uint32_t>(verts.size());
    verts.push_back(Vertex::Create(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec2(0.0f), color));
    for (int i = 0; i < segments; ++i) {
        uint32_t cur = baseIdx + static_cast<uint32_t>(i) * 2;
        uint32_t nxt = baseIdx + static_cast<uint32_t>(i + 1) * 2;
        inds.push_back(bottomCenter);
        inds.push_back(nxt);
        inds.push_back(cur);
    }

    // Top cap
    uint32_t topCenter = static_cast<uint32_t>(verts.size());
    verts.push_back(Vertex::Create(glm::vec3(0.0f, length, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f), color));
    for (int i = 0; i < segments; ++i) {
        uint32_t cur = baseIdx + static_cast<uint32_t>(i) * 2 + 1;
        uint32_t nxt = baseIdx + static_cast<uint32_t>(i + 1) * 2 + 1;
        inds.push_back(topCenter);
        inds.push_back(cur);
        inds.push_back(nxt);
    }
}

void EditorTools::BuildCone(std::vector<Vertex> &verts, std::vector<uint32_t> &inds, float radius, float height,
                            float baseY, int segments, const glm::vec3 &color)
{
    uint32_t baseIdx = static_cast<uint32_t>(verts.size());

    for (int i = 0; i <= segments; ++i) {
        float angle = static_cast<float>(i) / static_cast<float>(segments) * 2.0f * static_cast<float>(M_PI);
        float cx = std::cos(angle) * radius;
        float cz = std::sin(angle) * radius;
        float slopeLen = std::sqrt(radius * radius + height * height);
        glm::vec3 normal = glm::normalize(
            glm::vec3(cx / radius * height / slopeLen, radius / slopeLen, cz / radius * height / slopeLen));
        verts.push_back(Vertex::Create(glm::vec3(cx, baseY, cz), normal, glm::vec2(0.0f), color));
    }

    uint32_t tipIdx = static_cast<uint32_t>(verts.size());
    verts.push_back(
        Vertex::Create(glm::vec3(0.0f, baseY + height, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f), color));

    for (int i = 0; i < segments; ++i) {
        uint32_t cur = baseIdx + static_cast<uint32_t>(i);
        uint32_t nxt = baseIdx + static_cast<uint32_t>(i + 1);
        inds.push_back(cur);
        inds.push_back(nxt);
        inds.push_back(tipIdx);
    }

    uint32_t capCenter = static_cast<uint32_t>(verts.size());
    verts.push_back(Vertex::Create(glm::vec3(0.0f, baseY, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec2(0.0f), color));
    for (int i = 0; i < segments; ++i) {
        uint32_t cur = baseIdx + static_cast<uint32_t>(i);
        uint32_t nxt = baseIdx + static_cast<uint32_t>(i + 1);
        inds.push_back(capCenter);
        inds.push_back(nxt);
        inds.push_back(cur);
    }
}

// ============================================================================
// Geometry helper — torus (ring) for Rotate tool
// ============================================================================

void EditorTools::BuildTorus(std::vector<Vertex> &verts, std::vector<uint32_t> &inds, float majorRadius,
                             float tubeRadius, int majorSegs, int tubeSegs, const glm::vec3 &color)
{
    // Torus in the XZ plane (Y = up).
    // majorRadius = distance from centre to tube centre.
    // tubeRadius  = radius of the tube cross-section.
    uint32_t baseIdx = static_cast<uint32_t>(verts.size());

    for (int i = 0; i <= majorSegs; ++i) {
        float theta = static_cast<float>(i) / static_cast<float>(majorSegs) * 2.0f * static_cast<float>(M_PI);
        float cosTheta = std::cos(theta);
        float sinTheta = std::sin(theta);

        for (int j = 0; j <= tubeSegs; ++j) {
            float phi = static_cast<float>(j) / static_cast<float>(tubeSegs) * 2.0f * static_cast<float>(M_PI);
            float cosPhi = std::cos(phi);
            float sinPhi = std::sin(phi);

            float x = (majorRadius + tubeRadius * cosPhi) * cosTheta;
            float y = tubeRadius * sinPhi;
            float z = (majorRadius + tubeRadius * cosPhi) * sinTheta;

            glm::vec3 normal = glm::normalize(glm::vec3(cosPhi * cosTheta, sinPhi, cosPhi * sinTheta));

            verts.push_back(Vertex::Create(glm::vec3(x, y, z), normal, glm::vec2(0.0f), color));
        }
    }

    // Indices
    for (int i = 0; i < majorSegs; ++i) {
        for (int j = 0; j < tubeSegs; ++j) {
            uint32_t a = baseIdx + static_cast<uint32_t>(i * (tubeSegs + 1) + j);
            uint32_t b = baseIdx + static_cast<uint32_t>((i + 1) * (tubeSegs + 1) + j);
            uint32_t c = baseIdx + static_cast<uint32_t>((i + 1) * (tubeSegs + 1) + (j + 1));
            uint32_t d = baseIdx + static_cast<uint32_t>(i * (tubeSegs + 1) + (j + 1));
            inds.push_back(a);
            inds.push_back(b);
            inds.push_back(c);
            inds.push_back(a);
            inds.push_back(c);
            inds.push_back(d);
        }
    }
}

// ============================================================================
// Geometry helper — small cube for Scale tool endpoints
// ============================================================================

void EditorTools::BuildCube(std::vector<Vertex> &verts, std::vector<uint32_t> &inds, float halfSize, float centreY,
                            const glm::vec3 &color)
{
    // Axis-aligned cube centred at (0, centreY, 0)
    float lo = centreY - halfSize;
    float hi = centreY + halfSize;

    uint32_t base = static_cast<uint32_t>(verts.size());

    // 8 vertices
    glm::vec3 positions[8] = {
        {-halfSize, lo, -halfSize}, {+halfSize, lo, -halfSize}, {+halfSize, lo, +halfSize}, {-halfSize, lo, +halfSize},
        {-halfSize, hi, -halfSize}, {+halfSize, hi, -halfSize}, {+halfSize, hi, +halfSize}, {-halfSize, hi, +halfSize},
    };

    // 6 faces, 2 triangles each
    // Face normals
    static const glm::vec3 faceNormals[6] = {
        {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0},
    };
    static const int faceIndices[6][4] = {
        {0, 3, 2, 1}, // bottom (-Y)
        {4, 5, 6, 7}, // top (+Y)
        {0, 1, 5, 4}, // front (-Z)
        {2, 3, 7, 6}, // back (+Z)
        {0, 4, 7, 3}, // left (-X)
        {1, 2, 6, 5}, // right (+X)
    };

    for (int f = 0; f < 6; ++f) {
        uint32_t fbase = static_cast<uint32_t>(verts.size());
        for (int v = 0; v < 4; ++v) {
            verts.push_back(Vertex::Create(positions[faceIndices[f][v]], faceNormals[f], glm::vec2(0.0f), color));
        }
        inds.push_back(fbase);
        inds.push_back(fbase + 1);
        inds.push_back(fbase + 2);
        inds.push_back(fbase);
        inds.push_back(fbase + 2);
        inds.push_back(fbase + 3);
    }
}

void EditorTools::BuildPlaneQuad(std::vector<Vertex> &verts, std::vector<uint32_t> &inds, const glm::vec3 &origin,
                                 const glm::vec3 &axisU, const glm::vec3 &axisV, float offset, float size,
                                 const glm::vec3 &color)
{
    const glm::vec3 p0 = origin + axisU * offset + axisV * offset;
    const glm::vec3 p1 = origin + axisU * (offset + size) + axisV * offset;
    const glm::vec3 p2 = origin + axisU * (offset + size) + axisV * (offset + size);
    const glm::vec3 p3 = origin + axisU * offset + axisV * (offset + size);
    const glm::vec3 normal = glm::normalize(glm::cross(axisU, axisV));
    const uint32_t base = static_cast<uint32_t>(verts.size());

    verts.push_back(Vertex::Create(p0, normal, glm::vec2(0.0f, 0.0f), color));
    verts.push_back(Vertex::Create(p1, normal, glm::vec2(1.0f, 0.0f), color));
    verts.push_back(Vertex::Create(p2, normal, glm::vec2(1.0f, 1.0f), color));
    verts.push_back(Vertex::Create(p3, normal, glm::vec2(0.0f, 1.0f), color));

    inds.push_back(base + 0);
    inds.push_back(base + 1);
    inds.push_back(base + 2);
    inds.push_back(base + 0);
    inds.push_back(base + 2);
    inds.push_back(base + 3);

    verts.push_back(Vertex::Create(p0, -normal, glm::vec2(0.0f, 0.0f), color));
    verts.push_back(Vertex::Create(p3, -normal, glm::vec2(0.0f, 1.0f), color));
    verts.push_back(Vertex::Create(p2, -normal, glm::vec2(1.0f, 1.0f), color));
    verts.push_back(Vertex::Create(p1, -normal, glm::vec2(1.0f, 0.0f), color));

    inds.push_back(base + 4);
    inds.push_back(base + 5);
    inds.push_back(base + 6);
    inds.push_back(base + 4);
    inds.push_back(base + 6);
    inds.push_back(base + 7);
}

// ============================================================================
// Build translate handles: three arrows (cylinder + cone), built along +Y
// ============================================================================

void EditorTools::BuildTranslateHandleMeshes()
{
    constexpr float shaftRadius = 0.0175f;
    constexpr float shaftLength = 0.8f;
    constexpr float coneRadius = 0.0525f;
    constexpr float coneHeight = 0.2f;
    constexpr int segments = 12;

    glm::vec3 xColor = (m_highlightedAxis == HandleAxis::X) ? COLOR_HIGHLIGHT : COLOR_X_DEFAULT;
    glm::vec3 yColor = (m_highlightedAxis == HandleAxis::Y) ? COLOR_HIGHLIGHT : COLOR_Y_DEFAULT;
    glm::vec3 zColor = (m_highlightedAxis == HandleAxis::Z) ? COLOR_HIGHLIGHT : COLOR_Z_DEFAULT;
    glm::vec3 xyColor = (m_highlightedAxis == HandleAxis::XY) ? COLOR_HIGHLIGHT : glm::vec3(0.95f, 0.78f, 0.30f);
    glm::vec3 xzColor = (m_highlightedAxis == HandleAxis::XZ) ? COLOR_HIGHLIGHT : glm::vec3(0.92f, 0.42f, 0.42f);
    glm::vec3 yzColor = (m_highlightedAxis == HandleAxis::YZ) ? COLOR_HIGHLIGHT : glm::vec3(0.42f, 0.92f, 0.76f);

    m_arrowXVerts.clear();
    m_arrowXInds.clear();
    BuildCylinder(m_arrowXVerts, m_arrowXInds, shaftRadius, shaftLength, segments, xColor);
    BuildCone(m_arrowXVerts, m_arrowXInds, coneRadius, coneHeight, shaftLength, segments, xColor);

    m_arrowYVerts.clear();
    m_arrowYInds.clear();
    BuildCylinder(m_arrowYVerts, m_arrowYInds, shaftRadius, shaftLength, segments, yColor);
    BuildCone(m_arrowYVerts, m_arrowYInds, coneRadius, coneHeight, shaftLength, segments, yColor);

    m_arrowZVerts.clear();
    m_arrowZInds.clear();
    BuildCylinder(m_arrowZVerts, m_arrowZInds, shaftRadius, shaftLength, segments, zColor);
    BuildCone(m_arrowZVerts, m_arrowZInds, coneRadius, coneHeight, shaftLength, segments, zColor);

    m_planeXYVerts.clear();
    m_planeXYInds.clear();
    BuildPlaneQuad(m_planeXYVerts, m_planeXYInds, glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f),
                   glm::vec3(0.0f, 1.0f, 0.0f), PLANE_OFFSET, PLANE_SIZE, xyColor);

    m_planeXZVerts.clear();
    m_planeXZInds.clear();
    BuildPlaneQuad(m_planeXZVerts, m_planeXZInds, glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f),
                   glm::vec3(0.0f, 0.0f, 1.0f), PLANE_OFFSET, PLANE_SIZE, xzColor);

    m_planeYZVerts.clear();
    m_planeYZInds.clear();
    BuildPlaneQuad(m_planeYZVerts, m_planeYZInds, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
                   glm::vec3(0.0f, 0.0f, 1.0f), PLANE_OFFSET, PLANE_SIZE, yzColor);

    m_meshesBuilt = true;
}

// ============================================================================
// Build rotate handles: three torus rings in their respective planes
// ============================================================================
// Each ring is built in the XZ plane, then GetDrawCalls rotates it into
// the correct axis plane.  X-ring → ring around X axis (rotate 90° Y→Z),
// Y-ring → ring around Y axis (stays in XZ), Z-ring → ring around Z axis.
// ============================================================================

void EditorTools::BuildRotateHandleMeshes()
{
    constexpr float majorRadius = 0.85f; // matches translate arrow length roughly
    constexpr float tubeRadius = 0.0125f;
    constexpr int majorSegs = 48;
    constexpr int tubeSegs = 8;

    glm::vec3 xColor = (m_highlightedAxis == HandleAxis::X) ? COLOR_HIGHLIGHT : COLOR_X_DEFAULT;
    glm::vec3 yColor = (m_highlightedAxis == HandleAxis::Y) ? COLOR_HIGHLIGHT : COLOR_Y_DEFAULT;
    glm::vec3 zColor = (m_highlightedAxis == HandleAxis::Z) ? COLOR_HIGHLIGHT : COLOR_Z_DEFAULT;

    m_arrowXVerts.clear();
    m_arrowXInds.clear();
    BuildTorus(m_arrowXVerts, m_arrowXInds, majorRadius, tubeRadius, majorSegs, tubeSegs, xColor);

    m_arrowYVerts.clear();
    m_arrowYInds.clear();
    BuildTorus(m_arrowYVerts, m_arrowYInds, majorRadius, tubeRadius, majorSegs, tubeSegs, yColor);

    m_arrowZVerts.clear();
    m_arrowZInds.clear();
    BuildTorus(m_arrowZVerts, m_arrowZInds, majorRadius, tubeRadius, majorSegs, tubeSegs, zColor);

    m_meshesBuilt = true;
}

// ============================================================================
// Build scale handles: thin shaft + small cube at the endpoint
// ============================================================================

void EditorTools::BuildScaleHandleMeshes()
{
    constexpr float shaftRadius = 0.0175f;
    constexpr float shaftLength = 0.75f;
    constexpr float cubeHalf = 0.036f;
    constexpr int segments = 12;
    static constexpr glm::vec3 COLOR_CENTER_DEFAULT{0.72f, 0.72f, 0.74f};

    glm::vec3 xColor = (m_highlightedAxis == HandleAxis::X) ? COLOR_HIGHLIGHT : COLOR_X_DEFAULT;
    glm::vec3 yColor = (m_highlightedAxis == HandleAxis::Y) ? COLOR_HIGHLIGHT : COLOR_Y_DEFAULT;
    glm::vec3 zColor = (m_highlightedAxis == HandleAxis::Z) ? COLOR_HIGHLIGHT : COLOR_Z_DEFAULT;
    glm::vec3 xyColor = (m_highlightedAxis == HandleAxis::XY) ? COLOR_HIGHLIGHT : glm::vec3(0.95f, 0.78f, 0.30f);
    glm::vec3 xzColor = (m_highlightedAxis == HandleAxis::XZ) ? COLOR_HIGHLIGHT : glm::vec3(0.92f, 0.42f, 0.42f);
    glm::vec3 yzColor = (m_highlightedAxis == HandleAxis::YZ) ? COLOR_HIGHLIGHT : glm::vec3(0.42f, 0.92f, 0.76f);
    glm::vec3 centerColor = (m_highlightedAxis == HandleAxis::Center) ? COLOR_HIGHLIGHT : COLOR_CENTER_DEFAULT;

    m_arrowXVerts.clear();
    m_arrowXInds.clear();
    BuildCylinder(m_arrowXVerts, m_arrowXInds, shaftRadius, shaftLength, segments, xColor);
    BuildCube(m_arrowXVerts, m_arrowXInds, cubeHalf, shaftLength + cubeHalf, xColor);

    m_arrowYVerts.clear();
    m_arrowYInds.clear();
    BuildCylinder(m_arrowYVerts, m_arrowYInds, shaftRadius, shaftLength, segments, yColor);
    BuildCube(m_arrowYVerts, m_arrowYInds, cubeHalf, shaftLength + cubeHalf, yColor);

    m_arrowZVerts.clear();
    m_arrowZInds.clear();
    BuildCylinder(m_arrowZVerts, m_arrowZInds, shaftRadius, shaftLength, segments, zColor);
    BuildCube(m_arrowZVerts, m_arrowZInds, cubeHalf, shaftLength + cubeHalf, zColor);

    m_planeXYVerts.clear();
    m_planeXYInds.clear();
    BuildPlaneQuad(m_planeXYVerts, m_planeXYInds, glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f),
                   glm::vec3(0.0f, 1.0f, 0.0f), PLANE_OFFSET, PLANE_SIZE, xyColor);

    m_planeXZVerts.clear();
    m_planeXZInds.clear();
    BuildPlaneQuad(m_planeXZVerts, m_planeXZInds, glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f),
                   glm::vec3(0.0f, 0.0f, 1.0f), PLANE_OFFSET, PLANE_SIZE, xzColor);

    m_planeYZVerts.clear();
    m_planeYZInds.clear();
    BuildPlaneQuad(m_planeYZVerts, m_planeYZInds, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
                   glm::vec3(0.0f, 0.0f, 1.0f), PLANE_OFFSET, PLANE_SIZE, yzColor);

    m_centerCubeVerts.clear();
    m_centerCubeInds.clear();
    BuildCube(m_centerCubeVerts, m_centerCubeInds, CENTER_CUBE_HALF, 0.0f, centerColor);

    m_meshesBuilt = true;
}

void EditorTools::BuildRectHandleMeshes()
{
    const bool centerHovered = m_highlightedAxis == HandleAxis::RectCenter;
    for (std::size_t i = 0; i < m_rectHandleVerts.size(); ++i) {
        auto &verts = m_rectHandleVerts[i];
        auto &inds = m_rectHandleInds[i];
        verts.clear();
        inds.clear();
        const auto handle = static_cast<HandleAxis>(static_cast<int>(HandleAxis::RectLeft) + static_cast<int>(i));
        const glm::vec3 color =
            handle == m_highlightedAxis
                ? COLOR_RECT_HOVER
                : (centerHovered ? COLOR_RECT_CENTER_HOVER : (i < 4 ? COLOR_RECT_EDGE : COLOR_RECT_POINT));
        BuildCube(verts, inds, 0.5f, 0.0f, color);
    }
    for (std::size_t i = 0; i < m_rectMidpointVerts.size(); ++i) {
        auto &verts = m_rectMidpointVerts[i];
        auto &inds = m_rectMidpointInds[i];
        verts.clear();
        inds.clear();
        const auto handle = static_cast<HandleAxis>(static_cast<int>(HandleAxis::RectLeft) + static_cast<int>(i));
        const glm::vec3 color = handle == m_highlightedAxis
                                    ? COLOR_RECT_HOVER
                                    : (centerHovered ? COLOR_RECT_CENTER_HOVER : COLOR_RECT_POINT);
        BuildCube(verts, inds, 0.5f, 0.0f, color);
    }
    m_meshesBuilt = true;
}

// ============================================================================
// RebuildActiveMeshes — dispatcher
// ============================================================================

void EditorTools::RebuildActiveMeshes()
{
    switch (m_mode) {
    case ToolMode::Translate:
        BuildTranslateHandleMeshes();
        break;
    case ToolMode::Rotate:
        BuildRotateHandleMeshes();
        break;
    case ToolMode::Scale:
        BuildScaleHandleMeshes();
        break;
    case ToolMode::Rect:
        BuildRectHandleMeshes();
        break;
    default:
        m_meshesBuilt = true;
        break;
    }
    m_meshDirty = true;
}

// ============================================================================

void EditorTools::SetHighlightedAxis(HandleAxis axis)
{
    if (axis == m_highlightedAxis)
        return;
    m_highlightedAxis = axis;
    RebuildActiveMeshes();
}

// ============================================================================
// GetDrawCalls — produce draw calls for active axis/plane handles
// ============================================================================

EditorTools::RectFrame EditorTools::ResolveRectFrame(GameObject *object, const glm::vec3 &cameraPos)
{
    m_rectFrame = {};
    if (!object || !object->IsActiveInHierarchy())
        return m_rectFrame;

    if (m_rectFrameOverrideObjectId == object->GetID() && m_rectFrameOverride.valid) {
        m_rectFrame = m_rectFrameOverride;
        return m_rectFrame;
    }

    Transform *transform = object->GetTransform();
    if (!transform)
        return m_rectFrame;

    MeshRenderer *renderer = object->GetComponent<MeshRenderer>();
    glm::vec3 boundsMin(-0.5f);
    glm::vec3 boundsMax(0.5f);
    if (renderer) {
        boundsMin = renderer->GetLocalBoundsMin();
        boundsMax = renderer->GetLocalBoundsMax();
    }

    std::array<glm::vec3, 3> axes = {
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
    };
    std::array<float, 3> halfExtents{};
    glm::vec3 center(0.0f);

    // A transform-only parent uses the visible bounds of its descendants when
    // they exist.  This is an editor reference range, not a serialized size on
    // Empty/Camera/Light.  A leaf without renderers keeps the explicit unit
    // reference box below.
    bool usedDescendantBounds = false;
    if (!renderer && !object->GetChildren().empty()) {
        std::array<float, 3> projectedMin = {
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
        };
        std::array<float, 3> projectedMax = {
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest(),
        };
        if (m_localMode) {
            const glm::mat4 world = transform->GetWorldMatrix();
            for (int axis = 0; axis < 3; ++axis) {
                const glm::vec3 column(world[axis]);
                const float length = glm::length(column);
                if (length > 1.0e-6f)
                    axes[axis] = column / length;
            }
        }

        std::vector<GameObject *> pending;
        pending.reserve(object->GetChildCount());
        for (const auto &child : object->GetChildren())
            pending.push_back(child.get());
        while (!pending.empty()) {
            GameObject *candidate = pending.back();
            pending.pop_back();
            if (!candidate || !candidate->IsActiveInHierarchy())
                continue;
            if (MeshRenderer *childRenderer = candidate->GetComponent<MeshRenderer>()) {
                glm::vec3 worldMin;
                glm::vec3 worldMax;
                childRenderer->GetWorldBounds(worldMin, worldMax);
                for (int x = 0; x < 2; ++x) {
                    for (int y = 0; y < 2; ++y) {
                        for (int z = 0; z < 2; ++z) {
                            const glm::vec3 point(x ? worldMax.x : worldMin.x, y ? worldMax.y : worldMin.y,
                                                  z ? worldMax.z : worldMin.z);
                            for (int axis = 0; axis < 3; ++axis) {
                                const float projection = glm::dot(point, axes[axis]);
                                projectedMin[axis] = std::min(projectedMin[axis], projection);
                                projectedMax[axis] = std::max(projectedMax[axis], projection);
                            }
                        }
                    }
                }
                usedDescendantBounds = true;
            }
            for (const auto &child : candidate->GetChildren())
                pending.push_back(child.get());
        }

        if (usedDescendantBounds) {
            center = glm::vec3(0.0f);
            for (int axis = 0; axis < 3; ++axis) {
                const float middle = (projectedMin[axis] + projectedMax[axis]) * 0.5f;
                center += axes[axis] * middle;
                halfExtents[axis] = std::max((projectedMax[axis] - projectedMin[axis]) * 0.5f, 0.001f);
            }
        }
    }

    const bool useWorldBounds = !m_localMode || (renderer && renderer->IsVertexBufferWorldSpace());
    if (!usedDescendantBounds && useWorldBounds) {
        glm::vec3 worldMin;
        glm::vec3 worldMax;
        if (renderer) {
            renderer->GetWorldBounds(worldMin, worldMax);
        } else {
            const glm::mat4 world = transform->GetWorldMatrix();
            worldMin = glm::vec3(std::numeric_limits<float>::max());
            worldMax = glm::vec3(std::numeric_limits<float>::lowest());
            for (int x = 0; x < 2; ++x) {
                for (int y = 0; y < 2; ++y) {
                    for (int z = 0; z < 2; ++z) {
                        const glm::vec3 local(x ? boundsMax.x : boundsMin.x, y ? boundsMax.y : boundsMin.y,
                                              z ? boundsMax.z : boundsMin.z);
                        const glm::vec3 point = glm::vec3(world * glm::vec4(local, 1.0f));
                        worldMin = glm::min(worldMin, point);
                        worldMax = glm::max(worldMax, point);
                    }
                }
            }
        }
        center = (worldMin + worldMax) * 0.5f;
        const glm::vec3 extent = glm::max((worldMax - worldMin) * 0.5f, glm::vec3(0.001f));
        halfExtents = {extent.x, extent.y, extent.z};
    } else if (!usedDescendantBounds) {
        const glm::mat4 world = transform->GetWorldMatrix();
        const glm::vec3 localCenter = (boundsMin + boundsMax) * 0.5f;
        const glm::vec3 localExtent = glm::max((boundsMax - boundsMin) * 0.5f, glm::vec3(0.001f));
        center = glm::vec3(world * glm::vec4(localCenter, 1.0f));
        for (int axis = 0; axis < 3; ++axis) {
            const glm::vec3 column(world[axis]);
            const float length = std::max(glm::length(column), 0.001f);
            axes[axis] = column / length;
            halfExtents[axis] = std::max(localExtent[axis] * length, 0.001f);
        }
    }

    glm::vec3 view = cameraPos - center;
    if (glm::dot(view, view) < 1.0e-8f)
        view = glm::vec3(0.0f, 0.0f, 1.0f);
    else
        view = glm::normalize(view);

    constexpr std::array<std::array<int, 2>, 3> planeAxes = {{{0, 1}, {0, 2}, {1, 2}}};
    float bestFacing = -1.0f;
    int bestPlane = 0;
    for (int plane = 0; plane < static_cast<int>(planeAxes.size()); ++plane) {
        const auto &pair = planeAxes[plane];
        const glm::vec3 normal = glm::normalize(glm::cross(axes[pair[0]], axes[pair[1]]));
        const float facing = std::abs(glm::dot(normal, view));
        if (facing > bestFacing) {
            bestFacing = facing;
            bestPlane = plane;
        }
    }

    const int uIndex = planeAxes[bestPlane][0];
    const int vIndex = planeAxes[bestPlane][1];
    glm::vec3 axisU = axes[uIndex];
    glm::vec3 axisV = axes[vIndex];
    if (glm::dot(glm::cross(axisU, axisV), view) < 0.0f)
        axisV = -axisV;

    m_rectFrame.center = center;
    m_rectFrame.axisU = axisU;
    m_rectFrame.axisV = axisV;
    m_rectFrame.halfU = halfExtents[uIndex];
    m_rectFrame.halfV = halfExtents[vIndex];
    m_rectFrame.axisUIndex = uIndex;
    m_rectFrame.axisVIndex = vIndex;
    m_rectFrame.valid = true;
    return m_rectFrame;
}

DrawCallResult EditorTools::GetDrawCalls(std::shared_ptr<InxMaterial> material, uint64_t selectedObjId,
                                         Scene *activeScene, const glm::vec3 &cameraPos)
{
    DrawCallResult result;

    if (m_mode == ToolMode::None || selectedObjId == 0 || !activeScene) {
        return result;
    }

    if (!m_meshesBuilt) {
        RebuildActiveMeshes();
    }

    GameObject *selectedObj = activeScene->FindByID(selectedObjId);
    if (!selectedObj || !selectedObj->IsActiveInHierarchy()) {
        return result;
    }

    Transform *transform = selectedObj->GetTransform();
    if (!transform) {
        return result;
    }

    glm::vec3 objPos = transform->GetPosition();

    if (m_mode == ToolMode::Rect) {
        const RectFrame frame = ResolveRectFrame(selectedObj, cameraPos);
        if (!frame.valid)
            return result;
        const glm::vec3 &center = frame.center;
        const glm::vec3 &axisX = frame.axisU;
        const glm::vec3 &axisY = frame.axisV;
        const float halfWidth = frame.halfU;
        const float halfHeight = frame.halfV;
        const glm::vec3 axisZ = glm::normalize(glm::cross(axisX, axisY));

        glm::mat4 orientation(1.0f);
        orientation[0] = glm::vec4(axisX, 0.0f);
        orientation[1] = glm::vec4(axisY, 0.0f);
        orientation[2] = glm::vec4(axisZ, 0.0f);

        const float distance = glm::length(cameraPos - center);
        const float thickness = std::max(distance * 0.0036f * m_handleSize, 0.008f);
        const float cornerSize = thickness * 2.4f;
        const bool dirty = m_meshDirty;
        m_meshDirty = false;

        const std::array<glm::vec3, 8> offsets = {
            -axisX * halfWidth,
            axisX * halfWidth,
            -axisY * halfHeight,
            axisY * halfHeight,
            -axisX * halfWidth - axisY * halfHeight,
            axisX * halfWidth - axisY * halfHeight,
            -axisX * halfWidth + axisY * halfHeight,
            axisX * halfWidth + axisY * halfHeight,
        };
        const std::array<glm::vec3, 8> sizes = {
            glm::vec3(thickness, halfHeight * 2.0f + thickness, thickness),
            glm::vec3(thickness, halfHeight * 2.0f + thickness, thickness),
            glm::vec3(halfWidth * 2.0f + thickness, thickness, thickness),
            glm::vec3(halfWidth * 2.0f + thickness, thickness, thickness),
            glm::vec3(cornerSize),
            glm::vec3(cornerSize),
            glm::vec3(cornerSize),
            glm::vec3(cornerSize),
        };
        const std::array<uint64_t, 8> objectIds = {
            RECT_LEFT_ID,        RECT_RIGHT_ID,        RECT_BOTTOM_ID,   RECT_TOP_ID,
            RECT_BOTTOM_LEFT_ID, RECT_BOTTOM_RIGHT_ID, RECT_TOP_LEFT_ID, RECT_TOP_RIGHT_ID,
        };

        for (std::size_t i = 0; i < objectIds.size(); ++i) {
            DrawCall dc;
            dc.indexStart = 0;
            dc.indexCount = static_cast<uint32_t>(m_rectHandleInds[i].size());
            dc.worldMatrix = glm::translate(glm::mat4(1.0f), center + offsets[i] + axisZ * thickness * 0.55f) *
                             orientation * glm::scale(glm::mat4(1.0f), sizes[i]);
            dc.material = material;
            dc.objectId = objectIds[i];
            dc.identity = RenderProxyHandle::Synthetic(RenderDomain::EditorTool, dc.objectId).MakeDrawIdentity();
            dc.meshVertices = &m_rectHandleVerts[i];
            dc.meshIndices = &m_rectHandleInds[i];
            dc.forceBufferUpdate = dirty;
            result.drawCalls.push_back(dc);

            if (i < 4) {
                DrawCall midpoint;
                midpoint.indexStart = 0;
                midpoint.indexCount = static_cast<uint32_t>(m_rectMidpointInds[i].size());
                midpoint.worldMatrix =
                    glm::translate(glm::mat4(1.0f), center + offsets[i] + axisZ * thickness * 0.68f) * orientation *
                    glm::scale(glm::mat4(1.0f), glm::vec3(cornerSize * 0.82f));
                midpoint.material = material;
                midpoint.objectId = objectIds[i];
                midpoint.identity =
                    RenderProxyHandle::Synthetic(RenderDomain::EditorTool, EDITOR_TOOL_BASE_ID | (40 + i))
                        .MakeDrawIdentity();
                midpoint.meshVertices = &m_rectMidpointVerts[i];
                midpoint.meshIndices = &m_rectMidpointInds[i];
                midpoint.forceBufferUpdate = dirty;
                result.drawCalls.push_back(midpoint);
            }
        }
        return result;
    }

    float dist = glm::length(cameraPos - objPos);
    float scale = dist * 0.15f * m_handleSize;
    if (scale < 0.01f) {
        scale = 0.01f;
    }

    glm::mat4 baseTransform;
    if (m_localMode) {
        // Local mode: include object's world rotation so gizmo axes align
        // with the object's local coordinate system
        glm::quat worldRot = transform->GetWorldRotation();
        baseTransform = glm::translate(glm::mat4(1.0f), objPos) * glm::mat4_cast(worldRot) *
                        glm::scale(glm::mat4(1.0f), glm::vec3(scale));
    } else {
        baseTransform = glm::translate(glm::mat4(1.0f), objPos) * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
    }

    // ---- Per-mode axis rotations ----
    // Translate & Scale: arrow/shaft built along +Y → rotate to point along each axis.
    // Rotate: torus built in XZ plane (ring around Y) → rotate so each ring
    //         circles its respective axis.
    //
    // Translate/Scale rotations:
    //   X: rotate -90° around Z  (Y → X)
    //   Y: identity
    //   Z: rotate +90° around X  (Y → Z)
    //
    // Rotate rotations:
    //   X-ring: rotate +90° around Z  (XZ ring → YZ ring, circles X axis)
    //   Y-ring: identity              (XZ ring circles Y axis)
    //   Z-ring: rotate +90° around X  (XZ ring → XY ring, circles Z axis)

    glm::mat4 xRotation, yRotation, zRotation;

    if (m_mode == ToolMode::Rotate) {
        xRotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        yRotation = glm::mat4(1.0f);
        zRotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    } else {
        // Translate & Scale — same axis mapping
        xRotation = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        yRotation = glm::mat4(1.0f);
        zRotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    }

    const bool dirty = m_meshDirty;
    m_meshDirty = false;

    // X-axis draw call
    {
        DrawCall dc;
        dc.indexStart = 0;
        dc.indexCount = static_cast<uint32_t>(m_arrowXInds.size());
        dc.worldMatrix = baseTransform * xRotation;
        dc.material = material;
        dc.objectId = X_AXIS_ID;
        dc.identity = RenderProxyHandle::Synthetic(RenderDomain::EditorTool, dc.objectId).MakeDrawIdentity();
        dc.meshVertices = &m_arrowXVerts;
        dc.meshIndices = &m_arrowXInds;
        dc.forceBufferUpdate = dirty;
        result.drawCalls.push_back(dc);
    }

    // Y-axis draw call
    {
        DrawCall dc;
        dc.indexStart = 0;
        dc.indexCount = static_cast<uint32_t>(m_arrowYInds.size());
        dc.worldMatrix = baseTransform * yRotation;
        dc.material = material;
        dc.objectId = Y_AXIS_ID;
        dc.identity = RenderProxyHandle::Synthetic(RenderDomain::EditorTool, dc.objectId).MakeDrawIdentity();
        dc.meshVertices = &m_arrowYVerts;
        dc.meshIndices = &m_arrowYInds;
        dc.forceBufferUpdate = dirty;
        result.drawCalls.push_back(dc);
    }

    // Z-axis draw call
    {
        DrawCall dc;
        dc.indexStart = 0;
        dc.indexCount = static_cast<uint32_t>(m_arrowZInds.size());
        dc.worldMatrix = baseTransform * zRotation;
        dc.material = material;
        dc.objectId = Z_AXIS_ID;
        dc.identity = RenderProxyHandle::Synthetic(RenderDomain::EditorTool, dc.objectId).MakeDrawIdentity();
        dc.meshVertices = &m_arrowZVerts;
        dc.meshIndices = &m_arrowZInds;
        dc.forceBufferUpdate = dirty;
        result.drawCalls.push_back(dc);
    }

    if (m_mode == ToolMode::Translate || m_mode == ToolMode::Scale) {
        DrawCall xyDc;
        xyDc.indexStart = 0;
        xyDc.indexCount = static_cast<uint32_t>(m_planeXYInds.size());
        xyDc.worldMatrix = baseTransform;
        xyDc.material = material;
        xyDc.objectId = XY_PLANE_ID;
        xyDc.identity = RenderProxyHandle::Synthetic(RenderDomain::EditorTool, xyDc.objectId).MakeDrawIdentity();
        xyDc.meshVertices = &m_planeXYVerts;
        xyDc.meshIndices = &m_planeXYInds;
        xyDc.forceBufferUpdate = dirty;
        result.drawCalls.push_back(xyDc);

        DrawCall xzDc;
        xzDc.indexStart = 0;
        xzDc.indexCount = static_cast<uint32_t>(m_planeXZInds.size());
        xzDc.worldMatrix = baseTransform;
        xzDc.material = material;
        xzDc.objectId = XZ_PLANE_ID;
        xzDc.identity = RenderProxyHandle::Synthetic(RenderDomain::EditorTool, xzDc.objectId).MakeDrawIdentity();
        xzDc.meshVertices = &m_planeXZVerts;
        xzDc.meshIndices = &m_planeXZInds;
        xzDc.forceBufferUpdate = dirty;
        result.drawCalls.push_back(xzDc);

        DrawCall yzDc;
        yzDc.indexStart = 0;
        yzDc.indexCount = static_cast<uint32_t>(m_planeYZInds.size());
        yzDc.worldMatrix = baseTransform;
        yzDc.material = material;
        yzDc.objectId = YZ_PLANE_ID;
        yzDc.identity = RenderProxyHandle::Synthetic(RenderDomain::EditorTool, yzDc.objectId).MakeDrawIdentity();
        yzDc.meshVertices = &m_planeYZVerts;
        yzDc.meshIndices = &m_planeYZInds;
        yzDc.forceBufferUpdate = dirty;
        result.drawCalls.push_back(yzDc);
    }

    if (m_mode == ToolMode::Scale && !m_centerCubeInds.empty()) {
        DrawCall centerDc;
        centerDc.indexStart = 0;
        centerDc.indexCount = static_cast<uint32_t>(m_centerCubeInds.size());
        centerDc.worldMatrix = baseTransform;
        centerDc.material = material;
        centerDc.objectId = CENTER_ID;
        centerDc.identity =
            RenderProxyHandle::Synthetic(RenderDomain::EditorTool, centerDc.objectId).MakeDrawIdentity();
        centerDc.meshVertices = &m_centerCubeVerts;
        centerDc.meshIndices = &m_centerCubeInds;
        centerDc.forceBufferUpdate = dirty;
        result.drawCalls.push_back(centerDc);
    }

    auto depthKey = [&cameraPos](const DrawCall &dc) {
        if (!dc.meshVertices || dc.meshVertices->empty())
            return 0.0f;
        glm::vec3 center(0.0f);
        for (const auto &v : *dc.meshVertices)
            center += glm::vec3(dc.worldMatrix * glm::vec4(v.pos, 1.0f));
        center /= static_cast<float>(dc.meshVertices->size());
        const glm::vec3 delta = cameraPos - center;
        return glm::dot(delta, delta);
    };

    // The three rotation rings share one exact pivot and intersect each other.
    // Sorting their vertex-derived centroids lets tiny floating-point changes
    // reverse the winner at those intersections from one frame to the next.
    // Preserve the authored X/Y/Z order for a stable overlay instead.
    if (m_mode != ToolMode::Rotate) {
        std::stable_sort(result.drawCalls.begin(), result.drawCalls.end(),
                         [&](const DrawCall &a, const DrawCall &b) { return depthKey(a) > depthKey(b); });
    }

    return result;
}

} // namespace infernux
