#include <function/renderer/EditorTools.h>
#include <function/scene/GameObject.h>
#include <function/scene/MeshRenderer.h>
#include <function/scene/Scene.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

using namespace infernux;

int main()
{
    Scene scene("EditorTools");
    GameObject *object = scene.CreateGameObject("Selected");
    EditorTools tools;

    tools.SetToolMode(EditorTools::ToolMode::Scale);
    DrawCallResult scale = tools.GetDrawCalls(nullptr, object->GetID(), &scene, glm::vec3(3.0f, 4.0f, -5.0f));
    assert(scale.drawCalls.size() == 7);

    const std::array<uint64_t, 3> planeIds = {
        EditorTools::XY_PLANE_ID,
        EditorTools::XZ_PLANE_ID,
        EditorTools::YZ_PLANE_ID,
    };
    for (uint64_t id : planeIds) {
        const auto found = std::find_if(scale.drawCalls.begin(), scale.drawCalls.end(),
                                        [id](const DrawCall &draw) { return draw.objectId == id; });
        assert(found != scale.drawCalls.end());
        assert(found->indexCount != 0);
    }

    tools.SetToolMode(EditorTools::ToolMode::Rotate);
    for (const glm::vec3 camera : {glm::vec3(3.0f, 4.0f, -5.0f), glm::vec3(-5.0f, 2.0f, 6.0f)}) {
        DrawCallResult rotate = tools.GetDrawCalls(nullptr, object->GetID(), &scene, camera);
        assert(rotate.drawCalls.size() == 3);
        assert(rotate.drawCalls[0].objectId == EditorTools::X_AXIS_ID);
        assert(rotate.drawCalls[1].objectId == EditorTools::Y_AXIS_ID);
        assert(rotate.drawCalls[2].objectId == EditorTools::Z_AXIS_ID);
    }

    EditorTools::RectFrame authored;
    authored.center = glm::vec3(4.0f, 5.0f, 6.0f);
    authored.axisU = glm::vec3(0.0f, 0.0f, 1.0f);
    authored.axisV = glm::vec3(0.0f, 1.0f, 0.0f);
    authored.halfU = 2.5f;
    authored.halfV = 1.25f;
    authored.axisUIndex = 2;
    authored.axisVIndex = 1;
    authored.valid = true;
    tools.SetRectFrameOverride(object->GetID(), authored);
    const auto resolvedAuthored = tools.ResolveRectFrame(object, glm::vec3(3.0f, 4.0f, -5.0f));
    assert(resolvedAuthored.valid);
    assert(resolvedAuthored.center == authored.center);
    assert(resolvedAuthored.axisU == authored.axisU);
    assert(resolvedAuthored.axisV == authored.axisV);
    assert(resolvedAuthored.halfU == authored.halfU);
    assert(resolvedAuthored.halfV == authored.halfV);
    tools.ClearRectFrameOverride();
    const auto resolvedDefault = tools.ResolveRectFrame(object, glm::vec3(3.0f, 4.0f, -5.0f));
    assert(resolvedDefault.valid);
    assert(resolvedDefault.center != authored.center);

    // Empty objects, Cameras and Lights all share the Transform-owned
    // reference box when no renderer publishes visual bounds. Rect must not
    // disappear merely because the object has no MeshRenderer.
    tools.SetToolMode(EditorTools::ToolMode::Rect);
    object->GetTransform()->SetPosition(glm::vec3(1.0f, 2.0f, 3.0f));
    const auto emptyRect = tools.GetDrawCalls(nullptr, object->GetID(), &scene, glm::vec3(3.0f, 4.0f, -5.0f));
    assert(emptyRect.drawCalls.size() == 12);
    const auto emptyFrame = tools.GetRectFrame();
    assert(emptyFrame.valid);
    assert(glm::length(emptyFrame.center - glm::vec3(1.0f, 2.0f, 3.0f)) < 1.0e-5f);
    assert(emptyFrame.halfU > 0.0f);
    assert(emptyFrame.halfV > 0.0f);

    // A transform-only parent frames visible descendants instead of inventing
    // author data on the parent.  The child remains an ordinary renderer.
    GameObject *group = scene.CreateGameObject("Transform Group");
    GameObject *child = scene.CreateGameObject("Visible Child");
    child->SetParent(group, true);
    child->GetTransform()->SetPosition(glm::vec3(7.0f, 2.0f, 3.0f));
    MeshRenderer *childRenderer = child->AddComponent<MeshRenderer>();
    childRenderer->SetLocalBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
    const auto groupRect = tools.GetDrawCalls(nullptr, group->GetID(), &scene, glm::vec3(7.0f, 4.0f, -5.0f));
    assert(groupRect.drawCalls.size() == 12);
    const auto groupFrame = tools.GetRectFrame();
    assert(groupFrame.valid);
    assert(glm::length(groupFrame.center - glm::vec3(7.0f, 2.0f, 3.0f)) < 1.0e-4f);
    assert(groupFrame.halfU >= 0.999f);
    assert(groupFrame.halfV >= 0.999f);

    return 0;
}
