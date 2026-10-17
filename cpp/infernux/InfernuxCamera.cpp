/**
 * @file InfernuxCamera.cpp
 * @brief Infernux — Editor camera control methods
 *
 * Split from Infernux.cpp for maintainability.
 * Contains: GetEditorCamera, ProcessSceneViewInput.
 */

#include "Infernux.h"
#include <platform/input/InputManager.h>

namespace infernux
{

// ----------------------------------
// Editor Camera Access
// ----------------------------------

EditorCameraController *Infernux::GetEditorCamera()
{
    if (m_isCleanedUp) {
        return nullptr;
    }
    return &SceneManager::Instance().GetEditorCameraController();
}

// ----------------------------------
// Scene Camera Control
// ----------------------------------

void Infernux::ProcessSceneViewInput(float deltaTime, bool rightMouseDown, bool middleMouseDown, float mouseDeltaX,
                                     float mouseDeltaY, float scrollDelta, bool keyW, bool keyA, bool keyS, bool keyD,
                                     bool keyQ, bool keyE, bool keyShift)
{
    if (m_isCleanedUp) {
        return;
    }

    EditorCameraController &controller = SceneManager::Instance().GetEditorCameraController();

    // Scene drag is consumed natively every engine frame while capture is
    // active. Python still calls this entry point for wheel input and for the
    // non-captured compatibility path, but must not apply a second 60 Hz drag.
    if (InputManager::Instance().IsEditorMouseCaptureActive()) {
        if (scrollDelta != 0.0f)
            controller.OnMouseScroll(scrollDelta);
        return;
    }

    // Handle scroll
    if (scrollDelta != 0) {
        controller.OnMouseScroll(scrollDelta);
    }

    controller.ProcessInput(deltaTime, rightMouseDown, middleMouseDown, mouseDeltaX, mouseDeltaY, keyW, keyA, keyS,
                            keyD, keyQ, keyE, keyShift);
}

} // namespace infernux
