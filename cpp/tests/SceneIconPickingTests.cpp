#include <function/renderer/SceneIconPicking.h>

#include <cassert>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

int main()
{
    const glm::vec3 eye(0.0f, 0.0f, -10.0f);
    const glm::mat4 view = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 cameraToWorld = glm::inverse(view);
    glm::mat4 projection = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    projection[1][1] *= -1.0f;

    const auto ordinary =
        infernux::ProjectSceneIcon(glm::vec3(0.0f), cameraToWorld, view, projection, 800, 1.0f, glm::vec2(800.0f));
    assert(ordinary);
    // The light icon has visible diagonal rays near the quad corners. A
    // ray/sphere test misses this region even though it is rendered.
    assert(infernux::SceneIconContains(*ordinary, {420.0f, 420.0f}));
    assert(infernux::SceneIconContains(*ordinary, {423.0f, 400.0f}));
    assert(!infernux::SceneIconContains(*ordinary, {425.0f, 425.0f}));

    const auto highDpi =
        infernux::ProjectSceneIcon(glm::vec3(0.0f), cameraToWorld, view, projection, 800, 1.5f, glm::vec2(800.0f));
    assert(highDpi);
    assert(infernux::SceneIconContains(*highDpi, {432.0f, 432.0f}));

    // The old render target may be stretched by the Scene image widget while
    // a resize is pending. Hit bounds must follow what was drawn on screen.
    const auto stretched =
        infernux::ProjectSceneIcon(glm::vec3(0.0f), cameraToWorld, view, projection, 800, 1.0f, {1600.0f, 400.0f});
    assert(stretched);
    assert(std::abs((stretched->maximum.x - 4.0f) - 840.0f) < 0.01f);
    assert(std::abs((stretched->maximum.y - 4.0f) - 210.0f) < 0.01f);
    assert(infernux::SceneIconContains(*stretched, {838.0f, 208.0f}));

    // A rotated SceneView must use the camera's world-space billboard frame,
    // not a fixed world X/Y frame.  The icon remains centered at the same
    // projected pixel and its hit target follows the rotated view.
    const glm::vec3 rotatedEye(10.0f, 2.0f, -10.0f);
    const glm::mat4 rotatedView = glm::lookAt(rotatedEye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 rotatedCameraToWorld = glm::inverse(rotatedView);
    const auto rotated = infernux::ProjectSceneIcon(glm::vec3(0.0f), rotatedCameraToWorld, rotatedView, projection,
                                                    800, 1.0f, glm::vec2(800.0f));
    assert(rotated);
    assert(infernux::SceneIconContains(*rotated, {400.0f, 400.0f}));

    assert(!infernux::ProjectSceneIcon({0.0f, 0.0f, -20.0f}, cameraToWorld, view, projection, 800, 1.0f,
                                       glm::vec2(800.0f)));
    std::cout << "Scene icon projected picking tests passed\n";
    return 0;
}
