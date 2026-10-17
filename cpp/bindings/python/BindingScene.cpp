/**
 * @file BindingScene.cpp
 * @brief Register the Scene Python API in dependency order.
 */

#include "BindingSceneInternal.h"

namespace infernux
{

void RegisterSceneBindings(pybind11::module_ &m)
{
    using namespace scene_binding;
    BindUITransformDependencies(m);
    BindPrimitiveTypes(m);
    BindObjectHandle(m);
    BindComponent(m);
    BindTransform(m);
    BindMeshRenderer(m);
    BindLineRenderer(m);
    BindSkinnedMeshRenderer(m);
    BindSpriteRenderer(m);
    BindLight(m);
    BindPyComponentProxy(m);
    BindCamera(m);
    RegisterSceneComponentCasters();
    BindGameObject(m);
    BindSceneDocumentTypes(m);
    BindScene(m);
    BindSceneManager(m);
}

} // namespace infernux
