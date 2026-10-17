#pragma once

#include "BindingRegistration.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace infernux
{

class GameObject;
class InxMesh;
class InxResourceMeta;
class Scene;
struct Vertex;

enum class CoordinateSpace
{
    Self = 0,
    World = 1
};

enum class PrimitiveType
{
    Cube,
    Sphere,
    Capsule,
    Cylinder,
    Plane,
    Quad
};

namespace scene_binding
{

using FloatArray = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>;

// Shared creation helpers retain one primitive cache and one implementation of
// model identity resolution across renderer and Scene bindings.
void GetPrimitiveMeshData(PrimitiveType type, const std::vector<Vertex> *&outVertices,
                          const std::vector<uint32_t> *&outIndices, const char *&outDefaultName);
std::shared_ptr<InxMesh> GetBuiltinPrimitiveMeshAsset(const std::string &name);
GameObject *CreatePrimitiveObject(Scene *scene, PrimitiveType type, const std::string &name);
pybind11::list CreatePrimitiveObjectsBatch(Scene *scene, PrimitiveType type, size_t count,
                                           const std::string &namePrefix, bool withColliders);
pybind11::object InstantiateGameObjectsBatch(Scene &scene, GameObject *source, const FloatArray &positions,
                                             const std::optional<FloatArray> &rotations,
                                             const std::optional<FloatArray> &scales, GameObject *parent,
                                             bool instantiateInWorldSpace, bool returnObjects);
std::shared_ptr<const InxResourceMeta> GetModelMeta(const std::string &guid, const std::shared_ptr<InxMesh> &mesh);
std::string GetModelSubresourceId(const std::shared_ptr<const InxResourceMeta> &meta,
                                  const std::vector<std::string> &path);
GameObject *CreateModelObject(Scene *scene, const std::string &guid, const std::string &name);

void BindUITransformDependencies(pybind11::module_ &m);
void BindPrimitiveTypes(pybind11::module_ &m);
void BindObjectHandle(pybind11::module_ &m);
void BindComponent(pybind11::module_ &m);
void BindTransform(pybind11::module_ &m);
void BindMeshRenderer(pybind11::module_ &m);
void BindLineRenderer(pybind11::module_ &m);
void BindSkinnedMeshRenderer(pybind11::module_ &m);
void BindSpriteRenderer(pybind11::module_ &m);
void BindLight(pybind11::module_ &m);
void BindPyComponentProxy(pybind11::module_ &m);
void BindCamera(pybind11::module_ &m);
void RegisterSceneComponentCasters();
void BindGameObject(pybind11::module_ &m);
void BindSceneDocumentTypes(pybind11::module_ &m);
void BindScene(pybind11::module_ &m);
void BindSceneDocuments(NativeClass<Scene> &cls);
void BindSceneManager(pybind11::module_ &m);

} // namespace scene_binding
} // namespace infernux
