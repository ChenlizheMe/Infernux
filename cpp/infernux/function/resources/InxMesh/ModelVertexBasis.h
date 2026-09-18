#pragma once

struct aiScene;

namespace infernux
{
struct MeshImportSettings;

// Source-local triangle corners, before welding or cache reordering. Both the
// static mesh and skeletal companion consume this same published basis.
void BuildModelVertexBasis(const aiScene &scene, const MeshImportSettings &settings);
}
