#pragma once

#include <platform/filesystem/FileRevision.h>

#include <string>

namespace infernux
{

/// Identity of the primary artifact used to publish a mesh runtime version.
struct MeshArtifactReference
{
    std::string path;
    std::string sourceHash;
    FileRevision revision;
};

} // namespace infernux
