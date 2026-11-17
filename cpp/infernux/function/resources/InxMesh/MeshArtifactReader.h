#pragma once

#include "MeshArtifactReference.h"

#include <platform/filesystem/InxPath.h>

#include <cstdint>
#include <memory>
#include <string>

namespace infernux
{

class InxMesh;

/// Opens and plans on one file handle. Call Read under the worker's admitted
/// IO and workspace grant. A successful read releases the file handle.
class MeshArtifactReader final
{
  public:
    explicit MeshArtifactReader(const MeshArtifactReference &reference);

    [[nodiscard]] uint64_t FileBytes() const noexcept
    {
        return m_revision.data.size;
    }
    /// Includes the cooked decode workspace and the bounded read block.
    [[nodiscard]] uint64_t WorkspaceBytes() const noexcept;
    [[nodiscard]] std::shared_ptr<InxMesh> Read();

  private:
    FileInput m_file;
    const FileRevision m_revision;
    const std::string m_sourceHash;
    uint64_t m_workspace;
    size_t m_blockBytes;
};

} // namespace infernux
