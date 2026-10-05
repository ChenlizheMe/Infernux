#pragma once

#include "AtomicFile.h"

#include <cstdint>
#include <string>
#include <vector>

namespace infernux
{

struct DocumentTransactionEntry
{
    std::string path;
    std::string content;
    // Identity of the exact authored input consumed by this transaction.
    // nullopt is reserved for derived outputs with no author baseline.
    std::optional<AtomicFileState> expectedFileState;
};

struct DocumentTransactionFileState
{
    std::string path;
    uint64_t size = 0;
    int64_t modifiedNs = 0;
};

struct DocumentTransactionStats
{
    uint64_t entryCount = 0;
    uint64_t uncompressedBytes = 0;
    uint64_t journalBytes = 0;
    double serializeMilliseconds = 0.0;
    double journalWriteMilliseconds = 0.0;
    double applyMilliseconds = 0.0;
    std::vector<DocumentTransactionFileState> committedFileStates;
};

/// Durable write-ahead transaction for a set of authoritative documents.
/// Target paths and invalidations must remain inside projectRoot. The journal
/// stores project-relative paths so recovery remains valid after moving a project.
class DocumentTransaction final
{
  public:
    static DocumentTransactionStats Commit(const std::string &projectRoot, const std::string &journalPath,
                                           std::vector<DocumentTransactionEntry> entries,
                                           std::vector<std::string> invalidatedPaths);

    /// Replay and remove an existing journal. Returns false when no journal exists.
    /// Guarded targets must still match their input or this transaction's exact
    /// completed payload. A later author edit rejects recovery and retains the
    /// journal; resolving that edit belongs to the user's authoring/VCS workflow.
    static bool Recover(const std::string &projectRoot, const std::string &journalPath);
};

} // namespace infernux
