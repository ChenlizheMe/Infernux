#pragma once

#include "core/FoundationApi.h"

#include "AtomicFile.h"
#include "PathWriteScope.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace infernux
{

class FrameCpuStorageClaim;
class DocumentWriteTicket;
struct StagedDocumentWrite;

struct DocumentTransactionEntry
{
    DocumentTransactionEntry(std::string path, std::string content,
                             std::optional<AtomicFileState> expectedFileState = {})
        : path(std::move(path)), content(std::move(content)), expectedFileState(std::move(expectedFileState))
    {
    }

    std::shared_ptr<FrameCpuStorageClaim> preclaimedStorage;
    std::string path;
    std::string content;
    // Identity of the exact authored input consumed by this transaction.
    // nullopt is reserved for derived outputs with no author baseline.
    std::optional<AtomicFileState> expectedFileState;
    std::shared_ptr<StagedDocumentWrite> staged;
};

struct DocumentTransactionFileState
{
    std::string path;
    AtomicFileState state;
};

struct DocumentTransactionStats
{
    uint64_t entryCount = 0;
    uint64_t preparedBytes = 0;
    uint64_t journalBytes = 0;
    double prepareMilliseconds = 0.0;
    double serializeMilliseconds = 0.0;
    double journalWriteMilliseconds = 0.0;
    double applyMilliseconds = 0.0;
    std::vector<DocumentTransactionFileState> committedFileStates;
};

/// Durable write-ahead transaction for a set of authoritative documents.
/// Target paths and invalidations must remain inside projectRoot. The journal
/// refers to flushed sibling files using project-relative paths. Recovery
/// publishes remaining siblings or verifies outputs that already moved.
class DocumentTransaction final
{
  public:
    /// Queue a private, journal-compatible sibling without changing the target.
    INFERNUX_FOUNDATION_API static std::shared_ptr<DocumentWriteTicket> Stage(const std::string &projectRoot,
                                                                              DocumentTransactionEntry &entry);
    INFERNUX_FOUNDATION_API static DocumentTransactionStats Commit(const std::string &projectRoot,
                                                                   const std::string &journalPath,
                                                                   std::vector<DocumentTransactionEntry> entries,
                                                                   std::vector<std::string> invalidatedPaths,
                                                                   const PathWriteScope &writeScope = {});

    /// Replay and remove an existing journal. Returns false when no journal exists.
    /// Guarded targets must still match their input or this transaction's exact
    /// completed payload. A later author edit rejects recovery and retains the
    /// journal; resolving that edit belongs to the user's authoring/VCS workflow.
    INFERNUX_FOUNDATION_API static bool Recover(const std::string &projectRoot, const std::string &journalPath,
                                                const PathWriteScope &writeScope = {});
};

} // namespace infernux
