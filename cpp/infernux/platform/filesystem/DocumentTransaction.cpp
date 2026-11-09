#include "DocumentTransaction.h"

#include "DocumentStore.h"
#include "FileFingerprint.h"
#include "InxPath.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace infernux
{
namespace
{
constexpr std::string_view JournalMagic = "INXDTX-STAGED\n";
constexpr std::string_view StagedBasename = ".inx-stage";
constexpr std::string_view StagedNamePrefix = ".inx-stage.inx-tmp.";
constexpr uint64_t MaximumEntryCount = 1'000'000;
constexpr uint64_t MaximumJournalBytes = 2ULL << 30;
using Clock = std::chrono::steady_clock;

double ElapsedMilliseconds(Clock::time_point since)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

void AppendUint64(std::string &out, uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<char>((value >> shift) & 0xff));
}

uint64_t ReadUint64(std::string_view bytes, size_t &cursor)
{
    if (bytes.size() - cursor < sizeof(uint64_t))
        throw std::invalid_argument("document transaction journal is truncated");
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[cursor++])) << shift;
    return value;
}

void AppendField(std::string &out, std::string_view value)
{
    AppendUint64(out, value.size());
    out.append(value);
}

std::string ReadField(std::string_view bytes, size_t &cursor)
{
    const uint64_t size = ReadUint64(bytes, cursor);
    if (size > bytes.size() - cursor)
        throw std::invalid_argument("document transaction journal has an invalid field size");
    std::string value(bytes.substr(cursor, static_cast<size_t>(size)));
    cursor += static_cast<size_t>(size);
    return value;
}

using CanonicalParentCache = std::unordered_map<std::string, std::filesystem::path>;

std::filesystem::path NormalizeRoot(const std::string &projectRoot)
{
    const std::string root = ResolveFilesystemPath(projectRoot);
    if (root.empty())
        throw std::invalid_argument("document transaction project root cannot be empty");
    return ToFsPath(root);
}

std::filesystem::path RequireInsideRoot(const std::filesystem::path &root, const std::string &path,
                                        CanonicalParentCache &parents)
{
    if (path.empty())
        throw std::invalid_argument("document transaction path cannot be empty");
    const auto absolute = ToFsPath(NormalizeFilesystemPathLexically(path));
    const auto lexicalParent = absolute.parent_path();
    auto [parent, inserted] = parents.try_emplace(LexicalFilesystemPathKey(FromFsPath(lexicalParent)));
    if (inserted)
        parent->second = ToFsPath(ResolveFilesystemPath(FromFsPath(lexicalParent)));
    const auto candidate = parent->second / absolute.filename();
    if (!IsFilesystemPathKeyWithin(LexicalFilesystemPathKey(FromFsPath(candidate)),
                                   LexicalFilesystemPathKey(FromFsPath(root)), false))
        throw std::invalid_argument("document transaction path escapes project root: " + path);
    return candidate;
}

std::string ToRelativePath(const std::filesystem::path &root, const std::string &path, CanonicalParentCache &parents)
{
    std::string relative;
    if (!TryMakeRelativeFilesystemPath(FromFsPath(RequireInsideRoot(root, path, parents)), FromFsPath(root), relative))
        throw std::invalid_argument("invalid project-relative transaction path: " + path);
    return relative;
}

struct JournalPath
{
    std::string relative;
    std::string absolute;
};

struct JournalEntry
{
    JournalPath target;
    AtomicPreparedFile prepared;
    std::optional<AtomicFileState> expected;
    std::optional<AtomicFileState> published;
    std::string derivedPublishPath;
    bool ownsDerivedPublishPath = false;
};

struct Journal
{
    std::vector<JournalEntry> entries;
    std::vector<JournalPath> invalidated;
};

void ResolveJournalPaths(const std::filesystem::path &root, const std::string &journalPath, Journal &journal,
                         CanonicalParentCache &parents)
{
    std::unordered_set<std::string> paths{LexicalFilesystemPathKey(journalPath)};
    const auto registerPath = [&](const std::string &path) {
        if (!paths.insert(LexicalFilesystemPathKey(path)).second)
            throw std::invalid_argument("document transaction contains a duplicate target path: " + path);
    };
    const auto resolve = [&](JournalPath &path) {
        std::string portable;
        if (!TryNormalizePortableRelativePath(path.relative, portable))
            throw std::invalid_argument("journal contains an invalid relative path");
        path.absolute = FromFsPath(RequireInsideRoot(root, FromFsPath(root / ToFsPath(portable)), parents));
        registerPath(path.absolute);
    };
    for (auto &entry : journal.entries) {
        resolve(entry.target);
        const auto &name = entry.prepared.path;
        if (name.compare(0, StagedNamePrefix.size(), StagedNamePrefix) != 0 ||
            name.find_first_of("/\\:") != std::string::npos || !IsAtomicTemporaryPath(ToFsPath(name)))
            throw std::invalid_argument("journal contains an invalid prepared filename");
        const auto sibling = ToFsPath(entry.target.absolute).parent_path() / ToFsPath(name);
        if (!entry.expected) {
            const auto linkName = ".inx-publish" + name.substr(StagedBasename.size());
            entry.derivedPublishPath = FromFsPath(sibling.parent_path() / ToFsPath(linkName));
            registerPath(entry.derivedPublishPath);
        }
        entry.prepared.path = FromFsPath(sibling);
        registerPath(entry.prepared.path);
    }
    for (auto &path : journal.invalidated)
        resolve(path);
}

std::string SerializeJournal(const Journal &journal)
{
    std::string payload;
    AppendUint64(payload, journal.entries.size());
    AppendUint64(payload, journal.invalidated.size());
    for (const auto &entry : journal.entries) {
        AppendField(payload, entry.target.relative);
        AppendField(payload, FromFsPath(ToFsPath(entry.prepared.path).filename()));
        AppendUint64(payload, entry.prepared.state.size);
        AppendUint64(payload, entry.prepared.state.contentHash);
        const auto &state = entry.expected;
        AppendUint64(payload, !state ? 0 : (state->exists ? 2 : 1));
        if (state && state->exists) {
            AppendUint64(payload, state->size);
            AppendUint64(payload, static_cast<uint64_t>(state->modifiedNs));
            AppendUint64(payload, state->contentHash);
        }
    }
    for (const auto &path : journal.invalidated)
        AppendField(payload, path.relative);
    std::string bytes(JournalMagic);
    if (payload.size() > MaximumJournalBytes - JournalMagic.size() - sizeof(uint64_t))
        throw std::overflow_error("document transaction exceeds the journal size limit");
    AppendUint64(bytes, detail::FileFingerprint::Of(payload));
    bytes.append(payload);
    return bytes;
}

Journal DeserializeJournal(std::string_view bytes)
{
    if (bytes.size() < JournalMagic.size() + sizeof(uint64_t) || bytes.substr(0, JournalMagic.size()) != JournalMagic)
        throw std::invalid_argument("document transaction journal has an invalid header");
    size_t header = JournalMagic.size();
    const uint64_t checksum = ReadUint64(bytes, header);
    const auto payload = bytes.substr(header);
    if (checksum != detail::FileFingerprint::Of(payload))
        throw std::invalid_argument("document transaction journal checksum mismatch");

    size_t cursor = 0;
    const uint64_t entryCount = ReadUint64(payload, cursor);
    const uint64_t invalidationCount = ReadUint64(payload, cursor);
    if (entryCount == 0 || entryCount > MaximumEntryCount || invalidationCount > MaximumEntryCount)
        throw std::invalid_argument("document transaction journal has an invalid entry count");
    Journal journal;
    journal.entries.reserve(static_cast<size_t>(entryCount));
    journal.invalidated.reserve(static_cast<size_t>(invalidationCount));
    for (uint64_t index = 0; index < entryCount; ++index) {
        JournalEntry entry;
        entry.target.relative = ReadField(payload, cursor);
        entry.prepared.path = ReadField(payload, cursor);
        entry.prepared.state.exists = true;
        entry.prepared.state.size = ReadUint64(payload, cursor);
        entry.prepared.state.contentHash = ReadUint64(payload, cursor);
        const uint64_t guard = ReadUint64(payload, cursor);
        if (guard == 1) {
            entry.expected = AtomicFileState{};
        } else if (guard == 2) {
            AtomicFileState state;
            state.exists = true;
            state.size = ReadUint64(payload, cursor);
            const uint64_t modifiedBits = ReadUint64(payload, cursor);
            std::memcpy(&state.modifiedNs, &modifiedBits, sizeof(modifiedBits));
            state.contentHash = ReadUint64(payload, cursor);
            entry.expected = state;
        } else if (guard != 0) {
            throw std::invalid_argument("document transaction journal has an invalid baseline kind");
        }
        journal.entries.push_back(std::move(entry));
    }
    for (uint64_t index = 0; index < invalidationCount; ++index)
        journal.invalidated.push_back({ReadField(payload, cursor), {}});
    if (cursor != payload.size())
        throw std::invalid_argument("document transaction journal has trailing data");
    return journal;
}

std::string ReadJournal(const std::string &path)
{
    FileInput file(ToFsPath(path));
    const uint64_t size = file.Size();
    if (size > MaximumJournalBytes)
        throw std::invalid_argument("document transaction journal has an invalid file size");
    std::string bytes(static_cast<size_t>(size), '\0');
    file.Read(bytes.data(), bytes.size());
    return bytes;
}

std::array<PathWriteScope, 2> ClaimJournalFiles(const Journal &journal, const PathWriteScope &parent)
{
    std::vector<std::string> delegated, independent;
    const auto add = [&](const std::string &path) { (parent.Covers(path) ? delegated : independent).push_back(path); };
    for (const auto &entry : journal.entries)
        add(entry.target.absolute);
    for (const auto &path : journal.invalidated)
        add(path.absolute);
    return {PathWriteScope::BeginWrite(delegated, parent), PathWriteScope::BeginWrite(independent)};
}

PathWriteScope ClaimPreparedFiles(const Journal &journal, const PathWriteScope &parent)
{
    std::vector<std::string> paths;
    paths.reserve(journal.entries.size() * 2);
    const auto add = [&](const std::string &path) {
        if (!path.empty() && !parent.Covers(path))
            paths.push_back(path);
    };
    for (const auto &entry : journal.entries) {
        add(entry.prepared.path);
        add(entry.derivedPublishPath);
    }
    return PathWriteScope::BeginMutation(paths);
}

using DirectorySet = std::unordered_set<std::filesystem::path>;

DirectorySet PublicationDirectories(const Journal &journal)
{
    DirectorySet directories;
    for (const auto &entry : journal.entries)
        directories.insert(ToFsPath(entry.target.absolute).parent_path());
    return directories;
}

void PersistStagedDirectories(const std::filesystem::path &root, const DirectorySet &directories)
{
    // The journal may survive a power loss only if the entire path to each
    // staged file is also durable, including any newly created ancestors.
    DirectorySet ancestors;
    const auto rootKey = LexicalFilesystemPathKey(FromFsPath(root));
    for (const auto &directory : directories) {
        for (auto path = directory; IsFilesystemPathKeyWithin(LexicalFilesystemPathKey(FromFsPath(path)), rootKey);
             path = path.parent_path()) {
            if (!ancestors.insert(path).second)
                break;
            FlushFileDirectory(path);
            if (path == root)
                break;
        }
    }
}

void RemoveTransactionFile(const std::string &path)
{
    std::string error;
    if (!RemoveFileDurably(path, error))
        throw std::runtime_error("failed to remove transaction file '" + path + "': " + error);
}

// A batch can fail after some jobs have been accepted. Collect every receipt
// before releasing the path scopes or removing their unpublished staging files.
template <typename Submit, typename Collect> void RunWrites(size_t count, Submit submit, Collect collect)
{
    std::vector<std::pair<size_t, std::shared_ptr<DocumentWriteTicket>>> pending;
    pending.reserve(count);
    std::exception_ptr failure;
    for (size_t index = 0; index < count; ++index) {
        try {
            if (auto ticket = submit(index))
                pending.emplace_back(index, std::move(ticket));
        } catch (...) {
            failure = std::current_exception();
            break;
        }
    }
    for (const auto &[index, ticket] : pending) {
        try {
            ticket->Wait();
            const auto state = ticket->GetCommittedFileState();
            if (!state)
                throw std::runtime_error("transaction did not publish '" + ticket->GetPath() + "'");
            collect(index, *state);
        } catch (...) {
            if (!failure)
                failure = std::current_exception();
        }
    }
    if (failure)
        std::rethrow_exception(failure);
}

void DiscardStagedFiles(const Journal &journal) noexcept
{
    const auto remove = [](const std::string &path) {
        std::error_code ignored;
        std::filesystem::remove(NativeFilePath(ToFsPath(path)), ignored);
    };
    for (const auto &entry : journal.entries) {
        if (entry.ownsDerivedPublishPath)
            remove(entry.derivedPublishPath);
        if (entry.prepared.state.exists)
            remove(entry.prepared.path);
    }
}

struct StagedCleanup
{
    const Journal &journal;
    bool recorded = false;
    ~StagedCleanup()
    {
        if (!recorded)
            DiscardStagedFiles(journal);
    }
};

void StageWrites(Journal &journal, std::vector<DocumentTransactionEntry> &entries, const PathWriteScope &parent,
                 const PathWriteScope &preparedOwnership)
{
    RunWrites(
        journal.entries.size(),
        [&](size_t index) {
            const auto &path = journal.entries[index].prepared.path;
            const auto &owner = parent.Covers(path) ? parent : preparedOwnership;
            return DocumentStore::Instance().SubmitStaged(path, std::move(entries[index].content), owner,
                                                          std::move(entries[index].preclaimedStorage));
        },
        [&](size_t index, const AtomicFileState &state) { journal.entries[index].prepared.state = state; });
}

bool HasOutput(const AtomicFileState &actual, const AtomicFileState &expected)
{
    return actual.exists && actual.size == expected.size && actual.contentHash == expected.contentHash;
}

bool IsPreparedFile(const std::filesystem::path &path)
{
    const auto type = std::filesystem::symlink_status(path).type();
    if (type == std::filesystem::file_type::not_found)
        return false;
    if (type != std::filesystem::file_type::regular)
        throw std::runtime_error("transaction prepared path is not a regular file: " + FromFsPath(path));
    return true;
}

void PreparePublicationSources(Journal &journal, bool recovery)
{
    for (auto &entry : journal.entries) {
        if (entry.published)
            continue;
        const auto stage = NativeFilePath(ToFsPath(entry.prepared.path));
        if (!IsPreparedFile(stage))
            throw std::runtime_error("transaction staged file is missing: " + entry.prepared.path);
        if (entry.expected)
            continue;

        // Publish the link while the journal keeps the one staged payload for
        // recovery. Prepare every link before recording a fresh transaction.
        const auto link = NativeFilePath(ToFsPath(entry.derivedPublishPath));
        if (recovery && IsPreparedFile(link))
            std::filesystem::remove(link);
        std::filesystem::create_hard_link(stage, link);
        entry.ownsDerivedPublishPath = true;
    }
}

void CheckAuthoredBaselines(const Journal &journal)
{
    for (const auto &entry : journal.entries) {
        if (entry.expected && !(CaptureAtomicFileState(entry.target.absolute) == *entry.expected))
            throw std::runtime_error("document transaction target changed since input capture: " +
                                     entry.target.absolute);
    }
}

void CheckRecoveryInputs(Journal &journal)
{
    // The prepared file can have been consumed by an earlier publication.
    // Validate the entire write set before replaying any of its remaining files.
    for (auto &entry : journal.entries) {
        const auto expectedOutput = entry.prepared.state;
        const auto stage = NativeFilePath(ToFsPath(entry.prepared.path));
        entry.prepared.state = IsPreparedFile(stage) ? CaptureAtomicFileState(entry.prepared.path) : AtomicFileState{};
        if (entry.prepared.state.exists && !HasOutput(entry.prepared.state, expectedOutput))
            throw std::runtime_error("transaction prepared file changed: " + entry.prepared.path);
        if (!entry.expected) {
            if (!entry.prepared.state.exists)
                throw std::runtime_error("transaction derived file is missing: " + entry.prepared.path);
            continue;
        }

        const auto current = CaptureAtomicFileState(entry.target.absolute);
        if (HasOutput(current, expectedOutput)) {
            entry.published = current;
        } else if (!entry.prepared.state.exists) {
            throw std::runtime_error("transaction prepared file missing or published content changed: " +
                                     entry.target.absolute);
        } else if (!(current == *entry.expected)) {
            throw std::runtime_error("document transaction target changed since input capture: " +
                                     entry.target.absolute);
        }
    }
}

std::vector<DocumentTransactionFileState> ApplyJournal(const std::string &journalPath, Journal &journal,
                                                       const PathWriteScope &parent)
{
    RunWrites(
        journal.entries.size(),
        [&](size_t index) -> std::shared_ptr<DocumentWriteTicket> {
            auto &entry = journal.entries[index];
            const auto &path = entry.target.absolute;
            if (entry.published)
                return {};
            DocumentWriteOptions options;
            options.expectedFileState = entry.expected;
            if (parent.Covers(path))
                options.writeScope = parent;
            const auto &source = entry.expected ? entry.prepared.path : entry.derivedPublishPath;
            return DocumentStore::Instance().SubmitPrepared(path, {source, entry.prepared.state}, std::move(options));
        },
        [&](size_t index, const AtomicFileState &state) { journal.entries[index].published = state; });
    for (const auto &directory : PublicationDirectories(journal))
        FlushFileDirectory(directory);

    std::vector<DocumentTransactionFileState> files;
    files.reserve(journal.entries.size());
    for (const auto &entry : journal.entries)
        files.push_back(
            {LexicalFilesystemPathKey(entry.target.absolute), entry.published->size, entry.published->modifiedNs});
    for (const auto &path : journal.invalidated)
        RemoveTransactionFile(path.absolute);
    RemoveTransactionFile(journalPath);
    DiscardStagedFiles(journal);
    return files;
}

bool RecoverExisting(const std::filesystem::path &root, const std::string &journalPath, const PathWriteScope &parent,
                     CanonicalParentCache &parents)
{
    if (!std::filesystem::exists(NativeFilePath(ToFsPath(journalPath))))
        return false;
    auto journal = DeserializeJournal(ReadJournal(journalPath));
    ResolveJournalPaths(root, journalPath, journal, parents);
    const auto ownership = ClaimJournalFiles(journal, parent);
    const auto preparedOwnership = ClaimPreparedFiles(journal, parent);
    CheckRecoveryInputs(journal);
    PreparePublicationSources(journal, true);
    (void)ApplyJournal(journalPath, journal, parent);
    return true;
}

PathWriteScope OwnJournal(const std::string &path, const PathWriteScope &parent)
{
    return PathWriteScope::BeginMutation({path}, {}, parent.Covers(path) ? parent : PathWriteScope{});
}
} // namespace

DocumentTransactionStats DocumentTransaction::Commit(const std::string &projectRoot, const std::string &journalPath,
                                                     std::vector<DocumentTransactionEntry> entries,
                                                     std::vector<std::string> invalidatedPaths,
                                                     const PathWriteScope &writeScope)
{
    if (entries.empty() || entries.size() > MaximumEntryCount || invalidatedPaths.size() > MaximumEntryCount)
        throw std::invalid_argument("document transaction has an invalid entry count");
    const auto root = NormalizeRoot(projectRoot);
    CanonicalParentCache parents;
    const auto normalizedJournal = FromFsPath(RequireInsideRoot(root, journalPath, parents));
    const auto journalOwnership = OwnJournal(normalizedJournal, writeScope);
    RecoverExisting(root, normalizedJournal, writeScope, parents);

    const auto prepareStarted = Clock::now();
    for (auto &entry : entries)
        entry.path = ToRelativePath(root, entry.path, parents);
    std::sort(entries.begin(), entries.end(),
              [](const auto &left, const auto &right) { return left.path < right.path; });
    Journal journal;
    journal.entries.reserve(entries.size());
    for (const auto &entry : entries) {
        const auto name = FromFsPath(MakeAtomicTemporaryPath(ToFsPath(std::string(StagedBasename))));
        journal.entries.push_back({{entry.path, {}}, {name, {}}, entry.expectedFileState, {}, {}});
    }
    for (const auto &path : invalidatedPaths)
        journal.invalidated.push_back({ToRelativePath(root, path, parents), {}});
    ResolveJournalPaths(root, normalizedJournal, journal, parents);
    const auto ownership = ClaimJournalFiles(journal, writeScope);
    const auto preparedOwnership = ClaimPreparedFiles(journal, writeScope);
    StagedCleanup cleanup{journal};
    DocumentTransactionStats stats;
    stats.entryCount = entries.size();
    for (const auto &entry : entries)
        stats.preparedBytes += entry.content.size();
    auto directories = PublicationDirectories(journal);
    directories.insert(ToFsPath(normalizedJournal).parent_path());
    for (const auto &directory : directories)
        std::filesystem::create_directories(NativeFilePath(directory));
    StageWrites(journal, entries, writeScope, preparedOwnership);
    CheckAuthoredBaselines(journal);
    PreparePublicationSources(journal, false);
    PersistStagedDirectories(root, directories);
    stats.prepareMilliseconds = ElapsedMilliseconds(prepareStarted);

    const auto serializeStarted = Clock::now();
    const auto bytes = SerializeJournal(journal);
    stats.journalBytes = bytes.size();
    stats.serializeMilliseconds = ElapsedMilliseconds(serializeStarted);

    const auto journalStarted = Clock::now();
    const auto nativeJournal = NativeFilePath(ToFsPath(normalizedJournal));
    std::string error;
    const auto receipt = WriteTextFileAtomically(normalizedJournal, bytes, error, {FileWriteMode::Create, {}});
    if (!receipt) {
        // A rename can have succeeded before the directory flush failed.
        std::error_code inspectionError;
        cleanup.recorded = std::filesystem::exists(nativeJournal, inspectionError) || inspectionError;
        throw std::runtime_error("failed to persist document transaction journal: " + error);
    }
    if (receipt->publication != FilePublication::Written)
        throw std::runtime_error("document transaction journal already exists: " + normalizedJournal);
    cleanup.recorded = true;
    stats.journalWriteMilliseconds = ElapsedMilliseconds(journalStarted);

    const auto applyStarted = Clock::now();
    stats.committedFileStates = ApplyJournal(normalizedJournal, journal, writeScope);
    stats.applyMilliseconds = ElapsedMilliseconds(applyStarted);
    return stats;
}

bool DocumentTransaction::Recover(const std::string &projectRoot, const std::string &journalPath,
                                  const PathWriteScope &writeScope)
{
    const auto root = NormalizeRoot(projectRoot);
    CanonicalParentCache parents;
    const auto normalizedJournal = FromFsPath(RequireInsideRoot(root, journalPath, parents));
    const auto ownership = OwnJournal(normalizedJournal, writeScope);
    return RecoverExisting(root, normalizedJournal, writeScope, parents);
}

} // namespace infernux
