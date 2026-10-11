#include "DocumentTransaction.h"

#include "AtomicFile.h"
#include "DocumentStore.h"
#include "InxPath.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <zlib.h>

namespace infernux
{
namespace
{
constexpr std::string_view JournalMagic = "INXDTX3\n";
constexpr uint64_t MaximumEntryCount = 1'000'000;
constexpr uint64_t MaximumJournalBytes = 2ULL << 30;

uint64_t Fnv1a64(std::string_view bytes)
{
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
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

void AccumulateJournalSize(uint64_t &total, uint64_t bytes)
{
    if (bytes > MaximumJournalBytes - total)
        throw std::overflow_error("document transaction exceeds the journal size limit");
    total += bytes;
}

std::string ReadField(std::string_view bytes, size_t &cursor)
{
    const uint64_t size = ReadUint64(bytes, cursor);
    if (size > MaximumJournalBytes || size > bytes.size() - cursor)
        throw std::invalid_argument("document transaction journal has an invalid field size");
    std::string value(bytes.substr(cursor, static_cast<size_t>(size)));
    cursor += static_cast<size_t>(size);
    return value;
}

std::string CompressPayload(std::string_view payload)
{
    if (payload.size() > std::numeric_limits<uLong>::max())
        throw std::overflow_error("document transaction payload exceeds zlib limits");
    uLongf compressedSize = compressBound(static_cast<uLong>(payload.size()));
    std::string compressed(static_cast<size_t>(compressedSize), '\0');
    const int result =
        compress2(reinterpret_cast<Bytef *>(compressed.data()), &compressedSize,
                  reinterpret_cast<const Bytef *>(payload.data()), static_cast<uLong>(payload.size()), Z_BEST_SPEED);
    if (result != Z_OK)
        throw std::runtime_error("failed to compress document transaction journal");
    compressed.resize(static_cast<size_t>(compressedSize));
    return compressed;
}

std::string DecompressPayload(std::string_view compressed, uint64_t expectedSize)
{
    if (expectedSize > MaximumJournalBytes || expectedSize > std::numeric_limits<uLongf>::max() ||
        compressed.size() > std::numeric_limits<uLong>::max())
        throw std::invalid_argument("document transaction journal has invalid compression sizes");
    std::string payload(static_cast<size_t>(expectedSize), '\0');
    uLongf actualSize = static_cast<uLongf>(expectedSize);
    const int result =
        uncompress(reinterpret_cast<Bytef *>(payload.data()), &actualSize,
                   reinterpret_cast<const Bytef *>(compressed.data()), static_cast<uLong>(compressed.size()));
    if (result != Z_OK || actualSize != expectedSize)
        throw std::invalid_argument("document transaction journal decompression failed");
    return payload;
}

std::filesystem::path NormalizeRoot(const std::string &projectRoot)
{
    if (projectRoot.empty())
        throw std::invalid_argument("document transaction project root cannot be empty");
    const std::string root = ResolveFilesystemPath(projectRoot);
    if (root.empty())
        throw std::invalid_argument("failed to normalize document transaction root");
    return ToFsPath(root);
}

bool PathsEqual(const std::filesystem::path &left, const std::filesystem::path &right)
{
    return FilesystemPathsEquivalent(FromFsPath(left), FromFsPath(right));
}

using CanonicalParentCache = std::unordered_map<std::string, std::filesystem::path>;

std::filesystem::path RequireInsideRoot(const std::filesystem::path &root, const std::string &path,
                                        CanonicalParentCache *parentCache = nullptr)
{
    if (path.empty())
        throw std::invalid_argument("document transaction path cannot be empty");
    auto absolute = ToFsPath(NormalizeFilesystemPathLexically(path));
    if (parentCache) {
        const auto lexicalParent = absolute.parent_path();
        const std::string parentKey = LexicalFilesystemPathKey(FromFsPath(lexicalParent));
        auto cached = parentCache->find(parentKey);
        if (cached == parentCache->end()) {
            auto canonicalParent = ToFsPath(ResolveFilesystemPath(FromFsPath(lexicalParent)));
            cached = parentCache->emplace(parentKey, std::move(canonicalParent)).first;
        }
        absolute = cached->second / absolute.filename();
    } else {
        absolute = ToFsPath(ResolveFilesystemPath(FromFsPath(absolute)));
    }

    if (!IsFilesystemPathWithin(FromFsPath(absolute), FromFsPath(root)))
        throw std::invalid_argument("document transaction path escapes project root: " + path);
    return absolute;
}

std::string ToRelativePath(const std::filesystem::path &root, const std::string &path,
                           CanonicalParentCache *parentCache = nullptr)
{
    const auto absolute = RequireInsideRoot(root, path, parentCache);
    std::string relative;
    if (!TryMakeRelativeFilesystemPath(FromFsPath(absolute), FromFsPath(root), relative))
        throw std::invalid_argument("invalid project-relative transaction path: " + path);
    return relative;
}

std::string ResolveRelativePath(const std::filesystem::path &root, const std::string &relative,
                                CanonicalParentCache *parentCache = nullptr)
{
    std::string normalizedRelative;
    if (!TryNormalizePortableRelativePath(relative, normalizedRelative))
        throw std::invalid_argument("journal contains an invalid relative path");
    const std::filesystem::path relativePath = ToFsPath(normalizedRelative);
    return FromFsPath(RequireInsideRoot(root, FromFsPath(root / relativePath), parentCache));
}

struct Journal
{
    std::vector<DocumentTransactionEntry> entries;
    std::vector<std::string> invalidatedPaths;
};

std::string SerializeJournal(const std::filesystem::path &root, std::vector<DocumentTransactionEntry> entries,
                             std::vector<std::string> invalidatedPaths, uint64_t &uncompressedBytes)
{
    if (entries.empty())
        throw std::invalid_argument("document transaction requires at least one entry");
    if (entries.size() > MaximumEntryCount || invalidatedPaths.size() > MaximumEntryCount)
        throw std::overflow_error("document transaction exceeds the journal entry limit");

    CanonicalParentCache parentCache;
    for (auto &entry : entries)
        entry.path = ToRelativePath(root, entry.path, &parentCache);
    for (auto &path : invalidatedPaths)
        path = ToRelativePath(root, path, &parentCache);
    std::sort(entries.begin(), entries.end(),
              [](const auto &left, const auto &right) { return left.path < right.path; });
    std::sort(invalidatedPaths.begin(), invalidatedPaths.end());

    std::unordered_set<std::string> uniquePaths;
    uint64_t estimatedBytes = JournalMagic.size() + sizeof(uint64_t) * 3;
    for (const auto &entry : entries) {
        if (!uniquePaths.insert(entry.path).second)
            throw std::invalid_argument("document transaction contains a duplicate target path");
        if (static_cast<uint64_t>(entry.path.size()) > MaximumJournalBytes ||
            static_cast<uint64_t>(entry.content.size()) > MaximumJournalBytes)
            throw std::overflow_error("document transaction exceeds the journal size limit");
        AccumulateJournalSize(estimatedBytes, entry.path.size());
        AccumulateJournalSize(estimatedBytes, entry.content.size());
        AccumulateJournalSize(estimatedBytes,
                              sizeof(uint64_t) * (entry.expectedFileState && entry.expectedFileState->exists ? 6 : 3));
    }
    for (const auto &path : invalidatedPaths) {
        if (!uniquePaths.insert(path).second)
            throw std::invalid_argument("document transaction invalidates a target path");
        if (static_cast<uint64_t>(path.size()) > MaximumJournalBytes)
            throw std::overflow_error("document transaction exceeds the journal size limit");
        AccumulateJournalSize(estimatedBytes, path.size());
        AccumulateJournalSize(estimatedBytes, sizeof(uint64_t));
    }

    std::string payload;
    payload.reserve(static_cast<size_t>(estimatedBytes));
    AppendUint64(payload, entries.size());
    AppendUint64(payload, invalidatedPaths.size());
    for (const auto &entry : entries) {
        AppendField(payload, entry.path);
        AppendField(payload, entry.content);
        const auto &state = entry.expectedFileState;
        AppendUint64(payload, !state ? 0 : (state->exists ? 2 : 1));
        if (state && state->exists) {
            AppendUint64(payload, state->size);
            AppendUint64(payload, static_cast<uint64_t>(state->modifiedNs));
            AppendUint64(payload, state->contentHash);
        }
    }
    for (const auto &path : invalidatedPaths)
        AppendField(payload, path);
    uncompressedBytes = payload.size();

    const std::string compressed = CompressPayload(payload);
    std::string bytes(JournalMagic);
    AppendUint64(bytes, payload.size());
    AppendUint64(bytes, compressed.size());
    AppendUint64(bytes, Fnv1a64(payload));
    bytes.append(compressed);
    return bytes;
}

Journal DeserializeJournal(const std::filesystem::path &root, std::string_view bytes)
{
    if (bytes.size() < JournalMagic.size() + sizeof(uint64_t) * 3 ||
        bytes.substr(0, JournalMagic.size()) != JournalMagic)
        throw std::invalid_argument("document transaction journal has an invalid header");

    size_t envelopeCursor = JournalMagic.size();
    const uint64_t payloadSize = ReadUint64(bytes, envelopeCursor);
    const uint64_t compressedSize = ReadUint64(bytes, envelopeCursor);
    const uint64_t expectedChecksum = ReadUint64(bytes, envelopeCursor);
    if (compressedSize != bytes.size() - envelopeCursor)
        throw std::invalid_argument("document transaction journal has an invalid compressed size");
    const std::string payload = DecompressPayload(bytes.substr(envelopeCursor), payloadSize);
    if (expectedChecksum != Fnv1a64(payload))
        throw std::invalid_argument("document transaction journal checksum mismatch");

    size_t cursor = 0;
    const uint64_t entryCount = ReadUint64(payload, cursor);
    const uint64_t invalidationCount = ReadUint64(payload, cursor);
    if (entryCount == 0 || entryCount > MaximumEntryCount || invalidationCount > MaximumEntryCount)
        throw std::invalid_argument("document transaction journal has an invalid entry count");

    Journal journal;
    journal.entries.reserve(static_cast<size_t>(entryCount));
    journal.invalidatedPaths.reserve(static_cast<size_t>(invalidationCount));
    std::unordered_set<std::string> uniquePaths;
    CanonicalParentCache parentCache;
    for (uint64_t index = 0; index < entryCount; ++index) {
        const std::string relativePath = ReadField(payload, cursor);
        const std::string content = ReadField(payload, cursor);
        const std::string path = ResolveRelativePath(root, relativePath, &parentCache);
        if (!uniquePaths.insert(path).second)
            throw std::invalid_argument("document transaction journal contains a duplicate target");
        const uint64_t guard = ReadUint64(payload, cursor);
        std::optional<AtomicFileState> expected;
        if (guard == 1) {
            expected = AtomicFileState{};
        } else if (guard == 2) {
            AtomicFileState state;
            state.exists = true;
            state.size = ReadUint64(payload, cursor);
            const uint64_t modifiedBits = ReadUint64(payload, cursor);
            std::memcpy(&state.modifiedNs, &modifiedBits, sizeof(modifiedBits));
            state.contentHash = ReadUint64(payload, cursor);
            expected = state;
        } else if (guard != 0) {
            throw std::invalid_argument("document transaction journal has an invalid baseline kind");
        }
        journal.entries.push_back({path, content, expected});
    }
    for (uint64_t index = 0; index < invalidationCount; ++index) {
        const std::string path = ResolveRelativePath(root, ReadField(payload, cursor), &parentCache);
        if (!uniquePaths.insert(path).second)
            throw std::invalid_argument("document transaction journal invalidates a target");
        journal.invalidatedPaths.push_back(path);
    }
    if (cursor != payload.size())
        throw std::invalid_argument("document transaction journal has trailing data");
    return journal;
}

std::string ReadJournal(const std::string &journalPath)
{
    std::error_code sizeError;
    const uintmax_t size = std::filesystem::file_size(ToFsPath(journalPath), sizeError);
    if (sizeError || size > MaximumJournalBytes)
        throw std::invalid_argument("document transaction journal has an invalid file size");
    std::ifstream file(ToFsPath(journalPath), std::ios::in | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("failed to open document transaction journal: " + journalPath);
    std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad())
        throw std::runtime_error("failed to read document transaction journal: " + journalPath);
    return bytes;
}

std::vector<std::optional<AtomicFileState>> PreflightEntries(const std::vector<DocumentTransactionEntry> &entries,
                                                             bool recovery)
{
    std::vector<std::optional<AtomicFileState>> completed(entries.size());
    for (size_t index = 0; index < entries.size(); ++index) {
        const auto &entry = entries[index];
        if (!entry.expectedFileState)
            continue;
        const auto current = CaptureAtomicFileState(entry.path);
        if (current == *entry.expectedFileState)
            continue;
        // Recovery can encounter our already-published payload. Preserve that
        // file verbatim; a different author's later edit must never be replayed
        // over merely because an older journal still exists.
        if (recovery && current.exists) {
            const auto snapshot = ReadTextFileSnapshot(entry.path);
            if (snapshot.content == entry.content) {
                completed[index] = snapshot.state;
                continue;
            }
        }
        throw std::runtime_error("document transaction target changed since input capture: " + entry.path);
    }
    return completed;
}

std::vector<DocumentTransactionFileState> ApplyJournal(const std::string &journalPath, Journal journal, bool recovery)
{
    // Validate the whole write set before submitting any target (including
    // derived artifacts), then retain per-document CAS at actual replacement.
    const auto completed = PreflightEntries(journal.entries, recovery);
    std::vector<DocumentTransactionFileState> committedFiles;
    committedFiles.reserve(journal.entries.size());
    std::vector<std::shared_ptr<DocumentWriteTicket>> tickets;
    tickets.reserve(journal.entries.size());
    std::exception_ptr failure;
    for (size_t index = 0; index < journal.entries.size(); ++index) {
        auto &entry = journal.entries[index];
        try {
            if (PathsEqual(ToFsPath(entry.path), ToFsPath(journalPath)))
                throw std::invalid_argument("document transaction journal targets itself");
            if (completed[index]) {
                const auto &state = *completed[index];
                committedFiles.push_back({entry.path, state.size, state.modifiedNs});
                continue;
            }
            const auto parent = ToFsPath(entry.path).parent_path();
            if (!parent.empty())
                std::filesystem::create_directories(parent);
            DocumentWriteOptions options;
            options.expectedFileState = entry.expectedFileState;
            tickets.push_back(DocumentStore::Instance().Submit(entry.path, std::move(entry.content), options));
        } catch (...) {
            failure = std::current_exception();
            break;
        }
    }

    for (const auto &ticket : tickets) {
        try {
            ticket->Wait();
        } catch (...) {
            if (!failure)
                failure = std::current_exception();
        }
    }
    if (failure)
        std::rethrow_exception(failure);

    for (const auto &ticket : tickets) {
        const auto fileState = ticket->GetCommittedFileState();
        if (!fileState)
            throw std::runtime_error("document transaction could not fingerprint committed target: " +
                                     ticket->GetPath());
        committedFiles.push_back({ticket->GetPath(), fileState->size, fileState->modifiedNs});
    }

    for (const auto &path : journal.invalidatedPaths) {
        if (PathsEqual(ToFsPath(path), ToFsPath(journalPath)))
            throw std::invalid_argument("document transaction journal invalidates itself");
        std::string error;
        if (!RemoveFileDurably(path, error))
            throw std::runtime_error("failed to invalidate derived document '" + path + "': " + error);
    }
    std::string error;
    if (!RemoveFileDurably(journalPath, error))
        throw std::runtime_error("failed to remove document transaction journal: " + error);
    return committedFiles;
}
} // namespace

DocumentTransactionStats DocumentTransaction::Commit(const std::string &projectRoot, const std::string &journalPath,
                                                     std::vector<DocumentTransactionEntry> entries,
                                                     std::vector<std::string> invalidatedPaths)
{
    DocumentTransactionStats stats;
    stats.entryCount = entries.size();
    const auto root = NormalizeRoot(projectRoot);
    const std::string normalizedJournal = FromFsPath(RequireInsideRoot(root, journalPath));
    if (std::filesystem::exists(ToFsPath(normalizedJournal)))
        Recover(projectRoot, normalizedJournal);
    CanonicalParentCache parentCache;
    const std::string journalRelativePath = ToRelativePath(root, normalizedJournal, &parentCache);
    for (const auto &entry : entries) {
        if (ToRelativePath(root, entry.path, &parentCache) == journalRelativePath)
            throw std::invalid_argument("document transaction journal cannot be a target");
    }
    for (const auto &path : invalidatedPaths) {
        if (ToRelativePath(root, path, &parentCache) == journalRelativePath)
            throw std::invalid_argument("document transaction journal cannot invalidate itself");
    }

    const auto parent = ToFsPath(normalizedJournal).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    const auto serializeStarted = std::chrono::steady_clock::now();
    const std::string bytes =
        SerializeJournal(root, std::move(entries), std::move(invalidatedPaths), stats.uncompressedBytes);
    auto journal = DeserializeJournal(root, bytes);
    (void)PreflightEntries(journal.entries, false);
    stats.journalBytes = bytes.size();
    stats.serializeMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - serializeStarted).count();
    std::string error;
    const auto journalWriteStarted = std::chrono::steady_clock::now();
    if (!WriteTextFileAtomically(normalizedJournal, bytes, error))
        throw std::runtime_error("failed to persist document transaction journal: " + error);
    stats.journalWriteMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - journalWriteStarted).count();
    const auto applyStarted = std::chrono::steady_clock::now();
    stats.committedFileStates = ApplyJournal(normalizedJournal, std::move(journal), false);
    stats.applyMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - applyStarted).count();
    return stats;
}

bool DocumentTransaction::Recover(const std::string &projectRoot, const std::string &journalPath)
{
    const auto root = NormalizeRoot(projectRoot);
    const std::string normalizedJournal = FromFsPath(RequireInsideRoot(root, journalPath));
    if (!std::filesystem::exists(ToFsPath(normalizedJournal)))
        return false;
    (void)ApplyJournal(normalizedJournal, DeserializeJournal(root, ReadJournal(normalizedJournal)), true);
    return true;
}

} // namespace infernux
