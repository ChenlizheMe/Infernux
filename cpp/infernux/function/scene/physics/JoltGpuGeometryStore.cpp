#include "JoltGpuGeometryStore.h"
#include "JoltGpuTick.h"
#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
constexpr uint32_t UploadWords = 1024; // 16 KiB maximum packet, not a per-Host quota.
constexpr uint32_t CopyWords = 4096;   // 64 KiB maximum preservation packet.
size_t Fingerprint(const std::vector<UInt4> &words)
{
    size_t result = 0;
    for (const auto &word : words)
        for (const auto value : {word.x, word.y, word.z, word.w})
            result ^= size_t(value) + size_t(0x9e3779b9u) + (result << 6) + (result >> 2);
    return result;
}
bool Equal(UInt4 a, UInt4 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
} // namespace

bool JoltGpuGeometryLease::Contains(UInt4 reference) const noexcept
{
    if (reference.z || reference.w)
        return false;
    if (!reference.y)
        return reference.x == 0;
    if (!buffer || uint64_t(reference.x) + reference.y > wordCount ||
        !intervals::Contains(wordRanges, reference.x, uint64_t(reference.x) + reference.y))
        return false;
    return std::any_of(records.begin(), records.end(),
                       [&](const auto &record) { return record && Equal(record->reference, reference); });
}

JoltGpuGeometryStore::JoltGpuGeometryStore(rhi::ComputeHost &host) : m_host(host)
{
    const auto binding = host.device.GetCapabilities().portable.maxStorageBufferBinding;
    const uint64_t words = std::min<uint64_t>(UINT32_MAX, binding / sizeof(UInt4));
    if (!words || !host.device.GetCapabilities().IsStorageBufferRangeValid(0, words * sizeof(UInt4)))
        throw std::length_error("Device cannot bind the typed convex geometry arena");
    m_maxWords = static_cast<uint32_t>(words);
    m_free.reserve(2);
}

uint32_t JoltGpuGeometryStore::ReserveWords(uint32_t count)
{
    // At most live-record-count+1 free intervals exist. Reserve before creating
    // another live record so final-owner destruction can return words noexcept.
    m_free.reserve(m_offsets.size() + 2);
    for (auto it = m_free.begin(); it != m_free.end(); ++it)
        if (it->second - it->first >= count) {
            const auto first = static_cast<uint32_t>(it->first);
            it->first += count;
            if (it->first == it->second)
                m_free.erase(it);
            return first;
        }
    if (count > m_maxWords - m_highWaterWords)
        throw std::length_error("Live convex geometry ranges exceed device word/binding capacity");
    const auto first = m_highWaterWords;
    m_highWaterWords += count;
    return first;
}

void JoltGpuGeometryStore::FreeWords(uint32_t first, uint32_t count) noexcept
{
    intervals::Add(m_free, first, uint64_t(first) + count);
    if (!m_free.empty() && m_free.back().second == m_highWaterWords) {
        m_highWaterWords = static_cast<uint32_t>(m_free.back().first);
        m_free.pop_back();
    }
}

void JoltGpuGeometryStore::Release(uint32_t first, uint32_t count, size_t fingerprint,
                                   const JoltGpuGeometryRecord *recipe) noexcept
{
    const auto slot = m_offsets.find(first);
    if (slot == m_offsets.end())
        return; // Construction failed before its directory insertion.
    m_offsets.erase(slot);
    m_available.erase(first);
    m_recipes.erase(recipe);
    const auto [begin, end] = m_intern.equal_range(fingerprint);
    for (auto it = begin; it != end;)
        it = it->second.expired() ? m_intern.erase(it) : std::next(it);
    FreeWords(first, count);
}

std::shared_ptr<const JoltGpuGeometryHandle>
JoltGpuGeometryStore::Intern(std::shared_ptr<const JoltGpuGeometryRecord> record)
{
    if (!record)
        return {};
    if (weak_from_this().expired())
        throw std::logic_error("Convex geometry store requires its shared scene owner");
    if (const auto found = m_recipes.find(record.get()); found != m_recipes.end())
        if (auto existing = found->second.lock())
            return existing; // Stable native recipe: no repeated word scan/copy.
    ValidateJoltGpuGeometry(*record);
    const auto fingerprint = Fingerprint(record->words);
    const auto [first, last] = m_intern.equal_range(fingerprint);
    for (auto it = first; it != last; ++it)
        if (const auto old = it->second.lock();
            old && old->record->convexRadius == record->convexRadius &&
            old->record->words.size() == record->words.size() &&
            std::memcmp(old->record->words.data(), record->words.data(), record->words.size() * sizeof(UInt4)) == 0)
            return old;
    const uint32_t count = static_cast<uint32_t>(record->words.size());
    const uint32_t offset = ReserveWords(count);
    std::shared_ptr<const JoltGpuGeometryHandle> handle;
    try {
        auto weak = weak_from_this();
        handle = std::shared_ptr<const JoltGpuGeometryHandle>(
            new JoltGpuGeometryHandle{{offset, count, 0, 0}, std::move(record)},
            [weak, offset, count, fingerprint](const JoltGpuGeometryHandle *value) {
                if (auto store = weak.lock())
                    store->Release(offset, count, fingerprint, value->record.get());
                delete value;
            });
        m_offsets.emplace(offset, handle);
        m_recipes.emplace(handle->record.get(), handle);
        m_intern.emplace(fingerprint, handle);
    } catch (...) {
        m_offsets.erase(offset);
        if (handle)
            m_recipes.erase(handle->record.get());
        FreeWords(offset, count);
        throw;
    }
    return handle;
}

std::shared_ptr<const JoltGpuGeometryHandle> JoltGpuGeometryStore::Find(UInt4 reference) const
{
    if (!reference.y || reference.z || reference.w)
        return {};
    const auto found = m_offsets.find(reference.x);
    const auto handle = found != m_offsets.end() ? found->second.lock() : nullptr;
    return handle && Equal(handle->reference, reference) ? handle : nullptr;
}

uint32_t JoltGpuGeometryStore::RequireRecords(const JoltGpuGeometryHandles &records) const
{
    uint32_t required = 1;
    for (const auto &handle : records) {
        if (!handle)
            continue;
        const auto reference = handle->reference;
        if (!handle->record || Find(reference) != handle || reference.y != handle->record->words.size() ||
            uint64_t(reference.x) + reference.y > m_maxWords)
            throw std::invalid_argument("Convex geometry capture has no exact range in its ComputeHost store");
        required = std::max(required, reference.x + reference.y);
    }
    return required;
}

std::shared_ptr<const JoltGpuGeometryLease> JoltGpuGeometryStore::Lease(JoltGpuGeometryHandles records) const
{
    const uint32_t words = RequireRecords(records);
    if (!m_buffer || !m_sentinelAccepted)
        throw std::logic_error("Convex geometry sentinel was not accepted before capture");
    auto lease = std::make_shared<JoltGpuGeometryLease>();
    lease->buffer = m_buffer;
    lease->wordCount = words;
    lease->records = std::move(records);
    intervals::Add(lease->wordRanges, 0, 1);
    for (const auto &handle : lease->records) {
        if (!handle)
            continue;
        const auto found = m_available.find(handle->reference.x);
        if (found == m_available.end() || found->second.lock() != handle)
            throw std::logic_error("Convex record was not accepted in its captured buffer generation");
        intervals::Add(lease->wordRanges, handle->reference.x, uint64_t(handle->reference.x) + handle->reference.y);
    }
    for (const auto &[first, end] : lease->wordRanges)
        if (!m_buffer->HasInitializedRange(first * sizeof(UInt4), (end - first) * sizeof(UInt4)))
            throw std::logic_error("Convex geometry range has no accepted initialization proof");
    return lease;
}

struct JoltGpuGeometryStore::Upload : std::enable_shared_from_this<Upload>
{
    std::shared_ptr<JoltGpuGeometryStore> store;
    std::shared_ptr<rhi::ComputeBuffer> target;
    std::shared_ptr<const JoltGpuGeometryLease> guard;
    JoltGpuGeometryHandles records, copies, uploads;
    std::map<uint32_t, std::weak_ptr<const JoltGpuGeometryHandle>> available;
    intervals::Set copyRanges;
    uint32_t capacity = 0, required = 1, copyIndex = 0, copyOffset = 0, uploadIndex = 0, uploadOffset = 0;
    bool sentinelWritten = false, growing = false;
    std::function<void(std::shared_ptr<const JoltGpuGeometryLease>)> accepted;

    JoltGpuControlWork Work(std::shared_ptr<rhi::PreparedComputeBatch> packet, std::function<JoltGpuControlWork()> next)
    {
        JoltGpuControlWork result{std::move(packet), std::move(next), JoltGpuControlWork::ResumeAfter::Acceptance};
        result.geometry = guard; // The actual Tick receipt keeps ranges after Cancel.
        return result;
    }

    JoltGpuControlWork Next()
    {
        auto self = shared_from_this();
        if (!target) {
            const uint64_t doubled = std::max<uint64_t>(4096, uint64_t(store->m_capacityWords) * 2);
            capacity =
                static_cast<uint32_t>(std::min<uint64_t>(store->m_maxWords, std::max<uint64_t>(required, doubled)));
            target = rhi::ComputeBuffer::TryCreatePrivate(store->m_host, {capacity, rhi::ComputeScalarType::UInt32, 4});
            if (!target)
                return {nullptr, [self] { return self->Next(); }};
        }
        if (!guard) {
            // Ownership-only guard for accepted copy/upload receipts. Empty
            // wordRanges cannot be mistaken for initialized numerical geometry.
            auto held = std::make_shared<JoltGpuGeometryLease>();
            held->buffer = target;
            held->records = copies;
            held->records.insert(held->records.end(), uploads.begin(), uploads.end());
            guard = std::move(held);
        }
        if (copyIndex < copyRanges.size()) {
            const auto [first, end] = copyRanges[copyIndex];
            const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(CopyWords, end - first - copyOffset));
            const uint64_t offset = (first + copyOffset) * sizeof(UInt4), bytes = uint64_t(count) * sizeof(UInt4);
            auto packet = rhi::PrepareComputeBatch(store->m_host, {}, {}, {},
                                                   {{store->m_buffer, target, {offset, offset, bytes}}});
            return Work(packet, [self, count, end, first] {
                self->copyOffset += count;
                if (self->copyOffset == end - first) {
                    ++self->copyIndex;
                    self->copyOffset = 0;
                }
                return self->Next();
            });
        }
        if (!sentinelWritten) {
            auto packet =
                rhi::PrepareComputeBatch(store->m_host, {{target, 0, std::vector<uint8_t>(sizeof(UInt4))}}, {});
            return Work(packet, [self] {
                self->sentinelWritten = true;
                return self->Next();
            });
        }
        if (uploadIndex < uploads.size()) {
            const auto &handle = uploads[uploadIndex];
            const uint32_t count = std::min(UploadWords, handle->reference.y - uploadOffset);
            std::vector<uint8_t> bytes(uint64_t(count) * sizeof(UInt4));
            std::memcpy(bytes.data(), handle->record->words.data() + uploadOffset, bytes.size());
            auto packet = rhi::PrepareComputeBatch(
                store->m_host,
                {{target, uint64_t(handle->reference.x + uploadOffset) * sizeof(UInt4), std::move(bytes)}}, {});
            return Work(packet, [self, count] {
                self->uploadOffset += count;
                if (self->uploadOffset == self->uploads[self->uploadIndex]->reference.y) {
                    ++self->uploadIndex;
                    self->uploadOffset = 0;
                }
                return self->Next();
            });
        }
        // Only complete acceptance installs a compatible generation. A cancelled
        // partial writer has receipts/range guards but never publishes a lease.
        store->m_available.swap(available);
        store->m_buffer = target;
        store->m_capacityWords = capacity;
        store->m_sentinelAccepted = true;
        store->m_upload.reset();
        accepted(store->Lease(std::move(records)));
        return {};
    }
};

JoltGpuControlWork
JoltGpuGeometryStore::PrepareCapture(JoltGpuGeometryHandles records,
                                     std::function<void(std::shared_ptr<const JoltGpuGeometryLease>)> accepted)
{
    const uint32_t required = RequireRecords(records);
    const auto ready = [&](const auto &handle) {
        if (!handle)
            return true;
        const auto found = m_available.find(handle->reference.x);
        return found != m_available.end() && found->second.lock() == handle;
    };
    if (m_sentinelAccepted && std::all_of(records.begin(), records.end(), ready)) {
        accepted(Lease(std::move(records)));
        return {};
    }
    if (!m_upload.expired())
        throw std::logic_error("Convex geometry already has an unaccepted capture producer");
    auto upload = std::make_shared<Upload>();
    upload->store = shared_from_this();
    upload->required = required;
    upload->records = std::move(records);
    upload->accepted = std::move(accepted);
    upload->growing = !m_buffer || required > m_capacityWords;
    if (!upload->growing) {
        upload->target = m_buffer;
        upload->capacity = m_capacityWords;
        upload->sentinelWritten = m_sentinelAccepted;
    } else if (m_buffer) {
        if (m_sentinelAccepted) {
            intervals::Add(upload->copyRanges, 0, 1);
            upload->sentinelWritten = true;
        }
        for (const auto &[offset, weak] : m_available)
            if (auto handle = weak.lock()) {
                upload->copies.push_back(handle);
                intervals::Add(upload->copyRanges, offset, uint64_t(offset) + handle->reference.y);
                upload->required = std::max(upload->required, offset + handle->reference.y);
            }
    }
    std::map<uint32_t, std::shared_ptr<const JoltGpuGeometryHandle>> missing;
    for (const auto &handle : upload->records)
        if (!ready(handle))
            missing.emplace(handle->reference.x, handle);
    for (const auto &[offset, handle] : missing)
        upload->uploads.push_back(handle);
    // Allocate all directory nodes before accepting a writer; final compatible
    // generation installation is a noexcept map swap, never a partial commit.
    if (!upload->growing)
        upload->available = m_available;
    for (const auto &handle : upload->copies)
        upload->available[handle->reference.x] = handle;
    for (const auto &handle : upload->uploads)
        upload->available[handle->reference.x] = handle;
    m_upload = upload;
    return upload->Next();
}
} // namespace infernux
