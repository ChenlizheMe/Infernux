#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Core/TempAllocator.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace infernux
{

/// Retained scratch for one Jolt Update at a time. Jolt's job dependencies
/// serialize the LIFO operations, which may run on different worker threads.
class JoltCpuTempAllocator final : public JPH::TempAllocator
{
  public:
    explicit JoltCpuTempAllocator(size_t limit) : m_limit(limit - limit % Alignment)
    {
    }

    ~JoltCpuTempAllocator() override
    {
        JPH_ASSERT(IsEmpty());
    }

    void *Allocate(JPH::uint size) override
    {
        if (size == 0)
            return nullptr;
        const size_t bytes = AlignedBytes(size, m_limit - m_usedBytes);
        if (m_current == NoChunk || bytes > m_chunks[m_current].capacity - m_chunks[m_current].used)
            PushChunk(bytes);

        auto &chunk = m_chunks[m_current];
        void *address = chunk.data.get() + chunk.used;
        chunk.used += bytes;
        m_usedBytes += bytes;
        return address;
    }

    void Free(void *address, JPH::uint size) override
    {
        if (address == nullptr && size == 0)
            return;
        if (address == nullptr || size == 0 || m_current == NoChunk)
            Fail("invalid free", size);

        auto &chunk = m_chunks[m_current];
        const size_t bytes = AlignedBytes(size, chunk.used);
        if (chunk.data.get() + (chunk.used - bytes) != address)
            Fail("free out of LIFO order", size);

        chunk.used -= bytes;
        m_usedBytes -= bytes;
        if (chunk.used == 0) {
            m_activeBytes -= chunk.capacity;
            m_current = chunk.previous;
        }
    }

    [[nodiscard]] bool IsEmpty() const noexcept
    {
        return m_current == NoChunk;
    }

  private:
    static constexpr size_t Alignment = JPH_RVECTOR_ALIGNMENT;
    static constexpr size_t ChunkBytes = 64 * 1024;
    static constexpr size_t NoChunk = std::numeric_limits<size_t>::max();
    static_assert(ChunkBytes % Alignment == 0);

    struct AlignedFree
    {
        void operator()(std::byte *address) const noexcept
        {
            JPH::AlignedFree(address);
        }
    };

    struct Chunk
    {
        std::unique_ptr<std::byte, AlignedFree> data;
        size_t capacity = 0;
        size_t used = 0;
        size_t previous = NoChunk;
    };

    [[noreturn]] static void Fail(const char *reason, size_t bytes)
    {
        JPH::Trace("Jolt CPU scratch: %s (%zu bytes)", reason, bytes);
        std::abort();
    }

    static size_t AlignedBytes(size_t size, size_t available)
    {
        const size_t padding = (Alignment - size % Alignment) % Alignment;
        if (size > available || padding > available - size)
            Fail("aligned request exceeds available stack space", size);
        return size + padding;
    }

    void PushChunk(size_t bytes)
    {
        const size_t padding = (ChunkBytes - bytes % ChunkBytes) % ChunkBytes;
        const size_t capacity = bytes + std::min(padding, m_limit - bytes);
        // Keep nearby large requests reusable across differently sized
        // worlds, without letting a small request pin a large idle block.
        const size_t reuseLimit = capacity + std::min(bytes / 8, m_limit - capacity);
        size_t best = NoChunk;
        for (size_t i = 0; i < m_chunks.size(); ++i) {
            const auto &chunk = m_chunks[i];
            if (chunk.used == 0 && chunk.capacity >= bytes && chunk.capacity <= reuseLimit &&
                (best == NoChunk || chunk.capacity < m_chunks[best].capacity))
                best = i;
        }

        if (best == NoChunk)
            best = AllocateChunk(capacity);

        auto &chunk = m_chunks[best];
        chunk.previous = m_current;
        m_current = best;
        m_activeBytes += chunk.capacity;
    }

    size_t AllocateChunk(size_t capacity)
    {
        if (capacity > std::numeric_limits<size_t>::max() - m_activeBytes)
            Fail("active chunk size overflow", capacity);

        const size_t nextActiveBytes = m_activeBytes + capacity;
        m_peakActiveBytes = std::max(m_peakActiveBytes, nextActiveBytes);
        const size_t idleBudget = m_peakActiveBytes - nextActiveBytes;
        size_t slot = NoChunk;
        for (size_t i = 0; i < m_chunks.size(); ++i) {
            auto &chunk = m_chunks[i];
            if (chunk.used != 0)
                continue;
            if (chunk.data && m_reservedBytes - m_activeBytes > idleBudget) {
                m_reservedBytes -= chunk.capacity;
                chunk.data.reset();
                chunk.capacity = 0;
            }
            if (!chunk.data)
                slot = i;
        }

        if (slot == NoChunk) {
            slot = m_chunks.size();
            try {
                m_chunks.emplace_back();
            } catch (const std::bad_alloc &) {
                Fail("could not allocate chunk metadata", capacity);
            }
        }

        auto &chunk = m_chunks[slot];
        chunk.data.reset(static_cast<std::byte *>(JPH::AlignedAllocate(capacity, Alignment)));
        if (!chunk.data)
            Fail("could not allocate chunk", capacity);
        chunk.capacity = capacity;
        m_reservedBytes += capacity;
        return slot;
    }

    const size_t m_limit;
    std::vector<Chunk> m_chunks;
    size_t m_current = NoChunk;
    size_t m_usedBytes = 0;
    size_t m_activeBytes = 0, m_peakActiveBytes = 0, m_reservedBytes = 0;
};

} // namespace infernux
