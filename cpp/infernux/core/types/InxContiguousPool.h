#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

namespace infernux
{

template <typename T> class InxContiguousPool
{
  public:
    struct Handle
    {
        uint32_t index = UINT32_MAX;
        uint32_t generation = 0;

        [[nodiscard]] bool IsValid() const
        {
            return index != UINT32_MAX;
        }

        bool operator==(const Handle &rhs) const
        {
            return index == rhs.index && generation == rhs.generation;
        }

        bool operator!=(const Handle &rhs) const
        {
            return !(*this == rhs);
        }
    };

    [[nodiscard]] Handle Allocate()
    {
        if (m_freeListHead != UINT32_MAX) {
            uint32_t index = m_freeListHead;
            Slot &slot = m_slots[index];
            m_freeListHead = slot.nextFree;
            slot.alive = true;
            ++m_aliveCount;
            return Handle{index, slot.generation};
        }

        Slot slot;
        slot.alive = true;
        slot.generation = 1;
        slot.nextFree = UINT32_MAX;
        m_slots.push_back(slot);
        ++m_aliveCount;
        return Handle{static_cast<uint32_t>(m_slots.size() - 1), 1};
    }

    bool Free(Handle handle)
    {
        if (!IsAlive(handle)) {
            return false;
        }

        Slot &slot = m_slots[handle.index];
        slot.alive = false;
        ++slot.generation;
        slot.nextFree = m_freeListHead;
        m_freeListHead = handle.index;
        --m_aliveCount;
        return true;
    }

    [[nodiscard]] bool IsAlive(Handle handle) const
    {
        if (!handle.IsValid() || handle.index >= m_slots.size()) {
            return false;
        }
        const Slot &slot = m_slots[handle.index];
        return slot.alive && slot.generation == handle.generation;
    }

    [[nodiscard]] T &Get(Handle handle)
    {
        return m_slots[handle.index].value;
    }

    [[nodiscard]] const T &Get(Handle handle) const
    {
        return m_slots[handle.index].value;
    }

    [[nodiscard]] size_t Capacity() const
    {
        return m_slots.size();
    }

    [[nodiscard]] size_t AliveCount() const
    {
        return m_aliveCount;
    }

    /// Pre-allocate storage for at least @p n total slots.
    void Reserve(size_t n)
    {
        m_slots.reserve(n);
    }

    [[nodiscard]] const std::vector<Handle> GetAliveHandles() const
    {
        std::vector<Handle> handles;
        handles.reserve(m_aliveCount);
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_slots.size()); ++i) {
            const Slot &slot = m_slots[i];
            if (slot.alive) {
                handles.push_back(Handle{i, slot.generation});
            }
        }
        return handles;
    }

    /// Iterate all alive elements without allocating a handle vector.
    /// @p func receives (T &value).  Returning false from func stops early.
    template <typename Func> void ForEachAlive(Func &&func)
    {
        for (auto &slot : m_slots) {
            if (slot.alive) {
                if constexpr (std::is_same_v<decltype(func(slot.value)), bool>) {
                    if (!func(slot.value))
                        return;
                } else {
                    func(slot.value);
                }
            }
        }
    }

    template <typename Func> void ForEachAlive(Func &&func) const
    {
        for (const auto &slot : m_slots) {
            if (slot.alive) {
                if constexpr (std::is_same_v<decltype(func(slot.value)), bool>) {
                    if (!func(slot.value))
                        return;
                } else {
                    func(slot.value);
                }
            }
        }
    }

  private:
    struct Slot
    {
        T value{};
        uint32_t generation = 0;
        bool alive = false;
        uint32_t nextFree = UINT32_MAX;
    };

    std::vector<Slot> m_slots;
    uint32_t m_freeListHead = UINT32_MAX;
    size_t m_aliveCount = 0;
};

} // namespace infernux
