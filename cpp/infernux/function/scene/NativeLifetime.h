#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace infernux
{
class NativeLifetimeObserver
{
  public:
    virtual ~NativeLifetimeObserver() = default;
    virtual void RetireNativeObject() noexcept = 0;
};

/// Objects exposed through non-owning language bindings retire their borrowers
/// before storage is released. Unexposed objects allocate no observer storage.
/// Like Scene ownership, registration and retirement occur on the owner thread.
class NativeLifetimeOwner
{
  public:
    NativeLifetimeOwner() = default;
    NativeLifetimeOwner(const NativeLifetimeOwner &) = delete;
    NativeLifetimeOwner &operator=(const NativeLifetimeOwner &) = delete;
    // Borrowers belong to an address, not to movable payload. Both objects
    // remain alive after a move, with their original storage owners intact.
    NativeLifetimeOwner(NativeLifetimeOwner &&) noexcept
    {
    }
    NativeLifetimeOwner &operator=(NativeLifetimeOwner &&) noexcept
    {
        return *this;
    }
    ~NativeLifetimeOwner()
    {
        RetireNativeBorrowers();
    }

    void ObserveNativeLifetime(std::shared_ptr<NativeLifetimeObserver> observer)
    {
        m_borrowers.push_back(std::move(observer));
    }

    void ForgetNativeLifetimeObserver(const NativeLifetimeObserver *observer)
    {
        m_borrowers.erase(std::remove_if(m_borrowers.begin(), m_borrowers.end(),
                                         [observer](const auto &value) { return value.get() == observer; }),
                          m_borrowers.end());
    }

  private:
    void RetireNativeBorrowers() noexcept
    {
        std::vector<std::shared_ptr<NativeLifetimeObserver>> borrowers;
        borrowers.swap(m_borrowers);
        for (const auto &observer : borrowers)
            observer->RetireNativeObject();
    }
    std::vector<std::shared_ptr<NativeLifetimeObserver>> m_borrowers;
};
} // namespace infernux
