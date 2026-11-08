#pragma once

#include "core/FoundationApi.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernux
{

class PathWriteBusy final : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

/// One accepted operation's right to write paths. Ordinary writers coexist;
/// existing queues and expected-file-state checks still order their payloads.
/// A mutation excludes writers before moving/deleting a file or directory.
/// No lock is held while doing IO, waiting, or invoking a callback.
/// Keys use absolute lexical paths without file probes. Existing asset/Store
/// entry points supply their resolved paths; this is not a physical-alias lock.
class PathWriteScope
{
  public:
    PathWriteScope() = default;
    INFERNUX_FOUNDATION_API static PathWriteScope BeginWrite(const std::vector<std::string> &paths,
                                                             const PathWriteScope &parent = {});
    INFERNUX_FOUNDATION_API static PathWriteScope BeginMutation(const std::vector<std::string> &paths,
                                                                const std::vector<std::string> &directories = {},
                                                                const PathWriteScope &parent = {});
    [[nodiscard]] INFERNUX_FOUNDATION_API bool Covers(const std::string &path) const;
    [[nodiscard]] bool IsOpen() const noexcept
    {
        return static_cast<bool>(m_state);
    }
    /// Drop this holder. Accepted children retain their own ownership until
    /// actual publication/cleanup, independently of the caller's wrapper.
    void Close() noexcept
    {
        m_state.reset();
    }

  private:
    struct State;
    explicit PathWriteScope(std::shared_ptr<State> state) : m_state(std::move(state))
    {
    }
    static PathWriteScope Acquire(const std::vector<std::string> &paths, const std::vector<std::string> &directories,
                                  bool mutation, const PathWriteScope &parent);
    std::shared_ptr<State> m_state;
};

} // namespace infernux
