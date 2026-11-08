#include "PathWriteScope.h"
#include "InxPath.h"

#include <algorithm>
#include <mutex>

namespace infernux
{
struct PathWriteScope::State
{
    struct Range
    {
        std::string path;
        bool directory = false;
        bool Covers(const Range &other) const
        {
            return directory ? IsFilesystemPathKeyWithin(other.path, path) : !other.directory && other.path == path;
        }
        bool Overlaps(const Range &other) const
        {
            return path == other.path || (directory && IsFilesystemPathKeyWithin(other.path, path)) ||
                   (other.directory && IsFilesystemPathKeyWithin(path, other.path));
        }
    };
    struct Table
    {
        std::mutex mutex;
        std::vector<State *> active;
    };
    static std::shared_ptr<Table> SharedTable()
    {
        static auto table = std::make_shared<Table>();
        return table;
    }
    bool HasExactPath(const std::string &path) const
    {
        const auto exact =
            std::lower_bound(ranges.begin(), ranges.end(), path,
                             [](const Range &owned, const std::string &key) { return owned.path < key; });
        return exact != ranges.end() && exact->path == path;
    }
    bool Covers(const Range &range) const
    {
        if (!range.directory && HasExactPath(range.path))
            return true;
        return hasDirectories && std::any_of(ranges.begin(), ranges.end(), [&](const Range &owned) {
                   return owned.directory && owned.Covers(range);
               });
    }
    bool Overlaps(const Range &range) const
    {
        return HasExactPath(range.path) ||
               ((range.directory || hasDirectories) &&
                std::any_of(ranges.begin(), ranges.end(), [&](const Range &owned) { return range.Overlaps(owned); }));
    }
    bool Delegates(const State *owner) const
    {
        for (auto scope = this; scope; scope = scope->parent.get())
            if (scope == owner)
                return true;
        return false;
    }
    ~State()
    {
        if (!table)
            return; // Acquisition failed before registration.
        std::lock_guard lock(table->mutex);
        auto &active = table->active;
        active.erase(std::find(active.begin(), active.end(), this));
    }
    std::shared_ptr<Table> table;
    std::shared_ptr<State> parent;
    std::vector<Range> ranges;
    bool mutation = false;
    bool hasDirectories = false;
};

PathWriteScope PathWriteScope::BeginWrite(const std::vector<std::string> &paths, const PathWriteScope &parent)
{
    return Acquire(paths, {}, false, parent);
}

PathWriteScope PathWriteScope::BeginMutation(const std::vector<std::string> &paths,
                                             const std::vector<std::string> &directories, const PathWriteScope &parent)
{
    return Acquire(paths, directories, true, parent);
}

PathWriteScope PathWriteScope::Acquire(const std::vector<std::string> &paths,
                                       const std::vector<std::string> &directories, bool mutation,
                                       const PathWriteScope &parent)
{
    if (paths.empty() && directories.empty())
        return {};
    auto state = std::make_shared<State>();
    state->mutation = mutation;
    state->parent = parent.m_state;
    state->ranges.reserve(paths.size() + directories.size());
    const auto add = [&](const std::string &path, bool directory) {
        auto key = LexicalFilesystemPathKey(path);
        if (key.empty())
            throw std::invalid_argument("write scope requires a nonempty path");
        state->ranges.push_back({std::move(key), directory});
    };
    for (const auto &path : paths)
        add(path, false);
    for (const auto &path : directories)
        add(path, true);
    state->hasDirectories = !directories.empty();
    std::sort(state->ranges.begin(), state->ranges.end(),
              [](const auto &left, const auto &right) { return left.path < right.path; });

    if (state->parent) {
        if (mutation && !state->parent->mutation)
            throw std::invalid_argument("a writer cannot authorize a filesystem mutation");
        for (const auto &range : state->ranges)
            if (!state->parent->Covers(range))
                throw std::invalid_argument("write scope does not cover '" + range.path + "'");
        if (mutation)
            return parent; // A new holder, without extending the parent's paths.
    }
    auto table = State::SharedTable();
    std::lock_guard lock(table->mutex);
    for (const auto *active : table->active) {
        if (!mutation && !active->mutation)
            continue;
        if (!mutation && state->parent && state->parent->Delegates(active))
            continue;
        for (const auto &requested : state->ranges)
            if (active->Overlaps(requested))
                throw PathWriteBusy("path has an accepted writer or filesystem mutation: " + requested.path);
    }
    table->active.push_back(state.get());
    state->table = std::move(table);
    return PathWriteScope(std::move(state));
}

bool PathWriteScope::Covers(const std::string &path) const
{
    return m_state && m_state->Covers({LexicalFilesystemPathKey(path), false});
}

} // namespace infernux
