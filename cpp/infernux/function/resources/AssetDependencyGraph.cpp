#include "AssetDependencyGraph.h"

#include <core/log/InxLog.h>

namespace infernux
{

AssetDependencyGraph &AssetDependencyGraph::Instance()
{
    static AssetDependencyGraph instance;
    return instance;
}

// ============================================================================
// Asset → Asset dependencies
// ============================================================================

void AssetDependencyGraph::AddDependency(const std::string &userGuid, const std::string &dependencyGuid)
{
    if (userGuid.empty() || dependencyGuid.empty() || userGuid == dependencyGuid)
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    m_dependencies[userGuid].insert(dependencyGuid);
    m_dependents[dependencyGuid].insert(userGuid);
}

void AssetDependencyGraph::RemoveDependency(const std::string &userGuid, const std::string &dependencyGuid)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    auto fwdIt = m_dependencies.find(userGuid);
    if (fwdIt != m_dependencies.end()) {
        fwdIt->second.erase(dependencyGuid);
        if (fwdIt->second.empty())
            m_dependencies.erase(fwdIt);
    }

    auto revIt = m_dependents.find(dependencyGuid);
    if (revIt != m_dependents.end()) {
        revIt->second.erase(userGuid);
        if (revIt->second.empty())
            m_dependents.erase(revIt);
    }
}

void AssetDependencyGraph::ClearDependenciesOf(const std::string &userGuid)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    auto fwdIt = m_dependencies.find(userGuid);
    if (fwdIt == m_dependencies.end())
        return;

    // Remove this user from each dependency's reverse set
    for (const auto &depGuid : fwdIt->second) {
        auto revIt = m_dependents.find(depGuid);
        if (revIt != m_dependents.end()) {
            revIt->second.erase(userGuid);
            if (revIt->second.empty())
                m_dependents.erase(revIt);
        }
    }

    m_dependencies.erase(fwdIt);
}

void AssetDependencyGraph::RemoveAsset(const std::string &guid)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // Remove as a user (forward edges)
    auto fwdIt = m_dependencies.find(guid);
    if (fwdIt != m_dependencies.end()) {
        for (const auto &depGuid : fwdIt->second) {
            auto revIt = m_dependents.find(depGuid);
            if (revIt != m_dependents.end()) {
                revIt->second.erase(guid);
                if (revIt->second.empty())
                    m_dependents.erase(revIt);
            }
        }
        m_dependencies.erase(fwdIt);
    }

    // Remove as a dependency (reverse edges)
    auto revIt = m_dependents.find(guid);
    if (revIt != m_dependents.end()) {
        for (const auto &userGuid : revIt->second) {
            auto fwdIt2 = m_dependencies.find(userGuid);
            if (fwdIt2 != m_dependencies.end()) {
                fwdIt2->second.erase(guid);
                if (fwdIt2->second.empty())
                    m_dependencies.erase(fwdIt2);
            }
        }
        m_dependents.erase(revIt);
    }
}

void AssetDependencyGraph::SetDependencies(const std::string &userGuid,
                                           const std::unordered_set<std::string> &dependencyGuids)
{
    // No lock — ClearDependenciesOf and AddDependency each lock individually.
    // For atomicity we lock here instead.
    std::lock_guard<std::mutex> lock(m_mutex);

    // Clear old forward edges (inline to avoid double-lock)
    auto fwdIt = m_dependencies.find(userGuid);
    if (fwdIt != m_dependencies.end()) {
        for (const auto &depGuid : fwdIt->second) {
            auto revIt = m_dependents.find(depGuid);
            if (revIt != m_dependents.end()) {
                revIt->second.erase(userGuid);
                if (revIt->second.empty())
                    m_dependents.erase(revIt);
            }
        }
        fwdIt->second.clear();
    }

    // Add new edges
    for (const auto &depGuid : dependencyGuids) {
        if (depGuid.empty() || depGuid == userGuid)
            continue;
        m_dependencies[userGuid].insert(depGuid);
        m_dependents[depGuid].insert(userGuid);
    }

    // Clean up if no deps remain
    auto it = m_dependencies.find(userGuid);
    if (it != m_dependencies.end() && it->second.empty())
        m_dependencies.erase(it);
}

std::unordered_set<std::string> AssetDependencyGraph::GetDependencies(const std::string &guid) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_dependencies.find(guid);
    if (it != m_dependencies.end())
        return it->second;
    return {};
}

std::unordered_set<std::string> AssetDependencyGraph::GetDependents(const std::string &guid) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_dependents.find(guid);
    if (it != m_dependents.end())
        return it->second;
    return {};
}

bool AssetDependencyGraph::HasDependency(const std::string &userGuid, const std::string &dependencyGuid) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_dependencies.find(userGuid);
    if (it == m_dependencies.end())
        return false;
    return it->second.count(dependencyGuid) > 0;
}

// ============================================================================
// Event dispatch
// ============================================================================

void AssetDependencyGraph::RegisterCallback(ResourceType type, AssetEventCallback callback)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_callbacks[type].push_back(std::move(callback));
}

void AssetDependencyGraph::NotifyEvent(const std::string &guid, ResourceType type, AssetEvent event)
{
    // Copy dependents list to avoid holding lock during callbacks
    std::unordered_set<std::string> dependents;
    std::vector<AssetEventCallback> callbacks;

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto depIt = m_dependents.find(guid);
        if (depIt != m_dependents.end())
            dependents = depIt->second;

        auto cbIt = m_callbacks.find(type);
        if (cbIt != m_callbacks.end())
            callbacks = cbIt->second;
    }

    if (dependents.empty() || callbacks.empty()) {
        INXLOG_DEBUG("AssetDependencyGraph::NotifyEvent: guid=", guid, " type=", static_cast<int>(type),
                     " event=", static_cast<int>(event), " dependents=", dependents.size(),
                     " callbacks=", callbacks.size());
        return;
    }

    for (const auto &depGuid : dependents) {
        for (const auto &cb : callbacks) {
            cb(depGuid, guid, event);
        }
    }
}

// ============================================================================
// Diagnostics
// ============================================================================

size_t AssetDependencyGraph::GetEdgeCount() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    size_t count = 0;
    for (const auto &[_, deps] : m_dependencies)
        count += deps.size();
    return count;
}

size_t AssetDependencyGraph::GetNodeCount() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::unordered_set<std::string> nodes;
    for (const auto &[user, deps] : m_dependencies) {
        nodes.insert(user);
        for (const auto &dep : deps)
            nodes.insert(dep);
    }
    return nodes.size();
}

void AssetDependencyGraph::Clear()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_dependencies.clear();
    m_dependents.clear();
    // Keep callbacks — they are structural, not data
}

} // namespace infernux
