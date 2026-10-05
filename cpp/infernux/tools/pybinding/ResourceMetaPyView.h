#pragma once

#include <function/resources/InxResource/InxResourceMeta.h>
#include <memory>
#include <optional>
#include <utility>

namespace infernux
{

// A query retains its published generation without copying large model tables
// or exposing a mutable alias of the database-owned metadata to Python.
class ResourceMetaView
{
  public:
    static std::optional<ResourceMetaView> From(std::shared_ptr<const InxResourceMeta> metadata)
    {
        if (!metadata)
            return std::nullopt;
        return ResourceMetaView(std::move(metadata));
    }

    const InxResourceMeta &Get() const
    {
        return *m_metadata;
    }

  private:
    explicit ResourceMetaView(std::shared_ptr<const InxResourceMeta> metadata) : m_metadata(std::move(metadata))
    {
    }
    std::shared_ptr<const InxResourceMeta> m_metadata;
};

} // namespace infernux
