#include "InxPythonScriptLoader.hpp"

namespace infernux
{
void InxPythonScriptLoader::CreateMeta(AssetSource &source, InxResourceMeta &metadata) const
{
    // Python candidate compilation owns script structure; metadata needs its hash.
    metadata.InitFromSource(source, ResourceType::Script);
}
} // namespace infernux
