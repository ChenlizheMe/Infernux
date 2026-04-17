#include "InxTexture.h"

#include <function/resources/InxResource/InxResourceMeta.h>

#include <core/log/InxLog.h>

namespace infernux
{

// =============================================================================
// LoadImportSettings — read ALL settings from .meta file
// =============================================================================

bool InxTexture::LoadImportSettings(const std::string &filePath)
{
    std::string metaPath = InxResourceMeta::GetMetaFilePath(filePath);
    InxResourceMeta meta;
    if (!meta.LoadFromFile(metaPath))
        return false;

    if (meta.HasKey("texture_type")) {
        m_textureType = meta.GetDataAs<std::string>("texture_type");
    }
    if (meta.HasKey("srgb")) {
        m_srgb = meta.GetDataAs<bool>("srgb");
    }
    if (meta.HasKey("generate_mipmaps")) {
        m_generateMipmaps = meta.GetDataAs<bool>("generate_mipmaps");
    }
    if (meta.HasKey("max_size")) {
        m_maxSize = meta.GetDataAs<int>("max_size");
    }
    if (meta.HasKey("filter_mode")) {
        m_filterMode = meta.GetDataAs<std::string>("filter_mode");
    }
    if (meta.HasKey("wrap_mode")) {
        m_wrapMode = meta.GetDataAs<std::string>("wrap_mode");
    }
    if (meta.HasKey("aniso_level")) {
        m_anisoLevel = meta.GetDataAs<int>("aniso_level");
    }
    return true;
}

// =============================================================================
// Clone — Unity-style Object.Instantiate for textures
// =============================================================================

std::shared_ptr<InxTexture> InxTexture::Clone() const
{
    auto clone = std::make_shared<InxTexture>();

    // Copy metadata — the clone references the same image file on disk.
    clone->m_name = m_name + " (Instance)";
    clone->m_filePath = m_filePath; // Same source file
    // clone->m_guid intentionally left empty — runtime-only instance

    // Copy import settings
    clone->m_textureType = m_textureType;
    clone->m_srgb = m_srgb;
    clone->m_generateMipmaps = m_generateMipmaps;
    clone->m_maxSize = m_maxSize;
    clone->m_filterMode = m_filterMode;
    clone->m_wrapMode = m_wrapMode;
    clone->m_anisoLevel = m_anisoLevel;

    return clone;
}

} // namespace infernux
