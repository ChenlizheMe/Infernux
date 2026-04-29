#pragma once

#include "AssetImporter.h"
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/InxResource/InxResourceMeta.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace infernux
{

class AssetDatabase; // forward — used by MaterialImporter for path→GUID resolution

// ==========================================================================
// TextureImporter
// ==========================================================================

class TextureImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::Texture;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".psd", ".hdr", ".pic", ".pnm", ".pgm", ".ppm"};
    }

    bool Import(const ImportContext &ctx) override
    {
        // Texture loading is handled by InxTextureLoader in the asset pipeline.
        // This Import() ensures the meta file is created (already done by RegisterResource).
        if (!ctx.meta)
            return false;
        EnsureDefaultSettings(*ctx.meta);
        return true;
    }

    void EnsureDefaultSettings(InxResourceMeta &meta) override
    {
        if (!meta.HasKey("wrap_mode"))
            meta.AddMetadata("wrap_mode", std::string("repeat"));
        if (!meta.HasKey("filter_mode"))
            meta.AddMetadata("filter_mode", std::string("linear"));
        if (!meta.HasKey("generate_mipmaps"))
            meta.AddMetadata("generate_mipmaps", true);
        if (!meta.HasKey("srgb"))
            meta.AddMetadata("srgb", true);
        if (!meta.HasKey("max_size"))
            meta.AddMetadata("max_size", 2048);
    }
};

// ==========================================================================
// ShaderImporter
// ==========================================================================

class ShaderImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::Shader;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".vert", ".frag", ".geom", ".comp", ".tesc", ".tese"};
    }

    bool Import(const ImportContext &ctx) override
    {
        if (!ctx.meta)
            return false;
        EnsureDefaultSettings(*ctx.meta);
        return true;
    }

    void EnsureDefaultSettings(InxResourceMeta & /*meta*/) override
    {
        // Shader-specific settings can be added later (e.g. optimization level)
    }
};

// ==========================================================================
// MaterialImporter — scans .mat JSON to register texture/shader dependencies
// ==========================================================================

class MaterialImporter final : public AssetImporter
{
  public:
    /// Set the AssetDatabase pointer so we can resolve paths → GUIDs.
    void SetAssetDatabase(AssetDatabase *db)
    {
        m_assetDb = db;
    }

    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::Material;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".mat"};
    }

    bool Import(const ImportContext &ctx) override
    {
        if (!ctx.meta)
            return false;
        ScanDependencies(ctx);
        return true;
    }

    bool Reimport(const ImportContext &ctx) override
    {
        return Import(ctx); // re-scan deps on reimport
    }

  private:
    AssetDatabase *m_assetDb = nullptr;

    /// Parse .mat JSON → extract texture paths & shader paths → register as dependencies.
    void ScanDependencies(const ImportContext &ctx);
};

// ==========================================================================
// ScriptImporter
// ==========================================================================

class ScriptImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::Script;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".py"};
    }

    bool Import(const ImportContext &ctx) override
    {
        if (!ctx.meta)
            return false;
        return true;
    }
};

// ==========================================================================
// AudioImporter
// ==========================================================================

class AudioImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::Audio;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".wav"};
    }

    bool Import(const ImportContext &ctx) override
    {
        if (!ctx.meta)
            return false;
        EnsureDefaultSettings(*ctx.meta);
        return true;
    }

    void EnsureDefaultSettings(InxResourceMeta &meta) override
    {
        if (!meta.HasKey("force_mono"))
            meta.AddMetadata("force_mono", false);
        if (!meta.HasKey("load_in_background"))
            meta.AddMetadata("load_in_background", false);
        if (!meta.HasKey("quality"))
            meta.AddMetadata("quality", 1.0f);
    }
};

// ==========================================================================
// ModelImporter — handles 3D model files (.fbx, .obj, .gltf, .glb, …)
// ==========================================================================

class ModelImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::Mesh;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".fbx", ".obj", ".gltf", ".glb", ".dae", ".3ds", ".ply", ".stl"};
    }

    bool Import(const ImportContext &ctx) override;

    void EnsureDefaultSettings(InxResourceMeta &meta) override
    {
        if (!meta.HasKey("scale_factor"))
            meta.AddMetadata("scale_factor", 0.01f);
        if (!meta.HasKey("generate_normals"))
            meta.AddMetadata("generate_normals", true);
        if (!meta.HasKey("generate_tangents"))
            meta.AddMetadata("generate_tangents", true);
        if (!meta.HasKey("flip_uvs"))
            meta.AddMetadata("flip_uvs", true);
        if (!meta.HasKey("swap_uv_channels"))
            meta.AddMetadata("swap_uv_channels", false);
        if (!meta.HasKey("optimize_mesh"))
            meta.AddMetadata("optimize_mesh", true);
        const int importerVersion = meta.HasKey("importer_version") ? meta.GetDataAs<int>("importer_version") : 1;
        if (importerVersion < 2) {
            // Migrate old defaults to Unity-like behavior for DCC-authored assets.
            meta.AddMetadata("flip_uvs", true);
            if (!meta.HasKey("swap_uv_channels"))
                meta.AddMetadata("swap_uv_channels", false);
            meta.AddMetadata("importer_version", 2);
        }
    }
};

} // namespace infernux
