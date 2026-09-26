#pragma once

#include "AssetImporter.h"
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/InxMesh/MeshImportSettings.h>
#include <function/resources/InxResource/InxResourceMeta.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{

// ==========================================================================
// PrefabImporter
// ==========================================================================

class PrefabImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::DefaultText;
    }
    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".prefab"};
    }
    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override;
};

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
        return asset_formats::ToVector(asset_formats::kTextureExtensions);
    }

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override;

    void EnsureDefaultSettings(InxResourceMeta &meta) const override
    {
        if (!meta.HasKey("texture_type"))
            meta.AddMetadata("texture_type", std::string("default"));
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
        if (!meta.HasKey("aniso_level"))
            meta.AddMetadata("aniso_level", -1);
        if (!meta.HasKey("texture_compression"))
            meta.AddMetadata("texture_compression", std::string("auto"));
        if (!meta.HasKey("texture_compression_quality"))
            meta.AddMetadata("texture_compression_quality", std::string("normal"));
        static const std::unordered_set<std::string> targetFormats = {
            "auto", "rgba8", "rgba4444", "rgba16_unorm", "rgba16_float", "rgba32_float",
        };
        if (!meta.HasKey("texture_format") ||
            targetFormats.find(meta.GetDataAs<std::string>("texture_format")) == targetFormats.end())
            meta.AddMetadata("texture_format", std::string("auto"));
        const std::string textureType = meta.GetDataAs<std::string>("texture_type");
        if (textureType == "normal_map" || textureType == "data" || textureType == "vector_field" ||
            textureType == "sdf")
            meta.AddMetadata("srgb", false);
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
        return {".vert", ".frag"};
    }

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override
    {
        ImportArtifact artifact(request.metadata);
        EnsureDefaultSettings(artifact.metadata);
        return artifact;
    }

    void EnsureDefaultSettings(InxResourceMeta & /*meta*/) const override
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
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::Material;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".mat"};
    }

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override;

  private:
    [[nodiscard]] std::vector<std::string> ScanDependencies(const ImportRequest &request) const;
};

class PhysicMaterialImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::PhysicMaterial;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".physicmaterial"};
    }

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override
    {
        return ImportArtifact(request.metadata);
    }
};

// ==========================================================================
// RenderEffectImporter — validates effect sources and tracks asset edges
// ==========================================================================

class RenderEffectImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::RenderEffect;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".effect", ".effectgroup"};
    }

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override;

  private:
    [[nodiscard]] std::vector<std::string> ScanDependencies(const ImportRequest &request) const;
};

// ==========================================================================
// ParticleGraphImporter - validates particle graph sources and asset edges
// ==========================================================================

class ParticleGraphImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::ParticleGraph;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".particlegraph"};
    }

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override;

  private:
    [[nodiscard]] std::vector<std::string> ScanDependencies(const ImportRequest &request) const;
};

// ==========================================================================
// DataAssetImporter
// ==========================================================================

class RenderTextureImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::RenderTexture;
    }
    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".rendertexture"};
    }
    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override;
};

class DataAssetImporter final : public AssetImporter
{
  public:
    [[nodiscard]] ResourceType GetResourceType() const override
    {
        return ResourceType::DataAsset;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {".inxdata"};
    }

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override;

  private:
    [[nodiscard]] std::vector<std::string> ScanDependencies(const ImportRequest &request) const;
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

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override
    {
        return ImportArtifact(request.metadata);
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
        return asset_formats::ToVector(asset_formats::kAudioExtensions);
    }

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override
    {
        ImportArtifact artifact(request.metadata);
        EnsureDefaultSettings(artifact.metadata);
        return artifact;
    }

    void EnsureDefaultSettings(InxResourceMeta &meta) const override
    {
        if (!meta.HasKey("force_mono"))
            meta.AddMetadata("force_mono", false);
        if (!meta.HasKey("load_type"))
            meta.AddMetadata("load_type", std::string("decompress_on_load"));
        if (!meta.HasKey("load_in_background"))
            meta.AddMetadata("load_in_background", false);
        if (!meta.HasKey("quality"))
            meta.AddMetadata("quality", 1.0f);
        if (!meta.HasKey("compression_format"))
            meta.AddMetadata("compression_format", std::string("pcm"));
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
        return asset_formats::ToVector(asset_formats::kMeshExtensions);
    }

    [[nodiscard]] ImportArtifact Import(const ImportRequest &request) const override;
};

} // namespace infernux
