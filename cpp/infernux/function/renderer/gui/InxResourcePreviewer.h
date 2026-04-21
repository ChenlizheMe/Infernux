#pragma once

#include "InxGUIContext.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

// Forward declarations
class InxGUI;
class AssetDatabase;
class InxRenderer;

/// Display mode for preview visualization
enum class PreviewDisplayMode : int
{
    Default = 0,
    NormalMap = 1
};

/**
 * @brief Base interface for resource previewers
 *
 * Each previewer handles a specific type of resource (image, text, model, material, etc.)
 * and knows how to load and render it in the Inspector panel.
 */
class IResourcePreviewer
{
  public:
    virtual ~IResourcePreviewer() = default;

    /// @brief Get the type name of this previewer (e.g., "Image", "Text", "Model")
    virtual std::string GetTypeName() const = 0;

    /// @brief Get the list of supported file extensions (lowercase, with dot, e.g., ".png")
    virtual std::vector<std::string> GetSupportedExtensions() const = 0;

    /// @brief Check if this previewer can handle the given file
    virtual bool CanPreview(const std::string &filePath) const;

    /// @brief Load a resource for preview
    /// @return true if loaded successfully
    virtual bool Load(const std::string &filePath) = 0;

    /// @brief Render the preview in the given area
    /// @param ctx The GUI context
    /// @param availWidth Available width for rendering
    /// @param availHeight Available height for rendering
    virtual void Render(InxGUIContext *ctx, float availWidth, float availHeight) = 0;

    /// @brief Unload the current resource
    virtual void Unload() = 0;

    /// @brief Check if a resource is currently loaded
    virtual bool IsLoaded() const = 0;

    /// @brief Get the currently loaded file path
    virtual std::string GetLoadedPath() const = 0;

    /// @brief Get metadata about the loaded resource (for display in Inspector)
    /// @return Vector of key-value pairs (e.g., {"Width", "1920"}, {"Height", "1080"})
    virtual std::vector<std::pair<std::string, std::string>> GetMetadata() const = 0;

    /// @brief Set preview settings for visualization
    virtual void SetPreviewSettings(PreviewDisplayMode /*mode*/, int /*maxSize*/, bool /*srgb*/)
    {
    }
};

/**
 * @brief Manager for resource previewers
 *
 * Maintains a registry of previewers and provides a unified interface
 * for loading and rendering resource previews.
 */
class ResourcePreviewManager
{
  public:
    ResourcePreviewManager();
    ~ResourcePreviewManager();

    /// @brief Set the InxGUI instance (needed for texture upload)
    void SetGUI(InxGUI *gui);

    /// @brief Register a previewer
    void RegisterPreviewer(std::shared_ptr<IResourcePreviewer> previewer);

    /// @brief Check if there's a previewer for the given file extension
    bool HasPreviewer(const std::string &extension) const;

    /// @brief Get the previewer type name for a file extension
    std::string GetPreviewerTypeName(const std::string &extension) const;

    /// @brief Get all supported extensions
    std::vector<std::string> GetAllSupportedExtensions() const;

    /// @brief Load a file for preview
    /// @return true if loaded successfully
    bool LoadPreview(const std::string &filePath);

    /// @brief Render the current preview
    void RenderPreview(InxGUIContext *ctx, float availWidth, float availHeight);

    /// @brief Render metadata for the current preview
    void RenderMetadata(InxGUIContext *ctx);

    /// @brief Unload the current preview
    void UnloadPreview();

    /// @brief Check if a preview is currently loaded
    bool IsPreviewLoaded() const;

    /// @brief Get the currently loaded file path
    std::string GetLoadedPath() const;

    /// @brief Get the current previewer type name
    std::string GetCurrentTypeName() const;

    /// @brief Set preview settings (display mode, max size, sRGB) for live preview
    void SetPreviewSettings(int displayMode, int maxSize, bool srgb);

  private:
    IResourcePreviewer *FindPreviewer(const std::string &filePath) const;
    std::string GetExtension(const std::string &filePath) const;

    InxGUI *m_gui = nullptr;
    std::vector<std::shared_ptr<IResourcePreviewer>> m_previewers;
    std::unordered_map<std::string, IResourcePreviewer *> m_extensionMap; // extension -> previewer
    IResourcePreviewer *m_currentPreviewer = nullptr;
    PreviewDisplayMode m_displayMode = PreviewDisplayMode::Default;
    int m_maxSize = 0;
    bool m_srgb = true;
};

/**
 * @brief Image previewer - displays images (PNG, JPG, BMP, etc.)
 */
class ImagePreviewer : public IResourcePreviewer
{
  public:
    ImagePreviewer(InxGUI *gui);
    ~ImagePreviewer() override;

    std::string GetTypeName() const override
    {
        return "Image";
    }
    std::vector<std::string> GetSupportedExtensions() const override;

    bool Load(const std::string &filePath) override;
    void Render(InxGUIContext *ctx, float availWidth, float availHeight) override;
    void Unload() override;
    bool IsLoaded() const override
    {
        return m_textureId != 0;
    }
    std::string GetLoadedPath() const override
    {
        return m_loadedPath;
    }
    std::vector<std::pair<std::string, std::string>> GetMetadata() const override;
    void SetPreviewSettings(PreviewDisplayMode mode, int maxSize, bool srgb) override;

  private:
    void ApplyPreviewSettings();

    InxGUI *m_gui = nullptr;
    std::string m_loadedPath;
    uint64_t m_textureId = 0;
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    size_t m_fileSize = 0;
    std::vector<unsigned char> m_originalPixels;
    PreviewDisplayMode m_displayMode = PreviewDisplayMode::Default;
    int m_maxSize = 0;
    bool m_srgb = true;
};

/**
 * @brief Text previewer - displays text files
 */
class TextPreviewer : public IResourcePreviewer
{
  public:
    TextPreviewer();
    ~TextPreviewer() override = default;

    std::string GetTypeName() const override
    {
        return "Text";
    }
    std::vector<std::string> GetSupportedExtensions() const override;

    bool Load(const std::string &filePath) override;
    void Render(InxGUIContext *ctx, float availWidth, float availHeight) override;
    void Unload() override;
    bool IsLoaded() const override
    {
        return !m_content.empty() || m_loaded;
    }
    std::string GetLoadedPath() const override
    {
        return m_loadedPath;
    }
    std::vector<std::pair<std::string, std::string>> GetMetadata() const override;

  private:
    static constexpr size_t MAX_PREVIEW_SIZE = 64 * 1024; // 64KB

    std::string m_loadedPath;
    std::string m_content;
    std::vector<std::string> m_lines;
    size_t m_fileSize = 0;
    size_t m_lineCount = 0;
    bool m_truncated = false;
    bool m_loaded = false;
};

/**
 * @brief Binary previewer - shows info for binary files that can't be previewed
 */
class BinaryPreviewer : public IResourcePreviewer
{
  public:
    BinaryPreviewer() = default;
    ~BinaryPreviewer() override = default;

    std::string GetTypeName() const override
    {
        return "Binary";
    }
    std::vector<std::string> GetSupportedExtensions() const override;

    bool Load(const std::string &filePath) override;
    void Render(InxGUIContext *ctx, float availWidth, float availHeight) override;
    void Unload() override;
    bool IsLoaded() const override
    {
        return m_loaded;
    }
    std::string GetLoadedPath() const override
    {
        return m_loadedPath;
    }
    std::vector<std::pair<std::string, std::string>> GetMetadata() const override;

  private:
    std::string m_loadedPath;
    size_t m_fileSize = 0;
    bool m_loaded = false;
};

/**
 * @brief Material previewer - renders a PBR sphere with the material's properties
 */
class MaterialPreviewer : public IResourcePreviewer
{
  public:
    MaterialPreviewer(InxGUI *gui, int previewSize = 128);
    ~MaterialPreviewer() override;

    std::string GetTypeName() const override
    {
        return "Material";
    }
    std::vector<std::string> GetSupportedExtensions() const override;

    bool Load(const std::string &filePath) override;
    void Render(InxGUIContext *ctx, float availWidth, float availHeight) override;
    void Unload() override;
    bool IsLoaded() const override
    {
        return m_textureId != 0;
    }
    std::string GetLoadedPath() const override
    {
        return m_loadedPath;
    }
    std::vector<std::pair<std::string, std::string>> GetMetadata() const override;

    /// @brief Render a material file to RGBA pixels (for thumbnail generation).
    /// @param matFilePath Path to the .mat file
    /// @param size        Output image width and height (square)
    /// @param outPixels   Receives RGBA8 pixel data
    /// @param adb         Optional AssetDatabase for resolving texture GUIDs
    /// @param renderer    Optional renderer for GPU-based preview (falls back to CPU)
    /// @return true if successful
    static bool RenderToPixels(const std::string &matFilePath, int size, std::vector<unsigned char> &outPixels,
                               AssetDatabase *adb = nullptr, InxRenderer *renderer = nullptr);

    /// @brief Render from a JSON string instead of reading from disk.
    static bool RenderFromJson(const std::string &materialJson, int size, std::vector<unsigned char> &outPixels,
                               AssetDatabase *adb = nullptr, InxRenderer *renderer = nullptr);

  private:
    InxGUI *m_gui = nullptr;
    int m_previewSize;
    std::string m_loadedPath;
    uint64_t m_textureId = 0;
    std::string m_materialName;
    std::string m_shaderName;
};

} // namespace infernux
