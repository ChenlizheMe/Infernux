#pragma once

#include <function/resources/InxResource/InxResourceMeta.h>

#include <string>
#include <vector>

namespace infernux
{

/// @brief Texture data container holding raw pixel data from stb_image
struct InxTextureData
{
    std::vector<unsigned char> pixels; ///< RGBA pixel data
    int width = 0;                     ///< Image width in pixels
    int height = 0;                    ///< Image height in pixels
    int channels = 4;                  ///< Number of color channels (always forced to 4 for RGBA)
    std::string name;                  ///< Texture identifier/name
    std::string sourcePath;            ///< Original file path

    /// @brief Get total size in bytes
    size_t GetSizeBytes() const
    {
        return static_cast<size_t>(width) * height * channels;
    }

    /// @brief Check if texture data is valid
    bool IsValid() const
    {
        return !pixels.empty() && width > 0 && height > 0;
    }
};

/// @brief Pixel-loading utilities for raster images and static SVG.
///
/// Meta creation and loading is now handled by TextureLoader (IAssetLoader).
/// This class only provides static pixel-loading utilities.
class InxTextureLoader
{
  public:
    /// Select the P2/P3/P5/P6 decoder by source signature, before attempting stb.
    static bool IsPnmSource(const unsigned char *data, size_t dataSize);

    /// @brief Load texture directly from file path (convenience method)
    /// @param filePath Path to the texture file
    /// @param name Texture identifier name
    /// @return InxTextureData containing the loaded image
    /// svgMaxSize is the SVG longest edge; 0 means 8192. Raster images are unchanged.
    static InxTextureData LoadFromFile(const std::string &filePath, const std::string &name = "", int svgMaxSize = 0);

    /// @brief Load texture from memory buffer
    /// @param data Pointer to image file data in memory
    /// @param dataSize Size of data in bytes
    /// @param name Texture identifier name
    /// @return InxTextureData containing the loaded image
    static InxTextureData LoadFromMemory(const unsigned char *data, size_t dataSize, const std::string &name = "",
                                         int svgMaxSize = 0);

    /// @brief Create a solid color texture (for default/fallback textures)
    /// @param width Texture width
    /// @param height Texture height
    /// @param r Red component (0-255)
    /// @param g Green component (0-255)
    /// @param b Blue component (0-255)
    /// @param a Alpha component (0-255)
    /// @param name Texture identifier name
    /// @return InxTextureData containing the generated texture
    static InxTextureData CreateSolidColor(int width, int height, unsigned char r, unsigned char g, unsigned char b,
                                           unsigned char a, const std::string &name = "solid_color");

    /// @brief Create a checkerboard texture (for error/missing texture indication)
    /// @param width Texture width
    /// @param height Texture height
    /// @param checkerSize Size of each checker square
    /// @param name Texture identifier name
    /// @return InxTextureData containing the generated checkerboard texture
    static InxTextureData CreateCheckerboard(int width, int height, int checkerSize = 8,
                                             const std::string &name = "checkerboard");
};

} // namespace infernux
