#include <function/resources/AssetImporter/ConcreteImporters.h>
#include <function/resources/InxTexture/TextureArtifact.h>
#include <function/resources/InxTexture/TextureDecoder.h>
#include <platform/filesystem/InxPath.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct Case
{
    std::string source;
    std::vector<uint8_t> expected;
};

std::string Binary(const char *header, std::initializer_list<uint8_t> pixels)
{
    std::string result(header);
    for (const auto value : pixels)
        result.push_back(static_cast<char>(value));
    return result;
}

void Check(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}

struct TemporaryProject
{
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("infernux-texture-decode-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryProject() { Check(std::filesystem::create_directory(path), "temporary directory creation failed"); }
    ~TemporaryProject() { std::filesystem::remove_all(path); }
};
} // namespace

int main()
try {
    const std::vector<Case> cases = {
        {"P3\n1 1\n15\n15 0 5\n", {255, 0, 85, 255}},
        {"P2\n1 1\n1023\n512\n", {128, 128, 128, 255}},
        {Binary("P6\n1 1\n15\n", {15, 0, 5}), {255, 0, 85, 255}},
        {Binary("P5\n1 1\n15\n", {5}), {85, 85, 85, 255}},
        {Binary("P6\n1 1\n65535\n", {255, 255, 0, 0, 128, 0}), {255, 0, 128, 255}},
        {Binary("P5\n1 1\n1023\n", {2, 0}), {128, 128, 128, 255}},
        {Binary("P5\n1 1\n255\n", {35}), {35, 35, 35, 255}},
        {Binary("P6\n1 1\n255\n", {10, 13, 32}), {10, 13, 32, 255}},
        {"P3\n1 1\n255\n255 0\n", {}},
        {"P3\n1 1\n255\n255 0 invalid\n", {}},
        {"P2\n1 1\n15\n16\n", {}},
        {Binary("P6\n1 1\n255\n", {255, 0}), {}},
        {Binary("P5\n1 1\n1023\n", {2}), {}},
        {Binary("P6\n1 1\n15\n", {16, 0, 0}), {}},
        {"P3\n65535 65535\n255\n0\n", {}},
        {"", {}},
    };
    TemporaryProject project;
    size_t checks = 0;
    for (size_t index = 0; index < cases.size(); ++index) {
        const auto &test = cases[index];
        const auto filePath = project.path / (std::to_string(index) + ".ppm");
        {
            std::ofstream file(filePath, std::ios::binary);
            file.write(test.source.data(), static_cast<std::streamsize>(test.source.size()));
            Check(file.good(), "test source write failed");
        }
        infernux::ImportRequest request;
        request.sourcePath = infernux::FromFsPath(filePath);
        request.guid = "texture-decode-test-" + std::to_string(index);
        request.resourceType = infernux::ResourceType::Texture;
        request.metadata.Init(test.source.data(), test.source.size(), request.sourcePath, request.resourceType);
        request.metadata.AddMetadata("texture_compression", std::string("none"));
        request.metadata.AddMetadata("texture_format", std::string("rgba8"));
        request.metadata.AddMetadata("srgb", false);
        request.metadata.AddMetadata("generate_mipmaps", false);
        for (const bool import : {false, true}) {
            std::shared_ptr<const infernux::TextureCpuData> texture;
            try {
                if (import) {
                    const auto artifact = infernux::TextureImporter().Import(request);
                    texture = infernux::TextureArtifact::Deserialize(artifact.runtimeCpuArtifacts.at(0).bytes,
                        artifact.metadata.GetDataAs<std::string>("content_hash"));
                } else {
                    texture = infernux::TextureDecoder::DecodeMemory(
                        {test.source.begin(), test.source.end()}, request.metadata, request.sourcePath);
                }
            } catch (const std::exception &error) {
                if (!test.expected.empty())
                    throw;
                const std::string diagnostic(error.what());
                Check(diagnostic.find("failed to decode") != std::string::npos ||
                      diagnostic.find("texture source is empty") != std::string::npos,
                      "invalid source needs an explicit decode error");
            }
            if (test.expected.empty()) {
                Check(!texture, "invalid source published a CPU artifact");
            } else {
                Check(texture && texture->bytes == test.expected, "decoded pixels disagree with PNM sample values");
                Check(texture->format == infernux::TextureFormat::Rgba8UNorm && texture->mipLevels.size() == 1,
                      "unexpected texture output format");
                Check(texture->mipLevels[0].width == 1 && texture->mipLevels[0].height == 1,
                      "unexpected texture dimensions");
            }
            ++checks;
        }
    }
    std::cout << "TEXTURE_DECODE_OK checks=" << checks << '\n';
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
