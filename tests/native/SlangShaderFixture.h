#pragma once

#include <function/resources/ShaderAsset/SlangNativeModule.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernux::test
{

inline ShaderTargetCode CompileSlangFixture(const std::filesystem::path &path)
{
    std::ifstream source(path, std::ios::binary);
    if (!source)
        throw std::runtime_error("Cannot open shader fixture: " + path.generic_string());
    return CompileSlangNativeModule(
        {path.stem().stem().string(), path.generic_string(),
         std::string(std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>())},
        ShaderCodeTarget::SpirV);
}

inline std::vector<uint32_t> FixtureSpirv(const std::filesystem::path &path)
{
    const auto target = CompileSlangFixture(path);
    const auto &bytes = target.entries.front().code;
    std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

} // namespace infernux::test
