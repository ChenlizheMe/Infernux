#pragma once

#include <cstdint>
#include <cstring>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernux::bindings
{
inline std::vector<uint32_t> DecodeSpirvBytes(const pybind11::handle &value, const std::string &label)
{
    if (!pybind11::isinstance<pybind11::bytes>(value))
        throw std::invalid_argument(label + " must be SPIR-V bytes");
    const std::string bytes = pybind11::cast<pybind11::bytes>(value);
    if (bytes.size() < 5 * sizeof(uint32_t) || bytes.size() % sizeof(uint32_t) != 0)
        throw std::invalid_argument(label + " has an invalid SPIR-V byte size");
    std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    if (words.front() != 0x07230203u)
        throw std::invalid_argument(label + " has an invalid SPIR-V magic number");
    return words;
}
} // namespace infernux::bindings
