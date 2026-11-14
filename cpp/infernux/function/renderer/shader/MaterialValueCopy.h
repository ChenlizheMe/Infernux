#pragma once

#include "ShaderValueCopy.h"
#include <core/types/ColorSpace.h>
#include <function/resources/InxMaterial/MaterialProperty.h>

#include <array>
#include <stdexcept>
#include <type_traits>

namespace infernux
{

/// Copy logical material storage through a prepared target layout.
/// GPU booleans use one word; GLM matrices and material documents use columns.
inline void WriteMaterialValue(const ShaderValueCopy &copy, void *destination, MaterialPropertyType type,
                               const MaterialPropertyValue &value, const std::string &name)
{
    std::visit(
        [&](const auto &item) {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, std::string>) {
                return;
            } else if constexpr (std::is_same_v<T, bool>) {
                const uint32_t word = item ? 1u : 0u;
                copy.Write(destination, &word);
            } else if constexpr (std::is_same_v<T, glm::vec4>) {
                if (type == MaterialPropertyType::Color) {
                    const auto linear = inx::color::SrgbToLinear(item);
                    copy.Write(destination, &linear);
                } else {
                    copy.Write(destination, &item);
                }
            } else if constexpr (std::is_same_v<T, glm::mat3>) {
                std::array<float, 9> columns;
                for (glm::length_t column = 0; column < 3; ++column)
                    for (glm::length_t row = 0; row < 3; ++row)
                        columns[column * 3 + row] = item[column][row];
                copy.Write(destination, columns.data());
            } else if constexpr (std::is_same_v<T, std::vector<float>> || std::is_same_v<T, std::vector<glm::vec4>>) {
                if (copy.GetSourceBytes() != item.size() * sizeof(typename T::value_type))
                    throw std::invalid_argument("Material array size does not match the compiled parameter: " + name);
                copy.Write(destination, item.data());
            } else {
                copy.Write(destination, &item);
            }
        },
        value);
}

} // namespace infernux
