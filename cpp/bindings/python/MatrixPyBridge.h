#pragma once

#include <glm/glm.hpp>
#include <pybind11/numpy.h>
#include <string>

namespace infernux::binding
{
// Public 4x4 arrays use [row, column], independently of their NumPy strides.
inline glm::mat4 Matrix4FromPython(pybind11::handle value, const char *label)
{
    using Array = pybind11::array_t<float, pybind11::array::forcecast>;
    const auto array = Array::ensure(value);
    if (!array)
        throw pybind11::type_error(std::string(label) + " requires a numeric matrix");
    glm::mat4 result;
    if (array.ndim() == 2 && array.shape(0) == 4 && array.shape(1) == 4) {
        const auto values = array.unchecked<2>();
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 4; ++column)
                result[column][row] = values(row, column);
    } else {
        throw pybind11::value_error(std::string(label) + " must have shape (4, 4)");
    }
    return result;
}

inline pybind11::array_t<float> Matrix4ToPython(const glm::mat4 &matrix)
{
    pybind11::array_t<float> result({4, 4});
    auto values = result.mutable_unchecked<2>();
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            values(row, column) = matrix[column][row];
    return result;
}
} // namespace infernux::binding
