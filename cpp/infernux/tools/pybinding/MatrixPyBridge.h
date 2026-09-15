#pragma once

#include <glm/glm.hpp>
#include <pybind11/numpy.h>
#include <string>

namespace infernux::binding
{
// Public 4x4 arrays use [row, column], independently of their NumPy strides.
// Flat column-major sequences are retained only at older material/draw APIs.
inline glm::mat4 Matrix4FromPython(pybind11::handle value, const char *label, bool allowColumnMajorFlat = false)
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
    } else if (allowColumnMajorFlat && array.ndim() == 1 && array.shape(0) == 16) {
        const auto values = array.unchecked<1>();
        for (int index = 0; index < 16; ++index)
            result[index / 4][index % 4] = values(index);
    } else {
        throw pybind11::value_error(std::string(label) + " must have shape (4, 4)" +
                                    (allowColumnMajorFlat ? " or exactly 16 column-major numbers" : ""));
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
