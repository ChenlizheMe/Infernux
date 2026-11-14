#pragma once

#include <glm/glm.hpp>
#include <pybind11/numpy.h>
#include <string>
#include <vector>

namespace infernux::binding
{
// Public matrices use [row, column], independently of their NumPy strides.
template <glm::length_t Size> glm::mat<Size, Size, float> MatrixFromPython(pybind11::handle value, const char *label)
{
    using Array = pybind11::array_t<float, pybind11::array::forcecast>;
    const auto array = Array::ensure(value);
    if (!array)
        throw pybind11::type_error(std::string(label) + " requires a numeric matrix");
    glm::mat<Size, Size, float> result;
    if (array.ndim() == 2 && array.shape(0) == Size && array.shape(1) == Size) {
        const auto values = array.unchecked<2>();
        for (glm::length_t row = 0; row < Size; ++row)
            for (glm::length_t column = 0; column < Size; ++column)
                result[column][row] = values(row, column);
    } else {
        throw pybind11::value_error(std::string(label) + " must have shape (" + std::to_string(Size) + ", " +
                                    std::to_string(Size) + ")");
    }
    return result;
}

inline glm::mat4 Matrix4FromPython(pybind11::handle value, const char *label)
{
    return MatrixFromPython<4>(value, label);
}

inline pybind11::array_t<float> Matrix4ArrayToPython(const std::vector<glm::mat4> &matrices)
{
    pybind11::array_t<float> result(
        {static_cast<pybind11::ssize_t>(matrices.size()), pybind11::ssize_t(4), pybind11::ssize_t(4)});
    auto values = result.mutable_unchecked<3>();
    for (size_t index = 0; index < matrices.size(); ++index)
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 4; ++column)
                values(index, row, column) = matrices[index][column][row];
    return result;
}

template <glm::length_t Size> pybind11::array_t<float> MatrixToPython(const glm::mat<Size, Size, float> &matrix)
{
    pybind11::array_t<float> result({Size, Size});
    auto values = result.mutable_unchecked<2>();
    for (glm::length_t row = 0; row < Size; ++row)
        for (glm::length_t column = 0; column < Size; ++column)
            values(row, column) = matrix[column][row];
    return result;
}

inline pybind11::array_t<float> Matrix4ToPython(const glm::mat4 &matrix)
{
    return MatrixToPython(matrix);
}
} // namespace infernux::binding
