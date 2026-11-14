#pragma once

#include "MatrixPyBridge.h"
#include <function/resources/InxMaterial/MaterialProperty.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace infernux::binding
{
template <glm::length_t Size> glm::vec<Size, float> MaterialVectorFromPython(pybind11::handle value)
{
    if (!PySequence_Check(value.ptr()))
        return pybind11::cast<glm::vec<Size, float>>(value);
    const auto sequence = pybind11::reinterpret_borrow<pybind11::sequence>(value);
    if (pybind11::len(sequence) != Size)
        throw pybind11::value_error("Material vector has a different declared component count");
    glm::vec<Size, float> result;
    for (glm::length_t index = 0; index < Size; ++index)
        result[index] = pybind11::cast<float>(sequence[index]);
    return result;
}

inline std::vector<glm::vec4> MaterialVectorArrayFromPython(pybind11::handle value)
{
    const auto sequence = pybind11::reinterpret_borrow<pybind11::sequence>(value);
    std::vector<glm::vec4> result;
    result.reserve(pybind11::len(sequence));
    for (const auto item : sequence)
        result.push_back(MaterialVectorFromPython<4>(item));
    return result;
}

// Python conversion happens once at the public boundary. The declared shader
// type owns conversion for both shared materials and per-renderer overrides.
inline MaterialPropertyValue MaterialPropertyFromPython(MaterialPropertyType type, pybind11::handle value)
{
    namespace py = pybind11;
    switch (type) {
    case MaterialPropertyType::Float:
        return py::cast<float>(value);
    case MaterialPropertyType::Int:
        if (py::isinstance<py::bool_>(value))
            throw py::type_error("An int material parameter requires an integer");
        return py::cast<int>(value);
    case MaterialPropertyType::UInt:
        if (py::isinstance<py::bool_>(value))
            throw py::type_error("A uint material parameter requires an unsigned integer");
        return py::cast<uint32_t>(value);
    case MaterialPropertyType::Bool:
        if (!py::isinstance<py::bool_>(value))
            throw py::type_error("A bool material parameter requires True or False");
        return py::cast<bool>(value);
    case MaterialPropertyType::Float2:
        return MaterialVectorFromPython<2>(value);
    case MaterialPropertyType::Float3:
        return MaterialVectorFromPython<3>(value);
    case MaterialPropertyType::Float4:
    case MaterialPropertyType::Color:
        return MaterialVectorFromPython<4>(value);
    case MaterialPropertyType::Mat3:
        return MatrixFromPython<3>(value, "Material matrix");
    case MaterialPropertyType::Mat4:
        return MatrixFromPython<4>(value, "Material matrix");
    case MaterialPropertyType::FloatArray:
        return py::cast<std::vector<float>>(value);
    case MaterialPropertyType::Float4Array:
        return MaterialVectorArrayFromPython(value);
    case MaterialPropertyType::Texture2D:
        if (py::isinstance<py::str>(value))
            return py::cast<std::string>(value);
        return py::cast<std::string>(value.attr("guid"));
    }
    throw py::type_error("Unknown material parameter type");
}

template <glm::length_t Size> pybind11::list MaterialMatrixColumnsToPython(const glm::mat<Size, Size, float> &matrix)
{
    pybind11::list result;
    for (glm::length_t column = 0; column < Size; ++column)
        for (glm::length_t row = 0; row < Size; ++row)
            result.append(matrix[column][row]);
    return result;
}

template <typename Apply> void ApplyMaterialMatrixFromPython(pybind11::handle value, Apply apply)
{
    using Array = pybind11::array_t<float, pybind11::array::forcecast>;
    const auto array = Array::ensure(value);
    if (array && array.ndim() == 2 && array.shape(0) == 3)
        apply(MatrixFromPython<3>(array, "Material matrix"));
    else
        apply(MatrixFromPython<4>(value, "Material matrix"));
}
} // namespace infernux::binding
