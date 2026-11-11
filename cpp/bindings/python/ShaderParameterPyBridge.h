#pragma once

#include <function/renderer/shader/ShaderParameterValue.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pybind11::detail
{
template <> struct type_caster<infernux::ShaderParameterValue>
{
    using Value = infernux::ShaderParameterValue;
    PYBIND11_TYPE_CASTER(Value, const_name("ShaderParameterValue"));

    template <size_t N> static Value Vector(const sequence &values)
    {
        std::array<float, N> data;
        for (size_t index = 0; index < N; ++index)
            data[index] = pybind11::cast<float>(values[index]);
        return data;
    }

    template <size_t N> static Value Matrix(const sequence &rows)
    {
        std::array<float, N * N> data;
        for (size_t row = 0; row < N; ++row) {
            const auto columns = pybind11::cast<sequence>(rows[row]);
            if (columns.size() != N)
                throw value_error("Shader matrices have a fixed square shape");
            for (size_t column = 0; column < N; ++column)
                data[row * N + column] = pybind11::cast<float>(columns[column]);
        }
        return data;
    }

    bool load(handle source, bool)
    {
        if (PyBool_Check(source.ptr())) {
            value = source.ptr() == Py_True;
            return true;
        }
        if (!PySequence_Check(source.ptr())) {
            if (PyIndex_Check(source.ptr())) {
                const auto integer = pybind11::cast<int64_t>(source);
                if (integer < std::numeric_limits<int32_t>::min() || integer > std::numeric_limits<uint32_t>::max())
                    throw value_error("Shader integer parameters use 32-bit values");
                value = integer > std::numeric_limits<int32_t>::max() ? Value(static_cast<uint32_t>(integer))
                                                                      : Value(static_cast<int32_t>(integer));
                return true;
            }
            if (!PyNumber_Check(source.ptr()))
                return false;
            value = pybind11::cast<float>(source);
            return true;
        }
        if (PyUnicode_Check(source.ptr()) || PyBytes_Check(source.ptr()))
            return false;
        const auto values = reinterpret_borrow<sequence>(source);
        const auto count = values.size();
        if (count == 0)
            return false;
        if (PySequence_Check(values[0].ptr()) && !PyUnicode_Check(values[0].ptr())) {
            if (count == 3)
                value = Matrix<3>(values);
            else if (count == 4)
                value = Matrix<4>(values);
            else
                return false;
        } else {
            if (count == 2)
                value = Vector<2>(values);
            else if (count == 3)
                value = Vector<3>(values);
            else if (count == 4)
                value = Vector<4>(values);
            else
                return false;
        }
        return true;
    }

    template <size_t N> static handle CastVector(const Value &value)
    {
        return pybind11::cast(value.Read<std::array<float, N>>()).release();
    }
    template <size_t N> static handle CastMatrix(const Value &value)
    {
        const auto data = value.Read<std::array<float, N * N>>();
        tuple rows(N);
        for (size_t row = 0; row < N; ++row) {
            tuple columns(N);
            for (size_t column = 0; column < N; ++column)
                columns[column] = data[row * N + column];
            rows[row] = std::move(columns);
        }
        return rows.release();
    }

    static handle cast(const Value &value, return_value_policy, handle)
    {
        using Kind = infernux::ShaderParameterKind;
        switch (value.Kind()) {
        case Kind::Float:
            return float_(value.Read<float>()).release();
        case Kind::Int:
            return int_(value.Read<int32_t>()).release();
        case Kind::UInt:
            return int_(value.Read<uint32_t>()).release();
        case Kind::Bool:
            return bool_(value.Read<uint32_t>() != 0).release();
        case Kind::Vector2:
            return CastVector<2>(value);
        case Kind::Vector3:
            return CastVector<3>(value);
        case Kind::Vector4:
            return CastVector<4>(value);
        case Kind::Matrix3:
            return CastMatrix<3>(value);
        case Kind::Matrix4:
            return CastMatrix<4>(value);
        }
        return {};
    }
};
} // namespace pybind11::detail
