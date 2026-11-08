#pragma once

#include <array>
#include <pybind11/pybind11.h>
#include <type_traits>

namespace infernux::bindings
{
// Called with the GIL held. Custom Python easing is an observable callback;
// callers capture value-type input before entering it.
inline double CallPythonEase(PyObject *ease, PyObject *const *arguments, size_t count)
{
    const auto value =
        pybind11::reinterpret_steal<pybind11::object>(PyObject_Vectorcall(ease, arguments, count, nullptr));
    if (!value)
        throw pybind11::error_already_set();
    const double factor = PyFloat_AsDouble(value.ptr());
    if (factor == -1. && PyErr_Occurred())
        throw pybind11::error_already_set();
    return factor;
}

inline float SamplePreparedPythonEase(const pybind11::tuple &prepared, double progress)
{
    // This private tuple owns the callback, clock and immutable Single parameters.
    // They are prepared once when the ease changes, outside the frame loop.
    const auto duration =
        pybind11::reinterpret_steal<pybind11::object>(PyObject_CallNoArgs(PyTuple_GET_ITEM(prepared.ptr(), 1)));
    if (!duration)
        throw pybind11::error_already_set();
    const auto time = pybind11::float_(static_cast<float>(progress * PyFloat_AS_DOUBLE(duration.ptr())));
    PyObject *arguments[] = {time.ptr(), duration.ptr(), PyTuple_GET_ITEM(prepared.ptr(), 2),
                             PyTuple_GET_ITEM(prepared.ptr(), 3)};
    return static_cast<float>(CallPythonEase(PyTuple_GET_ITEM(prepared.ptr(), 0), arguments, 4));
}

template <typename T> inline pybind11::object EaseArgument(const T &argument)
{
    if constexpr (std::is_base_of_v<pybind11::handle, std::decay_t<T>>)
        return pybind11::reinterpret_borrow<pybind11::object>(argument);
    else
        return pybind11::cast(argument);
}

template <typename... Args> inline double SamplePythonEase(const pybind11::function &ease, const Args &...arguments)
{
    const std::array<pybind11::object, sizeof...(Args)> owners{EaseArgument(arguments)...};
    std::array<PyObject *, sizeof...(Args)> values;
    for (size_t index = 0; index < values.size(); ++index)
        values[index] = owners[index].ptr();
    return CallPythonEase(ease.ptr(), values.data(), values.size());
}
} // namespace infernux::bindings
