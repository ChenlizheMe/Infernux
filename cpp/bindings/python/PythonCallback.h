#pragma once

#include <pybind11/pybind11.h>

namespace infernux
{

/// Native frame/source ownership may end after the calling binding released
/// the GIL. Keep that lifetime rule in one place for every Python callback.
struct PythonCallback
{
    pybind11::object function;
    explicit PythonCallback(pybind11::object value) : function(std::move(value))
    {
    }
    ~PythonCallback()
    {
        if (!Py_IsInitialized()) {
            function.release();
            return;
        }
        pybind11::gil_scoped_acquire acquire;
        function = pybind11::object();
    }
};

} // namespace infernux
