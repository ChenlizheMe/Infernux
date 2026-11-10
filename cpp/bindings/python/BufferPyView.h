#pragma once

#include <pybind11/pybind11.h>
#include <stdexcept>

namespace infernux::bindings
{
inline pybind11::buffer_info ContiguousBuffer(pybind11::handle value, bool writable = false)
{
    auto info = pybind11::reinterpret_borrow<pybind11::buffer>(value).request(writable);
    if (!PyBuffer_IsContiguous(info.view(), 'C'))
        throw std::invalid_argument("memory buffer must be C-contiguous");
    return info;
}
} // namespace infernux::bindings
