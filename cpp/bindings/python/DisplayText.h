#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <string>

namespace infernux::bindings
{
// A UTF-16 animation can expose one half of a surrogate pair. Preserve the
// Python property and replace unpaired units only at the UTF-8 glyph boundary.
// Ordinary strings retain CPython's cached UTF-8 conversion path.
inline std::string TextForDisplay(const pybind11::str &text)
{
    Py_ssize_t size;
    if (const auto *bytes = PyUnicode_AsUTF8AndSize(text.ptr(), &size))
        return {bytes, size_t(size)};
    if (!PyErr_ExceptionMatches(PyExc_UnicodeEncodeError))
        throw pybind11::error_already_set();
    PyErr_Clear();
    const auto count = PyUnicode_GET_LENGTH(text.ptr());
    const auto kind = PyUnicode_KIND(text.ptr());
    const auto *data = PyUnicode_DATA(text.ptr());
    std::string output;
    output.reserve(size_t(count) * 3);
    for (Py_ssize_t i = 0; i < count; ++i) {
        uint32_t code = PyUnicode_READ(kind, data, i);
        if (code >= 0xd800 && code <= 0xdbff && i + 1 < count) {
            const auto low = PyUnicode_READ(kind, data, i + 1);
            if (low >= 0xdc00 && low <= 0xdfff) {
                code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                ++i;
            }
        }
        if (code >= 0xd800 && code <= 0xdfff)
            code = 0xfffd;
        if (code <= 0x7f)
            output.push_back(char(code));
        else if (code <= 0x7ff) {
            output.push_back(char(0xc0 | (code >> 6)));
            output.push_back(char(0x80 | (code & 0x3f)));
        } else if (code <= 0xffff) {
            output.push_back(char(0xe0 | (code >> 12)));
            output.push_back(char(0x80 | ((code >> 6) & 0x3f)));
            output.push_back(char(0x80 | (code & 0x3f)));
        } else {
            output.push_back(char(0xf0 | (code >> 18)));
            output.push_back(char(0x80 | ((code >> 12) & 0x3f)));
            output.push_back(char(0x80 | ((code >> 6) & 0x3f)));
            output.push_back(char(0x80 | (code & 0x3f)));
        }
    }
    return output;
}
} // namespace infernux::bindings
