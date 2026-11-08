#include "BindingRegistration.h"
#include "NumericBuffers.h"
#include "infernux/core/math/StringTween.h"
#include <cstring>
#include <optional>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace infernux
{
namespace
{
// Text copying is much cheaper per unit than numeric transcendental work.
// Releasing for a 512-unit label caused millisecond GIL handoff outliers under
// contention. Keep ordinary UI strings local; release once for large text jobs.
constexpr size_t StringReleaseThreshold = 1 << 20;

// These binding-only owners retain immutable Python text while the independent
// C++ plan remains Python-free. Refcounts and Unicode operations stay under GIL.
struct BoundStringSnapshot
{
    py::str start, end;
    tween::StringSnapshot native;
};

struct BoundStringSampler
{
    py::str start, end;
    tween::StringSampler native;
    bool asciiSlices;

    py::str Slice(double factor) const
    {
        const auto ranges = native.PlainSlices(factor);
        PyObject *result;
        if (!ranges.startCount)
            result = PyUnicode_Substring(end.ptr(), 0, ranges.shown);
        else if (!ranges.shown)
            result = PyUnicode_Substring(start.ptr(), ranges.startBegin, ranges.startBegin + ranges.startCount);
        else {
            result = PyUnicode_New(ranges.shown + ranges.startCount, 127);
            if (result) {
                auto *output = PyUnicode_1BYTE_DATA(result);
                std::memcpy(output, PyUnicode_1BYTE_DATA(end.ptr()), ranges.shown);
                std::memcpy(output + ranges.shown, PyUnicode_1BYTE_DATA(start.ptr()) + ranges.startBegin,
                            ranges.startCount);
            }
        }
        if (!result)
            throw py::error_already_set();
        return py::reinterpret_steal<py::str>(result);
    }
};

std::u16string ReadText(const py::str &text)
{
    const auto length = PyUnicode_GET_LENGTH(text.ptr());
    const int kind = PyUnicode_KIND(text.ptr());
    const void *data = PyUnicode_DATA(text.ptr());
    std::u16string result;
    result.reserve(length);
    for (Py_ssize_t i = 0; i < length; ++i) {
        const auto codepoint = PyUnicode_READ(kind, data, i);
        if (codepoint <= 0xffff)
            result.push_back(static_cast<char16_t>(codepoint));
        else {
            result.push_back(static_cast<char16_t>(0xd800 + ((codepoint - 0x10000) >> 10)));
            result.push_back(static_cast<char16_t>(0xdc00 + ((codepoint - 0x10000) & 0x3ff)));
        }
    }
    return result;
}

py::str WriteText(const std::u16string &text)
{
    const uint16_t marker = 1;
    int order = *reinterpret_cast<const unsigned char *>(&marker) ? -1 : 1;
    auto *result =
        PyUnicode_DecodeUTF16(reinterpret_cast<const char *>(text.data()),
                              static_cast<Py_ssize_t>(text.size() * sizeof(char16_t)), "surrogatepass", &order);
    if (!result)
        throw py::error_already_set();
    return py::reinterpret_steal<py::str>(result);
}
} // namespace

void RegisterStringTweenBindings(py::module_ &module)
{
    using namespace tween;
    py::class_<StringAlphabet, std::shared_ptr<StringAlphabet>>(module, "_StringScrambleAlphabet")
        .def(py::init([](const py::str &text, bool shuffle) {
                 auto units = ReadText(text);
                 const bindings::NumericWorkScope work(units.size(), StringReleaseThreshold);
                 return std::make_shared<StringAlphabet>(std::move(units), shuffle);
             }),
             py::arg("text"), py::arg("shuffle") = true)
        .def_property_readonly("characters", [](const StringAlphabet &self) {
            py::list result;
            for (char16_t unit : self.Text())
                result.append(WriteText(std::u16string(1, unit)));
            return result;
        });
    py::class_<BoundStringSampler>(module, "_StringTweenSampler")
        .def(
            "__call__",
            [](const BoundStringSampler &self, double progress, int cycles) {
                if (self.asciiSlices)
                    return self.Slice(self.native.Factor(progress));
                std::u16string result;
                {
                    const bindings::NumericWorkScope work(self.native.WorkSize(cycles), StringReleaseThreshold);
                    result = self.native.Sample(progress, cycles);
                }
                return WriteText(result);
            },
            py::arg("progress"), py::arg("cycles") = 0)
        .def(
            "interpolate",
            [](const BoundStringSampler &self, double factor, int cycles) {
                if (self.asciiSlices)
                    return self.Slice(factor);
                std::u16string result;
                {
                    const bindings::NumericWorkScope work(self.native.WorkSize(cycles), StringReleaseThreshold);
                    result = self.native.Interpolate(factor, cycles);
                }
                return WriteText(result);
            },
            py::arg("factor"), py::arg("cycles") = 0);
    py::class_<BoundStringSnapshot>(module, "_StringTweenSnapshot")
        .def(py::init([](const py::str &start, const py::str &end) {
            auto first = ReadText(start), last = ReadText(end);
            std::optional<StringSnapshot> snapshot;
            {
                const bindings::NumericWorkScope work(first.size() + last.size(), StringReleaseThreshold);
                snapshot.emplace(std::move(first), std::move(last));
            }
            return BoundStringSnapshot{start, end, std::move(*snapshot)};
        }))
        .def(
            "length", [](const BoundStringSnapshot &self, bool rich) { return self.native.Length(rich); },
            py::arg("rich_text") = false)
        .def(
            "start_length", [](const BoundStringSnapshot &self, bool rich) { return self.native.StartLength(rich); },
            py::arg("rich_text") = false)
        .def(
            "bind",
            [](const BoundStringSnapshot &self, EaseKernel ease, double duration, bool rich, bool relative, int mode,
               std::shared_ptr<const StringAlphabet> alphabet, int startLength, int changeLength) {
                const bool ascii = !rich && !relative && mode == 0 && PyUnicode_IS_ASCII(self.start.ptr()) &&
                                   PyUnicode_IS_ASCII(self.end.ptr()) &&
                                   size_t(PyUnicode_GET_LENGTH(self.start.ptr()) +
                                          PyUnicode_GET_LENGTH(self.end.ptr())) < StringReleaseThreshold;
                return BoundStringSampler{self.start, self.end,
                                          self.native.Bind(ease, duration, rich, relative, mode, std::move(alphabet),
                                                           startLength, changeLength),
                                          ascii};
            },
            py::arg("ease"), py::arg("duration"), py::arg("rich_text") = false, py::arg("relative") = false,
            py::arg("scramble_mode") = 0, py::arg("scramble_alphabet") = nullptr, py::arg("start_length") = -1,
            py::arg("change_length") = -1);
}
} // namespace infernux
