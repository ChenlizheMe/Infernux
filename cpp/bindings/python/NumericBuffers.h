#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace infernux::bindings
{
// A lock handoff costs more than a tiny numeric job, especially when another
// Python thread is busy. Keep small calls on the caller; release once for larger
// loops. This is an entry decision, never a per-element check or extra mutex.
class NumericWorkScope
{
  public:
    explicit NumericWorkScope(size_t count, size_t releaseThreshold = 512)
    {
        if (count >= releaseThreshold)
            release_.emplace();
    }

  private:
    std::optional<pybind11::gil_scoped_release> release_;
};

// Export/pin Python buffers at the boundary, then use only these native views
// while the GIL is released. No per-element validation or scratch allocation.
template <typename Input, typename Output, int Channels = 1> class NumericBuffers
{
  public:
    NumericBuffers(pybind11::buffer source, pybind11::buffer target)
        : source_(source.request()), target_(target.request(true))
    {
        count = source_.size;
        const auto outputSize = count * Channels;
        const bool flat = target_.ndim == 1 && target_.size == outputSize &&
                          (outputSize <= 1 || target_.strides[0] == sizeof(Output));
        const bool matrix = target_.ndim == 2 && target_.shape[0] == count && target_.shape[1] == Channels &&
                            (!count || (target_.strides[1] == sizeof(Output) &&
                                        (count <= 1 || target_.strides[0] == Channels * sizeof(Output))));
        if (source_.ndim != 1 || !source_.item_type_is_equivalent_to<Input>() ||
            (count > 1 && source_.strides[0] != sizeof(Input)) || !target_.item_type_is_equivalent_to<Output>() ||
            (!flat && !matrix))
            throw std::invalid_argument("numeric buffers require matching contiguous typed input/output layouts");
        input = static_cast<const char *>(source_.ptr);
        output = static_cast<char *>(target_.ptr);
        const auto from = reinterpret_cast<std::uintptr_t>(input);
        const auto to = reinterpret_cast<std::uintptr_t>(output);
        const bool inPlace = Channels == 1 && sizeof(Input) == sizeof(Output) && from == to;
        if (count && !inPlace && from < to + outputSize * sizeof(Output) && to < from + count * sizeof(Input))
            throw std::invalid_argument("numeric input/output buffers must not overlap");
    }

    Input Read(pybind11::ssize_t index) const
    {
        Input value;
        std::memcpy(&value, input + index * sizeof(Input), sizeof(Input));
        return value;
    }
    void Write(pybind11::ssize_t index, const Output *values) const
    {
        std::memcpy(output + index * Channels * sizeof(Output), values, Channels * sizeof(Output));
    }

    pybind11::ssize_t count;
    const char *input;
    char *output;

  private:
    pybind11::buffer_info source_, target_;
};
} // namespace infernux::bindings
