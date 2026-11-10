#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace infernux::tween
{
class NumericOverflow : public std::overflow_error
{
  public:
    NumericOverflow() : std::overflow_error("Unsigned scalar arithmetic overflow")
    {
    }
    explicit NumericOverflow(const char *message) : std::overflow_error(message)
    {
    }
};

namespace detail
{
// Bounded scratch arithmetic for a 96-bit decimal coefficient and a 64-bit
// integer operand. Five words cover every aligned intermediate; no allocation,
// compiler-specific 128-bit type, or optional numerical backend is needed.
struct UInt160
{
    std::array<uint32_t, 5> words{};
    explicit UInt160(uint64_t value = 0) : words{uint32_t(value), uint32_t(value >> 32), 0, 0, 0}
    {
    }
    bool Fits96() const
    {
        return (words[3] | words[4]) == 0;
    }
    bool Fits64() const
    {
        return (words[2] | words[3] | words[4]) == 0;
    }
    bool Maximum96() const
    {
        return Fits96() && (words[0] & words[1] & words[2]) == UINT32_MAX;
    }
    bool Odd() const
    {
        return words[0] & 1;
    }
    uint64_t Low64() const
    {
        return uint64_t(words[0]) | (uint64_t(words[1]) << 32);
    }
    int Compare(const UInt160 &other) const
    {
        for (int i = 4; i >= 0; --i)
            if (words[i] != other.words[i])
                return words[i] > other.words[i] ? 1 : -1;
        return 0;
    }
    void Add(const UInt160 &other)
    {
        uint64_t carry = 0;
        for (int i = 0; i < 5; ++i) {
            const uint64_t sum = uint64_t(words[i]) + other.words[i] + carry;
            words[i] = uint32_t(sum);
            carry = sum >> 32;
        }
    }
    void Subtract(const UInt160 &other)
    {
        uint64_t borrow = 0;
        for (int i = 0; i < 5; ++i) {
            const uint64_t difference = uint64_t(words[i]) - other.words[i] - borrow;
            words[i] = uint32_t(difference);
            borrow = difference >> 63;
        }
    }
    void Multiply32(uint32_t value)
    {
        uint64_t carry = 0;
        for (int i = 0; i < 5; ++i) {
            const uint64_t product = uint64_t(words[i]) * value + carry;
            words[i] = uint32_t(product);
            carry = product >> 32;
        }
    }
    void Multiply64(uint64_t value)
    {
        UInt160 high = *this;
        Multiply32(uint32_t(value));
        high.Multiply32(uint32_t(value >> 32));
        for (int i = 4; i > 0; --i)
            high.words[i] = high.words[i - 1];
        high.words[0] = 0;
        Add(high);
    }
    void MultiplyPower10(int exponent)
    {
        while (exponent--)
            Multiply32(10);
    }
    uint32_t Divide10()
    {
        uint64_t remainder = 0;
        for (int i = 4; i >= 0; --i) {
            const uint64_t value = (remainder << 32) | words[i];
            words[i] = uint32_t(value / 10);
            remainder = value % 10;
        }
        return uint32_t(remainder);
    }
};

inline bool RoundUp(uint32_t digit, bool sticky, const UInt160 &coefficient)
{
    return digit > 5 || (digit == 5 && (sticky || coefficient.Odd()));
}

inline void FitDecimal(UInt160 &coefficient, int &scale)
{
    uint32_t digit = 0;
    bool sticky = false, removed = false;
    while (!coefficient.Fits96() || (removed && coefficient.Maximum96() && RoundUp(digit, sticky, coefficient))) {
        if (!scale)
            throw NumericOverflow();
        sticky = sticky || digit != 0;
        digit = coefficient.Divide10();
        --scale;
        removed = true;
    }
    if (removed && RoundUp(digit, sticky, coefficient))
        coefficient.Add(UInt160(1));
}

struct DecimalFactor
{
    UInt160 coefficient;
    int scale;
    bool negative;
};

inline DecimalFactor ToDecimalFactor(float value)
{
    const double absolute = std::abs(static_cast<double>(value));
    if (!(absolute < 79228162514264337593543950336.0))
        throw NumericOverflow();
    if (absolute == 0)
        return {UInt160(), 0, false};
    constexpr double powers[] = {1,    1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,
                                 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
                                 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28};
    int scale = std::min(28, 6 - static_cast<int>(std::floor(std::log10(absolute))));
    uint64_t coefficient =
        static_cast<uint64_t>(std::nearbyint(scale >= 0 ? absolute * powers[scale] : absolute / powers[-scale]));
    if (!coefficient)
        return {UInt160(), 0, false};
    if (scale >= 0) {
        while (scale && coefficient && coefficient % 10 == 0) {
            coefficient /= 10;
            --scale;
        }
        return {UInt160(coefficient), scale, std::signbit(value)};
    }
    UInt160 result(coefficient);
    result.MultiplyPower10(-scale);
    return {result, 0, std::signbit(value)};
}

inline uint64_t DecimalUInt64Interpolate(uint64_t start, uint64_t change, float factor)
{
    if (factor == 0)
        return start;
    if (factor == 1) {
        if (change > UINT64_MAX - start)
            throw NumericOverflow();
        return start + change;
    }
    if (factor == -1) {
        if (change > start)
            throw NumericOverflow();
        return start - change;
    }
    auto value = ToDecimalFactor(factor);
    value.coefficient.Multiply64(change);
    FitDecimal(value.coefficient, value.scale);
    UInt160 initial(start);
    initial.MultiplyPower10(value.scale);
    if (!value.negative) {
        value.coefficient.Add(initial);
    } else if (initial.Compare(value.coefficient) >= 0) {
        initial.Subtract(value.coefficient);
        value.coefficient = initial;
        value.negative = false;
    } else {
        value.coefficient.Subtract(initial);
    }
    FitDecimal(value.coefficient, value.scale);
    while (value.scale--)
        value.coefficient.Divide10();
    if (!value.coefficient.Fits64() || (value.negative && value.coefficient.Low64() != 0))
        throw NumericOverflow();
    return value.coefficient.Low64();
}
} // namespace detail
} // namespace infernux::tween
