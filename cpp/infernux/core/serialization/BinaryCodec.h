#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace infernux::binary_codec
{

inline uint64_t Fnv1a64(std::string_view bytes)
{
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

template <typename UInt> inline void AppendUnsigned(std::string &output, UInt value)
{
    for (unsigned int shift = 0; shift < sizeof(UInt) * 8; shift += 8)
        output.push_back(static_cast<char>((value >> shift) & 0xffU));
}

inline void AppendU16(std::string &output, uint16_t value)
{
    AppendUnsigned(output, value);
}
inline void AppendU32(std::string &output, uint32_t value)
{
    AppendUnsigned(output, value);
}
inline void AppendU64(std::string &output, uint64_t value)
{
    AppendUnsigned(output, value);
}

inline void AppendFloat(std::string &output, float value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("artifact contains a non-finite float");
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    AppendU32(output, bits);
}

template <uint32_t MaximumStringBytes, bool AllowEmptyString = true> class Reader
{
  public:
    explicit Reader(std::string_view bytes) : m_bytes(bytes)
    {
    }

    template <typename UInt> UInt ReadUnsigned()
    {
        Require(sizeof(UInt));
        UInt value = 0;
        for (unsigned int shift = 0; shift < sizeof(UInt) * 8; shift += 8)
            value |= static_cast<UInt>(static_cast<UInt>(static_cast<unsigned char>(m_bytes[m_cursor++])) << shift);
        return value;
    }
    uint16_t ReadU16()
    {
        return ReadUnsigned<uint16_t>();
    }
    uint32_t ReadU32()
    {
        return ReadUnsigned<uint32_t>();
    }
    uint64_t ReadU64()
    {
        return ReadUnsigned<uint64_t>();
    }
    int32_t ReadI32()
    {
        const uint32_t bits = ReadU32();
        int32_t value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    float ReadFloat()
    {
        const uint32_t bits = ReadU32();
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value))
            throw std::invalid_argument("artifact contains a non-finite float");
        return value;
    }
    double ReadDouble()
    {
        const uint64_t bits = ReadU64();
        double value;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value))
            throw std::invalid_argument("artifact contains a non-finite double");
        return value;
    }
    std::string ReadString(bool allowEmpty = AllowEmptyString)
    {
        const uint32_t size = ReadU32();
        if ((!allowEmpty && size == 0) || size > MaximumStringBytes)
            throw std::invalid_argument("artifact contains an invalid string length");
        return std::string(ReadBytes(size));
    }
    std::string_view ReadBytes(uint64_t size)
    {
        if (size > std::numeric_limits<size_t>::max())
            throw std::invalid_argument("artifact payload exceeds addressable memory");
        Require(static_cast<size_t>(size));
        const auto result = m_bytes.substr(m_cursor, static_cast<size_t>(size));
        m_cursor += static_cast<size_t>(size);
        return result;
    }
    uint32_t ReadCount(uint32_t maximum = 100'000'000U, size_t minimumElementBytes = 1)
    {
        const uint32_t count = ReadU32();
        if (count > maximum || (minimumElementBytes != 0 && count > Remaining() / minimumElementBytes))
            throw std::invalid_argument("artifact count exceeds its format or payload limit");
        return count;
    }
    size_t Cursor() const noexcept
    {
        return m_cursor;
    }
    size_t Remaining() const noexcept
    {
        return m_bytes.size() - m_cursor;
    }
    bool AtEnd() const noexcept
    {
        return Remaining() == 0;
    }
    void Require(size_t size) const
    {
        if (size > Remaining())
            throw std::invalid_argument("artifact is truncated");
    }

  private:
    std::string_view m_bytes;
    size_t m_cursor = 0;
};

} // namespace infernux::binary_codec
