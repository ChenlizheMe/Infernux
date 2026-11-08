#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernux::binary_codec
{

inline constexpr uint64_t Fnv1a64Basis = 14695981039346656037ULL;

inline uint64_t Fnv1a64(std::string_view bytes, uint64_t hash = Fnv1a64Basis)
{
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
    explicit Reader(std::string_view bytes) : m_block(bytes), m_size(bytes.size())
    {
    }
    Reader(size_t size, std::function<std::string_view()> nextBlock) : m_size(size), m_nextBlock(std::move(nextBlock))
    {
    }

    template <typename UInt> UInt ReadUnsigned()
    {
        Require(sizeof(UInt));
        std::array<char, sizeof(UInt)> split;
        const char *bytes = m_block.data();
        if (m_block.size() < sizeof(UInt)) {
            ReadInto(split.data(), split.size());
            bytes = split.data();
        } else {
            Consume(sizeof(UInt));
        }
        UInt value = 0;
        for (unsigned int shift = 0; shift < sizeof(UInt) * 8; shift += 8)
            value |= static_cast<UInt>(static_cast<UInt>(static_cast<unsigned char>(*bytes++)) << shift);
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
        return ReadString(allowEmpty, [](uint32_t) {});
    }
    template <typename Charge> std::string ReadString(bool allowEmpty, Charge charge)
    {
        const uint32_t size = ReadU32();
        if ((!allowEmpty && size == 0) || size > MaximumStringBytes)
            throw std::invalid_argument("artifact contains an invalid string length");
        Require(size);
        charge(size);
        if (size <= m_block.size()) {
            std::string value(m_block.substr(0, size));
            Consume(size);
            return value;
        }
        std::string value(size, '\0');
        ReadInto(value.data(), value.size());
        return value;
    }
    bool ReadPrefix(std::string_view prefix)
    {
        Require(prefix.size());
        for (const unsigned char expected : prefix)
            if (ReadByte() != expected)
                return false;
        return true;
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
        return m_size - m_cursor;
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
    void ReadInto(char *destination, size_t remaining)
    {
        while (remaining) {
            Fill();
            const size_t count = (std::min)(m_block.size(), remaining);
            std::memcpy(destination, m_block.data(), count);
            Consume(count);
            destination += count;
            remaining -= count;
        }
    }

    void Fill()
    {
        if (!m_block.empty())
            return;
        m_block = m_nextBlock();
        if (m_block.empty())
            throw std::invalid_argument("artifact is truncated");
    }

    void Consume(size_t count)
    {
        m_block.remove_prefix(count);
        m_cursor += count;
    }

    unsigned char ReadByte()
    {
        Fill();
        const auto value = static_cast<unsigned char>(m_block.front());
        Consume(1);
        return value;
    }

    std::string_view m_block;
    size_t m_size;
    size_t m_cursor = 0;
    std::function<std::string_view()> m_nextBlock;
};

inline std::string_view CheckedPayload(std::string_view bytes)
{
    if (bytes.size() < sizeof(uint64_t))
        throw std::invalid_argument("artifact is truncated");
    const auto payload = bytes.substr(0, bytes.size() - sizeof(uint64_t));
    if (Reader<0>(bytes.substr(payload.size())).ReadU64() != Fnv1a64(payload))
        throw std::invalid_argument("artifact checksum mismatch");
    return payload;
}

} // namespace infernux::binary_codec
