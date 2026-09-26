#pragma once

#include <function/resources/InxMesh/InxMesh.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace infernux
{
namespace mesh_geometry_codec
{
struct Precision
{
    uint32_t position;
    uint32_t direction;
    uint32_t color;
    uint32_t uv;
    uint32_t weight;
};

inline Precision GetPrecision(MeshCompression compression)
{
    switch (compression) {
    case MeshCompression::Low:
        return {20, 10, 10, 16, 12};
    case MeshCompression::Medium:
        return {16, 8, 8, 12, 10};
    case MeshCompression::High:
        return {10, 6, 6, 8, 8};
    default:
        throw std::invalid_argument("mesh geometry quantization requires a compression level");
    }
}

inline void AppendU32(std::string &out, uint32_t value)
{
    for (unsigned shift = 0; shift < 32; shift += 8)
        out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

inline uint32_t ReadU32(std::string_view bytes, size_t &cursor)
{
    if (cursor > bytes.size() || sizeof(uint32_t) > bytes.size() - cursor)
        throw std::invalid_argument("compressed mesh vertex stream is truncated");
    uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[cursor++])) << shift;
    return value;
}

inline void AppendFloat(std::string &out, float value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("mesh compression requires finite vertex data");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    AppendU32(out, bits);
}

inline float ReadFloat(std::string_view bytes, size_t &cursor)
{
    const uint32_t bits = ReadU32(bytes, cursor);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value))
        throw std::invalid_argument("compressed mesh vertex stream contains a non-finite range");
    return value;
}

class BitWriter final
{
  public:
    void Write(uint32_t value, uint32_t bits)
    {
        if (bits == 0 || bits > 32 || (bits < 32 && value >= (uint32_t{1} << bits)))
            throw std::invalid_argument("mesh quantization value exceeds its bit width");
        for (uint32_t bit = 0; bit < bits; ++bit) {
            if (m_bit == 0)
                m_bytes.push_back('\0');
            if ((value >> bit) & 1U)
                m_bytes.back() = static_cast<char>(static_cast<unsigned char>(m_bytes.back()) | (1U << m_bit));
            m_bit = (m_bit + 1U) & 7U;
        }
    }
    [[nodiscard]] std::string Take()
    {
        return std::move(m_bytes);
    }

  private:
    std::string m_bytes;
    uint32_t m_bit = 0;
};

class BitReader final
{
  public:
    explicit BitReader(std::string_view bytes) : m_bytes(bytes)
    {
    }
    uint32_t Read(uint32_t bits)
    {
        if (bits == 0 || bits > 32 || m_cursor > m_bytes.size() * 8ULL || bits > m_bytes.size() * 8ULL - m_cursor)
            throw std::invalid_argument("compressed mesh vertex bitstream is truncated");
        uint32_t value = 0;
        for (uint32_t bit = 0; bit < bits; ++bit, ++m_cursor)
            value |= ((static_cast<unsigned char>(m_bytes[m_cursor / 8U]) >> (m_cursor & 7U)) & 1U) << bit;
        return value;
    }
    [[nodiscard]] size_t ConsumedBytes() const noexcept
    {
        return (m_cursor + 7U) / 8U;
    }

  private:
    std::string_view m_bytes;
    size_t m_cursor = 0;
};

inline uint32_t Quantize(float value, float minimum, float maximum, uint32_t bits)
{
    if (!std::isfinite(value) || !std::isfinite(minimum) || !std::isfinite(maximum) || maximum < minimum)
        throw std::invalid_argument("mesh compression requires finite ordered channel ranges");
    if (maximum == minimum)
        return 0;
    const uint32_t levels = bits == 32 ? std::numeric_limits<uint32_t>::max() : (uint32_t{1} << bits) - 1U;
    const double span = static_cast<double>(maximum) - static_cast<double>(minimum);
    const double normalized = std::clamp((static_cast<double>(value) - static_cast<double>(minimum)) / span, 0.0, 1.0);
    if (!std::isfinite(normalized))
        throw std::invalid_argument("mesh compression produced a non-finite normalized channel");
    return static_cast<uint32_t>(std::llround(normalized * levels));
}

inline float Dequantize(uint32_t value, float minimum, float maximum, uint32_t bits)
{
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || maximum < minimum)
        throw std::invalid_argument("compressed mesh vertex stream contains an invalid channel range");
    if (maximum == minimum)
        return minimum;
    const uint32_t levels = bits == 32 ? std::numeric_limits<uint32_t>::max() : (uint32_t{1} << bits) - 1U;
    if (value > levels)
        throw std::invalid_argument("compressed mesh vertex stream contains an out-of-range quantized value");
    const double decoded =
        static_cast<double>(minimum) + (static_cast<double>(maximum) - static_cast<double>(minimum)) *
                                           (static_cast<double>(value) / static_cast<double>(levels));
    if (!std::isfinite(decoded) || decoded < -static_cast<double>(std::numeric_limits<float>::max()) ||
        decoded > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::invalid_argument("compressed mesh vertex stream decodes to a non-finite channel");
    const float result = static_cast<float>(decoded);
    if (!std::isfinite(result))
        throw std::invalid_argument("compressed mesh vertex stream decodes outside the runtime float range");
    return result;
}

template <glm::length_t Length, typename T, glm::qualifier Q> inline bool IsFinite(const glm::vec<Length, T, Q> &value)
{
    for (glm::length_t component = 0; component < Length; ++component) {
        if (!std::isfinite(value[component]))
            return false;
    }
    return true;
}

inline glm::vec3 RequireUnitDirection(const glm::vec3 &value, const char *channel)
{
    if (!IsFinite(value))
        throw std::invalid_argument(std::string("compressed mesh vertex stream contains a non-finite ") + channel);
    const double lengthSquared = static_cast<double>(value.x) * value.x + static_cast<double>(value.y) * value.y +
                                 static_cast<double>(value.z) * value.z;
    if (!std::isfinite(lengthSquared) || lengthSquared == 0.0)
        throw std::invalid_argument(std::string("compressed mesh vertex stream contains a zero-length ") + channel);
    const double inverseLength = 1.0 / std::sqrt(lengthSquared);
    const glm::vec3 result(static_cast<float>(static_cast<double>(value.x) * inverseLength),
                           static_cast<float>(static_cast<double>(value.y) * inverseLength),
                           static_cast<float>(static_cast<double>(value.z) * inverseLength));
    if (!IsFinite(result))
        throw std::invalid_argument(std::string("compressed mesh vertex stream cannot normalize ") + channel);
    return result;
}

inline bool IsMissingDirection(const glm::vec3 &value) noexcept
{
    return value.x == 0.0f && value.y == 0.0f && value.z == 0.0f;
}

struct Ranges
{
    glm::vec3 positionMin{std::numeric_limits<float>::max()};
    glm::vec3 positionMax{std::numeric_limits<float>::lowest()};
    glm::vec3 colorMin{std::numeric_limits<float>::max()};
    glm::vec3 colorMax{std::numeric_limits<float>::lowest()};
    glm::vec2 uvMin{std::numeric_limits<float>::max()};
    glm::vec2 uvMax{std::numeric_limits<float>::lowest()};
    glm::vec2 uv1Min{std::numeric_limits<float>::max()};
    glm::vec2 uv1Max{std::numeric_limits<float>::lowest()};
};

inline std::string Encode(const std::vector<Vertex> &vertices, MeshCompression compression, bool secondaryUv = false)
{
    if (compression == MeshCompression::Off)
        throw std::invalid_argument("uncompressed vertices do not use the quantized codec");
    const Precision precision = GetPrecision(compression);
    Ranges ranges;
    if (vertices.empty()) {
        ranges.positionMin = ranges.positionMax = glm::vec3(0.0f);
        ranges.colorMin = ranges.colorMax = glm::vec3(0.0f);
        ranges.uvMin = ranges.uvMax = glm::vec2(0.0f);
        ranges.uv1Min = ranges.uv1Max = glm::vec2(0.0f);
    }
    for (const auto &vertex : vertices) {
        if (!IsFinite(vertex.pos) || !IsFinite(vertex.normal) || !IsFinite(vertex.tangent) || !IsFinite(vertex.color) ||
            !IsFinite(vertex.texCoord) || !IsFinite(vertex.texCoord1) || !IsFinite(vertex.boneWeights))
            throw std::invalid_argument("mesh compression requires finite vertex data");
        // A zero direction is the fixed-layout representation of a channel
        // intentionally omitted by normal_mode/tangent_mode=None. GEO2 keeps
        // that state explicitly; only authored directions are normalized.
        if (!IsMissingDirection(vertex.normal))
            (void)RequireUnitDirection(vertex.normal, "normal");
        const bool tangentPresent = !IsMissingDirection(glm::vec3(vertex.tangent));
        if (tangentPresent) {
            (void)RequireUnitDirection(glm::vec3(vertex.tangent), "tangent");
            if (vertex.tangent.w != -1.0f && vertex.tangent.w != 1.0f)
                throw std::invalid_argument("mesh compression requires tangent handedness to be -1 or +1");
        } else if (vertex.tangent.w != 0.0f) {
            throw std::invalid_argument("mesh compression requires an omitted tangent to be exactly zero");
        }
        ranges.positionMin = glm::min(ranges.positionMin, vertex.pos);
        ranges.positionMax = glm::max(ranges.positionMax, vertex.pos);
        ranges.colorMin = glm::min(ranges.colorMin, vertex.color);
        ranges.colorMax = glm::max(ranges.colorMax, vertex.color);
        ranges.uvMin = glm::min(ranges.uvMin, vertex.texCoord);
        ranges.uvMax = glm::max(ranges.uvMax, vertex.texCoord);
        ranges.uv1Min = glm::min(ranges.uv1Min, vertex.texCoord1);
        ranges.uv1Max = glm::max(ranges.uv1Max, vertex.texCoord1);
    }
    std::string output;
    for (glm::length_t i = 0; i < 3; ++i) {
        AppendFloat(output, ranges.positionMin[i]);
        AppendFloat(output, ranges.positionMax[i]);
    }
    for (glm::length_t i = 0; i < 3; ++i) {
        AppendFloat(output, ranges.colorMin[i]);
        AppendFloat(output, ranges.colorMax[i]);
    }
    for (glm::length_t i = 0; i < 2; ++i) {
        AppendFloat(output, ranges.uvMin[i]);
        AppendFloat(output, ranges.uvMax[i]);
    }
    if (secondaryUv)
        for (glm::length_t i = 0; i < 2; ++i) {
            AppendFloat(output, ranges.uv1Min[i]);
            AppendFloat(output, ranges.uv1Max[i]);
        }
    BitWriter writer;
    for (const auto &vertex : vertices) {
        for (glm::length_t i = 0; i < 3; ++i)
            writer.Write(Quantize(vertex.pos[i], ranges.positionMin[i], ranges.positionMax[i], precision.position),
                         precision.position);
        const bool normalPresent = !IsMissingDirection(vertex.normal);
        writer.Write(normalPresent ? 1U : 0U, 1);
        if (normalPresent) {
            for (glm::length_t i = 0; i < 3; ++i)
                writer.Write(Quantize(vertex.normal[i], -1.0f, 1.0f, precision.direction), precision.direction);
        }
        const bool tangentPresent = !IsMissingDirection(glm::vec3(vertex.tangent));
        writer.Write(tangentPresent ? 1U : 0U, 1);
        if (tangentPresent) {
            for (glm::length_t i = 0; i < 3; ++i)
                writer.Write(Quantize(vertex.tangent[i], -1.0f, 1.0f, precision.direction), precision.direction);
            writer.Write(vertex.tangent.w < 0.0f ? 0U : 1U, 1);
        }
        for (glm::length_t i = 0; i < 3; ++i)
            writer.Write(Quantize(vertex.color[i], ranges.colorMin[i], ranges.colorMax[i], precision.color),
                         precision.color);
        for (glm::length_t i = 0; i < 2; ++i)
            writer.Write(Quantize(vertex.texCoord[i], ranges.uvMin[i], ranges.uvMax[i], precision.uv), precision.uv);
        if (secondaryUv)
            for (glm::length_t i = 0; i < 2; ++i)
                writer.Write(Quantize(vertex.texCoord1[i], ranges.uv1Min[i], ranges.uv1Max[i], precision.uv),
                             precision.uv);
        for (glm::length_t i = 0; i < 4; ++i)
            writer.Write(vertex.boneIndices[i], 32);
        for (glm::length_t i = 0; i < 4; ++i)
            writer.Write(Quantize(vertex.boneWeights[i], 0.0f, 1.0f, precision.weight), precision.weight);
    }
    output += writer.Take();
    return output;
}

inline std::vector<Vertex> Decode(std::string_view bytes, size_t vertexCount, MeshCompression compression,
                                  bool secondaryUv = false)
{
    if (compression == MeshCompression::Off)
        throw std::invalid_argument("uncompressed vertices do not use the quantized codec");
    const Precision precision = GetPrecision(compression);
    size_t cursor = 0;
    Ranges ranges;
    for (glm::length_t i = 0; i < 3; ++i) {
        ranges.positionMin[i] = ReadFloat(bytes, cursor);
        ranges.positionMax[i] = ReadFloat(bytes, cursor);
    }
    for (glm::length_t i = 0; i < 3; ++i) {
        ranges.colorMin[i] = ReadFloat(bytes, cursor);
        ranges.colorMax[i] = ReadFloat(bytes, cursor);
    }
    for (glm::length_t i = 0; i < 2; ++i) {
        ranges.uvMin[i] = ReadFloat(bytes, cursor);
        ranges.uvMax[i] = ReadFloat(bytes, cursor);
    }
    if (secondaryUv)
        for (glm::length_t i = 0; i < 2; ++i) {
            ranges.uv1Min[i] = ReadFloat(bytes, cursor);
            ranges.uv1Max[i] = ReadFloat(bytes, cursor);
        }
    for (glm::length_t i = 0; i < 3; ++i) {
        if (ranges.positionMax[i] < ranges.positionMin[i] || ranges.colorMax[i] < ranges.colorMin[i])
            throw std::invalid_argument("compressed mesh vertex stream contains a reversed channel range");
    }
    for (glm::length_t i = 0; i < 2; ++i) {
        if (ranges.uvMax[i] < ranges.uvMin[i])
            throw std::invalid_argument("compressed mesh vertex stream contains a reversed channel range");
        if (secondaryUv && ranges.uv1Max[i] < ranges.uv1Min[i])
            throw std::invalid_argument("compressed mesh vertex stream contains a reversed secondary UV range");
    }
    BitReader reader(bytes.substr(cursor));
    std::vector<Vertex> vertices(vertexCount);
    for (auto &vertex : vertices) {
        for (glm::length_t i = 0; i < 3; ++i)
            vertex.pos[i] = Dequantize(reader.Read(precision.position), ranges.positionMin[i], ranges.positionMax[i],
                                       precision.position);
        if (reader.Read(1)) {
            for (glm::length_t i = 0; i < 3; ++i)
                vertex.normal[i] = Dequantize(reader.Read(precision.direction), -1.0f, 1.0f, precision.direction);
            vertex.normal = RequireUnitDirection(vertex.normal, "normal");
        } else {
            vertex.normal = glm::vec3(0.0f);
        }
        if (reader.Read(1)) {
            for (glm::length_t i = 0; i < 3; ++i)
                vertex.tangent[i] = Dequantize(reader.Read(precision.direction), -1.0f, 1.0f, precision.direction);
            const glm::vec3 tangent = RequireUnitDirection(glm::vec3(vertex.tangent), "tangent");
            vertex.tangent.x = tangent.x;
            vertex.tangent.y = tangent.y;
            vertex.tangent.z = tangent.z;
            vertex.tangent.w = reader.Read(1) ? 1.0f : -1.0f;
        } else {
            vertex.tangent = glm::vec4(0.0f);
        }
        for (glm::length_t i = 0; i < 3; ++i)
            vertex.color[i] =
                Dequantize(reader.Read(precision.color), ranges.colorMin[i], ranges.colorMax[i], precision.color);
        for (glm::length_t i = 0; i < 2; ++i)
            vertex.texCoord[i] = Dequantize(reader.Read(precision.uv), ranges.uvMin[i], ranges.uvMax[i], precision.uv);
        if (secondaryUv)
            for (glm::length_t i = 0; i < 2; ++i)
                vertex.texCoord1[i] =
                    Dequantize(reader.Read(precision.uv), ranges.uv1Min[i], ranges.uv1Max[i], precision.uv);
        for (glm::length_t i = 0; i < 4; ++i)
            vertex.boneIndices[i] = reader.Read(32);
        for (glm::length_t i = 0; i < 4; ++i)
            vertex.boneWeights[i] = Dequantize(reader.Read(precision.weight), 0.0f, 1.0f, precision.weight);
        if (!IsFinite(vertex.pos) || !IsFinite(vertex.color) || !IsFinite(vertex.texCoord) ||
            !IsFinite(vertex.texCoord1) || !IsFinite(vertex.boneWeights))
            throw std::invalid_argument("compressed mesh vertex stream decoded non-finite vertex data");
    }
    if (cursor + reader.ConsumedBytes() != bytes.size())
        throw std::invalid_argument("compressed mesh vertex stream contains trailing data");
    return vertices;
}
} // namespace mesh_geometry_codec
} // namespace infernux
