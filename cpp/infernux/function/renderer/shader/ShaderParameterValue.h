#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infernux
{

enum class ShaderParameterKind : uint8_t
{
    Float,
    Int,
    UInt,
    Bool,
    Vector2,
    Vector3,
    Vector4,
    Matrix3,
    Matrix4,
};

/// A small, owning logical value. Matrices use [row, column] order; target
/// padding and matrix order belong to the reflected ShaderValueCopy plan.
/// Large/variable data uses graph Buffer ports rather than parameter values.
class ShaderParameterValue
{
  public:
    ShaderParameterValue() = default;
    ShaderParameterValue(float value) : m_kind(ShaderParameterKind::Float)
    {
        Store(value);
    }
    ShaderParameterValue(int32_t value) : m_kind(ShaderParameterKind::Int)
    {
        Store(value);
    }
    ShaderParameterValue(uint32_t value) : m_kind(ShaderParameterKind::UInt)
    {
        Store(value);
    }
    ShaderParameterValue(bool value) : m_kind(ShaderParameterKind::Bool)
    {
        m_words[0] = value ? 1u : 0u;
    }

    template <size_t N> ShaderParameterValue(const std::array<float, N> &value)
    {
        static_assert(N == 2 || N == 3 || N == 4 || N == 9 || N == 16, "Unsupported shader value shape");
        if constexpr (N == 2)
            m_kind = ShaderParameterKind::Vector2;
        else if constexpr (N == 3)
            m_kind = ShaderParameterKind::Vector3;
        else if constexpr (N == 4)
            m_kind = ShaderParameterKind::Vector4;
        else if constexpr (N == 9)
            m_kind = ShaderParameterKind::Matrix3;
        else
            m_kind = ShaderParameterKind::Matrix4;
        std::memcpy(m_words.data(), value.data(), sizeof(value));
    }

    [[nodiscard]] ShaderParameterKind Kind() const noexcept
    {
        return m_kind;
    }
    [[nodiscard]] const void *Data() const noexcept
    {
        return m_words.data();
    }
    [[nodiscard]] size_t ByteSize() const noexcept
    {
        constexpr size_t sizes[] = {4, 4, 4, 4, 8, 12, 16, 36, 64};
        return sizes[static_cast<size_t>(m_kind)];
    }

    template <typename T> [[nodiscard]] T Read() const noexcept
    {
        static_assert(sizeof(T) <= sizeof(m_words));
        T result;
        std::memcpy(&result, m_words.data(), sizeof(result));
        return result;
    }

    /// Normalize at value publication, never while recording GPU commands.
    [[nodiscard]] ShaderParameterValue ConvertTo(ShaderParameterKind kind) const
    {
        if (kind == m_kind)
            return *this;
        if (kind == ShaderParameterKind::Float) {
            if (m_kind == ShaderParameterKind::Int)
                return static_cast<float>(Read<int32_t>());
            if (m_kind == ShaderParameterKind::UInt)
                return static_cast<float>(Read<uint32_t>());
        }
        if (kind == ShaderParameterKind::UInt && m_kind == ShaderParameterKind::Int && Read<int32_t>() >= 0)
            return static_cast<uint32_t>(Read<int32_t>());
        if (kind == ShaderParameterKind::Int && m_kind == ShaderParameterKind::UInt &&
            Read<uint32_t>() <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
            return static_cast<int32_t>(Read<uint32_t>());
        throw std::invalid_argument("Shader parameter value does not match its declared type or shape");
    }

    /// Scalar numeric view for diagnostics. GPU uploads retain the original
    /// type and consume Data() through the reflected copy plan.
    [[nodiscard]] float ScalarFloat() const
    {
        switch (m_kind) {
        case ShaderParameterKind::Float:
            return Read<float>();
        case ShaderParameterKind::Int:
            return static_cast<float>(Read<int32_t>());
        case ShaderParameterKind::UInt:
            return static_cast<float>(Read<uint32_t>());
        case ShaderParameterKind::Bool:
            return m_words[0] ? 1.0f : 0.0f;
        default:
            throw std::invalid_argument("This shader program accepts only scalar parameters");
        }
    }

    friend bool operator==(const ShaderParameterValue &left, const ShaderParameterValue &right) noexcept
    {
        return left.m_kind == right.m_kind && std::memcmp(left.Data(), right.Data(), left.ByteSize()) == 0;
    }

  private:
    template <typename T> void Store(T value) noexcept
    {
        std::memcpy(m_words.data(), &value, sizeof(value));
    }
    ShaderParameterKind m_kind = ShaderParameterKind::Float;
    std::array<uint32_t, 16> m_words{};
};

} // namespace infernux
