#pragma once

#include "ShaderProgramLayout.h"

#include <cstring>
#include <stdexcept>

namespace infernux
{

enum class ShaderMatrixOrder : uint8_t
{
    RowMajor,
    ColumnMajor,
};

/// Prepare a copy from tightly packed logical values to a target's reflected
/// storage. Build once with the program; upload only executes memcpy regions.
/// Arrays/structs concatenate their logical elements without target padding.
class ShaderValueCopy
{
  public:
    explicit ShaderValueCopy(const ShaderValueLayout &layout, ShaderMatrixOrder order = ShaderMatrixOrder::RowMajor)
    {
        Append(layout, layout.offset, order);
    }

    [[nodiscard]] size_t GetSourceBytes() const noexcept
    {
        return m_sourceBytes;
    }

    void Write(void *target, const void *source) const noexcept
    {
        for (const auto &region : m_regions)
            std::memcpy(static_cast<char *>(target) + region.target, static_cast<const char *>(source) + region.source,
                        region.bytes);
    }

  private:
    struct Region
    {
        size_t source;
        size_t target;
        size_t bytes;
    };
    size_t m_sourceBytes = 0;
    std::vector<Region> m_regions;

    static size_t ScalarBytes(ShaderScalarKind kind)
    {
        switch (kind) {
        case ShaderScalarKind::Int8:
        case ShaderScalarKind::UInt8:
            return 1;
        case ShaderScalarKind::Int16:
        case ShaderScalarKind::UInt16:
        case ShaderScalarKind::Float16:
            return 2;
        case ShaderScalarKind::Bool:
        case ShaderScalarKind::Int32:
        case ShaderScalarKind::UInt32:
        case ShaderScalarKind::Float32:
            return 4;
        case ShaderScalarKind::Int64:
        case ShaderScalarKind::UInt64:
        case ShaderScalarKind::Float64:
            return 8;
        default:
            throw std::invalid_argument("Shader value copy requires a concrete scalar type");
        }
    }

    void Add(size_t source, size_t target, size_t bytes)
    {
        if (!m_regions.empty()) {
            auto &last = m_regions.back();
            if (last.source + last.bytes == source && last.target + last.bytes == target) {
                last.bytes += bytes;
                return;
            }
        }
        m_regions.push_back({source, target, bytes});
    }

    void Append(const ShaderValueLayout &layout, size_t target, ShaderMatrixOrder order)
    {
        switch (layout.kind) {
        case ShaderValueKind::Struct:
            for (const auto &field : layout.fields)
                Append(field, target + field.offset, order);
            break;
        case ShaderValueKind::Array:
            for (uint64_t index = 0; index < layout.elementCount; ++index)
                Append(layout.fields.front(), target + index * layout.stride, order);
            break;
        case ShaderValueKind::Matrix: {
            const auto width = ScalarBytes(layout.scalar);
            const auto source = m_sourceBytes;
            for (uint32_t row = 0; row < layout.rows; ++row)
                for (uint32_t column = 0; column < layout.columns; ++column) {
                    const auto sourceIndex = order == ShaderMatrixOrder::RowMajor ? row * layout.columns + column
                                                                                  : column * layout.rows + row;
                    const auto destination =
                        layout.rowMajor ? row * layout.stride + column * width : column * layout.stride + row * width;
                    Add(source + sourceIndex * width, target + destination, width);
                }
            m_sourceBytes += layout.rows * layout.columns * width;
            break;
        }
        case ShaderValueKind::Vector:
        case ShaderValueKind::Scalar: {
            const auto bytes = ScalarBytes(layout.scalar) * layout.columns;
            Add(m_sourceBytes, target, bytes);
            m_sourceBytes += bytes;
            break;
        }
        }
    }
};

} // namespace infernux
