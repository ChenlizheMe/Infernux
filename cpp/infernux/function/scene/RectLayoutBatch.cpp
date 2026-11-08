#include "RectLayoutBatch.h"
#include <stdexcept>

namespace infernux
{
namespace
{
using Value = std::array<float, 3>;
template <glm::vec2 (RectTransformLayout::*Get)() const> Value ReadPair(const RectTransformLayout &layout)
{
    const auto value = (layout.*Get)();
    return {value.x, value.y, 0};
}
template <void (RectTransformLayout::*Set)(glm::vec2)> void WritePair(RectTransformLayout &layout, const Value &value)
{
    (layout.*Set)({value[0], value[1]});
}
template <int Axis> Value ReadSize(const RectTransformLayout &layout)
{
    return {layout.GetRect()[Axis + 2], 0, 0};
}
template <int Axis> void WriteSize(RectTransformLayout &layout, const Value &value)
{
    layout.SetAuthoringSizeWithCurrentAnchors(Axis, value[0]);
}
Value ReadPosition3D(const RectTransformLayout &layout)
{
    const auto value = layout.GetAnchoredPosition3D();
    return {value.x, value.y, value.z};
}
void WritePosition3D(RectTransformLayout &layout, const Value &value)
{
    layout.SetAnchoredPosition3D({value[0], value[1], value[2]});
}
} // namespace

RectLayoutBatch::RectLayoutBatch(std::vector<RectTransformLayout> layouts, const std::string &field)
    : m_layouts(std::move(layouts)), m_channels(2)
{
    if (field == "width") {
        m_read = ReadSize<0>;
        m_write = WriteSize<0>;
        m_channels = 1;
    } else if (field == "height") {
        m_read = ReadSize<1>;
        m_write = WriteSize<1>;
        m_channels = 1;
    } else if (field == "pivot") {
        m_read = ReadPair<&RectTransformLayout::GetPivot>;
        m_write = WritePair<&RectTransformLayout::SetPivot>;
    } else if (field == "anchor_min") {
        m_read = ReadPair<&RectTransformLayout::GetAnchorMin>;
        m_write = WritePair<&RectTransformLayout::SetAnchorMin>;
    } else if (field == "anchor_max") {
        m_read = ReadPair<&RectTransformLayout::GetAnchorMax>;
        m_write = WritePair<&RectTransformLayout::SetAnchorMax>;
    } else if (field == "size_delta") {
        m_read = ReadPair<&RectTransformLayout::GetSizeDelta>;
        m_write = WritePair<&RectTransformLayout::SetSizeDelta>;
    } else if (field == "anchored_position") {
        m_read = ReadPair<&RectTransformLayout::GetAnchoredPosition>;
        m_write = WritePair<&RectTransformLayout::SetAnchoredPosition>;
    } else if (field == "offset_min") {
        m_read = ReadPair<&RectTransformLayout::GetOffsetMin>;
        m_write = WritePair<&RectTransformLayout::SetOffsetMin>;
    } else if (field == "offset_max") {
        m_read = ReadPair<&RectTransformLayout::GetOffsetMax>;
        m_write = WritePair<&RectTransformLayout::SetOffsetMax>;
    } else if (field == "anchored_position3d") {
        m_read = ReadPosition3D;
        m_write = WritePosition3D;
        m_channels = 3;
    } else {
        throw std::invalid_argument("Unknown rectangle field: " + field);
    }
    Validate();
}

void RectLayoutBatch::Validate() const
{
    for (const auto &layout : m_layouts)
        layout.Validate();
}
} // namespace infernux
