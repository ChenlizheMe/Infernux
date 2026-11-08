#pragma once

#include "core/FoundationApi.h"

#include <array>
#include <cstdint>
#include <optional>

namespace infernux::ui
{
using RectPoint = std::array<float, 2>;

struct RectLayoutSpec
{
    RectPoint parentSize{}, parentPivot{0.5f, 0.5f};
    RectPoint anchorMin{0.5f, 0.5f}, anchorMax{0.5f, 0.5f};
    RectPoint pivot{0.5f, 0.5f}, sizeDelta{}, anchoredPosition{};
};

struct RectLayoutGeometry
{
    std::array<float, 4> rect{}; // local x, y, width, height
    RectPoint offsetMin{}, offsetMax{};
    std::array<std::array<float, 3>, 4> localCorners{};
};

struct RectLayoutDrive
{
    uint64_t owner = 0;
    RectPoint anchoredPosition{}, sizeDelta{};
    bool position = false;
    uint8_t sizeMask = 3;
};

enum class RectUnitMode : uint8_t
{
    Explicit,
    InheritUI,
    ScreenCanvas
};

// The scene store owns this optional local-XY parameterization. Transform's
// position array is its derived pose cache, not an independent layout pose.
struct RectTransformState
{
    RectLayoutSpec spec{{}, {.5f, .5f}, {.5f, .5f}, {.5f, .5f}, {.5f, .5f}, {100, 100}, {}};
    RectPoint anchorReference{};
    float positionScale = 1.f;
    RectUnitMode unitMode = RectUnitMode::Explicit;
    bool screenSpace = false;
    uint64_t revision = 0;
    RectPoint authoringParentSize{}, authoringParentPivot{.5f, .5f};
    bool drivenParent = false;
    std::optional<RectLayoutDrive> drive;

    INFERNUX_FOUNDATION_API void RefreshReference();
    INFERNUX_FOUNDATION_API RectPoint LocalPosition() const;
    INFERNUX_FOUNDATION_API RectPoint SetLocalPosition(RectPoint value);
    INFERNUX_FOUNDATION_API RectLayoutSpec AuthoringSpec() const;
    INFERNUX_FOUNDATION_API RectPoint AuthoringLocalPosition() const;
    INFERNUX_FOUNDATION_API void BeginDrive(uint64_t owner, bool position, uint8_t sizeMask = 3);
    INFERNUX_FOUNDATION_API void EndDrive();
};
} // namespace infernux::ui
