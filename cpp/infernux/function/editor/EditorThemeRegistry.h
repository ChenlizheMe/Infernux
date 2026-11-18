#pragma once

/**
 * @file EditorThemeRegistry.h
 * @brief Single source of truth + runtime theme manager for the editor.
 *
 * Token values live in EditorThemeTable.inl (X-macro base theme). At runtime
 * the registry holds an *active* theme (base + optional overrides) so the whole
 * editor can be re-skinned and switched from one place:
 *
 *   - C++ panels & ImGui global style read the ACTIVE theme via Colors()/Color().
 *   - Python's Theme class mirrors the active theme via the pybind getters.
 *   - ApplyImGuiColors() drives ImGui's global palette so EVERY built-in widget
 *     (C++ and Python) is themed from this one place.
 *
 * Switching is cheap: SetActiveTheme() swaps the active maps, bumps a generation
 * counter, and (via the binding) re-applies the ImGui palette once.
 */

#include <imgui.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{

class EditorThemeRegistry
{
  public:
    // Active-theme accessors (used by C++ panels and the Python mirror).
    static const std::unordered_map<std::string, ImVec4> &Colors();
    static const std::unordered_map<std::string, ImVec2> &Vec2s();
    static const std::unordered_map<std::string, float> &Floats();

    /// Look up a required active-theme token. Missing tokens are programming
    /// errors and throw instead of allowing each consumer to invent a value.
    static ImVec4 Color(const std::string &name);
    static ImVec2 Vec2(const std::string &name);
    static float Float(const std::string &name);

    // ── Theme switching ────────────────────────────────────────────────
    static std::vector<std::string> ThemeNames();
    static const std::string &ActiveTheme();
    /// Switch the active theme. No-op (returns false) for unknown names.
    static bool SetActiveTheme(const std::string &name);
    /// Bumped whenever the active theme changes — cheap change detection.
    static unsigned long Generation();

    // ── Contributed themes (packages) ──────────────────────────────────
    /// One package-authored theme. Tokens override the built-in base theme, so
    /// a package can ship a partial palette without restating every token.
    struct Definition
    {
        std::string name;
        std::unordered_map<std::string, ImVec4> colors;
        std::unordered_map<std::string, ImVec2> vec2s;
        std::unordered_map<std::string, float> floats;
    };

    /// Register or replace one theme under an owner (package reference).
    static void Register(const Definition &definition, const std::string &owner);
    /// Remove every theme contributed by one owner; falls back to the default.
    static void RemoveOwner(const std::string &owner);
    /// Replace an owner's whole catalog; optionally activate `preferred`.
    static void ReplaceOwner(const std::string &owner, const std::vector<Definition> &definitions,
                             const std::string &preferred);

    /// Push the active theme's palette into ImGui's global style colors.
    /// Themes every built-in widget at once (C++ and Python). Cheap; call on
    /// startup and once per theme switch.
    static void ApplyImGuiColors();
    static void ConfigureImGuiDimensions(ImGuiStyle &style);
};

} // namespace infernux
