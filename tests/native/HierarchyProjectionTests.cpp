// CPU ImGui frames exercise the production HierarchyPanel without a renderer.
// Seed the panel's pending Shift release state, then execute its real
// VisiblePreRender callback. Explicit-instantiation access avoids a product edit.
#include <algorithm>
#include <function/editor/HierarchyPanel.h>
#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace infernux;
template <typename Tag, typename Tag::type Member> struct AuditAccess
{
    friend typename Tag::type access(Tag)
    {
        return Member;
    }
};
struct PendingTag
{
    using type = uint64_t HierarchyPanel::*;
    friend type access(PendingTag);
};
struct ShiftTag
{
    using type = bool HierarchyPanel::*;
    friend type access(ShiftTag);
};
struct SearchTag
{
    using type = void (HierarchyPanel::*)(const char *);
    friend type access(SearchTag);
};
struct FilterTag
{
    using type = std::vector<GameObject *> (HierarchyPanel::*)(const std::vector<GameObject *> &);
    friend type access(FilterTag);
};
template struct AuditAccess<PendingTag, &HierarchyPanel::m_pendingSelectId>;
template struct AuditAccess<ShiftTag, &HierarchyPanel::m_pendingShift>;
template struct AuditAccess<SearchTag, &HierarchyPanel::SetSearchQuery>;
template struct AuditAccess<FilterTag, &HierarchyPanel::FilterForSearch>;
struct AuditHierarchy : HierarchyPanel
{
    using HierarchyPanel::VisiblePreRender;
};

// Extended production-panel contracts. Included after the basic probe helpers.
struct CollapseTag
{
    using type = std::unordered_set<uint64_t> HierarchyPanel::*;
    friend type access(CollapseTag);
};
struct DirtyTag
{
    using type = bool HierarchyPanel::*;
    friend type access(DirtyTag);
};
struct RefreshTag
{
    using type = void (HierarchyPanel::*)(Scene *, bool, bool);
    friend type access(RefreshTag);
};
template struct AuditAccess<CollapseTag, &HierarchyPanel::m_collapsedSceneWorldIds>;
template struct AuditAccess<DirtyTag, &HierarchyPanel::m_flatListDirty>;
template struct AuditAccess<RefreshTag, &HierarchyPanel::RefreshRootObjects>;
struct ProjectionRow
{
    GameObject *obj;
    bool sceneHeader;
};
template <typename Tag, auto Member> struct ProjectionAccess
{
    friend std::vector<ProjectionRow> projectionRows(Tag, const HierarchyPanel &panel)
    {
        std::vector<ProjectionRow> rows;
        for (const auto &item : panel.*Member)
            rows.push_back({item.obj, item.sceneHeader});
        return rows;
    }
};
struct RowsTag
{
    friend std::vector<ProjectionRow> projectionRows(RowsTag, const HierarchyPanel &);
};
template struct ProjectionAccess<RowsTag, &HierarchyPanel::m_flatItems>;

static int ExtendedCases(nlohmann::json &report)
{
    int failures = 0;
    const auto check = [](bool value, const char *message) {
        if (!value)
            throw std::runtime_error(message);
    };
    const auto run = [&](const char *name, const auto &body) {
        auto &manager = SceneManager::Instance();
        manager.Shutdown();
        auto *scene = manager.CreateScene("Projection primary");
        manager.SetActiveScene(scene);
        auto *a = scene->CreateGameObject("Match first");
        auto *child = scene->CreateGameObject("Unrelated child");
        child->SetParent(a, true);
        auto *b = scene->CreateGameObject("Match last");
        AuditHierarchy panel;
        panel.SetSceneHeaderSnapshot("Projection primary", true, "Prefab");
        panel.SetSelectionSnapshot({b->GetID()}, b->GetID());
        InxGUIContext context;
        const auto frame = [&] {
            ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_Always);
            panel.OnRender(&context);
            ImGui::Render();
        };
        const auto range = [&](uint64_t id) {
            std::vector<uint64_t> order;
            uint64_t selected = 0;
            panel.setOrderedIds = [&](const auto &ids) { order = ids; };
            panel.rangeSelectId = [&](uint64_t target) { selected = target; };
            panel.*access(PendingTag{}) = id;
            panel.*access(ShiftTag{}) = true;
            ImGui::NewFrame();
            panel.VisiblePreRender(&context);
            ImGui::EndFrame();
            panel.setOrderedIds = {};
            panel.rangeSelectId = {};
            check(selected == id, "range callback target changed");
            return order;
        };
        const auto rows = [&] {
            std::vector<uint64_t> ids;
            for (const auto &item : projectionRows(RowsTag{}, panel))
                if (item.obj)
                    ids.push_back(item.obj->GetID());
            return ids;
        };
        try {
            frame();
            frame();
            body(manager, scene, a, child, b, panel, frame, range, rows);
            report["cases"].push_back({{"case", name}, {"passed", true}});
        } catch (const std::exception &error) {
            ++failures;
            report["cases"].push_back({{"case", name}, {"passed", false}, {"error", error.what()}});
        }
        manager.Shutdown();
    };
    run("expansion_changes_before_release",
        [&](auto &, auto *, auto *a, auto *c, auto *b, auto &p, auto frame, auto range, auto) {
            p.SetExpandedObjectIds({a->GetID()});
            check(range(b->GetID()) == std::vector<uint64_t>{a->GetID(), c->GetID(), b->GetID()},
                  "new expansion not reflected");
            frame();
            p.SetExpandedObjectIds({});
            check(range(b->GetID()) == std::vector<uint64_t>{a->GetID(), b->GetID()}, "collapsed child selected");
        });
    run("search_transient_parent_context",
        [&](auto &, auto *, auto *a, auto *c, auto *b, auto &p, auto frame, auto range, auto rows) {
            (p.*access(SearchTag{}))("child");
            frame();
            check(rows() == std::vector<uint64_t>{a->GetID(), c->GetID()}, "matching descendant lost parent context");
            check(range(c->GetID()) == rows(), "search range differs from rows");
            p.ClearSearch();
            frame();
            check(rows() == std::vector<uint64_t>{a->GetID(), b->GetID()}, "search persisted expansion");
            check(range(b->GetID()) == rows(), "clear search reused search range");
        });
    run("search_changed_before_release",
        [&](auto &, auto *, auto *a, auto *c, auto *, auto &p, auto, auto range, auto) {
            (p.*access(SearchTag{}))("child");
            check(range(c->GetID()) == std::vector<uint64_t>{a->GetID(), c->GetID()}, "release used stale search");
        });
    run("hidden_push_changes_before_release", [&](auto &, auto *, auto *a, auto *c, auto *b, auto &p, auto frame,
                                                  auto range, auto) {
        p.SetExpandedObjectIds({a->GetID()});
        frame();
        p.SetRuntimeHiddenIds({c->GetID()});
        check(range(b->GetID()) == std::vector<uint64_t>{a->GetID(), b->GetID()}, "hidden child selected");
        p.SetRuntimeHiddenIds({a->GetID()});
        check(range(b->GetID()) == std::vector<uint64_t>{b->GetID()}, "hidden subtree selected");
        p.SetRuntimeHiddenIds({});
        check(range(b->GetID()) == std::vector<uint64_t>{a->GetID(), c->GetID(), b->GetID()}, "unhidden child missing");
    });
    run("hidden_pull_changes_search_cache",
        [&](auto &, auto *, auto *a, auto *c, auto *b, auto &p, auto frame, auto range, auto rows) {
            std::unordered_set<uint64_t> hidden;
            p.getRuntimeHiddenIds = [&] { return hidden; };
            (p.*access(SearchTag{}))("child");
            frame();
            hidden = {c->GetID()};
            frame();
            check(rows().empty(), "pull-hidden change kept cached search ancestor");
            hidden.clear();
            frame();
            check(range(c->GetID()) == std::vector<uint64_t>{a->GetID(), c->GetID()}, "pull unhide missing child");
            p.getRuntimeHiddenIds = {};
        });
    run("selected_hidden_root_stays_hidden",
        [&](auto &, auto *, auto *, auto *, auto *b, auto &p, auto frame, auto, auto rows) {
            p.SetRuntimeHiddenIds({b->GetID()});
            frame();
            const auto ids = rows();
            check(std::find(ids.begin(), ids.end(), b->GetID()) == ids.end(), "selection resurrected hidden root");
        });
    run("multiple_scenes_and_collapsed_selected_scene",
        [&](auto &m, auto *s, auto *a, auto *, auto *b, auto &p, auto frame, auto range, auto rows) {
            auto *other = m.CreateScene("Second");
            auto *d = other->CreateGameObject("Other root");
            p.SetSceneHeaderSnapshot("Primary", false, "");
            frame();
            check(range(d->GetID()) == std::vector<uint64_t>{a->GetID(), b->GetID(), d->GetID()},
                  "multiple scene order wrong");
            (p.*access(CollapseTag{})).insert(s->GetWorldId());
            p.*access(DirtyTag{}) = true;
            frame();
            check(rows() == std::vector<uint64_t>{d->GetID()}, "collapsed scene resurrected selected root");
            check(range(d->GetID()) == rows(), "collapsed scene in range");
        });
    run("inactive_scene_rename_updates_search",
        [&](auto &m, auto *, auto *, auto *, auto *b, auto &p, auto frame, auto range, auto rows) {
            auto *other = m.CreateScene("Second");
            auto *d = other->CreateGameObject("Other root");
            p.SetSceneHeaderSnapshot("Primary", false, "");
            (p.*access(SearchTag{}))("renamed");
            frame();
            check(rows().empty(), "search initial mismatch");
            d->SetName("Renamed elsewhere");
            frame();
            check(rows() == std::vector<uint64_t>{d->GetID()}, "inactive scene rename stale");
            check(range(d->GetID()) == rows(), "inactive search selection stale");
            d->SetName("Other root");
            frame();
            check(rows().empty(), "inactive scene reverse rename stale");
        });
    run("descendant_rename_invalidates_ancestors",
        [&](auto &, auto *, auto *a, auto *c, auto *, auto &p, auto frame, auto range, auto rows) {
            (p.*access(SearchTag{}))("renamed");
            frame();
            c->SetName("Renamed child");
            frame();
            check(rows() == std::vector<uint64_t>{a->GetID(), c->GetID()}, "ancestor search cache stale");
            c->SetName("No match");
            frame();
            check(rows().empty(), "ancestor remains after descendant rename");
        });
    run("rename_before_release", [&](auto &, auto *, auto *a, auto *, auto *b, auto &p, auto frame, auto range, auto) {
        (p.*access(SearchTag{}))("match");
        frame();
        a->SetName("No longer visible");
        check(range(b->GetID()) == std::vector<uint64_t>{b->GetID()}, "range rename invalidation delayed");
    });
    run("create_before_release", [&](auto &, auto *s, auto *a, auto *, auto *b, auto &p, auto frame, auto range, auto) {
        auto *fresh = s->CreateGameObject("New root");
        check(range(fresh->GetID()) == std::vector<uint64_t>{a->GetID(), b->GetID(), fresh->GetID()},
              "new root missing until render");
    });
    run("persistent_projection_and_rename",
        [&](auto &m, auto *, auto *a, auto *c, auto *b, auto &p, auto frame, auto range, auto) {
            m.Play();
            m.DontDestroyOnLoad(a);
            m.Update(0.0f);
            check(a->GetScene() == m.GetRuntimePersistentScene(), "fixture did not promote runtime root");
            p.SetExpandedObjectIds({a->GetID()});
            frame();
            check(range(c->GetID()) == std::vector<uint64_t>{b->GetID(), a->GetID(), c->GetID()},
                  "persistent range order missing");
            (p.*access(SearchTag{}))("new name");
            frame();
            c->SetName("New name");
            frame();
            check(range(c->GetID()) == std::vector<uint64_t>{a->GetID(), c->GetID()}, "persistent rename/search stale");
            p.ClearSearch();
            p.SetExpandedObjectIds({});
            frame();
            check(range(a->GetID()) == std::vector<uint64_t>{b->GetID(), a->GetID()},
                  "collapsed persistent child in range");
        });
    run("name_revision_does_not_invalidate_runtime_structure",
        [&](auto &, auto *s, auto *a, auto *, auto *, auto &p, auto frame, auto, auto) {
            const auto structure = s->GetStructureVersion();
            const auto names = s->GetObjectNameRevision();
            a->SetName(a->GetName());
            check(s->GetObjectNameRevision() == names, "same name increments revision");
            a->SetName("中文名称");
            check(s->GetObjectNameRevision() == names + 1, "rename did not increment once");
            check(s->GetStructureVersion() == structure, "rename rebuilt runtime object/component lists");
            (p.*access(RefreshTag{}))(s, false, false);
            check(!(p.*access(DirtyTag{})), "rename rebuilt rows without active search");
            (p.*access(SearchTag{}))("中文");
            frame();
            a->SetName(a->GetName());
            (p.*access(RefreshTag{}))(s, false, false);
            check(!(p.*access(DirtyTag{})), "same-name assignment rebuilt search rows");
            for (int i = 0; i < 10; ++i) {
                (p.*access(RefreshTag{}))(s, false, false);
                check(!(p.*access(DirtyTag{})), "idle refresh dirtied projection");
            }
            a->SetName("Changed");
            (p.*access(RefreshTag{}))(s, false, false);
            check(p.*access(DirtyTag{}), "changed name did not invalidate active search");
            GameObject detached("Detached");
            detached.SetName("Detached renamed");
            check(s->GetObjectNameRevision() == names + 2, "detached rename touched scene");
        });
    run("single_scene_mode_switch",
        [&](auto &, auto *, auto *a, auto *, auto *b, auto &p, auto frame, auto range, auto rows) {
            p.SetSceneHeaderSnapshot("Primary", false, "");
            frame();
            check(projectionRows(RowsTag{}, p).front().sceneHeader, "mode change kept prefab rows");
            p.SetSceneHeaderSnapshot("Primary", true, "Prefab");
            frame();
            check(!projectionRows(RowsTag{}, p).front().sceneHeader, "prefab mode kept scene header");
            check(range(b->GetID()) == std::vector<uint64_t>{a->GetID(), b->GetID()}, "mode change order wrong");
        });
    run("deleted_root_before_release_and_reveal",
        [&](auto &, auto *s, auto *a, auto *, auto *b, auto &p, auto frame, auto range, auto rows) {
            s->DestroyGameObject(a);
            s->ProcessPendingDestroys();
            p.SetSelectedObjectById(b->GetID(), false);
            check(range(b->GetID()) == std::vector<uint64_t>{b->GetID()}, "deleted subtree remained in selection");
            frame();
            check(rows() == std::vector<uint64_t>{b->GetID()}, "deleted subtree remained in rows");
        });
    for (const std::string reason : {"deleted", "hidden", "search", "no_scene"}) {
        const auto label = "pending_target_" + reason;
        run(label.c_str(), [&](auto &m, auto *s, auto *a, auto *, auto *b, auto &p, auto frame, auto, auto) {
            const auto target = a->GetID();
            p.*access(PendingTag{}) = target;
            p.*access(ShiftTag{}) = true;
            if (reason == "deleted") {
                s->DestroyGameObject(a);
                s->ProcessPendingDestroys();
            }
            if (reason == "hidden")
                p.SetRuntimeHiddenIds({target});
            if (reason == "search")
                (p.*access(SearchTag{}))("last");
            if (reason == "no_scene")
                m.Shutdown();
            bool called = false;
            std::vector<uint64_t> order;
            p.rangeSelectId = [&](uint64_t) { called = true; };
            p.setOrderedIds = [&](const auto &ids) { order = ids; };
            InxGUIContext context;
            ImGui::NewFrame();
            p.VisiblePreRender(&context);
            ImGui::EndFrame();
            check(!called, "release selected a target that is no longer visible");
            check(std::find(order.begin(), order.end(), target) == order.end(), "unavailable target in range order");
        });
    }
    return failures;
}

int main()
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(900, 900);
    io.DeltaTime = 1.0f / 60;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    InxGUIContext context;
    auto &manager = SceneManager::Instance();
    auto *scene = manager.CreateScene("Audit hierarchy range");
    manager.SetActiveScene(scene);
    auto *first = scene->CreateGameObject("Match first");
    auto *hidden = scene->CreateGameObject("Unrelated child");
    hidden->SetParent(first, true);
    auto *last = scene->CreateGameObject("Match last");
    const auto firstId = first->GetID(), hiddenId = hidden->GetID(), lastId = last->GetID();
    nlohmann::json report;
    report["method"] = "production VisiblePreRender, seeded pending Shift release; CPU ImGui only";
    report["ids"] = {{"first", firstId}, {"hidden_child", hiddenId}, {"last", lastId}};
    int failures = 0;
    for (const std::string mode : {"expanded_control", "collapsed", "search"}) {
        AuditHierarchy panel;
        panel.isPrefabMode = [] { return true; }; // suppress scene header rows
        panel.SetSelectionSnapshot({firstId}, firstId);
        if (mode == "expanded_control")
            panel.SetExpandedObjectIds({firstId});
        if (mode == "search")
            (panel.*access(SearchTag{}))("Match");
        std::vector<uint64_t> published;
        uint64_t target = 0;
        panel.setOrderedIds = [&](const std::vector<uint64_t> &ids) { published = ids; };
        panel.rangeSelectId = [&](uint64_t id) { target = id; };
        // Prime the real root/flat cache and first-selection reveal path.
        for (int frame = 0; frame < 2; ++frame) {
            ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(600, 600));
            panel.OnRender(&context);
            ImGui::Render();
        }
        panel.*access(PendingTag{}) = lastId;
        panel.*access(ShiftTag{}) = true;
        ImGui::NewFrame();
        panel.VisiblePreRender(&context);
        ImGui::EndFrame();
        const std::vector<uint64_t> expected = mode == "expanded_control"
                                                   ? std::vector<uint64_t>{firstId, hiddenId, lastId}
                                                   : std::vector<uint64_t>{firstId, lastId};
        const bool passed = published == expected && target == lastId;
        report["cases"].push_back({{"case", mode},
                                   {"expected_ordered_ids", expected},
                                   {"actual_ordered_ids", published},
                                   {"range_target", target},
                                   {"passed", passed}});
        failures += !passed;
    }
    for (const bool renameIntoMatch : {false, true}) {
        first->SetName(renameIntoMatch ? "Unrelated first" : "Match first");
        AuditHierarchy panel;
        panel.isPrefabMode = [] { return true; };
        panel.SetSelectionSnapshot({lastId}, lastId); // keep primary stable across rename
        (panel.*access(SearchTag{}))("Match");
        auto renderFrames = [&] {
            for (int frame = 0; frame < 2; ++frame) {
                ImGui::NewFrame();
                ImGui::SetNextWindowPos(ImVec2(0, 0));
                ImGui::SetNextWindowSize(ImVec2(600, 600));
                panel.OnRender(&context);
                ImGui::Render();
            }
        };
        auto visibleIds = [&] {
            std::vector<uint64_t> ids;
            const std::vector<GameObject *> roots{first, last};
            for (auto *obj : (panel.*access(FilterTag{}))(roots))
                ids.push_back(obj->GetID());
            return ids;
        };
        renderFrames();
        const auto before = visibleIds();
        const auto versionBefore = scene->GetStructureVersion();
        first->SetName(renameIntoMatch ? "Match first" : "Unrelated first");
        renderFrames();
        const std::vector<uint64_t> expected =
            renameIntoMatch ? std::vector<uint64_t>{firstId, lastId} : std::vector<uint64_t>{lastId};
        const auto actual = visibleIds();
        const bool passed = actual == expected;
        const std::string mode = renameIntoMatch ? "rename_into_search" : "rename_out_of_search";
        report["cases"].push_back({{"case", mode},
                                   {"before_visible_ids", before},
                                   {"expected_visible_ids", expected},
                                   {"actual_visible_ids", actual},
                                   {"structure_version_before", versionBefore},
                                   {"structure_version_after", scene->GetStructureVersion()},
                                   {"passed", passed}});
        failures += !passed;
        // Control: the public invalidation API must recover the correct projection.
        panel.InvalidateSceneStructureCache();
        renderFrames();
        const auto refreshed = visibleIds();
        const bool controlPassed = refreshed == expected;
        report["cases"].push_back({{"case", mode + "_invalidate_control"},
                                   {"expected_visible_ids", expected},
                                   {"actual_visible_ids", refreshed},
                                   {"passed", controlPassed}});
        failures += !controlPassed;
    }
    manager.Shutdown();
    failures += ExtendedCases(report);
    ImGui::DestroyContext();
    report["failures"] = failures;
    std::cout << report.dump(2) << '\n';
    return failures ? 1 : 0;
}
