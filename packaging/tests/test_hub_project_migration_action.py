from types import SimpleNamespace

from PySide6.QtWidgets import QApplication, QLabel, QScrollArea

from i18n import tr
from launcher import GameEngineLauncher
from ui_project_list import ProjectListPane
import ui_project_list as ui_project_list_module


def test_project_menu_migrates_the_clicked_project_not_previous_selection(tmp_path):
    app = QApplication.instance() or QApplication([])
    records = [
        SimpleNamespace(project_id=str(index), name=f"Project {index}",
                        created_at="", path=str(tmp_path))
        for index in range(2)
    ]
    pane = ProjectListPane(SimpleNamespace(all_projects=lambda: records))
    pane.select_project("0")
    requests = []
    window = SimpleNamespace(
        project_list=pane,
        viewmodel=SimpleNamespace(
            migrate_project=lambda parent: requests.append((pane.get_selected_project_id(), parent)),
        ),
    )
    pane.migrate_requested.connect(lambda project_id: GameEngineLauncher._migrate_project_from_card(window, project_id))
    actions = pane.project_cards["1"]._actions_menu.actions()
    migration = next(action for action in actions if action.text() == tr("Migrate Project"))
    migration.trigger()
    assert requests == [("1", window)]
    pane.close()


def test_long_project_identity_does_not_push_actions_outside_the_viewport(tmp_path):
    app = QApplication.instance() or QApplication([])
    record = SimpleNamespace(
        project_id="long", name="Long project name " * 20, created_at="",
        path=str(tmp_path / ("deep-" * 30)),
    )
    pane = ProjectListPane(SimpleNamespace(all_projects=lambda: [record]))
    pane.resize(700, 300)
    pane.show()
    app.processEvents()
    try:
        scroll = pane.findChild(QScrollArea)
        assert scroll.horizontalScrollBar().maximum() == 0
        button = pane.project_cards["long"]._actions_button
        right = button.mapTo(scroll.viewport(), button.rect().topRight())
        assert scroll.viewport().rect().contains(right)
    finally:
        pane.close()


def test_refresh_hides_retired_widgets_before_opening_a_modal(tmp_path):
    app = QApplication.instance() or QApplication([])
    records = []
    pane = ProjectListPane(SimpleNamespace(all_projects=lambda: records))
    pane.show()
    app.processEvents()
    empty_state = pane.card_layout.itemAt(0).widget()
    try:
        assert empty_state.isVisible()
        records.append(SimpleNamespace(project_id="new", name="New project",
                                       created_at="", path=str(tmp_path)))
        pane.refresh()
        # The caller may enter a modal loop before deferred deletion runs.
        assert not empty_state.isVisible()
    finally:
        pane.close()


def test_project_with_missing_local_engine_version_is_disabled(
    tmp_path, monkeypatch
):
    app = QApplication.instance() or QApplication([])
    project = tmp_path / "Project"
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "ProjectSettings" / "PythonRuntime.json").write_text(
        '{"pythonVersion": "3.13"}\n', encoding="utf-8"
    )
    (project / ".infernux-version").write_text("0.4.1\n", encoding="utf-8")

    class VersionManager:
        @staticmethod
        def is_installed(_version, _python_version=None):
            return False

    monkeypatch.setattr(ui_project_list_module, "is_frozen", lambda: True)
    record = SimpleNamespace(
        project_id="missing", name="Missing runtime", created_at="", path=str(project)
    )
    pane = ProjectListPane(
        SimpleNamespace(all_projects=lambda: [record]), VersionManager()
    )
    try:
        card = pane.project_cards["missing"]
        assert card.isEnabled() is False
        assert card.can_select is False
        assert any(
            label.text().startswith(tr("Install required version"))
            for label in card.findChildren(QLabel)
        )
        pane.select_project("missing")
        assert pane.get_selected_project_id() is None
    finally:
        pane.close()
