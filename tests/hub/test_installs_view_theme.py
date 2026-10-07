from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea, QWidget

from android_support import AndroidSupportManager
from i18n import tr
from style import StyleManager
from view.installs_view import _AndroidSupportCard, _configure_install_scroll_area


def test_catalog_failure_is_plain_selectable_text():
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from install_queue import InstallQueue
    from view.installs_view import InstallEditorDialog
    import time
    app = _app()
    class OfflineCatalog:
        def list_versions(self, **kwargs):
            raise RuntimeError("<urlopen error certificate verify failed>")
    dialog = InstallEditorDialog(OfflineCatalog(), InstallQueue(app))
    deadline = time.monotonic() + 5
    while "certificate verify failed" not in dialog._status.text() and time.monotonic() < deadline:
        QTest.qWait(10)
    dialog._fetch_thread.wait()
    assert "<urlopen error certificate verify failed>" in dialog._status.text()
    assert dialog._status.textFormat() == Qt.TextFormat.PlainText
    assert dialog._status.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert "hub.log" in dialog._status.text()
    dialog._fetch_thread._vm.list_versions = lambda **kwargs: []
    dialog._retry_fetch()
    deadline = time.monotonic() + 5
    while dialog._fetch_thread.isRunning() and time.monotonic() < deadline:
        QTest.qWait(10)
    dialog._fetch_thread.wait()
    QTest.qWait(10)
    assert "certificate" not in dialog._status.text()
    dialog.close()


def _app():
    return QApplication.instance() or QApplication([])


def test_install_dialog_scroll_surface_uses_dark_hub_palette():
    app = _app()
    scroll = QScrollArea()
    container = QWidget()
    _configure_install_scroll_area(scroll, container)
    scroll.setWidget(container)

    app.setStyleSheet(StyleManager.get_stylesheet(True))
    scroll.show()
    app.processEvents()

    assert scroll.objectName() == "installScrollArea"
    assert scroll.viewport().objectName() == "installViewport"
    assert container.objectName() == "installListContainer"
    assert scroll.viewport().palette().color(QPalette.ColorRole.Window).name() == "#191919"
    assert container.palette().color(QPalette.ColorRole.Window).name() == "#191919"

    scroll.close()


def test_common_message_boxes_use_shared_readable_metrics():
    stylesheet = StyleManager.get_stylesheet(True)

    assert "QMessageBox QLabel" in stylesheet
    assert "font-size: 15px" in stylesheet
    assert "min-width: 360px" in stylesheet
    assert "QMessageBox QPushButton" in stylesheet
    assert "min-height: 34px" in stylesheet


def test_android_support_card_exposes_install_before_any_project_plugin(
    tmp_path,
):
    _app()
    card = _AndroidSupportCard(AndroidSupportManager(tmp_path / "missing"))

    labels = [label.text() for label in card.findChildren(QLabel)]
    buttons = [button.text() for button in card.findChildren(QPushButton)]
    assert tr("Android compatibility") in labels
    assert tr("Required before the Android platform plugin can be imported. "
              "Large toolchains are installed once and shared by every project.") in labels
    assert buttons == [tr("Install")]

    card.close()
