"""Header plate for the Projects page."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt
from i18n import tr
from view.forge import PageHeader


class ControlPane(QWidget):
    """Projects page header: §01 plate with Open / New project actions.

    Launching lives on the rows themselves (button, double-click, Enter).
    """

    def __init__(self, viewmodel, style=None, parent=None):
        super().__init__(parent)
        self.viewmodel = viewmodel

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.header = PageHeader(
            "01", tr("PROJECTS"), tr("Projects"),
            tr("Create, open and launch your Infernux projects."),
        )
        main_layout.addWidget(self.header)

        self.btn_open = QPushButton(tr("Open"))
        self.btn_open.setObjectName("normalBtn")
        self.btn_open.setFixedHeight(36)
        self.btn_open.setMinimumWidth(92)
        self.btn_open.setToolTip(tr("Add an existing project folder to Hub"))
        self.btn_open.clicked.connect(lambda: self.viewmodel.open_existing_project(self))
        self.header.actions.addWidget(self.btn_open)

        self.btn_new = QPushButton("+  " + tr("New project"))
        self.btn_new.setObjectName("primaryBtn")
        self.btn_new.setFixedHeight(36)
        self.btn_new.setMinimumWidth(132)
        self.btn_new.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_new.clicked.connect(lambda: self.viewmodel.create_project(self))
        self.header.actions.addWidget(self.btn_new)
