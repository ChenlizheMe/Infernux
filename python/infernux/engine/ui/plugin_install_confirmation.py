"""Restart prompt for a loaded native dependency changed by installation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from infernux.engine.i18n import t
from infernux.engine.interaction import ModalService

from .editor_modal import (
    EditorModalAction,
    begin_editor_modal,
    end_editor_modal,
    render_editor_modal_actions,
)


class PluginInstallConfirmationCoordinator:
    MODAL_ID = "editor.plugin_install_confirmation"
    _instance: Optional["PluginInstallConfirmationCoordinator"] = None

    def __init__(self, modal_service: ModalService) -> None:
        self._modals = modal_service
        self._value = ""
        self._callback: Optional[Callable[[], None]] = None
        self._requested = False
        self._modals.register(
            self.MODAL_ID,
            is_active=lambda: self.is_active,
            render=self.render,
            cancel=self.cancel,
        )

    @classmethod
    def instance(cls) -> "PluginInstallConfirmationCoordinator":
        from infernux.engine.interaction import EditorInteractionCore

        core = EditorInteractionCore.instance()
        if core is None:
            raise RuntimeError("plugin install confirmation requires EditorInteractionCore")
        if cls._instance is None or cls._instance._modals is not core.modals:
            cls._instance = cls(core.modals)
        return cls._instance

    @property
    def is_active(self) -> bool:
        return self._callback is not None

    def request_restart(self, manager) -> bool:
        if self.is_active or not self._modals.activate(self.MODAL_ID, owner_id="plugins"):
            return False

        def restart() -> None:
            from infernux.engine.scene_manager import SceneFileManager
            from infernux_editor import restart_after_exit

            SceneFileManager.instance().request_close(
                on_confirmed=lambda: restart_after_exit(manager.project_root),
            )

        self._value = ", ".join(manager.pending_install["loaded"])
        self._callback = restart
        self._requested = True
        return True

    def render(self, ctx) -> bool:
        if not self.is_active:
            return False
        title = t("plugins.restart.title")
        request_open = self._requested
        self._requested = False
        if not begin_editor_modal(
            ctx,
            popup_id=f"{title}##plugin_install_confirmation",
            title=title,
            semantic_id=self.MODAL_ID,
            request_open=request_open,
            width=540.0,
            height=230.0,
        ):
            return False
        ctx.text_wrapped(t("plugins.restart.message"))
        ctx.spacing()
        ctx.text_wrapped(self._value)
        ctx.spacing()
        render_editor_modal_actions(
            ctx,
            (
                EditorModalAction(
                    t("plugins.restart.later"),
                    "cancel",
                    lambda: self._cancel(ctx),
                    kind="ghost",
                ),
                EditorModalAction(
                    t("plugins.restart.now"),
                    "confirm",
                    lambda: self._confirm(ctx),
                    kind="primary",
                ),
            ),
            semantic_prefix=self.MODAL_ID,
        )
        end_editor_modal(ctx)
        return True

    def _confirm(self, ctx) -> None:
        callback = self._callback
        ctx.close_current_popup()
        self._clear()
        if callback is not None:
            callback()

    def _cancel(self, ctx) -> None:
        ctx.close_current_popup()
        self.cancel()

    def cancel(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self._value = ""
        self._callback = None
        self._requested = False
        self._modals.deactivate(self.MODAL_ID)


__all__ = ["PluginInstallConfirmationCoordinator"]
