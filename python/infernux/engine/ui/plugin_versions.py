"""Compatible version selection for installed packages."""

from __future__ import annotations

from infernux.engine.i18n import t

from .plugin_install_progress import PluginInstallProgressService


class PluginVersionsView:
    def __init__(self) -> None:
        self.releases: dict[str, tuple[dict[str, object], ...]] = {}
        self.selection: dict[str, int] = {}
        self.messages: dict[str, str] = {}

    def render(self, ctx, manager, row) -> None:
        reference = str(row["reference"])
        current = str(row.get("version", ""))
        ctx.text_wrapped(t("plugins.versions.installed").format(version=current))
        if not manager.release_repository(reference):
            ctx.text_wrapped(t("plugins.versions.local"))
            return
        if ctx.button(t("plugins.versions.check") + "##plugin_check_versions"):
            self.check(manager, reference)
        versions = self.releases.get(reference, ())
        if versions:
            index = ctx.combo("##plugin_release_version", self.selection.get(reference, 0),
                              [str(item["version"]) for item in versions])
            if index != self.selection.get(reference, 0):
                self.messages.pop(reference, None)
            self.selection[reference] = index
            selected = versions[index]
            ctx.text_wrapped(str(selected.get("notes", "")) or t("plugins.versions.no_notes"))
            disabled = str(selected["version"]) == current
            ctx.begin_disabled(disabled)
            if ctx.button(t("plugins.versions.apply") + "##plugin_update_version"):
                self.download(manager, reference, str(selected["release_tag"]))
            ctx.end_disabled()
        elif reference in self.releases:
            ctx.text_wrapped(t("plugins.versions.none"))
        if self.messages.get(reference):
            ctx.text_wrapped(self.messages[reference])

    def check(self, manager, reference: str) -> None:
        def complete(ok, result, message):
            if ok:
                self.releases[reference] = tuple(result)
                self.selection[reference] = 0
                self.messages.pop(reference, None)
            else:
                self.messages[reference] = message

        self._begin(reference, lambda report: manager.available_releases(reference), complete)

    def download(self, manager, reference: str, tag: str) -> None:
        def observe(installed_reference, message):
            self.messages[reference] = message

        if not PluginInstallProgressService.instance().begin_package(
            manager=manager, label=reference, action="release", release_tag=tag, observer=observe,
        ):
            self.messages[reference] = t("plugins.install_progress.busy")

    def _begin(self, reference, work, complete) -> None:
        if not PluginInstallProgressService.instance().begin(
            label=reference, work=work, complete=complete,
        ):
            self.messages[reference] = t("plugins.install_progress.busy")
