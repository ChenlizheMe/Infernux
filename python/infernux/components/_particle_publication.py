"""Private particle candidates; the existing component owner retries them.

This holds publication intent, not an executor or another GPU resource cache.
Native upload tickets and the resident graph retain their original ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class _GpuParticlePublication:
    native: object
    component: object
    request: dict
    apply: Callable[[], None]
    metadata: object = None
    authoring_state: tuple = ()
    prepare: Callable[[], _GpuParticlePublication] | None = None
    current: Callable[[], bool] | None = None


class _GpuParticlePublicationBatch:
    """One atomic group, shared by its members until publication or withdrawal."""

    def __init__(self, items):
        self.items = tuple(items)
        self._attempt_opportunity = None
        native = self.items[0].native
        if any(item.native is not native for item in self.items):
            raise RuntimeError("particle publication crossed native engine instances")

    def bind(self):
        for item in self.items:
            previous = item.component._pending_gpu_publication
            if previous is not None:
                previous.cancel(rebuild=True)
        for item in self.items:
            item.component._pending_gpu_publication = self

    def _release(self):
        items, self.items = self.items, ()
        for item in items:
            component = item.component
            if component._pending_gpu_publication is self:
                component._pending_gpu_publication = None
        return items

    def cancel(self, *, rebuild=False):
        for item in self._release():
            if rebuild:
                item.component._runtime_rebuild_pending = True
            else:
                item.component._restore_serialized_overrides()

    def advance(self):
        from infernux.application import Application
        from infernux.lib import GpuParticlePublicationStatus as Status

        if not self.items:
            return True
        # Several members may update in one frame. They share this attempt;
        # manual hosts without an Engine frame are advanced by their caller.
        engine = Application._current_engine()
        opportunity = None if engine is None else int(engine._service_opportunity)
        if opportunity is not None and opportunity == self._attempt_opportunity:
            return False
        self._attempt_opportunity = opportunity
        if any(item.current is not None and not item.current() for item in self.items):
            self.cancel(rebuild=True)
            return True

        prepared = []
        for item in self.items:
            if item.prepare is not None and item.authoring_state != item.component._publication_authoring_state():
                updated = item.prepare()
                live_ids = {program['id'] for program in updated.request['programs']}
                updated.request['remove_ids'] = sorted(
                    (set(item.request['remove_ids']) | set(updated.request['remove_ids'])) - live_ids)
                item = updated
            prepared.append(item)
        self.items = tuple(prepared)
        status, error = self.items[0].native._replace_gpu_particle_graphs([item.request for item in self.items])
        if status == Status.Pending:
            return False
        if status == Status.Failed:
            raise RuntimeError(error)
        items = self._release()
        for item in items:
            item.apply()
        return True
