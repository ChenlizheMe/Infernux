"""Thread-safe file-system event coalescing for the asset import pipeline."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from infernux.engine.path_utils import path_key, portable_path, resolved_path


class AssetFsEventKind(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    META_DELETED = "meta_deleted"
    META_MODIFIED = "meta_modified"


@dataclass(frozen=True, slots=True)
class AssetFsEvent:
    kind: AssetFsEventKind
    path: str
    destination: str = ""
    guid_hint: str = ""
    observed_at: float = 0.0
    attempt: int = 0


@dataclass(slots=True)
class _PendingEvent:
    event: AssetFsEvent
    ready_at: float


def _absolute_path(path: str) -> str:
    if not path:
        raise ValueError("asset event path cannot be empty")
    return resolved_path(path)


def _path_key(path: str) -> str:
    return path_key(path)


_DOCUMENT_STORE_TEMP_PATTERN = re.compile(r"\.tmp\.\d+\.\d+$")


def is_document_store_temporary_path(path: str) -> bool:
    """Return whether *path* uses AtomicFile's exact temporary-file suffix."""
    if not path:
        return False
    normalized = portable_path(path)
    return _DOCUMENT_STORE_TEMP_PATTERN.search(normalized) is not None


def _event_references_temporary_path(event: AssetFsEvent) -> bool:
    return is_document_store_temporary_path(event.path) or is_document_store_temporary_path(event.destination)


class ImportCoordinator:
    """Accept watcher-thread events and publish coalesced main-thread work."""

    def __init__(
        self,
        *,
        debounce_seconds: float = 0.12,
        delete_grace_seconds: float = 0.75,
        meta_delete_grace_seconds: float = 0.6,
        retry_delay_seconds: float = 0.15,
        max_attempts: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min(debounce_seconds, delete_grace_seconds, meta_delete_grace_seconds, retry_delay_seconds) < 0:
            raise ValueError("coordinator delays must be non-negative")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._debounce_seconds = debounce_seconds
        self._delete_grace_seconds = delete_grace_seconds
        self._meta_delete_grace_seconds = meta_delete_grace_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._max_attempts = max_attempts
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingEvent] = {}
        self._inflight: dict[int, AssetFsEvent] = {}

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def submit(
        self,
        kind: AssetFsEventKind,
        path: str,
        *,
        destination: str = "",
        guid_hint: str = "",
        observed_at: float | None = None,
        debounce_seconds: float | None = None,
        source_registered: bool = True,
    ) -> None:
        if not isinstance(kind, AssetFsEventKind):
            raise TypeError("kind must be AssetFsEventKind")
        if debounce_seconds is not None and debounce_seconds < 0:
            raise ValueError("debounce_seconds must be non-negative or None")
        now = self._clock() if observed_at is None else observed_at
        normalized_path = _absolute_path(path)
        normalized_destination = _absolute_path(destination) if destination else ""
        with self._lock:
            if kind is AssetFsEventKind.MOVED and not source_registered:
                source_key = "asset:" + _path_key(normalized_path)
                pending = self._pending.get(source_key)
                predecessor = pending.event if pending and pending.event.kind is AssetFsEventKind.MOVED else next(
                    (active for active in self._inflight.values()
                     if active.kind is AssetFsEventKind.MOVED and self._event_key(active) == source_key), None,
                )
                if predecessor is not None:
                    # The DB still names A while A -> B is queued or owned.
                    # B -> C continues that identity; it is not a staging-file
                    # publication merely because B is not registered yet.
                    source_registered = True
                    guid_hint = predecessor.guid_hint or guid_hint
            published_source = ""
            if kind is AssetFsEventKind.MOVED and (
                not source_registered or is_document_store_temporary_path(normalized_path)
            ):
                if not normalized_destination or is_document_store_temporary_path(normalized_destination):
                    return
                published_source = normalized_path
                kind = AssetFsEventKind.MODIFIED
                normalized_path = normalized_destination
                normalized_destination = ""
            elif is_document_store_temporary_path(normalized_path) or is_document_store_temporary_path(
                normalized_destination
            ):
                return
            event = AssetFsEvent(
                kind=kind,
                path=normalized_path,
                destination=normalized_destination,
                guid_hint=guid_hint.strip(),
                observed_at=now,
            )
            if kind is AssetFsEventKind.MOVED and not event.destination:
                raise ValueError("moved event requires a destination")
            if published_source:
                # Rename publication consumes any queued events for the
                # staging file; only the destination's content survives.
                self._pending.pop("asset:" + _path_key(published_source), None)
            self._submit_locked(event, now, debounce_seconds=debounce_seconds)

    def _delay_for(
        self,
        kind: AssetFsEventKind,
        debounce_seconds: float | None = None,
    ) -> float:
        if kind is AssetFsEventKind.DELETED:
            return self._delete_grace_seconds
        if kind is AssetFsEventKind.META_DELETED:
            return self._meta_delete_grace_seconds
        return self._debounce_seconds if debounce_seconds is None else debounce_seconds

    def _event_key(self, event: AssetFsEvent) -> str:
        if event.kind in (AssetFsEventKind.META_DELETED, AssetFsEventKind.META_MODIFIED):
            return "meta:" + _path_key(event.path)
        target = event.destination if event.kind is AssetFsEventKind.MOVED else event.path
        return "asset:" + _path_key(target)

    def _find_guid_pair_locked(self, event: AssetFsEvent) -> tuple[str, _PendingEvent] | None:
        if not event.guid_hint or event.kind not in (AssetFsEventKind.CREATED, AssetFsEventKind.DELETED):
            return None
        opposite = (
            AssetFsEventKind.DELETED if event.kind is AssetFsEventKind.CREATED else AssetFsEventKind.CREATED
        )
        for key, pending in self._pending.items():
            candidate = pending.event
            # GUID pairing identifies a move between paths. Same-path events
            # retain their order: create/delete cancels; delete/create replaces.
            if (candidate.kind is opposite and candidate.guid_hint == event.guid_hint
                    and _path_key(candidate.path) != _path_key(event.path)):
                return key, pending
        return None

    def _submit_locked(
        self,
        event: AssetFsEvent,
        now: float,
        *,
        debounce_seconds: float | None = None,
    ) -> str | None:
        debounce = (
            self._debounce_seconds
            if debounce_seconds is None
            else debounce_seconds
        )
        pair = self._find_guid_pair_locked(event)
        if pair is not None:
            pair_key, pending = pair
            del self._pending[pair_key]
            deleted = event if event.kind is AssetFsEventKind.DELETED else pending.event
            created = event if event.kind is AssetFsEventKind.CREATED else pending.event
            event = AssetFsEvent(
                AssetFsEventKind.MOVED,
                deleted.path,
                destination=created.path,
                guid_hint=event.guid_hint,
                observed_at=event.observed_at,
            )

        if event.kind is AssetFsEventKind.MOVED:
            source_key = "asset:" + _path_key(event.path)
            destination_key = "asset:" + _path_key(event.destination)
            for key, pending in list(self._pending.items()):
                previous = pending.event
                if previous.kind is AssetFsEventKind.MOVED and _path_key(previous.destination) == _path_key(event.path):
                    event = replace(event, path=previous.path, guid_hint=event.guid_hint or previous.guid_hint)
                    del self._pending[key]
                    source_key = "asset:" + _path_key(event.path)
                    break
            self._pending.pop(source_key, None)
            self._pending.pop(destination_key, None)
            self._pending[self._event_key(event)] = _PendingEvent(
                event,
                now + self._delay_for(
                    AssetFsEventKind.MOVED,
                    debounce_seconds,
                ),
            )
            return self._event_key(event)

        key = self._event_key(event)
        previous_pending = self._pending.get(key)
        if previous_pending is None:
            self._pending[key] = _PendingEvent(
                event,
                now + self._delay_for(event.kind, debounce_seconds),
            )
            return key

        previous = previous_pending.event
        if event.kind in (AssetFsEventKind.META_DELETED, AssetFsEventKind.META_MODIFIED):
            self._pending[key] = _PendingEvent(
                event,
                now + self._delay_for(event.kind, debounce_seconds),
            )
            return key

        if previous.kind is AssetFsEventKind.MOVED:
            if event.kind is AssetFsEventKind.DELETED:
                collapsed = AssetFsEvent(
                    AssetFsEventKind.DELETED,
                    previous.path,
                    guid_hint=previous.guid_hint or event.guid_hint,
                    observed_at=event.observed_at,
                )
                self._pending.pop(key, None)
                self._pending[self._event_key(collapsed)] = _PendingEvent(
                    collapsed, now + self._delete_grace_seconds
                )
                return self._event_key(collapsed)
            else:
                previous_pending.event = replace(
                    previous, observed_at=event.observed_at, attempt=event.attempt,
                    guid_hint=event.guid_hint or previous.guid_hint,
                )
                previous_pending.ready_at = now + debounce
            return key

        if previous.kind is AssetFsEventKind.CREATED and event.kind is AssetFsEventKind.DELETED:
            # An older import can already own this identity. Cancelling only
            # the queued create/delete pair would hide its required retirement.
            if not any(self._event_key(active) == key for active in self._inflight.values()):
                del self._pending[key]
                return None
        if previous.kind is AssetFsEventKind.DELETED and event.kind is AssetFsEventKind.CREATED:
            event = replace(event, kind=AssetFsEventKind.MODIFIED)
        elif previous.kind is AssetFsEventKind.CREATED and event.kind is AssetFsEventKind.MODIFIED:
            event = replace(previous, observed_at=event.observed_at, attempt=event.attempt,
                            guid_hint=event.guid_hint or previous.guid_hint)
        elif event.kind is AssetFsEventKind.DELETED:
            pass
        elif previous.kind is AssetFsEventKind.DELETED:
            # Atomic replacement on Windows can report the old target as
            # deleted before the DocumentStore temp file is moved into place.
            # That temp move is normalized to MODIFIED above, so it must revive
            # the pending asset instead of allowing the stale delete to win.
            if event.kind is not AssetFsEventKind.MODIFIED:
                event = previous
        elif previous.kind is AssetFsEventKind.MODIFIED and event.kind is AssetFsEventKind.CREATED:
            event = replace(event, kind=AssetFsEventKind.MODIFIED)

        self._pending[key] = _PendingEvent(
            event,
            now + self._delay_for(event.kind, debounce_seconds),
        )
        return key

    def drain(
        self,
        *,
        force: bool = False,
        now: float | None = None,
        max_events: int | None = None,
    ) -> list[AssetFsEvent]:
        """Claim ready work; the owner must complete, retry or defer each event."""
        if max_events is not None and max_events < 1:
            raise ValueError("max_events must be positive or None")
        current = self._clock() if now is None else now
        with self._lock:
            ready = [
                (key, pending)
                for key, pending in self._pending.items()
                if force or pending.ready_at <= current
            ]
            ready.sort(key=lambda item: item[1].event.observed_at)
            if not force and max_events is not None:
                ready = ready[:max_events]
            for key, _pending in ready:
                del self._pending[key]
                self._inflight[id(_pending.event)] = _pending.event
        return [pending.event for _key, pending in ready]

    def complete(self, event: AssetFsEvent) -> None:
        """Release one owner claim without changing later watcher events."""
        with self._lock:
            self._inflight.pop(id(event), None)

    def _requeue_locked(self, event: AssetFsEvent, current: float, *, retry: bool, asset_published: bool) -> bool:
        if self._inflight.pop(id(event), None) is not event:
            return False
        if asset_published and event.kind in (AssetFsEventKind.CREATED, AssetFsEventKind.MOVED):
            # The owner has already published identity, but content work is
            # waiting. Repeating a create/move would undo that progress or
            # cancel a later deletion of the now-registered asset.
            event = replace(event, kind=AssetFsEventKind.MODIFIED,
                            path=event.destination or event.path, destination="")
        key = self._event_key(event)
        following = (key, self._pending[key]) if key in self._pending else None
        if following is None and event.kind not in (AssetFsEventKind.META_DELETED, AssetFsEventKind.META_MODIFIED):
            target = _path_key(event.destination or event.path)
            following = next(((k, p) for k, p in self._pending.items()
                              if p.event.kind is AssetFsEventKind.MOVED and _path_key(p.event.path) == target), None)
        if following is None:
            following = self._find_guid_pair_locked(event)
        if following is not None:
            # Compose in arrival order, not retry time: the old claim owns the
            # original move identity, while the successor owns content/budget
            # and its already established debounce or deletion grace deadline.
            following_key, pending = following
            del self._pending[following_key]
            self._pending[key] = _PendingEvent(event, current)
            result_key = self._submit_locked(pending.event, current)
            if result_key is not None:
                self._pending[result_key].ready_at = pending.ready_at
            return True
        if retry and event.attempt + 1 >= self._max_attempts:
            return False
        queued = replace(event, attempt=event.attempt + int(retry))
        self._pending[key] = _PendingEvent(queued, current + self._retry_delay_seconds)
        return True

    def retry(self, event: AssetFsEvent, *, now: float | None = None, asset_published: bool = False) -> bool:
        if _event_references_temporary_path(event):
            return False
        current = self._clock() if now is None else now
        with self._lock:
            return self._requeue_locked(event, current, retry=True, asset_published=asset_published)

    def defer(self, event: AssetFsEvent, *, now: float | None = None, asset_published: bool = False) -> None:
        """Requeue known in-flight owner work without consuming retry budget.

        A DocumentStore write has its own completion ticket.  Watcher echoes
        that arrive before that ticket completes are not failed imports, so
        counting them against the bounded importer retry budget can turn a
        healthy asynchronous save into a false terminal error.
        """
        if _event_references_temporary_path(event):
            return
        current = self._clock() if now is None else now
        with self._lock:
            self._requeue_locked(event, current, retry=False, asset_published=asset_published)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._inflight.clear()
