"""Thread-safe file-system event coalescing for the asset import pipeline."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable
from infernux.lib import NativeDocumentStore

from infernux.engine.path_utils import path_key as _path_key, portable_path, resolved_path


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
    unresolved_identity: bool = False
    unresolved_from: str = ""


@dataclass(slots=True)
class _PendingEvent:
    event: AssetFsEvent
    ready_at: float


def _absolute_path(path: str) -> str:
    if not path:
        raise ValueError("asset event path cannot be empty")
    return resolved_path(path)


def is_document_store_temporary_path(path: str) -> bool:
    """Return whether *path* uses AtomicFile's exact temporary-file suffix."""
    return bool(path) and NativeDocumentStore.is_temporary_path(path)


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
        self._dispatching: set[int] = set()

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
        unresolved_identity: bool = False,
    ) -> None:
        if not isinstance(kind, AssetFsEventKind):
            raise TypeError("kind must be AssetFsEventKind")
        if debounce_seconds is not None and debounce_seconds < 0:
            raise ValueError("debounce_seconds must be non-negative or None")
        now = self._clock() if observed_at is None else observed_at
        normalized_path = _absolute_path(path)
        normalized_destination = _absolute_path(destination) if destination else ""
        with self._lock:
            unresolved_from = ""
            first_observed = now
            if kind is AssetFsEventKind.MOVED and not source_registered:
                source_key = "asset:" + _path_key(normalized_path)
                pending = self._pending.get(source_key)
                predecessor = pending.event if pending and (
                    pending.event.kind is AssetFsEventKind.MOVED or pending.event.unresolved_identity
                ) else next(
                    (active for active in self._inflight.values()
                     if (active.kind is AssetFsEventKind.MOVED or active.unresolved_identity)
                     and self._event_key(active) == source_key), None,
                )
                if predecessor is not None and predecessor.kind is AssetFsEventKind.MOVED:
                    # The DB still names A while A -> B is queued or owned.
                    # B -> C continues that identity; it is not a staging-file
                    # publication merely because B is not registered yet.
                    source_registered = True
                    guid_hint = predecessor.guid_hint or guid_hint
                elif (predecessor is not None and predecessor.unresolved_identity and not guid_hint
                      and not is_document_store_temporary_path(normalized_path)):
                    unresolved_identity = True
                    unresolved_from = predecessor.unresolved_from
                    first_observed = predecessor.observed_at
                    if id(predecessor) in self._inflight:
                        unresolved_from = predecessor.path
            published_source = ""
            if kind is AssetFsEventKind.MOVED and (
                not source_registered or is_document_store_temporary_path(normalized_path)
            ):
                if not normalized_destination or is_document_store_temporary_path(normalized_destination):
                    return
                published_source = normalized_path
                kind = AssetFsEventKind.CREATED if unresolved_identity else AssetFsEventKind.MODIFIED
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
                observed_at=first_observed,
                unresolved_identity=unresolved_identity and kind is AssetFsEventKind.CREATED,
                unresolved_from=unresolved_from,
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
        if (not event.guid_hint or event.unresolved_identity
                or event.kind not in (AssetFsEventKind.CREATED, AssetFsEventKind.DELETED)):
            return None
        opposite = (
            AssetFsEventKind.DELETED if event.kind is AssetFsEventKind.CREATED else AssetFsEventKind.CREATED
        )
        for key, pending in self._pending.items():
            candidate = pending.event
            # GUID pairing identifies a move between paths. Same-path events
            # retain their order: create/delete cancels; delete/create replaces.
            if (candidate.kind is opposite and not candidate.unresolved_identity and candidate.guid_hint == event.guid_hint
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
            event = replace(event, kind=AssetFsEventKind.MODIFIED, unresolved_identity=False, unresolved_from="")
        elif previous.kind is AssetFsEventKind.CREATED and event.kind is AssetFsEventKind.MODIFIED:
            observed_at = previous.observed_at if previous.unresolved_identity else event.observed_at
            event = replace(previous, observed_at=observed_at, attempt=event.attempt,
                            guid_hint=event.guid_hint or previous.guid_hint)
        elif previous.unresolved_identity and event.unresolved_identity:
            event = replace(event, observed_at=min(previous.observed_at, event.observed_at),
                            unresolved_from=event.unresolved_from or previous.unresolved_from)
        elif event.kind is AssetFsEventKind.DELETED:
            pass
        elif previous.kind is AssetFsEventKind.DELETED:
            # Atomic replacement on Windows can report the old target as
            # deleted before the DocumentStore temp file is moved into place.
            # That temp move is normalized to MODIFIED above, so it must revive
            # the pending asset instead of allowing the stale delete to win.
            if event.kind is not AssetFsEventKind.MODIFIED:
                event = previous
        elif (previous.kind is AssetFsEventKind.MODIFIED and event.kind is AssetFsEventKind.CREATED
              and not event.unresolved_identity):
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
            ordered = sorted(self._pending.items(), key=lambda item: item[1].event.observed_at)
            # Hold registered deletes only for creates that were observed in
            # their grace window and are eligible for their first resolution.
            # A stream of newer creates or repeated deferrals cannot renew it.
            candidates = {
                key: self._identity_candidate_time(pending)
                for key, pending in ordered if pending.event.unresolved_identity
            }
            first_candidate = min(candidates.values(), default=float("inf"))
            protected_deletes = {
                key: pending.ready_at for key, pending in ordered
                if pending.event.kind is AssetFsEventKind.DELETED and pending.event.guid_hint
                and (bool(candidates) if force else first_candidate <= pending.ready_at)
            }
            if force:
                # A single finite batch may already have claimed both events
                # when the create's metadata identifies the move.
                ready = sorted(ordered, key=lambda item: item[0] in protected_deletes)
            elif max_events is not None:
                blocked = {
                    identity
                    for event in self._inflight.values()
                    for identity in self._identity_keys(self._event_key(event), event)
                }
                ready = []
                for key, pending in ordered:
                    identities = self._identity_keys(key, pending.event)
                    if (key not in protected_deletes and pending.ready_at <= current
                            and not blocked.intersection(identities)):
                        ready.append((key, pending))
                    blocked.update(identities)
                due_delete = max(
                    (deadline for deadline in protected_deletes.values() if deadline <= current),
                    default=float("-inf"),
                )
                ready.sort(key=lambda item: (
                    candidates.get(item[0], float("inf")) > due_delete,
                    not any(path.lower().endswith(".py")
                            for path in (item[1].event.path, item[1].event.destination)),
                ))
                ready = ready[:max_events]
            else:
                ready = [(key, pending) for key, pending in ordered
                         if key not in protected_deletes and pending.ready_at <= current]
            for key, _pending in ready:
                del self._pending[key]
                self._inflight[id(_pending.event)] = _pending.event
        return [pending.event for _key, pending in ready]

    def _identity_candidate_time(self, pending: _PendingEvent) -> float:
        window = max(self._debounce_seconds, self._retry_delay_seconds)
        return max(pending.event.observed_at, pending.ready_at - window)

    @staticmethod
    def _identity_keys(key: str, event: AssetFsEvent) -> tuple[str, ...]:
        identities = (key.partition(":")[2],)
        if event.kind is AssetFsEventKind.MOVED:
            identities += (_path_key(event.path),)
        if event.unresolved_from:
            identities += (_path_key(event.unresolved_from),)
        if event.guid_hint and not event.unresolved_identity:
            identities += ("guid:" + event.guid_hint,)
        return identities

    def begin_dispatch(self, event: AssetFsEvent) -> bool:
        """Start a claim unless a preceding create already consumed it."""
        with self._lock:
            claim = id(event)
            if self._inflight.get(claim) is not event or claim in self._dispatching:
                return False
            if event.kind is AssetFsEventKind.DELETED and event.guid_hint:
                # A move successor can arrive while an earlier create is
                # being dispatched from a captured batch.
                deadline = event.observed_at + self._delete_grace_seconds
                if any(item.event.unresolved_identity and self._identity_candidate_time(item) <= deadline
                       for item in self._pending.values()):
                    del self._inflight[claim]
                    self._pending.setdefault(self._event_key(event), _PendingEvent(event, deadline))
                    return False
            self._dispatching.add(claim)
            return True

    def resolve_created_move(self, event: AssetFsEvent, guid: str, source_path: str) -> AssetFsEvent | None:
        """Turn a claimed create and an undispatched delete into a claimed move."""
        guid = guid.strip()
        if event.kind is not AssetFsEventKind.CREATED or not guid or not source_path:
            return None
        source_key = "asset:" + _path_key(source_path)
        if source_key == self._event_key(event):
            return None
        with self._lock:
            if self._inflight.get(id(event)) is not event or self._has_source_successor_locked(event):
                return None
            pending = self._pending.get(source_key)
            deleted = pending.event if pending else next(
                (candidate for claim, candidate in self._inflight.items()
                 if claim not in self._dispatching and self._event_key(candidate) == source_key),
                None,
            )
            if deleted is None or deleted.kind is not AssetFsEventKind.DELETED or deleted.guid_hint != guid:
                return None
            move = AssetFsEvent(
                AssetFsEventKind.MOVED,
                deleted.path,
                destination=event.path,
                guid_hint=guid,
                observed_at=max(event.observed_at, deleted.observed_at),
            )
            if pending:
                del self._pending[source_key]
            else:
                del self._inflight[id(deleted)]
            del self._inflight[id(event)]
            self._dispatching.discard(id(event))
            self._inflight[id(move)] = move
            return move

    def complete(self, event: AssetFsEvent) -> None:
        """Release one owner claim without changing later watcher events."""
        with self._lock:
            self._inflight.pop(id(event), None)
            self._dispatching.discard(id(event))

    def has_source_successor(self, event: AssetFsEvent) -> bool:
        """Whether a later source event replaces content at this claimed path."""
        with self._lock:
            return self._has_source_successor_locked(event)

    def _has_source_successor_locked(self, event: AssetFsEvent) -> bool:
        target = _path_key(event.destination or event.path)
        return any(key.startswith("asset:") and target in self._identity_keys(key, item.event)
                   for key, item in self._pending.items())

    def _requeue_locked(
        self, event: AssetFsEvent, current: float, *, retry: bool, asset_published: bool, authored_guid: str,
    ) -> bool:
        if self._inflight.pop(id(event), None) is not event:
            return False
        self._dispatching.discard(id(event))
        if authored_guid and event.unresolved_identity:
            event = replace(event, guid_hint=authored_guid.strip())
        if asset_published and event.kind in (AssetFsEventKind.CREATED, AssetFsEventKind.MOVED):
            # The owner has already published identity, but content work is
            # waiting. Repeating a create/move would undo that progress or
            # cancel a later deletion of the now-registered asset.
            event = replace(event, kind=AssetFsEventKind.MODIFIED,
                            path=event.destination or event.path, destination="",
                            unresolved_identity=False, unresolved_from="")
        key = self._event_key(event)
        following = (key, self._pending[key]) if key in self._pending else None
        if following is None and event.kind not in (AssetFsEventKind.META_DELETED, AssetFsEventKind.META_MODIFIED):
            target = _path_key(event.destination or event.path)
            following = next(((k, p) for k, p in self._pending.items()
                              if ((p.event.kind is AssetFsEventKind.MOVED and _path_key(p.event.path) == target)
                                  or (p.event.unresolved_from and _path_key(p.event.unresolved_from) == target))), None)
        if following is None:
            following = self._find_guid_pair_locked(event)
        if following is not None:
            # Compose in arrival order, not retry time: the old claim owns the
            # original move identity, while the successor owns content/budget
            # and its already established debounce or deletion grace deadline.
            following_key, pending = following
            del self._pending[following_key]
            successor = pending.event
            if successor.unresolved_from and following_key != key:
                # The old path disappeared while identity was being read.
                # Keep the latest create and let its own sidecar confirm it.
                successor = replace(successor, guid_hint=event.guid_hint or successor.guid_hint, unresolved_from="")
            else:
                self._pending[key] = _PendingEvent(event, current)
            result_key = self._submit_locked(successor, current)
            if result_key is not None:
                self._pending[result_key].ready_at = pending.ready_at
            return True
        if retry and event.attempt + 1 >= self._max_attempts:
            return False
        queued = replace(event, attempt=event.attempt + int(retry))
        self._pending[key] = _PendingEvent(queued, current + self._retry_delay_seconds)
        return True

    def retry(
        self, event: AssetFsEvent, *, now: float | None = None, asset_published: bool = False, authored_guid: str = "",
    ) -> bool:
        if _event_references_temporary_path(event):
            return False
        current = self._clock() if now is None else now
        with self._lock:
            return self._requeue_locked(event, current, retry=True, asset_published=asset_published,
                                        authored_guid=authored_guid)

    def defer(
        self, event: AssetFsEvent, *, now: float | None = None, asset_published: bool = False, authored_guid: str = "",
    ) -> None:
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
            self._requeue_locked(event, current, retry=False, asset_published=asset_published, authored_guid=authored_guid)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._inflight.clear()
            self._dispatching.clear()
