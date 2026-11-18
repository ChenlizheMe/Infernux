"""Delivery cursors for named consumers and real conversation cycles."""

from collections import OrderedDict
from threading import RLock
import uuid


class ReportDelivery:
    def __init__(self, queries):
        self.queries = queries
        self._lock = RLock()
        self._reports = {}
        self._cycles = OrderedDict()
        self._seen_cycles = {}
        self._consumer_owners = {}
        queries.add_stop_listener(self.remove_query)

    def add(self, query_id, consumer, reports=3, mode="changes", owner="", close_query=True):
        if mode not in {"changes", "turns", "adaptive"}:
            raise ValueError("Report mode must be changes, turns or adaptive")
        if type(reports) is not int or reports < 1:
            raise ValueError("Report count must be positive")
        identity = uuid.uuid4().hex
        with self._lock:
            self.queries.snapshot(query_id)
            if len(self._reports) >= 1024:
                raise ValueError("Report delivery reached its 1024-observation capacity")
            if consumer not in self._consumer_owners and len(self._consumer_owners) >= 1024:
                raise ValueError("Report delivery reached its 1024-consumer capacity")
            self._consumer_owners.setdefault(consumer, set()).add(owner)
            seen = self._seen_cycles.setdefault(consumer, set())
            if consumer in self._cycles:
                seen.add(self._cycles[consumer])
            self._reports[identity] = {"id": identity, "query_id": query_id, "consumer": consumer,
                                       "remaining": reports, "mode": mode, "revision": -1,
                                       "owner": owner, "close_query": bool(close_query),
                                       "cycles": set(seen)}
        return {"id": identity, "query_id": query_id, "remaining": reports, "mode": mode,
                "turn_delivery_requires": "A real cycle_id from the client, hook or conversation_turn tool"}

    def consume(self, consumer, cycle_id=None, changes_only=False):
        delivered = []
        completed = []
        with self._lock:
            if cycle_id is not None:
                if not isinstance(cycle_id, str) or not cycle_id.strip():
                    raise ValueError("cycle_id must identify a real user turn")
                seen = self._seen_cycles.setdefault(consumer, set())
                replay = cycle_id in seen
                if not replay:
                    self._cycles[consumer] = cycle_id
                seen.add(cycle_id)
                if consumer in self._cycles:
                    self._cycles.move_to_end(consumer)
                while len(self._cycles) > 1024:
                    inactive = next(key for key in self._cycles if key not in self._consumer_owners)
                    del self._cycles[inactive]
                    self._seen_cycles.pop(inactive, None)
            # A hook without a reliable user-turn ID may read changed results;
            # the previous known turn must not suppress all its later changes.
            cycle_id = None if changes_only else cycle_id if cycle_id is not None else self._cycles.get(consumer)
            for report in tuple(self._reports.values()):
                if report["consumer"] != consumer or report["remaining"] == 0:
                    continue
                if changes_only and report["mode"] == "turns":
                    continue
                try:
                    current = self.queries.snapshot(report["query_id"])
                except Exception as exc:
                    if isinstance(exc, KeyError) or getattr(exc, "code", None) == "query.not_found":
                        completed.append(report["id"])
                        continue
                    raise
                if current["revision"] == 0 and report["mode"] != "turns":
                    continue
                if report["mode"] == "turns":
                    due = cycle_id is not None and cycle_id not in report["cycles"]
                else:
                    due = (report["revision"] != current["revision"]
                           and (cycle_id is None or cycle_id not in report["cycles"]))
                terminal = current["status"] == "closed" or current.get("terminal", False)
                due = due or terminal
                if not due:
                    continue
                already_reported = cycle_id is not None and cycle_id in report["cycles"]
                counted = not already_reported and (report["mode"] != "turns" or cycle_id is not None)
                if cycle_id is not None:
                    report["cycles"].add(cycle_id)
                report.update(revision=current["revision"], remaining=report["remaining"] - int(counted))
                delivered.append({"id": report["id"], "remaining": report["remaining"], "observation": current})
                if report["remaining"] == 0 or terminal:
                    completed.append(report["id"])
        for identity in completed:
            self.remove(identity)
        return delivered

    def remove(self, report_id):
        with self._lock:
            report = self._reports.pop(report_id, None)
            close_query = report is not None and report["close_query"] and not any(
                other["query_id"] == report["query_id"] for other in self._reports.values())
        if close_query:
            self.queries.stop(report["query_id"])
        return report is not None

    def remove_query(self, query_id):
        with self._lock:
            for identity, report in tuple(self._reports.items()):
                if report["query_id"] == query_id:
                    del self._reports[identity]

    def close_owner(self, owner):
        with self._lock:
            identities = [identity for identity, report in self._reports.items() if report["owner"] == owner]
        for identity in identities:
            self.remove(identity)
        with self._lock:
            # IDs live for the owning conversation/session, including after one
            # report exhausts its quota. Never evict a live ID and count it again.
            for consumer, owners in tuple(self._consumer_owners.items()):
                owners.discard(owner)
                if not owners:
                    del self._consumer_owners[consumer]
                    self._seen_cycles.pop(consumer, None)
                    self._cycles.pop(consumer, None)
        return len(identities)

    def close(self):
        with self._lock:
            self._reports.clear()
            self._cycles.clear()
            self._seen_cycles.clear()
            self._consumer_owners.clear()
