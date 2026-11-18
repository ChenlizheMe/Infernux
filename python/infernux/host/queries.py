"""Signal-driven queries. No frame polling occurs without a domain signal."""

from collections import defaultdict
from copy import deepcopy
from threading import Condition
import uuid

from .operations import OperationError, OperationKind


_services = set()


def emit_signal(name, payload=None):
    for service in tuple(_services):
        service.signal(name, payload)


class QueryService:
    def __init__(self, registry):
        from concurrent.futures import ThreadPoolExecutor

        self.registry = registry
        self._condition = Condition()
        self._queries = {}
        self._signals = defaultdict(set)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="InfernuxQuery")
        self._closed = False
        self._stop_listeners = []
        self._activity_listener = None
        _services.add(self)

    def start(self, operation, arguments=None, signals=None, owner=""):
        provider = self.registry.get(operation)
        if provider.schema.kind is not OperationKind.QUERY:
            raise OperationError("query.requires_read", "Only query operations can be observed")
        if signals is None:
            domain = operation.partition(".")[2].partition(".")[0]
            signals = {
                "scene": ("scene.changed", "world.changed"),
                "project": ("scene.changed", "runtime.changed", "assets.changed"),
                "asset": ("assets.changed",),
                "runtime": ("runtime.changed", "world.changed"),
                "player": ("player.changed",),
                "console": ("console.changed",),
                "capture": ("capture.completed",),
                "plugin": ("plugins.changed", "assets.changed", "operations.changed"),
                "theme": ("themes.changed", "assets.changed"),
            }.get(domain, ("world.changed",))
        signals = tuple(dict.fromkeys((*signals, "operations.changed")))
        identity = uuid.uuid4().hex
        with self._condition:
            if self._closed:
                raise OperationError("query.closed", "The observation service is closed")
            if len(self._queries) >= 256:
                raise OperationError("query.capacity", "The observation service reached its 256-query capacity")
            self._queries[identity] = {"id": identity, "operation": operation, "arguments": arguments or {},
                                       "signals": tuple(signals), "revision": 0, "status": "pending",
                                       "value": None, "error": None, "scheduled": False, "dirty": False,
                                       "owner": owner, "provider": provider, "terminal": False}
            for name in signals:
                self._signals[name].add(identity)
        self._notify_activity()
        self._schedule(identity)
        return {"id": identity, "status": "pending"}

    def signal(self, name, payload=None):
        with self._condition:
            identities = tuple(self._signals.get(name, ()))
        for identity in identities:
            self._schedule(identity)

    def _schedule(self, identity):
        with self._condition:
            query = self._queries.get(identity)
            if query is None or self._closed or query["terminal"]:
                return
            query["dirty"] = True
            if query["scheduled"]:
                return
            query["scheduled"] = True
        self._executor.submit(self._evaluate, identity)

    def _evaluate(self, identity):
        while True:
            with self._condition:
                query = self._queries.get(identity)
                if query is None or self._closed:
                    return
                query["dirty"] = False
                operation, arguments = query["operation"], query["arguments"]
            try:
                if self.registry.get(operation) is not query["provider"]:
                    raise OperationError("query.provider_changed", "The observed provider was retired; start a new observation")
                value = deepcopy(self.registry.execute(operation, arguments, capabilities=("*",),
                                 expected_kind=OperationKind.QUERY, expected_provider=query["provider"]))
                if self.registry.get(operation) is not query["provider"]:
                    raise OperationError("query.provider_changed", "The observed provider was retired during evaluation")
                error = None
                terminal = False
                finished = operation == "infernux.capture.status" and bool(value.get("terminal"))
            except Exception as exc:
                value, error = None, str(exc)
                terminal = isinstance(exc, OperationError) and exc.code in {
                    "operation.not_found", "operation.unavailable", "operation.provider_changed", "query.provider_changed",
                    "scene.object_not_found", "scene.component_not_found", "world.expired", "object.expired",
                }
                finished = False
            with self._condition:
                if identity not in self._queries:
                    return
                if query["revision"] == 0 or value != query["value"] or error != query["error"]:
                    query.update(value=value, error=error, status="closed" if terminal else "failed" if error else "ready",
                                 terminal=terminal or finished,
                                 revision=query["revision"] + 1)
                    self._condition.notify_all()
                if terminal or finished:
                    for name in query["signals"]:
                        self._signals[name].discard(identity)
                    query["scheduled"] = False
                    break
                if not query["dirty"]:
                    query["scheduled"] = False
                    return
        self._notify_activity()

    @staticmethod
    def _snapshot(query):
        return deepcopy({key: query[key] for key in ("id", "operation", "revision", "status", "value", "error", "terminal")})

    def snapshot(self, query_id):
        with self._condition:
            query = self._queries.get(query_id)
            if query is None:
                raise OperationError("query.not_found", f"Observation is closed or unknown: {query_id}")
            return self._snapshot(query)

    def wait(self, query_id, after=0, timeout=25.0):
        with self._condition:
            self._condition.wait_for(lambda: self._closed or query_id not in self._queries
                                     or self._queries[query_id]["terminal"]
                                     or self._queries[query_id]["revision"] > after, timeout=min(max(float(timeout), 0.0), 25.0))
            if self._closed or query_id not in self._queries:
                return {"id": query_id, "revision": after, "status": "closed", "value": None, "error": None}
            return self._snapshot(self._queries[query_id])

    def stop(self, query_id):
        with self._condition:
            query = self._queries.pop(query_id, None)
            if query is None:
                return False
            for name in query["signals"]:
                self._signals[name].discard(query_id)
            self._condition.notify_all()
        for callback in tuple(self._stop_listeners):
            callback(query_id)
        self._notify_activity()
        return True

    def add_stop_listener(self, callback):
        self._stop_listeners.append(callback)

    def has_subscribers(self, *signals):
        with self._condition:
            return any(self._signals.get(signal) for signal in signals)

    def set_activity_listener(self, callback):
        self._activity_listener = callback

    def _notify_activity(self):
        if self._activity_listener is not None:
            self._activity_listener()

    def close_owner(self, owner):
        with self._condition:
            identities = [identity for identity, query in self._queries.items() if query["owner"] == owner]
        for identity in identities:
            self.stop(identity)
        return len(identities)

    def close(self):
        _services.discard(self)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            identities = tuple(self._queries)
        for identity in identities:
            self.stop(identity)
        self._executor.shutdown(wait=False, cancel_futures=True)
