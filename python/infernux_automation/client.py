"""One request maps to one engine operation. Writes are never replayed."""

import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import uuid


class AutomationError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details


class AutomationClient:
    def __init__(self, project=None, *, endpoint=None, instance_id=None, timeout=30.0, session_id=None):
        if endpoint is None:
            descriptor = Path(project).resolve() / "Cache" / "Automation" / "engine.json"
            try:
                connection = json.loads(descriptor.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise AutomationError("engine.disconnected", f"Engine is not connected: {descriptor}") from exc
            try:
                endpoint, instance_id = connection["endpoint"], connection["instance_id"]
            except (KeyError, TypeError) as exc:
                raise AutomationError("engine.disconnected", f"Invalid engine descriptor: {descriptor}") from exc
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")) or not instance_id:
            raise AutomationError("engine.disconnected", "An engine endpoint and instance ID are required")
        self.endpoint = endpoint.rstrip("/")
        self.instance_id = instance_id
        self.timeout = timeout
        self.session_id = session_id or uuid.uuid4().hex

    def request(self, method, **arguments):
        return self._request(method, arguments, self.timeout)

    def _request(self, method, arguments, timeout):
        payload = json.dumps({"instance_id": self.instance_id, "session_id": self.session_id,
                              "arguments": arguments}, ensure_ascii=False).encode()
        request = Request(self.endpoint + "/" + method, payload, {"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                value = json.load(response)
        except HTTPError as exc:
            with exc:
                value = json.load(exc)
        except (URLError, TimeoutError, OSError) as exc:
            raise AutomationError("transport.unknown_outcome", f"Engine request did not finish: {method}. {exc}") from exc
        if not value["ok"]:
            error = value["error"]
            raise AutomationError(error["code"], error["message"], error.get("details"))
        return value["result"]

    def info(self):
        return self.request("info")

    def operations(self, query="", limit=50):
        return self.request("operations", query=query, limit=limit)

    def schema(self, operation):
        return self.request("schema", operation=operation)

    def call(self, operation, arguments=None):
        return self.request("call", operation=operation, arguments=arguments or {})

    def submit(self, operation, arguments=None):
        return self.request("job_submit", operation=operation, arguments=arguments or {})

    def job(self, job_id):
        return self.request("job_status", job_id=job_id)

    def cancel(self, job_id):
        return self.request("job_cancel", job_id=job_id)

    def observe(self, operation, arguments=None, signals=None):
        return self.request("query_start", operation=operation, arguments=arguments or {},
                            signals=list(signals) if signals is not None else None)

    def observation(self, query_id, *, after=None, timeout=25.0):
        if after is None:
            return self.request("query_snapshot", query_id=query_id)
        return self.request("query_wait", query_id=query_id, after=after, timeout=timeout)

    def stop_observing(self, query_id):
        return self.request("query_stop", query_id=query_id)

    def report(self, query_id, consumer, *, reports=3, mode="changes", close_query=True):
        return self.request("report_start", query_id=query_id, consumer=consumer,
                            reports=reports, mode=mode, close_query=close_query)

    def reports(self, consumer, *, cycle_id=None):
        return self.request("reports", consumer=consumer, cycle_id=cycle_id)

    def stop_reports(self, report_id):
        return self.request("report_stop", report_id=report_id)

    def keepalive(self):
        """Renew this client's observation leases without querying the world."""
        return self.request("session_ping")

    def close(self):
        """Release observations. Jobs and already-applied operations keep their real state."""
        return self._request("session_close", {}, min(self.timeout, 2.0))

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        try:
            self.close()
        except AutomationError:
            if _error[0] is None:
                raise
