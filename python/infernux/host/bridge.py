"""Local public engine endpoint, independent of MCP and any agent harness."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Condition, Thread
from time import monotonic

from .context import configure
from .operations import OperationError, OperationJobRegistry, OperationRegistry
from .queries import QueryService, emit_signal


class AutomationBridge:
    def __init__(self, project_root, *, registry=None, session_lease_seconds=90.0):
        self.context = configure(project_root)
        self.registry = registry or OperationRegistry.instance()
        self.jobs = OperationJobRegistry(self.registry)
        self.queries = QueryService(self.registry)
        from infernux_automation.delivery import ReportDelivery
        self.reports = ReportDelivery(self.queries)
        self._closed = False
        self._cleanups = []
        self._sessions = {}
        self._session_condition = Condition()
        self._session_lease_seconds = max(float(session_lease_seconds), 0.05)
        self._methods = {
            "info": self.info, "operations": self.operations,
            "schema": lambda operation: self.registry.get(operation).schema.document(),
            "call": self.call,
            "batch": lambda calls: self.registry.execute_batch(calls, capabilities=("*",)),
            "job_submit": lambda operation, arguments=None: {"job_id": self.jobs.submit(operation, arguments, capabilities=("*",))},
            "job_status": self.jobs.status, "job_cancel": self.jobs.cancel,
            "query_start": self.queries.start, "query_snapshot": self.queries.snapshot,
            "query_wait": self.queries.wait, "query_stop": self.queries.stop,
            "report_start": self.reports.add, "reports": self.reports.consume, "report_stop": self.reports.remove,
            "signal": lambda name, payload=None: emit_signal(name, payload),
        }
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                status = 200
                try:
                    document = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    if document["instance_id"] != bridge.context.instance_id:
                        raise OperationError("engine.instance_changed", "The selected engine instance is no longer current")
                    if bridge._closed:
                        raise OperationError("engine.closed", "Engine is closing")
                    method = self.path.strip("/")
                    session_id = str(document.get("session_id", ""))
                    arguments = dict(document.get("arguments", {}))
                    bridge._touch_session(session_id, create=method in {"query_start", "report_start"})
                    if method in {"query_start", "report_start"}:
                        arguments["owner"] = session_id
                    if method == "session_close":
                        result = bridge.close_session(session_id)
                    elif method == "session_ping":
                        result = {"connected": True}
                    else:
                        result = bridge._methods[method](**arguments)
                    response = {"ok": True, "result": result}
                except OperationError as exc:
                    status, response = 400, exc.envelope()
                except Exception as exc:
                    status, response = 400, OperationError("automation.failed", str(exc)).envelope()
                encoded = json.dumps(response, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # A disconnected caller does not replay/cancel a world mutation.

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = Thread(target=self.server.serve_forever, name="InfernuxAutomation", daemon=True)
        self.thread.start()
        self.descriptor = Path(project_root) / "Cache" / "Automation" / "engine.json"
        self.descriptor.parent.mkdir(parents=True, exist_ok=True)
        from infernux.core.document_store import write_document_text
        write_document_text(str(self.descriptor), json.dumps(self.info(), ensure_ascii=False) + "\n")
        self._catalog_subscription = self.registry.subscribe(lambda event: emit_signal("operations.changed", event))
        self._session_thread = Thread(target=self._expire_sessions, name="InfernuxObservationLease", daemon=True)
        self._session_thread.start()

    def _touch_session(self, session_id, *, create=False):
        if not session_id:
            return
        with self._session_condition:
            if create or session_id in self._sessions:
                self._sessions[session_id] = monotonic() + self._session_lease_seconds
                self._session_condition.notify_all()

    def close_session(self, session_id, *, expired_before=None):
        if not session_id:
            return {"queries_closed": 0, "reports_closed": 0}
        with self._session_condition:
            if expired_before is not None and self._sessions.get(session_id, float("inf")) > expired_before:
                return {"queries_closed": 0, "reports_closed": 0}
            self._sessions.pop(session_id, None)
            self._session_condition.notify_all()
            reports = self.reports.close_owner(session_id)
            queries = self.queries.close_owner(session_id)
        return {"queries_closed": queries, "reports_closed": reports}

    def _expire_sessions(self):
        # A crashed sidecar cannot send session_close. Expire only observation
        # leases, never jobs or applied writes; this loop does not touch frames.
        while True:
            with self._session_condition:
                if self._closed:
                    return
                now = monotonic()
                expired = [identity for identity, deadline in self._sessions.items() if deadline <= now]
                if not expired:
                    timeout = min(self._sessions.values()) - now if self._sessions else None
                    self._session_condition.wait(timeout=timeout)
                    continue
            for identity in expired:
                self.close_session(identity, expired_before=now)

    def info(self):
        return {"project": self.context.project_root, "instance_id": self.context.instance_id,
                "endpoint": self.endpoint, "pid": os.getpid(), "catalog_revision": self.registry.revision,
                "artifact_root": self.context.artifact_root}

    def operations(self, query="", limit=50):
        entries = self.registry.search(query, limit=limit) if query else self.registry.list()[:limit]
        return {"revision": self.registry.revision, "operations": entries}

    def call(self, operation, arguments=None):
        return self.registry.execute(operation, arguments, capabilities=("*",))

    def attach_editor_signals(self):
        from infernux.debug import DebugConsole
        from infernux.engine.play_mode import PlayModeManager
        from infernux.engine.interaction import AssetMutationService
        from .editor import EditorAutomationHost
        native = EditorAutomationHost.instance()._native_engine()
        native.set_capture_completion_callback(lambda capture_id: emit_signal("capture.completed", capture_id))
        self._cleanups.append(lambda: native.set_capture_completion_callback(None))
        console = DebugConsole.instance()
        logged = lambda entry: emit_signal("console.changed")
        console.add_listener(logged)
        self._cleanups.append(lambda: console.remove_listener(logged))
        assets = AssetMutationService.instance()
        if assets is not None:
            changed_assets = lambda event: emit_signal("assets.changed")
            assets.add_observer(changed_assets)
            self._cleanups.append(lambda: assets.remove_observer(changed_assets))
        play = PlayModeManager.instance()
        if play is not None:
            changed = lambda event: emit_signal("runtime.changed")
            play.add_state_change_listener(changed)
            self._cleanups.append(lambda: play.remove_state_change_listener(changed))
        from infernux.engine.bootstrap import EditorBootstrap
        from infernux.lib import SceneManager
        engine = EditorBootstrap.instance().engine
        getter = getattr(engine, "get_runtime_change_journal", None)
        journal = getter() if callable(getter) else None
        if journal is None:
            from infernux.engine.runtime_change_journal import runtime_change_journal

            journal = runtime_change_journal()
        self._attach_runtime_observations(journal, SceneManager.instance(), play)

    def _attach_runtime_observations(self, journal, scene_manager, play):
        from infernux.runtime_services import register_update
        from .commands import MainThreadCommandQueue
        previous = {"world": (journal.revision, int(scene_manager.get_global_transform_serial())), "clock": None}
        def published_state(_delta):
            # Read existing publication counters, never enumerate a world. This
            # callback is absent from the scheduler while no relevant watch exists.
            if self.queries.has_subscribers("world.changed"):
                stamp = journal.revision, int(scene_manager.get_global_transform_serial())
                if stamp != previous["world"]:
                    previous["world"] = stamp
                    emit_signal("world.changed", {"revision": stamp[0], "transform_serial": stamp[1]})
            if self.queries.has_subscribers("runtime.changed"):
                frame_count = int(getattr(scene_manager, "runtime_frame_count",
                                           scene_manager._frame_time.frame_count))
                step_sequence = int(getattr(play, "step_sequence", 0)) if play else 0
                stamp = (int(scene_manager.simulation_epoch), frame_count,
                         int(scene_manager.published_fixed_step_index), step_sequence)
                if stamp != previous["clock"]:
                    previous["clock"] = stamp
                    emit_signal("runtime.changed", {"epoch": stamp[0], "frame": stamp[1],
                                                    "fixed_step": stamp[2], "step_sequence": stamp[3]})
        registration = register_update(published_state, phase="service_update", enabled=False)
        def activity():
            def apply():
                registration.enabled = (not self._closed and
                    self.queries.has_subscribers("world.changed", "runtime.changed"))
            MainThreadCommandQueue.instance().submit("automation.observation.activity", apply)
        self.queries.set_activity_listener(activity)
        activity()
        self._cleanups.append(registration.close)
        self._cleanups.append(lambda: self.queries.set_activity_listener(None))
        return registration

    def close(self):
        with self._session_condition:
            if self._closed:
                return
            self._closed = True
            self._session_condition.notify_all()
        self._session_thread.join()
        self.registry.unsubscribe(self._catalog_subscription)
        self.queries.set_activity_listener(None)
        self.reports.close()
        self.queries.close()
        self.jobs.shutdown(wait=False)
        failures = []
        for cleanup in reversed(self._cleanups):
            try:
                cleanup()
            except Exception as exc:
                failures.append(str(exc))
        try:
            from .players import close_service
            close_service()
        except Exception as exc:
            failures.append(str(exc))
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
            try:
                current = json.loads(self.descriptor.read_text(encoding="utf-8"))
                if current.get("instance_id") == self.context.instance_id:
                    self.descriptor.unlink(missing_ok=True)
            except (FileNotFoundError, ValueError):
                pass
        if failures:
            raise RuntimeError("Automation shutdown completed with cleanup errors: " + "; ".join(failures))


_active = None


def start(project_root):
    global _active
    if _active is not None:
        _active.close()
    _active = AutomationBridge(project_root)
    return _active


def current():
    return _active


def stop():
    global _active
    bridge, _active = _active, None
    if bridge is not None:
        bridge.close()
