"""Early preload diagnostics must survive native Console attachment."""
from datetime import datetime
from threading import Thread

from Infernux.debug import DebugConsole, LogEntry, LogType
from Infernux.lib import ConsolePanel


def test_console_attachment_delivers_preload_history_once(capsys):
    console = DebugConsole()
    seen = []
    console.add_listener(seen.append)
    entries = [
        LogEntry("preload ready", LogType.LOG, datetime.now(), source_file="preload.py", source_line=7),
        LogEntry("preload warning", LogType.WARNING, datetime.now()),
        LogEntry("preload failed", LogType.ERROR, datetime.now(), stack_trace="startup traceback"),
    ]
    try:
        for entry in entries:
            console.log(entry)
        before = capsys.readouterr()
        panel = ConsolePanel()
        console.set_native_console(panel)
        console.set_native_console(panel)
        assert panel.get_info_count() == 1
        assert panel.get_warning_count() == 1
        assert panel.get_error_count() == 1
        snapshot = panel._get_visible_log_snapshot(10)
        assert [entry["message"] for entry in snapshot] == [entry.message for entry in entries]
        assert snapshot[0]["source_file"] == "preload.py"
        assert snapshot[0]["source_line"] == 7
        assert snapshot[2]["stack_trace"] == "startup traceback"
        assert seen == entries
        after = capsys.readouterr()
        assert before.out.count("preload ready") == 1
        assert before.err.count("preload failed") == 1
        assert not after.out and not after.err
    finally:
        console.set_native_console(None)
        console.clear()
        DebugConsole._instance = None


def test_console_reconnection_delivers_only_unpublished_retained_entries():
    console = DebugConsole()
    panel = ConsolePanel()
    try:
        console.set_native_console(panel)
        console.log(LogEntry("already published", LogType.LOG, datetime.now()))
        console.set_native_console(None)
        console._max_entries = 2
        console.log(LogEntry("expired", LogType.LOG, datetime.now()))
        console.log(LogEntry("corrected error", LogType.ERROR, datetime.now(), source_file="fixed.py"))
        console.log(LogEntry("new preload ready", LogType.LOG, datetime.now()))
        console.remove_source_entries("fixed.py")
        console.set_native_console(panel)
        snapshot = panel._get_visible_log_snapshot(10)
        assert [entry["message"] for entry in snapshot] == ["already published", "new preload ready"]
        console.set_native_console(None)
        console.log(LogEntry("cleared before attach", LogType.ERROR, datetime.now()))
        console.clear()
        replacement = ConsolePanel()
        console.set_native_console(replacement)
        assert replacement.get_info_count() == replacement.get_error_count() == 0
    finally:
        console.set_native_console(None)
        console.clear()
        DebugConsole._instance = None


def test_worker_logs_during_attachment_are_delivered_once():
    console = DebugConsole()
    panel = ConsolePanel()
    try:
        console.log(LogEntry("before worker", LogType.LOG, datetime.now()))
        worker = Thread(target=lambda: [console.log(LogEntry(f"worker {i}", LogType.LOG, datetime.now())) for i in range(20)])
        worker.start()
        console.set_native_console(panel)
        worker.join(timeout=2)
        assert not worker.is_alive()
        snapshot = panel._get_visible_log_snapshot(30)
        assert len(snapshot) == 21
        assert {entry["message"] for entry in snapshot} == {"before worker", *(f"worker {i}" for i in range(20))}
    finally:
        console.set_native_console(None)
        console.clear()
        DebugConsole._instance = None
