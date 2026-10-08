"""Loopback HTTP plus real Qt ownership/quit, with no external network."""
import ctypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import threading
import time
import traceback

from PySide6.QtCore import QTimer, QtMsgType, qInstallMessageHandler
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel
import shiboken6

import community_feed
from hub_utils import HubLaunchContext
from launcher import GameEngineLauncher
from view.discussion_view import DiscussionView


def spin(predicate):
    until = time.monotonic() + 5
    while not predicate() and time.monotonic() < until:
        QTest.qWait(5)
    assert predicate(), 'Qt/network condition did not complete'


def main():
    action, response = sys.argv[1:]
    if os.name == 'nt':
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    errors = []
    def error(*exc):
        errors.append(str(exc[1]))
        traceback.print_exception(*exc)
    sys.excepthook = error
    entered = threading.Event()
    requests = []
    body = json.dumps({'topic_list': {'filter': 'top', 'topics': [dict(
        id=7, slug='topic', title='Local topic', posts_count=3, views=5, like_count=2)]}}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            entered.set()
            try:
                if response == 'silent' or action == 'quit':
                    time.sleep(0.7)
                else:
                    time.sleep(0.05)
                self.send_response(503 if response == 'failure' else 200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                if response == 'trickle':
                    for chunk in body:
                        self.wfile.write(bytes([chunk]))
                        self.wfile.flush()
                        time.sleep(0.03)
                else:
                    self.wfile.write(b'{invalid' if response == 'malformed' else body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    community_feed.HOT_TOPICS_URL = f'http://127.0.0.1:{server.server_port}/top.json'
    DiscussionView.REQUEST_TIMEOUT_MS = 250
    hub = GameEngineLauncher(HubLaunchContext.SOURCE)
    view = hub.discussion_view
    start = time.monotonic()
    view.refresh()
    spin(entered.is_set)
    for _ in range(5):
        view.refresh()
    assert len(requests) == 1, requests
    if action == 'quit':
        QTimer.singleShot(0, hub.request_quit)
        app.exec()
        assert time.monotonic() - start < 0.6, 'optional feed blocked application exit'
        shiboken6.delete(hub)
    elif action == 'destroy':
        shiboken6.delete(view)
        QTest.qWait(40)
        hub.db.close()
        shiboken6.delete(hub)
    else:
        spin(view._refresh.isEnabled)
        labels = view.findChildren(QLabel)
        if response == 'success':
            assert any(label.text() == 'Local topic' for label in labels)
        else:
            assert any(label.property('kind') == 'error' for label in labels)
        if response in ('silent', 'trickle'):
            assert time.monotonic() - start < 0.6, 'request had no total deadline'
        # A completed/failed request must allow exactly one fresh request.
        view.refresh()
        spin(lambda: len(requests) == 2)
        spin(view._refresh.isEnabled)
        assert not errors, errors
        QTimer.singleShot(0, hub.request_quit)
        app.exec()
        shiboken6.delete(hub)
    assert not errors, errors
    print('COMMUNITY_LIFECYCLE_OK', flush=True)


if __name__ == '__main__':
    previous_handler = qInstallMessageHandler(
        lambda kind, context, message:
        print(message, file=sys.stderr, flush=True) if kind == QtMsgType.QtFatalMsg else None)
    try:
        main()
    finally:
        # Qt plugin unloading can log during process teardown. Retire the
        # probe's Python callback while the interpreter is still alive.
        qInstallMessageHandler(previous_handler)
