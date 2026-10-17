import threading
import time

from PySide6.QtCore import QCoreApplication, QEvent, QThread, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from install_queue import InstallQueue


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


def test_installations_are_serial_and_do_not_block_gui():
    app = QApplication.instance() or QApplication([])
    queue = InstallQueue(app)
    release = threading.Event()
    order = []
    publications = []
    workers = []
    queue.changed.connect(lambda: publications.append(QThread.currentThread()))
    queue.job_finished.connect(lambda _job: publications.append(QThread.currentThread()))
    gui_ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: gui_ticks.append(True))
    timer.start(10)

    def first(report):
        workers.append(QThread.currentThread())
        order.append("first")
        report("Downloading", 5_000_000_000, 8_000_000_000)
        assert release.wait(5)
        return "installed"

    a = queue.submit("python:3.13", "Python 3.13", first)
    assert queue.submit("python:3.13", "Duplicate", first) is a
    def second(report):
        workers.append(QThread.currentThread())
        order.append("second")

    b = queue.submit("android", "Android", second)
    try:
        wait_until(lambda: a.completed == 5_000_000_000 and len(gui_ticks) >= 3)
        assert a.total == 8_000_000_000
        assert b.state == "queued"
        assert order == ["first"]
    finally:
        release.set()
        wait_until(lambda: not queue.busy)
        timer.stop()
    assert order == ["first", "second"]
    assert a.state == b.state == "succeeded"
    assert a.result == "installed"
    assert len(workers) == 2
    assert all(thread != app.thread() for thread in workers)
    assert publications
    assert all(thread == app.thread() for thread in publications)


def test_failed_install_exposes_complete_error_and_allows_next_job_after_thread_cleanup():
    from view.job_widgets import JobStrip

    app = QApplication.instance() or QApplication([])
    previous_clipboard = app.clipboard().text()
    queue = InstallQueue(app)
    completions = []
    queue.job_finished.connect(lambda job: completions.append((job.state, queue._thread)))
    error = 'Invalid argument: \\\\?\\D:\\引擎\\' + 'long-path/' * 50 + 'Scripts/numba\nsecond line'

    def fail(report):
        raise RuntimeError(error)

    strip = JobStrip(queue, "python:3.13")
    strip.resize(420, 100)
    strip.show()
    try:
        failed = queue.submit("python:3.13", "Python", fail)
        next_job = queue.submit("engine:0.4", "Engine", lambda report: "ok")
        wait_until(lambda: not queue.busy)
        assert failed.error == "RuntimeError: " + error
        assert completions == [("failed", None), ("succeeded", None)]
        assert next_job.result == "ok"
        strip.refresh()
        assert strip._copy_error.isVisible()
        strip._copy_error.click()
        assert app.clipboard().text() == "RuntimeError: " + error
        queue.clear_finished()
        assert queue.jobs == []
    finally:
        app.clipboard().setText(previous_clipboard)
        strip.close()


def test_queued_install_can_be_removed_without_starting_it():
    app = QApplication.instance() or QApplication([])
    queue = InstallQueue(app)
    calls = []
    job = queue.submit("android", "Android", lambda report: calls.append(True))
    queue.cancel_queued(job)
    QTest.qWait(20)
    assert job.state == "cancelled"
    assert not queue.busy
    assert calls == []


def test_retired_queue_does_not_receive_deferred_next_job_callback():
    app = QApplication.instance() or QApplication([])
    queue = InstallQueue(app)
    idle = []
    queue.idle.connect(lambda: idle.append(True))
    job = queue.submit('runtime', 'Runtime', lambda _report: None)
    queue.cancel_queued(job)
    queue.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QTest.qWait(20)
    assert idle == []
