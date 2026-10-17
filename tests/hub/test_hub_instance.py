"""Real competing processes exercise the same lock/IPC used by the frozen Hub."""

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


CHILD = r'''
import os, sys, time
from pathlib import Path
from PySide6.QtCore import QCoreApplication, QTimer
from hub_instance import HubInstance
app = QCoreApplication([])
root = Path(sys.argv[1])
instance = HubInstance(root)
try:
    if not instance.acquire_or_activate(timeout_ms=int(sys.argv[3])):
        print("SECONDARY", flush=True)
        sys.exit(0)
    with (root / "owners").open("a") as stream:
        stream.write(str(os.getpid()) + "\n")
    print("PRIMARY", flush=True)
    time.sleep(float(sys.argv[2]))
    def activate():
        with (root / "activations").open("a") as stream:
            stream.write("activated\n")
    instance.set_activation_handler(activate)
    timer = QTimer()
    timer.timeout.connect(lambda: app.quit() if (root / "stop").exists() else None)
    timer.start(20)
    app.exec()
finally:
    instance.close()
'''


@pytest.fixture
def launch():
    children = []

    def start(root, delay=0, timeout_ms=5000):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging"))
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD, str(root), str(delay), str(timeout_ms)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        children.append(process)
        return process

    yield start
    for process in children:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Child process did not reach the expected state"
        time.sleep(0.02)


def test_simultaneous_launches_choose_one_owner_and_activate_it(tmp_path, launch):
    children = [launch(tmp_path, delay=0.5) for _ in range(6)]
    wait_until(lambda: sum(p.poll() is not None for p in children) == 5)
    owner = next(p for p in children if p.poll() is None)
    for process in children:
        if process is not owner:
            output, error = process.communicate(timeout=5)
            assert process.returncode == 0, f"pid={process.pid}; stdout={output!r}; stderr={error}"
            assert output.strip() == "SECONDARY", f"pid={process.pid}; stderr={error}"
    assert (tmp_path / "owners").read_text().splitlines() == [str(owner.pid)]
    assert len((tmp_path / "activations").read_text().splitlines()) == 5
    (tmp_path / "stop").touch()
    assert owner.wait(timeout=5) == 0
    assert (tmp_path / "hub-instance.lock").exists()  # Stable kernel-lock inode.
    assert not (tmp_path / "hub-instance.json").exists()


@pytest.mark.parametrize("crash", [False, True])
def test_exit_or_crash_allows_a_new_primary(tmp_path, launch, crash):
    first = launch(tmp_path)
    wait_until(lambda: (tmp_path / "owners").exists())
    if crash:
        first.kill()
    else:
        (tmp_path / "stop").touch()
    first.wait(timeout=5)
    (tmp_path / "stop").unlink(missing_ok=True)
    second = launch(tmp_path)
    wait_until(lambda: len((tmp_path / "owners").read_text().splitlines()) == 2)
    assert second.poll() is None
    (tmp_path / "stop").touch()
    assert second.wait(timeout=5) == 0


def test_busy_primary_is_not_replaced_when_activation_times_out(tmp_path, launch):
    owner = launch(tmp_path, delay=5)
    wait_until(lambda: (tmp_path / "owners").exists())
    second = launch(tmp_path, timeout_ms=200)
    _output, error = second.communicate(timeout=5)
    assert second.returncode != 0
    assert "did not respond" in error
    assert owner.poll() is None
    assert (tmp_path / "owners").read_text().splitlines() == [str(owner.pid)]


def test_independent_user_data_roots_have_independent_instances(tmp_path, launch):
    roots = [tmp_path / "first", tmp_path / "second"]
    children = [launch(root) for root in roots]
    wait_until(lambda: all((root / "owners").exists() for root in roots))
    assert all(p.poll() is None for p in children)
    for root in roots:
        (root / "stop").touch()
    assert all(p.wait(timeout=5) == 0 for p in children)


def test_live_lock_does_not_expire_with_age(tmp_path, launch):
    owner = launch(tmp_path)
    lock_path = tmp_path / "hub-instance.lock"
    wait_until(lambda: (tmp_path / "owners").exists())
    old = time.time() - 86400
    os.utime(lock_path, (old, old))
    second = launch(tmp_path)
    output, error = second.communicate(timeout=5)
    assert second.returncode == 0, error
    assert output.strip() == "SECONDARY"
    assert owner.poll() is None
