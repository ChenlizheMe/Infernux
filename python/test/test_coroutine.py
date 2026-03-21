"""Tests for InfEngine.coroutine — yield instructions, Coroutine, CoroutineScheduler."""

from __future__ import annotations

import time as stdlib_time

import pytest

from InfEngine.coroutine import (
    Coroutine,
    CoroutineScheduler,
    WaitForEndOfFrame,
    WaitForFixedUpdate,
    WaitForSeconds,
    WaitForSecondsRealtime,
    WaitUntil,
    WaitWhile,
)


# ═══════════════════════════════════════════════════════════════════════════
# Yield instruction unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWaitForSeconds:
    def test_duration_stored(self):
        w = WaitForSeconds(2.5)
        assert w.duration == 2.5

    def test_tick_not_ready(self):
        w = WaitForSeconds(1.0)
        assert w._tick(0.5) is False

    def test_tick_ready(self):
        w = WaitForSeconds(1.0)
        w._tick(0.5)
        assert w._tick(0.6) is True

    def test_repr(self):
        assert "1.0" in repr(WaitForSeconds(1.0))


class TestWaitForSecondsRealtime:
    def test_ready_after_duration(self):
        w = WaitForSecondsRealtime(0.0)  # 0 seconds = immediate
        assert w._is_ready() is True

    def test_not_ready_before_duration(self):
        w = WaitForSecondsRealtime(10.0)
        assert w._is_ready() is False

    def test_repr(self):
        assert "WaitForSecondsRealtime" in repr(WaitForSecondsRealtime(1))


class TestWaitForEndOfFrame:
    def test_repr(self):
        assert "WaitForEndOfFrame" in repr(WaitForEndOfFrame())


class TestWaitForFixedUpdate:
    def test_repr(self):
        assert "WaitForFixedUpdate" in repr(WaitForFixedUpdate())


class TestWaitUntil:
    def test_ready_when_true(self):
        w = WaitUntil(lambda: True)
        assert w._is_ready() is True

    def test_not_ready_when_false(self):
        w = WaitUntil(lambda: False)
        assert w._is_ready() is False


class TestWaitWhile:
    def test_ready_when_false(self):
        w = WaitWhile(lambda: False)
        assert w._is_ready() is True

    def test_not_ready_when_true(self):
        w = WaitWhile(lambda: True)
        assert w._is_ready() is False


# ═══════════════════════════════════════════════════════════════════════════
# Coroutine handle
# ═══════════════════════════════════════════════════════════════════════════

class TestCoroutineHandle:
    def test_initial_state(self):
        def gen():
            yield None
        co = Coroutine(gen())
        assert co.is_finished is False
        assert co._phase == "update"

    def test_repr_running(self):
        def gen():
            yield None
        co = Coroutine(gen())
        assert "running" in repr(co)

    def test_repr_finished(self):
        def gen():
            yield None
        co = Coroutine(gen())
        co._is_finished = True
        assert "finished" in repr(co)

    def test_unique_ids(self):
        def gen():
            yield None
        c1 = Coroutine(gen())
        c2 = Coroutine(gen())
        assert c1._id != c2._id


# ═══════════════════════════════════════════════════════════════════════════
# CoroutineScheduler
# ═══════════════════════════════════════════════════════════════════════════

class TestCoroutineScheduler:
    def test_start_runs_to_first_yield(self):
        steps = []

        def gen():
            steps.append("before")
            yield None
            steps.append("after")

        sched = CoroutineScheduler()
        co = sched.start(gen())
        assert steps == ["before"]
        assert co.is_finished is False

    def test_tick_advances_past_yield_none(self):
        steps = []

        def gen():
            steps.append(1)
            yield None
            steps.append(2)

        sched = CoroutineScheduler()
        sched.start(gen())
        sched.tick_update(0.016)
        assert steps == [1, 2]

    def test_generator_completes(self):
        def gen():
            yield None

        sched = CoroutineScheduler()
        co = sched.start(gen())
        sched.tick_update(0.016)
        assert co.is_finished is True
        assert sched.count == 0

    def test_wait_for_seconds(self):
        steps = []

        def gen():
            steps.append("start")
            yield WaitForSeconds(0.5)
            steps.append("done")

        sched = CoroutineScheduler()
        sched.start(gen())
        assert steps == ["start"]

        sched.tick_update(0.3)
        assert "done" not in steps

        sched.tick_update(0.3)
        assert "done" in steps

    def test_wait_until(self):
        flag = [False]
        steps = []

        def gen():
            yield WaitUntil(lambda: flag[0])
            steps.append("ready")

        sched = CoroutineScheduler()
        sched.start(gen())

        sched.tick_update(0.016)
        assert "ready" not in steps

        flag[0] = True
        sched.tick_update(0.016)
        assert "ready" in steps

    def test_wait_while(self):
        flag = [True]
        steps = []

        def gen():
            yield WaitWhile(lambda: flag[0])
            steps.append("ready")

        sched = CoroutineScheduler()
        sched.start(gen())

        sched.tick_update(0.016)
        assert "ready" not in steps

        flag[0] = False
        sched.tick_update(0.016)
        assert "ready" in steps

    def test_wait_for_fixed_update(self):
        steps = []

        def gen():
            yield WaitForFixedUpdate()
            steps.append("fixed")

        sched = CoroutineScheduler()
        sched.start(gen())

        # update phase should NOT advance this
        sched.tick_update(0.016)
        assert "fixed" not in steps

        # fixed_update phase should advance
        sched.tick_fixed_update(0.02)
        assert "fixed" in steps

    def test_wait_for_end_of_frame(self):
        steps = []

        def gen():
            yield WaitForEndOfFrame()
            steps.append("late")

        sched = CoroutineScheduler()
        sched.start(gen())

        sched.tick_update(0.016)
        assert "late" not in steps

        sched.tick_late_update(0.016)
        assert "late" in steps

    def test_nested_coroutine_wait(self):
        steps = []

        def inner():
            yield None
            steps.append("inner_done")

        def outer():
            sched2 = CoroutineScheduler()
            inner_co = sched2.start(inner())
            yield inner_co
            steps.append("outer_done")

        sched = CoroutineScheduler()
        sched.start(outer())

        # inner_co is not finished yet — outer waits
        sched.tick_update(0.016)
        # inner_co was started inline but outer is waiting on it
        # Since inner_co is managed by its own scheduler, we need to tick it separately
        # In this test: the outer just checks inner_co.is_finished which depends on inner sched
        assert "outer_done" not in steps

    def test_stop_coroutine(self):
        def gen():
            yield None
            yield None

        sched = CoroutineScheduler()
        co = sched.start(gen())
        sched.stop(co)
        assert co.is_finished is True
        assert sched.count == 0

    def test_stop_all(self):
        def gen():
            yield None
            yield None

        sched = CoroutineScheduler()
        c1 = sched.start(gen())
        c2 = sched.start(gen())
        sched.stop_all()
        assert c1.is_finished is True
        assert c2.is_finished is True
        assert sched.count == 0

    def test_count_property(self):
        def gen():
            yield None

        sched = CoroutineScheduler()
        sched.start(gen())
        sched.start(gen())
        assert sched.count == 2

    def test_exception_in_generator_finishes_coroutine(self):
        def gen():
            raise ValueError("boom")
            yield None  # unreachable

        sched = CoroutineScheduler()
        co = sched.start(gen())
        assert co.is_finished is True

    def test_many_coroutines(self):
        results = []

        def gen(idx):
            yield None
            results.append(idx)

        sched = CoroutineScheduler()
        for i in range(50):
            sched.start(gen(i))
        assert sched.count == 50

        sched.tick_update(0.016)
        assert len(results) == 50
        assert sched.count == 0
