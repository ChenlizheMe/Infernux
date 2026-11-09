"""Occurrence receipts and budgeted task release on the existing frame owner."""
from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns
import weakref

from infernux.application import Application
from infernux.execution import Clock, ClockStamp, PresentationStamp, capture_execution_scope, component_execution
from infernux.runtime_services import Signal, register_update

_CLOCK_PHASES = {
    'physics': 'physics_post_step', 'render': 'render_submitted',
    'presentation': 'render_presented', 'host': 'service_update',
}


@dataclass
class _Occurrence:
    sequence: int
    values: tuple
    stamp: ClockStamp
    host_frame: int
    observed_ns: int
    arguments: object = None


@dataclass
class _PeriodicOccurrences:
    sequence: int
    stamp: ClockStamp
    host_frame: int
    observed_ns: int
    count: int
    period: int

    def take(self):
        event = _Occurrence(self.sequence, (self.stamp,), self.stamp, self.host_frame, self.observed_ns)
        self.sequence += 1
        self.stamp = ClockStamp(self.stamp.kind, self.stamp.epoch, self.stamp.tick+self.period)
        self.count -= 1
        return event


class ProgramBinding:
    """Collect occurrences, then release one ordered request at a time.

    Providers run at release, not capture time. Retaining event values keeps
    their objects alive but does not freeze mutable data. Historical inputs
    must be owned samples. completed emits (sequence, request) on retirement.
    The request records when the binding observed the trigger separately from
    its due tick. Periods caught up together share one observation time.
    """
    def __init__(self, program, arguments, *, overlap, clock, options, period=None, signal=None):
        if not callable(arguments):
            raise TypeError('program arguments require a callable returning a mapping')
        if overlap not in ('queue', 'skip', 'latest'):
            raise ValueError("overlap must be 'queue', 'skip' or 'latest'")
        if not any(value.plan._is_current() for value in program._prepared.values()):
            raise RuntimeError('prepare the program signature before binding a trigger')
        engine = Application._current_engine()
        if engine is None:
            raise RuntimeError('program bindings require an active Engine')
        owner = engine._cpu_work(options.get('budget'))
        self._owner = weakref.ref(owner)
        self._scope = capture_execution_scope()
        self.program, self.arguments = program, arguments
        self.clock, self.overlap, self.options = clock, overlap, dict(options)
        self._trigger_clock = clock if clock is not None else Clock('host')
        self.request = None
        self.completed = Signal()
        self.last_trigger = self.last_release = None
        self.received_events = self.released_events = self.completed_events = self.failed_events = 0
        self.skipped_triggers = self.coalesced_triggers = self.abandoned_events = 0
        self.error = ''
        self._closed = self._releasing = self._reported = False
        self._pending = deque()
        self._pending_count = 0
        self._staged = None
        self._period, self._signal = period, signal
        self._epoch = self._next_tick = None
        # These infrastructure callbacks only collect/check permission. Author
        # code executes under the captured scope in _release.
        with component_execution(None):
            self._polls = tuple(register_update(self._poll, phase=phase, enabled=False)
                                for phase in ('service_update', 'physics_post_step'))
            if period is not None:
                stamp = clock.now
                self._epoch, self._next_tick = stamp.epoch, stamp.tick + period
                self._registration = register_update(self._on_clock, phase=_CLOCK_PHASES[clock.kind])
                self._listener = None
            else:
                self._registration = None
                self._listener = signal._add_receiver(self._on_signal)
        owner._program_bindings.add(self)
        if self._scope is not None:
            self._scope._requests.add(self)

    @property
    def closed(self):
        return self._closed

    @property
    def pending_events(self):
        return self._pending_count + int(self._staged is not None)

    @property
    def oldest_trigger(self):
        event = self._staged or (self._pending[0] if self._pending else None)
        return None if event is None else event.stamp

    def close(self):
        if self._closed:
            return
        self._closed = True
        for registration in self._polls:
            registration.close()
        if self._registration is not None:
            self._registration.close()
        if self._signal is not None:
            self._signal._remove_receiver(self._listener)
        if self.request is not None and not self.request.done:
            self.request.cancel()
        self.abandoned_events += self.pending_events
        self._pending.clear()
        self._pending_count = 0
        self._staged = None
        self.arguments = None
        owner = self._owner()
        if owner is not None:
            owner._program_bindings.discard(self)
        if self._scope is not None:
            self._scope._requests.discard(self)

    def cancel(self):
        self.close()

    def _alive(self):
        if self._closed:
            return False
        if (self._scope is not None and self._scope.state == 'closed'
                or self._epoch is not None and self.clock.now.epoch != self._epoch):
            self.close()
            return False
        return True

    def _on_clock(self, delta):
        if not self._alive():
            return
        stamp = self.clock.now
        if self._next_tick <= stamp.tick:
            count = (stamp.tick-self._next_tick)//self._period+1
            due_tick = self._next_tick
            self._next_tick += count*self._period
            if isinstance(stamp, PresentationStamp):
                # A callback registered during a frame joins the next
                # snapshot. Only the observed occurrence has a receipt.
                if stamp.tick == self._next_tick-self._period:
                    self._receive((stamp,), stamp)
                return
            due = ClockStamp(stamp.kind, stamp.epoch, due_tick)
            self._receive((due,), due, count=count, period=self._period)

    def _on_signal(self, *values):
        if self._alive():
            self._receive(values, self._trigger_clock.now)

    def _receive(self, values, stamp, *, count=1, period=0):
        sequence = self.received_events+1
        self.received_events += count
        self.last_trigger = (stamp if isinstance(stamp, PresentationStamp)
                             else ClockStamp(stamp.kind, stamp.epoch, stamp.tick+(count-1)*period))
        busy = self._staged is not None or bool(self._pending) or self.request is not None and not self.request.done
        if busy and self.overlap == 'skip':
            self.skipped_triggers += count
            return
        if self.overlap == 'latest':
            self.coalesced_triggers += self._pending_count+count-1
            self._pending.clear()
            self._pending_count = 0
            sequence, stamp = self.received_events, self.last_trigger
            values = (stamp,) if period else values
            count = 1
        elif self.overlap == 'skip':
            self.skipped_triggers += count-1
            count = 1
        frame = self._owner()._frame.opportunity
        observed_ns = perf_counter_ns()
        event = (_PeriodicOccurrences(sequence, stamp, frame, observed_ns, count, period) if count > 1
                 else _Occurrence(sequence, values, stamp, frame, observed_ns))
        self._pending.append(event)
        self._pending_count += count
        self._set_polling(True)

    def _set_polling(self, enabled):
        for registration in self._polls:
            registration.enabled = enabled

    def _permitted(self, owner):
        return (owner._accepting and owner._frame.active and owner._frame.accepting
                and owner._available(background=True)
                and (self._scope is None or self._scope.state in ('active', 'stepping')))

    def _can_release(self, owner):
        return self._permitted(owner) and len(owner.requests) < owner.budget.max_requests

    def _poll(self, delta):
        if not self._alive() or self._releasing:
            return
        if self._scope is not None and self._scope.state == 'paused':
            return
        owner = self._owner()
        if self.request is not None:
            if not self.request.done:
                return
            if not self._reported:
                if not self._permitted(owner):
                    return
                self._reported = True
                self.completed_events += 1
                self.failed_events += int(self.request.status != 'published')
                started = perf_counter_ns()
                owner._frame.begin_scope()
                try:
                    with component_execution(self._scope):
                        self.completed.emit(self.request.metrics['trigger_sequence'], self.request)
                finally:
                    owner._charge_host_interval(started)
                if not self._alive():
                    return
        if not self.pending_events:
            self._set_polling(False)
            return
        if not self._can_release(owner):
            return
        if self._staged is None:
            first = self._pending[0]
            if isinstance(first, _PeriodicOccurrences):
                self._staged = first.take()
                if not first.count:
                    self._pending.popleft()
            else:
                self._staged = self._pending.popleft()
            self._pending_count -= 1
        self._release(owner, self._staged)

    def _release(self, owner, event):
        started = perf_counter_ns()
        owner._frame.begin_scope()
        self._releasing = True
        try:
            with component_execution(self._scope):
                if event.arguments is None:
                    event.arguments = self.arguments(*event.values)
                if not self._alive() or not self._can_release(owner):
                    return
                _, key = self.program._arguments((), event.arguments)
                prepared = self.program._prepared.get(key)
                if (prepared is not None and prepared.optimizer is not None
                        and not owner._training_available(prepared.plan.root)):
                    return
                request = self.program.submit(**event.arguments, **self.options)
            self.request = request
            self._reported = False
            self._staged = None
            self.released_events += 1
            self.last_release = request.submission_stamp
            request.metrics.update(trigger_sequence=event.sequence, trigger_clock=event.stamp.kind,
                trigger_epoch=event.stamp.epoch, trigger_tick=event.stamp.tick,
                trigger_host_frame=event.host_frame, trigger_observed_ns=event.observed_ns,
                queued_host_frames=request.release_opportunity-event.host_frame)
            if self._period is not None and isinstance(event.stamp, PresentationStamp):
                request.metrics['trigger_accepted_ns'] = event.stamp.accepted_ns
        except BaseException as error:
            self.error = str(error)
            self.close()
            raise
        finally:
            self._releasing = False
            owner._charge_host_interval(started)
