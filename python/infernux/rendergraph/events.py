"""User labels and frozen inputs from accepted RenderGraph executions."""

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING
from weakref import ReferenceType, ref

from infernux.compute_source import BufferSource
from infernux.runtime_services import Signal
from .buffer_source import _GraphBufferLease

if TYPE_CHECKING:
    from infernux.lib._Infernux import _GraphEventViewSample


@dataclass(frozen=True, slots=True)
class RenderEvent:
    """One accepted pass occurrence. Device work can still be in flight.

    Compare `history_epoch` within a view to detect a temporal history reset.
    `view_sample` contains the scene camera matrices and jitter used in this
    occurrence. Its previous matrix comes from the last accepted draw in this
    view and is absent after a reset. Matrices are NumPy arrays in row, column
    order. The extent is the scene view size, independent of the dimensions
    of each pass output. Passes such as editor overlays can use a different
    projection.
    """

    label: str
    pass_name: str
    view: int
    generation: int
    occurrence: int
    history_epoch: int
    ordinal: int
    host_frame: int
    submission: tuple[int, int, int]
    view_sample: "_GraphEventViewSample | None"
    _leases: tuple[tuple[str, _GraphBufferLease], ...]
    _metadata_json: str
    _engine: ReferenceType

    @property
    def identity(self) -> tuple[int, int, int, str, int]:
        return self.view, self.generation, self.occurrence, self.pass_name, self.ordinal

    @property
    def completed(self) -> bool:
        """Whether the engine has observed this occurrence's GPU completion, without waiting."""
        from infernux.lib import _Infernux as native

        engine = self._engine()
        host = None if engine is None else engine.get_native_engine()
        return host is not None and native._is_graph_submission_completed(host, self.submission)

    @property
    def metadata(self):
        """Return this declaration's metadata without exposing shared mutable state."""
        return json.loads(self._metadata_json)

    @property
    def source(self) -> BufferSource | None:
        """Borrow a single export, or return None for an event without exports."""
        if len(self._leases) > 1:
            raise ValueError("render event has multiple sources, use .sources")
        return BufferSource(self._leases[0][1]) if self._leases else None

    @property
    def sources(self) -> dict[str, BufferSource]:
        """Borrow the named exports of this occurrence independently of other listeners."""
        return {name: BufferSource(lease) for name, lease in self._leases}


class _GraphSignal(Signal):
    def __init__(self, interest):
        super().__init__()
        self._interest = interest

    def _update_interest(self):
        self._interest.active = bool(self._receivers or self._listeners.listener_count)

    def add_listener(self, callback):
        listener = super().add_listener(callback)
        self._update_interest()
        return listener

    def remove_listener(self, listener):
        super().remove_listener(listener)
        self._update_interest()

    def _add_receiver(self, callback):
        receiver = super()._add_receiver(callback)
        self._update_interest()
        return receiver

    def _remove_receiver(self, receiver):
        super()._remove_receiver(receiver)
        self._update_interest()

    def emit(self, *values):
        raise TypeError("render graph events are emitted by accepted graph submissions")

    def _deliver(self, event):
        try:
            super().emit(event)
        finally:
            self._update_interest()


def events(label: str) -> Signal:
    """Observe the active renderer's accepted occurrences of this label.

    The signal works with ``program.on(signal, arguments)``. Each declaration
    carries its own pass and view identity, even when several share a ticket.
    """
    from infernux.application import Application
    from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
    from infernux.lib import _Infernux as native

    assert_runtime_owner_thread()
    if not isinstance(label, str) or not label:
        raise ValueError("a render graph event needs a nonempty string label")
    engine = Application._current_engine()
    if engine is None:
        raise RuntimeError("render graph events require an active Engine")
    if engine._graph_event_signals is None:
        engine._graph_event_signals = {}
    signals = engine._graph_event_signals
    signal = signals.get(label)
    if signal is None:
        signal = _GraphSignal(native._watch_graph_events(engine.get_native_engine(), label))
        signals[label] = signal
    return signal


def _dispatch_graph_events(engine):
    from infernux.debug import Debug
    from infernux.lib import _Infernux as native

    frame = engine._frame_work
    frame.begin_scope()
    try:
        host_frame = frame.opportunity
        engine_ref = ref(engine)
        for record in native._take_graph_events(engine.get_native_engine()):
            signal = engine._graph_event_signals.get(record.label)
            if signal is None or signal._interest is not record.interest or not signal._interest.active:
                continue
            try:
                leases = tuple((name, _GraphBufferLease(capture, engine)) for name, capture in record.sources)
                signal._deliver(RenderEvent(record.label, record.pass_name, record.view, record.generation,
                    record.occurrence, record.history_epoch, record.ordinal, host_frame, record.submission,
                    record.view_sample, leases, record.metadata, engine_ref))
            except Exception as error:
                Debug.log_exception(error)
    finally:
        frame.end_scope()
