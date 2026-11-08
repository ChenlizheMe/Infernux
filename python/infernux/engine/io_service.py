"""Numerical observations of the original native IO cursors, not an executor."""
from dataclasses import dataclass

from .service_windows import IoSpanSample, ServiceDemand, ServiceGroup, ServiceCycle


def span_key(span, chunk_bytes, target_ns):
    # An atom cannot amortize a larger extent over more callbacks. Require its
    # actual extent's observation (e.g. image bytes plus module symbol count).
    return ("io-span", span.operation, span.parameter, chunk_bytes, target_ns,
            span_extent(span) if span.kind == "atom" else
            span_extent(span).bit_length() if span.kind.endswith("_fragment") else
            (span.entry_bytes or 0).bit_length() if span.kind == "codec" else None)


def span_extent(span):
    if span.operation == "write.zstd_finish":
        return max(1, span.entry_bytes)
    return max(span.input_bytes or 0, span.output_bytes or 0, 1)


def io_host_key(work):
    return ("io-host", work.operation, work.parameter, bool(work.bytes))


@dataclass
class _SpanProgress:
    key: tuple
    entry: int | None
    units: int
    kind: str
    calls: int = 0
    execution_ns: int = 0
    host_ns: int = 0
    lag: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    regular: bool = True

    def sample(self):
        return IoSpanSample(self.units, self.calls,
            self.execution_ns, self.host_ns, self.lag,
            self.input_bytes if self.kind.startswith("codec") else None,
            self.output_bytes if self.kind.startswith("codec") else None)


@dataclass
class IoObservation:
    request: object
    execution_ns: int
    submissions: int
    work: object = None
    execution_key: tuple | None = None
    accepted_submission: int = 0
    opportunity: int = 0
    host_ns: int = 0
    recipe: object = None
    progress: object = None
    span: object = None
    grant: int = 0
    target_ns: int = 0
    streaming: bool = False

    def begin(self, task, maximum, target_ns, *, regular=True, collect_span=True):
        describe = getattr(task, "describe_remaining_work", None)
        if describe is None:
            return
        self.recipe, self.progress = describe(), getattr(task, "progress", None)
        if not collect_span:
            self.span = None  # A batch may finish several distinct recipe spans.
            return
        first = self.recipe[0][0][0] if self.recipe else None
        if first is None or first.kind in ("barrier", "input", "output"):
            return
        if (first.operation == "read.entry_end" and self.span is not None
                and self.span.kind == "codec"):
            return  # The decoder's final transition belongs to its real read.
        key = span_key(first, maximum, target_ns)
        if self.span is not None and self.span.entry == first.entry_index:
            if self.span.key == key or (first.kind.endswith("_fragment") and self.span.kind == first.kind
                                       and self.span.key[:5] == key[:5]):
                self.span.regular &= regular
                return  # Keep a fragment's initial class as its remainder shrinks.
            if self.span.key[1] == first.operation == "read.buffer":
                self.span.regular = False  # Capacity changed within this read's profile.
                return
        # A zero-input compression drain depends on the real complete entry,
        # not on an invented consumed byte or the compressed-size upper bound.
        self.span = _SpanProgress(key, first.entry_index, span_extent(first), first.kind, regular=regular)

    def complete(self, task, owner):
        if task.pending:
            return
        count = int(task.submissions)
        if count == self.submissions:
            return
        elapsed = int(task.execution_ns) - self.execution_ns
        owner._frame.observe_resource(owner._resource('io'), elapsed, self.opportunity, self.grant)
        target = self.target_ns or owner.budget._io_quantum_ns
        streaming, self.streaming = self.streaming, False
        self.grant = self.target_ns = 0
        metrics, work = self.request.metrics, self.work
        metrics["io_worker_execution_ns"] = metrics.get("io_worker_execution_ns", 0) + elapsed
        metrics["io_completed_submissions"] = metrics.get("io_completed_submissions", 0) + count - self.submissions
        metrics["io_execution_target_overruns"] = metrics.get("io_execution_target_overruns", 0) + (
            not streaming and count == self.submissions + 1 and elapsed > target)
        valid = (work is not None and count == self.accepted_submission
                 and task.state not in ("failed", "cancelled", "closed")
                 and self.request.status not in ("failed", "cancelled", "expired"))
        if valid:
            lag = max(0, owner._last_opportunity - self.opportunity)
            progress = getattr(task, "progress", None) if self.recipe is not None else None
            units = work.bytes
            if work.operation == "read.buffer":
                units = progress["buffer_work_bytes"] - self.progress["buffer_work_bytes"]
            elif work.operation == "read.directory":
                units = progress["read_bytes"] - self.progress["read_bytes"]
            elif work.operation == "write.batch":
                units = sum(progress[name] - self.progress[name] for name in ("written_bytes", "hashed_bytes"))
            owner._costs.observe(self.execution_key, units or 1, elapsed, lag=lag)
            metrics["io_execution_samples"] = metrics.get("io_execution_samples", 0) + (elapsed > 0)
            metrics["io_observation_lag"] = lag
            if self.recipe is not None:
                before = self.progress
                self.recipe, self.progress = task.describe_remaining_work(), progress
                fragment = self.span
                if fragment is not None:
                    reading = fragment.key[1].startswith("read.")
                    fragment.calls += 1
                    fragment.execution_ns = max(fragment.execution_ns, elapsed)
                    fragment.host_ns = max(fragment.host_ns, self.host_ns)
                    fragment.lag = max(fragment.lag, lag)
                    if fragment.kind.startswith("codec"):
                        fragment.input_bytes += self.progress["codec_input_bytes" if reading else "consumed_bytes"] - before[
                            "codec_input_bytes" if reading else "consumed_bytes"]
                        fragment.output_bytes += self.progress["decoded_bytes" if reading else "written_bytes"] - before[
                            "decoded_bytes" if reading else "written_bytes"]
                    first = self.recipe[0][0][0] if self.recipe else None
                    ended = (task.state == "ready" if reading and fragment.kind == "codec" else first is None or
                             first.operation != fragment.key[1] or
                             first.entry_index != fragment.entry)
                    if ended:
                        # Idle borrowing can change the batch count. Only a
                        # regular span can price a future deadline reservation.
                        if fragment.regular:
                            owner._costs.observe_io_span(fragment.key, fragment.sample())
                        metrics["io_completed_spans"] = metrics.get("io_completed_spans", 0) + 1
                        if fragment.kind.startswith("codec"):
                            metrics["codec_completed_fragments"] = metrics.get("codec_completed_fragments", 0) + 1
                            metrics["codec_actual_input_bytes"] = metrics.get("codec_actual_input_bytes", 0) + fragment.input_bytes
                            metrics["codec_actual_output_bytes"] = metrics.get("codec_actual_output_bytes", 0) + fragment.output_bytes
                        self.span = None
        else:
            self.span = None  # Cleanup and failed fragments never teach normal service.
        self.execution_ns += elapsed
        self.submissions, self.work = count, None

    def availability(self, costs, opportunity, observed_opportunity):
        if self.work is None:
            return ServiceDemand(reason="accepted IO has no normal-step observation")
        if self.streaming:
            return ServiceDemand(reason="streaming file work awaits its actual completion")
        lag = costs.lag(self.execution_key)
        if lag is None:
            return ServiceDemand(reason="accepted IO has no observed completion-lag profile")
        ready = self.opportunity + lag
        if observed_opportunity > ready:
            return ServiceDemand(reason="accepted IO exceeded its observed completion-lag profile")
        ready = max(ready, observed_opportunity + 1)
        return ServiceDemand(ready_at=max(opportunity, ready), occupied=(("io", ready),))


def recipe_service(recipe, owner, chunk_bytes, *, observation=None, pending=False, input_service=None):
    """Price native IO submissions from observed cursor progress."""
    groups, minimum, first = [], 0, True
    for spans, repeats in recipe:
        units = []
        for span in spans:
            if span.kind in ("barrier", "output"):
                return ServiceDemand(reason="IO still needs " + span.operation + " facts")
            if span.kind == "input":
                if input_service is None:
                    return ServiceDemand(reason="IO input consumer is unprofiled")
                supplied = input_service(span, repeats)
                if supplied.reason:
                    return supplied
                units.extend(supplied.groups)
                minimum = max(minimum, supplied.minimum_atom_ns)
                continue
            if span.kind in ("bytes", "upper_bytes", "bytes_fragment") and span.input_bytes is None:
                return ServiceDemand(reason="IO extent is not yet known: " + span.operation)
            active = None if not first or observation is None else observation.span
            if active is not None and not (active.entry == span.entry_index and (
                    active.key[1] == span.operation or span.operation == "read.entry_end" and active.kind == "codec")):
                active = None
            if active is not None and not active.regular:
                return ServiceDemand(reason="IO span used changing batch allowances")
            key = (active.key if active is not None else
                   span_key(span, chunk_bytes, owner.budget._io_quantum_ns))
            extent = span_extent(span) if active is None else active.units
            profile = owner._costs.io_span(key, extent)
            if profile is None:
                return ServiceDemand(reason="IO span is unprofiled: " + key[1])
            if span.kind == "atom":
                calls = 1
            else:
                calls = profile.calls
                if span.kind in ("bytes", "upper_bytes", "bytes_fragment") and span.input_bytes is not None:
                    calls = min(calls, extent)
                if active is not None:
                    calls -= active.calls
            calls -= int(first and pending)
            if calls < 0 or calls == 0 and not (first and pending):
                return ServiceDemand(reason="IO span exceeded its observed callback profile: " + key[1])
            execution = profile.execution_ns
            execution_limit = owner.budget._io_quantum_ns
            if key[1] == "read.buffer":
                execution_limit = owner.budget._io_frame_allowance_ns
                execution = max(execution, owner.budget._io_quantum_ns)
            if execution > execution_limit:
                return ServiceDemand(reason="observed IO policy exceeds its execution target: " + key[1], failure="host_target")
            if profile.host_ns > owner._max_background_credit():
                return ServiceDemand(reason="IO preparation/handoff exceeds its host allowance", failure="host_target")
            indivisible = (span.kind == 'atom' or
                           span.kind in ('bytes', 'bytes_fragment') and span_extent(span) <= 1)
            if calls:
                units.append(ServiceGroup('io', calls, profile.lag, profile.host_ns,
                                          execution, 0, indivisible))
            first = False
        if units:
            groups.append(ServiceCycle(tuple(units), repeats) if repeats > 1 else
                          units[0] if len(units) == 1 else ServiceCycle(tuple(units), 1))
    return ServiceDemand(tuple(groups), minimum_atom_ns=minimum)
