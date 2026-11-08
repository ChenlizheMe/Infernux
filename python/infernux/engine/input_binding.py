"""One host preparation gate for existing numerical request types."""


class InputBindings:
    def __init__(self, owner, sources, *, extra_bytes=0):
        self.slots = {name: source._pin() for name, source in sources.items()}
        self.leases = tuple(dict.fromkeys(self.slots.values()))
        if any(reference() is not owner for lease in self.leases
               if (reference := getattr(lease, 'request_owner', None)) is not None):
            raise ValueError('a request output belongs to its original Engine')
        self.extra_bytes = extra_bytes
        self.state = self._state()

    def _state(self):
        return tuple((lease.done, lease.ready, lease.error) for lease in self.leases)

    def poll(self):
        state = self._state()
        changed = state != self.state
        self.state = state
        return changed

    @property
    def ready(self):
        return all(ready for _, ready, _ in self.state)

    @property
    def error(self):
        return next((error or 'buffer producer was cancelled' for done, ready, error in self.state
                     if done and not ready), '')

    @property
    def producers(self):
        return tuple(dict.fromkeys(producer for lease in self.leases
            if (producer := getattr(lease, 'producer', None)) is not None and producer.status == 'pending'))

    def pending_render_captures(self, engine):
        return (lease.capture for lease in self.leases
                if hasattr(lease, 'capture') and lease.waits_for_render(engine))

    def waits_for_render(self, engine):
        return any(lease.waits_for_render(engine) for lease in self.leases if hasattr(lease, 'waits_for_render'))

    def bind(self, request, owner):
        acquired = {}
        try:
            for lease in self.leases:
                acquired[lease] = lease.acquire()
            request._bind_inputs({name: acquired[lease] for name, lease in self.slots.items()}, owner)
        finally:
            for buffer in acquired.values():
                buffer._close(flush=False)

    def close(self):
        # Releasing one request's references never cancels the shared producer.
        self.slots.clear()
        self.leases = ()
        self.extra_bytes = 0
